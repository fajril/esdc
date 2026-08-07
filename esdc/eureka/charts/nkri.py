"""NKRI-level Plotly figure generators for the Eureka dashboard."""

from __future__ import annotations

from collections import defaultdict

import plotly.graph_objects as go

from esdc.eureka.charts.colors import (
    ONSTREAM_PALETTE,
    PLOTLY_LAYOUT_DEFAULTS,
    TIMESERIES_PALETTE_AN,
    TIMESERIES_PALETTE_OC,
)
from esdc.eureka.queries import NKRITimeseriesRow, OnstreamRow

# ── Unit conversion helpers ────────────────────────────────────

# MSTBY (thousand STB/year) → MBOPD (thousand BOPD)
# X MSTBY × 1,000,000 STB/year → ÷365 → BOPD → ÷1000 → MBOPD
# = X × 1000 / 365 → but MSTBY is already thousand STB/yr,
# so MBOPD = MSTBY / 365


def _to_mbopd(mstby: float) -> float:
    return round(mstby / 365, 1) if mstby else 0.0


# BSCFY (billion SCF/year) → MMSCFD (million SCFD)
# X BSCFY × 1,000,000,000 SCF/year → ÷365 → SCFD → ÷1,000,000 → MMSCFD
# = X × 1000 / 365


def _to_mmscfd(bscfy: float) -> float:
    return round(bscfy * 1000 / 365, 1) if bscfy else 0.0


# ── Category definitions ───────────────────────────────────────

TIMESERIES_CATEGORIES: list[str] = [
    "Reserves",
    "Sales Potential Resources",
    "Contingent Resources (Exploitation)",
    "Contingent Resources (Exploration)",
    "Prospective Resources",
]

ONSTREAM_CATEGORIES: list[str] = [
    "Reserves",
    "Contingent Resources (Exploitation)",
    "Contingent Resources (Exploration)",
    "Prospective Resources",
]


def _agg_timeseries_oc(
    data: list[NKRITimeseriesRow],
) -> tuple[list[int], dict[str, list[float]]]:
    """Aggregate NKRITimeseriesRow into per-year per-category MBOPD."""
    by_year: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    years_set: set[int] = set()

    for row in data:
        yr = row.year
        years_set.add(yr)
        cls = _normalize_class(row.project_class)
        prefix = row.project_level

        if cls == "Reserves & GRR":
            by_year[yr]["Reserves"] += _to_mbopd(row.slf_oc)
            by_year[yr]["Sales Potential Resources"] += _to_mbopd(row.spf_oc)
        elif cls == "Contingent Resources":
            if prefix == "E":
                by_year[yr]["Contingent Resources (Exploitation)"] += _to_mbopd(
                    row.tpf_oc
                )
            elif prefix == "X":
                by_year[yr]["Contingent Resources (Exploration)"] += _to_mbopd(
                    row.tpf_oc
                )
        elif cls == "Prospective Resources":
            by_year[yr]["Prospective Resources"] += _to_mbopd(row.tpf_risked_oc)

    sorted_years = sorted(years_set)
    result: dict[str, list[float]] = {}
    for cat in TIMESERIES_CATEGORIES:
        result[cat] = [by_year[y].get(cat, 0.0) for y in sorted_years]

    return sorted_years, result


def _agg_timeseries_an(
    data: list[NKRITimeseriesRow],
) -> tuple[list[int], dict[str, list[float]]]:
    """Aggregate NKRITimeseriesRow into per-year per-category MMSCFD."""
    by_year: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    years_set: set[int] = set()

    for row in data:
        yr = row.year
        years_set.add(yr)
        cls = _normalize_class(row.project_class)
        prefix = row.project_level

        if cls == "Reserves & GRR":
            by_year[yr]["Reserves"] += _to_mmscfd(row.slf_an)
            by_year[yr]["Sales Potential Resources"] += _to_mmscfd(row.spf_an)
        elif cls == "Contingent Resources":
            if prefix == "E":
                by_year[yr]["Contingent Resources (Exploitation)"] += _to_mmscfd(
                    row.tpf_an
                )
            elif prefix == "X":
                by_year[yr]["Contingent Resources (Exploration)"] += _to_mmscfd(
                    row.tpf_an
                )
        elif cls == "Prospective Resources":
            by_year[yr]["Prospective Resources"] += _to_mmscfd(row.tpf_risked_an)

    sorted_years = sorted(years_set)
    result: dict[str, list[float]] = {}
    for cat in TIMESERIES_CATEGORIES:
        result[cat] = [by_year[y].get(cat, 0.0) for y in sorted_years]

    return sorted_years, result


def _normalize_class(raw: str | None) -> str:
    """Normalize raw project_class from nkri_timeseries."""
    if not raw:
        return ""
    if "Reserves" in raw:
        return "Reserves & GRR"
    if "Contingent" in raw or "Contigent" in raw:
        return "Contingent Resources"
    if "Prospective" in raw:
        return "Prospective Resources"
    return raw


# ── Chart generators ───────────────────────────────────────────


def nkri_timeseries_oc(
    data: list[NKRITimeseriesRow],
) -> go.Figure:
    """Stacked area chart: Oil + Condensate production rate forecast.

    Y-axis in MBOPD (thousand barrels of oil per day).
    Stacks: Reserves, Sales Potential, Contingent (Ex/En), Prospective.
    """
    years, values = _agg_timeseries_oc(data)
    if not years:
        return go.Figure()

    fig = go.Figure()
    fig.update_layout(
        yaxis_title="MBOPD",
        hovermode="x unified",
        colorway=[TIMESERIES_PALETTE_OC[c] for c in TIMESERIES_CATEGORIES],
        legend={
            "orientation": "h",
            "yanchor": "top",
            "y": -0.25,
            "xanchor": "center",
            "x": 0.5,
        },
        margin={"l": 50, "r": 30, "t": 30, "b": 80},
        **{
            k: v
            for k, v in PLOTLY_LAYOUT_DEFAULTS.items()
            if k not in ("margin", "legend")
        },
    )
    totals = [
        sum(values[c][i] for c in TIMESERIES_CATEGORIES) for i in range(len(years))
    ]
    fig.add_trace(
        go.Scatter(
            x=years,
            y=totals,
            mode="lines",
            line={"width": 0},
            showlegend=False,
            name="Total",
            hovertemplate="<b>Total</b>  %{y:,.1f} MBOPD<extra></extra>",
        )
    )
    for cat in TIMESERIES_CATEGORIES:
        fig.add_trace(
            go.Scatter(
                x=years,
                y=values[cat],
                mode="lines",
                stackgroup="one",
                name=cat,
                line={"width": 0},
                marker={"color": TIMESERIES_PALETTE_OC.get(cat, "#999")},
                hovertemplate=f"{cat}<br>%{{y:,.1f}} MBOPD<extra></extra>",
            )
        )
    return fig


def nkri_timeseries_an(
    data: list[NKRITimeseriesRow],
) -> go.Figure:
    """Stacked area chart: Assoc. + Non Assoc. Gas production rate forecast.

    Y-axis in MMSCFD (million standard cubic feet per day).
    Stacks: Reserves, Sales Potential, Contingent (Ex/En), Prospective.
    """
    years, values = _agg_timeseries_an(data)
    if not years:
        return go.Figure()

    fig = go.Figure()
    fig.update_layout(
        yaxis_title="MMSCFD",
        hovermode="x unified",
        colorway=[TIMESERIES_PALETTE_AN[c] for c in TIMESERIES_CATEGORIES],
        legend={
            "orientation": "h",
            "yanchor": "top",
            "y": -0.25,
            "xanchor": "center",
            "x": 0.5,
        },
        margin={"l": 50, "r": 30, "t": 30, "b": 80},
        **{
            k: v
            for k, v in PLOTLY_LAYOUT_DEFAULTS.items()
            if k not in ("margin", "legend")
        },
    )
    totals = [
        sum(values[c][i] for c in TIMESERIES_CATEGORIES) for i in range(len(years))
    ]
    fig.add_trace(
        go.Scatter(
            x=years,
            y=totals,
            mode="lines",
            line={"width": 0},
            showlegend=False,
            name="Total",
            hovertemplate="<b>Total</b>  %{y:,.1f} MMSCFD<extra></extra>",
        )
    )
    for cat in TIMESERIES_CATEGORIES:
        fig.add_trace(
            go.Scatter(
                x=years,
                y=values[cat],
                mode="lines",
                stackgroup="one",
                name=cat,
                line={"width": 0},
                marker={"color": TIMESERIES_PALETTE_AN.get(cat, "#999")},
                hovertemplate=f"{cat}<br>%{{y:,.1f}} MMSCFD<extra></extra>",
            )
        )
    return fig


def nkri_onstream_chart(
    data: list[OnstreamRow],
    report_year: int,
) -> go.Figure:
    """Stacked bar chart: Project counts by onstream year and class.

    Y-axis in number of projects.
    Stacks: Reserves, Contingent (Ex/En), Prospective.
    """
    by_year: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    years_set: set[int] = set()

    for row in data:
        yr = row.onstream_year
        years_set.add(yr)
        cls = row.project_class
        prefix = row.level_prefix

        if cls == "1. Reserves & GRR":
            by_year[yr]["Reserves"] += row.project_count
        elif cls == "2. Contingent Resources":
            if prefix == "E":
                by_year[yr]["Contingent Resources (Exploitation)"] += row.project_count
            elif prefix == "X":
                by_year[yr]["Contingent Resources (Exploration)"] += row.project_count
        elif cls == "3. Prospective Resources":
            by_year[yr]["Prospective Resources"] += row.project_count

    sorted_years = sorted(y for y in years_set if y >= report_year + 1)
    if not sorted_years:
        return go.Figure()

    fig = go.Figure()
    fig.update_layout(
        barmode="stack",
        yaxis_title="Count of Projects",
        hovermode="x unified",
        colorway=[ONSTREAM_PALETTE[c] for c in ONSTREAM_CATEGORIES],
        legend={
            "orientation": "h",
            "yanchor": "top",
            "y": -0.25,
            "xanchor": "center",
            "x": 0.5,
        },
        margin={"l": 50, "r": 30, "t": 30, "b": 80},
        **{
            k: v
            for k, v in PLOTLY_LAYOUT_DEFAULTS.items()
            if k not in ("margin", "legend")
        },
    )
    for cat in ONSTREAM_CATEGORIES:
        vals = [by_year[y].get(cat, 0) for y in sorted_years]
        fig.add_trace(
            go.Bar(
                x=sorted_years,
                y=vals,
                name=cat,
                marker={"color": ONSTREAM_PALETTE.get(cat, "#999")},
                hovertemplate=f"{cat}<br>%{{y}} projects<extra></extra>",
            )
        )
    return fig
