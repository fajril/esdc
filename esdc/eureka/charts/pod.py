"""POD-level Plotly figure generators for the Eureka dashboard."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable

import plotly.graph_objects as go

from esdc.eureka.charts.colors import PLOTLY_LAYOUT_DEFAULTS
from esdc.eureka.queries import PodTimeseriesRow

POD_CATEGORIES: list[str] = [
    "Reserves",
    "Contingent Exploitation",
    "Contingent Exploration",
    "Prospective",
]

POD_PALETTE_OC: dict[str, str] = {
    "Reserves": "#2EA043",
    "Contingent Exploitation": "#7EE787",
    "Contingent Exploration": "#A5D6A7",
    "Prospective": "#C5E1A5",
}

POD_PALETTE_AN: dict[str, str] = {
    "Reserves": "#DA3633",
    "Contingent Exploitation": "#FF7B72",
    "Contingent Exploration": "#FFA198",
    "Prospective": "#FFC1B6",
}


def _to_mbopd(mstby: float) -> float:
    return round(mstby / 365, 1) if mstby else 0.0


def _to_mmscfd(bscfy: float) -> float:
    return round(bscfy * 1000 / 365, 1) if bscfy else 0.0


def _agg_pod_timeseries(
    data: list[PodTimeseriesRow],
    field: str,
    converter: Callable[[float], float],
) -> tuple[list[int], dict[str, list[float]]]:
    by_year: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    years_set: set[int] = set()

    for row in data:
        yr = row.year
        years_set.add(yr)
        cat = row.category
        if cat not in POD_CATEGORIES:
            continue
        by_year[yr][cat] += converter(getattr(row, field))

    sorted_years = sorted(years_set)
    result: dict[str, list[float]] = {}
    for cat in POD_CATEGORIES:
        result[cat] = [by_year[y].get(cat, 0.0) for y in sorted_years]

    return sorted_years, result


def pod_timeseries_oc(data: list[PodTimeseriesRow]) -> go.Figure:
    years, values = _agg_pod_timeseries(data, "oc_mstby", _to_mbopd)
    if not years:
        return go.Figure()

    fig = go.Figure()
    fig.update_layout(
        yaxis_title="MBOPD",
        hovermode="x unified",
        colorway=[POD_PALETTE_OC[c] for c in POD_CATEGORIES],
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
    totals = [sum(values[c][i] for c in POD_CATEGORIES) for i in range(len(years))]
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
    for cat in POD_CATEGORIES:
        fig.add_trace(
            go.Scatter(
                x=years,
                y=values[cat],
                mode="lines",
                stackgroup="one",
                name=cat,
                line={"width": 0},
                marker={"color": POD_PALETTE_OC.get(cat, "#999")},
                hovertemplate=f"{cat}<br>%{{y:,.1f}} MBOPD<extra></extra>",
            )
        )
    return fig


def pod_timeseries_an(data: list[PodTimeseriesRow]) -> go.Figure:
    years, values = _agg_pod_timeseries(data, "an_bscfy", _to_mmscfd)
    if not years:
        return go.Figure()

    fig = go.Figure()
    fig.update_layout(
        yaxis_title="MMSCFD",
        hovermode="x unified",
        colorway=[POD_PALETTE_AN[c] for c in POD_CATEGORIES],
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
    totals = [sum(values[c][i] for c in POD_CATEGORIES) for i in range(len(years))]
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
    for cat in POD_CATEGORIES:
        fig.add_trace(
            go.Scatter(
                x=years,
                y=values[cat],
                mode="lines",
                stackgroup="one",
                name=cat,
                line={"width": 0},
                marker={"color": POD_PALETTE_AN.get(cat, "#999")},
                hovertemplate=f"{cat}<br>%{{y:,.1f}} MMSCFD<extra></extra>",
            )
        )
    return fig
