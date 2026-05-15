"""NKRI-level Plotly figure generators for the Eureka dashboard."""

from __future__ import annotations

import plotly.graph_objects as go

from esdc.eureka.charts.colors import (
    CLASS_PALETTE,
    FLUID_AN,
    FLUID_OC,
    PLOTLY_LAYOUT_DEFAULTS,
)
from esdc.eureka.queries import NKRIResourcesRow, get_nkri_resources


def _to_mmstb(val: float) -> float:
    """Convert MSTB → MMSTB (divide by 1000), rounded to 0 decimal."""
    if val is None:
        return 0.0
    return round(val / 1000, 0)


def _to_tscf(val: float) -> float:
    """Convert MSCF → TSCF (divide by 1000), rounded to 0 decimal."""
    if val is None:
        return 0.0
    return round(val / 1000, 0)


def _class_label(class_norm: str, stage_norm: str) -> str:
    """Get display label for resource class."""
    if class_norm == "Contingent Resources":
        if stage_norm == "Exploitation":
            return "Contingent Resources (Exploitation)"
        if stage_norm == "Exploration":
            return "Contingent Resources (Exploration)"
        return "Contingent Resources"
    return class_norm


def _split_reserves_grr(
    resources: list[NKRIResourcesRow],
) -> dict[str, dict[str, float]]:
    """Split 'Reserves & GRR' rows into separate Reserves and GRR entries."""
    groups: dict[str, dict[str, float]] = {}
    for r in resources:
        if r.project_class_norm == "Reserves & GRR":
            # Split into Reserves (res_*) and additional GRR (rec_* - res_*)
            res_oc = r.res_oc or 0
            res_an = r.res_an or 0
            rec_oc = r.rec_oc or 0
            rec_an = r.rec_an or 0
            grr_oc = max(0, rec_oc - res_oc)
            grr_an = max(0, rec_an - res_an)
            groups.setdefault("Reserves", {"oc": 0, "an": 0})
            groups.setdefault("GRR", {"oc": 0, "an": 0})
            groups["Reserves"]["oc"] += res_oc
            groups["Reserves"]["an"] += res_an
            groups["GRR"]["oc"] += grr_oc
            groups["GRR"]["an"] += grr_an
        else:
            label = _class_label(r.project_class_norm, r.project_stage_norm)
            groups.setdefault(label, {"oc": 0, "an": 0})
            groups[label]["oc"] += r.rec_oc or 0
            groups[label]["an"] += r.rec_an or 0
    return groups


CLASS_ORDER = [
    "Reserves",
    "GRR",
    "Contingent Resources (Exploitation)",
    "Contingent Resources (Exploration)",
    "Prospective Resources",
]


def nkri_volumetric_bar(resources: list[NKRIResourcesRow]) -> go.Figure:
    """Grouped bar chart: Resources by Classification (OC & AN).

    Shows 2P/2C/2U best estimate volumes grouped by resource class,
    with OC and AN side by side.
    """
    best = [r for r in resources if r.uncert_level == "2. Middle Value"]
    if not best:
        best = resources

    groups = _split_reserves_grr(best)
    labels = [lbl for lbl in CLASS_ORDER if lbl in groups]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="Oil+Condensate",
            x=labels,
            y=[_to_mmstb(groups[lbl]["oc"]) for lbl in labels],
            marker_color=FLUID_OC,
            hovertemplate="%{x}: %{y:,.0f} MMSTB<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            name="Gas+NGL",
            x=labels,
            y=[_to_tscf(groups[lbl]["an"]) for lbl in labels],
            marker_color=FLUID_AN,
            hovertemplate="%{x}: %{y:,.0f} TSCF<extra></extra>",
        )
    )

    fig.update_layout(
        title="Resources by Classification (Best Estimate)",
        barmode="group",
        yaxis_title="Volume",
        **PLOTLY_LAYOUT_DEFAULTS,
    )
    return fig


def nkri_resource_donut(resources: list[NKRIResourcesRow]) -> go.Figure:
    """Donut chart: Resource Split by fluid type (OC vs AN).

    Shows proportion of Oil+Condensate vs Gas+NGL across all resource classes.
    """
    best = [r for r in resources if r.uncert_level == "2. Middle Value"]
    if not best:
        best = resources

    total_oc = sum(r.rec_oc or 0 for r in best)
    total_an = sum(r.rec_an or 0 for r in best)

    fig = go.Figure(
        data=[
            go.Pie(
                labels=["Oil+Condensate", "Gas+NGL"],
                values=[_to_mmstb(total_oc), _to_tscf(total_an)],
                marker_colors=[FLUID_OC, FLUID_AN],
                hole=0.55,
                textinfo="label+percent",
                hovertemplate="%{label}<br>%{value:,.0f}<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        title="Resource Split (Best Estimate)",
        **PLOTLY_LAYOUT_DEFAULTS,
    )
    return fig


def nkri_yoy_comparison(year: int, available_years: list[int]) -> go.Figure:
    """Grouped bar chart: Year-over-Year comparison for all resource classes.

    Shows current year volumes by class for OC and AN.
    """
    sorted_years = sorted(available_years)
    if year not in sorted_years:
        return go.Figure()

    curr_data = get_nkri_resources(year)
    curr_best = [r for r in curr_data if r.uncert_level == "2. Middle Value"]
    if not curr_best:
        curr_best = curr_data

    groups = _split_reserves_grr(curr_best)
    labels = [lbl for lbl in CLASS_ORDER if lbl in groups]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name=f"{year} Oil+Condensate",
            x=labels,
            y=[_to_mmstb(groups[lbl]["oc"]) for lbl in labels],
            marker_color=FLUID_OC,
        )
    )
    fig.add_trace(
        go.Bar(
            name=f"{year} Gas+NGL",
            x=labels,
            y=[_to_tscf(groups[lbl]["an"]) for lbl in labels],
            marker_color=FLUID_AN,
        )
    )

    fig.update_layout(
        title=f"Resources by Class — {year}",
        barmode="group",
        yaxis_title="Volume",
        **PLOTLY_LAYOUT_DEFAULTS,
    )
    return fig


def nkri_hierarchy_sunburst(resources: list[NKRIResourcesRow]) -> go.Figure:
    """Sunburst chart: Hierarchical view of resource classes.

    Inner ring: resource class. Outer ring: OC/AN split.
    """
    best = [r for r in resources if r.uncert_level == "2. Middle Value"]
    if not best:
        best = resources

    groups = _split_reserves_grr(best)

    labels: list[str] = ["Total"]
    parents: list[str] = [""]
    values: list[float] = [0.0]
    colors: list[str] = [""]

    for cls in CLASS_ORDER:
        if cls not in groups:
            continue
        oc_val = _to_mmstb(groups[cls]["oc"])
        an_val = _to_tscf(groups[cls]["an"])

        labels.append(cls)
        parents.append("Total")
        values.append(oc_val + an_val)
        colors.append(CLASS_PALETTE.get(cls, "#999"))

        labels.append(f"{cls}<br>Oil+Cond")
        parents.append(cls)
        values.append(oc_val)
        colors.append(FLUID_OC)

        labels.append(f"{cls}<br>Gas+NGL")
        parents.append(cls)
        values.append(an_val)
        colors.append(FLUID_AN)

    total_vals = [v for i, v in enumerate(values) if parents[i] == "Total"]
    values[0] = sum(total_vals)

    fig = go.Figure(
        go.Sunburst(
            labels=labels,
            parents=parents,
            values=values,
            branchvalues="total",
            marker={"colors": colors},
            hovertemplate="%{label}<br>%{value:,.0f}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Resource Hierarchy",
        **PLOTLY_LAYOUT_DEFAULTS,
    )
    return fig
