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
            return "Contingent Exploit."
        if stage_norm == "Exploration":
            return "Contingent Explore."
        return "Contingent"
    return class_norm


def nkri_volumetric_bar(resources: list[NKRIResourcesRow]) -> go.Figure:
    """Grouped bar chart: Resources by Classification (OC & AN).

    Shows 2P/2C/2U best estimate volumes grouped by resource class,
    with OC and AN side by side.
    """
    # Filter to best estimate (2. Middle Value)
    best = [r for r in resources if r.uncert_level == "2. Middle Value"]
    if not best:
        best = resources

    # Group by class+stage
    groups: dict[str, dict[str, float]] = {}
    for r in best:
        label = _class_label(r.project_class_norm, r.project_stage_norm)
        if label not in groups:
            groups[label] = {"oc": 0, "an": 0}
        groups[label]["oc"] += r.rec_oc or 0
        groups[label]["an"] += r.rec_an or 0

    # Ordered categories
    class_order = [
        "Reserves & GRR",
        "Contingent Exploit.",
        "Contingent Explore.",
        "Prospective Resources",
    ]
    labels = [lbl for lbl in class_order if lbl in groups]

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
    """Stacked bar chart: Year-over-Year comparison for 2P/2C/2U.

    Shows delta values (current year minus previous year) by class for OC and AN.
    """
    sorted_years = sorted(available_years)
    if year not in sorted_years:
        return go.Figure()

    # For now, show current year resource volumes by class.
    # YoY deltas will be added in a future iteration.
    curr_data = get_nkri_resources(year)

    # Best estimate only
    curr_best = [r for r in curr_data if r.uncert_level == "2. Middle Value"]
    if not curr_best:
        curr_best = curr_data

    categories = ["Reserves & GRR", "Contingent Resources", "Prospective Resources"]

    # Current year values
    curr_oc: dict[str, float] = {}
    curr_an: dict[str, float] = {}
    for r in curr_best:
        label = r.project_class_norm
        if label in categories:
            curr_oc.setdefault(label, 0)
            curr_an.setdefault(label, 0)
            curr_oc[label] += r.rec_oc or 0
            curr_an[label] += r.rec_an or 0

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name=f"{year} OC",
            x=categories,
            y=[_to_mmstb(curr_oc.get(c, 0)) for c in categories],
            marker_color=FLUID_OC,
        )
    )
    fig.add_trace(
        go.Bar(
            name=f"{year} AN",
            x=categories,
            y=[_to_tscf(curr_an.get(c, 0)) for c in categories],
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
    """Sunburst chart: Hierarchical view Reserves → Contingent → Prospective.

    Inner ring: resource class. Outer ring: OC/AN split.
    """
    best = [r for r in resources if r.uncert_level == "2. Middle Value"]
    if not best:
        best = resources

    # Build sunburst data
    labels: list[str] = ["Total"]
    parents: list[str] = [""]
    values: list[float] = [0.0]
    colors: list[str] = [""]

    class_order = ["Reserves & GRR", "Contingent Resources", "Prospective Resources"]

    for cls in class_order:
        cls_rows = [r for r in best if r.project_class_norm == cls]
        if not cls_rows:
            continue

        oc_val = _to_mmstb(sum(r.rec_oc or 0 for r in cls_rows))
        an_val = _to_tscf(sum(r.rec_an or 0 for r in cls_rows))

        # Class node
        labels.append(cls)
        parents.append("Total")
        values.append(oc_val + an_val)
        colors.append(CLASS_PALETTE.get(cls, "#999"))

        # OC sub-node
        labels.append(f"{cls}<br>Oil+Cond")
        parents.append(cls)
        values.append(oc_val)
        colors.append(FLUID_OC)

        # AN sub-node
        labels.append(f"{cls}<br>Gas+NGL")
        parents.append(cls)
        values.append(an_val)
        colors.append(FLUID_AN)

    # Update total
    total_values = []
    for idx_pos, (_lbl, val) in enumerate(zip(labels, values, strict=False)):
        if parents[idx_pos] == "Total":
            total_values.append(val)
    values[0] = sum(total_values)

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
