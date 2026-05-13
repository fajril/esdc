"""RE1 helper utilities for production and forecast validation rules."""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    pass

from esdc.validate.rule_re0_helpers import (
    IDENTIFIER_COLS,
    _uncert_value,
)
from esdc.validate.rules import TOLERANCE

# ---------------------------------------------------------------------------
# Identifier columns
# ---------------------------------------------------------------------------

TS_IDENTIFIER_COLS: list[str] = [
    "report_year",
    "project_name",
    "wk_name",
    "field_name",
    "year",
]

# ---------------------------------------------------------------------------
# SQL builders for RE1 rule categories
# ---------------------------------------------------------------------------


def build_monotonic_sql(
    validated_column: str,
    uncert: str,
    table: str = "project_resources",
    tolerance: float = TOLERANCE,
) -> str:
    """Build SQL for monotonic ordering: current cumprod >= previous cumprod.

    Self-joins `table` on `project_id` to compare the same project's
    cumulative production across consecutive report years
    (l.report_year = h.report_year + 1).  Both sides are pinned to the
    same `uncert_level`.  Violation when the previous year minus
    current year > tolerance.

    Returns rows where COALESCE(h.{column}, 0) - COALESCE(l.{column}, 0) > tolerance.
    """
    ident_l = ", ".join(f"l.{c}" for c in IDENTIFIER_COLS)
    return (
        f"SELECT {ident_l},"
        f" l.{validated_column} AS val_ref,"
        f" h.{validated_column} AS val_cmp"
        f" FROM {table} l"
        f" JOIN {table} h"
        f" ON l.project_id = h.project_id"
        f" AND l.report_year = h.report_year + 1"
        f" AND l.uncert_level = '{_uncert_value(uncert)}'"
        f" AND h.uncert_level = '{_uncert_value(uncert)}'"
        f" WHERE COALESCE(h.{validated_column}, 0)"
        f" - COALESCE(l.{validated_column}, 0) > {tolerance}"
    )


def build_timeseries_sales_le_tpf_sql(
    sales_col: str,
    tpf_col: str,
    table: str = "project_timeseries",
    tolerance: float = TOLERANCE,
) -> str:
    """Build SQL for timeseries: sales forecast <= total potential per year.

    Operates on `project_timeseries` (no uncert_level column).
    Returns rows where COALESCE(sales_col, 0) - COALESCE(tpf_col, 0) > tolerance.
    """
    return (
        f"SELECT {', '.join(TS_IDENTIFIER_COLS)},"
        f" {sales_col} AS val_ref,"
        f" {tpf_col} AS val_cmp"
        f" FROM {table}"
        f" WHERE COALESCE({sales_col}, 0) - COALESCE({tpf_col}, 0) > {tolerance}"
    )


def build_forecast_sum_equals_reserve_sql(
    slf_col: str,
    reserve_col: str,
    uncert: str,
    tolerance: float = TOLERANCE,
) -> str:
    """Build SQL for cross-table: sum of sales forecast = 2P reserves.

    Joins `project_timeseries` (forecast data) with `project_resources`
    (reserves at 2P) on `project_id` + `report_year`.  Only future forecast
    years are summed (ts.year > pr.report_year).

    Returns rows where |SUM(slf) - reserve| > tolerance.
    """
    return (
        f"SELECT pr.report_year, pr.project_name, pr.wk_name, pr.field_name,"
        f" SUM(COALESCE(ts.{slf_col}, 0)) AS val_sum,"
        f" pr.{reserve_col} AS val_ref"
        f" FROM project_resources pr"
        f" JOIN project_timeseries ts"
        f" ON pr.project_id = ts.project_id"
        f" AND pr.report_year = ts.report_year"
        f" AND ts.year > pr.report_year"
        f" WHERE pr.uncert_level = '{_uncert_value(uncert)}'"
        f" GROUP BY pr.report_year, pr.project_name, pr.wk_name,"
        f" pr.field_name, pr.{reserve_col}"
        f" HAVING ABS(SUM(COALESCE(ts.{slf_col}, 0))"
        f" - COALESCE(pr.{reserve_col}, 0)) > {tolerance}"
    )


def build_forecast_sum_equals_resource_sql(
    tpf_col: str,
    resource_col: str,
    uncert: str,
    tolerance: float = TOLERANCE,
) -> str:
    """Build SQL for cross-table: sum of TPF = 2R GRR/CR/PR (P50).

    Same pattern as `build_forecast_sum_equals_reserve_sql` but
    comparing total-potential forecast with recoverable resources.
    """
    return (
        f"SELECT pr.report_year, pr.project_name, pr.wk_name, pr.field_name,"
        f" SUM(COALESCE(ts.{tpf_col}, 0)) AS val_sum,"
        f" pr.{resource_col} AS val_ref"
        f" FROM project_resources pr"
        f" JOIN project_timeseries ts"
        f" ON pr.project_id = ts.project_id"
        f" AND pr.report_year = ts.report_year"
        f" AND ts.year > pr.report_year"
        f" WHERE pr.uncert_level = '{_uncert_value(uncert)}'"
        f" GROUP BY pr.report_year, pr.project_name, pr.wk_name,"
        f" pr.field_name, pr.{resource_col}"
        f" HAVING ABS(SUM(COALESCE(ts.{tpf_col}, 0))"
        f" - COALESCE(pr.{resource_col}, 0)) > {tolerance}"
    )
