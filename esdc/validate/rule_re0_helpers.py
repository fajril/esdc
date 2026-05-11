"""RE0 helper utilities for volumetric validation rules."""

from __future__ import annotations

import logging
from enum import Enum

import duckdb

from esdc.selection import Severity
from esdc.validate.rules import TOLERANCE, Violation

logger = logging.getLogger(__name__)


class UncertLevel(str, Enum):
    LOW = "1. Low Value"
    MID = "2. Middle Value"
    HIGH = "3. High Value"


def _uncert_value(uncert: UncertLevel | str) -> str:
    return uncert.value if isinstance(uncert, UncertLevel) else uncert


# Column name mapping for rule definitions
VOL_COLUMNS = {
    "ioip": "prj_ioip",
    "igip": "prj_igip",
    "res_oil": "res_oil",
    "res_con": "res_con",
    "res_ga": "res_ga",
    "res_gn": "res_gn",
    "rec_oil": "rec_oil",
    "rec_con": "rec_con",
    "rec_ga": "rec_ga",
    "rec_gn": "rec_gn",
    "cprd_oil": "cprd_grs_oil",
    "cprd_con": "cprd_grs_con",
    "cprd_ga": "cprd_grs_ga",
    "cprd_gn": "cprd_grs_gn",
}

IDENTIFIER_COLS: list[str] = [
    "report_year",
    "project_name",
    "wk_name",
    "field_name",
]

FIELD_IDENTIFIER_COLS: list[str] = [
    "report_year",
    "wk_name",
    "field_name",
]

AGGREGATION_CONSISTENCY_IDENTIFIER_COLS: list[str] = [
    "report_year",
    "wk_name",
    "field_name",
    "project_stage",
    "project_class",
]


def _add_year_filter(sql: str, year: list[int] | None) -> str:
    """Append year filter to SQL query.

    For self-join queries (e.g., project_resources l JOIN project_resources h),
    the year filter is added to the ON clause using a qualified column name.
    For CTE-based queries (field-level aggregation), the year filter is added
    inside the CTE WHERE clause.
    For simple queries, the year filter is added to the WHERE clause.
    For cross-table joins (e.g., field_resources fr JOIN project_resources pr),
    the year filter uses the fr alias.
    """
    if not year:
        return sql
    year_list = ", ".join(str(y) for y in year)
    is_self_join = " JOIN " in sql and ("l." in sql or "h." in sql)
    if is_self_join:
        return sql.replace(
            "AND h.uncert_level",
            f"AND l.report_year IN ({year_list}) AND h.uncert_level",
            1,
        )
    is_pr_ts_join = "pr." in sql and "ts." in sql
    if is_pr_ts_join:
        return sql.replace(
            "WHERE pr.uncert_level",
            f"WHERE pr.report_year IN ({year_list}) AND pr.uncert_level",
            1,
        )
    is_cross_join = "fr." in sql and "pr." in sql
    if is_cross_join:
        return sql.replace(
            "WHERE fr.uncert_level",
            f"WHERE fr.report_year IN ({year_list}) AND fr.uncert_level",
            1,
        )
    if "WHERE" in sql:
        return sql.replace("WHERE", f"WHERE report_year IN ({year_list}) AND", 1)
    return sql + f" WHERE report_year IN ({year_list})"


def _add_year_filter_cte(sql: str, year: list[int] | None) -> str:
    """Add year filter inside CTE for field-level aggregation queries.

    Inserts the year filter into the CTE's WHERE clause (before GROUP BY).
    """
    if not year:
        return sql
    year_list = ", ".join(str(y) for y in year)
    if "WHERE" in sql:
        return sql.replace("WHERE", f"WHERE report_year IN ({year_list}) AND", 1)
    if "GROUP BY" in sql:
        return sql.replace(
            "GROUP BY",
            f"WHERE report_year IN ({year_list}) GROUP BY",
            1,
        )
    return sql


# ---------------------------------------------------------------------------
# Project-level SQL builders (project_resources table)
# ---------------------------------------------------------------------------


def build_non_negative_sql(
    validated_column: str,
    uncert: UncertLevel | str,
    table: str = "project_resources",
    tolerance: float = TOLERANCE,
) -> str:
    """Build SQL for Category A: column >= 0 at given uncert_level.

    Returns rows where COALESCE(column, 0) < -tolerance.
    Values between -tolerance and 0 are treated as zero (floating point noise).
    """
    return (
        f"SELECT {', '.join(IDENTIFIER_COLS)}, {validated_column}"
        f" FROM {table}"
        f" WHERE uncert_level = '{_uncert_value(uncert)}'"
        f" AND COALESCE({validated_column}, 0) < -{tolerance}"
    )


def build_ordering_sql(
    validated_column: str,
    compared_column: str,
    low_uncert: UncertLevel | str,
    high_uncert: UncertLevel | str,
    table: str = "project_resources",
    tolerance: float = TOLERANCE,
) -> str:
    """Build SQL for Category B: validated_column <= compared_column.

    Compares values at different uncertainty levels.
    Uses self-join to compare values at different uncertainty levels.
    Joins on project_id (which uniquely identifies a project-wk combination)
    to avoid cross-product false violations from shared project names.
    Returns rows where COALESCE(validated_column, 0)
    - COALESCE(compared_column, 0) > tolerance.
    """
    ident_l = ", ".join(f"l.{c}" for c in IDENTIFIER_COLS)
    return (
        f"SELECT {ident_l},"
        f" l.{validated_column} AS val_ref,"
        f" h.{compared_column} AS val_cmp"
        f" FROM {table} l"
        f" JOIN {table} h"
        f" ON l.project_id = h.project_id"
        f" AND l.report_year = h.report_year"
        f" AND l.uncert_level = '{_uncert_value(low_uncert)}'"
        f" AND h.uncert_level = '{_uncert_value(high_uncert)}'"
        f" WHERE COALESCE(l.{validated_column}, 0)"
        f" - COALESCE(h.{compared_column}, 0) > {tolerance}"
    )


def build_same_row_ordering_sql(
    validated_column: str,
    compared_column: str,
    uncert: UncertLevel | str,
    table: str = "project_resources",
    tolerance: float = TOLERANCE,
) -> str:
    """Build SQL for same-row ordering: validated_column <= compared_column.

    Used when both values are in the same row (e.g., res_oil vs rec_oil).
    Returns rows where COALESCE(validated_column, 0)
    - COALESCE(compared_column, 0) > tolerance.
    """
    return (
        f"SELECT {', '.join(IDENTIFIER_COLS)},"
        f" {validated_column} AS val_ref,"
        f" {compared_column} AS val_cmp"
        f" FROM {table}"
        f" WHERE uncert_level = '{_uncert_value(uncert)}'"
        f" AND COALESCE({validated_column}, 0)"
        f" - COALESCE({compared_column}, 0) > {tolerance}"
    )


def build_implication_sql(
    validated_column: str,
    cond_uncert: UncertLevel | str,
    compared_column: str,
    result_uncert: UncertLevel | str,
    table: str = "project_resources",
) -> str:
    """Build SQL for Category E: if validated_column > 0 then compared_column > 0.

    Self-join: find rows where the condition is satisfied but the result is zero.
    Joins on project_id to avoid cross-product false violations.
    """
    ident_h = ", ".join(f"h.{c}" for c in IDENTIFIER_COLS)
    return (
        f"SELECT {ident_h},"
        f" h.{validated_column} AS val_ref,"
        f" l.{compared_column} AS val_cmp"
        f" FROM {table} h"
        f" JOIN {table} l"
        f" ON h.project_id = l.project_id"
        f" AND h.report_year = l.report_year"
        f" AND h.uncert_level = '{_uncert_value(cond_uncert)}'"
        f" AND l.uncert_level = '{_uncert_value(result_uncert)}'"
        f" WHERE COALESCE(h.{validated_column}, 0) > 0"
        f" AND COALESCE(l.{compared_column}, 0) = 0"
    )


def build_zero_implication_sql(
    validated_column: str,
    cond_uncert: UncertLevel | str,
    compared_column: str,
    result_uncert: UncertLevel | str,
    table: str = "project_resources",
) -> str:
    """Build SQL for Category G: if validated_column=0 then compared_column=0.

    Self-join: find rows where condition column is zero but result column
    is non-zero. Joins on project_id to avoid cross-product false violations.
    """
    ident_h = ", ".join(f"h.{c}" for c in IDENTIFIER_COLS)
    return (
        f"SELECT {ident_h},"
        f" h.{validated_column} AS val_ref,"
        f" l.{compared_column} AS val_cmp"
        f" FROM {table} h"
        f" JOIN {table} l"
        f" ON h.project_id = l.project_id"
        f" AND h.report_year = l.report_year"
        f" AND h.uncert_level = '{_uncert_value(cond_uncert)}'"
        f" AND l.uncert_level = '{_uncert_value(result_uncert)}'"
        f" WHERE COALESCE(h.{validated_column}, 0) = 0"
        f" AND COALESCE(l.{compared_column}, 0) != 0"
    )


def build_reserve_vs_place_sql(
    validated_column: str,
    reserve_col: str,
    cumprod_col: str,
    uncert: UncertLevel | str,
    table: str = "project_resources",
    tolerance: float = TOLERANCE,
) -> str:
    """Build SQL for Category F: if place > 0, then reserve + cumprod < place.

    All values are in the same row (same uncert_level).
    Returns rows where place > 0 AND place - (reserve + cumprod) < tolerance.
    """
    return (
        f"SELECT {', '.join(IDENTIFIER_COLS)},"
        f" {validated_column} AS val_ref,"
        f" {reserve_col} AS val_cmp,"
        f" {cumprod_col} AS val_sum"
        f" FROM {table}"
        f" WHERE uncert_level = '{_uncert_value(uncert)}'"
        f" AND COALESCE({validated_column}, 0) > 0"
        f" AND COALESCE({validated_column}, 0) - COALESCE({reserve_col}, 0)"
        f" - COALESCE({cumprod_col}, 0) < {tolerance}"
    )


# ---------------------------------------------------------------------------
# Field-level SQL builders (field_resources table)
# ---------------------------------------------------------------------------


def build_field_non_negative_sql(
    validated_column: str,
    uncert: UncertLevel | str,
    table: str = "field_resources",
    tolerance: float = TOLERANCE,
) -> str:
    """Build SQL for field-level: SUM(column) per field >= 0 at given uncert_level.

    Aggregates column values across all projects within each field,
    then returns fields where the total is < -tolerance.
    """
    ident = ", ".join(FIELD_IDENTIFIER_COLS)
    return (
        f"SELECT {ident},"
        f" SUM(COALESCE({validated_column}, 0)) AS val_ref"
        f" FROM {table}"
        f" WHERE uncert_level = '{_uncert_value(uncert)}'"
        f" GROUP BY {ident}, uncert_level"
        f" HAVING SUM(COALESCE({validated_column}, 0)) < -{tolerance}"
    )


def build_field_ordering_sql(
    validated_column: str,
    compared_column: str,
    low_uncert: UncertLevel | str,
    high_uncert: UncertLevel | str,
    table: str = "field_resources",
    tolerance: float = TOLERANCE,
) -> str:
    """Build SQL for field-level ordering: SUM(validated_col) <= SUM(compared_col).

    Uses CTE to aggregate values per (wk_name, field_name, uncert_level),
    then self-joins to compare across uncertainty levels.
    Returns rows where validated - compared > tolerance.
    """
    ident = ", ".join(FIELD_IDENTIFIER_COLS)
    ident_l = ", ".join(f"l.{c}" for c in FIELD_IDENTIFIER_COLS)
    low_uv = _uncert_value(low_uncert)
    high_uv = _uncert_value(high_uncert)
    return (
        f"WITH field_agg AS ("
        f" SELECT {ident}, uncert_level,"
        f" SUM(COALESCE({validated_column}, 0)) AS {validated_column},"
        f" SUM(COALESCE({compared_column}, 0)) AS {compared_column}"
        f" FROM {table}"
        f" GROUP BY {ident}, uncert_level"
        f")"
        f" SELECT {ident_l},"
        f" l.{validated_column} AS val_ref,"
        f" h.{compared_column} AS val_cmp"
        f" FROM field_agg l"
        f" JOIN field_agg h"
        f" ON l.wk_name = h.wk_name"
        f" AND l.field_name = h.field_name"
        f" AND l.report_year = h.report_year"
        f" AND l.uncert_level = '{low_uv}'"
        f" AND h.uncert_level = '{high_uv}'"
        f" WHERE l.{validated_column} - h.{compared_column} > {tolerance}"
    )


# ---------------------------------------------------------------------------
# Aggregation consistency SQL builder (cross-table)
# ---------------------------------------------------------------------------


def build_aggregation_consistency_sql(
    validated_column: str,
    compared_column: str,
    uncert: UncertLevel | str,
    tolerance: float = TOLERANCE,
) -> str:
    """Build SQL for Category D: SUM(compared_column) must equal validated_column.

    Joins field_resources with project_resources on the GROUP BY keys
    (wk_id, field_id, report_year, project_stage, project_class, uncert_level)
    and checks that the sum of project-level values matches the field-level
    aggregated value within the given tolerance.

    Returns rows where ABS(SUM(compared_column) - validated_column) > tolerance.
    """
    ident = ", ".join(f"fr.{c}" for c in AGGREGATION_CONSISTENCY_IDENTIFIER_COLS)
    group_ident = ", ".join(f"fr.{c}" for c in AGGREGATION_CONSISTENCY_IDENTIFIER_COLS)
    return (
        f"SELECT {ident},"
        f" SUM(pr.{compared_column}) AS val_sum,"
        f" fr.{validated_column} AS val_field"
        f" FROM field_resources fr"
        f" JOIN project_resources pr"
        f" ON fr.wk_id = pr.wk_id"
        f" AND fr.field_id = pr.field_id"
        f" AND fr.report_year = pr.report_year"
        f" AND fr.project_stage = pr.project_stage"
        f" AND fr.project_class = pr.project_class"
        f" AND fr.uncert_level = pr.uncert_level"
        f" WHERE fr.uncert_level = '{_uncert_value(uncert)}'"
        f" GROUP BY {group_ident}, fr.{validated_column}"
        f" HAVING ABS(SUM(pr.{compared_column}) - fr.{validated_column}) > {tolerance}"
    )


# ---------------------------------------------------------------------------
# Violation builder
# ---------------------------------------------------------------------------


def _execute_and_build_violations(
    conn: duckdb.DuckDBPyConnection,
    sql: str,
    rule_id: str,
    description: str,
    severity: Severity,
    table: str,
    year: list[int] | None,
    extra_columns: list[str] | None = None,
    identifier_cols: list[str] | None = None,
    validated_column: str = "",
    compared_columns: list[str] | None = None,
    rule_group: str = "RE0",
) -> list[Violation]:
    """Execute SQL query and build Violation objects from results.

    This helper centralises the row-to-Violation conversion pattern
    used by all RE0 rule factories.
    """
    id_cols = identifier_cols or IDENTIFIER_COLS
    sql = _add_year_filter(sql, year)
    try:
        rows = conn.execute(sql).fetchall()
    except duckdb.Error:
        logger.exception("SQL execution failed for %s", rule_id)
        return []

    violations: list[Violation] = []
    for row in rows:
        identifiers = {id_cols[i]: str(row[i]) for i in range(len(id_cols))}
        current_values: dict[str, object] = {}

        extra_cols = extra_columns or []
        base = len(id_cols)
        for i, col_name in enumerate(extra_cols):
            current_values[col_name] = row[base + i]

        violations.append(
            Violation(
                rule_id=rule_id,
                rule_group=rule_group,
                description=description,
                severity=severity,
                table=table,
                validated_column=validated_column,
                compared_columns=compared_columns or [],
                identifiers=identifiers,
                current_values=current_values,
            ),
        )
    return violations
