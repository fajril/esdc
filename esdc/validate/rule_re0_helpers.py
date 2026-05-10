"""RE0 helper utilities for volumetric validation rules."""

from __future__ import annotations

import logging

import duckdb

from esdc.validate.rules import Violation

logger = logging.getLogger(__name__)

# Uncertainty level constants
UNCERT_LOW = "1. Low Value"
UNCERT_MID = "2. Middle Value"
UNCERT_HIGH = "3. High Value"

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

IDENTIFIER_COLS: list[str] = ["report_year", "project_name", "wk_name"]


def _add_year_filter(sql: str, year: list[int] | None) -> str:
    """Append year filter to SQL query."""
    if not year:
        return sql
    year_list = ", ".join(str(y) for y in year)
    if "WHERE" in sql:
        # Insert year filter right after WHERE
        return sql.replace("WHERE", f"WHERE report_year IN ({year_list}) AND", 1)
    return sql + f" WHERE report_year IN ({year_list})"


def build_non_negative_sql(
    column: str,
    uncert: str,
    table: str = "project_resources",
) -> str:
    """Build SQL for Category A: column >= 0 at given uncert_level.

    Returns rows where COALESCE(column, 0) < 0.
    """
    return (
        f"SELECT {', '.join(IDENTIFIER_COLS)}, {column}"
        f" FROM {table}"
        f" WHERE uncert_level = '{uncert}'"
        f" AND COALESCE({column}, 0) < 0"
    )


def build_ordering_sql(
    low_col: str,
    high_col: str,
    low_uncert: str,
    high_uncert: str,
    table: str = "project_resources",
) -> str:
    """Build SQL for Category B: low_col <= high_col across uncert_levels.

    Uses self-join to compare values at different uncertainty levels.
    Returns rows where COALESCE(low_col, 0) > COALESCE(high_col, 0).
    """
    ident_l = ", ".join(f"l.{c}" for c in IDENTIFIER_COLS)
    return (
        f"SELECT {ident_l}, l.{low_col} AS low_val, h.{high_col} AS high_val"
        f" FROM {table} l"
        f" JOIN {table} h"
        f" ON l.project_name = h.project_name"
        f" AND l.report_year = h.report_year"
        f" AND l.uncert_level = '{low_uncert}'"
        f" AND h.uncert_level = '{high_uncert}'"
        f" WHERE COALESCE(l.{low_col}, 0) > COALESCE(h.{high_col}, 0)"
    )


def build_same_row_ordering_sql(
    low_col: str,
    high_col: str,
    uncert: str,
    table: str = "project_resources",
) -> str:
    """Build SQL for same-row ordering: low_col <= high_col at same uncert_level.

    Used when both values are in the same row (e.g., res_oil vs rec_oil).
    Returns rows where COALESCE(low_col, 0) > COALESCE(high_col, 0).
    """
    return (
        f"SELECT {', '.join(IDENTIFIER_COLS)}, {low_col} AS low_val,"
        f" {high_col} AS high_val"
        f" FROM {table}"
        f" WHERE uncert_level = '{uncert}'"
        f" AND COALESCE({low_col}, 0) > COALESCE({high_col}, 0)"
    )


def build_implication_sql(
    cond_col: str,
    cond_uncert: str,
    result_col: str,
    result_uncert: str,
    table: str = "project_resources",
) -> str:
    """Build SQL for Category E: if cond_col > 0 then result_col > 0.

    Self-join: find rows where the condition is satisfied but the result is zero.
    """
    ident_h = ", ".join(f"h.{c}" for c in IDENTIFIER_COLS)
    return (
        f"SELECT {ident_h}, h.{cond_col} AS cond_val, l.{result_col} AS result_val"
        f" FROM {table} h"
        f" JOIN {table} l"
        f" ON h.project_name = l.project_name"
        f" AND h.report_year = l.report_year"
        f" AND h.uncert_level = '{cond_uncert}'"
        f" AND l.uncert_level = '{result_uncert}'"
        f" WHERE COALESCE(h.{cond_col}, 0) > 0"
        f" AND COALESCE(l.{result_col}, 0) = 0"
    )


def build_reserve_vs_place_sql(
    place_col: str,
    reserve_col: str,
    cumprod_col: str,
    uncert: str,
    table: str = "project_resources",
) -> str:
    """Build SQL for Category F: if place > 0, then reserve + cumprod < place.

    All values are in the same row (same uncert_level).
    Returns rows where place > 0 AND (reserve + cumprod) >= place.
    """
    return (
        f"SELECT {', '.join(IDENTIFIER_COLS)},"
        f" {place_col} AS place_val,"
        f" {reserve_col} AS reserve_val,"
        f" {cumprod_col} AS cumprod_val"
        f" FROM {table}"
        f" WHERE uncert_level = '{uncert}'"
        f" AND COALESCE({place_col}, 0) > 0"
        f" AND COALESCE({reserve_col}, 0) + COALESCE({cumprod_col}, 0)"
        f" >= COALESCE({place_col}, 0)"
    )


def _execute_and_build_violations(
    conn: duckdb.DuckDBPyConnection,
    sql: str,
    rule_id: str,
    description: str,
    severity: str,
    table: str,
    year: list[int] | None,
    extra_columns: list[str] | None = None,
) -> list[Violation]:
    """Execute SQL query and build Violation objects from results.

    This helper centralises the row-to-Violation conversion pattern
    used by all RE0 rule factories.
    """
    sql = _add_year_filter(sql, year)
    try:
        rows = conn.execute(sql).fetchall()
    except duckdb.Error:
        logger.exception("SQL execution failed for %s", rule_id)
        return []

    violations: list[Violation] = []
    for row in rows:
        identifiers = {
            IDENTIFIER_COLS[i]: str(row[i]) for i in range(len(IDENTIFIER_COLS))
        }
        current_values: dict[str, object] = {}

        # Extra columns start after the identifier columns
        extra_cols = extra_columns or []
        base = len(IDENTIFIER_COLS)
        for i, col_name in enumerate(extra_cols):
            current_values[col_name] = row[base + i]

        violations.append(
            Violation(
                rule_id=rule_id,
                rule_group="RE0",
                description=description,
                severity=severity,
                table=table,
                identifiers=identifiers,
                current_values=current_values,
            )
        )
    return violations
