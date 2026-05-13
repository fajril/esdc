"""RE2 helper utilities for material balance validation rules."""

from __future__ import annotations

from esdc.validate.rule_re0_helpers import (
    IDENTIFIER_COLS,
    _uncert_value,
)
from esdc.validate.rules import TOLERANCE

AGG_IDENT = "fr_result.wk_id, fr_result.field_id, fr_result.report_year"
AGG_SELECT = (
    "MIN(fr_result.report_year) AS report_year,"
    " MIN(fr_result.wk_name) AS wk_name,"
    " MIN(fr_result.field_name) AS field_name"
)


def build_material_balance_sql(
    validated_column: str,
    dcpy_columns: list[str],
    production_column: str,
    uncert: str,
    tolerance: float = TOLERANCE,
) -> str:
    """Build SQL for material balance validation.

    Checks: current = previous + discrepancies - delta_production.
    Self-joins project_resources to compare current vs previous year.

    val_cmp is the expected value: prev + sum(dcpy) - delta_production.
    Violation when ABS(curr - val_cmp) > tolerance.

    The WHERE clause joins on project_id and report_year offset, with both
    sides pinned to the same uncert_level.
    """
    ident_curr = ", ".join(f"curr.{c}" for c in IDENTIFIER_COLS)
    dcpy_add = " + ".join(f"COALESCE(curr.{c}, 0)" for c in dcpy_columns)
    return (
        f"SELECT {ident_curr},"
        f" COALESCE(curr.{validated_column}, 0) AS val_ref,"
        f" COALESCE(prev.{validated_column}, 0)"
        f" + ({dcpy_add})"
        f" - COALESCE(curr.{production_column}, 0)"
        f" + COALESCE(prev.{production_column}, 0) AS val_cmp"
        f" FROM project_resources curr"
        f" JOIN project_resources prev"
        f" ON curr.project_id = prev.project_id"
        f" AND curr.report_year = prev.report_year + 1"
        f" AND prev.uncert_level = '{_uncert_value(uncert)}'"
        f" WHERE curr.uncert_level = '{_uncert_value(uncert)}'"
        f" AND ABS("
        f"COALESCE(curr.{validated_column}, 0)"
        f" - COALESCE(prev.{validated_column}, 0)"
        f" - ({dcpy_add})"
        f" + COALESCE(curr.{production_column}, 0)"
        f" - COALESCE(prev.{production_column}, 0)"
        f") > {tolerance}"
    )


def build_eur_bounds_sql(
    field_column: str,
    eur_rec_column: str,
    eur_cprd_column: str,
    cond_uncert: str,
    result_uncert: str,
    tolerance: float = TOLERANCE,
) -> str:
    """Build SQL for RE2025/RE2026: total inplace > total EUR when EUR > 0.

    Aggregates across ALL project_class and project_stage per field.
    EUR = SUM(rec) + SUM(cprd_sls) computed across the entire field.
    Inplace = SUM(field_column) across the entire field.
    Violation when EUR > 0 AND inplace <= EUR.
    """
    eur_cond = f"(SUM(fr_cond.{eur_rec_column}) + SUM(fr_cond.{eur_cprd_column}))"
    inplace_result = f"SUM(fr_result.{field_column})"
    return (
        f"SELECT {AGG_SELECT},"
        f" {inplace_result} AS val_ref,"
        f" {eur_cond} AS val_cmp"
        f" FROM field_resources fr_result"
        f" JOIN field_resources fr_cond"
        f" ON fr_result.wk_id = fr_cond.wk_id"
        f" AND fr_result.field_id = fr_cond.field_id"
        f" AND fr_result.report_year = fr_cond.report_year"
        f" WHERE fr_result.uncert_level = '{_uncert_value(result_uncert)}'"
        f" AND fr_cond.uncert_level = '{_uncert_value(cond_uncert)}'"
        f" GROUP BY {AGG_IDENT}"
        f" HAVING {eur_cond} > {tolerance}"
        f" AND {inplace_result} - {eur_cond} < {tolerance}"
    )


def build_eur_implication_sql(
    field_column: str,
    eur_rec_column: str,
    eur_cprd_column: str,
    cond_uncert: str,
    result_uncert: str,
    tolerance: float = TOLERANCE,
) -> str:
    """Build SQL for RE2027/RE2028: if total EUR > 0 then total inplace > 0.

    Aggregates across ALL project_class and project_stage per field.
    Violation when EUR > 0 but inplace = 0.
    """
    eur_cond = f"(SUM(fr_cond.{eur_rec_column}) + SUM(fr_cond.{eur_cprd_column}))"
    inplace_result = f"SUM(fr_result.{field_column})"
    return (
        f"SELECT {AGG_SELECT},"
        f" {eur_cond} AS val_ref,"
        f" {inplace_result} AS val_cmp"
        f" FROM field_resources fr_result"
        f" JOIN field_resources fr_cond"
        f" ON fr_result.wk_id = fr_cond.wk_id"
        f" AND fr_result.field_id = fr_cond.field_id"
        f" AND fr_result.report_year = fr_cond.report_year"
        f" WHERE fr_result.uncert_level = '{_uncert_value(result_uncert)}'"
        f" AND fr_cond.uncert_level = '{_uncert_value(cond_uncert)}'"
        f" GROUP BY {AGG_IDENT}"
        f" HAVING {eur_cond} > {tolerance}"
        f" AND COALESCE({inplace_result}, 0) <= {tolerance}"
    )


def build_eur_greater_than_sql(
    field_column: str,
    eur_rec_column: str,
    eur_cprd_column: str,
    cond_uncert: str,
    result_uncert: str,
    tolerance: float = TOLERANCE,
) -> str:
    """Build SQL for RE2029/RE2030: total inplace > total EUR when EUR > 0.

    Aggregates across ALL project_class and project_stage per field.
    Both sides at the same uncert_level (rows aggregate with each other).
    Violation when EUR > 0 AND inplace <= EUR.
    """
    eur_cond = f"(SUM(fr_cond.{eur_rec_column}) + SUM(fr_cond.{eur_cprd_column}))"
    inplace_result = f"SUM(fr_result.{field_column})"
    return (
        f"SELECT {AGG_SELECT},"
        f" {inplace_result} AS val_ref,"
        f" {eur_cond} AS val_cmp"
        f" FROM field_resources fr_result"
        f" JOIN field_resources fr_cond"
        f" ON fr_result.wk_id = fr_cond.wk_id"
        f" AND fr_result.field_id = fr_cond.field_id"
        f" AND fr_result.report_year = fr_cond.report_year"
        f" WHERE fr_result.uncert_level = '{_uncert_value(result_uncert)}'"
        f" AND fr_cond.uncert_level = '{_uncert_value(cond_uncert)}'"
        f" GROUP BY {AGG_IDENT}"
        f" HAVING {eur_cond} > {tolerance}"
        f" AND {inplace_result} - {eur_cond} < {tolerance}"
    )
