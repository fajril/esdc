"""RE2 helper utilities for material balance validation rules."""

from __future__ import annotations

from esdc.validate.rule_re0_helpers import (
    AGGREGATION_CONSISTENCY_IDENTIFIER_COLS,
    IDENTIFIER_COLS,
    _uncert_value,
)
from esdc.validate.rules import TOLERANCE


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
    """Build SQL for RE2025/RE2026: field in-place > EUR when EUR > 0.

    Self-joins field_resources on (wk_id, field_id, report_year, project_stage,
    project_class). fr_result at result_uncert, fr_cond at cond_uncert.
    EUR = rec + cprd_sls (computed, not using pre-calculated eur column).
    Violation when EUR > 0 AND field_column <= EUR.
    """
    ident = ", ".join(f"fr_result.{c}" for c in AGGREGATION_CONSISTENCY_IDENTIFIER_COLS)
    return (
        f"SELECT {ident},"
        f" fr_result.{field_column} AS val_ref,"
        f" (fr_cond.{eur_rec_column} + fr_cond.{eur_cprd_column}) AS val_cmp"
        f" FROM field_resources fr_result"
        f" JOIN field_resources fr_cond"
        f" ON fr_result.wk_id = fr_cond.wk_id"
        f" AND fr_result.field_id = fr_cond.field_id"
        f" AND fr_result.report_year = fr_cond.report_year"
        f" AND fr_result.project_stage = fr_cond.project_stage"
        f" AND fr_result.project_class = fr_cond.project_class"
        f" WHERE fr_result.uncert_level = '{_uncert_value(result_uncert)}'"
        f" AND fr_cond.uncert_level = '{_uncert_value(cond_uncert)}'"
        f" AND (fr_cond.{eur_rec_column} + fr_cond.{eur_cprd_column}) > {tolerance}"
        f" AND fr_result.{field_column}"
        f" - (fr_cond.{eur_rec_column} + fr_cond.{eur_cprd_column}) < {tolerance}"
    )


def build_eur_implication_sql(
    field_column: str,
    eur_rec_column: str,
    eur_cprd_column: str,
    cond_uncert: str,
    result_uncert: str,
    tolerance: float = TOLERANCE,
) -> str:
    """Build SQL for RE2027/RE2028: if EUR > 0 then field > 0.

    Self-joins field_resources on (wk_id, field_id, report_year, project_stage,
    project_class). fr_result at result_uncert, fr_cond at cond_uncert.
    Violation when EUR > 0 but field_column = 0.
    """
    ident = ", ".join(f"fr_result.{c}" for c in AGGREGATION_CONSISTENCY_IDENTIFIER_COLS)
    return (
        f"SELECT {ident},"
        f" (fr_cond.{eur_rec_column} + fr_cond.{eur_cprd_column}) AS val_ref,"
        f" fr_result.{field_column} AS val_cmp"
        f" FROM field_resources fr_result"
        f" JOIN field_resources fr_cond"
        f" ON fr_result.wk_id = fr_cond.wk_id"
        f" AND fr_result.field_id = fr_cond.field_id"
        f" AND fr_result.report_year = fr_cond.report_year"
        f" AND fr_result.project_stage = fr_cond.project_stage"
        f" AND fr_result.project_class = fr_cond.project_class"
        f" WHERE fr_result.uncert_level = '{_uncert_value(result_uncert)}'"
        f" AND fr_cond.uncert_level = '{_uncert_value(cond_uncert)}'"
        f" AND (fr_cond.{eur_rec_column} + fr_cond.{eur_cprd_column}) > {tolerance}"
        f" AND COALESCE(fr_result.{field_column}, 0) <= {tolerance}"
    )


def build_eur_greater_than_sql(
    field_column: str,
    eur_rec_column: str,
    eur_cprd_column: str,
    cond_uncert: str,
    result_uncert: str,
    tolerance: float = TOLERANCE,
) -> str:
    """Build SQL for RE2029/RE2030: field > EUR when EUR > 0.

    Self-joins field_resources on (wk_id, field_id, report_year, project_stage,
    project_class). Both sides at the same uncert_level (row joins with itself).
    Violation when EUR > 0 AND field_column <= EUR.
    """
    ident = ", ".join(f"fr_result.{c}" for c in AGGREGATION_CONSISTENCY_IDENTIFIER_COLS)
    return (
        f"SELECT {ident},"
        f" fr_result.{field_column} AS val_ref,"
        f" (fr_cond.{eur_rec_column} + fr_cond.{eur_cprd_column}) AS val_cmp"
        f" FROM field_resources fr_result"
        f" JOIN field_resources fr_cond"
        f" ON fr_result.wk_id = fr_cond.wk_id"
        f" AND fr_result.field_id = fr_cond.field_id"
        f" AND fr_result.report_year = fr_cond.report_year"
        f" AND fr_result.project_stage = fr_cond.project_stage"
        f" AND fr_result.project_class = fr_cond.project_class"
        f" WHERE fr_result.uncert_level = '{_uncert_value(result_uncert)}'"
        f" AND fr_cond.uncert_level = '{_uncert_value(cond_uncert)}'"
        f" AND (fr_cond.{eur_rec_column} + fr_cond.{eur_cprd_column}) > {tolerance}"
        f" AND fr_result.{field_column}"
        f" - (fr_cond.{eur_rec_column} + fr_cond.{eur_cprd_column}) < {tolerance}"
    )
