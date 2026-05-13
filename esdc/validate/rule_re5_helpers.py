"""RE5 helper utilities: ProjectLevel enum, SQL builders, shared constants."""

from __future__ import annotations

from collections.abc import Sequence
from enum import Enum

import duckdb

from esdc.selection import Severity
from esdc.validate.rules import TOLERANCE, Violation

IDENTIFIER_COLS: list[str] = [
    "project_id",
    "report_year",
    "project_name",
    "wk_name",
    "field_name",
]

SALES_COLUMNS: list[str] = [
    "cprd_sls_oil",
    "cprd_sls_con",
    "cprd_sls_ga",
    "cprd_sls_gn",
]

REC_COLUMNS: list[str] = [
    "rec_oil",
    "rec_con",
    "rec_ga",
    "rec_gn",
]

RES_COLUMNS: list[str] = [
    "res_oil",
    "res_con",
    "res_ga",
    "res_gn",
]

GCF_COLUMNS: list[str] = [
    "gcf_srock",
    "gcf_res",
    "gcf_ts",
    "gcf_dyn",
]

DCPY_COLUMNS: list[str] = [
    "dcpy_um_oil",
    "dcpy_um_con",
    "dcpy_um_ga",
    "dcpy_um_gn",
    "dcpy_ppa_oil",
    "dcpy_ppa_con",
    "dcpy_ppa_ga",
    "dcpy_ppa_gn",
    "dcpy_wi_oil",
    "dcpy_wi_con",
    "dcpy_wi_ga",
    "dcpy_wi_gn",
    "dcpy_gtr_oil",
    "dcpy_gtr_con",
    "dcpy_gtr_ga",
    "dcpy_gtr_gn",
    "dcpy_cio_oil",
    "dcpy_cio_con",
    "dcpy_cio_ga",
    "dcpy_cio_gn",
]


class ProjectLevel(str, Enum):
    E0 = "E0. On Production"
    E1 = "E1. Production on Hold"
    E2 = "E2. Under Development"
    E3 = "E3. Justified for Development"
    E4 = "E4. Production Pending"
    E5 = "E5. Development Unclarified"
    E6 = "E6. Further Development"
    E7 = "E7. Production Not Viable"
    E8 = "E8. Further Development Not Viable"
    X0 = "X0. Development Pending"
    X1 = "X1. Discovery under Evaluation"
    X2 = "X2. Development Undetermined"
    X3 = "X3. Development Not Viable"
    X4 = "X4. Inconclusive Flow"
    X5 = "X5. Prospect"
    X6 = "X6. Lead"
    A1 = "A1. Dry"
    A2 = "A2. Dissolved"


ABANDONED_LEVELS: list[str] = [ProjectLevel.A1, ProjectLevel.A2]
E0_E1_E4_E7: list[str] = [
    ProjectLevel.E0,
    ProjectLevel.E1,
    ProjectLevel.E4,
    ProjectLevel.E7,
]
E0_E7_E8_A1_A2: list[str] = [
    ProjectLevel.E0,
    ProjectLevel.E7,
    ProjectLevel.E8,
    ProjectLevel.A1,
    ProjectLevel.A2,
]


def _pl_list(levels: Sequence[str]) -> str:
    return ", ".join(f"'{lv.value if isinstance(lv, ProjectLevel) else lv}'" for lv in levels)


def _add_year_filter(sql: str, year: list[int] | None) -> str:
    if not year:
        return sql
    year_list = ", ".join(str(y) for y in year)
    if "WHERE" in sql:
        return sql.replace("WHERE", f"WHERE report_year IN ({year_list}) AND", 1)
    return sql + f" WHERE report_year IN ({year_list})"


def _add_year_filter_self_join(
    sql: str, year: list[int] | None, alias: str = "curr"
) -> str:
    if not year:
        return sql
    year_list = ", ".join(str(y) for y in year)
    if "WHERE" in sql:
        return sql.replace(
            "WHERE", f"WHERE {alias}.report_year IN ({year_list}) AND", 1
        )
    return sql + f" WHERE {alias}.report_year IN ({year_list})"


def execute_and_build_violations(
    conn: duckdb.DuckDBPyConnection,
    sql: str,
    *,
    rule_id: str,
    description: str,
    severity: Severity,
    table: str,
    validated_column: str,
    compared_columns: list[str],
    identifier_cols: list[str] | None = None,
    extra_columns: list[str] | None = None,
    year: list[int] | None = None,
) -> list[Violation]:
    if identifier_cols is None:
        identifier_cols = IDENTIFIER_COLS
    rows = conn.execute(sql).fetchall()
    violations: list[Violation] = []
    n_ident = len(identifier_cols)
    for row in rows:
        identifiers = {identifier_cols[i]: str(row[i]) for i in range(n_ident)}
        extra_start = n_ident
        current: dict[str, object] = {}
        if extra_columns:
            for j, col in enumerate(extra_columns):
                current[col] = row[extra_start + j]
        violations.append(
            Violation(
                rule_id=rule_id,
                rule_group="RE5",
                description=description,
                severity=severity,
                table=table,
                validated_column=validated_column,
                compared_columns=compared_columns,
                identifiers=identifiers,
                current_values=current,
            )
        )
    return violations


# ---------------------------------------------------------------------------
# SQL Builders
# ---------------------------------------------------------------------------


def build_sales_implies_level_sql(
    allowed_levels: Sequence[str],
    tolerance: float = TOLERANCE,
) -> str:
    """RE5001/RE5041: If any sales increment > 0, project level must be in allowed set.

    Self-joins project_resources on project_id and uncert_level to compute
    production increment (curr - prev). Violation when any substance has
    positive increment AND project_level NOT IN allowed set.
    DISTINCT ON (project_id, report_year) deduplicates across uncert levels.
    """
    sales_prev = " + ".join(f"COALESCE(prev.{c}, 0)" for c in SALES_COLUMNS)
    sales_curr = " + ".join(f"COALESCE(curr.{c}, 0)" for c in SALES_COLUMNS)
    ident = ", ".join(f"curr.{c}" for c in IDENTIFIER_COLS)
    allowed = _pl_list(allowed_levels)
    return (
        f"SELECT DISTINCT ON (curr.project_id, curr.report_year) {ident}, curr.project_level AS val_ref,"
        f" ({sales_curr}) - ({sales_prev}) AS val_cmp"
        f" FROM project_resources curr"
        f" JOIN project_resources prev"
        f" ON curr.project_id = prev.project_id"
        f" AND curr.report_year = prev.report_year + 1"
        f" AND prev.uncert_level = curr.uncert_level"
        f" WHERE ({sales_curr}) - ({sales_prev}) > {tolerance}"
        f" AND curr.project_level NOT IN ({allowed})"
    )


def build_no_sales_implies_not_level_sql(
    disallowed_levels: Sequence[str],
    tolerance: float = TOLERANCE,
) -> str:
    """RE5002/RE5052: If no sales increment, project level must NOT be in disallowed set.

    DISTINCT ON (project_id, report_year) to deduplicate across uncert levels.
    """
    sales_prev = " + ".join(f"COALESCE(prev.{c}, 0)" for c in SALES_COLUMNS)
    sales_curr = " + ".join(f"COALESCE(curr.{c}, 0)" for c in SALES_COLUMNS)
    ident = ", ".join(f"curr.{c}" for c in IDENTIFIER_COLS)
    disallowed = _pl_list(disallowed_levels)
    return (
        f"SELECT DISTINCT ON (curr.project_id, curr.report_year) {ident}, curr.project_level AS val_ref,"
        f" ({sales_curr}) - ({sales_prev}) AS val_cmp"
        f" FROM project_resources curr"
        f" JOIN project_resources prev"
        f" ON curr.project_id = prev.project_id"
        f" AND curr.report_year = prev.report_year + 1"
        f" AND prev.uncert_level = curr.uncert_level"
        f" WHERE ABS(({sales_curr}) - ({sales_prev})) <= {tolerance}"
        f" AND curr.project_level IN ({disallowed})"
    )


def build_transition_sql(
    previous_level: str,
    allowed_levels: Sequence[str],
) -> str:
    """RE5009–RE5023: If previous_level = X, current must be in allowed set.

    Uses project_level_previous column. DISTINCT ON (project_id, report_year)
    to deduplicate across uncert levels.
    """
    allowed = _pl_list(allowed_levels)
    ident = ", ".join(IDENTIFIER_COLS)
    pl_val = previous_level.value if isinstance(previous_level, ProjectLevel) else previous_level
    return (
        f"SELECT DISTINCT ON (project_id, report_year) {ident},"
        f" project_level AS val_ref, project_level_previous AS val_cmp"
        f" FROM project_resources"
        f" WHERE project_level_previous = '{pl_val}'"
        f" AND project_level NOT IN ({allowed})"
    )


def build_groovy_transition_sql(
    previous_level: str,
    groovy_value: bool,
    required_level: str,
) -> str:
    """RE5003, RE5005, RE5006: Transition with groovy_isactive condition."""
    groovy_int = 1 if groovy_value else 0
    ident = ", ".join(IDENTIFIER_COLS)
    pl_val = previous_level.value if isinstance(previous_level, ProjectLevel) else previous_level
    req_val = required_level.value if isinstance(required_level, ProjectLevel) else required_level
    return (
        f"SELECT DISTINCT ON (project_id, report_year) {ident},"
        f" project_level AS val_ref, groovy_isactive AS val_cmp"
        f" FROM project_resources"
        f" WHERE project_level_previous = '{pl_val}'"
        f" AND groovy_isactive = {groovy_int}"
        f" AND project_level != '{req_val}'"
    )


def build_multi_year_transition_sql(
    previous_levels: Sequence[str],
    n_years: int,
    required_level: str,
    groovy_condition: bool | None = None,
    no_production: bool = True,
    tolerance: float = TOLERANCE,
) -> str:
    """RE5004, RE5005, RE5007, RE5008, RE5018: Multi-year transition.

    Joins curr with prev1, prev2, (prev3) to check consecutive years.
    If no_production=True, adds condition that current sales increment <= tolerance.
    """
    ident = ", ".join(f"curr.{c}" for c in IDENTIFIER_COLS)
    req_val = required_level.value if isinstance(required_level, ProjectLevel) else required_level
    conditions = [f"curr.project_level != '{req_val}'"]
    for i, pl in enumerate(previous_levels[:n_years], 1):
        alias = f"prev{i}"
        pl_val = pl.value if isinstance(pl, ProjectLevel) else pl
        conditions.append(f"{alias}.project_level = '{pl_val}'")

    join_clauses = ""
    for i in range(1, n_years + 1):
        alias = f"prev{i}"
        prev_alias = "curr" if i == 1 else f"prev{i - 1}"
        join_clauses += (
            f" JOIN project_resources {alias}"
            f" ON {alias}.project_id = {prev_alias}.project_id"
            f" AND {alias}.report_year = {prev_alias}.report_year - 1"
            f" AND {alias}.uncert_level = {prev_alias}.uncert_level"
        )

    if groovy_condition is not None:
        conditions.append(f"curr.groovy_isactive = {1 if groovy_condition else 0}")

    if no_production:
        sales_curr = " + ".join(f"COALESCE(curr.{c}, 0)" for c in SALES_COLUMNS)
        sales_prev1 = " + ".join(f"COALESCE(prev1.{c}, 0)" for c in SALES_COLUMNS)
        conditions.append(
            f"ABS(({sales_curr}) - ({sales_prev1})) <= {tolerance}"
        )

    where = " AND ".join(conditions)
    return (
        f"SELECT DISTINCT ON (curr.project_id, curr.report_year) {ident}, curr.project_level AS val_ref"
        f" FROM project_resources curr"
        f" {join_clauses}"
        f" WHERE {where}"
    )


def build_gcf_total_implies_level_sql(
    condition: str,
    required_levels: Sequence[str],
) -> str:
    """RE5024–RE5026: GCF total condition → level in set."""
    allowed = _pl_list(required_levels)
    ident = ", ".join(IDENTIFIER_COLS)
    return (
        f"SELECT DISTINCT ON (project_id, report_year) {ident},"
        f" project_level AS val_ref, gcf_total AS val_cmp"
        f" FROM project_resources"
        f" WHERE {condition}"
        f" AND project_level NOT IN ({allowed})"
    )


def build_gcf_element_not_neutral_sql(
    gcf_column: str,
) -> str:
    """RE5027–RE5030: If previous GCF element != 0.5, current must also != 0.5."""
    ident = ", ".join(f"curr.{c}" for c in IDENTIFIER_COLS)
    return (
        f"SELECT DISTINCT ON (curr.project_id, curr.report_year) {ident},"
        f" curr.{gcf_column} AS val_ref, prev.{gcf_column} AS val_cmp"
        f" FROM project_resources curr"
        f" JOIN project_resources prev"
        f" ON curr.project_id = prev.project_id"
        f" AND curr.report_year = prev.report_year + 1"
        f" AND prev.uncert_level = curr.uncert_level"
    f" WHERE prev.{gcf_column} != 0.5"
    f" AND curr.{gcf_column} = 0.5"
    )


def build_gcf_transition_sql() -> str:
    """RE5031: X6→X5 transition requires gcf_total >= prev gcf_total."""
    ident = ", ".join(f"curr.{c}" for c in IDENTIFIER_COLS)
    return (
        f"SELECT DISTINCT ON (curr.project_id, curr.report_year) {ident},"
        f" curr.gcf_total AS val_ref, prev.gcf_total AS val_cmp"
        f" FROM project_resources curr"
        f" JOIN project_resources prev"
        f" ON curr.project_id = prev.project_id"
        f" AND curr.report_year = prev.report_year + 1"
        f" AND prev.uncert_level = curr.uncert_level"
        f" WHERE prev.project_level = '{ProjectLevel.X6.value}'"
        f" AND curr.project_level = '{ProjectLevel.X5.value}'"
        f" AND prev.gcf_total > curr.gcf_total"
    )


def build_gcf_neutral_implies_lead_sql() -> str:
    """RE5032: Any GCF element = 0.5 → project level = X6."""
    ident = ", ".join(IDENTIFIER_COLS)
    gcf_checks = " OR ".join(f"{c} = 0.5" for c in GCF_COLUMNS)
    return (
        f"SELECT DISTINCT ON (project_id, report_year) {ident},"
        f" project_level AS val_ref"
        f" FROM project_resources"
        f" WHERE ({gcf_checks})"
        f" AND project_level != '{ProjectLevel.X6.value}'"
    )


def build_gcf_abandoned_binary_sql(
    gcf_column: str,
) -> str:
    """RE5033–RE5036: If abandoned, GCF must be 0 or 1."""
    ident = ", ".join(IDENTIFIER_COLS)
    abandoned = _pl_list(ABANDONED_LEVELS)
    return (
        f"SELECT DISTINCT ON (project_id, report_year) {ident},"
        f" {gcf_column} AS val_ref"
        f" FROM project_resources"
        f" WHERE project_level IN ({abandoned})"
    f" AND {gcf_column} != 0"
    f" AND {gcf_column} != 1"
    )


def build_gcf_monotonic_sql(
    gcf_column: str,
) -> str:
    """RE5037–RE5040: If prev GCF > 0.5 and not abandoned, current >= prev."""
    ident = ", ".join(f"curr.{c}" for c in IDENTIFIER_COLS)
    abandoned = _pl_list(ABANDONED_LEVELS)
    return (
        f"SELECT DISTINCT ON (curr.project_id, curr.report_year) {ident},"
        f" curr.{gcf_column} AS val_ref, prev.{gcf_column} AS val_cmp"
        f" FROM project_resources curr"
        f" JOIN project_resources prev"
        f" ON curr.project_id = prev.project_id"
        f" AND curr.report_year = prev.report_year + 1"
        f" AND prev.uncert_level = curr.uncert_level"
    f" WHERE prev.{gcf_column} > 0.5"
    f" AND curr.project_level NOT IN ({abandoned})"
    f" AND prev.{gcf_column} > curr.{gcf_column}"
    )


def build_level_mandatory_sql() -> str:
    """RE5042: Project level must not be null."""
    ident = ", ".join(IDENTIFIER_COLS)
    return (
        f"SELECT {ident}, project_level AS val_ref"
        f" FROM project_resources"
        f" WHERE project_level IS NULL"
    )


def build_gcf_must_be_filled_sql(
    gcf_column: str,
) -> str:
    """RE5044–RE5047: GCF element must not be null."""
    ident = ", ".join(IDENTIFIER_COLS)
    return (
        f"SELECT DISTINCT ON (project_id, report_year) {ident},"
        f" {gcf_column} AS val_ref"
        f" FROM project_resources"
        f" WHERE {gcf_column} IS NULL"
    )


def build_gcf_range_sql(
    gcf_column: str,
) -> str:
    """RE5048–RE5051: GCF element must be in [0, 1]."""
    ident = ", ".join(IDENTIFIER_COLS)
    return (
        f"SELECT DISTINCT ON (project_id, report_year) {ident},"
        f" {gcf_column} AS val_ref"
        f" FROM project_resources"
        f" WHERE ({gcf_column} < 0 OR {gcf_column} > 1)"
    )


def build_reserves_implies_not_abandoned_sql(
    uncert_level: str,
    columns: Sequence[str],
) -> str:
    """RE5043/RE5065: Hydrocarbon volume > 0 → project level not in {A1, A2}."""
    ident = ", ".join(IDENTIFIER_COLS)
    abandoned = _pl_list(ABANDONED_LEVELS)
    vol_sum = " + ".join(f"COALESCE({c}, 0)" for c in columns)
    return (
        f"SELECT {ident}, project_level AS val_ref,"
        f" ({vol_sum}) AS val_cmp"
        f" FROM project_resources"
        f" WHERE uncert_level = '{uncert_level}'"
        f" AND ({vol_sum}) > {TOLERANCE}"
        f" AND project_level IN ({abandoned})"
    )


def build_no_reserves_implies_level_sql(
    required_levels: Sequence[str],
) -> str:
    """RE5053: No reserves (rec at HIGH all 0) → level in required set."""
    ident = ", ".join(IDENTIFIER_COLS)
    required = _pl_list(required_levels)
    rec_sum = " + ".join(f"COALESCE({c}, 0)" for c in REC_COLUMNS)
    return (
        f"SELECT DISTINCT ON (project_id, report_year) {ident},"
        f" project_level AS val_ref"
        f" FROM project_resources"
        f" WHERE uncert_level = '3. High Value'"
        f" AND ({rec_sum}) <= {TOLERANCE}"
        f" AND project_level NOT IN ({required})"
    )


def build_level_implies_1p_required_sql(
    levels: Sequence[str],
) -> str:
    """RE5054: Level in {E1,E2,E3} → sum of 1P reserves > 0."""
    ident = ", ".join(IDENTIFIER_COLS)
    level_list = _pl_list(levels)
    res_sum = " + ".join(f"COALESCE({c}, 0)" for c in RES_COLUMNS)
    return (
        f"SELECT {ident}, project_level AS val_ref,"
        f" ({res_sum}) AS val_cmp"
        f" FROM project_resources"
        f" WHERE uncert_level = '1. Low Value'"
        f" AND project_level IN ({level_list})"
        f" AND ({res_sum}) <= {TOLERANCE}"
    )


def build_level_implies_1p_forbidden_sql(
    levels: Sequence[str],
) -> str:
    """RE5055: Level in {E4,...,X6} → 1P reserves must be 0."""
    ident = ", ".join(IDENTIFIER_COLS)
    level_list = _pl_list(levels)
    res_sum = " + ".join(f"COALESCE({c}, 0)" for c in RES_COLUMNS)
    return (
        f"SELECT {ident}, project_level AS val_ref,"
        f" ({res_sum}) AS val_cmp"
        f" FROM project_resources"
        f" WHERE uncert_level = '1. Low Value'"
        f" AND project_level IN ({level_list})"
        f" AND ({res_sum}) > {TOLERANCE}"
    )


def build_discrepancy_implies_remarks_sql(
    uncert_level: str,
) -> str:
    """RE5056–RE5058: If discrepancy > 0, project_remarks must not be null."""
    ident = ", ".join(IDENTIFIER_COLS)
    dcpy_sum = " + ".join(f"COALESCE({c}, 0)" for c in DCPY_COLUMNS)
    return (
        f"SELECT DISTINCT ON (project_id, report_year) {ident},"
        f" project_remarks AS val_ref"
        f" FROM project_resources"
        f" WHERE uncert_level = '{uncert_level}'"
        f" AND ({dcpy_sum}) > {TOLERANCE}"
        f" AND (project_remarks IS NULL OR TRIM(project_remarks) = '')"
    )


def build_ioip_igip_change_implies_remarks_sql(
    column: str,
    uncert_level: str,
) -> str:
    """RE5059–RE5064: If IOIP/IGIP changed, project_remarks must not be null."""
    ident = ", ".join(f"curr.{c}" for c in IDENTIFIER_COLS)
    return (
        f"SELECT {ident},"
        f" curr.project_remarks AS val_ref"
        f" FROM project_resources curr"
        f" JOIN project_resources prev"
        f" ON curr.project_id = prev.project_id"
        f" AND curr.report_year = prev.report_year + 1"
        f" AND prev.uncert_level = '{uncert_level}'"
        f" WHERE curr.uncert_level = '{uncert_level}'"
        f" AND ABS(COALESCE(curr.{column}, 0) - COALESCE(prev.{column}, 0)) > {TOLERANCE}"
        f" AND (curr.project_remarks IS NULL OR TRIM(curr.project_remarks) = '')"
    )


def build_onstream_required_sql(
    levels: Sequence[str],
) -> str:
    """RE5066: Level in set → onstream_actual must not be null."""
    ident = ", ".join(IDENTIFIER_COLS)
    level_list = _pl_list(levels)
    return (
        f"SELECT DISTINCT ON (project_id, report_year) {ident},"
        f" project_level AS val_ref, onstream_actual AS val_cmp"
        f" FROM project_resources"
        f" WHERE project_level IN ({level_list})"
    f" AND (onstream_actual IS NULL OR onstream_actual = '')"
    )


def _extract_year_sql(column: str = "onstream_actual") -> str:
    """Extract integer year from onstream_actual.

    onstream_actual may be a plain year or a date string in
    DD-MM-YYYY / YYYY-MM-DD / MM/DD/YYYY format.
    """
    return (
        f"CASE"
        f" WHEN regexp_matches({column}, '^\\d{{4}}$')"
        f" THEN CAST({column} AS INTEGER)"
        f" WHEN regexp_matches({column}, '^\\d{{2}}-\\d{{2}}-\\d{{4}}$')"
        f" THEN YEAR(STRPTIME({column}, '%d-%m-%Y'))"
        f" WHEN regexp_matches({column}, '^\\d{{4}}-\\d{{2}}-\\d{{2}}$')"
        f" THEN YEAR(STRPTIME({column}, '%Y-%m-%d'))"
        f" WHEN regexp_matches({column}, '^\\d{{2}}/\\d{{2}}/\\d{{4}}$')"
        f" THEN YEAR(STRPTIME({column}, '%m/%d/%Y'))"
        f" ELSE NULL"
        f" END"
    )


def build_onstream_before_report_year_sql() -> str:
    """RE5067: onstream_actual < report_year."""
    ident = ", ".join(IDENTIFIER_COLS)
    year_expr = _extract_year_sql()
    return (
        f"SELECT DISTINCT ON (project_id, report_year) {ident},"
        f" report_year AS val_ref, onstream_actual AS val_cmp"
        f" FROM project_resources"
    f" WHERE onstream_actual IS NOT NULL"
    f" AND onstream_actual != ''"
    f" AND {year_expr} IS NOT NULL"
    f" AND {year_expr} >= report_year"
    )


def build_level_implies_sales_positive_sql(
    levels: Sequence[str],
    tolerance: float = TOLERANCE,
) -> str:
    """RE5052: Level in {E0,E1,E4,E7} → production increment > 0."""
    ident = ", ".join(f"curr.{c}" for c in IDENTIFIER_COLS)
    level_list = _pl_list(levels)
    sales_prev = " + ".join(f"COALESCE(prev.{c}, 0)" for c in SALES_COLUMNS)
    sales_curr = " + ".join(f"COALESCE(curr.{c}, 0)" for c in SALES_COLUMNS)
    return (
        f"SELECT DISTINCT ON (curr.project_id, curr.report_year) {ident},"
        f" curr.project_level AS val_ref,"
        f" ({sales_curr}) - ({sales_prev}) AS val_cmp"
        f" FROM project_resources curr"
        f" JOIN project_resources prev"
        f" ON curr.project_id = prev.project_id"
        f" AND curr.report_year = prev.report_year + 1"
        f" AND prev.uncert_level = curr.uncert_level"
        f" WHERE curr.project_level IN ({level_list})"
        f" AND ({sales_curr}) - ({sales_prev}) <= {tolerance}"
    )


def build_sales_implies_level_set_sql(
    allowed_levels: Sequence[str],
    tolerance: float = TOLERANCE,
) -> str:
    """RE5041: If any sales > 0, level must be in allowed set."""
    ident = ", ".join(f"curr.{c}" for c in IDENTIFIER_COLS)
    allowed = _pl_list(allowed_levels)
    sales_prev = " + ".join(f"COALESCE(prev.{c}, 0)" for c in SALES_COLUMNS)
    sales_curr = " + ".join(f"COALESCE(curr.{c}, 0)" for c in SALES_COLUMNS)
    return (
        f"SELECT DISTINCT ON (curr.project_id, curr.report_year) {ident},"
        f" curr.project_level AS val_ref,"
        f" ({sales_curr}) - ({sales_prev}) AS val_cmp"
        f" FROM project_resources curr"
        f" JOIN project_resources prev"
        f" ON curr.project_id = prev.project_id"
        f" AND curr.report_year = prev.report_year + 1"
        f" AND prev.uncert_level = curr.uncert_level"
        f" WHERE ({sales_curr}) - ({sales_prev}) > {tolerance}"
        f" AND curr.project_level NOT IN ({allowed})"
    )
