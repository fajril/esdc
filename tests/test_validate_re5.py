"""Tests for RE5 (Maturity Level) validation rules."""

from __future__ import annotations

import duckdb

from esdc.validate.rule_re5_helpers import (
    E0_E1_E4_E7,
    ProjectLevel,
    _extract_year_sql,
    build_discrepancy_implies_remarks_sql,
    build_gcf_abandoned_binary_sql,
    build_gcf_element_not_neutral_sql,
    build_gcf_monotonic_sql,
    build_gcf_must_be_filled_sql,
    build_gcf_range_sql,
    build_gcf_total_implies_level_sql,
    build_groovy_transition_sql,
    build_ioip_igip_change_implies_remarks_sql,
    build_level_implies_sales_positive_sql,
    build_level_mandatory_sql,
    build_multi_year_transition_sql,
    build_no_sales_implies_not_level_sql,
    build_onstream_before_report_year_sql,
    build_onstream_required_sql,
    build_reserves_implies_not_abandoned_sql,
    build_sales_implies_level_set_sql,
    build_sales_implies_level_sql,
    build_transition_sql,
)
from esdc.validate.rules import TOLERANCE, get_rules_by_group

# ---------------------------------------------------------------------------
# SQL builder unit tests
# ---------------------------------------------------------------------------


class TestBuildSalesImpliesLevelSql:
    def test_self_join(self):
        sql = build_sales_implies_level_sql([ProjectLevel.E0])
        assert "JOIN project_resources prev" in sql
        assert "curr.report_year = prev.report_year + 1" in sql

    def test_sales_increment_check(self):
        sql = build_sales_implies_level_sql([ProjectLevel.E0])
        assert "cprd_sls_oil" in sql
        assert f"> {TOLERANCE}" in sql

    def test_allowed_levels_in_sql(self):
        sql = build_sales_implies_level_sql([ProjectLevel.E0])
        assert "'E0. On Production'" in sql


class TestBuildNoSalesImpliesNotLevelSql:
    def test_disallowed_levels(self):
        sql = build_no_sales_implies_not_level_sql([ProjectLevel.E0])
        assert "'E0. On Production'" in sql

    def test_self_join(self):
        sql = build_no_sales_implies_not_level_sql([ProjectLevel.E0])
        assert "curr.report_year = prev.report_year + 1" in sql


class TestBuildTransitionSql:
    def test_previous_level(self):
        sql = build_transition_sql(
            ProjectLevel.E2, [ProjectLevel.E0, ProjectLevel.E2, ProjectLevel.E5]
        )
        assert "project_level_previous = 'E2. Under Development'" in sql

    def test_allowed_levels(self):
        sql = build_transition_sql(
            ProjectLevel.E2, [ProjectLevel.E0, ProjectLevel.E2, ProjectLevel.E5]
        )
        assert "'E0. On Production'" in sql
        assert "'E2. Under Development'" in sql
        assert "'E5. Development Unclarified'" in sql

    def test_distinct_on(self):
        sql = build_transition_sql(ProjectLevel.E2, [ProjectLevel.E0])
        assert "DISTINCT ON" in sql


class TestBuildGroovyTransitionSql:
    def test_groovy_false(self):
        sql = build_groovy_transition_sql(ProjectLevel.E1, False, ProjectLevel.E4)
        assert "groovy_isactive = 0" in sql
        assert "project_level_previous = 'E1. Production on Hold'" in sql

    def test_groovy_true(self):
        sql = build_groovy_transition_sql(ProjectLevel.E4, True, ProjectLevel.E4)
        assert "groovy_isactive = 1" in sql


class TestBuildMultiYearTransitionSql:
    def test_three_years(self):
        sql = build_multi_year_transition_sql([ProjectLevel.E1] * 3, 3, ProjectLevel.E4)
        assert "prev1" in sql
        assert "prev2" in sql
        assert "prev3" in sql

    def test_two_years(self):
        sql = build_multi_year_transition_sql([ProjectLevel.X1] * 2, 2, ProjectLevel.X0)
        assert "prev1" in sql
        assert "prev2" in sql
        assert "prev3" not in sql

    def test_groovy_condition(self):
        sql = build_multi_year_transition_sql(
            [ProjectLevel.E2] * 3, 3, ProjectLevel.E5, groovy_condition=False
        )
        assert "groovy_isactive = 0" in sql

    def test_uncert_level_in_join(self):
        sql = build_multi_year_transition_sql([ProjectLevel.E4] * 3, 3, ProjectLevel.E7)
        assert "prev1.uncert_level = curr.uncert_level" in sql
        assert "prev2.uncert_level = prev1.uncert_level" in sql
        assert "prev3.uncert_level = prev2.uncert_level" in sql

    def test_distinct_on(self):
        sql = build_multi_year_transition_sql([ProjectLevel.E4] * 3, 3, ProjectLevel.E7)
        assert "DISTINCT ON (curr.project_id, curr.report_year)" in sql

    def test_no_production_default(self):
        sql = build_multi_year_transition_sql([ProjectLevel.E4] * 3, 3, ProjectLevel.E7)
        assert "cprd_sls_oil" in sql
        assert "ABS" in sql

    def test_no_production_disabled(self):
        sql = build_multi_year_transition_sql(
            [ProjectLevel.X1] * 2, 2, ProjectLevel.X0, no_production=False
        )
        assert "cprd_sls" not in sql


class TestBuildGcfTotalImpliesLevelSql:
    def test_gcf_between_0_and_1(self):
        sql = build_gcf_total_implies_level_sql(
            "gcf_total > 0 AND gcf_total < 1",
            [ProjectLevel.X5, ProjectLevel.X6],
        )
        assert "gcf_total" in sql
        assert "'X5. Prospect'" in sql
        assert "'X6. Lead'" in sql


class TestBuildGcfElementNotNeutralSql:
    def test_self_join(self):
        sql = build_gcf_element_not_neutral_sql("gcf_srock")
        assert "curr.report_year = prev.report_year + 1" in sql
        assert "gcf_srock" in sql

    def test_neutral_check(self):
        sql = build_gcf_element_not_neutral_sql("gcf_res")
        assert "0.5" in sql


class TestBuildGcfMonotonicSql:
    def test_self_join(self):
        sql = build_gcf_monotonic_sql("gcf_dyn")
        assert "curr.report_year = prev.report_year + 1" in sql

    def test_not_abandoned(self):
        sql = build_gcf_monotonic_sql("gcf_ts")
        assert "A1. Dry" in sql


class TestBuildGcfAbandonedBinarySql:
    def test_abandoned_levels(self):
        sql = build_gcf_abandoned_binary_sql("gcf_srock")
        assert "'A1. Dry'" in sql
        assert "'A2. Dissolved'" in sql


class TestBuildReservesImpliesNotAbandonedSql:
    def test_high_uncert(self):
        sql = build_reserves_implies_not_abandoned_sql(
            "3. High Value", ["rec_oil", "rec_con"]
        )
        assert "uncert_level = '3. High Value'" in sql
        assert "rec_oil" in sql


class TestBuildDiscrepancyImpliesRemarksSql:
    def test_uncert_level(self):
        sql = build_discrepancy_implies_remarks_sql("1. Low Value")
        assert "uncert_level = '1. Low Value'" in sql
        assert "project_remarks IS NULL" in sql


class TestBuildIOIPIGIPChangeImpliesRemarksSql:
    def test_self_join(self):
        sql = build_ioip_igip_change_implies_remarks_sql("prj_ioip", "2. Middle Value")
        assert "curr.report_year = prev.report_year + 1" in sql
        assert "prj_ioip" in sql
        assert "prev.uncert_level = '2. Middle Value'" in sql


class TestBuildOnstreamRequiredSql:
    def test_level_set(self):
        sql = build_onstream_required_sql(E0_E1_E4_E7)
        assert "'E0. On Production'" in sql
        assert "onstream_actual IS NULL" in sql


class TestBuildOnstreamBeforeReportYearSql:
    def test_year_extraction_and_comparison(self):
        sql = build_onstream_before_report_year_sql()
        assert "regexp_matches" in sql
        assert ">= report_year" in sql

    def test_filters_empty_and_unparseable(self):
        sql = build_onstream_before_report_year_sql()
        assert "onstream_actual IS NOT NULL" in sql
        assert "onstream_actual != ''" in sql


class TestExtractYearSql:
    def test_plain_year(self):
        sql = _extract_year_sql()
        assert "regexp_matches(onstream_actual, '^\\d{4}$')" in sql
        assert "CAST(onstream_actual AS INTEGER)" in sql

    def test_dd_mm_yyyy(self):
        sql = _extract_year_sql()
        assert "'%d-%m-%Y'" in sql

    def test_yyyy_mm_dd(self):
        sql = _extract_year_sql()
        assert "'%Y-%m-%d'" in sql

    def test_mm_dd_yyyy(self):
        sql = _extract_year_sql()
        assert "'%m/%d/%Y'" in sql


class TestBuildLevelImpliesSalesPositiveSql:
    def test_no_self_join(self):
        """RE5052: SQL is single-table; uses cumulative sales directly."""
        sql = build_level_implies_sales_positive_sql(E0_E1_E4_E7)
        assert "JOIN project_resources" not in sql
        assert "prev.report_year" not in sql

    def test_cumulative_having(self):
        """RE5052: SQL uses HAVING clause for cumulative sales = 0 check."""
        sql = build_level_implies_sales_positive_sql(E0_E1_E4_E7)
        assert "HAVING" in sql
        assert "MIN(COALESCE(cprd_sls_oil, 0))" in sql
        assert "MIN(COALESCE(cprd_sls_con, 0))" in sql
        assert "MIN(COALESCE(cprd_sls_ga, 0))" in sql
        assert "MIN(COALESCE(cprd_sls_gn, 0))" in sql
        assert f"<= {TOLERANCE}" in sql

    def test_level_set_in_where(self):
        """RE5052: SQL filters on project_level in disallowed set."""
        sql = build_level_implies_sales_positive_sql(E0_E1_E4_E7)
        assert "'E0. On Production'" in sql
        assert "'E1. Production on Hold'" in sql
        assert "'E4. Production Pending'" in sql
        assert "'E7. Production Not Viable'" in sql

    def test_group_by_id_cols(self):
        """RE5052: SQL groups by (project_id, report_year) to dedupe rows."""
        sql = build_level_implies_sales_positive_sql(E0_E1_E4_E7)
        assert "GROUP BY project_id, report_year" in sql
        assert "ANY_VALUE(project_name)" in sql


class TestBuildSalesImpliesLevelSetSql:
    def test_no_self_join(self):
        """RE5041: SQL is single-table; uses cumulative sales directly."""
        sql = build_sales_implies_level_set_sql(E0_E1_E4_E7)
        assert "JOIN project_resources" not in sql
        assert "prev.report_year" not in sql

    def test_cumulative_having(self):
        """RE5041: SQL uses HAVING clause for cumulative sales > 0 check."""
        sql = build_sales_implies_level_set_sql(E0_E1_E4_E7)
        assert "HAVING" in sql
        assert "MIN(COALESCE(cprd_sls_oil, 0))" in sql
        assert f"> {TOLERANCE}" in sql

    def test_disallowed_levels_in_where(self):
        """RE5041: SQL filters on project_level NOT IN allowed set."""
        sql = build_sales_implies_level_set_sql(E0_E1_E4_E7)
        assert "NOT IN" in sql
        assert "'E0. On Production'" in sql
        assert "'E1. Production on Hold'" in sql
        assert "'E4. Production Pending'" in sql
        assert "'E7. Production Not Viable'" in sql

    def test_group_by_id_cols(self):
        """RE5041: SQL groups by (project_id, report_year) to dedupe rows."""
        sql = build_sales_implies_level_set_sql(E0_E1_E4_E7)
        assert "GROUP BY project_id, report_year" in sql


class TestBuildGroovyTransitionSqlNoProduction:
    def test_re5003_includes_production_check(self):
        """RE5003: When require_no_production=True, SQL self-joins and checks sales increment."""
        sql = build_groovy_transition_sql(
            ProjectLevel.E1,
            groovy_value=False,
            required_level=ProjectLevel.E4,
            require_no_production=True,
        )
        assert "JOIN project_resources prev" in sql
        assert "prev.report_year + 1" in sql
        assert "ABS((COALESCE(curr.cprd_sls_oil, 0)" in sql
        assert "COALESCE(prev.cprd_sls_oil, 0)" in sql
        assert f"<= {TOLERANCE}" in sql

    def test_re5006_includes_production_check(self):
        """RE5006: When require_no_production=True, SQL self-joins and checks sales increment."""
        sql = build_groovy_transition_sql(
            ProjectLevel.E4,
            groovy_value=True,
            required_level=ProjectLevel.E4,
            require_no_production=True,
        )
        assert "JOIN project_resources prev" in sql
        assert "ABS((COALESCE(curr.cprd_sls_oil, 0)" in sql
        assert "COALESCE(prev.cprd_sls_oil, 0)" in sql
        assert f"<= {TOLERANCE}" in sql

    def test_re5013_no_production_check_by_default(self):
        """RE5013 regression: default (require_no_production=False) does NOT add production check."""
        sql = build_groovy_transition_sql(
            ProjectLevel.E7,
            groovy_value=True,
            required_level=ProjectLevel.E4,
        )
        assert "JOIN project_resources" not in sql
        assert "prev.report_year" not in sql

    def test_re5013_explicit_false_no_production_check(self):
        """RE5013 regression: explicit require_no_production=False keeps old behavior."""
        sql = build_groovy_transition_sql(
            ProjectLevel.E7,
            groovy_value=True,
            required_level=ProjectLevel.E4,
            require_no_production=False,
        )
        assert "JOIN project_resources" not in sql
        assert "cprd_sls_oil" not in sql


class TestBuildLevelMandatorySql:
    def test_null_check(self):
        sql = build_level_mandatory_sql()
        assert "project_level IS NULL" in sql


class TestBuildGcfMustBeFilledSql:
    def test_null_check(self):
        sql = build_gcf_must_be_filled_sql("gcf_srock")
        assert "gcf_srock IS NULL" in sql


class TestBuildGcfRangeSql:
    def test_range_check(self):
        sql = build_gcf_range_sql("gcf_res")
        assert "gcf_res" in sql
        assert "< 0" in sql
        assert "> 1" in sql


# ---------------------------------------------------------------------------
# Rule registration tests
# ---------------------------------------------------------------------------


class TestRuleRegistration:
    def test_re5_rules_count(self):
        re5_rules = get_rules_by_group("RE5")
        assert len(re5_rules) == 69

    def test_all_rule_ids_sequential(self):
        re5_rules = get_rules_by_group("RE5")
        ids = sorted(r.rule_id for r in re5_rules)
        expected = [f"RE5{i:03d}" for i in range(1, 70)]
        assert ids == expected

    def test_all_rules_have_description(self):
        re5_rules = get_rules_by_group("RE5")
        for rule_cls in re5_rules:
            assert rule_cls.description, f"{rule_cls.rule_id} missing description"

    def test_all_rules_have_formal(self):
        re5_rules = get_rules_by_group("RE5")
        for rule_cls in re5_rules:
            assert rule_cls.formal, f"{rule_cls.rule_id} missing formal"

    def test_all_rules_not_fixable(self):
        re5_rules = get_rules_by_group("RE5")
        for rule_cls in re5_rules:
            assert rule_cls.is_fixable is False

    def test_all_rules_apply_to_project_resources(self):
        re5_rules = get_rules_by_group("RE5")
        for rule_cls in re5_rules:
            assert rule_cls.applies_to_tables == ["project_resources"]


# ---------------------------------------------------------------------------
# Integration tests with DuckDB
# ---------------------------------------------------------------------------


def _make_conn() -> duckdb.DuckDBPyConnection:
    conn = duckdb.connect(":memory:")
    cols = [
        "report_year INTEGER",
        "project_id TEXT",
        "project_name TEXT",
        "wk_name TEXT",
        "field_name TEXT",
        "uncert_level TEXT",
        "project_level TEXT",
        "project_level_previous TEXT",
        "project_class TEXT",
        "project_stage TEXT",
        "cprd_sls_oil DOUBLE",
        "cprd_sls_con DOUBLE",
        "cprd_sls_ga DOUBLE",
        "cprd_sls_gn DOUBLE",
        "rec_oil DOUBLE",
        "rec_con DOUBLE",
        "rec_ga DOUBLE",
        "rec_gn DOUBLE",
        "res_oil DOUBLE",
        "res_con DOUBLE",
        "res_ga DOUBLE",
        "res_gn DOUBLE",
        "prj_ioip DOUBLE",
        "prj_igip DOUBLE",
        "gcf_srock DOUBLE",
        "gcf_res DOUBLE",
        "gcf_ts DOUBLE",
        "gcf_dyn DOUBLE",
        "gcf_total DOUBLE",
        "groovy_isactive INTEGER",
        "onstream_actual TEXT",
        "project_remarks TEXT",
        "dcpy_um_oil DOUBLE DEFAULT 0",
        "dcpy_ppa_oil DOUBLE DEFAULT 0",
        "dcpy_wi_oil DOUBLE DEFAULT 0",
        "dcpy_gtr_oil DOUBLE DEFAULT 0",
        "dcpy_gtr_con DOUBLE DEFAULT 0",
        "dcpy_gtr_ga DOUBLE DEFAULT 0",
        "dcpy_gtr_gn DOUBLE DEFAULT 0",
        "dcpy_cio_oil DOUBLE DEFAULT 0",
    ]
    conn.execute(f"CREATE TABLE project_resources ({', '.join(cols)})")
    return conn


class TestRE5001Integration:
    def test_production_and_not_e0_fails(self):
        conn = _make_conn()
        conn.execute(
            "INSERT INTO project_resources VALUES"
            "(2024, 'P1', 'P1', 'W1', 'F1', '1. Low Value',"
            f"'{ProjectLevel.E2.value}', '{ProjectLevel.E1.value}', '1. Reserves & GRR', '1. Exploitation',"
            "1200, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,"
            "0, 0, 0, 0, 0, 1, NULL, NULL, 0, 0, 0, 0, 0, 0, 0, 0)"
        )
        conn.execute(
            "INSERT INTO project_resources VALUES"
            "(2023, 'P1', 'P1', 'W1', 'F1', '1. Low Value',"
            f"'{ProjectLevel.E1.value}', '{ProjectLevel.E0.value}', '1. Reserves & GRR', '1. Exploitation',"
            "1000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,"
            "0, 0, 0, 0, 0, 1, NULL, NULL, 0, 0, 0, 0, 0, 0, 0, 0)"
        )
        from esdc.validate.rule_re5 import RE5001

        rule = RE5001()
        violations = rule.check(conn)
        assert len(violations) == 1
        assert violations[0].rule_id == "RE5001"

    def test_production_and_e0_passes(self):
        conn = _make_conn()
        conn.execute(
            "INSERT INTO project_resources VALUES"
            "(2024, 'P1', 'P1', 'W1', 'F1', '1. Low Value',"
            f"'{ProjectLevel.E0.value}', '{ProjectLevel.E0.value}', '1. Reserves & GRR', '1. Exploitation',"
            "1200, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,"
            "0, 0, 0, 0, 0, 1, NULL, NULL, 0, 0, 0, 0, 0, 0, 0, 0)"
        )
        conn.execute(
            "INSERT INTO project_resources VALUES"
            "(2023, 'P1', 'P1', 'W1', 'F1', '1. Low Value',"
            f"'{ProjectLevel.E0.value}', '{ProjectLevel.E0.value}', '1. Reserves & GRR', '1. Exploitation',"
            "1000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,"
            "0, 0, 0, 0, 0, 1, NULL, NULL, 0, 0, 0, 0, 0, 0, 0, 0)"
        )
        from esdc.validate.rule_re5 import RE5001

        rule = RE5001()
        violations = rule.check(conn)
        assert len(violations) == 0


class TestRE5009Integration:
    def test_e2_to_e4_fails(self):
        conn = _make_conn()
        conn.execute(
            "INSERT INTO project_resources VALUES"
            "(2024, 'P1', 'P1', 'W1', 'F1', '1. Low Value',"
            f"'{ProjectLevel.E4.value}', '{ProjectLevel.E2.value}', '1. Reserves & GRR', '1. Exploitation',"
            "0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,"
            "0, 0, 0, 0, 0, 0, NULL, NULL, 0, 0, 0, 0, 0, 0, 0, 0)"
        )
        from esdc.validate.rule_re5 import RE5009

        rule = RE5009()
        violations = rule.check(conn)
        assert len(violations) == 1

    def test_e2_to_e0_passes(self):
        conn = _make_conn()
        conn.execute(
            "INSERT INTO project_resources VALUES"
            "(2024, 'P1', 'P1', 'W1', 'F1', '1. Low Value',"
            f"'{ProjectLevel.E0.value}', '{ProjectLevel.E2.value}', '1. Reserves & GRR', '1. Exploitation',"
            "100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,"
            "0, 0, 0, 0, 0, 0, NULL, NULL, 0, 0, 0, 0, 0, 0, 0, 0)"
        )
        from esdc.validate.rule_re5 import RE5009

        rule = RE5009()
        violations = rule.check(conn)
        assert len(violations) == 0


class TestRE5024Integration:
    def test_gcf_between_0_1_and_not_x5_x6_fails(self):
        conn = _make_conn()
        conn.execute(
            "INSERT INTO project_resources VALUES"
            "(2024, 'P1', 'P1', 'W1', 'F1', '1. Low Value',"
            f"'{ProjectLevel.E0.value}', '{ProjectLevel.E0.value}', '1. Reserves & GRR', '1. Exploitation',"
            "0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,"
            "0, 0, 0.5, 0.5, 0.5, 1, NULL, NULL, 0, 0, 0, 0, 0, 0, 0, 0)"
        )
        from esdc.validate.rule_re5 import RE5024

        rule = RE5024()
        violations = rule.check(conn)
        assert len(violations) == 1

    def test_gcf_between_0_1_and_x5_passes(self):
        conn = _make_conn()
        conn.execute(
            "INSERT INTO project_resources VALUES"
            "(2024, 'P1', 'P1', 'W1', 'F1', '1. Low Value',"
            f"'{ProjectLevel.X5.value}', '{ProjectLevel.X5.value}', '3. Prospective Resources', '2. Exploration',"
            "0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,"
            "0, 0, 0.8, 0.7, 0.4, 1, NULL, NULL, 0, 0, 0, 0, 0, 0, 0, 0)"
        )
        from esdc.validate.rule_re5 import RE5024

        rule = RE5024()
        violations = rule.check(conn)
        assert len(violations) == 0


class TestRE5042Integration:
    def test_null_level_fails(self):
        conn = _make_conn()
        conn.execute(
            "INSERT INTO project_resources VALUES"
            "(2024, 'P1', 'P1', 'W1', 'F1', '1. Low Value',"
            "NULL, NULL, '1. Reserves & GRR', '1. Exploitation',"
            "0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,"
            "0, 0, 0, 0, 0, 0, NULL, NULL, 0, 0, 0, 0, 0, 0, 0, 0)"
        )
        from esdc.validate.rule_re5 import RE5042

        rule = RE5042()
        violations = rule.check(conn)
        assert len(violations) == 1


class TestRE5048Integration:
    def test_gcf_out_of_range_fails(self):
        conn = _make_conn()
        conn.execute(
            "INSERT INTO project_resources VALUES"
            "(2024, 'P1', 'P1', 'W1', 'F1', '1. Low Value',"
            f"'{ProjectLevel.X5.value}', '{ProjectLevel.X5.value}', '3. Prospective Resources', '2. Exploration',"
            "0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,"
            "50, 0, 0.5, 0.5, 0.5, 1, NULL, NULL, 0, 0, 0, 0, 0, 0, 0, 0)"
        )
        from esdc.validate.rule_re5 import RE5048

        rule = RE5048()
        violations = rule.check(conn)
        assert len(violations) == 1

    def test_gcf_in_range_passes(self):
        conn = _make_conn()
        conn.execute(
            "INSERT INTO project_resources VALUES"
            "(2024, 'P1', 'P1', 'W1', 'F1', '1. Low Value',"
            f"'{ProjectLevel.X5.value}', '{ProjectLevel.X5.value}', '3. Prospective Resources', '2. Exploration',"
            "0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,"
            "0.5, 0.5, 0.5, 0.5, 0.5, 1, NULL, NULL, 0, 0, 0, 0, 0, 0, 0, 0)"
        )
        from esdc.validate.rule_re5 import RE5048

        rule = RE5048()
        violations = rule.check(conn)
        assert len(violations) == 0


class TestRE5066Integration:
    def test_e0_without_onstream_fails(self):
        conn = _make_conn()
        conn.execute(
            "INSERT INTO project_resources VALUES"
            "(2024, 'P1', 'P1', 'W1', 'F1', '1. Low Value',"
            f"'{ProjectLevel.E0.value}', '{ProjectLevel.E0.value}', '1. Reserves & GRR', '1. Exploitation',"
            "100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,"
            "0, 0, 0, 0, 0, 0, NULL, NULL, 0, 0, 0, 0, 0, 0, 0, 0)"
        )
        from esdc.validate.rule_re5 import RE5066

        rule = RE5066()
        violations = rule.check(conn)
        assert len(violations) == 1

    def test_e0_with_onstream_passes(self):
        conn = _make_conn()
        conn.execute(
            "INSERT INTO project_resources VALUES"
            "(2024, 'P1', 'P1', 'W1', 'F1', '1. Low Value',"
            f"'{ProjectLevel.E0.value}', '{ProjectLevel.E0.value}', '1. Reserves & GRR', '1. Exploitation',"
            "100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,"
            "0, 0, 0, 0, 0, 0, '2020', NULL, 0, 0, 0, 0, 0, 0, 0, 0)"
        )
        from esdc.validate.rule_re5 import RE5066

        rule = RE5066()
        violations = rule.check(conn)
        assert len(violations) == 0

    def test_e0_with_empty_string_onstream_fails(self):
        conn = _make_conn()
        conn.execute(
            "INSERT INTO project_resources VALUES"
            "(2024, 'P1', 'P1', 'W1', 'F1', '1. Low Value',"
            f"'{ProjectLevel.E0.value}', '{ProjectLevel.E0.value}', '1. Reserves & GRR', '1. Exploitation',"
            "100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,"
            "0, 0, 0, 0, 0, 0, '', NULL, 0, 0, 0, 0, 0, 0, 0, 0)"
        )
        from esdc.validate.rule_re5 import RE5066

        rule = RE5066()
        violations = rule.check(conn)
        assert len(violations) == 1


class TestRE5067Integration:
    def test_onstream_equals_report_year_fails(self):
        conn = _make_conn()
        conn.execute(
            "INSERT INTO project_resources VALUES"
            "(2024, 'P1', 'P1', 'W1', 'F1', '1. Low Value',"
            f"'{ProjectLevel.E0.value}', '{ProjectLevel.E0.value}', '1. Reserves & GRR', '1. Exploitation',"
            "100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,"
            "0, 0, 0, 0, 0, 0, '2024', NULL, 0, 0, 0, 0, 0, 0, 0, 0)"
        )
        from esdc.validate.rule_re5 import RE5067

        rule = RE5067()
        violations = rule.check(conn)
        assert len(violations) == 1

    def test_onstream_before_report_year_passes(self):
        conn = _make_conn()
        conn.execute(
            "INSERT INTO project_resources VALUES"
            "(2024, 'P1', 'P1', 'W1', 'F1', '1. Low Value',"
            f"'{ProjectLevel.E0.value}', '{ProjectLevel.E0.value}', '1. Reserves & GRR', '1. Exploitation',"
            "100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,"
            "0, 0, 0, 0, 0, 0, '2020', NULL, 0, 0, 0, 0, 0, 0, 0, 0)"
        )
        from esdc.validate.rule_re5 import RE5067

        rule = RE5067()
        violations = rule.check(conn)
        assert len(violations) == 0

    def test_empty_string_onstream_not_counted(self):
        conn = _make_conn()
        conn.execute(
            "INSERT INTO project_resources VALUES"
            "(2024, 'P1', 'P1', 'W1', 'F1', '1. Low Value',"
            f"'{ProjectLevel.E0.value}', '{ProjectLevel.E0.value}', '1. Reserves & GRR', '1. Exploitation',"
            "100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,"
            "0, 0, 0, 0, 0, 0, '', NULL, 0, 0, 0, 0, 0, 0, 0, 0)"
        )
        from esdc.validate.rule_re5 import RE5067

        rule = RE5067()
        violations = rule.check(conn)
        assert len(violations) == 0

    def test_date_string_onstream_dd_mm_yyyy_passes(self):
        conn = _make_conn()
        conn.execute(
            "INSERT INTO project_resources VALUES"
            "(2024, 'P1', 'P1', 'W1', 'F1', '1. Low Value',"
            f"'{ProjectLevel.E0.value}', '{ProjectLevel.E0.value}', '1. Reserves & GRR', '1. Exploitation',"
            "100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,"
            "0, 0, 0, 0, 0, 0, '15-03-1977', NULL, 0, 0, 0, 0, 0, 0, 0, 0)"
        )
        from esdc.validate.rule_re5 import RE5067

        rule = RE5067()
        violations = rule.check(conn)
        assert len(violations) == 0

    def test_date_string_onstream_dd_mm_yyyy_fails(self):
        conn = _make_conn()
        conn.execute(
            "INSERT INTO project_resources VALUES"
            "(2024, 'P1', 'P1', 'W1', 'F1', '1. Low Value',"
            f"'{ProjectLevel.E0.value}', '{ProjectLevel.E0.value}', '1. Reserves & GRR', '1. Exploitation',"
            "100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,"
            "0, 0, 0, 0, 0, 0, '15-03-2025', NULL, 0, 0, 0, 0, 0, 0, 0, 0)"
        )
        from esdc.validate.rule_re5 import RE5067

        rule = RE5067()
        violations = rule.check(conn)
        assert len(violations) == 1

    def test_date_string_onstream_yyyy_mm_dd_passes(self):
        conn = _make_conn()
        conn.execute(
            "INSERT INTO project_resources VALUES"
            "(2024, 'P1', 'P1', 'W1', 'F1', '1. Low Value',"
            f"'{ProjectLevel.E0.value}', '{ProjectLevel.E0.value}', '1. Reserves & GRR', '1. Exploitation',"
            "100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,"
            "0, 0, 0, 0, 0, 0, '1977-03-15', NULL, 0, 0, 0, 0, 0, 0, 0, 0)"
        )
        from esdc.validate.rule_re5 import RE5067

        rule = RE5067()
        violations = rule.check(conn)
        assert len(violations) == 0

    def test_unparseable_onstream_skipped(self):
        conn = _make_conn()
        conn.execute(
            "INSERT INTO project_resources VALUES"
            "(2024, 'P1', 'P1', 'W1', 'F1', '1. Low Value',"
            f"'{ProjectLevel.E0.value}', '{ProjectLevel.E0.value}', '1. Reserves & GRR', '1. Exploitation',"
            "100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,"
            "0, 0, 0, 0, 0, 0, 'not-a-date', NULL, 0, 0, 0, 0, 0, 0, 0, 0)"
        )
        from esdc.validate.rule_re5 import RE5067

        rule = RE5067()
        violations = rule.check(conn)
        assert len(violations) == 0


class TestRE5005Integration:
    def _insert_e4_multi_year(
        self, conn: duckdb.DuckDBPyConnection, project_id: str, project_name: str
    ) -> None:
        for uncert in ("1. Low Value", "2. Middle Value", "3. High Value"):
            for year, level in [
                (2021, ProjectLevel.E4),
                (2022, ProjectLevel.E4),
                (2023, ProjectLevel.E4),
                (2024, ProjectLevel.E0),
            ]:
                conn.execute(
                    "INSERT INTO project_resources VALUES"
                    f"({year}, '{project_id}', '{project_name}', 'W1', 'F1', '{uncert}',"
                    f"'{level.value}', '{ProjectLevel.E4.value}', '1. Reserves & GRR', '1. Exploitation',"
                    "100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,"
                    "0, 0, 0, 0, 0, 0, NULL, NULL, 0, 0, 0, 0, 0, 0, 0, 0)"
                )

    def test_e4_3years_no_production_not_e7_fails_single_violation(self):
        conn = _make_conn()
        self._insert_e4_multi_year(conn, "P1", "PROJ1")
        from esdc.validate.rule_re5 import RE5005

        rule = RE5005()
        violations = rule.check(conn)
        assert len(violations) == 1

    def test_e4_3years_with_production_not_e7_no_violation(self):
        conn = _make_conn()
        for uncert in ("1. Low Value", "2. Middle Value", "3. High Value"):
            for year, level, sales_oil in [
                (2021, ProjectLevel.E4, 100),
                (2022, ProjectLevel.E4, 100),
                (2023, ProjectLevel.E4, 100),
                (2024, ProjectLevel.E0, 150),
            ]:
                conn.execute(
                    "INSERT INTO project_resources VALUES"
                    f"({year}, 'P1', 'PROJ1', 'W1', 'F1', '{uncert}',"
                    f"'{level.value}', '{ProjectLevel.E4.value}', '1. Reserves & GRR', '1. Exploitation',"
                    f"{sales_oil}, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,"
                    "0, 0, 0, 0, 0, 0, NULL, NULL, 0, 0, 0, 0, 0, 0, 0, 0)"
                )
        from esdc.validate.rule_re5 import RE5005

        rule = RE5005()
        violations = rule.check(conn)
        assert len(violations) == 0

    def test_e4_3years_no_production_is_e7_no_violation(self):
        conn = _make_conn()
        for uncert in ("1. Low Value", "2. Middle Value", "3. High Value"):
            for year, level in [
                (2021, ProjectLevel.E4),
                (2022, ProjectLevel.E4),
                (2023, ProjectLevel.E4),
                (2024, ProjectLevel.E7),
            ]:
                conn.execute(
                    "INSERT INTO project_resources VALUES"
                    f"({year}, 'P1', 'PROJ1', 'W1', 'F1', '{uncert}',"
                    f"'{level.value}', '{ProjectLevel.E4.value}', '1. Reserves & GRR', '1. Exploitation',"
                    "100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,"
                    "0, 0, 0, 0, 0, 0, NULL, NULL, 0, 0, 0, 0, 0, 0, 0, 0)"
                )
        from esdc.validate.rule_re5 import RE5005

        rule = RE5005()
        violations = rule.check(conn)
        assert len(violations) == 0


class TestRE5068Integration:
    def test_e0_reserves_exhausted_fails(self):
        conn = _make_conn()
        for year, res_oil, cprd_sls, dcpy_gtr in [
            (2023, 200, 900, 0),
            (2024, 100, 1200, -100),
        ]:
            conn.execute(
                "INSERT INTO project_resources VALUES"
                f"({year}, 'P1', 'P1', 'W1', 'F1', '1. Low Value',"
                f"'{ProjectLevel.E0.value}', '{ProjectLevel.E0.value}', '1. Reserves & GRR', '1. Exploitation',"
                f"{cprd_sls}, 0, 0, 0, 0, 0, 0, 0, {res_oil}, 0, 0, 0, 0, 0, "
                "0, 0, 0, 0, 0, 1, NULL, NULL, 0, 0, 0, "
                f"{dcpy_gtr}, 0, 0, 0, 0)"
            )
        from esdc.validate.rule_re5 import RE5068

        rule = RE5068()
        violations = rule.check(conn)
        assert len(violations) == 1

    def test_e0_reserves_not_exhausted_passes(self):
        conn = _make_conn()
        for year, res_oil, cprd_sls, dcpy_gtr in [
            (2023, 200, 900, 0),
            (2024, 150, 1000, 0),
        ]:
            conn.execute(
                "INSERT INTO project_resources VALUES"
                f"({year}, 'P1', 'P1', 'W1', 'F1', '1. Low Value',"
                f"'{ProjectLevel.E0.value}', '{ProjectLevel.E0.value}', '1. Reserves & GRR', '1. Exploitation',"
                f"{cprd_sls}, 0, 0, 0, 0, 0, 0, 0, {res_oil}, 0, 0, 0, 0, 0, "
                "0, 0, 0, 0, 0, 1, NULL, NULL, 0, 0, 0, "
                f"{dcpy_gtr}, 0, 0, 0, 0)"
            )
        from esdc.validate.rule_re5 import RE5068

        rule = RE5068()
        violations = rule.check(conn)
        assert len(violations) == 0

    def test_e0_no_reserves_passes(self):
        conn = _make_conn()
        for year in [2023, 2024]:
            conn.execute(
                "INSERT INTO project_resources VALUES"
                f"({year}, 'P1', 'P1', 'W1', 'F1', '1. Low Value',"
                f"'{ProjectLevel.E0.value}', '{ProjectLevel.E0.value}', '1. Reserves & GRR', '1. Exploitation',"
                "100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, "
                "0, 0, 0, 0, 0, 1, NULL, NULL, 0, 0, 0, 0, 0, 0, 0, 0)"
            )
        from esdc.validate.rule_re5 import RE5068

        rule = RE5068()
        violations = rule.check(conn)
        assert len(violations) == 0


class TestRE5069Integration:
    def test_2p_zero_exhausted_not_e0_fails(self):
        conn = _make_conn()
        for year, res_oil, cprd_sls, dcpy_gtr in [
            (2023, 200, 900, 0),
            (2024, 0, 1200, -100),
        ]:
            conn.execute(
                "INSERT INTO project_resources VALUES"
                f"({year}, 'P1', 'P1', 'W1', 'F1', '2. Middle Value',"
                f"'{ProjectLevel.E1.value}', '{ProjectLevel.E0.value}', '1. Reserves & GRR', '1. Exploitation',"
                f"{cprd_sls}, 0, 0, 0, 0, 0, 0, 0, {res_oil}, 0, 0, 0, 0, 0, "
                "0, 0, 0, 0, 0, 1, NULL, NULL, 0, 0, 0, "
                f"{dcpy_gtr}, 0, 0, 0, 0)"
            )
        from esdc.validate.rule_re5 import RE5069

        rule = RE5069()
        violations = rule.check(conn)
        assert len(violations) == 1

    def test_2p_zero_exhausted_is_e0_passes(self):
        conn = _make_conn()
        for year, res_oil, cprd_sls, dcpy_gtr in [
            (2023, 200, 900, 0),
            (2024, 0, 1200, -100),
        ]:
            conn.execute(
                "INSERT INTO project_resources VALUES"
                f"({year}, 'P1', 'P1', 'W1', 'F1', '2. Middle Value',"
                f"'{ProjectLevel.E0.value}', '{ProjectLevel.E0.value}', '1. Reserves & GRR', '1. Exploitation',"
                f"{cprd_sls}, 0, 0, 0, 0, 0, 0, 0, {res_oil}, 0, 0, 0, 0, 0, "
                "0, 0, 0, 0, 0, 1, NULL, NULL, 0, 0, 0, "
                f"{dcpy_gtr}, 0, 0, 0, 0)"
            )
        from esdc.validate.rule_re5 import RE5069

        rule = RE5069()
        violations = rule.check(conn)
        assert len(violations) == 0

    def test_2p_nonzero_passes(self):
        conn = _make_conn()
        for year in [2023, 2024]:
            conn.execute(
                "INSERT INTO project_resources VALUES"
                f"({year}, 'P1', 'P1', 'W1', 'F1', '2. Middle Value',"
                f"'{ProjectLevel.E4.value}', '{ProjectLevel.E0.value}', '1. Reserves & GRR', '1. Exploitation',"
                "100, 0, 0, 0, 100, 0, 0, 200, "
                "0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, "
                "1, NULL, NULL, 0, 0, 0, 0, 0, 0, 0, 0)"
            )
        from esdc.validate.rule_re5 import RE5069

        rule = RE5069()
        violations = rule.check(conn)
        assert len(violations) == 0


class TestProjectLevelEnum:
    def test_all_values_present(self):
        assert len(ProjectLevel) == 18

    def test_e0_value(self):
        assert ProjectLevel.E0 == "E0. On Production"

    def test_a2_value(self):
        assert ProjectLevel.A2 == "A2. Dissolved"

    def test_e7_value(self):
        assert ProjectLevel.E7 == "E7. Production Not Viable"

    def test_e8_value(self):
        assert ProjectLevel.E8 == "E8. Further Development Not Viable"

    def test_x3_value(self):
        assert ProjectLevel.X3 == "X3. Development Not Viable"

    def test_is_str(self):
        assert isinstance(ProjectLevel.E0, str)
