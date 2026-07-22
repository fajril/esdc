"""Comprehensive tests for RE2 (Material Balance) validation rules."""

from __future__ import annotations

import tempfile

import duckdb

from esdc.validate.rule_re0_helpers import UncertLevel
from esdc.validate.rule_re2_helpers import (
    build_eur_bounds_sql,
    build_eur_greater_than_sql,
    build_eur_implication_sql,
    build_material_balance_sql,
)
from esdc.validate.rules import TOLERANCE, get_all_rules, get_rules_by_group

# ---------------------------------------------------------------------------
# SQL builder unit tests
# ---------------------------------------------------------------------------


class TestBuildMaterialBalanceSql:
    """Unit tests for build_material_balance_sql."""

    def test_returns_string(self):
        sql = build_material_balance_sql(
            "rec_oil",
            ["dcpy_um_oil", "dcpy_ppa_oil"],
            "cprd_sls_oil",
            UncertLevel.LOW,
        )
        assert isinstance(sql, str)

    def test_self_join_on_project_id(self):
        sql = build_material_balance_sql(
            "rec_oil",
            ["dcpy_um_oil"],
            "cprd_sls_oil",
            UncertLevel.MID,
        )
        assert "curr.project_id = prev.project_id" in sql

    def test_joins_on_consecutive_years(self):
        sql = build_material_balance_sql(
            "rec_oil",
            ["dcpy_um_oil"],
            "cprd_sls_oil",
            UncertLevel.MID,
        )
        assert "curr.report_year = prev.report_year + 1" in sql

    def test_uncert_level_filter_on_both_sides(self):
        sql = build_material_balance_sql(
            "rec_con",
            ["dcpy_um_con"],
            "cprd_sls_con",
            UncertLevel.LOW,
        )
        assert "prev.uncert_level = '1. Low Value'" in sql
        assert "curr.uncert_level = '1. Low Value'" in sql

    def test_validated_column_in_select(self):
        sql = build_material_balance_sql(
            "rec_oil",
            ["dcpy_um_oil", "dcpy_ppa_oil", "dcpy_wi_oil"],
            "cprd_sls_oil",
            UncertLevel.MID,
        )
        assert "COALESCE(curr.rec_oil, 0) AS val_ref" in sql
        assert "prev.rec_oil" in sql

    def test_dcpy_columns_in_formula(self):
        sql = build_material_balance_sql(
            "rec_oil",
            ["dcpy_um_oil", "dcpy_ppa_oil", "dcpy_wi_oil"],
            "cprd_sls_oil",
            UncertLevel.MID,
        )
        assert "COALESCE(curr.dcpy_um_oil, 0)" in sql
        assert "COALESCE(curr.dcpy_ppa_oil, 0)" in sql
        assert "COALESCE(curr.dcpy_wi_oil, 0)" in sql

    def test_production_delta_in_formula(self):
        sql = build_material_balance_sql(
            "res_oil",
            ["dcpy_gtr_oil"],
            "cprd_sls_oil",
            UncertLevel.LOW,
        )
        assert "COALESCE(curr.cprd_sls_oil, 0)" in sql
        assert "COALESCE(prev.cprd_sls_oil, 0)" in sql

    def test_abs_tolerance_check(self):
        sql = build_material_balance_sql(
            "rec_oil",
            ["dcpy_um_oil"],
            "cprd_sls_oil",
            UncertLevel.MID,
        )
        assert f") > {TOLERANCE}" in sql
        assert "ABS(" in sql

    def test_custom_tolerance(self):
        sql = build_material_balance_sql(
            "rec_oil",
            ["dcpy_um_oil"],
            "cprd_sls_oil",
            UncertLevel.MID,
            tolerance=0.01,
        )
        assert ") > 0.01" in sql

    def test_single_dcpy_column_for_reserves(self):
        sql = build_material_balance_sql(
            "res_oil",
            ["dcpy_gtr_oil"],
            "cprd_sls_oil",
            UncertLevel.LOW,
        )
        assert "COALESCE(curr.dcpy_gtr_oil, 0)" in sql
        assert "dcpy_um_oil" not in sql


class TestBuildEurBoundsSql:
    """Unit tests for build_eur_bounds_sql."""

    def test_returns_string(self):
        sql = build_eur_bounds_sql(
            "ioip",
            "rec_oil",
            "cprd_sls_oil",
            UncertLevel.HIGH,
            UncertLevel.MID,
        )
        assert isinstance(sql, str)

    def test_self_joins_field_resources(self):
        sql = build_eur_bounds_sql(
            "ioip",
            "rec_oil",
            "cprd_sls_oil",
            UncertLevel.HIGH,
            UncertLevel.MID,
        )
        assert "field_resources fr_result" in sql
        assert "field_resources fr_cond" in sql
        assert "project_resources" not in sql

    def test_both_uncert_levels_in_filter(self):
        sql = build_eur_bounds_sql(
            "ioip",
            "rec_oil",
            "cprd_sls_oil",
            UncertLevel.HIGH,
            UncertLevel.MID,
        )
        assert "fr_cond.uncert_level = '3. High Value'" in sql
        assert "fr_result.uncert_level = '2. Middle Value'" in sql

    def test_eur_computed_from_rec_plus_cprd(self):
        sql = build_eur_bounds_sql(
            "ioip",
            "rec_oil",
            "cprd_sls_oil",
            UncertLevel.HIGH,
            UncertLevel.MID,
        )
        assert "fr_cond.rec_oil" in sql
        assert "fr_cond.cprd_sls_oil" in sql

    def test_group_by_and_having(self):
        sql = build_eur_bounds_sql(
            "ioip",
            "rec_oil",
            "cprd_sls_oil",
            UncertLevel.HIGH,
            UncertLevel.MID,
        )
        assert "GROUP BY" in sql
        assert "HAVING" in sql

    def test_no_project_class_or_stage_in_join(self):
        sql = build_eur_bounds_sql(
            "ioip",
            "rec_oil",
            "cprd_sls_oil",
            UncertLevel.HIGH,
            UncertLevel.MID,
        )
        assert "project_class" not in sql
        assert "project_stage" not in sql

    def test_aggregates_with_sum(self):
        sql = build_eur_bounds_sql(
            "ioip",
            "rec_oil",
            "cprd_sls_oil",
            UncertLevel.HIGH,
            UncertLevel.MID,
        )
        assert "SUM(fr_result.ioip)" in sql
        assert "SUM(fr_cond.rec_oil)" in sql
        assert "SUM(fr_cond.cprd_sls_oil)" in sql

    def test_violation_conditions_in_having(self):
        sql = build_eur_bounds_sql(
            "ioip",
            "rec_oil",
            "cprd_sls_oil",
            UncertLevel.HIGH,
            UncertLevel.MID,
        )
        assert (
            f"(SUM(fr_cond.rec_oil) + SUM(fr_cond.cprd_sls_oil)) > {TOLERANCE}" in sql
        )
        assert "HAVING" in sql


class TestBuildEurImplicationSql:
    """Unit tests for build_eur_implication_sql."""

    def test_returns_string(self):
        sql = build_eur_implication_sql(
            "igip",
            "rec_con",
            "cprd_sls_con",
            UncertLevel.HIGH,
            UncertLevel.LOW,
        )
        assert isinstance(sql, str)

    def test_self_joins_field_resources(self):
        sql = build_eur_implication_sql(
            "igip",
            "rec_con",
            "cprd_sls_con",
            UncertLevel.HIGH,
            UncertLevel.LOW,
        )
        assert "field_resources fr_result" in sql
        assert "field_resources fr_cond" in sql
        assert "project_resources" not in sql

    def test_both_uncert_levels_in_filter(self):
        sql = build_eur_implication_sql(
            "igip",
            "rec_con",
            "cprd_sls_con",
            UncertLevel.HIGH,
            UncertLevel.LOW,
        )
        assert "fr_cond.uncert_level = '3. High Value'" in sql
        assert "fr_result.uncert_level = '1. Low Value'" in sql

    def test_implication_conditions(self):
        sql = build_eur_implication_sql(
            "ioip",
            "rec_ga",
            "cprd_sls_ga",
            UncertLevel.HIGH,
            UncertLevel.LOW,
        )
        assert f"> {TOLERANCE}" in sql
        assert f"<= {TOLERANCE}" in sql
        assert "HAVING" in sql

    def test_group_by_and_having(self):
        sql = build_eur_implication_sql(
            "igip",
            "rec_con",
            "cprd_sls_con",
            UncertLevel.HIGH,
            UncertLevel.LOW,
        )
        assert "GROUP BY" in sql
        assert "HAVING" in sql

    def test_no_project_class_or_stage_in_join(self):
        sql = build_eur_implication_sql(
            "igip",
            "rec_con",
            "cprd_sls_con",
            UncertLevel.HIGH,
            UncertLevel.LOW,
        )
        assert "project_class" not in sql
        assert "project_stage" not in sql


class TestBuildEurGreaterThanSql:
    """Unit tests for build_eur_greater_than_sql."""

    def test_returns_string(self):
        sql = build_eur_greater_than_sql(
            "ioip",
            "rec_oil",
            "cprd_sls_oil",
            UncertLevel.LOW,
            UncertLevel.LOW,
        )
        assert isinstance(sql, str)

    def test_self_joins_field_resources(self):
        sql = build_eur_greater_than_sql(
            "igip",
            "rec_gn",
            "cprd_sls_gn",
            UncertLevel.LOW,
            UncertLevel.LOW,
        )
        assert "field_resources fr_result" in sql
        assert "field_resources fr_cond" in sql
        assert "project_resources" not in sql

    def test_both_uncert_levels_same(self):
        sql = build_eur_greater_than_sql(
            "ioip",
            "rec_oil",
            "cprd_sls_oil",
            UncertLevel.LOW,
            UncertLevel.LOW,
        )
        assert "fr_result.uncert_level = '1. Low Value'" in sql
        assert "fr_cond.uncert_level = '1. Low Value'" in sql

    def test_group_by_and_having(self):
        sql = build_eur_greater_than_sql(
            "ioip",
            "rec_oil",
            "cprd_sls_oil",
            UncertLevel.LOW,
            UncertLevel.LOW,
        )
        assert "GROUP BY" in sql
        assert "HAVING" in sql

    def test_no_project_class_or_stage_in_join(self):
        sql = build_eur_greater_than_sql(
            "ioip",
            "rec_oil",
            "cprd_sls_oil",
            UncertLevel.LOW,
            UncertLevel.LOW,
        )
        assert "project_class" not in sql
        assert "project_stage" not in sql

    def test_aggregates_with_sum(self):
        sql = build_eur_greater_than_sql(
            "ioip",
            "rec_oil",
            "cprd_sls_oil",
            UncertLevel.LOW,
            UncertLevel.LOW,
        )
        assert "SUM(fr_result.ioip)" in sql
        assert "SUM(fr_cond.rec_oil)" in sql
        assert "SUM(fr_cond.cprd_sls_oil)" in sql

    def test_violation_conditions_in_having(self):
        sql = build_eur_greater_than_sql(
            "ioip",
            "rec_oil",
            "cprd_sls_oil",
            UncertLevel.LOW,
            UncertLevel.LOW,
        )
        assert (
            f"(SUM(fr_cond.rec_oil) + SUM(fr_cond.cprd_sls_oil)) > {TOLERANCE}" in sql
        )
        assert "SUM(fr_result.ioip)" in sql


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _create_re2_test_table(conn: duckdb.DuckDBPyConnection) -> None:
    """Create a project_resources table for RE2 testing."""
    conn.execute("""
        CREATE TABLE project_resources (
            report_year INTEGER,
            project_name TEXT,
            wk_name TEXT,
            field_name TEXT,
            project_id TEXT,
            uncert_level TEXT,
            rec_oil REAL,
            rec_con REAL,
            rec_ga REAL,
            rec_gn REAL,
            res_oil REAL,
            res_con REAL,
            res_ga REAL,
            res_gn REAL,
            cprd_sls_oil REAL,
            cprd_sls_con REAL,
            cprd_sls_ga REAL,
            cprd_sls_gn REAL,
            dcpy_um_oil REAL,
            dcpy_um_con REAL,
            dcpy_um_ga REAL,
            dcpy_um_gn REAL,
            dcpy_ppa_oil REAL,
            dcpy_ppa_con REAL,
            dcpy_ppa_ga REAL,
            dcpy_ppa_gn REAL,
            dcpy_wi_oil REAL,
            dcpy_wi_con REAL,
            dcpy_wi_ga REAL,
            dcpy_wi_gn REAL,
            dcpy_uc_oil REAL,
            dcpy_uc_con REAL,
            dcpy_uc_ga REAL,
            dcpy_uc_gn REAL,
            dcpy_cio_oil REAL,
            dcpy_cio_con REAL,
            dcpy_cio_ga REAL,
            dcpy_cio_gn REAL,
            dcpy_gtr_oil REAL,
            dcpy_gtr_con REAL,
            dcpy_gtr_ga REAL,
            dcpy_gtr_gn REAL
        )
    """)


def _insert_project_row(
    conn: duckdb.DuckDBPyConnection,
    report_year: int,
    project_name: str,
    uncert: str,
    rec_oil: float = 0.0,
    rec_con: float = 0.0,
    rec_ga: float = 0.0,
    rec_gn: float = 0.0,
    res_oil: float = 0.0,
    res_con: float = 0.0,
    res_ga: float = 0.0,
    res_gn: float = 0.0,
    cprd_sls_oil: float = 0.0,
    cprd_sls_con: float = 0.0,
    cprd_sls_ga: float = 0.0,
    cprd_sls_gn: float = 0.0,
    dcpy_um_oil: float = 0.0,
    dcpy_um_con: float = 0.0,
    dcpy_um_ga: float = 0.0,
    dcpy_um_gn: float = 0.0,
    dcpy_ppa_oil: float = 0.0,
    dcpy_ppa_con: float = 0.0,
    dcpy_ppa_ga: float = 0.0,
    dcpy_ppa_gn: float = 0.0,
    dcpy_wi_oil: float = 0.0,
    dcpy_wi_con: float = 0.0,
    dcpy_wi_ga: float = 0.0,
    dcpy_wi_gn: float = 0.0,
    dcpy_uc_oil: float = 0.0,
    dcpy_uc_con: float = 0.0,
    dcpy_uc_ga: float = 0.0,
    dcpy_uc_gn: float = 0.0,
    dcpy_cio_oil: float = 0.0,
    dcpy_cio_con: float = 0.0,
    dcpy_cio_ga: float = 0.0,
    dcpy_cio_gn: float = 0.0,
    dcpy_gtr_oil: float = 0.0,
    dcpy_gtr_con: float = 0.0,
    dcpy_gtr_ga: float = 0.0,
    dcpy_gtr_gn: float = 0.0,
    wk_name: str = "WK1",
    field_name: str = "FLD1",
    project_id: str | None = None,
) -> None:
    if project_id is None:
        project_id = f"P-{hash((report_year, project_name, wk_name)) & 0xFFFFFF:06X}"
    conn.execute(
        "INSERT INTO project_resources VALUES ("
        "?, ?, ?, ?, ?, ?, "
        "?, ?, ?, ?, ?, ?, ?, ?, "
        "?, ?, ?, ?, "
        "?, ?, ?, ?, "
        "?, ?, ?, ?, "
        "?, ?, ?, ?, "
        "?, ?, ?, ?, "
        "?, ?, ?, ?, "
        "?, ?, ?, ?"
        ")",
        [
            report_year,
            project_name,
            wk_name,
            field_name,
            project_id,
            uncert,
            rec_oil,
            rec_con,
            rec_ga,
            rec_gn,
            res_oil,
            res_con,
            res_ga,
            res_gn,
            cprd_sls_oil,
            cprd_sls_con,
            cprd_sls_ga,
            cprd_sls_gn,
            dcpy_um_oil,
            dcpy_um_con,
            dcpy_um_ga,
            dcpy_um_gn,
            dcpy_ppa_oil,
            dcpy_ppa_con,
            dcpy_ppa_ga,
            dcpy_ppa_gn,
            dcpy_wi_oil,
            dcpy_wi_con,
            dcpy_wi_ga,
            dcpy_wi_gn,
            dcpy_uc_oil,
            dcpy_uc_con,
            dcpy_uc_ga,
            dcpy_uc_gn,
            dcpy_cio_oil,
            dcpy_cio_con,
            dcpy_cio_ga,
            dcpy_cio_gn,
            dcpy_gtr_oil,
            dcpy_gtr_con,
            dcpy_gtr_ga,
            dcpy_gtr_gn,
        ],
    )


# ---------------------------------------------------------------------------
# Integration tests for Category A: Material Balance GRR/CR/PR
# ---------------------------------------------------------------------------


class TestMaterialBalanceGRR:
    """Integration tests for RE2001-RE2012 (GRR material balance)."""

    def test_re2001_balanced(self):
        """Material balance satisfied: no violation.

        From docs: 1015 = 1000 + 50 + 30 + 20 + 10 + 5 - (200 - 100)
        ABS(1015 - 1000 - 115 + 200 - 100) = ABS(0) = 0
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            conn = duckdb.connect(f"{tmpdir}/test.duckdb")
            _create_re2_test_table(conn)
            _insert_project_row(
                conn,
                2024,
                "PROJ1",
                "1. Low Value",
                rec_oil=1015,
                cprd_sls_oil=200,
                dcpy_um_oil=50,
                dcpy_ppa_oil=30,
                dcpy_wi_oil=20,
                dcpy_uc_oil=10,
                dcpy_cio_oil=5,
                project_id="P-001",
            )
            _insert_project_row(
                conn,
                2023,
                "PROJ1",
                "1. Low Value",
                rec_oil=1000,
                cprd_sls_oil=100,
                project_id="P-001",
            )

            from esdc.validate.rule_re2 import RE2001

            rule = RE2001()
            violations = rule.check(conn)
            assert len(violations) == 0
            conn.close()

    def test_re2001_unbalanced(self):
        """Material balance violated: rec change doesn't match equation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            conn = duckdb.connect(f"{tmpdir}/test.duckdb")
            _create_re2_test_table(conn)
            _insert_project_row(
                conn,
                2024,
                "PROJ1",
                "1. Low Value",
                rec_oil=900,
                cprd_sls_oil=100,
                project_id="P-001",
            )
            _insert_project_row(
                conn,
                2023,
                "PROJ1",
                "1. Low Value",
                rec_oil=1000,
                cprd_sls_oil=90,
                project_id="P-001",
            )

            from esdc.validate.rule_re2 import RE2001

            rule = RE2001()
            violations = rule.check(conn)
            assert len(violations) == 1
            assert violations[0].rule_id == "RE2001"
            conn.close()

    def test_re2001_with_discrepancies_no_violation(self):
        """Balanced with discrepancies accounted for."""
        with tempfile.TemporaryDirectory() as tmpdir:
            conn = duckdb.connect(f"{tmpdir}/test.duckdb")
            _create_re2_test_table(conn)
            # rec_curr = 1015 = rec_prev(1000) + dcpy_um(50) + dcpy_ppa(30)
            #  + dcpy_wi(20) + dcpy_uc(10) + dcpy_cio(5)
            #  - (cprd_curr(200) - cprd_prev(100))
            # 1015 = 1000 + 50 + 30 + 20 + 10 + 5 - 100 = 1015
            _insert_project_row(
                conn,
                2024,
                "PROJ1",
                "1. Low Value",
                rec_oil=1015,
                cprd_sls_oil=200,
                dcpy_um_oil=50,
                dcpy_ppa_oil=30,
                dcpy_wi_oil=20,
                dcpy_uc_oil=10,
                dcpy_cio_oil=5,
                project_id="P-001",
            )
            _insert_project_row(
                conn,
                2023,
                "PROJ1",
                "1. Low Value",
                rec_oil=1000,
                cprd_sls_oil=100,
                project_id="P-001",
            )

            from esdc.validate.rule_re2 import RE2001

            rule = RE2001()
            violations = rule.check(conn)
            assert len(violations) == 0
            conn.close()

    def test_re2001_with_discrepancies_violation(self):
        """Unbalanced even with discrepancies."""
        with tempfile.TemporaryDirectory() as tmpdir:
            conn = duckdb.connect(f"{tmpdir}/test.duckdb")
            _create_re2_test_table(conn)
            # rec_curr = 900, but expected = 1000 + 115 - 100 = 1015
            _insert_project_row(
                conn,
                2024,
                "PROJ1",
                "1. Low Value",
                rec_oil=900,
                cprd_sls_oil=200,
                dcpy_um_oil=50,
                dcpy_ppa_oil=30,
                dcpy_wi_oil=20,
                dcpy_uc_oil=10,
                dcpy_cio_oil=5,
                project_id="P-001",
            )
            _insert_project_row(
                conn,
                2023,
                "PROJ1",
                "1. Low Value",
                rec_oil=1000,
                cprd_sls_oil=100,
                project_id="P-001",
            )

            from esdc.validate.rule_re2 import RE2001

            rule = RE2001()
            violations = rule.check(conn)
            assert len(violations) == 1
            conn.close()

    def test_no_previous_year_no_violation(self):
        """No previous year row means no comparison possible."""
        with tempfile.TemporaryDirectory() as tmpdir:
            conn = duckdb.connect(f"{tmpdir}/test.duckdb")
            _create_re2_test_table(conn)
            _insert_project_row(
                conn,
                2024,
                "PROJ1",
                "1. Low Value",
                rec_oil=100,
                project_id="P-001",
            )

            from esdc.validate.rule_re2 import RE2001

            rule = RE2001()
            violations = rule.check(conn)
            assert len(violations) == 0
            conn.close()


# ---------------------------------------------------------------------------
# Integration tests for Category B: Material Balance Reserves
# ---------------------------------------------------------------------------


class TestMaterialBalanceReserves:
    """Integration tests for RE2013-RE2024 (Reserves material balance)."""

    def test_re2013_balanced(self):
        """Reserves balance satisfied with dcpy_gtr."""
        with tempfile.TemporaryDirectory() as tmpdir:
            conn = duckdb.connect(f"{tmpdir}/test.duckdb")
            _create_re2_test_table(conn)
            # res_curr(1100) = res_prev(1000) + dcpy_gtr(200) - (200 - 100)
            _insert_project_row(
                conn,
                2024,
                "PROJ1",
                "1. Low Value",
                res_oil=1100,
                cprd_sls_oil=200,
                dcpy_gtr_oil=200,
                project_id="P-001",
            )
            _insert_project_row(
                conn,
                2023,
                "PROJ1",
                "1. Low Value",
                res_oil=1000,
                cprd_sls_oil=100,
                project_id="P-001",
            )

            from esdc.validate.rule_re2 import RE2013

            rule = RE2013()
            violations = rule.check(conn)
            assert len(violations) == 0
            conn.close()

    def test_re2013_unbalanced(self):
        """Reserves balance violated."""
        with tempfile.TemporaryDirectory() as tmpdir:
            conn = duckdb.connect(f"{tmpdir}/test.duckdb")
            _create_re2_test_table(conn)
            _insert_project_row(
                conn,
                2024,
                "PROJ1",
                "1. Low Value",
                res_oil=1000,
                cprd_sls_oil=200,
                dcpy_gtr_oil=200,
                project_id="P-001",
            )
            _insert_project_row(
                conn,
                2023,
                "PROJ1",
                "1. Low Value",
                res_oil=1000,
                cprd_sls_oil=100,
                project_id="P-001",
            )

            from esdc.validate.rule_re2 import RE2013

            rule = RE2013()
            violations = rule.check(conn)
            assert len(violations) == 1
            assert violations[0].rule_id == "RE2013"
            conn.close()


# ---------------------------------------------------------------------------
# Tolerance boundary tests
# ---------------------------------------------------------------------------


class TestToleranceBoundary:
    """Test that TOLERANCE boundary is correctly respected."""

    def test_material_balance_within_tolerance(self):
        """Difference within tolerance => no violation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            conn = duckdb.connect(f"{tmpdir}/test.duckdb")
            _create_re2_test_table(conn)
            diff = TOLERANCE / 2
            _insert_project_row(
                conn,
                2024,
                "PROJ1",
                "1. Low Value",
                res_oil=1000 + diff,
                cprd_sls_oil=100,
                project_id="P-001",
            )
            _insert_project_row(
                conn,
                2023,
                "PROJ1",
                "1. Low Value",
                res_oil=1000,
                cprd_sls_oil=100,
                project_id="P-001",
            )

            from esdc.validate.rule_re2 import RE2013

            rule = RE2013()
            violations = rule.check(conn)
            assert len(violations) == 0
            conn.close()

    def test_material_balance_at_tolerance_boundary(self):
        """Difference exactly at tolerance => no violation (< not <=)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            conn = duckdb.connect(f"{tmpdir}/test.duckdb")
            _create_re2_test_table(conn)
            _insert_project_row(
                conn,
                2024,
                "PROJ1",
                "1. Low Value",
                res_oil=1000 + TOLERANCE,
                cprd_sls_oil=100,
                project_id="P-001",
            )
            _insert_project_row(
                conn,
                2023,
                "PROJ1",
                "1. Low Value",
                res_oil=1000,
                cprd_sls_oil=100,
                project_id="P-001",
            )

            from esdc.validate.rule_re2 import RE2013

            rule = RE2013()
            violations = rule.check(conn)
            assert len(violations) == 0
            conn.close()

    def test_material_balance_above_tolerance(self):
        """Difference above tolerance => violation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            conn = duckdb.connect(f"{tmpdir}/test.duckdb")
            _create_re2_test_table(conn)
            diff = TOLERANCE + 0.0001
            _insert_project_row(
                conn,
                2024,
                "PROJ1",
                "1. Low Value",
                res_oil=1000 + diff,
                cprd_sls_oil=100,
                project_id="P-001",
            )
            _insert_project_row(
                conn,
                2023,
                "PROJ1",
                "1. Low Value",
                res_oil=1000,
                cprd_sls_oil=100,
                project_id="P-001",
            )

            from esdc.validate.rule_re2 import RE2013

            rule = RE2013()
            violations = rule.check(conn)
            assert len(violations) == 1
            conn.close()


# ---------------------------------------------------------------------------
# Integration tests for Category C: EUR rules
# ---------------------------------------------------------------------------


def _create_re2_field_test_table(conn: duckdb.DuckDBPyConnection) -> None:
    """Create a field_resources table for Category C testing."""
    conn.execute("""
        CREATE TABLE field_resources (
            report_year INTEGER,
            wk_name TEXT,
            field_name TEXT,
            project_stage TEXT,
            project_class TEXT,
            wk_id TEXT,
            field_id TEXT,
            uncert_level TEXT,
            ioip REAL,
            igip REAL,
            rec_oil REAL,
            rec_con REAL,
            rec_ga REAL,
            rec_gn REAL,
            cprd_sls_oil REAL,
            cprd_sls_con REAL,
            cprd_sls_ga REAL,
            cprd_sls_gn REAL
        )
    """)


def _insert_field_resource_row(
    conn: duckdb.DuckDBPyConnection,
    report_year: int,
    wk_name: str,
    field_name: str,
    uncert: str,
    ioip: float = 0.0,
    igip: float = 0.0,
    rec_oil: float = 0.0,
    rec_con: float = 0.0,
    rec_ga: float = 0.0,
    rec_gn: float = 0.0,
    cprd_sls_oil: float = 0.0,
    cprd_sls_con: float = 0.0,
    cprd_sls_ga: float = 0.0,
    cprd_sls_gn: float = 0.0,
    project_stage: str = "DEV",
    project_class: str = "COMM",
    wk_id: str | None = None,
    field_id: str | None = None,
) -> None:
    if wk_id is None:
        wk_id = f"WK-{wk_name}"
    if field_id is None:
        field_id = f"FLD-{field_name}"
    conn.execute(
        "INSERT INTO field_resources VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        [
            report_year,
            wk_name,
            field_name,
            project_stage,
            project_class,
            wk_id,
            field_id,
            uncert,
            ioip,
            igip,
            rec_oil,
            rec_con,
            rec_ga,
            rec_gn,
            cprd_sls_oil,
            cprd_sls_con,
            cprd_sls_ga,
            cprd_sls_gn,
        ],
    )


class TestEurBounds:
    """Integration tests for RE2025/RE2026 (EUR bounds)."""

    def test_re2025_no_violation(self):
        """IOIP Mid total > Oil EUR(P10) total => no violation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            conn = duckdb.connect(f"{tmpdir}/test.duckdb")
            _create_re2_field_test_table(conn)
            _insert_field_resource_row(
                conn,
                2024,
                "WK1",
                "FLD1",
                "2. Middle Value",
                ioip=1500,
            )
            _insert_field_resource_row(
                conn,
                2024,
                "WK1",
                "FLD1",
                "3. High Value",
                rec_oil=800,
                cprd_sls_oil=200,
            )

            from esdc.validate.rule_re2 import RE2025

            rule = RE2025()
            violations = rule.check(conn)
            assert len(violations) == 0
            conn.close()

    def test_re2025_violation(self):
        """IOIP Mid total <= Oil EUR(P10) total => violation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            conn = duckdb.connect(f"{tmpdir}/test.duckdb")
            _create_re2_field_test_table(conn)
            _insert_field_resource_row(
                conn,
                2024,
                "WK1",
                "FLD1",
                "2. Middle Value",
                ioip=800,
            )
            _insert_field_resource_row(
                conn,
                2024,
                "WK1",
                "FLD1",
                "3. High Value",
                rec_oil=800,
                cprd_sls_oil=200,
            )

            from esdc.validate.rule_re2 import RE2025

            rule = RE2025()
            violations = rule.check(conn)
            assert len(violations) == 1
            assert violations[0].rule_id == "RE2025"
            conn.close()

    def test_re2025_no_eur_no_violation(self):
        """EUR = 0 => no trigger, no violation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            conn = duckdb.connect(f"{tmpdir}/test.duckdb")
            _create_re2_field_test_table(conn)
            _insert_field_resource_row(
                conn,
                2024,
                "WK1",
                "FLD1",
                "2. Middle Value",
                ioip=0,
            )
            _insert_field_resource_row(
                conn,
                2024,
                "WK1",
                "FLD1",
                "3. High Value",
                rec_oil=0,
                cprd_sls_oil=0,
            )

            from esdc.validate.rule_re2 import RE2025

            rule = RE2025()
            violations = rule.check(conn)
            assert len(violations) == 0
            conn.close()

    def test_re2025_ignores_wrong_uncert_level(self):
        """Cond side at MID (not HIGH) => no match => no violation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            conn = duckdb.connect(f"{tmpdir}/test.duckdb")
            _create_re2_field_test_table(conn)
            _insert_field_resource_row(
                conn,
                2024,
                "WK1",
                "FLD1",
                "2. Middle Value",
                ioip=800,
            )
            _insert_field_resource_row(
                conn,
                2024,
                "WK1",
                "FLD1",
                "2. Middle Value",
                rec_oil=800,
                cprd_sls_oil=200,
            )

            from esdc.validate.rule_re2 import RE2025

            rule = RE2025()
            violations = rule.check(conn)
            assert len(violations) == 0
            conn.close()

    def test_re2025_multi_class_sums_eur_across_project_classes(self):
        """RE2025: IOIP Mid aggregates across project_class and project_stage.

        Two project_classes each contribute to EUR. Total EUR = 600+400=1000.
        Total IOIP Mid = 900+600=1500 > 1000 => no violation.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            conn = duckdb.connect(f"{tmpdir}/test.duckdb")
            _create_re2_field_test_table(conn)
            _insert_field_resource_row(
                conn,
                2024,
                "WK1",
                "FLD1",
                "2. Middle Value",
                ioip=900,
                project_class="COMM",
                project_stage="DEV",
            )
            _insert_field_resource_row(
                conn,
                2024,
                "WK1",
                "FLD1",
                "2. Middle Value",
                ioip=600,
                project_class="CONT",
                project_stage="EXP",
            )
            _insert_field_resource_row(
                conn,
                2024,
                "WK1",
                "FLD1",
                "3. High Value",
                rec_oil=400,
                cprd_sls_oil=200,
                project_class="COMM",
                project_stage="DEV",
            )
            _insert_field_resource_row(
                conn,
                2024,
                "WK1",
                "FLD1",
                "3. High Value",
                rec_oil=300,
                cprd_sls_oil=100,
                project_class="CONT",
                project_stage="EXP",
            )

            from esdc.validate.rule_re2 import RE2025

            rule = RE2025()
            violations = rule.check(conn)
            assert len(violations) == 0
            conn.close()

    def test_re2025_multi_class_violation_when_total_eur_exceeds_ioip(self):
        """RE2025: Total EUR across classes exceeds total IOIP => violation.

        COMM: ioip=300, rec=400, cprd=100 => eur=500
        CONT: ioip=300, rec=400, cprd=100 => eur=500
        Total IOIP = 600, Total EUR = 1000 => violation.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            conn = duckdb.connect(f"{tmpdir}/test.duckdb")
            _create_re2_field_test_table(conn)
            _insert_field_resource_row(
                conn,
                2024,
                "WK1",
                "FLD1",
                "2. Middle Value",
                ioip=300,
                project_class="COMM",
                project_stage="DEV",
            )
            _insert_field_resource_row(
                conn,
                2024,
                "WK1",
                "FLD1",
                "2. Middle Value",
                ioip=300,
                project_class="CONT",
                project_stage="EXP",
            )
            _insert_field_resource_row(
                conn,
                2024,
                "WK1",
                "FLD1",
                "3. High Value",
                rec_oil=400,
                cprd_sls_oil=100,
                project_class="COMM",
                project_stage="DEV",
            )
            _insert_field_resource_row(
                conn,
                2024,
                "WK1",
                "FLD1",
                "3. High Value",
                rec_oil=400,
                cprd_sls_oil=100,
                project_class="CONT",
                project_stage="EXP",
            )

            from esdc.validate.rule_re2 import RE2025

            rule = RE2025()
            violations = rule.check(conn)
            assert len(violations) == 1
            assert violations[0].rule_id == "RE2025"
            conn.close()

    def test_re2026_no_violation(self):
        """IGIP Mid total > NAG EUR(P10) total => no violation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            conn = duckdb.connect(f"{tmpdir}/test.duckdb")
            _create_re2_field_test_table(conn)
            _insert_field_resource_row(
                conn,
                2024,
                "WK1",
                "FLD1",
                "2. Middle Value",
                igip=1500,
            )
            _insert_field_resource_row(
                conn,
                2024,
                "WK1",
                "FLD1",
                "3. High Value",
                rec_gn=800,
                cprd_sls_gn=200,
            )

            from esdc.validate.rule_re2 import RE2026

            rule = RE2026()
            violations = rule.check(conn)
            assert len(violations) == 0
            conn.close()

    def test_re2026_violation(self):
        """IGIP Mid total <= NAG EUR(P10) total => violation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            conn = duckdb.connect(f"{tmpdir}/test.duckdb")
            _create_re2_field_test_table(conn)
            _insert_field_resource_row(
                conn,
                2024,
                "WK1",
                "FLD1",
                "2. Middle Value",
                igip=800,
            )
            _insert_field_resource_row(
                conn,
                2024,
                "WK1",
                "FLD1",
                "3. High Value",
                rec_gn=800,
                cprd_sls_gn=200,
            )

            from esdc.validate.rule_re2 import RE2026

            rule = RE2026()
            violations = rule.check(conn)
            assert len(violations) == 1
            assert violations[0].rule_id == "RE2026"
            conn.close()


class TestEurImplication:
    """Integration tests for RE2027/RE2028 (EUR implication)."""

    def test_re2027_no_violation(self):
        """Condensate EUR total > 0 and IGIP Low total > 0 => no violation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            conn = duckdb.connect(f"{tmpdir}/test.duckdb")
            _create_re2_field_test_table(conn)
            _insert_field_resource_row(
                conn,
                2024,
                "WK1",
                "FLD1",
                "1. Low Value",
                igip=500,
            )
            _insert_field_resource_row(
                conn,
                2024,
                "WK1",
                "FLD1",
                "3. High Value",
                rec_con=500,
                cprd_sls_con=500,
            )

            from esdc.validate.rule_re2 import RE2027

            rule = RE2027()
            violations = rule.check(conn)
            assert len(violations) == 0
            conn.close()

    def test_re2027_violation(self):
        """Condensate EUR total > 0 and IGIP Low total = 0 => violation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            conn = duckdb.connect(f"{tmpdir}/test.duckdb")
            _create_re2_field_test_table(conn)
            _insert_field_resource_row(
                conn,
                2024,
                "WK1",
                "FLD1",
                "1. Low Value",
                igip=0,
            )
            _insert_field_resource_row(
                conn,
                2024,
                "WK1",
                "FLD1",
                "3. High Value",
                rec_con=500,
                cprd_sls_con=500,
            )

            from esdc.validate.rule_re2 import RE2027

            rule = RE2027()
            violations = rule.check(conn)
            assert len(violations) == 1
            assert violations[0].rule_id == "RE2027"
            conn.close()

    def test_re2027_multi_class_sums_eur(self):
        """RE2027: EUR is summed across project_class and project_stage.

        COMM: rec_con=300, cprd_sls_con=200 => EUR=500
        CONT: rec_con=200, cprd_sls_con=100 => EUR=300
        Total EUR = 800, IGIP Low = 500 => no violation.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            conn = duckdb.connect(f"{tmpdir}/test.duckdb")
            _create_re2_field_test_table(conn)
            _insert_field_resource_row(
                conn,
                2024,
                "WK1",
                "FLD1",
                "1. Low Value",
                igip=500,
                project_class="COMM",
                project_stage="DEV",
            )
            _insert_field_resource_row(
                conn,
                2024,
                "WK1",
                "FLD1",
                "3. High Value",
                rec_con=300,
                cprd_sls_con=200,
                project_class="COMM",
                project_stage="DEV",
            )
            _insert_field_resource_row(
                conn,
                2024,
                "WK1",
                "FLD1",
                "3. High Value",
                rec_con=200,
                cprd_sls_con=100,
                project_class="CONT",
                project_stage="EXP",
            )

            from esdc.validate.rule_re2 import RE2027

            rule = RE2027()
            violations = rule.check(conn)
            assert len(violations) == 0
            conn.close()

    def test_re2028_no_violation(self):
        """Assoc Gas EUR total > 0 and IOIP Low total > 0 => no violation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            conn = duckdb.connect(f"{tmpdir}/test.duckdb")
            _create_re2_field_test_table(conn)
            _insert_field_resource_row(
                conn,
                2024,
                "WK1",
                "FLD1",
                "1. Low Value",
                ioip=500,
            )
            _insert_field_resource_row(
                conn,
                2024,
                "WK1",
                "FLD1",
                "3. High Value",
                rec_ga=500,
                cprd_sls_ga=500,
            )

            from esdc.validate.rule_re2 import RE2028

            rule = RE2028()
            violations = rule.check(conn)
            assert len(violations) == 0
            conn.close()

    def test_re2028_violation(self):
        """Assoc Gas EUR total > 0 and IOIP Low total = 0 => violation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            conn = duckdb.connect(f"{tmpdir}/test.duckdb")
            _create_re2_field_test_table(conn)
            _insert_field_resource_row(
                conn,
                2024,
                "WK1",
                "FLD1",
                "1. Low Value",
                ioip=0,
            )
            _insert_field_resource_row(
                conn,
                2024,
                "WK1",
                "FLD1",
                "3. High Value",
                rec_ga=500,
                cprd_sls_ga=500,
            )

            from esdc.validate.rule_re2 import RE2028

            rule = RE2028()
            violations = rule.check(conn)
            assert len(violations) == 1
            assert violations[0].rule_id == "RE2028"
            conn.close()


class TestEurGreaterThan:
    """Integration tests for RE2029/RE2030 (EUR greater than)."""

    def test_re2029_no_violation(self):
        """IOIP Low total > Oil EUR(P90) total => no violation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            conn = duckdb.connect(f"{tmpdir}/test.duckdb")
            _create_re2_field_test_table(conn)
            _insert_field_resource_row(
                conn,
                2024,
                "WK1",
                "FLD1",
                "1. Low Value",
                ioip=2000,
                rec_oil=500,
                cprd_sls_oil=500,
            )

            from esdc.validate.rule_re2 import RE2029

            rule = RE2029()
            violations = rule.check(conn)
            assert len(violations) == 0
            conn.close()

    def test_re2029_violation(self):
        """IOIP Low total <= Oil EUR(P90) total => violation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            conn = duckdb.connect(f"{tmpdir}/test.duckdb")
            _create_re2_field_test_table(conn)
            _insert_field_resource_row(
                conn,
                2024,
                "WK1",
                "FLD1",
                "1. Low Value",
                ioip=1000,
                rec_oil=500,
                cprd_sls_oil=500,
            )

            from esdc.validate.rule_re2 import RE2029

            rule = RE2029()
            violations = rule.check(conn)
            assert len(violations) == 1
            assert violations[0].rule_id == "RE2029"
            conn.close()

    def test_re2029_multi_class_sums_across_project_classes(self):
        """RE2029: IOIP Low and EUR are summed across project_class and project_stage.

        COMM DEV: ioip=800, rec_oil=300, cprd_sls_oil=100 => EUR=400
        CONT EXP: ioip=800, rec_oil=300, cprd_sls_oil=200 => EUR=500
        Total ioip = 1600, Total EUR = 900 => ioip > EUR => no violation.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            conn = duckdb.connect(f"{tmpdir}/test.duckdb")
            _create_re2_field_test_table(conn)
            _insert_field_resource_row(
                conn,
                2024,
                "WK1",
                "FLD1",
                "1. Low Value",
                ioip=800,
                rec_oil=300,
                cprd_sls_oil=100,
                project_class="COMM",
                project_stage="DEV",
            )
            _insert_field_resource_row(
                conn,
                2024,
                "WK1",
                "FLD1",
                "1. Low Value",
                ioip=800,
                rec_oil=300,
                cprd_sls_oil=200,
                project_class="CONT",
                project_stage="EXP",
            )

            from esdc.validate.rule_re2 import RE2029

            rule = RE2029()
            violations = rule.check(conn)
            assert len(violations) == 0
            conn.close()

    def test_re2029_multi_class_violation(self):
        """RE2029: Multi-class total EUR exceeds total ioip => violation.

        COMM DEV: ioip=300, rec_oil=400, cprd_sls_oil=100 => EUR=500
        CONT EXP: ioip=300, rec_oil=400, cprd_sls_oil=100 => EUR=500
        Total ioip = 600, Total EUR = 1000 => violation.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            conn = duckdb.connect(f"{tmpdir}/test.duckdb")
            _create_re2_field_test_table(conn)
            _insert_field_resource_row(
                conn,
                2024,
                "WK1",
                "FLD1",
                "1. Low Value",
                ioip=300,
                rec_oil=400,
                cprd_sls_oil=100,
                project_class="COMM",
                project_stage="DEV",
            )
            _insert_field_resource_row(
                conn,
                2024,
                "WK1",
                "FLD1",
                "1. Low Value",
                ioip=300,
                rec_oil=400,
                cprd_sls_oil=100,
                project_class="CONT",
                project_stage="EXP",
            )

            from esdc.validate.rule_re2 import RE2029

            rule = RE2029()
            violations = rule.check(conn)
            assert len(violations) == 1
            assert violations[0].rule_id == "RE2029"
            conn.close()

    def test_re2030_no_violation(self):
        """IGIP Low total > NAG EUR(P90) total => no violation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            conn = duckdb.connect(f"{tmpdir}/test.duckdb")
            _create_re2_field_test_table(conn)
            _insert_field_resource_row(
                conn,
                2024,
                "WK1",
                "FLD1",
                "1. Low Value",
                igip=2000,
                rec_gn=500,
                cprd_sls_gn=500,
            )

            from esdc.validate.rule_re2 import RE2030

            rule = RE2030()
            violations = rule.check(conn)
            assert len(violations) == 0
            conn.close()

    def test_re2030_violation(self):
        """IGIP Low total <= NAG EUR(P90) total => violation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            conn = duckdb.connect(f"{tmpdir}/test.duckdb")
            _create_re2_field_test_table(conn)
            _insert_field_resource_row(
                conn,
                2024,
                "WK1",
                "FLD1",
                "1. Low Value",
                igip=1000,
                rec_gn=500,
                cprd_sls_gn=500,
            )

            from esdc.validate.rule_re2 import RE2030

            rule = RE2030()
            violations = rule.check(conn)
            assert len(violations) == 1
            assert violations[0].rule_id == "RE2030"
            conn.close()


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------


class TestRe2Registry:
    """Test that all RE2 rules are registered."""

    def test_all_re2_rules_registered(self):
        re2_rules = get_rules_by_group("RE2")
        rule_ids = sorted(r.rule_id for r in re2_rules)
        expected = [
            "RE2001",
            "RE2002",
            "RE2003",
            "RE2004",
            "RE2005",
            "RE2006",
            "RE2007",
            "RE2008",
            "RE2009",
            "RE2010",
            "RE2011",
            "RE2012",
            "RE2013",
            "RE2014",
            "RE2015",
            "RE2016",
            "RE2017",
            "RE2018",
            "RE2019",
            "RE2020",
            "RE2021",
            "RE2022",
            "RE2023",
            "RE2024",
            "RE2025",
            "RE2026",
            "RE2027",
            "RE2028",
            "RE2029",
            "RE2030",
        ]
        assert rule_ids == expected

    def test_total_rule_count_increased(self):
        import esdc.validate.rule_re0  # noqa: F401
        import esdc.validate.rule_re1  # noqa: F401
        import esdc.validate.rule_re2  # noqa: F401
        import esdc.validate.rule_re9  # noqa: F401

        all_rules = get_all_rules()
        re2_count = len(get_rules_by_group("RE2"))
        re1_count = len(get_rules_by_group("RE1"))
        re0_count = len(get_rules_by_group("RE0"))
        re5_count = len(get_rules_by_group("RE5"))
        re9_count = len(get_rules_by_group("RE9"))
        assert re2_count == 30
        assert re1_count == 32
        assert re0_count == 66
        assert re9_count == 1
        assert len(all_rules) == re0_count + re1_count + re2_count + re5_count + re9_count

    def test_re2_rules_have_correct_group(self):
        import esdc.validate.rule_re2  # noqa: F401
        from esdc.validate.rule_re2 import RE2001, RE2025

        # rule_group is not a class attribute;
        # it's set via _execute_and_build_violations
        # Check that rules can be looked up via registry with RE2 prefix
        re2_rules = get_rules_by_group("RE2")
        assert len(re2_rules) == 30
        assert RE2001().rule_id.startswith("RE2")
        assert RE2025().rule_id.startswith("RE2")

    def test_re2025_re2026_are_warning_severity(self):
        from esdc.selection import Severity
        from esdc.validate.rule_re2 import RE2025, RE2026

        assert RE2025().severity == Severity.WARNING
        assert RE2026().severity == Severity.WARNING

    def test_re2027_to_re2030_are_strict_severity(self):
        from esdc.selection import Severity
        from esdc.validate.rule_re2 import RE2027, RE2028, RE2029, RE2030

        assert RE2027().severity == Severity.STRICT
        assert RE2028().severity == Severity.STRICT
        assert RE2029().severity == Severity.STRICT
        assert RE2030().severity == Severity.STRICT

    def test_all_re2_rules_not_fixable(self):
        re2_rules = get_rules_by_group("RE2")
        for rule_cls in re2_rules:
            rule = rule_cls()
            assert rule.is_fixable is False
