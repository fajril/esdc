"""Comprehensive tests for RE1 (Production and Forecast) validation rules."""

from __future__ import annotations

import tempfile
from pathlib import Path

import duckdb

from esdc.validate.rule_re0_helpers import UncertLevel
from esdc.validate.rule_re1_helpers import (
    build_forecast_sum_equals_reserve_sql,
    build_forecast_sum_equals_resource_sql,
    build_monotonic_sql,
    build_timeseries_sales_le_tpf_sql,
)
from esdc.validate.rules import TOLERANCE, get_all_rules, get_rules_by_group

# ---------------------------------------------------------------------------
# SQL builder unit tests
# ---------------------------------------------------------------------------


class TestBuildMonotonicSql:
    """Unit tests for build_monotonic_sql."""

    def test_returns_string(self):
        sql = build_monotonic_sql("cprd_grs_oil", UncertLevel.MID)
        assert isinstance(sql, str)

    def test_contains_both_column_aliases(self):
        sql = build_monotonic_sql("cprd_grs_oil", UncertLevel.MID)
        assert "l.cprd_grs_oil AS val_ref" in sql
        assert "h.cprd_grs_oil AS val_cmp" in sql

    def test_self_join_on_project_id(self):
        sql = build_monotonic_sql("cprd_grs_con", UncertLevel.MID)
        assert "l.project_id = h.project_id" in sql

    def test_joins_on_consecutive_years(self):
        sql = build_monotonic_sql("cprd_grs_ga", UncertLevel.MID)
        assert "l.report_year = h.report_year + 1" in sql

    def test_uncert_level_filter_on_both_sides(self):
        sql = build_monotonic_sql("cprd_grs_gn", UncertLevel.MID)
        assert "l.uncert_level = '2. Middle Value'" in sql
        assert "h.uncert_level = '2. Middle Value'" in sql

    def test_violation_condition(self):
        sql = build_monotonic_sql("cprd_sls_oil", UncertLevel.MID)
        assert (
            f"COALESCE(h.cprd_sls_oil, 0) - COALESCE(l.cprd_sls_oil, 0) > {TOLERANCE}"
            in sql
        )

    def test_includes_identifier_columns(self):
        sql = build_monotonic_sql("cprd_grs_oil", UncertLevel.MID)
        assert "l.report_year" in sql
        assert "l.project_name" in sql


class TestBuildTimeseriesSalesLeTpfSql:
    """Unit tests for build_timeseries_sales_le_tpf_sql."""

    def test_returns_string(self):
        sql = build_timeseries_sales_le_tpf_sql("slf_oil", "tpf_oil")
        assert isinstance(sql, str)

    def test_contains_both_columns(self):
        sql = build_timeseries_sales_le_tpf_sql("slf_con", "tpf_con")
        assert "slf_con AS val_ref" in sql
        assert "tpf_con AS val_cmp" in sql

    def test_from_project_timeseries(self):
        sql = build_timeseries_sales_le_tpf_sql("slf_ga", "tpf_ga")
        assert "FROM project_timeseries" in sql

    def test_violation_condition(self):
        sql = build_timeseries_sales_le_tpf_sql("slf_gn", "tpf_gn")
        assert f"COALESCE(slf_gn, 0) - COALESCE(tpf_gn, 0) > {TOLERANCE}" in sql

    def test_includes_year_column(self):
        sql = build_timeseries_sales_le_tpf_sql("slf_oil", "tpf_oil")
        assert "year" in sql


class TestBuildForecastSumEqualsReserveSql:
    """Unit tests for build_forecast_sum_equals_reserve_sql."""

    def test_returns_string(self):
        sql = build_forecast_sum_equals_reserve_sql(
            "slf_oil", "res_oil", UncertLevel.MID
        )
        assert isinstance(sql, str)

    def test_joins_timeseries_and_resources(self):
        sql = build_forecast_sum_equals_reserve_sql(
            "slf_con", "res_con", UncertLevel.MID
        )
        assert "project_timeseries ts" in sql
        assert "project_resources pr" in sql

    def test_join_conditions(self):
        sql = build_forecast_sum_equals_reserve_sql("slf_ga", "res_ga", UncertLevel.MID)
        assert "pr.project_id = ts.project_id" in sql
        assert "pr.report_year = ts.report_year" in sql
        assert "ts.year > pr.report_year" in sql

    def test_uncert_level_filter(self):
        sql = build_forecast_sum_equals_reserve_sql("slf_gn", "res_gn", UncertLevel.MID)
        assert "pr.uncert_level = '2. Middle Value'" in sql

    def test_having_clause_with_tolerance(self):
        sql = build_forecast_sum_equals_reserve_sql(
            "slf_oil", "res_oil", UncertLevel.MID
        )
        assert (
            f"HAVING ABS(SUM(COALESCE(ts.slf_oil, 0))"
            f" - COALESCE(pr.res_oil, 0)) > {TOLERANCE}" in sql
        )

    def test_group_by_clause(self):
        sql = build_forecast_sum_equals_reserve_sql(
            "slf_con", "res_con", UncertLevel.MID
        )
        assert "GROUP BY" in sql
        assert "pr.report_year" in sql
        assert "pr.project_name" in sql


class TestBuildForecastSumEqualsResourceSql:
    """Unit tests for build_forecast_sum_equals_resource_sql."""

    def test_returns_string(self):
        sql = build_forecast_sum_equals_resource_sql(
            "tpf_oil", "rec_oil", UncertLevel.MID
        )
        assert isinstance(sql, str)

    def test_joins_timeseries_and_resources(self):
        sql = build_forecast_sum_equals_resource_sql(
            "tpf_con", "rec_con", UncertLevel.MID
        )
        assert "project_timeseries ts" in sql
        assert "project_resources pr" in sql

    def test_having_clause_with_tolerance(self):
        sql = build_forecast_sum_equals_resource_sql(
            "tpf_ga", "rec_ga", UncertLevel.MID, tolerance=TOLERANCE
        )
        assert (
            f"HAVING ABS(SUM(COALESCE(ts.tpf_ga, 0))"
            f" - COALESCE(pr.rec_ga, 0)) > {TOLERANCE}" in sql
        )


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


def _create_re1_test_table(conn: duckdb.DuckDBPyConnection) -> None:
    """Create a project_resources table with cumprod columns for RE1 testing."""
    conn.execute("""
        CREATE TABLE project_resources (
            report_year INTEGER,
            project_name TEXT,
            wk_name TEXT,
            field_name TEXT,
            project_id TEXT,
            uncert_level TEXT,
            cprd_grs_oil REAL,
            cprd_grs_con REAL,
            cprd_grs_ga REAL,
            cprd_grs_gn REAL,
            cprd_sls_oil REAL,
            cprd_sls_con REAL,
            cprd_sls_ga REAL,
            cprd_sls_gn REAL,
            res_oil REAL,
            res_con REAL,
            res_ga REAL,
            res_gn REAL,
            rec_oil REAL,
            rec_con REAL,
            rec_ga REAL,
            rec_gn REAL
        )
    """)


def _create_timeseries_test_table(conn: duckdb.DuckDBPyConnection) -> None:
    """Create a project_timeseries table for RE1 testing."""
    conn.execute("""
        CREATE TABLE project_timeseries (
            report_year INTEGER,
            project_name TEXT,
            wk_name TEXT,
            field_name TEXT,
            project_id TEXT,
            year INTEGER,
            slf_oil REAL,
            slf_con REAL,
            slf_ga REAL,
            slf_gn REAL,
            tpf_oil REAL,
            tpf_con REAL,
            tpf_ga REAL,
            tpf_gn REAL
        )
    """)


def _insert_project_row(
    conn: duckdb.DuckDBPyConnection,
    report_year: int,
    project_name: str,
    uncert: str,
    cprd_grs_oil: float = 0.0,
    cprd_grs_con: float = 0.0,
    cprd_grs_ga: float = 0.0,
    cprd_grs_gn: float = 0.0,
    cprd_sls_oil: float = 0.0,
    cprd_sls_con: float = 0.0,
    cprd_sls_ga: float = 0.0,
    cprd_sls_gn: float = 0.0,
    res_oil: float = 0.0,
    res_con: float = 0.0,
    res_ga: float = 0.0,
    res_gn: float = 0.0,
    rec_oil: float = 0.0,
    rec_con: float = 0.0,
    rec_ga: float = 0.0,
    rec_gn: float = 0.0,
    wk_name: str = "WK1",
    field_name: str = "FLD1",
    project_id: str | None = None,
) -> None:
    if project_id is None:
        project_id = f"P-{hash((report_year, project_name, wk_name)) & 0xFFFFFF:06X}"
    conn.execute(
        "INSERT INTO project_resources"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            report_year,
            project_name,
            wk_name,
            field_name,
            project_id,
            uncert,
            cprd_grs_oil,
            cprd_grs_con,
            cprd_grs_ga,
            cprd_grs_gn,
            cprd_sls_oil,
            cprd_sls_con,
            cprd_sls_ga,
            cprd_sls_gn,
            res_oil,
            res_con,
            res_ga,
            res_gn,
            rec_oil,
            rec_con,
            rec_ga,
            rec_gn,
        ],
    )


def _insert_timeseries_row(
    conn: duckdb.DuckDBPyConnection,
    report_year: int,
    project_name: str,
    year: int,
    slf_oil: float = 0.0,
    slf_con: float = 0.0,
    slf_ga: float = 0.0,
    slf_gn: float = 0.0,
    tpf_oil: float = 0.0,
    tpf_con: float = 0.0,
    tpf_ga: float = 0.0,
    tpf_gn: float = 0.0,
    wk_name: str = "WK1",
    field_name: str = "FLD1",
    project_id: str | None = None,
) -> None:
    if project_id is None:
        project_id = f"P-{hash((report_year, project_name, wk_name)) & 0xFFFFFF:06X}"
    conn.execute(
        "INSERT INTO project_timeseries"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            report_year,
            project_name,
            wk_name,
            field_name,
            project_id,
            year,
            slf_oil,
            slf_con,
            slf_ga,
            slf_gn,
            tpf_oil,
            tpf_con,
            tpf_ga,
            tpf_gn,
        ],
    )


# ---------------------------------------------------------------------------
# Category A integration tests
# ---------------------------------------------------------------------------


class TestCategoryANonNegative:
    """Integration tests for RE1001-RE1004 and RE1009-RE1012."""

    def test_re1001_finds_negative_cumprod(self, tmp_path: Path):
        with tempfile.TemporaryDirectory() as tmpdir:
            conn = duckdb.connect(f"{tmpdir}/test.duckdb")
            _create_re1_test_table(conn)
            _insert_project_row(
                conn,
                report_year=2024,
                project_name="PROJ1",
                uncert="2. Middle Value",
                cprd_grs_oil=-10.0,
            )
            from esdc.validate.rule_re1 import RE1001

            rule = RE1001()
            violations = rule.check(conn)
            assert len(violations) == 1
            assert violations[0].rule_id == "RE1001"
            assert violations[0].validated_column == "cprd_grs_oil"
            assert violations[0].current_values["cprd_grs_oil"] == -10.0
            conn.close()

    def test_re1001_no_violations_when_positive(self, tmp_path: Path):
        with tempfile.TemporaryDirectory() as tmpdir:
            conn = duckdb.connect(f"{tmpdir}/test.duckdb")
            _create_re1_test_table(conn)
            _insert_project_row(
                conn,
                report_year=2024,
                project_name="PROJ1",
                uncert="2. Middle Value",
                cprd_grs_oil=100.0,
            )
            from esdc.validate.rule_re1 import RE1001

            rule = RE1001()
            violations = rule.check(conn)
            assert len(violations) == 0
            conn.close()

    def test_re1001_with_year_filter(self, tmp_path: Path):
        with tempfile.TemporaryDirectory() as tmpdir:
            conn = duckdb.connect(f"{tmpdir}/test.duckdb")
            _create_re1_test_table(conn)
            _insert_project_row(
                conn,
                report_year=2024,
                project_name="PROJ1",
                uncert="2. Middle Value",
                cprd_grs_oil=-10.0,
            )
            _insert_project_row(
                conn,
                report_year=2023,
                project_name="PROJ2",
                uncert="2. Middle Value",
                cprd_grs_oil=-20.0,
            )
            from esdc.validate.rule_re1 import RE1001

            rule = RE1001()
            violations = rule.check(conn, year=[2024])
            assert len(violations) == 1
            assert violations[0].identifiers["report_year"] == "2024"
            conn.close()

    def test_re1009_finds_negative_sales_cumprod(self, tmp_path: Path):
        with tempfile.TemporaryDirectory() as tmpdir:
            conn = duckdb.connect(f"{tmpdir}/test.duckdb")
            _create_re1_test_table(conn)
            _insert_project_row(
                conn,
                report_year=2024,
                project_name="PROJ1",
                uncert="2. Middle Value",
                cprd_sls_oil=-5.0,
            )
            from esdc.validate.rule_re1 import RE1009

            rule = RE1009()
            violations = rule.check(conn)
            assert len(violations) == 1
            assert violations[0].validated_column == "cprd_sls_oil"
            conn.close()


# ---------------------------------------------------------------------------
# Category B integration tests
# ---------------------------------------------------------------------------


class TestCategoryBMonotonic:
    """Integration tests for RE1013-RE1016 and RE1021-RE1024."""

    def test_re1013_finds_decreasing_cumprod(self, tmp_path: Path):
        with tempfile.TemporaryDirectory() as tmpdir:
            conn = duckdb.connect(f"{tmpdir}/test.duckdb")
            _create_re1_test_table(conn)
            pid = "P-123"
            # 2023: cumprod = 100
            _insert_project_row(
                conn,
                report_year=2023,
                project_name="PROJ1",
                uncert="2. Middle Value",
                cprd_grs_oil=100.0,
                project_id=pid,
            )
            # 2024: cumprod = 80 (decreased!)
            _insert_project_row(
                conn,
                report_year=2024,
                project_name="PROJ1",
                uncert="2. Middle Value",
                cprd_grs_oil=80.0,
                project_id=pid,
            )
            from esdc.validate.rule_re1 import RE1013

            rule = RE1013()
            violations = rule.check(conn)
            assert len(violations) == 1
            assert violations[0].rule_id == "RE1013"
            assert violations[0].validated_column == "cprd_grs_oil"
            assert violations[0].current_values["val_ref"] == 80.0
            assert violations[0].current_values["val_cmp"] == 100.0
            conn.close()

    def test_re1013_no_violation_when_increasing(self, tmp_path: Path):
        with tempfile.TemporaryDirectory() as tmpdir:
            conn = duckdb.connect(f"{tmpdir}/test.duckdb")
            _create_re1_test_table(conn)
            pid = "P-123"
            _insert_project_row(
                conn,
                report_year=2023,
                project_name="PROJ1",
                uncert="2. Middle Value",
                cprd_grs_oil=100.0,
                project_id=pid,
            )
            _insert_project_row(
                conn,
                report_year=2024,
                project_name="PROJ1",
                uncert="2. Middle Value",
                cprd_grs_oil=150.0,
                project_id=pid,
            )
            from esdc.validate.rule_re1 import RE1013

            rule = RE1013()
            violations = rule.check(conn)
            assert len(violations) == 0
            conn.close()

    def test_re1013_no_violation_when_consecutive_year_missing(self, tmp_path: Path):
        """If 2023 data exists but not 2024, the self-join finds no pair."""
        with tempfile.TemporaryDirectory() as tmpdir:
            conn = duckdb.connect(f"{tmpdir}/test.duckdb")
            _create_re1_test_table(conn)
            pid = "P-123"
            _insert_project_row(
                conn,
                report_year=2023,
                project_name="PROJ1",
                uncert="2. Middle Value",
                cprd_grs_oil=100.0,
                project_id=pid,
            )
            # No 2024 row
            from esdc.validate.rule_re1 import RE1013

            rule = RE1013()
            violations = rule.check(conn)
            assert len(violations) == 0
            conn.close()

    def test_re1021_sales_monotonic(self, tmp_path: Path):
        with tempfile.TemporaryDirectory() as tmpdir:
            conn = duckdb.connect(f"{tmpdir}/test.duckdb")
            _create_re1_test_table(conn)
            pid = "P-123"
            _insert_project_row(
                conn,
                report_year=2023,
                project_name="PROJ1",
                uncert="2. Middle Value",
                cprd_sls_oil=100.0,
                project_id=pid,
            )
            _insert_project_row(
                conn,
                report_year=2024,
                project_name="PROJ1",
                uncert="2. Middle Value",
                cprd_sls_oil=80.0,
                project_id=pid,
            )
            from esdc.validate.rule_re1 import RE1021

            rule = RE1021()
            violations = rule.check(conn)
            assert len(violations) == 1
            assert violations[0].rule_id == "RE1021"
            conn.close()

    def test_re1013_different_uncert_levels_not_compared(self, tmp_path: Path):
        """Cumprod at LOW should not be compared against cumprod at MID."""
        with tempfile.TemporaryDirectory() as tmpdir:
            conn = duckdb.connect(f"{tmpdir}/test.duckdb")
            _create_re1_test_table(conn)
            pid = "P-123"
            _insert_project_row(
                conn,
                report_year=2023,
                project_name="PROJ1",
                uncert="1. Low Value",
                cprd_grs_oil=100.0,
                project_id=pid,
            )
            _insert_project_row(
                conn,
                report_year=2024,
                project_name="PROJ1",
                uncert="2. Middle Value",
                cprd_grs_oil=80.0,
                project_id=pid,
            )
            from esdc.validate.rule_re1 import RE1013

            rule = RE1013()
            violations = rule.check(conn)
            # No match because uncert levels differ
            assert len(violations) == 0
            conn.close()


# ---------------------------------------------------------------------------
# Category C integration tests
# ---------------------------------------------------------------------------


class TestCategoryCSalesVsGross:
    """Integration tests for RE1029-RE1032."""

    def test_re1029_finds_sales_exceeding_gross(self, tmp_path: Path):
        with tempfile.TemporaryDirectory() as tmpdir:
            conn = duckdb.connect(f"{tmpdir}/test.duckdb")
            _create_re1_test_table(conn)
            _insert_project_row(
                conn,
                report_year=2024,
                project_name="PROJ1",
                uncert="2. Middle Value",
                cprd_sls_oil=150.0,
                cprd_grs_oil=100.0,
            )
            from esdc.validate.rule_re1 import RE1029

            rule = RE1029()
            violations = rule.check(conn)
            assert len(violations) == 1
            assert violations[0].rule_id == "RE1029"
            assert violations[0].validated_column == "cprd_sls_oil"
            assert violations[0].compared_columns == ["cprd_grs_oil"]
            conn.close()

    def test_re1029_no_violation_when_sales_equal_gross(self, tmp_path: Path):
        with tempfile.TemporaryDirectory() as tmpdir:
            conn = duckdb.connect(f"{tmpdir}/test.duckdb")
            _create_re1_test_table(conn)
            _insert_project_row(
                conn,
                report_year=2024,
                project_name="PROJ1",
                uncert="2. Middle Value",
                cprd_sls_oil=100.0,
                cprd_grs_oil=100.0,
            )
            from esdc.validate.rule_re1 import RE1029

            rule = RE1029()
            violations = rule.check(conn)
            assert len(violations) == 0
            conn.close()

    def test_re1029_no_violation_when_sales_less_than_gross(self, tmp_path: Path):
        with tempfile.TemporaryDirectory() as tmpdir:
            conn = duckdb.connect(f"{tmpdir}/test.duckdb")
            _create_re1_test_table(conn)
            _insert_project_row(
                conn,
                report_year=2024,
                project_name="PROJ1",
                uncert="2. Middle Value",
                cprd_sls_oil=80.0,
                cprd_grs_oil=100.0,
            )
            from esdc.validate.rule_re1 import RE1029

            rule = RE1029()
            violations = rule.check(conn)
            assert len(violations) == 0
            conn.close()


# ---------------------------------------------------------------------------
# Category D integration tests
# ---------------------------------------------------------------------------


class TestCategoryDSalesVsTpf:
    """Integration tests for RE1033-RE1036."""

    def test_re1033_finds_sales_exceeding_tpf(self, tmp_path: Path):
        with tempfile.TemporaryDirectory() as tmpdir:
            conn = duckdb.connect(f"{tmpdir}/test.duckdb")
            _create_re1_test_table(conn)
            _create_timeseries_test_table(conn)
            _insert_timeseries_row(
                conn,
                report_year=2024,
                project_name="PROJ1",
                year=2025,
                slf_oil=150.0,
                tpf_oil=100.0,
            )
            from esdc.validate.rule_re1 import RE1033

            rule = RE1033()
            violations = rule.check(conn)
            assert len(violations) == 1
            assert violations[0].rule_id == "RE1033"
            assert violations[0].validated_column == "slf_oil"
            assert violations[0].compared_columns == ["tpf_oil"]
            assert violations[0].identifiers["year"] == "2025"
            conn.close()

    def test_re1033_no_violation_when_sales_equal_tpf(self, tmp_path: Path):
        with tempfile.TemporaryDirectory() as tmpdir:
            conn = duckdb.connect(f"{tmpdir}/test.duckdb")
            _create_re1_test_table(conn)
            _create_timeseries_test_table(conn)
            _insert_timeseries_row(
                conn,
                report_year=2024,
                project_name="PROJ1",
                year=2025,
                slf_oil=100.0,
                tpf_oil=100.0,
            )
            from esdc.validate.rule_re1 import RE1033

            rule = RE1033()
            violations = rule.check(conn)
            assert len(violations) == 0
            conn.close()

    def test_re1033_no_violation_when_sales_less_than_tpf(self, tmp_path: Path):
        with tempfile.TemporaryDirectory() as tmpdir:
            conn = duckdb.connect(f"{tmpdir}/test.duckdb")
            _create_re1_test_table(conn)
            _create_timeseries_test_table(conn)
            _insert_timeseries_row(
                conn,
                report_year=2024,
                project_name="PROJ1",
                year=2025,
                slf_oil=80.0,
                tpf_oil=100.0,
            )
            from esdc.validate.rule_re1 import RE1033

            rule = RE1033()
            violations = rule.check(conn)
            assert len(violations) == 0
            conn.close()

    def test_re1033_with_year_filter(self, tmp_path: Path):
        with tempfile.TemporaryDirectory() as tmpdir:
            conn = duckdb.connect(f"{tmpdir}/test.duckdb")
            _create_re1_test_table(conn)
            _create_timeseries_test_table(conn)
            _insert_timeseries_row(
                conn,
                report_year=2024,
                project_name="PROJ1",
                year=2025,
                slf_oil=150.0,
                tpf_oil=100.0,
            )
            _insert_timeseries_row(
                conn,
                report_year=2023,
                project_name="PROJ1",
                year=2024,
                slf_oil=200.0,
                tpf_oil=100.0,
            )
            from esdc.validate.rule_re1 import RE1033

            rule = RE1033()
            violations = rule.check(conn, year=[2024])
            assert len(violations) == 1
            assert violations[0].identifiers["report_year"] == "2024"
            conn.close()


# ---------------------------------------------------------------------------
# Category E integration tests
# ---------------------------------------------------------------------------


class TestCategoryEForecastSumReserve:
    """Integration tests for RE1037-RE1040."""

    def test_re1037_finds_mismatch(self, tmp_path: Path):
        with tempfile.TemporaryDirectory() as tmpdir:
            conn = duckdb.connect(f"{tmpdir}/test.duckdb")
            _create_re1_test_table(conn)
            _create_timeseries_test_table(conn)
            pid = "P-123"
            # 2P reserves = 100
            _insert_project_row(
                conn,
                report_year=2024,
                project_name="PROJ1",
                uncert="2. Middle Value",
                res_oil=100.0,
                project_id=pid,
            )
            # Future forecasts: 2025=50, 2026=30 → sum=80 ≠ 100
            _insert_timeseries_row(
                conn,
                report_year=2024,
                project_name="PROJ1",
                year=2025,
                slf_oil=50.0,
                project_id=pid,
            )
            _insert_timeseries_row(
                conn,
                report_year=2024,
                project_name="PROJ1",
                year=2026,
                slf_oil=30.0,
                project_id=pid,
            )
            from esdc.validate.rule_re1 import RE1037

            rule = RE1037()
            violations = rule.check(conn)
            assert len(violations) == 1
            assert violations[0].rule_id == "RE1037"
            assert violations[0].validated_column == "slf_oil"
            assert violations[0].compared_columns == ["res_oil"]
            # val_sum should be 80 (50+30), val_ref should be 100
            assert violations[0].current_values["val_sum"] == 80.0
            assert violations[0].current_values["val_ref"] == 100.0
            conn.close()

    def test_re1037_no_violation_when_sum_matches(self, tmp_path: Path):
        with tempfile.TemporaryDirectory() as tmpdir:
            conn = duckdb.connect(f"{tmpdir}/test.duckdb")
            _create_re1_test_table(conn)
            _create_timeseries_test_table(conn)
            pid = "P-123"
            # 2P reserves = 100
            _insert_project_row(
                conn,
                report_year=2024,
                project_name="PROJ1",
                uncert="2. Middle Value",
                res_oil=100.0,
                project_id=pid,
            )
            # Future forecasts: 2025=60, 2026=40 → sum=100 = reserves
            _insert_timeseries_row(
                conn,
                report_year=2024,
                project_name="PROJ1",
                year=2025,
                slf_oil=60.0,
                project_id=pid,
            )
            _insert_timeseries_row(
                conn,
                report_year=2024,
                project_name="PROJ1",
                year=2026,
                slf_oil=40.0,
                project_id=pid,
            )
            from esdc.validate.rule_re1 import RE1037

            rule = RE1037()
            violations = rule.check(conn)
            assert len(violations) == 0
            conn.close()

    def test_re1037_no_violation_when_no_forecast_data(self, tmp_path: Path):
        """If no timeseries data exists, INNER JOIN finds nothing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            conn = duckdb.connect(f"{tmpdir}/test.duckdb")
            _create_re1_test_table(conn)
            _create_timeseries_test_table(conn)
            # Only project_resources, no timeseries
            _insert_project_row(
                conn,
                report_year=2024,
                project_name="PROJ1",
                uncert="2. Middle Value",
                res_oil=100.0,
                project_id="P-123",
            )
            from esdc.validate.rule_re1 import RE1037

            rule = RE1037()
            violations = rule.check(conn)
            assert len(violations) == 0
            conn.close()

    def test_re1037_only_future_years_count(self, tmp_path: Path):
        """Forecast at year=2024 (same as report_year) should NOT be counted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            conn = duckdb.connect(f"{tmpdir}/test.duckdb")
            _create_re1_test_table(conn)
            _create_timeseries_test_table(conn)
            pid = "P-123"
            _insert_project_row(
                conn,
                report_year=2024,
                project_name="PROJ1",
                uncert="2. Middle Value",
                res_oil=50.0,
                project_id=pid,
            )
            # Current year (2024) should NOT count (year > report_year)
            _insert_timeseries_row(
                conn,
                report_year=2024,
                project_name="PROJ1",
                year=2024,
                slf_oil=100.0,  # This should be ignored
                project_id=pid,
            )
            # Future year (2025) should count
            _insert_timeseries_row(
                conn,
                report_year=2024,
                project_name="PROJ1",
                year=2025,
                slf_oil=50.0,
                project_id=pid,
            )
            from esdc.validate.rule_re1 import RE1037

            rule = RE1037()
            violations = rule.check(conn)
            # val_sum should be 50 (only 2025 counts), which matches res_oil=50
            assert len(violations) == 0
            conn.close()


# ---------------------------------------------------------------------------
# Category F integration tests
# ---------------------------------------------------------------------------


class TestCategoryFForecastSumResource:
    """Integration tests for RE1041-RE1044."""

    def test_re1041_finds_mismatch(self, tmp_path: Path):
        with tempfile.TemporaryDirectory() as tmpdir:
            conn = duckdb.connect(f"{tmpdir}/test.duckdb")
            _create_re1_test_table(conn)
            _create_timeseries_test_table(conn)
            pid = "P-123"
            # P50 resources = 100
            _insert_project_row(
                conn,
                report_year=2024,
                project_name="PROJ1",
                uncert="2. Middle Value",
                rec_oil=100.0,
                project_id=pid,
            )
            # Future TPF: 2025=50, 2026=30 → sum=80 ≠ 100
            _insert_timeseries_row(
                conn,
                report_year=2024,
                project_name="PROJ1",
                year=2025,
                tpf_oil=50.0,
                project_id=pid,
            )
            _insert_timeseries_row(
                conn,
                report_year=2024,
                project_name="PROJ1",
                year=2026,
                tpf_oil=30.0,
                project_id=pid,
            )
            from esdc.validate.rule_re1 import RE1041

            rule = RE1041()
            violations = rule.check(conn)
            assert len(violations) == 1
            assert violations[0].rule_id == "RE1041"
            assert violations[0].validated_column == "tpf_oil"
            assert violations[0].compared_columns == ["rec_oil"]
            assert violations[0].current_values["val_sum"] == 80.0
            assert violations[0].current_values["val_ref"] == 100.0
            conn.close()

    def test_re1041_no_violation_when_sum_matches(self, tmp_path: Path):
        with tempfile.TemporaryDirectory() as tmpdir:
            conn = duckdb.connect(f"{tmpdir}/test.duckdb")
            _create_re1_test_table(conn)
            _create_timeseries_test_table(conn)
            pid = "P-123"
            # P50 resources = 100
            _insert_project_row(
                conn,
                report_year=2024,
                project_name="PROJ1",
                uncert="2. Middle Value",
                rec_oil=100.0,
                project_id=pid,
            )
            # Future TPF: 2025=60, 2026=40 → sum=100 = resources
            _insert_timeseries_row(
                conn,
                report_year=2024,
                project_name="PROJ1",
                year=2025,
                tpf_oil=60.0,
                project_id=pid,
            )
            _insert_timeseries_row(
                conn,
                report_year=2024,
                project_name="PROJ1",
                year=2026,
                tpf_oil=40.0,
                project_id=pid,
            )
            from esdc.validate.rule_re1 import RE1041

            rule = RE1041()
            violations = rule.check(conn)
            assert len(violations) == 0
            conn.close()


# ---------------------------------------------------------------------------
# Registry test
# ---------------------------------------------------------------------------


class TestRe1Registry:
    """Test that all RE1 rules are registered."""

    def test_all_re1_rules_registered(self):
        re1_rules = get_rules_by_group("RE1")
        rule_ids = sorted(r.rule_id for r in re1_rules)
        expected = [
            "RE1001",
            "RE1002",
            "RE1003",
            "RE1004",
            "RE1009",
            "RE1010",
            "RE1011",
            "RE1012",
            "RE1013",
            "RE1014",
            "RE1015",
            "RE1016",
            "RE1021",
            "RE1022",
            "RE1023",
            "RE1024",
            "RE1029",
            "RE1030",
            "RE1031",
            "RE1032",
            "RE1033",
            "RE1034",
            "RE1035",
            "RE1036",
            "RE1037",
            "RE1038",
            "RE1039",
            "RE1040",
            "RE1041",
            "RE1042",
            "RE1043",
            "RE1044",
        ]
        assert rule_ids == expected
        assert len(re1_rules) == 32

    def test_total_rule_count_increased(self):
        all_rules = get_all_rules()
        re1_count = len(get_rules_by_group("RE1"))
        re0_count = len(get_rules_by_group("RE0"))
        re9_count = len(get_rules_by_group("RE9"))
        assert re1_count == 32
        assert re0_count == 66
        assert re9_count == 1
        assert len(all_rules) == re0_count + re1_count + re9_count


# ---------------------------------------------------------------------------
# Tolerance boundary tests
# ---------------------------------------------------------------------------


class TestToleranceBoundary:
    """Tests that values within tolerance are not flagged as violations."""

    def test_non_negative_within_tolerance(self, tmp_path: Path):
        with tempfile.TemporaryDirectory() as tmpdir:
            conn = duckdb.connect(f"{tmpdir}/test.duckdb")
            _create_re1_test_table(conn)
            _insert_project_row(
                conn,
                report_year=2024,
                project_name="PROJ1",
                uncert="2. Middle Value",
                cprd_grs_oil=-0.0005,
            )
            from esdc.validate.rule_re1 import RE1001

            rule = RE1001()
            violations = rule.check(conn)
            assert len(violations) == 0
            conn.close()

    def test_non_negative_at_tolerance_boundary(self, tmp_path: Path):
        with tempfile.TemporaryDirectory() as tmpdir:
            conn = duckdb.connect(f"{tmpdir}/test.duckdb")
            _create_re1_test_table(conn)
            _insert_project_row(
                conn,
                report_year=2024,
                project_name="PROJ1",
                uncert="2. Middle Value",
                cprd_grs_oil=-TOLERANCE,
            )
            from esdc.validate.rule_re1 import RE1001

            rule = RE1001()
            violations = rule.check(conn)
            assert len(violations) == 0
            conn.close()

    def test_non_negative_above_tolerance(self, tmp_path: Path):
        with tempfile.TemporaryDirectory() as tmpdir:
            conn = duckdb.connect(f"{tmpdir}/test.duckdb")
            _create_re1_test_table(conn)
            _insert_project_row(
                conn,
                report_year=2024,
                project_name="PROJ1",
                uncert="2. Middle Value",
                cprd_grs_oil=-TOLERANCE - 0.0001,
            )
            from esdc.validate.rule_re1 import RE1001

            rule = RE1001()
            violations = rule.check(conn)
            assert len(violations) == 1
            conn.close()

    def test_sales_vs_gross_within_tolerance(self, tmp_path: Path):
        with tempfile.TemporaryDirectory() as tmpdir:
            conn = duckdb.connect(f"{tmpdir}/test.duckdb")
            _create_re1_test_table(conn)
            _insert_project_row(
                conn,
                report_year=2024,
                project_name="PROJ1",
                uncert="2. Middle Value",
                cprd_sls_oil=100.0005,
                cprd_grs_oil=100.0,
            )
            from esdc.validate.rule_re1 import RE1029

            rule = RE1029()
            violations = rule.check(conn)
            assert len(violations) == 0
            conn.close()

    def test_timeseries_within_tolerance(self, tmp_path: Path):
        with tempfile.TemporaryDirectory() as tmpdir:
            conn = duckdb.connect(f"{tmpdir}/test.duckdb")
            _create_re1_test_table(conn)
            _create_timeseries_test_table(conn)
            _insert_project_row(
                conn,
                report_year=2024,
                project_name="PROJ1",
                uncert="2. Middle Value",
            )
            _insert_timeseries_row(
                conn,
                report_year=2024,
                project_name="PROJ1",
                year=2025,
                slf_oil=7.208799,
                tpf_oil=7.208750,
            )
            from esdc.validate.rule_re1 import RE1033

            rule = RE1033()
            violations = rule.check(conn)
            assert len(violations) == 0
            conn.close()

    def test_timeseries_above_tolerance(self, tmp_path: Path):
        with tempfile.TemporaryDirectory() as tmpdir:
            conn = duckdb.connect(f"{tmpdir}/test.duckdb")
            _create_re1_test_table(conn)
            _create_timeseries_test_table(conn)
            _insert_project_row(
                conn,
                report_year=2024,
                project_name="PROJ1",
                uncert="2. Middle Value",
            )
            _insert_timeseries_row(
                conn,
                report_year=2024,
                project_name="PROJ1",
                year=2025,
                slf_oil=0.164,
                tpf_oil=0.054,
            )
            from esdc.validate.rule_re1 import RE1033

            rule = RE1033()
            violations = rule.check(conn)
            assert len(violations) == 1
            conn.close()
