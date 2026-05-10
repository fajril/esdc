"""Tests for RE0 validation helpers and rules."""

from __future__ import annotations

import duckdb

from esdc.validate.rule_re0_helpers import (
    IDENTIFIER_COLS,
    UNCERT_HIGH,
    UNCERT_LOW,
    UNCERT_MID,
    VOL_COLUMNS,
    _add_year_filter,
    _execute_and_build_violations,
    build_implication_sql,
    build_non_negative_sql,
    build_ordering_sql,
    build_reserve_vs_place_sql,
    build_same_row_ordering_sql,
)

# --- Helper SQL building tests ---

class TestBuildNonNegativeSql:
    def test_basic(self):
        sql = build_non_negative_sql("prj_ioip", UNCERT_LOW)
        assert "SELECT report_year, project_name, wk_name, prj_ioip" in sql
        assert "WHERE uncert_level = '1. Low Value'" in sql
        assert "COALESCE(prj_ioip, 0) < 0" in sql

    def test_with_custom_table(self):
        sql = build_non_negative_sql("res_oil", UNCERT_LOW, table="my_table")
        assert "FROM my_table" in sql

    def test_different_uncert(self):
        sql = build_non_negative_sql("rec_ga", UNCERT_MID)
        assert "uncert_level = '2. Middle Value'" in sql


class TestBuildOrderingSql:
    def test_basic(self):
        sql = build_ordering_sql(
            "prj_ioip", "prj_ioip", UNCERT_LOW, UNCERT_MID
        )
        assert "FROM project_resources l" in sql
        assert "JOIN project_resources h" in sql
        assert "l.uncert_level = '1. Low Value'" in sql
        assert "h.uncert_level = '2. Middle Value'" in sql
        assert "COALESCE(l.prj_ioip, 0) > COALESCE(h.prj_ioip, 0)" in sql

    def test_different_columns(self):
        sql = build_ordering_sql(
            "rec_oil", "rec_oil", UNCERT_LOW, UNCERT_MID
        )
        assert "l.rec_oil AS low_val" in sql
        assert "h.rec_oil AS high_val" in sql

    def test_custom_table(self):
        sql = build_ordering_sql(
            "res_oil", "res_oil", UNCERT_LOW, UNCERT_MID,
            table="my_table"
        )
        assert "FROM my_table l" in sql


class TestBuildSameRowOrderingSql:
    def test_basic(self):
        sql = build_same_row_ordering_sql(
            "res_oil", "rec_oil", UNCERT_LOW
        )
        assert "FROM project_resources" in sql
        assert "uncert_level = '1. Low Value'" in sql
        assert "COALESCE(res_oil, 0) > COALESCE(rec_oil, 0)" in sql


class TestBuildImplicationSql:
    def test_basic(self):
        sql = build_implication_sql(
            "res_oil", UNCERT_HIGH, "res_oil", UNCERT_LOW
        )
        assert "FROM project_resources h" in sql
        assert "JOIN project_resources l" in sql
        assert "h.uncert_level = '3. High Value'" in sql
        assert "l.uncert_level = '1. Low Value'" in sql
        assert "COALESCE(h.res_oil, 0) > 0" in sql
        assert "COALESCE(l.res_oil, 0) = 0" in sql

    def test_different_columns(self):
        sql = build_implication_sql(
            "res_con", UNCERT_HIGH, "res_con", UNCERT_LOW
        )
        assert "h.res_con AS cond_val" in sql
        assert "l.res_con AS result_val" in sql


class TestBuildReserveVsPlaceSql:
    def test_basic(self):
        sql = build_reserve_vs_place_sql(
            "prj_ioip", "res_oil", "cprd_grs_oil", UNCERT_LOW
        )
        assert "FROM project_resources" in sql
        assert "uncert_level = '1. Low Value'" in sql
        assert "COALESCE(prj_ioip, 0) > 0" in sql
        assert (
            "COALESCE(res_oil, 0) + COALESCE(cprd_grs_oil, 0) >= COALESCE(prj_ioip, 0)"
            in sql
        )


class TestAddYearFilter:
    def test_no_year(self):
        base = "SELECT * FROM table WHERE x = 1"
        assert _add_year_filter(base, None) == base

    def test_single_year(self):
        base = "SELECT * FROM table WHERE x = 1"
        result = _add_year_filter(base, [2024])
        assert "report_year IN (2024)" in result
        assert "AND x = 1" in result

    def test_multiple_years(self):
        base = "SELECT * FROM table WHERE x = 1"
        result = _add_year_filter(base, [2022, 2023, 2024])
        assert "report_year IN (2022, 2023, 2024)" in result

    def test_no_where_clause(self):
        base = "SELECT * FROM table"
        result = _add_year_filter(base, [2024])
        assert "WHERE report_year IN (2024)" in result


class TestExecuteAndBuildViolations:
    def test_builds_violations(self, tmp_path):
        db_path = tmp_path / "test.duckdb"
        conn = duckdb.connect(str(db_path))
        conn.execute("""
            CREATE TABLE project_resources (
                report_year INTEGER,
                project_name TEXT,
                wk_name TEXT,
                uncert_level TEXT,
                prj_ioip REAL
            )
        """)
        conn.execute(
            "INSERT INTO project_resources"
            " VALUES (2024, 'BadProj', 'WK1', '1. Low Value', -100)"
        )
        conn.execute(
            "INSERT INTO project_resources"
            " VALUES (2024, 'GoodProj', 'WK1', '1. Low Value', 100)"
        )

        sql = build_non_negative_sql("prj_ioip", UNCERT_LOW)
        violations = _execute_and_build_violations(
            conn,
            sql,
            rule_id="RE0001",
            description="test",
            severity="strict",
            table="project_resources",
            year=None,
            extra_columns=["prj_ioip"],
        )

        assert len(violations) == 1
        assert violations[0].rule_id == "RE0001"
        assert violations[0].rule_group == "RE0"
        assert violations[0].identifiers["project_name"] == "BadProj"
        assert violations[0].current_values["prj_ioip"] == -100.0

        conn.close()

    def test_year_filter(self, tmp_path):
        db_path = tmp_path / "test.duckdb"
        conn = duckdb.connect(str(db_path))
        conn.execute("""
            CREATE TABLE project_resources (
                report_year INTEGER,
                project_name TEXT,
                wk_name TEXT,
                uncert_level TEXT,
                prj_ioip REAL
            )
        """)
        conn.execute(
            "INSERT INTO project_resources"
            " VALUES (2023, 'Old', 'WK1', '1. Low Value', -50)"
        )
        conn.execute(
            "INSERT INTO project_resources"
            " VALUES (2024, 'New', 'WK1', '1. Low Value', -60)"
        )

        sql = build_non_negative_sql("prj_ioip", UNCERT_LOW)
        violations = _execute_and_build_violations(
            conn,
            sql,
            rule_id="RE0001",
            description="test",
            severity="strict",
            table="project_resources",
            year=[2024],
            extra_columns=["prj_ioip"],
        )

        assert len(violations) == 1
        assert violations[0].identifiers["project_name"] == "New"

        conn.close()

    def test_empty_result(self, tmp_path):
        db_path = tmp_path / "test.duckdb"
        conn = duckdb.connect(str(db_path))
        conn.execute("""
            CREATE TABLE project_resources (
                report_year INTEGER,
                project_name TEXT,
                wk_name TEXT,
                uncert_level TEXT,
                prj_ioip REAL
            )
        """)
        conn.execute(
            "INSERT INTO project_resources"
            " VALUES (2024, 'Good', 'WK1', '1. Low Value', 100)"
        )

        sql = build_non_negative_sql("prj_ioip", UNCERT_LOW)
        violations = _execute_and_build_violations(
            conn,
            sql,
            rule_id="RE0001",
            description="test",
            severity="strict",
            table="project_resources",
            year=None,
            extra_columns=["prj_ioip"],
        )

        assert len(violations) == 0
        conn.close()


# --- Constant tests ---

class TestConstants:
    def test_uncert_values(self):
        assert UNCERT_LOW == "1. Low Value"
        assert UNCERT_MID == "2. Middle Value"
        assert UNCERT_HIGH == "3. High Value"

    def test_identifier_cols(self):
        assert IDENTIFIER_COLS == ["report_year", "project_name", "wk_name"]

    def test_vol_columns(self):
        assert VOL_COLUMNS["ioip"] == "prj_ioip"
        assert VOL_COLUMNS["res_oil"] == "res_oil"
        assert VOL_COLUMNS["rec_oil"] == "rec_oil"
        assert VOL_COLUMNS["cprd_oil"] == "cprd_grs_oil"
