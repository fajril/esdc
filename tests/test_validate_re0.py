"""Tests for RE0 validation helpers and rules."""

from __future__ import annotations

import duckdb

from esdc.selection import Severity
from esdc.validate.rule_re0_helpers import (
    AGGREGATION_CONSISTENCY_IDENTIFIER_COLS,
    FIELD_IDENTIFIER_COLS,
    IDENTIFIER_COLS,
    VOL_COLUMNS,
    UncertLevel,
    _add_year_filter,
    _execute_and_build_violations,
    build_aggregation_consistency_sql,
    build_field_non_negative_sql,
    build_field_ordering_sql,
    build_implication_sql,
    build_non_negative_sql,
    build_ordering_sql,
    build_reserve_vs_place_sql,
    build_same_row_ordering_sql,
)

# --- Helper SQL building tests ---


class TestBuildNonNegativeSql:
    def test_basic(self):
        sql = build_non_negative_sql("prj_ioip", UncertLevel.LOW)
        assert "SELECT report_year, project_name, wk_name, field_name, prj_ioip" in sql
        assert "WHERE uncert_level = '1. Low Value'" in sql
        assert "COALESCE(prj_ioip, 0) < 0" in sql

    def test_with_custom_table(self):
        sql = build_non_negative_sql("res_oil", UncertLevel.LOW, table="my_table")
        assert "FROM my_table" in sql

    def test_different_uncert(self):
        sql = build_non_negative_sql("rec_ga", UncertLevel.MID)
        assert "uncert_level = '2. Middle Value'" in sql


class TestBuildOrderingSql:
    def test_basic(self):
        sql = build_ordering_sql(
            "prj_ioip", "prj_ioip", UncertLevel.LOW, UncertLevel.MID
        )
        assert "FROM project_resources l" in sql
        assert "JOIN project_resources h" in sql
        assert "ON l.project_id = h.project_id" in sql
        assert "AND l.report_year = h.report_year" in sql
        assert "l.uncert_level = '1. Low Value'" in sql
        assert "h.uncert_level = '2. Middle Value'" in sql
        assert "COALESCE(l.prj_ioip, 0) > COALESCE(h.prj_ioip, 0)" in sql

    def test_different_columns(self):
        sql = build_ordering_sql("rec_oil", "rec_oil", UncertLevel.LOW, UncertLevel.MID)
        assert "l.rec_oil AS val_ref" in sql
        assert "h.rec_oil AS val_cmp" in sql

    def test_custom_table(self):
        sql = build_ordering_sql(
            "res_oil", "res_oil", UncertLevel.LOW, UncertLevel.MID, table="my_table"
        )
        assert "FROM my_table l" in sql


class TestBuildSameRowOrderingSql:
    def test_basic(self):
        sql = build_same_row_ordering_sql("res_oil", "rec_oil", UncertLevel.LOW)
        assert "FROM project_resources" in sql
        assert "uncert_level = '1. Low Value'" in sql
        assert "COALESCE(res_oil, 0) > COALESCE(rec_oil, 0)" in sql
        assert "res_oil AS val_ref" in sql
        assert "rec_oil AS val_cmp" in sql


class TestBuildImplicationSql:
    def test_basic(self):
        sql = build_implication_sql(
            "res_oil", UncertLevel.HIGH, "res_oil", UncertLevel.LOW
        )
        assert "FROM project_resources h" in sql
        assert "JOIN project_resources l" in sql
        assert "ON h.project_id = l.project_id" in sql
        assert "AND h.report_year = l.report_year" in sql
        assert "h.uncert_level = '3. High Value'" in sql
        assert "l.uncert_level = '1. Low Value'" in sql
        assert "COALESCE(h.res_oil, 0) > 0" in sql
        assert "COALESCE(l.res_oil, 0) = 0" in sql

    def test_different_columns(self):
        sql = build_implication_sql(
            "res_con", UncertLevel.HIGH, "res_con", UncertLevel.LOW
        )
        assert "h.res_con AS val_ref" in sql
        assert "l.res_con AS val_cmp" in sql


class TestBuildReserveVsPlaceSql:
    def test_basic(self):
        sql = build_reserve_vs_place_sql(
            "prj_ioip", "res_oil", "cprd_grs_oil", UncertLevel.LOW
        )
        assert "FROM project_resources" in sql
        assert "uncert_level = '1. Low Value'" in sql
        assert "COALESCE(prj_ioip, 0) > 0" in sql
        assert (
            "COALESCE(res_oil, 0) + COALESCE(cprd_grs_oil, 0) >= COALESCE(prj_ioip, 0)"
            in sql
        )
        assert "prj_ioip AS val_ref" in sql
        assert "res_oil AS val_cmp" in sql
        assert "cprd_grs_oil AS val_sum" in sql


class TestBuildFieldNonNegativeSql:
    def test_basic(self):
        sql = build_field_non_negative_sql("ioip", UncertLevel.LOW)
        assert "SELECT report_year, wk_name, field_name" in sql
        assert "SUM(COALESCE(ioip, 0)) AS val_ref" in sql
        assert "FROM field_resources" in sql
        assert "GROUP BY report_year, wk_name, field_name, uncert_level" in sql
        assert "HAVING SUM(COALESCE(ioip, 0)) < 0" in sql

    def test_with_custom_table(self):
        sql = build_field_non_negative_sql("igip", UncertLevel.LOW, table="my_table")
        assert "FROM my_table" in sql

    def test_different_uncert(self):
        sql = build_field_non_negative_sql("ioip", UncertLevel.HIGH)
        assert "uncert_level = '3. High Value'" in sql


class TestBuildFieldOrderingSql:
    def test_basic(self):
        sql = build_field_ordering_sql("ioip", "ioip", UncertLevel.LOW, UncertLevel.MID)
        assert "WITH field_agg AS" in sql
        assert "SUM(COALESCE(ioip, 0)) AS ioip" in sql
        assert "GROUP BY report_year, wk_name, field_name, uncert_level" in sql
        assert "FROM field_agg l" in sql
        assert "JOIN field_agg h" in sql
        assert "ON l.wk_name = h.wk_name" in sql
        assert "AND l.field_name = h.field_name" in sql
        assert "l.uncert_level = '1. Low Value'" in sql
        assert "h.uncert_level = '2. Middle Value'" in sql
        assert "l.ioip AS val_ref" in sql
        assert "h.ioip AS val_cmp" in sql

    def test_different_columns(self):
        sql = build_field_ordering_sql(
            "igip", "igip", UncertLevel.MID, UncertLevel.HIGH
        )
        assert "SUM(COALESCE(igip, 0)) AS igip" in sql
        assert "l.igip AS val_ref" in sql
        assert "h.igip AS val_cmp" in sql

    def test_custom_table(self):
        sql = build_field_ordering_sql(
            "ioip",
            "ioip",
            UncertLevel.LOW,
            UncertLevel.MID,
            table="my_table",
        )
        assert "FROM my_table" in sql


class TestBuildAggregationConsistencySql:
    def test_basic(self):
        sql = build_aggregation_consistency_sql("ioip", "prj_ioip", UncertLevel.LOW)
        assert "FROM field_resources fr" in sql
        assert "JOIN project_resources pr" in sql
        assert "ON fr.wk_id = pr.wk_id" in sql
        assert "AND fr.field_id = pr.field_id" in sql
        assert "AND fr.report_year = pr.report_year" in sql
        assert "AND fr.project_stage = pr.project_stage" in sql
        assert "AND fr.project_class = pr.project_class" in sql
        assert "AND fr.uncert_level = pr.uncert_level" in sql
        assert "WHERE fr.uncert_level = '1. Low Value'" in sql
        assert "SUM(pr.prj_ioip) AS val_sum" in sql
        assert "fr.ioip AS val_field" in sql
        assert "HAVING ABS(SUM(pr.prj_ioip) - fr.ioip) > 0.001" in sql

    def test_different_column(self):
        sql = build_aggregation_consistency_sql("igip", "prj_igip", UncertLevel.MID)
        assert "SUM(pr.prj_igip) AS val_sum" in sql
        assert "fr.igip AS val_field" in sql
        assert "fr.uncert_level = '2. Middle Value'" in sql

    def test_identifier_columns(self):
        assert AGGREGATION_CONSISTENCY_IDENTIFIER_COLS == [
            "report_year",
            "wk_name",
            "field_name",
            "project_stage",
            "project_class",
        ]


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

    def test_self_join_year_filter(self):
        sql = build_ordering_sql(
            "prj_ioip", "prj_ioip", UncertLevel.LOW, UncertLevel.MID
        )
        result = _add_year_filter(sql, [2024])
        assert "l.report_year IN (2024)" in result

    def test_cross_join_year_filter(self):
        sql = build_aggregation_consistency_sql("ioip", "prj_ioip", UncertLevel.LOW)
        result = _add_year_filter(sql, [2024])
        assert "fr.report_year IN (2024)" in result
        assert "AND fr.uncert_level" in result


class TestExecuteAndBuildViolations:
    def test_builds_violations(self, tmp_path):
        db_path = tmp_path / "test.duckdb"
        conn = duckdb.connect(str(db_path))
        conn.execute("""
            CREATE TABLE project_resources (
                report_year INTEGER,
                project_name TEXT,
                wk_name TEXT,
                field_name TEXT,
                project_id TEXT,
                uncert_level TEXT,
                prj_ioip REAL
            )
        """)
        conn.execute(
            "INSERT INTO project_resources"
            " VALUES (2024, 'BadProj', 'WK1', 'FLD1', 'P-001', '1. Low Value', -100)"
        )
        conn.execute(
            "INSERT INTO project_resources"
            " VALUES (2024, 'GoodProj', 'WK1', 'FLD2', 'P-002', '1. Low Value', 100)"
        )

        sql = build_non_negative_sql("prj_ioip", UncertLevel.LOW)
        violations = _execute_and_build_violations(
            conn,
            sql,
            rule_id="RE0001",
            description="test",
            severity=Severity.STRICT,
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
                field_name TEXT,
                project_id TEXT,
                uncert_level TEXT,
                prj_ioip REAL
            )
        """)
        conn.execute(
            "INSERT INTO project_resources"
            " VALUES (2023, 'Old', 'WK1', 'FLD1', 'P-001', '1. Low Value', -50)"
        )
        conn.execute(
            "INSERT INTO project_resources"
            " VALUES (2024, 'New', 'WK1', 'FLD2', 'P-002', '1. Low Value', -60)"
        )

        sql = build_non_negative_sql("prj_ioip", UncertLevel.LOW)
        violations = _execute_and_build_violations(
            conn,
            sql,
            rule_id="RE0001",
            description="test",
            severity=Severity.STRICT,
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
                field_name TEXT,
                project_id TEXT,
                uncert_level TEXT,
                prj_ioip REAL
            )
        """)
        conn.execute(
            "INSERT INTO project_resources"
            " VALUES (2024, 'Good', 'WK1', 'FLD1', 'P-001', '1. Low Value', 100)"
        )

        sql = build_non_negative_sql("prj_ioip", UncertLevel.LOW)
        violations = _execute_and_build_violations(
            conn,
            sql,
            rule_id="RE0001",
            description="test",
            severity=Severity.STRICT,
            table="project_resources",
            year=None,
            extra_columns=["prj_ioip"],
        )

        assert len(violations) == 0
        conn.close()


# --- Constant tests ---


class TestConstants:
    def test_uncert_values(self):
        assert UncertLevel.LOW == "1. Low Value"
        assert UncertLevel.MID == "2. Middle Value"
        assert UncertLevel.HIGH == "3. High Value"

    def test_identifier_cols(self):
        assert IDENTIFIER_COLS == [
            "report_year",
            "project_name",
            "wk_name",
            "field_name",
        ]

    def test_field_identifier_cols(self):
        assert FIELD_IDENTIFIER_COLS == [
            "report_year",
            "wk_name",
            "field_name",
        ]

    def test_vol_columns(self):
        assert VOL_COLUMNS["ioip"] == "prj_ioip"
        assert VOL_COLUMNS["res_oil"] == "res_oil"
        assert VOL_COLUMNS["rec_oil"] == "rec_oil"
        assert VOL_COLUMNS["cprd_oil"] == "cprd_grs_oil"


# --- Integration tests for RE0 rules ---


def _create_re0_test_table(conn: duckdb.DuckDBPyConnection) -> None:
    """Create a full project_resources table for RE0 testing."""
    conn.execute("""
        CREATE TABLE project_resources (
            report_year INTEGER,
            project_name TEXT,
            wk_name TEXT,
            field_name TEXT,
            project_id TEXT,
            uncert_level TEXT,
            prj_ioip REAL,
            prj_igip REAL,
            rec_oil REAL,
            rec_con REAL,
            rec_ga REAL,
            rec_gn REAL,
            res_oil REAL,
            res_con REAL,
            res_ga REAL,
            res_gn REAL,
            cprd_grs_oil REAL,
            cprd_grs_con REAL,
            cprd_grs_ga REAL,
            cprd_grs_gn REAL
        )
    """)


def _insert_full_row(
    conn: duckdb.DuckDBPyConnection,
    year: int,
    name: str,
    uncert: str,
    ioip: float | None = 0,
    igip: float | None = 0,
    rec_oil: float | None = 0,
    rec_con: float | None = 0,
    rec_ga: float | None = 0,
    rec_gn: float | None = 0,
    res_oil: float | None = 0,
    res_con: float | None = 0,
    res_ga: float | None = 0,
    res_gn: float | None = 0,
    cprd_oil: float | None = 0,
    cprd_con: float | None = 0,
    cprd_ga: float | None = 0,
    cprd_gn: float | None = 0,
    wk_name: str = "WK1",
    field_name: str = "FLD1",
    project_id: str | None = None,
) -> None:
    if project_id is None:
        project_id = f"P-{hash((year, name, wk_name)) & 0xFFFFFF:06X}"
    conn.execute(
        "INSERT INTO project_resources"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            year,
            name,
            wk_name,
            field_name,
            project_id,
            uncert,
            ioip,
            igip,
            rec_oil,
            rec_con,
            rec_ga,
            rec_gn,
            res_oil,
            res_con,
            res_ga,
            res_gn,
            cprd_oil,
            cprd_con,
            cprd_ga,
            cprd_gn,
        ],
    )


def _create_field_test_table(conn: duckdb.DuckDBPyConnection) -> None:
    """Create a field_resources table for field-level RE0 testing."""
    conn.execute("""
        CREATE TABLE field_resources (
            report_year INTEGER,
            wk_name TEXT,
            field_name TEXT,
            uncert_level TEXT,
            ioip REAL,
            igip REAL,
            rec_oil REAL,
            rec_con REAL,
            rec_ga REAL,
            rec_gn REAL,
            res_oil REAL,
            res_con REAL,
            res_ga REAL,
            res_gn REAL,
            cprd_grs_oil REAL,
            cprd_grs_con REAL,
            cprd_grs_ga REAL,
            cprd_grs_gn REAL
        )
    """)


def _insert_field_row(
    conn: duckdb.DuckDBPyConnection,
    year: int,
    wk_name: str,
    field_name: str,
    uncert: str,
    ioip: float | None = 0,
    igip: float | None = 0,
    rec_oil: float | None = 0,
    rec_con: float | None = 0,
    rec_ga: float | None = 0,
    rec_gn: float | None = 0,
    res_oil: float | None = 0,
    res_con: float | None = 0,
    res_ga: float | None = 0,
    res_gn: float | None = 0,
    cprd_oil: float | None = 0,
    cprd_con: float | None = 0,
    cprd_ga: float | None = 0,
    cprd_gn: float | None = 0,
) -> None:
    conn.execute(
        "INSERT INTO field_resources"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            year,
            wk_name,
            field_name,
            uncert,
            ioip,
            igip,
            rec_oil,
            rec_con,
            rec_ga,
            rec_gn,
            res_oil,
            res_con,
            res_ga,
            res_gn,
            cprd_oil,
            cprd_con,
            cprd_ga,
            cprd_gn,
        ],
    )


# --- Field-level RE0 rule tests ---


class TestFieldLevelNonNegative:
    """Tests for RE0001, RE0002 (field-level non-negative checks)."""

    def test_re0001_finds_negative_ioip(self, tmp_path):
        db_path = tmp_path / "test.duckdb"
        conn = duckdb.connect(str(db_path))
        _create_field_test_table(conn)

        _insert_field_row(conn, 2024, "WK1", "FLD1", "1. Low Value", ioip=100)
        _insert_field_row(conn, 2024, "WK1", "FLD2", "1. Low Value", ioip=-50)

        from esdc.validate.rule_re0 import RE0001

        rule = RE0001()
        violations = rule.check(conn)

        assert len(violations) == 1
        assert violations[0].rule_id == "RE0001"
        assert violations[0].identifiers["wk_name"] == "WK1"
        assert violations[0].identifiers["field_name"] == "FLD2"
        assert violations[0].current_values["val_ref"] == -50.0
        conn.close()

    def test_re0001_no_violations(self, tmp_path):
        db_path = tmp_path / "test.duckdb"
        conn = duckdb.connect(str(db_path))
        _create_field_test_table(conn)

        _insert_field_row(conn, 2024, "WK1", "FLD1", "1. Low Value", ioip=100)
        _insert_field_row(conn, 2024, "WK1", "FLD2", "1. Low Value", ioip=0)

        from esdc.validate.rule_re0 import RE0001

        rule = RE0001()
        violations = rule.check(conn)
        assert len(violations) == 0
        conn.close()

    def test_re0001_aggregates_multiline_field(self, tmp_path):
        """Field with multiple project entries: SUM should be checked."""
        db_path = tmp_path / "test.duckdb"
        conn = duckdb.connect(str(db_path))
        _create_field_test_table(conn)

        _insert_field_row(conn, 2024, "WK1", "FLD1", "1. Low Value", ioip=200)
        _insert_field_row(conn, 2024, "WK1", "FLD1", "1. Low Value", ioip=-300)

        from esdc.validate.rule_re0 import RE0001

        rule = RE0001()
        violations = rule.check(conn)
        assert len(violations) == 1
        assert violations[0].identifiers["field_name"] == "FLD1"
        assert violations[0].current_values["val_ref"] == -100.0
        conn.close()

    def test_re0002_finds_negative_igip(self, tmp_path):
        db_path = tmp_path / "test.duckdb"
        conn = duckdb.connect(str(db_path))
        _create_field_test_table(conn)

        _insert_field_row(conn, 2024, "WK1", "FLD1", "1. Low Value", igip=-10)

        from esdc.validate.rule_re0 import RE0002

        rule = RE0002()
        violations = rule.check(conn)
        assert len(violations) == 1
        assert violations[0].rule_id == "RE0002"
        conn.close()


class TestFieldLevelOrdering:
    """Tests for RE0003-RE0006 (field-level ordering checks)."""

    def test_re0003_ioip_low_le_mid(self, tmp_path):
        db_path = tmp_path / "test.duckdb"
        conn = duckdb.connect(str(db_path))
        _create_field_test_table(conn)

        _insert_field_row(conn, 2024, "WK1", "FLD1", "1. Low Value", ioip=100)
        _insert_field_row(conn, 2024, "WK1", "FLD1", "2. Middle Value", ioip=50)

        from esdc.validate.rule_re0 import RE0003

        rule = RE0003()
        violations = rule.check(conn)
        assert len(violations) == 1
        assert violations[0].identifiers["field_name"] == "FLD1"
        assert violations[0].current_values["val_ref"] == 100.0
        assert violations[0].current_values["val_cmp"] == 50.0
        conn.close()

    def test_re0003_no_violations_when_equal(self, tmp_path):
        db_path = tmp_path / "test.duckdb"
        conn = duckdb.connect(str(db_path))
        _create_field_test_table(conn)

        _insert_field_row(conn, 2024, "WK1", "FLD1", "1. Low Value", ioip=100)
        _insert_field_row(conn, 2024, "WK1", "FLD1", "2. Middle Value", ioip=100)

        from esdc.validate.rule_re0 import RE0003

        rule = RE0003()
        violations = rule.check(conn)
        assert len(violations) == 0
        conn.close()

    def test_re0005_igip_low_le_mid(self, tmp_path):
        db_path = tmp_path / "test.duckdb"
        conn = duckdb.connect(str(db_path))
        _create_field_test_table(conn)

        _insert_field_row(conn, 2024, "WK1", "FLD1", "1. Low Value", igip=200)
        _insert_field_row(conn, 2024, "WK1", "FLD1", "2. Middle Value", igip=150)

        from esdc.validate.rule_re0 import RE0005

        rule = RE0005()
        violations = rule.check(conn)
        assert len(violations) == 1
        assert violations[0].current_values["val_ref"] == 200.0
        assert violations[0].current_values["val_cmp"] == 150.0
        conn.close()

    def test_field_ordering_aggregates_multiline(self, tmp_path):
        """Two entries for same field: SUM should be compared."""
        db_path = tmp_path / "test.duckdb"
        conn = duckdb.connect(str(db_path))
        _create_field_test_table(conn)

        _insert_field_row(conn, 2024, "WK1", "FLD1", "1. Low Value", ioip=200)
        _insert_field_row(conn, 2024, "WK1", "FLD1", "1. Low Value", ioip=100)
        _insert_field_row(conn, 2024, "WK1", "FLD1", "2. Middle Value", ioip=150)
        _insert_field_row(conn, 2024, "WK1", "FLD1", "2. Middle Value", ioip=50)

        from esdc.validate.rule_re0 import RE0003

        rule = RE0003()
        violations = rule.check(conn)
        assert len(violations) == 1
        assert violations[0].current_values["val_ref"] == 300.0
        assert violations[0].current_values["val_cmp"] == 200.0
        conn.close()

    def test_re0011_finds_negative_reserves(self, tmp_path):
        db_path = tmp_path / "test.duckdb"
        conn = duckdb.connect(str(db_path))
        _create_re0_test_table(conn)

        _insert_full_row(conn, 2024, "Bad", UncertLevel.LOW, res_oil=-10)

        from esdc.validate.rule_re0 import RE0011

        rule = RE0011()
        violations = rule.check(conn)

        assert len(violations) == 1
        assert violations[0].rule_id == "RE0011"
        assert violations[0].current_values["res_oil"] == -10.0
        conn.close()


class TestCategoryBOrdering:
    """Tests for RE0003-RE0006, RE0015-RE0030 (ordering via self-join)."""

    def test_re0003_ioip_low_le_mid(self, tmp_path):
        db_path = tmp_path / "test.duckdb"
        conn = duckdb.connect(str(db_path))
        _create_field_test_table(conn)

        _insert_field_row(conn, 2024, "WK1", "Good", "1. Low Value", ioip=100)
        _insert_field_row(conn, 2024, "WK1", "Good", "2. Middle Value", ioip=200)

        _insert_field_row(conn, 2024, "WK1", "Bad", "1. Low Value", ioip=300)
        _insert_field_row(conn, 2024, "WK1", "Bad", "2. Middle Value", ioip=200)

        from esdc.validate.rule_re0 import RE0003

        rule = RE0003()
        violations = rule.check(conn)

        assert len(violations) == 1
        assert violations[0].rule_id == "RE0003"
        assert violations[0].identifiers["wk_name"] == "WK1"
        assert violations[0].identifiers["field_name"] == "Bad"
        assert violations[0].current_values["val_ref"] == 300.0
        assert violations[0].current_values["val_cmp"] == 200.0
        conn.close()

    def test_re0003_no_violations_when_equal(self, tmp_path):
        db_path = tmp_path / "test.duckdb"
        conn = duckdb.connect(str(db_path))
        _create_field_test_table(conn)

        _insert_field_row(conn, 2024, "WK1", "Equal", "1. Low Value", ioip=100)
        _insert_field_row(conn, 2024, "WK1", "Equal", "2. Middle Value", ioip=100)

        from esdc.validate.rule_re0 import RE0003

        rule = RE0003()
        violations = rule.check(conn)
        assert len(violations) == 0
        conn.close()

    def test_re0023_reserves_1p_le_2p(self, tmp_path):
        db_path = tmp_path / "test.duckdb"
        conn = duckdb.connect(str(db_path))
        _create_re0_test_table(conn)

        _insert_full_row(conn, 2024, "Good", UncertLevel.LOW, res_oil=100)
        _insert_full_row(conn, 2024, "Good", UncertLevel.MID, res_oil=200)

        _insert_full_row(conn, 2024, "Bad", UncertLevel.LOW, res_oil=300)
        _insert_full_row(conn, 2024, "Bad", UncertLevel.MID, res_oil=200)

        from esdc.validate.rule_re0 import RE0023

        rule = RE0023()
        violations = rule.check(conn)

        assert len(violations) == 1
        assert violations[0].identifiers["project_name"] == "Bad"
        conn.close()

    def test_year_filter(self, tmp_path):
        db_path = tmp_path / "test.duckdb"
        conn = duckdb.connect(str(db_path))
        _create_field_test_table(conn)

        _insert_field_row(conn, 2023, "WK1", "Old", "1. Low Value", ioip=300)
        _insert_field_row(conn, 2023, "WK1", "Old", "2. Middle Value", ioip=200)

        _insert_field_row(conn, 2024, "WK1", "New", "1. Low Value", ioip=300)
        _insert_field_row(conn, 2024, "WK1", "New", "2. Middle Value", ioip=200)

        from esdc.validate.rule_re0 import RE0003

        rule = RE0003()
        violations = rule.check(conn, year=[2024])

        assert len(violations) == 1
        assert violations[0].identifiers["report_year"] == "2024"
        conn.close()

    def test_no_cross_work_area_false_violations(self, tmp_path):
        """Same project_name with different wk_name should NOT cross-join."""
        db_path = tmp_path / "test.duckdb"
        conn = duckdb.connect(str(db_path))
        _create_re0_test_table(conn)

        # Same project_name, two work areas — each has valid ordering
        # WK1: IOIP Low=100, Mid=200 (valid)
        # WK2: IOIP Low=500, Mid=600 (valid)
        # Without project_id join, this would produce a false violation
        # (Low from WK2 vs Mid from WK1: 500 > 200)
        _insert_full_row(
            conn,
            2024,
            "SharedProj",
            UncertLevel.LOW,
            ioip=100,
            wk_name="WK1",
            field_name="FLD1",
            project_id="P-AAA",
        )
        _insert_full_row(
            conn,
            2024,
            "SharedProj",
            UncertLevel.MID,
            ioip=200,
            wk_name="WK1",
            field_name="FLD1",
            project_id="P-AAA",
        )
        _insert_full_row(
            conn,
            2024,
            "SharedProj",
            UncertLevel.LOW,
            ioip=500,
            wk_name="WK2",
            field_name="FLD2",
            project_id="P-BBB",
        )
        _insert_full_row(
            conn,
            2024,
            "SharedProj",
            UncertLevel.MID,
            ioip=600,
            wk_name="WK2",
            field_name="FLD2",
            project_id="P-BBB",
        )

        from esdc.validate.rule_re0 import RE0003

        rule = RE0003()
        violations = rule.check(conn)

        # No violations because each project_id has valid Low <= Mid ordering
        assert len(violations) == 0
        conn.close()


class TestCategoryCReservesVsResources:
    """Tests for RE0031-RE0042 (same-row reserves <= resources)."""

    def test_re0031_reserves_le_resources(self, tmp_path):
        db_path = tmp_path / "test.duckdb"
        conn = duckdb.connect(str(db_path))
        _create_re0_test_table(conn)

        _insert_full_row(conn, 2024, "Good", UncertLevel.LOW, res_oil=100, rec_oil=200)
        _insert_full_row(conn, 2024, "Bad", UncertLevel.LOW, res_oil=300, rec_oil=200)

        from esdc.validate.rule_re0 import RE0031

        rule = RE0031()
        violations = rule.check(conn)

        assert len(violations) == 1
        assert violations[0].rule_id == "RE0031"
        assert violations[0].identifiers["project_name"] == "Bad"
        assert violations[0].current_values["val_ref"] == 300.0
        assert violations[0].current_values["val_cmp"] == 200.0
        conn.close()

    def test_re0031_different_uncert_levels(self, tmp_path):
        db_path = tmp_path / "test.duckdb"
        conn = duckdb.connect(str(db_path))
        _create_re0_test_table(conn)

        # Should only check same uncert_level
        _insert_full_row(conn, 2024, "Proj", UncertLevel.LOW, res_oil=100, rec_oil=50)
        _insert_full_row(conn, 2024, "Proj", UncertLevel.MID, res_oil=100, rec_oil=200)

        from esdc.validate.rule_re0 import RE0031

        rule = RE0031()
        violations = rule.check(conn)

        assert len(violations) == 1
        assert violations[0].identifiers["project_name"] == "Proj"
        conn.close()


def _create_agg_test_tables(conn: duckdb.DuckDBPyConnection) -> None:
    """Create project_resources and field_resources for agg consistency tests."""
    conn.execute("""
        CREATE TABLE project_resources (
            id INTEGER,
            report_year INTEGER,
            report_date TEXT,
            report_status TEXT,
            project_name TEXT,
            project_stage TEXT,
            project_class TEXT,
            project_level TEXT,
            uncert_level TEXT,
            wk_id TEXT,
            wk_name TEXT,
            field_id TEXT,
            field_name TEXT,
            prj_ioip REAL,
            prj_igip REAL,
            res_oil REAL,
            res_con REAL,
            res_ga REAL,
            res_gn REAL,
            rec_oil REAL,
            rec_con REAL,
            rec_ga REAL,
            rec_gn REAL,
            cprd_grs_oil REAL,
            cprd_grs_con REAL,
            cprd_grs_ga REAL,
            cprd_grs_gn REAL
        )
    """)
    conn.execute("""
        CREATE TABLE field_resources (
            report_year INTEGER,
            wk_id TEXT,
            wk_name TEXT,
            field_id TEXT,
            field_name TEXT,
            project_stage TEXT,
            project_class TEXT,
            uncert_level TEXT,
            ioip REAL,
            igip REAL,
            res_oil REAL,
            res_con REAL,
            res_ga REAL,
            res_gn REAL,
            rec_oil REAL,
            rec_con REAL,
            rec_ga REAL,
            rec_gn REAL,
            cprd_grs_oil REAL,
            cprd_grs_con REAL,
            cprd_grs_ga REAL,
            cprd_grs_gn REAL
        )
    """)


def _insert_project_row(
    conn: duckdb.DuckDBPyConnection,
    year: int,
    wk_id: str,
    wk_name: str,
    field_id: str,
    field_name: str,
    project_name: str,
    project_stage: str,
    project_class: str,
    uncert: str,
    prj_ioip: float | None = None,
    prj_igip: float | None = None,
) -> None:
    conn.execute(
        "INSERT INTO project_resources"
        " (id, report_year, report_date, report_status, project_name,"
        "  project_stage, project_class, project_level, uncert_level,"
        "  wk_id, wk_name, field_id, field_name,"
        "  prj_ioip, prj_igip,"
        "  res_oil, res_con, res_ga, res_gn,"
        "  rec_oil, rec_con, rec_ga, rec_gn,"
        "  cprd_grs_oil, cprd_grs_con, cprd_grs_ga, cprd_grs_gn)"
        " VALUES (?, ?, '2024-01-01', 'ACTIVE', ?,"
        "         ?, ?, 'LEVEL 1', ?,"
        "         ?, ?, ?, ?,"
        "         ?, ?,"
        "         0, 0, 0, 0,"
        "         0, 0, 0, 0,"
        "         0, 0, 0, 0)",
        [
            1,
            year,
            project_name,
            project_stage,
            project_class,
            uncert,
            wk_id,
            wk_name,
            field_id,
            field_name,
            prj_ioip,
            prj_igip,
        ],
    )


def _insert_field_row_agg(
    conn: duckdb.DuckDBPyConnection,
    year: int,
    wk_id: str,
    wk_name: str,
    field_id: str,
    field_name: str,
    project_stage: str,
    project_class: str,
    uncert: str,
    ioip: float | None = None,
    igip: float | None = None,
) -> None:
    conn.execute(
        "INSERT INTO field_resources"
        " (report_year, wk_id, wk_name, field_id, field_name,"
        "  project_stage, project_class, uncert_level,"
        "  ioip, igip,"
        "  res_oil, res_con, res_ga, res_gn,"
        "  rec_oil, rec_con, rec_ga, rec_gn,"
        "  cprd_grs_oil, cprd_grs_con, cprd_grs_ga, cprd_grs_gn)"
        " VALUES (?, ?, ?, ?, ?,"
        "         ?, ?, ?,"
        "         ?, ?,"
        "         0, 0, 0, 0,"
        "         0, 0, 0, 0,"
        "         0, 0, 0, 0)",
        [
            year,
            wk_id,
            wk_name,
            field_id,
            field_name,
            project_stage,
            project_class,
            uncert,
            ioip,
            igip,
        ],
    )


class TestCategoryDAggregationConsistency:
    """Tests for RE0043-RE0048 (SUM(project) must equal field value)."""

    def test_re0043_mismatch_detected(self, tmp_path):
        db_path = tmp_path / "test.duckdb"
        conn = duckdb.connect(str(db_path))
        _create_agg_test_tables(conn)

        _insert_project_row(
            conn,
            2024,
            "WK1",
            "Work Area 1",
            "F1",
            "Field One",
            "Proj A",
            "DISCOVERED",
            "CLASS A",
            UncertLevel.LOW,
            prj_ioip=100,
        )
        _insert_project_row(
            conn,
            2024,
            "WK1",
            "Work Area 1",
            "F1",
            "Field One",
            "Proj B",
            "DISCOVERED",
            "CLASS A",
            UncertLevel.LOW,
            prj_ioip=200,
        )
        _insert_field_row_agg(
            conn,
            2024,
            "WK1",
            "Work Area 1",
            "F1",
            "Field One",
            "DISCOVERED",
            "CLASS A",
            UncertLevel.LOW,
            ioip=500,
        )

        from esdc.validate.rule_re0 import RE0043

        rule = RE0043()
        violations = rule.check(conn)

        assert len(violations) == 1
        assert violations[0].rule_id == "RE0043"
        assert violations[0].identifiers["field_name"] == "Field One"
        assert violations[0].current_values["val_sum"] == 300.0
        assert violations[0].current_values["val_field"] == 500.0
        conn.close()

    def test_re0043_no_violation_when_equal(self, tmp_path):
        db_path = tmp_path / "test.duckdb"
        conn = duckdb.connect(str(db_path))
        _create_agg_test_tables(conn)

        _insert_project_row(
            conn,
            2024,
            "WK1",
            "Work Area 1",
            "F1",
            "Field One",
            "Proj A",
            "DISCOVERED",
            "CLASS A",
            UncertLevel.LOW,
            prj_ioip=100,
        )
        _insert_project_row(
            conn,
            2024,
            "WK1",
            "Work Area 1",
            "F1",
            "Field One",
            "Proj B",
            "DISCOVERED",
            "CLASS A",
            UncertLevel.LOW,
            prj_ioip=200,
        )
        _insert_field_row_agg(
            conn,
            2024,
            "WK1",
            "Work Area 1",
            "F1",
            "Field One",
            "DISCOVERED",
            "CLASS A",
            UncertLevel.LOW,
            ioip=300,
        )

        from esdc.validate.rule_re0 import RE0043

        rule = RE0043()
        violations = rule.check(conn)
        assert len(violations) == 0
        conn.close()

    def test_re0043_tolerance_not_triggered(self, tmp_path):
        db_path = tmp_path / "test.duckdb"
        conn = duckdb.connect(str(db_path))
        _create_agg_test_tables(conn)

        _insert_project_row(
            conn,
            2024,
            "WK1",
            "Work Area 1",
            "F1",
            "Field One",
            "Proj A",
            "DISCOVERED",
            "CLASS A",
            UncertLevel.LOW,
            prj_ioip=100,
        )
        _insert_project_row(
            conn,
            2024,
            "WK1",
            "Work Area 1",
            "F1",
            "Field One",
            "Proj B",
            "DISCOVERED",
            "CLASS A",
            UncertLevel.LOW,
            prj_ioip=200,
        )
        _insert_field_row_agg(
            conn,
            2024,
            "WK1",
            "Work Area 1",
            "F1",
            "Field One",
            "DISCOVERED",
            "CLASS A",
            UncertLevel.LOW,
            ioip=300.0005,
        )

        from esdc.validate.rule_re0 import RE0043

        rule = RE0043()
        violations = rule.check(conn)
        assert len(violations) == 0
        conn.close()

    def test_re0046_igip_mismatch(self, tmp_path):
        db_path = tmp_path / "test.duckdb"
        conn = duckdb.connect(str(db_path))
        _create_agg_test_tables(conn)

        _insert_project_row(
            conn,
            2024,
            "WK1",
            "Work Area 1",
            "F1",
            "Field One",
            "Proj A",
            "DEVELOPMENT",
            "CLASS B",
            UncertLevel.LOW,
            prj_igip=50,
        )
        _insert_field_row_agg(
            conn,
            2024,
            "WK1",
            "Work Area 1",
            "F1",
            "Field One",
            "DEVELOPMENT",
            "CLASS B",
            UncertLevel.LOW,
            igip=100,
        )

        from esdc.validate.rule_re0 import RE0046

        rule = RE0046()
        violations = rule.check(conn)

        assert len(violations) == 1
        assert violations[0].rule_id == "RE0046"
        conn.close()

    def test_year_filter(self, tmp_path):
        db_path = tmp_path / "test.duckdb"
        conn = duckdb.connect(str(db_path))
        _create_agg_test_tables(conn)

        _insert_project_row(
            conn,
            2023,
            "WK1",
            "Work Area 1",
            "F1",
            "Field Old",
            "Proj A",
            "DISCOVERED",
            "CLASS A",
            UncertLevel.LOW,
            prj_ioip=100,
        )
        _insert_field_row_agg(
            conn,
            2023,
            "WK1",
            "Work Area 1",
            "F1",
            "Field Old",
            "DISCOVERED",
            "CLASS A",
            UncertLevel.LOW,
            ioip=500,
        )

        _insert_project_row(
            conn,
            2024,
            "WK1",
            "Work Area 1",
            "F1",
            "Field New",
            "Proj B",
            "DISCOVERED",
            "CLASS A",
            UncertLevel.LOW,
            prj_ioip=200,
        )
        _insert_field_row_agg(
            conn,
            2024,
            "WK1",
            "Work Area 1",
            "F1",
            "Field New",
            "DISCOVERED",
            "CLASS A",
            UncertLevel.LOW,
            ioip=600,
        )

        from esdc.validate.rule_re0 import RE0043

        rule = RE0043()
        violations = rule.check(conn, year=[2024])

        assert len(violations) == 1
        assert violations[0].identifiers["report_year"] == "2024"
        conn.close()


class TestCategoryEImplication:
    """Tests for RE0049-RE0052 (if 3P > 0 then 1P > 0)."""

    def test_re0049_3p_implies_1p(self, tmp_path):
        db_path = tmp_path / "test.duckdb"
        conn = duckdb.connect(str(db_path))
        _create_re0_test_table(conn)

        # Good: 3P=100, 1P=50
        _insert_full_row(conn, 2024, "Good", UncertLevel.HIGH, res_oil=100)
        _insert_full_row(conn, 2024, "Good", UncertLevel.LOW, res_oil=50)

        # Bad: 3P=100, 1P=0
        _insert_full_row(conn, 2024, "Bad", UncertLevel.HIGH, res_oil=100)
        _insert_full_row(conn, 2024, "Bad", UncertLevel.LOW, res_oil=0)

        from esdc.validate.rule_re0 import RE0049

        rule = RE0049()
        violations = rule.check(conn)

        assert len(violations) == 1
        assert violations[0].rule_id == "RE0049"
        assert violations[0].identifiers["project_name"] == "Bad"
        assert violations[0].current_values["val_ref"] == 100.0
        assert violations[0].current_values["val_cmp"] == 0.0
        conn.close()

    def test_re0049_no_violation_when_3p_zero(self, tmp_path):
        db_path = tmp_path / "test.duckdb"
        conn = duckdb.connect(str(db_path))
        _create_re0_test_table(conn)

        _insert_full_row(conn, 2024, "No3P", UncertLevel.HIGH, res_oil=0)
        _insert_full_row(conn, 2024, "No3P", UncertLevel.LOW, res_oil=0)

        from esdc.validate.rule_re0 import RE0049

        rule = RE0049()
        violations = rule.check(conn)
        assert len(violations) == 0
        conn.close()

    def test_no_cross_work_area_false_violations(self, tmp_path):
        """Same project_name with different wk_name should NOT cross-join."""
        db_path = tmp_path / "test.duckdb"
        conn = duckdb.connect(str(db_path))
        _create_re0_test_table(conn)

        # WK1: 3P=100, 1P=50 (valid)
        # WK2: 3P=0, 1P=0 (valid, no implication triggered)
        _insert_full_row(
            conn,
            2024,
            "SharedProj",
            UncertLevel.HIGH,
            res_oil=100,
            wk_name="WK1",
            field_name="FLD1",
            project_id="P-AAA",
        )
        _insert_full_row(
            conn,
            2024,
            "SharedProj",
            UncertLevel.LOW,
            res_oil=50,
            wk_name="WK1",
            field_name="FLD1",
            project_id="P-AAA",
        )
        _insert_full_row(
            conn,
            2024,
            "SharedProj",
            UncertLevel.HIGH,
            res_oil=0,
            wk_name="WK2",
            field_name="FLD2",
            project_id="P-BBB",
        )
        _insert_full_row(
            conn,
            2024,
            "SharedProj",
            UncertLevel.LOW,
            res_oil=0,
            wk_name="WK2",
            field_name="FLD2",
            project_id="P-BBB",
        )

        from esdc.validate.rule_re0 import RE0049

        rule = RE0049()
        violations = rule.check(conn)

        assert len(violations) == 0
        conn.close()


class TestCategoryFReserveVsPlace:
    """Tests for RE0053-RE0058 (reserve + cumprod < in-place)."""

    def test_re0053_oil_reserve_cumprod_less_than_ioip(self, tmp_path):
        db_path = tmp_path / "test.duckdb"
        conn = duckdb.connect(str(db_path))
        _create_re0_test_table(conn)

        # Good: IOIP=1000, 1P=200, CumProd=100 → 300 < 1000
        _insert_full_row(
            conn,
            2024,
            "Good",
            UncertLevel.LOW,
            ioip=1000,
            res_oil=200,
            cprd_oil=100,
        )

        # Bad: IOIP=1000, 1P=800, CumProd=300 → 1100 >= 1000
        _insert_full_row(
            conn,
            2024,
            "Bad",
            UncertLevel.LOW,
            ioip=1000,
            res_oil=800,
            cprd_oil=300,
        )

        from esdc.validate.rule_re0 import RE0053

        rule = RE0053()
        violations = rule.check(conn)

        assert len(violations) == 1
        assert violations[0].rule_id == "RE0053"
        assert violations[0].identifiers["project_name"] == "Bad"
        assert violations[0].current_values["val_ref"] == 1000.0
        assert violations[0].current_values["val_cmp"] == 800.0
        assert violations[0].current_values["val_sum"] == 300.0
        conn.close()

    def test_re0053_no_violation_when_ioip_zero(self, tmp_path):
        db_path = tmp_path / "test.duckdb"
        conn = duckdb.connect(str(db_path))
        _create_re0_test_table(conn)

        # IOIP=0, so implication doesn't apply
        _insert_full_row(
            conn,
            2024,
            "Zero",
            UncertLevel.LOW,
            ioip=0,
            res_oil=100,
            cprd_oil=50,
        )

        from esdc.validate.rule_re0 import RE0053

        rule = RE0053()
        violations = rule.check(conn)
        assert len(violations) == 0
        conn.close()

    def test_re0056_gas_reserve_cumprod_less_than_igip(self, tmp_path):
        db_path = tmp_path / "test.duckdb"
        conn = duckdb.connect(str(db_path))
        _create_re0_test_table(conn)

        _insert_full_row(
            conn,
            2024,
            "Bad",
            UncertLevel.LOW,
            igip=1000,
            res_gn=800,
            cprd_gn=300,
        )

        from esdc.validate.rule_re0 import RE0056

        rule = RE0056()
        violations = rule.check(conn)

        assert len(violations) == 1
        assert violations[0].rule_id == "RE0056"
        conn.close()


class TestRegistry:
    """Test that all RE0 rules are properly registered."""

    def test_all_re0_rules_registered(self):
        from esdc.validate import get_all_rules

        rules = get_all_rules()
        re0_rules = [r for r in rules if r.rule_id.startswith("RE0")]
        assert len(re0_rules) == 58

        rule_ids = sorted(r.rule_id for r in re0_rules)
        expected = (
            [f"RE{i:04d}" for i in range(1, 49)]  # RE0001-RE0048
            + [f"RE{i:04d}" for i in range(49, 59)]  # RE0049-RE0058
        )
        assert rule_ids == expected
