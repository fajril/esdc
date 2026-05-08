"""Tests for esdc validate command and rules."""

from unittest.mock import patch

import duckdb
from typer.testing import CliRunner

from esdc.esdc import app
from esdc.selection import Severity
from esdc.validate import (
    ValidationResult,
    Violation,
    get_all_rules,
    get_rule,
    get_rules_by_group,
    render_formal,
    run_validation,
)
from esdc.validate.rule_re9 import RE9001

runner = CliRunner()

# --- Helpers ---


def _create_test_table(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute("""
        CREATE TABLE project_resources (
            report_year INTEGER,
            project_name TEXT,
            wk_name TEXT,
            project_isactive INTEGER,
            rec_oil REAL,
            rec_con REAL,
            rec_ga REAL,
            rec_gn REAL,
            rec_oc REAL,
            rec_an REAL,
            rec_oil_risked REAL,
            rec_con_risked REAL,
            rec_ga_risked REAL,
            rec_gn_risked REAL,
            rec_oc_risked REAL,
            rec_an_risked REAL,
            res_oil REAL,
            res_con REAL,
            res_ga REAL,
            res_gn REAL,
            res_oc REAL,
            res_an REAL,
            prj_ioip REAL,
            prj_igip REAL
        )
    """)


# --- Test Registry ---


class TestRegistry:
    def test_register_rule(self):
        assert get_rule("RE9001") is RE9001

    def test_get_rules_by_group(self):
        rules = get_rules_by_group("RE9")
        assert any(r is RE9001 for r in rules)

    def test_get_all_rules(self):
        rules = get_all_rules()
        assert any(r.rule_id == "RE9001" for r in rules)

    def test_get_nonexistent_rule(self):
        assert get_rule("RE9999") is None


# --- Test RE9001 ---


class TestRE9001:
    def test_rule_metadata(self):
        assert RE9001.rule_id == "RE9001"
        assert RE9001.severity == Severity.STRICT
        assert RE9001.is_fixable is True
        assert "project" in RE9001.formal
        assert len(RE9001.ZERO_COLUMNS) == 20  # type: ignore[attr-defined]

    def test_check_finds_violations(self, tmp_path):
        db_path = tmp_path / "test.duckdb"
        conn = duckdb.connect(str(db_path))
        _create_test_table(conn)

        # Active project with volumes -- NOT a violation
        conn.execute(
            "INSERT INTO project_resources VALUES "
            "(2024, 'ActiveProj', 'WK-1', 1, 100, 0, 0, 0, 0, 0, "
            "0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 200, 300)"
        )
        # Inactive project WITH nonzero volume -- SHOULD be a violation
        conn.execute(
            "INSERT INTO project_resources VALUES "
            "(2024, 'InactiveViol', 'WK-1', 0, 150, 0, 0, 0, 0, 0, "
            "0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 200, 0)"
        )
        # Inactive project all zero -- NOT a violation
        conn.execute(
            "INSERT INTO project_resources VALUES "
            "(2024, 'CleanInactive', 'WK-1', 0, 0, 0, 0, 0, 0, 0, "
            "0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)"
        )

        rule = RE9001()
        violations = rule.check(conn)
        assert len(violations) == 1
        assert violations[0].identifiers["project_name"] == "InactiveViol"
        assert violations[0].current_values["rec_oil"] == 150.0
        conn.close()

    def test_check_with_year_filter(self, tmp_path):
        db_path = tmp_path / "test.duckdb"
        conn = duckdb.connect(str(db_path))
        _create_test_table(conn)
        conn.execute(
            "INSERT INTO project_resources VALUES "
            "(2023, 'OldViolation', 'WK-1', 0, 100, 0, 0, 0, 0, 0, "
            "0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)"
        )
        conn.execute(
            "INSERT INTO project_resources VALUES "
            "(2024, 'NewViolation', 'WK-1', 0, 200, 0, 0, 0, 0, 0, "
            "0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)"
        )

        rule = RE9001()
        violations_2024 = rule.check(conn, year=[2024])
        assert len(violations_2024) == 1
        assert violations_2024[0].identifiers["report_year"] == "2024"

        violations_all = rule.check(conn)
        assert len(violations_all) == 2
        conn.close()

    def test_generate_fixes(self, tmp_path):
        db_path = tmp_path / "test.duckdb"
        conn = duckdb.connect(str(db_path))
        _create_test_table(conn)
        conn.execute(
            "INSERT INTO project_resources VALUES "
            "(2024, 'ViolProj', 'WK-1', 0, 150, 0, 0, 0, 0, 0, "
            "0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 200, 0)"
        )

        rule = RE9001()
        violations = rule.check(conn)
        fixes = rule.generate_fixes(violations)
        assert len(fixes) == 1
        fix_sql, fix_params = fixes[0]
        assert "UPDATE project_resources" in fix_sql
        assert "rec_oil = 0" in fix_sql
        assert "prj_ioip = 0" in fix_sql
        assert "project_name = ?" in fix_sql
        assert fix_params == ["ViolProj", 2024]

        # Execute fix and verify
        conn.execute(fix_sql, fix_params)
        result = conn.execute(
            "SELECT rec_oil, prj_ioip FROM project_resources "
            "WHERE project_name = 'ViolProj'"
        ).fetchone()
        assert result == (0.0, 0.0)
        conn.close()

    def test_no_violations(self, tmp_path):
        db_path = tmp_path / "test.duckdb"
        conn = duckdb.connect(str(db_path))
        _create_test_table(conn)
        conn.execute(
            "INSERT INTO project_resources VALUES "
            "(2024, 'GoodProj', 'WK-1', 0, 0, 0, 0, 0, 0, 0, "
            "0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)"
        )
        conn.execute(
            "INSERT INTO project_resources VALUES "
            "(2024, 'ActiveProj', 'WK-1', 1, 100, 0, 0, 0, 0, 0, "
            "0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 200, 300)"
        )

        rule = RE9001()
        violations = rule.check(conn)
        assert len(violations) == 0
        conn.close()


# --- Test render_formal ---


class TestRenderFormal:
    def test_implies(self):
        result = render_formal(r"A \implies B")
        assert "->" in result

    def test_forall(self):
        result = render_formal(r"\forall x \in S")
        assert "for all" in result

    def test_delta(self):
        result = render_formal(r"\Delta N")
        assert "delta" in result


# --- Test CLI ---


class TestValidateCommand:
    def test_validate_help(self):
        result = runner.invoke(app, ["validate", "--help"])
        assert result.exit_code == 0
        assert "validate" in result.stdout

    def test_validate_no_database(self, tmp_path):
        with patch(
            "esdc.validate.rules.Config.get_db_file",
            return_value=tmp_path / "nonexistent.duckdb",
        ):
            result = runner.invoke(app, ["validate"])
            assert result.exit_code == 0
            assert "not found" in result.stdout.lower()

    def test_validate_rule_not_found(self, tmp_path):
        with patch(
            "esdc.validate.rules.Config.get_db_file",
            return_value=tmp_path / "nonexistent.duckdb",
        ):
            result = runner.invoke(app, ["validate", "--rule", "RE9999"])
            assert result.exit_code == 0
            assert (
                "No matching" in result.stdout or "not found" in result.stdout.lower()
            )

    def test_validate_severity_filter_missing_db(self, tmp_path):
        with patch(
            "esdc.validate.rules.Config.get_db_file",
            return_value=tmp_path / "nonexistent.duckdb",
        ):
            result = runner.invoke(app, ["validate", "--severity", "strict"])
            assert result.exit_code == 0

    def test_validate_with_severity_invalid(self):
        with patch("esdc.validate.rules.run_validation", return_value=[]):
            result = runner.invoke(app, ["validate", "--severity", "not_a_severity"])
            assert result.exit_code == 1
            assert "Unknown" in result.stdout

    def test_validate_force_fix(self, tmp_path):
        db_path = tmp_path / "test.duckdb"
        conn = duckdb.connect(str(db_path))
        _create_test_table(conn)
        conn.execute(
            "INSERT INTO project_resources VALUES "
            "(2024, 'Bad', 'WK-1', 0, 150, 0, 0, 0, 0, 0, "
            "0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 200, 0)"
        )
        conn.close()

        with (
            patch("esdc.validate.rules.Config.get_db_file", return_value=db_path),
            patch("esdc.esdc.input", return_value=""),
        ):
            result = runner.invoke(app, ["validate", "--force-fix"])
            assert result.exit_code == 0
            assert "Warning: --force-fix" in result.stdout

        # Verify fix applied
        conn2 = duckdb.connect(str(db_path))
        rows = conn2.execute(
            "SELECT rec_oil, prj_ioip FROM project_resources WHERE project_name = 'Bad'"
        ).fetchall()
        assert len(rows) == 1
        assert rows[0] == (0.0, 0.0)
        conn2.close()


# --- Test run_validation integration ---


class TestRunValidationIntegration:
    def test_run_validation_all_rules(self, tmp_path):
        db_path = tmp_path / "test.duckdb"
        conn = duckdb.connect(str(db_path))
        _create_test_table(conn)
        conn.execute(
            "INSERT INTO project_resources VALUES "
            "(2024, 'Bad', 'WK-1', 0, 150, 0, 0, 0, 0, 0, "
            "0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)"
        )
        conn.close()

        with patch("esdc.validate.rules.Config.get_db_file", return_value=db_path):
            results = run_validation()
            assert len(results) >= 1
            re9001_result = next((r for r in results if r.rule_id == "RE9001"), None)
            assert re9001_result is not None
            assert re9001_result.total_violations == 1
            assert re9001_result.is_fixable is True

    def test_run_validation_no_db(self, tmp_path):
        with patch(
            "esdc.validate.rules.Config.get_db_file",
            return_value=tmp_path / "nonexistent.duckdb",
        ):
            results = run_validation()
            assert results == []


# --- Test ValidationResult / Violation dataclasses ---


class TestDataclasses:
    def test_violation_defaults(self):
        v = Violation(
            rule_id="RE9001",
            rule_group="RE9",
            description="test",
            severity=Severity.STRICT,
            table="project_resources",
            identifiers={"a": "1"},
            current_values={"b": 2},
        )
        assert v.fix_sql is None
        assert v.fix_applied is False

    def test_validation_result_defaults(self):
        r = ValidationResult(
            rule_id="RE9001",
            rule_group="RE9",
            description="test",
            formal="",
            severity=Severity.WARNING,
            is_fixable=True,
            total_violations=0,
            violations=[],
        )
        assert r.fix_applied_count == 0
