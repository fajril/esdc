"""Integration tests for fetch command."""

import re

import duckdb
import pytest
from typer.testing import CliRunner

from esdc.api_snapshot import get_project_resources_snapshot
from esdc.esdc import _load_file_as_csv, app
from esdc.pod_registry.store import get_sqlite_connection

from .constants import (
    PROJECT_RESOURCES_URL,
)

runner = CliRunner()


def _resources_url_for_year(year: int) -> re.Pattern:
    """Regex matching project-resources endpoint for a specific year."""
    return re.compile(
        rf"https://esdc\.skkmigas\.go\.id/api/v2/project-resources\?.*report-year={year}.*"
    )


def _timeseries_url_for_year(year: int) -> re.Pattern:
    """Regex matching project-timeseries endpoint for a specific year."""
    return re.compile(
        rf"https://esdc\.skkmigas\.go\.id/api/v2/project-timeseries\?.*report-year={year}.*"
    )


class TestFetchCreatesDatabase:
    """Tests that fetch command creates database correctly."""

    def test_fetch_creates_database(self, isolated_config, mock_esdc_api, monkeypatch):
        """First fetch run should create ~/.esdc/esdc.duckdb."""
        project_resources_data = [
            {
                "project_id": "PRJ-001",
                "project_name": "Test Project",
                "report_year": 2024,
                "project_stage": "DISCOVERED",
            }
        ]
        mock_esdc_api.add_json(PROJECT_RESOURCES_URL, json_data=project_resources_data)
        mock_esdc_api.add_json(
            _timeseries_url_for_year(2024),
            json_data=[{"project_id": "PRJ-001", "year": 2024, "cprd_grs_oil": 100.0}],
        )

        monkeypatch.setenv("ESDC_USER", "testuser")
        monkeypatch.setenv("ESDC_PASS", "testpass")

        result = runner.invoke(app, ["fetch", "--filetype", "json"])

        db_file = isolated_config / ".esdc" / "esdc.duckdb"
        assert db_file.exists(), f"Database not created at {db_file}"
        assert result.exit_code == 0


class TestFetchJsonToDatabase:
    """Tests that fetch with JSON persists data to database."""

    def test_fresh_fetch_reports_empty_upstream_timeseries(
        self, isolated_config, mock_esdc_api, monkeypatch
    ):
        mock_esdc_api.add_json(
            PROJECT_RESOURCES_URL,
            json_data=[
                {"project_id": "PRJ-001", "project_name": "Test", "report_year": 2025}
            ],
        )
        mock_esdc_api.add_json(_timeseries_url_for_year(2025), json_data=[])
        monkeypatch.setenv("ESDC_USER", "testuser")
        monkeypatch.setenv("ESDC_PASS", "testpass")

        result = runner.invoke(app, ["fetch"])

        assert result.exit_code == 0, result.output
        assert (
            "API returned no rows for project_timeseries (report year 2025)"
            in result.output
        )
        assert "FTS index skipped: project_timeseries" in result.output
        assert "Search indexes rebuilt" in result.output
        assert "HNSW embedding index dropped" not in result.output
        assert "✗" not in result.output
        with duckdb.connect(str(isolated_config / ".esdc" / "esdc.duckdb")) as conn:
            assert conn.execute(
                "SELECT COUNT(*) FROM project_resources"
            ).fetchone() == (1,)
            assert conn.execute(
                "SELECT COUNT(*) FROM information_schema.tables "
                "WHERE table_name = 'project_timeseries'"
            ).fetchone() == (0,)

    def test_fetch_json_to_database(self, isolated_config, mock_esdc_api, monkeypatch):
        """Fetch with JSON should persist data to database."""
        project_resources_data = [
            {
                "project_id": "PRJ-001",
                "project_name": "Test Project",
                "report_year": 2024,
                "project_stage": "DISCOVERED",
            }
        ]
        mock_esdc_api.add_json(PROJECT_RESOURCES_URL, json_data=project_resources_data)
        mock_esdc_api.add_json(
            _timeseries_url_for_year(2024),
            json_data=[{"project_id": "PRJ-001", "year": 2024, "cprd_grs_oil": 100.0}],
        )

        monkeypatch.setenv("ESDC_USER", "testuser")
        monkeypatch.setenv("ESDC_PASS", "testpass")

        result = runner.invoke(app, ["fetch", "--filetype", "json"])

        assert result.exit_code == 0

        db_file = isolated_config / ".esdc" / "esdc.duckdb"
        assert db_file.exists()

        conn = duckdb.connect(str(db_file))
        row = conn.execute("SELECT COUNT(*) FROM project_resources").fetchone()
        assert row is not None, "No rows returned"
        conn.close()

        count = row[0]
        assert count == 1, "Data should be persisted to project_resources table"

    def test_fetch_multi_year_timeseries_json(
        self, isolated_config, mock_esdc_api, monkeypatch
    ):
        """Fetch should download timeseries for all available years (>=2020)."""
        project_resources_data = [
            {
                "project_id": "PRJ-001",
                "project_name": "Year 2023",
                "report_year": 2023,
                "project_stage": "DISCOVERED",
            },
            {
                "project_id": "PRJ-002",
                "project_name": "Year 2024",
                "report_year": 2024,
                "project_stage": "DISCOVERED",
            },
        ]
        mock_esdc_api.add_json(PROJECT_RESOURCES_URL, json_data=project_resources_data)
        mock_esdc_api.add_json(
            _timeseries_url_for_year(2023),
            json_data=[{"project_id": "PRJ-001", "year": 2023, "cprd_grs_oil": 50.0}],
        )
        mock_esdc_api.add_json(
            _timeseries_url_for_year(2024),
            json_data=[{"project_id": "PRJ-002", "year": 2024, "cprd_grs_oil": 100.0}],
        )

        monkeypatch.setenv("ESDC_USER", "testuser")
        monkeypatch.setenv("ESDC_PASS", "testpass")

        result = runner.invoke(app, ["fetch", "--filetype", "json"])

        assert result.exit_code == 0

        db_file = isolated_config / ".esdc" / "esdc.duckdb"
        assert db_file.exists()

        conn = duckdb.connect(str(db_file))
        row = conn.execute("SELECT COUNT(*) FROM project_timeseries").fetchone()
        assert row is not None
        count = row[0]
        conn.close()

        assert count == 2, "Timeseries should contain data from both 2023 and 2024"

    def test_fetch_append_year_json(self, isolated_config, mock_esdc_api, monkeypatch):
        """Fetch --year should update both resources and timeseries.

        Verifies that --year updates both project_resources and
        project_timeseries for a specific year.
        """
        # --- Initial full fetch with 2024 data ---
        resources_2024 = [
            {
                "project_id": "PRJ-001",
                "project_name": "Old 2024",
                "report_year": 2024,
                "project_stage": "DISCOVERED",
            }
        ]
        mock_esdc_api.add_json(PROJECT_RESOURCES_URL, json_data=resources_2024)
        mock_esdc_api.add_json(
            _timeseries_url_for_year(2024),
            json_data=[{"project_id": "PRJ-001", "year": 2024, "cprd_grs_oil": 100.0}],
        )

        monkeypatch.setenv("ESDC_USER", "testuser")
        monkeypatch.setenv("ESDC_PASS", "testpass")

        result = runner.invoke(app, ["fetch", "--filetype", "json"])
        assert result.exit_code == 0

        db_file = isolated_config / ".esdc" / "esdc.duckdb"

        db_file = isolated_config / ".esdc" / "esdc.duckdb"
        conn = duckdb.connect(str(db_file))
        row = conn.execute("SELECT COUNT(*) FROM project_resources").fetchone()
        assert row is not None
        resources_count_before = row[0]
        row2 = conn.execute(
            "SELECT project_name FROM project_resources WHERE report_year = 2024"
        ).fetchone()
        assert row2 is not None
        resources_name_before = row2[0]
        conn.close()

        assert resources_count_before == 1
        assert resources_name_before == "Old 2024"

        # --- Now update with --year 2025 for both tables ---
        mock_esdc_api.add_json(
            _resources_url_for_year(2025),
            json_data=[
                {
                    "project_id": "PRJ-001",
                    "project_name": "New 2025",
                    "report_year": 2025,
                    "project_stage": "DISCOVERED",
                }
            ],
        )
        mock_esdc_api.add_json(
            _timeseries_url_for_year(2025),
            json_data=[{"project_id": "PRJ-001", "year": 2025, "cprd_grs_oil": 200.0}],
        )

        result = runner.invoke(app, ["fetch", "--filetype", "json", "--year", "2025"])
        assert result.exit_code == 0

        conn = duckdb.connect(str(db_file))
        # Both tables should have 2 rows (2024 + 2025)
        row = conn.execute("SELECT COUNT(*) FROM project_resources").fetchone()
        assert row is not None
        resources_count = row[0]
        row = conn.execute("SELECT COUNT(*) FROM project_timeseries").fetchone()
        assert row is not None
        timeseries_count = row[0]
        # Resources name for 2024 unchanged (still Old 2024)
        # Resources name for 2025 is New 2025
        row = conn.execute(
            "SELECT project_name FROM project_resources WHERE report_year = 2024"
        ).fetchone()
        assert row is not None
        name_2024 = row[0]
        row = conn.execute(
            "SELECT project_name FROM project_resources WHERE report_year = 2025"
        ).fetchone()
        assert row is not None
        name_2025 = row[0]
        row = conn.execute(
            "SELECT cprd_grs_oil FROM project_timeseries WHERE year = 2025"
        ).fetchone()
        assert row is not None
        cprd_2025 = row[0]
        conn.close()

        assert resources_count == 2, "Resources should contain data for 2024 and 2025"
        assert timeseries_count == 2, "Timeseries should contain data for 2024 and 2025"
        assert name_2024 == "Old 2024"
        assert name_2025 == "New 2025"
        assert cprd_2025 == 200.0

    def test_fetch_year_ignoring_upstream_filters_does_not_duplicate_rows(
        self, isolated_config, mock_esdc_api, monkeypatch
    ):
        initial = [
            {"project_id": "P23", "project_name": "23", "report_year": 2023},
            {"project_id": "P24", "project_name": "24", "report_year": 2024},
            {"project_id": "P25", "project_name": "25", "report_year": 2025},
        ]
        mock_esdc_api.add_json(PROJECT_RESOURCES_URL, json_data=initial)
        mock_esdc_api.add_json(_timeseries_url_for_year(2023), json_data=[])
        mock_esdc_api.add_json(_timeseries_url_for_year(2024), json_data=[])
        mock_esdc_api.add_json(_timeseries_url_for_year(2025), json_data=[])
        monkeypatch.setenv("ESDC_USER", "u")
        monkeypatch.setenv("ESDC_PASS", "p")
        assert runner.invoke(app, ["fetch", "--filetype", "json"]).exit_code == 0
        extra = initial + [{"project_id": "P26", "project_name": "26", "report_year": 2026}]
        mock_esdc_api.add_json(_resources_url_for_year(2024), json_data=extra)
        mock_esdc_api.add_json(
            _resources_url_for_year(2025),
            json_data=[
                {"project_id": "P3", "project_name": "3", "report_year": 2025},
                {"project_id": "P23", "project_name": "23", "report_year": 2023},
                {"project_id": "P24", "project_name": "24", "report_year": 2024},
                {"project_id": "P26", "project_name": "26", "report_year": 2026},
            ],
        )
        mock_esdc_api.add_json(_timeseries_url_for_year(2024), json_data=[])
        mock_esdc_api.add_json(_timeseries_url_for_year(2025), json_data=[])
        result = runner.invoke(
            app, ["fetch", "--filetype", "json", "--year", "2024", "--year", "2025"]
        )
        assert result.exit_code == 0, result.output
        import duckdb

        with duckdb.connect(str(isolated_config / ".esdc" / "esdc.duckdb")) as conn:
            assert conn.execute(
                "SELECT project_id, report_year FROM project_resources ORDER BY project_id"
            ).fetchall() == [("P23", 2023), ("P24", 2024), ("P3", 2025)]

    def test_malformed_second_year_does_not_mutate_snapshot(
        self, isolated_config, mock_esdc_api, monkeypatch
    ):
        initial = [{"project_id": "P24", "project_name": "24", "report_year": 2024}]
        mock_esdc_api.add_json(PROJECT_RESOURCES_URL, json_data=initial)
        mock_esdc_api.add_json(_timeseries_url_for_year(2024), json_data=[])
        monkeypatch.setenv("ESDC_USER", "u")
        monkeypatch.setenv("ESDC_PASS", "p")
        assert runner.invoke(app, ["fetch", "--filetype", "json"]).exit_code == 0
        sqlite_path = isolated_config / ".esdc" / "esdc.sqlite"
        conn = get_sqlite_connection(sqlite_path)
        before_snapshot = get_project_resources_snapshot(conn)
        before_rows = conn.execute("SELECT * FROM api_project_resources").fetchall()
        conn.close()
        with duckdb.connect(str(isolated_config / ".esdc" / "esdc.duckdb")) as db:
            before_duck = db.execute("SELECT * FROM project_resources").fetchall()
        mock_esdc_api.add_json(_resources_url_for_year(2024), json_data=initial)
        mock_esdc_api.add_json(
            _resources_url_for_year(2025), json_data=[{"invalid": 1}]
        )
        mock_esdc_api.add_json(_timeseries_url_for_year(2024), json_data=[])
        mock_esdc_api.add_json(_timeseries_url_for_year(2025), json_data=[])
        result = runner.invoke(
            app, ["fetch", "--filetype", "json", "--year", "2024", "--year", "2025"]
        )
        assert result.exit_code != 0
        conn = get_sqlite_connection(sqlite_path)
        assert get_project_resources_snapshot(conn) == before_snapshot
        assert conn.execute("SELECT * FROM api_project_resources").fetchall() == before_rows
        conn.close()
        with duckdb.connect(str(isolated_config / ".esdc" / "esdc.duckdb")) as db:
            assert db.execute("SELECT * FROM project_resources").fetchall() == before_duck

    def test_download_failure_second_year_preserves_sqlite_and_duckdb(
        self, isolated_config, mock_esdc_api, monkeypatch
    ):
        seed = [
            {"project_id": "P24", "project_name": "old24", "report_year": 2024},
            {"project_id": "P25", "project_name": "old25", "report_year": 2025},
        ]
        mock_esdc_api.add_json(PROJECT_RESOURCES_URL, json_data=seed)
        mock_esdc_api.add_json(_timeseries_url_for_year(2024), json_data=[])
        mock_esdc_api.add_json(_timeseries_url_for_year(2025), json_data=[])
        monkeypatch.setenv("ESDC_USER", "u")
        monkeypatch.setenv("ESDC_PASS", "p")
        assert runner.invoke(app, ["fetch", "--filetype", "json"]).exit_code == 0
        sqlite_path = isolated_config / ".esdc" / "esdc.sqlite"
        conn = get_sqlite_connection(sqlite_path)
        before = get_project_resources_snapshot(conn)
        conn.close()
        mock_esdc_api.add_json(
            _resources_url_for_year(2024),
            json_data=[
                {"project_id": "P24", "project_name": "new24", "report_year": 2024}
            ],
        )
        mock_esdc_api.add_status(_resources_url_for_year(2025), status=500)
        result = runner.invoke(
            app, ["fetch", "--filetype", "json", "--year", "2024", "--year", "2025"]
        )
        assert result.exit_code != 0
        conn = get_sqlite_connection(sqlite_path)
        after = get_project_resources_snapshot(conn)
        rows = [
            tuple(row)
            for row in conn.execute(
                "SELECT project_id, project_name FROM api_project_resources ORDER BY project_id"
            ).fetchall()
        ]
        conn.close()
        assert after == before and rows == [("P24", "old24"), ("P25", "old25")]
        import duckdb

        with duckdb.connect(str(isolated_config / ".esdc" / "esdc.duckdb")) as db:
            assert (
                db.execute(
                    "SELECT project_id, project_name FROM project_resources ORDER BY project_id"
                ).fetchall()
                == rows
            )

    def test_empty_selected_year_preserves_prior_rows_in_both_stores(
        self, isolated_config, mock_esdc_api, monkeypatch
    ):
        seed = [
            {"project_id": "P24", "project_name": "old24", "report_year": 2024},
            {"project_id": "P25", "project_name": "old25", "report_year": 2025},
        ]
        mock_esdc_api.add_json(PROJECT_RESOURCES_URL, json_data=seed)
        mock_esdc_api.add_json(_timeseries_url_for_year(2024), json_data=[])
        mock_esdc_api.add_json(_timeseries_url_for_year(2025), json_data=[])
        monkeypatch.setenv("ESDC_USER", "u")
        monkeypatch.setenv("ESDC_PASS", "p")
        assert runner.invoke(app, ["fetch", "--filetype", "json"]).exit_code == 0
        mock_esdc_api.add_json(
            _resources_url_for_year(2024),
            json_data=[
                {"project_id": "P24", "project_name": "new24", "report_year": 2024}
            ],
        )
        mock_esdc_api.add_json(_resources_url_for_year(2025), json_data=[])
        mock_esdc_api.add_json(_timeseries_url_for_year(2024), json_data=[])
        mock_esdc_api.add_json(_timeseries_url_for_year(2025), json_data=[])
        assert (
            runner.invoke(
                app, ["fetch", "--filetype", "json", "--year", "2024", "--year", "2025"]
            ).exit_code
            == 0
        )
        conn = get_sqlite_connection(isolated_config / ".esdc" / "esdc.sqlite")
        rows = [
            tuple(row)
            for row in conn.execute(
                "SELECT project_id, project_name FROM api_project_resources ORDER BY project_id"
            ).fetchall()
        ]
        conn.close()
        assert rows == [("P24", "new24"), ("P25", "old25")]
        import duckdb

        with duckdb.connect(str(isolated_config / ".esdc" / "esdc.duckdb")) as db:
            assert (
                db.execute(
                    "SELECT project_id, project_name FROM project_resources ORDER BY project_id"
                ).fetchall()
                == rows
            )

    def test_json_csv_parity_preserves_complete_values(
        self, isolated_config, mock_esdc_api, monkeypatch
    ):
        row = {
            "project_id": "0007",
            "project_name": "Parity",
            "report_year": 2024,
            "field_geom": "POINT (1 2)",
            "rec_oil": 12.5,
            "grov_current_situation": "line one\nline two",
        }
        mock_esdc_api.add_json(PROJECT_RESOURCES_URL, json_data=[row])
        mock_esdc_api.add_json(_timeseries_url_for_year(2024), json_data=[])
        monkeypatch.setenv("ESDC_USER", "u")
        monkeypatch.setenv("ESDC_PASS", "p")
        assert runner.invoke(app, ["fetch", "--filetype", "json"]).exit_code == 0
        conn = get_sqlite_connection(isolated_config / ".esdc" / "esdc.sqlite")
        json_snapshot = get_project_resources_snapshot(conn)
        json_rows = [
            tuple(row)
            for row in conn.execute(
                "SELECT project_id, report_year, field_geom, rec_oil, groovy_report FROM api_project_resources"
            ).fetchall()
        ]
        conn.close()
        csv_body = 'grov_current_situation;rec_oil;field_geom;report_year;project_name;project_id\n"line one\nline two";12.5;"POINT (1 2)";2024;Parity;0007\n'
        mock_esdc_api.add_csv(PROJECT_RESOURCES_URL, body=csv_body)
        mock_esdc_api.add_csv(
            _timeseries_url_for_year(2024), body="project_id;year\n0007;2024\n"
        )
        assert runner.invoke(app, ["fetch", "--filetype", "csv"]).exit_code == 0
        conn = get_sqlite_connection(isolated_config / ".esdc" / "esdc.sqlite")
        csv_snapshot = get_project_resources_snapshot(conn)
        csv_rows = [
            tuple(row)
            for row in conn.execute(
                "SELECT project_id, report_year, field_geom, rec_oil, groovy_report FROM api_project_resources"
            ).fetchall()
        ]
        conn.close()
        assert csv_rows == json_rows
        assert (
            csv_snapshot is not None
            and json_snapshot is not None
            and csv_snapshot.source_sha256 == json_snapshot.source_sha256
        )

    def test_header_only_csv_reload_preserves_snapshot(
        self, isolated_config, mock_esdc_api, monkeypatch, tmp_path
    ):
        seed = [{"project_id": "P24", "project_name": "old24", "report_year": 2024}]
        mock_esdc_api.add_json(PROJECT_RESOURCES_URL, json_data=seed)
        mock_esdc_api.add_json(_timeseries_url_for_year(2024), json_data=[])
        monkeypatch.setenv("ESDC_USER", "u")
        monkeypatch.setenv("ESDC_PASS", "p")
        assert runner.invoke(app, ["fetch", "--filetype", "json"]).exit_code == 0
        path = tmp_path / "empty.csv"
        path.write_text("project_id;project_name;report_year\n")
        conn = get_sqlite_connection(isolated_config / ".esdc" / "esdc.sqlite")
        before = get_project_resources_snapshot(conn)
        conn.close()
        _load_file_as_csv(str(path), "project_resources")
        conn = get_sqlite_connection(isolated_config / ".esdc" / "esdc.sqlite")
        assert get_project_resources_snapshot(conn) == before
        conn.close()

    @pytest.mark.parametrize(
        ("filetype", "body", "expected_exit"),
        [
            ("json", [], 0),
            ("json", {}, 1),
            ("json", None, 1),
            ("json", False, 1),
            ("csv", "project_id;project_name;report_year\n", 0),
        ],
    )
    def test_empty_or_malformed_full_project_response_preserves_truth(
        self, isolated_config, mock_esdc_api, monkeypatch, filetype, body, expected_exit
    ):
        monkeypatch.setenv("ESDC_USER", "u")
        monkeypatch.setenv("ESDC_PASS", "p")
        mock_esdc_api.add_json(
            PROJECT_RESOURCES_URL,
            json_data=[{"project_id": "KEEP", "project_name": "Keep", "report_year": 2024}],
        )
        mock_esdc_api.add_json(_timeseries_url_for_year(2024), json_data=[])
        assert runner.invoke(app, ["fetch", "--filetype", "json"]).exit_code == 0
        sqlite_path = isolated_config / ".esdc" / "esdc.sqlite"
        conn = get_sqlite_connection(sqlite_path)
        before_snapshot = get_project_resources_snapshot(conn)
        before_rows = conn.execute(
            "SELECT project_id, project_name, report_year FROM api_project_resources"
        ).fetchall()
        conn.close()
        with duckdb.connect(str(isolated_config / ".esdc" / "esdc.duckdb")) as db:
            before_duck = db.execute(
                "SELECT project_id, project_name, report_year FROM project_resources"
            ).fetchall()
        if filetype == "json":
            mock_esdc_api.add_json(PROJECT_RESOURCES_URL, json_data=body)
        else:
            mock_esdc_api.add_csv(PROJECT_RESOURCES_URL, body=body)
        mock_esdc_api.add_json(_timeseries_url_for_year(2024), json_data=[])
        result = runner.invoke(app, ["fetch", "--filetype", filetype])
        assert (result.exit_code != 0) == (expected_exit != 0), result.output
        conn = get_sqlite_connection(sqlite_path)
        assert get_project_resources_snapshot(conn) == before_snapshot
        assert conn.execute(
            "SELECT project_id, project_name, report_year FROM api_project_resources"
        ).fetchall() == before_rows
        conn.close()
        with duckdb.connect(str(isolated_config / ".esdc" / "esdc.duckdb")) as db:
            assert db.execute(
                "SELECT project_id, project_name, report_year FROM project_resources"
            ).fetchall() == before_duck

    def test_canonical_csv_full_fetch_and_local_reload_preserve_fields(
        self, isolated_config, mock_esdc_api, monkeypatch, tmp_path
    ):
        csv_body = (
            "project_id;report_year;groovy_report;basin86_id;basin86\n"
            "P-CAN;2024;Canonical report;B86-1;Basin 86\n"
        )
        monkeypatch.setenv("ESDC_USER", "u")
        monkeypatch.setenv("ESDC_PASS", "p")
        mock_esdc_api.add_csv(PROJECT_RESOURCES_URL, body=csv_body)
        mock_esdc_api.add_csv(_timeseries_url_for_year(2024), body="project_id;year\n")
        assert runner.invoke(app, ["fetch", "--filetype", "csv"]).exit_code == 0
        path = tmp_path / "canonical.csv"
        path.write_text(csv_body)
        _load_file_as_csv(str(path), "project_resources")
        conn = get_sqlite_connection(isolated_config / ".esdc" / "esdc.sqlite")
        assert conn.execute(
            "SELECT groovy_report, basin86_id, basin86 FROM api_project_resources"
        ).fetchone()[:3] == ("Canonical report", "B86-1", "Basin 86")
        conn.close()
        with duckdb.connect(str(isolated_config / ".esdc" / "esdc.duckdb")) as db:
            assert db.execute(
                "SELECT groovy_report, basin86_id, basin86 FROM project_resources"
            ).fetchone() == ("Canonical report", "B86-1", "Basin 86")

    def test_two_year_csv_failure_leaves_snapshot_unchanged(
        self, isolated_config, mock_esdc_api, monkeypatch
    ):
        monkeypatch.setenv("ESDC_USER", "u")
        monkeypatch.setenv("ESDC_PASS", "p")
        seed = [{"project_id": "KEEP", "project_name": "Keep", "report_year": 2024}]
        mock_esdc_api.add_json(PROJECT_RESOURCES_URL, json_data=seed)
        mock_esdc_api.add_json(_timeseries_url_for_year(2024), json_data=[])
        assert runner.invoke(app, ["fetch", "--filetype", "json"]).exit_code == 0
        conn = get_sqlite_connection(isolated_config / ".esdc" / "esdc.sqlite")
        before = get_project_resources_snapshot(conn)
        before_rows = conn.execute("SELECT * FROM api_project_resources").fetchall()
        conn.close()
        with duckdb.connect(str(isolated_config / ".esdc" / "esdc.duckdb")) as db:
            before_duck = db.execute("SELECT * FROM project_resources").fetchall()
        mock_esdc_api.add_csv(
            _resources_url_for_year(2024),
            body="project_id;report_year;project_name;groovy_report\nNEW24;2024;New;Canonical\n",
        )
        mock_esdc_api.add_status(_resources_url_for_year(2025), 500)
        result = runner.invoke(app, ["fetch", "--filetype", "csv", "--year", "2024", "--year", "2025"])
        assert result.exit_code != 0
        conn = get_sqlite_connection(isolated_config / ".esdc" / "esdc.sqlite")
        assert get_project_resources_snapshot(conn) == before
        assert conn.execute("SELECT * FROM api_project_resources").fetchall() == before_rows
        conn.close()
        with duckdb.connect(str(isolated_config / ".esdc" / "esdc.duckdb")) as db:
            assert db.execute("SELECT * FROM project_resources").fetchall() == before_duck

    def test_selected_year_without_matching_rows_is_noop(
        self, isolated_config, mock_esdc_api, monkeypatch
    ):
        monkeypatch.setenv("ESDC_USER", "u")
        monkeypatch.setenv("ESDC_PASS", "p")
        mock_esdc_api.add_json(
            PROJECT_RESOURCES_URL,
            json_data=[{"project_id": "KEEP", "report_year": 2024}],
        )
        mock_esdc_api.add_json(_timeseries_url_for_year(2024), json_data=[])
        assert runner.invoke(app, ["fetch", "--filetype", "json"]).exit_code == 0
        conn = get_sqlite_connection(isolated_config / ".esdc" / "esdc.sqlite")
        before = get_project_resources_snapshot(conn)
        conn.close()
        mock_esdc_api.add_json(
            _resources_url_for_year(2025),
            json_data=[{"project_id": "OTHER", "report_year": 2024}],
        )
        mock_esdc_api.add_json(_timeseries_url_for_year(2025), json_data=[])
        result = runner.invoke(app, ["fetch", "--filetype", "json", "--year", "2025"])
        assert result.exit_code == 0
        assert "No project-resource rows for report year 2025" in result.output
        conn = get_sqlite_connection(isolated_config / ".esdc" / "esdc.sqlite")
        assert get_project_resources_snapshot(conn) == before
        conn.close()


class TestFetchCsvToDatabase:
    """Tests that fetch with CSV persists data to database."""

    def test_fetch_csv_to_database(self, isolated_config, mock_esdc_api, monkeypatch):
        """Fetch with CSV should persist data to database."""
        csv_content = (
            "project_id;project_name;report_year;project_stage\n"
            "PRJ-002;CSV Project;2024;DEVELOPMENT"
        )
        csv_timeseries = "project_id;year;cprd_grs_oil\nPRJ-002;2024;200.0"
        mock_esdc_api.add_csv(PROJECT_RESOURCES_URL, body=csv_content)
        mock_esdc_api.add_csv(_timeseries_url_for_year(2024), body=csv_timeseries)

        monkeypatch.setenv("ESDC_USER", "testuser")
        monkeypatch.setenv("ESDC_PASS", "testpass")

        result = runner.invoke(app, ["fetch", "--filetype", "csv"])

        assert result.exit_code == 0

        db_file = isolated_config / ".esdc" / "esdc.duckdb"
        assert db_file.exists()

        conn = duckdb.connect(str(db_file))
        cursor = conn.execute("SELECT COUNT(*) FROM project_resources")
        row = cursor.fetchone()
        assert row is not None, "No rows returned"
        count = row[0]
        conn.close()

        assert count == 1, "Data should be persisted to project_resources table"


class TestFetchAuthFailure:
    """Tests that fetch handles auth failures correctly."""

    def test_fetch_auth_failure(
        self, isolated_config, mock_esdc_api, monkeypatch, caplog
    ):
        """API returns 401 should show clear error message."""
        import logging

        caplog.set_level(logging.WARNING)
        mock_esdc_api.add_status(PROJECT_RESOURCES_URL, status=401)

        monkeypatch.setenv("ESDC_USER", "baduser")
        monkeypatch.setenv("ESDC_PASS", "badpass")

        runner.invoke(app, ["fetch", "--filetype", "json"])

        # Check logs for warning messages
        log_messages = caplog.text
        assert "401" in log_messages or "Failed to download" in log_messages


class TestFetchMalformedJson:
    """Tests that fetch handles malformed JSON correctly."""

    def test_fetch_malformed_json(self, isolated_config, mock_esdc_api, monkeypatch):
        """Invalid JSON response should fail gracefully."""
        mock_esdc_api.add_json(
            PROJECT_RESOURCES_URL,
            json_data=[{"project_id": "PRJ-001", "invalid_column": "data"}],
        )
        mock_esdc_api.add_json(
            _timeseries_url_for_year(2024),
            json_data=[{"project_id": "PRJ-001", "year": 2024}],
        )

        monkeypatch.setenv("ESDC_USER", "testuser")
        monkeypatch.setenv("ESDC_PASS", "testpass")

        result = runner.invoke(app, ["fetch", "--filetype", "json"])

        assert result.exit_code == 0 or result.exception is not None


class TestFetchWithEnvCredentials:
    """Tests that fetch works with environment credentials."""

    def test_fetch_with_env_credentials(
        self, isolated_config, mock_esdc_api, monkeypatch
    ):
        """ESDC_USER and ESDC_PASS env vars should work."""
        project_resources_data = [
            {
                "project_id": "PRJ-003",
                "project_name": "Env Test",
                "report_year": 2024,
                "project_stage": "PRODUCTION",
            }
        ]
        mock_esdc_api.add_json(PROJECT_RESOURCES_URL, json_data=project_resources_data)
        mock_esdc_api.add_json(
            _timeseries_url_for_year(2024),
            json_data=[{"project_id": "PRJ-003", "year": 2024, "cprd_grs_oil": 300.0}],
        )

        monkeypatch.setenv("ESDC_USER", "envuser")
        monkeypatch.setenv("ESDC_PASS", "envpass")

        result = runner.invoke(app, ["fetch", "--filetype", "json"])

        assert result.exit_code == 0

        db_file = isolated_config / ".esdc" / "esdc.duckdb"
        assert db_file.exists()
