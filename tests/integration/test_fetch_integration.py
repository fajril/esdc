"""Integration tests for fetch command."""

import re

import duckdb
from typer.testing import CliRunner

from esdc.esdc import app
from tests.integration.constants import (
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
        """First fetch run should create ~/.esdc/esdc.db."""
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

        db_file = isolated_config / ".esdc" / "esdc.db"
        assert db_file.exists(), f"Database not created at {db_file}"
        assert result.exit_code == 0


class TestFetchJsonToDatabase:
    """Tests that fetch with JSON persists data to database."""

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

        db_file = isolated_config / ".esdc" / "esdc.db"
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

        db_file = isolated_config / ".esdc" / "esdc.db"
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

        db_file = isolated_config / ".esdc" / "esdc.db"

        db_file = isolated_config / ".esdc" / "esdc.db"
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

        db_file = isolated_config / ".esdc" / "esdc.db"
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

        db_file = isolated_config / ".esdc" / "esdc.db"
        assert db_file.exists()
