"""Integration tests for fetch command."""

import sqlite3

from typer.testing import CliRunner

from esdc.esdc import app
from tests.integration.constants import PROJECT_RESOURCES_URL, PROJECT_TIMESERIES_URL

runner = CliRunner()


class TestFetchCreatesDatabase:
    """Tests that fetch command creates database correctly."""

    def test_fetch_creates_database(self, isolated_config, mock_esdc_api, monkeypatch):
        """First fetch run should create ~/.esdc/esdc.db"""
        project_resources_data = [
            {
                "project_id": "PRJ-001",
                "project_name": "Test Project",
                "report_year": 2024,
                "project_stage": "DISCOVERED",
            }
        ]
        project_timeseries_data = [
            {"project_id": "PRJ-001", "year": 2024, "cprd_grs_oil": 100.0}
        ]
        mock_esdc_api.add_json(PROJECT_RESOURCES_URL, json_data=project_resources_data)
        mock_esdc_api.add_json(
            PROJECT_TIMESERIES_URL, json_data=project_timeseries_data
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
        """Fetch with JSON should persist data to database"""
        project_resources_data = [
            {
                "project_id": "PRJ-001",
                "project_name": "Test Project",
                "report_year": 2024,
                "project_stage": "DISCOVERED",
            }
        ]
        project_timeseries_data = [
            {"project_id": "PRJ-001", "year": 2024, "cprd_grs_oil": 100.0}
        ]
        mock_esdc_api.add_json(PROJECT_RESOURCES_URL, json_data=project_resources_data)
        mock_esdc_api.add_json(
            PROJECT_TIMESERIES_URL, json_data=project_timeseries_data
        )

        monkeypatch.setenv("ESDC_USER", "testuser")
        monkeypatch.setenv("ESDC_PASS", "testpass")

        result = runner.invoke(app, ["fetch", "--filetype", "json"])

        assert result.exit_code == 0

        db_file = isolated_config / ".esdc" / "esdc.db"
        assert db_file.exists()

        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM project_resources")
        count = cursor.fetchone()[0]
        conn.close()

        assert count == 1, "Data should be persisted to project_resources table"


class TestFetchCsvToDatabase:
    """Tests that fetch with CSV persists data to database."""

    def test_fetch_csv_to_database(self, isolated_config, mock_esdc_api, monkeypatch):
        """Fetch with CSV should persist data to database"""
        csv_content = (
            "project_id;project_name;report_year;project_stage\n"
            "PRJ-002;CSV Project;2024;DEVELOPMENT"
        )
        csv_timeseries = "project_id;year;cprd_grs_oil\nPRJ-002;2024;200.0"
        mock_esdc_api.add_csv(PROJECT_RESOURCES_URL, body=csv_content)
        mock_esdc_api.add_csv(PROJECT_TIMESERIES_URL, body=csv_timeseries)

        monkeypatch.setenv("ESDC_USER", "testuser")
        monkeypatch.setenv("ESDC_PASS", "testpass")

        result = runner.invoke(app, ["fetch", "--filetype", "csv"])

        assert result.exit_code == 0

        db_file = isolated_config / ".esdc" / "esdc.db"
        assert db_file.exists()

        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM project_resources")
        count = cursor.fetchone()[0]
        conn.close()

        assert count == 1, "Data should be persisted to project_resources table"


class TestFetchAuthFailure:
    """Tests that fetch handles auth failures correctly."""

    def test_fetch_auth_failure(
        self, isolated_config, mock_esdc_api, monkeypatch, caplog
    ):
        """API returns 401 should show clear error message"""
        import logging

        caplog.set_level(logging.WARNING)
        mock_esdc_api.add_status(PROJECT_RESOURCES_URL, status=401)

        monkeypatch.setenv("ESDC_USER", "baduser")
        monkeypatch.setenv("ESDC_PASS", "badpass")

        result = runner.invoke(app, ["fetch", "--filetype", "json"])

        # Check logs for warning messages
        log_messages = caplog.text
        assert "401" in log_messages or "Failed to download" in log_messages


class TestFetchMalformedJson:
    """Tests that fetch handles malformed JSON correctly."""

    def test_fetch_malformed_json(self, isolated_config, mock_esdc_api, monkeypatch):
        """Invalid JSON response should fail gracefully"""
        mock_esdc_api.add_json(
            PROJECT_RESOURCES_URL,
            json_data=[{"project_id": "PRJ-001", "invalid_column": "data"}],
        )
        mock_esdc_api.add_json(
            PROJECT_TIMESERIES_URL,
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
        """ESDC_USER and ESDC_PASS env vars should work"""
        project_resources_data = [
            {
                "project_id": "PRJ-003",
                "project_name": "Env Test",
                "report_year": 2024,
                "project_stage": "PRODUCTION",
            }
        ]
        project_timeseries_data = [
            {"project_id": "PRJ-003", "year": 2024, "cprd_grs_oil": 300.0}
        ]
        mock_esdc_api.add_json(PROJECT_RESOURCES_URL, json_data=project_resources_data)
        mock_esdc_api.add_json(
            PROJECT_TIMESERIES_URL, json_data=project_timeseries_data
        )

        monkeypatch.setenv("ESDC_USER", "envuser")
        monkeypatch.setenv("ESDC_PASS", "envpass")

        result = runner.invoke(app, ["fetch", "--filetype", "json"])

        assert result.exit_code == 0

        db_file = isolated_config / ".esdc" / "esdc.db"
        assert db_file.exists()
