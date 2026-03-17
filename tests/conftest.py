import pytest
from typer.testing import CliRunner


@pytest.fixture
def runner():
    """Create a fresh CliRunner for each test."""
    return CliRunner()


@pytest.fixture(autouse=True)
def reset_config_cache():
    """Reset Config cache before each test."""
    from esdc.configs import Config

    Config._config_cache = None
    yield


@pytest.fixture
def isolated_config(tmp_path, monkeypatch):
    """Create isolated config directory for integration tests.

    Sets ESDC_CONFIG_DIR to a temporary directory and ESDC_DB_FILE
    to avoid polluting the user's actual ~/.esdc/ directory.
    """
    config_dir = tmp_path / ".esdc"
    db_file = config_dir / "esdc.db"
    monkeypatch.setenv("ESDC_CONFIG_DIR", str(config_dir))
    monkeypatch.setenv("ESDC_DB_FILE", str(db_file))
    yield tmp_path


@pytest.fixture
def mock_esdc_api():
    """Factory to add mock ESDC API responses.

    Usage:
        mock = mock_esdc_api()
        mock.add_json("https://esdc.skkmigas.go.id/api/v2/...", {...})
        mock.add_csv("https://esdc.skkmigas.go.id/api/v2/...", "col1,col2\nval1,val2")
    """
    import re
    import responses as responses_lib

    responses_lib.start()

    def _add_response(method, pattern, **kwargs):
        url_pattern = re.compile(pattern)
        responses_lib.add(method, url_pattern, **kwargs)

    class MockAPI:
        def add_json(self, pattern, json_data, status=200):
            _add_response(responses_lib.GET, pattern, json=json_data, status=status)

        def add_csv(self, pattern, body, status=200):
            _add_response(responses_lib.GET, pattern, body=body, status=status)

        def add_status(self, pattern, status):
            _add_response(responses_lib.GET, pattern, status=status)

    yield MockAPI()

    responses_lib.stop()
    responses_lib.reset()


@pytest.fixture
def seeded_database(isolated_config):
    """Create database with sample project_resources data.

    Returns the tmp_path so tests can reference the database file.
    """
    from esdc.dbmanager import load_data_to_db

    sample_data = [
        ["PRJ-001", "Test Project Alpha", "2024", "DISCOVERED", "100.00"],
        ["PRJ-002", "Test Project Beta", "2024", "DEVELOPMENT", "200.00"],
        ["PRJ-003", "Test Project Gamma", "2023", "PRODUCTION", "300.00"],
    ]
    header = ["project_code", "project_name", "report_year", "project_stage", "budget"]

    # Ensure config is initialized
    from esdc.configs import Config

    Config.init_config()

    load_data_to_db(sample_data, header, "project_resources")

    return isolated_config
