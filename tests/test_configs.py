import os
from pathlib import Path
from unittest.mock import patch

from esdc.configs import Config


class TestConfigDir:
    """Tests for get_config_dir()."""

    def test_get_config_dir_default(self):
        """Test default config directory is ~/.esdc."""
        with patch.dict(os.environ, {}, clear=True):
            config_dir = Config.get_config_dir()
            assert config_dir == Path.home() / ".esdc"

    def test_get_config_dir_env_override(self, tmp_path):
        """Test ESDC_CONFIG_DIR environment variable overrides default."""
        with patch.dict(os.environ, {"ESDC_CONFIG_DIR": str(tmp_path)}):
            config_dir = Config.get_config_dir()
            assert config_dir == tmp_path


class TestConfigFile:
    """Tests for config file operations."""

    def test_get_config_file(self, tmp_path):
        """Test config file path is within config directory."""
        with patch.object(Config, "get_config_dir", return_value=tmp_path):
            config_file = Config.get_config_file()
            assert config_file == tmp_path / "config.yaml"

    def test_init_config_creates_directory(self, tmp_path):
        """Test init_config creates config directory if not exists."""
        with patch.object(Config, "get_config_dir", return_value=tmp_path / "esdc"):
            Config.init_config()
            assert (tmp_path / "esdc").exists()

    def test_init_config_creates_file(self, tmp_path):
        """Test init_config creates config file with defaults."""
        with patch.object(Config, "get_config_dir", return_value=tmp_path):
            Config.init_config()
            config_file = tmp_path / "config.yaml"
            assert config_file.exists()

    def test_init_config_does_not_overwrite_existing(self, tmp_path):
        """Test init_config does not overwrite existing config."""
        with patch.object(Config, "get_config_dir", return_value=tmp_path):
            config_file = tmp_path / "config.yaml"
            config_file.write_text("custom: value\n")
            Config.init_config()
            assert config_file.read_text() == "custom: value\n"


class TestApiUrl:
    """Tests for get_api_url()."""

    def test_get_api_url_env_override(self):
        """Test ESDC_URL environment variable has highest priority."""
        with patch.dict(os.environ, {"ESDC_URL": "https://custom.example.com/"}):
            assert Config.get_api_url() == "https://custom.example.com/"

    def test_get_api_url_config_file(self, tmp_path):
        """Test config file is used when no env var."""
        with patch.object(Config, "get_config_dir", return_value=tmp_path):
            config_file = tmp_path / "config.yaml"
            config_file.write_text("api_url: https://config.example.com/\n")
            assert Config.get_api_url() == "https://config.example.com/"

    def test_get_api_url_default(self, tmp_path):
        """Test default API URL when no env or config."""
        with patch.object(Config, "get_config_dir", return_value=tmp_path):
            assert Config.get_api_url() == Config.BASE_API_URL_V2


class TestCredentials:
    """Tests for get_credentials()."""

    def test_get_credentials_from_env(self):
        """Test credentials from environment variables."""
        with patch.dict(os.environ, {"ESDC_USER": "testuser", "ESDC_PASS": "testpass"}):
            username, password = Config.get_credentials()
            assert username == "testuser"
            assert password == "testpass"

    def test_get_credentials_from_prompt(self, mocker):
        """Test interactive prompt when env vars not set."""
        mocker.patch.dict(os.environ, {}, clear=True)
        mocker.patch("rich.prompt.Prompt.ask", side_effect=["promptuser", "promptpass"])

        username, password = Config.get_credentials()

        assert username == "promptuser"
        assert password == "promptpass"

    def test_get_credentials_partial_env(self, mocker):
        """Test partial credentials from env, prompt for missing."""
        mocker.patch.dict(os.environ, {"ESDC_USER": "envuser"}, clear=True)
        mocker.patch("rich.prompt.Prompt.ask", return_value="promptpass")

        username, password = Config.get_credentials()

        assert username == "envuser"
        assert password == "promptpass"


class TestDbFile:
    """Tests for database file path."""

    def test_get_db_file_env_override(self, tmp_path):
        """Test ESDC_DB_FILE environment variable overrides all."""
        db_path = tmp_path / "custom.db"
        with patch.dict(os.environ, {"ESDC_DB_FILE": str(db_path)}):
            assert Config.get_db_file() == db_path

    def test_get_db_file_config(self, tmp_path):
        """Test database path from config file."""
        with patch.object(Config, "get_config_dir", return_value=tmp_path):
            config_file = tmp_path / "config.yaml"
            config_file.write_text("database_path: /custom/path/db.db\n")
            assert Config.get_db_file() == Path("/custom/path/db.db")

    def test_get_db_file_default(self, tmp_path):
        """Test default database file path."""
        with patch.object(Config, "get_config_dir", return_value=tmp_path):
            assert Config.get_db_file() == tmp_path / "esdc.duckdb"

    def test_get_db_file_migrates_legacy_default(self, tmp_path):
        """Test legacy default database is renamed to the new default."""
        legacy_db = tmp_path / "esdc.db"
        default_db = tmp_path / "esdc.duckdb"
        legacy_db.write_text("legacy")

        with patch.object(Config, "get_config_dir", return_value=tmp_path):
            assert Config.get_db_file() == default_db

        assert default_db.read_text() == "legacy"
        assert not legacy_db.exists()

    def test_get_db_file_migrates_legacy_default_config(self, tmp_path):
        """Test generated legacy config path is migrated to the new default."""
        legacy_db = tmp_path / "esdc.db"
        default_db = tmp_path / "esdc.duckdb"
        legacy_db.write_text("legacy")
        config_file = tmp_path / "config.yaml"
        config_file.write_text(f"database_path: {legacy_db}\n")

        with patch.object(Config, "get_config_dir", return_value=tmp_path):
            assert Config.get_db_file() == default_db

        assert default_db.read_text() == "legacy"
        assert not legacy_db.exists()
        assert f"database_path: {default_db}" in config_file.read_text()

    def test_get_db_file_env_override_does_not_migrate_legacy_default(self, tmp_path):
        """Test ESDC_DB_FILE prevents default-path migration."""
        legacy_db = tmp_path / "esdc.db"
        env_db = tmp_path / "custom.db"
        legacy_db.write_text("legacy")

        with (
            patch.object(Config, "get_config_dir", return_value=tmp_path),
            patch.dict(os.environ, {"ESDC_DB_FILE": str(env_db)}),
        ):
            assert Config.get_db_file() == env_db

        assert legacy_db.exists()
        assert not (tmp_path / "esdc.duckdb").exists()

    def test_get_db_file_custom_config_does_not_migrate_legacy_default(self, tmp_path):
        """Test explicit database_path prevents default-path migration."""
        legacy_db = tmp_path / "esdc.db"
        custom_db = tmp_path / "custom.db"
        legacy_db.write_text("legacy")
        config_file = tmp_path / "config.yaml"
        config_file.write_text(f"database_path: {custom_db}\n")

        with patch.object(Config, "get_config_dir", return_value=tmp_path):
            assert Config.get_db_file() == custom_db

        assert legacy_db.exists()
        assert not (tmp_path / "esdc.duckdb").exists()

    def test_get_db_dir_env_override(self, tmp_path):
        """Test ESDC_DB_DIR environment variable."""
        with patch.dict(os.environ, {"ESDC_DB_DIR": str(tmp_path)}):
            assert Config.get_db_dir() == tmp_path

    def test_get_db_path_backwards_compat(self, tmp_path):
        """Test get_db_path returns directory (backwards compatibility)."""
        with patch.object(Config, "get_config_dir", return_value=tmp_path):
            assert Config.get_db_path() == tmp_path
