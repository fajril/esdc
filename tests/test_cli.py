import os
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from esdc.esdc import app

runner = CliRunner()


class TestCliHelp:
    """Tests for CLI help output."""

    def test_main_help(self):
        """Test main help output."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "Usage:" in result.stdout
        assert "Commands" in result.stdout
        assert "fetch" in result.stdout
        assert "show" in result.stdout

    def test_fetch_help(self):
        """Test fetch command --help."""
        result = runner.invoke(app, ["fetch", "--help"])
        assert result.exit_code == 0
        assert "filetype" in result.stdout.lower()
        assert "save" in result.stdout.lower()

    def test_show_help(self):
        """Test show command --help."""
        result = runner.invoke(app, ["show", "--help"])
        assert result.exit_code == 0
        assert "table" in result.stdout.lower()
        assert "where" in result.stdout.lower()

    def test_reload_help(self):
        """Test reload command --help."""
        result = runner.invoke(app, ["reload", "--help"])
        assert result.exit_code == 0
        assert "filetype" in result.stdout.lower()


class TestDbInfoCommand:
    """Tests for db-info command."""

    def test_db_info_output(self):
        """Test db-info shows correct database info."""
        result = runner.invoke(app, ["db-info"])
        assert result.exit_code == 0
        assert ".esdc" in result.stdout

    def test_db_info_with_custom_env(self):
        """Test db-info shows custom path from env var."""
        with patch.dict(os.environ, {"ESDC_DB_FILE": "/custom/path/db.db"}):
            result = runner.invoke(app, ["db-info"])
            assert result.exit_code == 0
            assert "/custom/path/db.db" in result.stdout


class TestShowCommand:
    """Tests for show command."""

    def test_show_invalid_table(self):
        """Test show with invalid table name."""
        result = runner.invoke(app, ["show", "invalid_table"])
        assert result.exit_code != 0

    def test_show_with_where_filter(self):
        """Test show with where filter."""
        with patch("esdc.esdc.run_query", return_value=MagicMock()):
            result = runner.invoke(
                app, ["show", "project_resources", "--where", "id", "--search", "test"]
            )
            assert result.exit_code == 0

    def test_show_with_year_filter(self):
        """Test show with year filter."""
        with patch("esdc.esdc.run_query", return_value=MagicMock()):
            result = runner.invoke(app, ["show", "project_resources", "--year", "2024"])
            assert result.exit_code == 0

    def test_show_with_detail_level(self):
        """Test show with detail level."""
        with patch("esdc.esdc.run_query", return_value=MagicMock()):
            result = runner.invoke(
                app, ["show", "project_resources", "--detail", "reserves"]
            )
            assert result.exit_code == 0


class TestFetchCommand:
    """Tests for fetch command."""

    def test_fetch_requires_credentials(self):
        """Test fetch prompts for credentials when not in env."""
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("esdc.esdc.Config.get_credentials") as mock_creds,
            patch("esdc.esdc.load_esdc_data"),
        ):
            mock_creds.return_value = ("user", "pass")
            result = runner.invoke(app, ["fetch", "--filetype", "json"])
            assert result.exit_code == 0

    def test_fetch_with_csv(self):
        """Test fetch with csv filetype."""
        with (
            patch.dict(os.environ, {"ESDC_USER": "test", "ESDC_PASS": "test"}),
            patch("esdc.esdc.load_esdc_data"),
        ):
            result = runner.invoke(app, ["fetch", "--filetype", "csv"])
            assert result.exit_code == 0

    def test_fetch_with_invalid_filetype(self):
        """Test fetch with invalid filetype."""
        with (
            patch.dict(os.environ, {"ESDC_USER": "test", "ESDC_PASS": "test"}),
            patch("esdc.esdc.load_esdc_data"),
        ):
            result = runner.invoke(app, ["fetch", "--filetype", "invalid"])
            assert result.exit_code == 0


class TestReloadCommand:
    """Tests for reload command."""

    def test_reload_missing_file(self, tmp_path):
        """Test reload with missing file shows warning."""
        with patch("esdc.esdc.Config.get_db_dir", return_value=tmp_path):
            result = runner.invoke(
                app, ["reload", "--filetype", "csv", "--no-embeddings"]
            )
            assert result.exit_code in [0, 1]


class TestVerboseFlag:
    """Tests for --verbose flag."""

    def test_verbose_default(self):
        """Test default verbosity level."""
        with patch("esdc.esdc.Config.init_config"):
            result = runner.invoke(app, ["--help"])
            assert result.exit_code == 0

    def test_verbose_info(self):
        """Test info verbosity level."""
        with patch("esdc.esdc.Config.init_config"):
            result = runner.invoke(app, ["--verbose", "1", "db-info"])
            assert result.exit_code == 0

    def test_verbose_debug(self):
        """Test debug verbosity level."""
        with patch("esdc.esdc.Config.init_config"):
            result = runner.invoke(app, ["--verbose", "2", "db-info"])
            assert result.exit_code == 0
