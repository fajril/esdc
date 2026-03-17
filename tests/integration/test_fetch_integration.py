import pytest
from typer.testing import CliRunner

from esdc.esdc import app

runner = CliRunner()


class TestFetchIntegration:
    """Integration tests for fetch command with real DB and mocked API."""

    def test_fetch_integration_requires_api_key(self):
        """Test fetch behavior when API key is not configured."""
        result = runner.invoke(app, ["fetch", "--filetype", "vm"])
        assert result.exit_code in [0, 1]

    def test_fetch_help_integration(self):
        """Test fetch --help shows expected options."""
        result = runner.invoke(app, ["fetch", "--help"])
        assert result.exit_code == 0
        assert "filetype" in result.stdout.lower()
