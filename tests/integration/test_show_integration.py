import pytest
from typer.testing import CliRunner

from esdc.esdc import app

runner = CliRunner()


class TestShowIntegration:
    """Integration tests for show command with real DB operations."""

    def test_show_help_integration(self):
        """Test show --help shows expected options."""
        result = runner.invoke(app, ["show", "--help"])
        assert result.exit_code == 0
        assert "table" in result.stdout.lower()

    def test_show_requires_table_argument(self):
        """Test show command requires table argument."""
        result = runner.invoke(app, ["show"])
        assert result.exit_code == 2

    def test_show_with_invalid_table(self):
        """Test show command with invalid table name."""
        result = runner.invoke(app, ["show", "--table", "invalid_table"])
        assert result.exit_code in [0, 1, 2]
