"""Integration tests for show command."""

import pytest
from typer.testing import CliRunner

from esdc.esdc import app

runner = CliRunner()


class TestShowCommand:
    """Tests for show command with real database."""

    def test_show_empty_database(self, isolated_config):
        """Show with no data should show warning message."""
        result = runner.invoke(app, ["show", "project_resources"])

        assert result.exit_code == 0

    def test_show_with_data(self, seeded_database):
        """Show with seeded data should display formatted table."""
        result = runner.invoke(app, ["show", "project_resources"])

        assert result.exit_code == 0
        assert "PRJ-001" in result.stdout or "Test Project Alpha" in result.stdout

    def test_show_with_where_filter(self, seeded_database):
        """Show with --where and --search should filter results."""
        result = runner.invoke(
            app,
            [
                "show",
                "project_resources",
                "--where",
                "project_name",
                "--search",
                "Alpha",
            ],
        )

        assert result.exit_code == 0
        assert "Alpha" in result.stdout
        assert "Beta" not in result.stdout

    def test_show_with_year_filter(self, seeded_database):
        """Show with --year should filter by report year."""
        result = runner.invoke(app, ["show", "project_resources", "--year", "2024"])

        assert result.exit_code == 0
        assert "2024" in result.stdout
        assert "2023" not in result.stdout

    def test_show_with_columns(self, seeded_database):
        """Show with --columns should select specific fields."""
        result = runner.invoke(
            app,
            ["show", "project_resources", "--columns", "project_name", "--output", "4"],
        )

        assert result.exit_code == 0
