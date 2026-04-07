"""Tests for ingest CLI commands."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from esdc.esdc import app

runner = CliRunner()


def test_ingest_help():
    """Test ingest command help."""
    result = runner.invoke(app, ["ingest", "--help"])
    assert result.exit_code == 0
    assert "ingest" in result.stdout.lower()


def test_ingest_file_command():
    """Test ingest file subcommand exists."""
    result = runner.invoke(app, ["ingest", "file", "--help"])
    assert result.exit_code == 0


def test_ingest_clear_command():
    """Test ingest clear subcommand exists."""
    result = runner.invoke(app, ["ingest", "clear", "--help"])
    assert result.exit_code == 0


def test_ingest_file_not_found():
    """Test ingest file with non-existent file."""
    result = runner.invoke(app, ["ingest", "file", "/nonexistent/file.md"])
    assert result.exit_code == 1
    assert "File not found" in result.output


def test_ingest_analyze_mode():
    """Test ingest with --analyze flag integration test."""
    test_file = Path("/tmp/esdc_test_docs/test_mom.md")
    if not test_file.exists():
        # Skip if test file not created
        return

    # This is an integration test that requires a real LLM
    # The basic command structure is tested in test_ingest_file_command
    # Actual pipeline execution is tested in tests/knowledge_graph/test_pipeline.py
