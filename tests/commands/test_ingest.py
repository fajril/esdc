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
