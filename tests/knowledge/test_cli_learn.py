from __future__ import annotations

from unittest.mock import patch

from typer.testing import CliRunner

from esdc.esdc import app
from esdc.knowledge.learn import LearnReport

runner = CliRunner()


def test_corpus_learn_invokes_run_learn_and_prints_report():
    report = LearnReport(
        docs_total=10,
        docs_processed=4,
        docs_skipped=6,
        edges_written=42,
        claims_written=12,
        dossiers_built=3,
        dossiers_skipped=5,
        proposals_pending=1,
    )
    with patch("esdc.knowledge.learn.run_learn", return_value=report) as mock:
        result = runner.invoke(app, ["corpus", "learn", "--limit", "4"])
    assert result.exit_code == 0
    mock.assert_called_once_with(force=False, dry_run=False, limit=4)
    assert "4" in result.output  # processed
    assert "42" in result.output  # edges
    assert "proposal" in result.output.lower()


def test_corpus_learn_dry_run_flag():
    report = LearnReport(docs_total=3, docs_processed=3, dry_run=True)
    with patch("esdc.knowledge.learn.run_learn", return_value=report) as mock:
        result = runner.invoke(app, ["corpus", "learn", "--dry-run"])
    assert result.exit_code == 0
    mock.assert_called_once_with(force=False, dry_run=True, limit=None)
    assert "dry" in result.output.lower()


def test_corpus_learn_reports_missing_provider():
    with patch(
        "esdc.knowledge.learn.run_learn",
        side_effect=ValueError("No provider configured. Run 'esdc configs'."),
    ):
        result = runner.invoke(app, ["corpus", "learn"])
    assert result.exit_code == 1
    assert "No provider configured" in result.output


def test_corpus_learn_init_guideline_flag(tmp_path):
    with patch(
        "esdc.knowledge.bootstrap.init_guideline",
        return_value=tmp_path / "guideline.yaml",
    ) as mock:
        result = runner.invoke(app, ["corpus", "learn", "--init-guideline"])
    assert result.exit_code == 0
    mock.assert_called_once_with(force=False)
    assert "Review" in result.output
