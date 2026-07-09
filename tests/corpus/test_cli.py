"""CLI-level UX tests for `esdc corpus` — validation and error paths only.

Pipeline behavior is covered in test_pipeline.py; these tests pin the
thin-CLI contract: bad flags and pipeline ValueErrors exit 1 with a
clean "Error:" line instead of a traceback.
"""

from typer.testing import CliRunner

import esdc.esdc as esdc_cli
from esdc.esdc import app

runner = CliRunner()


def test_commit_invalid_level_exits_1(tmp_path):
    result = runner.invoke(app, ["corpus", "commit", str(tmp_path), "--level", "galaxy"])
    assert result.exit_code == 1
    assert "Error: --level must be one of wk, field, project." in result.output


def test_commit_invalid_doc_type_exits_1(tmp_path):
    result = runner.invoke(
        app, ["corpus", "commit", str(tmp_path), "--doc-type", "invoice"]
    )
    assert result.exit_code == 1
    assert "Error: --doc-type must be one of" in result.output


def test_clear_without_yes_exits_1_with_counts(monkeypatch):
    class FakeStore:
        def counts(self):
            return {"documents": 0, "chunks": 0}

        def close(self):
            pass

    monkeypatch.setattr(esdc_cli, "_open_corpus_store", lambda: FakeStore())

    result = runner.invoke(app, ["corpus", "clear"])
    assert result.exit_code == 1
    assert "This deletes 0 documents and 0 chunks. Re-run with --yes." in result.output


def test_commit_model_mismatch_prints_clean_error(tmp_path, monkeypatch):
    import esdc.corpus.pipeline as pipeline

    def raise_mismatch(*args, **kwargs):
        raise ValueError(
            "[Corpus] embedding model changed. Run `esdc corpus reembed`."
        )

    # commit imports run_commit lazily from the pipeline module.
    monkeypatch.setattr(pipeline, "run_commit", raise_mismatch)

    result = runner.invoke(app, ["corpus", "commit", str(tmp_path)])
    assert result.exit_code == 1
    assert "Error: [Corpus] embedding model changed" in result.output
    assert "Traceback" not in result.output
