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


def test_extract_invalid_level_exits_1(tmp_path):
    result = runner.invoke(
        app, ["corpus", "extract", str(tmp_path), "--level", "galaxy"]
    )
    assert result.exit_code == 1
    assert "Error: --level must be one of wk, field, project." in result.output


def test_extract_invalid_doc_type_exits_1(tmp_path):
    result = runner.invoke(
        app, ["corpus", "extract", str(tmp_path), "--doc-type", "invoice"]
    )
    assert result.exit_code == 1
    assert "Error: --doc-type must be one of" in result.output


def test_extract_passes_overrides_to_pipeline(tmp_path, monkeypatch):
    import esdc.corpus.pipeline as pipeline

    captured = {}

    def fake_run_extract(paths, **kwargs):
        captured["paths"] = paths
        captured.update(kwargs)
        return pipeline.CorpusReport()

    monkeypatch.setattr(pipeline, "run_extract", fake_run_extract)

    result = runner.invoke(
        app, ["corpus", "extract", str(tmp_path), "--wk-name", "Rokan", "--level", "wk"]
    )
    assert result.exit_code == 0
    assert captured["wk_name"] == "Rokan"
    assert captured["level"] == "wk"


def test_commit_passes_skip_review_to_pipeline(tmp_path, monkeypatch):
    import esdc.corpus.pipeline as pipeline

    captured = {}

    def fake_run_commit(paths, **kwargs):
        captured.update(kwargs)
        return pipeline.CorpusReport()

    monkeypatch.setattr(pipeline, "run_commit", fake_run_commit)
    result = runner.invoke(app, ["corpus", "commit", str(tmp_path), "--skip-review"])
    assert result.exit_code == 0
    assert captured["skip_review"] is True


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


def test_migrate_doc_types_command_exists_exits_0_passes_dry_run(tmp_path, monkeypatch):
    import esdc.corpus.pipeline as pipeline

    captured = {}

    def fake_run_migrate_doc_types(paths, **kwargs):
        captured["paths"] = paths
        captured.update(kwargs)
        return pipeline.CorpusReport()

    monkeypatch.setattr(
        pipeline, "run_migrate_doc_types", fake_run_migrate_doc_types
    )

    result = runner.invoke(
        app, ["corpus", "migrate-doc-types", str(tmp_path), "--dry-run"]
    )
    assert result.exit_code == 0
    assert captured["dry_run"] is True


def test_migrate_doc_types_no_paths_still_runs_store_migration(monkeypatch):
    import esdc.corpus.pipeline as pipeline

    captured = {}

    def fake_run_migrate_doc_types(paths, **kwargs):
        captured["paths"] = paths
        captured.update(kwargs)
        return pipeline.CorpusReport()

    monkeypatch.setattr(
        pipeline, "run_migrate_doc_types", fake_run_migrate_doc_types
    )

    result = runner.invoke(app, ["corpus", "migrate-doc-types"])
    assert result.exit_code == 0
    assert captured["paths"] == []


def test_migrate_doc_types_value_error_exits_1(tmp_path, monkeypatch):
    import esdc.corpus.pipeline as pipeline

    def raise_mismatch(*args, **kwargs):
        raise ValueError("[Corpus] embedding model changed.")

    monkeypatch.setattr(pipeline, "run_migrate_doc_types", raise_mismatch)

    result = runner.invoke(app, ["corpus", "migrate-doc-types", str(tmp_path)])
    assert result.exit_code == 1
    assert "Error: [Corpus] embedding model changed" in result.output
    assert "Traceback" not in result.output


def test_entity_display_handles_legacy_plain_string():
    from esdc.esdc import _entity_display

    # Legacy row: store's suppress-parse left it a plain string.
    assert _entity_display({"wk_name": "Rokan"}) == "Rokan"
    # Normal parsed row.
    assert _entity_display({"wk_name": ["Rokan", "Mahakam"]}) == "Rokan, Mahakam"
    # Unparsed JSON string.
    assert _entity_display({"field_name": '["Duri"]'}) == "Duri"
    # Nothing set.
    assert _entity_display({}) == ""
