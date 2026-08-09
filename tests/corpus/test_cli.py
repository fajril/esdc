"""CLI-level UX tests for `esdc corpus` — validation and error paths only.

Pipeline behavior is covered in test_pipeline.py; these tests pin the
thin-CLI contract: bad flags and pipeline ValueErrors exit 1 with a
clean "Error:" line instead of a traceback.
"""

import pytest
from typer.testing import CliRunner

import esdc.esdc as esdc_cli
from esdc.esdc import app

runner = CliRunner()


def test_commit_rejects_override_flags(tmp_path):
    result = runner.invoke(
        app, ["corpus", "commit", str(tmp_path), "--wk-name", "Rokan"]
    )
    assert result.exit_code == 2
    assert "no such option" in result.output.lower()


def test_extract_invalid_level_exits_1(tmp_path):
    result = runner.invoke(
        app, ["corpus", "extract", str(tmp_path), "--level", "galaxy"]
    )
    assert result.exit_code == 1
    assert (
        "Error: --level must be one of wk, field, project, regulation." in result.output
    )


def test_extract_invalid_doc_type_exits_1(tmp_path):
    result = runner.invoke(
        app, ["corpus", "extract", str(tmp_path), "--doc-type", "invoice"]
    )
    assert result.exit_code == 1
    assert "Error: --doc-type must be one of" in result.output


def test_extract_invalid_topic_exits_1(tmp_path):
    result = runner.invoke(
        app, ["corpus", "extract", str(tmp_path), "--topic", "bogus"]
    )
    assert result.exit_code == 1
    assert "Error: --topic must be one of" in result.output


def test_extract_passes_topic_to_pipeline(tmp_path, monkeypatch):
    import esdc.corpus.pipeline as pipeline

    captured = {}

    def fake_run_extract(paths, **kwargs):
        captured.update(kwargs)
        return pipeline.CorpusReport()

    monkeypatch.setattr(pipeline, "run_extract", fake_run_extract)

    result = runner.invoke(app, ["corpus", "extract", str(tmp_path), "--topic", "wpnb"])
    assert result.exit_code == 0
    assert captured["topic"] == "wpnb"


def test_extract_level_regulation_with_entity_flag_exits_1(tmp_path):
    result = runner.invoke(
        app,
        [
            "corpus",
            "extract",
            str(tmp_path),
            "--level",
            "regulation",
            "--wk-name",
            "Rokan",
        ],
    )
    assert result.exit_code == 1
    assert "regulation documents cannot have wk/field/project entities" in result.output


def test_extract_doc_type_implies_regulation_with_entity_flag_exits_1(tmp_path):
    result = runner.invoke(
        app,
        [
            "corpus",
            "extract",
            str(tmp_path),
            "--doc-type",
            "uu",
            "--field-name",
            "Duri",
        ],
    )
    assert result.exit_code == 1
    assert "regulation documents cannot have wk/field/project entities" in result.output


def test_extract_explicit_level_conflicts_with_doc_type_rule_exits_1(tmp_path):
    result = runner.invoke(
        app,
        ["corpus", "extract", str(tmp_path), "--level", "wk", "--doc-type", "uu"],
    )
    assert result.exit_code == 1
    assert "conflicts" in result.output
    assert "doc_type 'uu' rule" in result.output


def test_extract_explicit_level_conflicts_with_topic_rule_exits_1(tmp_path):
    result = runner.invoke(
        app,
        ["corpus", "extract", str(tmp_path), "--level", "wk", "--topic", "pod"],
    )
    assert result.exit_code == 1
    assert "conflicts" in result.output
    assert "doc_topic 'pod' rule" in result.output


def test_extract_level_matches_doc_type_rule_no_conflict(tmp_path, monkeypatch):
    """Explicit --level equal to the implied rule's level is not a conflict."""
    import esdc.corpus.pipeline as pipeline

    monkeypatch.setattr(
        pipeline, "run_extract", lambda paths, **kwargs: pipeline.CorpusReport()
    )

    result = runner.invoke(
        app,
        [
            "corpus",
            "extract",
            str(tmp_path),
            "--level",
            "regulation",
            "--doc-type",
            "uu",
        ],
    )
    assert result.exit_code == 0


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
        raise ValueError("[Corpus] embedding model changed. Run `esdc corpus reembed`.")

    # commit imports run_commit lazily from the pipeline module.
    monkeypatch.setattr(pipeline, "run_commit", raise_mismatch)

    result = runner.invoke(app, ["corpus", "commit", str(tmp_path)])
    assert result.exit_code == 1
    assert "Error: [Corpus] embedding model changed" in result.output
    assert "Traceback" not in result.output


def test_meta_invalid_level_exits_1(tmp_path):
    result = runner.invoke(app, ["corpus", "meta", str(tmp_path), "--level", "bogus"])
    assert result.exit_code == 1
    assert (
        "Error: --level must be one of wk, field, project, regulation." in result.output
    )


def test_meta_invalid_doc_type_exits_1(tmp_path):
    result = runner.invoke(
        app, ["corpus", "meta", str(tmp_path), "--doc-type", "bogus"]
    )
    assert result.exit_code == 1
    assert "Error: --doc-type must be one of" in result.output


def test_meta_invalid_topic_exits_1(tmp_path):
    result = runner.invoke(app, ["corpus", "meta", str(tmp_path), "--topic", "bogus"])
    assert result.exit_code == 1
    assert "Error: --topic must be one of" in result.output


def test_meta_topic_passes_to_pipeline(tmp_path, monkeypatch):
    import esdc.corpus.pipeline as pipeline

    captured = {}

    def fake_run_meta(paths, **kwargs):
        captured.update(kwargs)
        return pipeline.CorpusReport()

    monkeypatch.setattr(pipeline, "run_meta", fake_run_meta)

    result = runner.invoke(app, ["corpus", "meta", str(tmp_path), "--topic", "wpnb"])
    assert result.exit_code == 0
    assert captured["topic"] == "wpnb"


def test_meta_level_regulation_with_entity_flag_exits_1(tmp_path):
    result = runner.invoke(
        app,
        [
            "corpus",
            "meta",
            str(tmp_path),
            "--level",
            "regulation",
            "--project-name",
            "POD Duri",
        ],
    )
    assert result.exit_code == 1
    assert "regulation documents cannot have wk/field/project entities" in result.output


def test_meta_explicit_level_conflicts_with_topic_rule_exits_1(tmp_path):
    result = runner.invoke(
        app, ["corpus", "meta", str(tmp_path), "--level", "field", "--topic", "afe"]
    )
    assert result.exit_code == 1
    assert "conflicts" in result.output
    assert "doc_topic 'afe' rule" in result.output


def test_meta_passes_overrides_to_pipeline(tmp_path, monkeypatch):
    import esdc.corpus.pipeline as pipeline

    captured = {}

    def fake_run_meta(paths, **kwargs):
        captured["paths"] = paths
        captured.update(kwargs)
        return pipeline.CorpusReport()

    monkeypatch.setattr(pipeline, "run_meta", fake_run_meta)

    result = runner.invoke(
        app,
        ["corpus", "meta", str(tmp_path), "--wk-name", "Rokan", "--reviewed"],
    )
    assert result.exit_code == 0
    assert captured["wk_name"] == "Rokan"
    assert captured["reviewed"] is True


def test_meta_no_flags_shows_table(tmp_path, monkeypatch):
    import esdc.corpus.pipeline as pipeline

    def fake_run_meta_show(paths):
        return [
            {
                "file": "a.corpus.md",
                "doc_type": "contract",
                "doc_topic": ["wpnb"],
                "doc_date": "2024-01-01",
                "doc_level": "wk",
                "wk_name": ["Rokan"],
                "field_name": None,
                "project_name": None,
                "reviewed": False,
            },
            {"file": "bad.corpus.md", "error": "missing YAML frontmatter"},
        ]

    called = {"run_meta": False}

    def fake_run_meta(paths, **kwargs):
        called["run_meta"] = True
        return pipeline.CorpusReport()

    monkeypatch.setattr(pipeline, "run_meta_show", fake_run_meta_show)
    monkeypatch.setattr(pipeline, "run_meta", fake_run_meta)

    result = runner.invoke(app, ["corpus", "meta", str(tmp_path)])
    assert result.exit_code == 0
    assert called["run_meta"] is False
    assert "a.corpus.md" in result.output
    assert "topic" in result.output
    # An unreadable sidecar's error lands in its own trailing `note`
    # column — never under doc_type.
    assert "note" in result.output
    assert "missing YAML frontmatter" in result.output
    bad_line = next(
        line for line in result.output.splitlines() if "bad.corpus.md" in line
    )
    bad_cells = [c.strip() for c in bad_line.split("|")]
    assert bad_cells[2] == ""  # doc_type cell stays clean for error rows
    good_line = next(
        line for line in result.output.splitlines() if "a.corpus.md" in line
    )
    assert "contract" in good_line
    assert "wpnb" in good_line
    assert "missing YAML frontmatter" not in good_line


def test_meta_regenerate_alone_is_valid_invocation(tmp_path, monkeypatch):
    """--regenerate alone (no other flags) must call run_meta, not show-mode."""
    import esdc.corpus.pipeline as pipeline

    captured = {}
    called = {"run_meta_show": False}

    def fake_run_meta(paths, **kwargs):
        captured.update(kwargs)
        return pipeline.CorpusReport()

    def fake_run_meta_show(paths):
        called["run_meta_show"] = True
        return []

    monkeypatch.setattr(pipeline, "run_meta", fake_run_meta)
    monkeypatch.setattr(pipeline, "run_meta_show", fake_run_meta_show)

    result = runner.invoke(app, ["corpus", "meta", str(tmp_path), "--regenerate"])
    assert result.exit_code == 0
    assert called["run_meta_show"] is False
    assert captured["regenerate"] is True


def test_meta_regenerate_value_error_exits_1(tmp_path, monkeypatch):
    import esdc.corpus.pipeline as pipeline

    def raise_no_model(*args, **kwargs):
        raise ValueError(
            "--regenerate requires a reachable metadata_model "
            "(set corpus.metadata_model in config)"
        )

    monkeypatch.setattr(pipeline, "run_meta", raise_no_model)

    result = runner.invoke(app, ["corpus", "meta", str(tmp_path), "--regenerate"])
    assert result.exit_code == 1
    assert "Error: --regenerate requires a reachable metadata_model" in result.output
    assert "Traceback" not in result.output


def test_meta_unknown_entity_prints_clean_error(tmp_path, monkeypatch):
    import esdc.corpus.pipeline as pipeline

    def raise_not_found(*args, **kwargs):
        raise ValueError("--wk-name 'Bogus' not found in database and no close matches")

    monkeypatch.setattr(pipeline, "run_meta", raise_not_found)

    result = runner.invoke(app, ["corpus", "meta", str(tmp_path), "--wk-name", "Bogus"])
    assert result.exit_code == 1
    assert "not found in database" in result.output
    assert "Traceback" not in result.output


def test_extract_unknown_entity_prints_clean_error(tmp_path, monkeypatch):
    import esdc.corpus.pipeline as pipeline

    def raise_not_found(*args, **kwargs):
        raise ValueError("--wk-name 'Bogus' not found in database and no close matches")

    monkeypatch.setattr(pipeline, "run_extract", raise_not_found)

    result = runner.invoke(
        app, ["corpus", "extract", str(tmp_path), "--wk-name", "Bogus"]
    )
    assert result.exit_code == 1
    assert "not found in database" in result.output
    assert "Traceback" not in result.output


def test_export_no_paths_no_all_exits_1():
    result = runner.invoke(app, ["corpus", "export"])
    assert result.exit_code == 1
    assert "Nothing to export: pass sidecar path(s) or --all." in result.output


def test_export_passes_all_to_pipeline(monkeypatch):
    import esdc.corpus.pipeline as pipeline

    captured = {}

    def fake_run_export(paths, **kwargs):
        captured["paths"] = paths
        captured.update(kwargs)
        return pipeline.CorpusReport()

    monkeypatch.setattr(pipeline, "run_export", fake_run_export)

    result = runner.invoke(app, ["corpus", "export", "--all"])
    assert result.exit_code == 0
    assert captured["paths"] == []
    assert captured["all_docs"] is True


def test_export_passes_paths_to_pipeline(tmp_path, monkeypatch):
    import esdc.corpus.pipeline as pipeline

    captured = {}

    def fake_run_export(paths, **kwargs):
        captured["paths"] = paths
        captured.update(kwargs)
        return pipeline.CorpusReport()

    monkeypatch.setattr(pipeline, "run_export", fake_run_export)

    sc = tmp_path / "doc.corpus.md"
    result = runner.invoke(app, ["corpus", "export", str(sc)])
    assert result.exit_code == 0
    assert captured["paths"] == [sc]
    assert captured["all_docs"] is False


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


def test_rename_invalid_doc_type_exits_1(tmp_path):
    result = runner.invoke(
        app, ["corpus", "rename", str(tmp_path), "--doc-type", "invoice"]
    )
    assert result.exit_code == 1
    assert "Error: --doc-type must be one of" in result.output


def test_rename_dry_run_is_default(tmp_path, monkeypatch):
    import esdc.corpus.rename as rename_mod
    from esdc.corpus.pipeline import CorpusReport
    from esdc.corpus.rename import RenamePlan

    captured = {}

    def fake_run_rename(paths, doc_type=None, apply=False):
        captured["apply"] = apply
        plan = RenamePlan(
            src=tmp_path / "scan.pdf",
            new_path=tmp_path / "letter - 2024.01.15 - Judul.pdf",
            doc_type="letter",
            doc_date="2024.01.15",
            title="Judul",
            source="sidecar",
            sidecar_src=None,
            sidecar_new=None,
            note="",
        )
        return CorpusReport(), [plan]

    monkeypatch.setattr(rename_mod, "run_rename", fake_run_rename)

    result = runner.invoke(app, ["corpus", "rename", str(tmp_path)])
    assert result.exit_code == 0
    assert captured["apply"] is False
    assert "letter - 2024.01.15 - Judul.pdf" in result.output
    assert "re-run with --yes" in result.output.lower()


def test_rename_yes_applies(tmp_path, monkeypatch):
    import esdc.corpus.rename as rename_mod
    from esdc.corpus.pipeline import CorpusReport
    from esdc.corpus.rename import RenamePlan

    captured = {}

    def fake_run_rename(paths, doc_type=None, apply=False):
        captured["apply"] = apply
        report = CorpusReport()
        report.processed.append("scan.pdf")
        plan = RenamePlan(
            src=tmp_path / "scan.pdf",
            new_path=tmp_path / "letter - 2024.01.15 - Judul.pdf",
            doc_type="letter",
            doc_date="2024.01.15",
            title="Judul",
            source="sidecar",
            sidecar_src=None,
            sidecar_new=None,
            note="",
        )
        return report, [plan]

    monkeypatch.setattr(rename_mod, "run_rename", fake_run_rename)

    result = runner.invoke(app, ["corpus", "rename", str(tmp_path), "--yes"])
    assert result.exit_code == 0
    assert captured["apply"] is True


# --- corpus eval: --init/--refresh + staleness gate -------------------------


class _FakeEvalStore:
    """Fake CorpusStore satisfying both the CLI and run_eval call paths.

    Combines the query-gen FakeStore (test_query_gen.py) and the
    evaluate FakeStore (test_evaluate.py) — corpus_eval and run_eval both
    instantiate `esdc.corpus.store.CorpusStore` internally, so a single
    fake must satisfy both call paths.
    """

    def __init__(self, docs):
        self._docs = {d["doc_id"]: d for d in docs}

    def list_documents(self):
        return [
            {"doc_id": d["doc_id"], "doc_type": d["doc_type"], "subject": d["subject"]}
            for d in self._docs.values()
        ]

    def fingerprint_rows(self):
        return [(d["doc_id"], d["file_hash"]) for d in self._docs.values()]

    def sample_content(self, doc_id, chunk_seed=None):
        return self._docs.get(doc_id)

    def document_bodies(self):
        return [(d["doc_id"], "", d.get("chunk_text", "")) for d in self._docs.values()]

    def search(self, query, limit=10, filters=None, rerank=None):
        doc_id = next(iter(self._docs))
        return {
            "status": "success",
            "results": [{"doc_id": doc_id, "file_name": f"{doc_id}.pdf"}],
            "count": 1,
        }

    def close(self):
        pass


def _eval_docs():
    return [
        {
            "doc_id": f"letter-{i}",
            "doc_type": "letter",
            "subject": f"subject {i}",
            "file_hash": f"h{i}",
            "chunk_text": f"body {i}",
        }
        for i in range(3)
    ]


@pytest.fixture
def fake_store(monkeypatch):
    store = _FakeEvalStore(_eval_docs())
    monkeypatch.setattr("esdc.corpus.store.CorpusStore", lambda: store)
    return store


@pytest.fixture
def fake_llm(monkeypatch):
    class _Resp:
        content = "generated query?"

    class _FakeLLM:
        def invoke(self, prompt):
            return _Resp()

    monkeypatch.setattr("esdc.providers.create_llm_from_config", lambda cfg: _FakeLLM())
    return _FakeLLM()


def _patch_queries_path(monkeypatch, path):
    from esdc.configs import Config

    monkeypatch.setattr(
        Config, "get_corpus_queries_path", classmethod(lambda cls: path)
    )


def _patch_provider_config(monkeypatch):
    from esdc.configs import Config

    monkeypatch.setattr(
        Config,
        "get_provider_config",
        classmethod(lambda cls: {"provider_type": "fake", "model": "x"}),
    )


def test_eval_missing_file_errors(monkeypatch, tmp_path):
    _patch_queries_path(monkeypatch, tmp_path / "corpus_queries.jsonl")

    result = runner.invoke(app, ["corpus", "eval"])
    assert result.exit_code == 1
    assert "--init" in result.output


def test_eval_init_generates_and_scores(monkeypatch, tmp_path, fake_store, fake_llm):
    """`--init` with no value must auto-size (the primary default workflow)."""
    path = tmp_path / "corpus_queries.jsonl"
    _patch_queries_path(monkeypatch, path)
    _patch_provider_config(monkeypatch)

    result = runner.invoke(app, ["corpus", "eval", "--init"])
    assert result.exit_code == 0, result.output
    assert "Generated" in result.output
    assert path.exists()


def test_eval_init_with_explicit_samples(monkeypatch, tmp_path, fake_store, fake_llm):
    path = tmp_path / "corpus_queries.jsonl"
    _patch_queries_path(monkeypatch, path)
    _patch_provider_config(monkeypatch)

    result = runner.invoke(app, ["corpus", "eval", "--init", "--samples", "2"])
    assert result.exit_code == 0, result.output
    assert "Generated" in result.output
    assert path.exists()


def test_eval_reports_per_class(monkeypatch, tmp_path):
    import esdc.corpus.evaluate as evaluate_mod
    from esdc.corpus.evaluate import ClassReport, EvalReport

    path = tmp_path / "queries.jsonl"
    path.write_text(
        '{"query": "q", "expected": ["d"], "class": "lookup"}\n', encoding="utf-8"
    )
    report = EvalReport(
        n_queries=1,
        pass_at={1: 1.0},
        mean_latency_ms=12.0,
        by_class={
            "lookup": ClassReport(n_queries=1, pass_at={1: 1.0}),
            "cross_reference": ClassReport(
                n_queries=2, pass_at={1: 0.5}, recall_at={1: 0.25}
            ),
        },
    )
    monkeypatch.setattr(evaluate_mod, "run_eval", lambda *a, **kw: report)

    result = runner.invoke(app, ["corpus", "eval", str(path), "--k", "1"])
    assert result.exit_code == 0, result.output
    assert "lookup" in result.output
    assert "cross_reference" in result.output
    assert "Recall@1" in result.output


def test_eval_reports_negative_abstention_hint(monkeypatch, tmp_path):
    import esdc.corpus.evaluate as evaluate_mod
    from esdc.corpus.evaluate import ClassReport, EvalReport

    path = tmp_path / "queries.jsonl"
    path.write_text(
        '{"query": "q", "expected": [], "class": "negative"}\n', encoding="utf-8"
    )
    report = EvalReport(
        n_queries=1,
        by_class={"negative": ClassReport(n_queries=1, abstention=None)},
    )
    monkeypatch.setattr(evaluate_mod, "run_eval", lambda *a, **kw: report)

    result = runner.invoke(app, ["corpus", "eval", str(path), "--k", "1"])
    assert result.exit_code == 0, result.output
    assert "--rerank" in result.output


def test_eval_refresh_preserves_non_lookup_rows(
    monkeypatch, tmp_path, fake_store, fake_llm
):
    """Reconcile keys on expected[0]; it must never see a multi-doc or negative row."""
    import json

    from esdc.corpus.sampling import corpus_fingerprint

    path = tmp_path / "corpus_queries.jsonl"
    _patch_queries_path(monkeypatch, path)
    _patch_provider_config(monkeypatch)
    meta = {
        "fingerprint": corpus_fingerprint(fake_store.fingerprint_rows()),
        "margin": 0.05,
        "n": 3,
        "ks": [1],
        "embedding_model": "qwen3",
        "generated_at": "2026-08-05T00:00:00",
    }
    path.write_text(
        json.dumps({"_meta": meta})
        + '\n{"query": "l", "expected": ["letter-0"], "class": "lookup",'
        ' "file_hash": "h0"}\n'
        '{"query": "x", "expected": ["letter-0", "letter-1"],'
        ' "class": "cross_reference"}\n'
        '{"query": "n", "expected": [], "class": "negative"}\n',
        encoding="utf-8",
    )

    result = runner.invoke(app, ["corpus", "eval", "--refresh"])
    assert result.exit_code == 0, result.output
    rows = [json.loads(x) for x in path.read_text().splitlines()[1:]]
    classes = [r.get("class") for r in rows]
    assert "negative" in classes
    assert "cross_reference" in classes


def test_eval_refresh_missing_file_errors(monkeypatch, tmp_path):
    """--refresh on a missing file must error cleanly.

    Must not raise a FileNotFoundError traceback.
    """
    _patch_queries_path(monkeypatch, tmp_path / "corpus_queries.jsonl")

    result = runner.invoke(app, ["corpus", "eval", "--refresh"])
    assert result.exit_code == 1
    assert "--init" in result.output
    assert result.exception is None or isinstance(result.exception, SystemExit)


def test_eval_stale_fingerprint_blocks(monkeypatch, tmp_path, fake_store):
    from esdc.corpus.query_gen import QueryMeta, write_query_file

    path = tmp_path / "corpus_queries.jsonl"
    _patch_queries_path(monkeypatch, path)
    meta = QueryMeta(
        fingerprint="deadbeef",
        margin=0.05,
        n=1,
        ks=[1, 5, 10],
        embedding_model="qwen3",
        generated_at="2026-07-25T00:00:00",
    )
    write_query_file(path, [{"query": "q", "expected": ["letter-0"]}], meta)

    result = runner.invoke(app, ["corpus", "eval"])
    assert result.exit_code == 1
    assert "Corpus changed" in result.output


def test_eval_stale_reports_changed_doc(monkeypatch, tmp_path, fake_store):
    """A doc re-ingested with edited content (same doc_id, new file_hash).

    Must be reported as "~1 changed", not folded into +added/-removed.
    """
    from esdc.corpus.query_gen import QueryMeta, write_query_file

    path = tmp_path / "corpus_queries.jsonl"
    _patch_queries_path(monkeypatch, path)

    # Same doc_ids as the live store (no additions/removals), but letter-0's
    # recorded file_hash ("OLD-HASH") no longer matches the store's current
    # hash ("h0") — simulating a content edit under the same doc_id.
    rows = [
        {"query": "q0", "expected": ["letter-0"], "file_hash": "OLD-HASH"},
        {"query": "q1", "expected": ["letter-1"], "file_hash": "h1"},
        {"query": "q2", "expected": ["letter-2"], "file_hash": "h2"},
    ]
    meta = QueryMeta(
        fingerprint="stale-fp",  # deliberately not matching the live fingerprint
        margin=0.05,
        n=3,
        ks=[1, 5, 10],
        embedding_model="qwen3",
        generated_at="2026-07-25T00:00:00",
    )
    write_query_file(path, rows, meta)

    result = runner.invoke(app, ["corpus", "eval"])
    assert result.exit_code == 1
    assert "Corpus changed" in result.output
    assert "+0 new" in result.output
    assert "-0 removed" in result.output
    assert "~1 changed" in result.output


def test_eval_refresh_prints_delta(monkeypatch, tmp_path, fake_store):
    """--refresh reconciles against a mutated live corpus and prints delta.

    Reads the query file, reconciles against the (mutated) live corpus,
    writes the updated file, and prints the +added/-removed delta.
    """
    from esdc.corpus.query_gen import QueryMeta, read_query_file, write_query_file
    from esdc.corpus.sampling import corpus_fingerprint

    path = tmp_path / "corpus_queries.jsonl"
    _patch_queries_path(monkeypatch, path)
    _patch_provider_config(monkeypatch)

    # Query file matching the store's current (3-doc) fingerprint.
    old_rows = [{"query": f"q{i}", "expected": [f"letter-{i}"]} for i in range(3)]
    meta = QueryMeta(
        fingerprint=corpus_fingerprint(fake_store.fingerprint_rows()),
        margin=0.05,
        n=3,
        ks=[1, 5, 10],
        embedding_model="qwen3",
        generated_at="2026-07-25T00:00:00",
    )
    write_query_file(path, old_rows, meta)

    # Mutate the corpus: drop letter-2, add letter-3 -> fingerprint changes.
    del fake_store._docs["letter-2"]
    fake_store._docs["letter-3"] = {
        "doc_id": "letter-3",
        "doc_type": "letter",
        "subject": "subject 3",
        "file_hash": "h3",
        "chunk_text": "body 3",
    }

    class _Resp:
        content = "refreshed query?"

    class _FakeLLM:
        def invoke(self, prompt):
            return _Resp()

    monkeypatch.setattr("esdc.providers.create_llm_from_config", lambda cfg: _FakeLLM())

    result = runner.invoke(app, ["corpus", "eval", "--refresh"])
    assert result.exit_code == 0, result.output
    assert "Refreshed:" in result.output
    assert "+1 new" in result.output
    assert "-1 removed" in result.output

    new_rows, new_meta = read_query_file(path)
    assert new_meta is not None
    new_ids = {r["expected"][0] for r in new_rows}
    assert "letter-2" not in new_ids
    assert "letter-3" in new_ids
    assert new_meta.fingerprint == corpus_fingerprint(fake_store.fingerprint_rows())


# --- corpus sync: rebuild the DuckDB mirror from the SQLite truth ----------


def test_corpus_sync_reports_refreshed_row_count(monkeypatch):
    """`esdc corpus sync` rebuilds the mirror and reports what it copied."""
    from esdc.corpus.mirror import MirrorReport

    class FakeStore:
        def __init__(self):
            self.ensure_tables_called = False

        def ensure_tables(self):
            self.ensure_tables_called = True

        def refresh_mirror(self):
            return MirrorReport(
                documents=3,
                orphan_chunks=1,
                registry={"m_pod": 2, "pod_document": 0},
                views=["v_document", "v_pod"],
            )

        def close(self):
            pass

    monkeypatch.setattr("esdc.corpus.store.CorpusStore", lambda: FakeStore())

    result = runner.invoke(app, ["corpus", "sync"])

    assert result.exit_code == 0, result.output
    assert "documents mirrored: 3" in result.output.lower()
    assert "orphan chunks removed: 1" in result.output.lower()
    assert "m_pod: 2" in result.output
    assert "views: v_document, v_pod" in result.output


def test_eval_refresh_reports_changed_count(monkeypatch, tmp_path, fake_store):
    """--refresh reports a "~C changed" segment for docs whose content changed.

    Same doc_id, new file_hash, between the old and new query file.
    """
    from esdc.corpus.query_gen import QueryMeta, write_query_file

    path = tmp_path / "corpus_queries.jsonl"
    _patch_queries_path(monkeypatch, path)
    _patch_provider_config(monkeypatch)

    # Query file recording a stale file_hash for letter-0; the live store's
    # sample_content/fingerprint_rows for letter-0 report "h0".
    old_rows = [
        {"query": "q0", "expected": ["letter-0"], "file_hash": "OLD-HASH"},
        {"query": "q1", "expected": ["letter-1"], "file_hash": "h1"},
        {"query": "q2", "expected": ["letter-2"], "file_hash": "h2"},
    ]
    meta = QueryMeta(
        fingerprint="stale-fp",
        margin=0.05,
        n=3,
        ks=[1, 5, 10],
        embedding_model="qwen3",
        generated_at="2026-07-25T00:00:00",
    )
    write_query_file(path, old_rows, meta)

    class _Resp:
        content = "refreshed query?"

    class _FakeLLM:
        def invoke(self, prompt):
            return _Resp()

    monkeypatch.setattr("esdc.providers.create_llm_from_config", lambda cfg: _FakeLLM())

    result = runner.invoke(app, ["corpus", "eval", "--refresh"])
    assert result.exit_code == 0, result.output
    assert "Refreshed:" in result.output
    assert "+0 new" in result.output
    assert "-0 removed" in result.output
    assert "~1 changed" in result.output


def test_reembed_stale_only_processes_flagged_documents(monkeypatch):
    """--stale re-embeds the detector's list, not the whole corpus."""
    from esdc.corpus.pipeline import CorpusReport

    calls: dict[str, list[str]] = {}

    class _Store:
        def __init__(self, *args, **kwargs):
            pass

        def ensure_tables(self, *args, **kwargs):
            return None

        def stale_embed_docs(self):
            return ["d1", "d2"]

        def close(self):
            return None

    def _fake_reembed_documents(doc_ids, store=None, progress=False):
        calls["doc_ids"] = list(doc_ids)
        report = CorpusReport()
        report.processed = list(doc_ids)
        report.embedding_model = "fake-model"
        return report

    def _fail_full_reembed(*args, **kwargs):
        raise AssertionError("--stale must not run the whole-corpus re-embed")

    monkeypatch.setattr("esdc.corpus.store.CorpusStore", _Store)
    monkeypatch.setattr(
        "esdc.corpus.pipeline.run_reembed_documents", _fake_reembed_documents
    )
    monkeypatch.setattr("esdc.corpus.pipeline.run_reembed", _fail_full_reembed)

    result = runner.invoke(app, ["corpus", "reembed", "--stale"])

    assert result.exit_code == 0, result.output
    assert calls["doc_ids"] == ["d1", "d2"]
    assert "2 stale document" in result.output


def test_reembed_stale_reports_a_clean_corpus_without_reembedding(monkeypatch):
    """Nothing stale -> say so and stop; never start the embedder."""

    class _Store:
        def __init__(self, *args, **kwargs):
            pass

        def ensure_tables(self, *args, **kwargs):
            return None

        def stale_embed_docs(self):
            return []

        def close(self):
            return None

    def _fail(*args, **kwargs):
        raise AssertionError("nothing is stale; no re-embed should run")

    monkeypatch.setattr("esdc.corpus.store.CorpusStore", _Store)
    monkeypatch.setattr("esdc.corpus.pipeline.run_reembed_documents", _fail)
    monkeypatch.setattr("esdc.corpus.pipeline.run_reembed", _fail)

    result = runner.invoke(app, ["corpus", "reembed", "--stale"])

    assert result.exit_code == 0, result.output
    assert "No stale documents" in result.output


def test_reembed_stale_rejects_an_embed_backend_override(monkeypatch):
    """Mixing models within one corpus is refused, not silently ignored."""

    def _fail(*args, **kwargs):
        raise AssertionError("must exit before touching the store or embedder")

    monkeypatch.setattr("esdc.corpus.store.CorpusStore", _fail)
    monkeypatch.setattr("esdc.corpus.pipeline.run_reembed_documents", _fail)
    monkeypatch.setattr("esdc.corpus.pipeline.run_reembed", _fail)

    result = runner.invoke(
        app, ["corpus", "reembed", "--stale", "--embed-backend", "ollama"]
    )

    assert result.exit_code == 1
    assert "--embed-backend applies only to a full re-embed" in result.output
