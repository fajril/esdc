"""Tests for the learning workflow."""

from __future__ import annotations

import json
import logging
import sqlite3

from esdc.knowledge.learn import run_learn
from esdc.knowledge.store import KnowledgeStore

EXTRACTION = {
    "entities": [{"type": "pod", "name": "POD I Duri"}],
    "claims": [
        {
            "type": "issue",
            "subject": "POD I Duri",
            "subject_type": "pod",
            "predicate": "delay_cause",
            "value": "rig availability",
            "evidence": "q",
        }
    ],
    "unknown_types": [{"kind": "claim_type", "name": "hse_incident", "why": "seen"}],
}


def _llm(prompt: str) -> str:
    if "case file" in prompt:
        return "## Approval & Revisions\nstub dossier"
    return json.dumps(EXTRACTION)


def test_full_run_processes_all_docs_and_builds_dossiers(sqlite_conn, duck_conn):
    report = run_learn(
        sqlite_conn=sqlite_conn,
        duck_conn=duck_conn,
        llm_caller=_llm,
        progress=False,
    )
    assert report.docs_total == 3
    assert report.docs_processed == 3
    assert report.docs_failed == 0
    assert report.edges_written > 0
    assert report.claims_written >= 3
    assert report.proposals_pending == 1
    # both linked pods get dossiers (DOC-A -> rev1 pod, DOC-B -> base pod)
    assert report.dossiers_built >= 2
    n = duck_conn.execute("SELECT COUNT(*) FROM knowledge_dossiers").fetchone()[0]
    assert n == report.dossiers_built


def test_second_run_is_incremental(sqlite_conn, duck_conn):
    run_learn(
        sqlite_conn=sqlite_conn, duck_conn=duck_conn, llm_caller=_llm, progress=False
    )
    calls = []

    def counting_llm(prompt):
        calls.append(prompt)
        return _llm(prompt)

    report = run_learn(
        sqlite_conn=sqlite_conn,
        duck_conn=duck_conn,
        llm_caller=counting_llm,
        progress=False,
    )
    assert report.docs_processed == 0
    assert report.docs_skipped == 3
    assert report.dossiers_built == 0
    assert calls == []  # no LLM calls at all on a no-change rerun


def test_dry_run_writes_nothing(sqlite_conn, duck_conn):
    report = run_learn(
        sqlite_conn=sqlite_conn,
        duck_conn=duck_conn,
        llm_caller=_llm,
        dry_run=True,
        progress=False,
    )
    assert report.dry_run is True
    assert report.docs_processed == 3  # "would process"
    store = KnowledgeStore(sqlite_conn)
    store.ensure_tables()
    assert store.learned_count() == 0
    assert sqlite_conn.execute("SELECT COUNT(*) FROM kg_edge").fetchone()[0] == 0


def test_failed_doc_is_counted_and_not_marked_learned(sqlite_conn, duck_conn):
    def flaky_llm(prompt):
        if "Pedoman Insentif" in prompt:
            return "not json at all"
        return _llm(prompt)

    report = run_learn(
        sqlite_conn=sqlite_conn,
        duck_conn=duck_conn,
        llm_caller=flaky_llm,
        progress=False,
    )
    assert report.docs_failed == 1
    assert report.docs_processed == 2
    # failed doc retries on next run
    report2 = run_learn(
        sqlite_conn=sqlite_conn, duck_conn=duck_conn, llm_caller=_llm, progress=False
    )
    assert report2.docs_processed == 1


def test_mirror_refresh_failure_is_warning_not_fatal(
    sqlite_conn, duck_conn, monkeypatch
):
    """A refresh_all failure during learn degrades to a warning, not a fatal error.

    A refresh_all failure (e.g. lost DuckDB write lock) must not abort
    learn -- everything above is already committed to the SQLite truth by
    then, so it's a stale mirror, not a failed learn run (same contract as
    the portal's _refresh_mirror_after_save).
    """
    import esdc.corpus.mirror as mirror

    def boom(conn, path):
        raise RuntimeError("lock held by another process")

    monkeypatch.setattr(mirror, "refresh_all", boom)

    report = run_learn(
        sqlite_conn=sqlite_conn, duck_conn=duck_conn, llm_caller=_llm, progress=False
    )
    assert report.docs_processed == 3
    assert report.docs_failed == 0


def test_in_memory_sqlite_skips_mirror_refresh_without_warning(
    sqlite_conn, duck_conn, caplog
):
    """An in-memory sqlite_conn must not trigger a mirror_refresh_failed warning.

    As some callers/tests inject, PRAGMA database_list's file column is ''
    for an in-memory connection -- Path('') resolves to '.', which
    ATTACHes the wrong database and used to fail the refresh on every
    learn run, logging a mirror_refresh_failed warning every time. The
    refresh should be skipped outright for an in-memory connection instead
    of attempted and warned about.
    """
    mem_conn = sqlite3.connect(":memory:")
    mem_conn.row_factory = sqlite3.Row
    mem_conn.execute("PRAGMA foreign_keys = ON")
    sqlite_conn.backup(mem_conn)

    try:
        with caplog.at_level(logging.WARNING):
            report = run_learn(
                sqlite_conn=mem_conn,
                duck_conn=duck_conn,
                llm_caller=_llm,
                progress=False,
            )
        assert report.docs_processed == 3
        assert report.docs_failed == 0
        assert not any(
            "mirror_refresh_failed" in record.message for record in caplog.records
        )
    finally:
        mem_conn.close()


def test_limit_caps_llm_docs(sqlite_conn, duck_conn):
    report = run_learn(
        sqlite_conn=sqlite_conn,
        duck_conn=duck_conn,
        llm_caller=_llm,
        limit=1,
        progress=False,
    )
    assert report.docs_processed == 1
    assert report.docs_skipped == 0  # remaining docs are pending, not skipped


def test_extraction_and_dossier_use_separate_bounded_callers(
    sqlite_conn, duck_conn, monkeypatch
):
    """Extraction calls carry bounds; dossier calls stay unbounded.

    run_learn must open the default LLM twice: once bounded for
    extract_knowledge, once unbounded for generate_pod_dossier, and use
    the matching caller in each phase.
    """
    import esdc.knowledge.learn as learn_mod
    from esdc.configs import Config

    monkeypatch.setattr(
        Config,
        "get_corpus_config",
        classmethod(
            lambda cls: {
                "extract_max_tokens": 8192,
                "extract_timeout_seconds": 300,
            }
        ),
    )

    opened: list[dict[str, object]] = []
    extraction_prompts: list[str] = []
    dossier_prompts: list[str] = []

    def fake_open_default_llm(
        *, max_output_tokens: int | None = None, timeout_seconds: float | None = None
    ):
        opened.append(
            {"max_output_tokens": max_output_tokens, "timeout_seconds": timeout_seconds}
        )
        if max_output_tokens is not None or timeout_seconds is not None:

            def extraction_caller(prompt: str) -> str:
                extraction_prompts.append(prompt)
                return json.dumps(EXTRACTION)

            return extraction_caller, "fake", "fake-model"

        def dossier_caller(prompt: str) -> str:
            dossier_prompts.append(prompt)
            return "## Approval & Revisions\nstub dossier"

        return dossier_caller, "fake", "fake-model"

    monkeypatch.setattr(learn_mod, "_open_default_llm", fake_open_default_llm)
    run_learn(
        sqlite_conn=sqlite_conn,
        duck_conn=duck_conn,
        progress=False,
    )

    assert opened == [
        {"max_output_tokens": 8192, "timeout_seconds": 300.0},
        {"max_output_tokens": None, "timeout_seconds": None},
    ]
    assert extraction_prompts
    assert dossier_prompts
    assert not set(extraction_prompts) & set(dossier_prompts)


def test_init_guideline_calls_open_default_llm_without_bounds(sqlite_conn, monkeypatch):
    """Guideline bootstrap must keep the unconstrained default caller."""
    import esdc.knowledge.learn as learn_mod
    from esdc.knowledge.bootstrap import init_guideline

    opened: list[dict[str, object]] = []
    payload = {
        "claim_types": {
            "economics": "NPV, IRR, capex. Predicate examples: npv_musd, irr_pct."
        },
        "doc_type_hints": {"letter": "Focus on approval terms."},
    }

    def fake_open_default_llm(**kwargs):
        opened.append(dict(kwargs))
        return lambda p: json.dumps(payload), "fake", "fake-model"

    monkeypatch.setattr(learn_mod, "_open_default_llm", fake_open_default_llm)
    init_guideline(sqlite_conn=sqlite_conn, force=True)

    assert opened == [{}]


def test_extraction_meta_carries_doc_id(sqlite_conn, duck_conn, monkeypatch):
    import esdc.knowledge.learn as learn_mod

    seen: list[dict[str, object]] = []
    real_extract = learn_mod.extract_knowledge

    def spy_extract(markdown, doc_meta, guideline, llm_caller):
        seen.append(dict(doc_meta))
        return real_extract(markdown, doc_meta, guideline, llm_caller)

    monkeypatch.setattr(learn_mod, "extract_knowledge", spy_extract)
    run_learn(
        sqlite_conn=sqlite_conn,
        duck_conn=duck_conn,
        llm_caller=_llm,
        progress=False,
    )

    assert seen
    assert all(meta.get("doc_id") for meta in seen)
