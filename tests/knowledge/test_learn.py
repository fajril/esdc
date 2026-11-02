"""Tests for the learning workflow."""

from __future__ import annotations

import json
import logging
import sqlite3

import pytest

from esdc.knowledge.freshness import knowledge_freshness
from esdc.knowledge.learn import run_learn
from esdc.knowledge.store import KnowledgeStore
from esdc.pod_registry.migrations import MigrationRefusedError
from tests.knowledge.conftest import POD_DURI_REV1, PROJECT_DURI, _seed_knowledge_world

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
    # pod_document rows belong to the canonical link write path, not to learn.
    assert report.pod_document_added == 0
    # Only the current identity is matchable: DOC-A's letter number resolves to
    # POD_DURI_REV1, while DOC-B's pod_name names a superseded master and stays
    # an unresolved mention. So exactly one POD gets a dossier.
    assert report.dossiers_built == 1
    assert report.unresolved_mentions == 3
    n = duck_conn.execute("SELECT COUNT(*) FROM knowledge_dossiers").fetchone()[0]
    assert n == report.dossiers_built
    assert duck_conn.execute(
        "SELECT entity_id FROM knowledge_dossiers WHERE entity_type = 'pod'"
    ).fetchone()[0] == POD_DURI_REV1
    edge = sqlite_conn.execute(
        "SELECT dst_id, method FROM kg_edge WHERE src_type = 'document' AND rel = 'ABOUT_POD'"
    ).fetchall()
    assert [(r["dst_id"], r["method"]) for r in edge] == [
        (POD_DURI_REV1, "letter_num_exact")
    ]


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
    # A dry run never creates the disposable cache either.
    assert duck_conn.execute(
        "SELECT COUNT(*) FROM duckdb_tables WHERE table_name = 'knowledge_dossiers'"
    ).fetchone()[0] == 0


def test_learn_state_hash_is_the_recomputed_canonical_content_hash(
    sqlite_conn, duck_conn
):
    """The recorded per-doc hash is recomputed from the canonical row, not trusted."""
    import hashlib

    from esdc.knowledge.guideline import load_guideline
    from esdc.knowledge.learn import _doc_source_hash

    report = run_learn(
        sqlite_conn=sqlite_conn,
        duck_conn=duck_conn,
        llm_caller=_llm,
        progress=False,
    )
    assert report.docs_processed == 3

    guideline = load_guideline()
    doc = sqlite_conn.execute(
        "SELECT file_hash FROM documents WHERE doc_id = 'DOC-A'"
    ).fetchone()
    expected = _doc_source_hash(doc["file_hash"], guideline.content_hash)
    recorded = sqlite_conn.execute(
        "SELECT source_hash FROM learn_state WHERE doc_id = 'DOC-A'"
    ).fetchone()
    assert recorded[0] == expected
    assert expected == hashlib.sha256(
        f"hash-a:{guideline.content_hash}".encode()
    ).hexdigest()


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


def test_failed_write_rolls_back_the_document_result(sqlite_conn, duck_conn, monkeypatch):
    """A write failure inside the result transaction leaves nothing committed.

    The provider stays fake. The fault is injected at the last write of the
    transaction, so the edge and claim produced for that document must be rolled
    back with it, the documents that succeeded keep their results, and a run that
    failed is never marked current.
    """
    extraction = {
        "entities": [{"type": "pod", "name": "POD I Duri Revisi 1"}],
        "claims": [
            {
                "type": "issue",
                "subject": "POD I Duri Revisi 1",
                "subject_type": "pod",
                "predicate": "delay_cause",
                "value": "rig availability",
                "evidence": "q",
            }
        ],
        "unknown_types": [],
    }

    def fake_llm(prompt: str) -> str:
        if "case file" in prompt:
            return "stub dossier"
        return json.dumps(extraction)

    seen_inside_transaction: list[int] = []
    original_mark_learned = KnowledgeStore.mark_learned

    def failing_mark_learned(
        self, doc_id, source_hash, *, basis_sha256=None, commit: bool = True
    ) -> None:
        if doc_id == "DOC-C":
            # The edge and claim are uncommitted writes in this same transaction.
            seen_inside_transaction.append(
                self._conn.execute(
                    "SELECT COUNT(*) FROM kg_edge "
                    "WHERE src_type = 'document' AND src_id = ?",
                    (doc_id,),
                ).fetchone()[0]
            )
            raise sqlite3.OperationalError("injected write failure")
        original_mark_learned(
            self, doc_id, source_hash, basis_sha256=basis_sha256, commit=commit
        )

    monkeypatch.setattr(KnowledgeStore, "mark_learned", failing_mark_learned)

    # Make the knowledge basis stale with a committed source change that is not
    # re-marked, so the end-of-run marker is observable rather than pre-set.
    sqlite_conn.execute(
        "UPDATE m_project SET project_name = 'Renamed master' WHERE project_id = ?",
        (PROJECT_DURI,),
    )
    sqlite_conn.commit()
    assert not knowledge_freshness(sqlite_conn).fresh

    report = run_learn(
        sqlite_conn=sqlite_conn,
        duck_conn=duck_conn,
        llm_caller=fake_llm,
        progress=False,
    )

    assert seen_inside_transaction == [1]
    assert report.docs_failed == 1
    assert report.docs_processed == 2
    # Rolled back together with the failed marker: nothing for DOC-C survived.
    assert sqlite_conn.execute(
        "SELECT COUNT(*) FROM kg_edge WHERE src_type = 'document' AND src_id = 'DOC-C'"
    ).fetchone()[0] == 0
    assert sqlite_conn.execute(
        "SELECT COUNT(*) FROM kg_claim WHERE doc_id = 'DOC-C'"
    ).fetchone()[0] == 0
    assert sqlite_conn.execute(
        "SELECT COUNT(*) FROM learn_state WHERE doc_id = 'DOC-C'"
    ).fetchone()[0] == 0
    # The documents that completed kept their results.
    assert sqlite_conn.execute("SELECT COUNT(*) FROM learn_state").fetchone()[0] == 2
    # A run with a failure is never marked current.
    assert not knowledge_freshness(sqlite_conn).fresh
    assert not sqlite_conn.in_transaction


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
    by_id = {meta["doc_id"]: meta for meta in seen}
    # The date carried into extraction is the document's own date, never the
    # current identity's approval date.
    assert by_id["DOC-A"]["doc_date"] == "2022-03-01"
    assert by_id["DOC-A"]["doc_date"] != "2022-01-01"


def test_learn_refuses_a_source_that_is_not_the_current_schema(duck_conn):
    """Validation happens before tables, the cache, or any provider call."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row

    def never_called(prompt: str) -> str:
        raise AssertionError("no provider work for an incompatible source")

    try:
        with pytest.raises(MigrationRefusedError):
            run_learn(
                sqlite_conn=conn,
                duck_conn=duck_conn,
                llm_caller=never_called,
                progress=False,
            )
        assert not conn.in_transaction
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' "
            "AND name IN ('kg_edge', 'kg_claim', 'learn_state')"
        ).fetchall()
        assert tables == []
        assert duck_conn.execute(
            "SELECT COUNT(*) FROM duckdb_tables WHERE table_name = 'knowledge_dossiers'"
        ).fetchone()[0] == 0
    finally:
        conn.close()


def test_learn_requires_an_idle_sqlite_connection(sqlite_conn, monkeypatch):
    """An active caller transaction is refused before the cache is opened."""
    import esdc.dbmanager as dbmanager

    calls: list[str] = []

    def tripwire(*_args, **_kwargs):
        calls.append("cache")
        raise AssertionError("the cache connector must not run for an active transaction")

    monkeypatch.setattr(dbmanager, "get_duckdb_connection", tripwire)

    sqlite_conn.execute("BEGIN")
    sqlite_conn.execute("SELECT 1 FROM m_pod").fetchone()
    try:
        with pytest.raises(RuntimeError, match="idle"):
            run_learn(
                sqlite_conn=sqlite_conn,
                llm_caller=_llm,
                progress=False,
            )
        assert calls == []
    finally:
        sqlite_conn.rollback()


def _tracked_source(tmp_path, factory):
    """Canonical current source on disk, re-opened through a close-tracking factory."""
    from esdc.pod_registry.store import get_sqlite_connection

    path = tmp_path / "owned-source.sqlite"
    seeded = get_sqlite_connection(path)
    _seed_knowledge_world(seeded)
    seeded.close()
    conn = sqlite3.connect(path, factory=factory)
    conn.row_factory = sqlite3.Row
    return conn


def test_learn_never_reaches_the_cache_for_an_invalid_supplied_source(monkeypatch):
    """The source is the single truth: it is validated before any cache work.

    An invalid supplied SQLite must be refused without opening, initialising, or
    writing the cache, and without any provider call. The caller's connection is
    left open and idle.
    """
    import esdc.dbmanager as dbmanager

    calls: list[str] = []

    def tripwire(*_args, **_kwargs):
        calls.append("cache")
        raise AssertionError("the cache connector must not run for an invalid source")

    monkeypatch.setattr(dbmanager, "get_duckdb_connection", tripwire)

    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("CREATE TABLE m_pod (id INTEGER PRIMARY KEY, pod_id TEXT)")

    def never_called(prompt: str) -> str:
        raise AssertionError("no provider work for an incompatible source")

    try:
        with pytest.raises(MigrationRefusedError):
            run_learn(sqlite_conn=conn, llm_caller=never_called, progress=False)
        assert calls == []
        assert not conn.in_transaction
        assert conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'kg_edge'"
        ).fetchone() is None
    finally:
        conn.close()


def test_learn_closes_the_owned_source_when_cache_acquisition_fails(tmp_path, monkeypatch):
    """A valid owned source is not leaked when the later cache acquisition fails."""
    import esdc.dbmanager as dbmanager
    from esdc.pod_registry import store as pod_store

    class Tracked(sqlite3.Connection):
        closed_for_probe = False

        def close(self):
            self.closed_for_probe = True
            super().close()

    calls: list[str] = []

    def tripwire(*_args, **_kwargs):
        calls.append("cache")
        raise RuntimeError("cache acquisition fails")

    owned = _tracked_source(tmp_path, Tracked)
    monkeypatch.setattr(pod_store, "get_sqlite_writer_connection", lambda: owned)
    monkeypatch.setattr(dbmanager, "get_duckdb_connection", tripwire)

    try:
        with pytest.raises(RuntimeError, match="cache acquisition fails"):
            run_learn(llm_caller=lambda _p: "not-called", progress=False)
        # Validation passed, so the cache was reached -- and the owned source closed.
        assert calls == ["cache"]
        assert owned.closed_for_probe
    finally:
        if not owned.closed_for_probe:
            owned.close()


def test_learn_leaves_a_supplied_source_caller_owned_when_cache_acquisition_fails(
    sqlite_conn, monkeypatch
):
    """Only connections learn acquired are closed; a supplied source stays usable."""
    import esdc.dbmanager as dbmanager

    def tripwire(*_args, **_kwargs):
        raise RuntimeError("cache acquisition fails")

    monkeypatch.setattr(dbmanager, "get_duckdb_connection", tripwire)

    with pytest.raises(RuntimeError, match="cache acquisition fails"):
        run_learn(sqlite_conn=sqlite_conn, llm_caller=_llm, progress=False)

    # Still usable and idle: the caller owns it.
    assert sqlite_conn.execute("SELECT COUNT(*) FROM m_pod").fetchone()[0] == 3
    assert not sqlite_conn.in_transaction


def test_learn_closes_both_owned_connections_even_when_one_close_raises(
    tmp_path, monkeypatch, caplog
):
    """One failing close must not skip the other owned connection.

    The original failure survives; the failing close is reported as a warning
    instead of replacing it. Narrowly lifecycle-focused: no cache work succeeds
    here, so the run stops at the first cache use.
    """
    import esdc.dbmanager as dbmanager
    from esdc.pod_registry import store as pod_store

    class CloseRaises(sqlite3.Connection):
        closed_for_probe = False

        def close(self):
            self.closed_for_probe = True
            super().close()
            raise RuntimeError("source close boom")

    class FailingCache:
        closed_for_probe = False

        def execute(self, *_args, **_kwargs):
            raise RuntimeError("cache work fails")

        def close(self):
            self.closed_for_probe = True

    owned = _tracked_source(tmp_path, CloseRaises)
    cache = FailingCache()
    monkeypatch.setattr(pod_store, "get_sqlite_writer_connection", lambda: owned)
    monkeypatch.setattr(dbmanager, "get_duckdb_connection", lambda *_args, **_kwargs: cache)

    try:
        with (
            caplog.at_level(logging.WARNING),
            pytest.raises(RuntimeError, match="cache work fails"),
        ):
            run_learn(llm_caller=lambda _p: "not-called", progress=False)
        assert owned.closed_for_probe
        assert cache.closed_for_probe
        assert any("close_failed" in record.message for record in caplog.records)
    finally:
        if not owned.closed_for_probe:
            owned.close()
