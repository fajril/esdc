"""`esdc corpus learn` orchestrator: eager knowledge reconstruction.

Phases:
1. Deterministic linking (registries, letter numbers, metadata) — no LLM.
2. Guideline-driven LLM extraction per new/changed document.
3. Registry-backed resolution of extracted mentions into edges/claims.
4. Dossier synthesis per POD, cached by source hash.

Incremental: per-doc learn_state hash = sha256(file_hash + guideline hash),
so both document changes and guideline edits retrigger exactly the right
work. Everything is precomputed here; chat serving never calls an LLM
for knowledge.
"""

from __future__ import annotations

import hashlib
import logging
import sqlite3
from collections.abc import Callable
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path

import duckdb

from esdc.console import console
from esdc.knowledge.dossier import ensure_dossier_table, generate_pod_dossier
from esdc.knowledge.extractor import extract_knowledge
from esdc.knowledge.guideline import load_guideline
from esdc.knowledge.linker import run_deterministic_linking
from esdc.knowledge.resolver import resolve_extraction
from esdc.knowledge.store import KnowledgeStore

logger = logging.getLogger(__name__)


@dataclass
class LearnReport:
    docs_total: int = 0
    docs_processed: int = 0
    docs_skipped: int = 0
    docs_failed: int = 0
    edges_written: int = 0
    pod_document_added: int = 0
    claims_written: int = 0
    unresolved_mentions: int = 0
    dossiers_built: int = 0
    dossiers_skipped: int = 0
    proposals_pending: int = 0
    dry_run: bool = False


def _doc_source_hash(file_hash: str, guideline_hash: str) -> str:
    return hashlib.sha256(f"{file_hash}:{guideline_hash}".encode()).hexdigest()


@contextmanager
def _status(progress: bool, msg: str):
    """Dim spinner around a phase; no-op when progress is disabled."""
    if progress:
        with console.status(msg):
            yield
    else:
        yield


def _open_default_llm(
    *,
    max_output_tokens: int | None = None,
    timeout_seconds: float | None = None,
) -> tuple[Callable[[str], str], str, str]:
    from esdc.configs import Config
    from esdc.providers import create_llm_from_config

    provider_config = Config.get_provider_config()
    if not provider_config:
        raise ValueError("No provider configured. Run 'esdc configs' first.")
    llm = create_llm_from_config(
        provider_config,
        max_output_tokens=max_output_tokens,
        timeout_seconds=timeout_seconds,
    )
    provider = str(
        provider_config.get("name") or provider_config.get("provider_type") or ""
    )
    model = str(provider_config.get("model") or "")
    return (lambda p: str(llm.invoke(p).content)), provider, model


def run_learn(
    *,
    force: bool = False,
    dry_run: bool = False,
    limit: int | None = None,
    sqlite_conn: sqlite3.Connection | None = None,
    duck_conn: duckdb.DuckDBPyConnection | None = None,
    llm_caller: Callable[[str], str] | None = None,
    provider: str = "",
    model: str = "",
    progress: bool = True,
) -> LearnReport:
    close_sqlite = close_duck = False
    if sqlite_conn is None:
        from esdc.pod_registry.store import get_sqlite_connection

        sqlite_conn = get_sqlite_connection()
        close_sqlite = True
    if duck_conn is None:
        from esdc.configs import Config
        from esdc.dbmanager import get_duckdb_connection

        duck_conn = get_duckdb_connection(Config.get_db_file(), read_only=False)
        close_duck = True

    report = LearnReport(dry_run=dry_run)
    try:
        guideline = load_guideline()
        store = KnowledgeStore(sqlite_conn)
        store.ensure_tables()

        docs = sqlite_conn.execute(
            "SELECT doc_id, file_hash, doc_type, doc_date, subject, markdown "
            "FROM documents ORDER BY doc_date, doc_id"
        ).fetchall()
        report.docs_total = len(docs)

        pending: list[sqlite3.Row] = []
        for doc in docs:
            src_hash = _doc_source_hash(doc["file_hash"], guideline.content_hash)
            if force or store.needs_learn(doc["doc_id"], src_hash):
                pending.append(doc)
            else:
                report.docs_skipped += 1
        if limit is not None:
            pending = pending[:limit]

        if dry_run:
            report.docs_processed = len(pending)
            return report

        # Phase 1: deterministic
        with _status(progress, "linking deterministic edges"):
            link_report = run_deterministic_linking(sqlite_conn, duck_conn, store)
        report.edges_written += link_report.edges_written
        report.pod_document_added = link_report.pod_document_added

        if llm_caller is None:
            from esdc.configs import Config

            corpus_cfg = Config.get_corpus_config()
            extraction_caller, provider, model = _open_default_llm(
                max_output_tokens=(
                    int(corpus_cfg.get("extract_max_tokens") or 0) or None
                ),
                timeout_seconds=(
                    float(corpus_cfg.get("extract_timeout_seconds") or 0) or None
                ),
            )
            dossier_caller, _, _ = _open_default_llm()
        else:
            extraction_caller = llm_caller
            dossier_caller = llm_caller

        # Phases 2+3: extraction + resolution
        from esdc.corpus.pipeline import _progress_with_status

        handle = None
        with ExitStack() as stack:
            if progress and pending:
                handle = stack.enter_context(
                    _progress_with_status("learn", len(pending), "docs")
                )

            for doc in pending:
                doc_id = doc["doc_id"]
                if handle is not None:
                    handle.file(doc_id)
                src_hash = _doc_source_hash(doc["file_hash"], guideline.content_hash)
                try:
                    meta = {
                        "doc_id": doc_id,
                        "doc_type": doc["doc_type"],
                        "doc_date": doc["doc_date"],
                        "subject": doc["subject"],
                    }
                    extraction = extract_knowledge(
                        doc["markdown"] or "", meta, guideline, extraction_caller
                    )
                    resolution = resolve_extraction(
                        doc_id, extraction, sqlite_conn, duck_conn
                    )
                    # replace this doc's previous LLM knowledge, keep deterministic
                    # edges (they are re-upserted by phase 1 on every run)
                    store.delete_doc_edges(doc_id)
                    if resolution.edges:
                        report.edges_written += store.upsert_edges(resolution.edges)
                    report.claims_written += store.replace_claims(
                        doc_id, resolution.claims
                    )
                    report.unresolved_mentions += len(resolution.unresolved)
                    for unknown in extraction.unknown_types:
                        store.bump_proposal(unknown["kind"], unknown["name"], doc_id)
                    store.mark_learned(doc_id, src_hash)
                    report.docs_processed += 1
                except Exception as e:  # noqa: BLE001 - one bad doc must not stop learn
                    logger.error("[Learn] doc_failed | doc_id=%s error=%s", doc_id, e)
                    report.docs_failed += 1
                if handle is not None:
                    handle.advance()

        # Deleting doc edges above also removed deterministic doc edges for
        # processed docs; restore them.
        with _status(progress, "restoring deterministic edges"):
            link_report = run_deterministic_linking(sqlite_conn, duck_conn, store)

        # Phase 4: dossiers for every POD with at least one document link
        ensure_dossier_table(duck_conn)
        pod_ids = [
            r[0]
            for r in sqlite_conn.execute(
                "SELECT DISTINCT dst_id FROM kg_edge "
                "WHERE rel = 'ABOUT_POD' ORDER BY dst_id"
            ).fetchall()
        ]
        handle = None
        with ExitStack() as stack:
            if progress and pod_ids:
                handle = stack.enter_context(
                    _progress_with_status("dossiers", len(pod_ids), "pods")
                )
            for pod_id in pod_ids:
                if handle is not None:
                    handle.file(pod_id)
                status = generate_pod_dossier(
                    pod_id,
                    sqlite_conn,
                    duck_conn,
                    store,
                    guideline.content_hash,
                    dossier_caller,
                    provider=provider,
                    model=model,
                    force=force,
                )
                if status == "built":
                    report.dossiers_built += 1
                else:
                    report.dossiers_skipped += 1
                if handle is not None:
                    handle.advance()

        report.proposals_pending = len(store.pending_proposals())

        # insert_document no longer writes the mirror row (Task 7), so the
        # mirror is only correct once the batch ends with a refresh. There
        # is no CorpusStore in scope here -- run_learn works directly
        # against the sqlite_conn/duck_conn it was given (which may be
        # test-injected, on non-default paths) -- so refresh the mirror
        # against those same connections rather than opening a second,
        # independently-pathed CorpusStore.
        from esdc.corpus.mirror import refresh_all

        sqlite_path_str = sqlite_conn.execute("PRAGMA database_list").fetchone()[2]
        if not sqlite_path_str:
            # In-memory sqlite_conn (tests inject this): PRAGMA
            # database_list's file column is '' for :memory: databases, so
            # Path('') would resolve to '.' and ATTACH the wrong database,
            # failing every time. There is no on-disk truth to refresh the
            # mirror from in that case, so skip rather than attempt-and-warn.
            logger.debug("[Learn] sqlite_conn is in-memory, skipping mirror refresh")
        else:
            sqlite_path = Path(sqlite_path_str)
            # Best-effort: DuckDB is single-writer, so this can lose the
            # write lock to a concurrent corpus command. Everything above is
            # already committed to the SQLite truth, so a refresh failure
            # here means the mirror is stale, not that learn failed -- same
            # non-fatal contract as the portal's _refresh_mirror_after_save.
            with _status(progress, "refreshing mirror"):
                try:
                    refresh_all(duck_conn, sqlite_path)
                except Exception as e:
                    logger.warning(
                        "[Learn] mirror_refresh_failed | error=%s -- run "
                        "`esdc corpus sync` to converge",
                        e,
                    )

        with _status(progress, "checkpointing"):
            duck_conn.execute("CHECKPOINT")

        try:
            from esdc.chat.tools import invalidate_tool_cache, reset_sql_cache

            reset_sql_cache()
            invalidate_tool_cache()
        except ImportError:
            pass
        return report
    finally:
        if close_sqlite:
            sqlite_conn.close()
        if close_duck:
            duck_conn.close()
