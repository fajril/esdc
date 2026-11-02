"""Per-POD dossier ("case file") synthesis, cached by source hash.

A dossier joins everything known about one POD: registry facts, revision
chain, linked documents, extracted claims, project data and live remarks.
Generated eagerly by `esdc corpus learn`; chat only ever reads them.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
from collections.abc import Callable, Mapping
from dataclasses import fields, is_dataclass
from datetime import date, datetime, timezone
from decimal import Decimal
from typing import Any

import duckdb

from esdc.knowledge.freshness import compute_knowledge_basis, knowledge_freshness
from esdc.knowledge.store import Claim, KnowledgeStore
from esdc.llm_text import strip_thinking_tags
from esdc.pod_registry.migrations import (
    MigrationRefusedError,
    _validate_existing_v2_registry,
)

logger = logging.getLogger(__name__)

_EXCERPT_CHARS = 4000

_DOSSIER_DDL = """
CREATE TABLE IF NOT EXISTS knowledge_dossiers (
    entity_type TEXT NOT NULL,
    entity_id   TEXT NOT NULL,
    entity_name TEXT,
    dossier_text TEXT NOT NULL,
    source_hash  TEXT NOT NULL,
    provider     TEXT,
    model        TEXT,
    generated_at TEXT,
    PRIMARY KEY (entity_type, entity_id)
)
"""


def ensure_dossier_table(duck_conn: duckdb.DuckDBPyConnection) -> None:
    duck_conn.execute(_DOSSIER_DDL)


def gather_pod_context(
    pod_id: str,
    sqlite_conn: sqlite3.Connection,
    duck_conn: duckdb.DuckDBPyConnection,
    store: KnowledgeStore,
) -> dict[str, Any]:
    """Read one POD's facts from the single current source.

    `duck_conn` is kept for caller compatibility while the instance graph is
    still on its pending conversion; the dossier cache is never a source of
    registry, project, field, working-area or remark facts.
    """
    pod_row = sqlite_conn.execute(
        """
        SELECT m.pod_id, m.pod_name, m.pod_letter_num, m.approval_date,
               m.rev_num, t.pod_type
        FROM m_pod m JOIN r_pod_type t ON t.id = m.pod_type_id
        WHERE m.pod_id = ?
        """,
        (pod_id,),
    ).fetchone()
    pod = dict(pod_row) if pod_row else {"pod_id": pod_id}

    edges = store.edges_for("pod", pod_id)
    revisions = sorted(
        {
            e.src_id if e.dst_id == pod_id else e.dst_id
            for e in edges
            if e.rel == "REVISES"
        }
    )

    doc_links = [e for e in edges if e.rel == "ABOUT_POD" and e.src_type == "document"]
    documents: list[dict[str, Any]] = []
    claims: list[Claim] = list(store.claims_for("pod", pod_id))
    seen_claim_ids = {
        (c.doc_id, c.claim_type, c.predicate, json.dumps(c.value, sort_keys=True))
        for c in claims
    }
    for link in sorted(doc_links, key=lambda e: e.src_id):
        row = sqlite_conn.execute(
            "SELECT doc_id, file_hash, doc_type, doc_date, subject, markdown "
            "FROM documents WHERE doc_id = ?",
            (link.src_id,),
        ).fetchone()
        if row is None:
            continue
        documents.append(
            {
                "doc_id": row["doc_id"],
                "file_hash": row["file_hash"],
                "doc_type": row["doc_type"],
                "doc_date": row["doc_date"],
                "subject": row["subject"],
                "excerpt": (row["markdown"] or "")[:_EXCERPT_CHARS],
                "link_method": link.method,
                "link_confidence": link.confidence,
            }
        )
        for c in store.claims_for_doc(row["doc_id"]):
            key = (
                c.doc_id,
                c.claim_type,
                c.predicate,
                json.dumps(c.value, sort_keys=True),
            )
            if key not in seen_claim_ids:
                seen_claim_ids.add(key)
                claims.append(c)

    project_ids = sorted({e.dst_id for e in edges if e.rel == "HAS_PROJECT"})
    projects: list[dict[str, Any]] = []
    for project_id in project_ids:
        # m_project is the current project authority; the latest-year raw API row
        # only supplies the report year and remarks. A master with no resource
        # facts is still reported, with NULL year and remarks.
        master = sqlite_conn.execute(
            "SELECT project_name FROM m_project WHERE project_id = ?",
            (project_id,),
        ).fetchone()
        if master is None:
            continue
        resource = sqlite_conn.execute(
            "SELECT report_year, project_remarks FROM api_project_resources "
            "WHERE project_id = ? "
            'ORDER BY CAST(report_year AS INTEGER) DESC, project_name, '
            '"project_remarks" LIMIT 1',
            (project_id,),
        ).fetchone()
        projects.append(
            {
                "project_id": str(project_id),
                "project_name": master["project_name"],
                "report_year": None if resource is None else resource["report_year"],
                "project_remarks": (
                    None if resource is None else resource["project_remarks"]
                ),
            }
        )

    return {
        "pod": pod,
        "revisions": revisions,
        "documents": documents,
        "claims": claims,
        "projects": projects,
    }


def _canonical_dossier_value(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return {
            "__type__": f"{type(value).__module__}.{type(value).__qualname__}",
            "fields": {
                item.name: _canonical_dossier_value(getattr(value, item.name))
                for item in fields(value)
            },
        }
    if isinstance(value, datetime):
        return {"__type__": "datetime", "value": value.isoformat()}
    if isinstance(value, date):
        return {"__type__": "date", "value": value.isoformat()}
    if isinstance(value, Decimal):
        return {"__type__": "Decimal", "value": str(value)}
    if isinstance(value, float):
        return {"__type__": "float", "value": value.hex()}
    if isinstance(value, bytes):
        return {"__type__": "bytes", "value": value.hex()}
    if isinstance(value, Mapping):
        items = [
            [_canonical_dossier_value(key), _canonical_dossier_value(item)]
            for key, item in value.items()
        ]
        items.sort(
            key=lambda pair: json.dumps(
                pair, sort_keys=True, ensure_ascii=False, separators=(",", ":")
            )
        )
        return {"__type__": "mapping", "items": items}
    if isinstance(value, list):
        return [_canonical_dossier_value(item) for item in value]
    if isinstance(value, tuple):
        return {
            "__type__": "tuple",
            "items": [_canonical_dossier_value(item) for item in value],
        }
    if isinstance(value, (set, frozenset)):
        items = [_canonical_dossier_value(item) for item in value]
        items.sort(
            key=lambda item: json.dumps(
                item, sort_keys=True, ensure_ascii=False, separators=(",", ":")
            )
        )
        return {"__type__": type(value).__name__, "items": items}
    if value is None or isinstance(value, (bool, int, str)):
        return value
    raise TypeError(f"unsupported dossier context value: {type(value).__name__}")


def _ordered_dossier_context(context: dict[str, Any]) -> dict[str, Any]:
    ordered = dict(context)
    for key in ("documents", "claims", "projects"):
        ordered[key] = sorted(
            context[key],
            key=lambda item: json.dumps(
                _canonical_dossier_value(item),
                sort_keys=True,
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        )
    ordered["revisions"] = sorted(context["revisions"])
    return ordered


def pod_source_hash(context: dict[str, Any], guideline_hash: str) -> str:
    """Hash every typed input used to build a POD dossier prompt."""
    payload = _canonical_dossier_value(
        {
            "context": _ordered_dossier_context(context),
            "guideline_hash": guideline_hash,
        }
    )
    canonical_json = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


def _build_dossier_prompt(context: dict[str, Any]) -> str:
    pod = context["pod"]
    doc_lines = (
        "\n\n".join(
            f"[{d['doc_id']}] {d['doc_type']} {d['doc_date'] or ''} — "
            f"{d['subject'] or ''}\n{d['excerpt']}"
            for d in context["documents"]
        )
        or "(no linked documents)"
    )
    claim_lines = (
        "\n".join(
            f"- [{c.doc_id}] {c.claim_type}.{c.predicate} = "
            f"{json.dumps(c.value, ensure_ascii=False)}"
            f"{f' (quote: {c.evidence})' if c.evidence else ''}"
            for c in context["claims"]
        )
        or "(no extracted claims)"
    )
    project_lines = (
        "\n".join(
            f"- {p['project_name']} ({p['project_id']}, report {p['report_year']}): "
            f"{p['project_remarks'] or 'no remarks'}"
            for p in context["projects"]
        )
        or "(no linked projects)"
    )
    return f"""You are building the definitive case file for one POD in the
Indonesian upstream oil & gas portfolio. Audience: an SKK Migas analyst.
Write in English; keep Indonesian terms where official (POD, WK, KKKS).

POD registry facts:
- pod_id: {pod.get("pod_id")}
- name: {pod.get("pod_name")}
- type: {pod.get("pod_type")}  revision: {pod.get("rev_num")}
- approval letter: {pod.get("pod_letter_num")}  date: {pod.get("approval_date")}
- revision chain (related pod_ids): {", ".join(context["revisions"]) or "none"}

Linked documents (excerpts):
{doc_lines}

Extracted claims:
{claim_lines}

Current project data:
{project_lines}

Write a markdown dossier with exactly these sections:
## Approval & Revisions
## Economics & Reserves
## Commitments & Schedule
## Meeting History & Decisions
## Current Situation & Issues
## Open Questions

Rules:
- Every fact must cite its source inline as [doc_id] or [project_remarks].
- Only state what the sources support; put gaps under Open Questions.
- Be complete but dense; no filler prose."""


def _replace_cached_dossier(
    duck_conn: duckdb.DuckDBPyConnection,
    pod_id: str,
    context: dict[str, Any],
    text: str,
    source_hash: str,
    provider: str,
    model: str,
) -> None:
    transaction_started = False
    try:
        duck_conn.execute("BEGIN TRANSACTION")
        transaction_started = True
        duck_conn.execute(
            "DELETE FROM knowledge_dossiers "
            "WHERE entity_type = 'pod' AND entity_id = ?",
            [pod_id],
        )
        duck_conn.execute(
            "INSERT INTO knowledge_dossiers VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [
                "pod",
                pod_id,
                context["pod"].get("pod_name"),
                text,
                source_hash,
                provider,
                model,
                datetime.now(timezone.utc).isoformat(),
            ],
        )
        duck_conn.execute("COMMIT")
        transaction_started = False
    except Exception:
        if transaction_started:
            try:
                duck_conn.execute("ROLLBACK")
            except Exception:
                logger.exception("failed to roll back DuckDB dossier replacement")
        raise


def generate_pod_dossier(
    pod_id: str,
    sqlite_conn: sqlite3.Connection,
    duck_conn: duckdb.DuckDBPyConnection,
    store: KnowledgeStore,
    guideline_hash: str,
    llm_caller: Callable[[str], str],
    provider: str = "",
    model: str = "",
    force: bool = False,
) -> str:
    if not _is_current_pod_source(sqlite_conn):
        raise MigrationRefusedError(
            "dossier generation requires the current POD source schema"
        )
    ensure_dossier_table(duck_conn)
    if sqlite_conn.in_transaction:
        raise RuntimeError("dossier generation requires an idle SQLite connection")
    sqlite_conn.execute("BEGIN")
    try:
        expected_basis = compute_knowledge_basis(sqlite_conn)
        context = _ordered_dossier_context(
            gather_pod_context(pod_id, sqlite_conn, duck_conn, store)
        )
        sqlite_conn.commit()
    except Exception:
        sqlite_conn.rollback()
        raise
    source_hash = pod_source_hash(context, guideline_hash)
    existing = duck_conn.execute(
        "SELECT source_hash, provider, model FROM knowledge_dossiers "
        "WHERE entity_type = 'pod' AND entity_id = ?",
        [pod_id],
    ).fetchone()
    if existing and existing[0] == source_hash and not force:
        return "skipped"

    # Preserve provenance when a forced rebuild omits provider/model settings.
    if existing:
        provider = provider or existing[1]
        model = model or existing[2]

    text = strip_thinking_tags(llm_caller(_build_dossier_prompt(context))).strip()
    sqlite_conn.execute("BEGIN IMMEDIATE")
    try:
        if compute_knowledge_basis(sqlite_conn) != expected_basis:
            sqlite_conn.rollback()
            return "stale"
        current_context = _ordered_dossier_context(
            gather_pod_context(pod_id, sqlite_conn, duck_conn, store)
        )
        if pod_source_hash(current_context, guideline_hash) != source_hash:
            sqlite_conn.rollback()
            return "stale"

        _replace_cached_dossier(
            duck_conn, pod_id, context, text, source_hash, provider, model
        )
        sqlite_conn.commit()
    except Exception:
        if sqlite_conn.in_transaction:
            sqlite_conn.rollback()
        raise
    logger.info("[Learn] dossier_built | pod=%s", pod_id)
    return "built"


def _is_current_pod_source(sqlite_conn: sqlite3.Connection) -> bool:
    """Read-only check that the supplied connection is the canonical current source.

    Uses the shared registry structure validator; it does not infer a second
    schema shape and never writes.

    Current metadata that cannot be read (a renamed or missing state column, for
    example) surfaces as a SQLite error from the validator. At this read boundary
    that means the source is not the canonical one, so the caller treats the entry
    as unavailable rather than serving last-good cache. Only source-validation
    errors are mapped here; errors from cache or provider work still propagate.
    """
    try:
        _validate_existing_v2_registry(sqlite_conn)
    except (MigrationRefusedError, sqlite3.Error):
        return False
    return True


def get_dossier(
    duck_conn: duckdb.DuckDBPyConnection,
    entity_type: str,
    entity_id: str,
    *,
    sqlite_conn: sqlite3.Connection | None = None,
) -> dict[str, Any] | None:
    row = duck_conn.execute(
        "SELECT entity_type, entity_id, entity_name, dossier_text, "
        "source_hash, provider, model, generated_at "
        "FROM knowledge_dossiers WHERE entity_type = ? AND entity_id = ?",
        [entity_type, entity_id],
    ).fetchone()
    if row is None:
        return None

    keys = (
        "entity_type",
        "entity_id",
        "entity_name",
        "dossier_text",
        "source_hash",
        "provider",
        "model",
        "generated_at",
    )

    # Only POD dossiers are POD-derived data. A non-POD cache row keeps the generic
    # behaviour: it is served as stored, without a source gate.
    if entity_type != "pod":
        return dict(zip(keys, row, strict=True))

    # A POD dossier is only served when the caller supplies the current POD source
    # it was derived from. Without that context there is no fallback, and nothing
    # is read or written on the caller's behalf.
    if sqlite_conn is None or not _is_current_pod_source(sqlite_conn):
        return None

    owns_snapshot = not sqlite_conn.in_transaction
    if owns_snapshot:
        sqlite_conn.execute("BEGIN")
    try:
        if not knowledge_freshness(sqlite_conn).fresh:
            if owns_snapshot:
                sqlite_conn.rollback()
            return None
        from esdc.knowledge.guideline import load_guideline

        context = _ordered_dossier_context(
            gather_pod_context(
                entity_id,
                sqlite_conn,
                duck_conn,
                KnowledgeStore(sqlite_conn),
            )
        )
        if pod_source_hash(context, load_guideline().content_hash) != row[4]:
            if owns_snapshot:
                sqlite_conn.rollback()
            return None
        if owns_snapshot:
            sqlite_conn.commit()
    except Exception:
        if owns_snapshot and sqlite_conn.in_transaction:
            sqlite_conn.rollback()
        raise

    return dict(zip(keys, row, strict=True))
