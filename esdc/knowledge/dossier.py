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
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

import duckdb

from esdc.knowledge.store import Claim, KnowledgeStore

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
    pod_row = sqlite_conn.execute(
        """
        SELECT m.pod_id, m.pod_name, m.pod_letter_num, m.approval_date,
               m.rev_num, t.pod_type
        FROM m_pod m JOIN r_pod_type t ON t.code = m.pod_type_code
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
        row = duck_conn.execute(
            """
            SELECT project_id, project_name, report_year, project_remarks
            FROM project_resources WHERE project_id = ?
            ORDER BY report_year DESC LIMIT 1
            """,
            [project_id],
        ).fetchone()
        if row:
            projects.append(
                {
                    "project_id": str(row[0]),
                    "project_name": row[1],
                    "report_year": row[2],
                    "project_remarks": row[3],
                }
            )

    return {
        "pod": pod,
        "revisions": revisions,
        "documents": documents,
        "claims": claims,
        "projects": projects,
    }


def pod_source_hash(context: dict[str, Any], guideline_hash: str) -> str:
    payload = {
        "doc_hashes": sorted(d["file_hash"] for d in context["documents"]),
        "claims": sorted(
            json.dumps(
                {
                    "doc_id": c.doc_id,
                    "type": c.claim_type,
                    "predicate": c.predicate,
                    "value": c.value,
                    "subject_id": c.subject_id,
                },
                sort_keys=True,
                ensure_ascii=False,
            )
            for c in context["claims"]
        ),
        "remarks": [p.get("project_remarks") for p in context["projects"]],
        "revisions": context["revisions"],
        "guideline": guideline_hash,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


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
    ensure_dossier_table(duck_conn)
    context = gather_pod_context(pod_id, sqlite_conn, duck_conn, store)
    source_hash = pod_source_hash(context, guideline_hash)
    existing = duck_conn.execute(
        "SELECT source_hash, provider, model FROM knowledge_dossiers "
        "WHERE entity_type = 'pod' AND entity_id = ?",
        [pod_id],
    ).fetchone()
    if existing and existing[0] == source_hash and not force:
        return "skipped"

    # A forced rebuild triggered without explicit provider/model (e.g. a
    # scheduled refresh) should not blow away known provenance metadata.
    if existing:
        provider = provider or existing[1]
        model = model or existing[2]

    text = llm_caller(_build_dossier_prompt(context))
    duck_conn.execute(
        "DELETE FROM knowledge_dossiers WHERE entity_type = 'pod' AND entity_id = ?",
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
    logger.info("[Learn] dossier_built | pod=%s", pod_id)
    return "built"


def get_dossier(
    duck_conn: duckdb.DuckDBPyConnection, entity_type: str, entity_id: str
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
    return dict(zip(keys, row, strict=True))
