"""Deterministic knowledge edges from registries and document metadata.

Phase 1 of `esdc corpus learn`. No LLM calls; safe to rerun any time.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass
from typing import Any

import duckdb

from esdc.corpus.context import _as_list
from esdc.corpus.pod_matcher import _norm
from esdc.knowledge.store import Edge, KnowledgeStore

logger = logging.getLogger(__name__)


@dataclass
class LinkReport:
    edges_written: int = 0
    pod_document_added: int = 0


def _canonical_names(
    duck_conn: duckdb.DuckDBPyConnection, column: str
) -> dict[str, str]:
    """Lowercase name -> canonical spelling, from project_resources."""
    rows = duck_conn.execute(
        f"SELECT DISTINCT {column} FROM project_resources WHERE {column} IS NOT NULL"
    ).fetchall()
    return {str(r[0]).strip().lower(): str(r[0]).strip() for r in rows if r[0]}


def _document_rows(sqlite_conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return sqlite_conn.execute(
        "SELECT doc_id, doc_number, field_name, wk_name, suggested_pod_ids "
        "FROM documents"
    ).fetchall()


def run_deterministic_linking(
    sqlite_conn: sqlite3.Connection,
    duck_conn: duckdb.DuckDBPyConnection,
    store: KnowledgeStore,
) -> LinkReport:
    report = LinkReport()
    edges: list[Edge] = []

    pod_rows = sqlite_conn.execute(
        "SELECT id, pod_id, pod_letter_num FROM m_pod"
    ).fetchall()
    letter_to_pod = {
        _norm(r["pod_letter_num"]): (r["id"], r["pod_id"])
        for r in pod_rows
        if r["pod_letter_num"]
    }
    valid_pod_ids = {r["pod_id"] for r in pod_rows}
    fields = _canonical_names(duck_conn, "field_name")
    wks = _canonical_names(duck_conn, "wk_name")

    # 1+2+3: per-document edges
    for doc in _document_rows(sqlite_conn):
        doc_id = doc["doc_id"]
        linked_pods: set[str] = set()

        hit = letter_to_pod.get(_norm(doc["doc_number"]))
        if hit:
            m_pod_pk, pod_id = hit
            edges.append(
                Edge(
                    "document",
                    doc_id,
                    "ABOUT_POD",
                    "pod",
                    pod_id,
                    confidence=1.0,
                    method="letter_num_exact",
                    evidence=f"doc_number={doc['doc_number']}",
                )
            )
            linked_pods.add(pod_id)
            cur = sqlite_conn.execute(
                "INSERT OR IGNORE INTO pod_document (pod_id, doc_id) VALUES (?, ?)",
                (m_pod_pk, doc_id),
            )
            report.pod_document_added += cur.rowcount

        raw = doc["suggested_pod_ids"]
        if raw:
            try:
                suggested = json.loads(raw)
            except (TypeError, ValueError):
                suggested = []
            for pod_id in suggested:
                if pod_id in valid_pod_ids and pod_id not in linked_pods:
                    edges.append(
                        Edge(
                            "document",
                            doc_id,
                            "ABOUT_POD",
                            "pod",
                            pod_id,
                            confidence=0.8,
                            method="suggested_promoted",
                        )
                    )
                    linked_pods.add(pod_id)

        for col, rel, dst_type, canon in (
            ("field_name", "ABOUT_FIELD", "field", fields),
            ("wk_name", "ABOUT_WK", "working_area", wks),
        ):
            for name in _as_list(doc[col]):
                canonical = canon.get(name.strip().lower())
                if canonical:
                    edges.append(
                        Edge(
                            "document",
                            doc_id,
                            rel,
                            dst_type,
                            canonical,
                            confidence=1.0,
                            method="metadata_exact",
                        )
                    )

    # 4: pod -> project
    for row in sqlite_conn.execute(
        "SELECT m.pod_id, pp.project_id FROM project_pod pp "
        "JOIN m_pod m ON m.id = pp.pod_id"
    ).fetchall():
        edges.append(
            Edge(
                "pod",
                row["pod_id"],
                "HAS_PROJECT",
                "project",
                row["project_id"],
                confidence=1.0,
                method="registry",
            )
        )

    # 5: pod revision chain
    for row in sqlite_conn.execute(
        "SELECT successor_id, predecessor_id FROM pod_revision"
    ).fetchall():
        edges.append(
            Edge(
                "pod",
                row["successor_id"],
                "REVISES",
                "pod",
                row["predecessor_id"],
                confidence=1.0,
                method="registry",
            )
        )

    # 6: project -> field -> wk at latest report_year
    latest: list[tuple[Any, ...]] = duck_conn.execute(
        """
        SELECT project_id, field_name, wk_name FROM (
            SELECT project_id, field_name, wk_name,
                   ROW_NUMBER() OVER (
                       PARTITION BY project_id ORDER BY report_year DESC
                   ) AS rn
            FROM project_resources
        ) WHERE rn = 1
        """
    ).fetchall()
    for project_id, field_name, wk_name in latest:
        if field_name:
            edges.append(
                Edge(
                    "project",
                    str(project_id),
                    "IN_FIELD",
                    "field",
                    str(field_name),
                    confidence=1.0,
                    method="registry",
                )
            )
        if field_name and wk_name:
            edges.append(
                Edge(
                    "field",
                    str(field_name),
                    "IN_WK",
                    "working_area",
                    str(wk_name),
                    confidence=1.0,
                    method="registry",
                )
            )

    report.edges_written = store.upsert_edges(edges) if edges else 0
    sqlite_conn.commit()
    logger.info(
        "[Learn] deterministic_linking | edges=%d pod_document_added=%d",
        report.edges_written,
        report.pod_document_added,
    )
    return report
