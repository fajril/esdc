"""Deterministic knowledge edges from registries and document metadata.

Phase 1 of `esdc corpus learn`. No LLM calls; safe to rerun any time.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass

import duckdb

from esdc.corpus.context import _as_list
from esdc.corpus.pod_matcher import PodMatcher, _norm
from esdc.knowledge.store import Edge, KnowledgeStore
from esdc.pod_registry.migrations import _validate_existing_v2_registry

logger = logging.getLogger(__name__)


@dataclass
class LinkReport:
    edges_written: int = 0
    pod_document_added: int = 0


def _sqlite_has_table(conn: sqlite3.Connection, table: str) -> bool:
    return (
        conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type IN ('table','view') AND name=?",
            [table],
        ).fetchone()
        is not None
    )


def _v2_document_rows(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    if not _sqlite_has_table(conn, "documents"):
        return []
    available = {
        str(row[1]) for row in conn.execute("PRAGMA table_info(documents)")
    }
    desired = (
        "doc_id",
        "doc_number",
        "pod_name",
        "field_name",
        "wk_name",
    )
    selected = tuple(name for name in desired if name in available)
    projection = ", ".join(f'"{name}"' for name in selected)
    return conn.execute(
        f"SELECT {projection} FROM documents ORDER BY doc_id"
    ).fetchall()


def _canonical_names(conn: sqlite3.Connection, column: str) -> dict[str, str]:
    """Lowercase name -> canonical spelling, from the canonical raw resource source."""
    names: set[str] = set()
    for row in _v2_resource_rows(conn):
        if column not in set(row.keys()):
            continue
        names.update(
            str(name).strip()
            for name in _as_list(row[column])
            if name and str(name).strip()
        )
    return {name.casefold(): name for name in names}


def _v2_resource_rows(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    table = "api_project_resources"
    if not _sqlite_has_table(conn, table):
        return []
    available = {
        str(row[1]) for row in conn.execute(f'PRAGMA table_info("{table}")')
    }
    required = {"project_id", "report_year"}
    if not required <= available:
        return []
    selected = tuple(
        name
        for name in (
            "project_id",
            "report_year",
            "project_name",
            "project_remarks",
            "field_name",
            "wk_name",
        )
        if name in available
    )
    projection = ", ".join(f'"{name}"' for name in selected)
    return conn.execute(
        f'SELECT {projection} FROM "{table}" '
        'ORDER BY project_id, CAST(report_year AS INTEGER) DESC, '
        'project_name, project_remarks, field_name, wk_name'
    ).fetchall()


def _run_v2_deterministic_linking(
    sqlite_conn: sqlite3.Connection,
    store: KnowledgeStore,
    *,
    commit: bool,
) -> LinkReport:
    report = LinkReport()
    edges: list[Edge] = []
    matcher = PodMatcher(sqlite_conn)
    pod_columns = {
        str(row[1]) for row in sqlite_conn.execute("PRAGMA table_info(m_pod)")
    }
    required = {
        "pod_id_seq",
        "pod_id",
        "pod_letter_num",
        "pod_name",
        "is_latest",
        "approval_date",
        "superseded_by",
    }
    if not required <= pod_columns:
        raise sqlite3.OperationalError("m_pod v2 columns are incomplete")

    pod_rows = sqlite_conn.execute(
        "SELECT pod_id_seq, pod_id, pod_letter_num, pod_name, is_latest, "
        "approval_date, superseded_by FROM m_pod "
        "WHERE pod_id IS NOT NULL AND trim(pod_id) <> ''"
    ).fetchall()
    pod_by_seq = {int(row["pod_id_seq"]): str(row["pod_id"]) for row in pod_rows}
    letter_candidates: dict[str, list[str]] = {}
    for row in pod_rows:
        if not bool(row["is_latest"]):
            continue
        letter = _norm(row["pod_letter_num"])
        if letter:
            letter_candidates.setdefault(letter, []).append(str(row["pod_id"]))

    fields = _canonical_names(sqlite_conn, "field_name")
    wks = _canonical_names(sqlite_conn, "wk_name")
    latest_resources: dict[str, sqlite3.Row] = {}
    for row in _v2_resource_rows(sqlite_conn):
        latest_resources.setdefault(str(row["project_id"]), row)

    for doc in _v2_document_rows(sqlite_conn):
        doc_columns = set(doc.keys())
        doc_id = str(doc["doc_id"])
        pod_id, method = matcher.resolve_exact(
            doc_number=doc["doc_number"] if "doc_number" in doc_columns else None,
            pod_names=(
                _as_list(doc["pod_name"])
                if "pod_name" in doc_columns
                else None
            ),
        )
        if pod_id is not None:
            evidence = (
                f"doc_number={doc['doc_number']}"
                if method == "letter_num_exact"
                else f"pod_name={doc['pod_name']}"
            )
            edges.append(
                Edge(
                    "document",
                    doc_id,
                    "ABOUT_POD",
                    "pod",
                    pod_id,
                    confidence=1.0,
                    method=method or "exact_current_identity",
                    evidence=evidence,
                )
            )
        for col, rel, dst_type, canon in (
            ("field_name", "ABOUT_FIELD", "field", fields),
            ("wk_name", "ABOUT_WK", "working_area", wks),
        ):
            if col not in doc_columns:
                continue
            for name in _as_list(doc[col]):
                canonical = canon.get(str(name).strip().casefold())
                if canonical:
                    edges.append(
                        Edge(
                            "document",
                            doc_id,
                            rel,
                            dst_type,
                            canonical,
                            confidence=1.0,
                            method="metadata_exact_current",
                        )
                    )

    # Existing pod_document rows are retained as verified explicit links.
    if _sqlite_has_table(sqlite_conn, "pod_document"):
        for row in sqlite_conn.execute(
            "SELECT pd.doc_id, m.pod_id FROM pod_document pd "
            "JOIN m_pod m ON m.pod_id_seq=pd.pod_id_seq "
            "WHERE m.pod_id IS NOT NULL AND trim(m.pod_id) <> ''"
        ).fetchall():
            edges.append(
                Edge(
                    "document",
                    str(row["doc_id"]),
                    "ABOUT_POD",
                    "pod",
                    str(row["pod_id"]),
                    confidence=1.0,
                    method="pod_document_explicit",
                    evidence="verified pod_document bridge",
                )
            )

    if _sqlite_has_table(sqlite_conn, "pod_project"):
        for row in sqlite_conn.execute(
            "SELECT m.pod_id, pp.project_id FROM pod_project pp "
            "JOIN m_pod m ON m.pod_id_seq=pp.pod_id_seq "
            "WHERE m.pod_id IS NOT NULL AND trim(m.pod_id) <> ''"
        ).fetchall():
            edges.append(
                Edge(
                    "pod",
                    str(row["pod_id"]),
                    "HAS_PROJECT",
                    "project",
                    str(row["project_id"]),
                    confidence=1.0,
                    method="registry_current",
                )
            )

    sqlite_conn.execute(
        "DELETE FROM kg_edge WHERE method='registry_current' "
        "AND rel IN ('REVISES','SUPERSEDES')"
    )
    for old in pod_rows:
        new_seq = old["superseded_by"]
        if new_seq is None or int(new_seq) not in pod_by_seq:
            continue
        new = next(row for row in pod_rows if int(row["pod_id_seq"]) == int(new_seq))
        properties = json.dumps(
            {"revision_effect": "full_replacement", "source": "m_pod.superseded_by"},
            sort_keys=True,
        )
        edges.append(
            Edge(
                "pod",
                str(new["pod_id"]),
                "REVISES",
                "pod",
                str(old["pod_id"]),
                confidence=1.0,
                method="registry_current",
                valid_from=str(new["approval_date"]),
                properties_json=properties,
            )
        )
        edges.append(
            Edge(
                "pod",
                str(new["pod_id"]),
                "SUPERSEDES",
                "pod",
                str(old["pod_id"]),
                confidence=1.0,
                method="registry_current",
                valid_from=str(new["approval_date"]),
                properties_json=properties,
            )
        )

    for project_id, row in latest_resources.items():
        resource_columns = set(row.keys())
        field_names = (
            _as_list(row["field_name"])
            if "field_name" in resource_columns
            else []
        )
        wk_names = (
            _as_list(row["wk_name"]) if "wk_name" in resource_columns else []
        )
        field_names_canonical = set(fields.values())
        wk_names_canonical = set(wks.values())
        field = next(
            (
                fields.get(name.strip().casefold())
                for name in field_names
                if name.strip() in field_names_canonical
            ),
            None,
        )
        wk = next(
            (
                wks.get(name.strip().casefold())
                for name in wk_names
                if name.strip() in wk_names_canonical
            ),
            None,
        )
        if field:
            edges.append(
                Edge(
                    "project", project_id, "IN_FIELD", "field", field,
                    confidence=1.0, method="registry_current"
                )
            )
        if field and wk:
            edges.append(
                Edge(
                    "field", field, "IN_WK", "working_area", wk,
                    confidence=1.0, method="registry_current"
                )
            )

    report.edges_written = store.upsert_edges(edges, commit=commit) if edges else 0
    if commit:
        sqlite_conn.commit()
    return report


def run_deterministic_linking(
    sqlite_conn: sqlite3.Connection,
    duck_conn: duckdb.DuckDBPyConnection,
    store: KnowledgeStore,
    *,
    commit: bool = True,
) -> LinkReport:
    """Rebuild registry-derived knowledge from the single current schema.

    ``duck_conn`` is kept for caller compatibility while the instance graph is
    still on its pending conversion; the current path reads only SQLite truth.

    The source is validated with the shared registry validator before anything is
    written: a wrong or missing current schema is refused, never reinterpreted or
    repaired here.
    """
    _validate_existing_v2_registry(sqlite_conn)
    report = _run_v2_deterministic_linking(sqlite_conn, store, commit=commit)
    logger.info(
        "[Learn] deterministic_linking | edges=%d pod_document_added=%d",
        report.edges_written,
        report.pod_document_added,
    )
    return report
