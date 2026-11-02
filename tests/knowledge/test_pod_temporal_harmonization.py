"""Temporal POD facts survive a real current-path replacement in every consumer.

The retired value-case and pod_revision model projected a revision chain from static
guidance. The current contract derives that chain from one latest-only pointer on
``m_pod``. This test follows that pointer through SQLite truth, the published DuckDB
projection and the real native instance graph, then performs an actual full-input
replacement through the public import path: the protected document bridge, the
permanent identity reservations and the raw document rows must survive it, the graph
must refuse stale knowledge until a real deterministic rebuild marks it current, and
every consumer must then report the change.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import duckdb
import pytest
from openpyxl import load_workbook

from esdc.chat.domain_knowledge.instance_graph import InstanceGraphManager
from esdc.knowledge.freshness import compute_knowledge_basis, mark_knowledge_current
from esdc.knowledge.linker import run_deterministic_linking
from esdc.knowledge.store import KnowledgeStore
from esdc.loaders import generate_pod_workbook_template
from esdc.pod_registry.contract import load_pod_contract
from esdc.pod_registry.publication import publish_v2_registry
from esdc.pod_registry.sqlite_import import (
    load_prepared_pod_batch,
    prepare_pod_import,
)
from tests.knowledge.conftest import (
    POD_DURI,
    POD_DURI_REV1,
    POD_KAMPUNG,
    PROJECT_DURI,
    PROJECT_KAMPUNG,
)
from tests.pod_registry.test_importer import ACTOR, INPUT_TABLES

TemporalRow = tuple[str, int, str | None]
Answer = dict[str, list[dict[str, Any]]]

# One pointer only: the 2019 master is superseded by the current 2022 revision.
# Successors are named by their business identity, not by a surrogate key.
TEMPORAL_TRUTH: list[TemporalRow] = [
    (POD_DURI, 0, POD_DURI_REV1),
    (POD_KAMPUNG, 1, None),
    (POD_DURI_REV1, 1, None),
]

TRUTH_QUERY = (
    "SELECT m.pod_id, m.is_latest, s.pod_id FROM m_pod m "
    "LEFT JOIN m_pod s ON s.pod_id_seq = m.superseded_by "
    "ORDER BY m.pod_id"
)

REVISION_EDGE = {
    "a.pod_id": POD_DURI_REV1,
    "b.pod_id": POD_DURI,
    "r.valid_from": "2022-01-01",
    "r.properties_json": (
        '{"revision_effect": "full_replacement", "source": "m_pod.superseded_by"}'
    ),
}
SUPERSEDE_EDGE = {
    "a.pod_id": POD_DURI_REV1,
    "b.pod_id": POD_DURI,
    "r.valid_from": "2022-01-01",
}
PROTECTED_DOC = "DOC-PROTECTED"
IMPORTED_NAME = "POD I Duri Revisi 1 (imported)"
ORIGINAL_NAMES = [
    (POD_DURI, "POD I Duri"),
    (POD_KAMPUNG, "POD Kampung Baru"),
    (POD_DURI_REV1, "POD I Duri Revisi 1"),
]
CHANGED_NAMES = [
    (POD_DURI, "POD I Duri"),
    (POD_KAMPUNG, "POD Kampung Baru"),
    (POD_DURI_REV1, IMPORTED_NAME),
]


@pytest.fixture(autouse=True)
def _reset_instance_graph_singleton():
    InstanceGraphManager.reset_for_tests()
    yield
    InstanceGraphManager.reset_for_tests()


def _read_truth(
    conn: duckdb.DuckDBPyConnection | sqlite3.Connection,
) -> list[TemporalRow]:
    return [
        (str(row[0]), int(bool(row[1])), None if row[2] is None else str(row[2]))
        for row in conn.execute(TRUTH_QUERY).fetchall()
    ]


def _projection_truth(duck_path: Path) -> list[TemporalRow]:
    conn = duckdb.connect(str(duck_path), read_only=True)
    try:
        return _read_truth(conn)
    finally:
        conn.close()


def _projection_names(duck_path: Path) -> list[tuple[str, str]]:
    conn = duckdb.connect(str(duck_path), read_only=True)
    try:
        return [
            (str(row[0]), str(row[1]))
            for row in conn.execute(
                "SELECT pod_id, pod_name FROM m_pod ORDER BY pod_id"
            ).fetchall()
        ]
    finally:
        conn.close()


def _sqlite_names(conn: sqlite3.Connection) -> list[tuple[str, str]]:
    return [
        (str(row[0]), str(row[1]))
        for row in conn.execute(
            "SELECT pod_id, pod_name FROM m_pod ORDER BY pod_id"
        ).fetchall()
    ]


def _raw_documents(conn: sqlite3.Connection) -> list[tuple[Any, ...]]:
    """Raw document fields; derived suggestions are excluded on purpose.

    Source reconciliation legitimately clears ``suggested_pod_ids``, so comparing the
    raw fields is what proves the protected row itself is untouched.
    """
    columns = [str(row[1]) for row in conn.execute("PRAGMA table_info(documents)")]
    columns.remove("suggested_pod_ids")
    selected = ", ".join(f'"{name}"' for name in columns)
    rows = conn.execute(f"SELECT {selected} FROM documents ORDER BY doc_id").fetchall()
    return [tuple(row) for row in rows]


def _protected_links(conn: sqlite3.Connection) -> list[tuple[Any, ...]]:
    rows = conn.execute("SELECT * FROM pod_document ORDER BY doc_id").fetchall()
    return [tuple(row) for row in rows]


def _ledger_reservations(conn: sqlite3.Connection) -> list[tuple[Any, ...]]:
    rows = conn.execute(
        "SELECT pod_id_seq, pod_id, pod_uuid FROM _pod_identity_ledger ORDER BY pod_id_seq"
    ).fetchall()
    return [tuple(row) for row in rows]


def _source_generation(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT source_generation FROM pod_registry_state").fetchone()
    return int(row[0])


def _graph_answers(mgr: InstanceGraphManager) -> Answer:
    return {
        "temporal_edges": mgr.query(
            "MATCH (a:POD)-[r:REVISES]->(b:POD) "
            "RETURN a.pod_id, b.pod_id, r.valid_from, r.properties_json "
            "ORDER BY b.pod_id"
        ),
        "supersede_edges": mgr.query(
            "MATCH (a:POD)-[r:SUPERSEDES]->(b:POD) "
            "RETURN a.pod_id, b.pod_id, r.valid_from ORDER BY b.pod_id"
        ),
        "document_links": mgr.query(
            "MATCH (d:Document)-[:ABOUT_POD]->(p:POD) "
            "RETURN d.doc_id, p.pod_id ORDER BY d.doc_id"
        ),
        "project_links": mgr.query(
            "MATCH (a:POD)-[:HAS_PROJECT]->(b:Project) "
            "RETURN a.pod_id, b.project_id ORDER BY a.pod_id"
        ),
        "pod_names": mgr.query(
            "MATCH (p:POD) RETURN p.pod_id, p.pod_name ORDER BY p.pod_id"
        ),
    }


def _pod_names(answers: Answer) -> list[tuple[str, str]]:
    return [(row["p.pod_id"], row["p.pod_name"]) for row in answers["pod_names"]]


def _rebuild(conn: sqlite3.Connection, duck_path: Path) -> None:
    """Real deterministic rebuild, then the knowledge current boundary."""
    # ``allow_stale`` only lets the writer run while the source has moved ahead of the
    # stored marker; no consumer may read a graph over stale knowledge.
    store = KnowledgeStore(conn, allow_stale=True)
    store.ensure_tables()
    duck = duckdb.connect(str(duck_path), read_only=True)
    try:
        run_deterministic_linking(conn, duck, store)
    finally:
        duck.close()
    conn.commit()
    conn.execute("BEGIN IMMEDIATE")
    assert mark_knowledge_current(conn, compute_knowledge_basis(conn))
    conn.commit()


def test_current_replacement_preserves_the_protected_bridge_in_every_consumer(
    sqlite_conn: sqlite3.Connection, tmp_path: Path
) -> None:
    sqlite_path = tmp_path / "esdc.sqlite"
    duck_path = tmp_path / "esdc.duckdb"

    assert _read_truth(sqlite_conn) == TEMPORAL_TRUTH

    current_seq = int(
        sqlite_conn.execute(
            "SELECT pod_id_seq FROM m_pod WHERE pod_id = ?", (POD_DURI_REV1,)
        ).fetchone()[0]
    )
    sqlite_conn.execute(
        "INSERT INTO documents (doc_id, file_name, file_hash, doc_type, doc_topic, "
        "markdown) VALUES (?, 'protected.pdf', 'hash-protected', 'letter', "
        "'[\"pod\"]', '# Keep')",
        (PROTECTED_DOC,),
    )
    sqlite_conn.execute(
        "INSERT INTO pod_document (pod_id_seq, doc_id) VALUES (?, ?)",
        (current_seq, PROTECTED_DOC),
    )
    sqlite_conn.commit()

    publish_v2_registry(sqlite_path, duck_path)
    _rebuild(sqlite_conn, duck_path)

    raw_before = _raw_documents(sqlite_conn)
    links_before = _protected_links(sqlite_conn)
    reservations_before = _ledger_reservations(sqlite_conn)
    generation_before = _source_generation(sqlite_conn)

    assert _projection_truth(duck_path) == TEMPORAL_TRUTH

    mgr = InstanceGraphManager(sqlite_path=sqlite_path)
    try:
        assert mgr.is_available()
        before = _graph_answers(mgr)
        assert before["temporal_edges"] == [REVISION_EDGE]
        assert before["supersede_edges"] == [SUPERSEDE_EDGE]
        # The exact letter match and the verified bridge both resolve the current POD.
        assert before["document_links"] == [
            {"d.doc_id": "DOC-A", "p.pod_id": POD_DURI_REV1},
            {"d.doc_id": PROTECTED_DOC, "p.pod_id": POD_DURI_REV1},
        ]
        assert before["project_links"] == [
            {"a.pod_id": POD_DURI, "b.project_id": PROJECT_DURI},
            {"a.pod_id": POD_KAMPUNG, "b.project_id": PROJECT_KAMPUNG},
            {"a.pod_id": POD_DURI_REV1, "b.project_id": PROJECT_DURI},
        ]
        assert _pod_names(before) == ORIGINAL_NAMES

        # Real current-path replacement: same identities, changed master name and
        # changed project relation, through the public prepare/apply/publish path.
        contract = load_pod_contract()
        template = generate_pod_workbook_template(
            output_path=tmp_path / "incoming.xlsx"
        )
        incoming = template.output_path
        workbook = load_workbook(incoming)
        try:
            for table in INPUT_TABLES:
                columns = [column.name for column in contract.tables[table].columns]
                quoted = ", ".join(f'"{name}"' for name in columns)
                sheet = workbook[table]
                has_seq = "pod_id_seq" in columns
                for row in sqlite_conn.execute(f'SELECT {quoted} FROM "{table}"'):
                    values = list(row)
                    if has_seq and values[columns.index("pod_id_seq")] == current_seq:
                        if table == "m_pod":
                            values[columns.index("pod_name")] = IMPORTED_NAME
                        elif table == "pod_project":
                            values[columns.index("project_id")] = PROJECT_KAMPUNG
                    sheet.append(values)
            workbook.save(incoming)
        finally:
            workbook.close()

        batch = prepare_pod_import(
            incoming, "xlsx", contract, target_path=sqlite_path, stage_snapshots=False
        )
        load_prepared_pod_batch(
            batch, sqlite_path=sqlite_path, duck_path=duck_path, actor=ACTOR
        )

        # Exactly one source generation advance, with the protected state intact.
        assert _source_generation(sqlite_conn) == generation_before + 1
        assert _raw_documents(sqlite_conn) == raw_before
        assert _protected_links(sqlite_conn) == links_before
        assert _ledger_reservations(sqlite_conn) == reservations_before
        assert _projection_truth(duck_path) == TEMPORAL_TRUTH
        # The replacement really applied on the source, not only in a projection.
        assert _sqlite_names(sqlite_conn) == CHANGED_NAMES

        # The same manager must refuse the now-stale knowledge rather than serve the
        # graph it had already built.
        assert not mgr.is_available()

        _rebuild(sqlite_conn, duck_path)
        assert mgr.is_available()
        after = _graph_answers(mgr)
        assert after["temporal_edges"] == before["temporal_edges"]
        assert after["supersede_edges"] == before["supersede_edges"]
        assert after["document_links"] == before["document_links"]
        assert _pod_names(after) == CHANGED_NAMES
        assert after["project_links"] == [
            {"a.pod_id": POD_DURI, "b.project_id": PROJECT_DURI},
            {"a.pod_id": POD_KAMPUNG, "b.project_id": PROJECT_KAMPUNG},
            {"a.pod_id": POD_DURI_REV1, "b.project_id": PROJECT_KAMPUNG},
        ]
        graph_names = _pod_names(after)
    finally:
        mgr.close()

    # Rebuild the projection from the same source with no live handle, and check the
    # captured graph answers still agree with it.
    publish_v2_registry(sqlite_path, duck_path)
    assert _projection_truth(duck_path) == TEMPORAL_TRUTH
    assert _projection_names(duck_path) == graph_names == CHANGED_NAMES
