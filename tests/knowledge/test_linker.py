"""Tests for entity linking."""

from __future__ import annotations

import json
import sqlite3
from datetime import date

import pytest

from esdc.knowledge.freshness import compute_knowledge_basis, mark_knowledge_current
from esdc.knowledge.linker import run_deterministic_linking
from esdc.knowledge.store import Edge, KnowledgeStore
from esdc.pod_registry.migrations import MigrationRefusedError
from tests.knowledge.conftest import (
    POD_DURI,
    POD_DURI_REV1,
    POD_KAMPUNG,
    PROJECT_DURI,
    _issue_pod,
)


def _run(sqlite_conn, duck_conn):
    store = KnowledgeStore(sqlite_conn)
    store.ensure_tables()
    report = run_deterministic_linking(sqlite_conn, duck_conn, store)
    return store, report


def _rels(edges, rel):
    return [e for e in edges if e.rel == rel]


def _pod_seq(sqlite_conn: sqlite3.Connection, pod_id: str) -> int:
    row = sqlite_conn.execute(
        "SELECT pod_id_seq FROM m_pod WHERE pod_id = ?", (pod_id,)
    ).fetchone()
    assert row is not None
    return int(row["pod_id_seq"])


def _finish_source_write(conn: sqlite3.Connection) -> None:
    """Close a source write the way reconciliation does.

    A source-row insert invalidates the knowledge basis, so the write boundary
    re-marks it for the same snapshot the readers measure. Without that the test
    would be reading a correctly-refused stale knowledge store.
    """
    if not conn.in_transaction:
        conn.execute("BEGIN IMMEDIATE")
    assert mark_knowledge_current(conn, compute_knowledge_basis(conn))
    conn.commit()


def test_letter_number_exact_match_links_document_to_current_identity(
    sqlite_conn, duck_conn
):
    store, report = _run(sqlite_conn, duck_conn)
    edges = _rels(store.edges_for("document", "DOC-A"), "ABOUT_POD")
    assert any(
        e.dst_id == POD_DURI_REV1
        and e.method == "letter_num_exact"
        and e.confidence == 1.0
        for e in edges
    )
    # Linking never writes pod_document rows: verified links are only read.
    assert sqlite_conn.execute("SELECT COUNT(*) FROM pod_document").fetchone()[0] == 0
    assert report.pod_document_added == 0


def test_suggested_pod_ids_are_review_only_and_never_become_edges(
    sqlite_conn, duck_conn
):
    """DOC-B carries a suggestion; a suggestion is not an accepted link."""
    store, report = _run(sqlite_conn, duck_conn)
    assert _rels(store.edges_for("document", "DOC-B"), "ABOUT_POD") == []
    assert sqlite_conn.execute(
        "SELECT suggested_pod_ids FROM documents WHERE doc_id = 'DOC-B'"
    ).fetchone()[0] == json.dumps([POD_DURI])
    assert sqlite_conn.execute(
        "SELECT COUNT(*) FROM pod_document WHERE doc_id = 'DOC-B'"
    ).fetchone()[0] == 0
    assert report.pod_document_added == 0


def test_existing_pod_document_row_is_read_as_an_explicit_edge(
    sqlite_conn, duck_conn
):
    """A verified link produces an edge for its pod; it is never promoted."""
    store = KnowledgeStore(sqlite_conn)
    store.ensure_tables()
    sqlite_conn.execute(
        "INSERT INTO pod_document (pod_id_seq, doc_id) VALUES (?, ?)",
        (_pod_seq(sqlite_conn, POD_DURI), "DOC-B"),
    )
    _finish_source_write(sqlite_conn)
    run_deterministic_linking(sqlite_conn, duck_conn, store)
    edges = _rels(store.edges_for("document", "DOC-B"), "ABOUT_POD")
    assert [(e.dst_id, e.method, e.confidence) for e in edges] == [
        (POD_DURI, "pod_document_explicit", 1.0)
    ]
    assert [
        tuple(row)
        for row in sqlite_conn.execute("SELECT pod_id_seq, doc_id FROM pod_document")
    ] == [(_pod_seq(sqlite_conn, POD_DURI), "DOC-B")]


def test_llm_edge_does_not_overwrite_a_protected_explicit_link(sqlite_conn, duck_conn):
    """A verified link keeps its method; a lower-confidence LLM edge merges into it.

    Document edges come from the write path only, and the confidence rule means an
    LLM edge can never replace the method recorded for an existing
    `pod_document_explicit` row on the same document and POD.
    """
    store = KnowledgeStore(sqlite_conn)
    store.ensure_tables()
    sqlite_conn.execute(
        "INSERT INTO pod_document (pod_id_seq, doc_id) VALUES (?, ?)",
        (_pod_seq(sqlite_conn, POD_DURI), "DOC-B"),
    )
    _finish_source_write(sqlite_conn)
    run_deterministic_linking(sqlite_conn, duck_conn, store)

    store.upsert_edges(
        [
            Edge(
                "document",
                "DOC-B",
                "ABOUT_POD",
                "pod",
                POD_DURI,
                confidence=0.7,
                method="llm_extracted",
                evidence="mention: POD I Duri",
            )
        ]
    )
    edges = _rels(store.edges_for("document", "DOC-B"), "ABOUT_POD")
    assert [(e.method, e.confidence) for e in edges] == [("pod_document_explicit", 1.0)]
    assert sqlite_conn.execute(
        "SELECT COUNT(*) FROM pod_document WHERE doc_id = 'DOC-B'"
    ).fetchone()[0] == 1


def test_document_edges_keep_the_document_date_not_the_approval_date(
    sqlite_conn, duck_conn
):
    """A document carries its own date; approval dates never replace it.

    Only the pointer-derived revision edges carry an approval date. A document
    dated before its POD's approval is still recorded with its own date, and the
    edge derived from it records no approval date.
    """
    store = KnowledgeStore(sqlite_conn)
    store.ensure_tables()
    sqlite_conn.execute(
        "INSERT INTO documents (doc_id, file_name, file_hash, doc_type, doc_number, "
        "doc_date, subject, wk_name, field_name) "
        "VALUES (?, 'early.pdf', 'hash-early', 'letter', ?, '2019-06-01', "
        "'Pre-approval letter', 'Rokan', 'Duri')",
        ("DOC-E", "SRT-0100/SKKMA0000/2022/S1"),
    )
    _finish_source_write(sqlite_conn)
    run_deterministic_linking(sqlite_conn, duck_conn, store)

    edges = _rels(store.edges_for("document", "DOC-E"), "ABOUT_POD")
    assert [(e.dst_id, e.method, e.valid_from) for e in edges] == [
        (POD_DURI_REV1, "letter_num_exact", None)
    ]
    stored_date = sqlite_conn.execute(
        "SELECT doc_date FROM documents WHERE doc_id = 'DOC-E'"
    ).fetchone()[0]
    assert stored_date == "2019-06-01"
    # The revision edge is the only place an approval date is recorded.
    revision = _rels(store.edges_for("pod", POD_DURI), "REVISES")
    assert [e.valid_from for e in revision] == ["2022-01-01"]


def test_field_and_wk_metadata_exact_edges_use_canonical_names(sqlite_conn, duck_conn):
    store, _ = _run(sqlite_conn, duck_conn)
    edges_a = store.edges_for("document", "DOC-A")
    assert any(e.rel == "ABOUT_FIELD" and e.dst_id == "Duri" for e in edges_a)
    assert any(e.rel == "ABOUT_WK" and e.dst_id == "Rokan" for e in edges_a)
    # regulation doc has no entity metadata -> no edges
    assert store.edges_for("document", "DOC-C") == []


def _insert_doc(conn: sqlite3.Connection, doc_id: str, wk_name: str, field_name: str) -> None:
    conn.execute(
        "INSERT INTO documents (doc_id, file_name, file_hash, doc_type, "
        "doc_number, wk_name, field_name) VALUES (?, ?, ?, 'letter', ?, ?, ?)",
        (
            doc_id,
            f"{doc_id}.pdf",
            f"hash-{doc_id}",
            f"NO-{doc_id}",
            wk_name,
            field_name,
        ),
    )
    _finish_source_write(conn)


def test_json_array_metadata_produces_edges(sqlite_conn, duck_conn):
    """Production stores these columns as JSON arrays, not bare strings."""
    _insert_doc(sqlite_conn, "DOC-J", '["Rokan"]', '["Duri"]')
    store, _ = _run(sqlite_conn, duck_conn)
    edges = store.edges_for("document", "DOC-J")
    assert any(e.rel == "ABOUT_FIELD" and e.dst_id == "Duri" for e in edges)
    assert any(e.rel == "ABOUT_WK" and e.dst_id == "Rokan" for e in edges)


def test_json_array_with_several_values_links_each(sqlite_conn, duck_conn):
    _insert_doc(sqlite_conn, "DOC-M", '["Rokan"]', '["Duri", "Kampung Baru"]')
    store, _ = _run(sqlite_conn, duck_conn)
    fields = sorted(
        e.dst_id for e in store.edges_for("document", "DOC-M") if e.rel == "ABOUT_FIELD"
    )
    assert fields == ["Duri", "Kampung Baru"]


def test_json_array_values_are_matched_case_insensitively(sqlite_conn, duck_conn):
    _insert_doc(sqlite_conn, "DOC-U", '["ROKAN"]', '["duri"]')
    store, _ = _run(sqlite_conn, duck_conn)
    edges = store.edges_for("document", "DOC-U")
    assert any(e.rel == "ABOUT_FIELD" and e.dst_id == "Duri" for e in edges)
    assert any(e.rel == "ABOUT_WK" and e.dst_id == "Rokan" for e in edges)


def test_unknown_and_empty_metadata_values_produce_no_edge(sqlite_conn, duck_conn):
    _insert_doc(sqlite_conn, "DOC-N", '["Tidak Ada WK"]', "[]")
    store, _ = _run(sqlite_conn, duck_conn)
    assert store.edges_for("document", "DOC-N") == []


def test_unparseable_metadata_value_is_treated_as_a_plain_name(sqlite_conn, duck_conn):
    """A malformed JSON string must not raise; it falls back to the raw value."""
    _insert_doc(sqlite_conn, "DOC-X", '["Rokan', "Duri")
    store, _ = _run(sqlite_conn, duck_conn)
    edges = store.edges_for("document", "DOC-X")
    assert any(e.rel == "ABOUT_FIELD" and e.dst_id == "Duri" for e in edges)
    assert not any(e.rel == "ABOUT_WK" for e in edges)


def test_registry_edges_projects_revisions_hierarchy(sqlite_conn, duck_conn):
    store, _ = _run(sqlite_conn, duck_conn)
    pod1 = store.edges_for("pod", POD_DURI)
    assert any(e.rel == "HAS_PROJECT" and e.dst_id == PROJECT_DURI for e in pod1)
    assert any(
        e.rel == "REVISES"
        and e.src_id == POD_DURI_REV1
        and e.dst_id == POD_DURI
        for e in pod1
    )
    prj = store.edges_for("project", PROJECT_DURI)
    assert any(e.rel == "IN_FIELD" and e.dst_id == "Duri" for e in prj)
    fld = store.edges_for("field", "Duri")
    assert any(e.rel == "IN_WK" and e.dst_id == "Rokan" for e in fld)


def test_matcher_indexes_only_current_identities(sqlite_conn, duck_conn):
    """A name that matches a superseded master produces no edge; pointers do."""
    store, _ = _run(sqlite_conn, duck_conn)
    # DOC-B names the superseded master exactly, and only current identities are
    # indexed, so nothing is linked from it.
    assert _rels(store.edges_for("document", "DOC-B"), "ABOUT_POD") == []
    current = store.edges_for("pod", POD_DURI_REV1)
    assert any(e.rel == "HAS_PROJECT" and e.dst_id == PROJECT_DURI for e in current)
    assert not [
        e for e in current if e.rel in ("REVISES", "SUPERSEDES") and e.dst_id == POD_DURI_REV1
    ]


def test_pointer_derived_revision_edges_with_exact_json_and_approval_date(
    sqlite_conn, duck_conn
):
    """The superseded_by pointer is the revision contract; there is no revision table."""
    store, _ = _run(sqlite_conn, duck_conn)
    assert not sqlite_conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='pod_revision'"
    ).fetchone()
    edges = _rels(store.edges_for("pod", POD_DURI), "REVISES")
    assert len(edges) == 1
    edge = edges[0]
    assert edge.src_id == POD_DURI_REV1
    assert edge.valid_from == "2022-01-01"
    assert edge.valid_to is None
    assert edge.properties_json == (
        '{"revision_effect": "full_replacement", "source": "m_pod.superseded_by"}'
    )
    supersedes = _rels(store.edges_for("pod", POD_DURI), "SUPERSEDES")
    assert len(supersedes) == 1
    assert supersedes[0].src_id == POD_DURI_REV1
    assert supersedes[0].valid_from == "2022-01-01"


def test_reconciliation_refreshes_or_drops_pointer_derived_edges(
    sqlite_conn, duck_conn
):
    """Registry REVISES/SUPERSEDES follow the pointer across reruns.

    Pointing a current POD at a new successor adds both edges; clearing the
    pointer drops them, because the latest-only pointer is the whole contract.
    """
    store, _ = _run(sqlite_conn, duck_conn)
    successor_seq = _issue_pod(
        sqlite_conn,
        approval_date=date(2024, 5, 1),
        daily_slot=4,
        rev_num=1,
        pod_name="Kampung Baru Rev 1",
        pod_letter_num="SRT-0300/SKKMA0000/2024/S1",
    )
    predecessor_seq = _pod_seq(sqlite_conn, POD_KAMPUNG)
    successor_id = sqlite_conn.execute(
        "SELECT pod_id FROM m_pod WHERE pod_id_seq = ?", (successor_seq,)
    ).fetchone()[0]

    sqlite_conn.execute(
        "UPDATE m_pod SET superseded_by = ?, is_latest = 0 WHERE pod_id_seq = ?",
        (successor_seq, predecessor_seq),
    )
    _finish_source_write(sqlite_conn)
    run_deterministic_linking(sqlite_conn, duck_conn, store)
    edges = store.edges_for("pod", POD_KAMPUNG)
    assert {e.rel for e in edges if e.src_id == successor_id} == {
        "REVISES",
        "SUPERSEDES",
    }

    sqlite_conn.execute(
        "UPDATE m_pod SET superseded_by = NULL, is_latest = 1 WHERE pod_id_seq = ?",
        (predecessor_seq,),
    )
    _finish_source_write(sqlite_conn)
    run_deterministic_linking(sqlite_conn, duck_conn, store)
    edges = store.edges_for("pod", POD_KAMPUNG)
    assert all(e.src_id != successor_id for e in _rels(edges, "REVISES"))
    assert _rels(edges, "SUPERSEDES") == []


def test_rerun_is_idempotent(sqlite_conn, duck_conn):
    store, _ = _run(sqlite_conn, duck_conn)
    first = sqlite_conn.execute("SELECT COUNT(*) FROM kg_edge").fetchone()[0]
    run_deterministic_linking(sqlite_conn, duck_conn, store)
    second = sqlite_conn.execute("SELECT COUNT(*) FROM kg_edge").fetchone()[0]
    assert first == second
    # No link promotion: the verified-link table is left exactly as seeded.
    assert sqlite_conn.execute("SELECT COUNT(*) FROM pod_document").fetchone()[0] == 0


def test_fresh_schema_has_temporal_columns(sqlite_conn):
    store = KnowledgeStore(sqlite_conn)
    store.ensure_tables()
    cols = {r[1] for r in sqlite_conn.execute("PRAGMA table_info(kg_edge)").fetchall()}
    assert {"valid_from", "valid_to", "properties_json"} <= cols


def test_legacy_kg_edge_migrated_in_place(sqlite_conn):
    sqlite_conn.execute(
        """
        CREATE TABLE IF NOT EXISTS kg_edge (
            src_type TEXT NOT NULL, src_id TEXT NOT NULL, rel TEXT NOT NULL,
            dst_type TEXT NOT NULL, dst_id TEXT NOT NULL,
            confidence REAL NOT NULL DEFAULT 1.0, method TEXT NOT NULL,
            evidence TEXT, learned_at TEXT DEFAULT (datetime('now')),
            UNIQUE (src_type, src_id, rel, dst_type, dst_id)
        )
        """
    )
    sqlite_conn.execute(
        "INSERT INTO kg_edge (src_type, src_id, rel, dst_type, dst_id, method) "
        "VALUES ('document', 'D', 'ABOUT_POD', 'pod', 'P', 'test')"
    )
    sqlite_conn.commit()
    store = KnowledgeStore(sqlite_conn)
    store.ensure_tables()
    cols = {r[1] for r in sqlite_conn.execute("PRAGMA table_info(kg_edge)").fetchall()}
    assert {"valid_from", "valid_to", "properties_json"} <= cols
    row = sqlite_conn.execute(
        "SELECT valid_from, valid_to, properties_json FROM kg_edge"
    ).fetchone()
    assert tuple(row) == (None, None, None)


def test_edge_temporal_fields_roundtrip(sqlite_conn):
    store = KnowledgeStore(sqlite_conn)
    store.ensure_tables()
    store.upsert_edges(
        [
            Edge(
                "pod",
                "NEW",
                "REVISES",
                "pod",
                "OLD",
                valid_from="2024-01-01",
                properties_json='{"a": 1}',
            )
        ]
    )
    edges = store.edges_for("pod", "OLD")
    assert len(edges) == 1
    e = edges[0]
    assert e.valid_from == "2024-01-01"
    assert e.valid_to is None
    assert e.properties_json == '{"a": 1}'


def test_linking_refuses_a_source_that_is_not_the_current_schema(
    sqlite_conn, duck_conn
):
    """Refusal happens before any derived table or edge is touched."""
    legacy = sqlite3.connect(":memory:")
    legacy.row_factory = sqlite3.Row
    legacy.executescript(
        """
        CREATE TABLE m_pod (pod_id INTEGER PRIMARY KEY, pod_name TEXT);
        CREATE TABLE pod_document (pod_id INTEGER, doc_id TEXT);
        """
    )
    store = KnowledgeStore(legacy)
    try:
        with pytest.raises(MigrationRefusedError):
            run_deterministic_linking(legacy, duck_conn, store)
        # Nothing was created, so the refusal is not a repair in disguise.
        assert legacy.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table'"
        ).fetchone()[0] == 2
        assert not legacy.in_transaction
    finally:
        legacy.close()


def test_current_schema_linking_still_writes_after_preflight(sqlite_conn, duck_conn):
    """The same canonical source passes the preflight and writes its edges."""
    store, report = _run(sqlite_conn, duck_conn)
    assert report.edges_written > 0
    assert report.pod_document_added == 0
    assert store.edges_for("document", "DOC-A")
