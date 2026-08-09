"""Tests for entity linking."""

from __future__ import annotations

from esdc.knowledge.linker import run_deterministic_linking
from esdc.knowledge.store import Edge, KnowledgeStore


def _run(sqlite_conn, duck_conn):
    store = KnowledgeStore(sqlite_conn)
    store.ensure_tables()
    report = run_deterministic_linking(sqlite_conn, duck_conn, store)
    return store, report


def _rels(edges, rel):
    return [e for e in edges if e.rel == rel]


def test_letter_number_exact_links_and_promotes_pod_document(sqlite_conn, duck_conn):
    store, report = _run(sqlite_conn, duck_conn)
    edges = _rels(store.edges_for("document", "DOC-A"), "ABOUT_POD")
    assert any(
        e.dst_id == "PL-2022-0002-2-2-1"
        and e.method == "letter_num_exact"
        and e.confidence == 1.0
        for e in edges
    )
    rows = sqlite_conn.execute("SELECT pod_id, doc_id FROM pod_document").fetchall()
    assert (2, "DOC-A") in [tuple(r) for r in rows]
    assert report.pod_document_added == 1


def test_suggested_pod_ids_promoted_without_pod_document(sqlite_conn, duck_conn):
    store, _ = _run(sqlite_conn, duck_conn)
    edges = _rels(store.edges_for("document", "DOC-B"), "ABOUT_POD")
    assert [(e.dst_id, e.method, e.confidence) for e in edges] == [
        ("PL-2019-0001-2-2-0", "suggested_promoted", 0.8)
    ]
    rows = sqlite_conn.execute(
        "SELECT doc_id FROM pod_document WHERE doc_id = 'DOC-B'"
    ).fetchall()
    assert rows == []


def test_field_and_wk_metadata_exact_edges_use_canonical_names(sqlite_conn, duck_conn):
    store, _ = _run(sqlite_conn, duck_conn)
    edges_a = store.edges_for("document", "DOC-A")
    assert any(e.rel == "ABOUT_FIELD" and e.dst_id == "Duri" for e in edges_a)
    assert any(e.rel == "ABOUT_WK" and e.dst_id == "Rokan" for e in edges_a)
    # regulation doc has no entity metadata -> no edges
    assert store.edges_for("document", "DOC-C") == []


def _insert_doc(conn, doc_id, wk_name, field_name):
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
    conn.commit()


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
    pod1 = store.edges_for("pod", "PL-2019-0001-2-2-0")
    assert any(e.rel == "HAS_PROJECT" and e.dst_id == "PRJ-001" for e in pod1)
    assert any(
        e.rel == "REVISES"
        and e.src_id == "PL-2022-0002-2-2-1"
        and e.dst_id == "PL-2019-0001-2-2-0"
        for e in pod1
    )
    prj = store.edges_for("project", "PRJ-001")
    assert any(e.rel == "IN_FIELD" and e.dst_id == "Duri" for e in prj)
    fld = store.edges_for("field", "Duri")
    assert any(e.rel == "IN_WK" and e.dst_id == "Rokan" for e in fld)


def test_rerun_is_idempotent(sqlite_conn, duck_conn):
    store, _ = _run(sqlite_conn, duck_conn)
    first = sqlite_conn.execute("SELECT COUNT(*) FROM kg_edge").fetchone()[0]
    run_deterministic_linking(sqlite_conn, duck_conn, store)
    second = sqlite_conn.execute("SELECT COUNT(*) FROM kg_edge").fetchone()[0]
    assert first == second
    pod_docs = sqlite_conn.execute("SELECT COUNT(*) FROM pod_document").fetchone()[0]
    assert pod_docs == 1


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


def _insert_revision(conn, successor, predecessor, effect, date, scope, prev_valid):
    conn.execute(
        "INSERT INTO pod_revision (successor_id, predecessor_id, revision_effect,"
        " effective_date, amended_scope, previous_remains_valid)"
        " VALUES (?, ?, ?, ?, ?, ?)",
        (successor, predecessor, effect, date, scope, prev_valid),
    )
    conn.commit()


def test_unknown_revision_links_but_never_supersedes(sqlite_conn, duck_conn):
    store = KnowledgeStore(sqlite_conn)
    store.ensure_tables()
    run_deterministic_linking(sqlite_conn, duck_conn, store)
    edges = store.edges_for("pod", "PL-2019-0001-2-2-0")
    assert _rels(edges, "SUPERSEDES") == []
    assert any(
        e.src_id == "PL-2022-0002-2-2-1"
        and e.valid_from is None
        and e.valid_to is None
        and e.properties_json
        == '{"amended_scope": null, "previous_remains_valid": null,'
        ' "revision_effect": "unknown"}'
        for e in _rels(edges, "REVISES")
    )


def test_partial_amendment_revision_edge_with_temporal_json(sqlite_conn, duck_conn):
    _insert_revision(
        sqlite_conn,
        "PL-2020-0003-2-2-0",
        "PL-2019-0001-2-2-0",
        "partial_amendment",
        "2020-02-15",
        "sec. 3",
        1,
    )
    store = KnowledgeStore(sqlite_conn)
    store.ensure_tables()
    run_deterministic_linking(sqlite_conn, duck_conn, store)
    edges = store.edges_for("pod", "PL-2019-0001-2-2-0")
    assert _rels(edges, "SUPERSEDES") == []
    assert any(
        e.src_id == "PL-2020-0003-2-2-0"
        and e.valid_from == "2020-02-15"
        and e.valid_to is None
        and e.properties_json
        == '{"amended_scope": "sec. 3", "previous_remains_valid": true,'
        ' "revision_effect": "partial_amendment"}'
        for e in _rels(edges, "REVISES")
    )


def test_full_replacement_revision_adds_supersedes_with_exact_json(
    sqlite_conn, duck_conn
):
    _insert_revision(
        sqlite_conn,
        "PL-2020-0003-2-2-0",
        "PL-2019-0001-2-2-0",
        "full_replacement",
        "2020-06-01",
        "whole pod",
        0,
    )
    store = KnowledgeStore(sqlite_conn)
    store.ensure_tables()
    run_deterministic_linking(sqlite_conn, duck_conn, store)
    edges = store.edges_for("pod", "PL-2019-0001-2-2-0")
    assert any(
        e.src_id == "PL-2020-0003-2-2-0"
        and e.valid_from == "2020-06-01"
        and e.valid_to is None
        and e.properties_json
        == '{"amended_scope": "whole pod", "previous_remains_valid": false,'
        ' "revision_effect": "full_replacement"}'
        for e in _rels(edges, "SUPERSEDES")
    )


def test_reconciliation_refreshes_or_drops_registry_revision_edges(
    sqlite_conn, duck_conn
):
    """Registry REVISES/SUPERSEDES follow pod_revision across reruns.

    An effect downgrade drops SUPERSEDES and refreshes equal-confidence
    temporal fields; a deleted pod_revision drops its REVISES.
    """
    store = KnowledgeStore(sqlite_conn)
    store.ensure_tables()
    successor = "PL-2020-0003-2-2-0"
    predecessor = "PL-2019-0001-2-2-0"

    _insert_revision(
        sqlite_conn,
        successor,
        predecessor,
        "full_replacement",
        "2020-06-01",
        "whole pod",
        0,
    )
    run_deterministic_linking(sqlite_conn, duck_conn, store)
    edges = store.edges_for("pod", predecessor)
    assert any(
        e.src_id == successor and e.method == "registry"
        for e in _rels(edges, "SUPERSEDES")
    )

    sqlite_conn.execute(
        "UPDATE pod_revision SET revision_effect = 'partial_amendment',"
        " effective_date = '2020-02-15', amended_scope = 'sec. 3',"
        " previous_remains_valid = 1"
        " WHERE successor_id = ? AND predecessor_id = ?",
        (successor, predecessor),
    )
    sqlite_conn.commit()
    run_deterministic_linking(sqlite_conn, duck_conn, store)
    edges = store.edges_for("pod", predecessor)
    assert _rels(edges, "SUPERSEDES") == []
    revises = [e for e in _rels(edges, "REVISES") if e.src_id == successor]
    assert len(revises) == 1
    assert revises[0].valid_from == "2020-02-15"
    assert revises[0].properties_json == (
        '{"amended_scope": "sec. 3", "previous_remains_valid": true,'
        ' "revision_effect": "partial_amendment"}'
    )

    sqlite_conn.execute(
        "DELETE FROM pod_revision WHERE successor_id = ? AND predecessor_id = ?",
        (successor, predecessor),
    )
    sqlite_conn.commit()
    run_deterministic_linking(sqlite_conn, duck_conn, store)
    edges = store.edges_for("pod", predecessor)
    assert all(e.src_id != successor for e in _rels(edges, "REVISES"))
