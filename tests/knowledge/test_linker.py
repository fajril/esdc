from __future__ import annotations

from esdc.knowledge.linker import run_deterministic_linking
from esdc.knowledge.store import KnowledgeStore


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
        (doc_id, f"{doc_id}.pdf", f"hash-{doc_id}", f"NO-{doc_id}",
         wk_name, field_name),
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
        e.dst_id for e in store.edges_for("document", "DOC-M")
        if e.rel == "ABOUT_FIELD"
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


def test_unparseable_metadata_value_is_treated_as_a_plain_name(
    sqlite_conn, duck_conn
):
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
