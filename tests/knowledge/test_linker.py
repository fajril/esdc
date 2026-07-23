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
