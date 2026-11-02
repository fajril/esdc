"""Tests for the SQLite -> DuckDB wholesale mirror refresh."""

from __future__ import annotations

import shutil
import sqlite3
from datetime import date
from pathlib import Path

import duckdb
import pytest

from esdc.corpus.mirror import (
    create_views,
    refresh_all,
    refresh_documents,
    refresh_registry,
    sweep_orphan_chunks,
)
from esdc.knowledge.store import Edge, KnowledgeStore
from esdc.pod_registry.identity import ApprovalKey, IssuedIdentity, issue_identity
from esdc.pod_registry.migrations import initialize_registry_v2

_SQLITE_DOCS = """
CREATE TABLE documents (
    doc_id TEXT PRIMARY KEY, file_name TEXT NOT NULL, file_path TEXT NOT NULL,
    file_hash TEXT NOT NULL UNIQUE, doc_type TEXT, doc_topic TEXT,
    doc_number TEXT, doc_date TEXT, subject TEXT, sender TEXT, recipient TEXT,
    doc_level TEXT, wk_name TEXT, field_name TEXT, project_name TEXT,
    pod_name TEXT, suggested_pod_ids TEXT, raw_entities TEXT, metadata TEXT,
    markdown TEXT NOT NULL, extraction_method TEXT NOT NULL,
    embedding_model TEXT NOT NULL, page_count INTEGER,
    ingested_at TEXT DEFAULT (datetime('now'))
)
"""

# Minimal old-shaped fixture SQL: test material only. The store no longer carries
# a superseded-schema constant, so refusal fixtures declare the shape they want.
_OLD_SHAPED_FIXTURE_SQL = """
CREATE TABLE m_pod (
    id INTEGER PRIMARY KEY,
    pod_id TEXT NOT NULL UNIQUE,
    pod_name TEXT,
    approval_seq INTEGER
);
CREATE TABLE project_pod (
    pod_id INTEGER NOT NULL REFERENCES m_pod(id),
    project_id TEXT NOT NULL,
    UNIQUE (pod_id, project_id)
);
CREATE TABLE pod_revision (
    successor_id TEXT NOT NULL,
    predecessor_id TEXT NOT NULL,
    UNIQUE (successor_id, predecessor_id)
);
"""


def _make_truth(path: Path, rows: list[dict]) -> None:
    conn = sqlite3.connect(path)
    conn.execute(_SQLITE_DOCS)
    for r in rows:
        conn.execute(
            "INSERT INTO documents (doc_id, file_name, file_path, file_hash, "
            "doc_type, doc_topic, doc_date, subject, wk_name, field_name, "
            "project_name, markdown, extraction_method, embedding_model, "
            "ingested_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                r["doc_id"],
                r["doc_id"] + ".pdf",
                "/tmp/" + r["doc_id"],
                "hash-" + r["doc_id"],
                r.get("doc_type", "surat"),
                r.get("doc_topic", '["pod"]'),
                r.get("doc_date", "2025-03-01"),
                r.get("subject", "s"),
                r.get("wk_name", '["Rokan"]'),
                r.get("field_name", '["Duri"]'),
                r.get("project_name", '["P1"]'),
                "# body",
                "docling",
                "qwen3",
                "2025-03-01 00:00:00",
            ),
        )
    conn.commit()
    conn.close()


def _latest_truth(path: Path, doc_id: str) -> tuple[Path, IssuedIdentity]:
    """Seed the canonical source: Jan 1 approval, order 1, and its document link."""
    _make_truth(path, [{"doc_id": doc_id}])
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA foreign_keys = ON")
    initialize_registry_v2(conn)
    conn.execute("BEGIN IMMEDIATE")
    identity = issue_identity(
        conn,
        key=ApprovalKey(date(2024, 1, 1), 1, 1, 1, 0),
        actor="FA:test-mirror",
    )
    conn.execute(
        "INSERT INTO m_pod (pod_id_seq, pod_uuid, approval_date, institution_id, "
        "pod_type_id, rev_num, pod_order_no, pod_id, pod_name, pod_letter_num, "
        "is_latest) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            identity.pod_id_seq,
            str(identity.pod_uuid),
            "2024-01-01",
            1,
            1,
            0,
            identity.pod_order_no,
            identity.pod_id,
            "Canonical POD",
            "LETTER-1",
            1,
        ),
    )
    conn.execute(
        "INSERT INTO pod_document (pod_id_seq, doc_id) VALUES (?, ?)",
        (identity.pod_id_seq, doc_id),
    )
    conn.execute(
        "UPDATE pod_registry_state SET source_generation = 3 WHERE singleton_id = 1"
    )
    conn.commit()
    conn.close()
    return path, identity


def test_refresh_all_publishes_v2_links_without_recreating_legacy_schema(
    tmp_path: Path,
):
    path = tmp_path / "v2-truth.sqlite"
    _make_truth(path, [{"doc_id": "d1"}])
    source = sqlite3.connect(path)
    source.row_factory = sqlite3.Row
    source.execute("PRAGMA foreign_keys=ON")
    initialize_registry_v2(source)
    source.execute(
        "INSERT INTO m_project (project_id, project_name, is_active) "
        "VALUES ('P-0000001-01', 'Project One', 1)"
    )
    source.commit()
    source.execute("BEGIN IMMEDIATE")
    identity = issue_identity(
        source,
        key=ApprovalKey(date(2024, 1, 1), 1, 1, 1, 0),
        actor="FA:test-corpus-sync",
    )
    source.execute(
        "INSERT INTO m_pod (pod_id_seq, pod_uuid, approval_date, institution_id, "
        "pod_type_id, rev_num, pod_order_no, pod_id, pod_name, pod_letter_num, "
        "is_latest) VALUES (?, ?, ?, 1, 1, 0, ?, ?, ?, 'LETTER-1', 1)",
        (
            identity.pod_id_seq,
            str(identity.pod_uuid),
            "2024-01-01",
            identity.pod_order_no,
            identity.pod_id,
            "Project One POD",
        ),
    )
    source.execute(
        "INSERT INTO pod_document (pod_id_seq, doc_id) VALUES (?, 'd1')",
        (identity.pod_id_seq,),
    )
    source.commit()
    store = KnowledgeStore(source, allow_stale=True)
    store.ensure_tables()
    store.upsert_edges(
        [Edge("document", "d1", "ABOUT_POD", "pod", "stale", method="llm_extracted")]
    )
    source.execute(
        "UPDATE pod_registry_state SET knowledge_generation=NULL WHERE singleton_id=1"
    )
    source.commit()
    source.close()

    conn = duckdb.connect()
    try:
        report = refresh_all(conn, path)
        assert report.registry["pod_document"] == 1
        assert conn.execute(
            "SELECT pod_id_seq, doc_id FROM pod_document"
        ).fetchall() == [(identity.pod_id_seq, "d1")]
        assert conn.execute(
            "SELECT pod_id FROM v_doc_pod_link WHERE doc_id='d1'"
        ).fetchall() == [(identity.pod_id,)]
        columns = {
            row[0]
            for row in conn.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_schema='main' AND table_name='pod_project'"
            ).fetchall()
        }
        assert "pod_id_seq" in columns
        assert "pod_id" not in columns
        assert conn.execute("SELECT COUNT(*) FROM kg_edge").fetchone() == (0,)
    finally:
        conn.close()


@pytest.fixture
def truth_path(tmp_path: Path) -> Path:
    path = tmp_path / "truth.sqlite"
    _make_truth(path, [{"doc_id": "d1"}, {"doc_id": "d2", "doc_date": "2026-07-01"}])
    return path


def test_refresh_documents_copies_rows_with_faithful_types(truth_path: Path):
    conn = duckdb.connect()

    copied = refresh_documents(conn, truth_path)

    assert copied == 2
    types = dict(
        conn.execute(
            "SELECT column_name, column_type FROM (DESCRIBE documents)"
        ).fetchall()
    )
    assert types["doc_date"] == "DATE"
    assert types["ingested_at"] == "TIMESTAMP"
    assert types["doc_topic"] == "JSON"
    assert types["field_name"] == "JSON"
    # the year filter used by _build_filter_clause must bind
    row = conn.execute(
        "SELECT COUNT(*) FROM documents WHERE EXTRACT(year FROM doc_date) = 2026"
    ).fetchone()
    assert row is not None
    assert row[0] == 1


def test_refresh_documents_wraps_legacy_bare_entity_names(tmp_path: Path):
    path = tmp_path / "legacy.sqlite"
    _make_truth(path, [{"doc_id": "d1", "field_name": "Duri"}])  # bare text, not JSON
    conn = duckdb.connect()

    refresh_documents(conn, path)

    row = conn.execute(
        "SELECT COUNT(*) FROM documents WHERE EXISTS ("
        "  SELECT 1 FROM json_each(documents.field_name) "
        "  WHERE CAST(value AS VARCHAR) ILIKE '%duri%')"
    ).fetchone()
    assert row is not None
    assert row[0] == 1


def test_refresh_documents_survives_a_malformed_doc_date(tmp_path, caplog):
    """One bad date must not fail the whole refresh (TRY_CAST, not CAST)."""
    path = tmp_path / "malformed.sqlite"
    _make_truth(
        path,
        [
            {"doc_id": "d1", "doc_date": "2025-03-01"},
            {"doc_id": "d2", "doc_date": "bukan tanggal"},  # malformed, non-ISO
            {"doc_id": "d3", "doc_date": "2026-07-01"},
        ],
    )
    conn = duckdb.connect()

    with caplog.at_level("WARNING"):
        copied = refresh_documents(conn, path)

    assert copied == 3  # the bad row is not dropped, only its doc_date is nulled
    rows = dict(conn.execute("SELECT doc_id, doc_date FROM documents").fetchall())
    assert rows["d1"] == date(2025, 3, 1)
    assert rows["d2"] is None
    assert rows["d3"] == date(2026, 7, 1)
    assert any(
        "doc_date" in r.message and "count=1" in r.message for r in caplog.records
    )


def test_refresh_documents_survives_apostrophe_in_truth_path(tmp_path: Path):
    """refresh_documents tolerates an apostrophe in the truth DB's path.

    A truth DB living under a directory containing an apostrophe (an
    ordinary macOS home directory, e.g. /Users/O'Brien/.esdc) must not break
    ATTACH. Unescaped interpolation of the path into the ATTACH statement
    would raise a DuckDB syntax/parser error here.
    """
    quirky_dir = tmp_path / "O'Brien"
    quirky_dir.mkdir()
    path = quirky_dir / "truth.sqlite"
    _make_truth(path, [{"doc_id": "d1"}, {"doc_id": "d2", "doc_date": "2026-07-01"}])
    conn = duckdb.connect()

    copied = refresh_documents(conn, path)

    assert copied == 2
    rows = dict(conn.execute("SELECT doc_id, doc_date FROM documents").fetchall())
    assert set(rows) == {"d1", "d2"}
    assert rows["d2"] == date(2026, 7, 1)


def test_refresh_documents_on_empty_truth_table_returns_zero(tmp_path: Path):
    """Fresh-install path: refresh can run before anything is committed."""
    path = tmp_path / "empty.sqlite"
    sconn = sqlite3.connect(path)
    sconn.execute(_SQLITE_DOCS)
    sconn.commit()
    sconn.close()
    conn = duckdb.connect()

    copied = refresh_documents(conn, path)

    assert copied == 0
    types = dict(
        conn.execute(
            "SELECT column_name, column_type FROM (DESCRIBE documents)"
        ).fetchall()
    )
    assert types["doc_date"] == "DATE"
    assert types["ingested_at"] == "TIMESTAMP"
    assert types["doc_topic"] == "JSON"
    assert types["field_name"] == "JSON"


def test_sweep_orphan_chunks_removes_chunks_of_deleted_documents(truth_path: Path):
    conn = duckdb.connect()
    refresh_documents(conn, truth_path)
    conn.execute(
        "CREATE TABLE document_chunks (chunk_id VARCHAR, doc_id VARCHAR, "
        "chunk_index INTEGER, section VARCHAR, chunk_text TEXT, embed_text TEXT)"
    )
    conn.execute(
        "INSERT INTO document_chunks VALUES "
        "('d1:0000','d1',0,NULL,'t','t'), ('gone:0000','gone',0,NULL,'t','t')"
    )

    deleted = sweep_orphan_chunks(conn)

    assert deleted == 1
    count_row = conn.execute("SELECT COUNT(*) FROM document_chunks").fetchone()
    assert count_row is not None
    assert count_row[0] == 1
    doc_row = conn.execute("SELECT doc_id FROM document_chunks").fetchone()
    assert doc_row is not None
    assert doc_row[0] == "d1"


def test_sweep_orphan_chunks_survives_a_null_doc_id_in_documents(truth_path: Path):
    """sweep_orphan_chunks still removes orphans when documents has a NULL doc_id.

    NOT IN against a subquery containing a NULL is UNKNOWN for every row,
    so it would silently delete nothing. NOT EXISTS must not have that
    landmine: the orphan is still removed and the non-orphan survives.
    """
    conn = duckdb.connect()
    refresh_documents(conn, truth_path)
    conn.execute("INSERT INTO documents (doc_id) VALUES (NULL)")
    conn.execute(
        "CREATE TABLE document_chunks (chunk_id VARCHAR, doc_id VARCHAR, "
        "chunk_index INTEGER, section VARCHAR, chunk_text TEXT, embed_text TEXT)"
    )
    conn.execute(
        "INSERT INTO document_chunks VALUES "
        "('d1:0000','d1',0,NULL,'t','t'), ('gone:0000','gone',0,NULL,'t','t')"
    )

    deleted = sweep_orphan_chunks(conn)

    assert deleted == 1
    count_row = conn.execute("SELECT COUNT(*) FROM document_chunks").fetchone()
    assert count_row is not None
    assert count_row[0] == 1
    doc_row = conn.execute("SELECT doc_id FROM document_chunks").fetchone()
    assert doc_row is not None
    assert doc_row[0] == "d1"


def test_sweep_orphan_chunks_on_fresh_install_returns_zero(truth_path: Path):
    """document_chunks does not exist yet on a fresh install; must not raise."""
    conn = duckdb.connect()
    refresh_documents(conn, truth_path)

    assert sweep_orphan_chunks(conn) == 0


def test_refresh_registry_copies_present_tables_and_skips_absent(tmp_path: Path):
    path = tmp_path / "reg.sqlite"
    conn_s = sqlite3.connect(path)
    conn_s.execute("CREATE TABLE kg_edge (src_id TEXT, rel TEXT, dst_id TEXT)")
    conn_s.execute("INSERT INTO kg_edge VALUES ('POD I Duri', 'ABOUT_POD', 'POD-1')")
    conn_s.commit()
    conn_s.close()
    # kg_claim deliberately absent — learn has never run
    conn = duckdb.connect()

    copied = refresh_registry(conn, path)

    assert copied == {"kg_edge": 1}
    assert "kg_claim" not in copied
    row = conn.execute("SELECT dst_id FROM kg_edge").fetchone()
    assert row is not None
    assert row[0] == "POD-1"


def test_refresh_registry_distinguishes_empty_present_table_from_absent(tmp_path: Path):
    """refresh_registry distinguishes an empty-but-present table from an absent one.

    A registry table that exists but holds zero rows (kg_edge right
    after `esdc corpus learn` creates its schema but before anything has
    been written) must still be copied and reported with count 0 —
    distinct from a table that does not exist at all (kg_claim, before
    `learn` has ever run), which must be skipped and must not appear as a
    key in the returned dict at all.
    """
    path = tmp_path / "reg_empty.sqlite"
    conn_s = sqlite3.connect(path)
    conn_s.execute("CREATE TABLE kg_edge (src_id TEXT, rel TEXT, dst_id TEXT)")
    # kg_edge intentionally left empty: the table exists, no rows yet.
    conn_s.commit()
    conn_s.close()
    # kg_claim deliberately absent
    conn = duckdb.connect()

    copied = refresh_registry(conn, path)

    assert copied == {"kg_edge": 0}
    assert "kg_claim" not in copied
    # not just the dict says 0 — the table must actually exist in DuckDB
    row = conn.execute("SELECT COUNT(*) FROM kg_edge").fetchone()
    assert row is not None
    assert row[0] == 0


def test_refresh_registry_drops_mirror_table_when_truth_table_disappears(
    tmp_path: Path,
):
    """refresh_registry drops the mirror table when its truth table disappears.

    A registry table present in one refresh but absent from the truth in
    a later refresh (older esdc.sqlite restored from backup, truth file
    swapped, KG state reset) must not leave stale rows sitting in DuckDB
    forever. The mirror table itself must be dropped, not just skipped.
    """
    path = tmp_path / "reg_disappear.sqlite"
    conn_s = sqlite3.connect(path)
    conn_s.execute("CREATE TABLE kg_edge (src_id TEXT, rel TEXT, dst_id TEXT)")
    conn_s.execute("INSERT INTO kg_edge VALUES ('POD I Duri', 'ABOUT_POD', 'POD-1')")
    conn_s.commit()
    conn_s.close()
    conn = duckdb.connect()

    first = refresh_registry(conn, path)

    assert first["kg_edge"] == 1
    row = conn.execute("SELECT COUNT(*) FROM kg_edge").fetchone()
    assert row is not None
    assert row[0] == 1

    conn_s = sqlite3.connect(path)
    conn_s.execute("DROP TABLE kg_edge")
    conn_s.commit()
    conn_s.close()

    second = refresh_registry(conn, path)

    assert "kg_edge" not in second
    with pytest.raises(duckdb.CatalogException):
        conn.execute("SELECT COUNT(*) FROM kg_edge")


def test_refresh_registry_stays_silent_when_table_was_never_mirrored(
    tmp_path: Path, caplog
):
    """refresh_registry stays silent about a table that was never mirrored.

    kg_edge/kg_claim have never existed in either store on most installs
    (`esdc corpus learn` has never run) — that is the state of the live
    database today. Refreshing must not log a "dropped" line for a table
    that was never there to drop, on this refresh or any subsequent one,
    since it would read as continuous data loss on every commit/learn/sync.
    """
    path = tmp_path / "reg_never_mirrored.sqlite"
    conn_s = sqlite3.connect(path)
    conn_s.execute("CREATE TABLE kg_edge (src_id TEXT, rel TEXT, dst_id TEXT)")
    conn_s.execute("INSERT INTO kg_edge VALUES ('POD I Duri', 'ABOUT_POD', 'POD-1')")
    conn_s.commit()
    conn_s.close()
    conn = duckdb.connect()

    with caplog.at_level("INFO"):
        refresh_registry(conn, path)
        second = refresh_registry(conn, path)  # steady state, run again

    assert "kg_claim" not in second
    assert not any("dropped" in r.message for r in caplog.records)


def test_refresh_registry_logs_when_a_stale_mirror_is_actually_dropped(
    tmp_path: Path, caplog
):
    """refresh_registry logs when it actually drops a stale mirror table.

    A table that WAS mirrored and then disappears from the truth is a
    genuine state change — removing mirrored rows — and must still log at
    info, unlike the never-mirrored case above.
    """
    path = tmp_path / "reg_real_drop.sqlite"
    conn_s = sqlite3.connect(path)
    conn_s.execute("CREATE TABLE kg_edge (src_id TEXT, rel TEXT, dst_id TEXT)")
    conn_s.execute("INSERT INTO kg_edge VALUES ('POD I Duri', 'ABOUT_POD', 'POD-1')")
    conn_s.commit()
    conn_s.close()
    conn = duckdb.connect()
    refresh_registry(conn, path)

    conn_s = sqlite3.connect(path)
    conn_s.execute("DROP TABLE kg_edge")
    conn_s.commit()
    conn_s.close()

    with caplog.at_level("INFO"):
        second = refresh_registry(conn, path)

    assert "kg_edge" not in second
    assert any(
        "dropped" in r.message and "kg_edge" in r.message for r in caplog.records
    )


def test_refresh_registry_copies_the_knowledge_graph_and_leaves_pod_tables_to_the_publisher(
    tmp_path: Path, caplog
):
    """The mirror copies the learned knowledge graph; the publisher owns the POD tables.

    `m_pod`, `r_institution`, `r_pod_type`, and `pod_document` are real SQLite
    operational tables — present in the truth on every production install — but
    their DuckDB shapes belong to the canonical publisher. `refresh_registry`
    must not raw-copy any of them even though they exist and hold rows in the
    truth, and it must not drop them either: retirement is the publisher's,
    inside the transaction that also republishes them.
    """
    path, identity = _latest_truth(tmp_path / "canonical.sqlite", "d1")
    source = sqlite3.connect(path)
    source.execute("CREATE TABLE kg_edge (edge_id INTEGER, predicate TEXT)")
    source.execute("INSERT INTO kg_edge VALUES (1, 'relates_to')")
    source.commit()
    source.close()

    conn = duckdb.connect()
    refresh_all(conn, path)
    projection_before = conn.execute("SELECT pod_id FROM m_pod").fetchall()

    with caplog.at_level("INFO"):
        copied = refresh_registry(conn, path)

    # The knowledge graph is still mirrored; none of the POD tables are.
    assert copied == {"kg_edge": 1}
    assert conn.execute("SELECT pod_id FROM m_pod").fetchall() == projection_before
    assert conn.execute("SELECT pod_id_seq, doc_id FROM pod_document").fetchall() == [
        (identity.pod_id_seq, "d1")
    ]
    assert not any("dropped" in r.message for r in caplog.records)
    conn.close()


def test_the_publisher_not_the_mirror_retires_a_raw_pod_mirror_inside_its_transaction(
    tmp_path: Path, caplog
):
    """Retiring a raw POD mirror belongs to the publisher, not to the mirror.

    A database left by an older build has raw copies of the SQLite operational
    tables sitting in DuckDB under the published names, including a
    `pod_document` keyed by the BIGINT surrogate instead of the contract's
    `pod_id_seq`. `refresh_registry` must leave that copy alone — dropping
    POD-owned tables outside the publisher's transaction is exactly what let a
    refused source destroy a projection it could not rebuild. The publisher is
    what replaces them, and only for a source that satisfies the contract.
    """
    path, identity = _latest_truth(tmp_path / "canonical.sqlite", "d1")
    conn = duckdb.connect()
    # Simulate exactly what the pre-fix refresh left behind.
    conn.execute("CREATE TABLE pod_document (pod_id BIGINT, doc_id VARCHAR)")
    conn.execute("INSERT INTO pod_document VALUES (1, 'd1')")

    with caplog.at_level("INFO"):
        copied = refresh_registry(conn, path)

    assert copied == {}
    assert conn.execute("SELECT pod_id, doc_id FROM pod_document").fetchall() == [(1, "d1")]
    assert not any("dropped" in r.message for r in caplog.records)

    refresh_all(conn, path)
    columns = {
        row[0]
        for row in conn.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_schema='main' AND table_name='pod_document'"
        ).fetchall()
    }
    assert "pod_id_seq" in columns and "pod_id" not in columns
    assert conn.execute("SELECT pod_id_seq, doc_id FROM pod_document").fetchall() == [
        (identity.pod_id_seq, "d1")
    ]
    conn.close()



def test_views_expose_both_grains(tmp_path: Path):
    """create_views' v_doc_pod_link/v_document expose both the link and document grain.

    create_views joins the shapes the canonical publisher produces: `pod_document`
    keyed on the contract's `pod_id_seq`, and `pod_registry` carrying the canonical
    string `pod_id`. There is no join left for an older column shape, so the grains
    are only exposed when the published shapes are the ones the contract documents.
    """
    path = tmp_path / "both.sqlite"
    _make_truth(path, [{"doc_id": "d1"}])
    conn = duckdb.connect()
    refresh_documents(conn, path)
    conn.execute(
        "CREATE TABLE pod_registry (pod_id_seq BIGINT, pod_id VARCHAR, pod_name VARCHAR)"
    )
    conn.executemany(
        "INSERT INTO pod_registry VALUES (?, ?, ?)",
        [(1, "POD-1", "A"), (2, "POD-2", "B")],
    )
    conn.execute("CREATE TABLE pod_document (pod_id_seq BIGINT, doc_id VARCHAR)")
    conn.executemany(
        "INSERT INTO pod_document VALUES (?, ?)", [(1, "d1"), (2, "d1")]
    )

    created = create_views(conn)

    assert set(created) == {"v_doc_pod_link", "v_document"}
    # fan-out grain: one row per (document, pod), resolved through pod_id_seq
    link_rows = conn.execute(
        "SELECT doc_id, pod_id FROM v_doc_pod_link ORDER BY pod_id"
    ).fetchall()
    assert link_rows == [("d1", "POD-1"), ("d1", "POD-2")]
    # document grain: one row per document, links aggregated
    doc_count = conn.execute("SELECT COUNT(*) FROM v_document").fetchone()
    assert doc_count is not None
    assert doc_count[0] == 1
    linked = conn.execute("SELECT linked_pod_ids FROM v_document").fetchone()
    assert linked is not None
    assert sorted(linked[0]) == ["POD-1", "POD-2"]
    conn.close()


def test_views_degrade_when_registry_absent(truth_path: Path):
    conn = duckdb.connect()
    refresh_documents(conn, truth_path)

    created = create_views(conn)

    assert created == ["v_document"]
    count_row = conn.execute("SELECT COUNT(*) FROM v_document").fetchone()
    assert count_row is not None
    assert count_row[0] == 2
    linked_row = conn.execute(
        "SELECT linked_pod_ids FROM v_document LIMIT 1"
    ).fetchone()
    assert linked_row is not None
    assert linked_row[0] == []


def test_views_degrade_when_only_one_pod_table_exists(tmp_path: Path):
    """create_views needs BOTH pod_registry and pod_document for the link view.

    Regression test. create_views used to treat either table's presence
    (COUNT >= 1) as enough and build v_doc_pod_link from a half-published
    registry, which made the JOIN dangle. Exactly one table existing must
    degrade to v_document only — with the stale link view dropped, never
    attempted — just like when neither table exists.
    """
    path = tmp_path / "half.sqlite"
    _make_truth(path, [{"doc_id": "d1"}])
    conn = duckdb.connect()
    refresh_documents(conn, path)
    conn.execute("CREATE TABLE pod_registry (pod_id VARCHAR, pod_name VARCHAR)")
    conn.execute("INSERT INTO pod_registry VALUES ('POD-1', 'A')")
    # pod_document deliberately absent — publish produced only one side.

    created = create_views(conn)

    assert created == ["v_document"]
    count_row = conn.execute("SELECT COUNT(*) FROM v_document").fetchone()
    assert count_row is not None
    assert count_row[0] == 1
    linked_row = conn.execute(
        "SELECT linked_pod_ids FROM v_document LIMIT 1"
    ).fetchone()
    assert linked_row is not None
    assert linked_row[0] == []
    with pytest.raises(duckdb.CatalogException):
        conn.execute("SELECT * FROM v_doc_pod_link")


def test_views_degrade_when_the_pod_pair_is_only_in_an_attached_catalog(
    tmp_path: Path,
) -> None:
    """The presence query must read the current catalog, not attached ones.

    `create_views` decides the link grain from the shapes published in the database it is
    writing to, and its unqualified names resolve there. A pair that exists only in an
    attached catalog cannot be joined, so counting it as the current pair makes the view
    dangle: the query raises instead of degrading. Only `documents` in the current catalog
    means `v_document` with an empty `linked_pod_ids`, exactly as when neither POD table
    exists.
    """
    path = tmp_path / "attached-only.sqlite"
    _make_truth(path, [{"doc_id": "d1"}])
    conn = duckdb.connect()
    refresh_documents(conn, path)
    conn.execute("ATTACH ':memory:' AS other")
    conn.execute("CREATE TABLE other.main.pod_document (pod_id_seq BIGINT, doc_id VARCHAR)")
    conn.execute(
        "CREATE VIEW other.main.pod_registry AS "
        "SELECT CAST(1 AS BIGINT) AS pod_id_seq, 'POD-1' AS pod_id"
    )

    created = create_views(conn)

    assert created == ["v_document"]
    count_row = conn.execute("SELECT COUNT(*) FROM v_document").fetchone()
    assert count_row is not None
    assert count_row[0] == 1
    linked_row = conn.execute("SELECT linked_pod_ids FROM v_document").fetchone()
    assert linked_row is not None
    assert linked_row[0] == []
    # The link view was never created in the current catalog.
    current_names = {
        row[0]
        for row in conn.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_catalog = current_database() AND table_schema = 'main'"
        ).fetchall()
    }
    assert "v_doc_pod_link" not in current_names
    assert {"pod_document", "pod_registry"} & current_names == set()
    conn.close()


def test_views_degrade_when_only_one_side_of_the_pair_is_in_the_current_catalog(
    tmp_path: Path,
) -> None:
    """One current-catalog table plus one attached table is not a current pair.

    The two sides have to be joinable in the database being written to. A mixed pair is a
    half-published registry from this catalog's point of view, so it degrades to
    `v_document` with empty links and the stale link view is dropped rather than built on
    a name that cannot resolve.
    """
    path = tmp_path / "mixed-catalog.sqlite"
    _make_truth(path, [{"doc_id": "d1"}])
    conn = duckdb.connect()
    refresh_documents(conn, path)
    conn.execute("CREATE TABLE pod_document (pod_id_seq BIGINT, doc_id VARCHAR)")
    conn.execute("INSERT INTO pod_document VALUES (1, 'd1')")
    conn.execute("ATTACH ':memory:' AS other")
    conn.execute(
        "CREATE VIEW other.main.pod_registry AS "
        "SELECT CAST(1 AS BIGINT) AS pod_id_seq, 'POD-1' AS pod_id"
    )

    created = create_views(conn)

    assert created == ["v_document"]
    linked_row = conn.execute("SELECT linked_pod_ids FROM v_document").fetchone()
    assert linked_row is not None
    assert linked_row[0] == []
    with pytest.raises(duckdb.CatalogException):
        conn.execute("SELECT * FROM v_doc_pod_link")
    conn.close()


def test_refresh_all_replaces_a_raw_pod_mirror_with_the_canonical_shape(tmp_path: Path):
    """refresh_all republishes the canonical shape over a raw surrogate mirror.

    Regression test. A pre-fix build raw-mirrored `pod_document` with the
    BIGINT `pod_id` surrogate foreign key on top of the published
    agent-documented shape. Under the single current schema the publisher owns
    that table, so a full refresh_all must leave `pod_document` holding
    `pod_id_seq`, expose the canonical string `pod_id` through `pod_registry`,
    and resolve the document link to that canonical id.
    """
    path, identity = _latest_truth(tmp_path / "canonical.sqlite", "d1")
    conn = duckdb.connect()
    # Simulate the raw mirror a pre-fix build left under the published name.
    conn.execute("CREATE TABLE pod_document (pod_id BIGINT, doc_id VARCHAR)")
    conn.execute("INSERT INTO pod_document VALUES (1, 'd1')")

    report = refresh_all(conn, path)

    columns = {
        row[0]
        for row in conn.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_schema='main' AND table_name='pod_document'"
        ).fetchall()
    }
    assert "pod_id_seq" in columns
    assert "pod_id" not in columns
    assert conn.execute("SELECT pod_id_seq, doc_id FROM pod_document").fetchall() == [
        (identity.pod_id_seq, "d1")
    ]
    assert conn.execute("SELECT pod_id FROM pod_registry").fetchall() == [
        (identity.pod_id,)
    ]
    assert report.registry["pod_document"] == 1


def test_refresh_all_repairs_a_pre_fix_database(tmp_path: Path):
    """refresh_all fully repairs a pre-fix database's stale raw POD mirrors.

    End-to-end repair check for the state a database built by the older build
    is in: raw copies of the SQLite operational tables sitting in DuckDB,
    including a BIGINT-keyed `pod_document` and a `v_doc_pod_link` built by
    joining those raw copies. One refresh_all must retire the raw copies,
    publish the current contract shapes from the truth, and rebuild both views
    against them so neither dangles nor keeps serving surrogate-keyed rows.
    """
    path, identity = _latest_truth(tmp_path / "canonical.sqlite", "d1")
    conn = duckdb.connect()
    refresh_documents(conn, path)
    # Simulate exactly what the pre-fix build left in DuckDB: raw mirrors
    # plus the view create_views used to build from them.
    conn.execute("CREATE TABLE m_pod (id INTEGER, pod_id VARCHAR, pod_name VARCHAR)")
    conn.execute("INSERT INTO m_pod VALUES (1, 'PL-2019-0001-2-2-0', 'Duri POD I')")
    conn.execute("CREATE TABLE r_institution (code INTEGER, institution VARCHAR)")
    conn.execute("INSERT INTO r_institution VALUES (9, 'Stale reference row')")
    conn.execute("CREATE TABLE project_pod (pod_id INTEGER, project_id VARCHAR)")
    conn.execute("INSERT INTO project_pod VALUES (1, 'PRJ-1')")
    conn.execute("CREATE TABLE pod_document (pod_id BIGINT, doc_id VARCHAR)")
    conn.execute("INSERT INTO pod_document VALUES (1, 'd1')")
    conn.execute("""
        CREATE VIEW v_doc_pod_link AS
        SELECT d.doc_id, p.pod_id, p.pod_name FROM documents d
        JOIN pod_document pd ON pd.doc_id = d.doc_id
        JOIN m_pod p ON p.id = pd.pod_id
    """)

    refresh_all(conn, path)

    # The raw surrogate shape is replaced by the contract shape built from the
    # truth, and the seeded reference rows replace the stale hand-written ones.
    assert conn.execute(
        "SELECT pod_id FROM m_pod ORDER BY pod_id"
    ).fetchall() == [(identity.pod_id,)]
    institutions = {
        row[0] for row in conn.execute("SELECT institution FROM r_institution").fetchall()
    }
    assert "Stale reference row" not in institutions
    assert "MESDM" in institutions
    assert conn.execute(
        "SELECT institution FROM r_institution WHERE institution = 'Stale reference row'"
    ).fetchall() == []
    assert conn.execute(
        "SELECT i.institution FROM m_pod p "
        "JOIN r_institution i ON i.id = p.institution_id"
    ).fetchall() == [("MESDM",)]
    assert conn.execute("SELECT doc_id, pod_id FROM v_doc_pod_link").fetchall() == [
        ("d1", identity.pod_id)
    ]
    linked_row = conn.execute("SELECT linked_pod_ids FROM v_document").fetchone()
    assert linked_row is not None
    assert linked_row[0] == [identity.pod_id]


def test_refresh_all_does_not_recreate_retired_pod_objects(tmp_path: Path):
    """One refresh_all publishes the current masters, not the superseded model.

    Stale objects left by the superseded POD model (revision registry,
    value-case table, economics views) are retired by the publisher; the
    documents mirror, the permanent identity ledger, and unrelated DuckDB
    tables are not touched.
    """
    path, identity = _latest_truth(tmp_path / "canonical.sqlite", "d1")
    conn = duckdb.connect()
    conn.execute("CREATE TABLE pod_revision (successor_id VARCHAR, predecessor_id VARCHAR)")
    conn.execute("INSERT INTO pod_revision VALUES ('X', 'Y')")
    conn.execute("CREATE TABLE pod_value_case (pod_id VARCHAR)")
    conn.execute("INSERT INTO pod_value_case VALUES ('X')")
    conn.execute("CREATE VIEW pod_economics AS SELECT 'stale' AS pod_id")
    conn.execute("CREATE TABLE unrelated_duck_table (value INTEGER)")
    conn.execute("INSERT INTO unrelated_duck_table VALUES (42)")

    report = refresh_all(conn, path)

    names = {
        row[0]
        for row in conn.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema='main'"
        ).fetchall()
    }
    assert not {
        "pod_economics",
        "pod_plan",
        "pod_project_economics",
        "pod_revision",
        "pod_value_case",
    } & names
    assert conn.execute("SELECT value FROM unrelated_duck_table").fetchall() == [
        (42,)
    ]
    assert conn.execute("SELECT COUNT(*) FROM _pod_identity_ledger").fetchone() == (1,)
    assert conn.execute(
        "SELECT pod_id_seq, doc_id FROM pod_document"
    ).fetchall() == [(identity.pod_id_seq, "d1")]
    assert report.documents == 1


def test_refresh_all_refuses_an_old_shaped_source_without_inventing_state(
    tmp_path: Path,
) -> None:
    """A source shaped for the superseded schema fails instead of falling back.

    The mirror must not invent an empty business source or serve the retired
    projection: the failure is reported, and everything already in DuckDB that
    the publisher did not own stays as it was.
    """
    path = tmp_path / "old-shaped-truth.sqlite"
    _make_truth(path, [{"doc_id": "d1"}])
    legacy = sqlite3.connect(path)
    legacy.executescript(_OLD_SHAPED_FIXTURE_SQL)
    legacy.commit()
    legacy.close()

    conn = duckdb.connect()
    refresh_documents(conn, path)
    conn.execute("CREATE VIEW pod_economics AS SELECT 'stale' AS pod_id")
    conn.execute("CREATE TABLE unrelated_duck_table (value INTEGER)")
    conn.execute("INSERT INTO unrelated_duck_table VALUES (42)")

    with pytest.raises(RuntimeError, match="missing tables"):
        refresh_all(conn, path)

    assert conn.execute("SELECT COUNT(*) FROM documents").fetchone() == (1,)
    assert conn.execute("SELECT value FROM unrelated_duck_table").fetchall() == [
        (42,)
    ]
    assert conn.execute("SELECT pod_id FROM pod_economics").fetchall() == [("stale",)]
    names = {
        row[0]
        for row in conn.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema='main'"
        ).fetchall()
    }
    assert not {"m_pod", "pod_registry", "pod_project", "pod_monitoring"} & names


def test_refresh_all_preserves_the_projection_when_an_invalid_source_is_refused(
    tmp_path: Path,
) -> None:
    """A refused source must not cost the mirror the projection it already had.

    The preservation must be real, not a publisher-only guarantee: the generic
    documents/KG refresh runs before the canonical publisher, so the read-only
    contract preflight has to happen before any caller-visible DuckDB mutation.
    A source with the corpus documents schema and an old-shaped `m_pod` is
    refused, and the previously published canonical projection stays as it was.
    """
    valid, identity = _latest_truth(tmp_path / "valid.sqlite", "d1")
    conn = duckdb.connect()
    refresh_all(conn, valid)
    projection_before = conn.execute(
        "SELECT pod_id, pod_name FROM m_pod ORDER BY pod_id"
    ).fetchall()
    documents_before = conn.execute(
        "SELECT doc_id FROM documents ORDER BY doc_id"
    ).fetchall()
    links_before = conn.execute(
        "SELECT pod_id_seq, doc_id FROM pod_document ORDER BY doc_id"
    ).fetchall()

    invalid = tmp_path / "invalid.sqlite"
    _make_truth(invalid, [{"doc_id": "d9"}])
    stale = sqlite3.connect(invalid)
    stale.executescript(_OLD_SHAPED_FIXTURE_SQL)
    stale.commit()
    source_tables_before = sorted(
        row[0]
        for row in stale.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    )
    stale.close()

    with pytest.raises(RuntimeError, match="missing tables") as excinfo:
        refresh_all(conn, invalid)

    assert "pod_project" in str(excinfo.value)
    assert conn.execute("SELECT pod_id, pod_name FROM m_pod ORDER BY pod_id").fetchall() == (
        projection_before
    )
    assert conn.execute("SELECT doc_id FROM documents ORDER BY doc_id").fetchall() == (
        documents_before
    )
    assert conn.execute(
        "SELECT pod_id_seq, doc_id FROM pod_document ORDER BY doc_id"
    ).fetchall() == links_before
    assert conn.execute("SELECT pod_id FROM pod_registry").fetchall() == [(identity.pod_id,)]

    # The preflight is read-only: it refuses the source without creating,
    # initializing, or upgrading it, so the source is left exactly as it was.
    check = sqlite3.connect(invalid)
    try:
        assert sorted(
            row[0]
            for row in check.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        ) == source_tables_before
        assert check.execute("PRAGMA journal_mode").fetchone()[0] != "wal"
    finally:
        check.close()
    conn.close()


def test_refresh_all_refuses_a_wrong_shape_source_before_mutating_duckdb(
    tmp_path: Path,
) -> None:
    """Every required table name present is not the contract, and order is what protects.

    A copy of a source that publishes keeps all the required table names but replaces
    one contract table with a wrong column, so a name-only check lets the mirror mutate
    DuckDB and then the publisher refuses the same file. The mirror must refuse it before
    its own documents and knowledge-graph work: the projection from the previous
    successful refresh, the raw mirrors, and the metadata all survive a source that can
    never publish. The copy also carries a new document row, so the assertions below show
    the refusal really happened before `refresh_documents` ran.
    """
    path, identity = _latest_truth(tmp_path / "canonical.sqlite", "d1")
    conn = duckdb.connect()
    refresh_all(conn, path)
    projection_before = conn.execute(
        "SELECT pod_id, pod_name FROM m_pod ORDER BY pod_id"
    ).fetchall()
    documents_before = conn.execute("SELECT doc_id FROM documents ORDER BY doc_id").fetchall()
    links_before = conn.execute(
        "SELECT pod_id_seq, doc_id FROM pod_document ORDER BY doc_id"
    ).fetchall()

    malformed = tmp_path / "wrong-shape.sqlite"
    shutil.copyfile(path, malformed)
    writer = sqlite3.connect(malformed)
    writer.execute(
        "INSERT INTO documents (doc_id, file_name, file_path, file_hash, markdown, "
        "extraction_method, embedding_model) VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("d9", "d9.pdf", "/tmp/d9", "hash-d9", "# body", "docling", "qwen3"),
    )
    writer.execute("DROP TABLE m_project")
    writer.execute("CREATE TABLE m_project (wrong TEXT)")
    writer.commit()
    writer.close()

    before = sqlite3.connect(malformed)
    journal_mode_before = before.execute("PRAGMA journal_mode").fetchone()[0]
    before.close()

    with pytest.raises(RuntimeError, match="ordered POD contract columns") as excinfo:
        refresh_all(conn, malformed)

    assert "m_project" in str(excinfo.value)
    assert conn.execute("SELECT doc_id FROM documents ORDER BY doc_id").fetchall() == (
        documents_before
    )
    assert conn.execute("SELECT pod_id, pod_name FROM m_pod ORDER BY pod_id").fetchall() == (
        projection_before
    )
    assert conn.execute(
        "SELECT pod_id_seq, doc_id FROM pod_document ORDER BY doc_id"
    ).fetchall() == links_before
    assert conn.execute("SELECT pod_id FROM pod_registry").fetchall() == [(identity.pod_id,)]

    # The preflight only reads: the refused source keeps the shape, rows, and journal
    # mode the writer left, and its read snapshot is ended so a writer can still lock it.
    check = sqlite3.connect(malformed)
    try:
        assert check.execute("PRAGMA journal_mode").fetchone()[0] == journal_mode_before
        assert [row[1] for row in check.execute("PRAGMA table_info(m_project)").fetchall()] == [
            "wrong"
        ]
        assert sorted(
            row[0] for row in check.execute("SELECT doc_id FROM documents").fetchall()
        ) == ["d1", "d9"]
        check.execute("BEGIN IMMEDIATE")
        check.commit()
    finally:
        check.close()
    conn.close()


def test_refresh_all_keeps_the_projection_when_publication_fails_inside_its_transaction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A publisher failure after the drops rolls its whole transaction back.

    The canonical publisher is atomic: it retires and replaces the POD objects
    inside one transaction, so a failure part-way through cannot leave the
    mirror with a half-published read side. The failure is injected after the
    drops — inside the transaction — which is the only way this preservation is
    real rather than a side effect of never having started the publish.
    """
    from esdc.loaders import _METADATA_TABLE
    from esdc.pod_registry import publication

    path, identity = _latest_truth(tmp_path / "canonical.sqlite", "d1")
    conn = duckdb.connect()
    refresh_all(conn, path)
    projection_before = conn.execute(
        "SELECT pod_id, pod_name FROM m_pod ORDER BY pod_id"
    ).fetchall()
    links_before = conn.execute(
        "SELECT pod_id_seq, doc_id FROM pod_document ORDER BY doc_id"
    ).fetchall()
    metadata_before = conn.execute(
        f"SELECT table_name FROM {_METADATA_TABLE} WHERE table_name = 'm_pod'"
    ).fetchall()

    def fail_after_the_drops(*args, **kwargs):
        raise RuntimeError("injected canonical publication failure")

    monkeypatch.setattr(publication, "_create_table_from_sqlite", fail_after_the_drops)
    with pytest.raises(RuntimeError, match="injected canonical publication failure"):
        refresh_all(conn, path)

    assert conn.execute("SELECT pod_id, pod_name FROM m_pod ORDER BY pod_id").fetchall() == (
        projection_before
    )
    assert conn.execute(
        "SELECT pod_id_seq, doc_id FROM pod_document ORDER BY doc_id"
    ).fetchall() == links_before
    # The dropped compatibility view and the metadata rows it deleted come back
    # with the rollback, so the read side is not left half-published.
    assert conn.execute("SELECT pod_id FROM pod_registry").fetchall() == [(identity.pod_id,)]
    assert conn.execute(
        f"SELECT table_name FROM {_METADATA_TABLE} WHERE table_name = 'm_pod'"
    ).fetchall() == metadata_before
    conn.close()
