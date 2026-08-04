"""Tests for the SQLite -> DuckDB wholesale mirror refresh."""

from __future__ import annotations

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
                r["doc_id"], r["doc_id"] + ".pdf", "/tmp/" + r["doc_id"],
                "hash-" + r["doc_id"], r.get("doc_type", "surat"),
                r.get("doc_topic", '["pod"]'), r.get("doc_date", "2025-03-01"),
                r.get("subject", "s"), r.get("wk_name", '["Rokan"]'),
                r.get("field_name", '["Duri"]'), r.get("project_name", '["P1"]'),
                "# body", "docling", "qwen3", "2025-03-01 00:00:00",
            ),
        )
    conn.commit()
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
    types = dict(conn.execute("SELECT column_name, column_type FROM (DESCRIBE documents)").fetchall())
    assert types["doc_date"] == "DATE"
    assert types["ingested_at"] == "TIMESTAMP"
    assert types["doc_topic"] == "JSON"
    assert types["field_name"] == "JSON"
    # the year filter used by _build_filter_clause must bind
    assert conn.execute(
        "SELECT COUNT(*) FROM documents WHERE EXTRACT(year FROM doc_date) = 2026"
    ).fetchone()[0] == 1


def test_refresh_documents_wraps_legacy_bare_entity_names(tmp_path: Path):
    path = tmp_path / "legacy.sqlite"
    _make_truth(path, [{"doc_id": "d1", "field_name": "Duri"}])  # bare text, not JSON
    conn = duckdb.connect()

    refresh_documents(conn, path)

    assert conn.execute(
        "SELECT COUNT(*) FROM documents WHERE EXISTS ("
        "  SELECT 1 FROM json_each(documents.field_name) "
        "  WHERE CAST(value AS VARCHAR) ILIKE '%duri%')"
    ).fetchone()[0] == 1


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
    rows = dict(
        conn.execute("SELECT doc_id, doc_date FROM documents").fetchall()
    )
    assert rows["d1"] == date(2025, 3, 1)
    assert rows["d2"] is None
    assert rows["d3"] == date(2026, 7, 1)
    assert any(
        "doc_date" in r.message and "count=1" in r.message for r in caplog.records
    )


def test_refresh_documents_survives_apostrophe_in_truth_path(tmp_path: Path):
    """A truth DB living under a directory containing an apostrophe (an
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
    assert conn.execute("SELECT COUNT(*) FROM document_chunks").fetchone()[0] == 1
    assert conn.execute(
        "SELECT doc_id FROM document_chunks"
    ).fetchone()[0] == "d1"


def test_sweep_orphan_chunks_survives_a_null_doc_id_in_documents(truth_path: Path):
    """NOT IN against a subquery containing a NULL is UNKNOWN for every row,
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
    assert conn.execute("SELECT COUNT(*) FROM document_chunks").fetchone()[0] == 1
    assert conn.execute(
        "SELECT doc_id FROM document_chunks"
    ).fetchone()[0] == "d1"


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
    assert conn.execute("SELECT dst_id FROM kg_edge").fetchone()[0] == "POD-1"


def test_refresh_registry_distinguishes_empty_present_table_from_absent(tmp_path: Path):
    """A registry table that exists but holds zero rows (kg_edge right
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
    assert conn.execute("SELECT COUNT(*) FROM kg_edge").fetchone()[0] == 0


def test_refresh_registry_drops_mirror_table_when_truth_table_disappears(tmp_path: Path):
    """A registry table present in one refresh but absent from the truth in
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
    assert conn.execute("SELECT COUNT(*) FROM kg_edge").fetchone()[0] == 1

    conn_s = sqlite3.connect(path)
    conn_s.execute("DROP TABLE kg_edge")
    conn_s.commit()
    conn_s.close()

    second = refresh_registry(conn, path)

    assert "kg_edge" not in second
    with pytest.raises(duckdb.CatalogException):
        conn.execute("SELECT COUNT(*) FROM kg_edge")


def test_refresh_registry_stays_silent_when_table_was_never_mirrored(tmp_path: Path, caplog):
    """kg_edge/kg_claim have never existed in either store on most installs
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


def test_refresh_registry_logs_when_a_stale_mirror_is_actually_dropped(tmp_path: Path, caplog):
    """A table that WAS mirrored and then disappears from the truth is a
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


def test_refresh_registry_no_longer_raw_mirrors_pod_tables(tmp_path: Path):
    """m_pod, r_institution, r_pod_type, project_pod, pod_document, and
    pod_revision are real operational SQLite tables — present in the
    truth on every production install — but `publish_pod_registry` now
    owns their read-side DuckDB shapes (see the module docstring).
    `refresh_registry` must not raw-copy any of them even though they
    exist and hold rows in the truth.
    """
    path = tmp_path / "pod_present.sqlite"
    conn_s = sqlite3.connect(path)
    conn_s.execute("CREATE TABLE r_institution (code INTEGER, institution TEXT)")
    conn_s.execute("INSERT INTO r_institution VALUES (2, 'SKK Migas')")
    conn_s.execute("CREATE TABLE r_pod_type (code INTEGER, pod_type TEXT)")
    conn_s.execute("INSERT INTO r_pod_type VALUES (2, 'POD I')")
    conn_s.execute(
        "CREATE TABLE m_pod (id INTEGER PRIMARY KEY, pod_id TEXT, pod_name TEXT, "
        "institution_code INTEGER, pod_type_code INTEGER)"
    )
    conn_s.execute(
        "INSERT INTO m_pod VALUES (1, 'PL-2019-0001-2-2-0', 'Duri POD I', 2, 2)"
    )
    conn_s.execute("CREATE TABLE project_pod (pod_id INTEGER, project_id TEXT)")
    conn_s.execute("INSERT INTO project_pod VALUES (1, 'PRJ-1')")
    conn_s.execute("CREATE TABLE pod_document (pod_id INTEGER, doc_id TEXT)")
    conn_s.execute("INSERT INTO pod_document VALUES (1, 'd1')")
    conn_s.execute(
        "CREATE TABLE pod_revision (successor_id TEXT, predecessor_id TEXT)"
    )
    conn_s.commit()
    conn_s.close()
    conn = duckdb.connect()

    copied = refresh_registry(conn, path)

    assert copied == {}
    for table in (
        "m_pod", "r_institution", "r_pod_type", "project_pod",
        "pod_document", "pod_revision",
    ):
        with pytest.raises(duckdb.CatalogException):
            conn.execute(f"SELECT * FROM {table}")


def test_refresh_registry_retires_stale_pod_mirrors_from_a_pre_fix_database(
    tmp_path: Path, caplog
):
    """A database refreshed by the pre-fix build of this module has raw
    copies of the six POD tables sitting in DuckDB, including a
    `pod_document` with the wrong BIGINT surrogate `pod_id` clobbering
    what `publish_pod_registry` produces under the same table name. The
    very next `refresh_registry` call must drop every one of them, using
    the same stale-table drop mechanism this module already had for a
    table vanishing from the truth — this is what repairs a user's live
    database on their next `esdc corpus sync`.
    """
    path = tmp_path / "pod_present.sqlite"
    conn_s = sqlite3.connect(path)
    conn_s.execute("CREATE TABLE r_institution (code INTEGER, institution TEXT)")
    conn_s.execute("INSERT INTO r_institution VALUES (2, 'SKK Migas')")
    conn_s.commit()
    conn_s.close()
    conn = duckdb.connect()
    # Simulate exactly what the pre-fix refresh_registry left behind.
    conn.execute("CREATE TABLE r_institution (code INTEGER, institution VARCHAR)")
    conn.execute("CREATE TABLE r_pod_type (code INTEGER, pod_type VARCHAR)")
    conn.execute("CREATE TABLE m_pod (id INTEGER, pod_id VARCHAR)")
    conn.execute("CREATE TABLE project_pod (pod_id INTEGER, project_id VARCHAR)")
    conn.execute(
        "CREATE TABLE pod_revision (successor_id VARCHAR, predecessor_id VARCHAR)"
    )
    conn.execute("CREATE TABLE pod_document (pod_id BIGINT, doc_id VARCHAR)")
    conn.execute("INSERT INTO pod_document VALUES (1, 'd1')")

    with caplog.at_level("INFO"):
        copied = refresh_registry(conn, path)

    assert copied == {}
    for table in (
        "m_pod", "r_institution", "r_pod_type", "project_pod",
        "pod_document", "pod_revision",
    ):
        with pytest.raises(duckdb.CatalogException):
            conn.execute(f"SELECT * FROM {table}")
    assert sum(
        1 for r in caplog.records if "dropped retired registry table" in r.message
    ) == 6


def test_views_expose_both_grains(tmp_path: Path):
    """create_views joins the *published* pod_registry/pod_document tables
    (canonical VARCHAR pod_id) now, not the raw m_pod/pod_document SQLite
    shapes — those are built directly here rather than via refresh_registry,
    which no longer produces them at all (see
    test_refresh_registry_no_longer_raw_mirrors_pod_tables).
    """
    path = tmp_path / "both.sqlite"
    _make_truth(path, [{"doc_id": "d1"}])
    conn = duckdb.connect()
    refresh_documents(conn, path)
    conn.execute("CREATE TABLE pod_registry (pod_id VARCHAR, pod_name VARCHAR)")
    conn.executemany(
        "INSERT INTO pod_registry VALUES (?, ?)",
        [("POD-1", "A"), ("POD-2", "B")],
    )
    conn.execute("CREATE TABLE pod_document (pod_id VARCHAR, doc_id VARCHAR)")
    conn.executemany(
        "INSERT INTO pod_document VALUES (?, ?)", [("POD-1", "d1"), ("POD-2", "d1")]
    )

    created = create_views(conn)

    assert set(created) == {"v_doc_pod_link", "v_document"}
    # fan-out grain: one row per (document, pod)
    assert conn.execute("SELECT COUNT(*) FROM v_doc_pod_link").fetchone()[0] == 2
    # document grain: one row per document, links aggregated
    assert conn.execute("SELECT COUNT(*) FROM v_document").fetchone()[0] == 1
    assert sorted(
        conn.execute("SELECT linked_pod_ids FROM v_document").fetchone()[0]
    ) == ["POD-1", "POD-2"]


def test_views_degrade_when_registry_absent(truth_path: Path):
    conn = duckdb.connect()
    refresh_documents(conn, truth_path)

    created = create_views(conn)

    assert created == ["v_document"]
    assert conn.execute("SELECT COUNT(*) FROM v_document").fetchone()[0] == 2
    assert conn.execute("SELECT linked_pod_ids FROM v_document LIMIT 1").fetchone()[0] == []


def test_refresh_all_produces_canonical_varchar_pod_id_not_bigint_fk(tmp_path: Path):
    """Regression test. refresh_registry used to raw-mirror pod_document
    (pod_id BIGINT — the m_pod.id surrogate foreign key) on top of whatever
    publish_pod_registry had published under the same table name (pod_id
    VARCHAR — the canonical PL-YYYY-XXXX-A-B-R id that
    pod_registry_schema.yaml documents to the chat agent). A full
    refresh_all must leave pod_document holding the published VARCHAR
    shape with the canonical id, not the surrogate BIGINT one.
    """
    from esdc.pod_registry.store import get_sqlite_connection

    path = tmp_path / "truth.sqlite"
    _make_truth(path, [{"doc_id": "d1"}])
    conn_s = get_sqlite_connection(path)
    conn_s.execute(
        "INSERT INTO r_institution (code, institution) VALUES (2, 'SKK Migas')"
    )
    conn_s.execute("INSERT INTO r_pod_type (code, pod_type) VALUES (2, 'POD I')")
    conn_s.execute(
        "INSERT INTO m_pod (id, pod_id, pod_name, pod_letter_num, approval_date,"
        " institution_code, pod_type_code, rev_num, approval_seq)"
        " VALUES (1, 'PL-2019-0001-2-2-0', 'Duri POD I', 'SRT-1', '2019-05-01',"
        " 2, 2, 0, 1)"
    )
    conn_s.execute("INSERT INTO pod_document (pod_id, doc_id) VALUES (1, 'd1')")
    conn_s.commit()
    conn_s.close()
    conn = duckdb.connect()

    refresh_all(conn, path)

    col_type = dict(
        conn.execute(
            "SELECT column_name, column_type FROM (DESCRIBE pod_document)"
        ).fetchall()
    )["pod_id"]
    assert col_type == "VARCHAR"
    assert conn.execute("SELECT pod_id, doc_id FROM pod_document").fetchall() == [
        ("PL-2019-0001-2-2-0", "d1")
    ]


def test_refresh_all_repairs_a_pre_fix_database(tmp_path: Path):
    """End-to-end repair check for the exact state a user's live database
    is in today: a pre-fix build's raw POD mirrors (including a
    BIGINT-keyed pod_document and a v_doc_pod_link view built by joining
    the raw m_pod/pod_document) sitting in DuckDB. A single refresh_all
    must retire the six raw tables, publish pod_registry/pod_project/
    pod_document under the canonical VARCHAR pod_id shape, and rebuild
    both views against the published tables so neither dangles nor keeps
    serving stale rows.
    """
    from esdc.pod_registry.store import get_sqlite_connection

    path = tmp_path / "truth.sqlite"
    _make_truth(path, [{"doc_id": "d1"}])
    conn_s = get_sqlite_connection(path)
    conn_s.execute(
        "INSERT INTO r_institution (code, institution) VALUES (2, 'SKK Migas')"
    )
    conn_s.execute("INSERT INTO r_pod_type (code, pod_type) VALUES (2, 'POD I')")
    conn_s.execute(
        "INSERT INTO m_pod (id, pod_id, pod_name, pod_letter_num, approval_date,"
        " institution_code, pod_type_code, rev_num, approval_seq)"
        " VALUES (1, 'PL-2019-0001-2-2-0', 'Duri POD I', 'SRT-1', '2019-05-01',"
        " 2, 2, 0, 1)"
    )
    conn_s.execute("INSERT INTO pod_document (pod_id, doc_id) VALUES (1, 'd1')")
    conn_s.commit()
    conn_s.close()

    conn = duckdb.connect()
    refresh_documents(conn, path)
    # Simulate exactly what the pre-fix build left in DuckDB: raw mirrors
    # plus the view create_views used to build from them.
    conn.execute("CREATE TABLE m_pod (id INTEGER, pod_id VARCHAR, pod_name VARCHAR)")
    conn.execute("INSERT INTO m_pod VALUES (1, 'PL-2019-0001-2-2-0', 'Duri POD I')")
    conn.execute("CREATE TABLE pod_document (pod_id BIGINT, doc_id VARCHAR)")
    conn.execute("INSERT INTO pod_document VALUES (1, 'd1')")
    conn.execute("""
        CREATE VIEW v_doc_pod_link AS
        SELECT d.doc_id, p.pod_id, p.pod_name FROM documents d
        JOIN pod_document pd ON pd.doc_id = d.doc_id
        JOIN m_pod p ON p.id = pd.pod_id
    """)

    refresh_all(conn, path)

    for table in ("m_pod", "r_institution", "r_pod_type", "project_pod", "pod_revision"):
        with pytest.raises(duckdb.CatalogException):
            conn.execute(f"SELECT * FROM {table}")
    col_type = dict(
        conn.execute(
            "SELECT column_name, column_type FROM (DESCRIBE pod_document)"
        ).fetchall()
    )["pod_id"]
    assert col_type == "VARCHAR"
    assert conn.execute("SELECT doc_id, pod_id FROM v_doc_pod_link").fetchall() == [
        ("d1", "PL-2019-0001-2-2-0")
    ]
    assert conn.execute("SELECT linked_pod_ids FROM v_document").fetchone()[0] == [
        "PL-2019-0001-2-2-0"
    ]
