"""Tests for the SQLite -> DuckDB wholesale mirror refresh."""

from __future__ import annotations

import sqlite3
from datetime import date
from pathlib import Path

import duckdb
import pytest

from esdc.corpus.mirror import (
    create_views,
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
    conn_s.execute("CREATE TABLE m_pod (id INTEGER PRIMARY KEY, pod_id TEXT, pod_name TEXT)")
    conn_s.execute("INSERT INTO m_pod VALUES (1, 'POD-1', 'Duri POD I')")
    conn_s.execute("CREATE TABLE project_pod (pod_id INTEGER, project_id TEXT)")
    conn_s.execute("INSERT INTO project_pod VALUES (1, 'PRJ-1')")
    conn_s.commit()
    conn_s.close()
    # kg_edge deliberately absent — learn has never run
    conn = duckdb.connect()

    copied = refresh_registry(conn, path)

    assert copied["m_pod"] == 1
    assert copied["project_pod"] == 1
    assert "kg_edge" not in copied
    assert conn.execute("SELECT pod_name FROM m_pod").fetchone()[0] == "Duri POD I"


def test_refresh_registry_distinguishes_empty_present_table_from_absent(tmp_path: Path):
    """A registry table that exists but holds zero rows (pod_document /
    pod_revision before `esdc corpus learn` has linked anything) must still
    be copied and reported with count 0 — distinct from a table that does
    not exist at all (kg_edge before `learn` has ever run), which must be
    skipped and must not appear as a key in the returned dict at all.
    """
    path = tmp_path / "reg_empty.sqlite"
    conn_s = sqlite3.connect(path)
    conn_s.execute(
        "CREATE TABLE m_pod (id INTEGER PRIMARY KEY, pod_id TEXT, pod_name TEXT)"
    )
    conn_s.execute("INSERT INTO m_pod VALUES (1, 'POD-1', 'Duri POD I')")
    conn_s.execute("CREATE TABLE pod_document (pod_id INTEGER, doc_id TEXT)")
    # pod_document intentionally left empty: the table exists, no rows yet.
    conn_s.commit()
    conn_s.close()
    # kg_edge deliberately absent — learn has never run
    conn = duckdb.connect()

    copied = refresh_registry(conn, path)

    assert copied["m_pod"] == 1
    assert copied["pod_document"] == 0
    assert "kg_edge" not in copied
    # not just the dict says 0 — the table must actually exist in DuckDB
    assert conn.execute("SELECT COUNT(*) FROM pod_document").fetchone()[0] == 0


def test_views_expose_both_grains(tmp_path: Path):
    path = tmp_path / "both.sqlite"
    _make_truth(path, [{"doc_id": "d1"}])
    conn_s = sqlite3.connect(path)
    conn_s.execute("CREATE TABLE m_pod (id INTEGER PRIMARY KEY, pod_id TEXT, pod_name TEXT)")
    conn_s.executemany(
        "INSERT INTO m_pod VALUES (?,?,?)", [(1, "POD-1", "A"), (2, "POD-2", "B")]
    )
    conn_s.execute("CREATE TABLE pod_document (pod_id INTEGER, doc_id TEXT)")
    conn_s.executemany("INSERT INTO pod_document VALUES (?,?)", [(1, "d1"), (2, "d1")])
    conn_s.commit()
    conn_s.close()
    conn = duckdb.connect()
    refresh_documents(conn, path)
    refresh_registry(conn, path)

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
