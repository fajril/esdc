"""Tests for the SQLite -> DuckDB wholesale mirror refresh."""

from __future__ import annotations

import sqlite3
from datetime import date
from pathlib import Path

import duckdb
import pytest

from esdc.corpus.mirror import refresh_documents

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


from esdc.corpus.mirror import sweep_orphan_chunks


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
