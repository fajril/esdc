"""Tests for the SQLite -> DuckDB wholesale mirror refresh."""

from __future__ import annotations

import sqlite3
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
