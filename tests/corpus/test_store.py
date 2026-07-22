import json
from pathlib import Path

import duckdb
import pytest

from esdc.corpus.chunker import Chunk
from esdc.corpus.store import CorpusStore


def _fake_vector(text: str) -> list[float]:
    """Deterministic text-dependent 3-dim vector, normalized.

    Buckets letters into three alphabet ranges so texts sharing
    characters point in similar directions — enough signal to make
    cosine ranking testable (a constant vector would rank everything
    equally and hide vector-search bugs).
    """
    v = [1.0, 1.0, 1.0]
    for ch in text.lower():
        if "a" <= ch <= "i":
            v[0] += 1.0
        elif "j" <= ch <= "r":
            v[1] += 1.0
        elif "s" <= ch <= "z":
            v[2] += 1.0
    norm = sum(x * x for x in v) ** 0.5
    return [x / norm for x in v]


class FakeEmbedder:
    model = "fake-embed"

    def generate_embedding(self, text: str) -> list[float]:
        return _fake_vector(text)

    def generate_embeddings_batch(self, texts: list[str]) -> list[list[float]]:
        return [_fake_vector(t) for t in texts]


class FakeEmbedder2(FakeEmbedder):
    model = "other-model"


@pytest.fixture
def store(tmp_path: Path):
    s = CorpusStore(db_path=tmp_path / "t.duckdb", embedder=FakeEmbedder())
    s.ensure_tables()
    yield s
    s.close()


DOC = {
    "doc_id": "abc123", "file_name": "s.pdf", "file_path": "/x/s.pdf",
    "file_hash": "ab" * 32, "doc_type": "surat", "doc_topic": None,
    "doc_number": "SRT-1", "doc_date": "2026-01-05", "subject": "Persetujuan",
    "sender": "SKK", "recipient": "KKKS", "doc_level": "field",
    "wk_name": "Rokan", "field_name": "Duri", "project_name": None,
    "pod_name": ["POD Mengoepeh"], "suggested_pod_ids": ["PL-2003-0005-3-2-0"],
    "raw_entities": "{}", "metadata": "{}", "markdown": "# Surat\nisi",
    "extraction_method": "native", "page_count": 1,
}


def test_insert_and_dedupe(store):
    assert store.document_exists(DOC["file_hash"]) is False
    store.insert_document(DOC, [Chunk(0, "Surat", "isi surat persetujuan")])
    assert store.document_exists(DOC["file_hash"]) is True
    assert store.counts() == {"documents": 1, "chunks": 1}


def test_force_replaces_document(store):
    store.insert_document(DOC, [Chunk(0, None, "versi lama")])
    store.delete_document(DOC["doc_id"])
    store.insert_document(DOC, [Chunk(0, None, "versi baru")])
    assert store.counts() == {"documents": 1, "chunks": 1}


def test_search_returns_inserted_doc(store):
    store.insert_document(DOC, [Chunk(0, "Surat", "persetujuan POD lapangan Duri")])
    store.rebuild_indexes()
    result = store.search("persetujuan POD", limit=5, filters=None)
    assert result["status"] == "success"
    assert result["results"][0]["doc_id"] == "abc123"
    assert result["results"][0]["file_name"] == "s.pdf"


def test_search_filter_excludes(store):
    store.insert_document(DOC, [Chunk(0, None, "persetujuan POD")])
    store.rebuild_indexes()
    result = store.search("persetujuan", limit=5, filters={"doc_type": "mom"})
    assert result["results"] == []


def test_search_results_include_doc_topic(store):
    doc = dict(DOC)
    doc["doc_topic"] = ["wpnb"]
    store.insert_document(doc, [Chunk(0, None, "persetujuan POD")])
    store.rebuild_indexes()
    result = store.search("persetujuan", limit=5, filters=None)
    assert result["results"][0]["doc_topic"] == ["wpnb"]


def test_get_document(store):
    store.insert_document(DOC, [Chunk(0, None, "isi")])
    doc = store.get_document("abc123")
    assert doc["markdown"] == "# Surat\nisi"
    assert store.get_document("nope") is None


def test_list_documents(store):
    store.insert_document(DOC, [Chunk(0, None, "a"), Chunk(1, None, "b")])
    docs = store.list_documents()
    assert len(docs) == 1
    assert docs[0]["doc_id"] == "abc123"
    assert docs[0]["n_chunks"] == 2


def test_clear(store):
    store.insert_document(DOC, [Chunk(0, None, "isi")])
    removed = store.clear()
    assert removed == {"documents": 1, "chunks": 1}
    assert store.counts() == {"documents": 0, "chunks": 0}
    store.ensure_tables()  # meta survives, no dim-mismatch error


def test_replace_chunks(store):
    store.insert_document(DOC, [Chunk(0, None, "lama")])
    store.replace_chunks("abc123", [Chunk(0, None, "baru"), Chunk(1, None, "baru2")])
    assert store.counts() == {"documents": 1, "chunks": 2}


def _doc_variant(doc_id: str, file_name: str) -> dict:
    doc = dict(DOC)
    doc["doc_id"] = doc_id
    doc["file_name"] = file_name
    doc["file_path"] = f"/x/{file_name}"
    doc["file_hash"] = (doc_id * 32)[:64]
    return doc


def test_search_ranking_multi_doc(store):
    # Each doc's chunk text is dominated by one letter bucket of the
    # fake embedder, so cosine ranking is text-dependent and decisive.
    store.insert_document(
        _doc_variant("doc-a", "a.pdf"), [Chunk(0, None, "abade beda ada gagah")]
    )
    store.insert_document(
        _doc_variant("doc-j", "j.pdf"), [Chunk(0, None, "jklm nopq lomp okon")]
    )
    store.insert_document(
        _doc_variant("doc-s", "s.pdf"), [Chunk(0, None, "stuv wxyz tusz vwyx")]
    )
    store.rebuild_indexes()

    result = store.search("abade beda gagah", limit=5, filters=None)
    assert result["status"] == "success"
    assert result["results"][0]["doc_id"] == "doc-a"

    truncated = store.search("abade beda gagah", limit=2, filters=None)
    assert truncated["count"] == 2
    assert len(truncated["results"]) == 2


def test_set_meta_dim_change_recreates_chunks(store):
    store.insert_document(DOC, [Chunk(0, None, "isi")])
    assert store.counts() == {"documents": 1, "chunks": 1}
    store.set_meta("new-model", dim=5)
    # chunks are derived data: recreated empty; documents preserved
    assert store.counts() == {"documents": 1, "chunks": 0}


def test_meta_mismatch_raises(tmp_path: Path):
    db_path = tmp_path / "mismatch.duckdb"
    s1 = CorpusStore(db_path=db_path, embedder=FakeEmbedder())
    s1.ensure_tables()
    s1.close()

    s2 = CorpusStore(db_path=db_path, embedder=FakeEmbedder2())
    with pytest.raises(ValueError, match="reembed"):
        s2.ensure_tables(validate_model=True)
    s2.close()


def test_search_not_available(tmp_path: Path):
    s = CorpusStore(db_path=tmp_path / "empty.duckdb", embedder=FakeEmbedder())
    result = s.search("anything", limit=5, filters=None)
    assert result["status"] == "not_available"
    s.close()


def test_ensure_tables_migrates_legacy_varchar_entities(tmp_path: Path):
    """ensure_tables must migrate legacy plain-text entity values.

    Pre-JSON-column DBs store plain text ('Rokan'); ensure_tables must
    wrap them into JSON arrays so json_each-based filters don't raise.
    """
    db = tmp_path / "corpus.duckdb"

    # Simulate a legacy DB: same table name, VARCHAR entity columns,
    # plain-text values.
    conn = duckdb.connect(str(db))
    conn.execute(
        """
        CREATE TABLE documents (
            doc_id VARCHAR PRIMARY KEY, file_name VARCHAR, file_path VARCHAR,
            file_hash VARCHAR, doc_type VARCHAR, doc_number VARCHAR,
            doc_date DATE, subject VARCHAR, sender VARCHAR, recipient VARCHAR,
            doc_level VARCHAR, wk_name VARCHAR, field_name VARCHAR,
            project_name VARCHAR, raw_entities JSON, metadata JSON,
            markdown TEXT NOT NULL, extraction_method VARCHAR,
            embedding_model VARCHAR, page_count INTEGER
        )
        """
    )
    conn.execute(
        "INSERT INTO documents (doc_id, file_name, file_path, file_hash, "
        "wk_name, markdown, extraction_method) "
        "VALUES ('abc', 'a.pdf', '/a.pdf', 'h1', 'Rokan', 'body', 'text')"
    )
    conn.close()

    store = CorpusStore(db_path=db, embedder=FakeEmbedder())
    try:
        store.ensure_tables()
        row = store._get_connection().execute(
            "SELECT wk_name FROM documents WHERE doc_id = 'abc'"
        ).fetchone()
        assert json.loads(row[0]) == ["Rokan"]
        # And a json_each-based filter must not raise:
        store._get_connection().execute(
            "SELECT count(*) FROM documents WHERE wk_name IS NOT NULL "
            "AND EXISTS (SELECT 1 FROM json_each(wk_name))"
        ).fetchone()
    finally:
        store.close()


def test_search_entity_filter_is_case_insensitive_substring(tmp_path: Path):
    store = CorpusStore(db_path=tmp_path / "c.duckdb", embedder=FakeEmbedder())
    try:
        store.ensure_tables()
        doc1 = _doc_variant("d1", "d1.pdf")
        doc1["wk_name"] = ["Rokan"]
        store.insert_document(doc1, [Chunk(0, None, "isi d1")])
        doc2 = _doc_variant("d2", "d2.pdf")
        doc2["wk_name"] = ["Bangkanai"]
        store.insert_document(doc2, [Chunk(0, None, "isi d2")])

        clause, params = store._build_filter_clause({"wk_name": "rokan"}, "d")
        rows = store._get_connection().execute(
            f"SELECT doc_id FROM documents d WHERE 1=1{clause}", params
        ).fetchall()
        assert [r[0] for r in rows] == ["d1"]

        # substring match too — "kan" is a substring of both "Rokan" and
        # "Bangkanai".
        clause, params = store._build_filter_clause({"wk_name": "kan"}, "d")
        rows = store._get_connection().execute(
            f"SELECT doc_id FROM documents d WHERE 1=1{clause}", params
        ).fetchall()
        assert {r[0] for r in rows} == {"d1", "d2"}
    finally:
        store.close()


def test_search_hydrated_docs_have_parsed_entity_lists(tmp_path: Path):
    """search() results must return wk_name as a list, not a JSON string."""
    store = CorpusStore(db_path=tmp_path / "c2.duckdb", embedder=FakeEmbedder())
    try:
        store.ensure_tables()
        doc = _doc_variant("d1", "d1.pdf")
        doc["wk_name"] = ["Rokan"]
        store.insert_document(doc, [Chunk(0, "Report", "drilling report content")])
        store.rebuild_indexes()
        result = store.search("drilling")
        assert result["status"] == "success"
        assert result["results"][0]["wk_name"] == ["Rokan"]
    finally:
        store.close()


def test_insert_and_get_pod_name_round_trip(store):
    store.insert_document(DOC, [Chunk(0, None, "isi")])
    got = store.get_document(DOC["doc_id"])
    assert got["pod_name"] == ["POD Mengoepeh"]
    assert got["suggested_pod_ids"] == ["PL-2003-0005-3-2-0"]
    docs = store.list_documents()
    assert docs[0]["pod_name"] == ["POD Mengoepeh"]


def test_insert_and_get_doc_topic_round_trip(store):
    doc = dict(DOC)
    doc["doc_topic"] = ["psc", "wpnb"]
    store.insert_document(doc, [Chunk(0, None, "isi")])
    got = store.get_document(doc["doc_id"])
    assert got["doc_topic"] == ["psc", "wpnb"]


def test_list_documents_includes_doc_topic(store):
    doc = dict(DOC)
    doc["doc_topic"] = ["pod_i"]
    store.insert_document(doc, [Chunk(0, None, "isi")])
    docs = store.list_documents()
    assert docs[0]["doc_topic"] == ["pod_i"]


def test_build_filter_clause_doc_topic_case_insensitive_substring(tmp_path: Path):
    store = CorpusStore(db_path=tmp_path / "c3.duckdb", embedder=FakeEmbedder())
    try:
        store.ensure_tables()
        doc1 = _doc_variant("d1", "d1.pdf")
        doc1["doc_topic"] = ["psc"]
        store.insert_document(doc1, [Chunk(0, None, "isi d1")])
        doc2 = _doc_variant("d2", "d2.pdf")
        doc2["doc_topic"] = ["wpnb"]
        store.insert_document(doc2, [Chunk(0, None, "isi d2")])

        clause, params = store._build_filter_clause({"doc_topic": "psc"}, "d")
        rows = store._get_connection().execute(
            f"SELECT doc_id FROM documents d WHERE 1=1{clause}", params
        ).fetchall()
        assert [r[0] for r in rows] == ["d1"]
    finally:
        store.close()


def test_search_filter_by_doc_topic_returns_matching_doc_only(tmp_path: Path):
    store = CorpusStore(db_path=tmp_path / "c4.duckdb", embedder=FakeEmbedder())
    try:
        store.ensure_tables()
        doc1 = _doc_variant("d1", "d1.pdf")
        doc1["doc_topic"] = ["psc"]
        store.insert_document(doc1, [Chunk(0, None, "persetujuan kontrak PSC")])
        doc2 = _doc_variant("d2", "d2.pdf")
        doc2["doc_topic"] = ["wpnb"]
        store.insert_document(doc2, [Chunk(0, None, "pengajuan WPNB")])
        store.rebuild_indexes()

        result = store.search(
            "persetujuan", limit=5, filters={"doc_topic": "psc"}
        )
        assert result["status"] == "success"
        assert {r["doc_id"] for r in result["results"]} == {"d1"}
    finally:
        store.close()


def test_ensure_tables_adds_doc_topic_column_to_legacy_documents_table(
    tmp_path: Path,
):
    """ensure_tables adds doc_topic column to legacy table.

    Must not crash on pre-doc_topic schema; json_each filters
    must work after migration.
    """
    db = tmp_path / "legacy_topic.duckdb"

    conn = duckdb.connect(str(db))
    conn.execute(
        """
        CREATE TABLE documents (
            doc_id VARCHAR PRIMARY KEY, file_name VARCHAR, file_path VARCHAR,
            file_hash VARCHAR, doc_type VARCHAR, doc_number VARCHAR,
            doc_date DATE, subject VARCHAR, sender VARCHAR, recipient VARCHAR,
            doc_level VARCHAR, wk_name JSON, field_name JSON,
            project_name JSON, raw_entities JSON, metadata JSON,
            markdown TEXT NOT NULL, extraction_method VARCHAR,
            embedding_model VARCHAR, page_count INTEGER
        )
        """
    )
    conn.execute(
        "INSERT INTO documents (doc_id, file_name, file_path, file_hash, "
        "markdown, extraction_method) "
        "VALUES ('abc', 'a.pdf', '/a.pdf', 'h1', 'body', 'text')"
    )
    conn.close()

    store = CorpusStore(db_path=db, embedder=FakeEmbedder())
    try:
        store.ensure_tables()  # must not crash
        row = store._get_connection().execute(
            "SELECT doc_topic FROM documents WHERE doc_id = 'abc'"
        ).fetchone()
        assert row[0] is None
        # json_each-based filter must not raise on the new column either.
        store._get_connection().execute(
            "SELECT count(*) FROM documents WHERE doc_topic IS NOT NULL "
            "AND EXISTS (SELECT 1 FROM json_each(doc_topic))"
        ).fetchone()
    finally:
        store.close()


# --------------------------------------------------------------------------
# SQLite source of truth + DuckDB mirror
# --------------------------------------------------------------------------


def _sqlite_doc_count(tmp_path: Path) -> int:
    import sqlite3

    conn = sqlite3.connect(tmp_path / "esdc.sqlite")
    try:
        return conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    finally:
        conn.close()


def test_insert_writes_sqlite_truth_and_duckdb_mirror(store, tmp_path):
    store.insert_document(DOC, [Chunk(0, "Surat", "isi surat")])
    assert _sqlite_doc_count(tmp_path) == 1
    # mirror row present for search joins / iris
    n = store._get_connection().execute(
        "SELECT COUNT(*) FROM documents"
    ).fetchone()[0]
    assert n == 1


def test_delete_removes_both_stores(store, tmp_path):
    store.insert_document(DOC, [Chunk(0, None, "isi")])
    store.delete_document(DOC["doc_id"])
    assert _sqlite_doc_count(tmp_path) == 0
    n = store._get_connection().execute(
        "SELECT COUNT(*) FROM documents"
    ).fetchone()[0]
    assert n == 0
    assert store.counts() == {"documents": 0, "chunks": 0}


class _RaisingConn:
    """Proxy around a real DuckDB connection that fails DELETEs on one table.

    DuckDBPyConnection.execute is a read-only attribute on the C extension
    type, so it cannot be monkeypatched directly — wrap the connection
    object instead and swap it in for ``store._conn``.
    """

    def __init__(self, real, table_to_fail: str):
        self._real = real
        self._table_to_fail = table_to_fail

    def execute(self, sql, *args, **kwargs):
        if "DELETE FROM" in sql and self._table_to_fail in sql:
            raise duckdb.Error("mirror locked")
        return self._real.execute(sql, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._real, name)


def test_delete_raises_and_keeps_sqlite_truth_when_mirror_delete_fails(
    store, tmp_path
):
    store.insert_document(DOC, [Chunk(0, None, "isi")])
    store._conn = _RaisingConn(store._get_connection(), store.CHUNK_TABLE)

    with pytest.raises(duckdb.Error):
        store.delete_document(DOC["doc_id"])

    # Mirror delete failed first, before the SQLite truth row was touched:
    # the doc is still fully visible, and a retry is possible.
    assert store.document_exists(DOC["file_hash"])
    assert _sqlite_doc_count(tmp_path) == 1


def test_orphaned_duckdb_rows_cleared_on_reinsert(store, tmp_path):
    import sqlite3

    store.insert_document(DOC, [Chunk(0, None, "isi")])
    # Simulate a failed earlier delete: sqlite row gone, duckdb rows remain.
    conn = sqlite3.connect(tmp_path / "esdc.sqlite")
    conn.execute("DELETE FROM documents")
    conn.commit()
    conn.close()
    assert not store.document_exists(DOC["file_hash"])
    # Re-insert must clear the orphans instead of violating PKs.
    store.insert_document(DOC, [Chunk(0, None, "isi baru")])
    assert _sqlite_doc_count(tmp_path) == 1
    assert store.counts()["chunks"] == 1


def test_exists_get_list_read_sqlite(store, tmp_path):

    store.insert_document(DOC, [Chunk(0, None, "isi")])
    # Mutate the mirror only; reads must reflect sqlite truth, not the mirror.
    store._get_connection().execute("DELETE FROM documents")
    assert store.document_exists(DOC["file_hash"])
    assert store.get_document(DOC["doc_id"]) is not None
    assert len(store.list_documents()) == 1
    assert store.find_doc_ids({"doc_type": "surat"}) == [
        (DOC["doc_id"], DOC["file_name"])
    ]


# --------------------------------------------------------------------------
# fill_blank_entities
# --------------------------------------------------------------------------


def _blank_entity_doc() -> dict:
    doc = dict(DOC)
    doc["wk_name"] = None
    doc["field_name"] = []
    doc["project_name"] = ["Existing Project"]
    return doc


def test_fill_blank_entities_fills_null_and_empty_leaves_non_empty(store):
    doc = _blank_entity_doc()
    store.insert_document(doc, [Chunk(0, None, "isi")])

    filled = store.fill_blank_entities(
        doc["doc_id"],
        {
            "wk_name": ["Sidecar WK"],
            "field_name": ["Sidecar Field"],
            "project_name": ["Sidecar Project"],
        },
    )

    assert filled == ["wk_name", "field_name"]
    result = store.get_document(doc["doc_id"])
    assert result["wk_name"] == ["Sidecar WK"]
    assert result["field_name"] == ["Sidecar Field"]
    # Non-empty stored value (e.g. portal-edited) is never clobbered.
    assert result["project_name"] == ["Existing Project"]


def test_fill_blank_entities_mirrors_to_duckdb(store):
    doc = _blank_entity_doc()
    store.insert_document(doc, [Chunk(0, None, "isi")])

    filled = store.fill_blank_entities(doc["doc_id"], {"wk_name": ["Sidecar WK"]})

    assert filled == ["wk_name"]
    mirror_row = (
        store._get_connection()
        .execute("SELECT wk_name FROM documents WHERE doc_id = ?", [doc["doc_id"]])
        .fetchone()
    )
    assert json.loads(mirror_row[0]) == ["Sidecar WK"]


def test_fill_blank_entities_no_blank_fields_returns_empty(store):
    doc = dict(DOC)
    doc["wk_name"] = ["Already Set"]
    doc["field_name"] = ["Already Set Field"]
    doc["project_name"] = ["Already Set Project"]
    store.insert_document(doc, [Chunk(0, None, "isi")])

    filled = store.fill_blank_entities(
        doc["doc_id"],
        {
            "wk_name": ["Sidecar WK"],
            "field_name": ["Sidecar Field"],
            "project_name": ["Sidecar Project"],
        },
    )

    assert filled == []
    result = store.get_document(doc["doc_id"])
    assert result["wk_name"] == ["Already Set"]
    assert result["field_name"] == ["Already Set Field"]
    assert result["project_name"] == ["Already Set Project"]


def test_fill_blank_entities_empty_sidecar_value_skips(store):
    doc = _blank_entity_doc()
    store.insert_document(doc, [Chunk(0, None, "isi")])

    filled = store.fill_blank_entities(
        doc["doc_id"], {"wk_name": None, "field_name": []}
    )

    assert filled == []
    result = store.get_document(doc["doc_id"])
    assert result["wk_name"] is None
    assert result["field_name"] == []


def test_fill_blank_entities_unknown_doc_id_returns_empty(store):
    assert store.fill_blank_entities("nope", {"wk_name": ["X"]}) == []
