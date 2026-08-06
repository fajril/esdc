import json
import sqlite3
from pathlib import Path

import duckdb
import pytest

from esdc.corpus.chunker import Chunk, chunk_markdown
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
    store.refresh_mirror()  # search hydrates/joins off the mirror, not the insert
    store.rebuild_indexes()
    result = store.search("persetujuan POD", limit=5, filters=None)
    assert result["status"] == "success"
    assert result["results"][0]["doc_id"] == "abc123"
    assert result["results"][0]["file_name"] == "s.pdf"


def test_search_filter_excludes(store):
    store.insert_document(DOC, [Chunk(0, None, "persetujuan POD")])
    store.refresh_mirror()  # filtered search joins the mirror
    store.rebuild_indexes()
    result = store.search("persetujuan", limit=5, filters={"doc_type": "mom"})
    assert result["results"] == []


def test_search_results_include_doc_topic(store):
    doc = dict(DOC)
    doc["doc_topic"] = ["wpnb"]
    store.insert_document(doc, [Chunk(0, None, "persetujuan POD")])
    store.refresh_mirror()  # doc_topic is hydrated from the mirror
    store.rebuild_indexes()
    result = store.search("persetujuan", limit=5, filters=None)
    assert result["results"][0]["doc_topic"] == ["wpnb"]


def test_get_document(store):
    store.insert_document(DOC, [Chunk(0, None, "isi")])
    store.refresh_mirror()  # get_document is a serving read off the mirror
    doc = store.get_document("abc123")
    assert doc["markdown"] == "# Surat\nisi"
    assert store.get_document("nope") is None


def test_list_documents(store):
    store.insert_document(DOC, [Chunk(0, None, "a"), Chunk(1, None, "b")])
    doc_no_chunks = dict(DOC)
    doc_no_chunks["doc_id"] = "nochunks1"
    doc_no_chunks["file_hash"] = "cd" * 32
    doc_no_chunks["file_name"] = "nochunks.pdf"
    store.insert_document(doc_no_chunks, [])
    store.refresh_mirror()  # list_documents is a serving read off the mirror
    docs = store.list_documents()
    by_id = {d["doc_id"]: d for d in docs}
    assert len(docs) == 2
    assert by_id["abc123"]["n_chunks"] == 2
    # A document with zero chunks must still appear in the results, with
    # n_chunks == 0 (not missing, not None) — the LEFT JOIN + COUNT(c.chunk_id)
    # must not drop or null out rows with no matching chunks.
    assert "nochunks1" in by_id
    assert by_id["nochunks1"]["n_chunks"] == 0


def test_clear(store):
    store.insert_document(DOC, [Chunk(0, None, "isi")])
    removed = store.clear()
    assert removed == {"documents": 1, "chunks": 1}
    assert store.counts() == {"documents": 0, "chunks": 0}
    store.ensure_tables()  # meta survives, no dim-mismatch error


def test_replace_chunks(store):
    store.insert_document(DOC, [Chunk(0, None, "lama")])
    store.replace_chunks(DOC, [Chunk(0, None, "baru"), Chunk(1, None, "baru2")])
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
        store.refresh_mirror()  # this test queries the DuckDB mirror directly

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
        store.refresh_mirror()  # wk_name is hydrated from the mirror
        store.rebuild_indexes()
        result = store.search("drilling")
        assert result["status"] == "success"
        assert result["results"][0]["wk_name"] == ["Rokan"]
    finally:
        store.close()


def test_insert_and_get_pod_name_round_trip(store):
    store.insert_document(DOC, [Chunk(0, None, "isi")])
    store.refresh_mirror()  # get_document/list_documents are serving reads
    got = store.get_document(DOC["doc_id"])
    assert got["pod_name"] == ["POD Mengoepeh"]
    assert got["suggested_pod_ids"] == ["PL-2003-0005-3-2-0"]
    docs = store.list_documents()
    assert docs[0]["pod_name"] == ["POD Mengoepeh"]


def test_insert_and_get_doc_topic_round_trip(store):
    doc = dict(DOC)
    doc["doc_topic"] = ["psc", "wpnb"]
    store.insert_document(doc, [Chunk(0, None, "isi")])
    store.refresh_mirror()  # get_document is a serving read off the mirror
    got = store.get_document(doc["doc_id"])
    assert got["doc_topic"] == ["psc", "wpnb"]


def test_list_documents_includes_doc_topic(store):
    doc = dict(DOC)
    doc["doc_topic"] = ["pod_i"]
    store.insert_document(doc, [Chunk(0, None, "isi")])
    store.refresh_mirror()  # list_documents is a serving read off the mirror
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
        store.refresh_mirror()  # this test queries the DuckDB mirror directly

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
        store.refresh_mirror()  # filtered search joins the mirror
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
    # The mirror row is produced by refresh_mirror(), not by insert itself.
    store.refresh_mirror()
    n = store._get_connection().execute(
        "SELECT COUNT(*) FROM documents"
    ).fetchone()[0]
    assert n == 1


def test_delete_removes_both_stores(store, tmp_path):
    store.insert_document(DOC, [Chunk(0, None, "isi")])
    store.refresh_mirror()  # populate the mirror so there's something to remove
    store.delete_document(DOC["doc_id"])
    assert _sqlite_doc_count(tmp_path) == 0
    # Chunks are deleted eagerly by delete_document, so search stops
    # surfacing this doc immediately...
    assert store.counts() == {"documents": 0, "chunks": 0}
    # ...and delete_document also removes the `documents` mirror row in
    # the same DuckDB transaction as the chunk delete, so it is gone
    # immediately too — not just after the next refresh. get_document/
    # list_documents/find_doc_ids are serving reads off this mirror, so
    # leaving the row behind would keep a deleted document visible.
    n = store._get_connection().execute(
        "SELECT COUNT(*) FROM documents"
    ).fetchone()[0]
    assert n == 0
    # A subsequent refresh is a no-op here: the truth row is already gone,
    # so the mirror stays empty.
    store.refresh_mirror()
    n = store._get_connection().execute(
        "SELECT COUNT(*) FROM documents"
    ).fetchone()[0]
    assert n == 0


def test_delete_document_removes_mirror_row_without_intervening_refresh(
    store, tmp_path
):
    """delete_document removes the mirror row immediately, with no refresh in between.

    Reproduces the Finding 1 bug: insert -> refresh -> delete must make
    the document disappear from every serving read (get_document,
    list_documents, find_doc_ids) immediately, with no refresh_mirror()
    call between the delete and the reads. Before the fix, delete_document
    left the mirror's `documents` row behind, so these all still returned
    the deleted document until the next refresh.
    """
    store.insert_document(DOC, [Chunk(0, None, "isi")])
    store.refresh_mirror()
    assert store.get_document(DOC["doc_id"]) is not None

    store.delete_document(DOC["doc_id"])

    assert store.get_document(DOC["doc_id"]) is None
    assert DOC["doc_id"] not in {d["doc_id"] for d in store.list_documents()}
    assert DOC["doc_id"] not in {
        doc_id for doc_id, _ in store.find_doc_ids({"doc_type": "surat"})
    }


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


def test_document_exists_reads_truth_independent_of_mirror_mutation(store, tmp_path):
    """document_exists (deciding read) is unaffected by mutating the DuckDB mirror directly.

    get_document/list_documents/find_doc_ids (serving
    reads) answer from that same mirror, so once it is wiped they go
    empty/None until the next refresh_mirror() — the inverse of the old
    contract, where every read here went to sqlite truth.
    """
    store.insert_document(DOC, [Chunk(0, None, "isi")])
    store.refresh_mirror()  # populate the mirror so there is something to wipe
    # Mutate the mirror only; document_exists must still see sqlite truth.
    store._get_connection().execute("DELETE FROM documents")
    assert store.document_exists(DOC["file_hash"])
    assert store.get_document(DOC["doc_id"]) is None
    assert store.list_documents() == []
    assert store.find_doc_ids({"doc_type": "surat"}) == []


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
    store.refresh_mirror()  # mirror row must exist for get_document below

    filled = store.fill_blank_entities(
        doc["doc_id"],
        {
            "wk_name": ["Sidecar WK"],
            "field_name": ["Sidecar Field"],
            "project_name": ["Sidecar Project"],
        },
    )

    assert filled == ["wk_name", "field_name"]
    # fill_blank_entities only writes the SQLite truth now (no more
    # row-by-row DuckDB mirror write); refresh before reading it back
    # through the mirror-backed get_document.
    store.refresh_mirror()
    result = store.get_document(doc["doc_id"])
    assert result["wk_name"] == ["Sidecar WK"]
    assert result["field_name"] == ["Sidecar Field"]
    # Non-empty stored value (e.g. portal-edited) is never clobbered.
    assert result["project_name"] == ["Existing Project"]


def test_fill_blank_entities_mirrors_to_duckdb(store):
    doc = _blank_entity_doc()
    store.insert_document(doc, [Chunk(0, None, "isi")])
    store.refresh_mirror()  # the mirror row must exist before fill_blank_entities

    filled = store.fill_blank_entities(doc["doc_id"], {"wk_name": ["Sidecar WK"]})
    assert filled == ["wk_name"]

    # fill_blank_entities only writes the SQLite truth; the mirror only
    # reflects it after the next refresh_mirror() (no more row-by-row
    # dual-write here).
    store.refresh_mirror()
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
    store.refresh_mirror()  # mirror row must exist for get_document below

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
    store.refresh_mirror()  # mirror row must exist for get_document below

    filled = store.fill_blank_entities(
        doc["doc_id"], {"wk_name": None, "field_name": []}
    )

    assert filled == []
    result = store.get_document(doc["doc_id"])
    assert result["wk_name"] is None
    assert result["field_name"] == []


def test_fill_blank_entities_unknown_doc_id_returns_empty(store):
    assert store.fill_blank_entities("nope", {"wk_name": ["X"]}) == []


def test_get_document_by_hash_returns_row_then_none(store):
    store.insert_document(DOC, [Chunk(0, None, "isi")])
    got = store.get_document_by_hash(DOC["file_hash"])
    assert got is not None
    assert got["doc_id"] == "abc123"
    assert got["file_hash"] == DOC["file_hash"]
    assert got["doc_date"] == DOC["doc_date"]
    assert got["subject"] == "Persetujuan"


def test_get_document_readers_return_same_row_shape(store):
    """Pin that get_document, get_document_by_id, and get_document_by_hash agree.

    get_document (DuckDB mirror) and get_document_by_id/by_hash (SQLite
    truth) share one column list + JSON-field set; pin that all three
    return the same keys so a future column add/rename/remove can't drift
    one reader out of sync with the other two silently.
    """
    store.insert_document(DOC, [Chunk(0, None, "isi")])
    store.refresh_mirror()  # get_document is a serving read off the mirror

    by_mirror = store.get_document(DOC["doc_id"])
    by_id = store.get_document_by_id(DOC["doc_id"])
    by_hash = store.get_document_by_hash(DOC["file_hash"])

    assert by_mirror is not None
    assert by_id is not None
    assert by_hash is not None
    assert by_mirror.keys() == by_id.keys() == by_hash.keys()
    # Both SQLite-truth readers hit the exact same row via different
    # columns -- their contents, not just their keys, must match.
    assert by_id == by_hash


def test_default_embedder_is_internal(monkeypatch, tmp_path):
    from esdc.corpus.embedder import MODEL_ID

    store = CorpusStore(
        db_path=tmp_path / "corpus.duckdb", sqlite_path=tmp_path / "esdc.sqlite"
    )
    assert store._embedder.model == MODEL_ID
    assert type(store._embedder).__name__ == "InternalEmbedder"
    assert store.get_document_by_hash("deadbeef") is None


# --------------------------------------------------------------------------
# embed_text: contextual prefix drives embeddings + FTS (chunk_text stays
# display-only)
# --------------------------------------------------------------------------


class RecordingEmbedder:
    model = "fake-model"

    def __init__(self):
        self.batch_calls = []
        self.calls = []  # every single-text generate_embedding() call, in order

    def generate_embedding(self, text):
        self.calls.append(text)
        return [0.1] * 8

    def generate_embeddings_batch(self, texts):
        self.batch_calls.append(list(texts))
        return [[0.1] * 8 for _ in texts]


@pytest.fixture
def store_with_doc_factory(tmp_path):
    stores = []

    def factory(chunk_size=3000, **doc_fields):
        store = CorpusStore(
            db_path=tmp_path / "corpus.duckdb",
            embedder=RecordingEmbedder(),
            sqlite_path=tmp_path / "esdc.sqlite",
        )
        store.ensure_tables()
        doc = dict(DOC)
        doc.update(doc_fields)
        chunks = chunk_markdown(doc["markdown"], chunk_size, min(300, chunk_size - 1))
        store.insert_document(doc, chunks)
        stores.append(store)
        return store, doc

    yield factory
    for s in stores:
        s.close()


def test_insert_stores_contextual_embed_text(store_with_doc_factory):
    """embed_text = prefix + section + chunk text; chunk_text untouched."""
    store, doc = store_with_doc_factory(
        doc_type="POD", subject="Pengembangan Merak", field_name=["Merak"]
    )
    row = store._get_connection().execute(
        "SELECT chunk_text, embed_text FROM document_chunks LIMIT 1"
    ).fetchone()
    chunk_text, embed_text = row
    assert "Merak" in embed_text
    assert embed_text.endswith(chunk_text)
    assert "Merak |" not in chunk_text  # display text has no prefix


def test_embedder_receives_contextual_text(store_with_doc_factory):
    """The vector is computed from embed_text, not chunk_text."""
    store, doc = store_with_doc_factory(
        doc_type="POD", subject="Pengembangan Merak", field_name=["Merak"]
    )
    embedded_texts = store._embedder.batch_calls[-1]
    assert all("Merak" in t for t in embedded_texts)


def test_keyword_search_matches_prefix_terms(store_with_doc_factory):
    """FTS runs over embed_text: doc-level entity terms hit every chunk."""
    store, doc = store_with_doc_factory(
        doc_type="POD", subject="Pengembangan Merak", field_name=["Merak"]
    )
    store.refresh_mirror()  # _keyword_search joins the mirror unconditionally
    store.rebuild_indexes()
    results = store._keyword_search("Merak", 10, None)
    assert results, "prefix term must be FTS-searchable"
    assert results[0]["embed_text"]


def test_search_over_retrieves_before_rrf(store_with_doc_factory, monkeypatch):
    store, _ = store_with_doc_factory(subject="Pengembangan Merak")
    seen = {}

    orig_vec = store._vector_search

    def spy_vector(embedding, limit, filters):
        seen["pool"] = limit
        return orig_vec(embedding, limit, filters)

    monkeypatch.setattr(store, "_vector_search", spy_vector)
    store.rebuild_indexes()
    store.search("produksi", limit=5)
    assert seen["pool"] == 50  # max(5 * 2, 50)


def test_search_rerank_reorders_top_pool(store_with_doc_factory, monkeypatch):
    import esdc.corpus.reranker as reranker_mod
    from esdc.corpus.reranker import Reranker

    Reranker._instance = None
    Reranker._failed = False

    class ReverseModel:
        def __init__(self):
            self.i = 0

        def embed(self, prompt):
            self.i += 1
            return [float(self.i), 0.0]  # later candidate wins

    monkeypatch.setattr(reranker_mod, "_load_reranker", lambda: ReverseModel())

    # chunk_size=20 forces the two sections into separate chunks so the
    # reranker has something to reorder.
    store, _ = store_with_doc_factory(
        chunk_size=20,
        subject="Pengembangan Merak",
        markdown="# A\n\nalpha konten\n\n# B\n\nbeta konten",
    )
    store.rebuild_indexes()
    baseline = store.search("konten", limit=2, rerank=False)
    reranked = store.search("konten", limit=2, rerank=True)
    assert reranked["status"] == "success"
    base_ids = [r["doc_id"] + r["chunk_text"] for r in baseline["results"]]
    rer_ids = [r["doc_id"] + r["chunk_text"] for r in reranked["results"]]
    assert rer_ids == list(reversed(base_ids))

    Reranker._instance = None
    Reranker._failed = False


def test_sample_content_default_is_first_chunk(store_with_doc_factory):
    store, doc = store_with_doc_factory(
        chunk_size=20, markdown="# A\n\nalpha satu\n\n# B\n\nbeta dua"
    )
    got = store.sample_content(doc["doc_id"])
    first = store._get_connection().execute(
        "SELECT chunk_text FROM document_chunks ORDER BY chunk_index LIMIT 1"
    ).fetchone()[0]
    assert got["chunk_text"] == first


def test_sample_content_seeded_pick_is_deterministic(store_with_doc_factory):
    store, doc = store_with_doc_factory(
        chunk_size=20, markdown="# A\n\nalpha satu\n\n# B\n\nbeta dua"
    )
    doc_id = doc["doc_id"]
    a = store.sample_content(doc_id, chunk_seed=doc_id)
    b = store.sample_content(doc_id, chunk_seed=doc_id)
    assert a["chunk_text"] == b["chunk_text"]
    assert a["chunk_text"]


def test_sample_content_seed_can_select_a_later_chunk(store_with_doc_factory):
    store, doc = store_with_doc_factory(
        chunk_size=20, markdown="# A\n\nalpha satu\n\n# B\n\nbeta dua"
    )
    texts = {
        store.sample_content(doc["doc_id"], chunk_seed=s)["chunk_text"]
        for s in ("a", "b", "c", "d", "e", "f", "g", "h")
    }
    assert len(texts) > 1  # the seed actually varies the pick


def test_search_surfaces_rerank_score(store_with_doc_factory, monkeypatch):
    """rerank_score must reach the payload; evaluate.py needs it for negatives."""
    import esdc.corpus.reranker as reranker_mod
    from esdc.corpus.reranker import Reranker

    Reranker._instance = None
    Reranker._failed = False

    class FixedModel:
        def embed(self, prompt):
            return [0.42, 0.0]

    monkeypatch.setattr(reranker_mod, "_load_reranker", lambda: FixedModel())

    # chunk_size=20 forces two chunks; _maybe_rerank is a no-op below two.
    store, _ = store_with_doc_factory(
        chunk_size=20,
        subject="Pengembangan Merak",
        markdown="# A\n\nalpha konten\n\n# B\n\nbeta konten",
    )
    store.rebuild_indexes()
    out = store.search("konten", limit=3, rerank=True)
    assert out["status"] == "success"
    assert out["results"][0]["rerank_score"] == 0.42

    Reranker._instance = None
    Reranker._failed = False


def test_search_omits_rerank_score_when_rerank_off(store_with_doc_factory):
    store, _ = store_with_doc_factory(subject="Pengembangan Merak")
    store.rebuild_indexes()
    out = store.search("konten", limit=3, rerank=False)
    assert out["status"] == "success"
    assert "rerank_score" not in out["results"][0]


def test_search_rerank_unavailable_falls_back(store_with_doc_factory, monkeypatch):
    import esdc.corpus.reranker as reranker_mod
    from esdc.corpus.reranker import Reranker

    Reranker._instance = None
    Reranker._failed = False

    def boom():
        raise RuntimeError("model missing")

    monkeypatch.setattr(reranker_mod, "_load_reranker", boom)

    store, _ = store_with_doc_factory(subject="Pengembangan Merak")
    store.rebuild_indexes()
    result = store.search("produksi", limit=5, rerank=True)
    assert result["status"] in ("success", "no_results")  # never error

    Reranker._instance = None
    Reranker._failed = False


def test_cap_per_doc_keeps_first_two_per_doc_in_incoming_order(store):
    """cap 2 over 10 entries from 2 docs -> 4 results, first two per doc kept.

    Incoming order is what rerank produced (best-scoring first), so
    "first two" is "two best-scoring" once this runs after _maybe_rerank.
    """
    merged = [
        {"doc_id": "a", "chunk_id": 0},
        {"doc_id": "a", "chunk_id": 1},
        {"doc_id": "b", "chunk_id": 2},
        {"doc_id": "a", "chunk_id": 3},
        {"doc_id": "b", "chunk_id": 4},
        {"doc_id": "a", "chunk_id": 5},
        {"doc_id": "b", "chunk_id": 6},
        {"doc_id": "a", "chunk_id": 7},
        {"doc_id": "b", "chunk_id": 8},
        {"doc_id": "b", "chunk_id": 9},
    ]
    result = store._cap_per_doc(merged, 2)
    assert len(result) == 4
    assert [r["chunk_id"] for r in result if r["doc_id"] == "a"] == [0, 1]
    assert [r["chunk_id"] for r in result if r["doc_id"] == "b"] == [2, 4]


def test_cap_per_doc_zero_is_identity(store):
    merged = [{"doc_id": "a", "chunk_id": 0}, {"doc_id": "b", "chunk_id": 1}]
    result = store._cap_per_doc(merged, 0)
    assert result is merged


def test_search_caps_chunks_per_document(store_with_doc_factory, monkeypatch):
    """No doc_id may appear more than max_chunks_per_doc times in results."""
    from esdc.configs import Config

    monkeypatch.setattr(
        Config,
        "get_corpus_config",
        classmethod(lambda cls: {"rerank": False, "max_chunks_per_doc": 2}),
    )
    store, _ = store_with_doc_factory(
        chunk_size=20,
        subject="Pengembangan Merak",
        markdown=(
            "# A\n\nkonten satu\n\n# B\n\nkonten dua\n\n"
            "# C\n\nkonten tiga\n\n# D\n\nkonten empat"
        ),
    )
    store.rebuild_indexes()
    n_chunks = store._get_connection().execute(
        "SELECT COUNT(*) FROM document_chunks"
    ).fetchone()[0]
    assert n_chunks > 2, "test is void unless the single doc had >2 chunks"

    result = store.search("konten", limit=10)
    assert result["status"] == "success"
    doc_ids = [r["doc_id"] for r in result["results"]]
    counts = {d: doc_ids.count(d) for d in set(doc_ids)}
    assert all(c <= 2 for c in counts.values())


# --------------------------------------------------------------------------
# query_instruct: Qwen3's asymmetric instruction prefix, query side only
# --------------------------------------------------------------------------


def test_search_embeds_query_with_instruct_prefix_when_enabled(
    store_with_doc_factory, monkeypatch
):
    from esdc.configs import Config

    monkeypatch.setattr(
        Config,
        "get_corpus_config",
        classmethod(lambda cls: {"rerank": False, "query_instruct": True}),
    )
    store, _ = store_with_doc_factory(subject="Pengembangan Merak")
    store.rebuild_indexes()
    store._embedder.calls.clear()

    store.search("produksi minyak", limit=3)

    assert store._embedder.calls, "search() must embed the query"
    query_call = store._embedder.calls[0]
    assert query_call.startswith("Instruct: ")
    assert "produksi minyak" in query_call


def test_search_embeds_raw_query_when_instruct_disabled(
    store_with_doc_factory, monkeypatch
):
    from esdc.configs import Config

    monkeypatch.setattr(
        Config,
        "get_corpus_config",
        classmethod(lambda cls: {"rerank": False, "query_instruct": False}),
    )
    store, _ = store_with_doc_factory(subject="Pengembangan Merak")
    store.rebuild_indexes()
    store._embedder.calls.clear()

    store.search("produksi minyak", limit=3)

    assert store._embedder.calls[0] == "produksi minyak"  # byte-identical to today


def test_aggregate_semantic_also_uses_query_instruct_prefix(
    store_with_doc_factory, monkeypatch
):
    from esdc.configs import Config

    monkeypatch.setattr(
        Config,
        "get_corpus_config",
        classmethod(lambda cls: {"rerank": False, "query_instruct": True}),
    )
    store, _ = store_with_doc_factory(subject="Pengembangan Merak")
    store.rebuild_indexes()
    store.refresh_mirror()  # _aggregate_semantic joins the documents mirror
    store._embedder.calls.clear()

    store.aggregate("produksi minyak", match="semantic")

    assert store._embedder.calls
    assert store._embedder.calls[0].startswith("Instruct: ")


def test_query_instruct_does_not_affect_chunk_embedding(
    store_with_doc_factory, monkeypatch
):
    """Enabling query_instruct must not leak the prefix into indexed chunks."""
    from esdc.configs import Config

    monkeypatch.setattr(
        Config,
        "get_corpus_config",
        classmethod(lambda cls: {"rerank": False, "query_instruct": True}),
    )
    store, doc = store_with_doc_factory(
        doc_type="POD", subject="Pengembangan Merak", field_name=["Merak"]
    )
    embedded_texts = store._embedder.batch_calls[-1]
    assert embedded_texts, "insert_document must batch-embed chunk text"
    assert all(not t.startswith("Instruct: ") for t in embedded_texts)
    assert all("Merak" in t for t in embedded_texts)


def test_corpus_config_isolated_in_tests():
    """Config isolation: search() must not read the real ~/.esdc config.

    search()'s _maybe_rerank calls Config.get_corpus_config(); the
    autouse _isolated_db_dirs fixture in conftest.py must also patch
    _load_config so tests never read the real ~/.esdc/config.yaml. If a
    dev machine has corpus.rerank: true set, an un-isolated test would
    trigger a real cross-encoder download and reorder search results.
    """
    from esdc.configs import Config

    assert Config.get_corpus_config()["rerank"] is False


# --------------------------------------------------------------------------
# Read helpers for eval query generation
# --------------------------------------------------------------------------


@pytest.fixture
def populated_store(store):
    store.insert_document(DOC, [Chunk(0, "Surat", "isi surat persetujuan")])
    return store


def test_fingerprint_rows_returns_doc_id_and_hash(populated_store):
    rows = populated_store.fingerprint_rows()
    assert all(len(r) == 2 for r in rows)
    ids = {r[0] for r in rows}
    assert ids  # non-empty; matches inserted docs


def test_sample_content_returns_first_chunk(populated_store):
    any_id = populated_store.fingerprint_rows()[0][0]
    content = populated_store.sample_content(any_id)
    assert content["doc_id"] == any_id
    assert "doc_type" in content and "subject" in content
    assert isinstance(content["chunk_text"], str)
    assert content["file_hash"] == DOC["file_hash"]


def test_sample_content_missing_doc_returns_none(populated_store):
    assert populated_store.sample_content("does-not-exist") is None


def test_document_bodies_returns_id_number_and_markdown(populated_store):
    rows = populated_store.document_bodies()
    assert rows
    doc_id, doc_number, markdown = rows[0]
    assert isinstance(doc_id, str)
    assert isinstance(markdown, str)
    assert doc_id == DOC["doc_id"]
    assert doc_number == DOC["doc_number"]
    assert markdown == DOC["markdown"]


def test_refresh_mirror_rebuilds_documents_from_sqlite_truth(tmp_path):
    """A row written only to the SQLite truth appears in the mirror after refresh."""
    from esdc.corpus.store import CorpusStore

    store = CorpusStore(
        db_path=tmp_path / "m.duckdb",
        embedder=FakeEmbedder(),
        sqlite_path=tmp_path / "m.sqlite",
    )
    store.ensure_tables()
    sconn = store._get_sqlite()
    sconn.execute(
        "INSERT INTO documents (doc_id, file_name, file_path, file_hash, "
        "doc_type, doc_date, markdown, extraction_method, embedding_model) "
        "VALUES ('sneaky','s.pdf','/tmp/s.pdf','h1','surat','2026-01-05','# x','docling','m')"
    )
    sconn.commit()
    assert store._get_connection().execute(
        "SELECT COUNT(*) FROM documents WHERE doc_id = 'sneaky'"
    ).fetchone()[0] == 0

    report = store.refresh_mirror()

    assert report.documents == 1
    assert store._get_connection().execute(
        "SELECT COUNT(*) FROM documents WHERE doc_id = 'sneaky'"
    ).fetchone()[0] == 1
    store.close()


def test_get_sqlite_self_heals_missing_raw_entities_metadata_ingested_at(
    tmp_path: Path,
):
    """A legacy documents table self-heals its missing columns via ALTER on connect.

    A SQLite documents table predating raw_entities/metadata/ingested_at
    self-heals via ALTER on connect.

    refresh_documents' `SELECT * REPLACE (...)` names those three columns
    explicitly, so a legacy table missing any of them broke every refresh
    (and, since Task 6, every corpus commit/learn/portal save). The two
    text columns get the same ADD COLUMN treatment already used for
    pod_name/suggested_pod_ids; ingested_at is added without its
    DEFAULT (datetime('now')) clause, since SQLite's ADD COLUMN only
    accepts a constant default.
    """
    import sqlite3

    sqlite_path = tmp_path / "legacy.sqlite"
    conn = sqlite3.connect(sqlite_path)
    conn.execute(
        """
        CREATE TABLE documents (
            doc_id TEXT PRIMARY KEY, file_name TEXT NOT NULL,
            file_path TEXT NOT NULL, file_hash TEXT NOT NULL UNIQUE,
            doc_type TEXT, doc_topic TEXT, doc_number TEXT, doc_date TEXT,
            subject TEXT, sender TEXT, recipient TEXT, doc_level TEXT,
            wk_name TEXT, field_name TEXT, project_name TEXT,
            pod_name TEXT, suggested_pod_ids TEXT,
            markdown TEXT NOT NULL, extraction_method TEXT NOT NULL,
            embedding_model TEXT NOT NULL, page_count INTEGER
        )
        """
    )
    conn.execute(
        "INSERT INTO documents (doc_id, file_name, file_path, file_hash, "
        "markdown, extraction_method, embedding_model) "
        "VALUES ('legacy1', 'a.pdf', '/a.pdf', 'h1', '# body', 'native', 'm')"
    )
    conn.commit()
    conn.close()

    store = CorpusStore(
        db_path=tmp_path / "legacy.duckdb",
        embedder=FakeEmbedder(),
        sqlite_path=sqlite_path,
    )
    try:
        store.ensure_tables()  # must not crash on the legacy schema

        cols = {
            row[1]
            for row in store._get_sqlite()
            .execute("PRAGMA table_info(documents)")
            .fetchall()
        }
        assert {"raw_entities", "metadata", "ingested_at"} <= cols

        # refresh_mirror's SELECT * REPLACE(...) must not raise either.
        report = store.refresh_mirror()
        assert report.documents == 1
    finally:
        store.close()


def test_insert_document_writes_truth_and_chunks_but_not_mirror(tmp_path):
    """The mirror row is produced by refresh, not by the insert path."""
    from esdc.corpus.chunker import Chunk
    from esdc.corpus.store import CorpusStore

    store = CorpusStore(
        db_path=tmp_path / "i.duckdb",
        embedder=FakeEmbedder(),
        sqlite_path=tmp_path / "i.sqlite",
    )
    store.ensure_tables()
    doc = {
        "doc_id": "d1", "file_name": "a.pdf", "file_path": "/tmp/a.pdf",
        "file_hash": "h1", "doc_type": "surat", "doc_date": "2026-01-01",
        "markdown": "# x", "extraction_method": "docling", "embedding_model": "m",
    }
    store.insert_document(doc, [Chunk(index=0, section=None, text="hello")])

    conn = store._get_connection()
    assert conn.execute("SELECT COUNT(*) FROM document_chunks").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 0
    assert store.document_exists("h1") is True  # truth has it

    store.refresh_mirror()
    assert conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 1
    store.close()


class _RaisingSqliteConn:
    """Proxy around a real sqlite3 connection that fails INSERTs into one table.

    Mirrors ``_RaisingConn`` above but for the SQLite side: only the
    truth-row INSERT fails, so the health-check ``SELECT 1`` that
    ``_get_sqlite()`` runs on an already-open connection still succeeds.
    """

    def __init__(self, real, table_to_fail: str):
        self._real = real
        self._table_to_fail = table_to_fail

    def execute(self, sql, *args, **kwargs):
        if "INSERT INTO" in sql and self._table_to_fail in sql:
            raise sqlite3.OperationalError("truth write failed")
        return self._real.execute(sql, *args, **kwargs)

    def __enter__(self):
        self._real.__enter__()
        return self

    def __exit__(self, *exc_info):
        return self._real.__exit__(*exc_info)

    def __getattr__(self, name):
        return getattr(self._real, name)


def test_insert_document_cleans_up_chunks_when_truth_write_fails(tmp_path):
    """insert_document cleans up already-written chunks when the truth write fails.

    If the SQLite commit-marker write raises, the chunks written just
    before it must be cleaned up so no orphaned embeddings survive, and
    the exception must still propagate (see insert_document's docstring).
    """
    store = CorpusStore(
        db_path=tmp_path / "fail.duckdb",
        embedder=FakeEmbedder(),
        sqlite_path=tmp_path / "fail.sqlite",
    )
    store.ensure_tables()
    doc = dict(DOC)
    doc["doc_id"] = "fails1"

    store._sconn = _RaisingSqliteConn(store._get_sqlite(), store.DOC_TABLE)

    with pytest.raises(sqlite3.OperationalError):
        store.insert_document(doc, [Chunk(0, "Section", "text")])

    conn = store._get_connection()
    assert (
        conn.execute(
            f"SELECT COUNT(*) FROM {store.CHUNK_TABLE} WHERE doc_id = ?",
            [doc["doc_id"]],
        ).fetchone()[0]
        == 0
    )
    assert conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 0
    store.close()


def test_serving_reads_use_the_mirror_and_deciding_reads_use_the_truth(tmp_path):
    """get_document answers from DuckDB; document_exists answers from SQLite."""
    from esdc.corpus.store import CorpusStore

    store = CorpusStore(
        db_path=tmp_path / "r.duckdb",
        embedder=FakeEmbedder(),
        sqlite_path=tmp_path / "r.sqlite",
    )
    store.ensure_tables()
    sconn = store._get_sqlite()
    sconn.execute(
        "INSERT INTO documents (doc_id, file_name, file_path, file_hash, "
        "doc_type, doc_date, markdown, extraction_method, embedding_model) "
        "VALUES ('d1','a.pdf','/tmp/a.pdf','h1','surat','2026-01-01','# body','docling','m')"
    )
    sconn.commit()

    # deciding read sees the truth immediately
    assert store.document_exists("h1") is True
    # serving read does not, until the mirror is refreshed
    assert store.get_document("d1") is None

    store.refresh_mirror()
    doc = store.get_document("d1")
    assert doc is not None and doc["markdown"] == "# body"
    store.close()


def test_bm25_predicate_escapes_single_quotes(tmp_path):
    """The escaping invariant lives in one place and survives extraction."""
    from esdc.corpus.store import CorpusStore

    store = CorpusStore(
        db_path=tmp_path / "q.duckdb",
        embedder=FakeEmbedder(),
        sqlite_path=tmp_path / "q.sqlite",
    )
    match_expr, clause, params = store._bm25_predicate("O'Brien", None)

    assert "O''Brien" in match_expr
    assert "match_bm25" in match_expr
    assert clause == ""
    assert params == []
    store.close()


def test_bm25_predicate_includes_filter_clause(tmp_path):
    from esdc.corpus.store import CorpusStore

    store = CorpusStore(
        db_path=tmp_path / "q2.duckdb",
        embedder=FakeEmbedder(),
        sqlite_path=tmp_path / "q2.sqlite",
    )
    _, clause, params = store._bm25_predicate("pod", {"doc_type": "surat"})

    assert "d.doc_type = ?" in clause
    assert params == ["surat"]
    store.close()


def test_hydrate_docs_returns_parsed_metadata(tmp_path):
    from esdc.corpus.chunker import Chunk
    from esdc.corpus.store import CorpusStore

    store = CorpusStore(
        db_path=tmp_path / "h.duckdb",
        embedder=FakeEmbedder(),
        sqlite_path=tmp_path / "h.sqlite",
    )
    store.ensure_tables()
    store.insert_document(
        {
            "doc_id": "d1", "file_name": "a.pdf", "file_path": "/x/a.pdf",
            "file_hash": "ab" * 32, "doc_type": "surat",
            "field_name": ["Duri"], "doc_date": "2026-01-05",
            "markdown": "# x", "extraction_method": "docling",
            "embedding_model": "fake-model",
        },
        [Chunk(0, None, "isi")],
    )
    store.refresh_mirror()

    docs = store._hydrate_docs(["d1"])

    assert docs["d1"]["file_name"] == "a.pdf"
    assert docs["d1"]["field_name"] == ["Duri"]  # parsed, not a JSON string
    store.close()


def test_corpus_unavailable_on_empty_corpus(tmp_path):
    from esdc.corpus.store import CorpusStore

    store = CorpusStore(
        db_path=tmp_path / "u.duckdb",
        embedder=FakeEmbedder(),
        sqlite_path=tmp_path / "u.sqlite",
    )
    store.ensure_tables()

    payload = store._corpus_unavailable()

    assert payload is not None
    assert payload["status"] == "not_available"
    store.close()


@pytest.fixture
def agg_store(tmp_path):
    """Corpus where 'separator' appears in one body and one subject only."""
    from esdc.corpus.chunker import Chunk
    from esdc.corpus.store import CorpusStore

    store = CorpusStore(
        db_path=tmp_path / "agg.duckdb",
        embedder=FakeEmbedder(),
        sqlite_path=tmp_path / "agg.sqlite",
    )
    store.ensure_tables()
    base = {
        "file_path": "/x/a.pdf", "doc_type": "surat", "doc_date": "2026-01-05",
        "extraction_method": "docling", "embedding_model": "fake-model",
    }
    # body mentions separator, twice -> must still count once
    store.insert_document(
        {**base, "doc_id": "body1", "file_name": "b1.pdf", "file_hash": "a" * 64,
         "subject": "Surat biasa", "markdown": "# x"},
        [Chunk(0, None, "pemasangan separator di lapangan"),
         Chunk(1, None, "separator kedua disebut lagi")],
    )
    # subject mentions separator, body does not -> must NOT count
    store.insert_document(
        {**base, "doc_id": "subj1", "file_name": "s1.pdf", "file_hash": "b" * 64,
         "subject": "Pengadaan separator", "markdown": "# y"},
        [Chunk(0, None, "isi tentang pompa dan pipa")],
    )
    # unrelated
    store.insert_document(
        {**base, "doc_id": "other", "file_name": "o.pdf", "file_hash": "c" * 64,
         "subject": "Lain lain", "doc_type": "mom", "markdown": "# z"},
        [Chunk(0, None, "rapat bulanan")],
    )
    store.rebuild_indexes()
    store.refresh_mirror()
    yield store
    store.close()


def test_aggregate_keyword_count_dedups_and_ignores_metadata_prefix(agg_store):
    result = agg_store.aggregate("separator", mode="count", match="keyword")

    assert result["status"] == "success"
    assert result["match"] == "keyword"
    assert result["approximate"] is False
    # body1 counted once despite two matching chunks; subj1 excluded
    assert result["count"] == 1
    assert result["doc_ids"] == ["body1"]


def test_aggregate_keyword_is_conjunctive(agg_store):
    result = agg_store.aggregate("separator pompa", mode="count", match="keyword")

    # no single chunk contains both terms
    assert result["count"] == 0
    assert result["status"] == "no_results"


def test_aggregate_keyword_conjunction_is_within_one_chunk(agg_store):
    """Terms split across two chunks of the same document do NOT match.

    body1 has 'separator' in chunk 0 and 'kedua' in chunk 1. Documented
    behavior, not an accident — see the plan's "embed_text trap" section.
    """
    result = agg_store.aggregate("separator kedua", mode="count", match="keyword")

    assert result["count"] == 1  # chunk 1 holds 'separator kedua' together
    assert (
        agg_store.aggregate("pemasangan kedua", mode="count", match="keyword")["count"]
        == 0
    )  # 'pemasangan' is chunk 0, 'kedua' is chunk 1


def test_aggregate_metadata_only_counts_by_filter(agg_store):
    result = agg_store.aggregate(None, mode="count", filters={"doc_type": "surat"})

    assert result["count"] == 2
    assert result["approximate"] is False
    assert result["match"] == "metadata"


def test_aggregate_semantic_mode_is_flagged_and_ranked(agg_store):
    result = agg_store.aggregate(
        "separator", mode="count", match="semantic", semantic_candidates=3
    )

    assert result["approximate"] is True
    assert result["match"] == "semantic"
    assert result["count"] == len(result["semantic_candidates"])
    assert result["count"] <= 3


def test_aggregate_semantic_candidates_not_limit_controls_the_ranking(tmp_path):
    """`limit` pages `list` mode only; it must never gate `mode="count"`.

    Regression guard for the top-K-pool bug: on a corpus bigger than the
    old `max(limit * 4, 200)` pool, a small `limit` used to silently
    shrink the semantic count. Uses filters to force _vector_search's
    JOIN/sequential-scan branch (deterministic, unlike the unfiltered
    HNSW-index path which is approximate even at large LIMITs).
    """
    from esdc.corpus.chunker import Chunk
    from esdc.corpus.store import CorpusStore

    store = CorpusStore(
        db_path=tmp_path / "big.duckdb",
        embedder=FakeEmbedder(),
        sqlite_path=tmp_path / "big.sqlite",
    )
    store.ensure_tables()
    base = {
        "file_path": "/x/a.pdf", "doc_type": "surat", "doc_date": "2026-01-05",
        "extraction_method": "docling", "embedding_model": "fake-model",
    }
    letters = "abcdefghijklmnopqrstuvwxyz"

    def word(i):
        return "".join(letters[(i + k) % 26] for k in range(3 + (i % 5)))

    n_docs = 250
    for i in range(n_docs):
        store.insert_document(
            {**base, "doc_id": f"doc{i}", "file_name": f"f{i}.pdf",
             "file_hash": format(i, "064x"), "subject": "Subjek acak",
             "markdown": "# x"},
            [Chunk(0, None, f"separator {word(i)}")],
        )
    store.rebuild_indexes()
    store.refresh_mirror()

    filters = {"doc_type": "surat"}
    a = store.aggregate(
        "separator", mode="count", match="semantic",
        semantic_candidates=5, limit=1, filters=filters,
    )
    b = store.aggregate(
        "separator", mode="count", match="semantic",
        semantic_candidates=5, limit=1000, filters=filters,
    )

    assert a["count"] == b["count"] == 5, "limit must not touch the ranking size"
    store.close()


def test_aggregate_list_mode_returns_documents_capped_by_limit(agg_store):
    result = agg_store.aggregate(None, mode="list", limit=2)

    assert result["count"] == 3          # full total, uncapped
    assert len(result["documents"]) == 2  # page capped by limit
    assert {"doc_id", "file_name", "subject"} <= set(result["documents"][0])


def test_aggregate_scalar_facet_sums_to_total(agg_store):
    result = agg_store.aggregate(None, mode="count", group_by="doc_type")

    facets = result["facets"]
    assert facets["dimension"] == "doc_type"
    assert facets["multi_valued"] is False
    assert sum(facets["values"].values()) == result["count"]
    assert facets["values"]["surat"] == 2


def test_aggregate_json_facet_is_flagged_multi_valued(agg_store):
    result = agg_store.aggregate(None, mode="count", group_by="doc_topic")

    assert result["facets"]["multi_valued"] is True


def test_aggregate_semantic_returns_ranked_scored_candidates(agg_store):
    rows = agg_store._aggregate_semantic("separator", candidates=3, filters=None)

    assert len(rows) <= 3
    assert {"doc_id", "similarity", "snippet"} <= set(rows[0])
    scores = [r["similarity"] for r in rows]
    assert scores == sorted(scores, reverse=True), "must be ranked, best first"


def test_aggregate_semantic_candidates_caps_the_list(agg_store):
    one = agg_store._aggregate_semantic("separator", candidates=1, filters=None)
    two = agg_store._aggregate_semantic("separator", candidates=2, filters=None)

    assert len(one) == 1
    assert len(two) == 2
    assert one[0]["doc_id"] == two[0]["doc_id"], "same ranking, just truncated"


def test_aggregate_semantic_honours_filters(agg_store):
    """A filter that excludes everything yields no candidates."""
    rows = agg_store._aggregate_semantic(
        "separator", candidates=10, filters={"doc_type": "zzz-nope"}
    )

    assert rows == []


def test_aggregate_hybrid_count_is_the_exact_total(agg_store):
    """Hybrid never inflates count with semantic-only documents."""
    kw = agg_store.aggregate("separator", mode="count", match="keyword")
    hy = agg_store.aggregate("separator", mode="count", match="hybrid")

    assert hy["count"] == kw["count"]
    assert hy["approximate"] is False, "the headline total is still exact"
    assert hy["doc_ids"] == kw["doc_ids"]


def test_aggregate_hybrid_provenance_reports_only_stable_numbers(agg_store):
    """Provenance carries the exact total and the ranking size, nothing else."""
    result = agg_store.aggregate("separator", mode="count", match="hybrid")

    p = result["provenance"]
    assert set(p) == {"exact_total", "semantic_extra", "semantic_extra_is_a_ranking"}
    assert p["exact_total"] == result["count"]
    assert p["semantic_extra_is_a_ranking"] is True


def test_aggregate_hybrid_provenance_exposes_no_knob_dependent_counts(agg_store):
    """Only the ranking size may move with semantic_candidates; the total may not.

    A keyword_only/both split was deliberately removed: measured on the
    live corpus it swung 29/5 to 7/27 purely with semantic_candidates
    while count stayed at 34, which is an artifact of the parameter, not
    a property of the corpus — the same trap similarity_threshold was.
    """
    small = agg_store.aggregate(
        "separator", mode="count", match="hybrid", semantic_candidates=1
    )
    large = agg_store.aggregate(
        "separator", mode="count", match="hybrid", semantic_candidates=50
    )

    assert small["count"] == large["count"]
    assert small["provenance"]["exact_total"] == large["provenance"]["exact_total"]
    moving = {
        k
        for k in small["provenance"]
        if small["provenance"][k] != large["provenance"][k]
    }
    assert moving <= {"semantic_extra"}, f"knob-dependent keys leaked: {moving}"


def test_aggregate_hybrid_semantic_candidates_exclude_keyword_hits(agg_store):
    """The tail is what keyword missed; exact hits are never repeated in it."""
    result = agg_store.aggregate("separator", mode="count", match="hybrid")

    kw_ids = set(result["doc_ids"])
    tail_ids = {c["doc_id"] for c in result["semantic_candidates"]}
    assert not (kw_ids & tail_ids)
    assert len(tail_ids) == result["provenance"]["semantic_extra"]


@pytest.fixture
def many_docs_store(tmp_path):
    """5 documents with long 'separator' bodies, for truncation tests.

    Enough documents to exercise limit-driven truncation in both list and
    count modes, and a chunk long enough (~1900 chars) to prove snippet
    trimming actually shrinks it.
    """
    from esdc.corpus.chunker import Chunk
    from esdc.corpus.store import CorpusStore

    store = CorpusStore(
        db_path=tmp_path / "many.duckdb",
        embedder=FakeEmbedder(),
        sqlite_path=tmp_path / "many.sqlite",
    )
    store.ensure_tables()
    base = {
        "file_path": "/x/a.pdf", "doc_type": "surat", "doc_date": "2026-01-05",
        "extraction_method": "docling", "embedding_model": "fake-model",
    }
    for i in range(5):
        store.insert_document(
            {**base, "doc_id": f"d{i}", "file_name": f"f{i}.pdf",
             "file_hash": format(i, "064x"), "subject": "Surat",
             "markdown": "# x"},
            [Chunk(0, None, f"pemasangan separator unit {i} " * 60)],
        )
    store.rebuild_indexes()
    store.refresh_mirror()
    yield store
    store.close()


def test_aggregate_list_mode_truncated_flags_and_note(many_docs_store):
    result = many_docs_store.aggregate(
        "separator", mode="list", match="keyword", limit=2
    )

    assert result["count"] == 5  # exhaustive total, unaffected by limit
    assert result["returned"] == 2
    assert result["truncated"] is True
    assert len(result["documents"]) == 2
    assert "note" in result
    assert "5" in result["note"]


def test_aggregate_list_mode_not_truncated_has_no_misleading_note(many_docs_store):
    result = many_docs_store.aggregate(
        "separator", mode="list", match="keyword", limit=10
    )

    assert result["count"] == 5
    assert result["returned"] == 5
    assert result["truncated"] is False
    assert "note" not in result


def test_aggregate_count_mode_doc_ids_bounded_by_limit(many_docs_store):
    result = many_docs_store.aggregate(
        "separator", mode="count", match="keyword", limit=2
    )

    assert result["count"] == 5  # exhaustive, unaffected by limit
    assert len(result["doc_ids"]) == 2  # bounded
    assert result["returned"] == 2
    assert result["truncated"] is True
    assert "note" in result


def test_aggregate_count_mode_doc_ids_not_truncated_when_under_limit(many_docs_store):
    result = many_docs_store.aggregate(
        "separator", mode="count", match="keyword", limit=50
    )

    assert result["count"] == 5
    assert len(result["doc_ids"]) == 5
    assert result["returned"] == 5
    assert result["truncated"] is False
    assert "note" not in result


def test_aggregate_matched_snippet_is_trimmed_and_contains_term(many_docs_store):
    result = many_docs_store.aggregate(
        "separator", mode="list", match="keyword", limit=1
    )

    snippet = result["documents"][0]["matched_snippet"]
    assert snippet is not None
    assert len(snippet) < 500  # materially shorter than the ~1900-char chunk
    assert "separator" in snippet.lower()


def test_aggregate_on_empty_corpus_is_not_available(tmp_path):
    from esdc.corpus.store import CorpusStore

    store = CorpusStore(
        db_path=tmp_path / "empty2.duckdb",
        embedder=FakeEmbedder(),
        sqlite_path=tmp_path / "empty2.sqlite",
    )
    store.ensure_tables()

    result = store.aggregate("apapun", mode="count")

    assert result["status"] == "not_available"


def test_build_filter_clause_supports_text_substring_columns(tmp_path):
    """sender/recipient/subject/doc_number filter by case-insensitive substring.

    Stored values are long institutional strings ("PERTAMINA BADAN
    PEMBINAAN PENGUSAHAAN KONTRAKTOR ASING..."), so an exact match would
    never hit; substring is the only usable form.
    """
    store = CorpusStore(
        db_path=tmp_path / "tf.duckdb",
        embedder=FakeEmbedder(),
        sqlite_path=tmp_path / "tf.sqlite",
    )
    clause, params = store._build_filter_clause(
        {"sender": "pertamina", "recipient": "skk"}, "d"
    )

    assert "d.sender ILIKE" in clause
    assert "d.recipient ILIKE" in clause
    assert params == ["pertamina", "skk"]
    store.close()


def test_build_filter_clause_supports_pod_name(tmp_path):
    """pod_name is a JSON array column like wk_name; it filters the same way."""
    store = CorpusStore(
        db_path=tmp_path / "pn.duckdb",
        embedder=FakeEmbedder(),
        sqlite_path=tmp_path / "pn.sqlite",
    )
    clause, params = store._build_filter_clause({"pod_name": "Bekasap"}, "d")

    assert "json_each(d.pod_name)" in clause
    assert params == ["Bekasap"]
    store.close()


def test_build_filter_clause_unchanged_without_new_keys(tmp_path):
    """Existing callers (search, find_doc_ids) see byte-identical output."""
    store = CorpusStore(
        db_path=tmp_path / "un.duckdb",
        embedder=FakeEmbedder(),
        sqlite_path=tmp_path / "un.sqlite",
    )
    clause, params = store._build_filter_clause(
        {"doc_type": "surat", "field_name": "Duri"}, "d"
    )

    assert clause.startswith(" AND d.doc_type = ?")
    assert "sender" not in clause
    assert params == ["surat", "Duri"]
    store.close()
    store.close()
