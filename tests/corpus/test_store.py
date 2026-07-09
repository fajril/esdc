from pathlib import Path

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
    "file_hash": "ab" * 32, "doc_type": "surat",
    "doc_number": "SRT-1", "doc_date": "2026-01-05", "subject": "Persetujuan",
    "sender": "SKK", "recipient": "KKKS", "doc_level": "field",
    "wk_name": "Rokan", "field_name": "Duri", "project_name": None,
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
        s2.ensure_tables()
    s2.close()


def test_search_not_available(tmp_path: Path):
    s = CorpusStore(db_path=tmp_path / "empty.duckdb", embedder=FakeEmbedder())
    result = s.search("anything", limit=5, filters=None)
    assert result["status"] == "not_available"
    s.close()
