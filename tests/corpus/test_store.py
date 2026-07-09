from pathlib import Path

import pytest

from esdc.corpus.chunker import Chunk
from esdc.corpus.store import CorpusStore


class FakeEmbedder:
    model = "fake-embed"

    def generate_embedding(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3]

    def generate_embeddings_batch(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3] for _ in texts]


class FakeEmbedder2:
    model = "other-model"

    def generate_embedding(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3]

    def generate_embeddings_batch(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3] for _ in texts]


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
