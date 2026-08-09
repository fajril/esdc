"""Stale embed_text detection and repair after entity edits."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from esdc.corpus.chunker import Chunk
from esdc.corpus.store import CorpusStore


class FakeEmbedder:
    model = "fake-model"

    def generate_embedding(self, text: str) -> list[float]:
        return [float(len(text) % 7), 1.0, 0.0, 0.5]

    def generate_embeddings_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.generate_embedding(t) for t in texts]


DOC = {
    "doc_id": "d1",
    "file_name": "s.pdf",
    "file_path": "/x/s.pdf",
    "file_hash": "ab" * 32,
    "doc_type": "surat",
    "doc_topic": ["pod"],
    "doc_date": "2026-01-05",
    "subject": "Persetujuan POD",
    "field_name": ["Duri"],
    "markdown": "# Surat\nisi surat",
    "extraction_method": "docling",
    "embedding_model": "fake-model",
}


def _edit_field_name(store: CorpusStore, doc_id: str, value: str) -> None:
    """Write an entity edit the way every real caller does.

    SQLite is the truth; the DuckDB `documents` copy the detector reads is
    rebuilt wholesale from it. Writing SQLite and refreshing is what the
    portal save and the commit batch both do — a test that UPDATEs the
    DuckDB table directly would be testing a path no caller takes.
    """
    sconn = store._get_sqlite()
    with sconn:
        sconn.execute(
            "UPDATE documents SET field_name = ? WHERE doc_id = ?", (value, doc_id)
        )
    store.refresh_mirror()


@pytest.fixture
def store(tmp_path: Path) -> Iterator[CorpusStore]:
    s = CorpusStore(
        db_path=tmp_path / "e.duckdb",
        embedder=FakeEmbedder(),
        sqlite_path=tmp_path / "e.sqlite",
    )
    s.ensure_tables()
    s.insert_document(DOC, [Chunk(0, "Surat", "isi surat")])
    # insert_document writes SQLite + DuckDB chunks, never the DuckDB
    # `documents` row (store.py:514). Without this refresh the detector's
    # JOIN has nothing on the documents side and every assertion below
    # passes vacuously. Same reason tests/corpus/test_chat_tools.py:105
    # refreshes after its inserts.
    s.refresh_mirror()
    yield s
    s.close()


def test_stale_embed_docs_empty_right_after_ingest(store: CorpusStore):
    assert store.stale_embed_docs() == []


def test_stale_embed_docs_flags_document_after_entity_edit(store: CorpusStore):
    _edit_field_name(store, "d1", '["Duri Field"]')

    assert store.stale_embed_docs() == ["d1"]


def test_stale_embed_docs_flags_a_name_shortened_to_its_own_prefix(
    store: CorpusStore,
):
    """'Dur' is a string prefix of the stored 'Duri' — but not of the header.

    Ingest wrote the prefix `surat | pod | Persetujuan POD | Duri`. Renaming
    the field to `Dur` leaves the OLD embed_text still starting with the new
    prefix, so a bare startswith() check would call this document current.
    Comparing the whole header segment is what catches it.
    """
    _edit_field_name(store, "d1", '["Dur"]')

    assert store.stale_embed_docs() == ["d1"]


def test_run_reembed_documents_refreshes_the_prefix(store: CorpusStore):
    from esdc.corpus.pipeline import run_reembed_documents

    _edit_field_name(store, "d1", '["Duri Field"]')
    assert store.stale_embed_docs() == ["d1"]

    report = run_reembed_documents(["d1"], store=store)

    assert report.processed == ["s.pdf"]
    assert report.failed == {}
    assert store.stale_embed_docs() == []
    embed_row = (
        store._get_connection()
        .execute("SELECT embed_text FROM document_chunks WHERE doc_id = 'd1'")
        .fetchone()
    )
    assert embed_row is not None
    embed_text = embed_row[0]
    assert "Duri Field" in embed_text
    assert "isi surat" in embed_text  # chunk_text preserved


def test_commit_reembeds_documents_whose_blank_entities_it_merged(
    tmp_path: Path, monkeypatch
):
    """A merge-only commit batch must not leave the chunks on the old prefix.

    Setup copied verbatim from
    tests/corpus/test_pipeline.py::test_commit_already_committed_fills_blank_and_preserves_portal_edit
    (search that module for "entities merged") — only the assertions differ:
    this test checks embed_text convergence instead of the merged column
    values.
    """
    from esdc.corpus import pipeline
    from tests.corpus.test_pipeline import (
        make_sidecar,
        make_store,
        patch_entity_resolver,
        patch_store_factory,
    )

    store = make_store(tmp_path)
    store.ensure_tables()
    patch_store_factory(monkeypatch, store)
    patch_entity_resolver(
        monkeypatch,
        matches={
            "Minas": {"entity_type": "field_name", "name": "Minas", "confidence": 1.0}
        },
    )

    file_hash = "cc" * 32
    doc_id = file_hash[:16]
    make_sidecar(
        tmp_path,
        "doc.pdf",
        reviewed=True,
        file_hash=file_hash,
        field_name=None,
    )
    report1 = pipeline.run_commit([tmp_path])
    assert report1.processed == ["doc.corpus.md"]
    doc = store.get_document(doc_id)
    assert doc is not None
    assert doc["field_name"] is None

    # Re-extract updates the same sidecar: field_name now populated.
    make_sidecar(
        tmp_path,
        "doc.pdf",
        reviewed=True,
        file_hash=file_hash,
        field_name="Minas",
    )

    report2 = pipeline.run_commit([tmp_path])

    assert any("entities merged" in p for p in report2.processed)
    assert store.stale_embed_docs() == []
    embed_row = (
        store._get_connection()
        .execute("SELECT embed_text FROM document_chunks WHERE doc_id = ?", (doc_id,))
        .fetchone()
    )
    assert embed_row is not None
    embed_text = embed_row[0]
    assert "Minas" in embed_text
    store.close()
