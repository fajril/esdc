# tests/corpus/test_store_probe.py
"""corpus_meta parity probe: seeded on write paths, enforced on mismatch."""

from __future__ import annotations

import hashlib
import json
import os

import duckdb
import pytest

from esdc import embedders
from esdc.corpus.chunker import Chunk
from esdc.corpus.store import CorpusStore


@pytest.fixture(autouse=True)
def _default_embedding_signature(monkeypatch):
    """Probe-memo tests run against one explicit effective embedding config."""
    monkeypatch.setattr(
        embedders,
        "config_signature",
        lambda backend=None: (
            "ollama",
            "http://localhost:11434",
            "qwen3-embedding-0.6b-q8_0",
            "",
        ),
    )


class _FixedEmbedder:
    model = "qwen3-embedding-0.6b-q8_0"

    def __init__(self, vec):
        self._vec = list(vec)

    def generate_embedding(self, text):
        return list(self._vec)

    def generate_embeddings_batch(self, texts):
        return [list(self._vec) for _ in texts]


class _CountingEmbedder:
    """Fixed-vector client that records what it was asked to embed."""

    def __init__(self, vec, name="qwen3-embedding-0.6b-q8_0"):
        self._vec = list(vec)
        self.model = name
        self.calls: list[tuple[str, str]] = []

    def generate_embedding(self, text):
        self.calls.append(
            ("probe", text) if text == embedders.PROBE_TEXT else ("query", text)
        )
        return list(self._vec)

    def generate_embeddings_batch(self, texts):
        if not texts:
            return []
        self.calls.append(("batch", texts[0]))
        return [list(self._vec) for _ in texts]

    def check_available(self) -> str | None:
        return None


class _CacheableEmbedder(_CountingEmbedder):
    """A client whose memo identity is stable across instances (HTTP-client shape)."""

    def __init__(self, vec, name="qwen3-embedding-0.6b-q8_0"):
        super().__init__(vec, name)
        self.cache_key = ("test", name, tuple(self._vec))


def _store(tmp_path, vec, name="c.duckdb"):
    return CorpusStore(
        db_path=tmp_path / name, embedder=_FixedEmbedder(vec), read_only=False
    )


def test_probe_column_created(tmp_path):
    s = _store(tmp_path, [1.0, 0.0, 0.0])
    s.ensure_tables(validate_model=True)
    cols = [
        r[0]
        for r in s._get_connection()
        .execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = 'corpus_meta'"
        )
        .fetchall()
    ]
    s.close()
    assert "probe_vec" in cols


def test_probe_seeded_on_first_write_path(tmp_path):
    s = _store(tmp_path, [1.0, 0.0, 0.0])
    s.ensure_tables(validate_model=True)
    stored_row = (
        s._get_connection().execute("SELECT probe_vec FROM corpus_meta").fetchone()
    )
    s.close()
    assert stored_row is not None
    stored = stored_row[0]
    assert json.loads(stored) == [1.0, 0.0, 0.0]


def test_matching_backend_accepts_without_rewriting_probe(tmp_path):
    s = _store(tmp_path, [1.0, 0.0, 0.0])
    s.ensure_tables(validate_model=True)
    before = (
        s._get_connection().execute("SELECT probe_vec FROM corpus_meta").fetchone()
    )
    s.close()
    assert before is not None

    s2 = _store(tmp_path, [0.999, 0.001, 0.0])
    s2.ensure_tables(validate_model=True)
    after = (
        s2._get_connection().execute("SELECT probe_vec FROM corpus_meta").fetchone()
    )
    s2.close()
    assert after is not None
    assert after[0] == before[0]
    assert json.loads(after[0]) == [1.0, 0.0, 0.0]


def test_drifted_backend_is_rejected(tmp_path):
    s = _store(tmp_path, [1.0, 0.0, 0.0])
    s.ensure_tables(validate_model=True)
    s.close()
    s2 = _store(tmp_path, [0.0, 1.0, 0.0])
    with pytest.raises(ValueError, match="parity probe failed"):
        s2.ensure_tables(validate_model=True)
    s2.close()


def test_read_path_leaves_probe_untouched_on_drift(tmp_path):
    s = _store(tmp_path, [1.0, 0.0, 0.0])
    s.ensure_tables(validate_model=True)
    before = (
        s._get_connection().execute("SELECT probe_vec FROM corpus_meta").fetchone()
    )
    s.close()
    assert before is not None

    s2 = _store(tmp_path, [0.0, 1.0, 0.0])
    s2.ensure_tables()  # validate_model=False -> read path, no probe check
    after = (
        s2._get_connection().execute("SELECT probe_vec FROM corpus_meta").fetchone()
    )
    s2.close()
    assert after is not None
    assert after[0] == before[0]
    assert json.loads(after[0]) == [1.0, 0.0, 0.0]


def test_set_meta_rebaselines_probe(tmp_path):
    s = _store(tmp_path, [1.0, 0.0, 0.0])
    s.ensure_tables(validate_model=True)
    s.close()
    s2 = _store(tmp_path, [0.0, 1.0, 0.0])
    s2.set_meta("qwen3-embedding-0.6b-q8_0", 3)
    stored_row = (
        s2._get_connection().execute("SELECT probe_vec FROM corpus_meta").fetchone()
    )
    s2.close()
    assert stored_row is not None
    stored = stored_row[0]
    assert json.loads(stored) == [0.0, 1.0, 0.0]


DOC = {
    "doc_id": "probe-doc",
    "file_name": "s.pdf",
    "file_path": "/x/s.pdf",
    "file_hash": "ab" * 32,
    "doc_type": "surat",
    "doc_topic": None,
    "doc_number": "SRT-1",
    "doc_date": "2026-01-05",
    "subject": "Persetujuan",
    "sender": "SKK",
    "recipient": "KKKS",
    "doc_level": "field",
    "wk_name": "Rokan",
    "field_name": "Duri",
    "project_name": None,
    "pod_name": ["POD Mengoepeh"],
    "suggested_pod_ids": ["PL-2003-0005-3-2-0"],
    "raw_entities": "{}",
    "metadata": "{}",
    "markdown": "# Surat\nisi surat persetujuan",
    "extraction_method": "native",
    "page_count": 1,
}


def _writer(tmp_path, embedder, name, sqlite_path=None):
    return CorpusStore(
        db_path=tmp_path / name,
        embedder=embedder,
        sqlite_path=sqlite_path or tmp_path / f"{name}.sqlite",
        read_only=False,
    )


def _stored_probe(db_path, table="corpus_meta"):
    conn = duckdb.connect(str(db_path), read_only=True)
    row = conn.execute(f"SELECT probe_vec FROM {table}").fetchone()
    conn.close()
    if row is None or row[0] is None:
        return None
    return json.loads(row[0]) if isinstance(row[0], str) else list(row[0])


def _vector(value) -> list[float]:
    """Read a stored vector however DuckDB hands it back (JSON text or list)."""
    return json.loads(value) if isinstance(value, str) else list(value)


def test_probe_match_keeps_vectors(tmp_path):
    """A client that reproduces the stored probe leaves the vector space untouched.

    The reader is a different service reporting a different wire name. Because its
    probe agrees with the stored one, the stored vectors stay usable as they are:
    a match may keep them, it never rewrites them or the pin that names them.
    """
    db_path = tmp_path / "keep.duckdb"
    writer = _writer(
        tmp_path, _CountingEmbedder([1.0, 0.0, 0.0], "space-a"), "keep.duckdb"
    )
    writer.ensure_tables(validate_model=True)
    writer.insert_document(DOC, [Chunk(0, "Surat", "isi surat persetujuan")])
    writer.refresh_mirror()
    writer.close()

    before = hashlib.sha256(db_path.read_bytes()).hexdigest()
    reader = CorpusStore(
        db_path=db_path,
        embedder=_CountingEmbedder([0.999, 0.001, 0.0], "space-b"),
        read_only=True,
    )
    out = reader.search("persetujuan", limit=5)
    reader.close()
    after = hashlib.sha256(db_path.read_bytes()).hexdigest()

    assert out["status"] == "success", out
    assert after == before, "a query must not rewrite vectors or probe metadata"
    assert _stored_probe(db_path) == [1.0, 0.0, 0.0]
    conn = duckdb.connect(str(db_path), read_only=True)
    chunk_row = conn.execute("SELECT embedding FROM document_chunks").fetchone()
    assert chunk_row is not None
    stored_chunk = _vector(chunk_row[0])
    conn.close()
    assert stored_chunk == [1.0, 0.0, 0.0]


def test_probe_mismatch_refuses_write(tmp_path):
    """A write against an unproven space is refused before anything is written."""
    db_path = tmp_path / "write.duckdb"
    writer = _writer(
        tmp_path, _CountingEmbedder([1.0, 0.0, 0.0], "stored-service"), "write.duckdb"
    )
    writer.ensure_tables(validate_model=True)
    writer.insert_document(DOC, [Chunk(0, "Surat", "isi surat persetujuan")])
    writer.close()

    drifted = _CountingEmbedder([0.0, 1.0, 0.0], "other-service")
    second = _writer(tmp_path, drifted, "write.duckdb")
    with pytest.raises(ValueError, match="parity probe failed") as exc:
        second.insert_document(
            {**DOC, "doc_id": "probe-doc-2", "file_hash": "cd" * 32},
            [Chunk(0, "Surat", "isi lain")],
        )
    second.close()

    # The refusal names both spaces: the one the vectors came from and the one asked
    # to write into it.
    assert "stored-service" in str(exc.value)
    assert "other-service" in str(exc.value)

    conn = duckdb.connect(str(db_path), read_only=True)
    count_row = conn.execute("SELECT COUNT(*) FROM document_chunks").fetchone()
    conn.close()
    assert count_row is not None
    rows = count_row[0]
    assert rows == 1, "the refused write left no chunk behind"
    assert _stored_probe(db_path) == [1.0, 0.0, 0.0]
    # The refusal is a probe check, not an embedding of the document text.
    assert drifted.calls == [("probe", embedders.PROBE_TEXT)]


def test_probe_mismatch_refuses_query(tmp_path):
    """A query that would embed against a mismatched space is refused, not run."""
    db_path = tmp_path / "query.duckdb"
    writer = _writer(
        tmp_path, _CountingEmbedder([1.0, 0.0, 0.0], "stored-service"), "query.duckdb"
    )
    writer.ensure_tables(validate_model=True)
    writer.insert_document(DOC, [Chunk(0, "Surat", "isi surat persetujuan")])
    writer.close()

    drifted = _CountingEmbedder([0.0, 1.0, 0.0], "other-service")
    reader = CorpusStore(db_path=db_path, embedder=drifted, read_only=True)
    out = reader.search("persetujuan", limit=5)
    reader.close()

    assert out["status"] == "error", out
    assert "other-service" in out["message"]
    assert "stored-service" in out["message"]
    # Refused before the query was embedded: only the probe was asked for.
    assert drifted.calls == [("probe", embedders.PROBE_TEXT)]
    assert _stored_probe(db_path) == [1.0, 0.0, 0.0]


def test_query_does_not_write_probe_metadata(tmp_path):
    """A query reads the probe; it never repairs it as a side effect."""
    db_path = tmp_path / "no-write.duckdb"
    writer = _writer(
        tmp_path,
        _CountingEmbedder([1.0, 0.0, 0.0], "stored-service"),
        "no-write.duckdb",
    )
    writer.ensure_tables(validate_model=True)
    writer.insert_document(DOC, [Chunk(0, "Surat", "isi surat persetujuan")])
    writer.close()

    conn = duckdb.connect(str(db_path))
    conn.execute("UPDATE corpus_meta SET probe_vec = NULL")
    conn.close()

    before = hashlib.sha256(db_path.read_bytes()).hexdigest()
    reader = CorpusStore(
        db_path=db_path, embedder=_CountingEmbedder([1.0, 0.0, 0.0]), read_only=True
    )
    out = reader.search("persetujuan", limit=5)
    reader.close()

    assert out["status"] == "error", out
    assert _stored_probe(db_path) is None, "the query seeded the probe"
    assert hashlib.sha256(db_path.read_bytes()).hexdigest() == before


def test_index_without_probe_is_not_solemnized_by_seeding_from_new_service(tmp_path):
    """A probe from a new service cannot certify vectors produced elsewhere.

    Seeding is only the migration path for a space that is still empty. Once
    vectors exist, the store must refuse to guess that the new service produced
    them.
    """
    db_path = tmp_path / "foreign.duckdb"
    writer = _writer(
        tmp_path, _CountingEmbedder([0.0, 1.0, 0.0], "space-a"), "foreign.duckdb"
    )
    writer.ensure_tables(validate_model=True)
    writer.insert_document(DOC, [Chunk(0, "Surat", "isi surat persetujuan")])
    writer.close()

    conn = duckdb.connect(str(db_path))
    conn.execute("UPDATE corpus_meta SET probe_vec = NULL")
    conn.close()

    new_service = _CountingEmbedder([1.0, 0.0, 0.0], "new-service")
    store = _writer(tmp_path, new_service, "foreign.duckdb")
    with pytest.raises(ValueError, match="new-service") as exc:
        store.ensure_tables(validate_model=True)
    store.close()

    assert "space-a" in str(exc.value)
    assert _stored_probe(db_path) is None, "the new service seeded the probe"
    conn = duckdb.connect(str(db_path), read_only=True)
    count_row = conn.execute("SELECT COUNT(*) FROM document_chunks").fetchone()
    assert count_row is not None
    assert count_row[0] == 1
    chunk_row = conn.execute("SELECT embedding FROM document_chunks").fetchone()
    assert chunk_row is not None
    stored = _vector(chunk_row[0])
    conn.close()
    assert stored == [0.0, 1.0, 0.0], "the foreign vectors were rewritten"

    # An empty space is still a migration path: seeding is allowed there.
    fresh = _writer(tmp_path, new_service, "empty.duckdb")
    fresh.ensure_tables(validate_model=True)
    fresh.close()
    assert _stored_probe(tmp_path / "empty.duckdb") == [1.0, 0.0, 0.0]


def test_index_without_probe_is_not_solemnized_even_when_the_pin_is_model_id(
    tmp_path,
):
    """Every client reports MODEL_ID, so the pin name cannot authorize a seed."""
    db_path = tmp_path / "model-id.duckdb"
    writer = _writer(
        tmp_path,
        _CountingEmbedder([0.0, 1.0, 0.0], embedders.MODEL_ID),
        "model-id.duckdb",
    )
    writer.ensure_tables(validate_model=True)
    writer.insert_document(DOC, [Chunk(0, "Surat", "isi surat persetujuan")])
    writer.close()

    conn = duckdb.connect(str(db_path))
    conn.execute("UPDATE corpus_meta SET probe_vec = NULL")
    conn.close()

    new_service = _CountingEmbedder([1.0, 0.0, 0.0], embedders.MODEL_ID)

    reader = CorpusStore(db_path=db_path, embedder=new_service, read_only=True)
    out = reader.search("persetujuan", limit=5)
    reader.close()
    assert out["status"] == "error", out
    assert "reembed" in out["message"]
    assert new_service.calls == [], "a query must not ask the service for a probe"
    assert _stored_probe(db_path) is None

    writer2 = _writer(tmp_path, new_service, "model-id.duckdb")
    with pytest.raises(ValueError, match="reembed"):
        writer2.ensure_tables(validate_model=True)
    writer2.close()
    assert not [c for c in new_service.calls if c[0] == "probe"], (
        "a refused write must not seed a probe"
    )
    assert _stored_probe(db_path) is None


def test_validation_runs_once_per_config_signature_and_db_signature(
    tmp_path, monkeypatch
):
    """Probe validation is memoized per (config signature, db signature)."""
    state = {"model": "service-a"}
    monkeypatch.setattr(
        embedders,
        "config_signature",
        lambda backend=None: (
            "ollama",
            "http://localhost:11434",
            state["model"],
            "",
        ),
    )

    client = _CountingEmbedder([1.0, 0.0, 0.0], "service-a")
    store = _writer(tmp_path, client, "memo.duckdb")
    store.ensure_tables(validate_model=True)
    # ensure_tables asks the client for a sample vector to read the dimension; the
    # parity probe is the first validation.
    assert client.calls == [
        ("query", "test"),
        ("probe", embedders.PROBE_TEXT),
    ]

    store.insert_document(DOC, [Chunk(0, "Surat", "isi surat persetujuan")])
    store.refresh_mirror()
    store.search("persetujuan", limit=5)
    store.search("compressor", limit=5)
    # One validation for the whole (config, db) pair: the probe was asked for once,
    # and every later operation embedded only what it was actually asked about.
    assert [kind for kind, _ in client.calls] == [
        "query",  # ensure_tables reads the dimension from a sample vector
        "probe",
        "batch",
        "query",
        "query",
    ]

    # A different effective config is a different space: validate once, then memoize.
    state["model"] = "service-b"
    store.search("flaring", limit=5)
    # The memo is keyed by the effective config: a different config is a different
    # space, so it is validated once more, once.
    assert [text for kind, text in client.calls if kind == "probe"] == [
        embedders.PROBE_TEXT,
        embedders.PROBE_TEXT,
    ]
    store.search("metering", limit=5)
    assert [text for kind, text in client.calls if kind == "probe"] == [
        embedders.PROBE_TEXT,
        embedders.PROBE_TEXT,
    ]

    # A validation that never succeeds is not memoized as verified: it is re-checked.
    store.close()
    drifted = _CountingEmbedder([0.0, 1.0, 0.0], "service-b")
    reader = CorpusStore(
        db_path=tmp_path / "memo.duckdb", embedder=drifted, read_only=True
    )
    out = reader.search("flaring", limit=5)
    assert out["status"] == "error", out
    reader.search("metering", limit=5)
    assert [c for c in drifted.calls if c[0] == "probe"] == [
        ("probe", embedders.PROBE_TEXT),
        ("probe", embedders.PROBE_TEXT),
    ]
    reader.close()


def test_validation_is_shared_across_store_instances(tmp_path):
    """D2: a fresh CorpusStore per tool call must not re-probe the same space."""
    writer = _writer(
        tmp_path, _CacheableEmbedder([1.0, 0.0, 0.0]), "shared.duckdb"
    )
    writer.ensure_tables(validate_model=True)
    writer.insert_document(DOC, [Chunk(0, "Surat", "isi surat persetujuan")])
    writer.close()
    embedders.clear_probe_memo()

    first = _CacheableEmbedder([1.0, 0.0, 0.0])
    reader1 = CorpusStore(
        db_path=tmp_path / "shared.duckdb", embedder=first, read_only=True
    )
    out1 = reader1.search("persetujuan", limit=5)
    reader1.close()

    second = _CacheableEmbedder([1.0, 0.0, 0.0])
    reader2 = CorpusStore(
        db_path=tmp_path / "shared.duckdb", embedder=second, read_only=True
    )
    out2 = reader2.search("persetujuan", limit=5)
    reader2.close()

    assert out1["status"] == "success", out1
    assert out2["status"] == "success", out2
    assert [c for c in first.calls if c[0] == "probe"] == [
        ("probe", embedders.PROBE_TEXT)
    ]
    assert [c for c in second.calls if c[0] == "probe"] == [], (
        "a second store on the same config/DB paid for another probe"
    )


def test_client_replaced_when_effective_config_changes(tmp_path, monkeypatch):
    """A store must not keep using a client resolved for a superseded config."""
    resolved: list[tuple] = []
    clients = {"service-a": _CountingEmbedder([1.0, 0.0, 0.0], "service-a")}
    state = {"model": "service-a", "api_key": "", "host": "http://localhost:11434"}

    def fake_get_embedder(backend=None):
        sig = embedders.config_signature()
        resolved.append(sig)
        return clients[sig[2]]

    monkeypatch.setattr(embedders, "get_embedder", fake_get_embedder)
    monkeypatch.setattr(
        embedders,
        "config_signature",
        lambda backend=None: (
            "ollama",
            state["host"],
            state["model"],
            state["api_key"],
        ),
    )

    store = CorpusStore(db_path=tmp_path / "swap.duckdb", read_only=True)
    assert store.embedder is clients["service-a"]
    assert store.embedder is clients["service-a"]
    assert len(resolved) == 1  # resolved once, then reused

    state["model"] = "service-b"
    clients["service-b"] = _CountingEmbedder([1.0, 0.0, 0.0], "service-b")
    assert store.embedder is clients["service-b"]
    assert len(resolved) == 2

    # A credential change is a different effective config too, so the client that
    # was resolved under the old one is replaced on the next use.
    state["api_key"] = "sk-test"
    assert store.embedder is clients["service-b"]
    assert len(resolved) == 3
    assert store.embedder is clients["service-b"]
    assert len(resolved) == 3
    store.close()


def test_memo_invalidated_when_the_stored_pin_or_probe_changes_without_replacing_the_file(
    tmp_path,
):
    """A metadata row rewritten underneath a cached client re-validates the space.

    The DB half of the memo key is the file's inode, and `esdc fetch`/`esdc reload`
    publish in place: the same inode can hold a different pin, dimension or probe. So
    the key carries the facts the validation was computed from, read from the row, and
    the entry is recorded as the state stands after a successful check. Request counts
    are the proof; nothing here looks inside the cache.
    """
    client = _CountingEmbedder([1.0, 0.0, 0.0], "space-a")
    db_path = tmp_path / "in-place.duckdb"
    store = _writer(tmp_path, client, "in-place.duckdb")
    store.ensure_tables(validate_model=True)
    store.insert_document(DOC, [Chunk(0, "Surat", "isi surat persetujuan")])
    # One validation for the space as it was when the first write ran.
    assert [text for kind, text in client.calls if kind == "probe"] == [
        embedders.PROBE_TEXT
    ]
    inode = os.stat(db_path).st_ino

    # Same file, same inode: another writer re-pinned it and stored a probe that
    # still agrees, so the next operation checks the space again, once.
    conn = duckdb.connect(str(db_path))
    conn.execute(
        "UPDATE corpus_meta SET embedding_model = ?, probe_vec = ?",
        ["space-b", json.dumps([0.999, 0.001, 0.0])],
    )
    conn.close()

    out = store.search("persetujuan", limit=5)
    assert out["status"] == "success", out
    assert os.stat(db_path).st_ino == inode, "the row was changed by replacing the file"
    assert [text for kind, text in client.calls if kind == "probe"] == [
        embedders.PROBE_TEXT,
        embedders.PROBE_TEXT,
    ]

    # The recorded key is the state after the successful check, so operations on that
    # state do not pay for the probe again.
    store.search("compressor", limit=5)
    assert [text for kind, text in client.calls if kind == "probe"] == [
        embedders.PROBE_TEXT,
        embedders.PROBE_TEXT,
    ]

    # A probe that no longer agrees is caught the same way, and a check that did not
    # succeed is not remembered as verified: each refused read checks again.
    conn = duckdb.connect(str(db_path))
    conn.execute("UPDATE corpus_meta SET probe_vec = ?", [json.dumps([0.0, 1.0, 0.0])])
    conn.close()
    refused = store.search("flaring", limit=5)
    assert refused["status"] == "error", refused
    assert "space-b" in refused["message"]
    assert [text for kind, text in client.calls if kind == "probe"] == [
        embedders.PROBE_TEXT,
        embedders.PROBE_TEXT,
        embedders.PROBE_TEXT,
    ]
    store.search("metering", limit=5)
    assert [text for kind, text in client.calls if kind == "probe"] == [
        embedders.PROBE_TEXT,
        embedders.PROBE_TEXT,
        embedders.PROBE_TEXT,
        embedders.PROBE_TEXT,
    ]
    store.close()


def test_readonly_reader_does_not_seed_probe(tmp_path):
    """A NULL probe stays NULL through a read-only validation.

    Seeding is a writer concern (`ensure_tables(validate_model=True)`);
    the reader path must not repair the pin as a side effect.
    """
    db_path = tmp_path / "ro_probe.duckdb"
    s = _store(tmp_path, [1.0, 0.0, 0.0], name="ro_probe.duckdb")
    s.ensure_tables(validate_model=True)
    s.close()

    conn = duckdb.connect(str(db_path))
    conn.execute("UPDATE corpus_meta SET probe_vec = NULL")
    conn.close()

    reader = CorpusStore(
        db_path=db_path, embedder=_FixedEmbedder([1.0, 0.0, 0.0]), read_only=True
    )
    reader.validate_readiness("search")  # read path validates schema only
    reader.close()

    conn = duckdb.connect(str(db_path), read_only=True)
    stored = conn.execute("SELECT probe_vec FROM corpus_meta").fetchone()
    conn.close()
    assert stored is not None
    assert stored[0] is None


def test_corpus_probe_memo_does_not_validate_stale_semantic_metadata(tmp_path):
    """A warm corpus probe must not certify a different, stale semantic space."""
    from esdc.search.semantic_resolver import SemanticResolver

    db_path = tmp_path / "cross-space.duckdb"
    client = _CacheableEmbedder([1.0, 0.0, 0.0])
    conn = duckdb.connect(str(db_path))
    probe = json.dumps([1.0, 0.0, 0.0])
    conn.execute(
        "CREATE TABLE corpus_meta "
        "(embedding_model VARCHAR, dim INTEGER, probe_vec JSON)"
    )
    conn.execute(
        "INSERT INTO corpus_meta VALUES (?, ?, ?)",
        ["qwen3-embedding-0.6b-q8_0", 3, probe],
    )
    conn.execute(
        "CREATE TABLE semantic_meta "
        "(embedding_model VARCHAR, dim INTEGER, probe_vec JSON, "
        "content_version INTEGER)"
    )
    conn.execute(
        "INSERT INTO semantic_meta VALUES (?, ?, ?, ?)",
        [
            "qwen3-embedding-0.6b-q8_0",
            3,
            probe,
            SemanticResolver.NARRATIVE_CONTENT_VERSION - 1,
        ],
    )
    conn.close()

    embedders.clear_probe_memo()
    try:
        store = CorpusStore(db_path=db_path, embedder=client, read_only=True)
        assert store._probe_gate("corpus_meta", allow_seed=False) is None
        store.close()

        before = duckdb.connect(str(db_path))
        rows_before = before.execute("SELECT * FROM semantic_meta").fetchall()
        before.close()

        resolver = SemanticResolver(
            db_path=db_path, embedder=client, read_only=True
        )
        out = resolver._ensure_semantic_meta()
        resolver.close()

        assert out is not None
        assert out["status"] == "not_available"
        assert "current project narrative" in out["message"]
        assert "esdc reload --embeddings-only" in out["message"]

        after = duckdb.connect(str(db_path))
        rows_after = after.execute("SELECT * FROM semantic_meta").fetchall()
        after.close()
        assert rows_after == rows_before
    finally:
        embedders.clear_probe_memo()
