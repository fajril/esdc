# tests/search/test_semantic_resolver_embedder.py
"""SemanticResolver embeds queries locally and accepts an injected embedder."""

from __future__ import annotations

import inspect
import json
import os
from typing import cast

import duckdb

from esdc.embedders import InternalEmbedder
from esdc.search.semantic_resolver import SemanticResolver


class _FixedEmbedder:
    model = "qwen3-embedding-0.6b-q8_0"

    def __init__(self, vec):
        self._vec = list(vec)
        self.calls = 0

    def generate_embedding(self, text):
        self.calls += 1
        return list(self._vec)

    def generate_embeddings_batch(self, texts):
        return [list(self._vec) for _ in texts]


def test_defaults_to_internal_embedder(tmp_path):
    r = SemanticResolver(db_path=tmp_path / "x.duckdb")
    assert isinstance(r._embedder, InternalEmbedder)


def test_accepts_injected_embedder(tmp_path):
    e = _FixedEmbedder([1.0])
    r = SemanticResolver(db_path=tmp_path / "x.duckdb", embedder=e)
    assert r._embedder is e


def test_model_param_is_gone():
    assert "model" not in inspect.signature(SemanticResolver.__init__).parameters


def test_query_path_uses_the_injected_embedder(tmp_path):
    e = _FixedEmbedder([1.0])
    db_path = tmp_path / "x.duckdb"
    duckdb.connect(str(db_path)).close()  # empty DB: reader opens, table absent
    r = SemanticResolver(db_path=db_path, embedder=e)
    # No embeddings table exists, so search short-circuits before embedding.
    out = r.search_by_text("anything")
    assert out["status"] == "not_available"
    assert e.calls == 0


def _seeded(tmp_path, vec, dim=None):
    """A resolver whose semantic_meta is pinned by an embedder producing vec.

    ``dim`` defaults to ``len(vec)`` so the stored probe length always
    matches the recorded dimension; pass it explicitly only to build a
    deliberately inconsistent pin.
    """
    if dim is None:
        dim = len(vec)
    r = SemanticResolver(
        db_path=tmp_path / "s.duckdb", embedder=_FixedEmbedder(vec), read_only=False
    )
    conn = r._get_connection()
    conn.execute(
        "CREATE TABLE IF NOT EXISTS semantic_meta "
        "(embedding_model VARCHAR, dim INTEGER, probe_vec JSON)"
    )
    conn.execute(
        "INSERT INTO semantic_meta VALUES (?, ?, ?)",
        ["qwen3-embedding-0.6b-q8_0", dim, json.dumps(list(vec))],
    )
    return r


def test_meta_table_name():
    assert SemanticResolver.SEMANTIC_META == "semantic_meta"


def test_matching_embedder_passes_pin(tmp_path):
    r = _seeded(tmp_path, [1.0, 0.0])
    assert r._ensure_semantic_meta() is None
    r.close()


def test_drifted_embedder_returns_not_available_not_exception(tmp_path):
    """Same-dimension cosine drift must degrade, not raise.

    Both probes have length 2, so the only thing that can reject them is
    the cosine comparison against PROBE_TOLERANCE — an orthogonal vector
    scores 0.0, well below it. The message must name the cosine, proving
    the dimension guard did not fire instead.
    """
    r = _seeded(tmp_path, [1.0, 0.0])
    r._embedder = _FixedEmbedder([0.0, 1.0])
    out = r._ensure_semantic_meta()
    r.close()
    assert out is not None
    assert out["status"] == "not_available"
    assert out["results"] == []
    assert "cosine" in out["message"]
    assert "dimension" not in out["message"]
    assert "esdc reload" in out["message"]


def test_legacy_space_without_meta_degrades_then_writable_repair(tmp_path):
    """A missing semantic_meta is read-only degradation, not query seeding.

    The reader must NOT create or seed the table. Repair is the advertised
    maintenance writer path (what ``esdc reload --embeddings-only`` runs:
    ``build_embeddings_table``); after it, the reader validates.
    """
    db_path = tmp_path / "s.duckdb"
    duckdb.connect(str(db_path)).close()  # reader opens an empty DB; no table
    r = SemanticResolver(db_path=db_path, embedder=_FixedEmbedder([1.0, 0.0]))
    out = r._ensure_semantic_meta()
    assert out is not None
    assert out["status"] == "not_available"
    assert "esdc reload --embeddings-only" in out["message"]
    # Reader never seeded the pin: the table was not created.
    present = r._get_connection().execute(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_name = 'semantic_meta'"
    ).fetchall()
    assert present == []
    r.close()

    # Writable maintenance path repairs the pin.
    writer = SemanticResolver(
        db_path=db_path, embedder=_FixedEmbedder([1.0, 0.0]), read_only=False
    )
    assert writer.build_embeddings_table() is True
    writer.close()

    # Reader now validates against the repaired pin.
    reader = SemanticResolver(db_path=db_path, embedder=_FixedEmbedder([1.0, 0.0]))
    assert reader._ensure_semantic_meta() is None
    reader.close()


def test_legacy_meta_missing_probe_column_repairable_by_reload(tmp_path):
    """Gap 6: a legacy semantic_meta without probe_vec is repairable.

    The reader advertises ``esdc reload --embeddings-only``; the writer it
    runs (``build_embeddings_table``) must actually add the probe column and
    (re)seed the probe, after which the reader validates.
    """
    db_path = tmp_path / "s.duckdb"
    legacy = SemanticResolver(
        db_path=db_path, embedder=_FixedEmbedder([1.0, 0.0]), read_only=False
    )
    conn = legacy._get_connection()
    conn.execute(
        "CREATE TABLE semantic_meta (embedding_model VARCHAR, dim INTEGER)"
    )
    conn.execute(
        "INSERT INTO semantic_meta (embedding_model, dim) VALUES (?, ?)",
        ["qwen3-embedding-0.6b-q8_0", 2],
    )
    out = legacy._ensure_semantic_meta()
    assert out is not None
    assert out["status"] == "not_available"
    assert "esdc reload --embeddings-only" in out["message"]
    legacy.close()

    writer = SemanticResolver(
        db_path=db_path, embedder=_FixedEmbedder([1.0, 0.0]), read_only=False
    )
    assert writer.build_embeddings_table() is True
    writer.close()

    reader = SemanticResolver(db_path=db_path, embedder=_FixedEmbedder([1.0, 0.0]))
    assert reader._ensure_semantic_meta() is None
    reader.close()


def test_pin_check_embeds_probe_only_once(tmp_path):
    """A stable DB file signature means the memo hits on the second call."""
    r = _seeded(tmp_path, [1.0, 0.0])
    assert r._ensure_semantic_meta() is None
    assert r._pin_verified_sig == r._db_signature()
    embedder = cast(_FixedEmbedder, r._embedder)
    embedder.calls = 0
    assert r._ensure_semantic_meta() is None
    r.close()
    assert embedder.calls == 0


def test_modern_database_returns_results_read_only(tmp_path):
    """A properly pinned database still serves results on a read-only reader.

    Seeded by a writable resolver, then served by a separate read-only
    resolver: the pin validates (memo held) and the vector search returns
    the stored row. This is the "preserve modern-database result behavior"
    guard for the new read-only validation path.
    """
    db_path = tmp_path / "s.duckdb"
    writer = SemanticResolver(
        db_path=db_path, embedder=_FixedEmbedder([1.0, 0.0]), read_only=False
    )
    assert writer.build_embeddings_table() is True
    conn = writer._get_connection()
    conn.execute(
        "INSERT INTO project_embeddings "
        "(project_id, report_year, table_name, project_remarks, embedding) "
        "VALUES (?, ?, ?, ?, ?)",
        ["P1", 2024, "project_resources", "masalah teknis sumur", [1.0, 0.0]],
    )
    writer.close()

    reader = SemanticResolver(
        db_path=db_path, embedder=_FixedEmbedder([1.0, 0.0]), read_only=True
    )
    out = reader.search_by_text("masalah teknis", limit=5)
    reader.close()
    assert out["status"] == "success"
    assert out["results"][0]["project_id"] == "P1"


def test_pin_recheck_on_db_signature_change(tmp_path):
    """A changed DB signature forces the pin to be re-verified.

    Covers e.g. esdc fetch/reload replacing the DB file underneath a
    cached resolver, instead of trusting a stale memo.
    """
    r = _seeded(tmp_path, [1.0, 0.0])
    assert r._ensure_semantic_meta() is None
    embedder = cast(_FixedEmbedder, r._embedder)
    embedder.calls = 0

    # Simulate the DB file changing underneath the cached resolver by
    # bumping its mtime, so the recorded signature no longer matches.
    st = r._db_path.stat()
    os.utime(r._db_path, ns=(st.st_atime_ns, st.st_mtime_ns + 1))

    assert r._ensure_semantic_meta() is None
    r.close()
    assert embedder.calls == 1


def test_build_writes_the_pin(tmp_path):
    r = SemanticResolver(
        db_path=tmp_path / "s.duckdb",
        embedder=_FixedEmbedder([1.0, 0.0]),
        read_only=False,
    )
    r.build_embeddings_table()
    row = (
        r._get_connection()
        .execute("SELECT embedding_model, dim FROM semantic_meta")
        .fetchone()
    )
    assert row is not None
    r.close()
    assert row[0] == "qwen3-embedding-0.6b-q8_0"
    assert row[1] == 2
