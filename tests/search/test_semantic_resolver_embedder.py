# tests/search/test_semantic_resolver_embedder.py
"""SemanticResolver embeds queries locally and accepts an injected embedder."""
from __future__ import annotations

import inspect
import json
import os

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
    r = SemanticResolver(db_path=tmp_path / "x.duckdb", embedder=e)
    # No embeddings table exists, so search short-circuits before embedding.
    out = r.search_by_text("anything")
    assert out["status"] == "not_available"
    assert e.calls == 0


def _seeded(tmp_path, vec, dim=1):
    """A resolver whose semantic_meta is pinned by an embedder producing vec."""
    r = SemanticResolver(db_path=tmp_path / "s.duckdb", embedder=_FixedEmbedder(vec))
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
    r = _seeded(tmp_path, [1.0, 0.0])
    r._embedder = _FixedEmbedder([0.0, 1.0])
    out = r._ensure_semantic_meta()
    r.close()
    assert out is not None
    assert out["status"] == "not_available"
    assert out["results"] == []
    assert "esdc reload" in out["message"]


def test_legacy_space_without_meta_seeds_on_query(tmp_path):
    r = SemanticResolver(
        db_path=tmp_path / "s.duckdb", embedder=_FixedEmbedder([1.0, 0.0])
    )
    assert r._ensure_semantic_meta() is None
    stored = r._get_connection().execute(
        "SELECT probe_vec FROM semantic_meta"
    ).fetchone()[0]
    r.close()
    assert json.loads(stored) == [1.0, 0.0]


def test_pin_check_embeds_probe_only_once(tmp_path):
    """A stable DB file signature means the memo hits on the second call."""
    r = _seeded(tmp_path, [1.0, 0.0])
    assert r._ensure_semantic_meta() is None
    r._embedder.calls = 0
    assert r._ensure_semantic_meta() is None
    r.close()
    assert r._embedder.calls == 0


def test_pin_recheck_on_db_signature_change(tmp_path):
    """A changed DB signature forces the pin to be re-verified.

    Covers e.g. esdc fetch/reload replacing the DB file underneath a
    cached resolver, instead of trusting a stale memo.
    """
    r = _seeded(tmp_path, [1.0, 0.0])
    assert r._ensure_semantic_meta() is None
    r._embedder.calls = 0

    # Simulate the DB file changing underneath the cached resolver by
    # bumping its mtime, so the recorded signature no longer matches.
    st = r._db_path.stat()
    os.utime(r._db_path, ns=(st.st_atime_ns, st.st_mtime_ns + 1))

    assert r._ensure_semantic_meta() is None
    r.close()
    assert r._embedder.calls == 1


def test_build_writes_the_pin(tmp_path):
    r = SemanticResolver(
        db_path=tmp_path / "s.duckdb", embedder=_FixedEmbedder([1.0, 0.0])
    )
    r.build_embeddings_table()
    row = r._get_connection().execute(
        "SELECT embedding_model, dim FROM semantic_meta"
    ).fetchone()
    r.close()
    assert row[0] == "qwen3-embedding-0.6b-q8_0"
    assert row[1] == 2
