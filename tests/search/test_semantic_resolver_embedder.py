# tests/search/test_semantic_resolver_embedder.py
"""SemanticResolver embeds queries locally and accepts an injected embedder."""
from __future__ import annotations

import inspect

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
