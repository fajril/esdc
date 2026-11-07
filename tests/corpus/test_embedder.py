# tests/corpus/test_embedder.py
"""The corpus embedder module is a shim over esdc.embedders."""

from __future__ import annotations

import esdc.embedders as emb
from esdc.corpus import embedder as shim


def test_shim_exports_model_id():
    assert shim.MODEL_ID == "qwen3-embedding-0.6b-q8_0"
    assert shim.MODEL_ID == emb.MODEL_ID


def test_shim_exports_the_client_protocol():
    assert shim.EmbeddingClient is emb.EmbeddingClient


def test_shim_exports_exactly_the_contract():
    """Only the identity and the shape are re-exported; no model behind them."""
    assert sorted(shim.__all__) == ["EmbeddingClient", "MODEL_ID"]


def test_shim_no_longer_aliases_the_retired_seam():
    for name in ("InternalEmbedder", "_get_model"):
        assert not hasattr(shim, name), name
