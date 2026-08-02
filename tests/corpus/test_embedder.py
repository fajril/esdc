# tests/corpus/test_embedder.py
"""The corpus embedder module is a shim over esdc.embedders."""
from __future__ import annotations

import esdc.embedders as emb
from esdc.corpus import embedder as shim


def test_shim_exports_model_id():
    assert shim.MODEL_ID == "qwen3-embedding-0.6b-q8_0"


def test_shim_exports_internal_embedder_class():
    assert shim.InternalEmbedder is emb.InternalEmbedder


def test_shim_exports_get_model():
    assert shim._get_model is emb._get_model
