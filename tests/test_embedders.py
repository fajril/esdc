# tests/test_embedders.py
"""Shared embedder backends: internal (llama.cpp), Ollama, OpenAI-compatible."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import esdc.embedders as emb


def _fake_model():
    m = MagicMock()
    # llama_cpp embed(str) -> list[float]; embed(list) -> list[list[float]]
    m.embed.side_effect = lambda x: (
        [0.1, 0.2, 0.3] if isinstance(x, str) else [[0.1, 0.2, 0.3] for _ in x]
    )
    return m


def test_model_id_is_backend_neutral():
    assert emb.MODEL_ID == "qwen3-embedding-0.6b-q8_0"


def test_internal_embedder_reports_model_id():
    assert emb.InternalEmbedder().model == emb.MODEL_ID


def test_internal_generate_embedding_returns_floats():
    with patch.object(emb, "_get_model", return_value=_fake_model()):
        vec = emb.InternalEmbedder().generate_embedding("halo")
    assert vec == [0.1, 0.2, 0.3]
    assert all(isinstance(x, float) for x in vec)


def test_internal_batch_empty_returns_empty():
    assert emb.InternalEmbedder().generate_embeddings_batch([]) == []


def test_internal_batch_maps_each_text():
    with patch.object(emb, "_get_model", return_value=_fake_model()):
        out = emb.InternalEmbedder().generate_embeddings_batch(["a", "b"])
    assert out == [[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]]


def test_corpus_shim_reexports_same_objects():
    from esdc.corpus import embedder as shim

    assert shim.InternalEmbedder is emb.InternalEmbedder
    assert shim.MODEL_ID == emb.MODEL_ID


def test_embedders_module_has_no_module_level_search_import():
    """esdc.search imports semantic_resolver, which will import this module."""
    import sys

    sys.modules.pop("esdc.search", None)
    sys.modules.pop("esdc.embedders", None)
    import esdc.embedders  # noqa: F401

    assert "esdc.search" not in sys.modules
