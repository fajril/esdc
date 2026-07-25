# tests/corpus/test_embedder.py
"""InternalEmbedder on llama.cpp (mocked model)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from esdc.corpus import embedder as emb_mod


def _fake_model():
    m = MagicMock()
    # llama_cpp embed(str) -> list[float]; embed(list) -> list[list[float]]
    m.embed.side_effect = lambda x: (
        [0.1, 0.2, 0.3] if isinstance(x, str) else [[0.1, 0.2, 0.3] for _ in x]
    )
    return m


def test_model_identifier_is_stable_backend_neutral_id():
    e = emb_mod.InternalEmbedder()
    assert e.model == "qwen3-embedding-0.6b-q8_0"


def test_generate_embedding_returns_float_list():
    with patch.object(emb_mod, "_get_model", return_value=_fake_model()):
        vec = emb_mod.InternalEmbedder().generate_embedding("halo")
    assert vec == [0.1, 0.2, 0.3]
    assert all(isinstance(x, float) for x in vec)


def test_generate_embeddings_batch_empty_returns_empty():
    assert emb_mod.InternalEmbedder().generate_embeddings_batch([]) == []


def test_generate_embeddings_batch_maps_each_text():
    with patch.object(emb_mod, "_get_model", return_value=_fake_model()):
        out = emb_mod.InternalEmbedder().generate_embeddings_batch(["a", "b"])
    assert out == [[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]]
