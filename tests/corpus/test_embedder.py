"""InternalEmbedder unit tests — never load a real fastembed model."""

import os

import pytest

import esdc.corpus.embedder as embedder_mod
from esdc.corpus.embedder import PINNED_MODEL, InternalEmbedder


class FakeTextEmbedding:
    def embed(self, texts):
        for _ in texts:
            yield [0.1, 0.2, 0.3]


@pytest.fixture
def fake_model(monkeypatch):
    monkeypatch.setattr(embedder_mod, "_get_model", lambda: FakeTextEmbedding())


def test_get_model_caches_under_esdc_models_dir(monkeypatch, tmp_path):
    """Real model downloads to <config_dir>/models, not the system temp dir.

    Keeps a warmed model alive across reboots / tmp purges for offline use.
    """
    import fastembed

    from esdc.configs import Config

    captured = {}

    class RecordingTextEmbedding:
        def __init__(self, model_name, cache_dir=None, **kwargs):
            captured["model_name"] = model_name
            captured["cache_dir"] = cache_dir

    monkeypatch.setattr(fastembed, "TextEmbedding", RecordingTextEmbedding)
    monkeypatch.setattr(Config, "get_config_dir", classmethod(lambda cls: tmp_path))
    monkeypatch.setattr(embedder_mod, "_model", None)

    embedder_mod._get_model()

    assert captured["model_name"] == PINNED_MODEL
    assert captured["cache_dir"] == str(tmp_path / "models")
    monkeypatch.setattr(embedder_mod, "_model", None)


def test_model_attr_is_prefixed_pin():
    e = InternalEmbedder()
    assert e.model == f"fastembed:{PINNED_MODEL}"


def test_generate_embedding_returns_float_list(fake_model):
    e = InternalEmbedder()
    vec = e.generate_embedding("halo dunia")
    assert vec == [0.1, 0.2, 0.3]
    assert all(isinstance(x, float) for x in vec)


def test_generate_embeddings_batch(fake_model):
    e = InternalEmbedder()
    vecs = e.generate_embeddings_batch(["a", "b"])
    assert vecs == [[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]]


def test_generate_embeddings_batch_empty_skips_model_load(monkeypatch):
    def boom():
        raise AssertionError("model must not load for empty input")

    monkeypatch.setattr(embedder_mod, "_get_model", boom)
    assert InternalEmbedder().generate_embeddings_batch([]) == []


@pytest.mark.skipif(
    os.environ.get("ESDC_EMBED_SMOKE") != "1",
    reason="real model download; set ESDC_EMBED_SMOKE=1 to run",
)
def test_real_model_smoke():
    e = InternalEmbedder()
    vec = e.generate_embedding("uji coba embedding bahasa Indonesia")
    assert len(vec) == 1024
