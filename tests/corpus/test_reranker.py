"""Reranker unit tests — never load a real cross-encoder."""

import os

import pytest

import esdc.corpus.reranker as reranker_mod
from esdc.corpus.reranker import (
    DEFAULT_RERANKER,
    Reranker,
    _register_custom,
    _resolve_model_name,
)


@pytest.fixture(autouse=True)
def reset_singleton():
    Reranker._instance = None
    Reranker._failed = False
    yield
    Reranker._instance = None
    Reranker._failed = False


class FakeEncoder:
    def rerank(self, query, texts):
        # reverse order: last text scores highest
        return list(range(len(texts)))


def test_get_returns_singleton(monkeypatch):
    monkeypatch.setattr(reranker_mod, "_load_encoder", lambda: FakeEncoder())
    a, b = Reranker.get(), Reranker.get()
    assert a is b is not None


def test_get_returns_none_and_remembers_failure(monkeypatch):
    calls = []

    def boom():
        calls.append(1)
        raise RuntimeError("no model")

    monkeypatch.setattr(reranker_mod, "_load_encoder", boom)
    assert Reranker.get() is None
    assert Reranker.get() is None
    assert len(calls) == 1  # failure cached, no reload storm


def test_rerank_returns_float_scores(monkeypatch):
    monkeypatch.setattr(reranker_mod, "_load_encoder", lambda: FakeEncoder())
    rr = Reranker.get()
    scores = rr.rerank("q", ["a", "b", "c"])
    assert scores == [0.0, 1.0, 2.0]


# --------------------------------------------------------------------------
# Configurable reranker model + custom-model registration
# --------------------------------------------------------------------------


def test_resolve_model_name_defaults_to_jina(monkeypatch):
    from esdc.configs import Config

    monkeypatch.setattr(Config, "get_corpus_config", classmethod(lambda cls: {}))
    assert _resolve_model_name() == DEFAULT_RERANKER
    assert DEFAULT_RERANKER == "jinaai/jina-reranker-v2-base-multilingual"


def test_resolve_model_name_from_config(monkeypatch):
    from esdc.configs import Config

    monkeypatch.setattr(
        Config,
        "get_corpus_config",
        classmethod(lambda cls: {"rerank_model": "BAAI/bge-reranker-v2-m3"}),
    )
    assert _resolve_model_name() == "BAAI/bge-reranker-v2-m3"


def test_bge_v2_m3_custom_mapping_is_correct():
    """Pre-wired Apache-2.0 Indonesian reranker maps to a fastembed ONNX repo.

    Single onnx/model.onnx + root tokenizer/config — the fastembed layout.
    """
    spec = reranker_mod._CUSTOM_RERANKERS["BAAI/bge-reranker-v2-m3"]
    assert spec["hf"] == "onnx-community/bge-reranker-v2-m3-ONNX"
    assert spec["model_file"] == "onnx/model.onnx"
    assert spec["license"] == "apache-2.0"


def test_register_custom_makes_bge_v2_m3_fastembed_supported():
    """Registration is metadata-only, no weight download.

    After registering, the name appears in fastembed's supported list.
    """
    from fastembed.rerank.cross_encoder import TextCrossEncoder

    _register_custom("BAAI/bge-reranker-v2-m3")
    names = {m["model"] for m in TextCrossEncoder.list_supported_models()}
    assert "BAAI/bge-reranker-v2-m3" in names
    # Idempotent: a second call must not raise on the already-registered name.
    _register_custom("BAAI/bge-reranker-v2-m3")


def test_register_custom_builtin_or_unknown_is_noop():
    # A fastembed built-in needs no registration.
    _register_custom("jinaai/jina-reranker-v2-base-multilingual")
    # An unknown name is left alone (TextCrossEncoder will raise a clear
    # error at construction, caught by Reranker.get()'s fallback).
    _register_custom("some/unregistered-model")


@pytest.mark.skipif(
    os.environ.get("ESDC_EMBED_SMOKE") != "1",
    reason="real model download; set ESDC_EMBED_SMOKE=1 to run",
)
def test_bge_v2_m3_real_smoke(monkeypatch):
    """Real integration smoke: config-select bge-reranker-v2-m3 and score.

    Loads via custom registration; gated behind ESDC_EMBED_SMOKE.
    """
    from esdc.configs import Config

    monkeypatch.setattr(
        Config,
        "get_corpus_config",
        classmethod(lambda cls: {"rerank_model": "BAAI/bge-reranker-v2-m3"}),
    )
    rr = Reranker.get()
    assert rr is not None
    scores = rr.rerank(
        "persetujuan pengembangan lapangan",
        ["Dokumen persetujuan POD lapangan Merak", "Resep rendang padang"],
    )
    assert len(scores) == 2
    assert scores[0] > scores[1]  # relevant doc ranks higher
