"""Warmup unit tests — never load a real model, never hit the network."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

import esdc.corpus.embedder as embedder_mod
import esdc.corpus.reranker as reranker_mod
from esdc.corpus import warmup as w
from esdc.corpus.reranker import Reranker
from esdc.corpus.warmup import run_warmup


class FakeTextEmbedding:
    def embed(self, texts):
        # llama.cpp-style: embed(str) -> vector; embed(list) -> list of vectors.
        if isinstance(texts, str):
            return [0.1, 0.2, 0.3]
        return [[0.1, 0.2, 0.3] for _ in texts]


class FakeRerankModel:
    def embed(self, prompt):
        return [0.9, 0.1]


@pytest.fixture(autouse=True)
def reset_singleton():
    Reranker._instance = None
    Reranker._failed = False
    yield
    Reranker._instance = None
    Reranker._failed = False


def test_warmup_embedder_ok_reranker_skipped_when_disabled():
    fake_emb = MagicMock()
    fake_emb.model = "qwen3-embedding-0.6b-q8_0"
    with (
        patch.object(w, "_build_embedder", return_value=fake_emb),
        patch.object(w.Config, "get_corpus_config", return_value={"rerank": False}),
        patch.object(w, "_resolve_model_name", return_value="rr-model"),
    ):
        results = w.run_warmup()
    by = {r.component: r for r in results}
    assert by["embedder"].ok is True
    assert by["reranker"].ok is True
    assert "skipped" in by["reranker"].detail


def test_warms_embedder_always(monkeypatch):
    monkeypatch.setattr(embedder_mod, "_get_model", lambda: FakeTextEmbedding())
    from esdc.configs import Config

    monkeypatch.setattr(
        Config, "get_corpus_config", classmethod(lambda cls: {"rerank": False})
    )

    results = run_warmup(rerank=False)

    embedder_result = next(r for r in results if r.component == "embedder")
    assert embedder_result.ok is True
    assert embedder_result.model == "qwen3-embedding-0.6b-q8_0"

    reranker_result = next(r for r in results if r.component == "reranker")
    assert reranker_result.ok is True
    assert "disabled" in reranker_result.detail or "skip" in reranker_result.detail


def test_warms_reranker_when_flag_true(monkeypatch):
    monkeypatch.setattr(embedder_mod, "_get_model", lambda: FakeTextEmbedding())
    monkeypatch.setattr(reranker_mod, "_load_reranker", lambda: FakeRerankModel())

    results = run_warmup(rerank=True)

    embedder_result = next(r for r in results if r.component == "embedder")
    reranker_result = next(r for r in results if r.component == "reranker")
    assert embedder_result.ok is True
    assert reranker_result.ok is True
    assert reranker_result.model == "ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF"


def test_rerank_none_follows_config(monkeypatch):
    from esdc.configs import Config

    monkeypatch.setattr(
        Config, "get_corpus_config", classmethod(lambda cls: {"rerank": True})
    )
    monkeypatch.setattr(embedder_mod, "_get_model", lambda: FakeTextEmbedding())
    monkeypatch.setattr(reranker_mod, "_load_reranker", lambda: FakeRerankModel())

    results = run_warmup()

    reranker_result = next(r for r in results if r.component == "reranker")
    assert reranker_result.ok is True


def test_embedder_failure_reported(monkeypatch):
    def boom():
        raise RuntimeError("no internet")

    monkeypatch.setattr(embedder_mod, "_get_model", boom)
    from esdc.configs import Config

    monkeypatch.setattr(
        Config, "get_corpus_config", classmethod(lambda cls: {"rerank": False})
    )

    results = run_warmup()

    embedder_result = next(r for r in results if r.component == "embedder")
    assert embedder_result.ok is False
    assert "no internet" in embedder_result.detail


def test_reranker_load_failure_reported(monkeypatch):
    monkeypatch.setattr(embedder_mod, "_get_model", lambda: FakeTextEmbedding())

    def boom():
        raise RuntimeError("no model")

    monkeypatch.setattr(reranker_mod, "_load_reranker", boom)

    results = run_warmup(rerank=True)

    reranker_result = next(r for r in results if r.component == "reranker")
    assert reranker_result.ok is False
