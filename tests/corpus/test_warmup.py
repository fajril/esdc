"""Warmup unit tests — service checks only, never a model load or a real request."""

from __future__ import annotations

from unittest.mock import MagicMock

import esdc.corpus.reranker as reranker_mod
import esdc.corpus.warmup as w
from esdc.corpus.warmup import run_warmup


def _embedder(message: str | None = None) -> MagicMock:
    client = MagicMock()
    client.model = "qwen3-embedding-0.6b-q8_0"
    client.check_available.return_value = message
    return client


def _reranker(error: Exception | None = None) -> MagicMock:
    client = MagicMock()
    if error is None:
        client.rerank.return_value = [0.5]
    else:
        client.rerank.side_effect = error
    return client


def _by(results: list) -> dict:
    return {r.component: r for r in results}


def test_embedder_check_is_the_service_availability_call(monkeypatch):
    emb = _embedder()
    monkeypatch.setattr(w, "get_embedder", lambda: emb)
    monkeypatch.setattr(
        w.Config, "get_corpus_config", classmethod(lambda cls: {"rerank": False})
    )

    result = _by(run_warmup())["embedder"]

    assert result.ok is True
    assert result.detail == "ready"
    assert result.model == emb.model
    emb.check_available.assert_called_once()


def test_embedder_service_failure_reports_its_message(monkeypatch):
    emb = _embedder("[Embedding] backend 'ollama' service at http://box returned HTTP 503.")
    monkeypatch.setattr(w, "get_embedder", lambda: emb)
    monkeypatch.setattr(
        w.Config, "get_corpus_config", classmethod(lambda cls: {"rerank": False})
    )

    result = _by(run_warmup())["embedder"]

    assert result.ok is False
    assert "503" in result.detail


def test_embedder_configuration_error_is_reported(monkeypatch):
    from esdc.embedders import EmbeddingConfigError

    def boom():
        raise EmbeddingConfigError("the in-process backend was removed")

    monkeypatch.setattr(w, "get_embedder", boom)
    monkeypatch.setattr(
        w.Config, "get_corpus_config", classmethod(lambda cls: {"rerank": False})
    )

    result = _by(run_warmup())["embedder"]

    assert result.ok is False
    assert "in-process" in result.detail


def test_rerank_disabled_makes_no_rerank_request(monkeypatch):
    monkeypatch.setattr(w, "get_embedder", lambda: _embedder())

    def forbidden():
        raise AssertionError("a disabled reranker must not be resolved")

    monkeypatch.setattr(w, "get_reranker", forbidden)
    monkeypatch.setattr(
        w.Config, "get_corpus_config", classmethod(lambda cls: {"rerank": False})
    )

    result = _by(run_warmup())["reranker"]

    assert result.ok is True
    assert "disabled" in result.detail


def test_rerank_enabled_checks_only_the_rerank_api(monkeypatch):
    rr = _reranker()
    monkeypatch.setattr(w, "get_embedder", lambda: _embedder())
    monkeypatch.setattr(w, "get_reranker", lambda: rr)
    monkeypatch.setattr(
        w.Config,
        "get_corpus_config",
        classmethod(
            lambda cls: {
                "rerank": True,
                "rerank_backend": "provider",
                "rerank_provider": "engine",
                "rerank_model": "rank-c",
            }
        ),
    )

    result = _by(run_warmup())["reranker"]

    assert result.ok is True
    assert result.model == "rank-c"
    rr.rerank.assert_called_once()
    query, texts = rr.rerank.call_args.args
    assert query == "warmup"
    assert isinstance(texts, list) and texts


def test_rerank_unavailable_is_reported_without_a_request(monkeypatch):
    monkeypatch.setattr(w, "get_embedder", lambda: _embedder())
    monkeypatch.setattr(w, "get_reranker", lambda: None)
    monkeypatch.setattr(
        w.Config,
        "get_corpus_config",
        classmethod(
            lambda cls: {"rerank": True, "rerank_backend": "provider", "rerank_model": "x"}
        ),
    )

    result = _by(run_warmup())["reranker"]

    assert result.ok is False
    assert "unavailable" in result.detail


def test_rerank_failure_is_reported(monkeypatch):
    rr = _reranker(RuntimeError("timeout"))
    monkeypatch.setattr(w, "get_embedder", lambda: _embedder())
    monkeypatch.setattr(w, "get_reranker", lambda: rr)
    monkeypatch.setattr(
        w.Config,
        "get_corpus_config",
        classmethod(
            lambda cls: {"rerank": True, "rerank_backend": "provider", "rerank_model": "x"}
        ),
    )

    result = _by(run_warmup())["reranker"]

    assert result.ok is False
    assert "timeout" in result.detail


def test_legacy_local_backend_has_no_load_path(monkeypatch):
    """The retired local value is reported through the public factory.

    There is no loader left for warmup to reach: nothing in-process can be
    loaded any more.
    """
    monkeypatch.setattr(reranker_mod, "_WARNED_RERANK_SIGNATURES", set())
    assert not hasattr(reranker_mod, "_load_reranker")
    assert not hasattr(reranker_mod, "Reranker")

    monkeypatch.setattr(w, "get_embedder", lambda: _embedder())
    monkeypatch.setattr(
        w.Config,
        "get_corpus_config",
        classmethod(
            lambda cls: {
                "rerank": True,
                "rerank_backend": "local",
                "rerank_model": "service-model-x",
            }
        ),
    )

    result = _by(run_warmup())["reranker"]

    assert result.ok is False
    assert "unavailable" in result.detail


def test_rerank_none_follows_config(monkeypatch):
    rr = _reranker()
    monkeypatch.setattr(w, "get_embedder", lambda: _embedder())
    monkeypatch.setattr(w, "get_reranker", lambda: rr)
    monkeypatch.setattr(
        w.Config,
        "get_corpus_config",
        classmethod(
            lambda cls: {"rerank": True, "rerank_backend": "provider", "rerank_model": "x"}
        ),
    )

    result = _by(run_warmup(rerank=None))["reranker"]

    assert result.ok is True
