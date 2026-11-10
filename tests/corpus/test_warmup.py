"""Warmup unit tests — service checks only, never a model load or a real request."""

from __future__ import annotations

from unittest.mock import MagicMock

import esdc.corpus.reranker as reranker_mod
import esdc.corpus.warmup as w
from esdc.corpus.warmup import run_warmup
from esdc.model_runtime import ResolvedTarget, ResolvedTask


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


def _rerank_task(
    *, enabled: bool = False, model: str = "rank-c", mode: str = "model"
) -> ResolvedTask:
    if mode != "model":
        return ResolvedTask(
            role="rerank", mode=mode, enabled=False  # type: ignore[arg-type]
        )
    return ResolvedTask(
        role="rerank",
        mode="model",
        enabled=enabled,
        primary=ResolvedTarget(
            connection="engine",
            provider={
                "name": "engine",
                "provider_type": "openai_compatible",
                "type": "openai_compatible",
                "base_url": "http://engine:4000/v1",
                "model": model,
            },
            model=model,
            options={},
        ),
    )


def _install_rerank(monkeypatch, task: ResolvedTask | None = None) -> None:
    chosen = task or _rerank_task()
    monkeypatch.setattr(
        w.Config,
        "resolve_model_task",
        classmethod(lambda cls, role: chosen),
    )


def _by(results: list) -> dict:
    return {r.component: r for r in results}


def test_embedder_check_is_the_service_availability_call(monkeypatch):
    emb = _embedder()
    monkeypatch.setattr(w, "get_embedder", lambda: emb)
    _install_rerank(monkeypatch)

    result = _by(run_warmup())["embedder"]

    assert result.ok is True
    assert result.detail == "ready"
    assert result.model == emb.model
    emb.check_available.assert_called_once()


def test_embedder_service_failure_reports_its_message(monkeypatch):
    emb = _embedder("[Embedding] backend 'ollama' service at http://box returned HTTP 503.")
    monkeypatch.setattr(w, "get_embedder", lambda: emb)
    _install_rerank(monkeypatch)

    result = _by(run_warmup())["embedder"]

    assert result.ok is False
    assert "503" in result.detail


def test_embedder_configuration_error_is_reported(monkeypatch):
    from esdc.embedders import EmbeddingConfigError

    def boom():
        raise EmbeddingConfigError("the in-process backend was removed")

    monkeypatch.setattr(w, "get_embedder", boom)
    _install_rerank(monkeypatch)

    result = _by(run_warmup())["embedder"]

    assert result.ok is False
    assert "in-process" in result.detail


def test_rerank_disabled_makes_no_rerank_request(monkeypatch):
    monkeypatch.setattr(w, "get_embedder", lambda: _embedder())

    def forbidden(**kwargs):
        raise AssertionError("a disabled reranker must not be resolved")

    monkeypatch.setattr(w, "get_reranker", forbidden)
    _install_rerank(monkeypatch, _rerank_task(enabled=False))

    result = _by(run_warmup())["reranker"]

    assert result.ok is True
    assert "disabled" in result.detail


def test_rerank_enabled_checks_only_the_rerank_api(monkeypatch):
    rr = _reranker()
    monkeypatch.setattr(w, "get_embedder", lambda: _embedder())
    monkeypatch.setattr(w, "get_reranker", lambda **kwargs: rr)
    _install_rerank(monkeypatch, _rerank_task(enabled=True, model="rank-c"))

    result = _by(run_warmup())["reranker"]

    assert result.ok is True
    assert result.model == "rank-c"
    rr.rerank.assert_called_once()
    query, texts = rr.rerank.call_args.args
    assert query == "warmup"
    assert isinstance(texts, list) and texts


def test_rerank_unavailable_is_reported_without_a_request(monkeypatch):
    monkeypatch.setattr(w, "get_embedder", lambda: _embedder())
    monkeypatch.setattr(w, "get_reranker", lambda **kwargs: None)
    _install_rerank(monkeypatch, _rerank_task(enabled=True, model="x"))

    result = _by(run_warmup())["reranker"]

    assert result.ok is False
    assert "unavailable" in result.detail


def test_rerank_failure_is_reported(monkeypatch):
    rr = _reranker(RuntimeError("timeout"))
    monkeypatch.setattr(w, "get_embedder", lambda: _embedder())
    monkeypatch.setattr(w, "get_reranker", lambda **kwargs: rr)
    _install_rerank(monkeypatch, _rerank_task(enabled=True, model="x"))

    result = _by(run_warmup())["reranker"]

    assert result.ok is False
    assert "timeout" in result.detail


def test_unset_rerank_is_skipped(monkeypatch):
    """An unset task is not an unavailable service; it is simply disabled."""
    monkeypatch.setattr(w, "get_embedder", lambda: _embedder())
    _install_rerank(monkeypatch, _rerank_task(mode="unset"))
    assert not hasattr(reranker_mod, "_load_reranker")
    assert not hasattr(reranker_mod, "Reranker")

    result = _by(run_warmup())["reranker"]

    assert result.ok is True
    assert "disabled" in result.detail


def test_rerank_none_follows_the_task_enabled_switch(monkeypatch):
    rr = _reranker()
    monkeypatch.setattr(w, "get_embedder", lambda: _embedder())
    monkeypatch.setattr(w, "get_reranker", lambda **kwargs: rr)
    _install_rerank(monkeypatch, _rerank_task(enabled=True, model="x"))

    result = _by(run_warmup(rerank=None))["reranker"]

    assert result.ok is True
