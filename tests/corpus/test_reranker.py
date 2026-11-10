# tests/corpus/test_reranker.py
"""get_reranker: the canonical Reranking task factory."""

from __future__ import annotations

import pytest

import esdc.corpus.provider_reranker as provider_reranker
from esdc.corpus import reranker as rr_mod
from esdc.corpus.provider_reranker import ProviderReranker
from esdc.model_runtime import ResolvedTarget, ResolvedTask

RERANK_PROVIDERS = {
    "engine": {
        "name": "engine",
        "provider_type": "openai_compatible",
        "type": "openai_compatible",
        "base_url": "http://engine:4000/v1",
        "api_key": "fixture-key",
        "model": "chat-a",
    },
    "gpu-box": {
        "type": "ollama",
        "provider_type": "ollama",
        "base_url": "http://gpu-box:11434",
    },
    "no-address": {
        "type": "openai_compatible",
        "provider_type": "openai_compatible",
    },
}


def _no_request(monkeypatch) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("the factory must not send a request")

    monkeypatch.setattr(provider_reranker.requests, "post", forbidden)


def _resolved(
    provider_name: str = "engine",
    *,
    model: str = "d1-2b",
    enabled: bool = True,
    mode: str = "model",
) -> ResolvedTask:
    if mode != "model":
        return ResolvedTask(
            role="rerank", mode=mode, enabled=False  # type: ignore[arg-type]
        )
    provider = dict(RERANK_PROVIDERS[provider_name])
    return ResolvedTask(
        role="rerank",
        mode="model",
        enabled=enabled,
        primary=ResolvedTarget(
            connection=provider_name,
            provider={**provider, "model": model},
            model=model,
            options={},
        ),
    )


def _install(monkeypatch, task: ResolvedTask) -> None:
    monkeypatch.setattr(
        rr_mod.Config,
        "resolve_model_task",
        classmethod(lambda cls, role: task),
    )


def test_no_in_process_reranker_remains():
    """The loader, registry, singleton and retired-local memo are gone."""
    for name in (
        "Reranker",
        "_load_reranker",
        "_resolve_model_name",
        "DEFAULT_RERANKER",
        "_RERANKER_GGUFS",
        "_rerank_prompt",
        "_load_lock",
        "_infer_lock",
        "_warn_retired_local",
        "_WARNED_RERANK_SIGNATURES",
        "_warn_lock",
        "atexit",
    ):
        assert not hasattr(rr_mod, name), name


def test_disabled_rerank_returns_none(monkeypatch):
    _no_request(monkeypatch)
    _install(monkeypatch, _resolved(enabled=False))
    assert rr_mod.get_reranker() is None


def test_unset_rerank_returns_none(monkeypatch):
    _no_request(monkeypatch)
    _install(monkeypatch, _resolved(mode="unset"))
    assert rr_mod.get_reranker() is None


def test_enabled_rerank_builds_the_client_with_its_assigned_model(monkeypatch):
    _no_request(monkeypatch)
    _install(monkeypatch, _resolved(model="d1-2b"))

    client = rr_mod.get_reranker()

    assert isinstance(client, ProviderReranker)
    assert client.model == "d1-2b"
    assert client.base_url == "http://engine:4000/v1"
    assert RERANK_PROVIDERS["engine"]["model"] == "chat-a"


def test_force_builds_the_client_even_when_the_task_is_disabled(monkeypatch):
    _no_request(monkeypatch)
    _install(monkeypatch, _resolved(enabled=False))

    client = rr_mod.get_reranker(force=True)

    assert isinstance(client, ProviderReranker)
    assert client.model == "d1-2b"


def test_legacy_gguf_model_id_is_passed_through_unchanged(monkeypatch):
    _no_request(monkeypatch)
    legacy = "ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF"
    _install(monkeypatch, _resolved(model=legacy))

    client = rr_mod.get_reranker()
    assert isinstance(client, ProviderReranker)
    assert client.model == legacy


def test_unsupported_connection_type_warns_and_returns_none(monkeypatch, caplog):
    _no_request(monkeypatch)
    _install(monkeypatch, _resolved("gpu-box"))

    with caplog.at_level("WARNING"):
        assert rr_mod.get_reranker() is None
    assert any("not OpenAI-compatible" in r.message for r in caplog.records)


def test_connection_without_base_url_warns_and_returns_none(monkeypatch, caplog):
    _no_request(monkeypatch)
    _install(monkeypatch, _resolved("no-address"))

    with caplog.at_level("WARNING"):
        assert rr_mod.get_reranker() is None
    assert any("has no base URL" in r.message for r in caplog.records)


@pytest.mark.parametrize("mode", ["chat", "ocr", "off", "unset"])
def test_non_model_modes_are_not_rerankable(monkeypatch, mode):
    _no_request(monkeypatch)
    _install(monkeypatch, _resolved(mode=mode))
    assert rr_mod.get_reranker(force=True) is None
