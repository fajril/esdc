# tests/corpus/test_reranker.py
"""get_reranker: the service factory, and what it does with a retired `local`."""

from __future__ import annotations

import pytest

import esdc.corpus.provider_reranker as provider_reranker
from esdc.corpus import reranker as rr_mod
from esdc.corpus.provider_reranker import ProviderReranker


def _no_request(monkeypatch) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("the factory must not send a request")

    monkeypatch.setattr(provider_reranker.requests, "post", forbidden)


def test_no_in_process_reranker_remains():
    """The loader, registry, singleton and teardown are gone, not hidden."""
    for name in (
        "Reranker",
        "_load_reranker",
        "_resolve_model_name",
        "DEFAULT_RERANKER",
        "_RERANKER_GGUFS",
        "_rerank_prompt",
        "_load_lock",
        "_infer_lock",
        "atexit",
    ):
        assert not hasattr(rr_mod, name), name
    assert hasattr(rr_mod, "_warn_lock")
    assert hasattr(rr_mod, "_WARNED_RERANK_SIGNATURES")


RERANK_PROVIDERS = {
    "engine": {
        "name": "engine",
        "provider_type": "openai_compatible",
        "base_url": "http://engine:4000/v1",
        "api_key": "fixture-key",
        "model": "chat-a",  # the saved Chat model, never the rerank model
    },
    "gpu-box": {"provider_type": "ollama", "base_url": "http://gpu-box:11434"},
    "no-address": {"provider_type": "openai_compatible", "model": "chat-a"},
}


def _factory_cfg(**overrides) -> dict:
    cfg = {
        "rerank_backend": "provider",
        "rerank_provider": "engine",
        "rerank_model": "d1-2b",
    }
    cfg.update(overrides)
    return cfg


def _stub_provider(monkeypatch, name: str):
    monkeypatch.setattr(
        rr_mod.Config,
        "get_referenced_provider",
        classmethod(lambda cls, n: RERANK_PROVIDERS[n]),
    )


def test_default_config_is_provider_and_off():
    """The service backend is the default; nothing here enables reranking."""
    cfg = rr_mod.Config.get_corpus_config()
    assert cfg.get("rerank") is False
    assert cfg.get("rerank_backend") == "provider"
    assert cfg.get("rerank_provider") == ""


def test_legacy_local_backend_warns_once_and_falls_back_to_rrf(monkeypatch, caplog):
    """D8: one warning per process per config signature, then RRF order."""
    monkeypatch.setattr(rr_mod, "_WARNED_RERANK_SIGNATURES", set())
    _no_request(monkeypatch)
    cfg = _factory_cfg(rerank_backend="local")
    monkeypatch.setattr(
        rr_mod.Config,
        "get_corpus_config",
        classmethod(lambda cls: cfg),
    )

    with caplog.at_level("WARNING"):
        assert rr_mod.get_reranker() is None
        assert rr_mod.get_reranker() is None

    warnings = [r.message for r in caplog.records if "retired" in r.message]
    assert len(warnings) == 1
    assert "provider" in warnings[0]
    # A stored value is reported, never repaired.
    assert cfg == _factory_cfg(rerank_backend="local")


def test_get_reranker_backend_defaults_to_provider(monkeypatch, caplog):
    """An absent backend means the service path, not the retired local path."""
    monkeypatch.setattr(rr_mod, "_WARNED_RERANK_SIGNATURES", set())
    _no_request(monkeypatch)
    monkeypatch.setattr(
        rr_mod.Config,
        "get_corpus_config",
        classmethod(lambda cls: {"rerank_model": "x"}),
    )
    with caplog.at_level("WARNING"):
        assert rr_mod.get_reranker() is None
    assert any("provider is not configured" in r.message for r in caplog.records)
    assert not any("retired" in r.message for r in caplog.records)


def test_get_reranker_provider_builds_the_http_client_with_the_explicit_model(
    monkeypatch,
):
    _no_request(monkeypatch)
    monkeypatch.setattr(
        rr_mod.Config,
        "get_corpus_config",
        classmethod(lambda cls: _factory_cfg()),
    )
    _stub_provider(monkeypatch, "engine")

    client = rr_mod.get_reranker()

    assert isinstance(client, ProviderReranker)
    assert client.model == "d1-2b"  # the explicit choice, not the provider chat model
    assert client.base_url == "http://engine:4000/v1"
    assert RERANK_PROVIDERS["engine"]["model"] == "chat-a"


def test_legacy_gguf_model_id_is_passed_through_unchanged(monkeypatch):
    """D5: a stored GGUF-shaped id is not reinterpreted; it is sent as the model."""
    _no_request(monkeypatch)
    legacy = _factory_cfg(rerank_model="ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF")
    monkeypatch.setattr(
        rr_mod.Config, "get_corpus_config", classmethod(lambda cls: legacy)
    )
    _stub_provider(monkeypatch, "engine")

    client = rr_mod.get_reranker()
    assert isinstance(client, ProviderReranker)
    assert client.model == legacy["rerank_model"]
    # The stored value itself is unchanged.
    assert legacy == _factory_cfg(rerank_model="ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF")


@pytest.mark.parametrize(
    ("cfg", "phrase"),
    [
        (_factory_cfg(rerank_provider=""), "provider is not configured"),
        (_factory_cfg(rerank_model=""), "model is not configured"),
        (_factory_cfg(rerank_provider="nope"), "not configured"),
        (
            _factory_cfg(rerank_provider="gpu-box"),
            "not supported for provider type ollama",
        ),
        (_factory_cfg(rerank_provider="no-address"), "has no base URL"),
        (_factory_cfg(rerank_backend="gateway"), "not supported"),
    ],
)
def test_get_reranker_invalid_selection_warns_and_returns_none(
    monkeypatch, caplog, cfg, phrase
):
    _no_request(monkeypatch)

    def referenced(cls, name):
        if name not in RERANK_PROVIDERS:
            raise ValueError(f"unknown provider {name}")
        return RERANK_PROVIDERS[name]

    monkeypatch.setattr(rr_mod.Config, "get_corpus_config", classmethod(lambda cls: cfg))
    monkeypatch.setattr(rr_mod.Config, "get_referenced_provider", classmethod(referenced))

    with caplog.at_level("WARNING"):
        assert rr_mod.get_reranker() is None

    assert any(phrase in r.message for r in caplog.records)
