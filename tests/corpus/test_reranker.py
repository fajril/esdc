# tests/corpus/test_reranker.py
"""Reranker on llama.cpp RANK pooling (mocked model) and the get_reranker factory."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

import esdc.corpus.provider_reranker as provider_reranker
from esdc.corpus import reranker as rr_mod
from esdc.corpus.provider_reranker import ProviderReranker


def setup_function(_):
    # reset singleton between tests
    rr_mod.Reranker._instance = None
    rr_mod.Reranker._failed = False


def test_resolve_model_name_default():
    with patch.object(rr_mod.Config, "get_corpus_config", return_value={}):
        assert rr_mod._resolve_model_name() == rr_mod.DEFAULT_RERANKER


def test_rerank_score_is_index_zero_of_rank_output():
    model = MagicMock()
    # relevant -> high P(yes); irrelevant -> low
    model.embed.side_effect = lambda pair: (
        [0.98, 0.02] if "450 juta" in pair else [0.30, 0.70]
    )
    r = rr_mod.Reranker(model)
    scores = r.rerank("cadangan?", ["Banyu Urip 450 juta barel", "keselamatan kerja"])
    assert scores[0] > scores[1]
    assert scores[0] == 0.98


def test_rerank_input_uses_official_qwen3_template():
    model = MagicMock()
    model.embed.return_value = [0.5, 0.5]
    rr_mod.Reranker(model).rerank("cadangan?", ["some doc"])
    prompt = model.embed.call_args[0][0]
    # GGUF does not bake the template; we must build it. Official format:
    # system prefix + <Instruct>/<Query>/<Document> + assistant/<think> suffix.
    assert "<Instruct>:" in prompt
    assert "<Query>: cadangan?" in prompt
    assert "<Document>: some doc" in prompt
    assert prompt.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")


def test_get_returns_none_and_caches_failure_on_load_error():
    with patch.object(rr_mod, "_load_reranker", side_effect=RuntimeError("boom")):
        assert rr_mod.Reranker.get() is None
        assert rr_mod.Reranker.get() is None  # cached, no second attempt


# --------------------------------------------------------------------------
# get_reranker: local keeps the loader, provider builds the HTTP client
# --------------------------------------------------------------------------

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


def _no_gguf_load(monkeypatch) -> list:
    """Record any loader attempt; the provider path must never reach llama.cpp."""
    loads: list[str] = []
    monkeypatch.setattr(rr_mod, "_load_reranker", lambda: loads.append("load"))
    return loads


def _no_request(monkeypatch) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("the factory must not send a request")

    monkeypatch.setattr(provider_reranker.requests, "post", forbidden)


def _stub_provider(monkeypatch, name: str):
    monkeypatch.setattr(
        rr_mod.Config,
        "get_referenced_provider",
        classmethod(lambda cls, n: RERANK_PROVIDERS[n]),
    )


def test_default_config_stays_local_and_off():
    """Nothing here enables reranking: the saved default is local and disabled."""
    cfg = rr_mod.Config.get_corpus_config()
    assert cfg.get("rerank") is False
    assert cfg.get("rerank_backend") == "local"
    assert cfg.get("rerank_provider") == ""


def test_get_reranker_local_calls_the_singleton_and_builds_no_http_client(monkeypatch):
    calls = []
    monkeypatch.setattr(
        rr_mod.Reranker,
        "get",
        classmethod(lambda cls: calls.append(cls) or None),
    )
    _no_request(monkeypatch)
    monkeypatch.setattr(
        rr_mod.Config,
        "get_corpus_config",
        classmethod(lambda cls: _factory_cfg(rerank_backend="local")),
    )

    assert rr_mod.get_reranker() is None
    assert calls == [rr_mod.Reranker]


def test_get_reranker_local_returns_the_loaded_singleton(monkeypatch):
    sentinel = rr_mod.Reranker(MagicMock())
    monkeypatch.setattr(rr_mod.Reranker, "get", classmethod(lambda cls: sentinel))
    monkeypatch.setattr(
        rr_mod.Config,
        "get_corpus_config",
        classmethod(lambda cls: {"rerank_backend": "local"}),
    )
    assert rr_mod.get_reranker() is sentinel


def test_get_reranker_backend_defaults_to_local(monkeypatch):
    sentinel = rr_mod.Reranker(MagicMock())
    monkeypatch.setattr(rr_mod.Reranker, "get", classmethod(lambda cls: sentinel))
    monkeypatch.setattr(
        rr_mod.Config, "get_corpus_config", classmethod(lambda cls: {"rerank_model": "x"})
    )
    assert rr_mod.get_reranker() is sentinel


def test_get_reranker_provider_builds_the_http_client_with_the_explicit_model(monkeypatch):
    loads = _no_gguf_load(monkeypatch)
    _no_request(monkeypatch)
    monkeypatch.setattr(rr_mod.Config, "get_corpus_config", classmethod(lambda cls: _factory_cfg()))
    _stub_provider(monkeypatch, "engine")

    client = rr_mod.get_reranker()

    assert isinstance(client, ProviderReranker)
    assert client.model == "d1-2b"  # the explicit choice, not the provider chat model
    assert client.base_url == "http://engine:4000/v1"
    assert loads == []  # no GGUF load for the provider path
    assert RERANK_PROVIDERS["engine"]["model"] == "chat-a"


@pytest.mark.parametrize(
    ("cfg", "phrase"),
    [
        (_factory_cfg(rerank_provider=""), "provider is not configured"),
        (_factory_cfg(rerank_model=""), "model is not configured"),
        (_factory_cfg(rerank_provider="nope"), "not configured"),
        (_factory_cfg(rerank_provider="gpu-box"), "not supported for provider type ollama"),
        (_factory_cfg(rerank_provider="no-address"), "has no base URL"),
        (_factory_cfg(rerank_backend="gateway"), "not supported"),
    ],
)
def test_get_reranker_invalid_selection_warns_and_returns_none(
    monkeypatch, caplog, cfg, phrase
):
    loads = _no_gguf_load(monkeypatch)
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
    assert loads == []  # an invalid selection never loads a model
