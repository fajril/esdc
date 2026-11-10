"""Canonical runtime routing tests for all inference tasks."""

import hashlib
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import yaml

import esdc.embedders as emb
from esdc.configs import Config
from esdc.corpus.ocr import create_ocr_client
from esdc.corpus.reranker import get_reranker
from esdc.corpus.task_models import text_caller_for_task
from esdc.providers import create_llm_from_resolved_task

API_KEY = "synthetic-conn-key"
BASE_URL = "http://gateway.example/v1"


def canonical_config() -> dict[str, Any]:
    return {
        "model_settings_version": 1,
        "connections": {
            "gateway": {
                "type": "openai_compatible",
                "base_url": BASE_URL,
                "api_key": API_KEY,
            }
        },
        "models": {
            "chat": {
                "mode": "model",
                "connection": "gateway",
                "model": "chat-model",
                "options": {"reasoning_effort": "medium"},
            },
            "embedding": {
                "mode": "model",
                "connection": "gateway",
                "model": "embed-model",
            },
            "rerank": {
                "mode": "model",
                "connection": "gateway",
                "model": "rerank-model",
                "enabled": True,
            },
            "ocr": {
                "mode": "model",
                "connection": "gateway",
                "model": "ocr-model",
            },
            "metadata": {
                "mode": "model",
                "connection": "gateway",
                "model": "meta-model",
                "options": {"reasoning_effort": "low"},
            },
            "cleanup": {
                "mode": "model",
                "connection": "gateway",
                "model": "clean-model",
            },
        },
    }


def write_config(tmp_path: Path, config: dict[str, Any]) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    Config._config_cache = None
    return path


@pytest.fixture(autouse=True)
def isolated_config_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[Path]:
    monkeypatch.setenv("ESDC_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("ESDC_DB_DIR", str(tmp_path))
    monkeypatch.setenv("ESDC_DB_FILE", str(tmp_path / "esdc.duckdb"))
    for name in (
        "ESDC_EMBEDDING_BACKEND",
        "ESDC_EMBEDDING_HOST",
        "ESDC_EMBEDDING_MODEL",
        "ESDC_EMBEDDING_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    Config._config_cache = None
    yield tmp_path
    Config._config_cache = None


class RecordedProvider:
    calls: list[dict[str, Any]] = []

    @classmethod
    def create_llm(cls, model=None, base_url=None, api_key=None, **kwargs):
        cls.calls.append(
            {
                "model": model,
                "base_url": base_url,
                "api_key": api_key,
                **kwargs,
            }
        )

        class _FakeLlm:
            def invoke(self, prompt):
                class _Response:
                    content = f"{model}:{prompt}"

                return _Response()

        return _FakeLlm()


class RecordedEmbedder:
    calls: list[dict[str, Any]] = []

    def __init__(self, host=None, model=None, api_key=None, **kwargs):
        self.calls.append(
            {"host": host, "model": model, "api_key": api_key, **kwargs}
        )


class RecordedOcr:
    calls: list[dict[str, Any]] = []

    def __init__(self, provider, model, timeout_seconds=None, **kwargs):
        self.calls.append(
            {
                "provider": dict(provider),
                "model": model,
                "timeout_seconds": timeout_seconds,
            }
        )

    def health_check(self) -> bool:
        return True


class RecordedReranker:
    calls: list[dict[str, Any]] = []

    def __init__(self, provider, model):
        self.calls.append({"provider": dict(provider), "model": model})


def _reset_records() -> None:
    RecordedProvider.calls = []
    RecordedEmbedder.calls = []
    RecordedOcr.calls = []
    RecordedReranker.calls = []


def test_all_inference_roles_use_their_resolved_assignment(
    isolated_config_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_config(isolated_config_dir, canonical_config())
    _reset_records()
    RecordedEmbedder.calls = []

    class FakeEmbedder(RecordedEmbedder):
        def __init__(self, host=None, model=None, api_key=None, **kwargs):
            super().__init__(host=host, model=model, api_key=api_key, **kwargs)

    monkeypatch.setattr(emb, "OpenAIEmbedder", FakeEmbedder)
    monkeypatch.setattr(
        "esdc.corpus.provider_ocr.OpenAICompatibleOcr", RecordedOcr
    )
    monkeypatch.setattr(
        "esdc.corpus.reranker.ProviderReranker", RecordedReranker
    )

    with patch("esdc.providers.get_provider", return_value=RecordedProvider):
        chat = Config.resolve_model_task("chat")
        create_llm_from_resolved_task(chat).invoke("chat")
        metadata = text_caller_for_task("metadata")
        cleanup = text_caller_for_task("cleanup")
        assert metadata is not None
        assert cleanup is not None
        metadata("metadata")
        cleanup("cleanup")

    embedding = emb.get_embedder()
    reranker = get_reranker()
    ocr = create_ocr_client(Config.get_corpus_config())

    assert embedding is not None
    assert reranker is not None
    assert ocr is not None

    text_models = [call["model"] for call in RecordedProvider.calls]
    assert text_models == ["chat-model", "meta-model", "clean-model"]
    assert RecordedProvider.calls[0]["reasoning_effort"] == "medium"
    assert RecordedProvider.calls[1]["reasoning_effort"] == "low"
    assert "reasoning_effort" not in RecordedProvider.calls[2]
    for index in (0, 1, 2):
        assert RecordedProvider.calls[index]["base_url"] == BASE_URL
        assert RecordedProvider.calls[index]["api_key"] == API_KEY

    embedding_call = RecordedEmbedder.calls[0]
    assert embedding_call["model"] == "embed-model"
    assert embedding_call["host"] == BASE_URL
    assert embedding_call["api_key"] == API_KEY

    assert RecordedReranker.calls[0]["model"] == "rerank-model"
    assert RecordedReranker.calls[0]["provider"]["base_url"] == BASE_URL
    assert RecordedOcr.calls[0]["model"] == "ocr-model"
    assert RecordedOcr.calls[0]["provider"]["base_url"] == BASE_URL

    non_chat_models = {
        embedding_call["model"],
        RecordedReranker.calls[0]["model"],
        RecordedOcr.calls[0]["model"],
        RecordedProvider.calls[1]["model"],
        RecordedProvider.calls[2]["model"],
    }
    assert "chat-model" not in non_chat_models


def test_unset_embedding_does_not_default_to_localhost(
    isolated_config_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    Config.init_config()
    _reset_records()
    monkeypatch.setattr(emb, "OpenAIEmbedder", RecordedEmbedder)

    with pytest.raises(emb.EmbeddingConfigError, match="not configured"):
        emb.get_embedder()

    assert RecordedEmbedder.calls == []


def test_embedding_backend_override_must_match_the_assigned_connection(
    isolated_config_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_config(isolated_config_dir, canonical_config())
    _reset_records()
    monkeypatch.setattr(emb, "OpenAIEmbedder", RecordedEmbedder)

    with pytest.raises(ValueError, match="does not match"):
        emb.get_embedder("ollama")

    assert RecordedEmbedder.calls == []


def test_embedding_signature_uses_the_effective_assignment(
    isolated_config_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_config(isolated_config_dir, canonical_config())

    signature = emb.config_signature()
    assert signature[0] == "openai"
    assert signature[1] == BASE_URL
    assert signature[2] == "embed-model"
    assert signature[3] == hashlib.sha256(API_KEY.encode()).hexdigest()

    monkeypatch.setenv("ESDC_EMBEDDING_HOST", "http://override.example/v1")
    monkeypatch.setenv("ESDC_EMBEDDING_MODEL", "override-embed")
    monkeypatch.setenv("ESDC_EMBEDDING_API_KEY", "override-key")

    overridden = emb.config_signature()
    assert overridden[0] == "openai"
    assert overridden[1] == "http://override.example/v1"
    assert overridden[2] == "override-embed"
    assert overridden[3] == hashlib.sha256(b"override-key").hexdigest()


def test_disabled_rerank_builds_no_client(
    isolated_config_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = canonical_config()
    config["models"]["rerank"]["enabled"] = False
    write_config(isolated_config_dir, config)
    _reset_records()
    monkeypatch.setattr(
        "esdc.corpus.reranker.ProviderReranker", RecordedReranker
    )

    assert get_reranker() is None
    assert RecordedReranker.calls == []


def test_text_task_modes_do_not_parse_model_strings(
    isolated_config_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = canonical_config()
    config["models"]["metadata"] = {"mode": "chat"}
    config["models"]["cleanup"] = {"mode": "off"}
    write_config(isolated_config_dir, config)
    _reset_records()

    with patch("esdc.providers.get_provider", return_value=RecordedProvider):
        metadata = text_caller_for_task("metadata")
        assert metadata is not None
        assert metadata("prompt") == "chat-model:prompt"
        assert text_caller_for_task("cleanup") is None

    assert [call["model"] for call in RecordedProvider.calls] == ["chat-model"]


def test_corpus_processing_config_no_longer_persists_model_fields() -> None:
    removed = {
        "rerank",
        "rerank_backend",
        "rerank_provider",
        "rerank_model",
        "ocr_model",
        "ocr_provider",
        "metadata_model",
        "metadata_provider",
        "cleanup_model",
        "cleanup_provider",
        "ollama_host",
        "n_gpu_layers",
    }
    assert removed.isdisjoint(Config.CORPUS_DEFAULTS)
    cfg = Config.get_corpus_config()
    assert removed.isdisjoint(cfg)
    assert cfg["chunk_size"] == 3000
    assert cfg["rerank_pool"] == 30
    assert cfg["ocr_dpi"] == 200
