# tests/test_configs_embedding.py
"""Embedding backend configuration keys, references and wizard registration."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

import esdc.configs as configs
import esdc.embedders as emb
from esdc.configs import ENUM_CHOICES, KEY_DESCRIPTIONS, MODEL_SECTIONS, Config
from esdc.model_settings import default_model_settings


@pytest.fixture(autouse=True)
def no_embedding_environment(monkeypatch):
    """Keep the ambient shell out of the precedence checks.

    Each unit still sets the condition it asserts; this fixture only guarantees
    that an unrelated value from the calling shell cannot decide the result.
    """
    for name in (
        "ESDC_EMBEDDING_BACKEND",
        "ESDC_EMBEDDING_HOST",
        "ESDC_EMBEDDING_API_KEY",
        "ESDC_EMBEDDING_MODEL",
    ):
        monkeypatch.delenv(name, raising=False)


def _no_config(monkeypatch, data=None):
    monkeypatch.setattr(Config, "_load_config", classmethod(lambda cls: data))


# Synthetic fixture provider. Its `model` is the Chat model and must never be
# read as the embedding model.
_ENGINE = {
    "provider_type": "openai_compatible",
    "base_url": "http://engine:4000/v1",
    "api_key": "fixture-key",
    "model": "chat-a",
    "reasoning_effort": "low",
}


def _embedding_config(**overrides):
    providers = overrides.pop("providers", {"engine": dict(_ENGINE)})
    connections = {}
    for name, entry in providers.items():
        connection = {"type": entry.get("provider_type") or entry.get("type")}
        for key in ("base_url", "api_key", "oauth", "auth_method", "api_version"):
            if key in entry:
                connection[key] = entry[key]
        connections[name] = connection

    data = {
        "model_settings_version": 1,
        "connections": connections,
        "models": default_model_settings(),
        "embedding_backend": "openai",
        "embedding_provider": "engine",
        "embedding_model": "embed-b",
        "embedding_api_key": "other-source-key",
    }
    data.update(overrides)

    provider_name = str(data.get("embedding_provider") or "")
    backend = str(data.get("embedding_backend") or "")
    model = str(data.get("embedding_model") or "")
    if backend != "local" and provider_name in connections and model:
        data["models"]["embedding"] = {
            "mode": "model",
            "connection": provider_name,
            "model": model,
        }
    return data


def _http_ok(payload):
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = payload
    return response


def test_backend_defaults_to_ollama(monkeypatch):
    _no_config(monkeypatch)
    monkeypatch.delenv("ESDC_EMBEDDING_BACKEND", raising=False)
    assert Config.get_embedding_backend() == "ollama"


def test_backend_reads_config(monkeypatch):
    _no_config(monkeypatch, {"embedding_backend": "local"})
    monkeypatch.delenv("ESDC_EMBEDDING_BACKEND", raising=False)
    assert Config.get_embedding_backend() == "local"


def test_backend_env_beats_config(monkeypatch):
    _no_config(monkeypatch, {"embedding_backend": "local"})
    monkeypatch.setenv("ESDC_EMBEDDING_BACKEND", "openai")
    assert Config.get_embedding_backend() == "openai"


def test_backend_is_normalized(monkeypatch):
    _no_config(monkeypatch)
    monkeypatch.setenv("ESDC_EMBEDDING_BACKEND", "  OLLAMA ")
    assert Config.get_embedding_backend() == "ollama"


def test_model_defaults_empty(monkeypatch):
    _no_config(monkeypatch)
    monkeypatch.delenv("ESDC_EMBEDDING_MODEL", raising=False)
    assert Config.get_embedding_model() == ""


def test_model_env_beats_config(monkeypatch):
    _no_config(monkeypatch, {"embedding_model": "from-config"})
    monkeypatch.setenv("ESDC_EMBEDDING_MODEL", "from-env")
    assert Config.get_embedding_model() == "from-env"


def test_api_key_env_beats_config(monkeypatch):
    _no_config(monkeypatch, {"embedding_api_key": "cfg"})
    monkeypatch.setenv("ESDC_EMBEDDING_API_KEY", "env")
    assert Config.get_embedding_api_key() == "env"


def test_api_key_defaults_empty(monkeypatch):
    _no_config(monkeypatch)
    monkeypatch.delenv("ESDC_EMBEDDING_API_KEY", raising=False)
    assert Config.get_embedding_api_key() == ""


def test_backend_enum_choices_registered():
    """The wizard offers services only: the retired in-process value is not a choice."""
    assert ENUM_CHOICES["embedding_backend"] == ["ollama", "openai"]
    assert "local" not in ENUM_CHOICES["embedding_backend"]


def test_retired_backend_description_names_a_service(monkeypatch):
    """A stored legacy value is described as needing a service, not repaired."""
    description = KEY_DESCRIPTIONS["embedding_backend"]
    assert "HTTP" in description
    assert "in-process" in description.lower() or "removed" in description.lower()
    for choice in ENUM_CHOICES["embedding_backend"]:
        assert choice in description
    _no_config(monkeypatch, {"embedding_backend": "local"})
    monkeypatch.delenv("ESDC_EMBEDDING_BACKEND", raising=False)
    # Reading is not validation: the stored value is returned as it is stored.
    assert Config.get_embedding_backend() == "local"


def test_new_keys_described():
    for key in ("embedding_backend", "embedding_model", "embedding_api_key"):
        assert key in KEY_DESCRIPTIONS


def test_new_keys_in_embeddings_wizard_section():
    section = MODEL_SECTIONS["Embeddings"]
    for key in (
        "embedding_backend",
        "embedding_provider",
        "embedding_host",
        "embedding_model",
        "embedding_api_key",
    ):
        assert key in section
    assert "embedding_provider" in section


def test_api_key_is_masked_by_existing_sensitive_suffix_rule():
    assert any("embedding_api_key".endswith(s) for s in configs.SENSITIVE_KEYS)


class TestEmbeddingProviderReference:
    """An active `embedding_provider` reference resolves host/key/model."""

    def test_reference_reuses_connection_not_chat_model(self, monkeypatch):
        _no_config(monkeypatch, _embedding_config())
        assert Config.get_embedding_backend() == "openai"
        assert Config.get_embedding_host() == "http://engine:4000/v1"
        assert Config.get_embedding_api_key() == "fixture-key"
        # The wire model is the explicit embedding model, never chat-a.
        assert Config.get_embedding_model() == "embed-b"

    def test_environment_beats_the_reference_for_every_part(self, monkeypatch):
        _no_config(monkeypatch, _embedding_config())
        monkeypatch.setenv("ESDC_EMBEDDING_HOST", "http://env-override:1234/v1")
        monkeypatch.setenv("ESDC_EMBEDDING_API_KEY", "env-key")
        monkeypatch.setenv("ESDC_EMBEDDING_MODEL", "env-model")
        assert Config.get_embedding_host() == "http://env-override:1234/v1"
        assert Config.get_embedding_api_key() == "env-key"
        assert Config.get_embedding_model() == "env-model"

    def test_reference_falls_back_to_saved_host_when_no_provider_address(
        self, monkeypatch
    ):
        """No address on the provider and no env value: the existing host wins."""
        provider = {"provider_type": "openai_compatible", "api_key": "fixture-key"}
        _no_config(
            monkeypatch,
            _embedding_config(
                embedding_host="http://legacy:1234/v1", providers={"engine": provider}
            ),
        )
        assert Config.get_embedding_host() == "http://legacy:1234/v1"

    def test_empty_reference_key_sends_no_auth(self, monkeypatch):
        """An empty provider key means no Authorization, not another saved key."""
        _no_config(
            monkeypatch,
            _embedding_config(providers={"engine": {**_ENGINE, "api_key": ""}}),
        )
        assert Config.get_embedding_api_key() == ""

        post = MagicMock(
            return_value=_http_ok({"data": [{"index": 0, "embedding": [1.0]}]})
        )
        monkeypatch.setattr(emb.requests, "post", post)
        emb.get_build_embedder().generate_embedding("x")
        assert "Authorization" not in post.call_args.kwargs["headers"]

    def test_unknown_active_reference_raises(self, monkeypatch):
        _no_config(monkeypatch, _embedding_config(embedding_provider="ghost"))
        with pytest.raises(ValueError, match="embedding_provider"):
            Config.get_embedding_host()

    def test_reference_type_backend_mismatch_raises(self, monkeypatch):
        """A gateway reference cannot be paired with the ollama backend."""
        _no_config(monkeypatch, _embedding_config(embedding_backend="ollama"))
        with pytest.raises(ValueError, match="embedding_backend"):
            Config.get_embedding_api_key()

    def test_unsupported_provider_type_raises(self, monkeypatch):
        _no_config(
            monkeypatch,
            _embedding_config(
                providers={"engine": {"provider_type": "anthropic", "model": "chat-a"}}
            ),
        )
        with pytest.raises(ValueError, match="embedding_provider"):
            Config.get_embedding_backend()

    def test_reference_with_no_address_anywhere_raises(self, monkeypatch):
        _no_config(
            monkeypatch,
            _embedding_config(
                embedding_host=None,
                providers={"engine": {"provider_type": "openai_compatible"}},
            ),
        )
        with pytest.raises(ValueError, match="embedding_provider"):
            Config.get_embedding_host()

    def test_local_backend_ignores_reference_and_keeps_stored_values(self, monkeypatch):
        data = _embedding_config(
            embedding_backend="local",
            embedding_host="http://legacy:1234/v1",
            embedding_api_key="legacy-key",
        )
        _no_config(monkeypatch, data)
        assert Config.get_embedding_backend() == "local"
        assert Config.get_embedding_host() == "http://legacy:1234/v1"
        assert Config.get_embedding_api_key() == "legacy-key"
        # The inactive reference is left exactly as stored.
        assert data["embedding_provider"] == "engine"

    def test_ollama_reference_uses_provider_host_and_keeps_pinned_model(
        self, monkeypatch
    ):
        ollama_provider = {
            "provider_type": "ollama",
            "base_url": "http://ollama-box:11434",
            "model": "chat-a",
        }
        _no_config(
            monkeypatch,
            _embedding_config(
                embedding_backend="ollama", providers={"engine": ollama_provider}
            ),
        )
        assert Config.get_embedding_backend() == "ollama"
        assert Config.get_embedding_host() == "http://ollama-box:11434"

        manager = MagicMock()
        import esdc.search.embedding_manager as em

        factory = MagicMock(return_value=manager)
        monkeypatch.setattr(em, "EmbeddingManager", factory)
        embedder = emb.get_build_embedder()
        assert isinstance(embedder, emb.OllamaEmbedder)
        assert embedder.host == "http://ollama-box:11434"
        # D7: the configured embedding model is what the service is asked for.
        # A provider reference supplies the address, never its chat model.
        assert factory.call_args.kwargs["model"] == "embed-b"
        # D1: identity stays the index pin, so a service alias change does not
        # repin corpus_meta.
        assert embedder.model == emb.MODEL_ID

    def test_ollama_reference_requires_an_explicit_model(self, monkeypatch):
        """Canonical mode model does not fall back to an implicit Ollama tag."""
        ollama_provider = {
            "provider_type": "ollama",
            "base_url": "http://ollama-box:11434",
            "model": "chat-a",
        }
        _no_config(
            monkeypatch,
            _embedding_config(
                embedding_backend="ollama",
                embedding_model="",
                providers={"engine": ollama_provider},
            ),
        )
        with pytest.raises(emb.EmbeddingConfigError, match="not configured"):
            emb.get_build_embedder()

    def test_get_build_embedder_uses_referenced_provider(self, monkeypatch):
        """Integration: the constructed embedder carries the reference values."""
        _no_config(monkeypatch, _embedding_config())
        post = MagicMock(
            return_value=_http_ok({"data": [{"index": 0, "embedding": [1.0]}]})
        )
        monkeypatch.setattr(emb.requests, "post", post)

        embedder = emb.get_build_embedder()
        assert isinstance(embedder, emb.OpenAIEmbedder)
        assert embedder.url == "http://engine:4000/v1/embeddings"
        assert embedder.wire_model == "embed-b"

        embedder.generate_embedding("sample text")
        post.assert_called_once()
        assert post.call_args.kwargs["json"]["model"] == "embed-b"
        assert post.call_args.kwargs["headers"]["Authorization"] == "Bearer fixture-key"
