# tests/test_configs_embedding.py
"""Embedding backend configuration keys and wizard registration."""
from __future__ import annotations

import esdc.configs as configs
from esdc.configs import ENUM_CHOICES, KEY_DESCRIPTIONS, MODEL_SECTIONS, Config


def _no_config(monkeypatch, data=None):
    monkeypatch.setattr(Config, "_load_config", classmethod(lambda cls: data))


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
    assert ENUM_CHOICES["embedding_backend"] == ["local", "ollama", "openai"]


def test_new_keys_described():
    for key in ("embedding_backend", "embedding_model", "embedding_api_key"):
        assert key in KEY_DESCRIPTIONS


def test_new_keys_in_embeddings_wizard_section():
    section = MODEL_SECTIONS["Embeddings"]
    for key in ("embedding_backend", "embedding_host", "embedding_model", "embedding_api_key"):
        assert key in section


def test_api_key_is_masked_by_existing_sensitive_suffix_rule():
    assert any("embedding_api_key".endswith(s) for s in configs.SENSITIVE_KEYS)
