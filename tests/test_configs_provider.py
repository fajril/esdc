"""Tests for provider configuration."""

from unittest.mock import patch

import pytest
import yaml

from esdc.configs import Config


@pytest.mark.allow_provider_config
class TestGetProviderConfig:
    """Tests for get_provider_config()."""

    def test_get_provider_config_returns_default_provider(self, tmp_path):
        """Test get_provider_config returns the canonical Chat primary."""
        with patch.object(Config, "get_config_dir", return_value=tmp_path):
            config_file = tmp_path / "config.yaml"
            config_file.write_text(
                """
model_settings_version: 1
connections:
  openai:
    type: openai
    api_key: sk-test
  ollama:
    type: ollama
    base_url: http://localhost:11434
models:
  chat:
    mode: model
    connection: ollama
    model: llama3.2
  embedding: {mode: unset}
  rerank: {mode: unset, enabled: false}
  ocr: {mode: unset}
  metadata: {mode: chat}
  cleanup: {mode: chat}
"""
            )
            provider_config = Config.get_provider_config()
            assert provider_config is not None
            assert provider_config["provider_type"] == "ollama"
            assert provider_config["base_url"] == "http://localhost:11434"
            assert provider_config["model"] == "llama3.2"

    def test_get_provider_config_no_default_provider(self, tmp_path):
        """Test get_provider_config returns None when Chat is unset."""
        with patch.object(Config, "get_config_dir", return_value=tmp_path):
            config_file = tmp_path / "config.yaml"
            config_file.write_text(
                """
model_settings_version: 1
connections:
  openai:
    type: openai
    api_key: sk-test
models:
  chat: {mode: unset}
  embedding: {mode: unset}
  rerank: {mode: unset, enabled: false}
  ocr: {mode: unset}
  metadata: {mode: chat}
  cleanup: {mode: chat}
"""
            )
            provider_config = Config.get_provider_config()
            assert provider_config is None

    def test_get_provider_config_no_providers_section(self, tmp_path):
        """Test get_provider_config returns None when no providers section."""
        with patch.object(Config, "get_config_dir", return_value=tmp_path):
            config_file = tmp_path / "config.yaml"
            config_file.write_text("api_url: https://example.com\n")
            # Clear cache
            provider_config = Config.get_provider_config()
            assert provider_config is None

    def test_get_provider_config_no_config(self, tmp_path):
        """Test get_provider_config returns None when no config file."""
        with patch.object(Config, "get_config_dir", return_value=tmp_path):
            # Clear cache
            provider_config = Config.get_provider_config()
            assert provider_config is None

    def test_get_provider_config_attaches_fallback_configs(self, tmp_path):
        """Test ordered Chat fallback targets attach adapter configs."""
        with patch.object(Config, "get_config_dir", return_value=tmp_path):
            config_file = tmp_path / "config.yaml"
            config_file.write_text(
                """
model_settings_version: 1
connections:
  deepseek:
    type: deepseek
    api_key: sk-deepseek
  openai:
    type: openai
    api_key: sk-openai
models:
  chat:
    mode: model
    connection: deepseek
    model: deepseek-v4-flash
    fallbacks:
      - connection: openai
        model: gpt-4o-mini
  embedding: {mode: unset}
  rerank: {mode: unset, enabled: false}
  ocr: {mode: unset}
  metadata: {mode: chat}
  cleanup: {mode: chat}
"""
            )

            provider_config = Config.get_provider_config()
            assert provider_config is not None
            assert provider_config["provider_type"] == "deepseek"
            assert len(provider_config["fallback_configs"]) == 1
            assert provider_config["fallback_configs"][0]["provider_type"] == "openai"
            assert provider_config["fallback_configs"][0]["model"] == "gpt-4o-mini"



class TestPersistProviderOauth:
    """Tests for persist_provider_oauth()."""

    def test_persist_provider_oauth_updates_matching_provider_only(self, tmp_path):
        """Only the named canonical connection's oauth section is updated."""
        with patch.object(Config, "get_config_dir", return_value=tmp_path):
            config_file = tmp_path / "config.yaml"
            config_file.write_text(
                """
model_settings_version: 1
connections:
  openai:
    type: openai
    auth_method: oauth
    oauth:
      access_token: old-access
      refresh_token: old-refresh
      expires_at: 1
  ollama:
    type: ollama
    base_url: http://localhost:11434
models:
  chat: {mode: unset}
  embedding: {mode: unset}
  rerank: {mode: unset, enabled: false}
  ocr: {mode: unset}
  metadata: {mode: chat}
  cleanup: {mode: chat}
"""
            )

            new_oauth = {
                "access_token": "new-access",
                "refresh_token": "rotated-refresh",
                "expires_at": 9999999999,
            }
            Config.persist_provider_oauth("openai", new_oauth)

            with open(config_file) as f:
                config = yaml.safe_load(f)

            assert config["connections"]["openai"]["oauth"] == new_oauth
            # Other connection fields/config are untouched.
            assert config["connections"]["openai"]["auth_method"] == "oauth"
            assert config["connections"]["ollama"] == {
                "type": "ollama",
                "base_url": "http://localhost:11434",
            }
            assert "providers" not in config

    def test_persist_provider_oauth_writes_file_with_0o600_permissions(self, tmp_path):
        """The rewritten config file must not be group/world readable."""
        with patch.object(Config, "get_config_dir", return_value=tmp_path):
            config_file = tmp_path / "config.yaml"
            config_file.write_text(
                """
model_settings_version: 1
connections:
  openai:
    type: openai
    oauth:
      access_token: old
models:
  chat: {mode: unset}
  embedding: {mode: unset}
  rerank: {mode: unset, enabled: false}
  ocr: {mode: unset}
  metadata: {mode: chat}
  cleanup: {mode: chat}
"""
            )
            config_file.chmod(0o644)

            Config.persist_provider_oauth("openai", {"access_token": "new"})

            mode = config_file.stat().st_mode & 0o777
            assert mode == 0o600

    def test_persist_provider_oauth_missing_provider_is_noop(self, tmp_path, caplog):
        """Unknown provider (e.g. env-only config): log a warning, don't crash."""
        with patch.object(Config, "get_config_dir", return_value=tmp_path):
            config_file = tmp_path / "config.yaml"
            config_file.write_text(
                """
providers:
  openai:
    provider_type: openai
    oauth:
      access_token: old
"""
            )
            original_contents = config_file.read_text()

            with caplog.at_level("WARNING"):
                Config.persist_provider_oauth("does-not-exist", {"access_token": "x"})

            assert config_file.read_text() == original_contents
            assert any("does-not-exist" in record.message for record in caplog.records)

    def test_persist_provider_oauth_missing_config_file_is_noop(self, tmp_path, caplog):
        """No config file on disk (e.g. env-only setup): no crash, warn instead."""
        with patch.object(Config, "get_config_dir", return_value=tmp_path):
            with caplog.at_level("WARNING"):
                Config.persist_provider_oauth("openai", {"access_token": "x"})

            assert not (tmp_path / "config.yaml").exists()
            assert any("openai" in record.message for record in caplog.records)
