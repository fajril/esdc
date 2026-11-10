"""Tests for provider configuration."""

import copy
import os
from unittest.mock import patch

import pytest
import yaml

from esdc.configs import Config


class TestGetProviders:
    """Tests for get_providers()."""

    def test_get_providers_returns_dict(self, tmp_path):
        """Test get_providers returns a dict."""
        with patch.object(Config, "get_config_dir", return_value=tmp_path):
            providers = Config.get_providers()
            assert isinstance(providers, dict)

    def test_get_providers_from_config(self, tmp_path):
        """Test get_providers returns canonical connections as adapters."""
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
  chat: {mode: unset}
  embedding: {mode: unset}
  rerank: {mode: unset, enabled: false}
  ocr: {mode: unset}
  metadata: {mode: chat}
  cleanup: {mode: chat}
"""
            )
            providers = Config.get_providers()
            assert "openai" in providers
            assert "ollama" in providers
            assert providers["openai"]["provider_type"] == "openai"
            assert providers["openai"]["type"] == "openai"
            assert "model" not in providers["openai"]

    def test_get_providers_empty_when_none(self, tmp_path):
        """Test get_providers returns empty dict when no providers."""
        with patch.object(Config, "get_config_dir", return_value=tmp_path):
            config_file = tmp_path / "config.yaml"
            config_file.write_text("api_url: https://example.com\n")
            providers = Config.get_providers()
            assert providers == {}


class TestSaveProvider:
    """Tests for save_provider()."""

    def test_save_provider_creates_new(self, tmp_path):
        """Test save_provider adds a new provider."""
        with patch.object(Config, "get_config_dir", return_value=tmp_path):
            Config.save_provider(
                "openai",
                {"provider_type": "openai", "api_key": "sk-test", "model": "gpt-4o"},
            )
            config_file = tmp_path / "config.yaml"
            with open(config_file) as f:
                config = yaml.safe_load(f)
            assert "providers" in config
            assert "openai" in config["providers"]
            assert config["providers"]["openai"]["api_key"] == "sk-test"

    def test_save_provider_updates_existing(self, tmp_path):
        """Test save_provider updates existing provider."""
        with patch.object(Config, "get_config_dir", return_value=tmp_path):
            config_file = tmp_path / "config.yaml"
            config_file.write_text(
                """
providers:
  openai:
    provider_type: openai
    api_key: sk-old
    model: gpt-4o
"""
            )

            Config.save_provider(
                "openai",
                {"provider_type": "openai", "api_key": "sk-new", "model": "gpt-4o"},
            )
            with open(config_file) as f:
                config = yaml.safe_load(f)
            assert config["providers"]["openai"]["api_key"] == "sk-new"

    def test_save_provider_adds_to_existing(self, tmp_path):
        """Test save_provider adds to existing providers."""
        with patch.object(Config, "get_config_dir", return_value=tmp_path):
            config_file = tmp_path / "config.yaml"
            config_file.write_text(
                """
providers:
  openai:
    provider_type: openai
    api_key: sk-test
"""
            )

            Config.save_provider(
                "ollama",
                {"provider_type": "ollama", "base_url": "http://localhost:11434"},
            )
            with open(config_file) as f:
                config = yaml.safe_load(f)
            assert "openai" in config["providers"]
            assert "ollama" in config["providers"]


class TestRemoveProvider:
    """Tests for remove_provider()."""

    def test_remove_provider_exists(self, tmp_path):
        """Test remove_provider removes existing provider."""
        with patch.object(Config, "get_config_dir", return_value=tmp_path):
            config_file = tmp_path / "config.yaml"
            config_file.write_text(
                """
providers:
  openai:
    provider_type: openai
  ollama:
    provider_type: ollama
"""
            )

            result = Config.remove_provider("openai")
            assert result is True
            with open(config_file) as f:
                config = yaml.safe_load(f)
            assert "openai" not in config["providers"]
            assert "ollama" in config["providers"]

    def test_remove_provider_not_exists(self, tmp_path):
        """Test remove_provider returns False for non-existent provider."""
        with patch.object(Config, "get_config_dir", return_value=tmp_path):
            config_file = tmp_path / "config.yaml"
            config_file.write_text(
                """
providers:
  openai:
    provider_type: openai
"""
            )

            result = Config.remove_provider("nonexistent")
            assert result is False
            with open(config_file) as f:
                config = yaml.safe_load(f)
            assert "openai" in config["providers"]

    def test_remove_provider_empty_providers(self, tmp_path):
        """Test remove_provider when no providers section exists."""
        with patch.object(Config, "get_config_dir", return_value=tmp_path):
            config_file = tmp_path / "config.yaml"
            config_file.write_text("api_url: https://example.com\n")

            result = Config.remove_provider("openai")
            assert result is False


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

    def test_get_provider_order_prepends_default_provider(self, tmp_path):
        """Chat primary remains first even when fallbacks are not ordered."""
        with patch.object(Config, "get_config_dir", return_value=tmp_path):
            config_file = tmp_path / "config.yaml"
            config_file.write_text(
                """
model_settings_version: 1
connections:
  deepseek:
    type: deepseek
  openai:
    type: openai
models:
  chat:
    mode: model
    connection: openai
    model: gpt-4o-mini
    fallbacks:
      - connection: deepseek
        model: deepseek-v4-flash
  embedding: {mode: unset}
  rerank: {mode: unset, enabled: false}
  ocr: {mode: unset}
  metadata: {mode: chat}
  cleanup: {mode: chat}
"""
            )

            assert Config.get_provider_order() == ["openai", "deepseek"]


class TestSetDefaultProvider:
    """Tests for set_default_provider()."""

    def test_set_default_provider(self, tmp_path):
        """Test set_default_provider sets the default provider."""
        with patch.object(Config, "get_config_dir", return_value=tmp_path):
            config_file = tmp_path / "config.yaml"
            config_file.write_text(
                """
providers:
  openai:
    provider_type: openai
  ollama:
    provider_type: ollama
"""
            )

            Config.set_default_provider("ollama")
            with open(config_file) as f:
                config = yaml.safe_load(f)
            assert config["default_provider"] == "ollama"

    def test_set_default_provider_moves_provider_order_to_front(self, tmp_path):
        """Test set_default_provider keeps failover order in sync."""
        with patch.object(Config, "get_config_dir", return_value=tmp_path):
            config_file = tmp_path / "config.yaml"
            config_file.write_text(
                """
default_provider: deepseek
provider_order:
  - deepseek
  - openai
providers:
  deepseek:
    provider_type: deepseek
  openai:
    provider_type: openai
"""
            )

            Config.set_default_provider("openai")
            with open(config_file) as f:
                config = yaml.safe_load(f)
            assert config["default_provider"] == "openai"
            assert config["provider_order"] == ["openai", "deepseek"]

    def test_set_provider_order_sets_default_provider(self, tmp_path):
        """Test set_provider_order stores order and updates default provider."""
        with patch.object(Config, "get_config_dir", return_value=tmp_path):
            config_file = tmp_path / "config.yaml"
            config_file.write_text(
                """
providers:
  deepseek:
    provider_type: deepseek
  openai:
    provider_type: openai
"""
            )

            Config.set_provider_order(["deepseek", "openai", "deepseek"])
            with open(config_file) as f:
                config = yaml.safe_load(f)
            assert config["default_provider"] == "deepseek"
            assert config["provider_order"] == ["deepseek", "openai"]

    def test_set_provider_order_rejects_unknown_provider(self, tmp_path):
        """Test set_provider_order validates provider names."""
        with patch.object(Config, "get_config_dir", return_value=tmp_path):
            config_file = tmp_path / "config.yaml"
            config_file.write_text(
                """
providers:
  deepseek:
    provider_type: deepseek
"""
            )

            with pytest.raises(ValueError, match="Unknown provider"):
                Config.set_provider_order(["deepseek", "openai"])

    def test_set_default_provider_creates_section(self, tmp_path):
        """Test set_default_provider works when no config exists."""
        with patch.object(Config, "get_config_dir", return_value=tmp_path):
            Config.set_default_provider("openai")
            config_file = tmp_path / "config.yaml"
            with open(config_file) as f:
                config = yaml.safe_load(f)
            assert config["default_provider"] == "openai"


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


class TestGetReferencedProvider:
    """A task role is handed a canonical connection adapter, never a cache ref."""

    _CONFIG_TEXT = """
model_settings_version: 1
connections:
  opencode:
    type: openai_compatible
  engine:
    type: openai_compatible
    base_url: http://engine:4000/v1
    api_key: fixture-key
    oauth:
      access_token: fixture-access
models:
  chat:
    mode: model
    connection: opencode
    model: chat-a
    fallbacks:
      - connection: engine
        model: chat-a
  embedding: {mode: unset}
  rerank: {mode: unset, enabled: false}
  ocr: {mode: unset}
  metadata: {mode: chat}
  cleanup: {mode: chat}
"""

    def test_returns_copy_with_saved_name(self, tmp_path):
        with patch.object(Config, "get_config_dir", return_value=tmp_path):
            (tmp_path / "config.yaml").write_text(self._CONFIG_TEXT)

            provider = Config.get_referenced_provider("engine")
            assert provider["name"] == "engine"
            assert provider["provider_type"] == "openai_compatible"
            assert provider["type"] == "openai_compatible"
            assert provider["base_url"] == "http://engine:4000/v1"
            assert provider["api_key"] == "fixture-key"
            assert "model" not in provider
            assert "reasoning_effort" not in provider

    def test_copy_is_deep_and_does_not_reach_the_cache(self, tmp_path):
        with patch.object(Config, "get_config_dir", return_value=tmp_path):
            (tmp_path / "config.yaml").write_text(self._CONFIG_TEXT)
            saved = Config.get_provider_config_by_name("engine")
            assert saved is not None

            provider = Config.get_referenced_provider("engine")
            assert provider is not saved
            assert provider["oauth"] is not saved["oauth"]

            provider["api_key"] = "wire-b"
            provider["oauth"]["access_token"] = "changed"
            fresh = Config.get_provider_config_by_name("engine")
            assert fresh is not None
            assert fresh["api_key"] == "fixture-key"
            assert fresh["oauth"]["access_token"] == "fixture-access"

    def test_unknown_reference_raises(self, tmp_path):
        with patch.object(Config, "get_config_dir", return_value=tmp_path):
            (tmp_path / "config.yaml").write_text(self._CONFIG_TEXT)
            with pytest.raises(ValueError, match="Unknown provider"):
                Config.get_referenced_provider("ghost")

    def test_non_mapping_provider_raises(self, tmp_path):
        with patch.object(Config, "get_config_dir", return_value=tmp_path):
            (tmp_path / "config.yaml").write_text(
                """
model_settings_version: 1
connections:
  engine: not-a-mapping
models:
  chat: {mode: unset}
  embedding: {mode: unset}
  rerank: {mode: unset, enabled: false}
  ocr: {mode: unset}
  metadata: {mode: chat}
  cleanup: {mode: chat}
"""
            )
            with pytest.raises(ValueError, match="must be a mapping"):
                Config.get_referenced_provider("engine")


class TestSetChatSelection:
    """Chat owns the provider's chat model, saved in one atomic write."""

    _CONFIG_TEXT = """
default_provider: opencode
provider_order:
  - opencode
  - engine
  - backup
providers:
  opencode:
    provider_type: openai_compatible
    base_url: http://gateway.example:4000/v1
    model: chat-a
  engine:
    provider_type: openai_compatible
    base_url: http://engine:4000/v1
    api_key: fixture-key
    model: chat-a
    reasoning_effort: low
    unknown_provider_key: kept
"""

    def test_default_order_and_model_change_together(self, tmp_path):
        with patch.object(Config, "get_config_dir", return_value=tmp_path):
            config_file = tmp_path / "config.yaml"
            config_file.write_text(self._CONFIG_TEXT)

            Config.set_chat_selection("engine", "chat-b")

            config = yaml.safe_load(config_file.read_text())
            assert config["default_provider"] == "engine"
            # The rest of the failover order keeps its order.
            assert config["provider_order"] == ["engine", "opencode", "backup"]
            assert config["providers"]["engine"]["model"] == "chat-b"
            # Other provider fields and unknown keys are preserved.
            assert config["providers"]["engine"]["reasoning_effort"] == "low"
            assert config["providers"]["engine"]["unknown_provider_key"] == "kept"
            assert config["providers"]["engine"]["api_key"] == "fixture-key"
            assert config["providers"]["engine"]["base_url"] == "http://engine:4000/v1"
            # A non-chat selection must not change another provider's model.
            assert config["providers"]["opencode"]["model"] == "chat-a"

    def test_saves_through_one_write(self, tmp_path):
        with patch.object(Config, "get_config_dir", return_value=tmp_path):
            config_file = tmp_path / "config.yaml"
            config_file.write_text(self._CONFIG_TEXT)

            replaces = []
            real_replace = os.replace

            def record_replace(src, dst):
                replaces.append((src, dst))
                return real_replace(src, dst)

            with patch.object(os, "replace", side_effect=record_replace):
                Config.set_chat_selection("engine", "chat-b")

            assert len(replaces) == 1

    def test_order_is_deduplicated(self, tmp_path):
        with patch.object(Config, "get_config_dir", return_value=tmp_path):
            config_file = tmp_path / "config.yaml"
            config_file.write_text(
                """
default_provider: opencode
provider_order:
  - opencode
  - engine
  - engine
providers:
  opencode:
    provider_type: openai_compatible
    model: chat-a
  engine:
    provider_type: openai_compatible
    model: chat-a
"""
            )

            Config.set_chat_selection("engine", "chat-b")
            config = yaml.safe_load(config_file.read_text())
            assert config["provider_order"] == ["engine", "opencode"]

    def test_provider_name_containing_dots_is_not_parsed_as_a_path(self, tmp_path):
        with patch.object(Config, "get_config_dir", return_value=tmp_path):
            config_file = tmp_path / "config.yaml"
            config_file.write_text(
                """
default_provider: opencode
providers:
  skk.private.22:
    provider_type: openai_compatible
    base_url: http://engine:4000/v1
    model: chat-a
"""
            )

            Config.set_chat_selection("skk.private.22", "chat-b")
            config = yaml.safe_load(config_file.read_text())
            assert config["default_provider"] == "skk.private.22"
            assert config["provider_order"] == ["skk.private.22"]
            assert config["providers"]["skk.private.22"]["model"] == "chat-b"

    def test_unknown_provider_raises(self, tmp_path):
        with patch.object(Config, "get_config_dir", return_value=tmp_path):
            config_file = tmp_path / "config.yaml"
            config_file.write_text(self._CONFIG_TEXT)
            before = config_file.read_text()

            with pytest.raises(ValueError, match="Unknown provider"):
                Config.set_chat_selection("ghost", "chat-b")

            assert config_file.read_text() == before



class TestGetProviderReferences:
    """Config keys that depend on one provider (Ticket 7).

    Every value below is a placeholder, never a real credential: the reference
    list is made of key names only, and no test here prints or asserts a value.
    """

    _CONFIG_TEXT = """
default_provider: engine
provider_order:
  - engine
  - other
embedding_backend: local
embedding_provider: engine
embedding_model: embedding-placeholder
corpus:
  metadata_model: provider:engine
  metadata_provider: engine
  cleanup_model: provider:engine
  cleanup_provider: engine
  ocr_model: vision-placeholder
  ocr_provider: engine
  rerank: false
  rerank_model: ggml-placeholder
  rerank_provider: engine
providers:
  engine:
    provider_type: openai_compatible
    base_url: http://engine.example/v1
    api_key: placeholder-not-a-secret
    model: chat-a
  other:
    provider_type: ollama
    model: chat-b
"""

    def test_returns_every_reference_in_the_documented_order(self, tmp_path):
        with patch.object(Config, "get_config_dir", return_value=tmp_path):
            (tmp_path / "config.yaml").write_text(self._CONFIG_TEXT)

            # Both model keys name a role provider, so each `provider:engine` there is
            # the model this task actually calls: a literal wire model, not a link.
            assert Config.get_provider_references("engine") == [
                "default_provider",
                "provider_order",
                "embedding_provider",
                "corpus.metadata_provider",
                "corpus.cleanup_provider",
                "corpus.ocr_provider",
                "corpus.rerank_provider",
            ]

    def test_legacy_links_are_reported_when_the_role_provider_is_empty(self, tmp_path):
        """A `provider:<name>` model is a link only while its role names no provider."""
        with patch.object(Config, "get_config_dir", return_value=tmp_path):
            (tmp_path / "config.yaml").write_text(
                """
default_provider: engine
provider_order:
  - engine
embedding_provider: engine
corpus:
  metadata_provider: ""
  metadata_model: provider:engine
  cleanup_provider: ""
  cleanup_model: provider:engine
  ocr_provider: engine
  rerank_provider: engine
providers:
  engine:
    provider_type: openai_compatible
    model: chat-a
"""
            )

            # The documented order is unchanged: an empty role provider is not a
            # reference, so the two legacy links sit at the end of the reported list.
            assert Config.get_provider_references("engine") == [
                "default_provider",
                "provider_order",
                "embedding_provider",
                "corpus.ocr_provider",
                "corpus.rerank_provider",
                "corpus.metadata_model",
                "corpus.cleanup_model",
            ]

    def test_a_literal_model_is_not_a_reference_when_its_role_has_a_provider(
        self, tmp_path
    ):
        """The role provider decides: a named provider makes the model a wire model."""
        with patch.object(Config, "get_config_dir", return_value=tmp_path):
            (tmp_path / "config.yaml").write_text(
                """
corpus:
  metadata_provider: other
  metadata_model: provider:engine
  cleanup_provider: other
  cleanup_model: provider:engine
providers:
  engine:
    provider_type: openai_compatible
    model: chat-a
  other:
    provider_type: ollama
    model: chat-b
"""
            )

            # Both roles call `other`; nothing here refers to engine.
            assert Config.get_provider_references("engine") == []
            assert Config.get_provider_references("other") == [
                "corpus.metadata_provider",
                "corpus.cleanup_provider",
            ]

    def test_the_removal_list_follows_the_same_rule(self, tmp_path):
        """Removal reports the role reference, never a literal model key."""
        with patch.object(Config, "get_config_dir", return_value=tmp_path):
            (tmp_path / "config.yaml").write_text(
                """
default_provider: engine
corpus:
  metadata_provider: engine
  metadata_model: provider:engine
  cleanup_provider: ""
  cleanup_model: provider:engine
providers:
  engine:
    provider_type: openai_compatible
    model: chat-a
"""
            )

            references = Config.get_provider_references("engine")
            # Metadata is a literal model behind a named provider: the role key is the
            # reference. Cleanup has no provider, so its model is a real link.
            assert references == [
                "default_provider",
                "corpus.metadata_provider",
                "corpus.cleanup_model",
            ]
            assert "corpus.metadata_model" not in references

    def test_includes_inactive_references(self, tmp_path):
        """A stored name is still a reference even when its task uses another source."""
        with patch.object(Config, "get_config_dir", return_value=tmp_path):
            (tmp_path / "config.yaml").write_text(
                """
default_provider: main
embedding_backend: local
embedding_provider: engine
corpus:
  rerank: false
  rerank_provider: engine
  metadata_model: main
  metadata_provider: engine
providers:
  engine:
    provider_type: openai_compatible
    model: chat-a
"""
            )

            # The local backend ignores embedding_provider and the rerank stage is
            # off, yet both stored names are reported.
            assert Config.get_provider_references("engine") == [
                "embedding_provider",
                "corpus.metadata_provider",
                "corpus.rerank_provider",
            ]

    def test_returns_empty_when_nothing_refers_to_the_provider(self, tmp_path):
        with patch.object(Config, "get_config_dir", return_value=tmp_path):
            (tmp_path / "config.yaml").write_text(
                """
default_provider: engine
provider_order:
  - engine
corpus:
  metadata_model: provider:engine
providers:
  engine:
    provider_type: openai_compatible
    model: chat-a
  other:
    provider_type: ollama
    model: chat-b
"""
            )

            # A configured provider that no key names, an unknown name, and an
            # empty name all give an empty list.
            assert Config.get_provider_references("other") == []
            assert Config.get_provider_references("ghost") == []
            assert Config.get_provider_references("") == []

    def test_provider_order_is_reported_once(self, tmp_path):
        with patch.object(Config, "get_config_dir", return_value=tmp_path):
            (tmp_path / "config.yaml").write_text(
                """
provider_order:
  - engine
  - engine
providers:
  engine:
    provider_type: openai_compatible
    model: chat-a
"""
            )

            assert Config.get_provider_references("engine") == ["provider_order"]

    def test_linked_model_keys_need_the_exact_provider_prefix(self, tmp_path):
        """A bare model name is not a reference; only 'provider:<name>' is."""
        with patch.object(Config, "get_config_dir", return_value=tmp_path):
            (tmp_path / "config.yaml").write_text(
                """
corpus:
  metadata_model: engine
  cleanup_model: provider:engine
providers:
  engine:
    provider_type: openai_compatible
    model: chat-a
"""
            )

            assert Config.get_provider_references("engine") == ["corpus.cleanup_model"]

    def test_only_key_names_are_returned(self, tmp_path):
        """No value, and so no credential or display label, can reach the caller."""
        with patch.object(Config, "get_config_dir", return_value=tmp_path):
            (tmp_path / "config.yaml").write_text(self._CONFIG_TEXT)

            references = Config.get_provider_references("engine")
            assert references
            for key in references:
                # Every item is one of the documented key names, never a value.
                assert key in Config.PROVIDER_REFERENCE_KEYS
                assert "placeholder" not in key
            # The provider's own fields are never reported, so a stored
            # credential can never reach the output.
            assert not any(key.startswith("providers.") for key in references)


class TestRenameProvider:
    """Rename one provider and move every reference in one atomic write."""

    _CONFIG_TEXT = """
default_provider: engine
provider_order:
  - engine
  - other
embedding_provider: engine
embedding_model: embedding-placeholder
corpus:
  metadata_model: provider:engine
  metadata_provider: engine
  cleanup_model: main
  cleanup_provider: engine
  ocr_model: vision-placeholder
  ocr_provider: engine
  rerank: true
  rerank_provider: engine
  ollama_host: http://engine.example:11434
providers:
  engine:
    provider_type: openai_compatible
    base_url: http://engine.example/v1
    api_key: placeholder-not-a-secret
    model: chat-a
    reasoning_effort: low
    auth_method: oauth
    oauth:
      access_token: placeholder-access
      refresh_token: placeholder-refresh
    unknown_provider_key: kept
  other:
    provider_type: ollama
    base_url: http://localhost:11434
    model: chat-b
"""

    def _write(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text(self._CONFIG_TEXT)
        return config_file

    def test_updates_every_reference_in_one_write(self, tmp_path):
        with patch.object(Config, "get_config_dir", return_value=tmp_path):
            config_file = self._write(tmp_path)

            replaces = []
            real_replace = os.replace

            def record_replace(src, dst):
                replaces.append((src, dst))
                return real_replace(src, dst)

            with patch.object(os, "replace", side_effect=record_replace):
                Config.rename_provider("engine", "desk")

            assert len(replaces) == 1

            config = yaml.safe_load(config_file.read_text())
            assert config["default_provider"] == "desk"
            assert config["provider_order"] == ["desk", "other"]
            assert config["embedding_provider"] == "desk"
            assert config["corpus"]["metadata_provider"] == "desk"
            assert config["corpus"]["cleanup_provider"] == "desk"
            assert config["corpus"]["ocr_provider"] == "desk"
            assert config["corpus"]["rerank_provider"] == "desk"
            # `provider:engine` here is a literal wire model: metadata has its own
            # provider, so the rename moves the provider key and leaves the model as is.
            assert config["corpus"]["metadata_model"] == "provider:engine"
            # A task with its own explicit model keeps it.
            assert config["corpus"]["cleanup_model"] == "main"
            assert config["corpus"]["ocr_model"] == "vision-placeholder"
            assert config["embedding_model"] == "embedding-placeholder"
            assert config["corpus"]["ollama_host"] == "http://engine.example:11434"

    def test_preserves_fields_oauth_unknown_keys_and_other_providers(self, tmp_path):
        with patch.object(Config, "get_config_dir", return_value=tmp_path):
            config_file = self._write(tmp_path)

            Config.rename_provider("engine", "desk")

            config = yaml.safe_load(config_file.read_text())
            entry = config["providers"]["desk"]
            assert entry["provider_type"] == "openai_compatible"
            assert entry["model"] == "chat-a"
            assert entry["reasoning_effort"] == "low"
            assert entry["auth_method"] == "oauth"
            assert entry["oauth"] == {
                "access_token": "placeholder-access",
                "refresh_token": "placeholder-refresh",
            }
            assert entry["unknown_provider_key"] == "kept"
            # Every other provider is left as it was.
            assert config["providers"]["other"] == {
                "provider_type": "ollama",
                "base_url": "http://localhost:11434",
                "model": "chat-b",
            }
            assert set(config["providers"]) == {"desk", "other"}
            assert "engine" not in config["providers"]

    def test_literal_model_strings_survive_the_rename_for_both_roles(self, tmp_path):
        """Only the role provider moves; a model string that names a provider stays."""
        with patch.object(Config, "get_config_dir", return_value=tmp_path):
            config_file = tmp_path / "config.yaml"
            config_file.write_text(
                """
corpus:
  metadata_provider: engine
  metadata_model: provider:engine
  cleanup_provider: engine
  cleanup_model: provider:engine
providers:
  engine:
    provider_type: openai_compatible
    model: chat-a
"""
            )

            Config.rename_provider("engine", "desk")

            config = yaml.safe_load(config_file.read_text())
            assert config["corpus"]["metadata_provider"] == "desk"
            assert config["corpus"]["cleanup_provider"] == "desk"
            # Both strings are the model the task calls, so the rename must not touch
            # them: `provider:engine` is the model id on that provider.
            assert config["corpus"]["metadata_model"] == "provider:engine"
            assert config["corpus"]["cleanup_model"] == "provider:engine"

    def test_a_legacy_link_is_rewritten_while_a_literal_model_is_not(self, tmp_path):
        """The same key holds a link or a wire model, depending on its role provider."""
        with patch.object(Config, "get_config_dir", return_value=tmp_path):
            config_file = tmp_path / "config.yaml"
            config_file.write_text(
                """
corpus:
  metadata_provider: ""
  metadata_model: provider:engine
  cleanup_provider: other
  cleanup_model: provider:engine
providers:
  engine:
    provider_type: openai_compatible
    model: chat-a
  other:
    provider_type: ollama
    model: chat-b
"""
            )

            Config.rename_provider("engine", "desk")

            config = yaml.safe_load(config_file.read_text())
            # Metadata has no provider, so its string is the legacy link: it follows
            # the rename. Cleanup names `other`, so its string stays as stored.
            assert config["corpus"]["metadata_model"] == "provider:desk"
            assert config["corpus"]["cleanup_model"] == "provider:engine"
            # The other role's provider is not engine, so it is left alone.
            assert config["corpus"]["cleanup_provider"] == "other"

    def test_duplicate_name_leaves_the_file_byte_identical(self, tmp_path):
        with patch.object(Config, "get_config_dir", return_value=tmp_path):
            config_file = self._write(tmp_path)
            before = config_file.read_bytes()

            replaces = []
            real_replace = os.replace

            def record_replace(src, dst):
                replaces.append((src, dst))
                return real_replace(src, dst)

            def attempt_rename():
                with pytest.raises(ValueError, match="already exists"):
                    Config.rename_provider("engine", "other")

            with patch.object(os, "replace", side_effect=record_replace):
                attempt_rename()

            # Validation happened before the first change, so nothing was written.
            assert replaces == []
            assert config_file.read_bytes() == before
            assert "engine" in yaml.safe_load(config_file.read_text())["provider_order"]

    def test_unknown_old_name_raises_and_leaves_the_file_unchanged(self, tmp_path):
        with patch.object(Config, "get_config_dir", return_value=tmp_path):
            config_file = self._write(tmp_path)
            before = config_file.read_bytes()

            with pytest.raises(ValueError, match="Unknown provider"):
                Config.rename_provider("ghost", "desk")

            assert config_file.read_bytes() == before

    def test_unknown_old_name_raises_even_when_the_names_match(self, tmp_path):
        """Validation runs before the same-name no-op, so nothing silently succeeds."""
        with patch.object(Config, "get_config_dir", return_value=tmp_path):
            config_file = self._write(tmp_path)
            before = config_file.read_bytes()

            replaces = []
            real_replace = os.replace

            def record_replace(src, dst):
                replaces.append((src, dst))
                return real_replace(src, dst)

            def attempt_rename():
                with pytest.raises(ValueError, match="Unknown provider"):
                    Config.rename_provider("ghost", "ghost")

            with patch.object(os, "replace", side_effect=record_replace):
                attempt_rename()

            assert replaces == []
            assert config_file.read_bytes() == before

    def test_empty_new_name_raises_before_any_write(self, tmp_path):
        with patch.object(Config, "get_config_dir", return_value=tmp_path):
            config_file = self._write(tmp_path)
            before = config_file.read_bytes()

            with pytest.raises(ValueError, match="cannot be empty"):
                Config.rename_provider("engine", "   ")

            assert config_file.read_bytes() == before

    def test_same_name_is_a_no_op(self, tmp_path):
        with patch.object(Config, "get_config_dir", return_value=tmp_path):
            config_file = self._write(tmp_path)
            before = config_file.read_bytes()

            replaces = []
            real_replace = os.replace

            def record_replace(src, dst):
                replaces.append((src, dst))
                return real_replace(src, dst)

            with patch.object(os, "replace", side_effect=record_replace):
                Config.rename_provider("engine", "engine")

            assert replaces == []
            assert config_file.read_bytes() == before

    def test_provider_order_keeps_its_order_and_deduplicates(self, tmp_path):
        with patch.object(Config, "get_config_dir", return_value=tmp_path):
            config_file = tmp_path / "config.yaml"
            config_file.write_text(
                """
provider_order:
  - other
  - engine
  - engine
providers:
  other:
    provider_type: ollama
    model: chat-b
  engine:
    provider_type: openai_compatible
    model: chat-a
"""
            )

            Config.rename_provider("engine", "desk")

            config = yaml.safe_load(config_file.read_text())
            assert config["provider_order"] == ["other", "desk"]
            assert config["providers"]["desk"]["model"] == "chat-a"

    def test_a_duplicate_already_stored_is_not_repaired(self, tmp_path):
        """Only the duplicate the rename itself creates is removed."""
        with patch.object(Config, "get_config_dir", return_value=tmp_path):
            config_file = tmp_path / "config.yaml"
            config_file.write_text(
                """
provider_order:
  - backup
  - backup
  - engine
providers:
  backup:
    provider_type: ollama
    model: chat-b
  engine:
    provider_type: openai_compatible
    model: chat-a
"""
            )

            Config.rename_provider("engine", "desk")

            config = yaml.safe_load(config_file.read_text())
            assert config["provider_order"] == ["backup", "backup", "desk"]

    def test_nothing_to_update_is_still_one_move_of_the_entry(self, tmp_path):
        """A provider with no references is renamed by itself, in one write."""
        with patch.object(Config, "get_config_dir", return_value=tmp_path):
            config_file = tmp_path / "config.yaml"
            config_file.write_text(
                """
default_provider: other
providers:
  other:
    provider_type: ollama
    model: chat-b
  engine:
    provider_type: openai_compatible
    model: chat-a
"""
            )

            replaces = []
            real_replace = os.replace

            def record_replace(src, dst):
                replaces.append((src, dst))
                return real_replace(src, dst)

            with patch.object(os, "replace", side_effect=record_replace):
                Config.rename_provider("engine", "desk")

            assert len(replaces) == 1
            config = yaml.safe_load(config_file.read_text())
            assert config["default_provider"] == "other"
            assert set(config["providers"]) == {"other", "desk"}
            assert config["providers"]["other"]["model"] == "chat-b"
            assert config["providers"]["desk"]["model"] == "chat-a"

    def test_draft_and_rename_are_one_write(self, tmp_path):
        """A draft rename updates the entry and every reference in one write."""
        with patch.object(Config, "get_config_dir", return_value=tmp_path):
            config_file = self._write(tmp_path)

            draft = {
                "provider_type": "openai_compatible",
                "base_url": "http://desk.example/v1",
                "model": "chat-new",
            }

            replaces = []
            real_replace = os.replace

            def record_replace(src, dst):
                replaces.append((src, dst))
                return real_replace(src, dst)

            with patch.object(os, "replace", side_effect=record_replace):
                Config.rename_provider("engine", "desk", provider_config=draft)

            # One write total: the key move, the references, and the draft fields.
            assert len(replaces) == 1

            config = yaml.safe_load(config_file.read_text())
            # The entry under the new key is the draft, not the saved fields.
            assert config["providers"]["desk"] == draft
            # The references moved with it in the same write.
            assert config["default_provider"] == "desk"
            assert config["provider_order"] == ["desk", "other"]
            assert config["embedding_provider"] == "desk"
            assert config["corpus"]["metadata_provider"] == "desk"
            # A literal wire model keeps its stored text even when the draft is applied
            # in the same write.
            assert config["corpus"]["metadata_model"] == "provider:engine"
            assert "engine" not in config["providers"]

    def test_failing_replace_keeps_file_and_cache_on_the_old_name(self, tmp_path):
        """The draft and the references cannot be split by a failed move."""
        with patch.object(Config, "get_config_dir", return_value=tmp_path):
            config_file = self._write(tmp_path)
            before = config_file.read_bytes()
            cache_before = Config._load_config()
            assert cache_before is not None
            snapshot = copy.deepcopy(cache_before)

            with (
                patch.object(os, "replace", side_effect=OSError("replace failed")),
                pytest.raises(OSError),
            ):
                Config.rename_provider(
                    "engine",
                    "desk",
                    provider_config={"provider_type": "openai_compatible", "model": "x"},
                )

            # Nothing half-applied: the file still holds the old key and the old
            # fields, and the cache is still the pre-save config.
            assert config_file.read_bytes() == before
            assert Config._config_cache is cache_before
            assert Config._config_cache == snapshot
            config = yaml.safe_load(config_file.read_text())
            assert "engine" in config["providers"]
            assert "desk" not in config["providers"]
            assert config["default_provider"] == "engine"
            assert config["corpus"]["metadata_model"] == "provider:engine"
            # Only its own temporary file existed, and it is gone after the failure.
            assert [p.name for p in tmp_path.iterdir()] == ["config.yaml"]

    def test_two_argument_rename_keeps_the_saved_entry_fields(self, tmp_path):
        """An omitted draft must not turn a rename into a field overwrite."""
        with patch.object(Config, "get_config_dir", return_value=tmp_path):
            config_file = self._write(tmp_path)

            Config.rename_provider("engine", "desk", provider_config=None)

            entry = yaml.safe_load(config_file.read_text())["providers"]["desk"]
            assert entry["model"] == "chat-a"
            assert entry["base_url"] == "http://engine.example/v1"
            assert entry["oauth"] == {
                "access_token": "placeholder-access",
                "refresh_token": "placeholder-refresh",
            }
            assert entry["unknown_provider_key"] == "kept"

    def test_draft_is_copied_and_not_aliased_into_the_saved_config(self, tmp_path):
        """The caller's draft stays out of the structure the write produced."""
        with patch.object(Config, "get_config_dir", return_value=tmp_path):
            self._write(tmp_path)

            draft = {"provider_type": "openai_compatible", "model": "chat-new"}
            Config.rename_provider("engine", "desk", provider_config=draft)

            cache = Config._config_cache
            assert cache is not None
            entry = cache["providers"]["desk"]
            assert entry is not draft
            assert entry == {"provider_type": "openai_compatible", "model": "chat-new"}
            draft["model"] = "changed after the write"
            assert entry["model"] == "chat-new"


class TestSaveProviderDoesNotMutateTheCache:
    """A provider write works on a copy, so a failed move cannot leak a draft.

    The write path is the same for a new key, an overwrite, and an edit that keeps
    the saved name: `save_provider` is called with the wizard's draft.
    """

    _CONFIG_TEXT = """
default_provider: engine
providers:
  engine:
    provider_type: openai_compatible
    base_url: http://engine.example/v1
    model: chat-a
    oauth:
      access_token: placeholder-access
    unknown_provider_key: kept
"""

    def _write(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text(self._CONFIG_TEXT)
        return config_file

    def _failed_save(self, config_file, name, draft):
        """One provider write whose os.replace fails."""
        cache = Config._load_config()
        assert cache is not None
        snapshot = copy.deepcopy(cache)
        before = config_file.read_bytes()

        with (
            patch.object(os, "replace", side_effect=OSError("replace failed")),
            pytest.raises(OSError),
        ):
            Config.save_provider(name, draft)

        # The file still holds the pre-save config, and the cache is still the
        # object it was before the call.
        assert config_file.read_bytes() == before
        assert Config._config_cache is cache
        assert Config._config_cache == snapshot
        # Only this call's own temporary file existed, and it is gone.
        assert [p.name for p in config_file.parent.iterdir()] == [config_file.name]
        return cache, snapshot

    def test_failed_new_provider_save_keeps_file_and_cache(self, tmp_path):
        """A rejected create must not put the draft into the cache."""
        with patch.object(Config, "get_config_dir", return_value=tmp_path):
            config_file = self._write(tmp_path)
            draft = {"provider_type": "openai_compatible", "model": "draft-model"}

            cache, _ = self._failed_save(config_file, "desk", draft)

            assert "desk" not in cache["providers"]
            assert cache["providers"] is not draft

            # A later unrelated save writes the pre-save cache, not the failed draft.
            Config.set_config_value("logging.level", "DEBUG")
            written = yaml.safe_load(config_file.read_text())
            assert written["logging"]["level"] == "DEBUG"
            assert "desk" not in written["providers"]
            assert written["providers"]["engine"]["model"] == "chat-a"

    def test_failed_overwrite_keeps_the_saved_entry_and_the_cache(self, tmp_path):
        """A rejected overwrite must not put the draft into the cache."""
        with patch.object(Config, "get_config_dir", return_value=tmp_path):
            config_file = self._write(tmp_path)
            draft = {"provider_type": "openai_compatible", "model": "draft-model"}

            cache, _ = self._failed_save(config_file, "engine", draft)

            entry = cache["providers"]["engine"]
            assert entry["model"] == "chat-a"
            assert entry["unknown_provider_key"] == "kept"
            assert entry is not draft

            Config.set_config_value("logging.level", "DEBUG")
            written = yaml.safe_load(config_file.read_text())
            assert written["providers"]["engine"]["model"] == "chat-a"
            assert "draft-model" not in str(written)

    def test_failed_edit_with_the_saved_name_keeps_file_and_cache(self, tmp_path):
        """The edit path is a save_provider call with an edited draft."""
        with patch.object(Config, "get_config_dir", return_value=tmp_path):
            config_file = self._write(tmp_path)
            draft = {
                "provider_type": "openai_compatible",
                "base_url": "http://desk-box:4000/v1",
                "model": "chat-a",
                "oauth": {"access_token": "placeholder-access"},
                "unknown_provider_key": "kept",
            }

            cache, snapshot = self._failed_save(config_file, "engine", draft)

            assert cache["providers"]["engine"]["base_url"] == (
                "http://engine.example/v1"
            )
            assert cache["providers"]["engine"] == snapshot["providers"]["engine"]
            assert cache["providers"]["engine"] is not draft

            Config.set_config_value("logging.level", "DEBUG")
            written = yaml.safe_load(config_file.read_text())
            assert written["providers"]["engine"]["base_url"] == (
                "http://engine.example/v1"
            )

    def test_successful_save_stores_a_copy_of_the_draft(self, tmp_path):
        """A write that succeeds stores a copy, never the caller's mapping."""
        with patch.object(Config, "get_config_dir", return_value=tmp_path):
            self._write(tmp_path)
            draft = {"provider_type": "openai_compatible", "model": "draft-model"}

            Config.save_provider("engine", draft)

            cache = Config._config_cache
            assert cache is not None
            entry = cache["providers"]["engine"]
            assert entry is not draft
            draft["model"] = "changed after the write"
            assert entry["model"] == "draft-model"


class TestOtherProviderWritesWorkOnACopy:
    """Provider writes work on a copy, as `save_provider` does.

    `update_provider_config` and `remove_provider` follow the same rule: the cached
    mapping is never mutated, so a failed ``os.replace`` leaves the file, the cache
    object, and its contents as they were, and an unsaved change cannot be picked up
    by a later write. The cache is updated only after the move succeeded.
    """

    _CONFIG_TEXT = """
default_provider: engine
providers:
  engine:
    provider_type: openai_compatible
    base_url: http://engine.example/v1
    model: chat-a
    oauth:
      access_token: placeholder-access
    unknown_provider_key: kept
  ollama:
    provider_type: ollama
    base_url: http://localhost:11434
    model: llama3.2
"""

    def _write(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text(self._CONFIG_TEXT)
        return config_file

    def _before(self, config_file):
        """The cache object as it is before the call, plus the file bytes."""
        cache = Config._load_config()
        assert cache is not None
        return cache, copy.deepcopy(cache), config_file.read_bytes()

    def test_failed_update_keeps_file_and_cache(self, tmp_path):
        """A rejected provider edit must not put the draft into the cache."""
        with patch.object(Config, "get_config_dir", return_value=tmp_path):
            config_file = self._write(tmp_path)
            cache, snapshot, before = self._before(config_file)
            draft = {"provider_type": "openai_compatible", "model": "draft-model"}

            with (
                patch.object(os, "replace", side_effect=OSError("replace failed")),
                pytest.raises(OSError),
            ):
                Config.update_provider_config("desk", draft)

            assert config_file.read_bytes() == before
            assert Config._config_cache is cache
            assert Config._config_cache == snapshot
            # Only this call's own temporary file existed, and it is gone.
            assert [p.name for p in config_file.parent.iterdir()] == [config_file.name]

            assert "desk" not in cache["providers"]
            assert cache["providers"] is not draft

            # A later unrelated save writes the pre-call cache, not the failed draft.
            Config.set_config_value("logging.level", "DEBUG")
            written = yaml.safe_load(config_file.read_text())
            assert written["logging"]["level"] == "DEBUG"
            assert "desk" not in written["providers"]
            assert written["providers"]["engine"]["model"] == "chat-a"

    def test_failed_removal_keeps_file_and_cache(self, tmp_path):
        """A rejected removal must not delete the provider from the cache."""
        with patch.object(Config, "get_config_dir", return_value=tmp_path):
            config_file = self._write(tmp_path)
            cache, snapshot, before = self._before(config_file)

            with (
                patch.object(os, "replace", side_effect=OSError("replace failed")),
                pytest.raises(OSError),
            ):
                Config.remove_provider("engine")

            assert config_file.read_bytes() == before
            assert Config._config_cache is cache
            assert Config._config_cache == snapshot
            assert cache is not None
            assert "engine" in cache["providers"]
            assert [p.name for p in config_file.parent.iterdir()] == [config_file.name]

            # A later unrelated save still writes the provider, not the deletion.
            Config.set_config_value("logging.level", "DEBUG")
            written = yaml.safe_load(config_file.read_text())
            assert written["logging"]["level"] == "DEBUG"
            assert "engine" in written["providers"]
            assert written["providers"]["engine"]["model"] == "chat-a"

    def test_successful_update_writes_once_and_updates_the_cache(self, tmp_path):
        """A write that succeeds stores a copy, never the caller's mapping."""
        with patch.object(Config, "get_config_dir", return_value=tmp_path):
            config_file = self._write(tmp_path)
            cache, snapshot, _ = self._before(config_file)
            draft = {"provider_type": "openai_compatible", "model": "draft-model"}

            with patch.object(os, "replace", wraps=os.replace) as moved:
                Config.update_provider_config("engine", draft)
            assert moved.call_count == 1

            after = Config._config_cache
            assert after is not None
            assert after is not cache
            entry = after["providers"]["engine"]
            assert entry is not draft
            assert entry == {
                "provider_type": "openai_compatible",
                "model": "draft-model",
            }
            # The object that was the cache before the call is still intact.
            assert cache == snapshot
            assert cache["providers"]["engine"]["model"] == "chat-a"

            written = yaml.safe_load(config_file.read_text())
            assert written["providers"]["engine"]["model"] == "draft-model"

            draft["model"] = "changed after the write"
            assert after["providers"]["engine"]["model"] == "draft-model"

    def test_successful_removal_writes_once_and_updates_the_cache(self, tmp_path):
        """A removal that succeeds updates the cache only after the move."""
        with patch.object(Config, "get_config_dir", return_value=tmp_path):
            config_file = self._write(tmp_path)
            cache, snapshot, _ = self._before(config_file)

            with patch.object(os, "replace", wraps=os.replace) as moved:
                assert Config.remove_provider("engine") is True
            assert moved.call_count == 1

            after = Config._config_cache
            assert after is not None
            assert after is not cache
            assert "engine" not in after["providers"]
            assert "ollama" in after["providers"]
            assert after["providers"] is not cache["providers"]
            # The object that was the cache before the call still has the provider.
            assert cache == snapshot
            assert "engine" in cache["providers"]

            written = yaml.safe_load(config_file.read_text())
            assert "engine" not in written["providers"]
            assert "ollama" in written["providers"]
