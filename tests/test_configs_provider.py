from unittest.mock import patch

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
        """Test get_providers returns providers from config file."""
        with patch.object(Config, "get_config_dir", return_value=tmp_path):
            config_file = tmp_path / "config.yaml"
            config_file.write_text(
                """
providers:
  openai:
    provider_type: openai
    api_key: sk-test
    model: gpt-4o
  ollama:
    provider_type: ollama
    base_url: http://localhost:11434
    model: llama3.2
"""
            )
            providers = Config.get_providers()
            assert "openai" in providers
            assert "ollama" in providers
            assert providers["openai"]["provider_type"] == "openai"

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


class TestGetProviderConfig:
    """Tests for get_provider_config()."""

    def test_get_provider_config_returns_default_provider(self, tmp_path):
        """Test get_provider_config returns the default provider's config."""
        with patch.object(Config, "get_config_dir", return_value=tmp_path):
            config_file = tmp_path / "config.yaml"
            config_file.write_text(
                """
default_provider: ollama
providers:
  openai:
    provider_type: openai
    api_key: sk-test
    model: gpt-4o
  ollama:
    provider_type: ollama
    base_url: http://localhost:11434
    model: llama3.2
"""
            )
            # Clear cache
            provider_config = Config.get_provider_config()
            assert provider_config is not None
            assert provider_config["provider_type"] == "ollama"
            assert provider_config["base_url"] == "http://localhost:11434"
            assert provider_config["model"] == "llama3.2"

    def test_get_provider_config_no_default_provider(self, tmp_path):
        """Test get_provider_config returns None when no default set."""
        with patch.object(Config, "get_config_dir", return_value=tmp_path):
            config_file = tmp_path / "config.yaml"
            config_file.write_text(
                """
providers:
  openai:
    provider_type: openai
    model: gpt-4o
"""
            )
            # Clear cache
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

    def test_set_default_provider_creates_section(self, tmp_path):
        """Test set_default_provider works when no config exists."""
        with patch.object(Config, "get_config_dir", return_value=tmp_path):
            Config.set_default_provider("openai")
            config_file = tmp_path / "config.yaml"
            with open(config_file) as f:
                config = yaml.safe_load(f)
            assert config["default_provider"] == "openai"
