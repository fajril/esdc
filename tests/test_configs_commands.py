"""Tests for the configs CLI commands."""

from unittest.mock import patch

import pytest

from esdc.configs import ENUM_CHOICES, KEY_DESCRIPTIONS, Config


class TestGetAllConfigFlat:
    """Tests for get_all_config_flat()."""

    def test_returns_default_keys_when_no_config(self, tmp_path):
        with patch.object(Config, "get_config_dir", return_value=tmp_path):
            Config._config_cache = None
            flat = Config.get_all_config_flat()
            assert "api_url" in flat
            assert "api.verify_ssl" in flat
            assert "logging.level" in flat
            assert "cache.sql_ttl" in flat
            assert "tool_format" in flat
            assert flat["api.verify_ssl"] is True

    def test_merges_user_config_over_defaults(self, tmp_path):
        with patch.object(Config, "get_config_dir", return_value=tmp_path):
            config_file = tmp_path / "config.yaml"
            config_file.write_text("api_url: https://custom.example.com/\n")
            Config._config_cache = None
            flat = Config.get_all_config_flat()
            assert flat["api_url"] == "https://custom.example.com/"
            assert "logging.level" in flat

    def test_includes_nested_keys(self, tmp_path):
        with patch.object(Config, "get_config_dir", return_value=tmp_path):
            Config._config_cache = None
            flat = Config.get_all_config_flat()
            assert "logging.file.enabled" in flat
            assert "logging.file.path" in flat
            assert "logging.server.level" in flat


class TestSetConfigValue:
    """Tests for set_config_value()."""

    def test_set_top_level_key(self, tmp_path):
        with patch.object(Config, "get_config_dir", return_value=tmp_path):
            config_file = tmp_path / "config.yaml"
            config_file.write_text("api_url: https://old.example.com/\n")
            Config._config_cache = None

            Config.set_config_value("api_url", "https://new.example.com/")
            Config._config_cache = None
            assert Config.get_api_url() == "https://new.example.com/"

    def test_set_nested_key(self, tmp_path):
        with patch.object(Config, "get_config_dir", return_value=tmp_path):
            config_file = tmp_path / "config.yaml"
            config_file.write_text("logging:\n  level: INFO\n")
            Config._config_cache = None

            Config.set_config_value("logging.level", "DEBUG")
            Config._config_cache = None
            config = Config._load_config()
            assert config is not None
            assert config["logging"]["level"] == "DEBUG"

    def test_set_boolean_key(self, tmp_path):
        with patch.object(Config, "get_config_dir", return_value=tmp_path):
            config_file = tmp_path / "config.yaml"
            config_file.write_text("api:\n  verify_ssl: true\n")
            Config._config_cache = None

            Config.set_config_value("api.verify_ssl", "false")
            Config._config_cache = None
            config = Config._load_config()
            assert config is not None
            assert config["api"]["verify_ssl"] is False

    def test_set_int_key(self, tmp_path):
        with patch.object(Config, "get_config_dir", return_value=tmp_path):
            config_file = tmp_path / "config.yaml"
            config_file.write_text("cache:\n  sql_ttl: 604800\n")
            Config._config_cache = None

            Config.set_config_value("cache.sql_ttl", "300")
            Config._config_cache = None
            config = Config._load_config()
            assert config is not None
            assert config["cache"]["sql_ttl"] == 300

    def test_creates_nested_structure(self, tmp_path):
        with patch.object(Config, "get_config_dir", return_value=tmp_path):
            config_file = tmp_path / "config.yaml"
            config_file.write_text("api_url: https://example.com/\n")
            Config._config_cache = None

            Config.set_config_value("logging.level", "WARNING")
            Config._config_cache = None
            config = Config._load_config()
            assert config is not None
            assert config["logging"]["level"] == "WARNING"


class TestResetConfig:
    """Tests for reset_config()."""

    def test_reset_all(self, tmp_path):
        with patch.object(Config, "get_config_dir", return_value=tmp_path):
            config_file = tmp_path / "config.yaml"
            config_file.write_text(
                "api_url: https://custom.example.com/\nlogging:\n  level: DEBUG\n"
            )
            Config._config_cache = None

            Config.reset_config()
            Config._config_cache = None
            assert Config.get_api_url() == Config.BASE_API_URL_V2

    def test_reset_single_key(self, tmp_path):
        with patch.object(Config, "get_config_dir", return_value=tmp_path):
            config_file = tmp_path / "config.yaml"
            config_file.write_text(
                "api_url: https://custom.example.com/\nlogging:\n  level: DEBUG\n"
            )
            Config._config_cache = None

            Config.reset_config("logging.level")
            Config._config_cache = None
            assert Config.get_api_url() == "https://custom.example.com/"
            logging_config = Config.get_logging_config()
            assert logging_config["level"] == "INFO"

    def test_reset_unknown_key_raises(self, tmp_path):
        with patch.object(Config, "get_config_dir", return_value=tmp_path):
            Config._config_cache = None
            with pytest.raises(KeyError, match="Unknown config key"):
                Config.reset_config("nonexistent.key")


class TestKeyDescriptions:
    """Tests for KEY_DESCRIPTIONS and ENUM_CHOICES coverage."""

    def test_all_default_keys_have_descriptions(self, tmp_path):
        """Every key from get_all_config_flat() must have a description."""
        with patch.object(Config, "get_config_dir", return_value=tmp_path):
            Config._config_cache = None
            flat = Config.get_all_config_flat()
            missing = [k for k in flat if k not in KEY_DESCRIPTIONS]
            assert not missing, f"Missing descriptions for: {missing}"

    def test_enum_values_include_defaults(self, tmp_path):
        """Every persistent ENUM_CHOICES key includes its default value."""
        with patch.object(Config, "get_config_dir", return_value=tmp_path):
            Config._config_cache = None
            flat = Config.get_all_config_flat()
            for key, choices in ENUM_CHOICES.items():
                assert key in flat, f"ENUM_CHOICES key {key!r} not in config"
                assert str(flat[key]) in choices, (
                    f"Default value {flat[key]!r} for {key!r} "
                    f"not in ENUM_CHOICES {choices}"
                )

    def test_boolean_keys_not_in_enum_choices(self):
        """Boolean keys use their own Select, so must not overlap with ENUM_CHOICES."""
        overlap = Config.BOOLEAN_KEYS & set(ENUM_CHOICES.keys())
        assert not overlap, f"Boolean keys should not be in ENUM_CHOICES: {overlap}"

    def test_legacy_model_keys_have_no_persistent_home(self, tmp_path):
        """Old provider/corpus model keys are conversion-only, not config keys."""
        legacy = (
            "embedding_backend",
            "embedding_host",
            "embedding_model",
            "embedding_api_key",
            "embedding_provider",
            "corpus.metadata_provider",
            "corpus.cleanup_provider",
            "corpus.ocr_provider",
            "corpus.rerank_backend",
            "corpus.rerank_provider",
            "corpus.ollama_host",
            "corpus.n_gpu_layers",
            "default_provider",
            "provider_order",
        )
        with patch.object(Config, "get_config_dir", return_value=tmp_path):
            Config._config_cache = None
            flat = Config.get_all_config_flat()
            assert "models.chat.mode" in flat
            for key in legacy:
                assert key not in flat, key
                assert key not in KEY_DESCRIPTIONS, key

    def test_reading_flat_config_does_not_rewrite_the_file(self, tmp_path):
        """Opening the config (here: --show) must not write defaults into it."""
        with patch.object(Config, "get_config_dir", return_value=tmp_path):
            config_file = tmp_path / "config.yaml"
            original = "api_url: https://custom.example.com/\n"
            config_file.write_text(original)
            Config._config_cache = None

            flat = Config.get_all_config_flat()
            assert flat["models.chat.mode"] == "unset"
            assert config_file.read_text() == original
            assert not [p for p in tmp_path.iterdir() if p.name != "config.yaml"]


class TestShowConfigOutput:
    """`configs --show` prints masked secrets with the shared formatter."""

    def test_show_masks_secrets_and_keeps_non_secrets_visible(self, runner, isolated_config):
        from esdc.commands.configs import configs_app

        api_key = "sk-synthetic-fixture-key"
        access_token = "synthetic-fixture-access-token"
        refresh_token = "synthetic-fixture-refresh-token"
        original = (
            "api_url: https://engine.example/v1\n"
            "providers:\n"
            "  engine:\n"
            "    provider_type: openai_compatible\n"
            "    base_url: http://engine:4000/v1\n"
            "    model: chat-a\n"
            f"    api_key: {api_key}\n"
            "    oauth:\n"
            f"      access_token: {access_token}\n"
            f"      refresh_token: {refresh_token}\n"
        )
        config_dir = isolated_config / ".esdc"
        config_dir.mkdir(parents=True, exist_ok=True)
        config_file = config_dir / "config.yaml"
        config_file.write_text(original)
        Config._config_cache = None

        result = runner.invoke(configs_app, ["--show"])

        assert result.exit_code == 0
        output = result.output

        # The secrets themselves never reach the terminal.
        for secret in (api_key, access_token, refresh_token):
            assert secret not in output
            # Only the first two and last two characters may appear.
            assert secret[2:-2] not in output

        assert "providers.engine.api_key = sk****ey" in output
        assert "providers.engine.oauth.access_token = sy****en" in output
        assert "providers.engine.oauth.refresh_token = sy****en" in output

        # Non-secret values stay readable.
        assert "api_url = https://engine.example/v1" in output
        assert "providers.engine.model = chat-a" in output

        # Showing the config is a read, not a write.
        assert config_file.read_text() == original
