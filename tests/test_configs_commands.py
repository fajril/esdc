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
            assert config["logging"]["level"] == "DEBUG"

    def test_set_boolean_key(self, tmp_path):
        with patch.object(Config, "get_config_dir", return_value=tmp_path):
            config_file = tmp_path / "config.yaml"
            config_file.write_text("api:\n  verify_ssl: true\n")
            Config._config_cache = None

            Config.set_config_value("api.verify_ssl", "false")
            Config._config_cache = None
            config = Config._load_config()
            assert config["api"]["verify_ssl"] is False

    def test_set_int_key(self, tmp_path):
        with patch.object(Config, "get_config_dir", return_value=tmp_path):
            config_file = tmp_path / "config.yaml"
            config_file.write_text("cache:\n  sql_ttl: 604800\n")
            Config._config_cache = None

            Config.set_config_value("cache.sql_ttl", "300")
            Config._config_cache = None
            config = Config._load_config()
            assert config["cache"]["sql_ttl"] == 300

    def test_creates_nested_structure(self, tmp_path):
        with patch.object(Config, "get_config_dir", return_value=tmp_path):
            config_file = tmp_path / "config.yaml"
            config_file.write_text("api_url: https://example.com/\n")
            Config._config_cache = None

            Config.set_config_value("logging.level", "WARNING")
            Config._config_cache = None
            config = Config._load_config()
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
        """ENUM_CHOICES must include the default value for each key."""
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
