"""Tests for application configuration."""

import copy
import os
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from esdc.config_fields import parse_config_input
from esdc.configs import (
    ENUM_CHOICES,
    KEY_DESCRIPTIONS,
    MODEL_SECTIONS,
    SETTINGS_SECTIONS,
    Config,
)


class TestConfigDir:
    """Tests for get_config_dir()."""

    def test_get_config_dir_default(self):
        """Test default config directory is ~/.esdc."""
        with patch.dict(os.environ, {}, clear=True):
            config_dir = Config.get_config_dir()
            assert config_dir == Path.home() / ".esdc"

    def test_get_config_dir_env_override(self, tmp_path):
        """Test ESDC_CONFIG_DIR environment variable overrides default."""
        with patch.dict(os.environ, {"ESDC_CONFIG_DIR": str(tmp_path)}):
            config_dir = Config.get_config_dir()
            assert config_dir == tmp_path


class TestConfigFile:
    """Tests for config file operations."""

    def test_get_config_file(self, tmp_path):
        """Test config file path is within config directory."""
        with patch.object(Config, "get_config_dir", return_value=tmp_path):
            config_file = Config.get_config_file()
            assert config_file == tmp_path / "config.yaml"

    def test_init_config_creates_directory(self, tmp_path):
        """Test init_config creates config directory if not exists."""
        with patch.object(Config, "get_config_dir", return_value=tmp_path / "esdc"):
            Config.init_config()
            assert (tmp_path / "esdc").exists()

    def test_init_config_creates_file(self, tmp_path):
        """Test init_config creates config file with defaults."""
        with patch.object(Config, "get_config_dir", return_value=tmp_path):
            Config.init_config()
            config_file = tmp_path / "config.yaml"
            assert config_file.exists()

    def test_init_config_does_not_overwrite_existing(self, tmp_path):
        """Test init_config does not overwrite existing config."""
        with patch.object(Config, "get_config_dir", return_value=tmp_path):
            config_file = tmp_path / "config.yaml"
            config_file.write_text("custom: value\n")
            Config.init_config()
            assert config_file.read_text() == "custom: value\n"


class TestApiUrl:
    """Tests for get_api_url()."""

    def test_get_api_url_env_override(self):
        """Test ESDC_URL environment variable has highest priority."""
        with patch.dict(os.environ, {"ESDC_URL": "https://custom.example.com/"}):
            assert Config.get_api_url() == "https://custom.example.com/"

    def test_get_api_url_config_file(self, tmp_path):
        """Test config file is used when no env var."""
        with patch.object(Config, "get_config_dir", return_value=tmp_path):
            config_file = tmp_path / "config.yaml"
            config_file.write_text("api_url: https://config.example.com/\n")
            assert Config.get_api_url() == "https://config.example.com/"

    def test_get_api_url_default(self, tmp_path):
        """Test default API URL when no env or config."""
        with patch.object(Config, "get_config_dir", return_value=tmp_path):
            assert Config.get_api_url() == Config.BASE_API_URL_V2


class TestCredentials:
    """Tests for get_credentials()."""

    def test_get_credentials_from_env(self):
        """Test credentials from environment variables."""
        with patch.dict(os.environ, {"ESDC_USER": "testuser", "ESDC_PASS": "testpass"}):
            username, password = Config.get_credentials()
            assert username == "testuser"
            assert password == "testpass"

    def test_get_credentials_from_prompt(self, mocker):
        """Test interactive prompt when env vars not set."""
        mocker.patch.dict(os.environ, {}, clear=True)
        mocker.patch("rich.prompt.Prompt.ask", side_effect=["promptuser", "promptpass"])

        username, password = Config.get_credentials()

        assert username == "promptuser"
        assert password == "promptpass"

    def test_get_credentials_partial_env(self, mocker):
        """Test partial credentials from env, prompt for missing."""
        mocker.patch.dict(os.environ, {"ESDC_USER": "envuser"}, clear=True)
        mocker.patch("rich.prompt.Prompt.ask", return_value="promptpass")

        username, password = Config.get_credentials()

        assert username == "envuser"
        assert password == "promptpass"


class TestDbFile:
    """Tests for database file path."""

    @pytest.fixture(autouse=True)
    def isolate_database_path_environment(self, monkeypatch):
        """Path units own their environment conditions.

        ESDC_DB_FILE wins over a configured path, so an ambient value from the calling
        shell would decide the result for every unit here. Each unit that wants one of
        these conditions still sets it inside the test; this fixture only guarantees that
        what a unit asserts is what that unit supplied.
        """
        monkeypatch.delenv("ESDC_DB_FILE", raising=False)
        monkeypatch.delenv("ESDC_DB_DIR", raising=False)

    def test_get_db_file_env_override(self, tmp_path):
        """Test ESDC_DB_FILE environment variable overrides all."""
        db_path = tmp_path / "custom.db"
        with patch.dict(os.environ, {"ESDC_DB_FILE": str(db_path)}):
            assert Config.get_db_file() == db_path

    def test_get_db_file_config(self, tmp_path):
        """Test database path from config file."""
        with patch.object(Config, "get_config_dir", return_value=tmp_path):
            config_file = tmp_path / "config.yaml"
            config_file.write_text("database_path: /custom/path/db.db\n")
            assert Config.get_db_file() == Path("/custom/path/db.db")

    def test_get_db_file_copies_chat_database_path_when_missing(self, tmp_path):
        """Test database.path is copied to database_path when top-level is absent."""
        db_path = tmp_path / "chat.duckdb"
        with patch.object(Config, "get_config_dir", return_value=tmp_path):
            config_file = tmp_path / "config.yaml"
            config_file.write_text(f"database:\n  path: {db_path}\n")

            assert Config.get_db_file() == db_path
            config_text = config_file.read_text()
            assert f"database_path: {db_path}" in config_text

    def test_get_db_file_default(self, tmp_path):
        """Test default database file path."""
        with patch.object(Config, "get_config_dir", return_value=tmp_path):
            assert Config.get_db_file() == tmp_path / "esdc.duckdb"

    def test_get_db_file_migrates_legacy_default(self, tmp_path):
        """Test legacy default database is renamed to the new default."""
        legacy_db = tmp_path / "esdc.db"
        default_db = tmp_path / "esdc.duckdb"
        legacy_db.write_text("legacy")

        with patch.object(Config, "get_config_dir", return_value=tmp_path):
            assert Config.get_db_file() == default_db

        assert default_db.read_text() == "legacy"
        assert not legacy_db.exists()

    def test_get_db_file_migrates_legacy_default_config(self, tmp_path):
        """Test generated legacy config path is migrated to the new default."""
        legacy_db = tmp_path / "esdc.db"
        default_db = tmp_path / "esdc.duckdb"
        legacy_db.write_text("legacy")
        config_file = tmp_path / "config.yaml"
        config_file.write_text(f"database_path: {legacy_db}\n")

        with patch.object(Config, "get_config_dir", return_value=tmp_path):
            assert Config.get_db_file() == default_db

        assert default_db.read_text() == "legacy"
        assert not legacy_db.exists()
        assert f"database_path: {default_db}" in config_file.read_text()

    def test_get_db_file_env_override_does_not_migrate_legacy_default(self, tmp_path):
        """Test ESDC_DB_FILE prevents default-path migration."""
        legacy_db = tmp_path / "esdc.db"
        env_db = tmp_path / "custom.db"
        legacy_db.write_text("legacy")

        with (
            patch.object(Config, "get_config_dir", return_value=tmp_path),
            patch.dict(os.environ, {"ESDC_DB_FILE": str(env_db)}),
        ):
            assert Config.get_db_file() == env_db

        assert legacy_db.exists()
        assert not (tmp_path / "esdc.duckdb").exists()

    def test_get_db_file_custom_config_does_not_migrate_legacy_default(self, tmp_path):
        """Test explicit database_path prevents default-path migration."""
        legacy_db = tmp_path / "esdc.db"
        custom_db = tmp_path / "custom.db"
        legacy_db.write_text("legacy")
        config_file = tmp_path / "config.yaml"
        config_file.write_text(f"database_path: {custom_db}\n")

        with patch.object(Config, "get_config_dir", return_value=tmp_path):
            assert Config.get_db_file() == custom_db

        assert legacy_db.exists()
        assert not (tmp_path / "esdc.duckdb").exists()

    def test_get_db_dir_env_override(self, tmp_path):
        """Test ESDC_DB_DIR environment variable."""
        with patch.dict(os.environ, {"ESDC_DB_DIR": str(tmp_path)}):
            assert Config.get_db_dir() == tmp_path

    def test_get_db_path_backwards_compat(self, tmp_path):
        """Test get_db_path returns directory (backwards compatibility)."""
        with patch.object(Config, "get_config_dir", return_value=tmp_path):
            assert Config.get_db_path() == tmp_path


class TestWizardSectionCoverage:
    """MODEL_SECTIONS + SETTINGS_SECTIONS must cover every editable key once.

    The wizard redesign groups flat config keys into a domain tree. This
    invariant guards against a new key silently falling out of the tree
    (no home) or landing in two groups at once.

    `corpus.n_gpu_layers` is the one deliberate exception: a value it has no
    implementation for any more is no longer offered for editing, but the key
    stays in defaults, descriptions and the int set so a stored value is still
    parsed and preserved.
    """

    # Keys the wizard no longer offers, while config still keeps them.
    _RETIRED_KEYS = {"corpus.n_gpu_layers"}

    def _editable_flat_keys(self) -> set[str]:
        flat_defaults = Config._flatten(Config.get_defaults())
        editable = set(flat_defaults.keys())
        editable |= {f"corpus.{k}" for k in Config.CORPUS_DEFAULTS}
        editable |= {
            "phoenix.enabled",
            "phoenix.collector_endpoint",
            "phoenix.project_name",
        }
        # default_provider / provider_order are edited only via the
        # provider CRUD flows, never via the flat key sections.
        editable -= {"default_provider", "provider_order"}
        return editable

    def test_union_matches_editable_keys(self):
        grouped_keys: list[str] = []
        for keys in {**MODEL_SECTIONS, **SETTINGS_SECTIONS}.values():
            grouped_keys.extend(keys)

        assert set(grouped_keys) == self._editable_flat_keys() - self._RETIRED_KEYS

    def test_no_key_duplicated_across_groups(self):
        grouped_keys: list[str] = []
        for keys in {**MODEL_SECTIONS, **SETTINGS_SECTIONS}.values():
            grouped_keys.extend(keys)

        assert len(grouped_keys) == len(set(grouped_keys))

    def test_no_orphan_keys(self):
        grouped_keys = {
            key
            for keys in {**MODEL_SECTIONS, **SETTINGS_SECTIONS}.values()
            for key in keys
        }
        orphans = self._editable_flat_keys() - grouped_keys - self._RETIRED_KEYS
        assert orphans == set()

    def test_the_retired_key_has_no_menu_home_but_is_still_preserved(self):
        grouped_keys = {
            key
            for keys in {**MODEL_SECTIONS, **SETTINGS_SECTIONS}.values()
            for key in keys
        }
        assert self._RETIRED_KEYS.isdisjoint(grouped_keys)

        for key in self._RETIRED_KEYS:
            assert key in KEY_DESCRIPTIONS, key
            assert key in Config.INT_KEYS, key

        # A stored value still parses and is preserved; nothing here drops the key.
        assert Config._flatten(Config.get_defaults())["corpus.n_gpu_layers"] == -1
        assert parse_config_input("corpus.n_gpu_layers", "6", {}) == 6


class TestBooleanAndIntKeyCoercion:
    """Guard the coercion sets that drive the wizard and YAML/env parsing.

    BOOLEAN_KEYS/INT_KEYS membership drives both wizard widget selection
    and Config._coerce_value's YAML/env normalization. A key whose default is
    bool/int but missing from these sets is silently saved/read as a string.
    """

    def test_phoenix_enabled_in_boolean_keys(self):
        assert "phoenix.enabled" in Config.BOOLEAN_KEYS

    def test_corpus_int_keys_in_int_keys(self):
        expected = {
            "corpus.chunk_size",
            "corpus.chunk_overlap",
            "corpus.ocr_dpi",
            "corpus.num_ctx",
            "corpus.min_chars_per_page",
        }
        assert expected <= Config.INT_KEYS

    def test_coerce_phoenix_enabled_true(self):
        assert Config._coerce_value("phoenix.enabled", "true") is True

    def test_coerce_phoenix_enabled_false(self):
        assert Config._coerce_value("phoenix.enabled", "false") is False

    def test_coerce_corpus_chunk_size(self):
        assert Config._coerce_value("corpus.chunk_size", "3000") == 3000

    def test_coerce_corpus_ocr_dpi(self):
        assert Config._coerce_value("corpus.ocr_dpi", "200") == 200

    def test_coerce_malformed_int_stays_string(self):
        assert Config._coerce_value("corpus.chunk_size", "abc") == "abc"

    def test_all_bool_and_int_defaults_are_registered(self):
        """Completeness guard over every default key.

        Bool defaults must land in BOOLEAN_KEYS and int defaults must land
        in INT_KEYS, so no sibling gap can reappear (floats like
        corpus.min_image_area are excluded on purpose).
        """
        flat_defaults = Config._flatten(Config.get_defaults())
        flat_defaults.update(
            {f"corpus.{k}": v for k, v in Config.CORPUS_DEFAULTS.items()}
        )
        for key, value in flat_defaults.items():
            if isinstance(value, bool):
                assert key in Config.BOOLEAN_KEYS, f"{key} missing from BOOLEAN_KEYS"
            elif isinstance(value, int):
                assert key in Config.INT_KEYS, f"{key} missing from INT_KEYS"


class TestCorpusExtractionBounds:
    """Defaults and descriptions for bounded extraction calls."""

    def test_extract_max_tokens_default(self):
        assert Config.CORPUS_DEFAULTS["extract_max_tokens"] == 8192

    def test_extract_timeout_seconds_default(self):
        assert Config.CORPUS_DEFAULTS["extract_timeout_seconds"] == 300

    def test_extract_keys_have_descriptions(self):
        assert "corpus.extract_max_tokens" in KEY_DESCRIPTIONS
        assert "corpus.extract_timeout_seconds" in KEY_DESCRIPTIONS

    def test_extract_keys_listed_in_wizard_sections(self):
        from esdc.configs import SETTINGS_SECTIONS

        section_keys = set(SETTINGS_SECTIONS["Corpus processing"])
        assert "corpus.extract_max_tokens" in section_keys
        assert "corpus.extract_timeout_seconds" in section_keys

    def test_extract_keys_coerce_as_ints(self):
        assert Config._coerce_value("corpus.extract_max_tokens", "0") == 0
        assert Config._coerce_value("corpus.extract_timeout_seconds", "0") == 0


ROLE_REFERENCE_KEYS = (
    "embedding_provider",
    "corpus.metadata_provider",
    "corpus.cleanup_provider",
    "corpus.ocr_provider",
    "corpus.rerank_provider",
)


class TestOptionalRoleReferences:
    """The six optional reference keys: defaults, descriptions, one home."""

    def test_role_reference_defaults_are_empty_strings(self):
        flat = Config._flatten(Config.get_defaults())
        for key in ROLE_REFERENCE_KEYS:
            assert flat[key] == "", key

    def test_rerank_backend_defaults_to_provider(self):
        assert (
            Config._flatten(Config.get_defaults())["corpus.rerank_backend"] == "provider"
        )

    def test_role_reference_keys_have_descriptions(self):
        for key in (*ROLE_REFERENCE_KEYS, "corpus.rerank_backend"):
            assert key in KEY_DESCRIPTIONS, key

    def test_rerank_backend_enum_registered_with_its_default(self):
        assert ENUM_CHOICES["corpus.rerank_backend"] == ["local", "provider"]

    def test_role_reference_keys_have_exactly_one_menu_location(self):
        grouped = [
            key
            for keys in {**MODEL_SECTIONS, **SETTINGS_SECTIONS}.values()
            for key in keys
        ]
        for key in (*ROLE_REFERENCE_KEYS, "corpus.rerank_backend"):
            assert grouped.count(key) == 1, key


class TestExistingConfigWithoutRoleReferences:
    """A saved config with none of the new keys behaves exactly as before."""

    _CONFIG_TEXT = """
api_url: https://config.example.com/
logging:
  level: INFO
embedding_host: http://legacy-embed:1234/v1
embedding_model: legacy-model
embedding_api_key: legacy-key
default_provider: opencode
provider_order:
  - opencode
  - engine
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
unknown_user_key: kept
"""

    def test_existing_getters_and_saved_values_are_unchanged(self, tmp_path, monkeypatch):
        for name in (
            "ESDC_EMBEDDING_BACKEND",
            "ESDC_EMBEDDING_HOST",
            "ESDC_EMBEDDING_API_KEY",
            "ESDC_EMBEDDING_MODEL",
        ):
            monkeypatch.delenv(name, raising=False)

        with patch.object(Config, "get_config_dir", return_value=tmp_path):
            config_file = tmp_path / "config.yaml"
            original = self._CONFIG_TEXT
            config_file.write_text(original)
            Config._config_cache = None

            assert Config.get_embedding_backend() == "ollama"
            assert Config.get_embedding_host() == "http://legacy-embed:1234/v1"
            assert Config.get_embedding_model() == "legacy-model"
            assert Config.get_embedding_api_key() == "legacy-key"
            assert Config.get_default_provider() == "opencode"
            assert Config.get_provider_order() == ["opencode", "engine"]

            config = Config._load_config()
            assert config is not None
            assert config["unknown_user_key"] == "kept"
            for key in ROLE_REFERENCE_KEYS:
                assert key not in config, key

            # Reading a config must not rewrite it.
            assert config_file.read_text() == original
            assert not [p for p in tmp_path.iterdir() if p.name != "config.yaml"]


class TestAtomicConfigSave:
    """One write per save, cache updated only when the file actually changed."""

    def test_multi_value_save_is_atomic(self, tmp_path):
        """Several dotted keys are applied to one dict and written once."""
        with patch.object(Config, "get_config_dir", return_value=tmp_path):
            config_file = tmp_path / "config.yaml"
            config_file.write_text("logging:\n  level: INFO\ncache:\n  sql_ttl: 604800\n")
            Config._config_cache = None

            replaces = []
            real_replace = os.replace

            def record_replace(src, dst):
                replaces.append((src, dst))
                return real_replace(src, dst)

            with patch.object(os, "replace", side_effect=record_replace):
                Config.set_config_values(
                    {
                        "logging.level": "DEBUG",
                        "corpus.rerank_backend": "provider",
                        "corpus.rerank_provider": "engine",
                    }
                )

            assert len(replaces) == 1
            config = yaml.safe_load(config_file.read_text())
            assert config["logging"]["level"] == "DEBUG"
            assert config["corpus"]["rerank_backend"] == "provider"
            assert config["corpus"]["rerank_provider"] == "engine"
            assert config["cache"]["sql_ttl"] == 604800
            # Only its own temporary file existed, and it is gone after success.
            assert [p.name for p in tmp_path.iterdir()] == ["config.yaml"]

    def test_failed_replace_preserves_file_and_cache(self, tmp_path):
        """A failed move leaves the old file and the pre-save cache intact."""
        with patch.object(Config, "get_config_dir", return_value=tmp_path):
            config_file = tmp_path / "config.yaml"
            config_file.write_text("logging:\n  level: INFO\n")
            Config._config_cache = None
            cache_before = Config._load_config()
            assert cache_before is not None
            snapshot = copy.deepcopy(cache_before)

            with (
                patch.object(os, "replace", side_effect=OSError("replace failed")),
                pytest.raises(OSError),
            ):
                Config.set_config_values({"logging.level": "DEBUG"})

            assert Config._config_cache is cache_before
            assert Config._config_cache == snapshot
            assert snapshot["logging"]["level"] == "INFO"
            assert yaml.safe_load(config_file.read_text())["logging"]["level"] == "INFO"
            # The temporary file this call created is removed, nothing else.
            assert [p.name for p in tmp_path.iterdir()] == ["config.yaml"]

    def test_save_preserves_existing_file_mode(self, tmp_path):
        with patch.object(Config, "get_config_dir", return_value=tmp_path):
            config_file = tmp_path / "config.yaml"
            config_file.write_text("logging:\n  level: INFO\n")
            config_file.chmod(0o640)
            Config._config_cache = None

            Config.set_config_value("logging.level", "DEBUG")

            assert config_file.stat().st_mode & 0o777 == 0o640

    def test_set_config_value_is_the_single_key_wrapper(self, tmp_path, monkeypatch):
        calls = []
        monkeypatch.setattr(
            Config,
            "set_config_values",
            classmethod(lambda cls, values: calls.append(values)),
        )
        Config.set_config_value("corpus.rerank_backend", "provider")
        assert calls == [{"corpus.rerank_backend": "provider"}]

    def test_provider_fields_are_not_saved_through_dotted_keys(self, tmp_path):
        """Provider entries keep their own saves; a dotted path must not reach them."""
        with patch.object(Config, "get_config_dir", return_value=tmp_path):
            config_file = tmp_path / "config.yaml"
            config_file.write_text("providers:\n  engine:\n    model: chat-a\n")
            Config._config_cache = None

            with pytest.raises(ValueError, match="dotted keys"):
                Config.set_config_values({"providers.engine.model": "chat-b"})

            config = yaml.safe_load(config_file.read_text())
            assert config["providers"]["engine"]["model"] == "chat-a"
