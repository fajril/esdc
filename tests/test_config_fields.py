"""Tests for the shared config field formatter and parser.

Synthetic values only, and no test writes the live config: the parser is a pure
function, and the one test that touches a file uses an isolated temporary
config directory.
"""

import pytest

from esdc.config_fields import format_config_value, parse_config_input
from esdc.configs import Config


class TestFormatConfigValue:
    """Display-only masking of secrets, with every other value shown as-is."""

    def test_api_key_is_masked(self):
        key = "providers.engine.api_key"
        assert format_config_value(key, "sk-1234567890abcdef") == "sk****ef"

    def test_oauth_access_token_is_masked(self):
        token = "synthetic-access-token"
        assert format_config_value("providers.engine.oauth.access_token", token) == (
            "sy****en"
        )

    def test_oauth_refresh_token_is_masked(self):
        token = "synthetic-refresh-token"
        assert format_config_value("providers.engine.oauth.refresh_token", token) == (
            "sy****en"
        )

    def test_short_secret_is_fully_masked(self):
        assert format_config_value("providers.engine.api_key", "abcd") == "****"

    def test_non_secret_values_stay_visible(self):
        assert format_config_value("api_url", "https://engine.example/v1") == (
            "https://engine.example/v1"
        )
        assert format_config_value("providers.engine.model", "chat-a") == "chat-a"
        assert format_config_value("logging.level", "INFO") == "INFO"
        assert format_config_value("corpus.chunk_size", 3000) == "3000"
        assert format_config_value("corpus.min_image_area", 0.05) == "0.05"

    def test_secret_key_is_matched_at_the_end_of_the_key(self):
        # A key that merely starts with a secret suffix is not a secret.
        assert format_config_value("api_key_file", "readable-value") == "readable-value"
        assert format_config_value("providers.engine.api_key", "readable-value") == (
            "re****ue"
        )


class TestParseConfigInput:
    """Input validation for one field, against the flat merged config."""

    def test_integer_key_coerces(self):
        assert parse_config_input("cache.sql_ttl", "300", {}) == 300

    def test_integer_error_names_the_field(self):
        with pytest.raises(ValueError, match="cache.sql_ttl"):
            parse_config_input("cache.sql_ttl", "abc", {})

    def test_integer_key_rejects_a_decimal(self):
        with pytest.raises(ValueError, match="corpus.chunk_size"):
            parse_config_input("corpus.chunk_size", "3000.5", {})

    def test_boolean_key_coerces_and_rejects_other_text(self):
        assert parse_config_input("api.verify_ssl", "true", {}) is True
        with pytest.raises(ValueError, match="api.verify_ssl"):
            parse_config_input("api.verify_ssl", "maybe", {})

    def test_decimal_field_is_stored_as_float(self):
        value = parse_config_input("corpus.min_image_area", "0.05", {})
        assert value == 0.05
        assert isinstance(value, float)

    def test_image_fraction_endpoints_are_valid(self):
        assert parse_config_input("corpus.min_image_area", "0", {}) == 0.0
        assert parse_config_input("corpus.min_image_area", "1", {}) == 1.0

    def test_image_fraction_outside_the_range_is_rejected(self):
        for raw in ("1.5", "-0.01", "nan", "inf"):
            with pytest.raises(ValueError, match="corpus.min_image_area"):
                parse_config_input("corpus.min_image_area", raw, {})

    def test_zero_limits_are_valid_sentinels(self):
        assert parse_config_input("semantic_search.embedding_batch_size", "0", {}) == 0
        values = {"corpus.chunk_size": 3000}
        assert parse_config_input("corpus.chunk_overlap", "0", values) == 0

    def test_negative_gpu_layers_is_valid(self):
        assert parse_config_input("corpus.n_gpu_layers", "-1", {}) == -1

    def test_empty_optional_host_is_valid(self):
        assert parse_config_input("embedding_host", "", {}) == ""
        assert parse_config_input("corpus.ollama_host", "", {}) == ""

    def test_chunk_size_must_be_positive(self):
        assert parse_config_input("corpus.chunk_size", "3000", {}) == 3000
        for raw in ("0", "-10"):
            with pytest.raises(ValueError, match="corpus.chunk_size"):
                parse_config_input("corpus.chunk_size", raw, {})

    def test_overlap_is_checked_against_the_draft_chunk_size(self):
        assert parse_config_input("corpus.chunk_overlap", "200", {"corpus.chunk_size": 3000}) == 200
        with pytest.raises(ValueError, match="corpus.chunk_overlap"):
            parse_config_input("corpus.chunk_overlap", "200", {"corpus.chunk_size": 100})

    def test_chunk_size_is_checked_against_the_draft_overlap(self):
        with pytest.raises(ValueError, match="corpus.chunk_size"):
            parse_config_input("corpus.chunk_size", "100", {"corpus.chunk_overlap": 100})

    def test_enum_matches_case_insensitively_and_returns_the_canonical_choice(self):
        assert parse_config_input("logging.level", "debug", {}) == "DEBUG"
        assert parse_config_input("tool_format", "Markdown", {}) == "markdown"
        assert parse_config_input("corpus.rerank_backend", "Provider", {}) == "provider"

    def test_enum_rejects_an_unknown_choice(self):
        with pytest.raises(ValueError, match="logging.level"):
            parse_config_input("logging.level", "VERBOSE", {})

    def test_plain_strings_are_returned_unchanged(self):
        assert parse_config_input("logging.file.max_size", "10MB", {}) == "10MB"
        assert parse_config_input("corpus.ocr_model", "glm-ocr", {}) == "glm-ocr"
        assert parse_config_input("providers.engine.base_url", "http://engine:4000/v1", {}) == (
            "http://engine:4000/v1"
        )
        assert parse_config_input("database_path", "/tmp/esdc.duckdb", {}) == "/tmp/esdc.duckdb"

    def test_parsing_never_saves(self, isolated_config):
        config_dir = isolated_config / ".esdc"
        config_dir.mkdir(parents=True, exist_ok=True)
        config_file = config_dir / "config.yaml"
        original = "logging:\n  level: INFO\n"
        config_file.write_text(original)
        Config._config_cache = None

        parse_config_input("logging.level", "DEBUG", {"logging.level": "INFO"})

        assert config_file.read_text() == original
        assert not [p for p in config_dir.iterdir() if p.name != "config.yaml"]
