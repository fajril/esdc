"""Tests for cache module."""

import pytest

from esdc.server.cache import (
    _hash_json_args,
    _hash_messages,
    clear_all_caches,
    get_cache_stats,
    get_parsed_json,
)


class TestHashFunctions:
    """Test hash generation functions."""

    def test_hash_messages_consistent(self):
        """Test that same messages produce same hash."""
        messages = [{"role": "user", "content": "Hello"}]
        hash1 = _hash_messages(messages)
        hash2 = _hash_messages(messages)
        assert hash1 == hash2
        assert len(hash1) == 16

    def test_hash_messages_different(self):
        """Test that different messages produce different hashes."""
        messages1 = [{"role": "user", "content": "Hello"}]
        messages2 = [{"role": "user", "content": "Goodbye"}]
        hash1 = _hash_messages(messages1)
        hash2 = _hash_messages(messages2)
        assert hash1 != hash2

    def test_hash_json_args_consistent(self):
        """Test that same JSON produces same hash."""
        json_str = '{"key": "value"}'
        hash1 = _hash_json_args(json_str)
        hash2 = _hash_json_args(json_str)
        assert hash1 == hash2
        assert len(hash1) == 8

    def test_hash_json_args_normalizes(self):
        """Test that normalized JSON produces same hash."""
        json1 = '{"key": "value", "num": 1}'
        json2 = '{"num": 1, "key": "value"}'  # Same content, different order
        hash1 = _hash_json_args(json1)
        hash2 = _hash_json_args(json2)
        assert hash1 == hash2


class TestGetParsedJson:
    """Test JSON parsing with cache."""

    def setup_method(self):
        """Clear cache before each test."""
        clear_all_caches()

    def test_parse_valid_json(self):
        """Test parsing valid JSON."""
        json_str = '{"key": "value"}'
        result = get_parsed_json(json_str)
        assert result == {"key": "value"}

    def test_parse_empty_string(self):
        """Test parsing empty string."""
        result = get_parsed_json("")
        assert result == {}

    def test_parse_invalid_json(self):
        """Test parsing invalid JSON returns empty dict."""
        result = get_parsed_json("not valid json")
        assert result == {}

    def test_parse_with_cache(self):
        """Test that parsing is cached."""
        json_str = '{"key": "value"}'

        # First call - parses
        result1 = get_parsed_json(json_str)

        # Second call - should use cache
        result2 = get_parsed_json(json_str)

        assert result1 == result2 == {"key": "value"}

        # Check cache stats
        stats = get_cache_stats()
        assert stats["json_cache_size"] > 0

    def test_cache_different_json(self):
        """Test that different JSON uses different cache entries."""
        json1 = '{"key1": "value1"}'
        json2 = '{"key2": "value2"}'

        result1 = get_parsed_json(json1)
        result2 = get_parsed_json(json2)

        assert result1 == {"key1": "value1"}
        assert result2 == {"key2": "value2"}

        stats = get_cache_stats()
        assert stats["json_cache_size"] == 2

    def test_cache_max_size_eviction(self):
        """Test that cache evicts old entries when max size reached."""
        from esdc.server.cache import MAX_JSON_CACHE_SIZE

        # Fill cache beyond max size
        for i in range(MAX_JSON_CACHE_SIZE + 10):
            get_parsed_json(f'{{"key{i}": "value{i}"}}')

        stats = get_cache_stats()
        # Cache should not exceed max size
        assert stats["json_cache_size"] <= MAX_JSON_CACHE_SIZE


class TestCacheStats:
    """Test cache statistics."""

    def setup_method(self):
        """Clear cache before each test."""
        clear_all_caches()

    def test_initial_stats(self):
        """Test initial cache stats are zero."""
        stats = get_cache_stats()
        assert stats["json_cache_size"] == 0
        assert "json_cache_max" in stats

    def test_stats_after_cache(self):
        """Test stats after caching."""
        get_parsed_json('{"key": "value"}')
        stats = get_cache_stats()
        assert stats["json_cache_size"] == 1

    def test_clear_all_caches(self):
        """Test that clear_all_caches empties the cache."""
        get_parsed_json('{"key": "value"}')
        assert get_cache_stats()["json_cache_size"] == 1

        clear_all_caches()
        assert get_cache_stats()["json_cache_size"] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
