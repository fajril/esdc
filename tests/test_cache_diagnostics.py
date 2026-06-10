"""Tests for cache diagnostics functions."""

from pathlib import Path
from unittest.mock import patch

import diskcache

from esdc.chat.tools import (
    _get_disk_cache_stats,
    get_sql_cache_stats,
    get_tool_cache_stats,
)


class TestGetDiskCacheStats:
    """Tests for _get_disk_cache_stats helper."""

    def test_nonexistent_directory(self, tmp_path: Path):
        """Test stats for a directory that doesn't exist."""
        with patch("esdc.configs.Config.get_cache_dir", return_value=tmp_path):
            result = _get_disk_cache_stats(
                cache=None,
                cache_dir_name="nonexistent",
            )
        assert result["entries"] == 0
        assert result["size_bytes"] == 0
        assert result["hits"] == 0
        assert result["misses"] == 0
        assert result["hit_rate"] is None

    def test_empty_cache_directory(self, tmp_path: Path):
        """Test stats for an empty cache directory."""
        cache_dir = tmp_path / "test_cache"
        cache_dir.mkdir()
        c = diskcache.Cache(str(cache_dir))
        c.close()

        with patch("esdc.configs.Config.get_cache_dir", return_value=tmp_path):
            result = _get_disk_cache_stats(
                cache=None,
                cache_dir_name="test_cache",
            )
        assert result["entries"] == 0
        assert result["hits"] == 0
        assert result["misses"] == 0

    def test_cache_with_hits_and_misses(self, tmp_path: Path):
        """Test stats from a cache that has recorded hits and misses."""
        cache_dir = tmp_path / "test_cache"
        cache_dir.mkdir()
        # Write data with statistics=True so hits/misses are tracked
        c = diskcache.Cache(str(cache_dir), statistics=True)
        c.set("key1", "value1")
        # Simulate a hit
        _ = c.get("key1")
        # Simulate a miss
        _ = c.get("nonexistent")
        c.close()

        # Now open a separate handle (like esdc status does) and read stats
        with patch("esdc.configs.Config.get_cache_dir", return_value=tmp_path):
            result = _get_disk_cache_stats(
                cache=None,
                cache_dir_name="test_cache",
            )
        assert result["entries"] == 1
        assert result["hits"] >= 1
        assert result["misses"] >= 1
        assert result["hit_rate"] is not None
        assert result["size_bytes"] > 0

    def test_race_condition_missing_directory(self, tmp_path: Path):
        """Test that FileNotFoundError is handled gracefully."""
        with patch("esdc.configs.Config.get_cache_dir", return_value=tmp_path):
            result = _get_disk_cache_stats(
                cache=None,
                cache_dir_name="will_not_exist",
            )
        assert result["entries"] == 0
        assert result["hits"] == 0
        assert result["misses"] == 0


class TestGetSqlCacheStats:
    """Tests for get_sql_cache_stats."""

    def test_returns_dict_with_expected_keys(self):
        """Test that get_sql_cache_stats returns all expected keys."""
        with (
            patch("esdc.chat.tools._sql_cache", None),
            patch("esdc.configs.Config.get_cache_dir", return_value=Path("/nonexistent")),
        ):
            result = get_sql_cache_stats()

        expected_keys = {
            "directory",
            "entries",
            "size_bytes",
            "size_limit",
            "hits",
            "misses",
            "hit_rate",
        }
        assert expected_keys.issubset(result.keys())


class TestGetToolCacheStats:
    """Tests for get_tool_cache_stats."""

    def test_returns_dict_with_expected_keys(self):
        """Test that get_tool_cache_stats returns all expected keys."""
        with (
            patch("esdc.chat.tools._tool_cache", None),
            patch("esdc.configs.Config.get_cache_dir", return_value=Path("/nonexistent")),
        ):
            result = get_tool_cache_stats()

        expected_keys = {
            "directory",
            "entries",
            "size_bytes",
            "size_limit",
            "hits",
            "misses",
            "hit_rate",
        }
        assert expected_keys.issubset(result.keys())


class TestCacheInvalidationTimestamp:
    """Tests for cache invalidation timestamp recording."""

    def test_record_and_read_timestamp(self, tmp_path: Path):
        """Test that _record_cache_invalidation writes and reads timestamp."""
        from esdc.dbmanager import (
            _record_cache_invalidation,
            get_last_cache_invalidation,
        )

        with patch("esdc.dbmanager.Config.get_cache_dir", return_value=tmp_path):
            _record_cache_invalidation(tmp_path / "sql_results")
            result = get_last_cache_invalidation()
        assert result is not None
        assert "T" in result

    def test_get_last_invalidated_no_file(self, tmp_path: Path):
        """Test get_last_cache_invalidation returns None when no file exists."""
        from esdc.dbmanager import get_last_cache_invalidation

        with patch("esdc.dbmanager.Config.get_cache_dir", return_value=tmp_path):
            result = get_last_cache_invalidation()
        assert result is None

    def test_invalidate_sql_cache_records_timestamp(self, tmp_path: Path):
        """Test that invalidate_sql_cache records a timestamp."""
        from esdc.dbmanager import get_last_cache_invalidation, invalidate_sql_cache

        with patch("esdc.dbmanager.Config.get_cache_dir", return_value=tmp_path):
            invalidate_sql_cache()
            result = get_last_cache_invalidation()
        assert result is not None

    def test_invalidate_tool_cache_records_timestamp(self, tmp_path: Path):
        """Test that invalidate_tool_cache records a timestamp."""
        from esdc.chat.tools import invalidate_tool_cache

        with (
            patch("esdc.configs.Config.get_cache_dir", return_value=tmp_path),
            patch("esdc.dbmanager.Config.get_cache_dir", return_value=tmp_path),
        ):
            invalidate_tool_cache()
            from esdc.dbmanager import get_last_cache_invalidation

            result = get_last_cache_invalidation()
        assert result is not None
