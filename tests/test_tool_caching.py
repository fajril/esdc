"""Tests for tool result caching."""

import json
from unittest.mock import MagicMock, patch


class TestToolCacheKey:
    """Test cache key generation for tools."""

    def test_tool_cache_key_deterministic(self):
        from esdc.chat.tools import _tool_cache_key

        key1 = _tool_cache_key("resolve_spatial", query_type="proximity", target="Duri")
        key2 = _tool_cache_key("resolve_spatial", query_type="proximity", target="Duri")
        assert key1 == key2

    def test_tool_cache_key_differs_for_different_args(self):
        from esdc.chat.tools import _tool_cache_key

        key1 = _tool_cache_key("resolve_spatial", query_type="proximity", target="Duri")
        key2 = _tool_cache_key(
            "resolve_spatial", query_type="proximity", target="Rokan"
        )
        assert key1 != key2

    def test_tool_cache_key_differs_for_different_tools(self):
        from esdc.chat.tools import _tool_cache_key

        key1 = _tool_cache_key("resolve_spatial", target="Duri")
        key2 = _tool_cache_key("get_schema", table_name="Duri")
        assert key1 != key2

    def test_tool_cache_key_none_handling(self):
        from esdc.chat.tools import _tool_cache_key

        key1 = _tool_cache_key(
            "resolve_spatial",
            query_type="proximity",
            target="Duri",
            wk_name=None,
        )
        key2 = _tool_cache_key(
            "resolve_spatial",
            query_type="proximity",
            target="Duri",
            wk_name="Mahakam",
        )
        assert key1 != key2


class TestResolveSpatialCaching:
    """Test that resolve_spatial uses caching correctly."""

    def test_resolve_spatial_caches_successful_result(self):
        from esdc.chat.tools import resolve_spatial

        with (
            patch("esdc.chat.tools._get_tool_cache") as mock_cache_fn,
            patch(
                "esdc.knowledge_graph.spatial_resolver.SpatialResolver"
            ) as MockResolver,
        ):
            mock_cache = MagicMock()
            mock_cache_fn.return_value = mock_cache
            mock_cache.__contains__ = MagicMock(return_value=False)
            mock_cache.set = MagicMock()

            mock_resolver = MagicMock()
            MockResolver.return_value = mock_resolver
            mock_resolver.find_fields_near_field.return_value = {
                "status": "success",
                "nearby_fields": [
                    {"field_name": "Duri", "wk_name": "Rokan", "distance_km": 10.5}
                ],
                "count": 1,
            }
            mock_resolver.close = MagicMock()

            result = resolve_spatial.invoke(
                {
                    "query_type": "proximity",
                    "target": "Duri",
                    "radius_km": 20.0,
                    "limit": 10,
                }
            )

            parsed = json.loads(result)
            assert parsed["status"] == "success"
            mock_cache.set.assert_called_once()

    def test_resolve_spatial_returns_cached_result(self):
        from esdc.chat.tools import resolve_spatial

        cached_result = json.dumps(
            {
                "status": "success",
                "nearby_fields": [
                    {
                        "field_name": "CachedResult",
                        "wk_name": "TestWK",
                        "distance_km": 5.0,
                    }
                ],
                "count": 1,
            }
        )

        with patch("esdc.chat.tools._get_tool_cache") as mock_cache_fn:
            mock_cache = MagicMock()
            mock_cache_fn.return_value = mock_cache
            mock_cache.__contains__ = MagicMock(return_value=True)
            mock_cache.__getitem__ = MagicMock(return_value=cached_result)

            result = resolve_spatial.invoke(
                {
                    "query_type": "proximity",
                    "target": "Duri",
                    "radius_km": 20.0,
                    "limit": 10,
                }
            )

            parsed = json.loads(result)
            assert parsed["status"] == "success"
            assert parsed["nearby_fields"][0]["field_name"] == "CachedResult"


class TestInvalidateToolCache:
    """Test that tool cache invalidation works."""

    def test_invalidate_tool_cache_callable(self):
        """invalidate_tool_cache should be importable and callable."""
        from esdc.chat.tools import invalidate_tool_cache

        assert callable(invalidate_tool_cache)

    def test_invalidate_tool_cache_clears_cache(self):
        """invalidate_tool_cache should clear the module-level cache."""
        from esdc.chat.tools import _tool_cache

        with patch("esdc.chat.tools._tool_cache") as mock_cache:
            from esdc.chat.tools import invalidate_tool_cache

            invalidate_tool_cache()
            mock_cache.clear.assert_called_once()
