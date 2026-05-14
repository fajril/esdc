"""Tests for semantic_search tool."""

import json
from unittest.mock import Mock, patch

from esdc.chat.tools import semantic_search


def test_semantic_search_tool_exists():
    """Test semantic_search tool is available."""
    assert semantic_search is not None


def test_semantic_search_by_text():
    """Test semantic search by text."""
    with patch(
        "esdc.search.semantic_resolver.SemanticResolver"
    ) as MockResolver:
        mock_resolver = Mock()
        mock_resolver.hybrid_search.return_value = {
            "status": "success",
            "count": 2,
            "results": [
                {"project_id": "P1", "similarity": 0.95},
                {"project_id": "P2", "similarity": 0.89},
            ],
        }
        mock_resolver.close = Mock()
        MockResolver.return_value = mock_resolver

        with patch("esdc.chat.tools._get_tool_cache") as mock_cache:
            cache = Mock()
            cache.__contains__ = Mock(return_value=False)
            mock_cache.return_value = cache

            result = semantic_search.invoke({"query": "proyek masalah teknis", "limit": 5})

            data = json.loads(result)
            assert data["status"] == "success"


def test_semantic_search_not_available():
    """Test fallback when embeddings not available."""
    with patch(
        "esdc.search.semantic_resolver.SemanticResolver"
    ) as MockResolver:
        mock_resolver = Mock()
        mock_resolver.hybrid_search.return_value = {
            "status": "not_available",
            "message": "No embeddings found",
            "results": [],
        }
        mock_resolver.close = Mock()
        MockResolver.return_value = mock_resolver

        with patch("esdc.chat.tools._get_tool_cache") as mock_cache:
            cache = Mock()
            cache.__contains__ = Mock(return_value=False)
            mock_cache.return_value = cache

            result = semantic_search.invoke({"query": "test query"})

            data = json.loads(result)
            assert data["status"] == "not_available"
