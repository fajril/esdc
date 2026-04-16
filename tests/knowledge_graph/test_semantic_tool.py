"""Tests for semantic_search tool."""

import json
from unittest.mock import Mock, patch

from esdc.chat.tools import semantic_search


def test_semantic_search_tool_exists():
    """Test semantic_search tool is available."""
    assert callable(semantic_search)


def test_semantic_search_by_text():
    """Test semantic search by text."""
    with patch(
        "esdc.knowledge_graph.semantic_resolver.SemanticResolver"
    ) as MockResolver:
        mock_resolver = Mock()
        mock_resolver.search_by_text.return_value = {
            "status": "success",
            "count": 2,
            "results": [
                {"uuid": "uuid-1", "similarity": 0.95},
                {"uuid": "uuid-2", "similarity": 0.89},
            ],
        }
        MockResolver.return_value = mock_resolver

        result = semantic_search.invoke({"query": "proyek masalah teknis", "limit": 5})

        data = json.loads(result)
        assert data["status"] == "success"
        assert len(data["results"]) == 2


def test_semantic_search_not_available():
    """Test fallback when embeddings not available."""
    with patch(
        "esdc.knowledge_graph.semantic_resolver.SemanticResolver"
    ) as MockResolver:
        mock_resolver = Mock()
        mock_resolver.search_by_text.return_value = {
            "status": "not_available",
            "message": "No embeddings found",
        }
        MockResolver.return_value = mock_resolver

        result = semantic_search.invoke({"query": "test query"})

        data = json.loads(result)
        assert data["status"] == "not_available"
