"""Tests for semantic_search tool."""

import json
from unittest.mock import Mock, patch

from esdc.chat.tools import semantic_search


def test_semantic_search_tool_exists():
    """Test semantic_search tool is available."""
    assert semantic_search is not None


def test_semantic_search_by_text(isolated_config):
    """Test semantic search by text.

    semantic_search now fans out to the document corpus too (see
    docs/plans/2026-07-13-improve-document-search-usage.md); isolated_config
    keeps that fan-out pointed at an empty tmp DuckDB instead of the real
    ~/.esdc store, so it deterministically returns documents=not_available.
    """
    with patch("esdc.search.semantic_resolver.SemanticResolver") as MockResolver:
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

            result = semantic_search.invoke(
                {"query": "proyek masalah teknis", "limit": 5}
            )

            data = json.loads(result)
            assert data["remarks"]["status"] == "success"
            assert data["documents"]["status"] == "not_available"


def test_semantic_search_not_available(isolated_config):
    """Test fallback when embeddings not available."""
    with patch("esdc.search.semantic_resolver.SemanticResolver") as MockResolver:
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
            # FTS fallback reports "error" (not "not_available") when the
            # DB file itself doesn't exist yet -- pre-existing behavior of
            # _search_remarks_via_fts, unrelated to the corpus fan-out.
            assert data["remarks"]["status"] == "error"
            assert data["documents"]["status"] == "not_available"
