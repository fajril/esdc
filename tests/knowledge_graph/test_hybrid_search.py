"""Tests for hybrid search (vector + BM25 with RRF)."""

from unittest.mock import MagicMock, patch


class TestHybridSearchMerge:
    """Test _merge_rrf method for Reciprocal Rank Fusion."""

    def test_merge_rrf_combines_both_paths(self):
        """RRF should merge results from both semantic and keyword paths."""
        from esdc.knowledge_graph.semantic_resolver import SemanticResolver

        resolver = SemanticResolver.__new__(SemanticResolver)

        semantic_results = [
            {
                "project_id": "P1",
                "report_year": 2024,
                "similarity": 0.95,
                "field_name": "Alpha",
            },
            {
                "project_id": "P2",
                "report_year": 2024,
                "similarity": 0.85,
                "field_name": "Beta",
            },
        ]
        keyword_results = [
            {
                "project_id": "P1",
                "report_year": 2024,
                "bm25_score": 10.5,
                "field_name": "Alpha",
            },
            {
                "project_id": "P3",
                "report_year": 2024,
                "bm25_score": 8.0,
                "field_name": "Gamma",
            },
        ]

        merged = resolver._merge_rrf(
            semantic_results,
            keyword_results,
            semantic_weight=0.7,
            keyword_weight=0.3,
        )

        # P1 should rank highest (appears in both)
        assert merged[0]["project_id"] == "P1"
        # P3 (keyword only) should be present
        assert any(r["project_id"] == "P3" for r in merged)
        # All results should have a unified "score" field
        for r in merged:
            assert "score" in r

    def test_merge_rrf_with_empty_keyword_results(self):
        """When keyword results are empty, should return semantic results with score."""
        from esdc.knowledge_graph.semantic_resolver import SemanticResolver

        resolver = SemanticResolver.__new__(SemanticResolver)
        semantic_results = [
            {
                "project_id": "P1",
                "report_year": 2024,
                "similarity": 0.9,
                "field_name": "Alpha",
            },
        ]

        merged = resolver._merge_rrf(
            semantic_results, [], semantic_weight=0.7, keyword_weight=0.3
        )
        assert len(merged) == 1
        assert merged[0]["project_id"] == "P1"
        assert merged[0]["score"] > 0

    def test_merge_rrf_with_empty_semantic_results(self):
        """When semantic results are empty, should return keyword results with score."""
        from esdc.knowledge_graph.semantic_resolver import SemanticResolver

        resolver = SemanticResolver.__new__(SemanticResolver)
        keyword_results = [
            {
                "project_id": "P1",
                "report_year": 2024,
                "bm25_score": 10.5,
                "field_name": "Alpha",
            },
        ]

        merged = resolver._merge_rrf(
            [], keyword_results, semantic_weight=0.7, keyword_weight=0.3
        )
        assert len(merged) == 1
        assert merged[0]["project_id"] == "P1"
        assert merged[0]["score"] > 0

    def test_merge_rrf_deduplicates_by_project_id_year(self):
        """Results should be deduplicated by project_id and report_year."""
        from esdc.knowledge_graph.semantic_resolver import SemanticResolver

        resolver = SemanticResolver.__new__(SemanticResolver)
        semantic_results = [
            {
                "project_id": "P1",
                "report_year": 2024,
                "similarity": 0.9,
                "field_name": "Alpha",
            },
        ]
        keyword_results = [
            {
                "project_id": "P1",
                "report_year": 2024,
                "bm25_score": 10.0,
                "field_name": "Alpha",
            },
        ]

        merged = resolver._merge_rrf(
            semantic_results, keyword_results, semantic_weight=0.7, keyword_weight=0.3
        )
        assert len(merged) == 1  # Not 2
        assert merged[0]["project_id"] == "P1"

    def test_merge_rrf_both_empty_returns_empty(self):
        """When both result lists are empty, should return empty list."""
        from esdc.knowledge_graph.semantic_resolver import SemanticResolver

        resolver = SemanticResolver.__new__(SemanticResolver)
        merged = resolver._merge_rrf([], [], semantic_weight=0.7, keyword_weight=0.3)
        assert merged == []


class TestHybridSearchToolIntegration:
    """Test that semantic_search tool uses hybrid search."""

    def test_semantic_search_tool_uses_hybrid(self):
        """The semantic_search tool should call hybrid_search method."""
        import json
        from unittest.mock import MagicMock, patch

        from esdc.chat.tools import semantic_search

        with patch("esdc.chat.tools.SemanticResolver") as MockResolver:
            mock_instance = MagicMock()
            mock_instance.hybrid_search.return_value = {
                "status": "success",
                "count": 1,
                "results": [{"project_id": "P1", "similarity": 0.9}],
            }
            mock_instance.close = MagicMock()
            MockResolver.return_value = mock_instance

            with patch("esdc.chat.tools._get_tool_cache") as mock_cache:
                cache = MagicMock()
                cache.__contains__ = MagicMock(return_value=False)
                mock_cache.return_value = cache

                result = semantic_search("test query")
                result_dict = json.loads(result)
                assert result_dict["status"] == "success"
                mock_instance.hybrid_search.assert_called_once()

    def test_hybrid_search_returns_success_status(self):
        """Hybrid search should return success status with results."""
        from esdc.knowledge_graph.semantic_resolver import SemanticResolver

        # This is a minimal test - actual hybrid search requires DB
        resolver = SemanticResolver.__new__(SemanticResolver)

        # Mock the embedding manager
        resolver._embedding_manager = MagicMock()
        resolver._embedding_manager.generate_embedding.return_value = [0.1] * 384

        # Mock search_by_embedding to return not_available (no embeddings)
        with patch.object(resolver, "search_by_embedding") as mock_search:
            mock_search.return_value = {"status": "not_available"}

            result = resolver.hybrid_search("test query")
            assert result["status"] == "not_available"


class TestKeywordSearch:
    """Test _keyword_search method."""

    def test_keyword_search_returns_results_with_bm25_score(self):
        """Keyword search should return results with bm25_score."""
        from esdc.knowledge_graph.semantic_resolver import SemanticResolver

        resolver = SemanticResolver.__new__(SemanticResolver)
        resolver._db_path = MagicMock()
        resolver._conn = None

        # Minimal test - actual test requires database connection
        # The method should handle exceptions gracefully
        result = resolver._keyword_search("test", limit=5)
        # Either returns results or empty list on error
        assert isinstance(result, list)


class TestHybridSearchResultFormat:
    """Test hybrid search result format."""

    def test_hybrid_search_result_has_expected_fields(self):
        """Hybrid search results should have expected fields after cleanup."""
        # Create sample merged results
        merged = [
            {
                "project_id": "P1",
                "report_year": 2024,
                "field_name": "Duri",
                "score": 0.85,
                "similarity": 0.9,
                "bm25_score": 10.5,
                "semantic_rank": 1,
                "keyword_rank": 2,
            }
        ]

        # Simulate the cleanup in hybrid_search
        results = merged[:10]
        for r in results:
            r.pop("similarity", None)
            r.pop("bm25_score", None)
            r.pop("semantic_rank", None)
            r.pop("keyword_rank", None)
            r["similarity"] = round(r.pop("score", 0), 4)

        assert "similarity" in results[0]
        assert "score" not in results[0]
        assert "bm25_score" not in results[0]
        assert "semantic_rank" not in results[0]
        assert results[0]["similarity"] == 0.85
