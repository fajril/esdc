"""Tests for SemanticResolver."""

import pytest
from unittest.mock import Mock, patch, MagicMock

from esdc.knowledge_graph.semantic_resolver import SemanticResolver


def test_semantic_resolver_initialization():
    """Test SemanticResolver can be initialized."""
    with patch("esdc.knowledge_graph.semantic_resolver.EmbeddingManager"):
        resolver = SemanticResolver()
        assert resolver is not None


def test_search_by_text():
    """Test semantic search by text query."""
    with patch("esdc.knowledge_graph.semantic_resolver.EmbeddingManager") as MockEmb:
        # Mock embedding manager
        mock_emb = Mock()
        mock_emb.generate_embedding.return_value = [0.1] * 1024
        MockEmb.return_value = mock_emb

        resolver = SemanticResolver()

        # Mock database connection - need to handle multiple execute calls
        resolver._get_connection = Mock()
        mock_conn = Mock()

        # First call: COUNT check, second call: actual search
        mock_cursor1 = Mock()
        mock_cursor1.fetchone.return_value = [1]  # Count > 0

        mock_cursor2 = Mock()
        # Result needs 5 columns: uuid, field_name, project_name, source_text, similarity
        mock_cursor2.fetchall.return_value = [
            (
                "uuid-1",
                "Field name",
                "Project name",
                "This is project remarks text",
                0.95,
            ),
        ]

        mock_conn.execute.side_effect = [mock_cursor1, mock_cursor2]
        resolver._get_connection.return_value = mock_conn

        result = resolver.search_by_text("proyek masalah teknis", limit=5)

        assert result["status"] == "success"
        assert len(result["results"]) == 1
        assert result["results"][0]["uuid"] == "uuid-1"


def test_search_by_embedding():
    """Test semantic search by pre-computed embedding."""
    with patch("esdc.knowledge_graph.semantic_resolver.EmbeddingManager"):
        resolver = SemanticResolver()
        resolver._get_connection = Mock()
        mock_conn = Mock()

        # Mock count check first (returns 1 document available)
        mock_cursor1 = Mock()
        mock_cursor1.fetchone.return_value = [1]

        # Mock search results (empty)
        mock_cursor2 = Mock()
        mock_cursor2.fetchall.return_value = []

        mock_conn.execute.side_effect = [mock_cursor1, mock_cursor2]
        resolver._get_connection.return_value = mock_conn

        query_embedding = [0.1] * 4096
        result = resolver.search_by_embedding(query_embedding, limit=10)

        assert result["status"] == "no_results"


def test_build_embeddings_table():
    """Test building embeddings table."""
    with patch("esdc.knowledge_graph.semantic_resolver.EmbeddingManager"):
        resolver = SemanticResolver()
        resolver._get_connection = Mock()
        mock_conn = Mock()
        resolver._get_connection.return_value = mock_conn

        result = resolver.build_embeddings_table()

        assert result is True


def test_search_not_available():
    """Test search when embeddings not available."""
    with patch("esdc.knowledge_graph.semantic_resolver.EmbeddingManager") as MockEmb:
        mock_emb = Mock()
        mock_emb.generate_embedding.return_value = [0.1] * 4096
        MockEmb.return_value = mock_emb

        resolver = SemanticResolver()
        resolver._get_connection = Mock()
        mock_conn = Mock()
        # Return 0 for count check
        mock_conn.execute.return_value.fetchall.return_value = []
        mock_conn.execute.return_value.fetchone.return_value = [0]
        resolver._get_connection.return_value = mock_conn

        result = resolver.search_by_text("test query")

        assert result["status"] == "not_available"
