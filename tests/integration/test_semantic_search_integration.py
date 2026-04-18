"""Integration tests for semantic search with real Ollama."""

import pytest


@pytest.mark.integration
@pytest.mark.skipif(not pytest.importorskip("ollama"), reason="Ollama not installed")
def test_end_to_end_semantic_search():
    """End-to-end test with actual Ollama and database.

    Requires:
    - Ollama running with qwen3-embedding:0.6b model
    - ESDC database with project_remarks data
    """
    import duckdb

    from esdc.configs import Config
    from esdc.knowledge_graph.embedding_manager import EmbeddingManager
    from esdc.knowledge_graph.semantic_resolver import SemanticResolver

    # Check prerequisites
    manager = EmbeddingManager(model="qwen3-embedding:0.6b")
    if not manager.health_check():
        pytest.skip("Ollama not available or model not loaded")

    # Check if database has data
    db_path = Config.get_db_file()
    if not db_path.exists():
        pytest.skip("Database not found")

    conn = duckdb.connect(str(db_path), read_only=True)
    result = conn.execute(
        "SELECT COUNT(*) FROM project_resources WHERE project_remarks IS NOT NULL"
    ).fetchone()
    if result is None or result[0] == 0:
        pytest.skip("No project_remarks data in database")
    conn.close()

    # Test 1: Generate embeddings
    resolver = SemanticResolver(model="qwen3-embedding:0.6b")
    resolver.build_embeddings_table()

    result = resolver.generate_and_store_embeddings(batch_size=50)
    assert result["status"] == "success"
    assert result["count"] > 0
    print(f"Generated {result['count']} embeddings")

    # Test 2: Semantic search
    search_result = resolver.search_by_text("proyek dengan reservoir kompleks", limit=3)
    assert search_result["status"] == "success"
    assert len(search_result["results"]) > 0

    # Check that results have expected fields
    for r in search_result["results"]:
        assert "uuid" in r
        assert "field_name" in r
        assert "similarity" in r
        assert 0 <= r["similarity"] <= 1

    resolver.close()
    print("✓ Integration test passed!")


@pytest.mark.integration
def test_semantic_search_fallback_no_ollama():
    """Test graceful fallback when Ollama is not available."""
    from esdc.knowledge_graph.embedding_manager import EmbeddingManager

    # Try to connect to non-existent Ollama
    manager = EmbeddingManager(model="nonexistent-model")

    # Health check should fail gracefully
    assert not manager.health_check()


@pytest.mark.integration
def test_semantic_search_no_embeddings():
    """Test behavior when no embeddings exist."""
    from esdc.knowledge_graph.semantic_resolver import SemanticResolver

    resolver = SemanticResolver()

    # This should return not_available, not error
    result = resolver.search_by_text("test query")

    assert result["status"] == "not_available"
    assert "message" in result
