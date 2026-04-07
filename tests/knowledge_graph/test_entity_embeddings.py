"""Tests for entity embedding store."""

import pickle
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from esdc.knowledge_graph.embeddings import EmbeddingGenerator
from esdc.knowledge_graph.entity_embeddings import EntityEmbeddingStore


@pytest.fixture
def mock_embedding_generator():
    """Create mock embedding generator."""
    generator = MagicMock(spec=EmbeddingGenerator)
    generator.EMBEDDING_DIM = 768

    def mock_generate(texts):
        # Return one embedding per text
        return [[0.1] * 768 for _ in texts]

    generator.generate.side_effect = mock_generate
    generator.generate_single.return_value = [0.1] * 768
    return generator


@pytest.fixture
def sample_entities():
    """Sample entities for testing."""
    return {
        "project": [
            {"name": "Arung Nowera", "esdc_id": "P001"},
            {"name": "Abadi", "esdc_id": "P002"},
            {"name": "Corridor", "esdc_id": "P003"},
        ],
        "field": [
            {"name": "Arung Field", "esdc_id": "F001"},
        ],
    }


def test_generate_embeddings(mock_embedding_generator, sample_entities):
    """Test embedding generation for entities."""
    store = EntityEmbeddingStore(embedding_generator=mock_embedding_generator)

    result = store.load_or_create(sample_entities, force_recreate=True)

    assert "project" in result
    assert "field" in result
    # Check that embeddings were generated for each entity type
    assert len(result["project"]) == 3  # 3 projects
    assert len(result["field"]) == 1  # 1 field
    # Check that each embedding has correct dimension
    assert all(len(emb) == 768 for emb in result["project"])
    assert all(len(emb) == 768 for emb in result["field"])


def test_get_embedding(mock_embedding_generator, sample_entities):
    """Test retrieving specific embedding."""
    store = EntityEmbeddingStore(embedding_generator=mock_embedding_generator)
    store.load_or_create(sample_entities, force_recreate=True)

    embedding = store.get_embedding("project", "Arung Nowera")

    assert embedding is not None
    assert len(embedding) == 768


def test_cache_staleness_detection(mock_embedding_generator, sample_entities, tmp_path):
    """Test that cache is detected as stale when entities change."""
    store = EntityEmbeddingStore(embedding_generator=mock_embedding_generator)
    store.CACHE_DIR = tmp_path

    # Generate and save cache
    store.load_or_create(sample_entities, force_recreate=True)

    # Modify entities
    modified_entities = {
        "project": [{"name": "Arung Nowera", "esdc_id": "P001"}],
    }

    # Should detect staleness
    assert store._is_cache_stale(modified_entities) is True


def test_save_and_load_cache(mock_embedding_generator, sample_entities, tmp_path):
    """Test cache persistence."""
    store = EntityEmbeddingStore(embedding_generator=mock_embedding_generator)
    store.CACHE_DIR = tmp_path

    # Generate and save
    store.load_or_create(sample_entities, force_recreate=True)
    assert len(store.embeddings) > 0

    # Create new store instance
    store2 = EntityEmbeddingStore(embedding_generator=mock_embedding_generator)
    store2.CACHE_DIR = tmp_path

    # Load from cache
    assert store2._load_cache() is True
    assert len(store2.embeddings) > 0


def test_cache_file_creation(mock_embedding_generator, sample_entities, tmp_path):
    """Test that cache file is created."""
    store = EntityEmbeddingStore(embedding_generator=mock_embedding_generator)
    store.CACHE_DIR = tmp_path

    store.load_or_create(sample_entities, force_recreate=True)

    cache_file = tmp_path / "entity_embeddings.pkl"
    assert cache_file.exists()


def test_clear_cache(mock_embedding_generator, sample_entities, tmp_path):
    """Test cache clearing."""
    store = EntityEmbeddingStore(embedding_generator=mock_embedding_generator)
    store.CACHE_DIR = tmp_path

    store.load_or_create(sample_entities, force_recreate=True)
    assert len(store.embeddings) > 0

    store.clear_cache()
    assert len(store.embeddings) == 0


def test_fallback_on_generation_failure(sample_entities):
    """Test fallback to zero vectors on generation failure."""
    failing_generator = MagicMock(spec=EmbeddingGenerator)
    failing_generator.EMBEDDING_DIM = 768
    failing_generator.generate.side_effect = Exception("Embedding failed")

    store = EntityEmbeddingStore(embedding_generator=failing_generator)
    result = store.load_or_create(sample_entities, force_recreate=True)

    assert "project" in result
    assert all(len(emb) == 768 for emb in result["project"])


def test_compute_entity_hash(mock_embedding_generator, sample_entities):
    """Test entity hash computation."""
    store = EntityEmbeddingStore(embedding_generator=mock_embedding_generator)

    hash1 = store._compute_entity_hash(sample_entities)
    hash2 = store._compute_entity_hash(sample_entities)

    assert hash1 == hash2
    assert len(hash1) == 32  # MD5 hex digest length


def test_empty_entities(mock_embedding_generator):
    """Test handling of empty entity list."""
    store = EntityEmbeddingStore(embedding_generator=mock_embedding_generator)

    result = store.load_or_create({"project": []}, force_recreate=True)

    assert "project" in result
    assert len(result["project"]) == 0
