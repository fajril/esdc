"""Tests for semantic entity matcher."""

import math
from unittest.mock import MagicMock

import pytest

from esdc.knowledge_graph.embeddings import EmbeddingGenerator
from esdc.knowledge_graph.semantic_matcher import SemanticEntityMatcher


@pytest.fixture
def mock_embedding_generator():
    """Create mock embedding generator."""
    generator = MagicMock(spec=EmbeddingGenerator)
    generator.EMBEDDING_DIM = 768

    def generate_side_effect(texts):
        # Return different embeddings based on text
        return [[0.1 + i * 0.01] * 768 for i in range(len(texts))]

    def generate_single_side_effect(text):
        # Document embedding
        return [0.1] * 768

    generator.generate.side_effect = generate_side_effect
    generator.generate_single.side_effect = generate_single_side_effect

    return generator


@pytest.fixture
def sample_master_entities():
    """Sample master entities."""
    return {
        "project": [
            {"name": "Arung Nowera", "esdc_id": "P001"},
            {"name": "Abadi", "esdc_id": "P002"},
        ],
        "field": [
            {"name": "Arung Field", "esdc_id": "F001"},
        ],
        "operator": [
            {"name": "Pertamina", "esdc_id": "OP001"},
        ],
    }


@pytest.fixture
def sample_master_embeddings():
    """Sample master embeddings (match sample_master_entities)."""
    return {
        "project": [
            [0.1] * 768,  # Arung Nowera
            [0.2] * 768,  # Abadi
        ],
        "field": [
            [0.1] * 768,  # Arung Field (similar to Arung Nowera)
        ],
        "operator": [
            [0.5] * 768,  # Pertamina (different)
        ],
    }


def test_find_matches_above_threshold(
    mock_embedding_generator, sample_master_entities, sample_master_embeddings
):
    """Test finding matches above similarity threshold."""
    matcher = SemanticEntityMatcher(
        embedding_generator=mock_embedding_generator, threshold=0.7
    )

    # Document embedding [0.1] * 768
    # Will match entities with similar embeddings
    matches = matcher.find_matches(
        document_text="Project Arung Nowera discussion",
        master_entities=sample_master_entities,
        master_embeddings=sample_master_embeddings,
    )

    assert len(matches) > 0
    assert all(m["similarity"] >= 0.7 for m in matches)
    assert all(m["source"] in ["esdc_db", "external"] for m in matches)


def test_find_matches_sorted_by_similarity(
    mock_embedding_generator, sample_master_entities, sample_master_embeddings
):
    """Test that matches are sorted by similarity descending."""
    matcher = SemanticEntityMatcher(
        embedding_generator=mock_embedding_generator, threshold=0.0
    )

    matches = matcher.find_matches(
        document_text="Project discussion",
        master_entities=sample_master_entities,
        master_embeddings=sample_master_embeddings,
    )

    # Check sorting
    for i in range(len(matches) - 1):
        assert matches[i]["similarity"] >= matches[i + 1]["similarity"]


def test_find_top_k(
    mock_embedding_generator, sample_master_entities, sample_master_embeddings
):
    """Test finding top-k candidates."""
    matcher = SemanticEntityMatcher(embedding_generator=mock_embedding_generator)

    candidates = matcher.find_top_k(
        document_text="Project discussion",
        master_entities=sample_master_entities,
        master_embeddings=sample_master_embeddings,
        k=2,
    )

    assert len(candidates) <= 2


def test_external_entities_included(
    mock_embedding_generator, sample_master_entities, sample_master_embeddings
):
    """Test that external entities are included in matching."""
    matcher = SemanticEntityMatcher(
        embedding_generator=mock_embedding_generator, threshold=0.0
    )

    matches = matcher.find_matches(
        document_text="Meeting with Kementerian ESDM",
        master_entities=sample_master_entities,
        master_embeddings=sample_master_embeddings,
    )

    # External entities should be in results
    external_matches = [m for m in matches if m["source"] == "external"]
    assert len(external_matches) > 0


def test_cosine_similarity_identical_vectors():
    """Test cosine similarity for identical vectors."""
    vec1 = [1.0, 0.0, 0.0]
    vec2 = [1.0, 0.0, 0.0]

    similarity = SemanticEntityMatcher.cosine_similarity(vec1, vec2)

    assert math.isclose(similarity, 1.0, rel_tol=1e-5)


def test_cosine_similarity_orthogonal_vectors():
    """Test cosine similarity for orthogonal vectors."""
    vec1 = [1.0, 0.0, 0.0]
    vec2 = [0.0, 1.0, 0.0]

    similarity = SemanticEntityMatcher.cosine_similarity(vec1, vec2)

    assert math.isclose(similarity, 0.0, rel_tol=1e-5)


def test_cosine_similarity_opposite_vectors():
    """Test cosine similarity for opposite vectors."""
    vec1 = [1.0, 0.0, 0.0]
    vec2 = [-1.0, 0.0, 0.0]

    similarity = SemanticEntityMatcher.cosine_similarity(vec1, vec2)

    assert math.isclose(similarity, -1.0, rel_tol=1e-5)


def test_cosine_similarity_zero_vectors():
    """Test cosine similarity for zero vectors."""
    vec1 = [0.0, 0.0, 0.0]
    vec2 = [1.0, 0.0, 0.0]

    similarity = SemanticEntityMatcher.cosine_similarity(vec1, vec2)

    assert similarity == 0.0


def test_cosine_similarity_different_dimensions():
    """Test cosine similarity for vectors with different dimensions."""
    vec1 = [1.0, 0.0]
    vec2 = [1.0, 0.0, 0.0]

    similarity = SemanticEntityMatcher.cosine_similarity(vec1, vec2)

    assert similarity == 0.0


def test_empty_entities(mock_embedding_generator, sample_master_embeddings):
    """Test handling of empty entity list."""
    matcher = SemanticEntityMatcher(embedding_generator=mock_embedding_generator)

    matches = matcher.find_matches(
        document_text="Document",
        master_entities={},
        master_embeddings=sample_master_embeddings,
    )

    # Should still include external entities
    external_matches = [m for m in matches if m["source"] == "external"]
    assert len(external_matches) > 0


def test_threshold_filtering(
    mock_embedding_generator, sample_master_entities, sample_master_embeddings
):
    """Test that threshold correctly filters matches."""
    high_threshold_matcher = SemanticEntityMatcher(
        embedding_generator=mock_embedding_generator, threshold=0.99
    )

    matches = high_threshold_matcher.find_matches(
        document_text="Project",
        master_entities=sample_master_entities,
        master_embeddings=sample_master_embeddings,
    )

    # With very high threshold, should have few or no matches
    assert len(matches) == 0 or all(m["similarity"] >= 0.99 for m in matches)


def test_default_threshold():
    """Test that default threshold is 0.7."""
    matcher = SemanticEntityMatcher()

    assert matcher.threshold == 0.7


def test_custom_threshold():
    """Test custom threshold setting."""
    matcher = SemanticEntityMatcher(threshold=0.85)

    assert matcher.threshold == 0.85
