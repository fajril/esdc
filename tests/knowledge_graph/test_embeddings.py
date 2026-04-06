from unittest.mock import MagicMock

from esdc.knowledge_graph.embeddings import EmbeddingGenerator


def test_generate_embeddings():
    """Test embedding generation for multiple texts."""
    mock_llm = MagicMock()
    mock_llm.embed_documents.return_value = [[0.1] * 768, [0.2] * 768]

    generator = EmbeddingGenerator(llm=mock_llm)

    texts = ["Test text one", "Test text two"]
    embeddings = generator.generate(texts)

    assert len(embeddings) == 2
    assert len(embeddings[0]) == 768  # nomic-embed-text dimension
    assert len(embeddings[1]) == 768


def test_generate_single():
    """Test single text embedding."""
    mock_llm = MagicMock()
    mock_llm.embed_query.return_value = [0.1] * 768

    generator = EmbeddingGenerator(llm=mock_llm)

    embedding = generator.generate_single("Test")

    assert len(embedding) == 768
    assert isinstance(embedding, list)


def test_batch_size():
    """Test that batch processing works correctly."""
    mock_llm = MagicMock()
    # First call returns 10 embeddings, second call returns 5 embeddings
    mock_llm.embed_documents.side_effect = [
        [[0.1] * 768] * 10,  # First batch
        [[0.2] * 768] * 5,  # Second batch
    ]

    generator = EmbeddingGenerator(llm=mock_llm, batch_size=10)

    texts = [f"Test {i}" for i in range(15)]
    embeddings = generator.generate(texts)

    assert len(embeddings) == 15
    # Verify embed_documents was called twice (batch 1: 10 items, batch 2: 5 items)
    assert mock_llm.embed_documents.call_count == 2
    # Verify embeddings are correct
    assert all(len(e) == 768 for e in embeddings)
