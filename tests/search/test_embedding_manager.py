"""Tests for EmbeddingManager."""

from unittest.mock import Mock, patch

from esdc.search.embedding_manager import EmbeddingManager


def test_embedding_manager_initialization():
    """Test EmbeddingManager can be initialized."""
    with patch("esdc.search.embedding_manager.ollama.Client"):
        manager = EmbeddingManager(model="qwen3-embedding:0.6b")
        assert manager.model == "qwen3-embedding:0.6b"


def test_embedding_manager_default_model():
    """Test EmbeddingManager uses default model."""
    with (
        patch("esdc.search.embedding_manager.ollama.Client"),
        patch("esdc.search.embedding_manager.Config._load_config") as mock_config,
    ):
        mock_config.return_value = None
        manager = EmbeddingManager()
        assert manager.model == EmbeddingManager.DEFAULT_MODEL


def test_generate_embedding_single():
    """Test generating embedding for single text."""
    with patch("esdc.search.embedding_manager.ollama.Client") as MockClient:
        mock_client = Mock()
        mock_response = Mock()
        mock_response.embeddings = [[0.1, 0.2, 0.3]]
        mock_client.embed.return_value = mock_response
        MockClient.return_value = mock_client

        manager = EmbeddingManager(model="qwen3-embedding:0.6b")
        embedding = manager.generate_embedding("This is a test")

        assert isinstance(embedding, list)
        assert all(isinstance(x, float) for x in embedding)


def test_generate_embeddings_batch():
    """Test generating embeddings for batch of texts."""
    with patch("esdc.search.embedding_manager.ollama.Client") as MockClient:
        mock_client = Mock()
        mock_response = Mock()
        mock_response.embeddings = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
        mock_client.embed.return_value = mock_response
        MockClient.return_value = mock_client

        manager = EmbeddingManager(model="qwen3-embedding:0.6b")
        texts = ["Text one", "Text two", "Text three"]
        embeddings = manager.generate_embeddings_batch(texts)

        assert len(embeddings) == 3


def test_health_check_success():
    """Test health check when Ollama is available."""
    with patch("esdc.search.embedding_manager.ollama.Client") as MockClient:
        mock_client = Mock()
        mock_client.show.return_value = {"name": "qwen3-embedding:0.6b"}
        MockClient.return_value = mock_client

        manager = EmbeddingManager(model="qwen3-embedding:0.6b")
        assert manager.health_check() is True


def test_health_check_failure():
    """Test health check when Ollama is not available."""
    with patch("esdc.search.embedding_manager.ollama.Client") as MockClient:
        mock_client = Mock()
        mock_client.show.side_effect = Exception("Ollama not running")
        MockClient.return_value = mock_client

        manager = EmbeddingManager(model="qwen3-embedding:0.6b")
        assert manager.health_check() is False


def test_generate_embeddings_batch_respects_batch_size():
    """Test that generate_embeddings_batch chunks by batch_size."""
    with patch("esdc.search.embedding_manager.ollama.Client") as MockClient:
        calls: list[int] = []

        class _FakeResponse:
            def __init__(self, n):
                self.embeddings = [[0.0] * 4 for _ in range(n)]

        mock_client = Mock()

        def fake_embed(model, input):
            calls.append(len(input))
            return _FakeResponse(len(input))

        mock_client.embed.side_effect = fake_embed
        MockClient.return_value = mock_client

        manager = EmbeddingManager(model="test", batch_size=2)
        result = manager.generate_embeddings_batch(["a", "b", "c", "d", "e"])

        assert len(result) == 5
        assert calls == [2, 2, 1]
