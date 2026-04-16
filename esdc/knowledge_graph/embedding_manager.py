"""Embedding generation manager using Ollama API.

Provides centralized embedding generation for semantic search
using Ollama's embedding models.
"""

from __future__ import annotations

import logging

import ollama

from esdc.configs import Config

logger = logging.getLogger(__name__)


class EmbeddingManager:
    """Generate embeddings using Ollama API.

    Default: qwen3-embedding:0.6b (639MB, 32K context, 4096 dimensions)
    Supports configurable model via config.yaml embedding_model key.
    """

    DEFAULT_MODEL = "qwen3-embedding:0.6b"
    DEFAULT_BATCH_SIZE = 100

    def __init__(
        self,
        model: str | None = None,
        host: str | None = None,
    ) -> None:
        """Initialize EmbeddingManager.

        Args:
            model: Ollama model name (default from config or qwen3-embedding:0.6b)
            host: Ollama host URL (default: None, uses localhost)
        """
        # Load from config if not provided
        if model is None:
            config = Config._load_config()
            model = (
                config.get("embedding_model", self.DEFAULT_MODEL)
                if config
                else self.DEFAULT_MODEL
            )

        self.model = model
        self._client = ollama.Client(host=host) if host else ollama.Client()

        logger.info(
            "[Embedding] initialized | model=%s",
            model,
        )

    def generate_embedding(
        self,
        text: str,
    ) -> list[float]:
        """Generate embedding for single text.

        Args:
            text: Input text to embed

        Returns:
            List of floats (embedding vector)
        """
        try:
            response = self._client.embed(
                model=self.model,
                input=text,
            )
            embedding = response.embeddings[0]

            logger.debug(
                "[Embedding] generated | model=%s text_len=%d",
                self.model,
                len(text),
            )

            return embedding

        except Exception as e:
            logger.error("[Embedding] failed | error=%s", e)
            raise

    def generate_embeddings_batch(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        """Generate embeddings for batch of texts.

        Args:
            texts: List of input texts

        Returns:
            List of embedding vectors
        """
        try:
            response = self._client.embed(
                model=self.model,
                input=texts,
            )

            logger.info(
                "[Embedding] batch generated | model=%s count=%d",
                self.model,
                len(texts),
            )

            return response.embeddings

        except Exception as e:
            logger.error("[Embedding] batch failed | error=%s", e)
            raise

    def health_check(self) -> bool:
        """Check if Ollama is available and model is loaded."""
        try:
            # Try to get model info
            self._client.show(self.model)
            return True
        except Exception:
            return False
