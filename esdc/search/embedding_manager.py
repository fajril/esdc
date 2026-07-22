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
        batch_size: int | None = None,
    ) -> None:
        """Initialize EmbeddingManager.

        Args:
            model: Ollama model name (default from config or qwen3-embedding:0.6b)
            host: Ollama host URL (default: None, uses localhost)
            batch_size: Max texts per embed request (default from config or 100)
        """
        if model is None:
            config = Config._load_config()
            resolved_model = (
                config.get("embedding_model", self.DEFAULT_MODEL)
                if config
                else self.DEFAULT_MODEL
            )
        else:
            resolved_model = model

        if batch_size is None:
            batch_size = Config.get_embedding_batch_size()
        self.batch_size: int = max(1, int(batch_size))

        if host is None:
            host = Config.get_embedding_host()
        self.model: str = resolved_model
        self._client = ollama.Client(host=host) if host else ollama.Client()

        logger.info(
            "[Embedding] initialized | model=%s",
            self.model,
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

            return list(embedding)

        except Exception as e:
            logger.error("[Embedding] failed | error=%s", e)
            raise

    def generate_embeddings_batch(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        """Generate embeddings for batch of texts, chunked by batch_size.

        Args:
            texts: List of input texts

        Returns:
            List of embedding vectors
        """
        results: list[list[float]] = []
        try:
            for start in range(0, len(texts), self.batch_size):
                slice_ = texts[start : start + self.batch_size]
                response = self._client.embed(model=self.model, input=slice_)
                results.extend(list(e) for e in response.embeddings)

            logger.info(
                "[Embedding] batch generated | model=%s count=%d batches=%d",
                self.model,
                len(texts),
                -(-len(texts) // self.batch_size) if texts else 0,
            )
            return results
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
