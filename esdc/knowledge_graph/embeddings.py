import logging
from typing import Any

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Generates embeddings for text using LangChain embedding models."""

    EMBEDDING_DIM = 768  # nomic-embed-text dimension

    def __init__(self, llm: Any = None, batch_size: int = 10):
        self.llm = llm
        self.batch_size = batch_size

        if self.llm is None:
            try:
                from langchain_ollama import OllamaEmbeddings

                from esdc.configs import Config

                # Try to get Ollama config from providers
                ollama_config = Config.get_provider_config_by_name("ollama")
                base_url = None
                if ollama_config:
                    base_url = ollama_config.get("base_url")

                self.llm = OllamaEmbeddings(model="nomic-embed-text", base_url=base_url)
            except Exception:
                logger.warning("Ollama not available, using mock embeddings")

    def generate(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for multiple texts in batches.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        embeddings: list[list[float]] = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]

            try:
                if self.llm is None:
                    batch_embeddings = [[0.1] * self.EMBEDDING_DIM for _ in batch]
                else:
                    batch_embeddings = self.llm.embed_documents(batch)
                    if not isinstance(batch_embeddings, list):
                        batch_embeddings = [[0.1] * self.EMBEDDING_DIM for _ in batch]
            except Exception as e:
                logger.warning(f"Embedding generation failed: {e}")
                batch_embeddings = [[0.0] * self.EMBEDDING_DIM for _ in batch]

            embeddings.extend(batch_embeddings)

        return embeddings

    def generate_single(self, text: str) -> list[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        try:
            if self.llm is None:
                return [0.1] * self.EMBEDDING_DIM

            embedding = self.llm.embed_query(text)

            if not isinstance(embedding, list):
                logger.warning("Invalid embedding format, returning zero vector")
                return [0.0] * self.EMBEDDING_DIM

            return embedding
        except Exception as e:
            logger.warning(f"Embedding generation failed: {e}")
            return [0.0] * self.EMBEDDING_DIM
