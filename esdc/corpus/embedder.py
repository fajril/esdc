"""Internal embedding backend: fastembed ONNX model, in-process.

Corpus embeddings are an internal concern of the application: vectors
are schema (corpus_meta pins model + dim), so the model is pinned in
code rather than user-configurable. LLM providers (Ollama, cloud) are
untouched — they serve chat/extract/learn only. First use downloads the
model from HuggingFace (~1-2 GB, cached on disk); everything after runs
offline on CPU.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

logger = logging.getLogger(__name__)

PINNED_MODEL = "intfloat/multilingual-e5-large"

_lock = threading.Lock()
_model: Any | None = None


def _get_model() -> Any:
    """Load (once) and return the fastembed TextEmbedding instance."""
    global _model
    with _lock:
        if _model is None:
            from fastembed import TextEmbedding

            try:
                _model = TextEmbedding(model_name=PINNED_MODEL)
            except Exception as e:
                raise RuntimeError(
                    f"[Corpus] failed to load embedding model {PINNED_MODEL}. "
                    "First use needs internet to download it (~1-2 GB, "
                    f"cached afterwards). Underlying error: {e}"
                ) from e
            logger.info("[Embedding] internal model loaded | model=%s", PINNED_MODEL)
        return _model


class InternalEmbedder:
    """fastembed-backed embedder, drop-in for EmbeddingManager's interface."""

    def __init__(self) -> None:
        self.model = f"fastembed:{PINNED_MODEL}"

    def generate_embedding(self, text: str) -> list[float]:
        """Generate an embedding for a single text."""
        vec = next(iter(_get_model().embed([text])))
        return [float(x) for x in vec]

    def generate_embeddings_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a batch of texts."""
        if not texts:
            return []
        return [[float(x) for x in vec] for vec in _get_model().embed(texts)]
