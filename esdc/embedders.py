# esdc/embedders.py
"""Embedding backends shared by every vector space in esdc.

Similarity (query embedding) always runs on InternalEmbedder — llama.cpp
on the pinned Qwen3-Embedding-0.6B GGUF, in-process, offline. Generation
(bulk embedding) runs on whichever backend `embedding_backend` selects.

Every backend reports MODEL_ID as its `.model`, never a backend-specific
wire name: that value is the vector-space identity written into
corpus_meta/semantic_meta and into per-row embedding_model columns.
Measured cross-backend cosine is >= 0.9986, so all backends write into one
cosine space; check_or_seed_probe enforces that at runtime.
"""
from __future__ import annotations

import logging
import threading
from typing import Any

logger = logging.getLogger(__name__)

# Stable identifier pinned in corpus_meta/semantic_meta. Backend-neutral on
# purpose: an Ollama or OpenAI-compatible server running the same weights
# writes into the same space under the same pin. Tied to EMBED_REPO/
# EMBED_FILE in llama_backend; changing the GGUF means changing this +
# reembedding.
MODEL_ID = "qwen3-embedding-0.6b-q8_0"

_load_lock = threading.Lock()
# Llama contexts are not thread-safe; every embed() call serializes on
# this. The chat server can run corpus tools concurrently.
_infer_lock = threading.Lock()
_model: Any | None = None


def _get_model() -> Any:
    """Load (once) and return the llama.cpp embedding model."""
    global _model
    with _load_lock:
        if _model is None:
            # Imports inside the try: a broken llama-cpp-python install
            # (ImportError) must surface the same actionable message as a
            # failed download.
            try:
                from esdc.corpus.llama_backend import (
                    EMBED_FILE,
                    EMBED_REPO,
                    load_llama,
                    resolve_gguf,
                )

                path = resolve_gguf(EMBED_REPO, EMBED_FILE)
                _model = load_llama(path)
            except Exception as e:
                raise RuntimeError(
                    f"[Embedding] failed to load embedding model {MODEL_ID}. "
                    "First use needs internet to download the GGUF (~600 MB, "
                    f"cached under ~/.esdc/models afterwards). Error: {e}"
                ) from e
            logger.info("[Embedding] internal model loaded | model=%s", MODEL_ID)
        return _model


class InternalEmbedder:
    """llama.cpp-backed embedder. Always used for query-time similarity."""

    def __init__(self) -> None:
        self.model = MODEL_ID

    def generate_embedding(self, text: str) -> list[float]:
        """Generate an embedding for a single text."""
        model = _get_model()
        with _infer_lock:
            return [float(x) for x in model.embed(text)]

    def generate_embeddings_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a batch of texts."""
        if not texts:
            return []
        model = _get_model()
        with _infer_lock:
            return [[float(x) for x in vec] for vec in model.embed(texts)]
