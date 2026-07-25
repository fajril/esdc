# esdc/corpus/embedder.py
"""Internal embedding backend: Qwen3 GGUF via llama.cpp, in-process.

Corpus embeddings are an internal concern: vectors are schema
(corpus_meta pins model + dim), so the model is pinned in code rather
than user-configurable. Runs in-process via llama-cpp-python on the
Qwen3-Embedding-0.6B GGUF — no Ollama daemon. Same weights Ollama would
run, so build-on-GPU + query-local stay in one cosine space. First use
downloads the GGUF (~600 MB, cached under ~/.esdc/models); everything
after runs offline. GPU offload via corpus.n_gpu_layers.
"""
from __future__ import annotations

import logging
import threading
from typing import Any

logger = logging.getLogger(__name__)

# Stable identifier pinned in corpus_meta. Backend-neutral on purpose
# (no "llamacpp:" prefix): a future Ollama build provider running the
# same GGUF writes into the same space under the same pin. Tied to
# EMBED_REPO/EMBED_FILE in llama_backend; changing the GGUF means
# changing this + reembedding.
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
                    f"[Corpus] failed to load embedding model {MODEL_ID}. "
                    "First use needs internet to download the GGUF (~600 MB, "
                    f"cached under ~/.esdc/models afterwards). Error: {e}"
                ) from e
            logger.info("[Embedding] internal model loaded | model=%s", MODEL_ID)
        return _model


class InternalEmbedder:
    """llama.cpp-backed embedder, drop-in for EmbeddingManager's interface."""

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
