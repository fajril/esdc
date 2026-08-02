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

import requests

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


def _unreachable(backend: str, target: str, err: Exception) -> str:
    """Actionable message for a generation backend that could not be reached."""
    return (
        f"[Embedding] backend '{backend}' could not reach {target}: {err}. "
        "Check the server is running and reachable, or set "
        "`embedding_backend: local` in ~/.esdc/config.yaml to embed "
        "in-process with no daemon."
    )


class OllamaEmbedder:
    """Generation backend over an Ollama daemon.

    Wraps EmbeddingManager rather than being handed to callers directly:
    EmbeddingManager.model doubles as the Ollama wire tag AND, under the
    CorpusStore embedder contract, the vector-space identity. Reporting the
    wire tag would repin corpus_meta on every backend switch and force a
    reembed for vectors that are 0.9995 identical.
    """

    OLLAMA_TAG = "qwen3-embedding:0.6b"

    def __init__(self, host: str | None = None) -> None:
        # Lazy import: esdc/search/__init__.py imports semantic_resolver,
        # which imports this module. A module-level import here would be a
        # circular import.
        from esdc.search.embedding_manager import EmbeddingManager

        self.host = host or "local ollama daemon"
        self.model = MODEL_ID
        self._mgr = EmbeddingManager(model=self.OLLAMA_TAG, host=host)

    def generate_embedding(self, text: str) -> list[float]:
        """Generate an embedding for a single text."""
        try:
            return [float(x) for x in self._mgr.generate_embedding(text)]
        except Exception as e:
            raise RuntimeError(_unreachable("ollama", self.host, e)) from e

    def generate_embeddings_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a batch of texts."""
        if not texts:
            return []
        try:
            return [
                [float(x) for x in v]
                for v in self._mgr.generate_embeddings_batch(texts)
            ]
        except Exception as e:
            raise RuntimeError(_unreachable("ollama", self.host, e)) from e

    def health_check(self) -> bool:
        """True when the daemon is reachable and the model is present."""
        return bool(self._mgr.health_check())


def _normalize_openai_url(host: str | None) -> str:
    """Build the /v1/embeddings URL from a base host.

    Accepts a base with or without a /v1 suffix and with or without a
    trailing slash, so http://box:8889, http://box:8889/ and
    http://box:8889/v1 all resolve to the same endpoint.
    """
    base = (host or "http://localhost:1234").rstrip("/")
    if not base.endswith("/v1"):
        base = f"{base}/v1"
    return f"{base}/embeddings"


class OpenAIEmbedder:
    """Generation backend over any OpenAI-compatible /v1/embeddings server.

    Verified against omlx; the same shape covers LM Studio, llama-server,
    vLLM and TEI. Unlike the other backends the wire model name cannot be
    pinned here — it depends on what the operator loaded — which is why
    check_or_seed_probe is the only guarantee that this server is serving
    the model the stored vectors came from.
    """

    def __init__(
        self,
        host: str | None,
        model: str,
        api_key: str | None = None,
        batch_size: int | None = None,
        timeout: int = 120,
    ) -> None:
        if not model:
            raise ValueError(
                "[Embedding] backend 'openai' requires the `embedding_model` "
                "config key set to the id the server uses for the embedding "
                "model (e.g. 'Qwen3-Embedding-0.6B-8bit'); run "
                "`esdc configs` or query the server's /v1/models."
            )
        from esdc.configs import Config

        self.url = _normalize_openai_url(host)
        self.wire_model = model
        self.model = MODEL_ID
        self.host = host or self.url
        self._api_key = api_key or None
        self._batch = int(batch_size or Config.get_embedding_batch_size())
        self._timeout = timeout

    def _post(self, batch: list[str]) -> list[list[float]]:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        try:
            resp = requests.post(
                self.url,
                json={"model": self.wire_model, "input": batch},
                headers=headers,
                timeout=self._timeout,
            )
        except Exception as e:
            raise RuntimeError(_unreachable("openai", self.url, e)) from e

        if resp.status_code >= 300:
            raise RuntimeError(
                f"[Embedding] {self.url} returned HTTP {resp.status_code}: "
                f"{str(resp.text)[:200]}. Check `embedding_model` matches a "
                "model the server has loaded, or set "
                "`embedding_backend: local` in ~/.esdc/config.yaml."
            )

        data = resp.json()["data"]
        # The OpenAI schema does not guarantee response order; each entry
        # carries its request index. Sorting is not optional — unsorted
        # results silently pair each text with another text's vector.
        ordered = sorted(data, key=lambda e: e.get("index", 0))
        return [[float(x) for x in e["embedding"]] for e in ordered]

    def generate_embedding(self, text: str) -> list[float]:
        """Generate an embedding for a single text."""
        return self._post([text])[0]

    def generate_embeddings_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a batch of texts, chunked by batch size."""
        if not texts:
            return []
        out: list[list[float]] = []
        for start in range(0, len(texts), self._batch):
            out.extend(self._post(texts[start : start + self._batch]))
        return out
