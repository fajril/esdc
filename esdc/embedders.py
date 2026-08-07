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

import atexit
import json
import logging
import math
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
# Set once _close_model begins so a late _get_model() cannot start a
# reload that would outlive the atexit handlers (that model would never
# be freed — the very SIGABRT this module removes).
_closing = False


def _close_model() -> None:
    """Free the llama.cpp model before the process exits.

    ggml-metal's dylib destructor runs at exit() and asserts every
    residency set was released (ggml-metal-device.m: GGML_ASSERT
    ([rsets->data count] == 0)). A model still alive then aborts the
    process with SIGABRT — exit 134 — after the work already succeeded.
    Dropping the last Python reference is not enough on its own: pytest
    keeps models alive in retained tracebacks, and a daemon thread's
    frame outlives interpreter finalization entirely.

    Both locks are taken with a timeout because atexit runs on the main
    thread while a daemon thread may still be inside _get_model() or
    embed(); freeing the model under them would be a use-after-free.
    Giving up restores the old abort — same outcome, no memory
    corruption. Lock order is always _load_lock then _infer_lock.
    """
    global _model, _closing
    _closing = True
    if not _load_lock.acquire(timeout=2.0):
        logger.warning("[Embedding] model loading at exit; skipping close")
        return
    try:
        if _model is None:
            return
        if not _infer_lock.acquire(timeout=2.0):
            logger.warning("[Embedding] model busy at exit; skipping close")
            return
        try:
            _model.close()
        finally:
            # Clear even if close() raises so a handler re-registered by
            # a load that finished mid-atexit never double-frees.
            _model = None
            _infer_lock.release()
    finally:
        _load_lock.release()


def _get_model() -> Any:
    """Load (once) and return the llama.cpp embedding model."""
    global _model
    with _load_lock:
        if _model is None:
            if _closing:
                # Shutdown has begun; a model loaded now would outlive
                # the atexit handlers and leak Metal buffers at exit().
                # Raising rather than returning None keeps this function's
                # contract — callers dereference the result immediately.
                raise RuntimeError(
                    "[Embedding] model load refused: interpreter shutdown has begun."
                )
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
            # Registered on load, not at import: a process that never
            # embeds owns no Metal buffers and needs no teardown.
            atexit.register(_close_model)
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
        n = len(batch)
        if len(ordered) != n or [e.get("index") for e in ordered] != list(range(n)):
            raise RuntimeError(
                f"[Embedding] {self.url} returned {len(ordered)} embeddings for a "
                f"batch of {n} with indexes {[e.get('index') for e in ordered]}; "
                f"expected exactly 0..{n - 1}. The server response is malformed or "
                "truncated. Check `embedding_model`/the server, or set "
                "`embedding_backend: local`."
            )
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


def get_build_embedder(backend: str | None = None) -> Any:
    """Construct the embedder used for BULK GENERATION.

    Never call this from a query path: similarity always runs on
    InternalEmbedder so search stays local, fast and offline.

    Args:
        backend: "local", "ollama" or "openai". None reads the
            `embedding_backend` config key (default "ollama").

    Returns:
        An embedder exposing generate_embedding, generate_embeddings_batch
        and a `.model` attribute equal to MODEL_ID.

    Raises:
        ValueError: the backend name is not one of the three known values.
    """
    from esdc.configs import Config

    name = (backend or Config.get_embedding_backend()).strip().lower()

    if name == "local":
        return InternalEmbedder()
    if name == "ollama":
        return OllamaEmbedder(host=Config.get_embedding_host())
    if name == "openai":
        return OpenAIEmbedder(
            host=Config.get_embedding_host(),
            model=Config.get_embedding_model(),
            api_key=Config.get_embedding_api_key(),
        )
    raise ValueError(
        f"[Embedding] unknown embedding_backend {name!r}; "
        "expected one of: local, ollama, openai"
    )


# Fixed sentence embedded by every backend to prove they share one cosine
# space. Never change it without reembedding every vector space.
PROBE_TEXT = "esdc corpus embedding parity probe"

# Worst measured cross-backend cosine is 0.998656 (ollama <-> mlx 8-bit),
# so 0.995 leaves ~0.004 of margin for kernel and batch variation while
# still rejecting real mismatches, which land far below 0.99.
PROBE_TOLERANCE = 0.995


def cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity. Scale-invariant, so normalized and raw vectors mix."""
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def check_or_seed_probe(conn: Any, meta_table: str, embedder: Any) -> None:
    """Verify `embedder` writes into the same space as the stored vectors.

    Seeds the probe when the column is NULL — that is the migration path
    for vector spaces created before probes existed. No-op when the meta
    table has no row yet; the caller seeds the pin first.

    Raises:
        ValueError: the embedder's dimension or direction disagrees with
            the stored probe beyond PROBE_TOLERANCE.
    """
    row = conn.execute(f"SELECT probe_vec FROM {meta_table} LIMIT 1").fetchone()
    if row is None:
        return

    current = embedder.generate_embedding(PROBE_TEXT)
    stored = row[0]

    if stored is None:
        conn.execute(f"UPDATE {meta_table} SET probe_vec = ?", [json.dumps(current)])
        logger.info("[Embedding] parity probe seeded | table=%s", meta_table)
        return

    ref = json.loads(stored) if isinstance(stored, str) else list(stored)

    if len(ref) != len(current):
        raise ValueError(
            f"[Embedding] embedding dimension changed for {meta_table} "
            f"(stored {len(ref)}, backend produces {len(current)}). The "
            "stored vectors are not comparable with this backend."
        )

    sim = cosine(current, ref)
    if sim < PROBE_TOLERANCE:
        raise ValueError(
            f"[Embedding] parity probe failed for {meta_table}: cosine "
            f"{sim:.6f} against the stored probe is below {PROBE_TOLERANCE}. "
            f"The backend reporting model={embedder.model!r} is not producing "
            "vectors in the same space as the stored ones — likely a "
            "different quantization, pooling mode or model entirely. Check "
            "`embedding_model`/`embedding_host`, or rebuild this space."
        )
    logger.debug("[Embedding] parity probe ok | table=%s cosine=%.6f", meta_table, sim)
