# esdc/embedders.py
"""Embedding clients shared by every vector space in esdc.

Query-time similarity and bulk generation both resolve through
`get_embedder()`, which builds a client for the configured HTTP service
(Ollama or any OpenAI-compatible `/v1/embeddings` server, localhost
included). `InternalEmbedder` remains only while its last callers move; it
is removed once nothing constructs it.

Every client reports MODEL_ID as its `.model`, never a service-specific
wire name: that value is the vector-space identity written into
corpus_meta/semantic_meta and into per-row embedding_model columns.
Measured cross-backend cosine is >= 0.9986, so all clients write into one
cosine space; check_or_seed_probe enforces that at runtime. The wire name
chooses the request, the probe proves the space.
"""

from __future__ import annotations

import atexit
import json
import logging
import math
import threading
from typing import Any, Protocol

import requests

logger = logging.getLogger(__name__)

# Stable identifier pinned in corpus_meta/semantic_meta. Backend-neutral on
# purpose: an Ollama or OpenAI-compatible server running the same weights
# writes into the same space under the same pin. Tied to EMBED_REPO/
# EMBED_FILE in llama_backend; changing the GGUF means changing this +
# reembedding.
MODEL_ID = "qwen3-embedding-0.6b-q8_0"

# An interactive query must not inherit the bulk-generation bound: a service
# that hangs would hold one chat turn for the whole batch timeout.
QUERY_TIMEOUT_SECONDS = 20.0
BATCH_TIMEOUT_SECONDS = 120.0


class EmbeddingConfigError(ValueError):
    """A stored backend value that no longer has an implementation.

    Raised when a client is asked for, never when a config file is read: a
    legacy value is reported, not repaired.
    """


class EmbeddingClient(Protocol):
    """One service connection in, one vector space out."""

    model: str

    def generate_embedding(self, text: str) -> list[float]: ...
    def generate_embeddings_batch(self, texts: list[str]) -> list[list[float]]: ...
    def check_available(self) -> str | None: ...


def _service_required(backend: str) -> str:
    """Actionable message for a retired in-process backend."""
    return (
        f"[Embedding] backend '{backend}' needs an HTTP embedding service. "
        "Run Ollama, omlx, or llama-server and set `embedding_backend` and "
        "`embedding_host`/`embedding_provider` to that service. The in-process "
        "backend was removed in this version."
    )


def _validate_vectors(
    vectors: list[list[Any]], expected: int, source: str
) -> list[list[float]]:
    """Reject a response that cannot be used as a vector space, as a whole.

    The values are checked as they arrived, before any conversion: count,
    dimension, numeric type, finiteness and a non-zero direction. A response
    that fails here fails entirely, so no partial set of vectors can be written
    into the index.

    Returns:
        The response as floats.

    Raises:
        RuntimeError: the response cannot be a vector space. The message names
            the source and the position, never the response body.
    """
    if len(vectors) != expected:
        raise RuntimeError(
            f"[Embedding] {source} returned {len(vectors)} embeddings for a "
            f"batch of {expected}. The response is malformed or truncated."
        )
    if not vectors:
        return []

    if not isinstance(vectors[0], (list, tuple)):
        raise RuntimeError(
            f"[Embedding] {source} returned a {type(vectors[0]).__name__} "
            "instead of a vector at position 0; the response is malformed."
        )

    dim = len(vectors[0])
    if dim == 0:
        raise RuntimeError(
            f"[Embedding] {source} returned an empty vector at position 0; "
            "it carries no direction and cannot be compared."
        )

    numbers: list[list[float]] = []
    for position, vector in enumerate(vectors):
        if not isinstance(vector, (list, tuple)):
            raise RuntimeError(
                f"[Embedding] {source} returned a {type(vector).__name__} "
                f"instead of a vector at position {position}; the response is "
                "malformed."
            )
        if len(vector) != dim:
            raise RuntimeError(
                f"[Embedding] {source} returned a vector of dimension "
                f"{len(vector)} at position {position}; expected {dim}."
            )
        row: list[float] = []
        for value in vector:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise RuntimeError(
                    f"[Embedding] {source} returned a component that is not a "
                    f"number ({type(value).__name__}) at position {position}; "
                    "a vector holds numbers."
                )
            number = float(value)
            if not math.isfinite(number):
                raise RuntimeError(
                    f"[Embedding] {source} returned a component that is not "
                    f"finite at position {position}; the response cannot be a "
                    "vector."
                )
            row.append(number)
        if all(number == 0.0 for number in row):
            raise RuntimeError(
                f"[Embedding] {source} returned a vector that is all zero at "
                f"position {position}; it carries no direction and cannot be "
                "compared."
            )
        numbers.append(row)
    return numbers


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


def error_text(err: Exception) -> str:
    """Describe a service failure without quoting the response body.

    An error that carries a status code means the service answered: its body
    can echo the credential or the request (the ollama SDK's ResponseError text
    is the server's body), so only the status is reported. A connection error
    has no body and its text is what the user needs.
    """
    status = getattr(err, "status_code", None)
    if status:
        return f"HTTP {status}"
    return str(err)


def _service_error(backend: str, target: str, err: Exception) -> str:
    """Actionable message for a service that failed.

    A service that answered with an error status is named as an error status,
    not as unreachable; a service that did not answer is named as unreachable.
    Neither message quotes the response body or suggests the retired
    in-process backend.
    """
    if getattr(err, "status_code", None):
        return (
            f"[Embedding] backend '{backend}' service at {target} returned "
            f"{error_text(err)}. Check `embedding_model` matches a model the "
            "service has loaded, and that the endpoint is an embeddings "
            "endpoint."
        )
    return (
        f"[Embedding] backend '{backend}' could not reach the embedding "
        f"service at {target}: {error_text(err)}. Check the service is running "
        "and reachable, and that `embedding_host`/`embedding_provider` point "
        "at it."
    )


class OllamaEmbedder:
    """Client for an Ollama service, used for query and for generation.

    The wire model is the configured one; an empty value falls back to the
    tag the current index was built against. `.model` stays MODEL_ID so a
    service alias change does not repin corpus_meta — the probe is what
    proves the space.

    The ollama SDK binds a timeout to the client rather than to a request,
    so the query path and the batch path hold separate clients.
    """

    OLLAMA_TAG = "qwen3-embedding:0.6b"

    def __init__(self, host: str | None = None, model: str | None = None) -> None:
        # Lazy import: esdc/search/__init__.py imports semantic_resolver,
        # which imports this module. A module-level import here would be a
        # circular import.
        from esdc.search.embedding_manager import EmbeddingManager

        self.host = host or "local Ollama service"
        self.wire_model = str(model or "").strip() or self.OLLAMA_TAG
        self.model = MODEL_ID
        self._query_mgr = EmbeddingManager(
            model=self.wire_model, host=host, timeout=QUERY_TIMEOUT_SECONDS
        )
        self._batch_mgr = EmbeddingManager(
            model=self.wire_model, host=host, timeout=BATCH_TIMEOUT_SECONDS
        )

    def generate_embedding(self, text: str) -> list[float]:
        """Generate an embedding for a single text."""
        try:
            raw = self._query_mgr.generate_embedding(text)
        except Exception as e:
            raise RuntimeError(_service_error("ollama", self.host, e)) from e
        # The response is validated as it arrived, so a bad value is reported
        # as a bad response rather than as a connection failure.
        return _validate_vectors([raw], 1, self.host)[0]

    def generate_embeddings_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a batch of texts.

        An empty batch makes no request.
        """
        if not texts:
            return []
        try:
            raw = self._batch_mgr.generate_embeddings_batch(texts)
        except Exception as e:
            raise RuntimeError(_service_error("ollama", self.host, e)) from e
        return _validate_vectors(raw, len(texts), self.host)

    def check_available(self) -> str | None:
        """Ask the embedding endpoint only; None when it answers.

        A response that is not a usable vector is reported as a bad response,
        not as an unreachable service, and nothing is raised: a caller asking
        whether the service answers should get a message, not an exception.
        """
        try:
            raw = self._query_mgr.generate_embedding("availability check")
        except Exception as e:
            return _service_error("ollama", self.host, e)
        try:
            _validate_vectors([raw], 1, self.host)
        except RuntimeError as e:
            return str(e)
        return None

    def health_check(self) -> bool:
        """True when the service answers on its embedding endpoint."""
        return self.check_available() is None


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
    """Client for any OpenAI-compatible /v1/embeddings server.

    Verified against omlx; the same shape covers LM Studio, llama-server,
    vLLM and TEI. The wire model name cannot be pinned here — it depends on
    what the operator loaded — which is why check_or_seed_probe is the only
    guarantee that this server is serving the model the stored vectors came
    from.

    One attempt per request, no retry: a service that fails is reported, not
    hammered. Query calls use the query timeout, batch calls the batch one.
    """

    def __init__(
        self,
        host: str | None,
        model: str,
        api_key: str | None = None,
        batch_size: int | None = None,
        query_timeout: float = QUERY_TIMEOUT_SECONDS,
        batch_timeout: float = BATCH_TIMEOUT_SECONDS,
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
        self._query_timeout = query_timeout
        self._batch_timeout = batch_timeout

    def _post(self, batch: list[str], timeout: float) -> list[list[float]]:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        try:
            resp = requests.post(
                self.url,
                json={"model": self.wire_model, "input": batch},
                headers=headers,
                timeout=timeout,
            )
        except Exception as e:
            raise RuntimeError(_service_error("openai", self.url, e)) from e

        if resp.status_code >= 300:
            # The response body is not quoted: it can echo the request or the
            # credential, and it is not needed to act on a status code.
            raise RuntimeError(
                f"[Embedding] {self.url} returned HTTP {resp.status_code}. "
                "Check `embedding_model` matches a model the server has "
                "loaded, and that the service is an embeddings endpoint."
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
                "truncated."
            )
        raw = [e["embedding"] for e in ordered]
        return _validate_vectors(raw, n, self.url)

    def generate_embedding(self, text: str) -> list[float]:
        """Generate a single embedding on the query timeout."""
        return self._post([text], self._query_timeout)[0]

    def generate_embeddings_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a batch of texts, chunked by batch size.

        An empty batch makes no request.
        """
        if not texts:
            return []
        out: list[list[float]] = []
        for start in range(0, len(texts), self._batch):
            out.extend(
                self._post(texts[start : start + self._batch], self._batch_timeout)
            )
        return out

    def check_available(self) -> str | None:
        """Ask the embedding endpoint only; None when it answers.

        Warm-up here means reachability, not a loaded model: the first real
        request can still pay a cold-start cost on the service.
        """
        try:
            self._post(["availability check"], self._query_timeout)
        except Exception as e:
            return str(e)
        return None


def get_embedder(backend: str | None = None) -> EmbeddingClient:
    """Construct the client for the configured HTTP embedding service.

    Query-time similarity and bulk generation resolve through the same
    factory: there is no in-process backend to fall back to. Construction
    makes no request — a client is built from config shape, and connectivity
    is only proven when a vector is asked for or `check_available()` runs.

    Args:
        backend: "ollama" or "openai". None reads the `embedding_backend`
            config key (default "ollama").

    Returns:
        An EmbeddingClient exposing generate_embedding,
        generate_embeddings_batch, check_available and a `.model` attribute
        equal to MODEL_ID.

    Raises:
        EmbeddingConfigError: the stored value has no implementation any more
            ("local"). A stored value is reported, never rewritten.
        ValueError: the backend name is not a known one.
    """
    from esdc.configs import Config

    name = (backend or Config.get_embedding_backend()).strip().lower()

    if name == "local":
        raise EmbeddingConfigError(_service_required(name))
    if name == "ollama":
        return OllamaEmbedder(
            host=Config.get_embedding_host(),
            model=Config.get_embedding_model(),
        )
    if name == "openai":
        return OpenAIEmbedder(
            host=Config.get_embedding_host(),
            model=Config.get_embedding_model(),
            api_key=Config.get_embedding_api_key(),
        )
    raise ValueError(
        f"[Embedding] unknown embedding_backend {name!r}; expected one of: "
        "ollama, openai. 'local' is retired: it needs an HTTP service."
    )


def get_build_embedder(backend: str | None = None) -> EmbeddingClient:
    """Compatibility alias for `get_embedder` (D6).

    The old contract — "similarity always runs on InternalEmbedder, never
    call this from a query path" — no longer holds: query and generation now
    resolve through the same factory. The name stays because existing callers
    and tests patch it.
    """
    return get_embedder(backend)


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


def probe_mismatch(stored: list[float], current: list[float]) -> str | None:
    """Describe how two probe vectors disagree, or None when they match.

    Pure comparison of dimension first, then cosine against PROBE_TOLERANCE.
    Shared by the writer-side ``check_or_seed_probe`` (which raises on the
    detail) and read-side semantic metadata validation (which degrades it to
    a ``not_available`` message), so both enforce one compatibility rule.
    The returned detail names the dimension or cosine, never a model label:
    a label alone does not prove two vectors share a space.
    """
    if len(stored) != len(current):
        return (
            f"stored probe dimension {len(stored)} does not match the active "
            f"embedder dimension {len(current)}"
        )
    sim = cosine(current, stored)
    if sim < PROBE_TOLERANCE:
        return f"stored probe cosine {sim:.6f} is below the required {PROBE_TOLERANCE}"
    return None


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

    mismatch = probe_mismatch(ref, current)
    if mismatch is not None:
        raise ValueError(
            f"[Embedding] parity probe failed for {meta_table}: {mismatch}. "
            f"The backend reporting model={embedder.model!r} is not producing "
            "vectors in the same space as the stored ones — likely a "
            "different quantization, pooling mode or model entirely. Check "
            "`embedding_model`/`embedding_host`, or rebuild this space."
        )
    logger.debug("[Embedding] parity probe ok | table=%s", meta_table)
