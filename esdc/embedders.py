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
import hashlib
import json
import logging
import math
import threading
from pathlib import Path
from typing import Any, NoReturn, Protocol

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


class EmbeddingError(RuntimeError):
    """A failure this module has already worded, so its message is body-free."""


class EmbeddingResponseError(EmbeddingError):
    """A service answered with something that cannot be a vector space.

    It is kept apart from a service failure so a caller can tell "the server is
    up but the answer is not a vector" from "the server did not answer".
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

    The numeric-type checks guard the JSON path. The ollama SDK's response model
    already narrows a component to float before the adapter sees it, so a `true`
    or a `"1.0"` arrives as a number there; count, dimension, finiteness and
    direction are what can still fail on that path.

    Returns:
        The response as floats.

    Raises:
        EmbeddingResponseError: the response cannot be a vector space. The
            message names the source and the position, never the response body.
    """
    if len(vectors) != expected:
        raise EmbeddingResponseError(
            f"[Embedding] {source} returned {len(vectors)} embeddings for a "
            f"batch of {expected}. The response is malformed or truncated."
        )
    if not vectors:
        return []

    if not isinstance(vectors[0], (list, tuple)):
        raise EmbeddingResponseError(
            f"[Embedding] {source} returned a {type(vectors[0]).__name__} "
            "instead of a vector at position 0; the response is malformed."
        )

    dim = len(vectors[0])
    if dim == 0:
        raise EmbeddingResponseError(
            f"[Embedding] {source} returned an empty vector at position 0; "
            "it carries no direction and cannot be compared."
        )

    numbers: list[list[float]] = []
    for position, vector in enumerate(vectors):
        if not isinstance(vector, (list, tuple)):
            raise EmbeddingResponseError(
                f"[Embedding] {source} returned a {type(vector).__name__} "
                f"instead of a vector at position {position}; the response is "
                "malformed."
            )
        if len(vector) != dim:
            raise EmbeddingResponseError(
                f"[Embedding] {source} returned a vector of dimension "
                f"{len(vector)} at position {position}; expected {dim}."
            )
        row: list[float] = []
        for value in vector:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise EmbeddingResponseError(
                    f"[Embedding] {source} returned a component that is not a "
                    f"number ({type(value).__name__}) at position {position}; "
                    "a vector holds numbers."
                )
            number = float(value)
            if not math.isfinite(number):
                raise EmbeddingResponseError(
                    f"[Embedding] {source} returned a component that is not "
                    f"finite at position {position}; the response cannot be a "
                    "vector."
                )
            row.append(number)
        if all(number == 0.0 for number in row):
            raise EmbeddingResponseError(
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


# Exceptions that mean the service did not answer. Their text is the failure
# itself, not a response body, so it is safe to show.
_TRANSPORT_ERRORS = (
    OSError,
    requests.exceptions.ConnectionError,
    requests.exceptions.Timeout,
)


def _status_message(backend: str, target: str, status: int) -> str:
    """Actionable message for a service that answered with an error status.

    The response body is not quoted: it can echo the request or the credential,
    and a status code is enough to act on.
    """
    return (
        f"[Embedding] backend '{backend}' service at {target} returned "
        f"HTTP {status}. Check that `embedding_model` matches a model the "
        "service has loaded, that the endpoint is an embeddings endpoint, and "
        "that the credential is accepted."
    )


def error_text(err: Exception) -> str:
    """Describe a service failure without quoting the response body.

    An error with a status code means the service answered, and its text is the
    server's body (the ollama SDK builds ResponseError from it), so only the
    status is reported. A connection or timeout error has no body and its text
    is what the user needs. Any other error may carry the body in its text, so
    only its type is reported.
    """
    status = getattr(err, "status_code", None)
    if status and status > 0:
        return f"HTTP {status}"
    if isinstance(err, _TRANSPORT_ERRORS):
        return str(err)
    return type(err).__name__


def _service_error(backend: str, target: str, err: Exception) -> str:
    """Actionable message for a service that failed.

    Three kinds of failure are named as what they are: a service that answered
    with an error status, a service that did not answer, and a service whose
    failure is neither, which is left undiagnosed rather than guessed at. No
    message quotes the response body or suggests the retired in-process backend.
    """
    status = getattr(err, "status_code", None)
    if status and status > 0:
        return _status_message(backend, target, status)
    if isinstance(err, _TRANSPORT_ERRORS):
        return (
            f"[Embedding] backend '{backend}' could not reach the embedding "
            f"service at {target}: {error_text(err)}. Check the service is "
            "running and reachable, and that "
            "`embedding_host`/`embedding_provider` point at it."
        )
    # The request completed or failed for a reason this module cannot name. The
    # checks for both readings are listed instead of picking one.
    return (
        f"[Embedding] backend '{backend}' service at {target} did not return "
        f"an embedding ({error_text(err)}). Check that the service is running "
        "and reachable, that `embedding_model` matches a model it has loaded, "
        "and that the endpoint is an embeddings endpoint."
    )


def _failure_message(backend: str, target: str, err: Exception) -> str:
    """Report a failure the way it happened.

    A failure this module worded keeps its own message; anything else is
    classified as a service failure. Both forms are body-free.
    """
    if isinstance(err, EmbeddingError):
        return str(err)
    return _service_error(backend, target, err)


def _raise_service_error(backend: str, target: str, err: Exception) -> NoReturn:
    """Report a failure without attaching a cause whose text is the body.

    A traceback prints `__cause__`, and a status error's text is the server's
    response body, which can echo the credential. The cause is kept only when
    its text is safe to print.
    """
    if isinstance(err, _TRANSPORT_ERRORS) and not getattr(err, "status_code", None):
        raise EmbeddingError(_failure_message(backend, target, err)) from err
    raise EmbeddingError(_failure_message(backend, target, err)) from None


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
        # Two clients built from the same connection are interchangeable for
        # memoization, so the probe cache can span store/resolver instances.
        self.cache_key = ("ollama", str(host or ""), self.wire_model)
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
            _raise_service_error("ollama", self.host, e)
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
            _raise_service_error("ollama", self.host, e)
        return _validate_vectors(raw, len(texts), self.host)

    def check_available(self) -> str | None:
        """Ask the embedding endpoint only; None when it answers.

        Nothing is raised: a caller asking whether a service answers should get a
        message, not an exception. A response that is not a usable vector is
        reported as a bad response, and the two kinds of failure are named
        through the same classifier as the request paths.
        """
        try:
            raw = self._query_mgr.generate_embedding("availability check")
            _validate_vectors([raw], 1, self.host)
        except Exception as e:
            return _failure_message("ollama", self.host, e)
        return None


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
        model = str(model or "").strip()
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
        # Two clients built from the same connection are interchangeable for
        # memoization, so the probe cache can span store/resolver instances.
        credential = hashlib.sha256(str(self._api_key or "").encode()).hexdigest()
        self.cache_key = ("openai", self.url, self.wire_model, credential)

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
            _raise_service_error("openai", self.url, e)

        if resp.status_code >= 300:
            raise EmbeddingError(_status_message("openai", self.url, resp.status_code))

        # Everything past here is an answer that is not a vector space, so it is
        # reported as a bad response, never as a service that failed.
        try:
            body = resp.json()
        except ValueError as e:
            raise EmbeddingResponseError(
                f"[Embedding] {self.url} answered with {type(e).__name__} instead "
                "of a JSON embedding response."
            ) from None
        data = body.get("data") if isinstance(body, dict) else None
        if not isinstance(data, list):
            raise EmbeddingResponseError(
                f"[Embedding] {self.url} did not return a 'data' list; the answer "
                "is not the embeddings schema this client expects."
            )
        if any(not isinstance(entry, dict) for entry in data):
            raise EmbeddingResponseError(
                f"[Embedding] {self.url} returned a row that is not an object; "
                "the answer is not the embeddings schema this client expects."
            )
        # The OpenAI schema does not guarantee response order; each entry
        # carries its request index. Sorting is not optional — unsorted
        # results silently pair each text with another text's vector. An index
        # that is not an integer is rejected before sorting, so a malformed
        # answer cannot escape as a TypeError from the comparison.
        indexes = [entry.get("index") for entry in data]
        if any(
            isinstance(index, bool) or not isinstance(index, int) for index in indexes
        ):
            raise EmbeddingResponseError(
                f"[Embedding] {self.url} returned an index that is not an integer; "
                "the answer is not the embeddings schema this client expects."
            )
        ordered = sorted(zip(indexes, data, strict=True), key=lambda pair: pair[0])
        n = len(batch)
        ordered_indexes = [index for index, _ in ordered]
        if len(ordered) != n or ordered_indexes != list(range(n)):
            raise EmbeddingResponseError(
                f"[Embedding] {self.url} returned {len(ordered)} embeddings for a "
                f"batch of {n} with indexes {ordered_indexes}; "
                f"expected exactly 0..{n - 1}. The response is malformed or "
                "truncated."
            )
        # A missing key is reported as a bad row by the validator, not as a KeyError.
        raw = [entry.get("embedding") for _, entry in ordered]
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
        request can still pay a cold-start cost on the service. Nothing is
        raised, and a bad response is classified the same way as on the request
        path.
        """
        try:
            self._post(["availability check"], self._query_timeout)
        except Exception as e:
            return _failure_message("openai", self.url, e)
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


def config_signature(backend: str | None = None) -> tuple[str, str, str, str]:
    """Identity of the effective config a client was resolved from.

    Which service answers, and with which weights, is decided by backend, host, wire
    model and credential together. A probe validation is only good for that exact
    combination: changing the wire model or the credential can point at a different
    deployment, so a validation recorded under the old one must not be reused.

    The credential is carried as a digest, never in the clear, so a signature can be
    compared or logged without exposing it.
    """
    from esdc.configs import Config

    name = (backend or Config.get_embedding_backend()).strip().lower()
    host = str(Config.get_embedding_host() or "").strip()
    wire = str(Config.get_embedding_model() or "").strip()
    credential = hashlib.sha256(
        str(Config.get_embedding_api_key() or "").encode()
    ).hexdigest()
    return (name, host, wire, credential)


def db_signature(path: Any) -> int | None:
    """Identity of a database file: its inode; None when it does not exist.

    The inode says which file a memo key belongs to, and it is stable across our own
    writes, so a validation is not thrown away by the write it is protecting. It is not
    enough on its own: `esdc fetch` and `esdc reload` publish into the same file, so the
    inode usually does not change. That is why the key also carries the stored pin,
    dimension and probe, which an in-place publish can change.
    """
    try:
        return Path(str(path)).stat().st_ino
    except OSError:
        return None


def probe_digest(value: Any) -> str | None:
    """Canonical hash of a stored probe, so a memo key can carry it cheaply.

    None when there is no probe to carry. An unreadable probe hashes to None as well:
    the check that reads it cannot succeed against it, so the difference never reaches
    the memo.
    """
    if value is None:
        return None
    try:
        vector = json.loads(value) if isinstance(value, str) else list(value)
    except (TypeError, ValueError):
        return None
    return hashlib.sha256(json.dumps([float(x) for x in vector]).encode()).hexdigest()


def probe_memo_key(
    conn: Any,
    meta_table: str,
    path: Any,
    config_sig: tuple[str, str, str, str] | None = None,
) -> tuple[Any, ...]:
    """The key a probe validation is memoized under.

    `(effective config, DB file, stored pin, stored dimension, stored probe)` — the
    facts a validation was computed from. Reading the row costs nothing beyond a SELECT;
    the service is not asked. A rewrite that changes any of them re-validates, which is
    what a file identity alone could not catch: `esdc fetch` and `esdc reload` publish
    in place, so the same inode can hold a different pin, dimension or probe. A
    writer's own writes change vectors, not the pin, the dimension or the probe, so
    they cannot invalidate the memo that protects them.

    Recorded by the caller as the state stands *after* a successful check, and only
    for a check that succeeded.
    """
    try:
        row = conn.execute(
            f"SELECT embedding_model, dim, probe_vec FROM {meta_table} LIMIT 1"
        ).fetchone()
    except Exception:
        # A legacy or absent metadata table has no stored facts to key on; the caller's
        # check reports that as a degradation rather than memoizing it.
        row = None
    stored_model, stored_dim, stored_probe = row if row else (None, None, None)
    return (
        config_sig if config_sig is not None else config_signature(),
        db_signature(path),
        stored_model,
        stored_dim,
        probe_digest(stored_probe),
    )


# Process-level memo for successful probe validations. D2 asks for one probe
# request per (effective config, DB state), not one per operation; a chat turn
# builds a fresh CorpusStore per tool call, so the memo cannot live only on the
# store. The client identity is part of the key: HTTP clients expose a stable
# `cache_key`, while an injected test client falls back to its object id.
_PROBE_MEMO: dict[tuple[Any, ...], None] = {}
_PROBE_MEMO_LOCK = threading.Lock()
_PROBE_MEMO_MAX = 2048


def client_identity(embedder: Any) -> tuple[Any, ...]:
    """Stable identity for a client built from the same connection."""
    key = getattr(embedder, "cache_key", None)
    if isinstance(key, tuple):
        try:
            hash(key)
        except TypeError:
            pass
        else:
            return key
    return ("instance", id(embedder))


def probe_is_verified(key: tuple[Any, ...], embedder: Any) -> bool:
    """True when this exact config/DB/client state already passed the probe."""
    full = key + (client_identity(embedder),)
    with _PROBE_MEMO_LOCK:
        return full in _PROBE_MEMO


def remember_probe_verified(key: tuple[Any, ...], embedder: Any) -> None:
    """Record a successful probe check for this exact state."""
    full = key + (client_identity(embedder),)
    with _PROBE_MEMO_LOCK:
        if len(_PROBE_MEMO) >= _PROBE_MEMO_MAX:
            _PROBE_MEMO.clear()
        _PROBE_MEMO[full] = None


def clear_probe_memo() -> None:
    """Drop every recorded probe validation (tests and explicit resets)."""
    with _PROBE_MEMO_LOCK:
        _PROBE_MEMO.clear()


def _row_count(conn: Any, table: str) -> int:
    """Rows in a vector table; 0 when it does not exist yet."""
    try:
        row = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
    except Exception:
        return 0
    return int(row[0]) if row else 0


def _probe_message(
    meta_table: str,
    stored_model: str | None,
    current_model: str | None,
    detail: str,
) -> str:
    """Refusal that names both spaces being compared, never a credential."""
    return (
        f"[Embedding] parity probe failed for {meta_table}: {detail}. "
        f"The client reporting model={current_model!r} cannot be trusted against "
        f"vectors pinned to {stored_model!r}."
    )


def probe_status(
    conn: Any, meta_table: str, embedder: Any, vector_table: str | None = None
) -> dict[str, Any]:
    """Read the stored probe and say whether `embedder` can be honoured against it.

    Reads only: a query must not repair the pin as a side effect of asking for a
    vector. Three readings come back:

    * ``match`` — the stored probe agrees with this client.
    * ``unverified`` — nothing is pinned yet. With the index behind the pin still
      empty there is no vector space to protect, so a writer may seed it.
    * ``error`` — the stored probe disagrees, or an index that already holds
      vectors has no probe to compare against. A probe from a new service cannot
      solemnize vectors it did not produce.

    Returns:
        ``status``, the two model names being compared, and the detail behind the
        reading. The message names both models; a model name is not proof of a space,
        but it is what tells a reader which two are being compared.
    """
    current_model = getattr(embedder, "model", None)
    row = conn.execute(
        f"SELECT embedding_model, probe_vec FROM {meta_table} LIMIT 1"
    ).fetchone()

    if row is None:
        return {
            "status": "unverified",
            "stored_model": None,
            "current_model": current_model,
            "detail": None,
            "message": None,
        }

    stored_model, stored = row[0], row[1]

    if stored is None:
        # Nothing to compare against. Seeding is only the migration path for a
        # space that is still empty. Once vectors exist, a probe this client just
        # made cannot certify vectors it did not produce — and a model name is not
        # proof of a space, because every client reports the same MODEL_ID.
        if not vector_table or _row_count(conn, vector_table) == 0:
            return {
                "status": "unverified",
                "stored_model": stored_model,
                "current_model": current_model,
                "detail": None,
                "message": None,
            }
        detail = (
            f"no stored probe for {meta_table} while {vector_table} already "
            "holds vectors — a probe from this service cannot certify vectors "
            "produced elsewhere. Run `esdc corpus reembed` or "
            "`esdc reload --embeddings-only` to rebuild the space"
        )
        return {
            "status": "error",
            "stored_model": stored_model,
            "current_model": current_model,
            "detail": detail,
            "message": _probe_message(meta_table, stored_model, current_model, detail),
        }

    ref = json.loads(stored) if isinstance(stored, str) else list(stored)
    detail = probe_mismatch(ref, embedder.generate_embedding(PROBE_TEXT))
    if detail is None:
        return {
            "status": "match",
            "stored_model": stored_model,
            "current_model": current_model,
            "detail": None,
            "message": None,
        }
    return {
        "status": "error",
        "stored_model": stored_model,
        "current_model": current_model,
        "detail": detail,
        "message": _probe_message(meta_table, stored_model, current_model, detail),
    }


def check_or_seed_probe(
    conn: Any, meta_table: str, embedder: Any, vector_table: str | None = None
) -> None:
    """Verify `embedder` writes into the same space as the stored vectors.

    Seeds the probe when the column is NULL — that is the migration path
    for vector spaces created before probes existed, and only for a space that
    is still empty. Seeding an index that already holds vectors would certify
    them as this client's work, which is why it is refused instead. No-op when
    the meta table has no row yet; the caller seeds the pin first.

    Raises:
        ValueError: the embedder's dimension or direction disagrees with
            the stored probe beyond PROBE_TOLERANCE, or a non-empty index
            has no probe to compare against.
    """
    status = probe_status(conn, meta_table, embedder, vector_table)

    if status["status"] == "match":
        logger.debug("[Embedding] parity probe ok | table=%s", meta_table)
        return

    if status["status"] == "unverified":
        if status["detail"] is None and status["stored_model"] is not None:
            current = embedder.generate_embedding(PROBE_TEXT)
            conn.execute(
                f"UPDATE {meta_table} SET probe_vec = ?", [json.dumps(current)]
            )
            logger.info("[Embedding] parity probe seeded | table=%s", meta_table)
        return

    raise ValueError(status["message"])
