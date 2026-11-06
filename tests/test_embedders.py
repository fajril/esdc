# tests/test_embedders.py
"""Shared embedding clients: HTTP services, plus the loader its last callers use.

The in-process loader and its tests are removed in Task 5, once nothing
constructs it.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import duckdb
import pytest

import esdc.embedders as emb


def _fake_model():
    m = MagicMock()
    # llama_cpp embed(str) -> list[float]; embed(list) -> list[list[float]]
    m.embed.side_effect = lambda x: (
        [0.1, 0.2, 0.3] if isinstance(x, str) else [[0.1, 0.2, 0.3] for _ in x]
    )
    return m


def test_model_id_is_backend_neutral():
    assert emb.MODEL_ID == "qwen3-embedding-0.6b-q8_0"


def test_internal_embedder_reports_model_id():
    assert emb.InternalEmbedder().model == emb.MODEL_ID


def test_internal_generate_embedding_returns_floats():
    with patch.object(emb, "_get_model", return_value=_fake_model()):
        vec = emb.InternalEmbedder().generate_embedding("halo")
    assert vec == [0.1, 0.2, 0.3]
    assert all(isinstance(x, float) for x in vec)


def test_internal_batch_empty_returns_empty():
    assert emb.InternalEmbedder().generate_embeddings_batch([]) == []


def test_internal_batch_maps_each_text():
    with patch.object(emb, "_get_model", return_value=_fake_model()):
        out = emb.InternalEmbedder().generate_embeddings_batch(["a", "b"])
    assert out == [[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]]


def test_corpus_shim_reexports_same_objects():
    from esdc.corpus import embedder as shim

    assert shim.InternalEmbedder is emb.InternalEmbedder
    assert shim.MODEL_ID == emb.MODEL_ID


def test_embedders_module_has_no_module_level_search_import():
    """esdc.search imports semantic_resolver, which will import this module."""
    import sys

    sys.modules.pop("esdc.search", None)
    sys.modules.pop("esdc.embedders", None)
    import esdc.embedders  # noqa: F401

    assert "esdc.search" not in sys.modules


def _patched_manager(monkeypatch, mgr) -> MagicMock:
    """Patch the lazily-imported EmbeddingManager at its source module."""
    import esdc.search.embedding_manager as em

    fake_cls = MagicMock(return_value=mgr)
    monkeypatch.setattr(em, "EmbeddingManager", fake_cls)
    return fake_cls


def test_ollama_reports_model_id_not_wire_tag(monkeypatch):
    mgr = MagicMock()
    _patched_manager(monkeypatch, mgr)
    e = emb.OllamaEmbedder(host="http://box:11434")
    assert e.model == emb.MODEL_ID
    assert e.model != e.OLLAMA_TAG


def test_ollama_uses_the_configured_wire_model(monkeypatch):
    """D7: the configured model is what the service is asked for."""
    mgr = MagicMock()
    factory = _patched_manager(monkeypatch, mgr)
    e = emb.OllamaEmbedder(host=None, model="Qwen3-Embedding-0.6B")
    assert factory.call_args.kwargs["model"] == "Qwen3-Embedding-0.6B"
    # D1: the wire name is for the request only; identity stays MODEL_ID.
    assert e.model == emb.MODEL_ID


def test_ollama_falls_back_to_the_legacy_tag_when_model_is_empty(monkeypatch):
    mgr = MagicMock()
    factory = _patched_manager(monkeypatch, mgr)
    emb.OllamaEmbedder(host=None, model="")
    assert factory.call_args.kwargs["model"] == emb.OllamaEmbedder.OLLAMA_TAG


def test_ollama_delegates_batch(monkeypatch):
    mgr = MagicMock()
    mgr.generate_embeddings_batch.return_value = [[1.0], [2.0]]
    _patched_manager(monkeypatch, mgr)
    assert emb.OllamaEmbedder(host=None).generate_embeddings_batch(["a", "b"]) == [
        [1.0],
        [2.0],
    ]


def test_ollama_batch_empty_short_circuits(monkeypatch):
    mgr = MagicMock()
    _patched_manager(monkeypatch, mgr)
    assert emb.OllamaEmbedder(host=None).generate_embeddings_batch([]) == []
    mgr.generate_embeddings_batch.assert_not_called()


def test_ollama_error_names_host_and_the_service_requirement(monkeypatch):
    mgr = MagicMock()
    mgr.generate_embedding.side_effect = ConnectionError("refused")
    _patched_manager(monkeypatch, mgr)
    e = emb.OllamaEmbedder(host="http://box:11434")
    try:
        e.generate_embedding("x")
    except RuntimeError as err:
        msg = str(err)
    else:
        raise AssertionError("expected RuntimeError")
    assert "http://box:11434" in msg
    assert "embedding service" in msg
    # The in-process escape hatch no longer exists, so a message must not
    # suggest it.
    assert "local" not in msg
    assert "refused" in msg


@pytest.mark.parametrize(
    "host,expected",
    [
        ("http://localhost:8889", "http://localhost:8889/v1/embeddings"),
        ("http://localhost:8889/", "http://localhost:8889/v1/embeddings"),
        ("http://localhost:8889/v1", "http://localhost:8889/v1/embeddings"),
        ("http://localhost:8889/v1/", "http://localhost:8889/v1/embeddings"),
    ],
)
def test_openai_url_normalization(host, expected):
    assert emb._normalize_openai_url(host) == expected


def test_openai_requires_a_wire_model():
    with pytest.raises(ValueError, match="embedding_model"):
        emb.OpenAIEmbedder(host="http://h:1/v1", model="")


def _http_ok(payload):
    r = MagicMock()
    r.status_code = 200
    r.json.return_value = payload
    return r


def test_openai_sorts_data_by_index(monkeypatch):
    """The OpenAI schema does not promise response order; entries carry index."""
    payload = {
        "data": [
            {"index": 2, "embedding": [3.0]},
            {"index": 0, "embedding": [1.0]},
            {"index": 1, "embedding": [2.0]},
        ]
    }
    post = MagicMock(return_value=_http_ok(payload))
    monkeypatch.setattr(emb.requests, "post", post)
    e = emb.OpenAIEmbedder(host="http://h:1/v1", model="m")
    assert e.generate_embeddings_batch(["a", "b", "c"]) == [[1.0], [2.0], [3.0]]


def test_openai_raises_on_short_data(monkeypatch):
    """Fewer entries than requested raises RuntimeError naming the URL.

    Otherwise this surfaces later as an opaque strict-zip ValueError.
    """
    payload = {
        "data": [
            {"index": 0, "embedding": [1.0]},
            {"index": 1, "embedding": [2.0]},
        ]
    }
    post = MagicMock(return_value=_http_ok(payload))
    monkeypatch.setattr(emb.requests, "post", post)
    e = emb.OpenAIEmbedder(host="http://h:1/v1", model="m")
    with pytest.raises(RuntimeError) as exc:
        e.generate_embeddings_batch(["a", "b", "c"])
    assert "http://h:1/v1/embeddings" in str(exc.value)


def test_openai_raises_on_empty_data(monkeypatch):
    """Empty data -> RuntimeError, not an IndexError from data[0]."""
    post = MagicMock(return_value=_http_ok({"data": []}))
    monkeypatch.setattr(emb.requests, "post", post)
    e = emb.OpenAIEmbedder(host="http://h:1/v1", model="m")
    with pytest.raises(RuntimeError):
        e.generate_embedding("x")


def test_openai_raises_on_duplicate_index(monkeypatch):
    """Duplicate/out-of-range indexes silently mispair text<->vector; must raise."""
    payload = {
        "data": [
            {"index": 0, "embedding": [1.0]},
            {"index": 0, "embedding": [2.0]},
        ]
    }
    post = MagicMock(return_value=_http_ok(payload))
    monkeypatch.setattr(emb.requests, "post", post)
    e = emb.OpenAIEmbedder(host="http://h:1/v1", model="m")
    with pytest.raises(RuntimeError):
        e.generate_embeddings_batch(["a", "b"])


def test_openai_omits_auth_header_when_no_key(monkeypatch):
    post = MagicMock(
        return_value=_http_ok({"data": [{"index": 0, "embedding": [1.0]}]})
    )
    monkeypatch.setattr(emb.requests, "post", post)
    emb.OpenAIEmbedder(host="http://h:1/v1", model="m").generate_embedding("x")
    assert "Authorization" not in post.call_args.kwargs["headers"]


def test_openai_sends_bearer_when_key_set(monkeypatch):
    post = MagicMock(
        return_value=_http_ok({"data": [{"index": 0, "embedding": [1.0]}]})
    )
    monkeypatch.setattr(emb.requests, "post", post)
    emb.OpenAIEmbedder(host="http://h:1/v1", model="m", api_key="k").generate_embedding(
        "x"
    )
    assert post.call_args.kwargs["headers"]["Authorization"] == "Bearer k"


def test_openai_splits_batches(monkeypatch):
    post = MagicMock(
        return_value=_http_ok({"data": [{"index": 0, "embedding": [1.0]}]})
    )
    monkeypatch.setattr(emb.requests, "post", post)
    e = emb.OpenAIEmbedder(host="http://h:1/v1", model="m", batch_size=1)
    e.generate_embeddings_batch(["a", "b", "c"])
    assert post.call_count == 3


def test_openai_non_2xx_raises_with_status_and_url(monkeypatch):
    r = MagicMock()
    r.status_code = 503
    r.text = "model not loaded"
    monkeypatch.setattr(emb.requests, "post", MagicMock(return_value=r))
    e = emb.OpenAIEmbedder(host="http://h:1/v1", model="m")
    with pytest.raises(RuntimeError) as exc:
        e.generate_embedding("x")
    msg = str(exc.value)
    assert "503" in msg
    assert "http://h:1/v1/embeddings" in msg
    # The response body is never quoted: it can echo the credential.
    assert "model not loaded" not in msg


def test_openai_transport_error_names_the_service_requirement(monkeypatch):
    import requests as rq

    monkeypatch.setattr(
        emb.requests, "post", MagicMock(side_effect=rq.ConnectionError("refused"))
    )
    e = emb.OpenAIEmbedder(host="http://h:1/v1", model="m")
    with pytest.raises(RuntimeError) as exc:
        e.generate_embedding("x")
    msg = str(exc.value)
    assert "embedding service" in msg
    assert "local" not in msg
    assert "refused" in msg


def test_openai_reports_model_id(monkeypatch):
    e = emb.OpenAIEmbedder(host="http://h:1/v1", model="Qwen3-Embedding-0.6B-8bit")
    assert e.model == emb.MODEL_ID
    assert e.wire_model == "Qwen3-Embedding-0.6B-8bit"


def test_factory_local_is_a_config_error(monkeypatch):
    from esdc.configs import Config

    monkeypatch.setattr(
        Config, "get_embedding_backend", classmethod(lambda cls: "local")
    )
    with pytest.raises(emb.EmbeddingConfigError) as exc:
        emb.get_embedder("local")
    msg = str(exc.value)
    assert "in-process" in msg
    assert "Ollama" in msg or "llama-server" in msg
    assert "embedding_backend" in msg
    # A stored value is reported, never repaired.
    assert Config.get_embedding_backend() == "local"


def test_factory_ollama_uses_configured_host(monkeypatch):
    mgr = MagicMock()
    _patched_manager(monkeypatch, mgr)
    from esdc.configs import Config

    monkeypatch.setattr(
        Config, "get_embedding_host", classmethod(lambda cls: "http://box:11434")
    )
    monkeypatch.setattr(
        Config, "get_embedding_model", classmethod(lambda cls: "Qwen3-Embedding-0.6B")
    )
    e = emb.get_embedder("ollama")
    assert isinstance(e, emb.OllamaEmbedder)
    assert e.host == "http://box:11434"
    assert e.wire_model == "Qwen3-Embedding-0.6B"


def test_factory_openai_passes_model_and_key(monkeypatch):
    from esdc.configs import Config

    monkeypatch.setattr(
        Config, "get_embedding_host", classmethod(lambda cls: "http://h:8889/v1")
    )
    monkeypatch.setattr(Config, "get_embedding_model", classmethod(lambda cls: "M"))
    monkeypatch.setattr(Config, "get_embedding_api_key", classmethod(lambda cls: "K"))
    e = emb.get_embedder("openai")
    assert isinstance(e, emb.OpenAIEmbedder)
    assert e.wire_model == "M"
    assert e.url == "http://h:8889/v1/embeddings"


def test_factory_none_resolves_through_config(monkeypatch):
    from esdc.configs import Config

    monkeypatch.setattr(
        Config, "get_embedding_backend", classmethod(lambda cls: "openai")
    )
    monkeypatch.setattr(Config, "get_embedding_model", classmethod(lambda cls: "M"))
    assert isinstance(emb.get_embedder(None), emb.OpenAIEmbedder)


def test_factory_explicit_arg_beats_config(monkeypatch):
    from esdc.configs import Config

    monkeypatch.setattr(
        Config, "get_embedding_backend", classmethod(lambda cls: "ollama")
    )
    # D6: the compatibility wrapper behaves like the factory, including for the
    # retired value.
    with pytest.raises(emb.EmbeddingConfigError):
        emb.get_build_embedder("local")
    # Pinned so the assertion does not depend on the machine's own config.
    monkeypatch.setattr(
        Config, "get_embedding_host", classmethod(lambda cls: "http://h:1/v1")
    )
    monkeypatch.setattr(Config, "get_embedding_model", classmethod(lambda cls: "M"))
    mgr = MagicMock()
    _patched_manager(monkeypatch, mgr)
    assert isinstance(emb.get_build_embedder("openai"), emb.OpenAIEmbedder)


def test_factory_unknown_backend_lists_valid_ones():
    with pytest.raises(ValueError) as exc:
        emb.get_embedder("lmstudio")
    msg = str(exc.value)
    assert "lmstudio" in msg
    for name in ("ollama", "openai"):
        assert name in msg
    # The retired value is named as retired, not as a valid choice.
    assert "local" in msg


def _meta_conn(probe=None, dim=3):
    conn = duckdb.connect(":memory:")
    conn.execute(
        "CREATE TABLE m (embedding_model VARCHAR, dim INTEGER, probe_vec JSON)"
    )
    conn.execute(
        "INSERT INTO m VALUES (?, ?, ?)",
        ["qwen3-embedding-0.6b-q8_0", dim, json.dumps(probe) if probe else None],
    )
    return conn


class _FixedEmbedder:
    model = "qwen3-embedding-0.6b-q8_0"

    def __init__(self, vec):
        self._vec = vec

    def generate_embedding(self, text):
        return list(self._vec)

    def generate_embeddings_batch(self, texts):
        return [list(self._vec) for _ in texts]


def test_probe_constants():
    assert emb.PROBE_TEXT == "esdc corpus embedding parity probe"
    assert emb.PROBE_TOLERANCE == 0.995


def test_cosine_is_scale_invariant():
    assert emb.cosine([1.0, 0.0], [5.0, 0.0]) == pytest.approx(1.0)


def test_probe_seeds_when_null():
    conn = _meta_conn(probe=None)
    emb.check_or_seed_probe(conn, "m", _FixedEmbedder([1.0, 0.0, 0.0]))
    row = conn.execute("SELECT probe_vec FROM m").fetchone()
    assert row is not None
    stored = row[0]
    assert json.loads(stored) == [1.0, 0.0, 0.0]


def test_probe_passes_on_near_identical_vector():
    conn = _meta_conn(probe=[1.0, 0.0, 0.0])
    emb.check_or_seed_probe(conn, "m", _FixedEmbedder([0.999, 0.001, 0.0]))


def test_probe_raises_on_drifted_vector():
    conn = _meta_conn(probe=[1.0, 0.0, 0.0])
    with pytest.raises(ValueError) as exc:
        emb.check_or_seed_probe(conn, "m", _FixedEmbedder([0.0, 1.0, 0.0]))
    assert "0.995" in str(exc.value)


def test_probe_raises_on_dim_change():
    conn = _meta_conn(probe=[1.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="dimension"):
        emb.check_or_seed_probe(conn, "m", _FixedEmbedder([1.0, 0.0]))


def test_probe_noop_when_meta_empty():
    conn = duckdb.connect(":memory:")
    conn.execute(
        "CREATE TABLE m (embedding_model VARCHAR, dim INTEGER, probe_vec JSON)"
    )
    emb.check_or_seed_probe(conn, "m", _FixedEmbedder([1.0]))  # must not raise


# --- Shared HTTP client contract ---


def test_construction_does_not_make_a_request(monkeypatch):
    post = MagicMock()
    monkeypatch.setattr(emb.requests, "post", post)
    mgr = MagicMock()
    _patched_manager(monkeypatch, mgr)
    emb.OpenAIEmbedder(host="http://h:1/v1", model="m")
    emb.OllamaEmbedder(host=None, model="m")
    post.assert_not_called()
    mgr.generate_embedding.assert_not_called()
    mgr.generate_embeddings_batch.assert_not_called()
    mgr.health_check.assert_not_called()


def test_empty_batch_makes_no_http_request(monkeypatch):
    post = MagicMock()
    monkeypatch.setattr(emb.requests, "post", post)
    e = emb.OpenAIEmbedder(host="http://h:1/v1", model="m")
    assert e.generate_embeddings_batch([]) == []
    post.assert_not_called()


def test_query_and_batch_use_separate_timeouts(monkeypatch):
    """D3: an interactive query must not inherit the batch timeout."""
    post = MagicMock(
        return_value=_http_ok({"data": [{"index": 0, "embedding": [1.0]}]})
    )
    monkeypatch.setattr(emb.requests, "post", post)
    e = emb.OpenAIEmbedder(host="http://h:1/v1", model="m", batch_size=1)
    e.generate_embedding("x")
    assert post.call_args.kwargs["timeout"] == emb.QUERY_TIMEOUT_SECONDS
    e.generate_embeddings_batch(["x"])
    assert post.call_args.kwargs["timeout"] == emb.BATCH_TIMEOUT_SECONDS
    # One attempt per request: no retry wrapper around the post call.
    assert post.call_count == 2


def test_ollama_clients_carry_the_query_and_batch_timeouts(monkeypatch):
    """The ollama SDK binds a timeout per client, so the paths hold separate ones."""
    mgr = MagicMock()
    factory = _patched_manager(monkeypatch, mgr)
    e = emb.OllamaEmbedder(host=None, model="m")
    kwargs = [call.kwargs for call in factory.call_args_list]
    assert emb.QUERY_TIMEOUT_SECONDS in [k.get("timeout") for k in kwargs]
    assert emb.BATCH_TIMEOUT_SECONDS in [k.get("timeout") for k in kwargs]
    assert all(k.get("model") == "m" for k in kwargs)
    # D1: two clients, one vector space.
    assert e.model == emb.MODEL_ID


@pytest.mark.parametrize(
    "vectors,expected",
    [
        ([[1.0, float("inf")]], "not finite"),
        ([[1.0, float("nan")]], "not finite"),
        ([[True, 1.0]], "bool"),
        ([["1.0", None]], "not a number"),
        ([[]], "empty vector"),
        ([[0.0, 0.0]], "all zero"),
        ([[1.0, 0.0], [1.0]], "dimension"),
    ],
)
def test_response_shape_validation_rejects_unusable_vectors(
    monkeypatch, vectors, expected
):
    payload = {"data": [{"index": i, "embedding": v} for i, v in enumerate(vectors)]}
    post = MagicMock(return_value=_http_ok(payload))
    monkeypatch.setattr(emb.requests, "post", post)
    e = emb.OpenAIEmbedder(host="http://h:1/v1", model="m", batch_size=10)
    texts = [f"t{i}" for i in range(len(vectors))]
    with pytest.raises(RuntimeError) as exc:
        e.generate_embeddings_batch(texts)
    assert expected in str(exc.value)
    # A rejected response is rejected whole: no partial vector set escapes.
    assert "http://h:1/v1/embeddings" in str(exc.value)


def test_ollama_path_rejects_a_bad_response_as_a_response(monkeypatch):
    """Values are checked as they arrived, so a bool is not read as a connection failure."""
    mgr = MagicMock()
    mgr.generate_embedding.return_value = [True, 1.0]
    _patched_manager(monkeypatch, mgr)
    e = emb.OllamaEmbedder(host="http://box:11434")
    with pytest.raises(RuntimeError) as exc:
        e.generate_embedding("x")
    msg = str(exc.value)
    assert "not a number" in msg
    assert "could not reach" not in msg
    assert "http://box:11434" in msg


def test_ollama_check_available_reports_a_bad_response_as_a_response(monkeypatch):
    mgr = MagicMock()
    mgr.generate_embedding.return_value = []
    _patched_manager(monkeypatch, mgr)
    e = emb.OllamaEmbedder(host="http://box:11434")
    message = e.check_available()
    assert message is not None
    # A response that answers but is not a vector is not an unreachable service.
    assert "could not reach" not in message
    assert "empty vector" in message


def test_response_that_is_not_a_vector_list_is_rejected(monkeypatch):
    """A row must be a sequence before its length is taken."""
    payload = {"data": [{"index": 0, "embedding": None}]}
    post = MagicMock(return_value=_http_ok(payload))
    monkeypatch.setattr(emb.requests, "post", post)
    e = emb.OpenAIEmbedder(host="http://h:1/v1", model="m", batch_size=10)
    with pytest.raises(RuntimeError) as exc:
        e.generate_embeddings_batch(["t0"])
    assert "instead of a vector" in str(exc.value)


class _StatusError(Exception):
    """Stand-in for the ollama SDK's ResponseError, whose text is the body."""

    def __init__(self, body: str, status_code: int) -> None:
        super().__init__(f"{body} (status code: {status_code})")
        self.status_code = status_code


def test_service_error_reports_status_without_the_body(monkeypatch):
    """The body can echo the credential, so only the status is reported."""
    mgr = MagicMock()
    mgr.generate_embedding.side_effect = _StatusError("Bearer sk-secret rejected", 401)
    _patched_manager(monkeypatch, mgr)
    e = emb.OllamaEmbedder(host="http://box:11434")
    with pytest.raises(RuntimeError) as exc:
        e.generate_embedding("x")
    msg = str(exc.value)
    assert "401" in msg
    assert "sk-secret" not in msg
    # An answering service is not described as unreachable.
    assert "could not reach" not in msg
    # No hint of the retired in-process backend.
    assert "local" not in msg


def test_transport_error_is_reported_as_unreachable(monkeypatch):
    """A failure with no response body has nothing to withhold."""
    mgr = MagicMock()
    mgr.generate_embedding.side_effect = ConnectionError("connection refused")
    _patched_manager(monkeypatch, mgr)
    e = emb.OllamaEmbedder(host="http://box:11434")
    with pytest.raises(RuntimeError) as exc:
        e.generate_embedding("x")
    msg = str(exc.value)
    assert "could not reach" in msg
    assert "connection refused" in msg
    assert "local" not in msg


@pytest.mark.parametrize(
    "failure",
    [_StatusError("server said sk-secret", 500), ConnectionError("connection refused")],
)
def test_check_available_returns_a_message_for_any_failure(monkeypatch, failure):
    """A caller asking whether a service answers should not have to catch."""
    mgr = MagicMock()
    mgr.generate_embedding.side_effect = failure
    _patched_manager(monkeypatch, mgr)
    e = emb.OllamaEmbedder(host="http://box:11434")
    message = e.check_available()
    assert message is not None
    assert "http://box:11434" in message
    if getattr(failure, "status_code", None):
        assert "sk-secret" not in message


def test_valid_response_passes_shape_validation(monkeypatch):
    payload = {"data": [{"index": 0, "embedding": [1.0, 0.5]}]}
    post = MagicMock(return_value=_http_ok(payload))
    monkeypatch.setattr(emb.requests, "post", post)
    e = emb.OpenAIEmbedder(host="http://h:1/v1", model="m", batch_size=10)
    assert e.generate_embeddings_batch(["t0"]) == [[1.0, 0.5]]


def test_check_available_uses_the_embedding_endpoint_only(monkeypatch):
    post = MagicMock(
        return_value=_http_ok({"data": [{"index": 0, "embedding": [1.0, 0.0]}]})
    )
    monkeypatch.setattr(emb.requests, "post", post)
    e = emb.OpenAIEmbedder(host="http://h:1/v1", model="m", api_key="secret")
    assert e.check_available() is None
    assert "/embeddings" in post.call_args.args[0]
    message = str(post.call_args.kwargs["json"])
    assert "secret" not in message


def test_check_available_reports_failure_without_leaking_the_credential(monkeypatch):
    import requests as rq

    monkeypatch.setattr(
        emb.requests, "post", MagicMock(side_effect=rq.ConnectionError("refused"))
    )
    e = emb.OpenAIEmbedder(host="http://h:1/v1", model="m", api_key="secret")
    message = e.check_available()
    assert message is not None
    assert "secret" not in message
    assert "refused" in message


def test_legacy_local_config_is_read_without_writing_and_cli_still_imports(
    monkeypatch, tmp_path
):
    from esdc.configs import Config

    monkeypatch.setattr(Config, "get_config_dir", classmethod(lambda cls: tmp_path))
    config_file = tmp_path / "config.yaml"
    original = "embedding_backend: local\n"
    config_file.write_text(original)
    Config._config_cache = None

    assert Config.get_embedding_backend() == "local"
    assert config_file.read_text() == original
    assert not [p for p in tmp_path.iterdir() if p.name != "config.yaml"]

    import esdc.esdc  # noqa: F401  - importing the CLI must not need a service
