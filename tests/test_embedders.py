# tests/test_embedders.py
"""Shared embedding clients: the HTTP service factory and its contracts.

There is no in-process loader any more, so nothing here patches a model object:
every client is built from config shape and exercised through a fake transport.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import duckdb
import pytest

import esdc.embedders as emb
from esdc.configs import Config
from esdc.model_runtime import ResolvedTarget, ResolvedTask


def _install_embedding_task(
    monkeypatch,
    *,
    backend: str = "openai",
    host: str | None = "http://h:1/v1",
    model: str = "M",
    api_key: str = "",
) -> None:
    provider_type = "openai_compatible" if backend == "openai" else "ollama"
    target = ResolvedTarget(
        connection="embed",
        provider={
            "name": "embed",
            "provider_type": provider_type,
            "type": provider_type,
            "base_url": host or "",
            "api_key": api_key,
            "model": model,
        },
        model=model,
        options={},
    )
    task = ResolvedTask(
        role="embedding",
        mode="model",
        enabled=True,
        primary=target,
    )
    monkeypatch.setattr(
        Config,
        "resolve_model_task",
        classmethod(lambda cls, role, apply_overrides=True: task),
    )


def test_model_id_is_backend_neutral():
    assert emb.MODEL_ID == "qwen3-embedding-0.6b-q8_0"


def test_no_in_process_seam_remains():
    """The retired loader is removed, not quietly aliased to the HTTP client."""
    for name in ("InternalEmbedder", "_get_model", "_close_model", "_model"):
        assert not hasattr(emb, name), name


def test_corpus_shim_reexports_the_contract():
    from esdc.corpus import embedder as shim

    assert shim.MODEL_ID == emb.MODEL_ID
    assert shim.EmbeddingClient is emb.EmbeddingClient
    assert sorted(shim.__all__) == ["EmbeddingClient", "MODEL_ID"]


def test_embedders_module_has_no_module_level_search_import():
    """esdc.search imports semantic_resolver, which will import this module.

    The originals are restored in `finally`. A fresh import creates a second
    module object, and both `sys.modules` and the `esdc` package attribute point
    at it, so callers that imported `esdc.embedders` first would keep the
    original while a later test patched the new one — the patch would not reach
    them and the test would reach a real service instead of a fake one.
    """
    import sys

    import esdc

    original_embedders = sys.modules.get("esdc.embedders")
    original_search = sys.modules.get("esdc.search")
    try:
        sys.modules.pop("esdc.search", None)
        sys.modules.pop("esdc.embedders", None)
        import esdc.embedders  # noqa: F401

        assert "esdc.search" not in sys.modules
    finally:
        for name, original in (
            ("esdc.embedders", original_embedders),
            ("esdc.search", original_search),
        ):
            attribute = name.split(".", 1)[1]
            if original is None:
                sys.modules.pop(name, None)
                if hasattr(esdc, attribute):
                    delattr(esdc, attribute)
                continue
            sys.modules[name] = original
            setattr(esdc, attribute, original)


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
    _install_embedding_task(monkeypatch, backend="openai")
    with pytest.raises(emb.EmbeddingConfigError) as exc:
        emb.get_embedder("local")
    msg = str(exc.value)
    assert "in-process" in msg
    assert "Ollama" in msg or "OpenAI-compatible" in msg
    assert "Embeddings task" in msg


def test_factory_ollama_uses_assigned_host_and_model(monkeypatch):
    mgr = MagicMock()
    _patched_manager(monkeypatch, mgr)
    _install_embedding_task(
        monkeypatch,
        backend="ollama",
        host="http://box:11434",
        model="Qwen3-Embedding-0.6B",
    )
    e = emb.get_embedder("ollama")
    assert isinstance(e, emb.OllamaEmbedder)
    assert e.host == "http://box:11434"
    assert e.wire_model == "Qwen3-Embedding-0.6B"


def test_factory_openai_passes_assigned_model_and_key(monkeypatch):
    _install_embedding_task(
        monkeypatch,
        backend="openai",
        host="http://h:8889/v1",
        model="M",
        api_key="K",
    )
    e = emb.get_embedder("openai")
    assert isinstance(e, emb.OpenAIEmbedder)
    assert e.wire_model == "M"
    assert e.url == "http://h:8889/v1/embeddings"


def test_factory_none_infers_backend_from_connection_type(monkeypatch):
    _install_embedding_task(monkeypatch, backend="openai", model="M")
    assert isinstance(emb.get_embedder(None), emb.OpenAIEmbedder)


def test_factory_explicit_backend_must_match_the_assignment(monkeypatch):
    _install_embedding_task(monkeypatch, backend="ollama")
    # D6: the compatibility wrapper behaves like the factory, including for the
    # retired value.
    with pytest.raises(emb.EmbeddingConfigError):
        emb.get_build_embedder("local")
    with pytest.raises(ValueError, match="does not match"):
        emb.get_build_embedder("openai")
    assert isinstance(emb.get_build_embedder("ollama"), emb.OllamaEmbedder)


def test_factory_unknown_backend_lists_valid_ones(monkeypatch):
    _install_embedding_task(monkeypatch, backend="openai", model="M")
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


def test_probe_constants_are_frozen():
    """Pinned because no default run exercises them behaviourally.

    Changing PROBE_TEXT re-embeds every space, and the PROBE_TOLERANCE boundary
    is asserted only in the skipif-gated `test_embed_parity_integration.py`.
    """
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


def test_timeout_values_are_the_confirmed_decisions():
    """D3: pin the numbers themselves, not only their use."""
    assert emb.QUERY_TIMEOUT_SECONDS == 20.0
    assert emb.BATCH_TIMEOUT_SECONDS == 120.0


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


def test_index_mismatch_survives_the_availability_check(monkeypatch):
    """The count and indexes are what say the pairing is wrong, so they are kept."""
    payload = {"data": [{"index": 1, "embedding": [1.0]}]}
    monkeypatch.setattr(emb.requests, "post", MagicMock(return_value=_http_ok(payload)))
    e = emb.OpenAIEmbedder(host="http://h:1/v1", model="m")
    message = e.check_available()
    assert message is not None
    assert "indexes" in message
    # The range is interpolated, so the message says what the pairing should have been.
    assert "expected exactly 0..0" in message
    assert "{n - 1}" not in message
    assert "did not return an embedding" not in message


def test_an_answer_that_is_not_json_is_a_bad_answer(monkeypatch):
    """A server that answers with anything else is not read as an unclassified failure."""
    response = MagicMock()
    response.status_code = 200
    response.json.side_effect = ValueError("Unexpected UTF-8 BOM")
    monkeypatch.setattr(emb.requests, "post", MagicMock(return_value=response))
    e = emb.OpenAIEmbedder(host="http://h:1/v1", model="m")
    message = e.check_available()
    assert message is not None
    assert "instead of a JSON" in message
    assert "could not reach" not in message
    assert "did not return an embedding" not in message


def test_an_answer_that_is_not_the_embeddings_schema_is_a_bad_answer(monkeypatch):
    payload = {"object": "list", "data": None}
    monkeypatch.setattr(emb.requests, "post", MagicMock(return_value=_http_ok(payload)))
    e = emb.OpenAIEmbedder(host="http://h:1/v1", model="m")
    message = e.check_available()
    assert message is not None
    assert "did not return a 'data' list" in message
    # A row without the key is reported by the validator, not as a KeyError.
    assert "KeyError" not in message


def test_ollama_path_rejects_a_bad_response_as_a_response(monkeypatch):
    """Values are checked as they arrived, so a bad answer is not a connection failure.

    The SDK's response model narrows a component to float before the adapter
    sees it, so what can still arrive unusable on this path is a non-finite value.
    """
    mgr = MagicMock()
    mgr.generate_embedding.return_value = [1.0, float("nan")]
    _patched_manager(monkeypatch, mgr)
    e = emb.OllamaEmbedder(host="http://box:11434")
    with pytest.raises(emb.EmbeddingResponseError) as exc:
        e.generate_embedding("x")
    msg = str(exc.value)
    assert "not finite" in msg
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


def test_mixed_index_types_are_a_bad_answer(monkeypatch):
    """A malformed index must not escape as a TypeError from sorting."""
    payload = {
        "data": [
            {"index": 1, "embedding": [1.0]},
            {"index": "0", "embedding": [2.0]},
        ]
    }
    post = MagicMock(return_value=_http_ok(payload))
    monkeypatch.setattr(emb.requests, "post", post)
    e = emb.OpenAIEmbedder(host="http://h:1/v1", model="m", batch_size=10)
    with pytest.raises(emb.EmbeddingResponseError) as exc:
        e.generate_embeddings_batch(["t0", "t1"])
    assert "index" in str(exc.value)
    assert "TypeError" not in str(exc.value)

    # The availability check reports the same bad answer, not a connectivity guess.
    message = e.check_available()
    assert message is not None
    assert "index" in message
    assert "TypeError" not in message


def test_ollama_valid_response_passes_shape_validation(monkeypatch):
    """The happy path: the adapter returns the service's vector as floats."""
    mgr = MagicMock()
    mgr.generate_embedding.return_value = [1.0, 0.5]
    _patched_manager(monkeypatch, mgr)
    e = emb.OllamaEmbedder(host="http://box:11434")
    assert e.generate_embedding("x") == [1.0, 0.5]


def test_ollama_check_available_returns_none_when_the_service_answers(monkeypatch):
    """A service that returns a usable vector is available."""
    mgr = MagicMock()
    mgr.generate_embedding.return_value = [1.0, 0.5]
    _patched_manager(monkeypatch, mgr)
    e = emb.OllamaEmbedder(host="http://box:11434")
    assert e.check_available() is None


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
    failure = ConnectionError("connection refused")
    mgr.generate_embedding.side_effect = failure
    _patched_manager(monkeypatch, mgr)
    e = emb.OllamaEmbedder(host="http://box:11434")
    with pytest.raises(RuntimeError) as exc:
        e.generate_embedding("x")
    msg = str(exc.value)
    assert "could not reach" in msg
    assert "connection refused" in msg
    assert "local" not in msg
    # A connection failure has no body, so the cause is safe to print.
    assert exc.value.__cause__ is failure


def test_a_bad_answer_stays_a_bad_answer(monkeypatch):
    """A caller can tell a service that answered badly from one that did not answer."""
    post = MagicMock(return_value=_http_ok({"data": [{"index": 0, "embedding": None}]}))
    monkeypatch.setattr(emb.requests, "post", post)
    e = emb.OpenAIEmbedder(host="http://h:1/v1", model="m")
    with pytest.raises(emb.EmbeddingResponseError) as exc:
        e.generate_embedding("x")
    assert "instead of a vector" in str(exc.value)
    assert "could not reach" not in str(exc.value)

    r = MagicMock()
    r.status_code = 503
    r.text = "model not loaded"
    monkeypatch.setattr(emb.requests, "post", MagicMock(return_value=r))
    with pytest.raises(RuntimeError) as exc:
        e.generate_embedding("x")
    assert not isinstance(exc.value, emb.EmbeddingResponseError)
    assert "returned HTTP 503" in str(exc.value)


def test_an_answer_that_is_not_a_known_failure_kind_stays_undiagnosed(monkeypatch):
    """A failure that is neither a status nor a connection error is not called unreachable.

    The ollama SDK uses httpx, whose timeout is not an OSError, so the message
    lists both checks instead of guessing which one it was. The SDK's own error
    defaults `status_code` to -1, which is not a status, so the guard is pinned
    here: its text is the body and must not be shown.
    """
    mgr = MagicMock()
    mgr.generate_embedding.side_effect = _StatusError("sk-secret in the body", -1)
    _patched_manager(monkeypatch, mgr)
    e = emb.OllamaEmbedder(host="http://box:11434")
    with pytest.raises(RuntimeError) as exc:
        e.generate_embedding("x")
    msg = str(exc.value)
    assert "did not return an embedding" in msg
    assert "could not reach" not in msg
    # A printed traceback shows the cause, so a body-bearing one is dropped.
    assert exc.value.__cause__ is None
    assert "sk-secret" not in msg
    # The two kinds of failure stay distinguishable: this one is not a bad answer.
    assert not isinstance(exc.value, emb.EmbeddingResponseError)


def test_httpx_timeout_is_not_reported_as_a_bad_answer(monkeypatch):
    """The most likely real failure keeps both checks in its message."""
    import httpx

    mgr = MagicMock()
    mgr.generate_embedding.side_effect = httpx.ReadTimeout("timed out")
    _patched_manager(monkeypatch, mgr)
    e = emb.OllamaEmbedder(host="http://box:11434")
    with pytest.raises(RuntimeError) as exc:
        e.generate_embedding("x")
    msg = str(exc.value)
    assert "did not return an embedding" in msg
    assert "reachable" in msg


@pytest.mark.parametrize(
    "failure,expected",
    [
        (_StatusError("server said sk-secret", 500), "returned HTTP 500"),
        (ConnectionError("connection refused"), "could not reach"),
    ],
)
def test_check_available_returns_a_message_for_any_failure(
    monkeypatch, failure, expected
):
    """A caller asking whether a service answers should not have to catch."""
    mgr = MagicMock()
    mgr.generate_embedding.side_effect = failure
    _patched_manager(monkeypatch, mgr)
    e = emb.OllamaEmbedder(host="http://box:11434")
    message = e.check_available()
    assert message is not None
    assert "http://box:11434" in message
    assert expected in message
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


def test_probe_memo_distinguishes_metadata_tables(tmp_path):
    """The same file/config/model/probe in two spaces must not share a memo entry."""
    db_path = tmp_path / "two-spaces.duckdb"
    conn = duckdb.connect(str(db_path))
    probe = json.dumps([1.0, 0.0, 0.0])
    for table in ("corpus_meta", "semantic_meta"):
        conn.execute(
            f"CREATE TABLE {table} "
            "(embedding_model VARCHAR, dim INTEGER, probe_vec JSON)"
        )
        conn.execute(
            f"INSERT INTO {table} VALUES (?, ?, ?)",
            ["qwen3-embedding-0.6b-q8_0", 3, probe],
        )
    signature = ("ollama", "http://host:11434", "model", "key")

    corpus = emb.probe_memo_key(conn, "corpus_meta", db_path, config_sig=signature)
    semantic = emb.probe_memo_key(
        conn, "semantic_meta", db_path, config_sig=signature
    )

    assert corpus != semantic
    assert corpus == emb.probe_memo_key(
        conn, "corpus_meta", db_path, config_sig=signature
    )
    conn.close()
