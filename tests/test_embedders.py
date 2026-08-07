# tests/test_embedders.py
"""Shared embedder backends: internal (llama.cpp), Ollama, OpenAI-compatible."""

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


def _patched_manager(monkeypatch, mgr):
    """Patch the lazily-imported EmbeddingManager at its source module."""
    import esdc.search.embedding_manager as em

    monkeypatch.setattr(em, "EmbeddingManager", MagicMock(return_value=mgr))
    return em.EmbeddingManager


def test_ollama_reports_model_id_not_wire_tag(monkeypatch):
    mgr = MagicMock()
    _patched_manager(monkeypatch, mgr)
    e = emb.OllamaEmbedder(host="http://box:11434")
    assert e.model == emb.MODEL_ID
    assert e.model != e.OLLAMA_TAG


def test_ollama_pins_wire_tag_ignoring_config(monkeypatch):
    mgr = MagicMock()
    factory = _patched_manager(monkeypatch, mgr)
    emb.OllamaEmbedder(host=None)
    # model= passed explicitly, so EmbeddingManager never falls back to the
    # embedding_model config key.
    assert factory.call_args.kwargs["model"] == "qwen3-embedding:0.6b"


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


def test_ollama_error_names_host_and_local_escape_hatch(monkeypatch):
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
    assert "embedding_backend: local" in msg


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
    assert "503" in str(exc.value)
    assert "http://h:1/v1/embeddings" in str(exc.value)


def test_openai_transport_error_names_local_escape_hatch(monkeypatch):
    import requests as rq

    monkeypatch.setattr(
        emb.requests, "post", MagicMock(side_effect=rq.ConnectionError("refused"))
    )
    e = emb.OpenAIEmbedder(host="http://h:1/v1", model="m")
    with pytest.raises(RuntimeError, match="embedding_backend: local"):
        e.generate_embedding("x")


def test_openai_reports_model_id(monkeypatch):
    e = emb.OpenAIEmbedder(host="http://h:1/v1", model="Qwen3-Embedding-0.6B-8bit")
    assert e.model == emb.MODEL_ID
    assert e.wire_model == "Qwen3-Embedding-0.6B-8bit"


def test_factory_local_returns_internal(monkeypatch):
    assert isinstance(emb.get_build_embedder("local"), emb.InternalEmbedder)


def test_factory_ollama_uses_configured_host(monkeypatch):
    mgr = MagicMock()
    _patched_manager(monkeypatch, mgr)
    from esdc.configs import Config

    monkeypatch.setattr(
        Config, "get_embedding_host", classmethod(lambda cls: "http://box:11434")
    )
    e = emb.get_build_embedder("ollama")
    assert isinstance(e, emb.OllamaEmbedder)
    assert e.host == "http://box:11434"


def test_factory_openai_passes_model_and_key(monkeypatch):
    from esdc.configs import Config

    monkeypatch.setattr(
        Config, "get_embedding_host", classmethod(lambda cls: "http://h:8889/v1")
    )
    monkeypatch.setattr(Config, "get_embedding_model", classmethod(lambda cls: "M"))
    monkeypatch.setattr(Config, "get_embedding_api_key", classmethod(lambda cls: "K"))
    e = emb.get_build_embedder("openai")
    assert isinstance(e, emb.OpenAIEmbedder)
    assert e.wire_model == "M"
    assert e.url == "http://h:8889/v1/embeddings"


def test_factory_none_falls_through_to_config(monkeypatch):
    from esdc.configs import Config

    monkeypatch.setattr(
        Config, "get_embedding_backend", classmethod(lambda cls: "local")
    )
    assert isinstance(emb.get_build_embedder(None), emb.InternalEmbedder)


def test_factory_explicit_arg_beats_config(monkeypatch):
    from esdc.configs import Config

    monkeypatch.setattr(
        Config, "get_embedding_backend", classmethod(lambda cls: "ollama")
    )
    assert isinstance(emb.get_build_embedder("local"), emb.InternalEmbedder)


def test_factory_unknown_backend_lists_valid_ones():
    with pytest.raises(ValueError) as exc:
        emb.get_build_embedder("lmstudio")
    msg = str(exc.value)
    assert "lmstudio" in msg
    for name in ("local", "ollama", "openai"):
        assert name in msg


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
    stored = conn.execute("SELECT probe_vec FROM m").fetchone()[0]
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
