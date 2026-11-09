# tests/test_embed_parity_integration.py
"""Opt-in: prove one real HTTP embedding service is internally consistent.

There is no in-process reference to compare against any more, so a service is
qualified by its own consistency — batch order and per-text agreement — and by
the probe stored in the index, which is the only proof that it serves the model
the stored vectors came from.

Skipped unless ESDC_EMBEDDING_HOST is set. This is the procedure for qualifying
a service before pointing a corpus at it:

    ESDC_EMBEDDING_HOST=http://localhost:8889/v1 \
    ESDC_EMBEDDING_BACKEND=openai \
    ESDC_EMBEDDING_MODEL=Qwen3-Embedding-0.6B-8bit \
    .venv/bin/pytest tests/test_embed_parity_integration.py -v

The first text is PROBE_TEXT, so the run also shows what the probe check would
say about this service.
"""

from __future__ import annotations

import math
import os

import pytest

from esdc.embedders import MODEL_ID, PROBE_TEXT, PROBE_TOLERANCE, cosine, get_embedder

pytestmark = pytest.mark.skipif(
    not os.environ.get("ESDC_EMBEDDING_HOST"),
    reason="set ESDC_EMBEDDING_HOST to run service parity checks",
)

TEXTS = [
    PROBE_TEXT,
    "Persetujuan POD I Lapangan Banyu Urip dengan cadangan minyak 450 MMSTB.",
    "The reservoir simulation model was history-matched against production data.",
]


def test_service_reports_the_canonical_vector_space_identity():
    """D1: the wire name chooses the request; `.model` is the index pin."""
    client = get_embedder()
    assert client.model == MODEL_ID


def test_response_is_usable_as_a_vector_space():
    client = get_embedder()
    vectors = client.generate_embeddings_batch(TEXTS)
    assert len(vectors) == len(TEXTS)
    for text, vector in zip(TEXTS, vectors, strict=True):
        assert vector, f"empty vector for {text!r}"
        assert all(math.isfinite(value) for value in vector), text
        assert any(value != 0.0 for value in vector), f"all-zero vector for {text!r}"


def test_batch_order_is_preserved():
    """A service that mispairs vectors would pass a single-text comparison."""
    client = get_embedder()
    batched = client.generate_embeddings_batch(TEXTS)
    for text, vector in zip(TEXTS, batched, strict=True):
        single = client.generate_embedding(text)
        assert len(single) == len(vector), f"dimension changed for {text!r}"
        sim = cosine(vector, single)
        assert sim >= PROBE_TOLERANCE, f"cosine {sim:.6f} too low for {text!r}"
