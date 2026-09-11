# tests/test_embed_parity_integration.py
"""Opt-in: prove a real remote backend shares the local cosine space.

Skipped unless ESDC_EMBEDDING_HOST is set. This is the procedure for
qualifying a new backend before pointing a corpus at it.

    ESDC_EMBEDDING_HOST=http://localhost:8889/v1 \
    ESDC_EMBEDDING_BACKEND=openai \
    ESDC_EMBEDDING_MODEL=Qwen3-Embedding-0.6B-8bit \
    .venv/bin/pytest tests/test_embed_parity_integration.py -v
"""

from __future__ import annotations

import os

import pytest

from esdc.embedders import (
    PROBE_TEXT,
    PROBE_TOLERANCE,
    InternalEmbedder,
    cosine,
    get_build_embedder,
)

pytestmark = pytest.mark.skipif(
    not os.environ.get("ESDC_EMBEDDING_HOST"),
    reason="set ESDC_EMBEDDING_HOST to run cross-backend parity checks",
)

TEXTS = [
    PROBE_TEXT,
    "Persetujuan POD I Lapangan Banyu Urip dengan cadangan minyak 450 MMSTB.",
    "The reservoir simulation model was history-matched against production data.",
]


def test_remote_backend_matches_local_space():
    local = InternalEmbedder()
    remote = get_build_embedder()

    lv = local.generate_embeddings_batch(TEXTS)
    rv = remote.generate_embeddings_batch(TEXTS)

    assert len(lv) == len(rv) == len(TEXTS)
    assert len(lv[0]) == len(rv[0]), "dimension mismatch between backends"

    for text, a, b in zip(TEXTS, lv, rv, strict=True):
        sim = cosine(a, b)
        assert sim >= PROBE_TOLERANCE, f"cosine {sim:.6f} too low for {text!r}"


def test_batch_order_is_preserved():
    """A backend that mispairs vectors would pass the parity test above."""
    remote = get_build_embedder()
    batched = remote.generate_embeddings_batch(TEXTS)
    for text, vec in zip(TEXTS, batched, strict=True):
        single = remote.generate_embedding(text)
        assert cosine(vec, single) >= PROBE_TOLERANCE
