"""Shared fixtures for search tests."""

import pytest

import esdc.embedders as embedders


@pytest.fixture(autouse=True)
def _default_embedding_signature(monkeypatch):
    """Search units run against one explicit effective embedding config.

    Probe memoization resolves the effective Embeddings assignment. Tests that
    inject their own client only need a stable signature; the embedder/config
    tests override this fixture locally.
    """
    monkeypatch.setattr(
        embedders,
        "config_signature",
        lambda backend=None: (
            "ollama",
            "http://localhost:11434",
            "qwen3-embedding-0.6b-q8_0",
            "",
        ),
    )
