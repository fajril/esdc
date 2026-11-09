"""Scoped offline dependencies for portal corpus tests."""

from __future__ import annotations

import pytest

import esdc.embedders


class _OfflineCorpusEmbedder:
    model = esdc.embedders.MODEL_ID

    def generate_embedding(self, text: str) -> list[float]:
        return [1.0, 0.0, 0.0]

    def generate_embeddings_batch(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0, 0.0] for _ in texts]


@pytest.fixture
def offline_corpus_embedder(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep corpus schema setup offline without bypassing its real code path.

    Repointed for Task 2: the portal no longer talks to an in-process model, it talks
    to the client the factory resolves. The guard is therefore that nothing here may
    open a real HTTP connection — the offline client stands in for the service, and any
    request that escapes the fixture is a test reaching the network.
    """

    def _no_http_connection(*args, **kwargs):
        raise AssertionError("portal tests must not open a real HTTP connection")

    client = _OfflineCorpusEmbedder()
    monkeypatch.setattr(esdc.embedders, "get_embedder", lambda backend=None: client)
    monkeypatch.setattr(esdc.embedders.requests, "post", _no_http_connection)
