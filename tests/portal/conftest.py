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
    """Keep corpus schema setup offline without bypassing its real code path."""

    def _unexpected_model_load() -> None:
        raise AssertionError("portal tests must not load the real embedding model")

    monkeypatch.setattr("esdc.corpus.embedder.InternalEmbedder", _OfflineCorpusEmbedder)
    monkeypatch.setattr(esdc.embedders, "_get_model", _unexpected_model_load)
