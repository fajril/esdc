"""Reranker unit tests — never load a real cross-encoder."""

import pytest

import esdc.corpus.reranker as reranker_mod
from esdc.corpus.reranker import Reranker


@pytest.fixture(autouse=True)
def reset_singleton():
    Reranker._instance = None
    Reranker._failed = False
    yield
    Reranker._instance = None
    Reranker._failed = False


class FakeEncoder:
    def rerank(self, query, texts):
        # reverse order: last text scores highest
        return list(range(len(texts)))


def test_get_returns_singleton(monkeypatch):
    monkeypatch.setattr(reranker_mod, "_load_encoder", lambda: FakeEncoder())
    a, b = Reranker.get(), Reranker.get()
    assert a is b is not None


def test_get_returns_none_and_remembers_failure(monkeypatch):
    calls = []

    def boom():
        calls.append(1)
        raise RuntimeError("no model")

    monkeypatch.setattr(reranker_mod, "_load_encoder", boom)
    assert Reranker.get() is None
    assert Reranker.get() is None
    assert len(calls) == 1  # failure cached, no reload storm


def test_rerank_returns_float_scores(monkeypatch):
    monkeypatch.setattr(reranker_mod, "_load_encoder", lambda: FakeEncoder())
    rr = Reranker.get()
    scores = rr.rerank("q", ["a", "b", "c"])
    assert scores == [0.0, 1.0, 2.0]
