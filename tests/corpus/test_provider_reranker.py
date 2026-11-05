"""Tests for reranking through a saved OpenAI-compatible provider connection.

Every request here is mocked; no test opens a socket, and every credential is
synthetic. No GGUF is loaded and no saved secret is used.
"""

import math
from typing import Any

import pytest
import requests

import esdc.corpus.provider_reranker as provider_reranker
from esdc.corpus.provider_reranker import ProviderReranker

BASE = "http://engine:4000/v1"
RERANK_URL = "http://engine:4000/v1/rerank"

# Synthetic saved provider. Its `model` is the Chat selection and must never be
# used as the rerank model.
PROVIDER = {
    "name": "engine",
    "provider_type": "openai_compatible",
    "base_url": BASE,
    "api_key": "fixture-key",
    "model": "chat-a",
}

TEXTS = ["first chunk", "second chunk"]


class _FakeResponse:
    """Stand-in for a requests.Response, so no connection is ever made."""

    def __init__(self, status: int, payload: object) -> None:
        self.status_code = status
        self._payload = payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            error = requests.exceptions.HTTPError(f"HTTP {self.status_code}")
            error.response = self
            raise error

    def json(self) -> object:
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


def _fake_http(monkeypatch, route: Any = None) -> list[dict]:
    """Route the mocked POST and record each call.

    With no route the fake answers with a matching complete result set, so a
    test that only inspects the request is not disturbed by the shape check.
    """
    calls: list[dict] = []

    def fake_post(url, headers=None, json=None, timeout=None):
        calls.append(
            {"url": url, "headers": dict(headers or {}), "json": json, "timeout": timeout}
        )
        if isinstance(route, Exception):
            raise route
        if route is None:
            documents = (json or {}).get("documents") or []
            return _FakeResponse(
                200,
                {
                    "results": [
                        {"index": i, "relevance_score": 0.5} for i in range(len(documents))
                    ]
                },
            )
        status, payload = route
        return _FakeResponse(status, payload)

    monkeypatch.setattr(provider_reranker.requests, "post", fake_post)
    return calls


def _reranker() -> ProviderReranker:
    return ProviderReranker(PROVIDER, "d1-2b")


def test_requires_a_base_url_and_a_model():
    with pytest.raises(ValueError, match="base URL"):
        ProviderReranker({"api_key": "fixture-key"}, "d1-2b")
    with pytest.raises(ValueError, match="model"):
        ProviderReranker(PROVIDER, "")


def test_request_shape_is_the_rerank_contract(monkeypatch):
    calls = _fake_http(monkeypatch)

    _reranker().rerank("cadangan?", TEXTS)

    call = calls[0]
    assert call["url"] == RERANK_URL
    assert call["json"] == {
        "model": "d1-2b",
        "query": "cadangan?",
        "documents": TEXTS,
        "top_n": 2,
        "return_documents": False,
    }
    assert call["timeout"] == (5.0, 30.0)
    assert call["headers"]["Content-Type"] == "application/json"
    assert call["headers"]["Authorization"] == "Bearer fixture-key"


def test_saved_address_prefix_is_preserved(monkeypatch):
    calls = _fake_http(monkeypatch)
    ProviderReranker({"base_url": "http://engine:4000/proxy/v1"}, "d1-2b").rerank(
        "q", ["a"]
    )
    assert calls[0]["url"] == "http://engine:4000/proxy/v1/rerank"


def test_empty_input_makes_no_request(monkeypatch):
    calls = _fake_http(monkeypatch)
    assert _reranker().rerank("cadangan?", []) == []
    assert calls == []


def test_scores_are_returned_in_input_order(monkeypatch):
    """The response is in rank order; the caller gets input order back."""
    _fake_http(
        monkeypatch,
        (200, {"results": [{"index": 1, "relevance_score": 0.9},
                           {"index": 0, "relevance_score": 0.2}]}),
    )

    scores = _reranker().rerank("cadangan?", TEXTS)
    assert scores == [0.2, 0.9]


@pytest.mark.parametrize(
    "payload",
    [
        {"results": [{"index": 0, "relevance_score": 0.5}]},  # one result missing
        {
            "results": [
                {"index": 0, "relevance_score": 0.5},
                {"index": 0, "relevance_score": 0.6},
            ]
        },  # duplicate index
        {
            "results": [
                {"index": 0, "relevance_score": 0.5},
                {"index": 2, "relevance_score": 0.6},
            ]
        },  # out of range
        {
            "results": [
                {"index": True, "relevance_score": 0.5},
                {"index": 1, "relevance_score": 0.6},
            ]
        },  # boolean index
        {
            "results": [
                {"index": 0.0, "relevance_score": 0.5},
                {"index": 1, "relevance_score": 0.6},
            ]
        },  # non-integer index
        {
            "results": [
                {"index": 0, "relevance_score": True},
                {"index": 1, "relevance_score": 0.6},
            ]
        },  # boolean score
        {
            "results": [
                {"index": 0, "relevance_score": "0.5"},
                {"index": 1, "relevance_score": 0.6},
            ]
        },  # nonnumeric score
        {
            "results": [
                {"index": 0, "relevance_score": float("nan")},
                {"index": 1, "relevance_score": 0.6},
            ]
        },  # NaN
        {
            "results": [
                {"index": 0, "relevance_score": float("inf")},
                {"index": 1, "relevance_score": 0.6},
            ]
        },  # Infinity
        {"results": [{"index": 0}, {"index": 1, "relevance_score": 0.6}]},  # no score
        {"results": ["not an object", {"index": 1, "relevance_score": 0.6}]},
        {"data": []},  # no results field
        [],  # results is not a mapping
    ],
)
def test_incomplete_or_invalid_results_reject_the_whole_response(monkeypatch, payload):
    calls = _fake_http(monkeypatch, (200, payload))

    with pytest.raises((RuntimeError, ValueError)):
        _reranker().rerank("cadangan?", TEXTS)

    # the whole response is rejected: no partial scores, and one attempt only
    assert len(calls) == 1
    assert calls[0]["json"]["top_n"] == len(TEXTS)


@pytest.mark.parametrize(
    ("route", "phrase"),
    [
        ((500, {}), "HTTP 500"),
        ((401, {}), "HTTP 401"),
        (requests.exceptions.Timeout("read timed out"), "timeout"),
        (requests.exceptions.ConnectionError("connection refused"), "connection error"),
        ((200, ValueError("Unexpected EOF")), "invalid JSON"),
    ],
)
def test_transport_failures_raise_a_short_reason(monkeypatch, route, phrase):
    calls = _fake_http(monkeypatch, route)
    with pytest.raises(RuntimeError) as exc_info:
        _reranker().rerank("cadangan?", TEXTS)

    assert phrase in str(exc_info.value)
    # no credential, and no response body, reaches the message
    assert "fixture-key" not in str(exc_info.value)
    assert len(calls) == 1


def test_one_attempt_with_no_retry(monkeypatch):
    calls = _fake_http(monkeypatch, requests.exceptions.ConnectionError("refused"))
    with pytest.raises(RuntimeError):
        _reranker().rerank("cadangan?", TEXTS)
    assert len(calls) == 1


def test_no_authorization_header_when_the_credential_is_empty(monkeypatch):
    calls = _fake_http(monkeypatch, (200, {"results": [{"index": 0, "relevance_score": 0.1}]}))
    ProviderReranker({**PROVIDER, "api_key": ""}, "d1-2b").rerank("q", ["a"])
    assert calls[0]["headers"] == {"Content-Type": "application/json"}


def test_oauth_token_is_used_when_no_api_key(monkeypatch):
    calls = _fake_http(monkeypatch, (200, {"results": [{"index": 0, "relevance_score": 0.1}]}))
    provider = {**PROVIDER, "api_key": "", "oauth": {"access_token": "tok"}}
    ProviderReranker(provider, "d1-2b").rerank("q", ["a"])
    assert calls[0]["headers"]["Authorization"] == "Bearer tok"


def test_returned_scores_are_plain_floats(monkeypatch):
    """Store compares them numerically; an integer score from a server is a float."""
    _fake_http(
        monkeypatch,
        (
            200,
            {
                "results": [
                    {"index": 1, "relevance_score": 1},
                    {"index": 0, "relevance_score": 0.5},
                ]
            },
        ),
    )
    scores = _reranker().rerank("q", ["a", "b"])
    assert scores == [0.5, 1.0]
    assert all(isinstance(s, float) and not isinstance(s, bool) for s in scores)
    assert all(math.isfinite(s) for s in scores)
