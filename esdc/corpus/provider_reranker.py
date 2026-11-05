# esdc/corpus/provider_reranker.py
"""Reranking through a saved OpenAI-compatible provider connection.

One POST to the normalized `/v1/rerank` endpoint of the saved address, with
the saved credential, the selected model, and `top_n = len(texts)`. One
attempt, no retry, and no tuning of the returned scores.

Server results come back in score order. Scores are put back at their input
`index` before they are returned, so the caller always receives them in the
order it passed. A response that is incomplete, duplicated, out of range, or
nonnumeric is rejected as a whole — never applied partially.

The credential and the response body never appear in a message or a log line.
Local GGUF scores and gateway scores mean different things; nothing here
converts one into the other.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from typing import Any, Protocol

import requests

from esdc.model_catalog import api_endpoint

logger = logging.getLogger(__name__)

# Request bound: 5-second connect, 30-second read, one attempt.
CONNECT_TIMEOUT_SECONDS = 5.0
READ_TIMEOUT_SECONDS = 30.0


class RerankClient(Protocol):
    """One query plus candidate texts in, one score per text out."""

    def rerank(self, query: str, texts: list[str]) -> list[float]: ...


class ProviderReranker:
    """Rerank client for one saved OpenAI-compatible provider and its model."""

    def __init__(self, provider: Mapping[str, Any], model: str) -> None:
        base_url = str(provider.get("base_url") or "").strip()
        model = str(model or "").strip()
        if not base_url:
            raise ValueError("ProviderReranker requires a base URL")
        if not model:
            raise ValueError("ProviderReranker requires a model")

        self.provider = provider
        self.model = model
        self.base_url = base_url

    def rerank(self, query: str, texts: list[str]) -> list[float]:
        """Score every text against the query, in the order the texts were passed.

        Empty input makes no request. Any HTTP, timeout, connection, parse, or
        shape failure raises; the caller keeps the order it already had.
        """
        if not texts:
            return []

        body = {
            "model": self.model,
            "query": query,
            "documents": texts,
            "top_n": len(texts),
            "return_documents": False,
        }

        try:
            response = requests.post(
                api_endpoint(self.base_url, "rerank"),
                headers=self._headers(),
                json=body,
                timeout=(CONNECT_TIMEOUT_SECONDS, READ_TIMEOUT_SECONDS),
            )
            response.raise_for_status()
            payload = response.json()
        except Exception as error:  # one attempt, no retry
            raise RuntimeError(_failure_reason(error)) from error

        return _scores_in_input_order(payload, len(texts))

    def _credential(self) -> str:
        """The saved connection credential, or an empty string (no auth header)."""
        oauth = self.provider.get("oauth")
        token = oauth.get("access_token") if isinstance(oauth, Mapping) else ""
        return str(self.provider.get("api_key") or token or "")

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        credential = self._credential()
        if credential:
            headers["Authorization"] = f"Bearer {credential}"
        return headers


def _scores_in_input_order(payload: Any, count: int) -> list[float]:
    """Validate the whole `results` list, then place each score at its index.

    The response is scored in rank order; returning by index is what restores
    the caller's input order. Anything incomplete or unusable rejects the whole
    response, so no partial set of scores can reach the caller.
    """
    results = payload.get("results") if isinstance(payload, Mapping) else None
    if not isinstance(results, list) or len(results) != count:
        raise ValueError("rerank returned an unexpected response shape")

    scores: list[float] = [0.0] * count
    seen: set[int] = set()

    for item in results:
        if not isinstance(item, Mapping):
            raise ValueError("rerank result is not an object")

        index = item.get("index")
        if isinstance(index, bool) or not isinstance(index, int):
            raise ValueError("rerank result index is not an integer")
        if not 0 <= index < count:
            raise ValueError("rerank result index is out of range")
        if index in seen:
            raise ValueError("rerank result index is duplicated")
        seen.add(index)

        raw_score = item.get("relevance_score")
        if isinstance(raw_score, bool) or not isinstance(raw_score, (int, float)):
            raise ValueError("rerank relevance_score is not numeric")
        score = float(raw_score)
        if not math.isfinite(score):
            raise ValueError("rerank relevance_score is not finite")

        scores[index] = score

    if len(seen) != count:
        raise ValueError("rerank response is missing a result")

    return scores


def _failure_reason(error: Exception) -> str:
    """Short, credential-free reason for a failed request (no response body)."""
    if isinstance(error, requests.exceptions.HTTPError):
        response = getattr(error, "response", None)
        status = getattr(response, "status_code", None)
        return f"HTTP {status}" if status is not None else "HTTP error"
    if isinstance(error, requests.exceptions.Timeout):
        return "timeout"
    if isinstance(error, requests.exceptions.ConnectionError):
        return "connection error"
    if isinstance(error, ValueError):
        return "invalid JSON"
    if isinstance(error, requests.exceptions.RequestException):
        return "request failed"

    return type(error).__name__
