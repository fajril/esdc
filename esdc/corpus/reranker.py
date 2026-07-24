"""Optional local reranker: fastembed cross-encoder, in-process.

Query-time second stage: RRF produces candidates, the cross-encoder
scores (query, chunk) pairs jointly and reorders the top pool. Always
optional — any load or scoring failure falls back to RRF order.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

PINNED_RERANKER = "jinaai/jina-reranker-v2-base-multilingual"


def _load_encoder() -> Any:
    """Load the fastembed cross-encoder (monkeypatch seam for tests)."""
    from fastembed.rerank.cross_encoder import TextCrossEncoder

    return TextCrossEncoder(model_name=PINNED_RERANKER)


class Reranker:
    """Singleton wrapper; get() returns None when the model can't load."""

    _instance: Reranker | None = None
    _failed: bool = False

    def __init__(self, encoder: Any) -> None:
        self._encoder = encoder

    @classmethod
    def get(cls) -> Reranker | None:
        if cls._failed:
            return None
        if cls._instance is None:
            try:
                cls._instance = cls(_load_encoder())
                logger.info(
                    "[Corpus] reranker loaded | model=%s", PINNED_RERANKER
                )
            except Exception as e:
                logger.warning(
                    "[Corpus] reranker unavailable, keeping RRF order | "
                    "model=%s error=%s",
                    PINNED_RERANKER,
                    e,
                )
                cls._failed = True
                return None
        return cls._instance

    def rerank(self, query: str, texts: list[str]) -> list[float]:
        """Score each text against the query; higher = more relevant."""
        return [float(s) for s in self._encoder.rerank(query, texts)]
