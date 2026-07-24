"""Optional local reranker: fastembed cross-encoder, in-process.

Query-time second stage: RRF produces candidates, the cross-encoder
scores (query, chunk) pairs jointly and reorders the top pool. Always
optional — any load or scoring failure falls back to RRF order.

The reranker model is configurable (`corpus.rerank_model`) because, unlike
embeddings, it produces no stored artifacts — swapping it changes only
runtime scoring, never the schema. The default is fastembed-native; a name
in `_CUSTOM_RERANKERS` (e.g. the Apache-2.0, Indonesian-capable
`BAAI/bge-reranker-v2-m3`) is registered with fastembed on demand via
`add_custom_model`, so no torch/sentence-transformers is ever pulled in.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Fastembed-native default. jina-v2 is CC-BY-NC-4.0 (non-commercial); for
# commercial/Indonesian use, set corpus.rerank_model to BAAI/bge-reranker-v2-m3
# below (Apache-2.0), registered on demand from a fastembed-compatible ONNX.
DEFAULT_RERANKER = "jinaai/jina-reranker-v2-base-multilingual"

# Rerankers not in fastembed's built-in list, registered lazily via
# TextCrossEncoder.add_custom_model. Keyed by the value a user puts in
# corpus.rerank_model. The ONNX repo must expose a single onnx/model.onnx
# plus root tokenizer.json/config.json (the fastembed layout).
_CUSTOM_RERANKERS: dict[str, dict[str, Any]] = {
    "BAAI/bge-reranker-v2-m3": {
        "hf": "onnx-community/bge-reranker-v2-m3-ONNX",
        "model_file": "onnx/model.onnx",
        "size_in_gb": 2.27,
        "license": "apache-2.0",
        "description": (
            "Multilingual cross-encoder (incl. Bahasa Indonesia), "
            "Apache-2.0, built on bge-m3."
        ),
    },
}


def _resolve_model_name() -> str:
    """Reranker model from corpus config, falling back to the default."""
    from esdc.configs import Config

    return Config.get_corpus_config().get("rerank_model") or DEFAULT_RERANKER


def _register_custom(model_name: str) -> None:
    """Register a non-builtin reranker with fastembed if needed.

    Metadata only — no weights download happens here. Unknown names are left
    for TextCrossEncoder to reject at construction, which Reranker.get()
    catches (falling back to RRF order).
    """
    from fastembed.rerank.cross_encoder import TextCrossEncoder

    supported = {m["model"] for m in TextCrossEncoder.list_supported_models()}
    if model_name in supported:
        return
    spec = _CUSTOM_RERANKERS.get(model_name)
    if spec is None:
        logger.warning(
            "[Corpus] rerank_model=%s is neither a fastembed built-in nor a "
            "known custom model; load will likely fall back to RRF order",
            model_name,
        )
        return
    from fastembed.common.model_description import ModelSource

    TextCrossEncoder.add_custom_model(
        model_name,
        sources=ModelSource(hf=spec["hf"]),
        model_file=spec["model_file"],
        description=spec.get("description", ""),
        license=spec.get("license", ""),
        size_in_gb=float(spec.get("size_in_gb", 0.0)),
    )


def _load_encoder() -> Any:
    """Load the configured fastembed cross-encoder (monkeypatch seam)."""
    from fastembed.rerank.cross_encoder import TextCrossEncoder

    from esdc.configs import Config

    model_name = _resolve_model_name()
    _register_custom(model_name)
    # Persist weights under ~/.esdc/models (follows ESDC_CONFIG_DIR), not the
    # volatile system temp dir fastembed defaults to.
    cache_dir = str(Config.get_config_dir() / "models")
    return TextCrossEncoder(model_name=model_name, cache_dir=cache_dir)


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
                    "[Corpus] reranker loaded | model=%s", _resolve_model_name()
                )
            except Exception as e:
                logger.warning(
                    "[Corpus] reranker unavailable, keeping RRF order | "
                    "model=%s error=%s",
                    _resolve_model_name(),
                    e,
                )
                cls._failed = True
                return None
        return cls._instance

    def rerank(self, query: str, texts: list[str]) -> list[float]:
        """Score each text against the query; higher = more relevant."""
        return [float(s) for s in self._encoder.rerank(query, texts)]
