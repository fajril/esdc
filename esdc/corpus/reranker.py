# esdc/corpus/reranker.py
"""Reranking through the configured HTTP service.

`get_reranker()` builds the client from the saved provider reference and the
explicit model id. Nothing is loaded in-process: a stored
`corpus.rerank_backend: local` value is retired, so it warns once per process
per config signature and returns None, leaving search in RRF order. The stored
value is reported, never rewritten.

Nothing here raises. An inactive or invalid selection warns and returns None,
and reranking itself stays gated by `corpus.rerank` at the call site — this
factory never enables it. A service score is a ranking signal, not a calibrated
probability.

Why the retired local model was not kept as the default (189-query eval,
2026-08-06): no gain on the dominant lookup class, a consistent loss on
cross_reference, and it scored template match rather than entity identity —
"persetujuan POFD Lapangan Volve" against "Persetujuan POFD Lapangan Securai"
scored 0.9999, with max(negative) above min(positive), so no threshold separates
them. That is why esdc/corpus/evaluate.py refuses to score the negative class
and why `corpus.negative_floor` was removed rather than tuned: a confidently
ranked result for a question the corpus cannot answer carries no signal that
nothing relevant exists.
"""

from __future__ import annotations

import logging
import threading

from esdc.configs import Config
from esdc.corpus.provider_reranker import ProviderReranker, RerankClient

logger = logging.getLogger(__name__)

# D8: one retired-local warning per process per config signature, so a chat
# server does not spam the log on every search. Kept apart from the probe memo.
_WARNED_RERANK_SIGNATURES: set[tuple[str, str, str]] = set()
_warn_lock = threading.Lock()


def _warn_retired_local(provider: str, model: str) -> None:
    """Warn once per process per config signature; never rewrite config."""
    signature = ("local", provider, model)
    with _warn_lock:
        if signature in _WARNED_RERANK_SIGNATURES:
            return
        _WARNED_RERANK_SIGNATURES.add(signature)
    logger.warning(
        "[Corpus] rerank backend 'local' is retired: in-process reranking was "
        "removed in this version. Keeping RRF order. Set corpus.rerank_backend "
        "to 'provider' and corpus.rerank_provider to an OpenAI-compatible "
        "/v1/rerank service."
    )


def get_reranker() -> RerankClient | None:
    """Return the reranker for the active corpus config, or None.

    `provider` (the default) builds the HTTP client from the saved reference and
    the explicit model. A stored `local` value is retired: it warns once per
    process per config signature and returns None, so search keeps its RRF
    order. Nothing here raises: an inactive or invalid selection warns and
    returns None. Reranking itself stays gated by `corpus.rerank` at the call
    site — this factory never enables it.
    """
    cfg = Config.get_corpus_config()
    backend = str(cfg.get("rerank_backend") or "provider").strip().lower()
    name = str(cfg.get("rerank_provider") or "").strip()
    model = str(cfg.get("rerank_model") or "").strip()

    if backend == "local":
        _warn_retired_local(name, model)
        return None

    if backend != "provider":
        logger.warning(
            "[Corpus] rerank backend %s is not supported; keeping RRF order",
            backend or "unknown",
        )
        return None

    if not name:
        logger.warning(
            "[Corpus] rerank provider is not configured; keeping RRF order"
        )
        return None
    if not model:
        logger.warning(
            "[Corpus] rerank model is not configured for provider '%s'; "
            "keeping RRF order",
            name,
        )
        return None

    try:
        provider = Config.get_referenced_provider(name)
    except ValueError:
        logger.warning(
            "[Corpus] rerank provider '%s' is not configured; keeping RRF order",
            name,
        )
        return None

    provider_type = str(
        provider.get("provider_type") or provider.get("type") or ""
    ).strip().lower()
    if provider_type != "openai_compatible":
        logger.warning(
            "[Corpus] rerank is not supported for provider type %s (%s); "
            "keeping RRF order",
            provider_type or "unknown",
            name,
        )
        return None

    if not str(provider.get("base_url") or "").strip():
        logger.warning(
            "[Corpus] rerank provider '%s' has no base URL; keeping RRF order",
            name,
        )
        return None

    return ProviderReranker(provider, model)
