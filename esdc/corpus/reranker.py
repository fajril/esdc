# esdc/corpus/reranker.py
"""Reranking through the canonical Reranking task.

`get_reranker()` builds the existing HTTP client from the resolved assignment:
the assigned OpenAI-compatible connection and its explicit wire model. Nothing
is loaded in-process. A disabled task returns None by default, so search keeps
its RRF order; an explicit ``force=True`` is used by the eval harness and the
warmup check.

Nothing here raises for an inactive or invalid selection: it warns and returns
None. A service score is a ranking signal, not a calibrated probability.

Why the retired local model was not kept (189-query eval, 2026-08-06): no gain
on the dominant lookup class, a consistent loss on cross_reference, and it
scored template match rather than entity identity — "persetujuan POFD Lapangan
Volve" against "Persetujuan POFD Lapangan Securai" scored 0.9999, with
max(negative) above min(positive), so no threshold separates them. That is why
esdc/corpus/evaluate.py refuses to score the negative class and why
`corpus.negative_floor` was removed rather than tuned: a confidently ranked
result for a question the corpus cannot answer carries no signal that nothing
relevant exists.
"""

from __future__ import annotations

import logging

from esdc.configs import Config
from esdc.corpus.provider_reranker import ProviderReranker, RerankClient
from esdc.model_settings import ModelConfigurationRequired

logger = logging.getLogger(__name__)


def get_reranker(*, force: bool = False) -> RerankClient | None:
    """Return the reranker for the resolved Reranking task, or None.

    The task's independent Enabled switch gates normal runtime use. ``force``
    keeps the existing explicit eval/warmup override without enabling the
    persisted task. Nothing here sends a request.
    """
    try:
        task = Config.resolve_model_task("rerank")
    except ModelConfigurationRequired as exc:
        logger.warning("[Corpus] %s", exc)
        return None

    if task.mode != "model" or task.primary is None:
        return None
    if not force and not task.enabled:
        return None

    provider = task.primary.provider
    provider_type = str(
        provider.get("provider_type") or provider.get("type") or ""
    ).strip().lower()
    if provider_type != "openai_compatible":
        logger.warning(
            "[Corpus] rerank connection %s is not OpenAI-compatible; keeping "
            "RRF order",
            task.primary.connection,
        )
        return None

    if not str(provider.get("base_url") or "").strip():
        logger.warning(
            "[Corpus] rerank connection %s has no base URL; keeping RRF order",
            task.primary.connection,
        )
        return None

    return ProviderReranker(provider, task.primary.model)
