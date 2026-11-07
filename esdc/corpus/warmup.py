"""Check the corpus embedding and reranking services.

No model is downloaded or loaded: each role's configured HTTP service is asked
with a synthetic input, so a later query does not discover a dead service.
"""

from __future__ import annotations

from dataclasses import dataclass

from esdc.configs import Config
from esdc.corpus.reranker import get_reranker
from esdc.embedders import MODEL_ID, get_embedder


@dataclass
class WarmupResult:
    component: str  # "embedder" | "reranker"
    model: str
    ok: bool
    detail: str  # "ready" or an error/skip message


def run_warmup(rerank: bool | None = None) -> list[WarmupResult]:
    """Check the embedder (always) and the reranker (per `rerank`/config).

    The embedder role uses the client's availability check. The reranker role
    sends a synthetic query and one synthetic document to the rerank endpoint;
    it never calls chat completions. Nothing here downloads a model.
    """
    results: list[WarmupResult] = []

    model_name = MODEL_ID
    try:
        emb = get_embedder()
        model_name = emb.model
        message = emb.check_available()
    except Exception as e:
        results.append(WarmupResult("embedder", model_name, False, str(e)))
    else:
        results.append(
            WarmupResult("embedder", model_name, message is None, message or "ready")
        )

    cfg = Config.get_corpus_config()
    want = rerank if rerank is not None else bool(cfg.get("rerank", False))
    rerank_model = str(cfg.get("rerank_model") or "").strip() or "(not configured)"

    if not want:
        results.append(
            WarmupResult("reranker", rerank_model, True, "skipped (rerank disabled)")
        )
        return results

    try:
        rr = get_reranker()
    except Exception as e:
        results.append(WarmupResult("reranker", rerank_model, False, str(e)))
        return results

    if rr is None:
        results.append(
            WarmupResult("reranker", rerank_model, False, "unavailable (see logs)")
        )
        return results

    try:
        rr.rerank("warmup", ["warmup"])
    except Exception as e:
        results.append(WarmupResult("reranker", rerank_model, False, str(e)))
    else:
        results.append(WarmupResult("reranker", rerank_model, True, "ready"))

    return results
