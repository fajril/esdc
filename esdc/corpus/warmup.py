"""Pre-fetch corpus models (embedder + optional reranker) for offline use.

Consumes the existing embedder/reranker seams to force a model download and
first-inference pass, so a later air-gapped run doesn't hit the network.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class WarmupResult:
    component: str  # "embedder" | "reranker"
    model: str
    ok: bool
    detail: str  # "ready" or an error/skip message


def run_warmup(rerank: bool | None = None) -> list[WarmupResult]:
    """Warm the embedder (always) and the reranker (per `rerank`/config)."""
    results: list[WarmupResult] = []

    from esdc.corpus.embedder import InternalEmbedder

    emb = InternalEmbedder()
    try:
        emb.generate_embedding("warmup")
        results.append(WarmupResult("embedder", emb.model, True, "ready"))
    except Exception as e:
        results.append(WarmupResult("embedder", emb.model, False, str(e)))

    from esdc.configs import Config
    from esdc.corpus.reranker import Reranker, _resolve_model_name

    if rerank is not None:
        want = rerank
    else:
        want = bool(Config.get_corpus_config().get("rerank", False))
    model_name = _resolve_model_name()

    if not want:
        results.append(
            WarmupResult("reranker", model_name, True, "skipped (rerank disabled)")
        )
    else:
        rr = Reranker.get()
        if rr is None:
            results.append(
                WarmupResult("reranker", model_name, False, "load failed (see logs)")
            )
        else:
            try:
                rr.rerank("warmup", ["a", "b"])
                results.append(WarmupResult("reranker", model_name, True, "ready"))
            except Exception as e:
                results.append(WarmupResult("reranker", model_name, False, str(e)))

    return results
