"""Pre-fetch corpus models (embedder + optional reranker) for offline use.

Forces a GGUF download + first-inference pass through the llama.cpp
embedder/reranker seams, so a later air-gapped run doesn't hit the
network.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from esdc.configs import Config
from esdc.corpus.reranker import Reranker, _resolve_model_name


@dataclass
class WarmupResult:
    component: str  # "embedder" | "reranker"
    model: str
    ok: bool
    detail: str  # "ready" or an error/skip message


def _build_embedder() -> Any:
    """Construct the corpus embedder (patchable seam for tests)."""
    from esdc.corpus.embedder import InternalEmbedder

    return InternalEmbedder()


def run_warmup(rerank: bool | None = None) -> list[WarmupResult]:
    """Warm the embedder (always) and the reranker (per `rerank`/config).

    Only the local backend is warmed. A provider backend has no in-process
    model to download or run, so it is reported as skipped: no GGUF load and
    no request, whichever non-local value the config holds.
    """
    results: list[WarmupResult] = []

    emb = _build_embedder()
    try:
        emb.generate_embedding("warmup")
        results.append(WarmupResult("embedder", emb.model, True, "ready"))
    except Exception as e:
        results.append(WarmupResult("embedder", emb.model, False, str(e)))

    cfg = Config.get_corpus_config()
    want = rerank if rerank is not None else bool(cfg.get("rerank", False))
    backend = str(cfg.get("rerank_backend") or "local").strip().lower()
    model_name = _resolve_model_name()

    if not want:
        results.append(
            WarmupResult("reranker", model_name, True, "skipped (rerank disabled)")
        )
        return results

    if backend != "local":
        detail = (
            "skipped (provider backend)"
            if backend == "provider"
            else "skipped (unsupported rerank backend)"
        )
        results.append(WarmupResult("reranker", model_name, True, detail))
        return results

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
