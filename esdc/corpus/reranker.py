# esdc/corpus/reranker.py
"""Optional local reranker: Qwen3-Reranker GGUF via llama.cpp, in-process.

Query-time second stage: RRF produces candidates, this scores
(query, chunk) pairs and reorders the top pool. Always optional — any
load or scoring failure falls back to RRF order.

Reranker output is runtime-only (no stored artifact), so the model is
configurable via corpus.rerank_model without reembedding. Scoring uses
llama.cpp RANK pooling: the pair is wrapped in the official Qwen3
rerank template (the GGUF does NOT bake it — llama.cpp only applies it
in llama-server's /rerank endpoint), and index 0 of the RANK output is
P("yes") — the relevance score.
"""
from __future__ import annotations

import logging
import threading
from typing import Any

from esdc.configs import Config

logger = logging.getLogger(__name__)

# Default reranker GGUF. ggml-org build carries the cls (yes/no) head
# required for llama.cpp RANK pooling — most community conversions lack
# it and score ~0 (llama.cpp#16407).
DEFAULT_RERANKER = "ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF"

# Known reranker GGUFs, keyed by the value a user puts in corpus.rerank_model.
_RERANKER_GGUFS: dict[str, tuple[str, str]] = {
    "ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF": (
        "ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF",
        "qwen3-reranker-0.6b-q8_0.gguf",
    ),
}

# Official Qwen3-Reranker prompt format. Scoring off-template still
# discriminates but is miscalibrated — always wrap pairs with this.
_PROMPT_PREFIX = (
    "<|im_start|>system\nJudge whether the Document meets the requirements "
    "based on the Query and the Instruct provided. Note that the answer can "
    'only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
)
_PROMPT_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
_INSTRUCT = (
    "Given a web search query, retrieve relevant passages that answer the query"
)

# Llama contexts are not thread-safe; rerank scoring serializes on this.
_infer_lock = threading.Lock()


def _rerank_prompt(query: str, doc: str) -> str:
    """Wrap a (query, doc) pair in the official Qwen3 rerank template."""
    return (
        f"{_PROMPT_PREFIX}<Instruct>: {_INSTRUCT}\n"
        f"<Query>: {query}\n<Document>: {doc}{_PROMPT_SUFFIX}"
    )


def _resolve_model_name() -> str:
    """Reranker model id from corpus config, falling back to the default."""
    return Config.get_corpus_config().get("rerank_model") or DEFAULT_RERANKER


def _load_reranker() -> Any:
    """Load the configured reranker GGUF with RANK pooling (monkeypatch seam).

    llama_cpp is imported here, not at module top: a broken
    llama-cpp-python install must be caught by Reranker.get() so search
    degrades to RRF order instead of erroring.
    """
    from llama_cpp import LLAMA_POOLING_TYPE_RANK

    from esdc.corpus.llama_backend import load_llama, resolve_gguf

    name = _resolve_model_name()
    if name in _RERANKER_GGUFS:
        repo, filename = _RERANKER_GGUFS[name]
    elif ":" in name:
        # Custom GGUFs are given as "repo_id:filename.gguf" — guessing a
        # filename for an arbitrary repo would just 404 confusingly.
        repo, filename = name.split(":", 1)
    else:
        logger.warning(
            "[Corpus] rerank_model=%s is not a known GGUF; expected "
            '"repo_id:filename.gguf" — falling back to default', name
        )
        repo, filename = _RERANKER_GGUFS[DEFAULT_RERANKER]
    path = resolve_gguf(repo, filename)
    return load_llama(path, pooling_type=LLAMA_POOLING_TYPE_RANK)


class Reranker:
    """Singleton wrapper; get() returns None when the model can't load."""

    _instance: Reranker | None = None
    _failed: bool = False

    def __init__(self, model: Any) -> None:
        self._model = model

    @classmethod
    def get(cls) -> Reranker | None:
        if cls._failed:
            return None
        if cls._instance is None:
            try:
                cls._instance = cls(_load_reranker())
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
        scores: list[float] = []
        with _infer_lock:
            for t in texts:
                out = self._model.embed(_rerank_prompt(query, t))
                scores.append(float(out[0]))
        return scores
