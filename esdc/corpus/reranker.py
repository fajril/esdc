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

MEASURED ON THIS CORPUS (2026-08-06) — READ BEFORE ENABLING corpus.rerank
=========================================================================
189-query eval, Qwen3-Reranker-0.6B-Q8_0, rerank_pool 30, n_gpu_layers -1
(full Metal offload), against the same set with rerank off:

    lookup (n=120)            Pass@1  78.3% -> 78.3%   (identical)
    cross_reference (n=30)    Pass@1  96.7% -> 86.7%   (-10.0 pp)
                              Recall@1/5/10  each -5.0 pp
    thematic (n=20)           Pass@1   0.0% -> 15.0%   (+15.0 pp)
    mean latency                69 ms -> 40 969 ms     (594x)

No gain on the dominant class, a consistent loss on cross_reference (all
four metrics, same direction — not noise), and the one clear win is on the
class whose labels are weakest.

Worse, it scores template match rather than entity identity. Top-1 P("yes")
for 19 queries about Norwegian fields — which an Indonesian upstream corpus
cannot answer — against 12 answerable lookups:

    negatives  min 0.0389  max 0.9999  mean 0.9363   (16 of 19 above 0.97)
    positives  min 0.9968  max 1.0000  mean 0.9995

    "persetujuan POFD Lapangan Volve"  ->  "Persetujuan POFD Lapangan
    Securai"  scores 0.9999

Both strings are POFD approval letters; the field name — carrying the whole
discriminative burden — does not move the score. The two negatives that did
score low were the ones whose *form* differed ("berapa cadangan terbukti
Lapangan Troll", a question rather than a letter title, 0.0389). Because
max(negative) exceeds min(positive), no threshold separates them: this is
why esdc/corpus/evaluate.py refuses to score the negative class and why
corpus.negative_floor was removed rather than tuned.

Consequence for chat: with rerank on, Document Search returns confidently
ranked context for questions the corpus cannot answer, with no signal that
nothing relevant exists.

Two things to try before concluding the model is at fault: _INSTRUCT below
is the stock English web-search instruction while these queries are
Indonesian, and rerank() scores strictly one pair at a time (1.37 s/pair for
a 0.6B model at full offload is anomalous — check n_batch and the serial
loop). Neither has been tested.
"""

from __future__ import annotations

import atexit
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
_INSTRUCT = "Given a web search query, retrieve relevant passages that answer the query"

# Llama contexts are not thread-safe; rerank scoring serializes on this.
_infer_lock = threading.Lock()
# Serializes singleton create/use against the shutdown close. Lock order
# is always _load_lock then _infer_lock (rerank() takes only _infer_lock).
_load_lock = threading.Lock()


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
            '"repo_id:filename.gguf" — falling back to default',
            name,
        )
        repo, filename = _RERANKER_GGUFS[DEFAULT_RERANKER]
    path = resolve_gguf(repo, filename)
    model = load_llama(path, pooling_type=LLAMA_POOLING_TYPE_RANK)
    # Registered here rather than in get() so the tests that monkeypatch
    # this loader never hand a fake model to teardown. See
    # esdc.embedders._close_model for why the free must be explicit.
    atexit.register(Reranker._close)
    return model


class Reranker:
    """Singleton wrapper; get() returns None when the model can't load."""

    _instance: Reranker | None = None
    _failed: bool = False

    def __init__(self, model: Any) -> None:
        self._model = model

    @classmethod
    def get(cls) -> Reranker | None:
        with _load_lock:
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

    @classmethod
    def _close(cls) -> None:
        """Free the reranker before the process exits.

        Same ggml-metal teardown assert as esdc.embedders._close_model,
        including the lock timeouts. `_failed` is set so a get() racing
        the close falls back to RRF order instead of handing out a
        wrapper around freed memory. Lock order is _load_lock then
        _infer_lock, mirroring esdc.embedders.
        """
        if not _load_lock.acquire(timeout=2.0):
            logger.warning("[Corpus] reranker loading at exit; skipping close")
            return
        try:
            inst = cls._instance
            if inst is None:
                return
            if not _infer_lock.acquire(timeout=2.0):
                logger.warning("[Corpus] reranker busy at exit; skipping close")
                return
            try:
                cls._instance = None
                cls._failed = True
                inst._model.close()
            finally:
                _infer_lock.release()
        finally:
            _load_lock.release()

    def rerank(self, query: str, texts: list[str]) -> list[float]:
        """Score each text against the query; higher = more relevant."""
        scores: list[float] = []
        with _infer_lock:
            if Reranker._failed:
                # get() handed this instance out before _close() freed the
                # model behind it. Only reachable at shutdown.
                raise RuntimeError("reranker closed at interpreter shutdown")
            for t in texts:
                out = self._model.embed(_rerank_prompt(query, t))
                scores.append(float(out[0]))
        return scores
