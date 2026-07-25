# esdc/corpus/llama_backend.py
"""Shared llama.cpp backend for corpus embeddings + rerank.

Loads pinned Qwen3 GGUFs in-process via llama-cpp-python. Weights are
cached under ~/.esdc/models (HF-cache layout, follows ESDC_CONFIG_DIR),
downloaded once, offline thereafter. No Ollama daemon involved.
"""
from __future__ import annotations

import logging

from huggingface_hub import hf_hub_download
from llama_cpp import Llama

from esdc.configs import Config

logger = logging.getLogger(__name__)

# Embedding GGUF is schema (corpus_meta pins model+dim). Q8_0, 1024-dim.
EMBED_REPO = "Qwen/Qwen3-Embedding-0.6B-GGUF"
EMBED_FILE = "Qwen3-Embedding-0.6B-Q8_0.gguf"

# Reranker GGUF is runtime-only. ggml-org build carries the cls head
# (yes/no classifier) llama.cpp needs for pooling_type=RANK; score =
# index 0 of the output. The rerank prompt template is NOT in the GGUF —
# Reranker.rerank builds it (llama.cpp only applies it server-side).
RERANK_REPO = "ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF"
RERANK_FILE = "qwen3-reranker-0.6b-q8_0.gguf"


def resolve_gguf(repo: str, filename: str) -> str:
    """Local path to a GGUF, downloading into ~/.esdc/models on first use.

    hf_hub_download is cache-first: it reuses the cached file and only
    hits the network when the file is missing, so a warmed model works
    offline.
    """
    cache_dir = str(Config.get_config_dir() / "models")
    return hf_hub_download(repo_id=repo, filename=filename, cache_dir=cache_dir)


def load_llama(
    gguf_path: str, *, pooling_type: int | None = None, n_ctx: int = 4096
) -> Llama:
    """Construct an embedding Llama for a GGUF.

    Always embedding=True (both embed and rerank use the embedding path;
    rerank differs only by pooling_type=RANK). n_gpu_layers comes from
    corpus config (0 = CPU; -1 = offload all layers on a GPU box).

    n_batch/n_ubatch are pinned to n_ctx: Llama.embed() truncates every
    input at n_batch tokens (default 512 — half a corpus chunk), and
    non-causal embedding models must fit each sequence in one physical
    batch (n_ubatch).
    """
    n_gpu_layers = int(Config.get_corpus_config().get("n_gpu_layers", 0))
    kwargs: dict = {
        "model_path": gguf_path,
        "embedding": True,
        "n_ctx": n_ctx,
        "n_batch": n_ctx,
        "n_ubatch": n_ctx,
        "n_gpu_layers": n_gpu_layers,
        "verbose": False,
    }
    if pooling_type is not None:
        kwargs["pooling_type"] = pooling_type
    logger.info(
        "[Corpus] loading GGUF | path=%s pooling=%s n_gpu_layers=%d",
        gguf_path,
        pooling_type,
        n_gpu_layers,
    )
    return Llama(**kwargs)
