# esdc/corpus/embedder.py
"""Backwards-compatible re-export of the shared embedder backends.

The implementation moved to esdc/embedders.py so esdc.search can use it
without importing esdc.corpus. Existing imports keep working.
"""

from __future__ import annotations

from esdc.embedders import (  # noqa: F401
    MODEL_ID,
    EmbeddingClient,
    InternalEmbedder,
    _get_model,
)

# InternalEmbedder and _get_model stay re-exported until every caller has moved to the
# factory client (Task 5); warm-up and the CLI still import them. EmbeddingClient is the
# shape callers should depend on from now on.
__all__ = ["MODEL_ID", "EmbeddingClient", "InternalEmbedder", "_get_model"]
