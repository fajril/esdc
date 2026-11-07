# esdc/corpus/embedder.py
"""Backwards-compatible re-export of the shared embedding client contract.

The implementation lives in esdc/embedders.py so esdc.search can use it
without importing esdc.corpus. Only the vector-space identity and the client
shape are re-exported here: callers resolve a client through
`esdc.embedders.get_embedder()`, and there is no in-process model behind this
module.
"""

from __future__ import annotations

from esdc.embedders import MODEL_ID, EmbeddingClient  # noqa: F401

__all__ = ["MODEL_ID", "EmbeddingClient"]
