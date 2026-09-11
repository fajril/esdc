# esdc/corpus/embedder.py
"""Backwards-compatible re-export of the shared embedder backends.

The implementation moved to esdc/embedders.py so esdc.search can use it
without importing esdc.corpus. Existing imports keep working.
"""

from __future__ import annotations

from esdc.embedders import (  # noqa: F401
    MODEL_ID,
    InternalEmbedder,
    _get_model,
)

__all__ = ["MODEL_ID", "InternalEmbedder", "_get_model"]
