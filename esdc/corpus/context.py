"""Deterministic chunk context: metadata prefix baked into embed/FTS text.

Contextual-retrieval variant without an LLM: sidecar frontmatter already
names the doc type, subject, and entities a query would mention, so the
prefix is assembled from the document row. chunk_text is never modified —
the prefix lives only in the derived embed_text column.
"""

from __future__ import annotations

import json
from typing import Any


def _as_list(val: Any) -> list[str]:
    """Coerce None / str / JSON-string-array / list into a list of strings."""
    if val is None:
        return []
    if isinstance(val, str):
        stripped = val.strip()
        if not stripped:
            return []
        if stripped.startswith("["):
            try:
                val = json.loads(stripped)
            except (ValueError, TypeError):
                return [stripped]
        else:
            return [stripped]
    if isinstance(val, list):
        return [str(v) for v in val if v]
    return [str(val)]


def build_context_prefix(doc: dict[str, Any]) -> str:
    """One line of document context: type | topics | subject | entities."""
    parts: list[str] = []
    if doc.get("doc_type"):
        parts.append(str(doc["doc_type"]))
    topics = _as_list(doc.get("doc_topic"))
    if topics:
        parts.append(", ".join(topics))
    subject = str(doc.get("subject") or "").strip()
    if subject:
        parts.append(subject)
    entities = (
        _as_list(doc.get("wk_name"))
        + _as_list(doc.get("field_name"))
        + _as_list(doc.get("project_name"))
        + _as_list(doc.get("pod_name"))
    )
    if entities:
        parts.append(", ".join(dict.fromkeys(entities)))
    return " | ".join(parts)


def build_embed_text(prefix: str, section: str | None, chunk_text: str) -> str:
    """Text that gets embedded and FTS-indexed; chunk_text stays for display."""
    header = " | ".join(p for p in (prefix, section) if p)
    return f"{header}\n\n{chunk_text}" if header else chunk_text
