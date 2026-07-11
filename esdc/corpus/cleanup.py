"""LLM formatting cleanup for extracted markdown (pre-review).

Runs at extract time, before human review, so any LLM mistake is caught
by the reviewer — commit stays LLM-free. Only *native* page segments are
cleaned: OCR page/image segments were already produced by an LLM. A
cleaned segment replaces the original only when it passes the guards
(no invented digit-runs, length within 0.5x-1.5x, non-empty, no
exception); otherwise the original segment is kept and counted as
rejected.
"""

import re
from collections.abc import Callable

CLEANUP_PROMPT = """You are reformatting one page of an Indonesian oil & gas \
official document that was auto-extracted from PDF to markdown. Fix ONLY formatting:
- correct heading levels (# only for real top-level sections)
- merge lines that were hard-wrapped mid-sentence
- fix tables: one line per row, remove stray <br> word-wraps inside cells
- fix broken emphasis like "s _tandalone_" -> "*standalone*"
- if the same picture-text block appears twice in a row, keep one copy
Do NOT reword, translate, summarize, or add anything. Keep every number, date, \
document number and name EXACTLY as written. Output only the cleaned markdown, \
no commentary.

Page content:
---
{segment}
---
Cleaned markdown:"""

# Same marker family the extractor emits and the chunker strips.
_MARKER_SPLIT_RE = re.compile(
    r"(<!--\s*page\s+\d+(?:\s+image\s+\d+)?:\s*\w+\s*-->\n?)"
)
_NATIVE_MARKER_RE = re.compile(r"<!--\s*page\s+\d+:\s*native(?:_docx|_md)?\s*-->")
_DIGIT_RUN_RE = re.compile(r"\d+")
_CODE_FENCE_RE = re.compile(r"^```[a-z]*\n|\n?```\s*$")


def _guard_ok(original: str, cleaned: str) -> bool:
    """Accept a cleaned segment only if it looks like a safe reformat."""
    if not cleaned.strip():
        return False
    ratio = len(cleaned) / max(len(original), 1)
    if not 0.5 <= ratio <= 1.5:
        return False
    invented = set(_DIGIT_RUN_RE.findall(cleaned)) - set(
        _DIGIT_RUN_RE.findall(original)
    )
    return not invented


def cleanup_markdown(
    markdown: str, caller: Callable[[str], str]
) -> tuple[str, int, int]:
    """Reformat native page segments via `caller`; markers pass through.

    Returns (cleaned_markdown, segments_cleaned, segments_rejected).
    """
    pieces = _MARKER_SPLIT_RE.split(markdown)
    out: list[str] = []
    n_cleaned = 0
    n_rejected = 0
    last_marker = ""
    for piece in pieces:
        if _MARKER_SPLIT_RE.fullmatch(piece):
            last_marker = piece
            out.append(piece)
            continue
        if not piece.strip() or not _NATIVE_MARKER_RE.search(last_marker):
            out.append(piece)
            continue
        try:
            cleaned = _CODE_FENCE_RE.sub(
                "", caller(CLEANUP_PROMPT.format(segment=piece)).strip()
            ).strip()
        except Exception:
            n_rejected += 1
            out.append(piece)
            continue
        if _guard_ok(piece, cleaned):
            out.append(cleaned + "\n\n")
            n_cleaned += 1
        else:
            n_rejected += 1
            out.append(piece)
    return "".join(out), n_cleaned, n_rejected
