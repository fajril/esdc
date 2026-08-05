"""Text helpers for handling raw LLM responses.

Deliberately stdlib-only and dependency-free so every subpackage can import
it without creating a cycle.
"""

from __future__ import annotations

import re

# Reasoning models (Qwen3 with thinking enabled, DeepSeek-R1) embed thinking
# blocks directly in the message content. Strip them before any parsing so
# neither the tags nor the reasoning prose reach a JSON slice or a title.
# Open and close are matched independently (no backreference): models are not
# consistent about which spelling they close with, and a mixed
# `<thinking>…</think>` must still be treated as one balanced block rather
# than truncated reasoning. `\b[^>]*>` tolerates attributes and newlines in
# the tag itself.
_THINKING_BLOCK_RE = re.compile(
    r"<(?:thinking|think)\b[^>]*>.*?</(?:thinking|think)\b[^>]*>",
    re.DOTALL | re.IGNORECASE,
)
_UNCLOSED_THINKING_RE = re.compile(
    r"<(?:thinking|think)\b[^>]*>.*\Z",
    re.DOTALL | re.IGNORECASE,
)
_THINKING_TAG_RE = re.compile(r"</?(?:thinking|think)\b[^>]*>", re.IGNORECASE)


def strip_thinking_tags(text: str) -> str:
    """Remove reasoning blocks (``<thinking>…</thinking>``) from LLM content.

    Balanced blocks go first. An opening tag still standing afterwards means
    the response was cut off mid-reasoning (token limit), so everything from
    it to the end is reasoning: drop it rather than let the prose reach the
    parser. The last pass clears any stray closing tag left with no opener.

    Returns ``text`` unchanged when it contains no thinking tags.
    """
    text = _THINKING_BLOCK_RE.sub("", text)
    text = _UNCLOSED_THINKING_RE.sub("", text)
    return _THINKING_TAG_RE.sub("", text)
