"""Helpers for surfacing media from agent output in a terminal UI."""

import re

_IMAGE_MD_RE = re.compile(r"!\[[^\]]*\]\(([^)\s]+)\)")


def extract_image_urls(markdown: str) -> list[str]:
    """Return image targets from markdown, in order, deduplicated."""
    seen: set[str] = set()
    urls: list[str] = []
    for match in _IMAGE_MD_RE.finditer(markdown or ""):
        url = match.group(1).strip()
        if url and url not in seen:
            seen.add(url)
            urls.append(url)
    return urls
