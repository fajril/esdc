"""Heading-aware markdown chunking.

Split on markdown headings first (a MoM's keputusan/action-items live
under their own headings), then enforce chunk_size with overlap.
Page provenance markers are review-time metadata, not content — strip
them so they never pollute embeddings.
"""

import re
from dataclasses import dataclass

HEADING_RE = re.compile(r"^(#{1,4})\s+(.+)$", re.MULTILINE)
PAGE_MARKER_RE = re.compile(r"<!--\s*page\s+\d+(?:\s+image\s+\d+)?:\s*\w+\s*-->\n?")


@dataclass
class Chunk:
    index: int
    section: str | None
    text: str


def _split_sections(markdown: str) -> list[tuple[str | None, str]]:
    matches = list(HEADING_RE.finditer(markdown))
    if not matches:
        return [(None, markdown)]
    sections: list[tuple[str | None, str]] = []
    if matches[0].start() > 0:
        sections.append((None, markdown[: matches[0].start()]))
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(markdown)
        sections.append((m.group(2).strip(), markdown[m.start() : end]))
    return sections


def chunk_markdown(markdown: str, chunk_size: int, overlap: int) -> list[Chunk]:
    """Chunk markdown into heading-aware pieces of at most chunk_size chars.

    Splits on headings (H1-H4) first so a section's content stays
    together; oversized sections are hard-split with overlap chars of
    context repeated at each boundary. Page provenance HTML comments
    are stripped before chunking. When several small sections pack into
    one chunk, section is the first heading in the chunk. Returns []
    for empty/whitespace-only input.
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    markdown = PAGE_MARKER_RE.sub("", markdown)
    chunks: list[Chunk] = []
    buffer = ""
    buffer_section: str | None = None

    def flush() -> None:
        nonlocal buffer
        text = buffer.strip()
        if text:
            chunks.append(Chunk(index=len(chunks), section=buffer_section, text=text))
        buffer = ""

    for section, body in _split_sections(markdown):
        if len(buffer) + len(body) > chunk_size and buffer:
            flush()
        if not buffer:
            buffer_section = section
        if len(body) <= chunk_size:
            buffer += body
            continue
        # oversized section: hard-split with overlap
        flush()
        buffer_section = section
        start = 0
        while start < len(body):
            piece = body[start : start + chunk_size]
            stripped = piece.strip()
            if stripped:
                chunks.append(Chunk(index=len(chunks), section=section, text=stripped))
            start += chunk_size - overlap
    flush()
    return chunks
