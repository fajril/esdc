"""Tiered per-page PDF -> markdown extraction.

Native text layer per page when present (pymupdf4llm); local vision-LLM
OCR only for pages without one. Never OCR a page with a healthy text
layer, never let the LLM rewrite native text. Every page is wrapped in
an HTML-comment marker so the human reviewer knows which pages came
from OCR and deserve extra scrutiny.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import fitz
import pymupdf4llm

logger = logging.getLogger(__name__)


class OcrClient(Protocol):
    def ocr_page(self, png_bytes: bytes) -> str: ...


@dataclass
class ExtractionResult:
    markdown: str  # with <!-- page N: method --> markers
    method: str  # 'native' | 'llm_ocr' | 'mixed'
    page_count: int
    pages_native: int
    pages_ocr: int


def extract_pdf(
    path: Path,
    ocr_client: OcrClient | None,
    min_chars_per_page: int = 50,
    ocr_dpi: int = 200,
) -> ExtractionResult:
    """Extract a PDF to markdown, tiering each page independently.

    Pages with a healthy native text layer (>= min_chars_per_page
    characters) are extracted verbatim via pymupdf4llm. Pages without
    one are rasterized and sent to ocr_client.ocr_page(). Raises
    ValueError if a page has no text layer and no ocr_client is given.
    """
    pages = pymupdf4llm.to_markdown(str(path), page_chunks=True)
    native_ok = [
        len((p.get("text") or "").strip()) >= min_chars_per_page for p in pages
    ]

    if not all(native_ok) and ocr_client is None:
        bad = [i + 1 for i, ok in enumerate(native_ok) if not ok]
        raise ValueError(
            f"{path.name}: scanned page(s) {bad} have no text layer and no OCR "
            "model is available. Check `ollama list` for the corpus.ocr_model."
        )

    doc = fitz.open(str(path)) if not all(native_ok) else None
    parts: list[str] = []
    for idx, page_md in enumerate(pages):
        if native_ok[idx]:
            parts.append(f"<!-- page {idx + 1}: native -->\n{page_md['text'].strip()}")
        else:
            pix = doc[idx].get_pixmap(dpi=ocr_dpi)
            text = ocr_client.ocr_page(pix.tobytes("png"))
            parts.append(f"<!-- page {idx + 1}: llm_ocr -->\n{text.strip()}")
    if doc is not None:
        doc.close()

    pages_native = sum(native_ok)
    pages_ocr = len(pages) - pages_native
    method = "native" if pages_ocr == 0 else "llm_ocr" if pages_native == 0 else "mixed"
    return ExtractionResult(
        markdown="\n\n".join(parts),
        method=method,
        page_count=len(pages),
        pages_native=pages_native,
        pages_ocr=pages_ocr,
    )
