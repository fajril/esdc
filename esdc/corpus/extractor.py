"""Tiered per-page PDF -> markdown extraction.

Native text layer per page when present (pymupdf4llm); local vision-LLM
OCR only for pages without one. Never OCR a page with a healthy text
layer, never let the LLM rewrite native text. Large embedded raster
images on native pages (tables saved as pictures) are clip-rendered and
OCR'd separately, appended below the page text. Every page is wrapped in
an HTML-comment marker so the human reviewer knows which pages came
from OCR and deserve extra scrutiny.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import docx
import fitz
import pymupdf4llm
from docx.table import Table
from docx.text.paragraph import Paragraph

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
    images_ocr: int = 0  # embedded images OCR'd on native pages
    images_skipped: int = 0  # significant images found but no OCR client


def _significant_image_rects(
    page: "fitz.Page", min_image_area: float
) -> list["fitz.Rect"]:
    """Bboxes of embedded raster images covering >= min_image_area of the page.

    Small images (logos, signatures, stamps) fall below the threshold and
    are ignored; large ones are almost always tables/charts saved as
    pictures, whose content the native text layer does not contain.
    """
    page_area = page.rect.get_area()
    if not page_area:
        return []
    rects = []
    for info in page.get_image_info():
        r = fitz.Rect(info["bbox"])
        if r.get_area() / page_area >= min_image_area:
            rects.append(r)
    return rects


def extract_pdf(
    path: Path,
    ocr_client: OcrClient | None,
    min_chars_per_page: int = 50,
    ocr_dpi: int = 200,
    min_image_area: float = 0.05,
) -> ExtractionResult:
    """Extract a PDF to markdown, tiering each page independently.

    Pages with a healthy native text layer (>= min_chars_per_page
    characters) are extracted verbatim via pymupdf4llm. Pages without
    one are rasterized and sent to ocr_client.ocr_page(). Raises
    ValueError if a page has no text layer and no ocr_client is given.

    Native pages are additionally scanned for embedded raster images
    covering >= min_image_area of the page (tables/charts saved as
    pictures); each such region is clip-rendered, OCR'd, and appended
    below the page text as a `<!-- page P image K: llm_ocr -->` block.
    Without an OCR client these images are counted in images_skipped
    instead of failing: the page still has its text layer.
    """
    pages = pymupdf4llm.to_markdown(str(path), page_chunks=True)
    if not pages:
        raise ValueError(f"{path.name}: no extractable pages")
    native_ok = [
        len((p.get("text") or "").strip()) >= min_chars_per_page for p in pages
    ]

    if not all(native_ok) and ocr_client is None:
        bad = [i + 1 for i, ok in enumerate(native_ok) if not ok]
        raise ValueError(
            f"{path.name}: scanned page(s) {bad} have no text layer and no OCR "
            "model is available. Check `ollama list` for the corpus.ocr_model."
        )

    doc = fitz.open(str(path))
    parts: list[str] = []
    images_ocr = 0
    images_skipped = 0
    try:
        for idx, page_md in enumerate(pages):
            if native_ok[idx]:
                part = f"<!-- page {idx + 1}: native -->\n{page_md['text'].strip()}"
                rects = _significant_image_rects(doc[idx], min_image_area)
                if ocr_client is None:
                    images_skipped += len(rects)
                else:
                    for img_no, rect in enumerate(rects, start=1):
                        pix = doc[idx].get_pixmap(dpi=ocr_dpi, clip=rect)
                        text = ocr_client.ocr_page(pix.tobytes("png"))
                        part += (
                            f"\n\n<!-- page {idx + 1} image {img_no}: llm_ocr -->"
                            f"\n{text.strip()}"
                        )
                        images_ocr += 1
                parts.append(part)
            else:
                pix = doc[idx].get_pixmap(dpi=ocr_dpi)
                text = ocr_client.ocr_page(pix.tobytes("png"))
                parts.append(f"<!-- page {idx + 1}: llm_ocr -->\n{text.strip()}")
    finally:
        doc.close()

    pages_native = sum(native_ok)
    pages_ocr = len(pages) - pages_native
    if pages_ocr == 0 and images_ocr == 0:
        method = "native"
    elif pages_native == 0:
        method = "llm_ocr"
    else:
        method = "mixed"
    return ExtractionResult(
        markdown="\n\n".join(parts),
        method=method,
        page_count=len(pages),
        pages_native=pages_native,
        pages_ocr=pages_ocr,
        images_ocr=images_ocr,
        images_skipped=images_skipped,
    )


def _escape_cell(text: str) -> str:
    return text.replace("|", "\\|").strip()


def _table_to_markdown(table: Table) -> str:
    """Convert a docx Table to a markdown table; first row is the header.

    Merged cells: python-docx repeats the merged text across every grid
    cell it spans, so those repeats are emitted as-is (documented
    limitation — no attempt to detect/collapse spans).
    """
    rows = [[_escape_cell(cell.text) for cell in row.cells] for row in table.rows]
    if not rows:
        return ""
    header = "| " + " | ".join(rows[0]) + " |"
    sep = "| " + " | ".join("---" for _ in rows[0]) + " |"
    lines = [header, sep]
    for row in rows[1:]:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def extract_docx(path: Path) -> ExtractionResult:
    """Extract a .docx source to markdown, walking the document in order.

    Headings become '#'-prefixed lines (level from the 'Heading N' style),
    plain paragraphs pass through verbatim, and tables become markdown
    tables (see `_table_to_markdown`). Embedded images are not OCR'd in
    v1 — they are only counted in `images_skipped`.
    """
    document = docx.Document(str(path))
    parts: list[str] = []
    for block in document.iter_inner_content():
        if isinstance(block, Paragraph):
            text = block.text.strip()
            if not text:
                continue
            style = (block.style.name or "") if block.style else ""
            if style.startswith("Heading"):
                level_str = style.removeprefix("Heading").strip()
                level = int(level_str) if level_str.isdigit() else 1
                parts.append(f"{'#' * level} {text}")
            else:
                parts.append(text)
        elif isinstance(block, Table):
            table_md = _table_to_markdown(block)
            if table_md:
                parts.append(table_md)

    images_skipped = len(document.inline_shapes)

    if not parts:
        raise ValueError(f"{path.name}: no extractable content")

    return ExtractionResult(
        markdown="<!-- page 1: native_docx -->\n" + "\n\n".join(parts),
        method="native_docx",
        page_count=1,
        pages_native=1,
        pages_ocr=0,
        images_skipped=images_skipped,
    )


def extract_markdown(path: Path) -> ExtractionResult:
    """Read a markdown source verbatim (frontmatter, if any, is kept as content)."""
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"{path.name}: no extractable content")
    return ExtractionResult(
        markdown=f"<!-- page 1: native_md -->\n{text}",
        method="native_md",
        page_count=1,
        pages_native=1,
        pages_ocr=0,
    )


SUPPORTED_EXTENSIONS = (".pdf", ".docx", ".md")


def extract_document(
    path: Path,
    ocr_client: OcrClient | None,
    min_chars_per_page: int = 50,
    ocr_dpi: int = 200,
    min_image_area: float = 0.05,
) -> ExtractionResult:
    """Route a source file to its format converter (see SUPPORTED_EXTENSIONS)."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return extract_pdf(
            path, ocr_client, min_chars_per_page, ocr_dpi, min_image_area
        )
    if suffix == ".docx":
        return extract_docx(path)
    if suffix == ".md":
        return extract_markdown(path)
    raise ValueError(f"{path.name}: unsupported format '{suffix}'")
