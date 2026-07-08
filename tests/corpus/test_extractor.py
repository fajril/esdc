from pathlib import Path

import fitz  # PyMuPDF, pulled in by pymupdf4llm
import pytest

from esdc.corpus.extractor import ExtractionResult, extract_pdf


@pytest.fixture
def text_pdf(tmp_path: Path) -> Path:
    """One page with a real text layer."""
    path = tmp_path / "surat.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Nomor: SRT-001/SKK/2026")
    page.insert_text((72, 100), "Perihal: Persetujuan POD Lapangan Duri tahun 2026")
    doc.save(path)
    return path


@pytest.fixture
def mixed_pdf(tmp_path: Path) -> Path:
    """Page 1 has text, page 2 is blank (scanned-page stand-in)."""
    path = tmp_path / "mixed.pdf"
    doc = fitz.open()
    page = doc.new_page()
    # insert_textbox wraps within the rect; a bare insert_text() line this
    # long overflows the page width and pymupdf4llm drops the whole span.
    page.insert_textbox(
        (72, 72, 523, 200),
        "Halaman digital dengan teks yang cukup panjang " * 3,
    )
    doc.new_page()
    doc.save(path)
    return path


@pytest.fixture
def scanned_pdf(tmp_path: Path) -> Path:
    """One blank page (scanned-page stand-in): no text layer at all."""
    path = tmp_path / "scan.pdf"
    doc = fitz.open()
    doc.new_page()
    doc.save(path)
    return path


class FakeOcr:
    def ocr_page(self, png_bytes: bytes) -> str:
        return "teks hasil OCR halaman scan"


def test_native_only(text_pdf):
    result = extract_pdf(text_pdf, ocr_client=None)
    assert isinstance(result, ExtractionResult)
    assert "SRT-001" in result.markdown
    assert "<!-- page 1: native -->" in result.markdown
    assert result.method == "native"
    assert result.pages_ocr == 0 and result.pages_native == 1


def test_mixed_uses_ocr_per_page(mixed_pdf):
    result = extract_pdf(mixed_pdf, ocr_client=FakeOcr())
    assert result.method == "mixed"
    assert "<!-- page 1: native -->" in result.markdown
    assert "<!-- page 2: llm_ocr -->" in result.markdown
    assert "hasil OCR" in result.markdown
    assert result.pages_native == 1 and result.pages_ocr == 1


def test_all_scanned_uses_llm_ocr(scanned_pdf):
    result = extract_pdf(scanned_pdf, ocr_client=FakeOcr())
    assert result.method == "llm_ocr"
    assert result.pages_native == 0 and result.pages_ocr == 1
    assert "<!-- page 1: llm_ocr -->" in result.markdown


def test_scanned_without_ocr_raises(mixed_pdf):
    with pytest.raises(ValueError, match="scanned page"):
        extract_pdf(mixed_pdf, ocr_client=None)
