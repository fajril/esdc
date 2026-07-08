from pathlib import Path

import pytest

from esdc.corpus.sidecar import read_sidecar, sidecar_path, write_sidecar


def test_sidecar_path():
    assert sidecar_path(Path("/x/surat.pdf")) == Path("/x/surat.corpus.md")


def test_roundtrip(tmp_path: Path):
    pdf = tmp_path / "mom.pdf"
    meta = {
        "source_file": "mom.pdf", "file_hash": "ab" * 32, "page_count": 2,
        "extraction_method": "mixed", "doc_type": "mom", "doc_number": None,
        "doc_date": "2026-03-01", "subject": "Monitoring POD", "sender": "SKK",
        "recipient": None, "doc_level": "project", "wk_name": "Rokan",
        "field_name": None, "project_name": "POD Duri", "extras": {"peserta": []},
    }
    body = "<!-- page 1: native -->\n# MoM\nisi"
    path = write_sidecar(pdf, meta, body)
    assert path == tmp_path / "mom.corpus.md"

    loaded_meta, loaded_body = read_sidecar(path)
    assert loaded_meta["doc_type"] == "mom"
    assert loaded_meta["file_hash"] == "ab" * 32
    assert loaded_body.strip() == body


def test_read_missing_frontmatter_raises(tmp_path: Path):
    bad = tmp_path / "x.corpus.md"
    bad.write_text("no frontmatter here")
    with pytest.raises(ValueError, match="frontmatter"):
        read_sidecar(bad)


def test_body_with_horizontal_rule_survives_roundtrip(tmp_path: Path):
    pdf = tmp_path / "surat.pdf"
    meta = {"file_hash": "cd" * 32}
    body = "# Judul\n\nparagraf pertama\n\n---\n\nparagraf setelah garis pemisah"
    path = write_sidecar(pdf, meta, body)

    _, loaded_body = read_sidecar(path)
    assert loaded_body.strip() == body
