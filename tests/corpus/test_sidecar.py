from pathlib import Path

import pytest

from esdc.corpus.sidecar import (
    read_sidecar,
    sidecar_path,
    write_sidecar,
    write_sidecar_file,
)


def test_sidecar_path():
    assert sidecar_path(Path("/x/surat.pdf")) == Path("/x/surat.corpus.md")


def test_write_sidecar_file_writes_to_exact_path(tmp_path):
    sc = tmp_path / "doc.corpus.md"
    write_sidecar_file(sc, {"file_hash": "aa", "reviewed": True}, "# Body")
    meta, body = read_sidecar(sc)
    assert meta["reviewed"] is True
    assert "# Body" in body


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
    with pytest.raises(ValueError, match="missing YAML frontmatter"):
        read_sidecar(bad)


def test_read_invalid_yaml_raises(tmp_path: Path):
    bad = tmp_path / "x.corpus.md"
    bad.write_text('---\nsubject: "unterminated quote\n---\nbody')
    with pytest.raises(ValueError, match="invalid YAML"):
        read_sidecar(bad)


def test_read_tolerates_leading_blank_line(tmp_path: Path):
    path = tmp_path / "x.corpus.md"
    path.write_text(f"\n---\nfile_hash: {'ef' * 32}\n---\nbody")
    meta, body = read_sidecar(path)
    assert meta["file_hash"] == "ef" * 32
    assert body.strip() == "body"


def test_body_with_horizontal_rule_survives_roundtrip(tmp_path: Path):
    pdf = tmp_path / "surat.pdf"
    meta = {"file_hash": "cd" * 32}
    body = "# Judul\n\nparagraf pertama\n\n---\n\nparagraf setelah garis pemisah"
    path = write_sidecar(pdf, meta, body)

    _, loaded_body = read_sidecar(path)
    assert loaded_body.strip() == body


def test_read_write_roundtrip_idempotent(tmp_path: Path):
    """Repeated read->write cycles must not grow the file with blank lines.

    run_meta makes rewrites a normal workflow, so each cycle re-feeding
    read_sidecar's body into write_sidecar_file has to produce byte-identical
    output instead of accumulating one leading + one trailing newline.
    """
    sc = tmp_path / "doc.corpus.md"
    write_sidecar_file(sc, {"file_hash": "aa" * 32, "reviewed": False}, "hello body")
    baseline = sc.read_bytes()

    for _ in range(3):
        meta, body = read_sidecar(sc)
        write_sidecar_file(sc, meta, body)
        assert sc.read_bytes() == baseline

    _, final_body = read_sidecar(sc)
    assert final_body.strip() == "hello body"


def test_roundtrip_preserves_internal_blank_lines_and_dashes(tmp_path: Path):
    sc = tmp_path / "doc.corpus.md"
    body = "# Judul\n\n\nparagraf pertama\n\n---\nparagraf setelah garis"
    write_sidecar_file(sc, {"file_hash": "bb" * 32}, body)

    for _ in range(3):
        meta, loaded_body = read_sidecar(sc)
        assert loaded_body.strip("\n") == body
        write_sidecar_file(sc, meta, loaded_body)

    _, final_body = read_sidecar(sc)
    assert final_body.strip("\n") == body
