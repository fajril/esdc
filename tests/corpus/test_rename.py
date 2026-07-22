import datetime

from esdc.corpus import rename


def test_format_doc_date_from_iso_string():
    assert rename.format_doc_date("2024-01-15") == "2024.01.15"


def test_format_doc_date_from_date_object():
    assert rename.format_doc_date(datetime.date(2024, 3, 5)) == "2024.03.05"


def test_format_doc_date_none_and_garbage():
    assert rename.format_doc_date(None) is None
    assert rename.format_doc_date("not-a-date") is None
    assert rename.format_doc_date("") is None


def test_sanitize_title_strips_illegal_chars_and_collapses_space():
    assert rename.sanitize_title("Persetujuan  POD/OPL: Duri?") == "Persetujuan POD OPL Duri"


def test_sanitize_title_none_and_empty():
    assert rename.sanitize_title(None) is None
    assert rename.sanitize_title("   ") is None
    assert rename.sanitize_title("///") is None


def test_sanitize_title_caps_length():
    out = rename.sanitize_title("x" * 300)
    assert out is not None and len(out) <= 150


from pathlib import Path

import esdc.corpus.rename as rename_mod
from esdc.corpus.sidecar import sidecar_path, write_sidecar_file


def _write_sidecar(src: Path, meta: dict) -> Path:
    return write_sidecar_file(sidecar_path(src), meta, "<!-- page 1: native -->\nbody")


def _stub_no_llm(monkeypatch):
    # No metadata model, no OCR: LLM tier is unavailable.
    monkeypatch.setattr(rename_mod.Config, "get_corpus_config", staticmethod(lambda: {
        "ocr_model": "x", "metadata_model": "", "num_ctx": 16384,
        "min_chars_per_page": 50, "ocr_dpi": 200, "min_image_area": 0.05,
        "ollama_host": None,
    }))
    monkeypatch.setattr(rename_mod, "_resolve_text_caller", lambda *a, **k: None)

    class _DeadOcr:
        def __init__(self, *a, **k):
            pass

        def health_check(self):
            return False

    monkeypatch.setattr(rename_mod, "OllamaVisionOcr", _DeadOcr)

    class _NoStore:
        def __init__(self, *a, **k):
            pass

        def get_document_by_hash(self, h):
            return None

        def close(self):
            pass

    monkeypatch.setattr(rename_mod, "CorpusStore", _NoStore)


def test_run_rename_uses_sidecar_metadata(tmp_path, monkeypatch):
    _stub_no_llm(monkeypatch)
    src = tmp_path / "scan001.pdf"
    src.write_bytes(b"%PDF-1.4 dummy")
    _write_sidecar(src, {
        "file_hash": "abc", "doc_type": "letter",
        "doc_date": "2024-01-15", "subject": "Persetujuan POD Duri",
    })

    report, plans = rename_mod.run_rename([tmp_path], apply=False)

    assert len(plans) == 1
    plan = plans[0]
    assert plan.source == "sidecar"
    assert plan.new_path.name == "letter - 2024.01.15 - Persetujuan POD Duri.pdf"
    assert plan.sidecar_new.name == "letter - 2024.01.15 - Persetujuan POD Duri.corpus.md"
    # dry-run: nothing moved
    assert src.exists()
    assert not plan.new_path.exists()


def test_run_rename_doc_type_override_beats_sidecar(tmp_path, monkeypatch):
    _stub_no_llm(monkeypatch)
    src = tmp_path / "x.pdf"
    src.write_bytes(b"%PDF-1.4 dummy")
    _write_sidecar(src, {
        "file_hash": "abc", "doc_type": "letter",
        "doc_date": "2024-01-15", "subject": "Judul",
    })

    _report, plans = rename_mod.run_rename([tmp_path], doc_type="mom", apply=False)
    assert plans[0].new_path.name == "mom - 2024.01.15 - Judul.pdf"


def test_run_rename_skips_when_field_unresolved(tmp_path, monkeypatch):
    _stub_no_llm(monkeypatch)
    src = tmp_path / "y.pdf"
    src.write_bytes(b"%PDF-1.4 dummy")
    _write_sidecar(src, {
        "file_hash": "abc", "doc_type": "letter",
        "doc_date": None, "subject": "Judul",
    })

    report, plans = rename_mod.run_rename([tmp_path], apply=False)
    assert plans[0].new_path is None
    assert "missing doc_date" in plans[0].note
    assert any("y.pdf" in s for s in report.skipped)


def test_run_rename_falls_back_to_db(tmp_path, monkeypatch):
    _stub_no_llm(monkeypatch)  # then override the store to return a hit

    class _HitStore:
        def __init__(self, *a, **k):
            pass

        def get_document_by_hash(self, h):
            return {"doc_type": "ba", "doc_date": "2023-12-01", "subject": "Berita Acara"}

        def close(self):
            pass

    monkeypatch.setattr(rename_mod, "CorpusStore", _HitStore)
    src = tmp_path / "z.pdf"
    src.write_bytes(b"%PDF-1.4 dummy")  # no sidecar

    _report, plans = rename_mod.run_rename([tmp_path], apply=False)
    assert plans[0].source == "db"
    assert plans[0].new_path.name == "ba - 2023.12.01 - Berita Acara.pdf"


def test_run_rename_llm_tier_receives_filename(tmp_path, monkeypatch):
    # metadata caller available; capture the filename passed to llm_extract.
    monkeypatch.setattr(rename_mod.Config, "get_corpus_config", staticmethod(lambda: {
        "ocr_model": "x", "metadata_model": "main", "num_ctx": 16384,
        "min_chars_per_page": 50, "ocr_dpi": 200, "min_image_area": 0.05,
        "ollama_host": None,
    }))
    monkeypatch.setattr(rename_mod, "_resolve_text_caller", lambda *a, **k: (lambda p: "{}"))

    class _DeadOcr:
        def __init__(self, *a, **k):
            pass

        def health_check(self):
            return False

    monkeypatch.setattr(rename_mod, "OllamaVisionOcr", _DeadOcr)

    class _NoStore:
        def __init__(self, *a, **k):
            pass

        def get_document_by_hash(self, h):
            return None

        def close(self):
            pass

    monkeypatch.setattr(rename_mod, "CorpusStore", _NoStore)

    class _Result:
        markdown = "<!-- page 1: native -->\nbody"

    monkeypatch.setattr(rename_mod, "extract_document", lambda *a, **k: _Result())

    seen = {}

    def fake_llm_extract(markdown, caller, filename=None):
        seen["filename"] = filename
        return {"doc_type": "note", "doc_date": "2022-06-30", "subject": "Catatan"}

    monkeypatch.setattr(rename_mod, "llm_extract", fake_llm_extract)

    src = tmp_path / "memo-final.pdf"
    src.write_bytes(b"%PDF-1.4 dummy")

    _report, plans = rename_mod.run_rename([tmp_path], apply=False)
    assert seen["filename"] == "memo-final.pdf"
    assert plans[0].source == "llm"
    assert plans[0].new_path.name == "note - 2022.06.30 - Catatan.pdf"


def test_run_rename_collision_within_batch(tmp_path, monkeypatch):
    _stub_no_llm(monkeypatch)
    a = tmp_path / "a.pdf"
    b = tmp_path / "b.pdf"
    for p in (a, b):
        p.write_bytes(b"%PDF-1.4 dummy")
    meta = {"file_hash": "h", "doc_type": "letter",
            "doc_date": "2024-01-15", "subject": "Same Title"}
    _write_sidecar(a, meta)
    _write_sidecar(b, meta)

    report, plans = rename_mod.run_rename([tmp_path], apply=False)
    # first claims the target, second is blocked
    resolved = [p for p in plans if p.new_path is not None]
    blocked = [p for p in plans if p.new_path is None]
    assert len(resolved) == 1
    assert len(blocked) == 1
    assert "target exists" in blocked[0].note
    assert report.failed


from esdc.corpus.sidecar import read_sidecar


def test_apply_renames_source_and_sidecar(tmp_path, monkeypatch):
    _stub_no_llm(monkeypatch)
    src = tmp_path / "scan.pdf"
    src.write_bytes(b"%PDF-1.4 dummy")
    _write_sidecar(src, {
        "file_hash": "abc", "source_file": "scan.pdf", "doc_type": "letter",
        "doc_date": "2024-01-15", "subject": "Judul Surat",
    })

    report, plans = rename_mod.run_rename([tmp_path], apply=True)

    new_src = tmp_path / "letter - 2024.01.15 - Judul Surat.pdf"
    new_sc = tmp_path / "letter - 2024.01.15 - Judul Surat.corpus.md"
    assert new_src.exists() and not src.exists()
    assert new_sc.exists()
    assert not (tmp_path / "scan.corpus.md").exists()
    meta, _body = read_sidecar(new_sc)
    assert meta["source_file"] == "letter - 2024.01.15 - Judul Surat.pdf"
    assert any("scan.pdf" in p for p in report.processed)


def test_apply_leaves_already_named_untouched(tmp_path, monkeypatch):
    _stub_no_llm(monkeypatch)
    name = "letter - 2024.01.15 - Judul.pdf"
    src = tmp_path / name
    src.write_bytes(b"%PDF-1.4 dummy")
    _write_sidecar(src, {
        "file_hash": "abc", "doc_type": "letter",
        "doc_date": "2024-01-15", "subject": "Judul",
    })

    report, _plans = rename_mod.run_rename([tmp_path], apply=True)
    assert src.exists()
    assert any("already named" in s for s in report.skipped)
    assert not report.processed
