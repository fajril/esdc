from contextlib import contextmanager
from pathlib import Path

import fitz
import pytest

from esdc.corpus import pipeline
from esdc.corpus.chunker import Chunk
from esdc.corpus.extractor import ExtractionResult
from esdc.corpus.sidecar import read_sidecar, sidecar_path, write_sidecar
from esdc.corpus.store import CorpusStore


class FakeEntityResolver:
    """Stand-in for EntityResolver that returns no matches by default."""

    def __init__(self, db=None, matches=None):
        self._matches = matches or {}

    def resolve(self, query):
        # Return first match for any key that contains the query
        for key, match in self._matches.items():
            if key.lower() in query.lower():
                return {
                    "status": "success",
                    "entities": [match],
                }
        return {"status": "failed", "entities": []}

    def resolve_name(self, name, entity_type):
        # Substring match against configured `matches`, filtered by type,
        # sorted by confidence DESC — mirrors the real resolve_name contract.
        hits = [
            match
            for key, match in self._matches.items()
            if match.get("entity_type") == entity_type and key.lower() in name.lower()
        ]
        return sorted(hits, key=lambda m: m["confidence"], reverse=True)

DEFAULT_CFG = {
    "chunk_size": 500,
    "chunk_overlap": 50,
    "ocr_model": "glm-ocr",
    "metadata_model": "fake-text-model",
    "ocr_dpi": 200,
    "num_ctx": 16384,
    "min_chars_per_page": 50,
    "min_image_area": 0.05,
}


class FakeOcr:
    """Stand-in for OllamaVisionOcr with a controllable health check."""

    def __init__(self, healthy: bool = True) -> None:
        self._healthy = healthy

    def health_check(self) -> bool:
        return self._healthy

    def query_image(self, png_bytes: bytes, prompt: str) -> str:
        return '{"doc_type": "surat", "doc_level": "wk"}'

    def ocr_page(self, png_bytes: bytes) -> str:
        return "ocr text"


class FakeEmbedder:
    """3-dim deterministic fake embedder, mirrors tests/corpus/test_store.py."""

    model = "fake-embed-v1"

    def generate_embedding(self, text: str) -> list[float]:
        return [1.0, 0.0, 0.0]

    def generate_embeddings_batch(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0, 0.0] for _ in texts]


class FakeEmbedder2:
    """A different model/dim, standing in for a post-reembed model swap."""

    model = "fake-embed-v2"

    def generate_embedding(self, text: str) -> list[float]:
        return [1.0, 0.0]

    def generate_embeddings_batch(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]


@pytest.fixture(autouse=True)
def patch_seams(tmp_path, monkeypatch):
    """Default seam patches every test starts from; tests override as needed.

    CorpusStore defaults to a tmp_path-rooted DB with no lookup tables, so
    ``_entity_resolver_or_none`` naturally resolves to "skip" (None) unless
    a test opts into a real/fake resolver — this keeps run_extract tests
    from ever touching the user's real ~/.esdc database.
    """
    monkeypatch.setattr(pipeline.Config, "get_corpus_config", lambda: dict(DEFAULT_CFG))
    monkeypatch.setattr(pipeline, "OllamaVisionOcr", lambda *a, **kw: FakeOcr(True))
    monkeypatch.setattr(
        pipeline,
        "llm_extract",
        lambda markdown, caller: {"doc_type": "surat", "doc_level": "wk"},
    )
    monkeypatch.setattr(
        pipeline,
        "CorpusStore",
        lambda *a, **kw: CorpusStore(
            db_path=tmp_path / "_default.duckdb", embedder=FakeEmbedder()
        ),
    )


def make_pdf(tmp_path: Path, name: str = "doc.pdf") -> Path:
    """A minimal real PDF (fitz needs a real file for the metadata-image path)."""
    path = tmp_path / name
    doc = fitz.open()
    doc.new_page()
    doc.save(path)
    doc.close()
    return path


def make_sidecar(
    tmp_path: Path,
    name: str,
    reviewed: bool,
    file_hash: str,
    body: str = "# Doc\nisi dokumen penting",
    **extra_meta,
) -> Path:
    """Write a .corpus.md sidecar directly (real sidecar module, no PDF needed)."""
    pdf = tmp_path / name
    meta = {
        "source_file": name,
        "file_hash": file_hash,
        "page_count": 1,
        "extraction_method": "native",
        "reviewed": reviewed,
        "doc_type": "surat",
        "doc_number": "1",
        "doc_date": "2026-01-01",
        "subject": "test",
        "sender": "A",
        "recipient": "B",
        "doc_level": "wk",
        "wk_name": None,
        "field_name": None,
        "project_name": None,
        "extras": {},
    }
    meta.update(extra_meta)
    write_sidecar(pdf, meta, body)
    return sidecar_path(pdf)


def make_store(tmp_path: Path, embedder=None, db_name: str = "corpus.duckdb") -> CorpusStore:
    return CorpusStore(db_path=tmp_path / db_name, embedder=embedder or FakeEmbedder())


def patch_store_factory(monkeypatch, store: CorpusStore) -> None:
    monkeypatch.setattr(pipeline, "CorpusStore", lambda *a, **kw: store)


def patch_entity_resolver(monkeypatch, matches=None):
    """Patch EntityResolver in pipeline to use FakeEntityResolver."""
    monkeypatch.setattr(
        pipeline,
        "EntityResolver",
        lambda db: FakeEntityResolver(db=db, matches=matches),
    )


def make_docx(tmp_path: Path, name: str = "doc.docx") -> Path:
    """A minimal real .docx source (heading + paragraph)."""
    from docx import Document

    d = Document()
    d.add_heading("Judul", level=1)
    d.add_paragraph("isi dokumen penting")
    path = tmp_path / name
    d.save(path)
    return path


# --------------------------------------------------------------------------
# _collect_sources
# --------------------------------------------------------------------------


def test_collect_sources_all_formats_excludes_sidecars(tmp_path):
    (tmp_path / "a.pdf").write_bytes(b"%PDF-1.4 fake")
    make_docx(tmp_path, "b.docx")
    (tmp_path / "c.md").write_text("# hi", encoding="utf-8")
    (tmp_path / "c.corpus.md").write_text("sidecar content", encoding="utf-8")
    (tmp_path / "d.txt").write_text("nope", encoding="utf-8")

    got = pipeline._collect_sources([tmp_path])
    names = {p.name for p in got}

    assert names == {"a.pdf", "b.docx", "c.md"}
    assert "c.corpus.md" not in names
    assert "d.txt" not in names


def test_collect_sources_collision_second_source_fails(tmp_path, monkeypatch):
    (tmp_path / "report.pdf").write_bytes(b"%PDF-1.4 fake")
    (tmp_path / "report.docx").write_bytes(b"PK fake docx bytes")

    def fake_extract_pdf(path, ocr_client, min_chars, dpi, min_image_area=0.05):
        return ExtractionResult("pdf body", "native", 1, 1, 0)

    def fake_extract_docx(path):
        return ExtractionResult("docx body", "native_docx", 1, 1, 0)

    monkeypatch.setattr(pipeline, "extract_pdf", fake_extract_pdf)
    monkeypatch.setattr(pipeline, "extract_docx", fake_extract_docx)

    report = pipeline.run_extract([tmp_path])

    assert len(report.processed) == 1
    assert len(report.failed) == 1
    failed_name, failed_msg = next(iter(report.failed.items()))
    assert failed_name in ("report.pdf", "report.docx")
    assert "sidecar path collision" in failed_msg
    # only one sidecar written, for the winner
    assert (tmp_path / "report.corpus.md").exists()


def test_extract_md_source_end_to_end(tmp_path, monkeypatch):
    src = tmp_path / "notes.md"
    src.write_text("# Judul\n\nisi dokumen penting", encoding="utf-8")

    report = pipeline.run_extract([src])
    assert report.failed == {}

    sidecar = tmp_path / "notes.corpus.md"
    assert sidecar.exists()
    meta, body = read_sidecar(sidecar)
    assert meta["extraction_method"] == "native_md"
    assert "isi dokumen penting" in body
    assert report.processed == ["notes.md [native — minimal review]"]


def test_extract_docx_source_end_to_end(tmp_path, monkeypatch):
    src = make_docx(tmp_path, "report.docx")

    report = pipeline.run_extract([src])
    assert report.failed == {}

    sidecar = tmp_path / "report.corpus.md"
    assert sidecar.exists()
    meta, body = read_sidecar(sidecar)
    assert meta["extraction_method"] == "native_docx"
    assert "isi dokumen penting" in body
    assert report.processed == ["report.docx [native — minimal review]"]


# --------------------------------------------------------------------------
# run_extract
# --------------------------------------------------------------------------


def test_extract_writes_sidecar_for_new_pdf(tmp_path, monkeypatch):
    pdf = make_pdf(tmp_path, "surat.pdf")

    def fake_extract(path, ocr_client, min_chars, dpi, min_image_area=0.05):
        return ExtractionResult(
            markdown="# Surat\nisi dokumen",
            method="native",
            page_count=1,
            pages_native=1,
            pages_ocr=0,
        )

    monkeypatch.setattr(pipeline, "extract_pdf", fake_extract)

    report = pipeline.run_extract([pdf])

    sidecar = tmp_path / "surat.corpus.md"
    assert sidecar.exists()
    assert report.failed == {}
    assert len(report.processed) == 1
    assert "surat.pdf" in report.processed[0]

    meta, body = read_sidecar(sidecar)
    assert meta["reviewed"] is False
    assert meta["extraction_method"] == "native"
    assert "isi dokumen" in body


def test_extract_skips_existing_sidecar(tmp_path, monkeypatch):
    pdf = make_pdf(tmp_path, "surat.pdf")
    write_sidecar(
        pdf, {"source_file": "surat.pdf", "file_hash": "x", "reviewed": False}, "old body"
    )

    called = []

    def fake_extract(path, ocr_client, min_chars, dpi, min_image_area=0.05):
        called.append(path)
        return ExtractionResult("new", "native", 1, 1, 0)

    monkeypatch.setattr(pipeline, "extract_pdf", fake_extract)

    report = pipeline.run_extract([pdf])
    assert called == []
    assert report.skipped == ["surat.pdf (sidecar exists)"]
    assert report.processed == []


def test_extract_force_overwrites(tmp_path, monkeypatch):
    pdf = make_pdf(tmp_path, "surat.pdf")
    write_sidecar(
        pdf, {"source_file": "surat.pdf", "file_hash": "x", "reviewed": True}, "old body"
    )

    def fake_extract(path, ocr_client, min_chars, dpi, min_image_area=0.05):
        return ExtractionResult("new body", "native", 1, 1, 0)

    monkeypatch.setattr(pipeline, "extract_pdf", fake_extract)

    report = pipeline.run_extract([pdf], force=True)
    assert report.skipped == []
    assert len(report.processed) == 1

    meta, body = read_sidecar(tmp_path / "surat.corpus.md")
    assert meta["reviewed"] is False
    assert "new body" in body


def test_extract_ocr_ratio_sort_and_flags(tmp_path, monkeypatch):
    pdf_native = make_pdf(tmp_path, "native.pdf")
    pdf_heavy = make_pdf(tmp_path, "heavy.pdf")
    pdf_light = make_pdf(tmp_path, "light.pdf")

    results = {
        "native.pdf": ExtractionResult("native body", "native", 2, 2, 0),
        "heavy.pdf": ExtractionResult("heavy body", "llm_ocr", 4, 0, 4),
        "light.pdf": ExtractionResult("light body", "mixed", 4, 3, 1),
    }

    def fake_extract(path, ocr_client, min_chars, dpi, min_image_area=0.05):
        return results[path.name]

    monkeypatch.setattr(pipeline, "extract_pdf", fake_extract)

    report = pipeline.run_extract([pdf_native, pdf_heavy, pdf_light])

    assert report.processed == [
        "heavy.pdf",
        "light.pdf",
        "native.pdf [native — minimal review]",
    ]
    assert any("heavy.pdf: 4/4 pages from OCR" in w for w in report.warnings)
    assert any("light.pdf: 1/4 pages from OCR" in w for w in report.warnings)
    assert not any("native.pdf" in w and "OCR" in w for w in report.warnings)


def test_extract_image_ocr_warnings(tmp_path, monkeypatch):
    pdf_imgs = make_pdf(tmp_path, "imgs.pdf")
    pdf_skip = make_pdf(tmp_path, "skip.pdf")

    results = {
        "imgs.pdf": ExtractionResult("body", "mixed", 3, 3, 0, images_ocr=2),
        "skip.pdf": ExtractionResult("body", "native", 3, 3, 0, images_skipped=1),
    }

    def fake_extract(path, ocr_client, min_chars, dpi, min_image_area=0.05):
        return results[path.name]

    monkeypatch.setattr(pipeline, "extract_pdf", fake_extract)

    report = pipeline.run_extract([pdf_imgs, pdf_skip])
    assert any(
        "imgs.pdf" in w and "2 embedded image(s) OCR'd" in w for w in report.warnings
    )
    assert any(
        "skip.pdf" in w and "1 embedded image(s) skipped" in w for w in report.warnings
    )


def test_extract_failure_isolated(tmp_path, monkeypatch):
    pdf_ok = make_pdf(tmp_path, "ok.pdf")
    pdf_bad = make_pdf(tmp_path, "bad.pdf")

    def fake_extract(path, ocr_client, min_chars, dpi, min_image_area=0.05):
        if path.name == "bad.pdf":
            raise ValueError("scanned page(s) [1] have no text layer")
        return ExtractionResult("ok body", "native", 1, 1, 0)

    monkeypatch.setattr(pipeline, "extract_pdf", fake_extract)

    report = pipeline.run_extract([pdf_ok, pdf_bad])
    assert "bad.pdf" in report.failed
    assert "scanned page" in report.failed["bad.pdf"]
    assert len(report.processed) == 1
    assert "ok.pdf" in report.processed[0]


def test_extract_metadata_prefill_failure_still_writes_sidecar(tmp_path, monkeypatch):
    pdf = make_pdf(tmp_path, "surat.pdf")

    def fake_extract(path, ocr_client, min_chars, dpi, min_image_area=0.05):
        return ExtractionResult("body text", "native", 1, 1, 0)

    monkeypatch.setattr(pipeline, "extract_pdf", fake_extract)

    def failing_llm_extract(markdown, caller):
        raise RuntimeError("model unavailable")

    monkeypatch.setattr(pipeline, "llm_extract", failing_llm_extract)

    report = pipeline.run_extract([pdf])
    assert any("metadata pre-fill failed" in w for w in report.warnings)

    meta, body = read_sidecar(tmp_path / "surat.corpus.md")
    assert meta["doc_type"] is None
    assert meta["doc_level"] is None
    assert "surat.pdf" not in report.failed
    assert len(report.processed) == 1


def test_extract_no_metadata_model_falls_back_to_ocr_image(tmp_path, monkeypatch):
    """metadata_model="" -> page-1 image sent to the (healthy) OCR client."""
    cfg = dict(DEFAULT_CFG)
    cfg["metadata_model"] = ""
    monkeypatch.setattr(pipeline.Config, "get_corpus_config", lambda: cfg)

    pdf = make_pdf(tmp_path, "surat.pdf")

    def fake_extract(path, ocr_client, min_chars, dpi, min_image_area=0.05):
        return ExtractionResult("body text", "native", 1, 1, 0)

    monkeypatch.setattr(pipeline, "extract_pdf", fake_extract)

    report = pipeline.run_extract([pdf])
    meta, _body = read_sidecar(tmp_path / "surat.corpus.md")
    assert meta["doc_type"] == "surat"
    assert meta["doc_level"] == "wk"
    assert report.failed == {}


def test_extract_no_metadata_model_no_ocr_skips_with_warning(tmp_path, monkeypatch):
    cfg = dict(DEFAULT_CFG)
    cfg["metadata_model"] = ""
    monkeypatch.setattr(pipeline.Config, "get_corpus_config", lambda: cfg)
    monkeypatch.setattr(pipeline, "OllamaVisionOcr", lambda *a, **kw: FakeOcr(False))

    pdf = make_pdf(tmp_path, "surat.pdf")

    def fake_extract(path, ocr_client, min_chars, dpi, min_image_area=0.05):
        assert ocr_client is None
        return ExtractionResult("body text", "native", 1, 1, 0)

    monkeypatch.setattr(pipeline, "extract_pdf", fake_extract)

    report = pipeline.run_extract([pdf])
    meta, _body = read_sidecar(tmp_path / "surat.corpus.md")
    assert meta["doc_type"] is None
    assert any("pre-fill skipped" in w for w in report.warnings)


# --------------------------------------------------------------------------
# _resolve_text_caller
# --------------------------------------------------------------------------


class FakeLLM:
    def invoke(self, prompt):
        class R:
            content = '{"doc_type": "mom"}'

        return R()


def test_resolve_text_caller_empty_returns_none():
    assert pipeline._resolve_text_caller("") is None
    assert pipeline._resolve_text_caller(None) is None


def test_resolve_text_caller_main_uses_provider(monkeypatch):
    import esdc.providers as providers

    monkeypatch.setattr(
        pipeline.Config, "get_provider_config", lambda: {"provider_type": "openai"}
    )
    monkeypatch.setattr(providers, "create_llm_from_config", lambda cfg: FakeLLM())

    caller = pipeline._resolve_text_caller("main")
    assert caller is not None
    assert caller("extract metadata") == '{"doc_type": "mom"}'


def test_resolve_text_caller_main_without_provider_returns_none(monkeypatch):
    monkeypatch.setattr(pipeline.Config, "get_provider_config", lambda: None)
    assert pipeline._resolve_text_caller("main") is None


def test_resolve_text_caller_ollama_name(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(pipeline, "_text_llm_caller", lambda model, host=None: sentinel)
    assert pipeline._resolve_text_caller("qwen3:8b") is sentinel


def test_prefill_metadata_main_model(monkeypatch, tmp_path):
    """metadata_model='main' routes prefill through the provider LLM."""
    import esdc.providers as providers

    monkeypatch.setattr(
        pipeline.Config, "get_provider_config", lambda: {"provider_type": "openai"}
    )
    monkeypatch.setattr(providers, "create_llm_from_config", lambda cfg: FakeLLM())

    cfg = dict(DEFAULT_CFG)
    cfg["metadata_model"] = "main"
    monkeypatch.setattr(pipeline.Config, "get_corpus_config", lambda: cfg)
    # patch_seams stubs llm_extract; restore the real one to prove the
    # provider caller's JSON flows through parsing.
    from esdc.corpus.metadata import llm_extract as real_llm_extract

    monkeypatch.setattr(pipeline, "llm_extract", real_llm_extract)

    pdf = make_pdf(tmp_path, "surat.pdf")

    def fake_extract(path, ocr_client, min_chars, dpi, min_image_area=0.05):
        return ExtractionResult("# Surat\nisi", "native", 1, 1, 0)

    monkeypatch.setattr(pipeline, "extract_pdf", fake_extract)

    report = pipeline.run_extract([pdf])
    assert report.failed == {}
    meta, _body = read_sidecar(tmp_path / "surat.corpus.md")
    assert meta["doc_type"] == "mom"


# --------------------------------------------------------------------------
# run_extract cleanup wiring
# --------------------------------------------------------------------------


def _extract_body(tmp_path, monkeypatch, cfg_overrides, markdown):
    cfg = dict(DEFAULT_CFG)
    cfg.update(cfg_overrides)
    monkeypatch.setattr(pipeline.Config, "get_corpus_config", lambda: cfg)

    pdf = make_pdf(tmp_path, "doc.pdf")

    def fake_extract(path, ocr_client, min_chars, dpi, min_image_area=0.05):
        return ExtractionResult(markdown, "native", 1, 1, 0)

    monkeypatch.setattr(pipeline, "extract_pdf", fake_extract)
    report = pipeline.run_extract([pdf])
    _meta, body = read_sidecar(tmp_path / "doc.corpus.md")
    return report, body


def test_extract_cleanup_applied_and_warned(tmp_path, monkeypatch):
    def fake_caller(prompt):
        return "# Judul\nbaris rapi"

    monkeypatch.setattr(
        pipeline,
        "_resolve_text_caller",
        lambda spec, host=None: fake_caller if spec == "main" else None,
    )
    report, body = _extract_body(
        tmp_path,
        monkeypatch,
        {"cleanup_model": "main", "metadata_model": ""},
        "<!-- page 1: native -->\n# **Judul**\nbaris jelek",
    )
    assert "baris rapi" in body
    assert "baris jelek" not in body
    assert any("1 page segment(s) reformatted" in w for w in report.warnings)


def test_extract_cleanup_off_by_default(tmp_path, monkeypatch):
    report, body = _extract_body(
        tmp_path,
        monkeypatch,
        {"cleanup_model": "", "metadata_model": ""},
        "<!-- page 1: native -->\n# **Judul**\nbaris jelek",
    )
    assert "baris jelek" in body
    assert not any("reformatted" in w for w in report.warnings)


def test_extract_cleanup_guard_rejection_warned(tmp_path, monkeypatch):
    def bad_caller(prompt):
        return "angka palsu 12345"

    monkeypatch.setattr(
        pipeline,
        "_resolve_text_caller",
        lambda spec, host=None: bad_caller if spec == "main" else None,
    )
    report, body = _extract_body(
        tmp_path,
        monkeypatch,
        {"cleanup_model": "main", "metadata_model": ""},
        "<!-- page 1: native -->\n# **Judul**\nbaris asli tanpa angka",
    )
    assert "baris asli tanpa angka" in body  # original kept
    assert any("failed cleanup guard" in w for w in report.warnings)


# --------------------------------------------------------------------------
# _apply_entity_resolution
# --------------------------------------------------------------------------


def test_apply_entity_resolution_none_stays_none_without_warning():
    report = pipeline.CorpusReport()
    meta = {"wk_name": None, "field_name": None, "project_name": None}
    resolver = FakeEntityResolver()

    pipeline._apply_entity_resolution(meta, resolver, report, "doc.pdf")

    assert meta["wk_name"] is None
    assert meta["field_name"] is None
    assert meta["project_name"] is None
    assert report.warnings == []
    assert "entity_warnings" not in meta


def test_apply_entity_resolution_unresolved_name_kept_with_warning():
    report = pipeline.CorpusReport()
    meta = {"wk_name": ["Nowhere Area"], "field_name": None, "project_name": None}
    resolver = FakeEntityResolver()  # no matches configured -> everything unresolved

    pipeline._apply_entity_resolution(meta, resolver, report, "doc.pdf")

    assert meta["wk_name"] == ["Nowhere Area"]  # kept, not dropped to None
    assert any(
        "wk_name 'Nowhere Area' unresolved — kept as-is, verify manually" in w
        for w in report.warnings
    )
    assert meta["entity_warnings"] == [
        "wk_name 'Nowhere Area' unresolved — kept as-is, verify manually"
    ]


def test_apply_entity_resolution_multiple_matches_all_kept():
    """resolver.resolve_name returns FINAL picks -- one raw name can legitimately
    map to several canonical rows (e.g. "Arung Nowera" -> two separate fields),
    so _apply_entity_resolution no longer collapses to a single best match.
    """
    report = pipeline.CorpusReport()
    meta = {"wk_name": ["Rokan"], "field_name": None, "project_name": None}

    class MultiMatchResolver:
        def resolve_name(self, name, entity_type):
            return [
                {"entity_type": "wk_name", "name": "WK Rokan", "confidence": 0.85},
                {"entity_type": "wk_name", "name": "Rokan", "confidence": 1.0},
            ]

    pipeline._apply_entity_resolution(meta, MultiMatchResolver(), report, "doc.pdf")

    assert meta["wk_name"] == ["WK Rokan", "Rokan"]  # every match kept, in order
    assert any(
        "wk_name 'Rokan' -> 'WK Rokan' (confidence 0.85)" in w for w in report.warnings
    )
    # the confidence==1.0 match doesn't get its own warning
    assert not any("-> 'Rokan'" in w for w in report.warnings)


def test_apply_entity_resolution_stale_warnings_cleared_on_clean_rerun():
    """A sidecar with leftover entity_warnings from a prior (buggy/unresolved)
    run must have them cleared once re-resolution comes back clean.
    """
    report = pipeline.CorpusReport()
    meta = {
        "wk_name": ["South Sumatera"],
        "field_name": None,
        "project_name": None,
        "entity_warnings": ["field_name 'Arung Nowera' unresolved — stale"],
    }
    resolver = FakeEntityResolver(
        matches={
            "South Sumatera": {
                "entity_type": "wk_name",
                "name": "South Sumatera",
                "confidence": 1.0,
            }
        }
    )

    pipeline._apply_entity_resolution(meta, resolver, report, "doc.pdf")

    assert "entity_warnings" not in meta


def test_apply_entity_resolution_no_warnings_key_absent():
    report = pipeline.CorpusReport()
    meta = {"wk_name": ["Rokan"], "field_name": None, "project_name": None}
    resolver = FakeEntityResolver(
        matches={"Rokan": {"entity_type": "wk_name", "name": "Rokan", "confidence": 1.0}}
    )

    pipeline._apply_entity_resolution(meta, resolver, report, "doc.pdf")

    assert "entity_warnings" not in meta
    assert report.warnings == []


# --------------------------------------------------------------------------
# run_extract entity resolution
# --------------------------------------------------------------------------


def test_extract_resolves_entities_with_canonical_names_and_raw_entities(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        pipeline,
        "llm_extract",
        lambda markdown, caller: {
            "doc_type": "surat",
            "doc_level": "wk",
            "wk_name": "Rokann",
        },
    )
    fake_resolver = FakeEntityResolver(
        matches={"Rokann": {"entity_type": "wk_name", "name": "Rokan", "confidence": 0.9}}
    )
    monkeypatch.setattr(
        pipeline, "_entity_resolver_or_none", lambda store, report: fake_resolver
    )

    pdf = make_pdf(tmp_path, "surat.pdf")

    def fake_extract(path, ocr_client, min_chars, dpi, min_image_area=0.05):
        return ExtractionResult("# Surat\nisi", "native", 1, 1, 0)

    monkeypatch.setattr(pipeline, "extract_pdf", fake_extract)

    report = pipeline.run_extract([pdf])
    assert report.failed == {}

    meta, _body = read_sidecar(tmp_path / "surat.corpus.md")
    assert meta["wk_name"] == ["Rokan"]
    assert meta["raw_entities"]["wk_name"] == ["Rokann"]
    assert any("confidence" in w for w in report.warnings)


def test_extract_no_resolver_keeps_raw_names_with_skip_warning(tmp_path, monkeypatch):
    monkeypatch.setattr(
        pipeline,
        "llm_extract",
        lambda markdown, caller: {
            "doc_type": "surat",
            "doc_level": "wk",
            "wk_name": "Rokan",
        },
    )
    monkeypatch.setattr(
        pipeline, "_entity_resolver_or_none", lambda store, report: None
    )

    pdf = make_pdf(tmp_path, "surat.pdf")

    def fake_extract(path, ocr_client, min_chars, dpi, min_image_area=0.05):
        return ExtractionResult("# Surat\nisi", "native", 1, 1, 0)

    monkeypatch.setattr(pipeline, "extract_pdf", fake_extract)

    pipeline.run_extract([pdf])
    meta, _body = read_sidecar(tmp_path / "surat.corpus.md")
    assert meta["wk_name"] == ["Rokan"]  # raw name kept as-is
    assert "raw_entities" not in meta


def test_extract_entity_resolver_or_none_skips_when_tables_missing(tmp_path, monkeypatch):
    """No lookup tables in the (fresh, empty) test DB -> resolution skipped once."""
    pdf = make_pdf(tmp_path, "surat.pdf")

    def fake_extract(path, ocr_client, min_chars, dpi, min_image_area=0.05):
        return ExtractionResult("# Surat\nisi", "native", 1, 1, 0)

    monkeypatch.setattr(pipeline, "extract_pdf", fake_extract)

    report = pipeline.run_extract([pdf])
    skip_warnings = [w for w in report.warnings if "entity resolution skipped" in w]
    assert len(skip_warnings) == 1
    assert "esdc fetch" in skip_warnings[0]


# --------------------------------------------------------------------------
# run_extract CLI overrides
# --------------------------------------------------------------------------


def test_extract_cli_override_beats_prefill(tmp_path, monkeypatch):
    monkeypatch.setattr(
        pipeline,
        "llm_extract",
        lambda markdown, caller: {
            "doc_type": "surat",
            "doc_level": "unknown",
            "wk_name": "LLM WK",
            "subject": "LLM Subject",
        },
    )

    pdf = make_pdf(tmp_path, "surat.pdf")

    def fake_extract(path, ocr_client, min_chars, dpi, min_image_area=0.05):
        return ExtractionResult("# Surat\nisi", "native", 1, 1, 0)

    monkeypatch.setattr(pipeline, "extract_pdf", fake_extract)

    report = pipeline.run_extract(
        [pdf], wk_name="Rokan", level="wk", project_name="POD Duri"
    )
    assert report.failed == {}

    meta, _body = read_sidecar(tmp_path / "surat.corpus.md")
    assert meta["wk_name"] == ["Rokan"]
    assert meta["doc_level"] == "wk"
    assert meta["project_name"] == ["POD Duri"]
    # non-overridden field keeps the LLM-prefilled value
    assert meta["subject"] == "LLM Subject"


def test_extract_override_none_keeps_prefill(tmp_path, monkeypatch):
    monkeypatch.setattr(
        pipeline,
        "llm_extract",
        lambda markdown, caller: {
            "doc_type": "surat",
            "doc_level": "wk",
            "wk_name": "LLM WK",
        },
    )

    pdf = make_pdf(tmp_path, "surat.pdf")

    def fake_extract(path, ocr_client, min_chars, dpi, min_image_area=0.05):
        return ExtractionResult("# Surat\nisi", "native", 1, 1, 0)

    monkeypatch.setattr(pipeline, "extract_pdf", fake_extract)

    report = pipeline.run_extract([pdf])
    assert report.failed == {}

    meta, _body = read_sidecar(tmp_path / "surat.corpus.md")
    assert meta["wk_name"] == ["LLM WK"]
    assert meta["doc_level"] == "wk"
    assert meta["doc_type"] == "surat"


def test_extract_override_goes_through_entity_resolution(tmp_path, monkeypatch):
    monkeypatch.setattr(
        pipeline,
        "llm_extract",
        lambda markdown, caller: {"doc_type": "surat", "doc_level": "wk"},
    )
    fake_resolver = FakeEntityResolver(
        matches={
            "Rokan": {"entity_type": "wk_name", "name": "Rokan", "confidence": 0.9}
        }
    )
    monkeypatch.setattr(
        pipeline, "_entity_resolver_or_none", lambda store, report: fake_resolver
    )

    pdf = make_pdf(tmp_path, "surat.pdf")

    def fake_extract(path, ocr_client, min_chars, dpi, min_image_area=0.05):
        return ExtractionResult("# Surat\nisi", "native", 1, 1, 0)

    monkeypatch.setattr(pipeline, "extract_pdf", fake_extract)

    report = pipeline.run_extract([pdf], wk_name="Rokan")
    assert report.failed == {}

    meta, _body = read_sidecar(tmp_path / "surat.corpus.md")
    assert meta["wk_name"] == ["Rokan"]
    assert meta["raw_entities"]["wk_name"] == ["Rokan"]
    assert any("confidence" in w for w in report.warnings)


# --------------------------------------------------------------------------
# run_commit
# --------------------------------------------------------------------------


def test_commit_pending_review_skipped_reviewed_ingested(tmp_path, monkeypatch):
    store = make_store(tmp_path)
    store.ensure_tables()
    patch_store_factory(monkeypatch, store)
    patch_entity_resolver(monkeypatch)

    make_sidecar(tmp_path, "pending.pdf", reviewed=False, file_hash="aa" * 32)
    make_sidecar(tmp_path, "ready.pdf", reviewed=True, file_hash="bb" * 32)

    report = pipeline.run_commit([tmp_path])

    assert report.skipped == ["pending.corpus.md (pending review)"]
    assert report.processed == ["ready.corpus.md"]
    assert store.document_exists("bb" * 32) is True
    store.close()


def test_commit_skip_review_ingests_pending_and_writes_back(tmp_path, monkeypatch):
    store = make_store(tmp_path)
    store.ensure_tables()
    patch_store_factory(monkeypatch, store)
    patch_entity_resolver(monkeypatch)
    sc = make_sidecar(tmp_path, "pending.pdf", reviewed=False, file_hash="aa" * 32,
                      subject="orig subject")

    report = pipeline.run_commit([tmp_path], skip_review=True)

    assert report.processed == ["pending.corpus.md"]
    assert report.skipped == []
    assert any("ingested without review" in w for w in report.warnings)
    meta, body = read_sidecar(sc)
    assert meta["reviewed"] is True          # flipped
    assert meta["subject"] == "orig subject" # everything else untouched
    store.close()


def test_commit_skip_review_dry_run_no_write_back(tmp_path, monkeypatch):
    store = make_store(tmp_path)
    store.ensure_tables()
    patch_store_factory(monkeypatch, store)
    patch_entity_resolver(monkeypatch)
    sc = make_sidecar(tmp_path, "pending.pdf", reviewed=False, file_hash="aa" * 32,
                      subject="orig subject")

    report = pipeline.run_commit([tmp_path], skip_review=True, dry_run=True)
    assert report.processed == ["pending.corpus.md"]
    meta, _ = read_sidecar(sc)
    assert meta["reviewed"] is False         # dry run never touches the file
    store.close()


def test_commit_skip_review_reviewed_sidecar_not_rewritten(tmp_path, monkeypatch):
    store = make_store(tmp_path)
    store.ensure_tables()
    patch_store_factory(monkeypatch, store)
    patch_entity_resolver(monkeypatch)
    sc = make_sidecar(tmp_path, "ready.pdf", reviewed=True, file_hash="bb" * 32)

    mtime_before = sc.stat().st_mtime_ns
    report = pipeline.run_commit([tmp_path], skip_review=True)
    assert report.processed == ["ready.corpus.md"]
    assert not any("ingested without review" in w for w in report.warnings)
    assert sc.stat().st_mtime_ns == mtime_before  # file untouched
    store.close()


def test_commit_without_skip_review_still_skips_pending(tmp_path, monkeypatch):
    store = make_store(tmp_path)
    store.ensure_tables()
    patch_store_factory(monkeypatch, store)
    patch_entity_resolver(monkeypatch)
    make_sidecar(tmp_path, "pending.pdf", reviewed=False, file_hash="aa" * 32)

    report = pipeline.run_commit([tmp_path])
    assert report.skipped == ["pending.corpus.md (pending review)"]
    store.close()


def test_commit_dedupe_and_force(tmp_path, monkeypatch):
    store = make_store(tmp_path)
    store.ensure_tables()
    patch_store_factory(monkeypatch, store)
    patch_entity_resolver(monkeypatch)

    make_sidecar(
        tmp_path,
        "doc.pdf",
        reviewed=True,
        file_hash="cc" * 32,
        body="# Doc\nisi pertama yang lumayan panjang",
    )

    report1 = pipeline.run_commit([tmp_path])
    assert report1.processed == ["doc.corpus.md"]
    assert store.counts()["documents"] == 1

    report2 = pipeline.run_commit([tmp_path])
    assert report2.skipped == ["doc.corpus.md (already committed)"]
    assert store.counts()["documents"] == 1

    report3 = pipeline.run_commit([tmp_path], force=True)
    assert report3.processed == ["doc.corpus.md"]
    assert store.counts()["documents"] == 1
    store.close()


def test_commit_cli_override_beats_frontmatter(tmp_path, monkeypatch):
    store = make_store(tmp_path)
    store.ensure_tables()
    patch_store_factory(monkeypatch, store)
    patch_entity_resolver(monkeypatch)

    file_hash = "dd" * 32
    make_sidecar(tmp_path, "doc.pdf", reviewed=True, file_hash=file_hash, doc_type="mom")

    pipeline.run_commit([tmp_path], doc_type="surat")

    doc = store.get_document(file_hash[:16])
    assert doc["doc_type"] == "surat"
    store.close()


def test_commit_entity_resolution(tmp_path, monkeypatch):
    store = make_store(tmp_path)
    store.ensure_tables()
    patch_store_factory(monkeypatch, store)
    patch_entity_resolver(
        monkeypatch,
        matches={
            "Rokan": {
                "entity_type": "wk_name",
                "name": "Rokan",
                "confidence": 1.0,
            },
            "Duri": {
                "entity_type": "field_name",
                "name": "Duri",
                "confidence": 0.8,
            },
        },
    )

    file_hash = "ee" * 32
    make_sidecar(
        tmp_path,
        "doc.pdf",
        reviewed=True,
        file_hash=file_hash,
        wk_name="Rokan",
        field_name="Duriii Field Area",
        project_name="Unknown Project XYZ",
    )

    report = pipeline.run_commit([tmp_path])

    assert not any("wk_name" in w for w in report.warnings)
    assert any("field_name" in w and "confidence" in w for w in report.warnings)
    assert any("project_name" in w and "unresolved" in w for w in report.warnings)

    doc = store.get_document(file_hash[:16])
    assert doc["wk_name"] == ["Rokan"]
    # Unresolved names are kept as-is for manual review, never dropped.
    assert doc["project_name"] == ["Unknown Project XYZ"]
    store.close()


def test_commit_entity_warnings_and_raw_entities_dont_leak_into_doc_columns(
    tmp_path, monkeypatch
):
    store = make_store(tmp_path)
    store.ensure_tables()
    patch_store_factory(monkeypatch, store)
    patch_entity_resolver(monkeypatch)  # no matches -> everything unresolved

    file_hash = "12" * 32
    make_sidecar(
        tmp_path,
        "doc.pdf",
        reviewed=True,
        file_hash=file_hash,
        wk_name="Some Area",
        raw_entities={"wk_name": ["Some Area"], "field_name": None, "project_name": None},
    )

    pipeline.run_commit([tmp_path])

    doc = store.get_document(file_hash[:16])
    assert "entity_warnings" not in doc
    assert "entity_warnings" not in (doc.get("metadata") or {})
    # sidecar-authored raw_entities is preferred over the commit-time resolve.
    assert doc["raw_entities"] == {
        "wk_name": ["Some Area"],
        "field_name": None,
        "project_name": None,
    }
    store.close()


def test_commit_already_committed_skips_before_resolution(tmp_path, monkeypatch):
    """Skip-on-already-committed must short-circuit before resolution runs.

    Re-commits of already-ingested docs should never re-invoke the resolver.
    """
    store = make_store(tmp_path)
    store.ensure_tables()
    patch_store_factory(monkeypatch, store)

    class CountingResolver:
        def __init__(self):
            self.calls = 0

        def resolve_name(self, name, entity_type):
            self.calls += 1
            return []

    counting_resolver = CountingResolver()
    monkeypatch.setattr(pipeline, "EntityResolver", lambda db: counting_resolver)

    file_hash = "13" * 32
    make_sidecar(
        tmp_path, "doc.pdf", reviewed=True, file_hash=file_hash, wk_name="Rokan"
    )

    report1 = pipeline.run_commit([tmp_path])
    assert report1.processed == ["doc.corpus.md"]
    calls_after_first = counting_resolver.calls
    assert calls_after_first > 0

    report2 = pipeline.run_commit([tmp_path])
    assert report2.skipped == ["doc.corpus.md (already committed)"]
    assert counting_resolver.calls == calls_after_first  # unchanged: not re-invoked
    store.close()


def test_commit_dry_run_writes_nothing(tmp_path, monkeypatch):
    store = make_store(tmp_path)
    store.ensure_tables()
    patch_store_factory(monkeypatch, store)
    patch_entity_resolver(monkeypatch)

    insert_calls = []
    orig_insert = store.insert_document

    def spy_insert(doc, chunks):
        insert_calls.append(doc)
        orig_insert(doc, chunks)

    monkeypatch.setattr(store, "insert_document", spy_insert)

    make_sidecar(tmp_path, "doc.pdf", reviewed=True, file_hash="ff" * 32)

    report = pipeline.run_commit([tmp_path], dry_run=True)

    assert report.processed == ["doc.corpus.md"]
    assert insert_calls == []
    assert store.counts()["documents"] == 0
    store.close()


def test_commit_bad_sidecar_isolated(tmp_path, monkeypatch):
    store = make_store(tmp_path)
    store.ensure_tables()
    patch_store_factory(monkeypatch, store)
    patch_entity_resolver(monkeypatch)

    bad = tmp_path / "bad.corpus.md"
    bad.write_text("no frontmatter at all")

    make_sidecar(tmp_path, "good.pdf", reviewed=True, file_hash="11" * 32)

    report = pipeline.run_commit([tmp_path])

    assert "bad.corpus.md" in report.failed
    assert report.processed == ["good.corpus.md"]
    store.close()


# --------------------------------------------------------------------------
# run_status
# --------------------------------------------------------------------------


def test_status_all_states(tmp_path, monkeypatch):
    store = make_store(tmp_path)
    store.ensure_tables()
    patch_store_factory(monkeypatch, store)

    (tmp_path / "raw.pdf").write_bytes(b"%PDF-1.4 fake")

    make_sidecar(tmp_path, "pending.pdf", reviewed=False, file_hash="22" * 32)
    make_sidecar(tmp_path, "ready.pdf", reviewed=True, file_hash="33" * 32)
    make_sidecar(tmp_path, "done.pdf", reviewed=True, file_hash="44" * 32)

    store.insert_document(
        {
            "doc_id": ("44" * 32)[:16],
            "file_name": "done.pdf",
            "file_path": "x",
            "file_hash": "44" * 32,
            "markdown": "isi",
            "extraction_method": "native",
        },
        [Chunk(0, None, "isi dokumen")],
    )

    results = {r["file"]: r["state"] for r in pipeline.run_status([tmp_path])}

    assert results["raw.pdf"] == "not extracted"
    assert results["pending.corpus.md"] == "pending review"
    assert results["ready.corpus.md"] == "ready to commit"
    assert results["done.corpus.md"] == "committed"
    store.close()


def test_status_tolerates_store_error(tmp_path, monkeypatch):
    def broken_store(*a, **kw):
        raise RuntimeError("db locked")

    monkeypatch.setattr(pipeline, "CorpusStore", broken_store)
    make_sidecar(tmp_path, "ready.pdf", reviewed=True, file_hash="55" * 32)

    results = {r["file"]: r["state"] for r in pipeline.run_status([tmp_path])}
    assert results["ready.corpus.md"] == "unknown (db error)"


# --------------------------------------------------------------------------
# run_reembed
# --------------------------------------------------------------------------


def test_reembed_updates_meta_and_chunks_with_failure_isolation(tmp_path, monkeypatch):
    store1 = make_store(tmp_path, embedder=FakeEmbedder())
    store1.ensure_tables()
    doc_a = {
        "doc_id": "aaaa111122223333",
        "file_name": "a.pdf",
        "file_path": "x",
        "file_hash": "aa" * 32,
        "markdown": "# A\nisi dokumen a yang lumayan panjang",
        "extraction_method": "native",
    }
    doc_b = {
        "doc_id": "bbbb111122223333",
        "file_name": "b.pdf",
        "file_path": "x",
        "file_hash": "bb" * 32,
        "markdown": "# B\nisi dokumen b yang lumayan panjang",
        "extraction_method": "native",
    }
    store1.insert_document(doc_a, [Chunk(0, None, "isi dokumen a")])
    store1.insert_document(doc_b, [Chunk(0, None, "isi dokumen b")])
    store1.close()

    store2 = CorpusStore(db_path=tmp_path / "corpus.duckdb", embedder=FakeEmbedder2())
    monkeypatch.setattr(pipeline, "CorpusStore", lambda *a, **kw: store2)

    orig_replace = store2.replace_chunks

    def flaky_replace(doc_id, chunks):
        if doc_id == doc_b["doc_id"]:
            raise RuntimeError("embedding backend down")
        return orig_replace(doc_id, chunks)

    monkeypatch.setattr(store2, "replace_chunks", flaky_replace)

    report = pipeline.run_reembed()

    assert "a.pdf" in report.processed
    assert "b.pdf" in report.failed

    updated_a = store2.get_document(doc_a["doc_id"])
    assert updated_a["embedding_model"] == "fake-embed-v2"
    store2.close()


def test_run_commit_fails_sidecar_missing_file_hash(tmp_path, monkeypatch):
    store = make_store(tmp_path)
    store.ensure_tables()
    patch_store_factory(monkeypatch, store)
    patch_entity_resolver(monkeypatch)

    # Write a sidecar WITHOUT file_hash in its frontmatter
    pdf = tmp_path / "nohash.pdf"
    meta = {
        "source_file": "nohash.pdf",
        "page_count": 1,
        "extraction_method": "native",
        "reviewed": True,
        "doc_type": "surat",
        "doc_number": "1",
        "doc_date": "2026-01-01",
        "subject": "test",
        "sender": "A",
        "recipient": "B",
        "doc_level": "wk",
    }
    write_sidecar(pdf, meta, "# Doc\ncontent here")

    report = pipeline.run_commit([tmp_path])

    assert any("file_hash" in msg for msg in report.failed.values())
    assert not report.processed
    store.close()


# --------------------------------------------------------------------------
# progress bars
# --------------------------------------------------------------------------


def test_progress_with_status_helper_builds_and_updates():
    with pipeline._progress_with_status("extract", 2, "files") as p:
        p.file("surat.pdf")
        p.status("parsing PDF")
        p.advance()
        p.status("resolve entities")
        p.advance()
    # No assertion beyond "did not raise" + both tasks completed.
    # (Rendering correctness is visual; this pins the contract/signature.)


class FakeProgressHandle:
    """Records file()/status()/advance() calls for assertions."""

    def __init__(self):
        self.files = []
        self.statuses = []
        self.advances = 0

    def file(self, name):
        self.files.append(name)

    def status(self, phase):
        self.statuses.append(phase)

    def advance(self):
        self.advances += 1


def fake_progress_with_status_factory(captured):
    @contextmanager
    def factory(verb, total, unit):
        handle = FakeProgressHandle()
        captured.update(verb=verb, total=total, unit=unit, handle=handle)
        yield handle
    return factory


def test_extract_shows_progress_bar(tmp_path, monkeypatch):
    captured = {}
    monkeypatch.setattr(
        pipeline, "_progress_with_status", fake_progress_with_status_factory(captured)
    )
    pdf = make_pdf(tmp_path, "surat.pdf")

    def fake_extract(path, ocr_client, min_chars, dpi, min_image_area=0.05):
        return ExtractionResult("# Surat\nisi", "native", 1, 1, 0)

    monkeypatch.setattr(pipeline, "extract_pdf", fake_extract)

    report = pipeline.run_extract([pdf])
    assert report.failed == {}
    assert captured["verb"] == "extract"
    assert captured["total"] == 1
    assert captured["unit"] == "files"
    handle = captured["handle"]
    assert handle.files == ["surat.pdf"]
    assert handle.advances == 1
    # phase-level status updates flowed through
    assert "parsing PDF" in handle.statuses
    assert "prefill metadata" in handle.statuses
    assert "write sidecar" in handle.statuses


def test_commit_shows_progress_bar(tmp_path, monkeypatch):
    captured = {}
    monkeypatch.setattr(
        pipeline, "_progress_with_status", fake_progress_with_status_factory(captured)
    )
    store = make_store(tmp_path)
    store.ensure_tables()
    patch_store_factory(monkeypatch, store)
    patch_entity_resolver(monkeypatch)
    make_sidecar(tmp_path, "doc.pdf", reviewed=True, file_hash="dd" * 32)

    report = pipeline.run_commit([tmp_path])
    assert report.processed == ["doc.corpus.md"]
    assert captured["verb"] == "commit"
    assert captured["unit"] == "docs"
    assert captured["total"] == 1
    handle = captured["handle"]
    assert handle.files == ["doc.corpus.md"]
    assert handle.advances == 1
    assert "resolve entities" in handle.statuses
    assert "embed + insert" in handle.statuses
    store.close()


def test_extract_passes_ollama_host_to_ocr_and_text_caller(tmp_path, monkeypatch):
    cfg = dict(DEFAULT_CFG)
    cfg["ollama_host"] = "http://gpu-box:11434"
    cfg["metadata_model"] = "qwen3:8b"
    monkeypatch.setattr(pipeline.Config, "get_corpus_config", lambda: cfg)

    ocr_kwargs = {}
    monkeypatch.setattr(
        pipeline,
        "OllamaVisionOcr",
        lambda *a, **kw: (ocr_kwargs.update(kw), FakeOcr(True))[1],
    )
    caller_hosts = []
    monkeypatch.setattr(
        pipeline,
        "_text_llm_caller",
        lambda model, host=None: caller_hosts.append(host) or (lambda p: "{}"),
    )

    pdf = make_pdf(tmp_path, "surat.pdf")

    def fake_extract(path, ocr_client, min_chars, dpi, min_image_area=0.05):
        return ExtractionResult("# Surat\nisi", "native", 1, 1, 0)

    monkeypatch.setattr(pipeline, "extract_pdf", fake_extract)

    pipeline.run_extract([pdf])
    assert ocr_kwargs.get("host") == "http://gpu-box:11434"
    assert caller_hosts == ["http://gpu-box:11434"]
