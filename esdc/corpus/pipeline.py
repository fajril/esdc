"""Extract/commit pipeline orchestration for `esdc corpus`.

Two-step human-in-the-loop flow:

1. ``run_extract`` — PDFs -> reviewable ``.corpus.md`` sidecars (native/OCR
   tiered text + LLM-prefilled, unreviewed metadata).
2. ``run_commit`` — reviewed sidecars (``reviewed: true``) -> chunked,
   embedded rows in ``CorpusStore``.

``run_status`` reports where files sit in that pipeline; ``run_reembed``
rebuilds chunk embeddings after an embedding-model change.

Every function returns/uses ``CorpusReport`` (or a list of dicts for
status) as the single source of truth the CLI prints from; one item
failing never aborts the batch.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import fitz
import ollama

from esdc.chat.domain_knowledge.entity_resolver_lib import EntityResolver
from esdc.configs import Config
from esdc.corpus.chunker import chunk_markdown
from esdc.corpus.extractor import extract_pdf
from esdc.corpus.metadata import (
    METADATA_PROMPT_IMAGE,
    llm_extract,
    normalize_entity_fields,
    normalize_metadata,
    parse_llm_json,
)
from esdc.corpus.ocr import OllamaVisionOcr
from esdc.corpus.sidecar import read_sidecar, sidecar_path, write_sidecar
from esdc.corpus.store import CorpusStore

logger = logging.getLogger(__name__)

# documents columns that go through canonical-entity resolution in run_commit.
ENTITY_FIELDS = ("wk_name", "field_name", "project_name")


@dataclass
class CorpusReport:
    """Batch result every pipeline function reports through; CLI-print truth."""

    processed: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    # Set by run_reembed only: the embedding model chunks were rebuilt with.
    embedding_model: str = ""


def _collect_pdfs(paths: list[Path]) -> list[Path]:
    """Resolve a mix of PDF files and directories into a sorted, deduped list."""
    found: set[Path] = set()
    for p in paths:
        if p.is_file():
            if p.suffix.lower() == ".pdf":
                found.add(p)
        elif p.is_dir():
            found.update(p.rglob("*.pdf"))
    return sorted(found)


def _collect_sidecars(paths: list[Path]) -> list[Path]:
    """Resolve a mix of ``.corpus.md`` files and directories, sorted, deduped."""
    found: set[Path] = set()
    for p in paths:
        if p.is_file():
            if p.name.endswith(".corpus.md"):
                found.add(p)
        elif p.is_dir():
            found.update(p.rglob("*.corpus.md"))
    return sorted(found)


def _text_llm_caller(model: str) -> Any:
    """A prompt->text callable backed by a text-only Ollama chat model."""
    client = ollama.Client()

    def call(prompt: str) -> str:
        response = client.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0},
        )
        return response["message"]["content"]

    return call


def _resolve_text_caller(model_spec: str | None) -> Any | None:
    """Turn a corpus model config string into a prompt->text callable.

    "" / None -> None (feature off). "main" -> the default chat provider
    (may be a cloud API — sends document text off-machine; user opt-in).
    Anything else -> a local Ollama model. Returns None when "main" is
    requested but no provider is configured or construction fails.
    """
    if not model_spec:
        return None
    if model_spec != "main":
        return _text_llm_caller(model_spec)

    provider_config = Config.get_provider_config()
    if not provider_config:
        return None
    try:
        import esdc.providers as providers

        llm = providers.create_llm_from_config(provider_config)
    except Exception as e:
        logger.warning("[Corpus] main-model caller unavailable: %s", e)
        return None

    def call(prompt: str) -> str:
        return str(llm.invoke(prompt).content)

    return call


def _prefill_metadata(
    pdf: Path,
    markdown: str,
    text_caller: Any | None,
    ocr_client: Any | None,
    cfg: dict[str, Any],
    report: CorpusReport,
    name: str,
) -> dict[str, Any]:
    """Best-effort metadata pre-fill. Never raises: failure is a warning."""
    try:
        if text_caller is not None:
            return llm_extract(markdown, text_caller)
        if ocr_client is not None:
            doc = fitz.open(str(pdf))
            try:
                pix = doc[0].get_pixmap(dpi=cfg.get("ocr_dpi", 200))
                png_bytes = pix.tobytes("png")
            finally:
                doc.close()
            raw = ocr_client.query_image(png_bytes, METADATA_PROMPT_IMAGE)
            return normalize_metadata(parse_llm_json(raw))
        report.warnings.append(
            f"{name}: no metadata model configured and OCR unavailable "
            "— metadata pre-fill skipped"
        )
        return {}
    except Exception as e:
        report.warnings.append(f"{name}: metadata pre-fill failed ({e})")
        return {}


def run_extract(paths: list[Path], force: bool = False) -> CorpusReport:
    """Parse PDFs to reviewable ``.corpus.md`` sidecars (extract/commit step 1)."""
    report = CorpusReport()
    cfg = Config.get_corpus_config()
    pdfs = _collect_pdfs(paths)

    ocr = OllamaVisionOcr(cfg["ocr_model"], num_ctx=cfg.get("num_ctx", 16384))
    ocr_client = ocr if ocr.health_check() else None

    metadata_caller = _resolve_text_caller(cfg.get("metadata_model"))
    if cfg.get("metadata_model") and metadata_caller is None:
        report.warnings.append(
            f"metadata_model '{cfg['metadata_model']}' unavailable — "
            "falling back to image-based metadata prefill"
        )

    # (name, pages_ocr, page_count) collected so the final report can be
    # sorted by OCR ratio DESC once, instead of per-file.
    entries: list[tuple[str, int, int]] = []

    for pdf in pdfs:
        name = pdf.name
        if sidecar_path(pdf).exists() and not force:
            report.skipped.append(f"{name} (sidecar exists)")
            continue

        try:
            file_hash = hashlib.sha256(pdf.read_bytes()).hexdigest()
            result = extract_pdf(
                pdf,
                ocr_client,
                cfg["min_chars_per_page"],
                cfg["ocr_dpi"],
                cfg["min_image_area"],
            )
            meta_fields = _prefill_metadata(
                pdf, result.markdown, metadata_caller, ocr_client, cfg, report, name
            )

            meta = {
                "source_file": name,
                "file_hash": file_hash,
                "page_count": result.page_count,
                "extraction_method": result.method,
                "reviewed": False,
                "doc_type": meta_fields.get("doc_type"),
                "doc_number": meta_fields.get("doc_number"),
                "doc_date": meta_fields.get("doc_date"),
                "subject": meta_fields.get("subject"),
                "sender": meta_fields.get("sender"),
                "recipient": meta_fields.get("recipient"),
                "doc_level": meta_fields.get("doc_level"),
                "wk_name": meta_fields.get("wk_name"),
                "field_name": meta_fields.get("field_name"),
                "project_name": meta_fields.get("project_name"),
                "extras": meta_fields.get("extras"),
            }
            meta = normalize_entity_fields(meta)
            write_sidecar(pdf, meta, result.markdown)

            if result.pages_ocr > 0:
                report.warnings.append(
                    f"{name}: {result.pages_ocr}/{result.page_count} pages "
                    "from OCR — review carefully"
                )
            if result.images_ocr > 0:
                report.warnings.append(
                    f"{name}: {result.images_ocr} embedded image(s) OCR'd "
                    "— review the appended table/chart text carefully"
                )
            if result.images_skipped > 0:
                report.warnings.append(
                    f"{name}: {result.images_skipped} embedded image(s) skipped "
                    "(no OCR model) — table/chart content may be missing"
                )
            entries.append((name, result.pages_ocr, result.page_count))
        except Exception as e:
            report.failed[name] = str(e)

    entries.sort(
        key=lambda e: (e[1] / e[2] if e[2] else 0.0),
        reverse=True,
    )
    for name, pages_ocr, _page_count in entries:
        report.processed.append(
            name if pages_ocr > 0 else f"{name} [native — minimal review]"
        )

    return report


def _apply_entity_resolution(
    meta: dict[str, Any],
    resolver: EntityResolver,
    report: CorpusReport,
    name: str,
) -> dict[str, Any | None]:
    """Resolve wk_name/field_name/project_name in-place on `meta`; return raw values.

    Input can be a list of names (e.g. ["ARUNG", "NOWERA"]) or a single string.
    Stores all matches with confidence >= 0.8.
    """
    min_confidence = 0.8
    raw_entities: dict[str, Any | None] = {}
    for key in ENTITY_FIELDS:
        raw = meta.get(key)
        raw_entities[key] = raw
        if raw is None:
            meta[key] = None
            continue

        # Normalize to list for uniform handling
        raw_list = raw if isinstance(raw, list) else [raw]
        all_resolved: list[str] = []
        all_warnings: list[str] = []

        for raw_name in raw_list:
            if not raw_name:
                continue
            result = resolver.resolve(str(raw_name))
            if result["status"] != "success" or not result["entities"]:
                all_warnings.append(f"{key} '{raw_name}' unresolved")
                continue
            # Find matches for this specific field type with sufficient confidence
            matches = [
                e for e in result["entities"]
                if e.get("entity_type") == key
                and e.get("confidence", 0) >= min_confidence
            ]
            if matches:
                for m in matches:
                    if m["name"] not in all_resolved:
                        all_resolved.append(m["name"])
                    if m["confidence"] < 1.0:
                        all_warnings.append(
                            f"{key} '{raw_name}' -> '{m['name']}' "
                            f"(confidence {m['confidence']})"
                        )
            else:
                all_warnings.append(f"{key} '{raw_name}' unresolved")

        meta[key] = all_resolved if all_resolved else None
        for w in all_warnings:
            report.warnings.append(f"{name}: {w}")
    return raw_entities


def run_commit(
    paths: list[Path],
    level: str | None = None,
    doc_type: str | None = None,
    wk_name: str | None = None,
    field_name: str | None = None,
    project_name: str | None = None,
    force: bool = False,
    dry_run: bool = False,
) -> CorpusReport:
    """Ingest reviewed ``.corpus.md`` sidecars into the corpus store."""
    report = CorpusReport()
    cfg = Config.get_corpus_config()
    sidecars = _collect_sidecars(paths)

    overrides = {
        "doc_level": level,
        "doc_type": doc_type,
        "wk_name": wk_name,
        "field_name": field_name,
        "project_name": project_name,
    }

    store = CorpusStore()
    any_processed = False
    try:
        # also needed to reach the conn for canonical names
        store.ensure_tables(validate_model=True)
        resolver = EntityResolver(db=store._get_connection())

        for sc in sidecars:
            name = sc.name
            try:
                meta, body = read_sidecar(sc)
            except ValueError as e:
                report.failed[name] = str(e)
                continue

            if meta.get("reviewed") is not True:
                report.skipped.append(f"{name} (pending review)")
                continue

            for key, value in overrides.items():
                if value is not None:
                    meta[key] = value
            meta = normalize_metadata(meta)
            meta = normalize_entity_fields(meta)

            raw_entities = _apply_entity_resolution(meta, resolver, report, name)

            file_hash = meta.get("file_hash")
            if not file_hash:
                report.failed[name] = (
                    "missing file_hash in sidecar frontmatter; "
                    "re-run `esdc corpus extract` to regenerate it"
                )
                continue
            exists = store.document_exists(file_hash)
            if exists and not force:
                report.skipped.append(f"{name} (already committed)")
                continue

            chunks = chunk_markdown(body, cfg["chunk_size"], cfg["chunk_overlap"])
            if not chunks:
                report.failed[name] = "no content"
                continue

            doc = {
                "doc_id": (file_hash or "")[:16],
                "file_name": meta.get("source_file"),
                "file_path": str(sc),
                "file_hash": file_hash,
                "doc_type": meta.get("doc_type"),
                "doc_number": meta.get("doc_number"),
                "doc_date": meta.get("doc_date"),
                "subject": meta.get("subject"),
                "sender": meta.get("sender"),
                "recipient": meta.get("recipient"),
                "doc_level": meta.get("doc_level"),
                "wk_name": meta.get("wk_name"),
                "field_name": meta.get("field_name"),
                "project_name": meta.get("project_name"),
                "raw_entities": json.dumps(raw_entities),
                "metadata": json.dumps(meta.get("extras") or {}),
                "markdown": body,
                "extraction_method": meta.get("extraction_method"),
                "embedding_model": store._embedder.model,
                "page_count": meta.get("page_count"),
            }

            if dry_run:
                report.processed.append(name)
                any_processed = True
                continue

            if exists and force:
                store.delete_document(doc["doc_id"])
            store.insert_document(doc, chunks)
            report.processed.append(name)
            any_processed = True

        if any_processed and not dry_run:
            store.rebuild_indexes()
    finally:
        store.close()

    return report


def run_status(paths: list[Path]) -> list[dict[str, str]]:
    """Report each sidecar/PDF's place in the extract -> review -> commit flow."""
    sidecars = _collect_sidecars(paths)
    pdfs = _collect_pdfs(paths)
    sidecar_set = set(sidecars)

    results: list[dict[str, str]] = []
    # (result index, file_hash) for sidecars needing a DB existence check.
    pending: list[tuple[int, str | None]] = []

    for sc in sidecars:
        name = sc.name
        try:
            meta, _body = read_sidecar(sc)
        except ValueError as e:
            results.append({"file": name, "state": f"unknown (bad sidecar: {e})"})
            continue

        if meta.get("reviewed") is not True:
            results.append({"file": name, "state": "pending review"})
        else:
            results.append({"file": name, "state": ""})
            pending.append((len(results) - 1, meta.get("file_hash")))

    if pending:
        try:
            store = CorpusStore()
            store.ensure_tables()
            try:
                for idx, file_hash in pending:
                    exists = bool(file_hash) and store.document_exists(file_hash)
                    results[idx]["state"] = "committed" if exists else "ready to commit"
            finally:
                store.close()
        except Exception as e:
            logger.warning("[Corpus] status DB check failed: %s", e)
            for idx, _file_hash in pending:
                results[idx]["state"] = "unknown (db error)"

    for pdf in pdfs:
        if sidecar_path(pdf) not in sidecar_set:
            results.append({"file": pdf.name, "state": "not extracted"})

    return results


def run_reembed() -> CorpusReport:
    """Rebuild chunk embeddings for every document with the current embedder.

    ``CorpusStore.ensure_tables`` (default ``validate_model=False``)
    initializes a fresh store or, if tables already exist, just reads the
    pinned model/dim from ``corpus_meta`` without raising on a mismatch —
    that mismatch is exactly the situation this command exists to fix.
    ``set_meta`` then re-pins the new model/dim, recreating
    document_chunks if the dimension changed, and every document's chunks
    are regenerated from its stored markdown.
    """
    report = CorpusReport()
    store = CorpusStore()
    try:
        store.ensure_tables()

        new_model = store._embedder.model
        new_dim = len(store._embedder.generate_embedding("test"))
        store.set_meta(new_model, new_dim)
        report.embedding_model = new_model

        cfg = Config.get_corpus_config()
        for summary in store.list_documents():
            doc_id = summary["doc_id"]
            name = summary.get("file_name") or doc_id
            try:
                doc = store.get_document(doc_id)
                if doc is None:
                    report.failed[name] = "document not found"
                    continue
                chunks = chunk_markdown(
                    doc["markdown"], cfg["chunk_size"], cfg["chunk_overlap"]
                )
                store.replace_chunks(doc_id, chunks)
                report.processed.append(name)
            except Exception as e:
                report.failed[name] = str(e)

        store.rebuild_indexes()
    finally:
        store.close()

    return report
