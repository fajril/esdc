"""Rename corpus source files (and their sidecars) to a canonical structure.

Target name: ``DOC_TYPE - YYYY.MM.DD - title.<ext>`` where DOC_TYPE is
``doc_type``, the date is the ISO ``doc_date`` (issue date) rendered with
dots, and title is ``subject``. The three fields are resolved per file
through a ladder: existing ``.corpus.md`` sidecar frontmatter -> committed
corpus DB (by file_hash) -> LLM/OCR inference (with the filename passed as
a hint). Dry-run is the default; on-disk renames happen only when the CLI
passes ``apply=True``.
"""

import datetime
import hashlib
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fitz

from esdc.configs import Config
from esdc.corpus.extractor import extract_document
from esdc.corpus.metadata import (
    llm_extract,
    metadata_image_prompt,
    normalize_metadata,
    parse_llm_json,
)
from esdc.corpus.ocr import OllamaVisionOcr
from esdc.corpus.pipeline import CorpusReport, _collect_sources, _resolve_text_caller
from esdc.corpus.sidecar import read_sidecar, sidecar_path
from esdc.corpus.store import CorpusStore

_FIELDS = ("doc_type", "doc_date", "subject")

logger = logging.getLogger(__name__)

_ILLEGAL_TITLE_CHARS = re.compile(r'[\\/:*?"<>|\x00-\x1f]')
_MAX_TITLE_CHARS = 150


def format_doc_date(value: Any) -> str | None:
    """Render an ISO date (str or date/datetime) as ``YYYY.MM.DD``; None if invalid."""
    if value is None or value == "":
        return None
    if isinstance(value, (datetime.date, datetime.datetime)):
        d: datetime.date = value
    else:
        try:
            d = datetime.date.fromisoformat(str(value).strip())
        except ValueError:
            return None
    return f"{d.year:04d}.{d.month:02d}.{d.day:02d}"


def sanitize_title(title: Any) -> str | None:
    """Make a filesystem-safe title fragment; None when empty after cleaning."""
    if not title:
        return None
    cleaned = _ILLEGAL_TITLE_CHARS.sub(" ", str(title))
    cleaned = re.sub(r"\s+", " ", cleaned).strip().strip(".").strip()
    if not cleaned:
        return None
    return cleaned[:_MAX_TITLE_CHARS].strip()


@dataclass
class RenamePlan:
    src: Path
    new_path: Path | None
    doc_type: str | None
    doc_date: str | None
    title: str | None
    source: str
    sidecar_src: Path | None
    sidecar_new: Path | None
    note: str = ""


class _LlmContext:
    """Lazily-built OCR client + text caller shared across a rename batch."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        self._cfg = cfg
        self._built = False
        self.ocr_client: Any | None = None
        self.metadata_caller: Any | None = None

    def _build(self) -> None:
        if self._built:
            return
        self._built = True
        cfg = self._cfg
        host = cfg.get("ollama_host") or None
        ocr = OllamaVisionOcr(
            cfg["ocr_model"], num_ctx=cfg.get("num_ctx", 16384), host=host
        )
        self.ocr_client = ocr if ocr.health_check() else None
        self.metadata_caller = _resolve_text_caller(cfg.get("metadata_model"), host)

    def infer(self, src: Path) -> dict[str, Any]:
        """Best-effort LLM/OCR metadata for src; {} when unavailable/failed."""
        self._build()
        cfg = self._cfg
        try:
            if self.metadata_caller is not None:
                result = extract_document(
                    src,
                    self.ocr_client,
                    cfg["min_chars_per_page"],
                    cfg["ocr_dpi"],
                    cfg["min_image_area"],
                )
                return llm_extract(
                    result.markdown, self.metadata_caller, filename=src.name
                )
            if self.ocr_client is not None and src.suffix.lower() == ".pdf":
                doc = fitz.open(str(src))
                try:
                    pix = doc[0].get_pixmap(dpi=cfg.get("ocr_dpi", 200))
                    png = pix.tobytes("png")
                finally:
                    doc.close()
                raw = self.ocr_client.query_image(png, metadata_image_prompt(src.name))
                return normalize_metadata(parse_llm_json(raw))
        except Exception as e:  # inference is best-effort; never fatal
            logger.warning("[Corpus] rename inference failed for %s: %s", src.name, e)
        return {}


def _resolve_fields(
    src: Path,
    doc_type_override: str | None,
    llm: _LlmContext,
) -> tuple[dict[str, Any], Path | None, str]:
    """Return (fields, sidecar_src, doc_date_source) via sidecar -> DB -> LLM."""
    fields: dict[str, Any] = dict.fromkeys(_FIELDS)
    if doc_type_override is not None:
        fields["doc_type"] = doc_type_override
    sidecar_src: Path | None = None
    source = "-"

    def _fill(meta: dict[str, Any], tier: str) -> None:
        nonlocal source
        for key in _FIELDS:
            if fields[key] is None and meta.get(key) is not None:
                fields[key] = meta.get(key)
                if key == "doc_date":
                    source = tier

    # Tier 1: sidecar
    sc = sidecar_path(src)
    if sc.exists():
        sidecar_src = sc
        try:
            meta, _body = read_sidecar(sc)
            _fill(meta, "sidecar")
        except ValueError:
            pass

    # Tier 2: committed DB (by file_hash)
    if any(fields[k] is None for k in _FIELDS):
        try:
            store = CorpusStore()
            try:
                file_hash = hashlib.sha256(src.read_bytes()).hexdigest()
                row = store.get_document_by_hash(file_hash)
                if row is not None:
                    _fill(row, "db")
            finally:
                store.close()
        except Exception as e:
            logger.debug("[Corpus] rename DB lookup skipped for %s: %s", src.name, e)

    # Tier 3: LLM/OCR inference
    if any(fields[k] is None for k in _FIELDS):
        _fill(llm.infer(src), "llm")

    return fields, sidecar_src, source


def run_rename(
    paths: list[Path],
    doc_type: str | None = None,
    apply: bool = False,
) -> tuple[CorpusReport, list[RenamePlan]]:
    """Plan (and, when apply=True, perform) canonical renames of source files.

    Returns (report, plans). Dry-run (apply=False) builds plans without
    touching disk. Execution is handled in _apply_plans (Task 5).
    """
    report = CorpusReport()
    cfg = Config.get_corpus_config()
    sources = _collect_sources(paths)
    llm = _LlmContext(cfg)

    plans: list[RenamePlan] = []
    claimed: set[Path] = set()

    for src in sources:
        fields, sidecar_src, source = _resolve_fields(src, doc_type, llm)
        dt = fields["doc_type"]
        dd = format_doc_date(fields["doc_date"])
        title = sanitize_title(fields["subject"])

        missing = [
            name
            for name, val in (("doc_type", dt), ("doc_date", dd), ("title", title))
            if not val
        ]
        if missing:
            note = f"unresolved: missing {', '.join(missing)}"
            report.skipped.append(f"{src.name} ({note})")
            plans.append(
                RenamePlan(src, None, dt, dd, title, "-", sidecar_src, None, note)
            )
            continue

        new_path = src.with_name(f"{dt} - {dd} - {title}{src.suffix}")
        sidecar_new = sidecar_path(new_path) if sidecar_src is not None else None

        if new_path == src:
            report.skipped.append(f"{src.name} (already named)")
            plans.append(
                RenamePlan(
                    src,
                    new_path,
                    dt,
                    dd,
                    title,
                    source,
                    sidecar_src,
                    sidecar_new,
                    "already named",
                )
            )
            continue

        conflict = None
        for target in (new_path, sidecar_new):
            if target is None:
                continue
            if target in claimed or (target.exists() and target != src):
                conflict = target
                break
        if conflict is not None:
            note = f"target exists: {conflict.name}"
            report.failed[src.name] = note
            plans.append(
                RenamePlan(src, None, dt, dd, title, source, sidecar_src, None, note)
            )
            continue

        claimed.add(new_path)
        if sidecar_new is not None:
            claimed.add(sidecar_new)
        plans.append(
            RenamePlan(
                src, new_path, dt, dd, title, source, sidecar_src, sidecar_new, ""
            )
        )

    if apply:
        _apply_plans(plans, report)  # defined in Task 5

    return report, plans


def _apply_plans(plans: list[RenamePlan], report: CorpusReport) -> None:
    """Perform the disk renames for resolved plans; record results in report."""
    from esdc.corpus.sidecar import write_sidecar_file

    for plan in plans:
        if plan.new_path is None or plan.note:
            continue
        try:
            plan.src.rename(plan.new_path)
            if plan.sidecar_src is not None and plan.sidecar_new is not None:
                meta, body = read_sidecar(plan.sidecar_src)
                meta["source_file"] = plan.new_path.name
                write_sidecar_file(plan.sidecar_new, meta, body)
                if plan.sidecar_src != plan.sidecar_new:
                    plan.sidecar_src.unlink()
            report.processed.append(plan.src.name)
        except OSError as e:
            report.failed[plan.src.name] = str(e)
