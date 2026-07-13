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
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import fitz
import ollama
from rich.console import Group
from rich.live import Live
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn

from esdc.chat.domain_knowledge.doc_schema import legacy_doc_type_map, legacy_topic_seed
from esdc.chat.domain_knowledge.entity_registry import ENTITY_REGISTRY
from esdc.chat.domain_knowledge.entity_resolver_lib import EntityResolver
from esdc.configs import Config
from esdc.console import console
from esdc.corpus.chunker import chunk_markdown
from esdc.corpus.cleanup import cleanup_markdown
from esdc.corpus.extractor import (
    SUPPORTED_EXTENSIONS,
    extract_docx,
    extract_markdown,
    extract_pdf,
)
from esdc.corpus.metadata import (
    ENTITY_FIELDS,
    METADATA_PROMPT_IMAGE,
    apply_doc_level_rules,
    doc_level_rule,
    llm_extract,
    normalize_entity_fields,
    normalize_metadata,
    normalize_topic,
    parse_llm_json,
    remap_legacy_doc_type,
    seed_topic_from_legacy,
)
from esdc.corpus.ocr import OllamaVisionOcr
from esdc.corpus.sidecar import (
    read_sidecar,
    sidecar_path,
    write_sidecar,
    write_sidecar_file,
)
from esdc.corpus.store import CorpusStore

logger = logging.getLogger(__name__)


@dataclass
class CorpusReport:
    """Batch result every pipeline function reports through; CLI-print truth."""

    processed: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    # Set by run_reembed only: the embedding model chunks were rebuilt with.
    embedding_model: str = ""


def _collect_sources(paths: list[Path]) -> list[Path]:
    """Resolve a mix of source files/dirs into a sorted, deduped list.

    Collects every ``SUPPORTED_EXTENSIONS`` file (``.pdf``, ``.docx``,
    ``.md``). ``.md`` files that are our own sidecars (``*.corpus.md``)
    are never collected as sources.
    """
    found: set[Path] = set()
    for p in paths:
        if p.is_file():
            suffix = p.suffix.lower()
            if suffix in SUPPORTED_EXTENSIONS and not p.name.lower().endswith(
                ".corpus.md"
            ):
                found.add(p)
        elif p.is_dir():
            for match in p.rglob("*"):
                if (
                    match.is_file()
                    and match.suffix.lower() in SUPPORTED_EXTENSIONS
                    and not match.name.lower().endswith(".corpus.md")
                ):
                    found.add(match)
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


def _text_llm_caller(model: str, host: str | None = None) -> Any:
    """A prompt->text callable backed by a text-only Ollama chat model."""
    client = ollama.Client(host=host)

    def call(prompt: str) -> str:
        response = client.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0},
        )
        return response["message"]["content"]

    return call


def _resolve_text_caller(
    model_spec: str | None, host: str | None = None
) -> Any | None:
    """Turn a corpus model config string into a prompt->text callable.

    "" / None -> None (feature off). "main" -> the default chat provider
    (may be a cloud API — sends document text off-machine; user opt-in).
    "provider:<name>" -> the named provider from config.yaml's providers
    section; the prefix is explicit because provider names can collide
    with Ollama model names (e.g. "mistral"), so bare names stay Ollama.
    Anything else -> an Ollama model on `host` (default local daemon).
    Returns None when "main"/"provider:<name>" is requested but no such
    provider is configured or construction fails.
    """
    if not model_spec:
        return None

    if model_spec.startswith("provider:"):
        name = model_spec[len("provider:"):]
        named_config = Config.get_provider_config_by_name(name)
        if not isinstance(named_config, dict):
            logger.warning(
                "[Corpus] unknown provider '%s' in corpus model spec %r",
                name,
                model_spec,
            )
            return None
        # Same shape get_provider_configs_by_priority produces for "main".
        provider_config = dict(named_config)
        provider_config.setdefault("name", name)
        provider_config.setdefault(
            "provider_type", provider_config.get("type") or name
        )
    elif model_spec != "main":
        return _text_llm_caller(model_spec, host)
    else:
        provider_config = Config.get_provider_config()
        if not provider_config:
            return None

    try:
        import esdc.providers as providers

        llm = providers.create_llm_from_config(provider_config)
    except Exception as e:
        logger.warning("[Corpus] %s-model caller unavailable: %s", model_spec, e)
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
    """Best-effort metadata pre-fill. Never raises: failure is a warning.

    The image fallback renders the source's first page, so it only applies
    to PDFs; non-PDF sources without a text model skip pre-fill entirely.
    """
    try:
        if text_caller is not None:
            return llm_extract(markdown, text_caller)
        if ocr_client is not None and pdf.suffix.lower() == ".pdf":
            doc = fitz.open(str(pdf))
            try:
                pix = doc[0].get_pixmap(dpi=cfg.get("ocr_dpi", 200))
                png_bytes = pix.tobytes("png")
            finally:
                doc.close()
            raw = ocr_client.query_image(png_bytes, METADATA_PROMPT_IMAGE)
            return normalize_metadata(parse_llm_json(raw))
        report.warnings.append(
            f"{name}: no metadata model configured and first-page image "
            "fallback unavailable — metadata pre-fill skipped"
        )
        return {}
    except Exception as e:
        report.warnings.append(f"{name}: metadata pre-fill failed ({e})")
        return {}


def _entity_resolver_or_none(
    store: CorpusStore, report: CorpusReport
) -> EntityResolver | None:
    """Build an EntityResolver if the canonical lookup tables exist, else None.

    Never raises: any failure (missing tables, no DB yet, connection error)
    is reported as a single warning and resolution is skipped for the run.
    """
    lookup_tables = {
        ENTITY_REGISTRY[key].lookup_table
        for key in ("wk_name", "field_name", "project_name")
    }
    try:
        conn = store._get_connection()
        placeholders = ", ".join("?" for _ in lookup_tables)
        row = conn.execute(
            "SELECT COUNT(*) FROM information_schema.tables "
            f"WHERE table_name IN ({placeholders})",
            list(lookup_tables),
        ).fetchone()
        if row is None or row[0] < len(lookup_tables):
            raise RuntimeError("lookup tables missing")
        return EntityResolver(db=conn)
    except Exception:
        report.warnings.append(
            "entity resolution skipped — canonical tables missing; "
            "run `esdc fetch` first"
        )
        return None


def _apply_overrides(meta: dict[str, Any], overrides: dict[str, Any]) -> None:
    """Apply non-None CLI override values onto metadata in place."""
    for key, value in overrides.items():
        if value is not None:
            meta[key] = value


_ENTITY_OVERRIDE_FLAGS = {
    "wk_name": "--wk-name",
    "field_name": "--field-name",
    "project_name": "--project-name",
}


def _validate_entity_hierarchy(
    resolver: EntityResolver,
    given: dict[str, str],
    canonical: dict[str, str],
) -> None:
    """Reject CLI entity overrides that don't form a valid hierarchy.

    Checks that field_name belongs to wk_name, and project_name belongs
    to field_name + wk_name, by querying project_resources. `given` holds
    the raw CLI values (for error messages); `canonical` holds the
    resolved canonical names (for filtering). Levels that resolved to
    multiple canonical names are skipped — ambiguity is warned elsewhere.
    """
    wk = canonical.get("wk_name")
    field = canonical.get("field_name")
    project = given.get("project_name")

    if wk and field:
        matches = resolver.resolve_name(
            field, "field_name", parent_filter={"wk_name": wk}
        )
        if not matches:
            flag_field = _ENTITY_OVERRIDE_FLAGS["field_name"]
            flag_wk = _ENTITY_OVERRIDE_FLAGS["wk_name"]
            raise ValueError(
                f"{flag_field} '{given['field_name']}' does not belong to "
                f"{flag_wk} '{given['wk_name']}' in the database"
            )

    if wk and project:
        parent = {"wk_name": wk}
        parent_desc = f"{_ENTITY_OVERRIDE_FLAGS['wk_name']} '{given['wk_name']}'"
        if field:
            parent["field_name"] = field
            parent_desc += (
                f" / {_ENTITY_OVERRIDE_FLAGS['field_name']} '{given['field_name']}'"
            )
        matches = resolver.resolve_name(project, "project_name", parent_filter=parent)
        if not matches:
            flag_project = _ENTITY_OVERRIDE_FLAGS["project_name"]
            raise ValueError(
                f"{flag_project} '{project}' does not belong to {parent_desc} "
                f"in the database"
            )


def _validate_entity_overrides(
    resolver: EntityResolver | None,
    overrides: dict[str, Any],
) -> None:
    """Reject CLI entity overrides that don't resolve against canonical tables.

    Raises ValueError before any sidecar is touched. Only explicit CLI
    values are validated strictly — LLM-prefilled/frontmatter values stay
    warn-only so unreviewed data never jams the pipeline. Also validates
    that the given wk_name/field_name/project_name form a consistent
    hierarchy (see `_validate_entity_hierarchy`), using the canonical
    resolved names rather than the raw CLI values.
    """
    given = {
        key: overrides[key]
        for key in _ENTITY_OVERRIDE_FLAGS
        if overrides.get(key) is not None
    }
    if not given:
        return
    if resolver is None:
        raise ValueError(
            "cannot validate --wk-name/--field-name/--project-name against "
            "the database — corpus DB unavailable (run `esdc fetch`, or "
            "close other esdc instances holding the DB lock)"
        )
    canonical: dict[str, str] = {}
    for key, value in given.items():
        matches = resolver.resolve_name(str(value), key)
        if matches:
            if len(matches) == 1:
                canonical[key] = matches[0]["name"]
            continue
        flag = _ENTITY_OVERRIDE_FLAGS[key]
        suggestions = resolver.suggest_names(str(value), key)
        if suggestions:
            listed = ", ".join(f"'{s}'" for s in suggestions)
            raise ValueError(
                f"{flag} '{value}' not found in database; closest matches: {listed}"
            )
        raise ValueError(f"{flag} '{value}' not found in database and no close matches")

    # Hierarchy check: wk -> field -> project must be consistent
    _validate_entity_hierarchy(resolver, given, canonical)


def _warn_vocab_demotions(
    raw_type: Any,
    raw_level: Any,
    meta: dict[str, Any],
    report: CorpusReport,
    name: str,
) -> None:
    """Surface out-of-vocab values that normalize_metadata clamps silently.

    Legacy aliases and case variants are legitimate normalizations, not
    demotions — only values that had no canonical form get a warning.
    """
    if raw_type is not None and meta.get("doc_type") == "others":
        canon = raw_type.lower() if isinstance(raw_type, str) else raw_type
        canon = legacy_doc_type_map().get(canon, canon)
        if canon != "others":
            report.warnings.append(
                f"{name}: doc_type '{raw_type}' not in vocab — stored as 'others'"
            )
    if raw_level is not None and meta.get("doc_level") == "unknown":
        canon = raw_level.lower() if isinstance(raw_level, str) else raw_level
        if canon != "unknown":
            report.warnings.append(
                f"{name}: doc_level '{raw_level}' not in vocab — stored as 'unknown'"
            )


def _warn_rule_effects(
    pre_rule_meta: dict[str, Any],
    meta: dict[str, Any],
    report: CorpusReport,
    name: str,
) -> None:
    """Surface deterministic doc_level_rule effects applied silently.

    These come from ``normalize_metadata`` (or the meta-command's surgical
    equivalent). ``pre_rule_meta`` is the metadata as it stood right before the legacy
    doc_type remap/rule stage ran; ``meta`` is the final, rule-applied
    result. Three effects are surfaced: a legacy doc_type remap (+ topic
    seed), a rule that changed doc_level, and a "regulation" rule clearing
    wk_name/field_name/project_name.
    """
    raw_type = pre_rule_meta.get("doc_type")
    canon_raw_type = raw_type.lower() if isinstance(raw_type, str) else raw_type
    if canon_raw_type in legacy_doc_type_map():
        seeded = legacy_topic_seed().get(canon_raw_type)
        suffix = f" + topic '{seeded}'" if seeded else ""
        report.warnings.append(
            f"{name}: doc_type '{raw_type}' remapped to "
            f"'{meta.get('doc_type')}'{suffix}"
        )

    rule = doc_level_rule(meta.get("doc_type"), meta.get("doc_topic"))
    if rule is not None:
        kind, key, level = rule
        raw_level = pre_rule_meta.get("doc_level")
        canon_raw_level = raw_level.lower() if isinstance(raw_level, str) else raw_level
        if canon_raw_level != level:
            report.warnings.append(
                f"{name}: doc_level '{raw_level}' overridden to '{level}' "
                f"({kind} '{key}' rule)"
            )
        if level == "regulation":
            stripped = [
                key for key in ENTITY_FIELDS if pre_rule_meta.get(key) not in (None, [])
            ]
            if stripped:
                report.warnings.append(
                    f"{name}: doc_level 'regulation' — cleared {', '.join(stripped)}"
                )


def _apply_meta_rules(meta: dict[str, Any]) -> dict[str, Any]:
    """Canonicalize + apply doc_level rules the way ``esdc corpus meta`` should.

    Unlike ``normalize_metadata`` (the full commit-time clamp), this never
    invents a value for a field the sidecar/override didn't touch: a
    missing doc_type stays missing, an untouched doc_level stays whatever
    it was, and an unrecognized (non-legacy) doc_type is left alone rather
    than demoted to "others" — that clamp only makes sense at commit, the
    last gate before the searchable corpus. A legacy doc_type alias already
    present is still remapped (+ seeds its doc_topic) and the deterministic
    doc_level rules still apply, since both are meant to self-heal a
    sidecar the moment ``meta`` touches it.
    """
    out = dict(meta)
    raw_type = out.get("doc_type")
    seeded_topic = seed_topic_from_legacy(raw_type)
    out["doc_type"] = remap_legacy_doc_type(raw_type)

    raw_topic = out.get("doc_topic")
    if raw_topic is None and seeded_topic is not None:
        raw_topic = [seeded_topic]
    out["doc_topic"] = normalize_topic(raw_topic)

    return apply_doc_level_rules(out)


class _ProgressHandle:
    """Drives a two-line rich progress display: a bar plus a status line.

    The bar's own task tracks file-count progress; the status line shows
    the current file and phase (``surat.pdf · resolve entities``), updated
    in place. Both lines share one Live so they refresh together.
    """

    def __init__(self, bar: Progress, bar_id: int, status: Progress, status_id: int):
        self._bar = bar
        self._bar_id = bar_id
        self._status = status
        self._status_id = status_id
        self._file = ""

    def file(self, name: str) -> None:
        """Set the current file; clears any prior phase text."""
        self._file = name[:40]
        self._render()

    def status(self, phase: str) -> None:
        """Update the phase shown after the current file name."""
        self._render(phase)

    def advance(self) -> None:
        self._bar.advance(self._bar_id)

    def _render(self, phase: str = "") -> None:
        text = self._file
        if phase:
            text = f"{text} · {phase}" if text else phase
        self._status.update(self._status_id, description=text)


@contextmanager
def _progress_with_status(verb: str, total: int, unit: str):
    """A file-count bar with a dim, in-place status line rendered below it."""
    bar = Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=36),
        TextColumn(f"[green]{{task.completed}}/{{task.total}} {unit}"),
        TimeElapsedColumn(),
        console=console,
    )
    status = Progress(TextColumn("  [dim]{task.description}"), console=console)
    bar_id = bar.add_task(verb, total=total)
    status_id = status.add_task("")
    with Live(Group(bar, status), console=console, refresh_per_second=10):
        yield _ProgressHandle(bar, bar_id, status, status_id)


def run_extract(
    paths: list[Path],
    level: str | None = None,
    doc_type: str | None = None,
    topic: str | None = None,
    wk_name: str | None = None,
    field_name: str | None = None,
    project_name: str | None = None,
    force: bool = False,
) -> CorpusReport:
    """Parse sources (PDF/docx/md) to reviewable ``.corpus.md`` sidecars.

    (extract/commit step 1)
    """
    report = CorpusReport()
    cfg = Config.get_corpus_config()
    sources = _collect_sources(paths)

    # Collision guard: distinct sources that would write to the same
    # sidecar path (e.g. report.pdf + report.docx -> report.corpus.md).
    # The first (in sorted order) wins; the rest fail with a clear message
    # instead of silently overwriting each other's sidecar.
    by_sidecar: dict[Path, list[Path]] = {}
    for src in sources:
        by_sidecar.setdefault(sidecar_path(src), []).append(src)
    sources = []
    for group in by_sidecar.values():
        sources.append(group[0])
        for dup in group[1:]:
            report.failed[dup.name] = f"sidecar path collision with {group[0].name}"

    overrides = {
        "doc_level": level,
        "doc_type": doc_type,
        "doc_topic": topic,
        "wk_name": wk_name,
        "field_name": field_name,
        "project_name": project_name,
    }

    ollama_host = cfg.get("ollama_host") or None
    ocr = OllamaVisionOcr(
        cfg["ocr_model"], num_ctx=cfg.get("num_ctx", 16384), host=ollama_host
    )
    ocr_client = ocr if ocr.health_check() else None

    metadata_caller = _resolve_text_caller(cfg.get("metadata_model"), ollama_host)
    if cfg.get("metadata_model") and metadata_caller is None:
        report.warnings.append(
            f"metadata_model '{cfg['metadata_model']}' unavailable — "
            "falling back to image-based metadata prefill"
        )

    cleanup_caller = _resolve_text_caller(cfg.get("cleanup_model"), ollama_host)
    if cfg.get("cleanup_model") and cleanup_caller is None:
        report.warnings.append(
            f"cleanup_model '{cfg['cleanup_model']}' unavailable — "
            "formatting cleanup skipped"
        )

    store = CorpusStore()
    try:
        resolver = _entity_resolver_or_none(store, report)
        _validate_entity_overrides(resolver, overrides)

        # (name, pages_ocr, page_count) collected so the final report can be
        # sorted by OCR ratio DESC once, instead of per-file.
        entries: list[tuple[str, int, int]] = []

        with _progress_with_status("extract", len(sources), "files") as p:
            for src in sources:
                name = src.name
                p.file(name)
                if sidecar_path(src).exists() and not force:
                    report.skipped.append(f"{name} (sidecar exists)")
                    p.advance()
                    continue

                try:
                    file_hash = hashlib.sha256(src.read_bytes()).hexdigest()
                    suffix = src.suffix.lower()
                    if suffix == ".pdf":
                        p.status("parsing PDF")
                        result = extract_pdf(
                            src,
                            ocr_client,
                            cfg["min_chars_per_page"],
                            cfg["ocr_dpi"],
                            cfg["min_image_area"],
                        )
                    elif suffix == ".docx":
                        p.status("parsing docx")
                        result = extract_docx(src)
                    else:
                        p.status("reading markdown")
                        result = extract_markdown(src)
                    markdown = result.markdown
                    if cleanup_caller is not None:
                        p.status("cleanup formatting")
                        markdown, n_cleaned, n_rejected = cleanup_markdown(
                            markdown, cleanup_caller
                        )
                        if n_cleaned:
                            report.warnings.append(
                                f"{name}: {n_cleaned} page segment(s) reformatted by "
                                "cleanup model — verify against the original PDF"
                            )
                        if n_rejected:
                            report.warnings.append(
                                f"{name}: {n_rejected} segment(s) failed cleanup guard "
                                "— original text kept"
                            )
                    p.status("prefill metadata")
                    meta_fields = _prefill_metadata(
                        src, markdown, metadata_caller, ocr_client, cfg, report, name
                    )

                    meta = {
                        "source_file": name,
                        "file_hash": file_hash,
                        "page_count": result.page_count,
                        "extraction_method": result.method,
                        "reviewed": False,
                        "doc_type": meta_fields.get("doc_type"),
                        "doc_topic": meta_fields.get("doc_topic"),
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
                    _apply_overrides(meta, overrides)
                    # A --topic override is a single string; normalize it to
                    # the list form doc_topic is stored as.
                    if overrides.get("doc_topic") is not None:
                        meta["doc_topic"] = normalize_topic(overrides["doc_topic"])
                    meta = normalize_entity_fields(meta)

                    if resolver is not None:
                        p.status("resolve entities")
                        raw_entities = _apply_entity_resolution(
                            meta, resolver, report, name
                        )
                        if any(v is not None for v in raw_entities.values()):
                            meta["raw_entities"] = raw_entities

                    p.status("write sidecar")
                    write_sidecar(src, meta, markdown)

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
                            f"{name}: {result.images_skipped} embedded image(s) "
                            "skipped (no OCR model) — table/chart content may be "
                            "missing"
                        )
                    entries.append((name, result.pages_ocr, result.page_count))
                except Exception as e:
                    report.failed[name] = str(e)
                finally:
                    p.advance()

        entries.sort(
            key=lambda e: (e[1] / e[2] if e[2] else 0.0),
            reverse=True,
        )
        for name, pages_ocr, _page_count in entries:
            report.processed.append(
                name if pages_ocr > 0 else f"{name} [native — minimal review]"
            )
    finally:
        store.close()

    return report


def _apply_entity_resolution(
    meta: dict[str, Any],
    resolver: EntityResolver,
    report: CorpusReport,
    name: str,
) -> dict[str, Any | None]:
    """Resolve wk_name/field_name/project_name in-place on `meta`; return raw values.

    Input can be a list of names (e.g. ["ARUNG", "NOWERA"]) or a single string.
    `resolver.resolve_name` now returns the final confident picks for a raw
    name -- possibly more than one, since one raw string can legitimately
    name several real entities (e.g. "Arung Nowera" naming two separate
    fields). Resolution is hierarchy-aware: once a level resolves to a
    single unambiguous canonical name, that name is passed as
    `parent_filter` when resolving the next level (``wk_name`` ->
    ``field_name`` -> ``project_name``, per ``ENTITY_FIELDS``), since
    ``project_resources`` is denormalized (wk/field/project on the same
    row).

    Two failure modes for a raw name that doesn't resolve under its parent:
    - **Wrong hierarchy**: the name resolves to something when the parent
      filter is dropped (i.e. it names a real entity, just under a
      different wk/field). This is a proven cross-hierarchy mismatch, so
      the value is DROPPED (not kept) and a warning names what it matched
      and which parent it violated — the sidecar must never hold an
      invalid wk -> field -> project combination.
    - **Unknown name**: the name doesn't resolve even without the parent
      filter. This is unproven — it might just be a typo or an entity
      missing from the DB — so it's kept as-is in `meta[key]` for a
      reviewer to fix, exactly as before.

    `meta[key]` ends up holding every resolved (or kept-as-is-unknown) name,
    deduped. `meta["entity_warnings"]` collects the human-readable reasons
    why, and is cleared entirely on a clean re-run so a stale warning from a
    prior resolution never lingers after the sidecar is fixed.
    """
    raw_entities: dict[str, Any | None] = {}
    all_warnings: list[str] = []
    # Canonical names resolved so far, keyed by ENTITY_FIELDS level — used
    # as parent_filter for the next level. Only an actual, unambiguous
    # resolver match is added here; a kept-as-is raw (unresolved) name must
    # never constrain a child level's resolution.
    resolved_context: dict[str, str] = {}

    for key in ENTITY_FIELDS:
        raw = meta.get(key)
        raw_entities[key] = raw
        if raw is None:
            meta[key] = None
            continue

        # Normalize to list for uniform handling
        raw_list = raw if isinstance(raw, list) else [raw]
        resolved: list[str] = []
        matched_names: set[str] = set()  # subset of `resolved` from real matches

        for raw_name in raw_list:
            if not raw_name:
                continue
            parent_filter = resolved_context or None
            matches = resolver.resolve_name(
                str(raw_name), key, parent_filter=parent_filter
            )

            if not matches and parent_filter:
                # Diagnose: does the name exist under a DIFFERENT parent
                # (wrong hierarchy) or nowhere at all (unknown)? A
                # wrong-hierarchy name is DROPPED — the sidecar must never
                # hold an invalid wk -> field -> project combination.
                # Unknown names fall through to the kept-as-is handling
                # below, unchanged from before.
                diagnosis = resolver.resolve_name(
                    str(raw_name), key, parent_filter=None
                )
                if diagnosis:
                    parent_desc = ", ".join(
                        f"{k}='{v}'" for k, v in parent_filter.items()
                    )
                    matched = ", ".join(f"'{m['name']}'" for m in diagnosis)
                    all_warnings.append(
                        f"{key} '{raw_name}' matches {matched} but does not "
                        f"belong to {parent_desc} — dropped, verify hierarchy"
                    )
                    continue

            if not matches:
                if raw_name not in resolved:
                    resolved.append(raw_name)
                # Give the reviewer something to copy-paste: the closest
                # canonical names, fuzzy-ranked (a typo like "Rokann"
                # matches nothing via resolve_name's substring lookup).
                suggestions = resolver.suggest_names(str(raw_name), key)
                if suggestions:
                    listed = ", ".join(f"'{s}'" for s in suggestions)
                    all_warnings.append(
                        f"{key} '{raw_name}' unresolved — kept as-is; "
                        f"closest matches: {listed}"
                    )
                else:
                    all_warnings.append(
                        f"{key} '{raw_name}' unresolved — kept as-is, verify manually"
                    )
                continue
            for m in matches:
                if m["name"] not in resolved:
                    resolved.append(m["name"])
                matched_names.add(m["name"])
                if m["confidence"] < 1.0:
                    all_warnings.append(
                        f"{key} '{raw_name}' -> '{m['name']}' "
                        f"(confidence {m['confidence']})"
                    )

        meta[key] = resolved if resolved else None

        # Only an unambiguous single resolution can safely constrain child
        # levels, and only when it came from an actual resolver match (a
        # kept-as-is raw/unknown name must not filter child resolution).
        if resolved and len(resolved) == 1 and resolved[0] in matched_names:
            resolved_context[key] = resolved[0]
        elif resolved and len(resolved) > 1:
            logger.debug(
                "entity hierarchy: %s resolved to %d names, "
                "skipping parent_filter for child levels",
                key,
                len(resolved),
            )

    if all_warnings:
        meta["entity_warnings"] = all_warnings
        for w in all_warnings:
            report.warnings.append(f"{name}: {w}")
    else:
        meta.pop("entity_warnings", None)
    return raw_entities


def run_commit(
    paths: list[Path],
    skip_review: bool = False,
    force: bool = False,
    dry_run: bool = False,
) -> CorpusReport:
    """Ingest reviewed ``.corpus.md`` sidecars into the corpus store."""
    report = CorpusReport()
    cfg = Config.get_corpus_config()
    sidecars = _collect_sidecars(paths)

    store = CorpusStore()
    any_processed = False
    try:
        # also needed to reach the conn for canonical names
        store.ensure_tables(validate_model=True)
        resolver = EntityResolver(db=store._get_connection())

        with _progress_with_status("commit", len(sidecars), "docs") as p:
            for sc in sidecars:
                name = sc.name
                p.file(name)
                try:
                    try:
                        p.status("read sidecar")
                        meta, body = read_sidecar(sc)
                    except ValueError as e:
                        report.failed[name] = str(e)
                        continue

                    raw_meta = dict(meta)
                    pending = meta.get("reviewed") is not True
                    if pending and not skip_review:
                        report.skipped.append(f"{name} (pending review)")
                        continue

                    raw_type = meta.get("doc_type")
                    raw_level = meta.get("doc_level")
                    pre_rule_meta = dict(meta)
                    meta = normalize_metadata(meta)
                    _warn_vocab_demotions(raw_type, raw_level, meta, report, name)
                    _warn_rule_effects(pre_rule_meta, meta, report, name)
                    meta = normalize_entity_fields(meta)

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

                    # Safety net for hand-edited sidecars: re-resolve even
                    # though extract already did this once.
                    p.status("resolve entities")
                    resolver_raws = _apply_entity_resolution(
                        meta, resolver, report, name
                    )
                    sidecar_raws = meta.pop("raw_entities", None)
                    meta.pop("entity_warnings", None)

                    p.status("chunk markdown")
                    chunks = chunk_markdown(
                        body, cfg["chunk_size"], cfg["chunk_overlap"]
                    )
                    if not chunks:
                        report.failed[name] = "no content"
                        continue

                    doc = {
                        "doc_id": (file_hash or "")[:16],
                        "file_name": meta.get("source_file"),
                        "file_path": str(sc),
                        "file_hash": file_hash,
                        "doc_type": meta.get("doc_type"),
                        "doc_topic": meta.get("doc_topic"),
                        "doc_number": meta.get("doc_number"),
                        "doc_date": meta.get("doc_date"),
                        "subject": meta.get("subject"),
                        "sender": meta.get("sender"),
                        "recipient": meta.get("recipient"),
                        "doc_level": meta.get("doc_level"),
                        "wk_name": meta.get("wk_name"),
                        "field_name": meta.get("field_name"),
                        "project_name": meta.get("project_name"),
                        "raw_entities": json.dumps(sidecar_raws or resolver_raws),
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

                    p.status("embed + insert")
                    if exists and force:
                        store.delete_document(doc["doc_id"])
                    store.insert_document(doc, chunks)
                    if pending:
                        raw_meta["reviewed"] = True
                        write_sidecar_file(sc, raw_meta, body)
                        report.warnings.append(
                            f"{name}: ingested without review (--skip-review)"
                        )
                    report.processed.append(name)
                    any_processed = True
                except Exception as e:
                    report.failed[name] = str(e)
                finally:
                    p.advance()

        if any_processed and not dry_run:
            store.rebuild_indexes()
    finally:
        store.close()

    return report


def run_status(paths: list[Path]) -> list[dict[str, str]]:
    """Report each sidecar/source's place in the extract -> review -> commit flow."""
    sidecars = _collect_sidecars(paths)
    sources = _collect_sources(paths)
    sidecar_set = set(sidecars)

    # A file-arg source (e.g. `doc.pdf`) never matches the .corpus.md glob
    # above, so its sidecar would otherwise go undetected even though it
    # sits right next to it on disk. Add it explicitly, deduped, keeping
    # deterministic order (globbed sidecars first, then source-derived ones).
    for src in sources:
        candidate = sidecar_path(src)
        if candidate not in sidecar_set and candidate.exists():
            sidecars.append(candidate)
            sidecar_set.add(candidate)

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

    for src in sources:
        if sidecar_path(src) not in sidecar_set:
            results.append({"file": src.name, "state": "not extracted"})

    return results


# LLM-owned fields --regenerate replaces wholesale, before overrides/rules run.
_REGENERATE_FIELDS = (
    "doc_type",
    "doc_topic",
    "doc_number",
    "doc_date",
    "subject",
    "sender",
    "recipient",
    "doc_level",
    "wk_name",
    "field_name",
    "project_name",
    "extras",
)


def run_meta(
    paths: list[Path],
    level: str | None = None,
    doc_type: str | None = None,
    topic: str | None = None,
    wk_name: str | None = None,
    field_name: str | None = None,
    project_name: str | None = None,
    reviewed: bool | None = None,
    regenerate: bool = False,
) -> CorpusReport:
    """Bulk-edit ``.corpus.md`` sidecar frontmatter in place.

    Persists overrides to the sidecar (unlike the old commit-time overrides,
    which never reached the file). Re-runs entity resolution as a safety net
    and warns when the sidecar is already committed, since the store copy
    only updates on ``commit --force``.

    ``regenerate=True`` re-runs the LLM metadata analysis on each sidecar's
    existing markdown body (no re-parse/OCR) and replaces the LLM-owned
    fields with the fresh result before overrides/rules apply — explicit
    flags in the same invocation still win. Requires a reachable
    ``metadata_model``; raises before touching any file otherwise.
    """
    report = CorpusReport()
    sidecars = _collect_sidecars(paths)
    if not sidecars:
        # Nothing to mutate — skip the store open (and override validation)
        # entirely rather than paying for a DB connection to do no work.
        return report

    overrides = {
        "doc_level": level,
        "doc_type": doc_type,
        "doc_topic": topic,
        "wk_name": wk_name,
        "field_name": field_name,
        "project_name": project_name,
    }

    regen_caller = None
    if regenerate:
        cfg = Config.get_corpus_config()
        ollama_host = cfg.get("ollama_host") or None
        regen_caller = _resolve_text_caller(cfg.get("metadata_model"), ollama_host)
        if regen_caller is None:
            raise ValueError(
                "--regenerate requires a reachable metadata_model "
                "(set corpus.metadata_model in config)"
            )

    store: CorpusStore | None = None
    resolver = None
    try:
        try:
            store = CorpusStore()
            store.ensure_tables()
            resolver = _entity_resolver_or_none(store, report)
        except Exception as e:
            logger.warning("[Corpus] meta DB unavailable: %s", e)
            report.warnings.append(
                "corpus DB unavailable — committed-status check and entity "
                "resolution skipped"
            )
            store = None

        # Explicit CLI entity values must exist in the canonical tables;
        # fail fast before any sidecar is rewritten.
        _validate_entity_overrides(resolver, overrides)

        with _progress_with_status("meta", len(sidecars), "docs") as p:
            for sc in sidecars:
                name = sc.name
                p.file(name)
                try:
                    try:
                        p.status("read sidecar")
                        meta, body = read_sidecar(sc)
                    except ValueError as e:
                        report.failed[name] = str(e)
                        continue

                    if regenerate:
                        p.status("regenerate metadata")
                        regen_fields = llm_extract(body, regen_caller)
                        for key in _REGENERATE_FIELDS:
                            meta[key] = regen_fields.get(key)

                    _apply_overrides(meta, overrides)
                    if reviewed is not None:
                        meta["reviewed"] = reviewed
                    elif regenerate:
                        meta["reviewed"] = False

                    pre_rule_meta = dict(meta)
                    meta = _apply_meta_rules(meta)
                    _warn_rule_effects(pre_rule_meta, meta, report, name)
                    meta = normalize_entity_fields(meta)

                    if resolver is not None:
                        p.status("resolve entities")
                        existing_raws = meta.get("raw_entities") or {}
                        resolved_raws = _apply_entity_resolution(
                            meta, resolver, report, name
                        )
                        # Overridden fields get the user-supplied value as their
                        # new raw; untouched fields keep their original raw.
                        for key in ("wk_name", "field_name", "project_name"):
                            if overrides[key] is not None:
                                existing_raws[key] = overrides[key]
                            elif key not in existing_raws:
                                existing_raws[key] = resolved_raws.get(key)
                        if any(v is not None for v in existing_raws.values()):
                            meta["raw_entities"] = existing_raws

                    file_hash = meta.get("file_hash")
                    if (
                        store is not None
                        and file_hash
                        and store.document_exists(file_hash)
                    ):
                        report.warnings.append(
                            f"{name}: already committed — run "
                            "`esdc corpus commit --force` to apply the new "
                            "metadata to the corpus"
                        )

                    p.status("write sidecar")
                    write_sidecar_file(sc, meta, body)
                    report.processed.append(name)
                except Exception as e:
                    report.failed[name] = str(e)
                finally:
                    p.advance()
    finally:
        if store is not None:
            store.close()

    return report


def run_meta_show(paths: list[Path]) -> list[dict[str, Any]]:
    """Read-only metadata listing for ``.corpus.md`` sidecars."""
    rows: list[dict[str, Any]] = []
    for sc in _collect_sidecars(paths):
        try:
            meta, _body = read_sidecar(sc)
        except ValueError as e:
            rows.append({"file": sc.name, "error": str(e)})
            continue
        rows.append(
            {
                "file": sc.name,
                "doc_type": meta.get("doc_type"),
                "doc_topic": meta.get("doc_topic"),
                "doc_date": meta.get("doc_date"),
                "doc_level": meta.get("doc_level"),
                "wk_name": meta.get("wk_name"),
                "field_name": meta.get("field_name"),
                "project_name": meta.get("project_name"),
                "reviewed": meta.get("reviewed"),
            }
        )
    return rows


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
