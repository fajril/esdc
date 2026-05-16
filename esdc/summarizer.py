"""LLM-backed executive summaries for Eureka resource dashboards."""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal

import duckdb
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn

from esdc.configs import Config
from esdc.console import console
from esdc.db_security import _load_sql_script
from esdc.dbmanager import _ensure_duckdb_database, get_duckdb_connection

logger = logging.getLogger(__name__)

EntityLevel = Literal["field", "working_area", "nkri"]
SummarizeTarget = Literal["all", "field", "working_area", "nkri"]

SUMMARY_TABLE = "resource_summaries"
SUMMARY_COLUMNS_SQL = f"""
CREATE TABLE IF NOT EXISTS {SUMMARY_TABLE} (
    entity_level TEXT NOT NULL,
    report_year INTEGER NOT NULL,
    entity_id TEXT NOT NULL,
    entity_name TEXT,
    summary_json TEXT NOT NULL,
    summary_text TEXT,
    source_hash TEXT NOT NULL,
    source_level TEXT NOT NULL,
    provider TEXT,
    model TEXT,
    generated_at TEXT NOT NULL,
    PRIMARY KEY (entity_level, report_year, entity_id)
)
"""

SUMMARY_FIELDS = {
    "headline": "",
    "executive_summary": "",
    "current_situation": "",
    "key_challenges": [],
    "solution_proposals": [],
    "production_or_reserve_opportunities": [],
    "management_attention": [],
    "data_quality_notes": [],
    "ksmi_context": {
        "resource_classes": [],
        "project_levels": [],
        "constraint_types": [],
        "mentions_groovy": False,
        "mentions_pod_or_pse": False,
    },
    "source_coverage": {
        "source_items_reviewed": 0,
        "source_items_with_material_issues": 0,
    },
}

KSMI_PROMPT_CONTEXT = """
Konteks KSMI ringkas:
- KSMI membedakan Reserves & GRR, Contingent Resources, dan Prospective Resources.
- Risk berbeda dari uncertainty: risk terkait probabilitas kejadian countable,
  sedangkan uncertainty terkait rentang estimasi volume setelah akumulasi dikonfirmasi.
- Project level E0-E8, X0-X6, A1-A2 merefleksikan kematangan proyek dan peluang
  komersial. WAP adalah acuan status tahunan.
- GROOVY adalah strategi jangka panjang yang memuat executive summary, current
  situation, key challenges, solution proposal, dan timeline.
- POD/POP/POFD/OPL/OPLL adalah izin berproduksi; PSE bukan izin berproduksi.
- Kendala penting dapat berupa teknis, komersial, regulasi/legal, sosial-lingkungan,
  fasilitas, data, atau ketidakpastian subsurface.
""".strip()


@dataclass(frozen=True)
class SummaryRunResult:
    """Counts from one summarize run."""

    fields_created: int = 0
    fields_skipped: int = 0
    working_areas_created: int = 0
    working_areas_skipped: int = 0
    nkri_created: int = 0
    nkri_skipped: int = 0

    def add(self, level: EntityLevel, created: int, skipped: int) -> SummaryRunResult:
        values = self.__dict__.copy()
        if level == "field":
            values["fields_created"] += created
            values["fields_skipped"] += skipped
        elif level == "working_area":
            values["working_areas_created"] += created
            values["working_areas_skipped"] += skipped
        else:
            values["nkri_created"] += created
            values["nkri_skipped"] += skipped
        return SummaryRunResult(**values)


class SummaryDependencyError(RuntimeError):
    """Raised when an aggregate summary depends on missing lower-level summaries."""


class SummaryLookupError(RuntimeError):
    """Raised when a requested summary cannot be found."""


def ensure_summary_table(conn: duckdb.DuckDBPyConnection) -> None:
    """Create the resource summary annotation table if needed."""
    conn.execute(SUMMARY_COLUMNS_SQL)


def refresh_resource_views(conn: duckdb.DuckDBPyConnection) -> None:
    """Recreate static resource views so summary columns are exposed."""
    try:
        script = _load_sql_script("create_esdc_view.sql")
        for statement in [s.strip() for s in script.split(";") if s.strip()]:
            conn.execute(statement)
    except duckdb.Error as exc:
        console.print(
            "[yellow]Warning:[/yellow] resource views were not refreshed: "
            f"{exc}"
        )


def summarize_resources(
    *,
    year: int,
    target: str = "all",
    name: str | None = None,
    force: bool = False,
    retry: int = 0,
) -> SummaryRunResult:
    """Run field/WK/NKRI executive summary generation for one report year."""
    normalized_target = _normalize_summarize_target(target)
    if normalized_target in {"all", "nkri"} and name:
        raise ValueError(f"esdc summarize {normalized_target} does not accept a name.")

    db_path = Config.get_db_file()
    _ensure_duckdb_database(db_path)
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    provider_config = Config.get_provider_config()
    if not provider_config:
        raise ValueError("No provider configured. Run 'esdc chat --setup' first.")

    from esdc.providers import create_llm_from_config

    llm = create_llm_from_config(provider_config)
    provider_name = str(
        provider_config.get("name") or provider_config.get("provider_type") or ""
    )
    model_name = str(provider_config.get("model") or "")

    conn = get_duckdb_connection(db_path)
    result = SummaryRunResult()
    try:
        ensure_summary_table(conn)
        if normalized_target in {"all", "field"}:
            field_rows = None
            if normalized_target == "field" and name:
                field_rows = [_resolve_field(conn, year, name)]
            created, skipped = _summarize_fields(
                conn,
                llm,
                year,
                provider_name,
                model_name,
                force,
                field_rows=field_rows,
                retry=retry,
            )
            result = result.add("field", created, skipped)
        if normalized_target in {"all", "working_area"}:
            _assert_level_complete(conn, year, "field")
            working_area_rows = None
            if normalized_target == "working_area" and name:
                working_area_rows = [_resolve_working_area(conn, year, name)]
            created, skipped = _summarize_working_areas(
                conn,
                llm,
                year,
                provider_name,
                model_name,
                force,
                working_area_rows=working_area_rows,
                retry=retry,
            )
            result = result.add("working_area", created, skipped)
        if normalized_target in {"all", "nkri"}:
            _assert_level_complete(conn, year, "working_area")
            created, skipped = _summarize_nkri(
                conn, llm, year, provider_name, model_name, force, retry=retry
            )
            result = result.add("nkri", created, skipped)
        refresh_resource_views(conn)
        conn.execute("CHECKPOINT")
    finally:
        conn.close()

    from esdc.chat.tools import invalidate_tool_cache, reset_sql_cache

    reset_sql_cache()
    invalidate_tool_cache()
    return result


def _normalize_summarize_target(target: str) -> SummarizeTarget:
    normalized = target.strip().lower().replace("-", "_")
    aliases = {
        "all": "all",
        "field": "field",
        "fields": "field",
        "wk": "working_area",
        "wa": "working_area",
        "working_area": "working_area",
        "working_areas": "working_area",
        "work_area": "working_area",
        "nkri": "nkri",
        "national": "nkri",
        "nasional": "nkri",
    }
    if normalized not in aliases:
        raise ValueError("Summarize target must be one of: all, field, wk, nkri.")
    return aliases[normalized]  # type: ignore[return-value]


def get_resource_summary(
    *,
    level: str,
    year: int,
    name: str | None = None,
) -> dict[str, Any]:
    """Fetch one generated resource summary for CLI display."""
    db_path = Config.get_db_file()
    _ensure_duckdb_database(db_path)
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    normalized_level = _normalize_summary_level(level)
    conn = get_duckdb_connection(db_path, read_only=True)
    try:
        if normalized_level == "field":
            if not name:
                raise SummaryLookupError("Field summary requires a field name.")
            entity_id, entity_name = _resolve_field(conn, year, name)
        elif normalized_level == "working_area":
            if not name:
                raise SummaryLookupError("Working area summary requires a WK name.")
            entity_id, entity_name = _resolve_working_area(conn, year, name)
        else:
            entity_id, entity_name = "NKRI", "NKRI"

        row = conn.execute(
            f"""
            SELECT entity_level, report_year, entity_id, entity_name, summary_json,
                   summary_text, source_level, provider, model, generated_at
            FROM {SUMMARY_TABLE}
            WHERE entity_level = ?
              AND report_year = ?
              AND entity_id = ?
            """,
            [normalized_level, year, str(entity_id)],
        ).fetchone()
        if row is None:
            raise SummaryLookupError(
                _missing_summary_message(
                    normalized_level, year, str(entity_name or entity_id)
                )
            )
        summary_json = json.loads(row[4])
        return {
            "entity_level": row[0],
            "report_year": row[1],
            "entity_id": row[2],
            "entity_name": row[3],
            "summary": summary_json,
            "summary_text": row[5],
            "source_level": row[6],
            "provider": row[7],
            "model": row[8],
            "generated_at": row[9],
        }
    except duckdb.CatalogException as exc:
        raise SummaryLookupError(
            f"No summaries have been generated yet. Run 'esdc summarize all --year "
            f"{year}' first."
        ) from exc
    finally:
        conn.close()


def _normalize_summary_level(level: str) -> EntityLevel:
    normalized = level.strip().lower().replace("-", "_")
    aliases = {
        "field": "field",
        "fields": "field",
        "wk": "working_area",
        "wa": "working_area",
        "working_area": "working_area",
        "working_areas": "working_area",
        "work_area": "working_area",
        "nkri": "nkri",
        "national": "nkri",
        "nasional": "nkri",
    }
    if normalized not in aliases:
        raise SummaryLookupError(
            "Summary level must be one of: field, wk, working_area, nkri."
        )
    return aliases[normalized]  # type: ignore[return-value]


def _missing_summary_message(level: EntityLevel, year: int, name: str) -> str:
    if level == "field":
        return (
            f"No field summary found for {name} in {year}. Run: "
            f'esdc summarize field "{name}" --year {year}'
        )
    if level == "working_area":
        return (
            f"No working area summary found for {name} in {year}. Run: "
            f'esdc summarize wk "{name}" --year {year}'
        )
    return (
        f"No NKRI summary found for {year}. Run: "
        f"esdc summarize nkri --year {year}"
    )


def _summarize_fields(
    conn: duckdb.DuckDBPyConnection,
    llm: Any,
    year: int,
    provider: str,
    model: str,
    force: bool,
    field_rows: list[tuple[Any, Any]] | None = None,
    retry: int = 0,
) -> tuple[int, int]:
    rows = field_rows
    if rows is None:
        rows = conn.execute(
            """
            SELECT
                COALESCE(NULLIF(field_id, ''), field_name) AS entity_id,
                MIN(field_name) AS entity_name
            FROM project_resources
            WHERE report_year = ?
            GROUP BY COALESCE(NULLIF(field_id, ''), field_name)
            ORDER BY entity_name
            """,
            [year],
        ).fetchall()
    created = skipped = 0
    description = "Summarizing fields"
    if field_rows and len(field_rows) == 1:
        description = f"Summarizing fields: {field_rows[0][1]}"
    with _summary_progress() as progress:
        task = progress.add_task(description, total=len(rows))
        for entity_id, entity_name in rows:
            source_items = _field_source_items(conn, year, entity_id)
            metrics = _field_metrics(conn, year, entity_id)
            did_create = _summarize_entity(
                conn=conn,
                llm=llm,
                level="field",
                year=year,
                entity_id=str(entity_id),
                entity_name=str(entity_name or entity_id),
                source_level="project_remarks",
                source_items=source_items,
                metrics=metrics,
                provider=provider,
                model=model,
                force=force,
                retry=retry,
            )
            created += int(did_create)
            skipped += int(not did_create)
            progress.advance(task)
    return created, skipped


def _summarize_working_areas(
    conn: duckdb.DuckDBPyConnection,
    llm: Any,
    year: int,
    provider: str,
    model: str,
    force: bool,
    working_area_rows: list[tuple[Any, Any]] | None = None,
    retry: int = 0,
) -> tuple[int, int]:
    rows = working_area_rows
    if rows is None:
        rows = conn.execute(
            """
            SELECT
                COALESCE(NULLIF(wk_id, ''), wk_name) AS entity_id,
                MIN(wk_name) AS entity_name
            FROM project_resources
            WHERE report_year = ?
            GROUP BY COALESCE(NULLIF(wk_id, ''), wk_name)
            ORDER BY entity_name
            """,
            [year],
        ).fetchall()
    created = skipped = 0
    description = "Summarizing working areas"
    if working_area_rows and len(working_area_rows) == 1:
        description = f"Summarizing working areas: {working_area_rows[0][1]}"
    with _summary_progress() as progress:
        task = progress.add_task(description, total=len(rows))
        for entity_id, entity_name in rows:
            source_items = _working_area_source_items(conn, year, entity_id)
            metrics = _working_area_metrics(conn, year, entity_id)
            did_create = _summarize_entity(
                conn=conn,
                llm=llm,
                level="working_area",
                year=year,
                entity_id=str(entity_id),
                entity_name=str(entity_name or entity_id),
                source_level="field_summary",
                source_items=source_items,
                metrics=metrics,
                provider=provider,
                model=model,
                force=force,
                retry=retry,
            )
            created += int(did_create)
            skipped += int(not did_create)
            progress.advance(task)
    return created, skipped


def _summarize_nkri(
    conn: duckdb.DuckDBPyConnection,
    llm: Any,
    year: int,
    provider: str,
    model: str,
    force: bool,
    retry: int = 0,
) -> tuple[int, int]:
    source_items = _nkri_source_items(conn, year)
    metrics = _nkri_metrics(conn, year)
    with _summary_progress() as progress:
        task = progress.add_task("Summarizing NKRI", total=1)
        did_create = _summarize_entity(
            conn=conn,
            llm=llm,
            level="nkri",
            year=year,
            entity_id="NKRI",
            entity_name="NKRI",
            source_level="wk_summary",
            source_items=source_items,
            metrics=metrics,
            provider=provider,
            model=model,
            force=force,
            retry=retry,
        )
        progress.advance(task)
    return int(did_create), int(not did_create)


def _summary_progress() -> Progress:
    return Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=36),
        TextColumn("[green]{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    )


def _resolve_field(
    conn: duckdb.DuckDBPyConnection, year: int, field_name: str
) -> tuple[Any, Any]:
    normalized = field_name.strip().lower()
    if not normalized:
        raise ValueError("Field name cannot be empty.")

    exact_rows = conn.execute(
        """
        SELECT
            COALESCE(NULLIF(field_id, ''), field_name) AS entity_id,
            MIN(field_name) AS entity_name
        FROM project_resources
        WHERE report_year = ?
          AND lower(trim(field_name)) = ?
        GROUP BY COALESCE(NULLIF(field_id, ''), field_name)
        ORDER BY entity_name
        """,
        [year, normalized],
    ).fetchall()
    if len(exact_rows) == 1:
        return exact_rows[0]
    if len(exact_rows) > 1:
        _raise_ambiguous_field(field_name, exact_rows)

    fallback_rows = conn.execute(
        """
        SELECT
            COALESCE(NULLIF(field_id, ''), field_name) AS entity_id,
            MIN(field_name) AS entity_name
        FROM project_resources
        WHERE report_year = ?
          AND field_name ILIKE ?
        GROUP BY COALESCE(NULLIF(field_id, ''), field_name)
        ORDER BY entity_name
        """,
        [year, f"%{field_name.strip()}%"],
    ).fetchall()
    if len(fallback_rows) == 1:
        return fallback_rows[0]
    if len(fallback_rows) > 1:
        _raise_ambiguous_field(field_name, fallback_rows)
    raise ValueError(f"No field named '{field_name}' found for report year {year}.")


def _resolve_working_area(
    conn: duckdb.DuckDBPyConnection, year: int, wk_name: str
) -> tuple[Any, Any]:
    normalized = wk_name.strip().lower()
    if not normalized:
        raise SummaryLookupError("Working area name cannot be empty.")

    exact_rows = conn.execute(
        """
        SELECT
            COALESCE(NULLIF(wk_id, ''), wk_name) AS entity_id,
            MIN(wk_name) AS entity_name
        FROM project_resources
        WHERE report_year = ?
          AND lower(trim(wk_name)) = ?
        GROUP BY COALESCE(NULLIF(wk_id, ''), wk_name)
        ORDER BY entity_name
        """,
        [year, normalized],
    ).fetchall()
    if len(exact_rows) == 1:
        return exact_rows[0]
    if len(exact_rows) > 1:
        _raise_ambiguous_entity("working area", wk_name, exact_rows)

    fallback_rows = conn.execute(
        """
        SELECT
            COALESCE(NULLIF(wk_id, ''), wk_name) AS entity_id,
            MIN(wk_name) AS entity_name
        FROM project_resources
        WHERE report_year = ?
          AND wk_name ILIKE ?
        GROUP BY COALESCE(NULLIF(wk_id, ''), wk_name)
        ORDER BY entity_name
        """,
        [year, f"%{wk_name.strip()}%"],
    ).fetchall()
    if len(fallback_rows) == 1:
        return fallback_rows[0]
    if len(fallback_rows) > 1:
        _raise_ambiguous_entity("working area", wk_name, fallback_rows)
    raise SummaryLookupError(
        f"No working area named '{wk_name}' found for report year {year}."
    )


def _raise_ambiguous_field(field_name: str, rows: list[tuple[Any, Any]]) -> None:
    _raise_ambiguous_entity("field", field_name, rows)


def _raise_ambiguous_entity(
    entity_label: str, entity_name: str, rows: list[tuple[Any, Any]]
) -> None:
    candidates = ", ".join(f"{name} ({entity_id})" for entity_id, name in rows[:10])
    more = "" if len(rows) <= 10 else f", and {len(rows) - 10} more"
    raise ValueError(
        f"Ambiguous {entity_label} name '{entity_name}'. "
        f"Matching {entity_label}s: {candidates}{more}."
    )


def _summarize_entity(
    *,
    conn: duckdb.DuckDBPyConnection,
    llm: Any,
    level: EntityLevel,
    year: int,
    entity_id: str,
    entity_name: str,
    source_level: str,
    source_items: list[dict[str, Any]],
    metrics: dict[str, Any],
    provider: str,
    model: str,
    force: bool,
    retry: int = 0,
) -> bool:
    source_hash = _hash_source(source_items, metrics)
    if not force and _existing_hash_matches(conn, level, year, entity_id, source_hash):
        return False

    if not _has_nonempty_source(source_items):
        summary = _empty_summary(level, entity_name, source_items)
    else:
        source_items_for_prompt = _source_items_with_nonempty_source(source_items)
        prompt = build_summary_prompt(
            level=level,
            year=year,
            entity_name=entity_name,
            source_level=source_level,
            source_items=source_items_for_prompt,
            metrics=metrics,
            source_quality=_source_quality_context(source_items),
        )
        last_error: str | None = None
        for attempt in range(retry + 1):
            retry_prompt = (
                prompt + _retry_feedback(last_error) if last_error else prompt
            )
            content, actual_provider, actual_model = _invoke_llm_with_metadata(
                llm,
                retry_prompt,
                fallback_provider=provider,
                fallback_model=model,
            )
            provider = actual_provider
            model = actual_model
            try:
                summary = _parse_summary_response(content)
                break
            except (json.JSONDecodeError, ValueError) as e:
                msg = (
                    f"[yellow]Attempt {attempt + 1}/{retry + 1} failed[/yellow] "
                    f"for {level} '[bold]{entity_name}[/bold]' "
                    f"({type(e).__name__}): {e}"
                )
                logger.warning("%s | Response: %.200s", msg, content.strip()[:200])
                console.print(msg)
                if attempt < retry:
                    last_error = str(e)
                else:
                    raise
        else:
            raise RuntimeError(
                "Retry loop exhausted without success or raise."
            )
        summary = _normalize_summary(summary, len(source_items_for_prompt))

    summary_json = json.dumps(summary, ensure_ascii=False, sort_keys=True)
    summary_text = _summary_text(summary)
    generated_at = datetime.now(timezone.utc).isoformat()
    conn.execute(
        f"""
        INSERT OR REPLACE INTO {SUMMARY_TABLE}
            (entity_level, report_year, entity_id, entity_name, summary_json,
             summary_text, source_hash, source_level, provider, model, generated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            level,
            year,
            entity_id,
            entity_name,
            summary_json,
            summary_text,
            source_hash,
            source_level,
            provider,
            model,
            generated_at,
        ],
    )
    return True


def build_summary_prompt(
    *,
    level: EntityLevel,
    year: int,
    entity_name: str,
    source_level: str,
    source_items: list[dict[str, Any]],
    metrics: dict[str, Any],
    source_quality: dict[str, Any] | None = None,
) -> str:
    """Build the executive summary prompt sent to the configured LLM."""
    level_label = {
        "field": "FIELD",
        "working_area": "WORKING AREA",
        "nkri": "NKRI",
    }[level]
    source_json = json.dumps(source_items, ensure_ascii=False, indent=2)
    metrics_json = json.dumps(metrics, ensure_ascii=False, indent=2)
    source_quality_json = json.dumps(
        source_quality or _source_quality_context(source_items),
        ensure_ascii=False,
        indent=2,
    )
    aggregate_note = (
        "Project remarks tetap tersedia di level project; ringkasan ini harus "
        "menjadi sintesis level field."
        if level == "field"
        else "Sumber adalah summary level bawah; sintesis tema lintas aset dan "
        "hindari sekadar menggabungkan ulang semua detail."
    )
    return f"""
Anda adalah analis senior SKK Migas yang menyusun executive summary untuk
dashboard Eureka.

Level ringkasan: {level_label}
Entitas: {entity_name}
Tahun laporan: {year}
Sumber utama: {source_level}

{KSMI_PROMPT_CONTEXT}

Tujuan:
- Memberi manajemen gambaran jelas mengenai current situation, isu utama, kendala,
  risiko, solusi/tindak lanjut, dan hal yang perlu keputusan manajemen.
- Menjaga semua informasi material dari source agar tidak hilang.
- {aggregate_note}

Aturan:
1. Jangan menghilangkan isu, kendala, risiko, solusi, atau tindak lanjut material.
2. Jika beberapa source menyampaikan isu yang sama, gabungkan menjadi satu tema dan
   sebutkan cakupannya.
3. Jika ada isu spesifik yang material, tetap sebutkan meskipun hanya muncul sekali.
4. Jangan membuat asumsi baru di luar source dan technical context.
5. Jangan mengubah angka, status, nama project, field, WK, atau istilah KSMI.
6. Jika ada opportunity peningkatan produksi, optimasi produksi, percepatan onstream,
   EOR/IOR, workover, infill, facility debottlenecking, atau kegiatan lain yang
   dapat menaikkan produksi, highlight.
7. Jika ada opportunity penambahan cadangan/resources, maturation dari Contingent
   atau Prospective ke Reserves, unlock volume, revisi POD/OPL/POFD, atau pengurangan
   risiko/ketidakpastian, highlight.
8. Gunakan angka teknis hanya sebagai konteks materialitas, bukan pengganti isi source.
9. Tulis dalam Bahasa Indonesia formal, ringkas, dan cocok untuk executive dashboard.
10. Isi data_quality_notes hanya jika source_quality menunjukkan remarks kosong,
   terlalu pendek, placeholder, atau tidak informatif. Jika tidak ada masalah
   kualitas data, data_quality_notes wajib berupa array kosong [] dan jangan
   menulis catatan seperti "tidak ada isu kualitas data".
11. Bila masalah kualitas remarks material untuk manajemen, masukkan juga ke
   management_attention dengan bahasa netral seperti "Operator belum membuat
   remarks yang cukup informatif".

Technical context key metrics:
{metrics_json}

Source quality:
{source_quality_json}

Source items:
{source_json}

Kembalikan hanya JSON valid dengan struktur:
{{
  "headline": string,
  "executive_summary": string,
  "current_situation": string,
  "key_challenges": string[],
  "solution_proposals": string[],
  "production_or_reserve_opportunities": string[],
  "management_attention": string[],
  "data_quality_notes": string[],
  "ksmi_context": {{
    "resource_classes": string[],
    "project_levels": string[],
    "constraint_types": string[],
    "mentions_groovy": boolean,
    "mentions_pod_or_pse": boolean
  }},
  "source_coverage": {{
    "source_items_reviewed": number,
    "source_items_with_material_issues": number
  }}
}}

Batas panjang:
- headline maksimal 25 kata.
- executive_summary maksimal 120 kata.
- setiap array maksimal 5 butir, kecuali management_attention maksimal 3 butir.
""".strip()


def _field_source_items(
    conn: duckdb.DuckDBPyConnection, year: int, entity_id: str
) -> list[dict[str, Any]]:
    rows = conn.execute(
        f"""
        SELECT {', '.join(_available_columns(conn, 'project_resources', [
            'project_id', 'project_name', 'field_name', 'wk_name', 'project_class',
            'project_stage', 'project_level', 'uncert_level', 'project_remarks'
        ]))}
        FROM project_resources
        WHERE report_year = ?
          AND COALESCE(NULLIF(field_id, ''), field_name) = ?
        ORDER BY project_name
        """,
        [year, entity_id],
    ).fetchall()
    cols = _available_columns(
        conn,
        "project_resources",
        [
            "project_id",
            "project_name",
            "field_name",
            "wk_name",
            "project_class",
            "project_stage",
            "project_level",
            "uncert_level",
            "project_remarks",
        ],
    )
    return [dict(zip(cols, row, strict=False)) for row in rows]


def _working_area_source_items(
    conn: duckdb.DuckDBPyConnection, year: int, entity_id: str
) -> list[dict[str, Any]]:
    rows = conn.execute(
        f"""
        SELECT DISTINCT
            COALESCE(NULLIF(pr.field_id, ''), pr.field_name) AS field_id,
            MIN(pr.field_name) AS field_name,
            rs.summary_text
        FROM project_resources pr
        JOIN {SUMMARY_TABLE} rs
          ON rs.entity_level = 'field'
         AND rs.report_year = pr.report_year
         AND rs.entity_id = COALESCE(NULLIF(pr.field_id, ''), pr.field_name)
        WHERE pr.report_year = ?
          AND COALESCE(NULLIF(pr.wk_id, ''), pr.wk_name) = ?
        GROUP BY COALESCE(NULLIF(pr.field_id, ''), pr.field_name), rs.summary_text
        ORDER BY field_name
        """,
        [year, entity_id],
    ).fetchall()
    return [
        {"field_id": row[0], "field_name": row[1], "field_summary": row[2]}
        for row in rows
    ]


def _nkri_source_items(
    conn: duckdb.DuckDBPyConnection, year: int
) -> list[dict[str, Any]]:
    rows = conn.execute(
        f"""
        SELECT entity_id, entity_name, summary_text
        FROM {SUMMARY_TABLE}
        WHERE entity_level = 'working_area'
          AND report_year = ?
        ORDER BY entity_name
        """,
        [year],
    ).fetchall()
    return [
        {"wk_id": row[0], "wk_name": row[1], "wk_summary": row[2]} for row in rows
    ]


def _field_metrics(
    conn: duckdb.DuckDBPyConnection, year: int, entity_id: str
) -> dict[str, Any]:
    return _project_resource_metrics(
        conn,
        year,
        "COALESCE(NULLIF(field_id, ''), field_name) = ?",
        [entity_id],
    )


def _working_area_metrics(
    conn: duckdb.DuckDBPyConnection, year: int, entity_id: str
) -> dict[str, Any]:
    return _project_resource_metrics(
        conn,
        year,
        "COALESCE(NULLIF(wk_id, ''), wk_name) = ?",
        [entity_id],
    )


def _nkri_metrics(conn: duckdb.DuckDBPyConnection, year: int) -> dict[str, Any]:
    return _project_resource_metrics(conn, year, "1 = 1", [])


def _project_resource_metrics(
    conn: duckdb.DuckDBPyConnection,
    year: int,
    where_sql: str,
    params: list[Any],
) -> dict[str, Any]:
    columns = _column_set(conn, "project_resources")
    numeric_candidates = [
        "res_oc",
        "res_an",
        "rec_oc",
        "rec_an",
        "rec_oc_risked",
        "rec_an_risked",
        "prj_ioip",
        "prj_igip",
        "rate_sls_oc",
        "rate_sls_an",
        "cprd_sls_oc",
        "cprd_sls_an",
    ]
    sums = {
        name: f"SUM({name}) AS {name}"
        for name in numeric_candidates
        if name in columns
    }
    count_parts = ["COUNT(*) AS project_count"]
    if "field_id" in columns:
        count_parts.append("COUNT(DISTINCT field_id) AS field_count")
    elif "field_name" in columns:
        count_parts.append("COUNT(DISTINCT field_name) AS field_count")
    if "wk_id" in columns:
        count_parts.append("COUNT(DISTINCT wk_id) AS wk_count")
    elif "wk_name" in columns:
        count_parts.append("COUNT(DISTINCT wk_name) AS wk_count")
    sql = (
        "SELECT "
        + ", ".join(count_parts + list(sums.values()))
        + f" FROM project_resources WHERE report_year = ? AND {where_sql}"
    )
    row = conn.execute(sql, [year, *params]).fetchone()
    names = [part.split(" AS ")[-1] for part in count_parts + list(sums.values())]
    metrics = dict(zip(names, row or (), strict=False))
    for col in ("project_class", "project_stage", "project_level", "uncert_level"):
        if col in columns:
            metrics[f"{col}_mix"] = _value_mix(conn, col, year, where_sql, params)
    return metrics


def _value_mix(
    conn: duckdb.DuckDBPyConnection,
    column: str,
    year: int,
    where_sql: str,
    params: list[Any],
) -> list[dict[str, Any]]:
    rows = conn.execute(
        f"""
        SELECT {column}, COUNT(*) AS count
        FROM project_resources
        WHERE report_year = ? AND {where_sql}
        GROUP BY {column}
        ORDER BY count DESC, {column}
        LIMIT 10
        """,
        [year, *params],
    ).fetchall()
    return [{"value": row[0], "count": row[1]} for row in rows]


def _assert_level_complete(
    conn: duckdb.DuckDBPyConnection, year: int, level: EntityLevel
) -> None:
    if level == "field":
        expected = conn.execute(
            """
            SELECT COUNT(*)
            FROM (
                SELECT DISTINCT COALESCE(NULLIF(field_id, ''), field_name)
                FROM project_resources
                WHERE report_year = ?
            )
            """,
            [year],
        ).fetchone()[0]
    else:
        expected = conn.execute(
            """
            SELECT COUNT(*)
            FROM (
                SELECT DISTINCT COALESCE(NULLIF(wk_id, ''), wk_name)
                FROM project_resources
                WHERE report_year = ?
            )
            """,
            [year],
        ).fetchone()[0]
    actual = conn.execute(
        f"""
        SELECT COUNT(*)
        FROM {SUMMARY_TABLE}
        WHERE entity_level = ?
          AND report_year = ?
        """,
        [level, year],
    ).fetchone()[0]
    if expected == 0:
        raise SummaryDependencyError(
            f"No project_resources rows found for report year {year}."
        )
    if actual < expected:
        name = "field summaries" if level == "field" else "working area summaries"
        raise SummaryDependencyError(
            f"Missing {name} for {year}: expected {expected}, found {actual}. "
            "Run the lower-level summarize step first."
        )


def _existing_hash_matches(
    conn: duckdb.DuckDBPyConnection,
    level: EntityLevel,
    year: int,
    entity_id: str,
    source_hash: str,
) -> bool:
    row = conn.execute(
        f"""
        SELECT source_hash
        FROM {SUMMARY_TABLE}
        WHERE entity_level = ?
          AND report_year = ?
          AND entity_id = ?
        """,
        [level, year, entity_id],
    ).fetchone()
    return bool(row and row[0] == source_hash)


def _hash_source(source_items: list[dict[str, Any]], metrics: dict[str, Any]) -> str:
    payload = json.dumps(
        {"source_items": source_items, "metrics": metrics},
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _invoke_llm_with_metadata(
    llm: Any,
    prompt: str,
    *,
    fallback_provider: str,
    fallback_model: str,
) -> tuple[str, str, str]:
    response = llm.invoke(prompt)
    content = getattr(response, "content", response)
    if isinstance(content, list):
        content_text = "\n".join(str(item) for item in content)
    else:
        content_text = str(content)

    provider = (
        getattr(llm, "last_provider_name", None)
        or getattr(llm, "_esdc_provider_name", None)
        or fallback_provider
    )
    model = (
        getattr(llm, "last_model_name", None)
        or getattr(llm, "_esdc_model_name", None)
        or fallback_model
    )
    return content_text, str(provider or ""), str(model or "")


def _repair_json(raw: str) -> str:
    """Repair common LLM JSON formatting issues before parsing."""
    s = raw.strip()
    # Remove trailing comma before closing braces/brackets
    s = re.sub(r",\s*}", "}", s)
    s = re.sub(r",\s*]", "]", s)
    # Quote unquoted keys: {key: value} -> {"key": value}
    s = re.sub(r"([{,]\s*)([a-zA-Z_]\w*)\s*:", r'\1"\2":', s)
    # Replace Python booleans/null
    s = s.replace(": True", ": true").replace(": False", ": false")
    s = s.replace(": None", ": null")
    return s


def _parse_summary_response(content: str) -> dict[str, Any]:
    cleaned = content.strip()
    if cleaned.startswith("<think>"):
        end_idx = cleaned.find("</think>")
        if end_idx != -1:
            cleaned = cleaned[end_idx + len("</think>"):].strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1:
        cleaned = cleaned[start : end + 1]
    before_repair = cleaned
    cleaned = _repair_json(cleaned)
    if cleaned != before_repair:
        logger.debug("JSON repair applied: %.150s", before_repair)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        logger.warning(
            "Invalid JSON from LLM (after repair). Raw[%.300s] | Repaired[%.300s]",
            content.strip()[:300],
            cleaned[:300],
        )
        raise
    if not isinstance(parsed, dict):
        raise ValueError("Summary response must be a JSON object.")
    return parsed


def _retry_feedback(error: str) -> str:
    return (
        f"\n\nPercobaan sebelumnya gagal dengan error JSON: {error}\n"
        "Hanya kembalikan JSON valid sesuai struktur yang diminta."
    )


def _normalize_summary(summary: dict[str, Any], source_count: int) -> dict[str, Any]:
    normalized = json.loads(json.dumps(SUMMARY_FIELDS))
    for key in normalized:
        if key in summary:
            normalized[key] = summary[key]
    coverage = normalized.setdefault("source_coverage", {})
    coverage.setdefault("source_items_reviewed", source_count)
    coverage.setdefault("source_items_with_material_issues", 0)
    normalized["data_quality_notes"] = _normalize_data_quality_notes(
        normalized.get("data_quality_notes")
    )
    return normalized


def _normalize_data_quality_notes(value: Any) -> list[str]:
    if value is None:
        return []
    values = value if isinstance(value, list) else [value]
    notes = [str(item).strip() for item in values if str(item).strip()]
    benign_phrases = (
        "tidak ada isu kualitas data",
        "tidak ada masalah kualitas data",
        "tidak ada catatan kualitas data",
        "kualitas data memadai",
        "remarks sudah memadai",
        "no data quality issue",
        "no data quality issues",
    )
    return [
        note
        for note in notes
        if not any(phrase in note.lower() for phrase in benign_phrases)
    ]


def _empty_summary(
    level: EntityLevel, entity_name: str, source_items: list[dict[str, Any]]
) -> dict[str, Any]:
    label = {"field": "field", "working_area": "WK", "nkri": "NKRI"}[level]
    summary = json.loads(json.dumps(SUMMARY_FIELDS))
    summary["headline"] = f"Tidak ada remarks material untuk {label} {entity_name}."
    summary["executive_summary"] = (
        "Tidak terdapat remarks material yang dapat disintesis untuk periode ini."
    )
    summary["current_situation"] = "Tidak ada remarks material pada source."
    summary["data_quality_notes"] = [
        "Source tidak memiliki remarks yang cukup informatif untuk diringkas."
    ]
    summary["source_coverage"]["source_items_reviewed"] = len(source_items)
    return summary


def _summary_text(summary: dict[str, Any]) -> str:
    parts = [
        str(summary.get("headline") or "").strip(),
        str(summary.get("executive_summary") or "").strip(),
    ]
    return "\n\n".join(part for part in parts if part)


def _has_nonempty_source(source_items: list[dict[str, Any]]) -> bool:
    for item in source_items:
        for key in ("project_remarks", "field_summary", "wk_summary", "summary_text"):
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                return True
    return False


def _source_items_with_nonempty_source(
    source_items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    source_keys = ("project_remarks", "field_summary", "wk_summary", "summary_text")
    return [
        item
        for item in source_items
        if any(
            isinstance(item.get(key), str) and item.get(key, "").strip()
            for key in source_keys
        )
    ]


def _source_quality_context(source_items: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(source_items)
    empty = 0
    low_quality: list[dict[str, Any]] = []
    for item in source_items:
        text = _source_item_text(item)
        if not text:
            empty += 1
            continue
        if _is_low_quality_source_text(text):
            low_quality.append(
                {
                    "project_id": item.get("project_id"),
                    "project_name": item.get("project_name"),
                    "field_name": item.get("field_name"),
                    "wk_name": item.get("wk_name"),
                    "text": text[:160],
                    "reason": (
                        "Remarks terlalu pendek, placeholder, atau tidak informatif."
                    ),
                }
            )
    return {
        "source_items_total": total,
        "source_items_with_empty_text": empty,
        "source_items_with_low_quality_text": len(low_quality),
        "low_quality_examples": low_quality[:10],
    }


def _source_item_text(item: dict[str, Any]) -> str:
    for key in ("project_remarks", "field_summary", "wk_summary", "summary_text"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _is_low_quality_source_text(text: str) -> bool:
    normalized = " ".join(text.lower().strip().split())
    if normalized in {
        "-",
        "--",
        "n/a",
        "na",
        "nil",
        "none",
        "null",
        "test",
        "testing",
        "asdf",
        "qwerty",
        "ok",
        "oke",
        "done",
        "clear",
        "no issue",
        "tidak ada",
        "tidak ada remarks",
        "belum ada",
    }:
        return True
    words = [word for word in normalized.replace(".", " ").split() if word]
    if len(words) <= 2:
        return True
    alnum = [char for char in normalized if char.isalnum()]
    return bool(alnum and len(set(alnum)) <= 2)


def _available_columns(
    conn: duckdb.DuckDBPyConnection, table: str, candidates: list[str]
) -> list[str]:
    columns = _column_set(conn, table)
    available = [column for column in candidates if column in columns]
    if not available:
        raise ValueError(f"None of the requested columns exist in {table}.")
    return available


def _column_set(conn: duckdb.DuckDBPyConnection, table: str) -> set[str]:
    with contextlib.suppress(duckdb.Error):
        rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
        return {row[1] for row in rows}
    return set()
