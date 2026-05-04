"""LangChain tool for standardized simple data queries.

Provides ``simple_data_query`` — a single tool that builds, executes,
and formats aggregate SQL for common ESDC data questions.
"""

import json
import logging
from typing import Any

from langchain_core.tools import tool

from esdc.configs import Config
from esdc.dbmanager import get_duckdb_connection
from esdc.selection import TableName
from esdc.view_builder import (
    QUERY_TYPE_DETAIL,
    build_smart_query,
)

logger = logging.getLogger(__name__)

ENTITY_LEVEL_TABLE: dict[str, TableName] = {
    "field": TableName.FIELD_RESOURCES,
    "work_area": TableName.WA_RESOURCES,
    "national": TableName.NKRI_RESOURCES,
}

QUERY_TYPE_DISPLAY: dict[str, str] = {
    "reserves": "Cadangan (Reserves)",
    "resources": "Potensi (Resources/GRR)",
    "contingent": "Contingent Resources",
    "prospective": "Prospective Resources (Risked)",
    "cumprod": "Produksi Kumulatif",
    "prodrate": "Rate Produksi",
}

UNCERTAINTY_DISPLAY: dict[str, str] = {
    "P90": "P90 (Low)",
    "P50": "P50 (Best Estimate)",
    "P10": "P10 (High)",
    "1P": "1P (P90/Low)",
    "2P": "2P (P50/Mid)",
    "3P": "3P (P10/High)",
    "1C": "1C (Low)",
    "2C": "2C (Mid)",
    "3C": "3C (High)",
    "1U": "1U (Low)",
    "2U": "2U (Mid)",
    "3U": "3U (High)",
    "1R": "1R (Low)",
    "2R": "2R (Mid)",
    "3R": "3R (High)",
}

# Map (query_type, generic_input) -> context-specific label
CONTEXT_UNCERTAINTY_MAP: dict[str, dict[str, str]] = {
    "reserves": {"P90": "1P", "P50": "2P", "P10": "3P"},
    "resources": {"P90": "1R", "P50": "2R", "P10": "3R"},
    "contingent": {"P90": "1C", "P50": "2C", "P10": "3C"},
    "prospective": {"P90": "1U", "P50": "2U", "P10": "3U"},
    "cumprod": {"P90": "P90", "P50": "P50", "P10": "P10"},
    "prodrate": {"P90": "P90", "P50": "P50", "P10": "P10"},
}


def _resolve_uncertainty_label(query_type: str, uncertainty: str) -> str:
    """Resolve uncertainty shorthand to context-aware display label.

    Generic inputs (P90/P50/P10) are converted to type-specific labels:
      P50 + resources -> "2R (P50/Mid)"
      P50 + reserves -> "2P (P50/Mid)"
      P50 + contingent -> "2C (P50/Mid)"
    Type-specific inputs (2P/2C/2U/2R) are used directly.
    """
    context_map = CONTEXT_UNCERTAINTY_MAP.get(query_type, {})
    specific = context_map.get(uncertainty)
    if specific:
        return UNCERTAINTY_DISPLAY.get(specific, specific)
    return UNCERTAINTY_DISPLAY.get(uncertainty, uncertainty)


def _format_summary(
    query_type: str,
    entity_name: str | None,
    entity_level: str,
    uncertainty: str,
    report_year: int | list[int] | None,
    results: list[dict[str, Any]],
    columns: list[str],
) -> str:
    """Format query results into a human-readable summary string."""
    if not results:
        return "No data found for the specified criteria."

    entity_display = entity_name if entity_name else "Nasional"
    type_display = QUERY_TYPE_DISPLAY.get(query_type, query_type)
    uncert_display = _resolve_uncertainty_label(query_type, uncertainty)

    if isinstance(report_year, list):
        year_display = f" (trend {', '.join(str(y) for y in report_year)})"
    elif report_year:
        year_display = f" (WAP {report_year})"
    else:
        year_display = ""

    lines: list[str] = []
    header = f"{type_display} - {entity_display}{year_display} [{uncert_display}]:"
    lines.append(header)

    # Helper: format numeric columns
    def _fmt_row(row: dict[str, Any], skip_cols: set[str]) -> str:
        parts: list[str] = []
        for col in columns:
            if col in skip_cols:
                continue
            val = row.get(col)
            if val is not None and isinstance(val, (int, float)):
                parts.append(
                    f"{col}: {val:,.2f}"
                    if isinstance(val, float)
                    else f"{col}: {val:,}"
                )
        return ", ".join(parts)

    if "project_class" in columns:
        # Grouped by project_class (potensi)
        for row in results:
            pc = row.get("project_class", "")
            detail = _fmt_row(
                row,
                {"project_class", "project_stage", "report_year", "uncert_level"},
            )
            lines.append(f"  {pc}: {detail}")
    elif len(results) == 1 and "report_year" not in columns:
        # Single aggregate result
        lines.append(f"  {_fmt_row(results[0], {'report_year', 'uncert_level'})}")
    else:
        # Trend/comparison (grouped by report_year)
        for row in results:
            yr = row.get("report_year", "")
            lines.append(f"  {yr}: {_fmt_row(row, {'report_year'})}")

    return "\n".join(lines)


@tool("Simple Data Query")
def simple_data_query(
    query_type: str,
    entity_level: str,
    entity_name: str | None = None,
    uncertainty: str = "P50",
    report_year: int | list[int] | None = None,
) -> str:
    """Execute a standardized aggregate query for simple factual data.

    Use for questions like:
    - "berapa reserves WK X?"
    - "berapa resources yang ada di WK X?"
    - "berapa contingent resources nasional terbaru?"
    - "berapa prospective resources risked lapangan Y?"
    - "berapa produksi kumulatif lapangan Z?"
    - "berapa rate produksi lapangan Z?"

    Parameters
    ----------
    query_type : str
        One of: ``reserves``, ``resources``, ``contingent``, ``prospective``,
        ``cumprod``, ``prodrate``.
    entity_level : str
        One of: ``field``, ``work_area``, ``national``.
    entity_name : str | None
        Entity name (e.g. "Rokan", "Duri").  ``None`` for national queries.
    uncertainty : str
        Uncertainty level.  Default ``"P50"``.
        Generic: ``P90``, ``P50``, ``P10``.
        Reserves: ``1P``, ``2P``, ``3P``.
        Contingent: ``1C``, ``2C``, ``3C``.
        Prospective: ``1U``, ``2U``, ``3U``.
        GRR: ``1R``, ``2R``, ``3R``.
    report_year : int | list[int] | None
        ``None`` → latest available year. ``int`` → single year.
        ``list`` → trend/comparison across years.

    Returns:
    -------
    str
        JSON with ``sql``, ``results``, ``summary``, and metadata.
    """
    # Validate
    valid_types = set(QUERY_TYPE_DETAIL.keys())
    if query_type not in valid_types:
        types_str = ", ".join(sorted(valid_types))
        raise ValueError(
            f"Invalid query_type '{query_type}'. Must be one of: {types_str}"
        )

    valid_levels = set(ENTITY_LEVEL_TABLE.keys())
    if entity_level not in valid_levels:
        levels_str = ", ".join(sorted(valid_levels))
        raise ValueError(
            f"Invalid entity_level '{entity_level}'. Must be one of: {levels_str}"
        )

    # Normalise report_year to list for build_smart_query
    if report_year is None:
        years = None
    elif isinstance(report_year, list):
        years = report_year
    else:
        years = [report_year]

    table = ENTITY_LEVEL_TABLE[entity_level]

    result = build_smart_query(
        query_type=query_type,
        table=table,
        entity_name=entity_name,
        uncertainty=uncertainty,
        report_years=years,
    )

    sql = result["sql"]
    params = result["params"]

    db_path = Config.get_db_file()
    if not db_path.exists():
        return json.dumps(
            {"error": "Database not found. Run 'esdc fetch --save' first."}
        )

    conn = None
    try:
        conn = get_duckdb_connection(db_path, read_only=True)
        rows = conn.execute(sql, params).fetchall()
        columns = [desc[0] for desc in conn.description] if conn.description else []

        # Convert to list of dicts
        results = [dict(zip(columns, row, strict=False)) for row in rows]

        # Derive actual report year(s)
        actual_year: int | list[int] | None = None
        if years and len(years) == 1:
            actual_year = years[0]
        elif years and len(years) > 1:
            actual_year = years
        elif results and "report_year" in results[0]:
            actual_year = results[0].get("report_year")

        summary = _format_summary(
            query_type=query_type,
            entity_name=entity_name,
            entity_level=entity_level,
            uncertainty=uncertainty,
            report_year=actual_year,
            results=results,
            columns=columns,
        )

        return json.dumps(
            {
                "query_type": query_type,
                "entity": entity_name or "Nasional",
                "entity_level": entity_level,
                "uncertainty": _resolve_uncertainty_label(query_type, uncertainty),
                "report_year": actual_year,
                "sql": sql,
                "params": [str(p) for p in params],
                "results": results,
                "summary": summary,
            },
            default=str,
        )
    except Exception:
        logger.exception("Error executing simple_data_query")
        return json.dumps({"error": "Failed to execute query"})
    finally:
        if conn:
            conn.close()
