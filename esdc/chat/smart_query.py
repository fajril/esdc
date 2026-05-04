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
    "cadangan": "Cadangan (Reserves)",
    "potensi": "Potensi (Resources/GRR)",
    "contingent": "Contingent Resources",
    "prospective": "Prospective Resources (Risked)",
    "cumprod": "Produksi Kumulatif",
    "prodrate": "Rate Produksi",
}

UNCERTAINTY_DISPLAY: dict[str, str] = {
    "1P": "1P (Proven/Low)",
    "2P": "2P (Probable/Mid)",
    "3P": "3P (Possible/High)",
    "probable": "Probable (2P-1P)",
    "1C": "1C (Low)",
    "2C": "2C (Mid)",
    "3C": "3C (High)",
}


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
    uncert_display = UNCERTAINTY_DISPLAY.get(uncertainty, uncertainty)

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
    uncertainty: str = "2P",
    report_year: int | list[int] | None = None,
) -> str:
    """Execute a standardized aggregate query for simple factual data.

    Use for questions like:
    - "berapa cadangan WK X?"
    - "berapa potensi yang ada di WK X?"
    - "berapa contingent resources nasional terbaru?"
    - "berapa prospective resources risked lapangan Y?"
    - "berapa produksi kumulatif lapangan Z?"
    - "berapa rate produksi lapangan Z?"

    Parameters
    ----------
    query_type : str
        One of: ``cadangan``, ``potensi``, ``contingent``, ``prospective``,
        ``cumprod``, ``prodrate``.
    entity_level : str
        One of: ``field``, ``work_area``, ``national``.
    entity_name : str | None
        Entity name (e.g. "Rokan", "Duri").  ``None`` for national queries.
    uncertainty : str
        Uncertainty level: ``1P``, ``2P``, ``3P``, ``probable``, ``1C``,
        ``2C``, ``3C``.  Default ``"2P"``.
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
                "uncertainty": UNCERTAINTY_DISPLAY.get(uncertainty, uncertainty),
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
