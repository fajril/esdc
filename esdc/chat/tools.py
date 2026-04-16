# Standard library
import asyncio
import hashlib
import logging
import re
from typing import Annotated, Any

# Third-party
import diskcache
import duckdb
from langchain.tools import tool

logger = logging.getLogger(__name__)

# Maximum rows to return to prevent context window overflow
MAX_QUERY_ROWS = 50

_sql_cache: diskcache.Cache | None = None

_FTS_TABLES: dict[str, str] = {
    "project_resources": "fts_main_project_resources",
    "project_timeseries": "fts_main_project_timeseries",
}

_VIEW_TO_BASE: dict[str, tuple[str, str]] = {
    "field_resources": ("project_resources", "field_id"),
    "wa_resources": ("project_resources", "wk_id"),
    "field_timeseries": ("project_timeseries", "field_id"),
    "wa_timeseries": ("project_timeseries", "wk_id"),
}

_FTS_COLUMNS: set[str] = {
    "project_name",
    "field_name",
    "wk_name",
    "province",
    "basin128",
    "operator_name",
    "project_remarks",
    "vol_remarks",
}

_ILIKE_PATTERN = re.compile(
    r"(\w+)\s+ILIKE\s+'%([^']+)%'",
    re.IGNORECASE,
)


def _rewrite_with_fts(query: str) -> str:
    """Rewrite ILIKE '%keyword%' patterns to use FTS match_bm25() where possible.

    Strategy:
    - For base tables (project_resources, project_timeseries):
      Add `fts_main_{table}.match_bm25(uuid, 'keyword') IS NOT NULL` condition
    - For views (field_resources, wa_resources, etc.):
      Add subquery filter against the base table's FTS index
    - Always keep the original ILIKE as a secondary filter
    """
    ilike_matches = _ILIKE_PATTERN.findall(query)
    if not ilike_matches:
        return query

    fts_eligible = [
        (col, keyword) for col, keyword in ilike_matches if col in _FTS_COLUMNS
    ]
    if not fts_eligible:
        return query

    from_table_match = re.search(r"\bFROM\s+(\w+)", query, re.IGNORECASE)
    if not from_table_match:
        return query

    table_name = from_table_match.group(1).lower()

    if table_name in _FTS_TABLES:
        fts_name = _FTS_TABLES[table_name]
        keywords = " ".join(kw for _, kw in fts_eligible)
        escape_kw = keywords.replace("'", "''")
        fts_condition = f"{fts_name}.match_bm25(uuid, '{escape_kw}') IS NOT NULL"
        where_match = re.search(r"\bWHERE\b", query, re.IGNORECASE)
        if where_match:
            insert_pos = where_match.end()
            query = query[:insert_pos] + f" {fts_condition} AND" + query[insert_pos:]
        else:
            query += f" WHERE {fts_condition}"
        logger.debug(
            "[FTS] rewrite_base | table=%s keywords='%s'",
            table_name,
            keywords,
        )

    elif table_name in _VIEW_TO_BASE:
        base_table, join_col = _VIEW_TO_BASE[table_name]
        fts_name = _FTS_TABLES[base_table]
        keywords = " ".join(kw for _, kw in fts_eligible)
        escape_kw = keywords.replace("'", "''")
        fts_subquery = (
            f"SELECT {join_col} FROM {base_table} "
            f"WHERE {fts_name}.match_bm25(uuid, '{escape_kw}') IS NOT NULL"
        )
        fts_condition = f"{join_col} IN ({fts_subquery})"
        where_match = re.search(r"\bWHERE\b", query, re.IGNORECASE)
        if where_match:
            insert_pos = where_match.end()
            query = query[:insert_pos] + f" {fts_condition} AND" + query[insert_pos:]
        else:
            query += f" WHERE {fts_condition}"
        logger.debug(
            "[FTS] rewrite_view | view=%s base=%s join=%s keywords='%s'",
            table_name,
            base_table,
            join_col,
            keywords,
        )

    return query


def _get_cache() -> diskcache.Cache:
    """Get or create the SQL results cache.

    The cache uses permanent storage (no TTL) because:
    - ESDC data only changes when 'esdc reload' is run
    - Cache is automatically invalidated via invalidate_sql_cache() during reload
    - Between reloads, data is static so cache can persist indefinitely

    Returns:
        diskcache.Cache instance for sql_results directory
    """
    global _sql_cache
    if _sql_cache is None:
        from esdc.configs import Config

        cache_dir = Config.get_cache_dir() / "sql_results"
        cache_dir.mkdir(parents=True, exist_ok=True)
        _sql_cache = diskcache.Cache(str(cache_dir))
    return _sql_cache


def _get_cache_key(sql: str) -> str:
    return hashlib.sha256(sql.encode()).hexdigest()


def _validate_table_name(name: str | None) -> str | None:
    """Validate table name to prevent SQL injection.

    Only allows alphanumeric characters, underscores, and hyphens.
    """
    if not name:
        return None
    if re.match(r"^[a-zA-Z0-9_-]+$", name):
        return name
    return None


def get_db_connection(db_path: str | None = None) -> duckdb.DuckDBPyConnection:
    """Get a database connection.

    Caller is responsible for closing the connection.
    For context manager usage, wrap with 'with get_db_connection() as conn:'
    """
    from pathlib import Path

    if not db_path:
        from esdc.configs import Config

        db_path = str(Config.get_chat_db_path())

    path_obj = Path(db_path)
    if not path_obj.exists():
        raise FileNotFoundError(
            f"Database file not found: {db_path}\n"
            f"Please run 'esdc fetch' to download the database, "
            f"or check your configuration in ~/.esdc/config.yaml"
        )

    if path_obj.exists():
        with open(path_obj, "rb") as f:
            if f.read(6) == b"SQLite":
                raise RuntimeError(
                    f"Database at {db_path} is in SQLite format. "
                    f"Run 'esdc fetch --save' to rebuild in DuckDB format."
                )

    conn = duckdb.connect(str(db_path), read_only=True)
    return conn


@tool("SQL Executor")
async def execute_sql(
    query: Annotated[
        str, "A valid SQL SELECT query to execute against the ESDC database."
    ],
    db_path: Annotated[str | None, "Optional path to the database file."] = None,
) -> str:
    """Execute a SQL query against the ESDC database and return results as a formatted table.

    Use this tool when the user wants to query data from the database.
    Only SELECT queries are allowed for safety.

    This is an async tool that runs the query in a thread pool to avoid blocking
    the event loop, keeping the UI responsive during database operations.
    """
    # Run the synchronous database query in a thread pool
    return await asyncio.get_event_loop().run_in_executor(
        None, _execute_sql_sync, query, db_path
    )


def _execute_sql_sync(query: str, db_path: str | None = None) -> str:
    """Synchronous SQL execution (runs in thread pool to avoid blocking)."""
    query = query.strip()

    if not query.lower().startswith("select"):
        return "Error: Only SELECT queries are allowed. Attempted to execute: " + query

    cache = _get_cache()
    cache_key = _get_cache_key(query)
    if cache_key in cache:
        logger.debug("[SQL] cache_hit | query=%s", query[:80])
        return str(cache[cache_key])

    logger.debug("[SQL] cache_miss | query=%s", query[:80])

    original_query = query
    rewritten = _rewrite_with_fts(query)
    if rewritten != query:
        logger.debug("[FTS] query_rewritten | original=%s", query[:80])
        logger.debug("[FTS] query_rewritten | rewritten=%s", rewritten[:80])
        query = rewritten

    conn = None
    try:
        conn = get_db_connection(db_path)
        logger.debug("[SQL] connection_established")

        try:
            result = conn.execute(query)
            logger.debug("[SQL] query_executed")
        except duckdb.Error as e:
            fts_failed = "match_bm25" in query or "fts_main_" in query
            if fts_failed and query != original_query:
                logger.debug(
                    "[FTS] fts_query_failed | falling back to original | error=%s",
                    e,
                )
                query = original_query
                result = conn.execute(query)
                logger.debug("[SQL] query_executed (fallback)")
            else:
                raise

        if result.description:
            columns = [description[0] for description in result.description]
            rows = result.fetchall()
            logger.debug("[SQL] rows_fetched | count=%d", len(rows))

            if not rows:
                return "Query executed successfully. No results returned."

            max_rows = 50
            total_rows = len(rows)
            if len(rows) > max_rows:
                rows = rows[:max_rows]
                truncated = True
            else:
                truncated = False

            row_strings = []
            for row in rows:
                row_strings.append(" | ".join(str(value) for value in row))

            header = " | ".join(columns)
            separator = "-" * len(header)

            formatted_result = f"{header}\n{separator}\n" + "\n".join(row_strings)
            if truncated:
                formatted_result += (
                    f"\n\n... ({total_rows - max_rows} more rows not shown)"
                )
            formatted_result += f"\n\n({total_rows} rows returned)"

            # Store result in cache (permanent storage - cleared only on esdc reload)
            cache.set(cache_key, formatted_result)
            logger.debug("[SQL] result_formatted_and_cached | rows=%d", total_rows)
            return formatted_result
        else:
            return "Query executed successfully. No results to display."

    except FileNotFoundError as e:
        logger.debug("[SQL] file_not_found | error=%s", e)
        return str(e)
    except duckdb.Error as e:
        logger.debug("[SQL] duckdb_error | error=%s", e)
        return f"SQL Error: {str(e)}"
    except Exception as e:
        logger.debug("[SQL] unexpected_error | error=%s", e)
        return f"Error: {str(e)}"
    finally:
        if conn:
            conn.close()


@tool("Schema Inspector")
def get_schema(
    table_name: Annotated[
        str | None,
        "The name of the table to get schema for. If not provided, returns schema for all tables.",
    ] = None,
) -> str:
    """Get the schema (column names and types) for tables in the ESDC database.

    Use this tool to understand the structure of tables.
    """
    safe_table_name = _validate_table_name(table_name)
    if table_name and not safe_table_name:
        return f"Error: Invalid table name '{table_name}'. Only alphanumeric, underscore, and hyphen allowed."

    conn = None
    try:
        conn = get_db_connection()

        if safe_table_name:
            result = conn.execute(f"DESCRIBE {safe_table_name}")
            columns = result.fetchall()

            if not columns:
                return f"Table '{table_name}' not found."

            output = f"Schema for '{table_name}':\n"
            output += "Column | Type | Nullable | Default\n"
            output += "-" * 50 + "\n"

            for col in columns:
                nullable = "Yes" if col[2] == "YES" else "No"
                default_val = col[4] if col[4] is not None else ""
                output += f"{col[0]} | {col[1]} | {nullable} | {default_val}\n"

            return output
        else:
            result = conn.execute(
                "SELECT table_name as name "
                "FROM information_schema.tables "
                "WHERE table_schema = 'main'"
            )
            tables = result.fetchall()

            output = "Available tables:\n"
            for table in tables:
                tbl_name = table[0]
                tbl_result = conn.execute(f"DESCRIBE {tbl_name}")
                tbl_columns = tbl_result.fetchall()
                col_names = ", ".join(col[0] for col in tbl_columns)
                output += f"- {tbl_name}: {col_names}\n"

            return output

    except FileNotFoundError as e:
        return str(e)
    except duckdb.Error as e:
        return f"SQL Error: {str(e)}"
    except Exception as e:
        return f"Error: {str(e)}"
    finally:
        if conn:
            conn.close()


@tool("Table Lister")
def list_tables() -> str:
    """List all available tables and views in the ESDC database.

    Use this tool to see what data is available.
    """
    try:
        conn = get_db_connection()

        result = conn.execute(
            "SELECT table_name as name, table_type as type "
            "FROM information_schema.tables "
            "WHERE table_schema = 'main' "
            "ORDER BY table_type, table_name"
        )
        items = result.fetchall()

        if not items:
            conn.close()
            return "No tables or views found in the database."

        output = "Available tables and views:\n\n"

        tables = [item for item in items if item[1] == "BASE TABLE"]
        views = [item for item in items if item[1] == "VIEW"]

        if tables:
            output += "Tables:\n"
            for table in tables:
                count_result = conn.execute(f"SELECT COUNT(*) FROM {table[0]}")
                count_row = count_result.fetchone()
                count = count_row[0] if count_row else 0
                output += f"  - {table[0]} ({count} rows)\n"

        if views:
            output += "\nViews:\n"
            for view in views:
                output += f"  - {view[0]}\n"

        conn.close()
        return output

    except FileNotFoundError as e:
        return str(e)
    except duckdb.Error as e:
        return f"SQL Error: {str(e)}"
    except Exception as e:
        return f"Error: {str(e)}"


@tool("Model Checker")
def list_available_models(provider_type: str = "ollama") -> str:
    """List available models for a given provider type.

    Args:
        provider_type: The provider type (ollama, openai, openai_compatible)
    """
    try:
        from esdc.providers import get_provider

        provider_class = get_provider(provider_type)
        if not provider_class:
            return f"Unknown provider type: {provider_type}"

        models = provider_class.list_models()

        if not models:
            return f"No models available for {provider_type}. The provider may not be configured."

        result = f"Available models for {provider_type}:\n"
        for model in models:
            result += f"  - {model}\n"

        return result

    except Exception as e:
        return f"Error: {str(e)}"


@tool("Table Selector")
def get_recommended_table(
    entity_type: Annotated[
        str,
        "Type of entity being queried: 'field', 'work_area', 'wa', 'national', 'nkri', or 'project'. "
        "Use 'field' for field-level queries, 'work_area' or 'wa' for work area queries, "
        "'national' or 'nkri' for national-level queries, 'project' for project-specific queries.",
    ],
    needs_project_detail: Annotated[
        bool,
        "Set to True if you need project-specific columns like project_name, project_remarks, "
        "or project-level breakdowns. When False (default), uses pre-aggregated views for better performance.",
    ] = False,
) -> str:
    """Get the recommended database table or view for a query.

    This tool helps optimize query performance by selecting the right aggregation level.
    Pre-aggregated views (field_resources, wa_resources, nkri_resources) are much faster
    than querying project_resources when you don't need project-level details.

    WHEN TO USE:
    - Call this BEFORE writing SQL queries to select the optimal table
    - Use for field-level, work area-level, or national-level aggregate queries

    Returns:
    JSON string with:
    - table: Recommended table name
    - explanation: Why this table is recommended
    - hierarchy: The aggregation level of this table

    Examples:
    - Field totals: entity_type='field' → {'table': 'field_resources', ...}
    - Work area summary: entity_type='work_area' → {'table': 'wa_resources', ...}
    - National statistics: entity_type='national' → {'table': 'nkri_resources', ...}
    - Project breakdown: entity_type='field', needs_project_detail=True → {'table': 'project_resources', ...}
    """
    import json

    from esdc.chat.domain_knowledge import get_table_for_query

    try:
        table = get_table_for_query(
            entity_type=entity_type, require_detail=needs_project_detail
        )

        hierarchy = {
            "project_resources": "project-level (most detailed)",
            "field_resources": "field-level (pre-aggregated)",
            "wa_resources": "work area-level (pre-aggregated)",
            "nkri_resources": "national-level (pre-aggregated)",
        }

        entity_descriptions = {
            "field": "field-level data",
            "work_area": "work area-level data",
            "wa": "work area-level data",
            "national": "national-level data",
            "nkri": "national-level data",
            "project": "project-specific data",
        }

        entity_key = entity_type.lower().replace(" ", "_")
        if entity_key in ["wilayah_kerja", "work_area", "wa"]:
            entity_desc = "work area-level data"
        elif entity_key in ["lapangan", "field"]:
            entity_desc = "field-level data"
        elif entity_key in ["nasional", "national", "nkri"]:
            entity_desc = "national-level data"
        else:
            entity_desc = entity_descriptions.get(entity_key, "aggregated data")

        explanation = f"Recommended for {entity_desc}"
        if needs_project_detail:
            explanation += " with project-level breakdown"

        return json.dumps(
            {
                "table": table,
                "explanation": explanation,
                "hierarchy": hierarchy.get(table, "aggregation"),
            }
        )

    except Exception as e:
        return json.dumps(
            {
                "table": "project_resources",
                "explanation": f"Defaulting to project_resources due to error: {str(e)}",
                "hierarchy": "project-level (most detailed)",
            }
        )


@tool("Uncertainty Resolver")
def resolve_uncertainty_level(
    level: Annotated[
        str,
        "Uncertainty level from user query. Examples: '1P', '2P', '3P', 'proven', 'probable', 'possible', "
        "'1C', '2C', '3C', '1R', '2R', '3R', '1U', '2U', '3U', 'terbukti' (proven), 'mungkin' (probable), "
        "'harapan' (possible). Case-insensitive.",
    ],
    volume_type: Annotated[
        str,
        "Type of volume being queried: 'reserves' or 'cadangan' for reserves, "
        "'resources' or 'sumber_daya' for resources, 'grr' for GRR, "
        "'contingent' for Contingent Resources, 'prospective' for Prospective Resources.",
    ] = "reserves",
) -> str:
    """Resolve uncertainty level to database filter values and SQL conditions.

    CRITICAL for 'probable' and 'possible' which are CALCULATED values (not in database):
    - 'probable' = 2P - 1P (Middle - Low) - REQUIRES CASE statements
    - 'possible' = 3P - 2P (High - Middle) - REQUIRES CASE statements

    These calculated values ONLY apply to RESERVES (not resources!).

    WHEN TO USE:
    - Call this when user mentions uncertainty levels (1P/2P/3P, proven/probable/possible)
    - Use the returned SQL fragment in WHERE clauses or CASE statements
    - Check 'warnings' field for validation errors

    Returns:
    JSON string with:
    - db_value: Filter value for uncert_level column (or None for calculated)
    - type: 'direct' or 'calculated'
    - calculation: Formula for calculated values (e.g., '2P - 1P')
    - sql_template: SQL fragment for calculated values
    - warnings: List of validation warnings
    - filter_column: Column to filter (always 'uncert_level')

    Examples:
    - resolve_uncertainty_level('2P', 'reserves') → direct value '2. Middle Value'
    - resolve_uncertainty_level('probable', 'reserves') → calculated, returns CASE template
    - resolve_uncertainty_level('probable', 'resources') → ERROR, reserves only
    """
    import json

    from esdc.chat.domain_knowledge import get_uncertainty_filter, get_uncertainty_spec

    try:
        spec = get_uncertainty_spec(level, volume_type=volume_type)

        if spec is None:
            valid_levels = "1P, 2P, 3P, proven, probable, possible, 1C, 2C, 3C, 1R, 2R, 3R, 1U, 2U, 3U"
            return json.dumps(
                {
                    "error": f"Unknown uncertainty level: '{level}'",
                    "valid_levels": valid_levels,
                    "suggestion": f"Try one of: {valid_levels}",
                }
            )

        result = {
            "level": level.lower(),
            "volume_type": volume_type,
            "type": spec.type,
            "db_value": spec.db_value,
            "calculation": spec.calculation,
            "is_cumulative": spec.is_cumulative,
            "reserves_only": spec.reserves_only,
            "description": spec.description,
            "warnings": [],
        }

        if spec.type == "calculated" and spec.sql_template:
            result["sql_template"] = spec.sql_template
            result["usage"] = (
                "Use this SQL template in your SELECT clause. "
                "Replace {column} with your column name (e.g., res_oc, res_an)."
            )

        filter_value = get_uncertainty_filter(level)
        result["filter_column"] = "uncert_level"
        result["filter_value"] = filter_value

        if spec.reserves_only and volume_type.lower() not in [
            "reserves",
            "cadangan",
            "reserve",
        ]:
            result["warnings"].append(
                f"'{level}' only applies to reserves. For {volume_type}, use 1C/2C/3C (contingent) or 1U/2U/3U (prospective)."
            )

        return json.dumps(result, indent=2)

    except ValueError as e:
        return json.dumps({"error": str(e), "level": level, "volume_type": volume_type})
    except Exception as e:
        return json.dumps(
            {
                "error": f"Unexpected error: {str(e)}",
                "level": level,
                "volume_type": volume_type,
            }
        )


@tool("Timeseries Column Guide")
def get_timeseries_columns(
    data_type: Annotated[
        str,
        "Type of timeseries data needed: 'forecast' (future production volumes), "
        "'historical' (cumulative production), or 'rate' (production rates per year). "
        "Default is 'forecast'.",
    ] = "forecast",
    forecast_type: Annotated[
        str,
        "Type of forecast when data_type='forecast': 'tpf' (Total Potential Forecast), "
        "'slf' (Sales Forecast), 'spf' (Sales Potential Forecast), 'crf' (Contingent Resources Forecast), "
        "'prf' (Prospective Resources Forecast), 'ciof' (Consumed in Operation Forecast), "
        "or 'lossf' (Loss Production Forecast). Default is 'tpf'.",
    ] = "tpf",
    substance: Annotated[
        str,
        "Substance suffix: 'oil' (oil only), 'con' (condensate only), 'ga' (associated gas), "
        "'gn' (non-associated gas), 'oc' (oil + condensate combined), or 'an' (total gas). "
        "Default is 'oc'.",
    ] = "oc",
) -> str:
    """Get the correct column names for timeseries queries.

    CRITICAL: This tool prevents common errors where the model confuses rate_* columns
    (historical production RATES) with tpf_* columns (forecast VOLUMES).

    WHEN TO USE:
    - ALWAYS call this BEFORE writing SQL for timeseries/forecast queries
    - When user asks about "forecast", "perkiraan", "proyeksi", "peak production"
    - When querying project_timeseries, field_timeseries, wa_timeseries, or nkri_timeseries

    COLUMN CATEGORIES:
    1. Forecast VOLUMES (USE FOR FORECASTS): tpf_*, slf_*, spf_*, crf_*, prf_*
       - Units: MSTB (oil), BSCF (gas) - these are VOLUMES, not rates
       - Example: tpf_oc = forecast oil+condensate volume in MSTB

    2. Historical CUMULATIVE: cprd_grs_*, cprd_sls_*
       - Units: MSTB (oil), BSCF (gas) - cumulative production volumes
       - Example: cprd_grs_oc = cumulative gross oil+condensate in MSTB

    3. Production RATES: rate_*
       - Units: MSTB/Y (oil), BSCF/Y (gas) - RATES per year, NOT volumes
       - Example: rate_oc = production rate in MSTB per year
       - NEVER use for forecast queries!

    UNIT DIFFERENCE:
    - tpf_oc = 1000 MSTB means 1 million barrels total volume
    - rate_oc = 1000 MSTB/Y means 1000 barrels per year production rate
    These are completely different measurements!

    Returns:
    JSON string with:
    - column: The column name to use (e.g., "tpf_oc")
    - description: Human-readable description
    - unit: Unit abbreviation (MSTB, BSCF, MSTB/Y, BSCF/Y)
    - unit_description: Detailed unit explanation
    - category: Column category (forecast, historical, rate)
    - tables: Applicable tables
    - warning: Important warning about column usage
    - incorrect_alternatives: Columns NOT to use (commonly confused)
    - examples: Example SQL queries

    Examples:
    - get_timeseries_columns("forecast", "tpf", "oc") → tpf_oc for forecast volumes
    - get_timeseries_columns("forecast", "slf", "an") → slf_an for sales forecast gas
    - get_timeseries_columns("historical", substance="oc") → cprd_grs_oc for cumulative
    - get_timeseries_columns("rate", substance="oc") → rate_oc for production rate

    IMPORTANT: For forecast queries, the model often incorrectly selects rate_* columns.
    ALWAYS use this tool to validate your column selection before writing SQL.
    """
    import json

    from esdc.chat.domain_knowledge import (
        get_timeseries_columns as _get_timeseries_columns,
    )

    try:
        result = _get_timeseries_columns(
            data_type=data_type,
            forecast_type=forecast_type,
            substance=substance,
        )

        return json.dumps(result, indent=2)

    except Exception as e:
        return json.dumps(
            {
                "error": f"Error getting timeseries columns: {str(e)}",
                "data_type": data_type,
                "forecast_type": forecast_type,
                "substance": substance,
            }
        )


@tool("Resources Column Guide")
def get_resources_columns(
    volume_type: Annotated[
        str,
        "Type of volume: 'reserves' (commercial reserves only), 'resources' (GRR/Contingent/Prospective), "
        "or 'risked' (prospective resources with geological chance factor applied). Default is 'reserves'.",
    ] = "reserves",
    substance: Annotated[
        str,
        "Substance suffix: 'oil' (oil only), 'con' (condensate only), 'ga' (associated gas), "
        "'gn' (non-associated gas), 'oc' (oil + condensate combined), or 'an' (total gas). "
        "Default is 'oc'.",
    ] = "oc",
) -> str:
    """Get the correct column names for static resource queries.

    CRITICAL: This tool prevents confusion between res_* (reserves) and rec_* (resources) columns.
    The model often confuses these two similar prefixes.

    WHEN TO USE:
    - ALWAYS call this BEFORE writing SQL for resource/reserves queries
    - When user asks about "cadangan" (reserves), "sumber daya" (resources), or "GRR"
    - When querying project_resources, field_resources, wa_resources, or nkri_resources

    COLUMN CATEGORIES:
    1. Reserves (res_*): Commercial reserves only - use for "cadangan" queries
       - Columns: res_oil, res_con, res_ga, res_gn, res_oc, res_an
       - Only projects with project_class = '1. Reserves & GRR'

    2. Resources (rec_*): All recoverable resources - use for "sumber daya" queries
       - Columns: rec_oil, rec_con, rec_ga, rec_gn, rec_oc, rec_an, rec_mboe
       - Includes Reserves + GRR + Contingent + Prospective

    3. Risked Resources (rec_*_risked): Prospective resources with GCF applied
       - Columns: rec_oil_risked, rec_con_risked, etc.
       - Only applies to prospective resources

    PREFIX CONFUSION:
    - res_* = Reserves (commercial only, "cadangan")
    - rec_* = Resources (all recoverable, "sumber daya")
    - These are completely different! res_oc ≠ rec_oc

    Returns:
    JSON string with:
    - column: The column name to use (e.g., "res_oc" or "rec_oc")
    - description: Human-readable description
    - unit: Unit (MSTB or BSCF)
    - category: Column category (reserves, resources, resources_risked)
    - tables: Applicable tables
    - warning: Important warning about res/rec confusion
    - incorrect_alternatives: Columns NOT to use
    - examples: Example SQL queries

    Examples:
    - get_resources_columns("reserves", "oc") → res_oc for reserves
    - get_resources_columns("resources", "an") → rec_an for resources
    - get_resources_columns("risked", "oil") → rec_oil_risked for risked prospective

    IMPORTANT: Always call this tool to validate column selection.
    The difference between res_* and rec_* is critical - they are NOT interchangeable.
    """
    import json

    from esdc.chat.domain_knowledge import (
        get_resources_columns as _get_resources_columns,
    )

    try:
        result = _get_resources_columns(
            volume_type=volume_type,
            substance=substance,
        )

        return json.dumps(result, indent=2)

    except Exception as e:
        return json.dumps(
            {
                "error": f"Error getting resources columns: {str(e)}",
                "volume_type": volume_type,
                "substance": substance,
            }
        )


@tool("Problem Cluster Search")
def search_problem_cluster(
    query: Annotated[
        str,
        "Search term for problem cluster. Can be partial name (e.g., 'subsurface', 'uneconomic'), "
        "cluster code (e.g., '1.1.1', '2.2'), or keyword from the problem description.",
    ],
) -> str:
    """Search for problem cluster definitions when user asks about project issues or specific cluster terms.

    CRITICAL: Use this tool when user asks about:
    - Problem cluster definitions (e.g., "apa arti subsurface uncertainty?")
    - What specific problem terms mean (e.g., "what is uneconomic?")
    - Questions about project problems or obstacles
    - Any cluster code references (e.g., "1.1.1", "2.2", "3.1.2")

    This tool searches the official problem cluster taxonomy with 20 categories
    covering Technical, Economics, Legal, and Social/Environment issues.

    Returns:
    JSON string with:
    - clusters: List of matching problem clusters (max 3)
    - explanation: Full formatted explanation of the top result
    - code: Problem cluster code (e.g., "1.1.1")
    - name: Problem cluster name
    - category: Hierarchical category (e.g., "Technical > Subsurface")
    - definition: Full Indonesian definition
    - examples: List of example scenarios

    Examples:
    - search_problem_cluster("subsurface") → Subsurface Uncertainty (1.1.1)
    - search_problem_cluster("uneconomic") → Uneconomic (2.2)
    - search_problem_cluster("1.1.1") → Exact code match for Subsurface Uncertainty
    - search_problem_cluster("AMDAL") → AMDAL (3.1.2)
    """
    import json

    from esdc.chat.domain_knowledge import (
        get_cluster_explanation,
        search_problem_clusters,
    )

    try:
        results = search_problem_clusters(query, limit=3)

        if not results:
            return json.dumps(
                {
                    "error": f"No problem cluster found matching '{query}'",
                    "suggestion": "Try searching for keywords like: subsurface, data, uneconomic, AMDAL, permit, etc.",
                    "available_categories": [
                        "Technical > Subsurface (1.1.x)",
                        "Technical > Non Subsurface (1.2.x)",
                        "Economics (2.x)",
                        "Legal > Law and Regulations (3.1.x)",
                        "Legal > T&C Contracts (3.2.x)",
                        "Social and Environment (4.x)",
                    ],
                }
            )

        # Get detailed explanation for top result
        top_result = results[0]
        explanation = get_cluster_explanation(top_result["code"])

        return json.dumps(
            {
                "clusters": [
                    {
                        "code": r["code"],
                        "name": r["name"],
                        "category": r["category"],
                        "match_score": r.get("match_score", 0),
                    }
                    for r in results
                ],
                "top_result": {
                    "code": top_result["code"],
                    "name": top_result["name"],
                    "category": top_result["category"],
                },
                "explanation": explanation,
            },
            indent=2,
            ensure_ascii=False,  # Preserve Indonesian characters
        )

    except Exception as e:
        return json.dumps(
            {
                "error": f"Error searching problem clusters: {str(e)}",
                "query": query,
            }
        )


@tool("Knowledge Traversal")
def knowledge_traversal(
    query: Annotated[
        str,
        "Natural language query to resolve entities and match patterns "
        "against the knowledge graph. Examples: 'cadangan Duri 2024', "
        "'profil produksi Abadi', 'top 5 lapangan di WK Rokan'.",
    ],
    return_multiple: Annotated[
        bool,
        "If True, return all matching entities instead of single best match. "
        "Use when user asks for multiple matches "
        "(e.g., 'lapangan yang ada kata duri apa saja').",
    ] = False,
) -> str:
    """Resolve entities and match query patterns from the ESDC knowledge graph.

    This tool traverses the knowledge graph to identify entities
    (fields, working areas, operators, years, uncertainty levels) and match
    query patterns from natural language. It returns structured context that
    enables single-shot SQL generation, reducing multi-round tool calling
    to 1-2 calls.

    WHEN TO USE:
    - Call this BEFORE writing SQL queries to resolve entity names
    - When user mentions specific field names, working areas, operators, or years
    - When user asks about reserves (cadangan), production (produksi), etc.
    - When you need to determine the correct table/view and WHERE conditions

    FALLBACK: If this tool returns status='failed' or status='ambiguous',
    fall back to multi-round tool calling (get_schema,
    get_recommended_table, resolve_uncertainty_level, etc.)

    Returns:
    JSON string with:
    - status: "success", "ambiguous", or "failed"
    - entities: List of resolved entities with type, id, name, confidence
    - pattern: Best matching query pattern from graph schema
    - suggested_table: Recommended table/view for the query
    - where_conditions: Suggested WHERE clauses
    - required_columns: Columns likely needed
    - confidence: Overall confidence score (0.0-1.0)

    Examples:
    - knowledge_traversal("cadangan Duri 2024")
      → Entity: Field=Duri, Year=2024, Pattern: cadangan, Table: field_resources
    - knowledge_traversal("profil produksi Abadi")
      → Entity: Field=Abadi, Pattern: profil_produksi, Table: field_timeseries
    - knowledge_traversal("isu water cut di lapangan Duri")
      → Entity: Field=Duri, Pattern: issues_remarks, Table: field_resources
    """
    import json

    from esdc.knowledge_graph.resolver import KnowledgeTraversalResolver

    try:
        conn = get_db_connection()
        try:
            resolver = KnowledgeTraversalResolver(db=conn)
            result = resolver.resolve(query=query, return_multiple=return_multiple)
            result["query"] = query

            if result.get("pattern") and result["pattern"].get("cypher_template"):
                result["cypher_available"] = True
            else:
                result["cypher_available"] = False

            return json.dumps(result, indent=2, ensure_ascii=False)
        finally:
            conn.close()

    except FileNotFoundError as e:
        return json.dumps(
            {
                "status": "failed",
                "fallback": "multi_round",
                "message": str(e),
                "query": query,
            }
        )
    except Exception as e:
        logger.error("[KG] traversal_error | query=%s error=%s", query, e)
        return json.dumps(
            {
                "status": "failed",
                "fallback": "multi_round",
                "message": f"Knowledge traversal error: {str(e)}",
                "query": query,
            }
        )


@tool("Cypher Executor")
async def execute_cypher(
    query: Annotated[
        str,
        "A valid Cypher query to execute against the ESDC knowledge graph. "
        "Use this for graph traversal queries like finding nearby fields, "
        "tracing relationships, or multi-hop entity resolution.",
    ],
) -> str:
    """Execute a Cypher query against the ESDC knowledge graph.

    Use this tool when knowledge_traversal indicates cypher_available=True
    or when you need graph traversal (spatial proximity, relationships).

    Supports parameterized queries using $param_name syntax.

    Returns:
    JSON string with:
    - status: "success" or "error"
    - results: List of result rows as dictionaries
    - row_count: Number of rows returned

    Examples:
    - execute_cypher("MATCH (f:Field {field_name: 'Duri'}) RETURN f.field_name, f.field_lat")
    - execute_cypher("MATCH (f1:Field)-[:LOCATED_NEAR]->(f2:Field) WHERE f1.field_name = 'Duri' AND f2.distance_km < 20 RETURN f2.field_name, f2.distance_km")
    """
    import json

    try:
        return await asyncio.get_event_loop().run_in_executor(
            None, _execute_cypher_sync, query
        )
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})


def _execute_cypher_sync(query: str) -> str:
    """Synchronous Cypher execution."""
    import json

    from esdc.knowledge_graph.ladybug_manager import LadybugDBManager

    manager = LadybugDBManager()
    if not manager.initialize():
        return json.dumps(
            {
                "status": "error",
                "message": "Knowledge graph not available. Run 'esdc load --kg' first.",
            }
        )

    try:
        results = manager.execute_cypher(query)
        return json.dumps(
            {
                "status": "success",
                "results": results,
                "row_count": len(results),
            },
            indent=2,
            ensure_ascii=False,
        )
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})
    finally:
        manager.close()


@tool("Spatial Resolver")
def resolve_spatial(
    query_type: Annotated[
        str,
        "Type of spatial query: 'proximity' (fields near a field), "
        "'working_area' (fields in a working area), 'distance' (between two fields), "
        "'coordinates' (get field coordinates), 'nearest_from_coords' (find nearest from lat/long), "
        "'field_clusters' (cluster fields by proximity), 'adjacent_wk' (find adjacent working areas), "
        "or 'average_distance' (average distance between multiple fields).",
    ],
    target: Annotated[
        str | dict,
        "For proximity: field name. For working_area: working area name. "
        "For distance: comma-separated 'field1, field2'. For coordinates: field name. "
        "For nearest_from_coords: dict with 'lat', 'long', 'entity_type' ('field' or 'working_area'). "
        "For field_clusters: dict with 'max_distance_km', 'min_cluster_size'. "
        "For adjacent_wk: dict with 'wk_name', 'max_distance_km'. "
        "For average_distance: dict with 'field_names' as list.",
    ],
    radius_km: Annotated[
        float,
        "For proximity queries: search radius in kilometers (default: 20).",
    ] = 20.0,
    limit: Annotated[
        int,
        "Maximum number of results to return (default: 10).",
    ] = 10,
    wk_name: Annotated[
        str | None,
        "Optional working area name to scope results. "
        "When provided, field lookups filter to the specified working area. "
        "Use when the query mentions a working area context like "
        "'lapangan X di WK Y' or 'field X in working area Y'.",
    ] = None,
) -> str:
    """Execute spatial queries using DuckDB's native spatial capabilities.

    Use this tool for:
    - Finding fields within a radius of another field
    - Getting fields in a working area
    - Calculating distance between fields
    - Getting field coordinates
    - Finding nearest entities from arbitrary coordinates
    - Clustering fields by proximity
    - Finding adjacent working areas
    - Calculating average distance between multiple fields

    Returns:
    JSON string with query results.

    Examples:
    - resolve_spatial("proximity", "Duri", 20) -> Fields within 20km of Duri
    - resolve_spatial("working_area", "Rokan") -> All fields in Rokan working area
    - resolve_spatial("distance", "Duri, Bekapai") -> Distance between Duri and Bekapai
    - resolve_spatial("coordinates", "Duri") -> Lat/long of Duri field
    - resolve_spatial("nearest_from_coords", '{"lat": 1.5, "long": 101.3, "entity_type": "field", "radius_km": 20}')
    - resolve_spatial("field_clusters", '{"max_distance_km": 20, "min_cluster_size": 2}')
    - resolve_spatial("adjacent_wk", '{"wk_name": "Rokan", "max_distance_km": 20}')
    - resolve_spatial("average_distance", '{"field_names": ["Duri", "Rokan", "Belanak"]}')
    """
    import json

    from esdc.knowledge_graph.spatial_resolver import SpatialResolver

    logger.debug(
        "[SPATIAL_START] query_type=%s | target=%s | radius_km=%s | wk_name=%s",
        query_type,
        target,
        radius_km,
        wk_name,
    )

    resolver = SpatialResolver()

    try:
        if query_type == "proximity":
            result = resolver.find_fields_near_field(
                field_name=target,
                radius_km=radius_km,
                limit=limit,
                wk_name=wk_name,
            )
        elif query_type == "working_area":
            result = resolver.find_fields_in_working_area(wk_name=target, limit=limit)
        elif query_type == "distance":
            parts = [p.strip() for p in target.split(",")]
            if len(parts) != 2:
                logger.debug("[SPATIAL_ERR] error=invalid_format")
                return json.dumps(
                    {
                        "status": "error",
                        "message": "Distance query requires 'field1, field2' format",
                    }
                )
            result = resolver.calculate_distance(
                from_field=parts[0],
                to_field=parts[1],
                wk_name=wk_name,
            )
        elif query_type == "coordinates":
            result = resolver.get_field_coordinates(
                field_name=target,
                wk_name=wk_name,
            )
        elif query_type == "nearest_from_coords":
            try:
                params = target if isinstance(target, dict) else json.loads(target)
                result = resolver.find_nearest_from_coordinates(
                    lat=float(params.get("lat", 0.0)),
                    long=float(params.get("long", 0.0)),
                    entity_type=params.get("entity_type", "field"),
                    radius_km=float(params.get("radius_km", radius_km)),
                    limit=int(params.get("limit", limit)),
                )
            except (json.JSONDecodeError, ValueError, TypeError) as e:
                logger.debug("[SPATIAL_ERR] error=invalid_params")
                return json.dumps(
                    {
                        "status": "error",
                        "message": f"Invalid format for nearest_from_coords: {e}. Expected dict with lat, long, entity_type",
                    }
                )
        elif query_type == "field_clusters":
            try:
                params = target if isinstance(target, dict) else json.loads(target)
                result = resolver.find_field_clusters(
                    max_distance_km=float(params.get("max_distance_km", radius_km)),
                    min_cluster_size=int(params.get("min_cluster_size", 2)),
                )
            except (json.JSONDecodeError, ValueError, TypeError) as e:
                logger.debug("[SPATIAL_ERR] error=invalid_params")
                return json.dumps(
                    {
                        "status": "error",
                        "message": f"Invalid format for field_clusters: {e}. Expected dict with max_distance_km, min_cluster_size",
                    }
                )
        elif query_type == "adjacent_wk":
            try:
                params = target if isinstance(target, dict) else json.loads(target)
                result = resolver.find_adjacent_working_areas(
                    wk_name=params.get("wk_name", ""),
                    max_distance_km=float(params.get("max_distance_km", radius_km)),
                    limit=int(params.get("limit", limit)),
                )
            except (json.JSONDecodeError, ValueError, TypeError) as e:
                logger.debug("[SPATIAL_ERR] error=invalid_params")
                return json.dumps(
                    {
                        "status": "error",
                        "message": f"Invalid format for adjacent_wk: {e}. Expected dict with wk_name, max_distance_km",
                    }
                )
        elif query_type == "average_distance":
            try:
                params = target if isinstance(target, dict) else json.loads(target)
                field_names = params.get("field_names", [])
                if not isinstance(field_names, list) or len(field_names) < 2:
                    logger.debug("[SPATIAL_ERR] error=insufficient_fields")
                    return json.dumps(
                        {
                            "status": "error",
                            "message": "average_distance requires at least 2 field names in 'field_names' list",
                        }
                    )
                result = resolver.calculate_average_distance(field_names=field_names)
            except (json.JSONDecodeError, ValueError, TypeError) as e:
                logger.debug("[SPATIAL_ERR] error=invalid_params")
                return json.dumps(
                    {
                        "status": "error",
                        "message": f"Invalid format for average_distance: {e}. Expected dict with field_names list",
                    }
                )
        else:
            logger.debug("[SPATIAL_ERR] error=unknown_query_type")
            return json.dumps(
                {
                    "status": "error",
                    "message": f"Unknown query_type: {query_type}. "
                    "Use: proximity, working_area, distance, coordinates, nearest_from_coords, field_clusters, adjacent_wk, or average_distance",
                }
            )

        result_count = (
            len(result.get("nearby_fields", [])) if isinstance(result, dict) else 0
        )
        logger.debug(
            "[SPATIAL_OK] query_type=%s | results=%d", query_type, result_count
        )
        return json.dumps(result, indent=2, ensure_ascii=False)

    except Exception as e:
        logger.error(
            "[SPATIAL_ERR] query_failed | type=%s target=%s error=%s",
            query_type,
            target,
            e,
        )
        return json.dumps(
            {
                "status": "error",
                "message": str(e),
                "query_type": query_type,
                "target": target,
            }
        )
    finally:
        resolver.close()


@tool("Semantic Search")
def semantic_search(
    query: Annotated[
        str,
        "Natural language query to search for semantically similar documents.",
    ],
    limit: Annotated[
        int,
        "Maximum number of results (default: 10).",
    ] = 10,
    report_year: Annotated[
        int | None,
        "Filter by report year (e.g., 2024). Optional.",
    ] = None,
    field_name: Annotated[
        str | None,
        "Filter by field name (ILIKE pattern, e.g., '%Duri%'). Optional.",
    ] = None,
    pod_name: Annotated[
        str | None,
        "Filter by POD name (ILIKE pattern, e.g., '%POD%'). Optional.",
    ] = None,
    wk_name: Annotated[
        str | None,
        "Filter by working area name (ILIKE pattern, e.g., '%Rokan%'). Optional.",
    ] = None,
    province: Annotated[
        str | None,
        "Filter by province (ILIKE pattern, e.g., '%Riau%'). Optional.",
    ] = None,
    basin128: Annotated[
        str | None,
        "Filter by basin (ILIKE pattern, e.g., '%Sumatera%'). Optional.",
    ] = None,
    project_class: Annotated[
        str | None,
        "Filter by project class (ILIKE pattern, e.g., '%Contigent%'). Optional.",
    ] = None,
    project_stage: Annotated[
        str | None,
        "Filter by project stage (ILIKE pattern, e.g., '%Exploration%'). Optional.",
    ] = None,
    project_level: Annotated[
        str | None,
        "Filter by project level (ILIKE pattern, e.g., '%E0%'). Optional.",
    ] = None,
    operator_name: Annotated[
        str | None,
        "Filter by operator name (ILIKE pattern, e.g., '%Pertamina%'). Optional.",
    ] = None,
    operator_group: Annotated[
        str | None,
        "Filter by operator group (ILIKE pattern, e.g., '%Pertamina Hulu%'). Optional.",
    ] = None,
    wk_subgroup: Annotated[
        str | None,
        "Filter by working area subgroup (ILIKE pattern, e.g., '%Upstream%'). Optional.",
    ] = None,
    wk_regionisasi_ngi: Annotated[
        str | None,
        "Filter by NGI region (ILIKE pattern, e.g., '%Sumatera%'). Optional.",
    ] = None,
    wk_area_perwakilan_skkmigas: Annotated[
        str | None,
        "Filter by SKK Migas region (ILIKE pattern, e.g., '%Duri%'). Optional.",
    ] = None,
) -> str:
    """Search for documents by semantic similarity to the query.

    Use this tool when:
    - User asks about concepts, meanings, or topics (not exact keywords)
    - User queries like: "proyek dengan masalah X", "lapangan yang sulit"
    - FTS returns no results or insufficient results
    - User wants to filter by year, field, working area, etc.

    Returns:
    JSON string with:
    - status: "success", "no_results", "not_available", "fallback_to_fts", or "error"
    - results: List of similar documents with similarity scores and contextual columns
    - count: Number of results
    - message: Additional information (e.g., fallback explanation)

    Examples:
    - semantic_search("proyek dengan reservoir kompleks") -> Find projects with complex reservoir
    - semantic_search("masalah produksi", 5) -> Top 5 production issues
    - semantic_search("tidak ekonomis", report_year=2024) -> Economic issues in 2024
    - semantic_search("kendala teknis", field_name="%Duri%") -> Technical issues in Duri field
    """
    import json

    from esdc.knowledge_graph.semantic_resolver import SemanticResolver

    resolver = SemanticResolver()

    # Build filters dict from optional parameters
    filters: dict[str, Any] = {}
    if report_year is not None:
        filters["report_year"] = report_year
    if field_name is not None:
        filters["field_name"] = field_name
    if pod_name is not None:
        filters["pod_name"] = pod_name
    if wk_name is not None:
        filters["wk_name"] = wk_name
    if province is not None:
        filters["province"] = province
    if basin128 is not None:
        filters["basin128"] = basin128
    if project_class is not None:
        filters["project_class"] = project_class
    if project_stage is not None:
        filters["project_stage"] = project_stage
    if project_level is not None:
        filters["project_level"] = project_level
    if operator_name is not None:
        filters["operator_name"] = operator_name
    if operator_group is not None:
        filters["operator_group"] = operator_group
    if wk_subgroup is not None:
        filters["wk_subgroup"] = wk_subgroup
    if wk_regionisasi_ngi is not None:
        filters["wk_regionisasi_ngi"] = wk_regionisasi_ngi
    if wk_area_perwakilan_skkmigas is not None:
        filters["wk_area_perwakilan_skkmigas"] = wk_area_perwakilan_skkmigas

    try:
        result = resolver.search_by_text(
            query=query,
            limit=limit,
            filters=filters if filters else None,
        )

        # If embeddings not available, fallback to FTS search
        if result.get("status") == "not_available":
            logger.info("[Semantic] embeddings not available, falling back to FTS")
            # For FTS fallback, we can't apply all filters, but we can try with basic ones
            fallback_result = _search_remarks_via_fts(query, limit, "project_resources")
            return json.dumps(fallback_result, indent=2, ensure_ascii=False)

        return json.dumps(result, indent=2, ensure_ascii=False)

    except Exception as e:
        logger.error("[Semantic] tool failed | query=%s error=%s", query, e)
        return json.dumps(
            {
                "status": "error",
                "message": str(e),
                "query": query,
            }
        )
    finally:
        resolver.close()


def _search_remarks_via_fts(
    query: str,
    limit: int = 10,
    table_name: str | None = None,
) -> dict:
    """Fallback FTS search on project_remarks when embeddings unavailable.

    Searches project_remarks using FTS match_bm25 for keyword-based results.
    """
    import duckdb

    from esdc.configs import Config

    target_table = table_name or "project_resources"
    db_file = Config.get_db_file()

    if not db_file.exists():
        return {
            "status": "error",
            "message": "Database file not found. Run 'esdc fetch --save' first.",
            "results": [],
            "count": 0,
        }

    try:
        # Use FTS to search project_remarks
        escaped_query = query.replace("'", "''")
        sql = f"""
            SELECT
                pr.uuid,
                pr.field_name,
                pr.project_name,
                SUBSTRING(pr.project_remarks, 1, 300) as source_text,
                fts_main_{target_table}.match_bm25(
                    pr.uuid, '{escaped_query}'
                ) as relevance_score
            FROM {target_table} pr
            JOIN fts_main_{target_table} ON fts_main_{target_table}.uuid = pr.uuid
            WHERE fts_main_{target_table}.match_bm25(
                pr.uuid, '{escaped_query}'
            ) IS NOT NULL
            ORDER BY relevance_score DESC
            LIMIT {limit}
        """

        conn = duckdb.connect(str(db_file), read_only=True)
        try:
            df = conn.execute(sql).fetchdf()
        finally:
            conn.close()

        if df is None or df.empty:
            return {
                "status": "no_results",
                "message": "No matching documents found via FTS fallback.",
                "results": [],
                "count": 0,
            }

        # Convert to results format matching semantic search
        results = []
        for _, row in df.iterrows():
            results.append(
                {
                    "uuid": row.get("uuid", ""),
                    "field_name": row.get("field_name", ""),
                    "project_name": row.get("project_name", ""),
                    "source_text": row.get("source_text", "")[:200] + "..."
                    if len(str(row.get("source_text", ""))) > 200
                    else str(row.get("source_text", "")),
                    "similarity": round(float(row.get("relevance_score", 0)), 4),
                }
            )

        return {
            "status": "fallback_to_fts",
            "message": (
                "Semantic search not available (embeddings not generated). "
                "Using keyword-based FTS search as fallback. "
                "Run 'esdc reload --embeddings-only' to enable semantic search."
            ),
            "count": len(results),
            "results": results,
        }

    except Exception as e:
        logger.error("[Semantic] FTS fallback failed | query=%s error=%s", query, e)
        return {
            "status": "not_available",
            "message": (
                "Semantic search not available and FTS fallback failed. "
                f"Error: {e}. "
                "Run 'esdc reload --embeddings-only' to enable semantic search."
            ),
            "results": [],
            "count": 0,
        }
