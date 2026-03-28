import asyncio
import re
import sqlite3
from typing import Annotated

from langchain.tools import tool


# Maximum rows to return to prevent context window overflow
MAX_QUERY_ROWS = 50


def _validate_table_name(name: str | None) -> str | None:
    """Validate table name to prevent SQL injection.

    Only allows alphanumeric characters, underscores, and hyphens.
    """
    if not name:
        return None
    if re.match(r"^[a-zA-Z0-9_-]+$", name):
        return name
    return None


def get_db_connection(db_path: str | None = None) -> sqlite3.Connection:
    """Get a database connection.

    Caller is responsible for closing the connection.
    For context manager usage, wrap with 'with get_db_connection() as conn:'
    """
    if not db_path:
        from esdc.configs import Config

        db_path = str(Config.get_chat_db_path())

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


@tool
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

    conn = None
    try:
        conn = get_db_connection(db_path)
        cursor = conn.cursor()

        cursor.execute(query)

        if cursor.description:
            columns = [description[0] for description in cursor.description]
            rows = cursor.fetchall()

            if not rows:
                return "Query executed successfully. No results returned."

            # Limit results to prevent context window overflow
            MAX_ROWS = 50
            total_rows = len(rows)
            if len(rows) > MAX_ROWS:
                rows = rows[:MAX_ROWS]
                truncated = True
            else:
                truncated = False

            row_strings = []
            for row in rows:
                row_strings.append(" | ".join(str(value) for value in row))

            header = " | ".join(columns)
            separator = "-" * len(header)

            result = f"{header}\n{separator}\n" + "\n".join(row_strings)
            if truncated:
                result += f"\n\n... ({total_rows - MAX_ROWS} more rows not shown)"
            result += f"\n\n({total_rows} rows returned)"

            return result
        else:
            return "Query executed successfully. No results to display."

    except sqlite3.Error as e:
        return f"SQL Error: {str(e)}"
    except Exception as e:
        return f"Error: {str(e)}"
    finally:
        if conn:
            conn.close()


@tool
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
        cursor = conn.cursor()

        if safe_table_name:
            cursor.execute(f"PRAGMA table_info({safe_table_name})")
            columns = cursor.fetchall()

            if not columns:
                return f"Table '{table_name}' not found."

            result = f"Schema for '{table_name}':\n"
            result += "Column | Type | Nullable | Default\n"
            result += "-" * 50 + "\n"

            for col in columns:
                result += (
                    f"{col[1]} | {col[2]} | {'No' if col[3] else 'Yes'} | {col[4]}\n"
                )

            return result
        else:
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
            tables = cursor.fetchall()

            result = "Available tables:\n"
            for table in tables:
                table_name = table[0]
                cursor.execute(f"PRAGMA table_info({table_name})")
                columns = cursor.fetchall()
                col_names = ", ".join(col[1] for col in columns)
                result += f"- {table_name}: {col_names}\n"

            return result

    except Exception as e:
        return f"Error: {str(e)}"
    finally:
        if conn:
            conn.close()


@tool
def list_tables() -> str:
    """List all available tables and views in the ESDC database.

    Use this tool to see what data is available.
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute(
            "SELECT name, type FROM sqlite_master WHERE type IN ('table', 'view') AND name NOT LIKE 'sqlite_%' ORDER BY type, name"
        )
        items = cursor.fetchall()

        if not items:
            conn.close()
            return "No tables or views found in the database."

        result = "Available tables and views:\n\n"

        tables = [item for item in items if item[1] == "table"]
        views = [item for item in items if item[1] == "view"]

        if tables:
            result += "Tables:\n"
            for table in tables:
                cursor.execute(f"SELECT COUNT(*) FROM {table[0]}")
                count = cursor.fetchone()[0]
                result += f"  - {table[0]} ({count} rows)\n"

        if views:
            result += "\nViews:\n"
            for view in views:
                result += f"  - {view[0]}\n"

        conn.close()
        return result

    except Exception as e:
        return f"Error: {str(e)}"


@tool
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


@tool
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

    RETURNS:
    JSON string with:
    - table: Recommended table name
    - explanation: Why this table is recommended
    - hierarchy: The aggregation level of this table

    EXAMPLES:
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


@tool
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

    RETURNS:
    JSON string with:
    - db_value: Filter value for uncert_level column (or None for calculated)
    - type: 'direct' or 'calculated'
    - calculation: Formula for calculated values (e.g., '2P - 1P')
    - sql_template: SQL fragment for calculated values
    - warnings: List of validation warnings
    - filter_column: Column to filter (always 'uncert_level')

    EXAMPLES:
    - resolve_uncertainty_level('2P', 'reserves') → direct value '2. Middle Value'
    - resolve_uncertainty_level('probable', 'reserves') → calculated, returns CASE template
    - resolve_uncertainty_level('probable', 'resources') → ERROR, reserves only
    """
    import json

    from esdc.chat.domain_knowledge import get_uncertainty_spec, get_uncertainty_filter

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
