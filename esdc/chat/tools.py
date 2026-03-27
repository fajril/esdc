import asyncio
import re
import sqlite3
from pathlib import Path
from typing import Annotated

from langchain.tools import tool

from esdc.chat.schema_loader import SchemaLoader

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
