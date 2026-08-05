# Standard library
import asyncio
import hashlib
import json
import logging
import re
import shutil
import threading
from typing import Annotated, Any

# Third-party
import diskcache
import duckdb
from langchain.tools import tool

# First-party (no circular deps — doc_schema.py only imports yaml/stdlib)
from esdc.chat.domain_knowledge.doc_schema import enum_values, render_tool_context

logger = logging.getLogger("esdc.chat.tools")

# Maximum rows to return to prevent context window overflow
MAX_QUERY_ROWS = 50

_sql_cache: diskcache.Cache | None = None
_tool_cache: diskcache.Cache | None = None

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
        _sql_cache = diskcache.Cache(
            str(cache_dir), size_limit=500_000_000, statistics=True
        )
    return _sql_cache


def _get_cache_key(sql: str) -> str:
    return hashlib.sha256(sql.encode()).hexdigest()


def _get_tool_cache() -> diskcache.Cache:
    """Get or create the tool results cache (for non-SQL tools).

    Same pattern as SQL cache: permanent storage, invalidated on esdc reload.
    """
    global _tool_cache
    if _tool_cache is None:
        from esdc.configs import Config

        cache_dir = Config.get_cache_dir() / "tool_results"
        cache_dir.mkdir(parents=True, exist_ok=True)
        _tool_cache = diskcache.Cache(
            str(cache_dir), size_limit=500_000_000, statistics=True
        )
    return _tool_cache


_corpus_embedder = None


def _get_corpus_embedder():
    """Lazily create and reuse one InternalEmbedder for corpus tools.

    The embedder loads an in-process llama.cpp Qwen3 embedding model; recreating it per
    tool call wasted setup time. The CorpusStore/DuckDB connection is
    deliberately NOT cached (short-lived connections avoid file-lock
    conflicts with the corpus CLI).
    """
    global _corpus_embedder
    if _corpus_embedder is None:
        from esdc.corpus.embedder import InternalEmbedder

        _corpus_embedder = InternalEmbedder()
    return _corpus_embedder


_semantic_resolver_tls = threading.local()


def _get_semantic_resolver():
    """Reuse one SemanticResolver PER THREAD for the semantic_search tool.

    Reusing the resolver instance (not just the embedder) is what lets its
    DB-signature-keyed semantic_meta pin memo actually pay off: a fresh
    SemanticResolver() per call meant the memo never survived past a single
    tool invocation. semantic_search is a sync LangChain tool run on a
    threadpool worker, and the chat server serves requests concurrently, so
    a single module-global resolver would let two threads share one
    DuckDBPyConnection -- a non-thread-safe object -- and race on
    resolver.close() (thread X nulling self._conn while thread Y is
    mid-query). Caching per-thread instead keeps the memo win without any
    cross-thread sharing: each thread gets its own resolver (and its own
    connection), and resolver.close() in the caller's finally block only
    ever affects that thread's own connection.
    """
    resolver = getattr(_semantic_resolver_tls, "resolver", None)
    if resolver is None:
        from esdc.search.semantic_resolver import SemanticResolver

        resolver = SemanticResolver()
        _semantic_resolver_tls.resolver = resolver
    return resolver


def _get_disk_cache_stats(
    cache: diskcache.Cache | None,
    cache_dir_name: str,
    size_limit: int = 500_000_000,
) -> dict[str, Any]:
    """Get statistics from a diskcache.Cache instance.

    Opens a temporary read-only handle (without statistics=True) so that
    esdc status does not interfere with the live cache's hit/miss counters.
    Stats are persisted by diskcache in its internal SQLite database, so
    hits/misses are readable even from a separate process.

    Args:
        cache: The global cache handle, or None if not yet initialized.
        cache_dir_name: Subdirectory name (e.g. "sql_results" or "tool_results").
        size_limit: Maximum cache size in bytes.

    Returns:
        Dict with cache diagnostics.
    """
    from esdc.configs import Config

    cache_dir = Config.get_cache_dir() / cache_dir_name

    if not cache_dir.exists():
        return {
            "directory": str(cache_dir),
            "entries": 0,
            "size_bytes": 0,
            "size_limit": size_limit,
            "hits": 0,
            "misses": 0,
            "hit_rate": None,
        }

    # Open a temporary handle to read stats from disk.
    # Use statistics=False (default) to avoid incrementing counters
    # in this process — we only want to read what the live process wrote.
    try:
        temp_cache = diskcache.Cache(str(cache_dir))
    except (FileNotFoundError, OSError):
        return {
            "directory": str(cache_dir),
            "entries": 0,
            "size_bytes": 0,
            "size_limit": size_limit,
            "hits": 0,
            "misses": 0,
            "hit_rate": None,
        }
    try:
        stats = temp_cache.stats()  # type: ignore[union-attr]
        hits: int = stats[0]  # type: ignore[assignment]
        misses: int = stats[1]  # type: ignore[assignment]
        entries = len(temp_cache)  # type: ignore[arg-type]
        volume = temp_cache.volume()  # type: ignore[union-attr]
        limit = temp_cache.size_limit  # type: ignore[attr-defined]
    except (FileNotFoundError, OSError):
        hits, misses, entries, volume, limit = 0, 0, 0, 0, size_limit
    finally:
        temp_cache.close()
    total = hits + misses
    return {
        "directory": str(cache_dir),
        "entries": entries,
        "size_bytes": volume,
        "size_limit": limit,
        "hits": hits,
        "misses": misses,
        "hit_rate": hits / total if total > 0 else None,
    }


def get_sql_cache_stats() -> dict[str, Any]:
    """Get SQL cache statistics for diagnostics.

    Returns:
        Dict with cache size, entries, hits, misses, and hit rate.
    """
    return _get_disk_cache_stats(_sql_cache, "sql_results")


def get_tool_cache_stats() -> dict[str, Any]:
    """Get tool cache statistics for diagnostics.

    Returns:
        Dict with cache size, entries, hits, misses, and hit rate.
    """
    return _get_disk_cache_stats(_tool_cache, "tool_results")


def _tool_cache_key(tool_name: str, **kwargs: Any) -> str:
    """Generate a deterministic cache key from tool name and arguments."""
    sorted_args = json.dumps(kwargs, sort_keys=True, default=str)
    return f"{tool_name}:{hashlib.sha256(sorted_args.encode()).hexdigest()}"


def invalidate_tool_cache() -> None:
    """Clear the tool results cache directory."""
    global _tool_cache
    if _tool_cache is not None:
        _tool_cache.clear()
        _tool_cache = None
    from esdc.configs import Config
    from esdc.dbmanager import _record_cache_invalidation

    cache_dir = Config.get_cache_dir() / "tool_results"
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
        logger.info("Tool cache invalidated: %s", cache_dir)
    _record_cache_invalidation(cache_dir)


def reset_sql_cache() -> None:
    """Reset the SQL results cache (in-memory + on-disk).

    Called by dbmanager after reload to ensure the stale diskcache.Cache
    object doesn't serve outdated results.
    """
    global _sql_cache
    if _sql_cache is not None:
        _sql_cache.clear()
        _sql_cache = None
    logger.info("SQL cache reset (in-memory handle cleared)")


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

    from esdc.dbmanager import get_duckdb_connection

    conn = get_duckdb_connection(db_path, read_only=True)
    return conn


@tool("SQL Executor")
async def execute_sql(
    query: Annotated[
        str, "A valid SQL SELECT query to execute against the ESDC database."
    ],
    db_path: Annotated[str | None, "Optional path to the database file."] = None,
) -> str:
    """Execute a SQL query against the ESDC database.

    Returns results as a formatted table. Use this tool when the user wants
    to query data from the database.
    Only SELECT queries are allowed for safety.

    For domain context about KSMI levels, entities, or transitions,
    call knowledge_traversal first.

    This is an async tool that runs the query in a thread pool to avoid blocking
    the event loop, keeping the UI responsive during database operations.
    """
    try:
        return await asyncio.wait_for(
            asyncio.get_running_loop().run_in_executor(
                None, _execute_sql_sync, query, db_path
            ),
            timeout=30,
        )
    except asyncio.TimeoutError:
        logger.error("[SQL] Query timed out after 30s: %s", query[:80])
        return "Error: Query timed out after 30 seconds. Try simplifying your query."


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
                has_fts_rewrite = query != original_query and (
                    "match_bm25" in query or "fts_main_" in query
                )
                if has_fts_rewrite:
                    logger.debug(
                        "[FTS] zero_results_with_fts | falling back to original query"
                    )
                    result = conn.execute(original_query)
                    if result.description:
                        rows = result.fetchall()
                        query = original_query
                        logger.debug("[FTS] fallback_rows | count=%d", len(rows))
                    if not rows:
                        return "Query executed successfully. No results returned."
                else:
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
        "The name of the table to get schema for. If not provided, returns schema for all tables.",  # noqa: E501
    ] = None,
) -> str:
    """Get the schema (column names and types) for tables in the ESDC database.

    Use this tool to understand the structure of tables.
    """
    safe_table_name = _validate_table_name(table_name)
    if table_name and not safe_table_name:
        return f"Error: Invalid table name '{table_name}'. Only alphanumeric, underscore, and hyphen allowed."  # noqa: E501

    cache = _get_tool_cache()
    cache_key = _tool_cache_key("get_schema", table_name=safe_table_name)
    if cache_key in cache:
        logger.debug("[CACHE] hit | tool=get_schema key=%s", cache_key[:16])
        return str(cache[cache_key])

    logger.debug("[CACHE] miss | tool=get_schema key=%s", cache_key[:16])

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

            cache.set(cache_key, output)
            logger.debug("[CACHE] stored | tool=get_schema key=%s", cache_key[:16])
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

            cache.set(cache_key, output)
            logger.debug("[CACHE] stored | tool=get_schema key=%s", cache_key[:16])
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
    cache = _get_tool_cache()
    cache_key = _tool_cache_key("list_tables")
    if cache_key in cache:
        logger.debug("[CACHE] hit | tool=list_tables key=%s", cache_key[:16])
        return str(cache[cache_key])

    logger.debug("[CACHE] miss | tool=list_tables key=%s", cache_key[:16])

    conn = None
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

        from esdc.dbmanager import get_last_updated

        last_updated = get_last_updated(conn)
        if last_updated:
            output += f"\nData last updated: {last_updated}\n"

        cache.set(cache_key, output)
        logger.debug("[CACHE] stored | tool=list_tables key=%s", cache_key[:16])
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
            return f"No models available for {provider_type}. The provider may not be configured."  # noqa: E501

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
        "Type of entity being queried: 'field', 'work_area', 'wa', 'national', 'nkri', or 'project'. "  # noqa: E501
        "Use 'field' for field-level queries, 'work_area' or 'wa' for work area queries, "  # noqa: E501
        "'national' or 'nkri' for national-level queries, 'project' for project-specific queries.",  # noqa: E501
    ],
    needs_project_detail: Annotated[
        bool,
        "Set to True if you need project-specific columns like project_name, project_remarks, "  # noqa: E501
        "or project-level breakdowns. When False (default), uses pre-aggregated views for better performance.",  # noqa: E501
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
    - Project breakdown: entity_type='field',
    needs_project_detail=True → {'table': 'project_resources', ...}
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
                "explanation": f"Defaulting to project_resources due to error: {str(e)}",  # noqa: E501
                "hierarchy": "project-level (most detailed)",
            }
        )


@tool("Uncertainty Resolver")
def resolve_uncertainty_level(
    level: Annotated[
        str,
        "Uncertainty level from user query. Examples: '1P', '2P', '3P', 'proven', 'probable', 'possible', "  # noqa: E501
        "'1C', '2C', '3C', '1R', '2R', '3R', '1U', '2U', '3U', 'terbukti' (proven), 'mungkin' (probable), "  # noqa: E501
        "'harapan' (possible). Case-insensitive.",
    ],
    volume_type: Annotated[
        str,
        "Type of volume being queried: 'reserves' or 'cadangan' for reserves, "
        "'resources' or 'sumber_daya' for resources, 'grr' for GRR, "
        "'contingent' for Contingent Resources, 'prospective' for Prospective Resources.",  # noqa: E501
    ] = "reserves",
) -> str:
    """Resolve uncertainty level to database filter values and SQL conditions.

    CRITICAL for 'probable' and 'possible' which are
    CALCULATED values (not in database):
    - 'probable' = 2P - 1P (Middle - Low) - REQUIRES CASE statements
    - 'possible' = 3P - 2P (High - Middle) - REQUIRES CASE statements

    These calculated values ONLY apply to RESERVES (not resources!).

    WHEN TO USE:
    - Call this when user mentions uncertainty levels (1P/2P/3P,
    proven/probable/possible)
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
    - resolve_uncertainty_level('probable', 'reserves') → calculated,
    returns CASE template
    - resolve_uncertainty_level('probable', 'resources') → ERROR, reserves only
    """
    import json

    from esdc.chat.domain_knowledge import get_uncertainty_filter, get_uncertainty_spec

    try:
        spec = get_uncertainty_spec(level, volume_type=volume_type)

        if spec is None:
            valid_levels = "1P, 2P, 3P, proven, probable, possible, 1C, 2C, 3C, 1R, 2R, 3R, 1U, 2U, 3U"  # noqa: E501
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
                f"'{level}' only applies to reserves. For {volume_type}, use 1C/2C/3C (contingent) or 1U/2U/3U (prospective)."  # noqa: E501
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
        "'slf' (Sales Forecast), 'spf' (Sales Potential Forecast), 'crf' (Contingent Resources Forecast), "  # noqa: E501
        "'prf' (Prospective Resources Forecast), 'ciof' (Consumed in Operation Forecast), "  # noqa: E501
        "or 'lossf' (Loss Production Forecast). Default is 'tpf'.",
    ] = "tpf",
    substance: Annotated[
        str,
        "Substance suffix: 'oil' (oil only), 'con' (condensate only), 'ga' (associated gas), "  # noqa: E501
        "'gn' (non-associated gas), 'oc' (oil + condensate combined), or 'an' (total gas). "  # noqa: E501
        "Default is 'oc'.",
    ] = "oc",
) -> str:
    """Get the correct column names for timeseries queries.

    CRITICAL: This tool prevents common errors where the model confuses rate_* columns
    (historical production RATES) with tpf_* columns (forecast VOLUMES).

    WHEN TO USE:
    - ALWAYS call this BEFORE writing SQL for timeseries/forecast queries
    - When user asks about "forecast", "perkiraan", "proyeksi", "peak production"
    - When querying project_timeseries, field_timeseries, wa_timeseries,
    or nkri_timeseries

    COLUMN CATEGORIES:
    1. Forecast VOLUMES (USE FOR FORECASTS): tpf_*, slf_*, spf_*, crf_*, prf_*
       - Units: MSTB (oil), BSCF (gas) - these are VOLUMES, not rates
       - Example: tpf_oc = forecast oil+condensate volume in MSTB

    2. Production RATES: rate_*
       - Units: MSTB/Y (oil), BSCF/Y (gas) - RATES per year, NOT volumes
       - Example: rate_oc = production rate in MSTB per year
       - NEVER use for forecast queries!

    ⚠️ CUMULATIVE PRODUCTION (cprd_grs_*, cprd_sls_*) are NOT available via this tool.
    Use get_resources_columns(volume_type="cumulative_production") instead.
    Cumulative production must ALWAYS be queried from *_resources, NEVER *_timeseries.

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
    - get_timeseries_columns("rate", substance="oc") → rate_oc for production rate
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
        "Type of volume: 'reserves' (commercial reserves only), 'resources' (GRR/Contingent/Prospective), "  # noqa: E501
        "or 'risked' (prospective resources with geological chance factor applied). Default is 'reserves'.",  # noqa: E501
    ] = "reserves",
    substance: Annotated[
        str,
        "Substance suffix: 'oil' (oil only), 'con' (condensate only), 'ga' (associated gas), "  # noqa: E501
        "'gn' (non-associated gas), 'oc' (oil + condensate combined), or 'an' (total gas). "  # noqa: E501
        "Default is 'oc'.",
    ] = "oc",
) -> str:
    """Get the correct column names for static resource queries.

    CRITICAL: This tool prevents confusion between
    res_* (reserves) and rec_* (resources) columns.
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

    3. Risked Resources (rec_*_risked): Universal shortcut for all resource classes
       - Columns: rec_oil_risked, rec_con_risked, etc.
       - GRR/Contingent: GCF=1, identical to rec_*
       - Prospective: GCF<1, differs from rec_*
       - Use for "all resources" without project_class filter

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
        "Search term for problem cluster. Can be partial name (e.g., 'subsurface', 'uneconomic'), "  # noqa: E501
        "cluster code (e.g., '1.1.1', '2.2'), or keyword from the problem description.",
    ],
) -> str:
    """Search for problem cluster definitions when user asks about project issues.

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
                    "suggestion": "Try searching for keywords like: subsurface, data, uneconomic, AMDAL, permit, etc.",  # noqa: E501
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


@tool("Entity Resolver")
def entity_resolver(
    query: Annotated[
        str,
        "Natural language query to resolve entities and match patterns "
        "against the knowledge graph. Examples: 'cadangan Duri 2024', "
        "'profil produksi Abadi', 'top 5 lapangan di WK Rokan'.",
    ],
    return_multiple: Annotated[
        bool,
        "If True, return all matching entities instead of single best match. "
        "Defaults to True so ambiguous or partial entity names surface all "
        "matches. Set False only when a single best match is required.",
    ] = True,
) -> str:
    """Resolve entities and match query patterns from the ESDC knowledge graph.

    This tool resolves entity names (fields, working areas, operators, years)
    and matches query patterns from natural language. It returns structured
    context that enables single-shot SQL generation, reducing multi-round
    tool calling to 1-2 calls.

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
    - entity_resolver("cadangan Duri 2024")
      → Entity: Field=Duri, Year=2024, Pattern: cadangan, Table: field_resources
    - entity_resolver("profil produksi Abadi")
      → Entity: Field=Abadi, Pattern: profil_produksi, Table: field_timeseries
    - entity_resolver("isu water cut di lapangan Duri")
      → Entity: Field=Duri, Pattern: issues_remarks, Table: field_resources
    """
    import json

    from esdc.chat.domain_knowledge.entity_resolver_lib import EntityResolver

    cache = _get_tool_cache()
    cache_key = _tool_cache_key(
        "entity_resolver", query=query, return_multiple=return_multiple
    )
    if cache_key in cache:
        logger.debug("[CACHE] hit | tool=entity_resolver key=%s", cache_key[:16])
        return str(cache[cache_key])

    logger.debug("[CACHE] miss | tool=entity_resolver key=%s", cache_key[:16])

    try:
        conn = get_db_connection()
        try:
            resolver = EntityResolver(db=conn)
            result = resolver.resolve(query=query, return_multiple=return_multiple)
            result["query"] = query

            result_str = json.dumps(result, indent=2, ensure_ascii=False)
            if result.get("status") in ("success", "ambiguous"):
                cache.set(cache_key, result_str)
                logger.debug(
                    "[CACHE] stored | tool=entity_resolver key=%s", cache_key[:16]
                )
            return result_str
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
                "message": f"Entity resolver error: {str(e)}",
                "query": query,
            }
        )


@tool("Spatial Resolver")
def resolve_spatial(
    query_type: Annotated[
        str,
        "Type of spatial query: 'proximity' (fields near a field), "
        "'working_area' (fields in a working area), 'distance' (between two fields), "
        "'coordinates' (get field coordinates), 'nearest_from_coords' (find nearest from lat/long), "  # noqa: E501
        "'field_clusters' (cluster fields by proximity), 'adjacent_wk' (find adjacent working areas), "  # noqa: E501
        "or 'average_distance' (average distance between multiple fields).",
    ],
    target: Annotated[
        str | dict,
        "For proximity: field name. For working_area: working area name. "
        "For distance: comma-separated 'field1, field2'. For coordinates: field name. "
        "For nearest_from_coords: dict with 'lat', 'long', 'entity_type' ('field' or 'working_area'). "  # noqa: E501
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
    - resolve_spatial("nearest_from_coords", '{"lat": 1.5, "long": 101.3,
    "entity_type": "field", "radius_km": 20}')
    - resolve_spatial("field_clusters", '{"max_distance_km": 20,
    "min_cluster_size": 2}')
    - resolve_spatial("adjacent_wk", '{"wk_name": "Rokan", "max_distance_km": 20}')
    - resolve_spatial("average_distance", '{"field_names": ["Duri", "Rokan",
    "Belanak"]}')
    """
    import json

    from esdc.search.spatial_resolver import SpatialResolver

    logger.debug(
        "[SPATIAL_START] query_type=%s | target=%s | radius_km=%s | wk_name=%s",
        query_type,
        target,
        radius_km,
        wk_name,
    )

    cache = _get_tool_cache()
    cache_key = _tool_cache_key(
        "resolve_spatial",
        query_type=query_type,
        target=str(target),
        radius_km=radius_km,
        limit=limit,
        wk_name=wk_name,
    )
    if cache_key in cache:
        logger.debug("[CACHE] hit | tool=resolve_spatial key=%s", cache_key[:16])
        return str(cache[cache_key])

    logger.debug("[CACHE] miss | tool=resolve_spatial key=%s", cache_key[:16])

    resolver = SpatialResolver()

    target_str: str = ""
    target_dict: dict[str, Any] = {}
    if isinstance(target, dict):
        target_dict = target
    else:
        target_str = str(target)

    try:
        if query_type == "proximity":
            result = resolver.find_fields_near_field(
                field_name=target_str,
                radius_km=radius_km,
                limit=limit,
                wk_name=wk_name,
            )
        elif query_type == "working_area":
            result = resolver.find_fields_in_working_area(
                wk_name=target_str, limit=limit
            )
        elif query_type == "distance":
            parts = [p.strip() for p in target_str.split(",")]
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
                field_name=target_str,
                wk_name=wk_name,
            )
        elif query_type == "nearest_from_coords":
            try:
                params = target_dict or json.loads(target_str)
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
                        "message": f"Invalid format for nearest_from_coords: {e}. Expected dict with lat, long, entity_type",  # noqa: E501
                    }
                )
        elif query_type == "field_clusters":
            try:
                params = target_dict or json.loads(target_str)
                result = resolver.find_field_clusters(
                    max_distance_km=float(params.get("max_distance_km", radius_km)),
                    min_cluster_size=int(params.get("min_cluster_size", 2)),
                )
            except (json.JSONDecodeError, ValueError, TypeError) as e:
                logger.debug("[SPATIAL_ERR] error=invalid_params")
                return json.dumps(
                    {
                        "status": "error",
                        "message": f"Invalid format for field_clusters: {e}. Expected dict with max_distance_km, min_cluster_size",  # noqa: E501
                    }
                )
        elif query_type == "adjacent_wk":
            try:
                params = target_dict or json.loads(target_str)
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
                        "message": f"Invalid format for adjacent_wk: {e}. Expected dict with wk_name, max_distance_km",  # noqa: E501
                    }
                )
        elif query_type == "average_distance":
            try:
                params = target_dict or json.loads(target_str)
                field_names = params.get("field_names", [])
                if not isinstance(field_names, list) or len(field_names) < 2:
                    logger.debug("[SPATIAL_ERR] error=insufficient_fields")
                    return json.dumps(
                        {
                            "status": "error",
                            "message": "average_distance requires at least 2 field names in 'field_names' list",  # noqa: E501
                        }
                    )
                result = resolver.calculate_average_distance(field_names=field_names)
            except (json.JSONDecodeError, ValueError, TypeError) as e:
                logger.debug("[SPATIAL_ERR] error=invalid_params")
                return json.dumps(
                    {
                        "status": "error",
                        "message": f"Invalid format for average_distance: {e}. Expected dict with field_names list",  # noqa: E501
                    }
                )
        else:
            logger.debug("[SPATIAL_ERR] error=unknown_query_type")
            return json.dumps(
                {
                    "status": "error",
                    "message": f"Unknown query_type: {query_type}. "
                    "Use: proximity, working_area, distance, coordinates, nearest_from_coords, field_clusters, adjacent_wk, or average_distance",  # noqa: E501
                }
            )

        result_count = (
            len(result.get("nearby_fields", [])) if isinstance(result, dict) else 0
        )
        logger.debug(
            "[SPATIAL_OK] query_type=%s | results=%d", query_type, result_count
        )
        result_str = json.dumps(result, indent=2, ensure_ascii=False)
        if isinstance(result, dict) and result.get("status") in (
            "success",
            "no_results",
        ):
            cache.set(cache_key, result_str)
            logger.debug("[CACHE] stored | tool=resolve_spatial key=%s", cache_key[:16])
        return result_str

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
        "Bilingual Indonesian/English concept query (5+ words) describing issues, "
        "problems, or characteristics in project_remarks. The database contains "
        "bilingual remarks (Indonesian narrative + English technical terms). "
        "Examples: 'proyek dengan masalah reservoir heterogen', "
        "'kendala teknis water injection'. "
        "DO NOT use for keyword matching on project_name (EOR, waterflood, water cut) "
        "or single words. For those, use execute_sql with ILIKE on project_name.",
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
        "Filter by working area subgroup (ILIKE pattern, e.g., '%Upstream%'). Optional.",  # noqa: E501
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
    """Search project remarks AND official documents by semantic similarity.

    Every call fans out to two sources and returns both:
    - project_remarks (via hybrid semantic + FTS search)
    - the ingested document corpus (surat, MoM, berita acara)

    Use this tool when:
    - User asks about concepts, meanings, or topics (not exact keywords)
    - User queries like: "proyek dengan masalah X", "lapangan yang sulit"
    - FTS returns no results or insufficient results
    - User wants to filter by year, field, working area, etc.

    Returns:
    JSON string with two top-level sections:
    - remarks: {status, results, count, message} — same shape as before,
      status is "success", "no_results", "not_available", "fallback_to_fts",
      or "error"
    - documents: {status, results, count, message} — hits from the document
      corpus, or status="not_available"/"error" if no corpus is ingested or
      the corpus lookup failed. A corpus failure never affects the remarks
      section. If documents is "not_available", do not mention documents in
      the answer unless the user specifically asked about them.

    Examples:
    - semantic_search("proyek dengan reservoir kompleks") ->
    Find projects with complex reservoir
    - semantic_search("masalah produksi", 5) -> Top 5 production issues
    - semantic_search("tidak ekonomis", report_year=2024) -> Economic issues in 2024
    - semantic_search("kendala teknis", field_name="%Duri%") ->
    Technical issues in Duri field
    """
    import json

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

    cache = _get_tool_cache()
    # v2 key: the return envelope changed from flat {status, ...} to
    # {remarks, documents}. The tool cache is permanent on disk, so old
    # flat-shape entries must never hit.
    cache_key = _tool_cache_key(
        "semantic_search_v2", query=query, limit=limit, **filters
    )
    if cache_key in cache:
        logger.debug("[CACHE] hit | tool=semantic_search key=%s", cache_key[:16])
        return str(cache[cache_key])

    logger.debug("[CACHE] miss | tool=semantic_search key=%s", cache_key[:16])

    resolver = _get_semantic_resolver()

    try:
        remarks_result = resolver.hybrid_search(
            query=query,
            limit=limit,
            filters=filters if filters else None,
        )

        # If embeddings not available, fallback to FTS search
        if remarks_result.get("status") == "not_available":
            logger.info("[Semantic] embeddings not available, falling back to FTS")
            remarks_result = _search_remarks_via_fts(query, limit, "project_resources")

    except Exception as e:
        logger.error("[Semantic] tool failed | query=%s error=%s", query, e)
        remarks_result = {
            "status": "error",
            "message": str(e),
            "query": query,
        }
    finally:
        resolver.close()

    # Fan out to the document corpus so issue/topic queries surface official
    # documents too. A corpus failure must never break the remarks result.
    documents_result: dict[str, Any] = {"status": "not_available"}
    try:
        from esdc.corpus.store import CorpusStore

        store = CorpusStore(embedder=_get_corpus_embedder())
        try:
            documents_result = store.search(
                query=query,
                limit=5,
                filters=_map_remarks_filters_to_corpus(filters),
            )
        finally:
            store.close()
    except Exception as e:
        logger.warning("[SemanticSearch] corpus fan-out failed: %s", e)
        documents_result = {"status": "error", "message": str(e)}

    result_str = json.dumps(
        {"remarks": remarks_result, "documents": documents_result},
        indent=2,
        ensure_ascii=False,
        default=str,
    )
    # Cache only when BOTH sections are definitive. The cache is not
    # invalidated by `esdc corpus commit`, so caching a not_available/error
    # documents section would freeze it even after a corpus is ingested.
    if remarks_result.get("status") in (
        "success",
        "no_results",
    ) and documents_result.get("status") in ("success", "no_results"):
        cache.set(cache_key, result_str)
        logger.debug("[CACHE] stored | tool=semantic_search key=%s", cache_key[:16])
    return result_str


def _search_remarks_via_fts(
    query: str,
    limit: int = 10,
    table_name: str | None = None,
) -> dict:
    """Fallback FTS search on project_remarks when embeddings unavailable.

    Searches project_remarks using FTS match_bm25 for keyword-based results.
    """
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

        from esdc.dbmanager import get_duckdb_connection

        conn = get_duckdb_connection(db_file, read_only=True)
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


def _map_remarks_filters_to_corpus(filters: dict[str, Any]) -> dict[str, Any]:
    """Map semantic_search's remarks filters to CorpusStore's filter schema.

    Only wk_name, field_name, project_name, and report_year (-> year) have
    equivalents in the document corpus; the rest (pod_name, province,
    basin128, project_class/stage/level, operator_*, wk_subgroup,
    wk_regionisasi_ngi, wk_area_perwakilan_skkmigas) are remarks-only and
    are dropped.
    """
    corpus_filters: dict[str, Any] = {}
    if "wk_name" in filters:
        corpus_filters["wk_name"] = filters["wk_name"]
    if "field_name" in filters:
        corpus_filters["field_name"] = filters["field_name"]
    if "project_name" in filters:
        corpus_filters["project_name"] = filters["project_name"]
    if "report_year" in filters:
        corpus_filters["year"] = filters["report_year"]
    return corpus_filters


_DOC_TYPE_VALUES = enum_values("doc_type")
_DOC_TOPIC_VALUES = enum_values("doc_topic")
_DOC_SCHEMA_CONTEXT = render_tool_context()


def _doc_filters_from_args(**kwargs: Any) -> dict[str, Any]:
    """Collect the non-None corpus filter arguments into a filters dict."""
    return {k: v for k, v in kwargs.items() if v is not None}


@tool("Document Search")
def search_documents(
    query: Annotated[
        str,
        "Bilingual Indonesian/English query about official documents "
        "(surat, minutes of meeting, berita acara). "
        "Example: 'persetujuan POD lapangan Duri 2025'.",
    ],
    limit: Annotated[int, "Maximum results (default 5)."] = 5,
    doc_type: Annotated[str | None, f"Filter: {', '.join(_DOC_TYPE_VALUES)}."] = None,
    doc_topic: Annotated[
        str | None, f"Filter by business topic: {', '.join(_DOC_TOPIC_VALUES)}."
    ] = None,
    year: Annotated[int | None, "Filter by document year."] = None,
    wk_name: Annotated[str | None, "Filter by working area (ILIKE pattern)."] = None,
    field_name: Annotated[str | None, "Filter by field name (ILIKE pattern)."] = None,
    project_name: Annotated[
        str | None, "Filter by project name (ILIKE pattern)."
    ] = None,
) -> str:
    """Search ingested official documents by meaning.

    Use this tool when:
    - User asks about official documents: surat, minutes of meeting (MoM),
      berita acara ingested via `esdc corpus`
    - User references correspondence, approvals, or meeting decisions:
      "surat tentang X", "MoM pembahasan Y", "dokumen persetujuan Z"
    - User wants document hits filtered by type, topic (POD, WP&B, PSC,
      ...), year, working area, field, or project

    Use this tool directly when the user asks about a specific document or
    document type (surat, MoM, berita acara, "POD I Revisi 2") — its
    doc_type/doc_topic/year filters give precise hits. For broad issue/topic
    exploration, semantic_search already includes a documents section.
    DO NOT use for reserves/production numbers (use execute_sql).
    DO NOT call entity_resolver first — this tool takes free-text names
    directly.

    Returns:
    JSON string with:
    - status: "success", "no_results", "not_available", or "error"
    - results: List of matching chunks with doc_id, file_name, doc_type,
      doc_topic, doc_date, subject, wk_name, field_name, project_name, section,
      chunk_text, and relevance score (RRF fusion, small magnitudes
      ~0.01-0.03 are normal)
    - count: Number of results
    - message: Additional information (e.g., how to ingest documents)

    Use read_document(doc_id) to fetch the full text of a hit.

    Examples:
    - search_documents("persetujuan POD lapangan Duri") -> POD approval letters
    - search_documents("pembahasan work program", doc_type="mom") -> MoM hits
    - search_documents("rencana kerja", doc_topic="wpnb") -> WP&B documents
    - search_documents("berita acara serah terima", year=2025) -> 2025 BA docs
    """
    filters = _doc_filters_from_args(
        doc_type=doc_type,
        doc_topic=doc_topic,
        year=year,
        wk_name=wk_name,
        field_name=field_name,
        project_name=project_name,
    )

    cache = _get_tool_cache()
    cache_key = _tool_cache_key("search_documents", query=query, limit=limit, **filters)
    if cache_key in cache:
        logger.debug("[CACHE] hit | tool=search_documents key=%s", cache_key[:16])
        return str(cache[cache_key])

    logger.debug("[CACHE] miss | tool=search_documents key=%s", cache_key[:16])

    store = None
    try:
        from esdc.corpus.store import CorpusStore

        store = CorpusStore(embedder=_get_corpus_embedder())
        # Mirror the CLI's _open_corpus_store: heal schema drift (e.g. a
        # pre-branch DuckDB missing the embed_text column) before search
        # runs its SELECT, so an upgraded install doesn't error on every
        # chat search until the user happens to run a corpus CLI command.
        # validate_model stays False (default) — chat search must not
        # hard-fail on an embedding-model mismatch.
        store.ensure_tables()
        result = store.search(
            query=query,
            limit=limit,
            filters=filters if filters else None,
        )

        if result.get("status") == "not_available":
            # Keep the store's diagnostic and append the actionable steps.
            hint = (
                "Run: esdc corpus extract <folder>, review the sidecars, "
                "then esdc corpus commit <folder>"
            )
            store_msg = result.get("message")
            result["message"] = f"{store_msg} {hint}" if store_msg else hint

        result_str = json.dumps(result, indent=2, ensure_ascii=False, default=str)
        if result.get("status") in ("success", "no_results"):
            cache.set(cache_key, result_str)
            logger.debug(
                "[CACHE] stored | tool=search_documents key=%s", cache_key[:16]
            )
        return result_str

    except Exception as e:
        logger.error("[DocSearch] tool failed | query=%s error=%s", query, e)
        return json.dumps(
            {
                "status": "error",
                "message": str(e),
                "query": query,
            }
        )
    finally:
        if store is not None:
            store.close()


# search_documents is a langchain StructuredTool; the LLM-facing text used
# for tool-calling is `.description` (captured from the function docstring
# at decoration time), not `.__doc__` (which on the StructuredTool instance
# resolves to the wrapper class's own docstring). Append the schema-derived
# field guide the same way esdc/chat/openterminal.py's
# `run_command.description = ...` does.
search_documents.description = (
    search_documents.description
    + "\n\nDocument metadata schema:\n"
    + _DOC_SCHEMA_CONTEXT
)


@tool("Document Aggregator")
def aggregate_documents(
    query: Annotated[
        str | None,
        "Term or phrase to match in document BODY text. Leave empty for a "
        "pure metadata count/list by doc_type/doc_topic/year/entity.",
    ] = None,
    mode: Annotated[
        str, "'count' (exhaustive total) or 'list' (deduped documents + count)."
    ] = "count",
    match: Annotated[
        str,
        "'hybrid' (default) = exact count plus semantically-similar "
        "candidates. 'keyword' = exact only. 'semantic' = ranking only.",
    ] = "hybrid",
    group_by: Annotated[
        str | None,
        "Facet dimension: year, doc_type, doc_level, doc_topic, wk_name, "
        "field_name, project_name.",
    ] = None,
    semantic_candidates: Annotated[
        int,
        "How many semantically-similar documents to return alongside the "
        "exact count. Does not affect count.",
    ] = 20,
    limit: Annotated[
        int, "Max documents returned in list mode. Counts are always exhaustive."
    ] = 50,
    doc_type: Annotated[str | None, f"Filter: {', '.join(_DOC_TYPE_VALUES)}."] = None,
    doc_topic: Annotated[
        str | None, f"Filter by business topic: {', '.join(_DOC_TOPIC_VALUES)}."
    ] = None,
    doc_level: Annotated[str | None, "Filter by document level."] = None,
    year: Annotated[int | None, "Filter by document year."] = None,
    wk_name: Annotated[str | None, "Filter by working area (ILIKE pattern)."] = None,
    field_name: Annotated[str | None, "Filter by field name (ILIKE pattern)."] = None,
    project_name: Annotated[
        str | None, "Filter by project name (ILIKE pattern)."
    ] = None,
    pod_name: Annotated[str | None, "Filter by POD name (ILIKE pattern)."] = None,
    sender: Annotated[
        str | None,
        "Filter by sending party ('dari X'), substring match. On an approval "
        "letter the SENDER is the approving authority, so 'disetujui oleh "
        "Menteri ESDM' means sender='Menteri ESDM' (also SKK Migas, BPMA).",
    ] = None,
    recipient: Annotated[
        str | None,
        "Filter by receiving party ('untuk X' / 'kepada X'), substring match. "
        "On an approval letter this is the KKKS being approved.",
    ] = None,
    subject: Annotated[
        str | None, "Filter by letter subject line, substring match."
    ] = None,
    doc_number: Annotated[
        str | None, "Filter by document/letter number, substring match."
    ] = None,
) -> str:
    """Count or list ALL documents matching a criterion — not the top few.

    Use this tool when:
    - The user asks HOW MANY documents: "berapa dokumen ...", "how many
      documents ..."
    - The user asks WHICH documents, exhaustively: "dokumen apa saja ...",
      "dokumen mana saja ...", "list all documents that ..."
    - The user wants a breakdown by year/type/topic/entity (use group_by)

    DO NOT use search_documents for these — it returns only the top few
    passages, so any count derived from it is wrong.

    match="hybrid" (default) gives you both: `count` is the EXACT number
    of documents whose body literally contains every query term, and
    `semantic_candidates` lists documents that are semantically related
    but did NOT contain the terms, each with a similarity score.

    How to report a hybrid result:
    - `count` is the answer. It is exact, reproducible, and safe to state
      as a number.
    - `semantic_candidates` are SUGGESTIONS, not part of the count. Say
      "N documents contain the term; M others appear related and may be
      worth reviewing". Never add the two together into one figure.
    - `provenance.exact_total` equals `count` and is a real total.
      `provenance.semantic_extra` is how many candidates were RETURNED —
      the size of a ranking you requested, not a measurement. Ask for 200
      and it says 200. Never report it as "200 related documents exist".

    match="keyword" skips the semantic pass entirely when you only want
    the defensible count. match="semantic" returns just the ranking; there
    `count` means "candidates returned", NOT a total, and approximate is
    true -- say so.

    There is no similarity threshold: absolute similarity scores are not
    comparable between queries (a 0.5 cutoff selects 6 documents for one
    query and 342 for another on this corpus), so the semantic side is
    always a ranking, capped by semantic_candidates.

    Offshore/onshore is a SQL attribute, not a document field: use
    execute_sql with is_offshore for that, not this tool.

    Returns:
    JSON string with status, mode, match, approximate, count, and either
    doc_ids (count mode) or documents (list mode), plus facets when
    group_by is set. Facets over multi-valued columns (doc_topic,
    wk_name, field_name, project_name) are flagged multi_valued and do
    NOT sum to count.

    `count` is ALWAYS the complete, exhaustive total over every matching
    document — it never shrinks because of `limit`. The doc_ids/documents
    ARRAY, however, is capped at `limit` items. When the array holds
    fewer items than `count`, the payload adds `returned` (items in the
    array) and `truncated: true`, plus a `note` string spelling out that
    the array is a partial page. NEVER report `returned` or the array's
    length as if it were the answer to "how many" — always report `count`,
    and when `truncated` is true, say the list you're showing is partial
    (e.g. "150 documents match; showing the first 50") rather than
    presenting the partial array as the complete set. Raise `limit` or add
    filters if the user needs to see more of the array itself.

    Examples:
    - aggregate_documents("separator", mode="list") -> every doc mentioning it
    - aggregate_documents("akan onstream 2026", match="semantic", year=2026)
    - aggregate_documents(mode="count", doc_topic="pod", group_by="year")
    """
    filters = _doc_filters_from_args(
        doc_type=doc_type,
        doc_topic=doc_topic,
        doc_level=doc_level,
        year=year,
        wk_name=wk_name,
        field_name=field_name,
        project_name=project_name,
        pod_name=pod_name,
        sender=sender,
        recipient=recipient,
        subject=subject,
        doc_number=doc_number,
    )

    cache = _get_tool_cache()
    cache_key = _tool_cache_key(
        "aggregate_documents",
        query=query,
        mode=mode,
        match=match,
        group_by=group_by,
        semantic_candidates=semantic_candidates,
        limit=limit,
        **filters,
    )
    if cache_key in cache:
        logger.debug("[CACHE] hit | tool=aggregate_documents key=%s", cache_key[:16])
        return str(cache[cache_key])

    store = None
    try:
        from esdc.corpus.store import CorpusStore

        store = CorpusStore(embedder=_get_corpus_embedder())
        store.ensure_tables()
        result = store.aggregate(
            query=query,
            mode=mode,
            match=match,
            group_by=group_by,
            semantic_candidates=semantic_candidates,
            limit=limit,
            filters=filters if filters else None,
        )

        if result.get("status") == "not_available":
            hint = (
                "Run: esdc corpus extract <folder>, review the sidecars, "
                "then esdc corpus commit <folder>"
            )
            store_msg = result.get("message")
            result["message"] = f"{store_msg} {hint}" if store_msg else hint

        result_str = json.dumps(result, indent=2, ensure_ascii=False, default=str)
        if result.get("status") in ("success", "no_results"):
            cache.set(cache_key, result_str)
        return result_str

    except Exception as e:
        logger.error("[DocAggregate] tool failed | query=%s error=%s", query, e)
        return json.dumps({"status": "error", "message": str(e), "query": query})
    finally:
        if store is not None:
            store.close()


# Same reasoning as search_documents.description above: the LLM-facing
# text is `.description`, captured at decoration time.
aggregate_documents.description = (
    aggregate_documents.description
    + "\n\nDocument metadata schema:\n"
    + _DOC_SCHEMA_CONTEXT
)

# `similarity_threshold` was removed rather than deprecated (see the
# docstring above): a caller that still passes it must get a loud error,
# not a silent no-op. Pydantic v2's default is to ignore unrecognized
# fields, so without this the removed kwarg would be dropped quietly and
# the caller would never learn it did nothing. Forbidding extras on this
# tool's schema turns that into a ValidationError.
aggregate_documents.args_schema.model_config["extra"] = "forbid"
aggregate_documents.args_schema.model_rebuild(force=True)


@tool("Document Reader")
def read_document(
    doc_id: Annotated[str, "doc_id returned by search_documents."],
    max_chars: Annotated[
        int, "Truncate markdown to this many chars (default 20000)."
    ] = 20000,
) -> str:
    """Fetch full markdown + metadata of one ingested document as JSON.

    Use this tool when:
    - search_documents returned a hit and the user needs the full document
      text (quotes, summaries, detailed answers)
    - User asks to read a specific ingested document by its doc_id

    Returns:
    JSON string with:
    - status: "success", "not_found", or "error"
    - document: Full metadata row (file_name, doc_type, doc_date, subject,
      sender, recipient, wk_name, field_name, project_name, ...) with
      markdown truncated to max_chars
    - document.truncated: true when markdown was cut at max_chars

    Examples:
    - read_document("a1b2c3") -> full text of document a1b2c3
    - read_document("a1b2c3", max_chars=5000) -> first 5000 chars only
    """
    store = None
    try:
        from esdc.corpus.store import CorpusStore

        store = CorpusStore(embedder=_get_corpus_embedder())
        doc = store.get_document(doc_id)
        if doc is None:
            logger.debug("[DocRead] not_found | doc_id=%s", doc_id)
            return json.dumps({"status": "not_found", "doc_id": doc_id})

        # Drop fields that are noise for the chat agent: embedding
        # bookkeeping and raw JSON-string blobs (raw_entities, metadata)
        # whose useful parts are already promoted to top-level columns.
        for noise_field in ("embedding_model", "raw_entities", "metadata"):
            doc.pop(noise_field, None)
        markdown = doc.get("markdown") or ""
        doc["truncated"] = len(markdown) > max_chars
        doc["markdown"] = markdown[:max_chars]
        logger.debug(
            "[DocRead] success | doc_id=%s truncated=%s", doc_id, doc["truncated"]
        )
        return json.dumps(
            {"status": "success", "document": doc},
            indent=2,
            ensure_ascii=False,
            default=str,
        )

    except Exception as e:
        logger.error("[DocRead] tool failed | doc_id=%s error=%s", doc_id, e)
        return json.dumps(
            {
                "status": "error",
                "message": str(e),
                "doc_id": doc_id,
            }
        )
    finally:
        if store is not None:
            store.close()


def _format_find_results(
    results: list[dict[str, Any]],
) -> str:
    """Format FTS find results into readable text."""
    if not results:
        return "No matching entities found."
    lines = []
    for i, r in enumerate(results, 1):
        lines.append(f"### {i}. {r.get('code', 'N/A')} — {r.get('name', 'N/A')}")
        lines.append(f"   Type: {r.get('source_table', r.get('entity_type', 'N/A'))}")
        if r.get("definition"):
            defn = r["definition"]
            if len(defn) > 300:
                defn = defn[:300] + "..."
            lines.append(f"   Definition: {defn}")
        if r.get("aliases"):
            aliases = r["aliases"]
            if isinstance(aliases, str):
                aliases = aliases.split("||")
            lines.append(f"   Aliases: {', '.join(str(a) for a in aliases[:5])}")
        lines.append(f"   Relevance: {r.get('score', 0):.4f}")
        lines.append("")
    return "\n".join(lines)


def _format_traverse_results(
    entity_code: str,
    relationship: str,
    results: list[dict[str, Any]],
) -> str:
    """Format graph traversal results into readable text."""
    rel_labels = {
        "CLASSIFIED_AS": "classified as",
        "REPORTED_AS": "reported as",
        "CAN_TRANSITION_TO": "can transition to",
        "HAS_LEVEL": "has level",
        "BELONGS_TO_FRAMEWORK": "belongs to framework",
        "HAS_SUBSTANCE": "has substance",
    }
    rel_label = rel_labels.get(relationship, relationship)
    if not results:
        return f"Entity '{entity_code}' has no {rel_label} relationships."
    lines = [f"Entity '{entity_code}' {rel_label}:"]
    for r in results:
        code = r.get("code", "N/A")
        name = r.get("name", "N/A")
        rtype = r.get("type", "")
        lines.append(f"  - {code}: {name} ({rtype})")
    return "\n".join(lines)


@tool("Knowledge Traversal")
def knowledge_traversal(
    topic: Annotated[
        str,
        "Knowledge topic to retrieve. "
        "One of: 'definition' (concept definitions), "
        "'level' (project maturity levels E0-X6, A1-A2), "
        "'transition' (level transition rules, reachability matrix, WAP constraints), "
        "'formula' (volume formulas, GRR, EUR, Gross/Net/Sales), "
        "'hierarchy' (classification hierarchy tree), "
        "'entity' (entities: WAP, PSE, GROOVY, etc.), "
        "'document' (PSE, GROOVY, izin berproduksi), "
        "'commercial' (commercial factors), "
        "'all' (entire knowledge base).",
    ],
    entity: Annotated[
        str | None,
        "Optional entity name to narrow results. "
        "Examples: 'E0', 'GRR', 'PSE', 'GROOVY', "
        "'DokumenPenentuanStatusEksplorasi', 'SalesPotentialResources', 'TBS'. "
        "If provided, returns only that entity regardless of topic.",
    ] = None,
    relationship: Annotated[
        str | None,
        "Optional relationship type for graph traversal. "
        "Use to find related entities. Examples: "
        "'CLASSIFIED_AS' (what classification a level belongs to), "
        "'REPORTED_AS' (what volume types a classification reports as), "
        "'CAN_TRANSITION_TO' (what levels a level can transition to), "
        "'HAS_LEVEL' (what levels belong to the KSMI framework), "
        "'BELONGS_TO_FRAMEWORK' (what entities belong to KSMI). "
        "Only effective when entity is also provided.",
    ] = None,
    include_reachability: Annotated[
        bool,
        "If True and topic is 'transition' or 'level', auto-append the full "
        "reachability matrix (Level → Allowed Targets) to the output. The "
        "queried entity, if any, is highlighted with a marker. Prevents "
        "common reasoning errors like claiming E3 can transition to E4 "
        "(only E0, E2, E5 are valid targets for E3). Default: True.",
    ] = True,
) -> str:
    """Retrieve domain knowledge — definitions, rules, transitions, formulas.

    Traverses the KSMI knowledge schema to provide detailed information about:
    - Project maturity levels (E0-On Production, E1-Production on Hold, etc.)
    - Level transition rules (which levels can transition to which)
    - WAP constraints (max WAP duration, GROOVY dispensation)
    - Volume formulas (EUR, GRR, Gross/Net/Sales relationships)
    - Document semantics (PSE vs Izin Berproduksi, GROOVY)
    - is_pod_approved / is_pse_approved logic per level
    - Classification hierarchy (Reserves > GRR, Contingent, Prospective)

    The short table in the system prompt covers level codes and basic rules.
    Use this tool for ANY detailed question about KSMI concepts.

    When 'relationship' is provided with 'entity', performs a graph traversal
    to find related entities. For example:
    - entity='Reserves', relationship='REPORTED_AS' → returns gross, net, sales
    - entity='E0', relationship='CAN_TRANSITION_TO' → returns E1, E4, E7

    When 'include_reachability' is True (default) and topic is 'transition'
    or 'level', the output is automatically extended with a compact
    reachability matrix covering all 18 levels (E0-E8, X0-X6, A1, A2).
    The queried entity, if provided, is visually highlighted.

    Entity lookup also consults a general oil & gas glossary of non-KSMI
    commercial/financing terms (e.g. TBS = Trustee Borrowing Scheme) that
    may appear in document text but are not part of the KSMI framework.

    Returns formatted text with definitions, key concepts, and rules.
    """
    if entity:
        try:
            from esdc.loaders import lookup_loaded_schema

            loaded_schema_result = lookup_loaded_schema(entity)
            if loaded_schema_result:
                return loaded_schema_result
        except Exception as e:
            logger.warning("[LoadedSchema-KG] lookup_failed | error=%s", e)

        try:
            from esdc.chat.domain_knowledge.glossary import glossary_lookup

            glossary_result = glossary_lookup(entity)
            if glossary_result:
                return glossary_result
        except Exception as e:
            logger.warning("[Glossary] lookup_failed | error=%s", e)

    base_output, matrix_text = _query_graph(
        entity, relationship, topic, include_reachability
    )

    if base_output is not None:
        if matrix_text and matrix_text not in base_output:
            return f"{base_output}\n\n---\n\n{matrix_text}"
        return base_output

    if matrix_text is not None:
        return matrix_text

    from esdc.chat.domain_knowledge.ksmi_loader import ksmi_retrieve

    return ksmi_retrieve(topic=topic, entity=entity)


_REACHABILITY_TOPICS = frozenset({"transition", "level"})


def _query_graph(
    entity: str | None,
    relationship: str | None,
    topic: str,
    include_reachability: bool,
) -> tuple[str | None, str | None]:
    """Query the KSMI graph for entity info and reachability matrix.

    Returns:
        Tuple of (base_output, matrix_text). Either or both may be None
        if the graph is unavailable or the queries return no results.
    """
    from esdc.chat.domain_knowledge.ksmi_graph_manager import KSMIGraphManager

    base_output: str | None = None
    matrix_text: str | None = None
    try:
        mgr = KSMIGraphManager()
        if entity and relationship:
            results = mgr.traverse(entity, relationship)
            if results:
                base_output = _format_traverse_results(entity, relationship, results)
        if base_output is None and entity:
            results = mgr.find_all(entity)
            if results:
                base_output = _format_find_results(results)
        if include_reachability and topic.lower() in _REACHABILITY_TOPICS:
            try:
                matrix_text = mgr.format_reachability(highlight=entity)
            except Exception as e:
                logger.warning("[KSMI-KG] reachability_format_error | %s", e)
    except Exception as e:
        logger.warning("[KSMI-KG] graph_fallback | error=%s", e)
    return base_output, matrix_text


def _get_instance_graph():
    """Seam for tests; returns the singleton instance graph manager."""
    from esdc.chat.domain_knowledge.instance_graph import get_instance_graph

    return get_instance_graph()


def _get_knowledge_context(
    entity_type: str, entity_id: str
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Fetch dossier (duckdb) and claims (sqlite) for one resolved entity."""
    from esdc.knowledge.dossier import get_dossier
    from esdc.knowledge.store import KnowledgeStore
    from esdc.pod_registry.store import get_sqlite_connection

    dossier = None
    conn = None
    try:
        conn = get_db_connection()
        dossier = get_dossier(conn, entity_type, entity_id)
    except Exception as e:  # table may not exist before first learn
        logger.debug("[ExploreEntity] no dossier | %s", e)
    finally:
        if conn:
            conn.close()

    claims: list[dict[str, Any]] = []
    sconn = get_sqlite_connection()
    try:
        store = KnowledgeStore(sconn)
        store.ensure_tables()
        claims = [
            {
                "doc_id": c.doc_id,
                "type": c.claim_type,
                "predicate": c.predicate,
                "value": c.value,
                "evidence": c.evidence,
            }
            for c in store.claims_for(entity_type, entity_id)
        ]
    finally:
        sconn.close()
    return dossier, claims


@tool("Entity Knowledge Explorer")
def explore_entity(
    entity: Annotated[
        str,
        "Entity name to explore: POD name, field name, working area, "
        "project name, or document subject. Free text, bilingual. "
        "Example: 'POD I Duri Revisi 1'.",
    ],
    entity_type: Annotated[
        str | None,
        "Optional filter: 'pod', 'project', 'field', 'working_area', "
        "'document'. Leave empty to search all types.",
    ] = None,
) -> str:
    """Explore everything known about one entity via the knowledge graph.

    Built by `esdc corpus learn`. For a POD this returns its full dossier
    (approval, economics, commitments, meeting history, current issues —
    with [doc_id] citations), all related entities (documents about it,
    projects under it, revision chain, field/WK), and extracted claims.

    Use this tool when:
    - The user asks a broad question about one POD/field/project/WK:
      "bagaimana keekonomian POD X", "status proyek Y", "ceritakan POD Z"
    - You need the connections: which MoMs discussed a POD, what a POD
      revised, which projects implement it
    - A search_documents hit mentions a POD and you want its full context

    Follow-ups: use read_document(doc_id) on any cited doc_id; use
    execute_sql for current numbers.
    DO NOT use for aggregate portfolio queries (use execute_sql).

    Returns JSON with entity, dossier (markdown), related (edges grouped
    by relation), claims, status.
    """
    from esdc.chat.domain_knowledge.instance_graph import _TYPE_TO_LABEL

    if entity_type:
        # Normalize LLM-provided variants ('POD', 'working area', ...) to the
        # canonical keys graph.find() understands, same convention as the
        # entity_key normalization in get_recommended_table (tools.py:737).
        entity_type = entity_type.strip().lower().replace(" ", "_")
        if entity_type not in _TYPE_TO_LABEL:
            return json.dumps(
                {
                    "status": "error",
                    "message": (
                        f"Unknown entity_type '{entity_type}'. Valid: pod, "
                        "project, field, working_area, document."
                    ),
                }
            )

    cache = _get_tool_cache()
    cache_key = _tool_cache_key(
        "explore_entity", entity=entity, entity_type=entity_type
    )
    if cache_key in cache:
        return str(cache[cache_key])

    try:
        graph = _get_instance_graph()
        if not graph.is_available():
            return json.dumps(
                {
                    "status": "not_available",
                    "message": (
                        "Knowledge graph not built yet. Run: esdc corpus learn"
                    ),
                }
            )
        hits = graph.find(entity, top_k=5, entity_type=entity_type)
        if entity_type:
            # Harmless safety net: find() already restricts to entity_type
            # when given, so this should be a no-op in practice.
            hits = [h for h in hits if h["entity_type"] == entity_type]
        if not hits:
            return json.dumps(
                {
                    "status": "not_found",
                    "message": f"No entity matching '{entity}'.",
                }
            )
        top = hits[0]
        related = graph.neighbors(top["entity_type"], top["entity_id"])
        dossier, claims = _get_knowledge_context(top["entity_type"], top["entity_id"])
        result = json.dumps(
            {
                "status": "success",
                "entity": top,
                "other_matches": hits[1:],
                "dossier": dossier["dossier_text"] if dossier else None,
                "related": related,
                "claims": claims,
                "message": (
                    None
                    if dossier
                    else "No dossier for this entity yet (dossiers exist for "
                    "PODs after `esdc corpus learn`)."
                ),
            },
            indent=2,
            ensure_ascii=False,
            default=str,
        )
        cache.set(cache_key, result)
        return result
    except Exception as e:
        logger.error("[ExploreEntity] failed | entity=%s error=%s", entity, e)
        return json.dumps({"status": "error", "message": str(e)})
