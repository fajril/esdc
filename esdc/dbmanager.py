import contextlib
import logging
import shutil
import time
from pathlib import Path

import duckdb
import pandas as pd

from esdc.configs import Config
from esdc.console import console
from esdc.db_security import SQLSanitizer, _load_sql_script
from esdc.selection import TableName


def get_duckdb_connection(
    db_path: str | Path, read_only: bool = False
) -> duckdb.DuckDBPyConnection:
    """Open a DuckDB connection with the VSS extension loaded.

    The VSS extension must be loaded before any operation on a database
    that contains HNSW indexes, including CHECKPOINT and WAL replay.
    Failing to load VSS before checkpointing causes:
    ``Missing Extension Error: Cannot bind index, unknown index type 'HNSW'``.

    INSTALL is idempotent (no-op if already installed).
    LOAD is idempotent (no-op if already loaded).
    """
    conn = duckdb.connect(str(db_path), read_only=read_only)
    with contextlib.suppress(Exception):
        conn.execute("INSTALL vss")
    conn.execute("LOAD vss")
    return conn


def _is_sqlite_database(db_path: Path) -> bool:
    """Check if a database file is in SQLite format."""
    if not db_path.exists():
        return False
    try:
        with open(db_path, "rb") as f:
            header = f.read(16)
        return header[:6] == b"SQLite"
    except OSError:
        return False


def _ensure_duckdb_database(db_path: Path) -> None:
    """Remove old SQLite database if it exists.

    DuckDB cannot open SQLite-format database files. The user needs
    to re-fetch data to rebuild in DuckDB format.
    """
    if _is_sqlite_database(db_path):
        logging.warning(
            "Found SQLite-format database. Removing it for DuckDB migration. "
            "Data will be rebuilt on fetch."
        )
        db_path.unlink()


def _create_fts_indexes(conn: duckdb.DuckDBPyConnection) -> None:
    """Create Full-Text Search indexes for fast text matching."""
    try:
        conn.execute("INSTALL fts")
        conn.execute("LOAD fts")
    except duckdb.Error:
        logging.warning("FTS extension not available, skipping FTS index creation")
        return

    fts_configs = [
        {
            "table": "project_resources",
            "id_col": "uuid",
            "index_cols": [
                "project_name",
                "field_name",
                "wk_name",
                "province",
                "basin128",
                "operator_name",
                "project_remarks",
                "vol_remarks",
                "basin86",
                "wk_id",
                "field_id",
                "field_name_previous",
                "project_name_previous",
                "pod_name",
                "operator_group",
                "wk_subgroup",
            ],
        },
        {
            "table": "project_timeseries",
            "id_col": "uuid",
            "index_cols": [
                "project_name",
                "field_name",
                "wk_name",
                "province",
                "basin128",
                "operator_name",
                "project_remarks",
                "vol_remarks",
                "basin86",
                "wk_id",
                "field_id",
                "field_name_previous",
                "project_name_previous",
                "pod_name",
                "operator_group",
                "wk_subgroup",
            ],
        },
    ]

    existing_tables = {
        row[0]
        for row in conn.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'main'"
        ).fetchall()
    }

    for config in fts_configs:
        table_name = config["table"]
        if table_name not in existing_tables:
            logging.debug(
                "Skipping FTS index for %s (table not yet loaded)", table_name
            )
            continue

        id_col = config["id_col"]
        cols_str = ", ".join(f"'{c}'" for c in config["index_cols"])
        try:
            conn.execute(
                f"PRAGMA create_fts_index('{table_name}', '{id_col}', "
                f"{cols_str}, lower=1, strip_accents=1, stemmer='', stopwords='', overwrite=1)"
            )
            logging.debug("FTS index created for %s", table_name)
        except duckdb.Error as e:
            logging.warning("Failed to create FTS index for %s: %s", table_name, e)

    btree_indexes = [
        (
            "idx_project_resources_report_year",
            "project_resources",
            "report_year DESC",
        ),
        (
            "idx_project_resources_agg",
            "project_resources",
            "report_year, wk_id, field_id, project_class, project_level",
        ),
        (
            "idx_project_timeseries_report_year",
            "project_timeseries",
            "report_year DESC",
        ),
        (
            "idx_project_timeseries_agg",
            "project_timeseries",
            "report_year, wk_id, field_id, project_class, project_level, year",
        ),
    ]

    for idx_name, table_name, columns in btree_indexes:
        if table_name not in existing_tables:
            continue
        try:
            conn.execute(
                f"CREATE INDEX IF NOT EXISTS {idx_name} ON {table_name}({columns})"
            )
            logging.debug("B-tree index created: %s", idx_name)
        except duckdb.Error as e:
            logging.warning("Failed to create B-tree index %s: %s", idx_name, e)


def reindex_fts() -> None:
    """Rebuild FTS and B-tree indexes on existing database tables.

    Connects to the existing DuckDB database and creates/recreates
    Full-Text Search indexes and B-tree indexes without reloading data.
    """
    db_path = Config.get_db_file()
    if not db_path.exists():
        logging.error("Database not found at %s. Run 'esdc fetch' first.", db_path)
        return

    logging.info("Rebuilding FTS indexes on %s", db_path)
    conn = get_duckdb_connection(db_path)
    try:
        _create_fts_indexes(conn)
        conn.execute("CHECKPOINT")
        logging.info("FTS indexes rebuilt successfully.")
    except duckdb.Error as e:
        logging.error("Failed to rebuild FTS indexes: %s", e)
    finally:
        conn.close()


def load_data_to_db(
    content: list[list[str]], header: list[str], table_name: str
) -> None:
    """Load data into the ESDC database.

    Args:
        content: The data rows to load into the database.
        header: Column names for the data.
        table_name: The name of the table to load the data into.

    Raises:
        duckdb.Error: If there is an error connecting to the database
            or executing a query.
    """
    create_table_query = {
        "project_resources": "create_table_project_resources.sql",
        "project_timeseries": "create_table_project_timeseries.sql",
    }
    start_time = time.monotonic()
    label = f"Loading {table_name}" if len(content) > 1000 else f"{table_name}"

    def _status(step: str) -> str:
        elapsed = time.monotonic() - start_time
        return f"[dim]{label}: {step} [elapsed {elapsed:.1f}s][/dim]"

    _ensure_duckdb_database(Config.get_db_file())
    if not Config.get_db_dir().exists():
        Config.get_db_dir().mkdir(parents=True, exist_ok=True)
    conn = get_duckdb_connection(Config.get_db_file())

    with console.status(_status("preparing")) as status:
        try:
            status.update(_status("creating schema"))
            _execute_sql_script(conn, create_table_query[table_name])

            status.update(_status(f"inserting {len(content):,} rows"))
            placeholders = ", ".join(["?" for _ in header])
            stmt = (
                f"INSERT INTO {table_name} "
                f"({', '.join(header)}) VALUES ({placeholders})"
            )
            conn.executemany(stmt, content)

            status.update(_status("creating uuid columns"))
            _execute_sql_script(
                conn,
                "create_column_uuid.sql",
                replacements={"{table_name}": table_name},
            )

            if table_name == "project_resources":
                status.update(_status("creating resource views"))
                _execute_sql_script(conn, "create_project_resources_uuid.sql")
                _execute_sql_script(conn, "create_project_resources_is_discovered.sql")
                _execute_sql_script(conn, "create_project_resources_project_stage.sql")
                _execute_sql_script(conn, "create_esdc_view.sql")

            if table_name == "project_timeseries":
                status.update(_status("creating timeseries views"))
                _execute_sql_script(conn, "create_timeseries_views.sql")

            if table_name in ("project_resources", "project_timeseries"):
                status.update(_status("building search indexes"))
                _create_fts_indexes(conn)

            status.update(_status("checkpointing"))
            conn.execute("CHECKPOINT")

            elapsed = time.monotonic() - start_time
            console.print(
                f"[green]✓[/green] {label}: "
                f"{len(content):,} rows loaded in {elapsed:.1f}s"
            )
        finally:
            conn.close()

    invalidate_sql_cache()
    from esdc.chat.tools import invalidate_tool_cache, reset_sql_cache

    reset_sql_cache()
    invalidate_tool_cache()


def run_query(
    table: TableName,
    where: str | None = None,
    like: str | None = None,
    years: list[int] | None = None,
    details: list[str] | None = None,
    columns: str | list[str] = "",
) -> pd.DataFrame | None:
    """Execute a parameterized query on the specified table in the database.

    Args:
        table: The table to query.
        where: The column name for the WHERE clause. Defaults to None.
        like: The value for the ILIKE clause. Defaults to None.
        years: The years to filter by. Defaults to None.
        details: Detail levels to show (reserves, resources,
            inplace, cumprod, rate, all).
        columns: The columns to select. Defaults to "".

    Returns:
        The query results as a pandas DataFrame, or None if the query fails.
    """
    if not Config.get_db_file().exists():
        logging.error(
            """
        Database file does not exist. Try to run this command first:

            esdc fetch --save
        """
        )
        return None

    if _is_sqlite_database(Config.get_db_file()):
        logging.error(
            "Database is in SQLite format. "
            "Run 'esdc fetch --save' to rebuild in DuckDB format."
        )
        return None

    try:
        query, params = SQLSanitizer.build_query(
            table, where=where, like=like, years=years, details=details, columns=columns
        )
        conn = get_duckdb_connection(Config.get_db_file(), read_only=True)
        try:
            df = conn.execute(query, params).fetchdf()
        finally:
            conn.close()
    except duckdb.Error as e:
        logging.error("Cannot query data. Message: %s", e)
        return None

    return df


def _execute_sql_script(
    conn: duckdb.DuckDBPyConnection,
    script: str | Path,
    replacements: dict[str, str] | None = None,
) -> None:
    """Execute a SQL script that may contain multiple statements.

    Splits on ';' and executes each non-empty statement.

    Args:
        conn: Active duckdb connection.
        script: Either a SQL script filename (in esdc/sql/) or
            raw SQL content to execute directly.
        replacements: Optional dict of template replacements.
            e.g. {"{table_name}": "project_resources"}
    """
    if isinstance(script, Path) or not script.strip().startswith(
        ("-", "C", "D", "A", "S", "I", "U")
    ):
        sql_content = _load_sql_script(str(script))
    else:
        sql_content = script

    if replacements:
        for placeholder, value in replacements.items():
            sql_content = sql_content.replace(placeholder, value)

    statements = [s.strip() for s in sql_content.split(";") if s.strip()]
    for stmt in statements:
        conn.execute(stmt)


def invalidate_sql_cache() -> None:
    """Clear the SQL results cache directory."""
    cache_dir = Config.get_cache_dir() / "sql_results"
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
        logging.info("SQL cache invalidated: %s", cache_dir)
