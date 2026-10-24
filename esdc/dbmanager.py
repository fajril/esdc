"""DuckDB database lifecycle, import, and export management."""

import contextlib
import datetime
import itertools
import logging
import time
import uuid
from collections.abc import Sequence
from pathlib import Path

import diskcache
import duckdb
import pandas as pd

from esdc.configs import Config
from esdc.console import console
from esdc.db_security import SQLSanitizer, _load_sql_script
from esdc.fetching import PreparedTimeseriesSlice
from esdc.selection import TableName


def get_duckdb_connection(
    db_path: str | Path,
    read_only: bool = True,
    *,
    allow_install: bool = True,
) -> duckdb.DuckDBPyConnection:
    """Open a DuckDB connection with VSS and Spatial extensions loaded.

    Read-only by default: DuckDB allows many concurrent read-only
    processes but a read-write connection demands an exclusive lock on
    the file, blocking everything else (`esdc serve` vs `esdc status`).
    Write paths must opt in explicitly with ``read_only=False``; a write
    attempt on a read-only connection fails loudly instead of silently
    holding the exclusive lock. Note: ``read_only=True`` cannot create a
    missing database file.

    The VSS extension must be loaded before any operation on a database
    that contains HNSW indexes, including CHECKPOINT and WAL replay.
    Failing to load VSS before checkpointing causes:
    ``Missing Extension Error: Cannot bind index, unknown index type 'HNSW'``.

    INSTALL is best effort for Spatial because offline callers may already have
    the extension installed locally.  LOAD is required: callers that query
    geometry must observe a clear error instead of silently falling back to
    the raw ``field_geom`` WKT column.
    """
    if allow_install:
        conn = duckdb.connect(str(db_path), read_only=read_only)
    else:
        conn = duckdb.connect(
            str(db_path),
            read_only=read_only,
            config={"autoinstall_known_extensions": "false"},
        )
    try:
        if allow_install:
            with contextlib.suppress(Exception):
                conn.execute("INSTALL vss")
        conn.execute("LOAD vss")
        if allow_install:
            try:
                conn.execute("INSTALL spatial")
            except duckdb.Error as exc:
                logging.debug("Spatial extension install skipped: %s", exc)
        conn.execute("LOAD spatial")
    except BaseException as exc:
        if isinstance(exc, duckdb.Error):
            label = "VSS" if "vss" in str(exc).lower() else "Spatial"
            wrapped = duckdb.Error(f"Unable to load {label} extension: {exc}")
        else:
            wrapped = exc
        try:
            conn.close()
        except BaseException as cleanup:
            if isinstance(wrapped, (KeyboardInterrupt, SystemExit)):
                wrapped.args = (
                    *wrapped.args,
                    f"closing DuckDB connection also failed: {cleanup}",
                )
            else:
                logging.exception("Could not close DuckDB after extension load failure")
        if wrapped is exc:
            raise
        raise wrapped from exc
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
    """Refuse SQLite files without changing their contents."""
    if _is_sqlite_database(db_path):
        raise ValueError(
            f"SQLite-format database cannot be opened as DuckDB: {db_path}"
        )


def _create_fts_indexes(
    conn: duckdb.DuckDBPyConnection,
    *,
    tables: Sequence[str] | None = None,
    extensions_loaded: bool = False,
    allow_install: bool = True,
) -> dict[str, str]:
    """Create Full-Text Search indexes for fast text matching.

    Returns a dict mapping table names to their FTS creation result:
    "ok" on success, or an error message on failure.
    """
    fts_results: dict[str, str] = {}

    supported_tables = ("project_resources", "project_timeseries")
    selected_tables = (
        supported_tables if tables is None else tuple(dict.fromkeys(tables))
    )
    unsupported = set(selected_tables) - set(supported_tables)
    if unsupported:
        raise ValueError(
            f"Unsupported index table(s): {', '.join(sorted(unsupported))}"
        )

    try:
        if not extensions_loaded:
            if allow_install:
                conn.execute("INSTALL fts")
            conn.execute("LOAD fts")
    except duckdb.Error:
        logging.warning("FTS extension not available, skipping FTS index creation")
        for table in selected_tables:
            fts_results[table] = "FTS extension not available"
        return fts_results

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
                "groovy_report",
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
                "basin86_id",
                "wk_id",
                "field_id",
                "pod_name",
                "operator_group",
                "wk_subgroup",
            ],
        },
    ]
    fts_configs = [c for c in fts_configs if c["table"] in selected_tables]

    existing_tables = {
        row[0]
        for row in conn.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'main'"
        ).fetchall()
    }

    for config in fts_configs:
        table_name = config["table"]
        if not table_name or table_name not in existing_tables:
            logging.debug(
                "Skipping FTS index for %s (table not yet loaded)",
                table_name or "<empty>",
            )
            fts_results[table_name or "<empty>"] = "table not loaded"
            continue

        row_count = (
            conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone() or (0,)
        )[0]
        if row_count == 0:
            logging.info("Skipping FTS index for %s (0 rows)", table_name)
            fts_results[table_name] = "skipped (0 rows)"
            continue

        id_col = config["id_col"]
        actual_columns = {
            row[0]
            for row in conn.execute(
                f"SELECT column_name FROM information_schema.columns "
                f"WHERE table_name = '{table_name}'"
            ).fetchall()
        }
        available_cols = [c for c in config["index_cols"] if c in actual_columns]
        if len(available_cols) != len(config["index_cols"]):
            missing = [c for c in config["index_cols"] if c not in actual_columns]
            logging.warning(
                "Skipping missing FTS columns for %s: %s",
                table_name,
                missing,
            )
        if not available_cols:
            logging.warning("No valid FTS columns for %s, skipping", table_name)
            fts_results[table_name] = "no valid FTS columns"
            continue
        cols_str = ", ".join(f"'{c}'" for c in available_cols)
        console.print(f"  Building FTS index for {table_name} ({row_count:,} rows)...")
        try:
            conn.execute(
                f"PRAGMA create_fts_index('{table_name}', "
                f"'{id_col}', {cols_str}, "
                f"lower=1, strip_accents=1, overwrite=1)"
            )
            logging.info("FTS index created for %s", table_name)
            fts_results[table_name] = "ok"
            console.print(f"  [green]✓[/green] FTS index: {table_name}")
        except duckdb.Error as e:
            error_msg = str(e)
            if "Catalog Error" in error_msg and "does not exist" in error_msg:
                logging.error(
                    "DuckDB FTS index creation failed for '%s'. Original error: %s",
                    table_name,
                    error_msg,
                )
            else:
                logging.warning("Failed to create FTS index for %s: %s", table_name, e)
            fts_results[table_name] = str(e)
            console.print(f"  [red]✗[/red] FTS index: {table_name} — {e}")

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
    btree_indexes = [item for item in btree_indexes if item[1] in selected_tables]

    if any(t in existing_tables for _, t, _ in btree_indexes):
        console.print("  Creating B-tree indexes...")
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
            previous = fts_results.get(table_name, "ok")
            fts_results[table_name] = f"{previous}; {idx_name}: {e}"

    return fts_results


def reindex_fts(*, strict: bool = False, allow_install: bool = True) -> None:
    """Rebuild indexes on the configured database, best-effort by default."""
    if allow_install:
        _reindex_fts_for_path(Config.get_db_file(), strict=strict)
    else:
        _reindex_fts_for_path(
            Config.get_db_file(), strict=strict, allow_install=False
        )


def _reindex_fts_for_path(
    db_path: Path, *, strict: bool = False, allow_install: bool = True
) -> None:
    """Rebuild FTS and B-tree indexes on existing database tables.

    Connects to the existing DuckDB database and creates/recreates
    Full-Text Search indexes and B-tree indexes without reloading data.

    Drops the HNSW embedding index before checkpointing to avoid
    DuckDB internal errors with duplicate keys in HNSW wrappers.
    If an HNSW index was present, report the command to regenerate it.
    """
    if not db_path.exists():
        logging.error("Database not found at %s. Run 'esdc fetch' first.", db_path)
        return

    start_time = time.monotonic()

    def _step(step: str) -> str:
        elapsed = time.monotonic() - start_time
        return f"[dim]Rebuilding indexes: {step} [elapsed {elapsed:.1f}s][/dim]"

    console.print("[bold]Rebuilding search indexes...[/bold]")
    logging.info("Rebuilding FTS indexes on %s", db_path)
    if allow_install:
        conn = get_duckdb_connection(db_path, read_only=False)
    else:
        conn = get_duckdb_connection(
            db_path, read_only=False, allow_install=False
        )
    hnsw_dropped = False
    try:
        hnsw_exists = (
            conn.execute(
                "SELECT 1 FROM duckdb_indexes() "
                "WHERE schema_name = 'main' AND index_name = 'idx_hnsw_embeddings'"
            ).fetchone()
            is not None
        )
        if hnsw_exists:
            with console.status(_step("dropping HNSW index")):
                conn.execute("DROP INDEX main.idx_hnsw_embeddings")
                hnsw_dropped = True

        with console.status(_step("building FTS indexes")):
            if allow_install:
                fts_results = _create_fts_indexes(conn)
            else:
                fts_results = _create_fts_indexes(conn, allow_install=False)

        with console.status(_step("creating B-tree indexes")):
            pass  # B-tree indexes already created inside _create_fts_indexes

        with console.status(_step("checkpointing")):
            conn.execute("CHECKPOINT")

        elapsed = time.monotonic() - start_time
        logging.info("FTS indexes rebuilt successfully.")

        fts_ok = [t for t, r in fts_results.items() if r == "ok"]
        skip_reasons = {"table not loaded", "skipped (0 rows)"}
        fts_failed = [
            t for t, r in fts_results.items() if r != "ok" and r not in skip_reasons
        ]
        existing_tables = {
            row[0]
            for row in conn.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema='main'"
            ).fetchall()
        }
        populated_tables = {
            table
            for table in ("project_resources", "project_timeseries")
            if table in existing_tables
            and (count := conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone())
            is not None
            and count[0] > 0
        }
        for table, result in fts_results.items():
            if result in skip_reasons:
                console.print(f"[dim]FTS index skipped: {table} — {result}[/dim]")

        if fts_ok and not fts_failed:
            console.print(
                f"[green]✓[/green] Search indexes rebuilt "
                f"({', '.join(fts_ok)}) in {elapsed:.1f}s"
            )
        elif fts_ok and fts_failed:
            for t in fts_failed:
                console.print(f"[red]✗[/red] FTS index: {t} — {fts_results[t]}")
        elif fts_failed:
            console.print(f"[red]✗[/red] FTS indexes failed in {elapsed:.1f}s")
        else:
            console.print("[dim]No populated tables available for FTS indexing.[/dim]")
        if strict:
            failures = {
                table: fts_results.get(table, "index result missing")
                for table in populated_tables
                if fts_results.get(table) != "ok"
            }
            if failures:
                raise RuntimeError(f"Indexing failed for populated tables: {failures}")
    except duckdb.Error as e:
        logging.error("Failed to rebuild FTS indexes: %s", e)
        console.print(f"[red]✗[/red] Failed to rebuild indexes: {e}")
        if strict:
            raise
    finally:
        if hnsw_dropped:
            console.print(
                "[yellow]:warning:  HNSW embedding index dropped "
                "(required for checkpoint). "
                "Run `[dim]esdc reload --embeddings-only[/dim]` "
                "to regenerate.[/yellow]"
            )
        conn.close()


def get_last_updated(conn: duckdb.DuckDBPyConnection) -> str | None:
    """Retrieve the last_updated timestamp from the _metadata table.

    Returns the ISO 8601 timestamp string of the last successful data fetch,
    or None if the _metadata table does not exist or has no entry.
    """
    try:
        result = conn.execute(
            "SELECT value FROM _metadata WHERE key = 'last_updated'"
        ).fetchone()
        return result[0] if result else None
    except duckdb.Error:
        return None


def load_data_to_db(
    content: list[list[str]], header: list[str], table_name: str
) -> None:
    """Load supported database rows through their canonical ingestion path."""
    if table_name == "project_resources":
        raise ValueError(
            "project_resources must be ingested through the canonical SQLite "
            "resource ingestion path"
        )
    if table_name != "project_timeseries":
        raise ValueError(f"Unsupported table: {table_name}")
    if not content:
        return

    from esdc.fetch_publication import ingest_timeseries_rows
    from esdc.pod_registry.store import get_esdc_sqlite_path

    ingest_timeseries_rows(
        content,
        header,
        duckdb_path=Config.get_db_file(),
        sqlite_path=get_esdc_sqlite_path(),
    )


def _stage_timeseries_slices(
    conn: duckdb.DuckDBPyConnection,
    slices: Sequence["PreparedTimeseriesSlice"],
) -> str | None:
    """Stage validated timeseries rows in a typed, operation-owned table."""
    if not slices:
        return None
    from esdc.fetching import (
        iter_spooled_rows,
        validate_timeseries_header,
    )

    typed_slices = tuple(slices)
    canonical_types = _timeseries_canonical_types()
    target_exists = conn.execute(
        "SELECT 1 FROM information_schema.tables "
        "WHERE table_schema='main' AND table_name='project_timeseries'"
    ).fetchone()
    target_types: dict[str, str] = {}
    if target_exists is not None:
        target_types = {
            row[0]: row[1]
            for row in conn.execute(
                "SELECT column_name, data_type FROM information_schema.columns "
                "WHERE table_schema='main' AND table_name='project_timeseries'"
            ).fetchall()
        }
    stage = f"project_timeseries_stage_{uuid.uuid4().hex}"
    union_columns = tuple(
        column
        for column in canonical_types
        if any(column in prepared.header for prepared in typed_slices)
    )
    definitions = ", ".join(
        f"{column} {target_types.get(column, canonical_types[column])}"
        for column in union_columns
    )
    conn.execute(f"CREATE TEMP TABLE {stage} ({definitions})")
    total_rows = 0
    for prepared in typed_slices:
        try:
            header = validate_timeseries_header(prepared.header)
            if not set(header) <= set(canonical_types):
                raise ValueError(
                    "project_timeseries header is outside the canonical schema"
                )
            if prepared.row_count <= 0:
                raise ValueError("timeseries staging accepts nonempty slices only")
            placeholders = ", ".join("?" for _ in header)
            header_sql = ", ".join(header)
            insert_sql = f"INSERT INTO {stage} ({header_sql}) VALUES ({placeholders})"
            count = 0
            rows = iter(iter_spooled_rows(prepared))
            while batch := list(itertools.islice(rows, 1000)):
                if any(
                    row[header.index("report_year")] != prepared.report_year
                    for row in batch
                ):
                    raise ValueError(
                        "project_timeseries staged row does not match year "
                        f"{prepared.report_year}"
                    )
                conn.executemany(insert_sql, batch)
                count += len(batch)
            if count != prepared.row_count:
                raise ValueError(
                    "project_timeseries staged "
                    f"{count} rows for year {prepared.report_year}; "
                    f"expected {prepared.row_count}"
                )
            total_rows += count
        except Exception as exc:
            from esdc.fetching import FetchFailure

            raise FetchFailure(
                f"project_timeseries staging failed: {exc}",
                phase="publication",
                table="project_timeseries",
                report_year=prepared.report_year,
            ) from exc
    staged_count = conn.execute(f"SELECT COUNT(*) FROM {stage}").fetchone()
    if staged_count is None:
        raise RuntimeError("project_timeseries staging count returned no row")
    actual_count = staged_count[0]
    if actual_count != total_rows:
        raise ValueError("project_timeseries staging row count mismatch")
    return stage


def _timeseries_canonical_columns() -> tuple[str, ...]:
    return tuple(_timeseries_canonical_types())


def _timeseries_canonical_types() -> dict[str, str]:
    source = _load_sql_script("create_table_project_timeseries.sql")
    start = source.index("CREATE TABLE IF NOT EXISTS project_timeseries")
    body_start = source.index("(\n", start) + 2
    body_end = source.index("\n);", body_start)
    body = source[body_start:body_end]
    result: dict[str, str] = {}
    for line in body.splitlines():
        column = line.split("--", 1)[0].strip().rstrip(",").strip()
        if not column:
            continue
        parts = column.split(None, 1)
        if len(parts) != 2 or parts[0] in {"id", "uuid"}:
            continue
        result[parts[0]] = parts[1]
    from esdc.fetching import TIMESERIES_COLUMNS

    if set(result) != TIMESERIES_COLUMNS:
        raise RuntimeError("timeseries canonical schema and validator columns differ")
    return result


def _apply_timeseries_stage(
    conn: duckdb.DuckDBPyConnection,
    stage: str,
    slices: Sequence["PreparedTimeseriesSlice"],
) -> None:
    """Apply staged year replacements inside the caller's transaction."""
    if not slices:
        return
    typed_slices = tuple(slices)
    exists = conn.execute(
        "SELECT 1 FROM information_schema.tables "
        "WHERE table_schema='main' AND table_name='project_timeseries'"
    ).fetchone()
    if exists is None:
        _execute_sql_script(conn, "create_table_project_timeseries.sql")
    columns = {
        row[0]
        for row in conn.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_schema='main' AND table_name='project_timeseries'"
        ).fetchall()
    }
    if "uuid" not in columns:
        conn.execute("ALTER TABLE project_timeseries ADD COLUMN uuid VARCHAR")
        columns.add("uuid")
    supplied = tuple(dict.fromkeys(c for s in typed_slices for c in s.header))
    missing = set(supplied) - columns
    if missing:
        raise ValueError(
            "project_timeseries target is missing supplied canonical columns: "
            + ", ".join(sorted(missing))
        )
    for prepared in typed_slices:
        conn.execute(
            "DELETE FROM project_timeseries WHERE report_year = ?",
            [prepared.report_year],
        )
        applicable = tuple(
            column for column in _timeseries_canonical_columns() if column in columns
        )
        insert_columns = ", ".join((*applicable, "uuid"))
        select_columns = ", ".join(
            column if column in supplied else f"NULL AS {column}"
            for column in applicable
        )
        conn.execute(
            f"INSERT INTO project_timeseries ({insert_columns}) "
            f"SELECT {select_columns}, gen_random_uuid()::VARCHAR FROM {stage} "
            "WHERE report_year = ?",
            [prepared.report_year],
        )
        result = conn.execute(
            "SELECT COUNT(*) FROM project_timeseries WHERE report_year = ?",
            [prepared.report_year],
        ).fetchone()
        if result is None:
            raise RuntimeError("project_timeseries apply count returned no row")
        count = result[0]
        if count != prepared.row_count:
            raise ValueError(
                "project_timeseries applied "
                f"{count} rows for year {prepared.report_year}; "
                f"expected at least {prepared.row_count}"
            )
    _execute_sql_script(conn, "create_timeseries_views.sql")
    _execute_sql_script(conn, "create_table_metadata.sql")
    indexes = _create_fts_indexes(
        conn,
        tables=("project_timeseries",),
        extensions_loaded=True,
    )
    result = indexes.get("project_timeseries")
    if result != "ok":
        raise RuntimeError(f"project_timeseries required index work failed: {result}")
    conn.execute(f"DROP TABLE {stage}")


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


def check_indexes(conn: duckdb.DuckDBPyConnection) -> dict:
    """Check the status of FTS, B-tree, and embedding indexes.

    Returns a dict with keys:
        fts_indexes: list of dicts with table, exists, column_count
        btree_indexes: list of dicts with name, table, exists
        embeddings: dict with table_exists, row_count, hnsw_exists
    """
    result: dict = {
        "fts_indexes": [],
        "btree_indexes": [],
        "embeddings": {
            "table_exists": False,
            "row_count": 0,
            "hnsw_exists": False,
        },
    }

    fts_tables = ["project_resources", "project_timeseries"]
    existing_tables = {
        row[0]
        for row in conn.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'main'"
        ).fetchall()
    }

    fts_schemas = {
        row[0]
        for row in conn.execute(
            "SELECT schema_name FROM information_schema.schemata "
            "WHERE schema_name LIKE 'fts_main_%'"
        ).fetchall()
    }

    for table in fts_tables:
        fts_schema = f"fts_main_{table}"
        col_count = 0
        if fts_schema in fts_schemas:
            try:
                cols = conn.execute(
                    f"SELECT column_name FROM information_schema.columns "
                    f"WHERE table_schema = '{fts_schema}' AND table_name = 'fields'"
                ).fetchall()
                col_count = len(cols)
            except duckdb.Error:
                col_count = 0
        result["fts_indexes"].append(
            {
                "table": table,
                "exists": fts_schema in fts_schemas,
                "column_count": col_count,
            }
        )

    btree_expected = [
        ("idx_project_resources_report_year", "project_resources"),
        ("idx_project_resources_agg", "project_resources"),
        ("idx_project_timeseries_report_year", "project_timeseries"),
        ("idx_project_timeseries_agg", "project_timeseries"),
    ]

    existing_indexes = {
        row[0]
        for row in conn.execute("SELECT index_name FROM duckdb_indexes()").fetchall()
    }

    for idx_name, table in btree_expected:
        result["btree_indexes"].append(
            {
                "name": idx_name,
                "table": table,
                "exists": idx_name in existing_indexes,
            }
        )

    if "project_embeddings" in existing_tables:
        result["embeddings"]["table_exists"] = True
        try:
            row = conn.execute("SELECT COUNT(*) FROM project_embeddings").fetchone()
            result["embeddings"]["row_count"] = (row or (0,))[0]
        except duckdb.Error:
            result["embeddings"]["row_count"] = 0

    result["embeddings"]["hnsw_exists"] = "idx_hnsw_embeddings" in existing_indexes

    return result


def check_table_stats(conn: duckdb.DuckDBPyConnection) -> list[dict]:
    """Check row counts per report_year for each data table.

    Returns list of dicts with keys:
        table: table name
        total: total row count
        years: list of (report_year, count) tuples
    """
    tables = ["project_resources", "project_timeseries"]
    result = []

    existing_tables = {
        row[0]
        for row in conn.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'main'"
        ).fetchall()
    }

    for table in tables:
        if table not in existing_tables:
            result.append({"table": table, "total": 0, "years": [], "exists": False})
            continue

        total = (conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone() or (0,))[0]
        rows = conn.execute(
            f"SELECT report_year, COUNT(*) FROM {table} "
            f"GROUP BY report_year ORDER BY report_year"
        ).fetchall()

        result.append(
            {
                "table": table,
                "total": total,
                "years": [(row[0], row[1]) for row in rows],
                "exists": True,
            }
        )

    return result


def verify_indexes(conn: duckdb.DuckDBPyConnection) -> dict:
    """Run functional verification on FTS, HNSW, and B-tree indexes.

    Returns a dict with keys:
        fts: list of dicts with table, functional, result_count
        hnsw: dict with functional, result_count
        btree: list of dicts with name, functional
    """
    result: dict = {
        "fts": [],
        "hnsw": {"functional": False, "result_count": 0},
        "btree": [],
    }

    existing_tables = {
        row[0]
        for row in conn.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'main'"
        ).fetchall()
    }

    fts_schemas = {
        row[0]
        for row in conn.execute(
            "SELECT schema_name FROM information_schema.schemata "
            "WHERE schema_name LIKE 'fts_main_%'"
        ).fetchall()
    }

    fts_tests = [
        ("project_resources", "duri"),
        ("project_timeseries", "duri"),
    ]
    for table, test_query in fts_tests:
        fts_schema = f"fts_main_{table}"
        if table not in existing_tables or fts_schema not in fts_schemas:
            result["fts"].append(
                {"table": table, "functional": False, "result_count": 0}
            )
            continue

        try:
            conn.execute("INSTALL fts")
            conn.execute("LOAD fts")
        except duckdb.Error:
            pass

        try:
            count = (
                conn.execute(
                    f"SELECT COUNT(*) FROM {table} "
                    f"WHERE {fts_schema}.match_bm25(uuid, '{test_query}') "
                    f"IS NOT NULL"
                ).fetchone()
                or (0,)
            )[0]
            result["fts"].append(
                {
                    "table": table,
                    "functional": count > 0,
                    "result_count": count,
                }
            )
        except duckdb.Error:
            result["fts"].append(
                {"table": table, "functional": False, "result_count": 0}
            )

    if "project_embeddings" in existing_tables:
        try:
            dim = (
                conn.execute(
                    "SELECT array_length(embedding) FROM project_embeddings LIMIT 1"
                ).fetchone()
                or (0,)
            )[0]

            import numpy as np

            test_vec = ",".join(str(v) for v in np.random.rand(dim))
            count = (
                conn.execute(
                    f"SELECT COUNT(*) FROM project_embeddings "
                    f"WHERE array_cosine_similarity(embedding, "
                    f"[{test_vec}]::FLOAT[{dim}]) > 0.5"
                ).fetchone()
                or (0,)
            )[0]
            result["hnsw"] = {"functional": True, "result_count": count}
        except (duckdb.Error, Exception):
            result["hnsw"] = {"functional": False, "result_count": 0}

    btree_expected = [
        ("idx_project_resources_report_year", "project_resources"),
        ("idx_project_resources_agg", "project_resources"),
        ("idx_project_timeseries_report_year", "project_timeseries"),
        ("idx_project_timeseries_agg", "project_timeseries"),
    ]

    existing_indexes = {
        row[0]
        for row in conn.execute("SELECT index_name FROM duckdb_indexes()").fetchall()
    }

    for idx_name, table in btree_expected:
        if table not in existing_tables:
            result["btree"].append({"name": idx_name, "functional": False})
            continue

        if idx_name not in existing_indexes:
            result["btree"].append({"name": idx_name, "functional": False})
            continue

        try:
            col = "report_year" if "report_year" in idx_name else "wk_id"
            conn.execute(
                f"SELECT COUNT(*) FROM {table} WHERE {col} = 2024 LIMIT 1"
            ).fetchone()
            result["btree"].append({"name": idx_name, "functional": True})
        except duckdb.Error:
            result["btree"].append({"name": idx_name, "functional": False})

    return result


def invalidate_sql_cache() -> None:
    """Clear SQL results in place so independent handles observe invalidation."""
    from esdc.chat.tools import reset_sql_cache

    failures: list[BaseException] = []
    try:
        reset_sql_cache()
    except BaseException as exc:
        failures.append(exc)
    cache_dir = Config.get_cache_dir() / "sql_results"
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        with diskcache.Cache(str(cache_dir)) as cache:
            cache.clear()
        logging.info("SQL cache invalidated: %s", cache_dir)
        _record_cache_invalidation(cache_dir)
    except BaseException as exc:
        failures.append(exc)
    if failures:
        interruption = next(
            (
                error
                for error in failures
                if isinstance(error, (KeyboardInterrupt, SystemExit))
            ),
            None,
        )
        if interruption is not None:
            interruption.args = (
                *interruption.args,
                "SQL cache invalidation also reported: "
                + "; ".join(
                    str(error) for error in failures if error is not interruption
                ),
            )
            raise interruption
        raise RuntimeError(
            "SQL cache invalidation failed: "
            + "; ".join(str(error) for error in failures)
        ) from failures[0]


def _record_cache_invalidation(cache_dir: Path) -> None:
    """Record the timestamp of cache invalidation.

    Stores timestamp in a sibling file so it survives cache directory deletion.
    """
    from esdc.configs import Config

    cache_root = Config.get_cache_dir()
    timestamp_file = cache_root / ".last_invalidated"
    now = datetime.datetime.now().isoformat()
    try:
        timestamp_file.write_text(now)
    except OSError as e:
        logging.warning("Could not record cache invalidation timestamp: %s", e)


def get_last_cache_invalidation() -> str | None:
    """Get the timestamp of the last cache invalidation.

    Returns:
        ISO timestamp string or None if never invalidated.
    """
    from esdc.configs import Config

    cache_root = Config.get_cache_dir()
    timestamp_file = cache_root / ".last_invalidated"
    if timestamp_file.exists():
        try:
            return timestamp_file.read_text().strip()
        except OSError:
            return None
    return None
