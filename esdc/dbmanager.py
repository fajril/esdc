import logging
import shutil
from pathlib import Path

import duckdb
import pandas as pd

from esdc.configs import Config
from esdc.db_security import SQLSanitizer, _load_sql_script
from esdc.selection import TableName


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
    logging.info("Connecting to the database.")
    _ensure_duckdb_database(Config.get_db_file())
    if not Config.get_db_dir().exists():
        Config.get_db_dir().mkdir(parents=True, exist_ok=True)
        logging.info("Database does not exist. Creating new database.")
        logging.debug("Database location: %s", Config.get_db_dir())
    conn = duckdb.connect(str(Config.get_db_file()))
    try:
        logging.debug("creating table %s in database", table_name)
        _execute_sql_script(conn, create_table_query[table_name])
        column_names = ", ".join(["?" for _ in header])
        insert_stmt = (
            f"INSERT INTO {table_name} ({', '.join(header)}) VALUES ({column_names})"
        )
        logging.debug("Inserting table data %s into the database.", table_name)
        try:
            conn.executemany(insert_stmt, content)
        except duckdb.Error as e:
            logging.debug("insert statement: %s", insert_stmt)
            raise duckdb.Error(str(e)) from e

        logging.debug("Creating uuid column for table %s", table_name)
        _execute_sql_script(
            conn,
            "create_column_uuid.sql",
            replacements={"{table_name}": table_name},
        )

        if table_name == "project_resources":
            logging.debug("Creating project_uuid column.")
            _execute_sql_script(conn, "create_project_resources_uuid.sql")

            logging.debug("Creating is_discovered column.")
            _execute_sql_script(conn, "create_project_resources_is_discovered.sql")

            logging.debug("Creating project_stage column.")
            _execute_sql_script(conn, "create_project_resources_project_stage.sql")

            logging.debug("Creating table view for field, working area, nkri.")
            _execute_sql_script(conn, "create_esdc_view.sql")

        if table_name == "project_timeseries":
            logging.debug("Creating timeseries views for field, wa, nkri.")
            _execute_sql_script(conn, "create_timeseries_views.sql")

        conn.execute("CHECKPOINT")
        logging.info("Table %s is loaded into database.", table_name)
    finally:
        conn.close()

    invalidate_sql_cache()


def run_query(
    table: TableName,
    where: str | None = None,
    like: str | None = None,
    year: int | None = None,
    output: int = 0,
    columns: str | list[str] = "",
) -> pd.DataFrame | None:
    """Execute a parameterized query on the specified table in the database.

    Args:
        table: The table to query.
        where: The column name for the WHERE clause. Defaults to None.
        like: The value for the LIKE clause. Defaults to None.
        year: The year to filter by. Defaults to None.
        output: The output format. Defaults to 0.
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
            table, where=where, like=like, year=year, output=output, columns=columns
        )
        conn = duckdb.connect(str(Config.get_db_file()), read_only=True)
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
