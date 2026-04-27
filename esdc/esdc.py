"""ESDC Data Management Module.

This module provides functionality for managing data
related to the ESDC (https://esdc.skkmigas.go.id).
It includes commands for fetching and displaying data from various resources,
as well as loading data into a SQLite database.
The module utilizes the Typer library for command-line interface (CLI) interactions
and Rich for enhanced logging and output formatting.

Key Features:
- Fetch data from the ESDC API in various formats (CSV, JSON, ZIP).
- Load data into a SQLite database.
- Display data from specific tables with filtering options.
- Save output data to files.

Dependencies:
- pandas: For data manipulation and storage.
- requests: For making HTTP requests to the ESDC API.
- rich: For enhanced terminal output and logging.
- pyyaml: For loading configuration from config.yaml.
- sqlite3: For database operations.

Commands:
- init: Initializes the application and fetches data.
- fetch: Downloads data from the ESDC API and saves it to a specified file type.
- reload: Reloads data from existing binary files into the database.
- show: Displays data from a specified table with optional filters.

Usage:
Run the module from the command line to access the available commands and options.
"""

import csv
import gzip
import io
import json
import logging
import os
import time
import warnings
from collections.abc import Iterable
from contextlib import closing
from datetime import date
from pathlib import Path
from typing import Annotated

import pandas as pd
import requests
import rich
import typer
from rich.logging import RichHandler
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)
from tabulate import tabulate

from esdc.commands.configs import configs_app  # noqa: E402
from esdc.configs import Config  # noqa: E402
from esdc.console import console  # noqa: E402
from esdc.dbmanager import (  # noqa: E402
    _ensure_duckdb_database,
    get_duckdb_connection,
    load_data_to_db,
    run_query,
)
from esdc.selection import ApiVer, FileType, TableName  # noqa: E402

TABLES: tuple[TableName, TableName] = (
    TableName.PROJECT_RESOURCES,
    TableName.PROJECT_TIMESERIES,
)

app = typer.Typer(no_args_is_help=False)
app.add_typer(configs_app, name="configs")


@app.callback()
def main(verbose: int = 0):
    """
    Main function to set up logging and log level.

    Args:
        verbose (int, optional): Verbosity level. Defaults to 0.

    Returns:
        None

    Notes:
        This function sets up the logging configuration using the RichHandler.
        The log level is set based on the verbosity level:
            - verbose >= 2: DEBUG
            - verbose == 1: INFO
            - verbose == 0: WARNING
    """
    Config.init_config()
    # Only configure logging if not already configured (e.g., by app.py)
    if not logging.root.handlers:
        handler = RichHandler(show_time=False)
        handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
        logging.root.addHandler(handler)
        logger = logging.getLogger()
        if verbose >= 2:
            logger.setLevel(logging.DEBUG)
        elif verbose == 1:
            logger.setLevel(logging.INFO)
        else:
            logger.setLevel(logging.WARNING)
        logger.info(
            "Log level set to %s", logging.getLevelName(logger.getEffectiveLevel())
        )


# ... (previous imports remain unchanged)


@app.command()
def fetch(
    filetype: str = typer.Option("json", help="Options: csv, json"),
    save: bool = typer.Option(
        False,
        "--save/--no-save",
        help="Save fetched data to ~/.esdc/ directory.",
    ),
    no_reload: bool = typer.Option(
        False,
        "--no-reload",
        help="Only download data, skip loading into database (implies --save).",
    ),
    no_reindex: bool = typer.Option(
        False,
        "--no-reindex",
        help=(
            "Skip rebuilding FTS and B-tree indexes after loading data. "
            "By default, indexes are rebuilt automatically so ILIKE text "
            "searches work correctly for the newly-fetched data."
        ),
    ),
    year: Annotated[
        list[int] | None,
        typer.Option(
            help=(
                "Specific report year(s) to fetch. "
                "Can specify multiple: --year 2024 --year 2025"
            )
        ),
    ] = None,
) -> None:
    """Fetch data from ESDC and optionally load into the database.

    By default, downloads data and loads it into the database, then
    rebuilds FTS/B-tree indexes so ILIKE text searches work correctly.
    Use --no-reindex to skip index rebuilding after loading.
    Use --no-reload to download and save data without loading into the database.
    Use --year to fetch and update specific year(s) for both resources and
    timeseries.
    """
    username, password = Config.get_credentials()

    if year:
        year = sorted(set(year))
        logging.info("Will fetch data for specific year(s): %s", year)

    should_save = save or no_reload
    should_reload = not no_reload
    should_reindex = not no_reindex

    if filetype == "csv":
        load_esdc_data(
            filetype=FileType.CSV,
            to_file=should_save,
            reload=should_reload,
            username=username,
            password=password,
            years=year,
            reindex=should_reindex,
        )
    elif filetype == "json":
        load_esdc_data(
            filetype=FileType.JSON,
            to_file=should_save,
            reload=should_reload,
            username=username,
            password=password,
            years=year,
            reindex=should_reindex,
        )
    else:
        logging.warning("File type %s is not available.", filetype)


@app.command()
def reload(
    filetype: Annotated[
        str | None, typer.Option(help="Options: csv, json, zip")
    ] = "csv",
    reindex_only: Annotated[
        bool,
        typer.Option(
            "--reindex-only",
            help="Rebuild FTS/B-tree indexes only, without reloading data.",
        ),
    ] = False,
    no_embeddings: Annotated[
        bool,
        typer.Option(
            "--no-embeddings",
            help="Skip semantic embeddings generation.",
        ),
    ] = False,
    embeddings_only: Annotated[
        bool,
        typer.Option(
            "--embeddings-only",
            help="Only regenerate embeddings, skip data reload.",
        ),
    ] = False,
) -> None:
    """Reload data from binary files and save it to a file.

    By default, also generates semantic embeddings for project_remarks.
    Use --no-embeddings to skip embedding generation.
    Use --embeddings-only to only regenerate embeddings without reloading data.

    Args:
        filetype: The type of file to save the data to. Defaults to "csv".
        reindex_only: If True,
        only rebuild FTS and B-tree indexes without reloading data.
        no_embeddings: If True, skip semantic embeddings generation.
        embeddings_only: If True, only regenerate embeddings without reloading data.

    Returns:
        None
    """
    # Handle embeddings-only mode
    if embeddings_only:
        _generate_embeddings()
        return

    # Handle reindex-only mode
    if reindex_only:
        from esdc.dbmanager import reindex_fts

        reindex_fts()
        return

    # Normal reload
    db_dir = Config.get_db_dir()
    for table in TABLES:
        filename = db_dir / f"{table.value}.{filetype}"
        if filename.exists():
            if filetype == "csv":
                _load_file_as_csv(str(filename), table.value)
            elif filetype == "json":
                _load_file_as_json(str(filename), table.value)
            else:
                logging.debug(
                    "failed to load %s. Unknown %s format", filename, filetype
                )
        else:
            logging.warning("File %s is not found.", filename)

    # Generate embeddings after reload (unless disabled)
    if not no_embeddings:
        _generate_embeddings()


def _generate_embeddings() -> None:
    """Generate semantic embeddings for project_remarks with progress bar."""
    from esdc.configs import Config
    from esdc.knowledge_graph.embedding_manager import EmbeddingManager
    from esdc.knowledge_graph.semantic_resolver import SemanticResolver

    logger = logging.getLogger(__name__)

    logger.info("Starting semantic embeddings generation process")
    console.print("[bold blue]Generating Semantic Embeddings[/bold blue]")

    db_path = Config.get_db_file()
    if not db_path.exists():
        logger.warning(
            f"Database not found at {db_path}, skipping embeddings generation"
        )
        console.print(
            f"[yellow]Warning: Database not found at {db_path}, skipping embeddings[/yellow]"  # noqa: E501
        )
        return

    # Check if Ollama is available
    embedding_manager = EmbeddingManager()
    logger.info(f"Initialized embedding manager with model: {embedding_manager.model}")

    if not embedding_manager.health_check():
        logger.warning("Ollama not available, cannot generate embeddings")
        console.print(
            "[yellow]Warning: Ollama not available, skipping embeddings generation[/yellow]"  # noqa: E501
        )
        console.print(
            "[dim]To generate embeddings later, run: esdc reload --embeddings-only[/dim]"  # noqa: E501
        )
        return

    logger.info(f"Ollama is available, model {embedding_manager.model} is loaded")
    resolver = SemanticResolver(db_path=db_path)

    try:
        # Drop existing embeddings table if it exists to ensure fresh start
        logger.debug("Dropping existing embeddings table if present")
        conn = resolver._get_connection()
        conn.execute("DROP TABLE IF EXISTS project_embeddings")
        conn.execute("DROP INDEX IF EXISTS idx_hnsw_embeddings")

        # Build table
        logger.debug("Creating embeddings table in DuckDB")
        console.print("Creating embeddings table...")
        resolver.build_embeddings_table()

        # Count total documents first (using resolver's connection)
        total_docs = resolver.count_documents_with_remarks("project_resources")

        if total_docs == 0:
            logger.info("No documents with project_remarks found")
            console.print("[yellow]No documents found to generate embeddings[/yellow]")
            return

        batch_size = Config.get_embedding_batch_size()
        total_batches = (total_docs + batch_size - 1) // batch_size

        logger.info(
            f"Starting embedding generation: {total_docs} documents, {total_batches} batches, batch_size={batch_size}"  # noqa: E501
        )

        # Create progress bar
        with Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=40),
            "[progress.percentage]{task.percentage:>3.0f}%",
            "•",
            TextColumn("[green]{task.completed}/{task.total} docs"),
            "•",
            TimeElapsedColumn(),
            "•",
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(
                f"Processing with {embedding_manager.model}", total=total_docs
            )

            # Progress callback function
            def update_progress(current: int, total: int) -> None:
                progress.update(task, completed=current)

            # Generate embeddings with progress tracking
            result = resolver.generate_and_store_embeddings(
                table_name="project_resources",
                batch_size=batch_size,
                progress_callback=update_progress,
            )

        if result["status"] == "success":
            logger.info(f"Successfully generated {result['count']} embeddings")
            console.print(
                f"[green]Success![/green] Generated {result['count']} embeddings"
            )
            console.print(
                "[dim]Semantic search is now available via 'semantic_search' tool[/dim]"
            )
        else:
            logger.warning(
                f"Embedding generation completed with warning: {result.get('message')}"
            )
            console.print(
                f"[yellow]Warning: {result.get('message', 'Unknown error')}[/yellow]"
            )

    except Exception as e:
        logger.error(f"Embedding generation failed: {e}", exc_info=True)
        console.print(f"[yellow]Warning: Embedding generation failed: {e}[/yellow]")
        console.print("[dim]Semantic search will not be available[/dim]")
    finally:
        resolver.close()


@app.command()
def show(
    table: Annotated[str, typer.Argument(help="Table name.")],
    where: Annotated[str | None, typer.Option(help="Column to search.")] = None,
    search: Annotated[str | None, typer.Option(help="Filter value")] = "",
    year: Annotated[
        list[int] | None,
        typer.Option(min=2019, help="Filter year value. Can specify multiple."),
    ] = None,
    detail: Annotated[
        list[str] | None,
        typer.Option(
            help="Detail level: reserves, resources, resources_risked, inplace, cumprod, rate, all. "  # noqa: E501
            "Can specify multiple. Defaults to resources.",
        ),
    ] = None,
    save: bool = typer.Option(
        False,
        "--save/--no-save",
        help="Save output to Excel file in current directory.",
    ),
    columns: Annotated[
        str, typer.Option(help="Select specific columns (space-separated)")
    ] = "",
):
    """Show data from a specific table.

    Args:
        table: The name of the table to show data from.
        where: The column to search. Defaults to None.
        search: A search keyword to apply to the selected column.
        year: Filter year value(s). Can specify multiple: --year 2024 --year 2025.
        detail: Detail level(s) to show. Defaults to 'resources'.
            Options: reserves, resources, resources_risked, inplace, cumprod, rate, all.
        save: Whether to save the output data to an Excel file.
        columns: A space-separated list of column(s) to select.

    Returns:
        None
    """
    if columns.strip():
        columns_splitted: list[str] | str = columns.split(" ")
    else:
        columns_splitted = ""

    years_list = year if year else None
    details_list = detail if detail else None

    df = run_query(
        table=TableName(table),
        where=where,
        like=search,
        years=years_list,
        details=details_list,
        columns=columns_splitted,
    )
    if df is not None:
        pd.options.display.float_format = "{:,.2f}".format
        formatted_df = df.map(lambda x: f"{x:<,.2f}" if isinstance(x, float) else x)
        formatted_table = tabulate(
            formatted_df.to_dict("records"),
            headers="keys",
            tablefmt="psql",
            showindex=False,
            stralign="right",
        )
        rich.print(formatted_table)
        if save:
            today = date.today().strftime("%Y%m%d")
            df.to_excel(
                f"view_{table}_{today}.xlsx", index=False, sheet_name="resources report"
            )
    else:
        logging.warning("Unable to show data. The query is none.")


def _detect_report_years(db_path: Path, min_year: int = 2020) -> list[int]:
    """Query available report_year from project_resources, starting from min_year."""
    if not db_path.exists():
        logging.warning(
            "Database not found at %s. Cannot detect report years.", db_path
        )
        return []

    try:
        conn = get_duckdb_connection(db_path, read_only=True)
        try:
            result = conn.execute(
                f"SELECT DISTINCT report_year FROM project_resources "
                f"WHERE report_year >= {min_year} ORDER BY report_year"
            ).fetchall()
        finally:
            conn.close()

        years = [row[0] for row in result if row[0] is not None]
        logging.info(
            "Detected report years (>= %d) for timeseries: %s", min_year, years
        )
        return years
    except Exception:
        logging.exception("Failed to detect report years from database.")
        return []


_CREATE_TABLE_SCRIPTS = {
    "project_resources": "create_table_project_resources.sql",
    "project_timeseries": "create_table_project_timeseries.sql",
}


def _append_to_table(
    table_name: str,
    header: list[str],
    content: list[list[str]],
    append_years: list[int],
) -> None:
    """Append data: delete existing rows for given years, then insert new rows."""
    start_time = time.monotonic()
    db_path = Config.get_db_file()
    _ensure_duckdb_database(db_path)
    if not Config.get_db_dir().exists():
        Config.get_db_dir().mkdir(parents=True, exist_ok=True)

    year_label = ", ".join(str(y) for y in append_years)

    def _status(step: str) -> str:
        elapsed = time.monotonic() - start_time
        return (
            f"[dim]{table_name} ({year_label}): {step} [elapsed {elapsed:.1f}s][/dim]"
        )

    with console.status(_status("preparing")) as status:
        conn = get_duckdb_connection(db_path)
        try:
            # Ensure table exists; only create if missing
            status.update(_status("creating schema"))
            schema_stmt = (
                "SELECT 1 FROM information_schema.tables "
                f"WHERE table_name = '{table_name}'"
            )
            table_exists = conn.execute(schema_stmt).fetchone()

            if not table_exists:
                status.update(_status("creating table"))
                from esdc.db_security import _load_sql_script

                script_name = _CREATE_TABLE_SCRIPTS.get(table_name)
                if script_name is None:
                    raise ValueError(
                        f"No create script mapped for table '{table_name}'"
                    )
                create_sql = _load_sql_script(script_name)
                statements = [s.strip() for s in create_sql.split(";") if s.strip()]
                for stmt in statements:
                    conn.execute(stmt)

            # Delete existing rows for the specified years
            status.update(_status("deleting old rows"))
            delete_stmt = (
                f"DELETE FROM {table_name} WHERE report_year IN ({year_label})"
            )
            conn.execute(delete_stmt)

            # Insert new rows
            status.update(_status(f"inserting {len(content):,} rows"))
            placeholders = ", ".join(["?" for _ in header])
            insert_stmt = (
                f"INSERT INTO {table_name} ({', '.join(header)}) "
                f"VALUES ({placeholders})"
            )
            conn.executemany(insert_stmt, content)

            # ------------------------------------------------------------------
            # Work around DuckDB HNSW checkpoint crash: drop the index before
            # CHECKPOINT.  The index will be rebuilt by
            # `esdc reload --embeddings-only` when embeddings are regenerated
            # for the updated data.
            # ------------------------------------------------------------------
            status.update(_status("checkpointing"))
            conn.execute("DROP INDEX IF EXISTS idx_hnsw_embeddings")
            conn.execute("CHECKPOINT")

            elapsed = time.monotonic() - start_time
            console.print(
                f"[green]✓[/green] {table_name} ({year_label}): "
                f"{len(content):,} rows updated in {elapsed:.1f}s"
            )
        finally:
            conn.close()


def _fetch_and_parse_table(
    table: TableName,
    filetype: FileType,
    to_file: bool,
    username: str,
    password: str,
    report_year: int | None = None,
) -> tuple[list[list[str]], list[str]] | None:
    """Download and parse a single table."""
    url = esdc_url_builder(
        table_name=table, file_type=filetype, report_year=report_year
    )
    logging.info("downloading from %s", url)
    data = esdc_downloader(url, username, password)
    if data is None:
        logging.warning("Failed to download %s data.", table.value)
        return None

    if to_file:
        filename = table.value
        if report_year is not None:
            filename = f"{table.value}_{report_year}"
        save_path = Config.get_db_dir() / f"{filename}.{filetype.value}"
        logging.debug("Save data as %s", save_path)
        with open(save_path, "wb") as f:
            _ = f.write(data)

    if filetype == FileType.CSV:
        decoded_data = data.decode("utf-8").splitlines()
        return _read_csv(decoded_data)
    elif filetype == FileType.JSON:
        parsed_json = json.loads(data)
        if not parsed_json:
            return None
        header = list(parsed_json[0].keys())
        content = [list(item.values()) for item in parsed_json]
        return content, header
    return None


def load_esdc_data(
    filetype: FileType = FileType.CSV,
    to_file: bool = True,
    reload: bool = True,
    username: str = "",
    password: str = "",
    years: list[int] | None = None,
    reindex: bool = True,
) -> None:
    """Download data from the ESDC API and optionally load into the database.

    Parameters
    ----------
    filetype : FileType
        The file type for the downloaded data. Currently supports "csv" and "json".
    to_file : bool
        Whether to save the downloaded data to a file. Defaults to True.
    reload : bool
        Whether to load the downloaded data into the database. Defaults to True.
        When False, data is only downloaded and saved (not loaded).
    username : str
        The username for authenticating with the ESDC API.
    password : str
        The password for authenticating with the ESDC API.
    years : list[int] | None
        Specific report year(s) to fetch. When provided both
        ``project_resources`` and ``project_timeseries`` are updated for
        those years using append mode.
    reindex : bool
        If True (default), rebuild FTS and B-tree indexes after loading data.
        This ensures ILIKE text searches work correctly for the newly-fetched data.
        Set to False to skip reindexing (e.g. via --no-reindex).
    """
    # ------------------------------------------------------------------
    # 1) Full-replace mode (default)
    # ------------------------------------------------------------------
    if years is None:
        # Resources: full replace
        resources_result = _fetch_and_parse_table(
            TableName.PROJECT_RESOURCES,
            filetype,
            to_file,
            username,
            password,
        )
        if resources_result is not None and reload:
            load_data_to_db(
                resources_result[0],
                resources_result[1],
                TableName.PROJECT_RESOURCES.value,
            )

        # Detect years from DB after resources loaded
        timeseries_years = _detect_report_years(Config.get_db_file(), min_year=2020)
        if not timeseries_years:
            logging.warning("No report years found for timeseries. Skipping.")
            return

        # Timeseries: full replace, fetching per year
        all_timeseries_content: list[list[str]] = []
        timeseries_header: list[str] = []
        for year in timeseries_years:
            result = _fetch_and_parse_table(
                TableName.PROJECT_TIMESERIES,
                filetype,
                to_file,
                username,
                password,
                report_year=year,
            )
            if result is not None:
                all_timeseries_content.extend(result[0])
                timeseries_header = result[1]

        if all_timeseries_content and reload:
            load_data_to_db(
                all_timeseries_content,
                timeseries_header,
                TableName.PROJECT_TIMESERIES.value,
            )

        # Full-replace mode: load_data_to_db already calls _create_fts_indexes,
        # but we still reindex here as a safety pass when flag is set.
        if reload and reindex:
            from esdc.dbmanager import reindex_fts

            reindex_fts()

        return

    # ------------------------------------------------------------------
    # 2) Per-year append mode (--year)
    # ------------------------------------------------------------------
    if reload:
        console.print(
            "[dim]Per-year mode: updating project_resources and "
            f"project_timeseries for years {years}[/dim]"
        )

    for year in sorted(set(years)):
        for table in TABLES:
            result = _fetch_and_parse_table(
                table,
                filetype,
                to_file,
                username,
                password,
                report_year=year,
            )
            if result is None or not reload:
                continue
            _append_to_table(table.value, result[1], result[0], [year])

    if reload and reindex:
        from esdc.dbmanager import reindex_fts

        reindex_fts()


def esdc_url_builder(
    table_name: TableName,
    api_ver: ApiVer = ApiVer.V2,
    verbose: int = 3,
    report_year: int | None = None,
    file_type: FileType = FileType.CSV,
) -> str:
    """
    Build an ESDC URL based on the provided parameters.

    Parameters
    ----------
    table_name : TableName
        The table name to query.
    api_ver : ApiVer, optional
        The API version to use. Defaults to ApiVer.V2.
    verbose : int, optional
        The verbosity level. Defaults to 3.
    report_year : int, optional
        The report year to filter by. Defaults to None.
    file_type : FileType, optional
        The file type to request. Defaults to FileType.CSV.

    Returns:
    -------
    str
        The constructed ESDC URL.

    Notes:
    -----
    The URL is constructed by concatenating the base ESDC URL, API version,
    table name, verbosity level, and file type.
    If a report year is provided, it is added as a query parameter.
    For example, the url for project_resources table is:
    https://esdc.skkmigas.go.id/api/v2/project-resources?verbose=3&output=csv
    """
    url = Config.get_api_url().rstrip("/") + api_ver.value
    tables = {
        TableName.PROJECT_RESOURCES: "project-resources",
        TableName.PROJECT_TIMESERIES: "project-timeseries",
    }
    url = f"{url}/{tables[table_name]}?verbose={verbose}"

    # TODO this is temporary fix since as of 2024-07-06
    # the API for time series does not support all year selection
    # remove this conditional if the API for project_timeseries is fixed.
    if report_year is not None:
        url = f"{url}&report-year={report_year}"
    url = f"{url}&output={file_type.value}"

    return url


def _load_file_as_csv(file: str, table_name):
    with open(file, "rb") as f:
        data = f.read()
    decoded_data = data.decode("utf-8").splitlines()
    content, header = _read_csv(decoded_data)
    load_data_to_db(content, header, table_name)


def _load_file_as_json(file: str, table_name):
    with open(file, "rb") as f:
        data = f.read()
    parsed_json = json.loads(data)
    if not parsed_json:
        return
    header = parsed_json[0].keys()
    content = [list(item.values()) for item in parsed_json]
    load_data_to_db(content, header, table_name)


def esdc_downloader(url: str, username: str = "", password: str = "") -> bytes | None:
    """
    Download a file from a URL using the requests library and return its content.

    Parameters
    ----------
    url : str
        The URL of the file to download.

    Returns:
    -------
    bytes | None
        The content of the downloaded file as bytes, or None if the download failed.

    Raises:
    ------
    requests.exceptions.RequestException
        If there is a request error while downloading the file.

    """
    try:
        logging.info("requesting data to server...")
        logging.debug(url)
        verify_ssl = Config.get_verify_ssl()
        if not verify_ssl:
            warnings.warn(
                "SSL certificate verification is disabled. "
                "Set api.verify_ssl: true in config.yaml or ESDC_VERIFY_SSL=true "
                "to enable certificate verification.",
                stacklevel=2,
            )
            from urllib3.exceptions import InsecureRequestWarning

            warnings.filterwarnings("ignore", category=InsecureRequestWarning)

        response = requests.get(
            url, auth=(username, password), stream=True, timeout=300, verify=verify_ssl
        )

        if response.status_code == 200:
            file_size = int(response.headers.get("Content-Length", 0))
            logging.debug("File size is %s bytes", file_size)
            logging.debug(
                "Encoding format: %s", response.headers.get("Content-Encoding")
            )

            with (
                closing(io.BytesIO()) as f,
                Progress(
                    *Progress.get_default_columns(),
                    TransferSpeedColumn(),
                    DownloadColumn(binary_units=True),
                ) as progress,
            ):
                if file_size > 0:
                    task_id = progress.add_task(
                        f"[cyan]Downloading {round(file_size / 1e6)} MB...",
                        total=file_size,
                        unit="B",
                        transfer=True,
                        speed_unit="B/s",
                    )
                else:
                    task_id = progress.add_task(
                        "[cyan]Downloading...",
                        total=None,
                        unit="B",
                        transfer=True,
                        speed_unit="B/s",
                    )
                if response.headers.get("Content-Encoding") == "gzip":
                    with gzip.GzipFile(fileobj=response.raw, mode="rb") as gz:
                        while True:
                            chunk = gz.read(size=8192)
                            if not chunk:
                                break
                            _ = f.write(chunk)
                            _ = progress.update(task_id, advance=len(chunk))
                else:
                    for chunk in response.iter_content(chunk_size=8192):
                        _ = f.write(chunk)
                        _ = progress.update(task_id, advance=len(chunk))

                logging.info(
                    "File downloaded successfully to memory (Size: %s bytes)",
                    f.tell(),
                )
                return f.getvalue()

        logging.warning("File download failed. Status code: %s", response.status_code)
        return None
    except requests.exceptions.RequestException as e:
        logging.error("Request error while downloading the file. Error: %s", str(e))
        return None


def _read_csv(file: str | Iterable[str]) -> tuple[list[list[str]], list[str]]:
    """Reads a CSV file and returns its contents as a tuple.

    Returns a tuple of (data, header) where data is a list of lists of strings
    and header is a list of strings.

    Args:
        file: The path to the CSV file as a string,
        or an iterable of strings representing the CSV data.

    Returns:
        tuple[list[list[str]], list[str]]: A tuple containing the data
        and header of the CSV file.
    """

    class _EsdcDialect(csv.Dialect):
        delimiter = ";"
        quotechar = '"'
        doublequote = True
        lineterminator = "\n"
        quoting = csv.QUOTE_MINIMAL

    csv.register_dialect("esdc", _EsdcDialect)

    if isinstance(file, str):
        with open(file, newline="", encoding="utf-8") as csvfile:
            reader = csv.reader(csvfile, dialect="esdc")
    else:
        reader = csv.reader(file, dialect="esdc")
    header = next(reader)
    data = []
    for row in reader:
        data.append(row)
    return data, header


@app.command(name="chat")
def chat(setup: bool = False):
    """Start the interactive chat TUI."""
    from esdc.configs import Config

    if setup or not Config.has_chat_config():
        rich.print(
            "[bold yellow]No provider configured.[/bold yellow] "
            "Run '[cyan]esdc configs[/cyan]' to set one up."
        )
        return

    if Config.has_chat_config():
        from esdc.chat.app import ESDCChatApp

        app = ESDCChatApp()
        app.run()
    else:
        rich.print("[yellow]Setup incomplete. Chat cannot start.[/yellow]")


@app.command(name="status")
def status() -> None:
    """Show database location, configuration, and index status."""
    db_dir = Config.get_db_dir()
    db_file = Config.get_db_file()

    rich.print(f"[bold]Database directory:[/bold] {db_dir}")
    rich.print(f"[bold]Database file:[/bold] {db_file}")

    if os.environ.get("ESDC_DB_DIR"):
        rich.print("  (from [cyan]ESDC_DB_DIR[/cyan] environment variable)")
    if os.environ.get("ESDC_DB_FILE"):
        rich.print("  (from [cyan]ESDC_DB_FILE[/cyan] environment variable)")

    if not db_file.exists():
        rich.print(
            "[yellow]Database exists: No[/yellow] "
            "(run '[cyan]esdc fetch --save[/cyan]' to create)"
        )
        return

    rich.print("[green]Database exists: Yes[/green]")

    try:
        from esdc.dbmanager import check_indexes, get_duckdb_connection

        conn = get_duckdb_connection(db_file)
        try:
            status = check_indexes(conn)
        finally:
            conn.close()
    except Exception as e:
        rich.print(f"[yellow]Could not check indexes: {e}[/yellow]")
        return

    rich.print()
    rich.print("[bold]FTS Indexes:[/bold]")
    for fts in status["fts_indexes"]:
        icon = "[green]✅[/green]" if fts["exists"] else "[red]❌[/red]"
        detail = ""
        if fts["exists"]:
            detail = f" ({fts['column_count']} columns)"
        rich.print(f"  {icon} FTS {fts['table']}{detail}")

    rich.print()
    rich.print("[bold]B-tree Indexes:[/bold]")
    for bt in status["btree_indexes"]:
        icon = "[green]✅[/green]" if bt["exists"] else "[red]❌[/red]"
        rich.print(f"  {icon} {bt['name']}")

    rich.print()
    rich.print("[bold]Embeddings:[/bold]")
    emb = status["embeddings"]
    table_icon = "[green]✅[/green]" if emb["table_exists"] else "[red]❌[/red]"
    rich.print(f"  {table_icon} project_embeddings table", end="")
    if emb["table_exists"]:
        rich.print(f" ({emb['row_count']:,} rows)")
    else:
        rich.print()
    hnsw_icon = "[green]✅[/green]" if emb["hnsw_exists"] else "[red]❌[/red]"
    rich.print(f"  {hnsw_icon} HNSW index idx_hnsw_embeddings")


@app.command(name="serve")
def serve(
    web: bool = typer.Option(True, "--web", help="Run web server"),
    port: int = typer.Option(3334, "--port", "-p", help="Server port"),
    host: str = typer.Option("0.0.0.0", "--host", "-h", help="Server host"),
    log_level: str = typer.Option("info", "--log-level", help="Uvicorn log level"),
) -> None:
    """Start OpenAI-compatible API server.

    This command starts a web server that provides an OpenAI-compatible API
    for the ESDC agent. This allows tools like OpenWebUI to connect to ESDC
    as an external provider.

    Args:
        web: Whether to run the web server (default: True)
        port: Port to run the server on (default: 3334)
        host: Host to bind the server to (default: 0.0.0.0)
        log_level: Uvicorn log level (default: info)
    """
    from esdc.server.app import run_server

    if web:
        rich.print(
            f"[bold green]Starting ESDC server on http://{host}:{port}[/bold green]"
        )
        rich.print(
            f"[dim]API documentation available at http://{host}:{port}/docs[/dim]"
        )
        run_server(host=host, port=port, log_level=log_level)


@app.command(name="load-kg")
def load_kg() -> None:
    """Build the knowledge graph from the ESDC database.

    Creates a LadybugDB graph database with nodes and relationships
    for fields, working areas, projects, operators, and reports.
    Uses zero-copy ATTACH to DuckDB for data loading.
    """
    from esdc.knowledge_graph.ladybug_manager import LadybugDBManager

    db_file = Config.get_db_file()
    if not db_file.exists():
        rich.print("[red]Database not found. Run 'esdc fetch --save' first.[/red]")
        return

    rich.print("[bold cyan]Building knowledge graph...[/bold cyan]")
    manager = LadybugDBManager()
    if manager.build_graph(db_file):
        schema_info = manager.get_schema_info()
        table_count = len(schema_info.get("tables", []))
        rich.print(
            f"[green]Knowledge graph built successfully![/green] ({table_count} tables)"
        )
    else:
        rich.print(
            "[red]Failed to build knowledge graph. Check logs for details.[/red]"
        )
    manager.close()


if __name__ == "__main__":
    app()
