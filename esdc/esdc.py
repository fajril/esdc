"""ESDC Data Management Module.

This module provides functionality for managing data
related to the ESDC (https://esdc.skkmigas.go.id).
It includes commands for fetching and displaying data from various resources,
as well as loading data into a DuckDB database.
The module utilizes the Typer library for command-line interface (CLI) interactions
and Rich for enhanced logging and output formatting.

Key Features:
- Fetch data from the ESDC API in various formats (CSV, JSON, ZIP).
- Load data into a DuckDB database.
- Display data from specific tables with filtering options.
- Save output data to files.

Dependencies:
- pandas: For data manipulation and storage.
- requests: For making HTTP requests to the ESDC API.
- rich: For enhanced terminal output and logging.
- pyyaml: For loading configuration from config.yaml.
- duckdb: For database operations.

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
from typing import Annotated, Any

import pandas as pd
import requests
import rich
import typer
from rich.logging import RichHandler
from rich.panel import Panel
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
    _execute_sql_script,
    get_duckdb_connection,
    load_data_to_db,
    run_query,
)
from esdc.loaders import (  # noqa: E402
    LoadSchemaError,
    SpreadsheetLoadError,
    copy_pod_schema_template,
    generate_schema_template_from_excel,
    load_excel_to_duckdb,
    load_pod_workbook_to_duckdb,
    print_load_result,
)
from esdc.selection import ApiVer, FileType, Severity, TableName  # noqa: E402
from esdc.summarizer import (  # noqa: E402
    SummaryDependencyError,
    SummaryLookupError,
    _strategic_summary_text,
    get_resource_summary,
    summarize_resources,
)
from esdc.validate import ValidationResult, run_validation

TABLES: tuple[TableName, TableName] = (
    TableName.PROJECT_RESOURCES,
    TableName.PROJECT_TIMESERIES,
)

app = typer.Typer(no_args_is_help=False)
schema_app = typer.Typer(invoke_without_command=True, no_args_is_help=True)
corpus_app = typer.Typer(no_args_is_help=True)
app.add_typer(configs_app, name="configs")
app.add_typer(schema_app, name="schema")
app.add_typer(
    corpus_app,
    name="corpus",
    help="Ingest official PDF documents (surat, MoM) for iris document search.",
)


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


@schema_app.callback()
def schema_command(
    generate: Annotated[
        bool,
        typer.Option(
            "--generate",
            help="Generate a starter YAML schema from an Excel .xlsx file.",
        ),
    ] = False,
    from_excel: Annotated[
        Path | None,
        typer.Option(
            "--from-excel",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            help="Excel .xlsx file to inspect.",
        ),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            file_okay=True,
            dir_okay=False,
            writable=True,
            help=(
                "Schema YAML output path. Defaults to "
                "./<excel-stem>.schema.yaml."
            ),
        ),
    ] = None,
    overwrite: Annotated[
        bool,
        typer.Option(
            "--overwrite",
            help="Replace the output schema file if it already exists.",
        ),
    ] = False,
    schema_pod: Annotated[
        bool,
        typer.Option(
            "--schema-pod",
            help="Generate the built-in POD workbook template.",
        ),
    ] = False,
) -> None:
    """Generate and inspect spreadsheet schemas and POD workbook templates."""
    if schema_pod:
        try:
            destination = copy_pod_schema_template(
                output_path=output,
                overwrite=overwrite,
            )
        except (LoadSchemaError, SpreadsheetLoadError) as e:
            typer.echo(f"Error: {e}")
            raise typer.Exit(1) from None
        typer.echo(f"Generated POD workbook template at {destination}")
        return

    if not generate:
        typer.echo("Error: specify --generate or --schema-pod.")
        raise typer.Exit(1) from None
    if from_excel is None:
        typer.echo("Error: --from-excel is required when using --generate.")
        raise typer.Exit(1) from None
    try:
        result = generate_schema_template_from_excel(
            from_excel,
            output_path=output,
            overwrite=overwrite,
        )
    except (LoadSchemaError, SpreadsheetLoadError) as e:
        typer.echo(f"Error: {e}")
        raise typer.Exit(1) from None

    typer.echo(
        f"Generated schema template for '{result.table_name}' "
        f"from sheet '{result.sheet_name}' with {result.column_count:,} columns"
    )
    typer.echo(f"Schema: {result.output_path}")


@app.command()
def load(
    from_excel: Annotated[
        Path,
        typer.Option(
            "--from-excel",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            help="Excel .xlsx file to load.",
        ),
    ],
    schema: Annotated[
        Path | None,
        typer.Option(
            "--schema",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            help="YAML schema/data dictionary for the Excel table.",
        ),
    ] = None,
    schema_pod: Annotated[
        bool,
        typer.Option(
            "--schema-pod",
            help="Use the built-in POD schema template.",
        ),
    ] = False,
) -> None:
    """Load a spreadsheet into DuckDB and register its data dictionary.

    Run `esdc schema --generate --from-excel data.xlsx` to create a starter schema.
    Use `--schema-pod` to load POD workbook data with the built-in POD template.
    """
    if (schema is None) == (not schema_pod):
        typer.echo("Error: specify exactly one of --schema or --schema-pod.")
        raise typer.Exit(1) from None
    try:
        if schema_pod:
            results = load_pod_workbook_to_duckdb(from_excel)
            for result in results:
                print_load_result(result)
            return
        schema_path = schema
        assert schema_path is not None
        result = load_excel_to_duckdb(from_excel, schema_path)
    except (LoadSchemaError, SpreadsheetLoadError) as e:
        typer.echo(f"Error: {e}")
        raise typer.Exit(1) from None
    print_load_result(result)


@app.command()
def summarize(
    target: Annotated[
        str,
        typer.Argument(help="Target to summarize: all, field, wk, or nkri."),
    ],
    name: Annotated[
        str | None,
        typer.Argument(help="Field or working area name. Omit for all/nkri."),
    ] = None,
    year: Annotated[
        int | None,
        typer.Option(
            "--year",
            help="Report year to summarize.",
        ),
    ] = None,
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            help="Regenerate summaries even when the source hash is unchanged.",
        ),
    ] = False,
    retry: Annotated[
        int,
        typer.Option(
            "--retry",
            help="Max LLM attempts (1 = no retry, 2 = one retry, etc). Default 1.",
        ),
    ] = 1,
    from_wk: Annotated[
        str | None,
        typer.Option(
            "--from-wk",
            help="Only summarize fields belonging to this working area.",
        ),
    ] = None,
) -> None:
    """Generate LLM executive summaries for Eureka resource dashboards."""
    if year is None:
        typer.echo("Error: --year is required.")
        raise typer.Exit(1) from None

    try:
        result = summarize_resources(
            year=year,
            target=target,
            name=name,
            force=force,
            retry=retry,
            from_wk=from_wk,
        )
    except (FileNotFoundError, ValueError, SummaryDependencyError) as e:
        typer.echo(f"Error: {e}")
        raise typer.Exit(1) from None

    typer.echo(
        "Summary complete: "
        f"fields {result.fields_created} created/{result.fields_skipped} skipped; "
        f"working areas {result.working_areas_created} created/"
        f"{result.working_areas_skipped} skipped; "
        f"nkri {result.nkri_created} created/{result.nkri_skipped} skipped; "
        f"tokens {result.total_tokens_processed:,} processed"
    )


@app.command()
def summary(
    level: Annotated[
        str,
        typer.Argument(help="Summary level: field, wk, working_area, or nkri."),
    ],
    name: Annotated[
        str | None,
        typer.Argument(help="Field or working area name. Omit for nkri."),
    ] = None,
    year: Annotated[
        int | None,
        typer.Option(
            "--year",
            help="Report year to show.",
        ),
    ] = None,
    json_output: Annotated[
        bool,
        typer.Option(
            "--json",
            help="Print the raw summary JSON payload.",
        ),
    ] = False,
) -> None:
    """Show a generated executive summary."""
    if year is None:
        typer.echo("Error: --year is required.")
        raise typer.Exit(1) from None

    try:
        data = get_resource_summary(level=level, year=year, name=name)
    except (FileNotFoundError, SummaryLookupError, ValueError) as e:
        typer.echo(f"Error: {e}")
        raise typer.Exit(1) from None

    if json_output:
        typer.echo(json.dumps(data, ensure_ascii=False, indent=2))
        return

    rich.print(
        Panel(
            _format_summary_for_cli(data),
            title=f"{data['entity_name']} · {data['report_year']}",
            subtitle=f"{data['entity_level']} summary",
            border_style="cyan",
        )
    )


def _format_summary_for_cli(data: dict) -> str:
    summary_data = data.get("summary") or {}
    lines: list[str] = []

    if data.get("source_level") == "strategic_analysis":
        summary_text = _strategic_summary_text(summary_data).strip()
        if summary_text:
            lines.append(summary_text)
        else:
            strategic_summary = summary_data.get("summary") or {}
            rendered_findings = _render_summary_value(
                strategic_summary.get("key_findings")
            )
            rendered_recommendations = _render_summary_value(
                strategic_summary.get("recommendations")
            )
            if rendered_findings:
                lines.extend(["[bold cyan]Key Findings[/bold cyan]", rendered_findings])
            if rendered_recommendations:
                lines.extend(
                    [
                        "",
                        "[bold cyan]Recommendations[/bold cyan]",
                        rendered_recommendations,
                    ]
                )
        meta_parts = [
            f"source={data.get('source_level') or '-'}",
            f"provider={data.get('provider') or '-'}",
            f"model={data.get('model') or '-'}",
            f"generated={data.get('generated_at') or '-'}",
        ]
        lines.extend(["", f"[dim]{' · '.join(meta_parts)}[/dim]"])
        return "\n".join(lines)

    headline = str(summary_data.get("headline") or "").strip()
    if headline:
        lines.append(f"[bold]{headline}[/bold]")

    executive = str(summary_data.get("executive_summary") or "").strip()
    if executive:
        lines.extend(["", executive])

    sections = [
        ("Current Situation", summary_data.get("current_situation")),
        ("Key Challenges", summary_data.get("key_challenges")),
        ("Solution Proposals", summary_data.get("solution_proposals")),
        (
            "Production / Reserve Opportunities",
            summary_data.get("production_or_reserve_opportunities"),
        ),
        ("Management Attention", summary_data.get("management_attention")),
        ("Data Quality Notes", summary_data.get("data_quality_notes")),
    ]
    for title, value in sections:
        rendered = _render_summary_value(value)
        if rendered:
            lines.extend(["", f"[bold cyan]{title}[/bold cyan]", rendered])

    meta_parts = [
        f"source={data.get('source_level') or '-'}",
        f"provider={data.get('provider') or '-'}",
        f"model={data.get('model') or '-'}",
        f"generated={data.get('generated_at') or '-'}",
    ]
    lines.extend(["", f"[dim]{' · '.join(meta_parts)}[/dim]"])
    return "\n".join(lines)


def _render_summary_value(value) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        items = [str(item).strip() for item in value if str(item).strip()]
        return "\n".join(f"• {item}" for item in items)
    text = str(value).strip()
    return text


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
    from esdc.search.embedding_manager import EmbeddingManager
    from esdc.search.semantic_resolver import SemanticResolver

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

            status.update(_status("recording metadata"))
            _execute_sql_script(conn, "create_table_metadata.sql")

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


def _print_cache_subsection(name: str, stats: dict[str, Any]) -> None:
    """Print a cache subsection in status output."""
    rich.print()
    rich.print(f"  [bold]{name}[/bold]")
    rich.print(f"      Path: {stats['directory']}")
    rich.print(
        f"      Entries: {stats['entries']:,} | "
        f"Size: {_humanize_bytes(stats['size_bytes'])} / "
        f"{_humanize_bytes(stats['size_limit'])}"
    )
    _print_hit_rate(stats.get("hits", 0), stats.get("misses", 0))


def _print_hit_rate(hits: int, misses: int) -> None:
    """Print hit rate line with color coding."""
    total = hits + misses
    if total == 0:
        rich.print("      Hits: 0 | Misses: 0 | Hit rate: N/A (no activity)")
        return
    rate = hits / total
    rate_color = "green" if rate >= 0.8 else "yellow" if rate >= 0.5 else "red"
    rich.print(
        f"      Hits: {hits:,} | Misses: {misses:,} | "
        f"Hit rate: [{rate_color}]{rate:.1%}[/{rate_color}]"
    )


def _humanize_bytes(n: int) -> str:
    """Convert bytes to human-readable string."""
    n = max(n, 0)
    if n < 1024:
        return f"{n} B"
    if n < 1024**2:
        return f"{n / 1024:.1f} KB"
    if n < 1024**3:
        return f"{n / (1024 ** 2):.1f} MB"
    return f"{n / (1024 ** 3):.1f} GB"


def _status_fetch_report() -> bool:
    """Print DB location/env vars, tables + per-year rows, last updated.

    Returns:
        True if the database file exists and could be read (report
        printed in full); False if a short "Database exists: No" or
        error message was printed instead and callers should stop
        further status reporting (mirrors the pre-split monolith, which
        returned immediately in both cases).
    """
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
        return False

    rich.print("[green]Database exists: Yes[/green]")

    try:
        from esdc.dbmanager import (
            check_table_stats,
            get_duckdb_connection,
            get_last_updated,
        )

        conn = get_duckdb_connection(db_file)
        try:
            table_stats = check_table_stats(conn)
            last_updated = get_last_updated(conn)
        finally:
            conn.close()
    except Exception as e:
        rich.print(f"[yellow]Could not check indexes: {e}[/yellow]")
        return False

    if last_updated:
        rich.print(f"[bold]Last updated:[/bold] {last_updated}")
    else:
        from datetime import datetime

        mtime = datetime.fromtimestamp(db_file.stat().st_mtime)
        rich.print(f"[bold]Last updated:[/bold] {mtime:%Y-%m-%d %H:%M:%S} (file mtime)")

    rich.print()
    rich.print("[bold]Tables:[/bold]")
    for ts in table_stats:
        if not ts["years"]:
            icon = "[red]❌[/red]"
            rich.print(f"  {icon} {ts['table']}: not loaded")
            continue
        icon = "[green]✅[/green]"
        rich.print(f"  {icon} {ts['table']} ({ts['total']:,} rows):")
        for year, count in ts["years"]:
            rich.print(f"      {year}: {count:,} rows")

    return True


def _status_index_report(verify: bool) -> None:
    """Print FTS, B-tree, and embeddings/HNSW index status.

    Args:
        verify: When True, additionally runs functional verification on
            the indexes (slower) and prints a Verification section.
    """
    from esdc.dbmanager import check_indexes, get_duckdb_connection

    db_file = Config.get_db_file()

    if not db_file.exists():
        rich.print(
            "[yellow]Database exists: No[/yellow] "
            "(run '[cyan]esdc fetch --save[/cyan]' to create)"
        )
        return

    try:
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

    if not verify:
        return

    rich.print()
    rich.print("[bold]Verification:[/bold]")

    try:
        from esdc.dbmanager import verify_indexes

        conn = get_duckdb_connection(db_file)
        try:
            vf = verify_indexes(conn)
        finally:
            conn.close()
    except Exception as e:
        rich.print(f"[yellow]  Could not verify indexes: {e}[/yellow]")
        return

    for fts in vf["fts"]:
        if fts["functional"]:
            icon = "[green]✅[/green]"
            detail = f"functional ({fts['result_count']} results)"
        else:
            icon = "[red]❌[/red]"
            detail = "not functional"
        rich.print(f"  {icon} FTS {fts['table']}: {detail}")

    if vf["hnsw"]["functional"]:
        icon = "[green]✅[/green]"
        detail = f"functional ({vf['hnsw']['result_count']} results)"
    else:
        icon = "[red]❌[/red]"
        detail = "not functional"
    rich.print(f"  {icon} HNSW search: {detail}")

    for bt in vf["btree"]:
        if bt["functional"]:
            icon = "[green]✅[/green]"
            detail = "query OK"
        else:
            icon = "[yellow]⚠[/yellow]"
            detail = "not verified (compound index)"
        rich.print(f"  {icon} {bt['name']}: {detail}")


def _status_cache_report() -> None:
    """Print SQL/tool/JSON cache diagnostics and last invalidation time."""
    rich.print()
    rich.print("[bold]Cache:[/bold]")
    cache_dir = Config.get_cache_dir()
    rich.print(f"  Directory: {cache_dir}")

    try:
        from esdc.chat.tools import get_sql_cache_stats, get_tool_cache_stats
        from esdc.dbmanager import get_last_cache_invalidation
        from esdc.server.cache import get_cache_stats

        sql_stats = get_sql_cache_stats()
        tool_stats = get_tool_cache_stats()
        json_stats = get_cache_stats()

        _print_cache_subsection("SQL Results Cache", sql_stats)
        _print_cache_subsection("Tool Results Cache", tool_stats)

        # JSON cache
        rich.print()
        rich.print("  [bold]JSON Parsing Cache (RAM):[/bold]")
        rich.print(
            f"      Entries: {json_stats['json_cache_size']} /"
            f" {json_stats['json_cache_max']}"
        )
        _print_hit_rate(
            json_stats.get("json_cache_hits", 0),
            json_stats.get("json_cache_misses", 0),
        )
        rich.print("      Note: In-memory only, resets on restart")

        # Last invalidated
        last_invalidated = get_last_cache_invalidation()
        if last_invalidated:
            rich.print()
            rich.print(f"  [bold]Last cache invalidated:[/bold] {last_invalidated}")
    except Exception as e:
        rich.print(f"[yellow]  Could not check cache: {e}[/yellow]")


_CORPUS_TABLES = ("documents", "document_chunks", "corpus_meta")


def _status_corpus_report() -> None:
    """Print the corpus store section.

    Doc/chunk counts, doc_type breakdown, pinned embedding model/dim,
    and chunks FTS + HNSW index status.

    Read-only and never instantiates CorpusStore (that would drag in the
    embedder / require Ollama to be running). Connects directly via
    get_duckdb_connection and guards every query with an
    information_schema check, mirroring how dbmanager.check_indexes
    inspects duckdb_indexes().
    """
    from esdc.dbmanager import get_duckdb_connection

    db_file = Config.get_db_file()

    if not db_file.exists():
        rich.print(
            "[yellow]Database exists: No[/yellow] "
            "(run '[cyan]esdc fetch --save[/cyan]' to create)"
        )
        return

    try:
        conn = get_duckdb_connection(db_file)
        try:
            existing_tables = {
                row[0]
                for row in conn.execute(
                    "SELECT table_name FROM information_schema.tables "
                    "WHERE table_schema = 'main'"
                ).fetchall()
            }

            if not set(_CORPUS_TABLES) <= existing_tables:
                rich.print("[bold]Corpus:[/bold]")
                rich.print(
                    "  [yellow]not initialized[/yellow] "
                    "(run '[cyan]esdc corpus commit[/cyan]')"
                )
                return

            doc_count = (
                conn.execute("SELECT COUNT(*) FROM documents").fetchone() or (0,)
            )[0]
            chunk_count = (
                conn.execute("SELECT COUNT(*) FROM document_chunks").fetchone()
                or (0,)
            )[0]
            doc_type_rows = conn.execute(
                "SELECT COALESCE(doc_type, 'unknown'), COUNT(*) FROM documents "
                "GROUP BY 1 ORDER BY 1"
            ).fetchall()
            meta_row = conn.execute(
                "SELECT embedding_model, dim FROM corpus_meta LIMIT 1"
            ).fetchone()

            fts_schemas = {
                row[0]
                for row in conn.execute(
                    "SELECT schema_name FROM information_schema.schemata "
                    "WHERE schema_name LIKE 'fts_main_%'"
                ).fetchall()
            }
            fts_exists = "fts_main_document_chunks" in fts_schemas

            existing_indexes = {
                row[0]
                for row in conn.execute(
                    "SELECT index_name FROM duckdb_indexes()"
                ).fetchall()
            }
            hnsw_exists = "idx_hnsw_chunks" in existing_indexes
        finally:
            conn.close()
    except Exception as e:
        rich.print(f"[yellow]Could not check corpus: {e}[/yellow]")
        return

    rich.print("[bold]Corpus:[/bold]")
    rich.print(f"  {doc_count:,} documents, {chunk_count:,} chunks")
    if meta_row:
        rich.print(f"  Embedding model: {meta_row[0]} (dim={meta_row[1]})")
    else:
        rich.print("  [yellow]Embedding model: unknown (corpus_meta empty)[/yellow]")

    rich.print()
    rich.print("  [bold]Document types:[/bold]")
    for doc_type, count in doc_type_rows:
        rich.print(f"      {doc_type}: {count:,}")

    rich.print()
    fts_icon = "[green]✅[/green]" if fts_exists else "[red]❌[/red]"
    rich.print(f"  {fts_icon} FTS document_chunks")
    hnsw_icon = "[green]✅[/green]" if hnsw_exists else "[red]❌[/red]"
    rich.print(f"  {hnsw_icon} HNSW index idx_hnsw_chunks")


def _summary_line_fetch(conn: Any) -> str:
    """Compact 'Database:'/'Tables:' lines for the bare `esdc status` summary."""
    from esdc.dbmanager import check_table_stats, get_last_updated

    db_file = Config.get_db_file()
    last_updated = get_last_updated(conn)
    updated_suffix = f" (last updated {last_updated})" if last_updated else ""
    table_stats = check_table_stats(conn)
    loaded = sum(1 for ts in table_stats if ts["years"])
    total = len(table_stats)
    return (
        f"[bold]Database:[/bold] {db_file} [green]✅[/green]{updated_suffix}\n"
        f"[bold]Tables:[/bold]   {loaded}/{total} loaded"
    )


def _summary_line_index(conn: Any) -> str:
    """Compact 'Indexes:' line for the bare `esdc status` summary."""
    from esdc.dbmanager import check_indexes

    idx = check_indexes(conn)
    fts_missing = sum(1 for f in idx["fts_indexes"] if not f["exists"])
    btree_missing = sum(1 for b in idx["btree_indexes"] if not b["exists"])
    hnsw_ok = idx["embeddings"]["hnsw_exists"]

    fts_part = (
        "FTS [green]✅[/green]"
        if fts_missing == 0
        else f"FTS [red]❌[/red] ({fts_missing} missing)"
    )
    btree_part = (
        "B-tree [green]✅[/green]"
        if btree_missing == 0
        else f"B-tree [red]❌[/red] ({btree_missing} missing)"
    )
    hnsw_part = "HNSW [green]✅[/green]" if hnsw_ok else "HNSW [red]❌[/red]"
    return f"[bold]Indexes:[/bold]  {fts_part}  {btree_part}  {hnsw_part}"


def _summary_line_corpus(conn: Any) -> str:
    """Compact 'Corpus:' line for the bare `esdc status` summary."""
    existing_tables = {
        row[0]
        for row in conn.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'main'"
        ).fetchall()
    }
    if not set(_CORPUS_TABLES) <= existing_tables:
        return (
            "[bold]Corpus:[/bold]   [yellow]not initialized[/yellow] "
            "(run [cyan]esdc corpus commit[/cyan])"
        )

    doc_count = (conn.execute("SELECT COUNT(*) FROM documents").fetchone() or (0,))[0]
    chunk_count = (
        conn.execute("SELECT COUNT(*) FROM document_chunks").fetchone() or (0,)
    )[0]
    meta_row = conn.execute(
        "SELECT embedding_model, dim FROM corpus_meta LIMIT 1"
    ).fetchone()
    model = meta_row[0] if meta_row else "unknown"
    return (
        f"[bold]Corpus:[/bold]   {doc_count:,} documents, "
        f"{chunk_count:,} chunks ({model})"
    )


def _hit_rate_str(hits: int, misses: int) -> str:
    """Compact hit-rate fragment; '–' for RAM-only stats with no activity."""
    total = hits + misses
    if total == 0:
        return "–"
    return f"{hits / total:.0%} hit"


def _summary_line_cache() -> str:
    """Compact 'Cache:' line for the bare `esdc status` summary."""
    from esdc.chat.tools import get_sql_cache_stats, get_tool_cache_stats

    sql_stats = get_sql_cache_stats()
    tool_stats = get_tool_cache_stats()
    sql_rate = _hit_rate_str(sql_stats.get("hits", 0), sql_stats.get("misses", 0))
    tool_rate = _hit_rate_str(tool_stats.get("hits", 0), tool_stats.get("misses", 0))
    return f"[bold]Cache:[/bold]    SQL {sql_rate} · Tool {tool_rate}"


status_app = typer.Typer(no_args_is_help=False)
app.add_typer(
    status_app,
    name="status",
    help="Show database location, configuration, and index status.",
)


@status_app.callback(invoke_without_command=True)
def status_main(ctx: typer.Context) -> None:
    """Show a compact one-glance summary of all domains.

    Detail: esdc status fetch|index|corpus|cache.
    """
    if ctx.invoked_subcommand is not None:
        return

    db_file = Config.get_db_file()
    if not db_file.exists():
        rich.print(
            "[yellow]Database exists: No[/yellow] "
            "(run '[cyan]esdc fetch --save[/cyan]' to create)"
        )
        return

    from esdc.dbmanager import get_duckdb_connection

    try:
        conn = get_duckdb_connection(db_file)
    except Exception as e:
        rich.print(f"[yellow]Could not open database: {e}[/yellow]")
        return

    summary_fns = (_summary_line_fetch, _summary_line_index, _summary_line_corpus)
    try:
        for summary_fn in summary_fns:
            try:
                rich.print(summary_fn(conn))
            except Exception as e:
                rich.print(f"[yellow]could not check: {e}[/yellow]")
    finally:
        conn.close()

    try:
        rich.print(_summary_line_cache())
    except Exception as e:
        rich.print(f"[yellow]could not check cache: {e}[/yellow]")

    rich.print()
    rich.print("Detail: esdc status fetch|index|corpus|cache")


@status_app.command(name="fetch")
def status_fetch_cmd() -> None:
    """DB location/env vars, tables + per-year rows, last updated."""
    _status_fetch_report()


@status_app.command(name="index")
def status_index_cmd(
    verify: Annotated[
        bool,
        typer.Option(
            "--verify",
            help="Run functional verification on indexes (slower).",
        ),
    ] = False,
) -> None:
    """FTS, B-tree, and embeddings + HNSW index status."""
    _status_index_report(verify)


@status_app.command(name="corpus")
def status_corpus_cmd() -> None:
    """Corpus store summary.

    Doc/chunk counts, doc_type breakdown, pinned embedding model/dim,
    chunks FTS + HNSW status.

    Per-file pipeline state: esdc corpus status <paths>.
    """
    _status_corpus_report()


@status_app.command(name="cache")
def status_cache_cmd() -> None:
    """SQL/tool/JSON cache stats, last invalidated."""
    _status_cache_report()


@app.command(name="eureka")
def eureka(
    port: int = typer.Option(2030, "--port", "-p", help="Dashboard port"),
    host: str = typer.Option("0.0.0.0", "--host", "-h", help="Dashboard host"),
    log_level: str = typer.Option("info", "--log-level", help="Log level"),
    year: int | None = typer.Option(None, "--year", "-y", help="Report year"),
) -> None:
    """Launch Eureka — Resources Knowledge Pages dashboard."""
    from esdc.eureka.app import run_eureka

    rich.print(
        f"[bold green]Starting Eureka dashboard on "
        f"http://{host}:{port}/eureka/[/bold green]"
    )
    run_eureka(host=host, port=port, log_level=log_level, year=year)


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


@app.command(name="validate")
def validate(
    rule: Annotated[
        list[str] | None,
        typer.Option("--rule", "-r", help="Specific rule ID(s) to run (e.g., RE9001)."),
    ] = None,
    group: Annotated[
        list[str] | None,
        typer.Option(
            "--group",
            "-g",
            help="Rule group(s) to run (e.g., RE9).",
        ),
    ] = None,
    force_fix: Annotated[
        bool,
        typer.Option(
            "--force-fix",
            help="Apply fixes to violations (writes to DB).",
        ),
    ] = False,
    severity: Annotated[
        str | None,
        typer.Option(
            "--severity",
            help="Filter by severity: strict, warning, info.",
        ),
    ] = None,
    year: Annotated[
        list[int] | None,
        typer.Option(
            "--year",
            min=2019,
            help="Filter by report year(s). Can specify multiple.",
        ),
    ] = None,
    verbose: Annotated[
        int,
        typer.Option(
            "--verbose",
            "-v",
            count=True,
            help=(
                "Verbosity level: "
                "0=group summary (default), "
                "1=per-rule summary, "
                "2=full detail."
            ),
        ),
    ] = 0,
    save: Annotated[
        bool,
        typer.Option(
            "--save/--no-save",
            help="Save violations to file.",
        ),
    ] = False,
    save_format: Annotated[
        str,
        typer.Option(
            "--save-format",
            help="Save format: xlsx, csv, json.",
        ),
    ] = "xlsx",
) -> None:
    """Validate ESDC data against business rules.

    Rules are organized by group IDs (RE0-RE9).  Each rule has a unique
    rule ID such as ``RE9001`` defined via a formal mathematical
    expression.

    Verbosity levels:
        0 (default): Group summary only — total violations per rule group.
        1 (-v): Per-rule summary — rule ID, formula, and violation count.
        2 (-vv): Full detail — individual violations with values.

    Examples:
    --------
    Run all rules (group summary only)::

        esdc validate

    Per-rule summary::

        esdc validate -v

    Full detail with individual violations::

        esdc validate -vv

    Run a specific group::

        esdc validate --group RE9

    Run a specific rule::

        esdc validate --rule RE9001

    Fix violations in the database::

        esdc validate --force-fix

    Filter by severity::

        esdc validate --severity strict

    Validate a specific year::

        esdc validate --year 2024

    Save violations to file::

        esdc validate --save
        esdc validate --save --save-format csv
        esdc validate --save --save-format json
    """
    # Validate severity early so invalid value fails fast
    if severity:
        try:
            Severity(severity)
        except ValueError:
            rich.print(
                f"[red]Unknown severity: {severity}. Use: strict, warning, info[/red]"
            )
            raise typer.Exit(1) from None

    # Import rules so that the @register_rule decorator fires.
    import esdc.validate.rule_re0  # noqa: F401
    import esdc.validate.rule_re1  # noqa: F401
    import esdc.validate.rule_re2  # noqa: F401
    import esdc.validate.rule_re9  # noqa: F401

    if force_fix:
        rich.print(
            "[bold red]Warning: --force-fix will modify the database![/bold red]"
        )
        rich.print("[dim]Press Ctrl+C to cancel, or Enter to continue...[/dim]")
        try:
            input()
        except KeyboardInterrupt:
            raise typer.Abort() from None

    results = run_validation(
        rule_ids=rule,
        groups=group,
        force_fix=force_fix,
        year=year,
    )

    if severity:
        results = [r for r in results if r.severity == Severity(severity)]

    if not results:
        rich.print("[green]No validation rules matched.[/green]")
        return

    total_violations = sum(r.total_violations for r in results)
    total_fixed = sum(r.fix_applied_count for r in results)
    results_with_violations = [r for r in results if r.total_violations > 0]

    if verbose >= 2:
        _print_full_detail(results_with_violations)
    elif verbose == 1:
        _print_rule_summary(results_with_violations)

    # Always print group summary (and total) unless verbose >= 2
    # which already includes a per-rule summary ending with totals.
    if verbose < 2:
        _print_group_summary(results)

    rich.print("")
    if total_violations == 0:
        rich.print("[green]✓ All validations passed![/green]")
    elif force_fix and total_fixed > 0:
        rich.print(
            f"[green]✓ Applied {total_fixed} fixes out of "
            f"{total_violations} violations[/green]"
        )
        if total_fixed < total_violations:
            remaining = total_violations - total_fixed
            rich.print(
                f"[yellow]⚠ {remaining} violation(s) remaining "
                f"(manual review required)[/yellow]"
            )
    else:
        rich.print(
            f"[yellow]⚠ Found {total_violations} violation(s). "
            f"Use --force-fix to apply fixes.[/yellow]"
        )

    if save:
        _save_violations(results, save_format)


def _severity_icon(severity: Severity) -> str:
    icons = {
        Severity.STRICT: "⛔",
        Severity.WARNING: "🟡",
        Severity.INFO: "🔵",
    }
    return icons.get(severity, "⚪")


def _print_group_summary(results: list[ValidationResult]) -> None:
    from collections import defaultdict

    group_totals: dict[str, int] = defaultdict(int)
    for r in results:
        group_totals[r.rule_group] += r.total_violations

    for group in sorted(group_totals):
        total = group_totals[group]
        rich.print(f"  {group}: {total:,} violations")

    total = sum(group_totals.values())
    rich.print(f"  Total: {total:,} violations")


def _print_rule_summary(results: list[ValidationResult]) -> None:
    for result in results:
        icon = _severity_icon(result.severity)
        rich.print(f"\n{icon} [bold]{result.rule_id}[/bold]: {result.description}")

        if result.formal:
            rich.print(f"   {result.formal}")

        rich.print(f"   Violations: [bold]{result.total_violations}[/bold]")


def _print_full_detail(results: list[ValidationResult]) -> None:
    for result in results:
        icon = _severity_icon(result.severity)
        rich.print(f"\n{icon} [bold]{result.rule_id}[/bold]: {result.description}")

        if result.formal:
            rich.print(f"   {result.formal}")

        fixable_text = "Yes" if result.is_fixable else "No"
        fixable_style = "green" if result.is_fixable else "yellow"
        rich.print(f"   Fixable: [{fixable_style}]{fixable_text}[/{fixable_style}]")

        rich.print(f"   Violations: [bold]{result.total_violations}[/bold]")

        for v in result.violations[:20]:
            ids_text = ", ".join(f"{k}={val}" for k, val in v.identifiers.items())
            rich.print(f"   - {ids_text}")

            shown = {
                k: val
                for k, val in v.current_values.items()
                if val is not None and k != "project_isactive"
            }
            if shown:
                cols_text = ", ".join(f"{k}={val}" for k, val in shown.items())
                rich.print(f"     {cols_text}")
            if v.fix_applied:
                rich.print("     [green]✓ Fixed[/green]")

        if result.total_violations > 20:
            rich.print(f"   ... and {result.total_violations - 20} more")


def _save_violations(results: list[ValidationResult], fmt: str) -> None:
    """Collect all violations into a DataFrame and save to file."""
    from datetime import datetime

    import pandas as pd

    fmt = (fmt or "xlsx").strip().lower()
    if fmt not in ("xlsx", "csv", "json"):
        rich.print(f"[red]Unknown format: {fmt}. Use xlsx, csv, or json.[/red]")
        raise typer.Exit(1) from None

    rows: list[dict[str, object]] = []
    for result in results:
        for v in result.violations:
            rows.append(
                {
                    "report_year": v.identifiers.get("report_year", ""),
                    "wk_name": v.identifiers.get("wk_name", ""),
                    "field_name": v.identifiers.get("field_name", ""),
                    "project_name": v.identifiers.get("project_name", ""),
                    "validated_column": v.validated_column,
                    "severity": v.severity.value,
                    "rule_id": v.rule_id,
                    "description": v.description,
                }
            )

    if not rows:
        rich.print("[green]No violations to save.[/green]")
        return

    df = pd.DataFrame(rows)
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    filename = f"val_result_{timestamp}.{fmt}"

    if fmt == "xlsx":
        df.to_excel(filename, index=False, engine="openpyxl")
    elif fmt == "csv":
        df.to_csv(filename, index=False)
    elif fmt == "json":
        df.to_json(filename, orient="records", indent=2)

    rich.print(f"[green]Saved {len(rows)} violations to {filename}[/green]")


def _print_corpus_report(report) -> None:
    """Print a processed/skipped/failed summary + warnings; exit 1 on total failure.

    Partial failure exits 0 by design: batch progress is preserved and each
    failure is listed per file, so scripts that need stricter semantics should
    parse the FAILED lines rather than rely on the exit code.
    """
    rows = [
        ("processed", len(report.processed)),
        ("skipped", len(report.skipped)),
        ("failed", len(report.failed)),
    ]
    rich.print(tabulate(rows, headers=["", "count"], tablefmt="psql"))
    for name, error in report.failed.items():
        typer.echo(f"  FAILED {name}: {error}", err=True)
    for warning in report.warnings:
        typer.echo(f"  Warning: {warning}", err=True)
    if report.failed and not report.processed:
        raise typer.Exit(1)


def _validate_corpus_overrides(
    level: str | None,
    doc_type: str | None,
    topic: str | None = None,
    wk_name: str | None = None,
    field_name: str | None = None,
    project_name: str | None = None,
) -> None:
    from esdc.corpus.metadata import DOC_LEVELS, DOC_TOPICS, DOC_TYPES, doc_level_rule

    allowed_levels = tuple(lv for lv in DOC_LEVELS if lv != "unknown")
    if level is not None and level not in allowed_levels:
        typer.echo(
            f"Error: --level must be one of {', '.join(allowed_levels)}.", err=True
        )
        raise typer.Exit(1)
    if doc_type is not None and doc_type not in DOC_TYPES:
        typer.echo(
            f"Error: --doc-type must be one of {', '.join(DOC_TYPES)}.", err=True
        )
        raise typer.Exit(1)
    if topic is not None and topic not in DOC_TOPICS:
        typer.echo(
            f"Error: --topic must be one of {', '.join(DOC_TOPICS)}.", err=True
        )
        raise typer.Exit(1)

    rule = doc_level_rule(doc_type, [topic] if topic is not None else None)
    implied_level = rule[2] if rule is not None else None

    entity_given = any(v is not None for v in (wk_name, field_name, project_name))
    effective_level = level or implied_level
    if effective_level == "regulation" and entity_given:
        typer.echo(
            "Error: regulation documents cannot have wk/field/project entities.",
            err=True,
        )
        raise typer.Exit(1)

    if level is not None and implied_level is not None and level != implied_level:
        kind, key, _ = rule
        typer.echo(
            f"Error: --level {level} conflicts with the {kind} '{key}' rule "
            f"(implies {implied_level}).",
            err=True,
        )
        raise typer.Exit(1)


def _open_corpus_store():
    """Open a CorpusStore with tables ensured, or exit 1 with a clear error."""
    from esdc.corpus.store import CorpusStore

    store = CorpusStore()
    try:
        store.ensure_tables()
    except ValueError as e:
        store.close()
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1) from None
    return store


@corpus_app.command()
def extract(
    paths: Annotated[
        list[Path],
        typer.Argument(
            exists=True, help="Source file(s) (.pdf, .docx, .md) or folder(s)."
        ),
    ],
    level: Annotated[
        str | None,
        typer.Option(
            "--level", help="Override doc_level: wk, field, project, regulation."
        ),
    ] = None,
    doc_type: Annotated[
        str | None, typer.Option("--doc-type", help="Override doc_type.")
    ] = None,
    topic: Annotated[
        str | None,
        typer.Option("--topic", help="Set doc_topic (single value)."),
    ] = None,
    wk_name: Annotated[
        str | None, typer.Option("--wk-name", help="Override wk_name.")
    ] = None,
    field_name: Annotated[
        str | None, typer.Option("--field-name", help="Override field_name.")
    ] = None,
    project_name: Annotated[
        str | None, typer.Option("--project-name", help="Override project_name.")
    ] = None,
    force: Annotated[
        bool,
        typer.Option("--force", help="Re-extract even if a sidecar already exists."),
    ] = False,
) -> None:
    """Parse .pdf/.docx/.md sources to reviewable .corpus.md sidecars (step 1 of 2)."""
    from esdc.corpus.pipeline import run_extract

    _validate_corpus_overrides(level, doc_type, topic, wk_name, field_name, project_name)

    try:
        report = run_extract(
            paths,
            level=level,
            doc_type=doc_type,
            topic=topic,
            wk_name=wk_name,
            field_name=field_name,
            project_name=project_name,
            force=force,
        )
    except ValueError as e:  # e.g. --wk-name value not in canonical tables
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1) from None
    _print_corpus_report(report)
    if report.processed:
        typer.echo("Review the .corpus.md files, then run: esdc corpus commit <folder>")


@corpus_app.command()
def commit(
    paths: Annotated[
        list[Path],
        typer.Argument(exists=True, help="Sidecar .corpus.md file(s) or folder(s)."),
    ],
    skip_review: Annotated[
        bool,
        typer.Option(
            "--skip-review",
            help="Ingest sidecars still pending review; marks reviewed on success.",
        ),
    ] = False,
    force: Annotated[
        bool, typer.Option("--force", help="Re-ingest even if already committed.")
    ] = False,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Validate and report without writing.")
    ] = False,
) -> None:
    """Ingest reviewed .corpus.md sidecars into the searchable corpus (step 2 of 2)."""
    from esdc.corpus.pipeline import run_commit

    try:
        report = run_commit(
            paths,
            skip_review=skip_review,
            force=force,
            dry_run=dry_run,
        )
    except ValueError as e:  # e.g. embedding-model mismatch -> `corpus reembed`
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1) from None
    _print_corpus_report(report)


@corpus_app.command(name="status")
def corpus_status(
    paths: Annotated[
        list[Path],
        typer.Argument(exists=True, help="PDF/sidecar file(s) or folder(s) to check."),
    ],
) -> None:
    """Show each PDF/sidecar's place in the extract -> review -> commit pipeline.

    Store-level summary: esdc status corpus.
    """
    from esdc.corpus.pipeline import run_status

    rows = run_status(paths)
    rich.print(
        tabulate(
            [(r["file"], r["state"]) for r in rows],
            headers=["file", "state"],
            tablefmt="psql",
        )
    )
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["state"]] = counts.get(r["state"], 0) + 1
    for state, n in sorted(counts.items()):
        typer.echo(f"  {state}: {n}")


@corpus_app.command(name="meta")
def corpus_meta(
    paths: Annotated[
        list[Path],
        typer.Argument(exists=True, help="Sidecar .corpus.md file(s) or folder(s)."),
    ],
    level: Annotated[
        str | None,
        typer.Option(
            "--level", help="Set doc_level: wk, field, project, regulation."
        ),
    ] = None,
    doc_type: Annotated[
        str | None, typer.Option("--doc-type", help="Set doc_type.")
    ] = None,
    topic: Annotated[
        str | None,
        typer.Option("--topic", help="Set doc_topic (single value)."),
    ] = None,
    wk_name: Annotated[
        str | None, typer.Option("--wk-name", help="Set wk_name.")
    ] = None,
    field_name: Annotated[
        str | None, typer.Option("--field-name", help="Set field_name.")
    ] = None,
    project_name: Annotated[
        str | None, typer.Option("--project-name", help="Set project_name.")
    ] = None,
    reviewed: Annotated[
        bool | None,
        typer.Option("--reviewed/--no-reviewed", help="Set the reviewed flag."),
    ] = None,
) -> None:
    """Show or bulk-set .corpus.md frontmatter metadata.

    Without flags: print a metadata table. With flags: write the values
    into each sidecar's frontmatter (re-running entity resolution).
    """
    from esdc.corpus.pipeline import run_meta, run_meta_show

    _validate_corpus_overrides(level, doc_type, topic, wk_name, field_name, project_name)

    values = (level, doc_type, topic, wk_name, field_name, project_name, reviewed)
    if all(v is None for v in values):
        rows = run_meta_show(paths)
        table = [
            (
                r["file"],
                r.get("doc_type"),
                _topic_display(r.get("doc_topic")),
                r.get("doc_date"),
                r.get("doc_level"),
                _entity_display(r),
                r.get("reviewed"),
                r.get("error") or "",
            )
            for r in rows
        ]
        headers = [
            "file", "doc_type", "topic", "doc_date", "doc_level", "entity",
            "reviewed", "note",
        ]
        rich.print(tabulate(table, headers=headers, tablefmt="psql"))
        return

    try:
        report = run_meta(
            paths,
            level=level,
            doc_type=doc_type,
            topic=topic,
            wk_name=wk_name,
            field_name=field_name,
            project_name=project_name,
            reviewed=reviewed,
        )
    except ValueError as e:  # e.g. --wk-name value not in canonical tables
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1) from None
    _print_corpus_report(report)


def _topic_display(topic: Any) -> str:
    """Join a doc_topic list for table display; blank for None/empty."""
    if not topic:
        return ""
    if isinstance(topic, list):
        return ", ".join(str(t) for t in topic)
    return str(topic)


def _entity_display(d: dict) -> str:
    """Join first non-empty entity field for display."""
    for key in ("wk_name", "field_name", "project_name"):
        val = d.get(key)
        if not val:
            continue
        if isinstance(val, str):
            try:
                val = json.loads(val)
            except (json.JSONDecodeError, TypeError):
                return val  # legacy plain-text value — show as-is
        if isinstance(val, list):
            return ", ".join(str(v) for v in val)
        return str(val)
    return ""


@corpus_app.command(name="list")
def list_documents() -> None:
    """List all documents committed to the corpus."""
    store = _open_corpus_store()
    try:
        docs = store.list_documents()
    finally:
        store.close()

    rows = [
        (
            d["doc_id"],
            d["file_name"],
            d["doc_type"],
            d["doc_date"],
            d["doc_level"],
            _entity_display(d),
            d["n_chunks"],
        )
        for d in docs
    ]
    headers = [
        "doc_id", "file_name", "doc_type", "doc_date",
        "doc_level", "entity", "n_chunks",
    ]
    rich.print(tabulate(rows, headers=headers, tablefmt="psql"))


@corpus_app.command()
def remove(
    doc_ids: Annotated[list[str], typer.Argument(help="Document ID(s) to remove.")],
) -> None:
    """Remove document(s) from the corpus (does not touch files on disk)."""
    store = _open_corpus_store()
    removed = 0
    try:
        for doc_id in doc_ids:
            if store.get_document(doc_id) is None:
                typer.echo(f"Not found: {doc_id}", err=True)
                continue
            store.delete_document(doc_id)
            removed += 1
        if removed:
            store.rebuild_indexes()
    finally:
        store.close()
    typer.echo(f"Removed {removed} document(s).")


@corpus_app.command()
def clear(
    yes: Annotated[
        bool, typer.Option("--yes", help="Confirm deletion of the entire corpus.")
    ] = False,
) -> None:
    """Delete all documents and chunks from the corpus."""
    store = _open_corpus_store()
    try:
        counts = store.counts()
        if not yes:
            typer.echo(
                f"This deletes {counts['documents']} documents and "
                f"{counts['chunks']} chunks. Re-run with --yes."
            )
            raise typer.Exit(1)
        removed = store.clear()
        store.rebuild_indexes()
    finally:
        store.close()
    typer.echo(
        f"Removed {removed['documents']} documents and {removed['chunks']} chunks."
    )


@corpus_app.command()
def reembed() -> None:
    """Rebuild chunk embeddings for the whole corpus after an embedding-model change."""
    from esdc.corpus.pipeline import run_reembed

    report = run_reembed()
    _print_corpus_report(report)
    typer.echo(
        f"Re-embedded {len(report.processed)} document(s) "
        f"with model '{report.embedding_model}'."
    )




if __name__ == "__main__":
    app()
