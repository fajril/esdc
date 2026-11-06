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
import hashlib
import io
import json
import logging
import os
import sqlite3
import warnings
from collections.abc import Iterable, Mapping, Sequence
from contextlib import closing
from dataclasses import replace
from datetime import date, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Annotated, Any, cast

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
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)
from tabulate import tabulate

from esdc.api_project_resources import (
    _ALLOWED_SOURCE_FIELDS,
    normalize_project_resources,
)
from esdc.api_projection import (
    ProjectPublication,
    publish_project_resources,
)
from esdc.api_snapshot import (
    ApiProjectResourceRow,
    JSONScalar,
    get_project_resources_snapshot,
)
from esdc.commands.analytics import (  # noqa: E402
    bootstrap_timeseries_source_command,
    rebuild_api_analytics_command,
    verify_analytics_migration_command,
)
from esdc.commands.configs import configs_app  # noqa: E402
from esdc.commands.spatial import spatial_app  # noqa: E402
from esdc.configs import Config  # noqa: E402
from esdc.console import console  # noqa: E402
from esdc.dbmanager import (  # noqa: E402
    get_duckdb_connection,
    load_data_to_db,
    run_query,
)
from esdc.fetch_publication import publish_prepared_fetch
from esdc.fetching import (
    FetchFailure,
    FetchOutcome,
    FetchPhase,
    PreparedFetch,
    PreparedTimeseriesSlice,
    parse_report_year,
    save_payload_atomically,
    spool_timeseries,
    validate_timeseries_header,
)
from esdc.loaders import (  # noqa: E402
    LoadSchemaError,
    SpreadsheetLoadError,
    copy_pod_schema_template,
    generate_schema_template_from_excel,
    load_excel_to_duckdb,
    print_load_result,
)
from esdc.pod_registry.api_context import read_pod_api_context  # noqa: E402
from esdc.pod_registry.api_import import (  # noqa: E402
    INPUT_TABLES,
    PodApiImportError,
    prepare_pod_api_batch,
)
from esdc.pod_registry.api_payload import (  # noqa: E402
    PodApiPayloadError,
    parse_pod_api_payload,
)
from esdc.pod_registry.contract import load_pod_contract  # noqa: E402
from esdc.pod_registry.normalization import normalize_project_id  # noqa: E402
from esdc.pod_registry.sqlite_import import (  # noqa: E402
    PodImportError,
    PodImportProjectionError,
    load_prepared_pod_batch,
    prepare_pod_import,
)
from esdc.pod_registry.store import (
    get_esdc_sqlite_path,
    get_sqlite_connection,
)
from esdc.selection import (  # noqa: E402
    ApiVer,
    FileType,
    PodApiTable,
    Severity,
    TableName,
)
from esdc.source_publication import (
    committed_pod_source_generation,
    get_fetch_publication_status,
)
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
app.command(name="bootstrap-timeseries-source")(bootstrap_timeseries_source_command)
app.command(name="rebuild-api-analytics")(rebuild_api_analytics_command)
app.command(name="verify-analytics-migration")(verify_analytics_migration_command)
app.add_typer(configs_app, name="configs")
app.add_typer(spatial_app, name="spatial")
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
            help=("Schema YAML output path. Defaults to ./<excel-stem>.schema.yaml."),
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


def _read_pod_crosswalk(path: Path) -> dict[int, int]:
    import json

    def _object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        seen_raw: set[str] = set()
        for key, _ in pairs:
            if key in seen_raw:
                raise PodImportError(
                    [f"Duplicate raw JSON key in POD crosswalk: {key}"]
                )
            seen_raw.add(key)
        return dict(pairs)

    try:
        data = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_object,
        )
    except (OSError, UnicodeError) as exc:
        raise PodImportError([f"Could not read POD crosswalk: {exc}"]) from None
    except json.JSONDecodeError as exc:
        raise PodImportError([f"Invalid JSON in POD crosswalk: {exc}"]) from None
    except PodImportError:
        raise

    if not isinstance(data, dict):
        raise PodImportError(["POD crosswalk must be a JSON object."])

    mapping: dict[int, int] = {}
    seen_normalized: set[int] = set()

    for raw_key, raw_value in data.items():
        try:
            old_id = int(raw_key)
        except ValueError:
            raise PodImportError([f"Invalid POD crosswalk key: {raw_key!r}"]) from None

        if old_id in seen_normalized:
            raise PodImportError(
                [f"Duplicate normalized POD crosswalk key: {old_id}"]
            ) from None

        seen_normalized.add(old_id)

        if isinstance(raw_value, bool) or not isinstance(raw_value, int):
            raise PodImportError(
                [f"Invalid POD crosswalk target for key {raw_key!r}"]
            ) from None

        mapping[old_id] = raw_value

    return mapping


@app.command()
def load(
    from_excel: Annotated[
        Path | None,
        typer.Option(
            "--from-excel",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            help="Excel .xlsx file to load.",
        ),
    ] = None,
    from_sqlite: Annotated[
        Path | None,
        typer.Option(
            "--from-sqlite",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            help="Read-only SQLite POD source to import.",
        ),
    ] = None,
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
    pod_registry: Annotated[
        bool,
        typer.Option(
            "--pod-registry",
            help="Import the master-only POD profile and preserve target facts.",
        ),
    ] = False,
    pod_crosswalk: Annotated[
        Path | None,
        typer.Option(
            "--pod-crosswalk",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            help=(
                "Approved JSON object mapping old numeric POD IDs to new "
                "pod_id_seq values for a one-time full replacement."
            ),
        ),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="Prepare and validate a POD source without opening a target writer.",
        ),
    ] = False,
) -> None:
    """Load contract-driven POD sources or generic Excel tables.

    Run `esdc schema --generate --from-excel data.xlsx` to create a starter schema.
    Use `--schema-pod` for the complete five-input Excel POD profile; SQLite
    selects the full profile automatically. Use `--pod-registry` for the
    three-table master-only profile. Replacing an old POD target requires a full
    load and a reviewed `--pod-crosswalk` mapping for protected document links.
    """
    if pod_crosswalk is not None and (schema is not None or pod_registry):
        typer.echo(
            "Error: --pod-crosswalk applies only to a full POD load; it cannot be "
            "combined with generic --schema or --pod-registry."
        )
        raise typer.Exit(1) from None

    if schema is not None:
        if schema_pod or pod_registry:
            typer.echo("Error: generic --schema cannot be combined with a POD profile.")
            raise typer.Exit(1) from None
        if from_excel is None or from_sqlite is not None:
            typer.echo(
                "Error: generic --schema requires exactly one --from-excel source."
            )
            raise typer.Exit(1) from None
        if dry_run:
            typer.echo("Error: --dry-run is supported only for POD registry imports.")
            raise typer.Exit(1) from None
    else:
        if schema_pod and pod_registry:
            typer.echo(
                "Error: --schema-pod and --pod-registry select different POD profiles."
            )
            raise typer.Exit(1) from None
        sources = sum([from_excel is not None, from_sqlite is not None])
        if sources != 1:
            typer.echo("Error: specify exactly one of --from-excel or --from-sqlite.")
            raise typer.Exit(1) from None
        if from_excel is not None and not (schema_pod or pod_registry):
            typer.echo("Error: Excel POD loads require --schema-pod or --pod-registry.")
            raise typer.Exit(1) from None

    if schema is None:
        profile = "master-only" if pod_registry else "full"
        source = "xlsx" if from_excel is not None else "sqlite"
        source_path = from_excel if from_excel is not None else from_sqlite
        assert source_path is not None
        batch = None
        try:
            approved_map = (
                _read_pod_crosswalk(pod_crosswalk)
                if pod_crosswalk is not None
                else None
            )
            batch = prepare_pod_import(
                source_path,
                source,
                profile=profile,
                approved_crosswalk=approved_map,
            )
            for table, rows in batch.rows.items():
                typer.echo(f"  {table}: {len(rows):,} rows")
            if batch.staged_snapshots is not None:
                typer.echo(
                    "  snapshots: "
                    f"{batch.staged_snapshots.resource_row_count:,} resource rows, "
                    f"{batch.staged_snapshots.timeseries_row_count:,} timeseries rows"
                )
            if dry_run:
                typer.echo(
                    f"POD {profile} source validation passed; target was not "
                    "opened for writing."
                )
                return
            result, publication = load_prepared_pod_batch(
                batch,
                sqlite_path=get_esdc_sqlite_path(),
                duck_path=Config.get_db_file(),
            )
            typer.echo(
                f"SQLite POD source committed at generation {result.source_generation}."
            )
            if publication is None:
                typer.echo("DuckDB publication was skipped.")
            else:
                typer.echo("POD registry published to DuckDB.")
            return
        except PodImportProjectionError as e:
            typer.echo(f"Error: {e}")
            typer.echo(
                "The SQLite source committed. Retry publication with "
                "'esdc rebuild-api-analytics'; "
                "do not re-import the source."
            )
            raise typer.Exit(1) from None
        except PodImportError as e:
            typer.echo("Error: POD import failed:")
            for message in e.errors:
                typer.echo(f"  - {message}")
            raise typer.Exit(1) from None
        finally:
            if batch is not None:
                batch.cleanup_staging()

    try:
        schema_path = schema
        assert schema_path is not None
        assert from_excel is not None
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

    try:
        if filetype == "csv":
            _ = load_esdc_data(
                filetype=FileType.CSV,
                to_file=should_save,
                reload=should_reload,
                username=username,
                password=password,
                years=year,
                reindex=should_reindex,
            )
        elif filetype == "json":
            _ = load_esdc_data(
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
    except Exception as exc:
        context = ""
        if isinstance(exc, FetchFailure):
            context_parts = [f"phase={exc.phase}"]
            if exc.table is not None:
                context_parts.append(f"table={exc.table}")
            if exc.report_year is not None:
                context_parts.append(f"year={exc.report_year}")
            context = " (" + ", ".join(context_parts) + ")"
        typer.echo(f"Fetch failed{context}: {exc}")
        if isinstance(exc, FetchFailure) and exc.sqlite_committed:
            # The committed source and the pending mirror are two different facts.
            generation = committed_pod_source_generation(get_esdc_sqlite_path())
            if generation is None:
                console.print("SQLite source committed.")
            else:
                console.print(f"SQLite source committed at generation {generation}.")
            if exc.recovery_command is not None or not exc.duckdb_committed:
                console.print("DuckDB publication pending.")
                recovery = exc.recovery_command or "esdc rebuild-api-analytics"
                console.print(f"Run '{recovery}' to complete the mirror.")
        raise typer.Exit(1) from None


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
    embed_backend: Annotated[
        str | None,
        typer.Option(
            "--embed-backend",
            help="Override embedding_backend for this run: local, ollama or openai.",
        ),
    ] = None,
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
        embed_backend: Override embedding_backend for this run
        (local, ollama or openai).

    Returns:
        None
    """
    # Handle embeddings-only mode
    if embeddings_only:
        _generate_embeddings(embed_backend=embed_backend)
        return

    # Handle reindex-only mode
    if reindex_only:
        from esdc.dbmanager import reindex_fts

        reindex_fts()
        return

    _reload_files(filetype)
    from esdc.pod_registry.publication import PodPublicationReport, has_v2_state

    sqlite_path = get_esdc_sqlite_path()
    if has_v2_state(sqlite_path):
        from esdc.pod_registry.publish import publish_pod_registry

        publication = publish_pod_registry(sqlite_path, Config.get_db_file())
        if (
            isinstance(publication, PodPublicationReport)
            and publication.projection_error is not None
        ):
            typer.echo(
                "POD registry source was "
                f"{'committed' if publication.source_committed else 'not committed'}; "
                f"DuckDB publication failed: {publication.projection_error}",
                err=True,
            )
            raise typer.Exit(code=1)

    # Generate embeddings after reload (unless disabled)
    if not no_embeddings:
        _generate_embeddings(embed_backend=embed_backend)


def _reload_files(filetype: str | None) -> None:
    """Validate all available API files before one joint publication."""
    if filetype not in (FileType.CSV.value, FileType.JSON.value):
        logging.debug("Unknown reload format %s", filetype)
        return

    db_dir = Config.get_db_dir()
    resource_rows: tuple[ApiProjectResourceRow, ...] = ()
    timeseries_slices: list[PreparedTimeseriesSlice] = []
    origin = "api_csv" if filetype == FileType.CSV.value else "api_json"

    with TemporaryDirectory(prefix="esdc-reload-") as temporary_directory:
        for table in TABLES:
            filename = db_dir / f"{table.value}.{filetype}"
            if not filename.exists():
                logging.warning("File %s is not found.", filename)
                continue
            payload = filename.read_bytes()
            if table == TableName.PROJECT_RESOURCES:
                if filetype == FileType.CSV.value:
                    rows, header = _read_csv(io.StringIO(payload.decode("utf-8")))
                    resource_rows = _csv_rows_to_normalized_project_resources(
                        rows, header
                    )
                else:
                    parsed = json.loads(
                        payload, object_pairs_hook=_json_object_without_duplicate_keys
                    )
                    resource_rows = _normalize_project_resources_json(parsed)
                continue

            if filetype == FileType.CSV.value:
                rows, header = _parse_table_payload(payload, table, FileType.CSV, None)
            else:
                rows, header = _parse_timeseries_json(payload, None)
            if not rows:
                continue
            if not isinstance(header, (list, tuple)) or any(
                not isinstance(column, str) for column in header
            ):
                raise ValueError("project_timeseries header must contain column names")
            names = validate_timeseries_header(header)
            year_index = names.index("report_year")
            by_year: dict[int, list[Sequence[JSONScalar]]] = {}
            for index, raw in enumerate(rows):
                if len(raw) != len(names):
                    raise ValueError(
                        f"project_timeseries record {index} has wrong width"
                    )
                try:
                    year = parse_report_year(raw[year_index])
                except ValueError as exc:
                    raise ValueError(
                        f"project_timeseries record {index} column report_year: {exc}"
                    ) from exc
                values = list(raw)
                values[year_index] = year
                by_year.setdefault(year, []).append(values)
            for year, year_rows in sorted(by_year.items()):
                prepared_slice = spool_timeseries(
                    names,
                    year_rows,
                    report_year=year,
                    directory=Path(temporary_directory),
                    origin=origin,
                )
                if prepared_slice is not None:
                    timeseries_slices.append(prepared_slice)

        prepared = PreparedFetch(
            resource_rows,
            None,
            tuple(timeseries_slices),
            (),
            (),
            timeseries_mode=("replace_all" if timeseries_slices else "replace_years"),
        )
        if prepared.resource_rows or prepared.timeseries:
            publish_prepared_fetch(
                prepared,
                duckdb_path=Config.get_db_file(),
                sqlite_path=get_esdc_sqlite_path(),
                reindex=True,
            )


def _generate_embeddings(embed_backend: str | None = None) -> None:
    """Generate semantic embeddings for project_remarks with progress bar.

    ``embed_backend`` overrides the ``embedding_backend`` config key for
    this run only. Query-time similarity always runs locally regardless.
    """
    from esdc.configs import Config
    from esdc.embedders import get_build_embedder
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

    try:
        embedder = get_build_embedder(embed_backend)
    except ValueError as e:
        # A stored value that has no implementation is reported, not repaired.
        typer.echo(f"Error: {e}")
        raise typer.Exit(1) from None

    logger.info(f"Initialized embedder with model: {embedder.model}")

    # Every client is an HTTP service now, so the check reports what that
    # service said: a status, a connection failure, or a response that is not a
    # vector. The message is the diagnosis, so it is what the user sees.
    message = embedder.check_available()
    if message:
        logger.warning(message)
        console.print(f"[yellow]Warning: {message}[/yellow]")
        console.print(
            "[dim]To generate embeddings later, run: esdc reload --embeddings-only[/dim]"  # noqa: E501
        )
        return

    resolver = SemanticResolver(db_path=db_path, embedder=embedder, read_only=False)

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
                f"Processing with {embedder.model}", total=total_docs
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


_V2_POD_SHOW_TABLES = frozenset(
    {
        "m_project",
        "m_pod",
        "pod_registry",
        "pod_project",
        "pod_document",
        "pod_monitoring_api",
        "pod_forecast_api",
        "r_institution",
        "r_pod_type",
        "r_pod_stage",
        "pod_registry_state",
        "pod_knowledge_state",
        "pod_read_model_snapshot",
        "pod_read_model_diagnostic",
    }
)


def _show_v2_pod_table(
    table: str,
    *,
    where: str | None,
    search: str | None,
    years: list[int] | None,
    columns: list[str] | str,
) -> pd.DataFrame | None:
    """Query a published v2 POD object after checking its current schema."""
    conn = get_duckdb_connection(Config.get_db_file(), read_only=True)
    try:
        exists = conn.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema='main' AND table_name=?",
            [table],
        ).fetchone()
        if exists is None:
            logging.warning("POD registry object %s is not published.", table)
            return None
        available = {
            str(row[0])
            for row in conn.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_schema='main' AND table_name=?",
                [table],
            ).fetchall()
        }
        selected = columns if isinstance(columns, list) else columns.split()
        if selected and any(column not in available for column in selected):
            raise ValueError("requested column is not in the published POD schema")
        projection = (
            ", ".join(f'"{column}"' for column in selected) if selected else "*"
        )
        query = f'SELECT {projection} FROM "{table}"'
        predicates: list[str] = []
        params: list[str | int] = []
        if where and search:
            if where not in available:
                raise ValueError("search column is not in the published POD schema")
            predicates.append(f'CAST("{where}" AS VARCHAR) ILIKE ?')
            params.append(f"%{search}%")
        if years:
            year_column = next(
                (
                    name
                    for name in (
                        "report_year",
                        "year_frcast",
                        "reported_date",
                        "approval_date",
                    )
                    if name in available
                ),
                None,
            )
            if year_column is None:
                raise ValueError(f"{table} has no year or date column")
            year_expression = (
                f'CAST("{year_column}" AS INTEGER)'
                if year_column in {"report_year", "year_frcast"}
                else f'EXTRACT(YEAR FROM TRY_CAST("{year_column}" AS DATE))'
            )
            predicates.append(f"{year_expression} IN ({', '.join('?' for _ in years)})")
            params.extend(years)
        if predicates:
            query += " WHERE " + " AND ".join(predicates)
        return conn.execute(query, params).fetchdf()
    finally:
        conn.close()


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

    if table in _V2_POD_SHOW_TABLES:
        df = _show_v2_pod_table(
            table,
            where=where,
            search=search,
            years=years_list,
            columns=columns_splitted,
        )
    else:
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
    """Replace selected timeseries years through canonical ingestion."""
    if table_name == "project_resources":
        raise ValueError(
            "project_resources must be ingested through the canonical SQLite "
            "resource ingestion path"
        )
    if table_name != "project_timeseries":
        raise ValueError(f"Unsupported append table: {table_name}")
    if not content or not append_years:
        return
    if any(
        isinstance(year, bool) or not isinstance(year, int) for year in append_years
    ):
        raise ValueError("append_years must contain integers")

    from esdc.fetch_publication import _ingest_timeseries_rows
    from esdc.pod_registry.store import get_esdc_sqlite_path

    names = validate_timeseries_header(header)
    year_index = names.index("report_year")
    selected = set(append_years)
    rows = []
    for index, raw in enumerate(content):
        if len(raw) != len(names):
            raise ValueError(f"project_timeseries record {index} has wrong width")
        year = parse_report_year(raw[year_index])
        if year in selected:
            values = cast(list[JSONScalar], list(raw))
            values[year_index] = year
            rows.append(values)
    if not rows:
        return
    _ingest_timeseries_rows(
        rows,
        names,
        duckdb_path=Config.get_db_file(),
        sqlite_path=get_esdc_sqlite_path(),
        replace_all=False,
        reindex=True,
        origin="api_json",
    )


def _fetch_and_parse_table(
    table: TableName,
    filetype: FileType,
    to_file: bool,
    username: str,
    password: str,
    report_year: int | None = None,
) -> (
    tuple[list[list[str]], list[str]]
    | tuple[list[list[JSONScalar]], list[str]]
    | list[Mapping[str, JSONScalar]]
    | None
):
    """Download and validate one table using the shared fetch parsers."""
    data, url = _download_table_bytes(table, filetype, username, password, report_year)
    try:
        result = _parse_table_payload(data, table, filetype, report_year)
    except FetchFailure:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise _fetch_failure(
            f"Unable to parse {table.value} response: {exc}",
            table,
            report_year,
            phase="parse",
        ) from exc
    except (ValueError, csv.Error, StopIteration) as exc:
        raise _fetch_failure(
            f"Invalid {table.value} response: {exc}",
            table,
            report_year,
            phase="validation",
        ) from exc

    if to_file:
        _save_download(table, filetype, data, report_year)
    if (
        table == TableName.PROJECT_TIMESERIES
        and isinstance(result, tuple)
        and not result[0]
    ):
        return None
    return result


def _fetch_failure(
    message: str,
    table: TableName | PodApiTable | str,
    report_year: int | None,
    *,
    phase: FetchPhase,
) -> FetchFailure:
    name = table.value if isinstance(table, (TableName, PodApiTable)) else str(table)
    scope = f" report year {report_year}" if report_year is not None else ""
    return FetchFailure(
        f"{name}{scope}: {message}",
        phase=phase,
        table=name,
        report_year=report_year,
    )


def _pod_validation_failure(
    exc: PodApiImportError, selected_years: tuple[int, ...] | None
) -> FetchFailure:
    """Keep the record index and the request scope that produced a batch failure."""
    reason = exc.reason
    if exc.row is not None:
        reason = f"record {exc.row}: {reason}"
    year: int | None = None
    if selected_years and exc.table_name == PodApiTable.POD_MONITORING.value:
        if len(selected_years) == 1:
            # One selected year is exactly the request scope the record came from.
            year = selected_years[0]
        else:
            # A combined batch cannot attribute a record index to one year reliably, so
            # the message names the years it covered instead of reporting a silent None.
            years = ", ".join(str(value) for value in selected_years)
            reason = f"{reason} across selected years {years}"
    return _fetch_failure(reason, exc.table_name, year, phase="validation")


def _download_table_bytes(
    table: TableName | PodApiTable,
    filetype: FileType,
    username: str,
    password: str,
    report_year: int | None,
) -> tuple[bytes, str]:
    # The five POD endpoints answer JSON only; the snapshot file format does not
    # change that, so a CSV request still asks for a JSON body here.
    requested_type = FileType.JSON if isinstance(table, PodApiTable) else filetype
    url = esdc_url_builder(
        table_name=table, file_type=requested_type, report_year=report_year
    )
    logging.info("downloading from %s", url)
    scope = f" (report year {report_year})" if report_year is not None else ""
    console.print(f"Fetching {table.value}{scope}...")
    data = esdc_downloader(url, username, password)
    if data is None:
        logging.warning("Failed to download %s data.", table.value)
        raise _fetch_failure(
            "download failed (no response payload)",
            table,
            report_year,
            phase="download",
        )
    return data, url


def _save_download(
    table: TableName | PodApiTable,
    filetype: FileType,
    data: bytes,
    report_year: int | None,
) -> None:
    # A POD body is saved as the exact bytes it answered, which are always JSON.
    requested_type = FileType.JSON if isinstance(table, PodApiTable) else filetype
    filename = table.value
    if report_year is not None:
        filename = f"{table.value}_{report_year}"
    save_path = Config.get_db_dir() / f"{filename}.{requested_type.value}"
    logging.debug("Save data as %s", save_path)
    save_payload_atomically(save_path, data)


def _json_object_without_duplicate_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _parse_timeseries_json(
    data: bytes, report_year: int | None
) -> tuple[list[list[JSONScalar]], list[str]]:
    parsed = json.loads(data, object_pairs_hook=_json_object_without_duplicate_keys)
    if not isinstance(parsed, list):
        raise ValueError("project_timeseries JSON must be an array of objects")
    if not parsed:
        return [], []
    first = parsed[0]
    if not isinstance(first, dict) or any(not isinstance(key, str) for key in first):
        raise ValueError(
            "project_timeseries record 0 must be an object with string keys"
        )
    header = validate_timeseries_header(tuple(first))
    expected_keys = set(header)
    content: list[list[JSONScalar]] = []
    for index, item in enumerate(parsed):
        if not isinstance(item, dict) or any(not isinstance(key, str) for key in item):
            raise ValueError(f"project_timeseries record {index} must be an object")
        if set(item) != expected_keys:
            missing = sorted(expected_keys - set(item))
            extra = sorted(set(item) - expected_keys)
            raise ValueError(
                f"project_timeseries record {index} keys differ from header; "
                f"missing={missing}, extra={extra}"
            )
        row: list[JSONScalar] = []
        for name in header:
            value = item[name]
            if value is not None and not isinstance(value, (str, int, float, bool)):
                raise ValueError(
                    f"project_timeseries record {index} column {name} must be scalar"
                )
            row.append(value)
        try:
            year = parse_report_year(row[header.index("report_year")])
        except ValueError as exc:
            raise ValueError(
                f"project_timeseries record {index} column report_year: {exc}"
            ) from exc
        row[header.index("report_year")] = year
        content.append(row)
    return content, list(header)


def _parse_table_payload(
    data: bytes,
    table: TableName,
    filetype: FileType,
    report_year: int | None,
    *,
    validate_resources: bool = True,
) -> (
    tuple[list[list[str]], list[str]]
    | tuple[list[list[JSONScalar]], list[str]]
    | list[Mapping[str, JSONScalar]]
):
    if filetype == FileType.CSV:
        content, header = _read_csv(io.StringIO(data.decode("utf-8"), newline=""))
        if table == TableName.PROJECT_TIMESERIES:
            names = validate_timeseries_header(header)
            for index, row in enumerate(content):
                if len(row) != len(names):
                    raise ValueError(
                        f"project_timeseries CSV record {index} has wrong width"
                    )
            year_index = names.index("report_year")
            for index, row in enumerate(content):
                try:
                    row[year_index] = str(parse_report_year(row[year_index]))
                except ValueError as exc:
                    raise ValueError(
                        f"project_timeseries record {index} column report_year: {exc}"
                    ) from exc
            return content, list(names)
        if table == TableName.PROJECT_RESOURCES and validate_resources:
            _csv_rows_to_normalized_project_resources(content, header)
        return content, header

    if table == TableName.PROJECT_TIMESERIES:
        content, header = _parse_timeseries_json(data, report_year)
        if not content:
            console.print(
                f"[yellow]API returned no rows for {table.value}"
                f"{f' (report year {report_year})' if report_year is not None else ''}"
                "; "
                "no data loaded from this response.[/yellow]"
            )
        return content, header

    parsed = json.loads(data, object_pairs_hook=_json_object_without_duplicate_keys)
    if table == TableName.PROJECT_RESOURCES:
        if validate_resources:
            _normalize_project_resources_json(parsed)
        if parsed == []:
            console.print(
                f"[yellow]API returned no rows for {table.value}; "
                "no data loaded from this response.[/yellow]"
            )
        return parsed
    raise ValueError(f"unsupported ESDC table {table.value}")


def prepare_fetch(
    *,
    filetype: FileType,
    to_file: bool,
    username: str,
    password: str,
    years: Sequence[int] | None,
    directory: Path,
    sqlite_path: Path | None = None,
) -> PreparedFetch:
    """Download and validate all fetch inputs without touching either database."""
    selected_years = None if years is None else tuple(sorted(set(years)))
    contract = load_pod_contract()
    target_path = (
        Path(sqlite_path) if sqlite_path is not None else get_esdc_sqlite_path()
    )
    # One read-only snapshot of the POD target, closed before any request is made.
    try:
        context = read_pod_api_context(target_path, contract)
    except PodImportError as exc:
        raise _fetch_failure(
            str(exc), PodApiTable.MASTER_DATA_POD, None, phase="preflight"
        ) from exc

    def canonical_project_id(value: str | None) -> str | None:
        if value is None:
            return None
        try:
            return normalize_project_id(value)
        except ValueError:
            # A stated alphanumeric identifier is preserved as given; only the
            # documented compact form is expanded for v2 storage here.
            return value

    resource_rows: list[ApiProjectResourceRow] = []
    resource_years: list[int] = []
    skipped_resource_years: list[int] = []

    requests: tuple[int | None, ...] = (
        (None,) if selected_years is None else selected_years
    )
    for year in requests:
        payload, _ = _download_table_bytes(
            TableName.PROJECT_RESOURCES, filetype, username, password, year
        )
        try:
            parsed = _parse_table_payload(
                payload,
                TableName.PROJECT_RESOURCES,
                filetype,
                year,
                validate_resources=False,
            )
            if filetype == FileType.JSON:
                if not isinstance(parsed, list):
                    raise ValueError("project_resources JSON must be an array")
                normalized = _normalize_project_resources_json(parsed)
            else:
                if not isinstance(parsed, tuple):
                    raise ValueError("project_resources CSV must be tabular")
                if not all(
                    isinstance(value, str) for row in parsed[0] for value in row
                ):
                    raise ValueError("project_resources CSV must contain text values")
                normalized = _csv_rows_to_normalized_project_resources(
                    cast(Sequence[Sequence[str]], parsed[0]), parsed[1]
                )
            normalized = tuple(
                replace(row, project_id=canonical_project_id(row.project_id))
                for row in normalized
            )
            if year is None:
                resource_rows.extend(normalized)
            else:
                scoped = [row for row in normalized if row.report_year == year]
                resource_rows.extend(scoped)
                # A requested year stays recorded even when it answers valid-empty.
                resource_years.append(year)
                if not scoped:
                    skipped_resource_years.append(year)
            if to_file:
                _save_download(TableName.PROJECT_RESOURCES, filetype, payload, year)
        except FetchFailure:
            raise
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise _fetch_failure(
                f"Unable to parse response: {exc}",
                TableName.PROJECT_RESOURCES,
                year,
                phase="parse",
            ) from exc
        except (ValueError, csv.Error, StopIteration) as exc:
            raise _fetch_failure(
                f"Invalid response: {exc}",
                TableName.PROJECT_RESOURCES,
                year,
                phase="validation",
            ) from exc

    timeseries_years = selected_years
    if timeseries_years is None:
        timeseries_years = tuple(
            sorted(
                {
                    row.report_year
                    for row in resource_rows
                    if isinstance(row.report_year, int)
                    and not isinstance(row.report_year, bool)
                }
            )
        )

    timeseries: list[PreparedTimeseriesSlice] = []
    skipped_timeseries_years: list[int] = []
    # Every requested year is observed because a failed request stops the batch;
    # a valid-empty year is recorded as skipped *and* as observed.
    observed_timeseries_years: list[int] = list(timeseries_years)
    for year in timeseries_years:
        payload, _ = _download_table_bytes(
            TableName.PROJECT_TIMESERIES, filetype, username, password, year
        )
        try:
            parsed = _parse_table_payload(
                payload, TableName.PROJECT_TIMESERIES, filetype, year
            )
            if not isinstance(parsed, tuple):
                raise ValueError("project_timeseries response must be tabular")
            rows, header = parsed
            if filetype == FileType.JSON and not rows:
                prepared = None
            else:
                prepared = spool_timeseries(
                    header,
                    rows,
                    report_year=year,
                    directory=directory,
                    origin="api_csv" if filetype == FileType.CSV else "api_json",
                )
            if prepared is None:
                skipped_timeseries_years.append(year)
            else:
                timeseries.append(prepared)
            if to_file:
                _save_download(TableName.PROJECT_TIMESERIES, filetype, payload, year)
            del parsed, rows, header
        except FetchFailure:
            raise
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise _fetch_failure(
                f"Unable to parse response: {exc}",
                TableName.PROJECT_TIMESERIES,
                year,
                phase="parse",
            ) from exc
        except (ValueError, csv.Error, StopIteration) as exc:
            raise _fetch_failure(
                f"Invalid response: {exc}",
                TableName.PROJECT_TIMESERIES,
                year,
                phase="validation",
            ) from exc
        finally:
            del payload

    # The masters and the approved forecast are requested once globally; monitoring
    # follows the deduplicated, sorted selection, or answers one full request when
    # no year is selected. `--year` never filters the forecast year.
    pod_requests: list[tuple[PodApiTable, int | None]] = [
        (table, None)
        for table in (
            PodApiTable.MASTER_DATA_PROJECT,
            PodApiTable.MASTER_DATA_POD,
            PodApiTable.POD_PROJECT,
        )
    ]
    monitoring_scope = (None,) if selected_years is None else selected_years
    pod_requests.extend((PodApiTable.POD_MONITORING, year) for year in monitoring_scope)
    pod_requests.append((PodApiTable.POD_FORECAST, None))

    bodies: dict[str, list[bytes]] = {table: [] for table in INPUT_TABLES}
    saved: list[tuple[PodApiTable, int | None, bytes]] = []
    for table, year in pod_requests:
        payload, _ = _download_table_bytes(
            table, FileType.JSON, username, password, year
        )
        try:
            parse_pod_api_payload(payload, table_name=table.value, contract=contract)
        except PodApiPayloadError as exc:
            raise _fetch_failure(
                f"Invalid response: {exc.reason}", table, year, phase="parse"
            ) from exc
        bodies[table.value].append(payload)
        saved.append((table, year, payload))

    try:
        batch = prepare_pod_api_batch(
            bodies,
            existing_rows=context.existing_rows,
            reserved_pod_identities=context.reserved_pod_identities,
            monitoring_years=selected_years,
            contract=contract,
        )
    except PodApiImportError as exc:
        raise _pod_validation_failure(exc, selected_years) from exc
    batch = replace(batch, target_basis=context.target_basis)

    # Linked POD documents must be covered by the prepared masters. Master data is a
    # global replace, so the prepared rows are the whole final scope: a stored row the
    # new source omits cannot rescue its link. An orphan link aborts here rather than
    # guessing a numeric POD identity.
    prepared_sequences = {
        int(str(row.get("pod_id_seq")))
        for row in batch.rows[PodApiTable.MASTER_DATA_POD.value]
        if row.get("pod_id_seq") is not None
    }
    orphaned = sorted(set(context.linked_pod_ids) - prepared_sequences)
    if orphaned:
        raise _fetch_failure(
            "linked POD documents reference POD sequences the prepared masters do not "
            f"cover: {', '.join(str(sequence) for sequence in orphaned)}; they must be "
            "reconciled by an explicit crosswalk, never inferred",
            PodApiTable.MASTER_DATA_POD,
            None,
            phase="preflight",
        )

    # Nothing is saved until the whole batch and its links validate.
    if to_file:
        for table, year, payload in saved:
            _save_download(table, FileType.JSON, payload, year)

    return PreparedFetch(
        tuple(resource_rows),
        None if selected_years is None else tuple(resource_years),
        tuple(timeseries),
        tuple(skipped_resource_years),
        tuple(skipped_timeseries_years),
        timeseries_observed_years=tuple(sorted(observed_timeseries_years)),
        timeseries_complete_observation=(
            selected_years is None and bool(timeseries_years)
        ),
        pod_batch=batch,
        resources_observed=True,
    )


def _commit_and_publish_project_resources(
    payload: Sequence[Mapping[str, object]],
    *,
    years: Sequence[int] | None = None,
) -> ProjectPublication:
    incoming = _normalize_project_resources_json(list(payload))
    return _commit_normalized_project_resources(incoming, years=years)


def _commit_normalized_project_resources(
    rows: Sequence[ApiProjectResourceRow], *, years: Sequence[int] | None = None
) -> ProjectPublication:
    """Persist one already-normalized project-resource batch and publish once."""
    if not rows:
        raise ValueError("project resources response contains no rows")
    selected_years = tuple(dict.fromkeys(years or ()))
    incoming = tuple(rows)
    if years is not None:
        if not selected_years:
            raise ValueError("at least one report year is required")
        incoming = tuple(row for row in incoming if row.report_year in selected_years)
        if not incoming:
            raise ValueError(
                "project resources response contains no selected-year rows"
            )
    sqlite_path = get_esdc_sqlite_path()
    outcome = publish_prepared_fetch(
        PreparedFetch(
            incoming,
            None if years is None else selected_years,
            (),
            (),
            (),
        ),
        duckdb_path=Config.get_db_file(),
        sqlite_path=sqlite_path,
        reindex=True,
    )
    if (
        outcome.resource_revision is None
        or outcome.resource_source_sha256 is None
        or outcome.resource_invalid_geometry_count is None
    ):
        raise RuntimeError("project resources publication returned no revision")
    return ProjectPublication(
        outcome.resource_revision,
        outcome.resource_row_count,
        outcome.resource_source_sha256,
        outcome.resource_invalid_geometry_count,
    )


def _normalize_project_resources_json(
    payload: object,
) -> tuple[ApiProjectResourceRow, ...]:
    """Validate a raw JSON response before normalizing it."""
    if not isinstance(payload, list):
        raise ValueError("project_resources JSON must be a list of objects")
    scalar_rows: list[Mapping[str, JSONScalar]] = []
    for index, item in enumerate(payload):
        if not isinstance(item, Mapping):
            raise ValueError(f"record {index} must be an object")
        scalar_row: dict[str, JSONScalar] = {}
        for key, value in item.items():
            if not isinstance(key, str):
                raise ValueError(f"record {index} keys must be strings")
            if value is not None and not isinstance(value, (str, int, float, bool)):
                raise ValueError(f"record {index} field {key!r} must be a JSON scalar")
            scalar_row[key] = value
        if "report_year" not in scalar_row:
            raise ValueError(f"record {index} column report_year is required")
        try:
            scalar_row["report_year"] = parse_report_year(scalar_row["report_year"])
        except ValueError as exc:
            raise ValueError(f"record {index} column report_year: {exc}") from exc
        scalar_rows.append(scalar_row)
    return normalize_project_resources(scalar_rows).rows


def _hash_normalized_rows(rows: Sequence[ApiProjectResourceRow]) -> str:
    encoded = sorted(
        json.dumps(
            tuple(
                getattr(row, field)
                for field in ApiProjectResourceRow.__dataclass_fields__
            ),
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        for row in rows
    )
    return hashlib.sha256(b"[" + b",".join(encoded) + b"]").hexdigest()


def refresh_project_resources_from_api(
    *, username: str, password: str, report_year: int | None
) -> ProjectPublication:
    """Fetch, normalize, commit, and publish project resources from the API."""
    payload, _ = _download_table_bytes(
        TableName.PROJECT_RESOURCES,
        FileType.JSON,
        username,
        password,
        report_year,
    )
    try:
        parsed = _parse_table_payload(
            payload,
            TableName.PROJECT_RESOURCES,
            FileType.JSON,
            report_year,
            validate_resources=False,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise _fetch_failure(
            f"Unable to parse response: {exc}",
            TableName.PROJECT_RESOURCES,
            report_year,
            phase="parse",
        ) from exc
    if not isinstance(parsed, list):
        raise RuntimeError("project resources API returned no JSON rows")
    rows = _normalize_project_resources_json(parsed)
    if not rows:
        raise RuntimeError("project resources API returned no JSON rows")
    selected = None if report_year is None else (report_year,)
    if selected is not None:
        rows = tuple(row for row in rows if row.report_year == report_year)
        if not rows:
            raise RuntimeError(
                "project resources response contains no selected-year rows"
            )
    return _commit_normalized_project_resources(rows, years=selected)


def rebuild_project_resources_projection() -> ProjectPublication:
    """Retry publication of the latest committed or stale SQLite revision."""
    sqlite_path = get_esdc_sqlite_path()
    conn = get_sqlite_connection(sqlite_path)
    try:
        snapshot = get_project_resources_snapshot(conn)
    finally:
        conn.close()
    if snapshot is None:
        raise RuntimeError("no project resources snapshot is available")
    if snapshot.publication_state not in ("committed", "analytics_stale", "published"):
        raise RuntimeError(
            "project resources revision "
            f"{snapshot.revision} is already {snapshot.publication_state}"
        )
    duckdb_path = Config.get_db_file()
    from esdc.fetch_publication import ensure_timeseries_baseline

    try:
        ensure_timeseries_baseline(duckdb_path, sqlite_path, allow_install=False)
    except ValueError as exc:
        raise ValueError(
            f"project_timeseries is pending or differs from SQLite; "
            f"run 'esdc rebuild-api-analytics' before resources-only recovery: {exc}"
        ) from exc
    return publish_project_resources(
        duckdb_path, sqlite_path, snapshot.revision, allow_install=False
    )


@app.command(name="rebuild-project-resources")
def rebuild_project_resources_cmd() -> None:
    """Rebuild DuckDB project-resource analytics from SQLite without refetching."""
    try:
        publication = rebuild_project_resources_projection()
    except Exception as exc:
        typer.echo(f"Unable to rebuild project resources: {exc}")
        raise typer.Exit(1) from None
    typer.echo(
        f"Published project resources revision {publication.revision} "
        f"({publication.row_count:,} rows; "
        f"invalid geometry: {publication.invalid_geometry_count})."
    )


def load_esdc_data(
    filetype: FileType = FileType.CSV,
    to_file: bool = True,
    reload: bool = True,
    username: str = "",
    password: str = "",
    years: list[int] | None = None,
    reindex: bool = True,
) -> FetchOutcome:
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
    selected_years = None if years is None else tuple(sorted(set(years)))
    with TemporaryDirectory(prefix="esdc-fetch-") as temporary_directory:
        prepared = prepare_fetch(
            filetype=filetype,
            to_file=to_file,
            username=username,
            password=password,
            years=selected_years,
            directory=Path(temporary_directory),
        )
        if not reload:
            console.print(
                "[green]Fetch payloads validated; no database changes.[/green]"
            )
            # The gate needs a committed source; this route stopped before it.
            console.print("POD source generation: none created.")
            console.print(
                "Nothing committed to SQLite and the mirror gate was not touched."
            )
            return FetchOutcome(
                None,
                None,
                0,
                None,
                (),
                0,
                (),
                prepared.skipped_resource_years,
                prepared.skipped_timeseries_years,
            )
        outcome = publish_prepared_fetch(
            prepared,
            duckdb_path=Config.get_db_file(),
            sqlite_path=get_esdc_sqlite_path(),
            reindex=reindex,
        )
    if (
        outcome.resource_revision is None
        and not outcome.updated_timeseries_years
        and not outcome.updated_resource_years
    ):
        if outcome.skipped_resource_years:
            skipped = ", ".join(map(str, outcome.skipped_resource_years))
            console.print(
                "[yellow]No data changes; No project-resource rows for report year "
                f"{skipped}; existing data was retained.[/yellow]"
            )
        else:
            console.print(
                "[yellow]No data changes; existing data was retained.[/yellow]"
            )
        return outcome
    if outcome.updated_resource_years:
        resource_year_label = ", ".join(map(str, outcome.updated_resource_years))
    elif outcome.resource_revision is None:
        resource_year_label = "unchanged"
    else:
        resource_year_label = "full snapshot"
    invalid_geometry_label = (
        outcome.resource_invalid_geometry_count
        if outcome.resource_invalid_geometry_count is not None
        else "unchanged"
    )
    console.print(
        "[green]Fetch published successfully.[/green] "
        f"Resources: {resource_year_label}; "
        f"revision={outcome.resource_revision or 'unchanged'}; "
        f"DuckDB publication revision={outcome.resource_revision or 'unchanged'}; "
        f"rows={outcome.resource_row_count}; "
        f"sha256={outcome.resource_source_sha256 or 'unchanged'}; "
        "invalid_geometry="
        f"{invalid_geometry_label}."
    )
    console.print(
        "Timeseries: updated years "
        f"{', '.join(map(str, outcome.updated_timeseries_years)) or 'none'}; "
        f"rows={outcome.timeseries_rows_written}; skipped years "
        f"{', '.join(map(str, outcome.skipped_timeseries_years)) or 'none'}."
    )
    console.print(
        "Resources: skipped years "
        f"{', '.join(map(str, outcome.skipped_resource_years)) or 'none'}."
    )
    if outcome.pod_source_generation is not None:
        # The combined gate is read from committed SQLite truth, not from this call's
        # flags: the pair and the POD projection must both agree before freshness is
        # claimed, and a pending part is reported separately from the committed source.
        gate = get_fetch_publication_status(
            sqlite_path=get_esdc_sqlite_path(), duckdb_path=Config.get_db_file()
        )
        console.print(
            f"POD source generation {outcome.pod_source_generation}: SQLite committed."
        )
        for component in gate.components:
            console.print(f"Gate {component.name}: {component.status}")
        for component in gate.components:
            if component.status == "published" and component.rows_written == 0:
                console.print(
                    f"{component.name}: observed-empty snapshot published."
                )
            elif component.status == "absent":
                console.print(f"{component.name}: never observed.")
        if gate.fresh:
            console.print(
                f"Whole fetch is fresh for generation {outcome.pod_source_generation}."
            )
        else:
            unpublished = [
                f"{component.name} ({component.status})"
                for component in gate.components
                if component.status != "published"
            ]
            generation_label = (
                f" for generation {outcome.pod_source_generation}"
                if outcome.pod_source_generation is not None
                else ""
            )
            console.print(
                f"Whole fetch is not fresh{generation_label}: "
                f"{', '.join(unpublished)}."
            )
            if gate.recovery_command is not None:
                console.print(f"Run '{gate.recovery_command}' to complete the mirror.")
            else:
                console.print(
                    "Fetch or load the missing source data before rebuilding."
                )
    return outcome


def esdc_url_builder(
    table_name: TableName | PodApiTable,
    api_ver: ApiVer | None = None,
    verbose: int = 3,
    report_year: int | None = None,
    file_type: FileType = FileType.CSV,
) -> str:
    """
    Build an ESDC URL based on the provided parameters.

    Parameters
    ----------
    table_name : TableName | PodApiTable
        The table name to query.
    api_ver : ApiVer | None, optional
        The API version to use. Defaults to V1 for a POD table and V2 otherwise;
        an explicit version is always honored.
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

    A POD table uses its configured relative endpoint path under the API version
    and always requests JSON output, so a CSV snapshot format does not change it:
    https://esdc.skkmigas.go.id/api/v1/master-data-pod?verbose=3&output=json
    """
    pod_table = isinstance(table_name, PodApiTable)
    default_version = ApiVer.V1 if pod_table else ApiVer.V2
    version = api_ver if api_ver is not None else default_version
    url = Config.get_api_url().rstrip("/") + version.value
    tables = {
        TableName.PROJECT_RESOURCES: "project-resources",
        TableName.PROJECT_TIMESERIES: "project-timeseries",
    }
    if pod_table:
        segment = Config.get_pod_endpoint(table_name.value)
    else:
        segment = tables[table_name]
    url = f"{url}/{segment}?verbose={verbose}"

    # TODO this is temporary fix since as of 2024-07-06
    # the API for time series does not support all year selection
    # remove this conditional if the API for project_timeseries is fixed.
    if report_year is not None:
        url = f"{url}&report-year={report_year}"
    # POD endpoints answer JSON only, whatever the snapshot file format is.
    output = FileType.JSON if pod_table else file_type
    url = f"{url}&output={output.value}"

    return url


def _load_file_as_csv(file: str, table_name):
    with open(file, "rb") as f:
        data = f.read()
    content, header = _read_csv(io.StringIO(data.decode("utf-8")))
    if table_name in (TableName.PROJECT_RESOURCES, TableName.PROJECT_RESOURCES.value):
        rows = _csv_rows_to_normalized_project_resources(content, header)
        if rows:
            _commit_normalized_project_resources(rows)
        else:
            console.print("[yellow]No project-resource rows; no data loaded.[/yellow]")
        return
    if table_name in (TableName.PROJECT_TIMESERIES, TableName.PROJECT_TIMESERIES.value):
        if not content:
            return
        from esdc.fetch_publication import _ingest_timeseries_rows
        from esdc.pod_registry.store import get_esdc_sqlite_path

        _ingest_timeseries_rows(
            content,
            header,
            duckdb_path=Config.get_db_file(),
            sqlite_path=get_esdc_sqlite_path(),
            replace_all=True,
            reindex=True,
            origin="api_csv",
        )
        return
    load_data_to_db(content, list(header), table_name)


def _csv_rows_to_normalized_project_resources(
    content: Sequence[Sequence[str]], header: Sequence[str]
) -> tuple[ApiProjectResourceRow, ...]:
    names = list(header)
    if len(names) != len(set(names)):
        raise ValueError("project_resources CSV contains duplicate headers")
    fields = set(ApiProjectResourceRow.__dataclass_fields__)
    unknown = sorted(set(names) - _ALLOWED_SOURCE_FIELDS - fields)
    if unknown:
        raise ValueError(
            "project_resources CSV contains unknown headers: " + ", ".join(unknown)
        )
    raw_grov = [name for name in names if name.startswith("grov_")]
    if "groovy_report" in names and raw_grov:
        raise ValueError(
            "project_resources CSV cannot combine raw GROV fields with groovy_report"
        )
    rows: list[ApiProjectResourceRow] = []
    for index, values in enumerate(content):
        if len(values) != len(names):
            raise ValueError(f"project_resources CSV row {index} has wrong width")
        data: dict[str, JSONScalar] = {
            name: (value if value != "" else None)
            for name, value in zip(names, values, strict=True)
        }
        report_year = data.get("report_year")
        if report_year is None:
            raise ValueError(
                f"project_resources CSV row {index} column report_year is required"
            )
        try:
            data["report_year"] = parse_report_year(report_year)
        except ValueError as exc:
            raise ValueError(
                f"project_resources CSV row {index} column report_year: {exc}"
            ) from exc
        if raw_grov:
            raw = {
                key: value
                for key, value in data.items()
                if key in _ALLOWED_SOURCE_FIELDS
                and key not in {"basin86", "basin86_id"}
            }
            normalized = normalize_project_resources([raw]).rows[0]
            overlays = {
                key: value
                for key, value in data.items()
                if key in fields and key not in raw_grov
            }
            rows.append(replace(normalized, **overlays) if overlays else normalized)
            continue
        project_id = data.pop("project_id", None)
        if project_id is not None and not isinstance(project_id, str):
            raise ValueError("project_id must be a string or null")
        rows.append(ApiProjectResourceRow(project_id=project_id, **data))
    return tuple(rows)


def _csv_rows_to_project_resources(
    content: Sequence[Sequence[str]], header: Sequence[str]
) -> list[dict[str, JSONScalar]]:
    """Compatibility adapter returning raw mappings for existing callers."""
    rows = _csv_rows_to_normalized_project_resources(content, header)
    return [
        {
            field: getattr(row, field)
            for field in ApiProjectResourceRow.__dataclass_fields__
        }
        for row in rows
    ]


def _load_file_as_json(file: str, table_name):
    with open(file, "rb") as f:
        data = f.read()
    parsed_json = json.loads(
        data, object_pairs_hook=_json_object_without_duplicate_keys
    )
    if table_name in (TableName.PROJECT_RESOURCES, TableName.PROJECT_RESOURCES.value):
        _normalize_project_resources_json(parsed_json)
        if parsed_json:
            _commit_and_publish_project_resources(parsed_json)
        else:
            console.print("[yellow]No project-resource rows; no data loaded.[/yellow]")
        return
    if table_name in (TableName.PROJECT_TIMESERIES, TableName.PROJECT_TIMESERIES.value):
        content, header = _parse_timeseries_json(data, None)
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
        return
    if not parsed_json:
        return
    header = tuple(parsed_json[0].keys())
    if any(set(item) != set(header) for item in parsed_json):
        raise ValueError("project_timeseries JSON records must have identical keys")
    content = [[item[key] for key in header] for item in parsed_json]
    load_data_to_db(content, list(header), table_name)


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
def chat():
    """Start the interactive chat TUI."""
    from esdc.configs import Config

    if not Config.has_chat_config():
        rich.print(
            "[bold yellow]No provider configured.[/bold yellow] "
            "Run '[cyan]esdc configs[/cyan]' first."
        )
        return

    from esdc.chat.app import ESDCChatApp

    app_ = ESDCChatApp()
    app_.run()


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
        return f"{n / (1024**2):.1f} MB"
    return f"{n / (1024**3):.1f} GB"


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
        _print_api_source_status(db_file, get_esdc_sqlite_path())
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

        conn = get_duckdb_connection(db_file, allow_install=False)
        try:
            table_stats = check_table_stats(conn)
            last_updated = get_last_updated(conn)
        finally:
            conn.close()
    except Exception as e:
        _print_api_source_status(db_file, get_esdc_sqlite_path())
        rich.print(f"[yellow]Could not check indexes: {e}[/yellow]")
        return False

    if last_updated:
        rich.print(f"[bold]Last updated:[/bold] {last_updated}")
    else:
        from datetime import datetime

        mtime = datetime.fromtimestamp(db_file.stat().st_mtime)
        rich.print(f"[bold]Last updated:[/bold] {mtime:%Y-%m-%d %H:%M:%S} (file mtime)")

    api_status = _print_api_source_status(db_file, get_esdc_sqlite_path())

    rich.print()
    rich.print("[bold]Tables:[/bold]")
    for ts in table_stats:
        if not ts.get("exists", bool(ts["years"])):
            icon = "[red]❌[/red]"
            rich.print(f"  {icon} {ts['table']}: not loaded")
            continue
        if ts["table"] == "project_timeseries" and ts["total"] == 0:
            state = api_status.get("project_timeseries", {}).get("status", "unverified")
            icon = "[green]✅[/green]" if state == "published" else "[yellow]⚠[/yellow]"
            label = (
                "observed empty; published"
                if state == "published"
                else f"0 rows; {state}"
            )
            rich.print(f"  {icon} {ts['table']}: {label}")
            continue
        if not ts["years"]:
            rich.print(f"  [yellow]⚠[/yellow] {ts['table']}: 0 rows")
            continue
        icon = "[green]✅[/green]"
        rich.print(f"  {icon} {ts['table']} ({ts['total']:,} rows):")
        for year, count in ts["years"]:
            rich.print(f"      {year}: {count:,} rows")

    return True


def _print_api_source_status(db_file: Path, sqlite_path: Path):
    """Report both API snapshots and DuckDB markers using read-only connections."""
    from esdc.analytics_rebuild import ApiSourceStatus, get_api_analytics_status

    try:
        sources: dict[str, ApiSourceStatus] = get_api_analytics_status(
            duckdb_path=db_file, sqlite_path=sqlite_path
        )
    except Exception as exc:
        rich.print(f"[red]Could not check API source status: {exc}[/red]")
        return {}
    rich.print()
    rich.print("[bold]API source and analytical lineage:[/bold]")
    for table, report in sources.items():
        snapshot = report["snapshot"]
        lineage = report["lineage"]
        rich.print(f"  {table}: {report['status']} — {report['detail']}")
        if snapshot is None:
            rich.print("    SQLite: snapshot absent")
        else:
            rich.print(
                "    SQLite: "
                f"revision={snapshot['revision']}, rows={snapshot['row_count']}, "
                f"sha256={snapshot['source_sha256']}, "
                f"state={snapshot['publication_state']}"
            )
        if lineage is None:
            rich.print("    DuckDB: lineage absent")
        else:
            rich.print(
                "    DuckDB: "
                f"revision={lineage['revision']}, "
                f"rows={lineage['source_row_count']}, "
                f"sha256={lineage['source_sha256']}"
            )
            if table == "project_resources":
                rich.print(f"    Invalid geometry: {lineage['invalid_geometry_count']}")
    from esdc.knowledge.freshness import knowledge_freshness_markers
    from esdc.pod_registry.publication import get_pod_registry_status
    from esdc.pod_registry.store import get_sqlite_readonly_connection

    source = get_sqlite_readonly_connection(sqlite_path)
    if source is not None:
        try:
            is_v2 = (
                source.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' "
                    "AND name='pod_registry_state'"
                ).fetchone()
                is not None
            )
            if is_v2:
                registry = get_pod_registry_status(sqlite_path, db_file)
                knowledge = knowledge_freshness_markers(source)
                rich.print()
                rich.print("[bold]POD publication and knowledge:[/bold]")
                rich.print(
                    f"  Business projection: {registry.state} — "
                    f"{registry.message or 'generation markers agree'}"
                )
                rich.print(
                    "  Knowledge: "
                    f"{'fresh' if knowledge.fresh else 'stale'} — "
                    f"source generation={knowledge.source_generation}, "
                    f"knowledge generation={knowledge.knowledge_generation}"
                )
        finally:
            source.close()
    return sources


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
        conn = get_duckdb_connection(db_file, allow_install=False)
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

        conn = get_duckdb_connection(db_file, allow_install=False)
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
        conn = get_duckdb_connection(db_file, allow_install=False)
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
                conn.execute("SELECT COUNT(*) FROM document_chunks").fetchone() or (0,)
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
        conn = get_duckdb_connection(db_file, allow_install=False)
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
    port: int = typer.Option(3334, "--port", "-p", help="Server port"),
    host: str = typer.Option("0.0.0.0", "--host", "-h", help="Server host"),
    log_level: str = typer.Option("info", "--log-level", help="Uvicorn log level"),
) -> None:
    """Start OpenAI-compatible API server.

    Provides an OpenAI-compatible API for the ESDC agent so tools like
    OpenWebUI can connect to ESDC as an external provider.

    Args:
        port: Port to run the server on (default: 3334)
        host: Host to bind the server to (default: 0.0.0.0)
        log_level: Uvicorn log level (default: info)
    """
    from esdc.server.app import run_server

    rich.print(f"[bold green]Starting ESDC server on http://{host}:{port}[/bold green]")
    rich.print(f"[dim]API documentation available at http://{host}:{port}/docs[/dim]")
    run_server(host=host, port=port, log_level=log_level)


@app.command(name="portal")
def portal(
    port: int = typer.Option(13334, "--port", "-p", help="Portal port"),
    host: str = typer.Option("127.0.0.1", "--host", "-h", help="Portal host"),
    log_level: str = typer.Option("info", "--log-level", help="Log level"),
) -> None:
    """Launch the POD registry portal (Excel-like master data editor)."""
    from esdc.portal.app import run_portal

    rich.print(f"[bold green]Starting POD portal on http://{host}:{port}/[/bold green]")
    run_portal(host=host, port=port, log_level=log_level)


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
        typer.echo(f"Error: --topic must be one of {', '.join(DOC_TOPICS)}.", err=True)
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

    if level is not None and rule is not None and level != rule[2]:
        kind, key, implied_level = rule
        typer.echo(
            f"Error: --level {level} conflicts with the {kind} '{key}' rule "
            f"(implies {implied_level}).",
            err=True,
        )
        raise typer.Exit(1)


def _open_corpus_store(*, read_only: bool = True, operation: str = "document"):
    """Open a CorpusStore for a CLI read/write, or exit 1 with a clear error.

    Reader mode (the default) validates the committed corpus schema without
    creating or migrating anything; writer mode is an explicit opt-in that
    repairs/initializes the schema. Query commands must NOT pass
    ``read_only=False`` merely to inherit automatic migrations.

    Reader validation is operation-specific (see ``validate_readiness``) so a
    command only requires the tables/columns it actually reads -- e.g.
    ``corpus list`` validates ``list`` (documents + the chunk-count join)
    rather than the narrow ``document`` lookup shape.
    """
    from esdc.corpus.store import CorpusStore

    if read_only:
        store = CorpusStore(read_only=True)
        try:
            store.validate_readiness(operation)
        except ValueError as e:
            store.close()
            typer.echo(f"Error: {e}", err=True)
            raise typer.Exit(1) from None
        return store

    store = CorpusStore(read_only=False)
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

    _validate_corpus_overrides(
        level, doc_type, topic, wk_name, field_name, project_name
    )

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


@corpus_app.command(name="rename")
def corpus_rename(
    paths: Annotated[
        list[Path],
        typer.Argument(
            exists=True, help="Source file(s) (.pdf, .docx, .md) or folder(s)."
        ),
    ],
    doc_type: Annotated[
        str | None, typer.Option("--doc-type", help="Override doc_type.")
    ] = None,
    yes: Annotated[
        bool,
        typer.Option("--yes", help="Apply the renames (default: dry-run preview)."),
    ] = False,
) -> None:
    """Rename sources to `DOC_TYPE - YYYY.MM.DD - title.<ext>` (dry-run by default).

    Resolves doc_type/doc_date/title from an existing .corpus.md sidecar, the
    committed corpus DB, or LLM/OCR inference (filename passed as a hint).
    Renames the matching .corpus.md sidecar in lockstep. Preview only unless
    --yes is given.
    """
    from esdc.corpus.rename import run_rename

    _validate_corpus_overrides(None, doc_type)

    report, plans = run_rename(paths, doc_type=doc_type, apply=yes)

    table = [
        (
            p.src.name,
            "->",
            p.new_path.name if p.new_path is not None else "",
            p.doc_type or "",
            p.doc_date or "",
            p.title or "",
            p.source,
            p.note,
        )
        for p in plans
    ]
    headers = [
        "file",
        "",
        "new name",
        "doc_type",
        "doc_date",
        "title",
        "source",
        "note",
    ]
    rich.print(tabulate(table, headers=headers, tablefmt="psql"))

    if not yes:
        typer.echo("Dry run — re-run with --yes to apply.")
        return
    _print_corpus_report(report)


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
    embed_backend: Annotated[
        str | None,
        typer.Option(
            "--embed-backend",
            help="Override embedding_backend for this run: local, ollama or openai.",
        ),
    ] = None,
) -> None:
    """Ingest reviewed .corpus.md sidecars into the searchable corpus (step 2 of 2)."""
    from esdc.corpus.pipeline import run_commit

    try:
        report = run_commit(
            paths,
            skip_review=skip_review,
            force=force,
            dry_run=dry_run,
            embed_backend=embed_backend,
        )
    except (ValueError, RuntimeError) as e:
        # e.g. embedding-model mismatch -> `corpus reembed`
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1) from None
    _print_corpus_report(report)


@corpus_app.command()
def learn(
    force: Annotated[
        bool,
        typer.Option("--force", help="Reprocess all docs and rebuild dossiers."),
    ] = False,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Report what would be processed; no writes."),
    ] = False,
    limit: Annotated[
        int | None,
        typer.Option("--limit", help="Process at most N documents (smoke runs)."),
    ] = None,
    init_guideline: Annotated[
        bool,
        typer.Option(
            "--init-guideline",
            help="Draft ~/.esdc/guideline.yaml from the corpus via LLM, "
            "then exit. Review it before running learn.",
        ),
    ] = False,
) -> None:
    """Reconstruct the knowledge graph from the committed corpus (eager)."""
    from esdc.knowledge import learn as learn_mod

    if init_guideline:
        from esdc.knowledge.bootstrap import init_guideline as init_fn

        try:
            path = init_fn(force=force)
        except (ValueError, FileExistsError, FileNotFoundError) as e:
            typer.echo(f"Error: {e}", err=True)
            raise typer.Exit(1) from None
        rich.print(f"[green]Guideline draft written:[/green] {path}")
        rich.print("Review and edit it, then run: [cyan]esdc corpus learn[/cyan]")
        return

    try:
        report = learn_mod.run_learn(force=force, dry_run=dry_run, limit=limit)
    except (ValueError, FileNotFoundError) as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1) from None

    if report.dry_run:
        rich.print(
            f"[yellow]Dry run:[/yellow] would process "
            f"{report.docs_processed}/{report.docs_total} documents "
            f"({report.docs_skipped} up to date)."
        )
        return

    rich.print("[bold]Corpus learn complete[/bold]")
    rich.print(
        f"  Documents: {report.docs_processed} processed, "
        f"{report.docs_skipped} skipped, {report.docs_failed} failed "
        f"(of {report.docs_total})"
    )
    rich.print(
        f"  Edges: {report.edges_written} written "
        f"({report.pod_document_added} pod_document links promoted)"
    )
    rich.print(
        f"  Claims: {report.claims_written} "
        f"| unresolved mentions: {report.unresolved_mentions}"
    )
    rich.print(
        f"  Dossiers: {report.dossiers_built} built, "
        f"{report.dossiers_skipped} up to date"
    )
    if report.proposals_pending:
        rich.print(
            f"[yellow]  {report.proposals_pending} schema proposal(s) pending "
            f"— review with: esdc corpus proposals[/yellow]"
        )
    if report.docs_failed:
        rich.print(
            f"[red]  {report.docs_failed} document(s) failed — "
            f"rerun 'esdc corpus learn' to retry.[/red]"
        )


@corpus_app.command()
def proposals() -> None:
    """List pending schema proposals discovered by `esdc corpus learn`."""
    from esdc.knowledge.store import KnowledgeStore
    from esdc.pod_registry.store import (
        get_sqlite_readonly_connection,
        is_missing_table_error,
    )

    conn = get_sqlite_readonly_connection()
    if conn is None:
        props = []
    else:
        try:
            store = KnowledgeStore(conn)
            try:
                props = store.pending_proposals()
            except sqlite3.OperationalError as exc:
                if not is_missing_table_error(exc):
                    raise
                props = []
        finally:
            conn.close()
    if not props:
        rich.print("No pending proposals.")
        return
    for p in props:
        rich.print(
            f"[cyan]{p['proposal_type']}[/cyan] {p['name']} "
            f"— seen in {p['evidence_count']} doc(s): "
            f"{', '.join(p['sample_doc_ids'][:5])}"
        )
    rich.print(
        "\nTo adopt a proposal, add it to esdc/knowledge/guideline.yaml "
        "and rerun: esdc corpus learn"
    )


@corpus_app.command(name="sync")
def corpus_sync() -> None:
    """Rebuild DuckDB projections from the existing SQLite source."""
    from esdc.configs import Config
    from esdc.corpus.mirror import refresh_all
    from esdc.dbmanager import get_duckdb_connection
    from esdc.pod_registry.store import get_esdc_sqlite_path

    database_path = Config.get_db_file()
    database_path.parent.mkdir(parents=True, exist_ok=True)
    conn = get_duckdb_connection(
        database_path,
        read_only=False,
        allow_install=False,
    )
    try:
        report = refresh_all(conn, get_esdc_sqlite_path())
        typer.echo(f"documents mirrored: {report.documents}")
        if report.orphan_chunks:
            typer.echo(f"orphan chunks removed: {report.orphan_chunks}")
        for table, n in sorted(report.registry.items()):
            typer.echo(f"  {table}: {n}")
        typer.echo(f"views: {', '.join(report.views)}")
    finally:
        conn.close()


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
        typer.Option("--level", help="Set doc_level: wk, field, project, regulation."),
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
    regenerate: Annotated[
        bool,
        typer.Option(
            "--regenerate",
            help=(
                "Re-run LLM metadata analysis on each sidecar's existing body "
                "(requires a reachable metadata_model); resets reviewed to "
                "false unless --reviewed/--no-reviewed is also passed."
            ),
        ),
    ] = False,
) -> None:
    """Show or bulk-set .corpus.md frontmatter metadata.

    Without flags: print a metadata table. With flags: write the values
    into each sidecar's frontmatter (re-running entity resolution).
    """
    from esdc.corpus.pipeline import run_meta, run_meta_show

    _validate_corpus_overrides(
        level, doc_type, topic, wk_name, field_name, project_name
    )

    values = (level, doc_type, topic, wk_name, field_name, project_name, reviewed)
    if not regenerate and all(v is None for v in values):
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
            "file",
            "doc_type",
            "topic",
            "doc_date",
            "doc_level",
            "entity",
            "reviewed",
            "note",
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
            regenerate=regenerate,
        )
    except ValueError as e:  # e.g. --wk-name value not in canonical tables, or
        # --regenerate with no reachable metadata_model
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
    store = _open_corpus_store(operation="list")
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
        "doc_id",
        "file_name",
        "doc_type",
        "doc_date",
        "doc_level",
        "entity",
        "n_chunks",
    ]
    rich.print(tabulate(rows, headers=headers, tablefmt="psql"))


@corpus_app.command()
def remove(
    doc_ids: Annotated[
        list[str] | None, typer.Argument(help="Document ID(s) to remove.")
    ] = None,
    doc_type: Annotated[
        str | None, typer.Option("--doc-type", help="Remove all docs of this type.")
    ] = None,
    year: Annotated[
        int | None, typer.Option("--year", help="Filter by document year.")
    ] = None,
    wk_name: Annotated[
        str | None, typer.Option("--wk-name", help="Filter by working area name.")
    ] = None,
    field_name: Annotated[
        str | None, typer.Option("--field-name", help="Filter by field name.")
    ] = None,
    project_name: Annotated[
        str | None, typer.Option("--project-name", help="Filter by project name.")
    ] = None,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="List matches without deleting.")
    ] = False,
    yes: Annotated[
        bool, typer.Option("--yes", help="Confirm filter-based deletion.")
    ] = False,
) -> None:
    """Remove document(s) from the corpus (does not touch files on disk).

    Sidecars are untouched: a later `esdc corpus commit` re-ingests them.
    """
    filters = {
        k: v
        for k, v in {
            "doc_type": doc_type,
            "year": year,
            "wk_name": wk_name,
            "field_name": field_name,
            "project_name": project_name,
        }.items()
        if v is not None
    }
    explicit = list(doc_ids or [])
    if not explicit and not filters:
        typer.echo(
            "Nothing to remove: pass doc id(s) or a filter "
            "(--doc-type/--year/--wk-name/--field-name/--project-name). "
            "To delete everything, use `esdc corpus clear`.",
            err=True,
        )
        raise typer.Exit(1)

    # Only a deletion that will actually run needs write access: a dry-run
    # preview and a filter match still awaiting --yes are both readers.
    read_only = dry_run or (bool(filters) and not yes)
    store = _open_corpus_store(read_only=read_only)
    removed = 0
    try:
        targets: list[tuple[str, str]] = []
        if filters:
            matched = store.find_doc_ids(filters)
            typer.echo(f"Matched {len(matched)} document(s):")
            for _doc_id, file_name in matched:
                typer.echo(f"  {file_name}")
            targets.extend(matched)
        for doc_id in explicit:
            if all(doc_id != t[0] for t in targets):
                targets.append((doc_id, doc_id))

        if dry_run:
            typer.echo(f"Dry run — {len(targets)} document(s) would be removed.")
            return
        if filters and not yes:
            typer.echo(
                f"This deletes {len(targets)} document(s). Re-run with --yes.",
                err=True,
            )
            raise typer.Exit(1)

        _warn_pod_links([t[0] for t in targets])

        for doc_id, _name in targets:
            if store.get_document(doc_id) is None:
                typer.echo(f"Not found: {doc_id}", err=True)
                continue
            store.delete_document(doc_id)
            removed += 1
        if removed:
            store.rebuild_indexes()
    finally:
        store.close()
    typer.echo(
        f"Removed {removed} document(s). Sidecars kept — "
        "`esdc corpus commit` would re-ingest them."
    )


def _warn_pod_links(doc_ids: list[str]) -> None:
    """Warn when removed docs are linked in the registry's pod_document table.

    Links are kept: doc_id is derived from source bytes, so a re-committed
    document returns under the same id. Tolerates the table (or the whole
    registry db) not existing.
    """
    if not doc_ids:
        return
    try:
        from esdc.pod_registry.store import get_sqlite_readonly_connection

        conn = get_sqlite_readonly_connection()
        if conn is None:
            return
        try:
            placeholders = ", ".join("?" for _ in doc_ids)
            rows = conn.execute(
                "SELECT DISTINCT doc_id FROM pod_document "
                f"WHERE doc_id IN ({placeholders})",
                doc_ids,
            ).fetchall()
        finally:
            conn.close()
    except Exception:
        return
    linked = [r["doc_id"] for r in rows]
    if linked:
        typer.echo(
            f"Warning: {len(linked)} removed document(s) are linked to PODs "
            "in the registry (links kept; re-commit restores the same doc_id): "
            + ", ".join(linked),
            err=True,
        )


@corpus_app.command()
def clear(
    yes: Annotated[
        bool, typer.Option("--yes", help="Confirm deletion of the entire corpus.")
    ] = False,
) -> None:
    """Delete all documents and chunks from the corpus."""
    store = _open_corpus_store(read_only=False)
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
def export(
    paths: Annotated[
        list[Path] | None,
        typer.Argument(help="Sidecar .corpus.md file(s) to regenerate from the DB."),
    ] = None,
    all_docs: Annotated[
        bool,
        typer.Option("--all", help="Regenerate every committed document's sidecar."),
    ] = False,
) -> None:
    """Regenerate .corpus.md sidecars from the DB (documents table is truth).

    Rebuilds each sidecar's frontmatter + body from its committed row —
    including portal edits and Task-4 commit-time entity merges — so a
    deleted or stale sidecar can be reconstructed. Overwrites the file(s)
    at their DB-recorded file_path; run `esdc corpus commit` afterward
    only if you want to re-ingest (a matching hash is already committed,
    so a plain re-commit is a no-op).
    """
    from esdc.corpus.pipeline import run_export

    if not all_docs and not paths:
        typer.echo("Nothing to export: pass sidecar path(s) or --all.", err=True)
        raise typer.Exit(1)

    report = run_export(paths or [], all_docs=all_docs)
    _print_corpus_report(report)


@corpus_app.command()
def reembed(
    embed_backend: Annotated[
        str | None,
        typer.Option(
            "--embed-backend",
            help="Override embedding_backend for this run: local, ollama or openai.",
        ),
    ] = None,
    stale: Annotated[
        bool,
        typer.Option(
            "--stale",
            help="Only re-embed documents whose chunk context prefix is "
            "out of date (e.g. after portal entity edits).",
        ),
    ] = False,
) -> None:
    """Rebuild chunk embeddings: whole corpus, or --stale after entity edits."""
    from esdc.corpus.pipeline import run_reembed, run_reembed_documents
    from esdc.corpus.store import CorpusStore

    if stale:
        if embed_backend is not None:
            # --stale re-embeds a subset with the model the rest of the
            # corpus already uses; honouring a different backend here would
            # leave one corpus holding vectors from two models, which no
            # similarity comparison can span.
            typer.echo(
                "Error: --stale re-embeds with the corpus's current model; "
                "--embed-backend applies only to a full re-embed. Drop one.",
                err=True,
            )
            raise typer.Exit(1)
        store = CorpusStore(read_only=False)
        try:
            store.ensure_tables()
            doc_ids = store.stale_embed_docs()
            if not doc_ids:
                typer.echo("No stale documents — every chunk prefix is current.")
                return
            report = run_reembed_documents(doc_ids, store=store, progress=True)
        finally:
            store.close()
        _print_corpus_report(report)
        typer.echo(
            f"Re-embedded {len(report.processed)} stale document(s) "
            f"with model '{report.embedding_model}'."
        )
        return

    try:
        report = run_reembed(embed_backend=embed_backend)
    except (ValueError, RuntimeError) as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1) from None
    _print_corpus_report(report)
    typer.echo(
        f"Re-embedded {len(report.processed)} document(s) "
        f"with model '{report.embedding_model}'."
    )


@corpus_app.command(name="eval")
def corpus_eval(
    queries: Annotated[
        Path | None,
        typer.Argument(
            help="JSONL query set (default: ~/.esdc/corpus_queries.jsonl)",
        ),
    ] = None,
    init: Annotated[
        bool,
        typer.Option(
            "--init",
            help="Generate query set (auto-sized from --margin unless "
            "--samples given).",
        ),
    ] = False,
    samples: Annotated[
        int | None,
        typer.Option(
            "--samples", "-n", help="Explicit sample size for --init (default: auto)."
        ),
    ] = None,
    refresh: Annotated[
        bool,
        typer.Option("--refresh", help="Incrementally sync query set to corpus."),
    ] = False,
    margin: Annotated[
        float, typer.Option("--margin", help="CI half-width for auto sample size.")
    ] = 0.05,
    seed: Annotated[int, typer.Option("--seed", help="Sampling seed.")] = 42,
    ks: str = typer.Option("1,5,10", "--k", help="Comma-separated k values"),
    classes: str = typer.Option(
        "lookup,cross_reference,thematic",
        "--classes",
        help="Query classes to generate with --init "
        "(negatives are hand-authored, never generated)",
    ),
    rerank: bool | None = typer.Option(
        None,
        "--rerank/--no-rerank",
        help="Force rerank on/off (default: corpus.rerank config)",
    ),
) -> None:
    """Score retrieval quality (Pass@k, latency) against a query set.

    With no query set present, run `--init` first to generate one.
    """
    import esdc.corpus.evaluate as evaluate_mod
    from esdc.corpus.query_gen import (
        _make_meta,
        generate,
        generate_cross_reference,
        generate_thematic,
        read_query_file,
        reconcile,
        write_query_file,
    )
    from esdc.corpus.sampling import corpus_fingerprint
    from esdc.corpus.store import CorpusStore

    path = queries or Config.get_corpus_queries_path()
    k_values = tuple(int(k.strip()) for k in ks.split(",") if k.strip())

    def _llm_call():
        cfg = Config.get_provider_config()
        if not cfg:
            typer.echo(
                "Error: no LLM provider configured; query synthesis needs one.",
                err=True,
            )
            raise typer.Exit(1)
        import esdc.providers as providers

        llm = providers.create_llm_from_config(cfg)
        return lambda prompt: str(llm.invoke(prompt).content)

    def _progress_run(fn):
        with Progress(
            SpinnerColumn(),
            *Progress.get_default_columns(),
            TimeElapsedColumn(),
        ) as progress:
            task = progress.add_task("[cyan]Synthesizing queries...", total=None)

            def cb(done: int, total: int) -> None:
                progress.update(task, total=total, completed=done)

            return fn(cb)

    if init or refresh:
        if refresh and not path.exists():
            typer.echo(
                f"No query set at {path}. Run: esdc corpus eval --init", err=True
            )
            raise typer.Exit(1)
        store = CorpusStore()
        try:
            call = _llm_call()
            wanted = {c.strip() for c in classes.split(",") if c.strip()}
            if init:
                rows, meta = _progress_run(
                    lambda cb: generate(
                        store,
                        call,
                        margin=margin,
                        n=samples,
                        seed=seed,
                        ks=k_values,
                        progress_cb=cb,
                    )
                )
                if "cross_reference" in wanted:
                    rows += generate_cross_reference(store, call, seed=seed)
                if "thematic" in wanted:
                    rows += generate_thematic(store, call, seed=seed)
                # meta.n must count every class, not just the lookup pass.
                meta = _make_meta(store, rows, margin, k_values)
                write_query_file(path, rows, meta)
                rich.print(f"Generated {len(rows)} queries → {path}")
            else:  # --refresh
                all_old, old_meta = read_query_file(path)
                # reconcile keys on expected[0], which is meaningless for a
                # multi-document row and destructive for a negative one
                # (expected == []). Only lookup rows go through it.
                old_rows = [
                    r
                    for r in all_old
                    if str(r.get("class") or "lookup_legacy").startswith("lookup")
                ]
                carried = [r for r in all_old if r not in old_rows]
                rows, meta = _progress_run(
                    lambda cb: reconcile(
                        store,
                        call,
                        old_rows,
                        old_meta,
                        seed=seed,
                        progress_cb=cb,
                    )
                )
                rows = rows + carried
                meta = _make_meta(store, rows, meta.margin, meta.ks)
                write_query_file(path, rows, meta)
                # Delta is reported over the lookup rows only — the carried
                # classes are untouched by definition.
                new_lookup = [r for r in rows if r not in carried]
                old_ids = {r["expected"][0] for r in old_rows if r.get("expected")}
                new_ids = {r["expected"][0] for r in new_lookup if r.get("expected")}
                added = len(new_ids - old_ids)
                removed = len(old_ids - new_ids)
                old_by_id = {
                    r["expected"][0]: r.get("file_hash")
                    for r in old_rows
                    if r.get("expected")
                }
                new_by_id = {
                    r["expected"][0]: r.get("file_hash")
                    for r in new_lookup
                    if r.get("expected")
                }
                changed = sum(
                    1
                    for d in (old_by_id.keys() & new_by_id.keys())
                    if old_by_id[d] and new_by_id[d] and old_by_id[d] != new_by_id[d]
                )
                rich.print(
                    f"Refreshed: +{added} new, -{removed} removed, "
                    f"~{changed} changed → {len(rows)} queries"
                )
        finally:
            store.close()
    else:
        if not path.exists():
            typer.echo(
                f"No query set at {path}. Run: esdc corpus eval --init", err=True
            )
            raise typer.Exit(1)
        _rows, meta = read_query_file(path)
        if meta is None:
            rich.print(
                "[yellow]Legacy query file (no fingerprint); scoring as-is.[/yellow]"
            )
        else:
            store = CorpusStore()
            try:
                fp_rows = store.fingerprint_rows()
                live = corpus_fingerprint(fp_rows)
                live_ids = {r[0] for r in fp_rows}
            finally:
                store.close()
            if live != meta.fingerprint:
                live_hash = dict(fp_rows)
                existing_ids = {r["expected"][0] for r in _rows if r.get("expected")}
                added = len(live_ids - existing_ids)
                removed = len(existing_ids - live_ids)
                changed = sum(
                    1
                    for r in _rows
                    if r.get("expected")
                    and r["expected"][0] in live_hash
                    and r.get("file_hash")
                    and r["file_hash"] != live_hash[r["expected"][0]]
                )
                typer.echo(
                    f"Corpus changed (+{added} new, -{removed} removed, "
                    f"~{changed} changed). Rerun with --refresh or --init.",
                    err=True,
                )
                raise typer.Exit(1)

    report = evaluate_mod.run_eval(path, ks=k_values, rerank=rerank)
    rich.print(f"Queries scored: {report.n_queries}")
    for cls in sorted(report.by_class):
        cr = report.by_class[cls]
        rich.print(f"\n[bold]{cls}[/bold] (n={cr.n_queries})")
        for k in k_values:
            if k in cr.pass_at:
                rich.print(f"  Pass@{k}:   {cr.pass_at[k] * 100:.1f}%")
            if k in cr.recall_at:
                rich.print(f"  Recall@{k}: {cr.recall_at[k] * 100:.1f}%")
        if cr.abstention is not None:
            rich.print(f"  Abstention: {cr.abstention * 100:.1f}%")
        elif cls == "negative":
            rich.print("  Abstention: not scored (needs --rerank)")
    if report.by_class:
        # Blended across classes of unequal difficulty — kept for
        # continuity, dimmed so nobody quotes it across set versions.
        blended = report.pass_at.get(k_values[0], 0.0) * 100
        rich.print(
            f"\n[dim]All classes blended (not comparable across query-set "
            f"versions): Pass@{k_values[0]} {blended:.1f}%[/dim]"
        )
    else:
        for k in k_values:
            rich.print(f"  Pass@{k}: {report.pass_at.get(k, 0.0) * 100:.1f}%")
    rich.print(f"  Mean latency: {report.mean_latency_ms:.0f} ms")
    for f in report.failures:
        rich.print(f"[yellow]  skipped: {f}[/yellow]")
    if not report.n_queries:
        raise typer.Exit(code=1)


@corpus_app.command(name="warmup")
def corpus_warmup(
    rerank: bool | None = typer.Option(
        None,
        "--rerank/--no-rerank",
        help="Also warm the reranker (default: corpus.rerank config)",
    ),
) -> None:
    """Pre-download corpus models (embedder + optional reranker) for offline use."""
    from esdc.corpus.warmup import run_warmup

    results = run_warmup(rerank=rerank)
    failed = False
    for r in results:
        mark = "[green]OK[/green]" if r.ok else "[red]FAIL[/red]"
        rich.print(f"{mark} {r.component}: {r.model} — {r.detail}")
        failed = failed or not r.ok
    if failed:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
