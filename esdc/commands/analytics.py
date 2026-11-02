"""Commands for inspecting and maintaining analytical source snapshots."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Annotated

import duckdb
import typer

from esdc.analytics_rebuild import AnalyticsRebuildReport, rebuild_api_analytics
from esdc.analytics_rehearsal import run_analytics_rehearsal
from esdc.configs import Config
from esdc.pod_registry.store import get_esdc_sqlite_path
from esdc.source_publication import (
    committed_pod_source_generation,
    publish_committed_pod_sources,
)
from esdc.timeseries_bootstrap import bootstrap_timeseries_source


def rebuild_api_analytics_command() -> None:
    """Rebuild both API analytical projections from committed SQLite snapshots."""
    sqlite_path = get_esdc_sqlite_path()
    generation = committed_pod_source_generation(sqlite_path)
    try:
        if generation is None:
            # A database without POD uses the generic raw-source recovery route.
            report = rebuild_api_analytics(
                duckdb_path=Config.get_db_file(), sqlite_path=sqlite_path
            )
        else:
            # The v2 route publishes committed truth only: no HTTP, no source re-apply,
            # no new revision UUID and no new source generation.
            report = publish_committed_pod_sources(
                sqlite_path,
                Config.get_db_file(),
                expected_generation=generation,
            )
    except (OSError, RuntimeError, ValueError, sqlite3.Error, duckdb.Error) as exc:
        typer.echo(f"API analytics recovery failed: {exc}", err=True)
        raise typer.Exit(code=1) from None
    typer.echo(json.dumps(_report_payload(report), sort_keys=True))


def _report_payload(report: AnalyticsRebuildReport) -> dict[str, object]:
    payload: dict[str, object] = {
        "output_path": str(report.output_path),
        "source_revisions": report.source_revisions,
        "source_hashes": report.source_hashes,
        "subsystem_status": report.subsystem_status,
        "object_counts": report.object_counts,
        "requires_regeneration": report.requires_regeneration,
    }
    # The optional fields appear only for the route that has them, so the legacy payload
    # stays exactly as it was; the v2 payload reports each component's real status.
    if report.components:
        payload["components"] = [
            {
                "name": component.name,
                "status": component.status,
                "rows_written": component.rows_written,
                "revision": component.revision,
                "generation": component.generation,
                "error": component.error,
            }
            for component in report.components
        ]
        payload["fresh"] = report.fresh
    if report.recovery_command is not None:
        payload["recovery_command"] = report.recovery_command
    return payload


def bootstrap_timeseries_source_command(
    apply: Annotated[
        bool,
        typer.Option(
            "--apply",
            help=(
                "Import the legacy source into canonical SQLite and commit a baseline."
            ),
        ),
    ] = False,
) -> None:
    """Inspect or bootstrap a legacy DuckDB project_timeseries table."""
    try:
        report = bootstrap_timeseries_source(
            duckdb_path=Config.get_db_file(),
            sqlite_path=get_esdc_sqlite_path(),
            apply=apply,
        )
    except (OSError, RuntimeError, ValueError, sqlite3.Error) as exc:
        typer.echo(f"Bootstrap refused: {exc}", err=True)
        raise typer.Exit(code=1) from None

    action = "Applied" if report.applied else "Dry run"
    typer.echo(
        f"{action}: {report.row_count} legacy timeseries rows; "
        f"{report.null_year_rows} with NULL report_year; "
        f"source SHA-256 {report.source_sha256}."
    )
    typer.echo("Source types: " + json.dumps(report.source_types, sort_keys=True))
    if not report.applied:
        typer.echo("No SQLite database or schema was created. Pass --apply to import.")


def verify_analytics_migration_command(
    baseline_dir: Annotated[
        Path,
        typer.Option(
            "--baseline-dir",
            help="Closed directory containing esdc.sqlite and esdc.duckdb.",
        ),
    ],
    fixture_manifest: Annotated[
        Path,
        typer.Option(
            "--fixture-manifest", help="Complete offline JSON fixture manifest."
        ),
    ],
    output_dir: Annotated[
        Path,
        typer.Option(
            "--output-dir", help="New directory for isolated clones and report.json."
        ),
    ],
) -> None:
    """Rehearse analytics migration against closed clones and offline fixtures."""
    try:
        report_path = run_analytics_rehearsal(
            baseline_dir=baseline_dir,
            fixture_manifest=fixture_manifest,
            output_dir=output_dir,
        )
    except (OSError, RuntimeError, ValueError, sqlite3.Error, duckdb.Error) as exc:
        report_path = output_dir / "report.json"
        retained = (
            f" Report retained at {report_path}." if report_path.is_file() else ""
        )
        typer.echo(f"Analytics migration rehearsal failed: {exc}.{retained}", err=True)
        raise typer.Exit(code=1) from None
    typer.echo(str(report_path))
