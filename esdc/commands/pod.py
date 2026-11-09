"""Explicit POD registry maintenance commands."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Annotated

import typer

from esdc.pod_registry.economics_upgrade import (
    EconomicsUpgradeReport,
    inspect_economics_upgrade,
    upgrade_economics_columns,
)
from esdc.pod_registry.migrations import MigrationRefusedError
from esdc.sqlite_runtime import configure_writer, require_wal_runtime


def _payload(report: EconomicsUpgradeReport) -> dict[str, object]:
    return {
        "blockers": list(report.blockers),
        "renames": [list(rename) for rename in report.renames],
        "status": report.status,
    }


def _read_only(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)


def _validate_backup(database: Path, backup: Path) -> None:
    if backup.is_symlink() or backup.exists():
        raise RuntimeError(f"backup path already exists: {backup}")
    if backup.resolve() == database.resolve():
        raise RuntimeError("backup path must differ from the database path")
    for suffix in ("-wal", "-shm", "-journal"):
        if str(backup) == f"{database}{suffix}":
            raise RuntimeError("backup path must not be a database sidecar file")


def _create_backup(database: Path, backup: Path) -> None:
    source = _read_only(database)
    destination = sqlite3.connect(backup)
    try:
        source.backup(destination)
    finally:
        destination.close()
        source.close()


def _verify_backup(backup: Path) -> None:
    conn = _read_only(backup)
    try:
        report = inspect_economics_upgrade(conn)
    finally:
        conn.close()
    if report.status != "upgrade_required":
        raise RuntimeError("backup does not contain the expected pre-upgrade layout")


def _apply(database: Path, backup: Path) -> None:
    require_wal_runtime()
    created_backup = False
    # mode=rw refuses a path that disappeared after Typer's parse-time check instead
    # of creating a new empty database file.
    writer = sqlite3.connect(f"{database.resolve().as_uri()}?mode=rw", uri=True)
    try:
        configure_writer(writer)
        writer.execute("BEGIN IMMEDIATE")
        report = inspect_economics_upgrade(writer)
        if report.status == "blocked":
            raise MigrationRefusedError("; ".join(report.blockers))
        if report.status == "current":
            writer.rollback()
            typer.echo(json.dumps(_payload(report), sort_keys=True))
            return

        _validate_backup(database, backup)
        _create_backup(database, backup)
        created_backup = True
        _verify_backup(backup)
        applied = upgrade_economics_columns(writer)
        writer.commit()
    except BaseException:
        if writer.in_transaction:
            writer.rollback()
        if created_backup:
            typer.echo(f"Pre-upgrade backup retained at: {backup}", err=True)
        raise
    finally:
        writer.close()
    typer.echo(json.dumps(_payload(applied), sort_keys=True))


def upgrade_pod_economics_command(
    database: Annotated[
        Path,
        typer.Option(
            "--database",
            exists=True,
            dir_okay=False,
            readable=True,
            help="Existing POD registry SQLite database to inspect or upgrade.",
        ),
    ],
    apply: Annotated[
        bool,
        typer.Option(
            "--apply",
            help="Apply the upgrade. Requires a distinct, new --backup path.",
        ),
    ] = False,
    backup: Annotated[
        Path | None,
        typer.Option(
            "--backup",
            help="New file path that receives the pre-upgrade SQLite backup.",
        ),
    ] = None,
) -> None:
    """Inspect or explicitly upgrade POD economics column names to contract 2."""
    if apply and backup is None:
        typer.echo("Error: --apply requires --backup.", err=True)
        raise typer.Exit(code=1)
    try:
        if not apply:
            conn = _read_only(database)
            try:
                report = inspect_economics_upgrade(conn)
            finally:
                conn.close()
            typer.echo(json.dumps(_payload(report), sort_keys=True))
            return
        assert backup is not None
        _apply(database, backup)
    except (
        ValueError,
        sqlite3.Error,
        OSError,
        RuntimeError,
        MigrationRefusedError,
    ) as exc:
        typer.echo(f"POD economics upgrade failed: {exc}", err=True)
        raise typer.Exit(code=1) from None
