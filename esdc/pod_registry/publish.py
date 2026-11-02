# esdc/pod_registry/publish.py
"""Snapshot the SQLite POD registry into DuckDB for iris text-to-SQL."""

from __future__ import annotations

import contextlib
import sqlite3
from pathlib import Path

from esdc.configs import Config
from esdc.pod_registry.contract import load_pod_contract
from esdc.pod_registry.migrations import _validate_existing_v2_registry
from esdc.pod_registry.ownership import SUPPORT_TABLES
from esdc.pod_registry.publication import (
    PodPublicationReport,
    publish_v2_registry,
)
from esdc.pod_registry.store import get_esdc_sqlite_path, get_sqlite_readonly_connection


def source_preflight_error(sqlite_path: Path | None = None) -> str | None:
    """Report why a source cannot satisfy the current contract, without writing.

    Read-only: the source is opened without a WAL writer upgrade, mkdir, or schema
    initialization, and no rows are hashed — `publish_pod_registry` does the full
    validation, including the row hashes, when it publishes. Both checks run on one
    read snapshot that is ended and closed on every path: first the table set this
    publisher reads, then the existing current-schema validator
    (`migrations._validate_existing_v2_registry`, which is read-only compatible) for
    the state row, the incompatible marker, and the ordered contract columns. Table
    names alone are not the contract — a source that keeps every required name but
    re-creates `m_project` with a wrong column cannot publish. This is why the shape
    check belongs here: a caller that mutates DuckDB before publishing (the corpus
    mirror) must refuse such a source before its own mutations, instead of destroying
    a projection the refused source can no longer rebuild.
    """
    try:
        source = get_sqlite_readonly_connection(sqlite_path)
    except sqlite3.Error as exc:
        return f"source preflight failed: {exc}"
    if source is None:
        return "SQLite source is unavailable"
    try:
        source.execute("BEGIN")
        contract = load_pod_contract()
        missing = [
            name
            for name in (*contract.tables, *SUPPORT_TABLES)
            if source.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", [name]
            ).fetchone()
            is None
        ]
        if missing:
            return (
                "source preflight failed: v2 SQLite source is missing tables: "
                + ", ".join(missing)
            )
        _validate_existing_v2_registry(source)
        return None
    except (sqlite3.Error, ValueError, RuntimeError) as exc:
        return f"source preflight failed: {exc}"
    finally:
        # The read snapshot is ended on every path, including the refusal paths.
        with contextlib.suppress(sqlite3.Error):
            source.rollback()
        source.close()


def publish_pod_registry(
    sqlite_path: Path | None = None, db_path: Path | None = None
) -> PodPublicationReport:
    """Replace the DuckDB POD projections with one current source snapshot.

    There is one contract and one path: the canonical snapshot published by
    `publish_v2_registry`. The source is validated read-only before DuckDB is
    opened, so a missing, empty, corrupt, or shape-incompatible source is
    reported as a publication failure and never rewrites the existing
    projection. Caches are invalidated only after a successful publication.
    """
    source_path = sqlite_path or get_esdc_sqlite_path()
    target = db_path or Config.get_db_file()

    report = publish_v2_registry(source_path, target)
    if report.projection_error is None:
        from esdc.chat.tools import invalidate_tool_cache, reset_sql_cache

        reset_sql_cache()
        invalidate_tool_cache()
    return report
