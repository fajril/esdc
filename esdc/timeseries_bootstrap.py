"""Explicit, transactional import of a legacy DuckDB timeseries baseline."""

from __future__ import annotations

import hashlib
import sqlite3
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import date, datetime, time, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

import duckdb

from esdc.api_snapshot import JSONScalar
from esdc.timeseries_snapshot import (
    TIMESERIES_SOURCE_COLUMNS,
    TIMESERIES_SOURCE_TYPES,
    CanonicalTimeseriesRow,
    normalize_timeseries_row,
    record_timeseries_provenance,
    replace_timeseries_in_transaction,
)

_EXCLUDED_DUCKDB_COLUMNS = frozenset(
    {
        "id",
        "uuid",
        "geom",
        "geometry",
        "field_geom",
        "field_geometry",
        "embedding",
        "embeddings",
    }
)
_INTEGER_TYPES = frozenset(
    {
        "TINYINT",
        "SMALLINT",
        "INTEGER",
        "INT",
        "BIGINT",
        "HUGEINT",
        "UTINYINT",
        "USMALLINT",
        "UINTEGER",
        "UBIGINT",
    }
)


@dataclass(frozen=True)
class TimeseriesBootstrapReport:
    """Summary of a dry-run or committed legacy import."""

    applied: bool
    row_count: int
    source_sha256: str
    null_year_rows: int
    source_types: dict[str, str]


@dataclass(frozen=True)
class _SourceFingerprint:
    file_sha256: str
    file_size: int
    file_mtime_ns: int
    row_count: int
    null_year_rows: int
    normalized_sha256: str
    source_types: dict[str, str]


def _file_fingerprint(path: Path) -> tuple[str, int, int]:
    digest = hashlib.sha256()
    with path.open("rb") as source_file:
        for block in iter(lambda: source_file.read(1024 * 1024), b""):
            digest.update(block)
    stat = path.stat()
    return digest.hexdigest(), stat.st_size, stat.st_mtime_ns


def _source_types(conn: duckdb.DuckDBPyConnection) -> dict[str, str]:
    try:
        description = conn.execute(
            "DESCRIBE SELECT * FROM project_timeseries"
        ).fetchall()
    except duckdb.Error as exc:
        raise ValueError(
            "DuckDB project_timeseries source table is unavailable"
        ) from exc
    if not description:
        raise ValueError("DuckDB project_timeseries source schema is empty")

    types = {str(column[0]): str(column[1]).upper() for column in description}
    missing = sorted(set(TIMESERIES_SOURCE_COLUMNS) - types.keys())
    if missing:
        raise ValueError(
            "DuckDB project_timeseries schema is missing source columns: "
            + ", ".join(missing)
        )
    unexpected = sorted(
        set(types) - set(TIMESERIES_SOURCE_COLUMNS) - _EXCLUDED_DUCKDB_COLUMNS
    )
    if unexpected:
        raise ValueError(
            "DuckDB project_timeseries schema has unsupported columns: "
            + ", ".join(unexpected)
        )
    for column in TIMESERIES_SOURCE_COLUMNS:
        _validate_type(column, types[column])
    return {column: types[column] for column in TIMESERIES_SOURCE_COLUMNS}


def _validate_type(column: str, source_type: str) -> None:
    expected = TIMESERIES_SOURCE_TYPES[column]
    normalized = source_type.split("(", 1)[0].strip()
    if expected == "TEXT":
        allowed = (
            normalized in {"VARCHAR", "CHAR", "BPCHAR", "TEXT", "STRING"}
            or normalized in {"DATE", "TIME", "TIMESTAMP", "TIMESTAMP WITH TIME ZONE"}
        )
    elif expected == "INTEGER":
        allowed = normalized in _INTEGER_TYPES
    else:
        allowed = (
            normalized in {"FLOAT", "REAL", "DOUBLE", "DECIMAL", "NUMERIC"}
            or normalized in _INTEGER_TYPES
        )
    if not allowed:
        raise ValueError(
            f"DuckDB project_timeseries column {column} has unsafe type "
            f"{source_type!r} for analytical type {expected}"
        )


def _source_value(column: str, value: Any, source_type: str) -> JSONScalar:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Decimal):
        return format(value, "f")
    if isinstance(value, (date, datetime, time)):
        normalized_type = source_type.split("(", 1)[0].strip()
        if TIMESERIES_SOURCE_TYPES[column] != "TEXT" or normalized_type not in {
            "DATE",
            "TIME",
            "TIMESTAMP",
            "TIMESTAMP WITH TIME ZONE",
        }:
            raise ValueError(
                f"DuckDB project_timeseries column {column} contains a date/time "
                "value for an incompatible declared type"
            )
        return value.isoformat()
    raise ValueError(
        f"DuckDB project_timeseries column {column} contains unsupported "
        f"value type {type(value).__name__}"
    )


def _iter_source_rows(
    conn: duckdb.DuckDBPyConnection, source_types: dict[str, str]
) -> Iterator[CanonicalTimeseriesRow]:
    columns = ", ".join(f'"{column}"' for column in TIMESERIES_SOURCE_COLUMNS)
    ordering = ", ".join(f'"{column}"' for column in TIMESERIES_SOURCE_COLUMNS)
    cursor = conn.execute(
        f"SELECT {columns} FROM project_timeseries ORDER BY {ordering}"
    )
    while batch := cursor.fetchmany(1000):
        for raw_values in batch:
            values = tuple(
                _source_value(column, value, source_types[column])
                for column, value in zip(
                    TIMESERIES_SOURCE_COLUMNS, raw_values, strict=True
                )
            )
            yield normalize_timeseries_row(
                TIMESERIES_SOURCE_COLUMNS, values, origin="legacy_duckdb"
            )


def _read_source_fingerprint(
    conn: duckdb.DuckDBPyConnection, path: Path, source_types: dict[str, str]
) -> _SourceFingerprint:
    digest = hashlib.sha256()
    count = 0
    null_years = 0
    for row in _iter_source_rows(conn, source_types):
        digest.update(row.row_json.encode("utf-8"))
        digest.update(b"\n")
        count += 1
        null_years += row.report_year is None
    file_hash, file_size, file_mtime_ns = _file_fingerprint(path)
    return _SourceFingerprint(
        file_hash,
        file_size,
        file_mtime_ns,
        count,
        null_years,
        digest.hexdigest(),
        dict(source_types),
    )


def _has_table(conn: sqlite3.Connection, name: str) -> bool:
    return (
        conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
        ).fetchone()
        is not None
    )


def _assert_canonical_absent(conn: sqlite3.Connection) -> None:
    if _has_table(conn, "api_project_timeseries") and conn.execute(
        "SELECT 1 FROM api_project_timeseries LIMIT 1"
    ).fetchone():
        raise ValueError(
            "canonical timeseries rows already exist; "
            "bootstrap will not overwrite them"
        )
    if _has_table(conn, "api_snapshot") and conn.execute(
        "SELECT 1 FROM api_snapshot WHERE source_table='project_timeseries' LIMIT 1"
    ).fetchone():
        raise ValueError(
            "canonical timeseries snapshot already exists; "
            "bootstrap will not overwrite it"
        )


def _ensure_snapshot_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """CREATE TABLE IF NOT EXISTS api_snapshot (
            source_table TEXT PRIMARY KEY,
            revision TEXT NOT NULL,
            row_count INTEGER NOT NULL,
            source_sha256 TEXT NOT NULL,
            fetched_at TEXT NOT NULL,
            publication_state TEXT NOT NULL CHECK (
                publication_state IN ('committed', 'published', 'analytics_stale')
            ),
            invalid_geometry_count INTEGER NOT NULL DEFAULT 0
                CHECK (invalid_geometry_count >= 0)
        )"""
    )


def _check_sqlite_absence(sqlite_path: Path) -> None:
    if not sqlite_path.exists():
        return
    uri = f"{sqlite_path.resolve().as_uri()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    try:
        _assert_canonical_absent(conn)
    finally:
        conn.close()


def bootstrap_timeseries_source(
    *, duckdb_path: Path, sqlite_path: Path, apply: bool = False
) -> TimeseriesBootstrapReport:
    """Inspect or explicitly import a complete legacy DuckDB timeseries table.

    The default run is read-only and never creates SQLite state. ``apply=True``
    writes the canonical rows, committed snapshot, and source provenance in one
    SQLite transaction, after rechecking the same DuckDB read-only connection.
    """
    source_path = duckdb_path.expanduser().resolve(strict=True)
    try:
        source_conn = duckdb.connect(str(source_path), read_only=True)
    except duckdb.Error as exc:
        raise ValueError(f"cannot open DuckDB source read-only: {exc}") from exc
    try:
        source_conn.execute("BEGIN TRANSACTION")
        source_types = _source_types(source_conn)
        first = _read_source_fingerprint(source_conn, source_path, source_types)
        _check_sqlite_absence(sqlite_path)
        if not apply:
            return TimeseriesBootstrapReport(
                False,
                first.row_count,
                first.file_sha256,
                first.null_year_rows,
                dict(first.source_types),
            )

        sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        sqlite_conn = sqlite3.connect(sqlite_path, isolation_level=None)
        try:
            second_types = _source_types(source_conn)
            second = _read_source_fingerprint(source_conn, source_path, second_types)
            if second != first:
                raise RuntimeError(
                    "DuckDB project_timeseries source changed during bootstrap; "
                    "no canonical changes were committed"
                )

            sqlite_conn.execute("BEGIN IMMEDIATE")
            _assert_canonical_absent(sqlite_conn)
            _ensure_snapshot_table(sqlite_conn)
            from esdc.timeseries_snapshot import ensure_timeseries_tables

            ensure_timeseries_tables(sqlite_conn)
            revision = f"legacy-{uuid.uuid4().hex}"
            fetched_at = datetime.now(timezone.utc).isoformat()
            replace_timeseries_in_transaction(
                sqlite_conn,
                _iter_source_rows(source_conn, source_types),
                years=None,
                revision=revision,
                fetched_at=fetched_at,
            )
            record_timeseries_provenance(
                sqlite_conn,
                revision=revision,
                origin="legacy_duckdb",
                details={
                    "source_path": str(source_path),
                    "source_sha256": first.file_sha256,
                    "source_size": first.file_size,
                    "source_mtime_ns": first.file_mtime_ns,
                    "normalized_source_sha256": first.normalized_sha256,
                    "source_types": first.source_types,
                    "row_count": first.row_count,
                    "null_year_rows": first.null_year_rows,
                },
            )
            sqlite_conn.commit()
        except BaseException:
            if sqlite_conn.in_transaction:
                sqlite_conn.rollback()
            raise
        finally:
            sqlite_conn.close()
        return TimeseriesBootstrapReport(
            True,
            first.row_count,
            first.file_sha256,
            first.null_year_rows,
            dict(first.source_types),
        )
    finally:
        source_conn.close()
