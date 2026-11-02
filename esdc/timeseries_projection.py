"""Typed DuckDB projection primitives for canonical SQLite timeseries rows."""

from __future__ import annotations

import json
import re
import sqlite3
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import duckdb

from esdc.db_security import _load_sql_script
from esdc.timeseries_snapshot import (
    TIMESERIES_SOURCE_COLUMNS,
    TIMESERIES_SOURCE_TYPES,
    get_timeseries_snapshot,
    iter_timeseries_rows,
)

_TYPE_PATTERN = re.compile(
    r"(?:VARCHAR|TEXT|CHAR|BPCHAR|STRING|DATE|TIME|TIMESTAMP(?: WITH TIME ZONE)?|"
    r"TINYINT|SMALLINT|INTEGER|INT|BIGINT|HUGEINT|UTINYINT|USMALLINT|UINTEGER|"
    r"UBIGINT|FLOAT|REAL|DOUBLE|DECIMAL\([0-9]{1,3}, ?[0-9]{1,3}\)|"
    r"NUMERIC(?:\([0-9]{1,3}, ?[0-9]{1,3}\))?)\Z",
    re.ASCII | re.IGNORECASE,
)
_NUMERIC_TYPES = {
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
    "FLOAT",
    "REAL",
    "DOUBLE",
    "NUMERIC",
}


@dataclass(frozen=True)
class TimeseriesStage:
    table_name: str
    row_count: int
    years: tuple[int, ...] | None


class TimeseriesProjectionError(ValueError):
    """A canonical timeseries slice cannot be represented by its SQL types."""

    def __init__(self, message: str, *, report_year: int | None) -> None:
        super().__init__(message)
        self.report_year = report_year


def _canonical_types() -> dict[str, str]:
    source = _load_sql_script("create_table_project_timeseries.sql")
    start = source.index("CREATE TABLE IF NOT EXISTS project_timeseries")
    body_start = source.index("(\n", start) + 2
    body_end = source.index("\n);", body_start)
    result: dict[str, str] = {}
    for line in source[body_start:body_end].splitlines():
        column = line.split("--", 1)[0].strip().rstrip(",").strip()
        if not column:
            continue
        parts = column.split(None, 1)
        if len(parts) == 2 and parts[0] not in {"id", "uuid"}:
            result[parts[0]] = parts[1]
    if tuple(result) != TIMESERIES_SOURCE_COLUMNS:
        raise RuntimeError("timeseries analytical DDL does not match source columns")
    return result


def _safe_type(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = " ".join(value.upper().split())
    if _TYPE_PATTERN.fullmatch(normalized):
        if normalized.startswith(("DECIMAL(", "NUMERIC(")):
            match = re.fullmatch(
                r"(?:DECIMAL|NUMERIC)\(([0-9]+), ?([0-9]+)\)", normalized
            )
            if match is None:
                return None
            precision, scale = map(int, match.groups())
            if not 1 <= precision <= 38 or not 0 <= scale <= precision:
                return None
        return normalized
    return None


def _table_types(conn: duckdb.DuckDBPyConnection) -> dict[str, str]:
    return dict(
        conn.execute(
            "SELECT column_name, data_type FROM information_schema.columns "
            "WHERE table_schema='main' AND table_name='project_timeseries'"
        ).fetchall()
    )


def _legacy_type_overrides(source: sqlite3.Connection) -> dict[str, str]:
    snapshot = get_timeseries_snapshot(source)
    if (
        snapshot is None
        or not source.execute(
            "SELECT 1 FROM api_project_timeseries WHERE origin='legacy_duckdb' LIMIT 1"
        ).fetchone()
    ):
        return {}
    row = source.execute(
        "SELECT origin, details_json FROM api_snapshot_provenance "
        "WHERE source_table='project_timeseries' AND revision=?",
        (snapshot.revision,),
    ).fetchone()
    if row is None:
        raise ValueError("retained legacy timeseries rows lack current provenance")
    try:
        details = json.loads(row[1])
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("retained legacy timeseries provenance is malformed") from exc
    if not isinstance(details, dict):
        raise ValueError("retained legacy timeseries provenance is malformed")
    if row[0] not in {"legacy_duckdb", "api_json", "api_csv"}:
        raise ValueError("retained legacy timeseries provenance has an invalid origin")
    bootstrap = row[0] == "legacy_duckdb"
    type_key = "source_types" if bootstrap else "legacy_type_overrides"
    raw_types = details.get(type_key)
    if not isinstance(raw_types, dict):
        raise ValueError("retained legacy timeseries provenance lacks numeric types")
    expected_keys = (
        set(TIMESERIES_SOURCE_COLUMNS)
        if bootstrap
        else {
            column
            for column, source_type in TIMESERIES_SOURCE_TYPES.items()
            if source_type != "TEXT"
        }
    )
    if set(raw_types) != expected_keys:
        raise ValueError("legacy timeseries source type map is incomplete")
    result: dict[str, str] = {}
    for column in TIMESERIES_SOURCE_COLUMNS:
        value = raw_types.get(column)
        if TIMESERIES_SOURCE_TYPES[column] == "TEXT":
            if bootstrap and _safe_type(value) is None:
                raise ValueError(f"legacy timeseries type for {column} is unsafe")
            continue
        sql_type = _safe_type(value)
        if sql_type is None or not (
            sql_type in _NUMERIC_TYPES
            or sql_type.startswith("DECIMAL(")
            or sql_type.startswith("NUMERIC(")
        ):
            raise ValueError(f"legacy timeseries numeric type for {column} is unsafe")
        result[column] = sql_type
    return result


def stage_canonical_timeseries(
    conn: duckdb.DuckDBPyConnection,
    source: sqlite3.Connection,
    *,
    years: Sequence[int] | None,
) -> TimeseriesStage:
    """Copy verified canonical SQLite rows into a bounded DuckDB temp table."""
    selected = None if years is None else tuple(dict.fromkeys(years))
    if selected == ():
        raise ValueError("timeseries stage requires at least one selected year")
    canonical = _canonical_types()
    exists = (
        conn.execute(
            "SELECT 1 FROM information_schema.tables WHERE table_schema='main' "
            "AND table_name='project_timeseries'"
        ).fetchone()
        is not None
    )
    types = _table_types(conn) if exists else canonical | _legacy_type_overrides(source)
    if set(TIMESERIES_SOURCE_COLUMNS) - set(types):
        raise ValueError("analytical timeseries schema is missing canonical columns")
    safe_types = {
        column: _safe_type(types[column]) for column in TIMESERIES_SOURCE_COLUMNS
    }
    if any(value is None for value in safe_types.values()):
        raise ValueError("analytical timeseries schema contains an unsupported type")
    table = f"project_timeseries_stage_{uuid.uuid4().hex}"
    definitions = ", ".join(
        f"{name} {safe_types[name]}" for name in TIMESERIES_SOURCE_COLUMNS
    )
    conn.execute(f"CREATE TEMP TABLE {table} ({definitions})")
    placeholders = ", ".join("?" for _ in TIMESERIES_SOURCE_COLUMNS)
    insert = f"INSERT INTO {table} VALUES ({placeholders})"
    count = 0
    batch: list[tuple[Any, ...]] = []
    batch_year: int | None = None

    def insert_batch() -> None:
        try:
            conn.executemany(insert, batch)
        except duckdb.Error as exc:
            raise TimeseriesProjectionError(
                f"project_timeseries report year {batch_year}: {exc}",
                report_year=batch_year,
            ) from exc

    for row in iter_timeseries_rows(source, years=selected):
        if batch and row.report_year != batch_year:
            insert_batch()
            count += len(batch)
            batch.clear()
        batch_year = row.report_year
        values = json.loads(row.row_json)
        batch.append(tuple(values[column] for column in TIMESERIES_SOURCE_COLUMNS))
        if len(batch) == 1000:
            insert_batch()
            count += len(batch)
            batch.clear()
    if batch:
        insert_batch()
        count += len(batch)
    return TimeseriesStage(table, count, selected)


def _valid_stage_name(name: str) -> bool:
    return (
        re.fullmatch(r"project_timeseries_stage_[0-9a-f]{32}", name, re.ASCII)
        is not None
    )


def _create_target(conn: duckdb.DuckDBPyConnection, stage_name: str) -> None:
    if not _valid_stage_name(stage_name):
        raise ValueError("unsafe timeseries stage table name")
    stage_columns = {
        column: type_name
        for column, type_name, *_ in conn.execute(f"DESCRIBE {stage_name}").fetchall()
    }
    if set(TIMESERIES_SOURCE_COLUMNS) - set(stage_columns):
        raise ValueError("timeseries stage is missing canonical source columns")
    safe_types = {
        column: _safe_type(stage_columns[column])
        for column in TIMESERIES_SOURCE_COLUMNS
    }
    if any(value is None for value in safe_types.values()):
        raise ValueError("timeseries stage contains an unsupported type")
    definitions = ", ".join(
        f"{column} {safe_types[column]}" for column in TIMESERIES_SOURCE_COLUMNS
    )
    conn.execute("CREATE SEQUENCE seq_project_timeseries START 1")
    conn.execute(
        "CREATE TABLE project_timeseries ("
        "id INTEGER PRIMARY KEY DEFAULT nextval('seq_project_timeseries'), "
        f"{definitions}, uuid VARCHAR)"
    )


def apply_timeseries_stage(
    conn: duckdb.DuckDBPyConnection, stage: TimeseriesStage
) -> None:
    """Apply one staged canonical slice set inside the caller's transaction."""
    if not _valid_stage_name(stage.table_name):
        raise ValueError("unsafe timeseries stage table name")
    exists = (
        conn.execute(
            "SELECT 1 FROM information_schema.tables WHERE table_schema='main' "
            "AND table_name='project_timeseries'"
        ).fetchone()
        is not None
    )
    if not exists:
        _create_target(conn, stage.table_name)
    types = _table_types(conn)
    missing = set(TIMESERIES_SOURCE_COLUMNS) - set(types)
    if missing:
        raise ValueError("analytical timeseries schema is missing canonical columns")
    if "uuid" not in types:
        conn.execute("ALTER TABLE project_timeseries ADD COLUMN uuid VARCHAR")
    if stage.years is None:
        conn.execute("DELETE FROM project_timeseries")
    else:
        placeholders = ",".join("?" for _ in stage.years)
        conn.execute(
            f"DELETE FROM project_timeseries WHERE report_year IN ({placeholders})",
            list(stage.years),
        )
    columns = ", ".join(TIMESERIES_SOURCE_COLUMNS)
    conn.execute(
        f"INSERT INTO project_timeseries ({columns}, uuid) "
        f"SELECT {columns}, gen_random_uuid()::VARCHAR FROM {stage.table_name}"
    )
    conn.execute(f"DROP TABLE {stage.table_name}")
    from esdc.dbmanager import _create_fts_indexes, _execute_sql_script

    _execute_sql_script(conn, "create_timeseries_views.sql")
    _execute_sql_script(conn, "create_table_metadata.sql")
    if stage.row_count:
        result = _create_fts_indexes(
            conn, tables=("project_timeseries",), extensions_loaded=True
        ).get("project_timeseries")
        if result != "ok":
            raise RuntimeError(
                f"project_timeseries required index work failed: {result}"
            )
