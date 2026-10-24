"""Independent, read-only audit of the SQLite timeseries analytical mirror."""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path

import duckdb

from esdc.db_security import _load_sql_script
from esdc.timeseries_snapshot import (
    TIMESERIES_SOURCE_COLUMNS,
    get_timeseries_snapshot,
    iter_timeseries_rows,
    timeseries_fingerprint,
)

_SAFE_TYPE = re.compile(
    r"(?:VARCHAR|TEXT|CHAR|DATE|TIME|TIMESTAMP|TIMESTAMP WITH TIME ZONE|"
    r"TINYINT|SMALLINT|INTEGER|INT|BIGINT|"
    r"HUGEINT|UTINYINT|USMALLINT|UINTEGER|UBIGINT|FLOAT|REAL|DOUBLE|"
    r"DECIMAL\([0-9]{1,2}, ?[0-9]{1,2}\))\Z",
    re.ASCII | re.IGNORECASE,
)


def _valid_input_origins(value: object, current_origin: str) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and all(
            isinstance(item, str) and item in {"api_json", "api_csv"} for item in value
        )
        and current_origin in value
    )


def _canonical_types() -> dict[str, str]:
    """Read column declarations directly from the canonical SQL file."""
    source = _load_sql_script("create_table_project_timeseries.sql")
    start = source.index("CREATE TABLE IF NOT EXISTS project_timeseries")
    body_start = source.index("(\n", start) + 2
    body_end = source.index("\n);", body_start)
    declarations: dict[str, str] = {}
    for line in source[body_start:body_end].splitlines():
        line = line.split("--", 1)[0].strip().rstrip(",")
        if not line or line.startswith("--"):
            continue
        parts = line.split(None, 1)
        if len(parts) == 2 and parts[0] not in {"id", "uuid"}:
            declarations[parts[0]] = " ".join(parts[1].upper().split())
    if tuple(declarations) != TIMESERIES_SOURCE_COLUMNS:
        raise ValueError(
            "canonical timeseries SQL declarations do not match source fields"
        )
    return declarations


def _duck_type(sql_type: str) -> str:
    normalized = " ".join(sql_type.upper().split())
    normalized = re.sub(r"\(\s*([0-9]+)\s*,\s*([0-9]+)\s*\)", r"(\1,\2)", normalized)
    normalized = {"TEXT": "VARCHAR", "REAL": "FLOAT", "INT": "INTEGER"}.get(
        normalized, normalized
    )
    if normalized.startswith("NUMERIC("):
        normalized = "DECIMAL" + normalized.removeprefix("NUMERIC")
    return normalized


def _expected_types(
    source: sqlite3.Connection, declarations: dict[str, str]
) -> tuple[dict[str, str], dict[str, set[str]]]:
    """Apply only validated current provenance to canonical SQL declarations."""
    snapshot = get_timeseries_snapshot(source)
    if snapshot is None:
        raise ValueError("SQLite project_timeseries snapshot lineage is missing")
    origins = source.execute(
        "SELECT DISTINCT origin FROM api_project_timeseries"
    ).fetchall()
    has_legacy = any(row[0] == "legacy_duckdb" for row in origins)
    types = dict(declarations)
    compatible: dict[str, set[str]] = {
        name: {_duck_type(value)} for name, value in declarations.items()
    }
    provenance = source.execute(
        "SELECT origin, details_json FROM api_snapshot_provenance "
        "WHERE source_table='project_timeseries' AND revision=?",
        (snapshot.revision,),
    ).fetchone()
    if provenance is None or provenance[0] not in {
        "legacy_duckdb",
        "api_json",
        "api_csv",
    }:
        raise ValueError("project_timeseries lacks valid current provenance")
    try:
        details = json.loads(provenance[1])
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("legacy timeseries provenance is malformed") from exc
    if not isinstance(details, dict):
        raise ValueError("project_timeseries provenance must be an object")
    if not origins:
        input_origins = details.get("input_origins")
        if (
            provenance[0] not in {"api_json", "api_csv"}
            or details.get("observed_empty") is not True
            or not _valid_input_origins(input_origins, provenance[0])
        ):
            raise ValueError(
                "empty timeseries lacks explicit observed-empty provenance"
            )
        normalized = {name: _duck_type(value) for name, value in types.items()}
        return normalized, compatible
    elif has_legacy:
        if provenance[0] == "legacy_duckdb":
            if not isinstance(details.get("source_types"), dict):
                raise ValueError(
                    "legacy timeseries provenance source type map is missing"
                )
        else:
            input_origins = details.get("input_origins")
            if provenance[0] not in {"api_json", "api_csv"} or not _valid_input_origins(
                input_origins, provenance[0]
            ):
                raise ValueError(
                    "legacy timeseries current input provenance is invalid"
                )
    else:
        input_origins = details.get("input_origins")
        if provenance[0] not in {"api_json", "api_csv"} or not _valid_input_origins(
            input_origins, provenance[0]
        ):
            raise ValueError("timeseries current input provenance is invalid")
        normalized = {name: _duck_type(value) for name, value in types.items()}
        return normalized, compatible
    if provenance[0] == "legacy_duckdb":
        source_types = details.get("source_types")
        if not isinstance(source_types, dict) or set(source_types) != set(declarations):
            raise ValueError("legacy timeseries source type map is incomplete")
    else:
        source_types = details.get("legacy_type_overrides")
        numeric_columns = {
            name
            for name, declaration in declarations.items()
            if declaration not in {"TEXT", "VARCHAR"}
        }
        if not isinstance(source_types, dict) or set(source_types) != numeric_columns:
            raise ValueError(
                "legacy timeseries inherited numeric type map is incomplete"
            )
        legacy_provenance = source.execute(
            "SELECT details_json FROM api_snapshot_provenance "
            "WHERE source_table='project_timeseries' AND origin='legacy_duckdb' "
            "ORDER BY rowid DESC LIMIT 1"
        ).fetchone()
        if legacy_provenance is not None:
            try:
                legacy_details = json.loads(legacy_provenance[0])
            except (TypeError, json.JSONDecodeError) as exc:
                raise ValueError("legacy timeseries provenance is malformed") from exc
            legacy_types = (
                legacy_details.get("source_types")
                if isinstance(legacy_details, dict)
                else None
            )
            if not isinstance(legacy_types, dict) or set(legacy_types) != set(
                declarations
            ):
                raise ValueError("legacy timeseries source type map is incomplete")
            for name, value in legacy_types.items():
                if (
                    not isinstance(value, str)
                    or _SAFE_TYPE.fullmatch(value.upper()) is None
                ):
                    raise ValueError(
                        f"legacy timeseries provenance type for {name} is unsafe"
                    )
                if _duck_type(declarations[name]) in {"VARCHAR", "CHAR", "TEXT"}:
                    legacy_normalized = _duck_type(value)
                    if legacy_normalized not in {
                        "VARCHAR",
                        "CHAR",
                        "TEXT",
                        "DATE",
                        "TIME",
                        "TIMESTAMP",
                        "TIMESTAMP WITH TIME ZONE",
                    }:
                        raise ValueError(
                            f"legacy timeseries text type for {name} is unsafe"
                        )
                    compatible[name].add(legacy_normalized)
    for name, value in source_types.items():
        if not isinstance(value, str) or _SAFE_TYPE.fullmatch(value.upper()) is None:
            raise ValueError(f"legacy timeseries provenance type for {name} is unsafe")
        if name not in types:
            raise ValueError(
                f"legacy timeseries provenance contains unknown type {name}"
            )
        numeric = _duck_type(declarations[name]) not in {"VARCHAR", "CHAR", "TEXT"}
        normalized = _duck_type(value)
        if numeric and not (
            normalized
            in {
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
            }
            or normalized.startswith("DECIMAL(")
        ):
            raise ValueError(f"legacy timeseries numeric type for {name} is unsafe")
        if not numeric and normalized not in {
            "VARCHAR",
            "CHAR",
            "TEXT",
            "DATE",
            "TIME",
            "TIMESTAMP",
            "TIMESTAMP WITH TIME ZONE",
        }:
            raise ValueError(f"legacy timeseries text type for {name} is unsafe")
        # Legacy DATE/TIME/TIMESTAMP declarations are supported source inputs,
        # but the fresh canonical analytical schema stores these text fields as
        # VARCHAR. Only legacy numeric precision/type overrides are inherited.
        if numeric:
            types[name] = normalized
            compatible[name] = {normalized}
        else:
            compatible[name].add(normalized)
    return {name: _duck_type(value) for name, value in types.items()}, compatible


def compare_timeseries_projection(
    *, sqlite_path: Path, duckdb_path: Path, sample_limit: int = 10
) -> dict[str, object]:
    """Compare canonical SQLite rows and a DuckDB projection without writes.

    Full parity is duplicate-sensitive and uses DuckDB ``EXCEPT ALL`` in a
    private scratch database. Source and target rows are transferred in batches
    of at most 1000; generated ``id`` and ``uuid`` fields are deliberately
    excluded from semantic parity.
    """
    issues: list[str] = []
    samples: list[dict[str, object]] = []
    canonical_rows = projected_rows = value_difference_count = 0
    lineage_matches = False
    sqlite_conn: sqlite3.Connection | None = None
    target: duckdb.DuckDBPyConnection | None = None
    scratch: duckdb.DuckDBPyConnection | None = None
    try:
        sqlite_conn = sqlite3.connect(
            sqlite_path.resolve().as_uri() + "?mode=ro", uri=True
        )
        sqlite_conn.execute("BEGIN")
        target = duckdb.connect(str(duckdb_path), read_only=True)
        declarations = _canonical_types()
        expected_types, compatible_types = _expected_types(sqlite_conn, declarations)
        snapshot = get_timeseries_snapshot(sqlite_conn)
        if snapshot is None:
            raise ValueError("SQLite project_timeseries snapshot lineage is missing")
        canonical_rows, content_hash = timeseries_fingerprint(sqlite_conn)
        if (
            snapshot.row_count != canonical_rows
            or snapshot.source_sha256 != content_hash
        ):
            issues.append("SQLite timeseries snapshot hash or row count is corrupt")
        for row in iter_timeseries_rows(sqlite_conn):
            if row.report_year is not None and not isinstance(row.report_year, int):
                issues.append("SQLite timeseries year index has an invalid type")
                break
        info = target.execute(
            "SELECT column_name, data_type FROM information_schema.columns "
            "WHERE table_schema='main' AND table_name='project_timeseries'"
        ).fetchall()
        actual_types = {name: data_type.upper() for name, data_type in info}
        if not actual_types:
            raise ValueError("DuckDB project_timeseries table is missing")
        for name in expected_types:
            actual = actual_types.get(name)
            if actual not in compatible_types[name]:
                issues.append(
                    f"DuckDB project_timeseries type for {name} is "
                    f"{actual}, expected one of {sorted(compatible_types[name])}"
                )
        effective_types = {
            name: actual_types[name]
            if actual_types.get(name) in compatible_types[name]
            else expected
            for name, expected in expected_types.items()
        }
        projected_count = target.execute(
            "SELECT COUNT(*) FROM project_timeseries"
        ).fetchone()
        assert projected_count is not None
        projected_rows = int(projected_count[0])
        publication_exists = target.execute(
            "SELECT 1 FROM information_schema.tables WHERE table_schema='main' "
            "AND table_name='api_publication'"
        ).fetchone()
        publication = None
        if publication_exists:
            publication = target.execute(
                "SELECT revision, source_row_count, source_sha256 FROM api_publication "
                "WHERE source_table='project_timeseries'"
            ).fetchone()
        lineage_matches = publication is not None and (
            publication[0] == snapshot.revision
            and publication[1] == snapshot.row_count
            and publication[2] == snapshot.source_sha256
            and snapshot.publication_state == "published"
        )
        if not lineage_matches:
            issues.append(
                "SQLite snapshot and DuckDB timeseries publication lineage differ"
            )

        scratch = duckdb.connect()
        scratch.execute("BEGIN")
        definitions = ", ".join(
            f'"{name}" {effective_types[name]}' for name in TIMESERIES_SOURCE_COLUMNS
        )
        scratch.execute(f"CREATE TABLE expected_rows ({definitions})")
        scratch.execute(f"CREATE TABLE actual_rows ({definitions})")
        columns = ", ".join(f'"{name}"' for name in TIMESERIES_SOURCE_COLUMNS)
        placeholders = ", ".join("?" for _ in TIMESERIES_SOURCE_COLUMNS)
        casts = ", ".join(
            f"CAST(? AS {effective_types[name]})" for name in TIMESERIES_SOURCE_COLUMNS
        )
        expected_insert = f"INSERT INTO expected_rows VALUES ({casts})"
        actual_cursor = target.execute(f"SELECT {columns} FROM project_timeseries")
        while batch := actual_cursor.fetchmany(1000):
            scratch.executemany(
                f"INSERT INTO actual_rows VALUES ({placeholders})", batch
            )
        batch_values: list[tuple[object, ...]] = []
        for row in iter_timeseries_rows(sqlite_conn):
            values = json.loads(row.row_json)
            batch_values.append(
                tuple(values[name] for name in TIMESERIES_SOURCE_COLUMNS)
            )
            if len(batch_values) == 1000:
                scratch.executemany(expected_insert, batch_values)
                batch_values.clear()
        if batch_values:
            scratch.executemany(expected_insert, batch_values)
        scratch.execute("COMMIT")
        difference_sql = (
            f"SELECT * FROM ((SELECT {columns} FROM actual_rows EXCEPT ALL "
            f"SELECT {columns} FROM expected_rows) UNION ALL "
            f"(SELECT {columns} FROM expected_rows EXCEPT ALL "
            f"SELECT {columns} FROM actual_rows))"
        )
        difference_count = scratch.execute(
            f"SELECT COUNT(*) FROM ({difference_sql}) d"
        ).fetchone()
        assert difference_count is not None
        value_difference_count = int(difference_count[0])
        if value_difference_count:
            cur = scratch.execute(
                f"SELECT * FROM ({difference_sql}) d LIMIT ?", [max(0, sample_limit)]
            )
            samples = [
                dict(zip(TIMESERIES_SOURCE_COLUMNS, row, strict=True))
                for row in cur.fetchall()
            ]
        if value_difference_count:
            issues.append(
                "DuckDB timeseries rows differ from canonical SQLite values "
                "or multiplicities"
            )
        if canonical_rows != projected_rows:
            issues.append("SQLite and DuckDB timeseries row counts differ")
    except (sqlite3.Error, duckdb.Error, OSError, ValueError, TypeError) as exc:
        issues.append(f"Timeseries projection could not be audited: {exc}")
    finally:
        if scratch is not None:
            scratch.close()
        if target is not None:
            target.close()
        if sqlite_conn is not None:
            sqlite_conn.close()
    return {
        "ok": not issues,
        "canonical_rows": canonical_rows,
        "projected_rows": projected_rows,
        "value_difference_count": value_difference_count,
        "lineage_matches": lineage_matches,
        "issues": issues,
        "difference_samples": samples,
    }
