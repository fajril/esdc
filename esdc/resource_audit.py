"""Independent read-only parity checks for the project-resource mirror."""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import duckdb

from esdc.api_projection import get_resource_publication
from esdc.api_snapshot import (
    ApiProjectResourceRow,
    _persisted_hash,
    get_project_resources_snapshot,
    iter_project_resources_rows,
)
from esdc.db_security import _load_sql_script

_EXCLUDED_COLUMNS = {"id", "uuid", "field_geom", "field_geometry"}


@dataclass(frozen=True)
class ResourceParityReport:
    canonical_rows: int
    projected_rows: int
    identity_difference_count: int
    source_multiset_difference_count: int
    metadata_matches: bool
    invalid_geometry_count: int
    issues: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return (
            self.canonical_rows == self.projected_rows
            and self.identity_difference_count == 0
            and self.source_multiset_difference_count == 0
            and self.metadata_matches
            and not self.issues
        )


def _canonical_types() -> dict[str, str]:
    source = _load_sql_script("create_table_project_resources.sql")
    match = re.search(
        r"CREATE TABLE IF NOT EXISTS project_resources\s*\((.*?)\n\);", source, re.S
    )
    if match is None:
        raise RuntimeError("canonical project_resources SQL schema was not found")
    declarations: dict[str, str] = {}
    for line in match.group(1).splitlines():
        declaration = line.strip().split("--", 1)[0].rstrip(",")
        parsed = re.match(
            r"([a-z_][a-z0-9_]*)\s+([A-Z]+(?:\([0-9, ]+\))?)", declaration, re.I
        )
        if parsed:
            declarations[parsed.group(1)] = parsed.group(2).upper()
    ddl = ", ".join(
        f'"{name}" {data_type}'
        for name, data_type in declarations.items()
        if name not in _EXCLUDED_COLUMNS and data_type != "GEOMETRY"
    )
    scratch = duckdb.connect()
    try:
        scratch.execute(f"CREATE TABLE canonical_source_types ({ddl})")
        result = {
            row[0]: row[1].upper()
            for row in scratch.execute("DESCRIBE canonical_source_types").fetchall()
        }
        return result
    finally:
        scratch.close()


def _first(value: object) -> str:
    return value[0] if isinstance(value, str) and value else ""


def _expected_value(row: ApiProjectResourceRow, name: str) -> object:
    if name == "is_discovered":
        code = _first(row.project_class)
        return 1 if code in {"1", "2"} else 0 if code == "3" else None
    if name == "project_stage":
        return {
            "E": "1. Exploitation",
            "X": "2. Exploration",
            "A": "3. Abandoned",
        }.get(_first(row.project_level))
    return getattr(row, name)


def _readonly_sqlite(path: Path) -> sqlite3.Connection:
    uri = path.resolve().as_uri() + "?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _difference_count(conn: duckdb.DuckDBPyConnection, columns: tuple[str, ...]) -> int:
    names = ", ".join(f'"{name}"' for name in columns)
    query = (
        f"SELECT COUNT(*) FROM ((SELECT {names} FROM actual_resources "
        f"EXCEPT ALL SELECT {names} FROM expected_resources) UNION ALL "
        f"(SELECT {names} FROM expected_resources EXCEPT ALL "
        f"SELECT {names} FROM actual_resources)) differences"
    )
    row = conn.execute(query).fetchone()
    if row is None:
        raise RuntimeError("EXCEPT ALL count returned no row")
    return int(row[0])


def _count(
    conn: duckdb.DuckDBPyConnection,
    query: str,
    parameters: Sequence[object] | None = None,
) -> int:
    values = list(parameters) if parameters is not None else []
    row = conn.execute(query, values).fetchone()
    if row is None:
        raise RuntimeError("count query returned no row")
    return int(row[0])


def _count_invalid_legacy_geometry(rows: list[ApiProjectResourceRow]) -> int:
    wkts = [
        str(row.field_geom)
        for row in rows
        if row.field_geom is not None and str(row.field_geom).strip()
    ]
    if not wkts:
        return 0
    scratch = duckdb.connect()
    try:
        scratch.execute("LOAD spatial")
        invalid_count = 0
        for offset in range(0, len(wkts), 1000):
            batch = wkts[offset : offset + 1000]
            placeholders = ", ".join("(?)" for _ in batch)
            invalid_count += _count(
                scratch,
                "SELECT COUNT(*) FROM (VALUES "
                + placeholders
                + ") AS legacy_wkt(value) WHERE NOT coalesce("
                "ST_IsValid(TRY(ST_GeomFromText(value))), false)",
                batch,
            )
        return invalid_count
    finally:
        scratch.close()


def compare_resource_projection(
    conn: duckdb.DuckDBPyConnection, sqlite_path: Path
) -> ResourceParityReport:
    """Compare canonical SQLite rows with the caller's published DuckDB mirror."""
    issues: list[str] = []
    source_conn: sqlite3.Connection | None = None
    canonical_rows = 0
    projected_rows = 0
    identity_difference_count = 0
    source_multiset_difference_count = 0
    metadata_matches = False
    invalid_geometry_count = 0
    try:
        source_conn = _readonly_sqlite(sqlite_path)
        source_conn.execute("BEGIN")
        tables = {
            row[0]
            for row in source_conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        if "api_project_resources" not in tables:
            issues.append("SQLite canonical api_project_resources table is missing")
            snapshot = None
            rows = []
        else:
            source_columns = {
                row["name"]
                for row in source_conn.execute(
                    "PRAGMA table_info(api_project_resources)"
                )
            }
            canonical = tuple(ApiProjectResourceRow.__dataclass_fields__)
            missing = set(canonical) - source_columns
            if missing:
                issues.append(
                    "SQLite canonical fields are missing: " + ", ".join(sorted(missing))
                )
            snapshot = (
                get_project_resources_snapshot(source_conn)
                if "api_snapshot" in tables
                else None
            )
            if snapshot is None:
                issues.append("SQLite project_resources snapshot lineage is missing")
            actual_count, actual_hash = _persisted_hash(source_conn)
            canonical_rows = actual_count
            if snapshot is not None and (
                snapshot.row_count != actual_count
                or snapshot.source_sha256 != actual_hash
            ):
                issues.append(
                    "SQLite snapshot count/hash does not match canonical rows"
                )
            rows = list(iter_project_resources_rows(source_conn)) if not missing else []
    except (sqlite3.Error, OSError, ValueError) as exc:
        issues.append(f"SQLite source could not be audited: {exc}")
        rows = []
        snapshot = None
    finally:
        if source_conn is not None:
            source_conn.close()

    canonical_schema = _canonical_types()
    columns = tuple(
        name
        for name in ApiProjectResourceRow.__dataclass_fields__
        if name not in _EXCLUDED_COLUMNS
    )
    for name in columns:
        if name not in canonical_schema:
            issues.append(f"Canonical SQL schema is missing source field {name}")

    target_columns = {
        row[0]: row[1].upper()
        for row in conn.execute(
            "SELECT column_name, data_type FROM duckdb_columns() "
            "WHERE database_name=current_database() "
            "AND table_name = 'project_resources'"
        ).fetchall()
    }
    if not target_columns:
        issues.append("DuckDB project_resources projection is missing")
    else:
        projected_rows = _count(conn, "SELECT COUNT(*) FROM project_resources")
    obsolete_columns = {"field_geom", "field_geometry"} & set(target_columns)
    if obsolete_columns:
        issues.append(
            "DuckDB project_resources has obsolete geometry columns: "
            + ", ".join(sorted(obsolete_columns))
            + "; rebuild the analytical database to replace the old projection"
        )
    for name in (*columns, "id", "uuid"):
        expected_type = canonical_schema.get(name)
        if name == "id":
            expected_type = "INTEGER"
        if name == "uuid":
            expected_type = "VARCHAR"
        if expected_type and name not in target_columns:
            issues.append(f"DuckDB projection column {name} is missing")
        elif expected_type and target_columns.get(name) != expected_type:
            issues.append(
                f"DuckDB projection type for {name} is "
                f"{target_columns.get(name)}, expected {expected_type}"
            )

    publication_exists = conn.execute(
        "SELECT 1 FROM information_schema.tables "
        "WHERE table_catalog=current_database() "
        "AND table_schema='main' "
        "AND table_name='api_publication'"
    ).fetchone()
    try:
        publication = get_resource_publication(conn) if publication_exists else None
    except (duckdb.Error, ValueError) as exc:
        publication = None
        issues.append(f"DuckDB publication lineage could not be read: {exc}")
    if snapshot is None:
        if publication is not None:
            issues.append(
                "DuckDB publication lineage exists without a canonical SQLite snapshot"
            )
    elif publication is None:
        issues.append("DuckDB resource publication lineage is missing")
    else:
        metadata_matches = (
            publication.revision == snapshot.revision
            and publication.source_row_count == snapshot.row_count
            and publication.source_sha256 == snapshot.source_sha256
            and publication.invalid_geometry_count == snapshot.invalid_geometry_count
            and snapshot.publication_state == "published"
        )
        if not metadata_matches:
            issues.append(
                "SQLite snapshot and DuckDB publication metadata do not match"
            )

    if target_columns and all(
        name in target_columns and target_columns[name] == canonical_schema.get(name)
        for name in columns
    ):
        try:
            actual_values = conn.execute(
                "SELECT "
                + ", ".join(f'"{name}"' for name in columns)
                + " FROM project_resources"
            ).fetchall()
            scratch = duckdb.connect()
            try:
                ddl = ", ".join(
                    f'"{name}" {canonical_schema[name]}' for name in columns
                )
                scratch.execute(f"CREATE TABLE actual_resources ({ddl})")
                scratch.execute(f"CREATE TABLE expected_resources ({ddl})")
                placeholders = ", ".join("?" for _ in columns)
                scratch.executemany(
                    f"INSERT INTO actual_resources VALUES ({placeholders})",
                    actual_values,
                )
                scratch.executemany(
                    f"INSERT INTO expected_resources VALUES ({placeholders})",
                    [
                        tuple(_expected_value(row, name) for name in columns)
                        for row in rows
                    ],
                )
                identity_difference_count = _difference_count(
                    scratch, ("project_id", "report_year", "uncert_level")
                )
                source_multiset_difference_count = _difference_count(scratch, columns)
            finally:
                scratch.close()
        except (duckdb.Error, KeyError) as exc:
            issues.append(f"Typed source comparison could not be completed: {exc}")

    try:
        invalid_geometry_count = _count_invalid_legacy_geometry(rows)
    except duckdb.Error as exc:
        issues.append(f"Legacy source geometry could not be checked: {exc}")

    if snapshot is not None and publication is not None:
        if snapshot.invalid_geometry_count != invalid_geometry_count:
            issues.append(
                "SQLite snapshot invalid-geometry count does not match raw WKT"
            )
        if publication.invalid_geometry_count != invalid_geometry_count:
            issues.append(
                "DuckDB publication invalid-geometry count does not match raw WKT"
            )
        metadata_matches = metadata_matches and (
            snapshot.invalid_geometry_count == invalid_geometry_count
            and publication.invalid_geometry_count == invalid_geometry_count
        )

    return ResourceParityReport(
        canonical_rows,
        projected_rows,
        identity_difference_count,
        source_multiset_difference_count,
        metadata_matches,
        invalid_geometry_count,
        tuple(issues),
    )
