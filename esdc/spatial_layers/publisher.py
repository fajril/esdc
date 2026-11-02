"""Publish the authoritative SQLite spatial-layer store into DuckDB."""

from __future__ import annotations

import contextlib
import sqlite3
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import duckdb

from esdc.dbmanager import get_duckdb_connection
from esdc.pod_registry.store import (
    get_sqlite_readonly_connection,
    is_missing_table_error,
)

__all__ = ["PublicationResult", "publish_spatial_layers"]


class SpatialLayerReader(Protocol):
    """Small source-reader contract kept independent of DuckDB."""

    def execute(self, sql: str, parameters: Any = (), /) -> Any: ...

    def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class PublicationResult:
    """Counts and names produced by one successful publication."""

    layer_count: int
    feature_count: int
    table_names: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _Layer:
    layer_id: str
    name: str
    version: int
    source_crs: str | None
    version_id: str

    @property
    def table_name(self) -> str:
        return f"user_spatial_{self.layer_id.replace('-', '')}"


def _open_source_reader(path: Path) -> SpatialLayerReader | None:
    """Open the portable SQLite-backed source."""
    return get_sqlite_readonly_connection(path)


def _read_layers(reader: SpatialLayerReader) -> list[_Layer]:
    try:
        rows = reader.execute(
            """SELECT l.id, l.name, v.version, v.source_crs, v.id
               FROM spatial_layer AS l
               JOIN spatial_layer_version AS v ON v.layer_id = l.id
               WHERE l.archived_at IS NULL
                 AND v.version = (
                     SELECT MAX(v2.version)
                     FROM spatial_layer_version AS v2
                     WHERE v2.layer_id = l.id
                 )
               ORDER BY l.id"""
        ).fetchall()
    except sqlite3.OperationalError as exc:
        if not is_missing_table_error(exc):
            raise
        return []
    return [
        _Layer(str(row[0]), str(row[1]), int(row[2]), row[3], str(row[4]))
        for row in rows
    ]


def _read_features(
    reader: SpatialLayerReader, version_id: str
) -> Iterator[tuple[int, bytes, str]]:
    rows = reader.execute(
        """SELECT id, geometry_wkb, properties_json
           FROM spatial_layer_feature
           WHERE version_id = ?
           ORDER BY id""",
        (version_id,),
    ).fetchall()
    for feature_id, geometry, properties in rows:
        if geometry is None:
            raise ValueError(f"feature {feature_id} has no geometry")
        yield int(feature_id), bytes(geometry), str(properties)


def _sql_literal(value: str | None) -> str:
    if value is None:
        return "NULL"
    return "'" + value.replace("'", "''") + "'"


def _catalog_view(conn: duckdb.DuckDBPyConnection, layers: list[_Layer]) -> None:
    conn.execute("DROP VIEW IF EXISTS spatial_layer_catalog")
    if not layers:
        conn.execute(
            """CREATE VIEW spatial_layer_catalog AS
               SELECT CAST(NULL AS VARCHAR) AS layer_id,
                      CAST(NULL AS VARCHAR) AS name,
                      CAST(NULL AS INTEGER) AS version,
                      CAST(NULL AS VARCHAR) AS source_crs,
                      CAST(NULL AS VARCHAR) AS table_name
               WHERE FALSE"""
        )
        return
    selects = [
        f"SELECT {_sql_literal(layer.layer_id)} AS layer_id, "
        f"{_sql_literal(layer.name)} AS name, {layer.version} AS version, "
        f"{_sql_literal(layer.source_crs)} AS source_crs, "
        f"{_sql_literal(layer.table_name)} AS table_name"
        for layer in layers
    ]
    conn.execute("CREATE VIEW spatial_layer_catalog AS " + " UNION ALL ".join(selects))


def publish_spatial_layers(duckdb_path: Path, sqlite_path: Path) -> PublicationResult:
    """Replace published user-layer objects in one independent DuckDB transaction."""
    reader = _open_source_reader(Path(sqlite_path))
    conn: duckdb.DuckDBPyConnection | None = None
    try:
        conn = get_duckdb_connection(Path(duckdb_path), read_only=False)
        layers = _read_layers(reader) if reader is not None else []
        conn.execute("BEGIN")
        feature_count = 0
        stale_tables = conn.execute(
            """SELECT table_name FROM information_schema.tables
               WHERE table_schema = 'main' AND table_name LIKE 'user_spatial_%'"""
        ).fetchall()
        for (table_name,) in stale_tables:
            escaped = str(table_name).replace('"', '""')
            conn.execute(f'DROP TABLE IF EXISTS "{escaped}"')
        for layer in layers:
            table = f'"{layer.table_name}"'
            conn.execute(
                f"CREATE TABLE {table} (feature_id BIGINT, "
                "properties_json VARCHAR, geometry GEOMETRY)"
            )
            if reader is not None:
                for feature_id, geometry, properties in _read_features(
                    reader, layer.version_id
                ):
                    conn.execute(
                        f"INSERT INTO {table} VALUES (?, ?, ST_GeomFromWKB(?))",
                        (feature_id, properties, geometry),
                    )
                    feature_count += 1
        _catalog_view(conn, layers)
        conn.execute("COMMIT")
        return PublicationResult(
            len(layers),
            feature_count,
            tuple(layer.table_name for layer in layers),
        )
    except Exception:
        if conn is not None:
            with contextlib.suppress(duckdb.Error):
                conn.execute("ROLLBACK")
        raise
    finally:
        if reader is not None:
            reader.close()
        if conn is not None:
            conn.close()
