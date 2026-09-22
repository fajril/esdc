"""Authoritative portable SQLite store for managed spatial layers."""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import uuid
from collections.abc import Iterable, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast

from .models import LayerKind, SpatialLayer, SpatialLayerVersion, utc_timestamp

__all__ = [
    "SpatialLayerStore",
    "SpatialLayerCapabilityError",
    "get_sqlite_connection",
]


class SpatialLayerCapabilityError(RuntimeError):
    """Raised when SQLite lacks the built-in capabilities required here."""


def get_sqlite_connection(path: Path) -> sqlite3.Connection:
    """Open an ordinary SQLite writer connection with foreign keys enabled."""
    connection = sqlite3.connect(str(path))
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


_DDL = """
CREATE TABLE IF NOT EXISTS spatial_layer (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    kind TEXT NOT NULL CHECK (kind IN ('overlay', 'basemap')),
    created_at TEXT NOT NULL,
    archived_at TEXT
);
CREATE UNIQUE INDEX IF NOT EXISTS spatial_layer_active_name
    ON spatial_layer(name) WHERE archived_at IS NULL;
CREATE TABLE IF NOT EXISTS spatial_layer_version (
    id TEXT PRIMARY KEY,
    layer_id TEXT NOT NULL REFERENCES spatial_layer(id),
    version INTEGER NOT NULL,
    source_path TEXT NOT NULL,
    source_sha256 TEXT NOT NULL,
    source_crs TEXT,
    geometry_type TEXT,
    feature_count INTEGER NOT NULL DEFAULT 0,
    min_x REAL,
    min_y REAL,
    max_x REAL,
    max_y REAL,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    UNIQUE(layer_id, version)
);
CREATE TABLE IF NOT EXISTS spatial_layer_feature (
    id INTEGER PRIMARY KEY,
    version_id TEXT NOT NULL REFERENCES spatial_layer_version(id),
    properties_json TEXT NOT NULL DEFAULT '{}',
    geometry_wkb BLOB NOT NULL,
    min_x REAL NOT NULL,
    max_x REAL NOT NULL,
    min_y REAL NOT NULL,
    max_y REAL NOT NULL
);
CREATE VIRTUAL TABLE IF NOT EXISTS spatial_layer_feature_rtree
    USING rtree(feature_id, min_x, max_x, min_y, max_y);
"""


def _row_layer(row: sqlite3.Row) -> SpatialLayer:
    return SpatialLayer(
        id=row["id"], name=row["name"], kind=cast(LayerKind, row["kind"]),
        archived_at=row["archived_at"], created_at=row["created_at"],
    )


def _row_version(row: sqlite3.Row) -> SpatialLayerVersion:
    bounds = None
    if row["min_x"] is not None:
        bounds = (row["min_x"], row["min_y"], row["max_x"], row["max_y"])
    return SpatialLayerVersion(
        id=row["id"], layer_id=row["layer_id"], version=row["version"],
        source_path=row["source_path"], source_sha256=row["source_sha256"],
        source_crs=row["source_crs"], geometry_type=row["geometry_type"],
        feature_count=row["feature_count"], bounds=bounds,
        metadata_json=row["metadata_json"], created_at=row["created_at"],
    )


class SpatialLayerStore:
    """Transactional registry and feature store backed by portable SQLite."""

    def __init__(
        self, path: Path, connection: sqlite3.Connection | None = None
    ) -> None:
        self.path = Path(path)
        self.connection = connection or get_sqlite_connection(self.path)
        self.connection.row_factory = sqlite3.Row
        self._transaction_depth = 0
        self._initialize_schema()

    @contextmanager
    def transaction(self):
        """Run registry/version/feature writes as one immediate transaction."""
        outer = self._transaction_depth == 0
        if outer:
            self.connection.execute("BEGIN IMMEDIATE")
        self._transaction_depth += 1
        try:
            yield self
        except Exception:
            self._transaction_depth -= 1
            if outer:
                self.connection.rollback()
            raise
        else:
            self._transaction_depth -= 1
            if outer:
                self.connection.commit()

    @contextmanager
    def _write_context(self):
        if self._transaction_depth:
            yield
        else:
            with self.connection:
                yield

    def _initialize_schema(self) -> None:
        with self.connection:
            self.connection.executescript(_DDL)
            columns = {row[1] for row in self.connection.execute(
                "PRAGMA table_info(spatial_layer_feature)"
            )}
            for column, definition in (
                ("geometry_wkb", "BLOB"), ("min_x", "REAL"), ("max_x", "REAL"),
                ("min_y", "REAL"), ("max_y", "REAL"),
            ):
                if column not in columns:
                    self.connection.execute(
                        "ALTER TABLE spatial_layer_feature ADD COLUMN "
                        f"{column} {definition}"
                    )
            try:
                self.connection.execute(
                    "CREATE VIRTUAL TABLE IF NOT EXISTS spatial_layer_feature_rtree "
                    "USING rtree(feature_id, min_x, max_x, min_y, max_y)"
                )
            except sqlite3.Error as exc:
                raise SpatialLayerCapabilityError(
                    "SQLite RTree support is required for spatial layers"
                ) from exc

    def _table_exists(self, name: str) -> bool:
        return bool(self.connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
        ).fetchone())

    def create_layer(self, name: str, kind: LayerKind = "overlay") -> SpatialLayer:
        if kind not in ("overlay", "basemap"):
            raise ValueError("kind must be 'overlay' or 'basemap'")
        now = utc_timestamp()
        layer = SpatialLayer(str(uuid.uuid4()), name, kind, created_at=now)
        try:
            with self._write_context():
                self.connection.execute(
                    "INSERT INTO spatial_layer(id,name,kind,created_at) "
                    "VALUES (?,?,?,?)",
                    (layer.id, layer.name, layer.kind, now),
                )
        except sqlite3.IntegrityError as exc:
            raise ValueError(
                f"an active spatial layer named {name!r} already exists"
            ) from exc
        return layer

    def create_version(
        self,
        layer_id: str,
        source: Path,
        *,
        source_crs: str | None = None,
        geometry_type: str | None = None,
        feature_count: int = 0,
        bounds: Sequence[float] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> SpatialLayerVersion:
        """Create one provenance version for the supplied source."""
        source_path = Path(source)
        source_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
        row = self.connection.execute(
            "SELECT COALESCE(MAX(version), 0) + 1 "
            "FROM spatial_layer_version WHERE layer_id=?",
            (layer_id,),
        ).fetchone()
        version_number = int(row[0])
        version_id = str(uuid.uuid4())
        now = utc_timestamp()
        normalized_bounds = (
            tuple(float(value) for value in bounds) if bounds is not None else None
        )
        if normalized_bounds is not None and len(normalized_bounds) != 4:
            raise ValueError("bounds must contain four values")
        record = SpatialLayerVersion(
            id=version_id, layer_id=layer_id, version=version_number,
            source_path=str(source_path), source_sha256=source_sha256,
            source_crs=source_crs, geometry_type=geometry_type,
            feature_count=feature_count,
            bounds=cast(tuple[float, float, float, float] | None, normalized_bounds),
            metadata_json=json.dumps(metadata or {}, sort_keys=True), created_at=now,
        )
        with self._write_context():
            self.connection.execute(
                """INSERT INTO spatial_layer_version
                (id,layer_id,version,source_path,source_sha256,source_crs,geometry_type,
                 feature_count,min_x,min_y,max_x,max_y,metadata_json,created_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (record.id, record.layer_id, record.version, record.source_path,
                 record.source_sha256, record.source_crs, record.geometry_type,
                 record.feature_count, *(record.bounds or (None, None, None, None)),
                 record.metadata_json, record.created_at),
            )
        return record

    add_version = create_version

    def insert_features(
        self,
        version_id: str,
        features: Iterable[Mapping[str, Any]],
    ) -> int:
        count = 0
        with self._write_context():
            for feature in features:
                geometry = feature.get(
                    "geometry_wkb", feature.get("geometry", feature.get("wkb"))
                )
                properties = feature.get("properties", {})
                if not isinstance(geometry, (bytes, bytearray, memoryview)):
                    raise TypeError("feature geometry must be WKB bytes")
                bounds = feature.get("bounds")
                if bounds is None or len(bounds) != 4:
                    raise ValueError("feature bounds must contain four values")
                try:
                    normalized_bounds = tuple(float(value) for value in bounds)
                except (TypeError, ValueError) as exc:
                    raise ValueError("feature bounds must be numeric") from exc
                if not all(math.isfinite(value) for value in normalized_bounds):
                    raise ValueError("feature bounds must be finite")
                self.connection.execute(
                    "INSERT INTO spatial_layer_feature "
                    "(version_id,properties_json,geometry_wkb,min_x,max_x,min_y,max_y) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        version_id,
                        json.dumps(properties, sort_keys=True),
                        bytes(geometry),
                        normalized_bounds[0], normalized_bounds[2],
                        normalized_bounds[1], normalized_bounds[3],
                    ),
                )
                feature_id = self.connection.execute(
                    "SELECT last_insert_rowid()"
                ).fetchone()[0]
                self.connection.execute(
                    "INSERT INTO spatial_layer_feature_rtree "
                    "(feature_id,min_x,max_x,min_y,max_y) VALUES (?,?,?,?,?)",
                    (feature_id, normalized_bounds[0], normalized_bounds[2],
                     normalized_bounds[1], normalized_bounds[3]),
                )
                count += 1
            self.connection.execute(
                "UPDATE spatial_layer_version SET feature_count=? WHERE id=?",
                (count, version_id),
            )
        return count

    def list_layers(self, *, include_archived: bool = False) -> list[SpatialLayer]:
        sql = "SELECT * FROM spatial_layer"
        if not include_archived:
            sql += " WHERE archived_at IS NULL"
        sql += " ORDER BY name"
        return [_row_layer(row) for row in self.connection.execute(sql)]

    def resolve(self, identifier: str) -> SpatialLayer | None:
        row = self.connection.execute(
            "SELECT * FROM spatial_layer "
            "WHERE (id=? OR name=?) AND archived_at IS NULL",
            (identifier, identifier),
        ).fetchone()
        return _row_layer(row) if row else None

    def archive(self, identifier: str) -> SpatialLayer:
        layer = self.resolve(identifier)
        if layer is None:
            raise KeyError(f"active spatial layer not found: {identifier}")
        archived_at = utc_timestamp()
        with self.connection:
            self.connection.execute(
                "UPDATE spatial_layer SET archived_at=? WHERE id=?",
                (archived_at, layer.id),
            )
        return SpatialLayer(
            layer.id, layer.name, layer.kind, archived_at, layer.created_at
        )

    def purge_metadata(self, layer_id: str) -> None:
        """Delete one layer's registry rows by exact resolved ID.

        Only the layer ID is accepted. Re-resolving a display name here could
        delete an archived layer that shares its name with the active layer the
        caller confirmed, so callers must resolve first and pass that exact ID.
        """
        row = self.connection.execute(
            "SELECT id FROM spatial_layer WHERE id=?", (layer_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"spatial layer not found: {layer_id}")
        with self.connection:
            self.connection.execute(
                "DELETE FROM spatial_layer_feature_rtree WHERE feature_id IN "
                "(SELECT f.id FROM spatial_layer_feature AS f "
                "JOIN spatial_layer_version AS v ON v.id=f.version_id "
                "WHERE v.layer_id=?)", (row["id"],)
            )
            self.connection.execute(
                "DELETE FROM spatial_layer_feature WHERE version_id IN "
                "(SELECT id FROM spatial_layer_version WHERE layer_id=?)", (row["id"],)
            )
            self.connection.execute(
                "DELETE FROM spatial_layer_version WHERE layer_id=?", (row["id"],)
            )
            self.connection.execute(
                "DELETE FROM spatial_layer WHERE id=?", (row["id"],)
            )

    def versions(self, identifier: str) -> list[SpatialLayerVersion]:
        layer = self.resolve(identifier)
        if layer is None:
            return []
        return [
            _row_version(row)
            for row in self.connection.execute(
                "SELECT * FROM spatial_layer_version "
                "WHERE layer_id=? ORDER BY version",
                (layer.id,),
            )
        ]

    def close(self) -> None:
        self.connection.close()
