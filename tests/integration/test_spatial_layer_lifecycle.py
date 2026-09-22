"""End-to-end lifecycle and routing-boundary tests for user spatial layers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, cast

import duckdb
import pytest

from esdc.chat.query_classifier import route_query
from esdc.chat.skills import discover_skills, select_skills_for_route
from esdc.spatial_layers.importer import SpatialLayerImportError, import_layer
from esdc.spatial_layers.publisher import publish_spatial_layers
from esdc.spatial_layers.store import SpatialLayerStore


def _geojson(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "crs": {"type": "name", "properties": {"name": "EPSG:4326"}},
                "features": [
                    {
                        "type": "Feature",
                        "geometry": {"type": "Point", "coordinates": [106.8, -6.2]},
                        "properties": {"name": "Jakarta", "kind": "city"},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def _open_duckdb_read_connection(path: Path) -> duckdb.DuckDBPyConnection:
    connection = duckdb.connect(str(path), read_only=True)
    connection.execute("LOAD spatial")
    return connection


def test_geojson_lifecycle_registry_rtree_and_duckdb_mirror(tmp_path: Path) -> None:
    source = tmp_path / "cities.geojson"
    sqlite_path = tmp_path / "spatial.sqlite"
    duckdb_path = tmp_path / "esdc.duckdb"
    _geojson(source)
    store = SpatialLayerStore(sqlite_path)

    try:
        version = import_layer(
            source,
            "cities",
            None,
            "overlay",
            "detailed",
            store=cast(Any, store),
        )
        layer = store.resolve("cities")
        assert layer is not None
        assert version.source_sha256 == hashlib.sha256(source.read_bytes()).hexdigest()
        assert version.source_path == str(source)
        assert version.source_crs == "EPSG:4326"
        assert version.feature_count == 1
        assert "attribute_schema" in version.metadata_json
        assert store.connection.execute(
            "SELECT COUNT(*) FROM spatial_layer_feature WHERE version_id=?",
            (version.id,),
        ).fetchone()[0] == 1
        assert store.connection.execute(
            "SELECT COUNT(*) FROM spatial_layer_feature_rtree"
        ).fetchone()[0] == 1
    finally:
        store.close()

    published = publish_spatial_layers(duckdb_path, sqlite_path)
    assert published.layer_count == 1
    assert published.feature_count == 1
    table_name = published.table_names[0]

    connection = _open_duckdb_read_connection(duckdb_path)
    try:
        assert connection.execute(
            "SELECT COUNT(*) FROM spatial_layer_catalog"
        ).fetchone() == (1,)
        assert connection.execute(
            f'SELECT COUNT(*), COUNT(*) FILTER (WHERE geometry IS NOT NULL) FROM "{table_name}"'
        ).fetchone() == (1, 1)
        assert connection.execute(
            f"SELECT ST_GeometryType(geometry) FROM \"{table_name}\""
        ).fetchone() == ("POINT",)
        assert connection.execute(
            f"SELECT ST_Covers(ST_GeomFromText(?), geometry) FROM \"{table_name}\"",
            ("POLYGON ((106 -7, 108 -7, 108 -5, 106 -5, 106 -7))",),
        ).fetchone() == (True,)
    finally:
        connection.close()

    store = SpatialLayerStore(sqlite_path)
    try:
        store.archive("cities")
    finally:
        store.close()
    republished = publish_spatial_layers(duckdb_path, sqlite_path)
    assert republished.layer_count == 0
    assert republished.feature_count == 0
    connection = _open_duckdb_read_connection(duckdb_path)
    try:
        assert connection.execute(
            "SELECT COUNT(*) FROM spatial_layer_catalog"
        ).fetchone() == (0,)
        assert connection.execute(
            "SELECT COUNT(*) FROM information_schema.tables "
            "WHERE table_name LIKE 'user_spatial_%'"
        ).fetchone() == (0,)
    finally:
        connection.close()


def _bowtie_geojson(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "crs": {"type": "name", "properties": {"name": "EPSG:4326"}},
                "features": [
                    {
                        "type": "Feature",
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [[[0, 0], [1, 1], [1, 0], [0, 1], [0, 0]]],
                        },
                        "properties": {},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def test_invalid_geometry_import_leaves_no_partial_store_rows(tmp_path: Path) -> None:
    source = tmp_path / "bowtie.geojson"
    sqlite_path = tmp_path / "spatial.sqlite"
    _bowtie_geojson(source)
    store = SpatialLayerStore(sqlite_path)
    try:
        with pytest.raises(SpatialLayerImportError, match=r"feature 0.*invalid"):
            import_layer(
                source,
                "bowtie",
                None,
                "overlay",
                "detailed",
                store=cast(Any, store),
            )
        for table in (
            "spatial_layer",
            "spatial_layer_version",
            "spatial_layer_feature",
            "spatial_layer_feature_rtree",
        ):
            assert store.connection.execute(
                f"SELECT COUNT(*) FROM {table}"
            ).fetchone()[0] == 0
    finally:
        store.close()


def test_spatial_routes_keep_authoritative_tools_for_visualization_requests() -> None:
    ordinary = route_query("Which fields are within a 10 km radius of Duri?")
    visualization = route_query("Show a map overlay of fields within a 10 km radius of Duri.")

    assert ordinary.allowed_tools == ("Spatial Resolver", "SQL Executor")
    assert visualization.allowed_tools == ordinary.allowed_tools
    assert ordinary.skill_names == visualization.skill_names == ("spatial_analysis",)
    selected = select_skills_for_route(visualization, discover_skills())
    assert len(selected) == 1
    assert "spatial_layer_catalog" in (selected[0].instructions or "")
