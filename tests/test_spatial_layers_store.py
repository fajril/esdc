"""Focused tests for the portable managed spatial-layer registry."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from esdc.spatial_layers.models import SpatialLayerVersion
from esdc.spatial_layers.store import SpatialLayerStore, get_sqlite_connection


def test_connection_is_ordinary_sqlite(tmp_path: Path) -> None:
    connection = get_sqlite_connection(tmp_path / "spatial.sqlite")
    try:
        assert connection.execute("PRAGMA foreign_keys").fetchone() == (1,)
    finally:
        connection.close()


def test_schema_has_wkb_and_builtin_rtree(tmp_path: Path) -> None:
    store = SpatialLayerStore(tmp_path / "spatial.sqlite")
    try:
        columns = {row[1] for row in store.connection.execute("PRAGMA table_info(spatial_layer_feature)")}
        assert {"geometry_wkb", "min_x", "max_x", "min_y", "max_y"} <= columns
        assert store.connection.execute(
            "SELECT sql FROM sqlite_master WHERE name='spatial_layer_feature_rtree'"
        ).fetchone()[0].lower().startswith("create virtual table")
    finally:
        store.close()


def test_created_layer_version_and_feature_rtree_rows(tmp_path: Path) -> None:
    source = tmp_path / "input.geojson"
    source.write_text("{}", encoding="utf-8")
    store = SpatialLayerStore(tmp_path / "spatial.sqlite")
    try:
        layer = store.create_layer("roads", "overlay")
        version = store.create_version(layer.id, source, source_crs="EPSG:4326")
        assert isinstance(version, SpatialLayerVersion)
        assert version.source_sha256 == hashlib.sha256(source.read_bytes()).hexdigest()
        assert store.insert_features(version.id, [{
            "geometry_wkb": b"wkb", "bounds": (1, 2, 3, 4), "properties": {"x": 1}
        }]) == 1
        assert tuple(store.connection.execute(
            "SELECT geometry_wkb,min_x,max_x,min_y,max_y FROM spatial_layer_feature"
        ).fetchone()) == (b"wkb", 1.0, 3.0, 2.0, 4.0)
        assert store.connection.execute(
            "SELECT feature_id,min_x,max_x,min_y,max_y FROM spatial_layer_feature_rtree"
        ).fetchone()[1:] == (1.0, 3.0, 2.0, 4.0)
    finally:
        store.close()


def test_missing_bounds_rejected_before_write(tmp_path: Path) -> None:
    store = SpatialLayerStore(tmp_path / "spatial.sqlite")
    try:
        layer = store.create_layer("roads")
        source = tmp_path / "source"
        source.write_text("x", encoding="utf-8")
        version = store.create_version(layer.id, source)
        with pytest.raises(ValueError, match="bounds"):
            store.insert_features(version.id, [{"geometry_wkb": b"wkb"}])
        assert tuple(store.connection.execute("SELECT COUNT(*) FROM spatial_layer_feature").fetchone()) == (0,)
    finally:
        store.close()


@pytest.mark.parametrize("bounds", [(0, 0, float("inf"), 1), (0, 0, 1, float("-inf"))])
def test_non_finite_bounds_rejected_before_write(tmp_path: Path, bounds: tuple[float, ...]) -> None:
    store = SpatialLayerStore(tmp_path / "spatial.sqlite")
    try:
        layer = store.create_layer("roads")
        source = tmp_path / "source"
        source.write_text("x", encoding="utf-8")
        version = store.create_version(layer.id, source)
        with pytest.raises(ValueError, match="finite"):
            store.insert_features(version.id, [{"geometry_wkb": b"wkb", "bounds": bounds}])
        assert tuple(store.connection.execute("SELECT COUNT(*) FROM spatial_layer_feature").fetchone()) == (0,)
        assert tuple(store.connection.execute("SELECT COUNT(*) FROM spatial_layer_feature_rtree").fetchone()) == (0,)
    finally:
        store.close()


def test_duplicate_archive_and_atomic_purge(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.write_text("x", encoding="utf-8")
    store = SpatialLayerStore(tmp_path / "spatial.sqlite")
    try:
        first = store.create_layer("roads")
        with pytest.raises(ValueError, match="already exists"):
            store.create_layer("roads")
        version = store.create_version(first.id, source)
        store.insert_features(version.id, [{"geometry_wkb": b"wkb", "bounds": (0, 0, 1, 1)}])
        assert store.archive(first.id).active is False
        replacement = store.create_layer("roads")
        assert replacement.id != first.id
        store.purge_metadata(first.id)
        assert tuple(store.connection.execute("SELECT COUNT(*) FROM spatial_layer_feature").fetchone()) == (0,)
        assert tuple(store.connection.execute("SELECT COUNT(*) FROM spatial_layer_feature_rtree").fetchone()) == (0,)
    finally:
        store.close()


def _seed_layer(store: SpatialLayerStore, name: str, source: Path, wkb: bytes) -> tuple[str, str]:
    layer = store.create_layer(name)
    version = store.create_version(layer.id, source)
    store.insert_features(
        version.id, [{"geometry_wkb": wkb, "bounds": (0, 0, 1, 1)}]
    )
    return layer.id, version.id


def _count(store: SpatialLayerStore, table: str, where: str = "", params: tuple = ()) -> int:
    sql = f"SELECT COUNT(*) FROM {table}"
    if where:
        sql += f" WHERE {where}"
    return int(store.connection.execute(sql, params).fetchone()[0])


def test_purge_by_exact_id_ignores_reused_display_name(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.write_text("x", encoding="utf-8")
    store = SpatialLayerStore(tmp_path / "spatial.sqlite")
    try:
        archived_id, archived_version_id = _seed_layer(store, "roads", source, b"a")
        store.archive(archived_id)
        active_id, active_version_id = _seed_layer(store, "roads", source, b"b")

        store.purge_metadata(active_id)

        assert _count(store, "spatial_layer", "id=?", (active_id,)) == 0
        assert _count(store, "spatial_layer_version", "layer_id=?", (active_id,)) == 0
        assert _count(store, "spatial_layer_feature", "version_id=?", (active_version_id,)) == 0
        assert _count(store, "spatial_layer", "id=?", (archived_id,)) == 1
        assert _count(store, "spatial_layer_version", "layer_id=?", (archived_id,)) == 1
        assert _count(store, "spatial_layer_feature", "version_id=?", (archived_version_id,)) == 1
        assert _count(store, "spatial_layer_feature_rtree") == 1
    finally:
        store.close()


def test_purge_rejects_display_name_as_identifier(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.write_text("x", encoding="utf-8")
    store = SpatialLayerStore(tmp_path / "spatial.sqlite")
    try:
        layer_id, _ = _seed_layer(store, "roads", source, b"a")

        with pytest.raises(KeyError, match="not found"):
            store.purge_metadata("roads")

        assert _count(store, "spatial_layer", "id=?", (layer_id,)) == 1
        assert _count(store, "spatial_layer_feature") == 1
        assert _count(store, "spatial_layer_feature_rtree") == 1
    finally:
        store.close()
