"""Focused tests for local vector import and canonicalization."""

from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pyogrio.raw
import pytest
from shapely import wkb
from shapely.geometry import Point, Polygon

from esdc.spatial_layers.importer import SpatialLayerImportError, import_layer


class RecordingStore:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []

    def transaction(self):
        return self._transaction()

    @contextmanager
    def _transaction(self):
        snapshot = self.calls.copy()
        try:
            yield self
        except Exception:
            self.calls[:] = snapshot
            raise

    def create_layer(self, name: str, kind: str) -> Any:
        self.calls.append(("layer", (name, kind)))
        return type("Layer", (), {"id": "layer-1"})()

    def create_version(self, layer_id: str, source: Path, **kwargs: Any) -> Any:
        self.calls.append(("version", kwargs))
        return type(
            "Version", (), {
                "id": "version-1",
                "source_sha256": hashlib.sha256(Path(source).read_bytes()).hexdigest(),
            }
        )()

    def insert_features(self, version_id: str, features: Any) -> int:
        features = list(features)
        self.calls.append(("features", features))
        return len(features)


class FailingStore(RecordingStore):
    def insert_features(self, version_id: str, features: Any) -> int:
        super().insert_features(version_id, features)
        raise RuntimeError("writer failed")


def _geojson(path: Path, *, crs: str | None = "EPSG:4326") -> None:
    payload = {
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [106.8, -6.2]},
            "properties": {"name": "Jakarta", "count": 1},
        }],
    }
    if crs is not None:
        payload["crs"] = {"type": "name", "properties": {"name": crs}}
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_import_normalizes_and_records_features(tmp_path: Path) -> None:
    source = tmp_path / "cities.geojson"
    _geojson(source)
    store = RecordingStore()

    version = import_layer(source, "cities", None, "basemap", "detailed", store=store)

    assert version.source_sha256 == hashlib.sha256(source.read_bytes()).hexdigest()
    assert store.calls[0] == ("layer", ("cities", "basemap"))
    metadata = store.calls[1][1]
    assert metadata["geometry_type"] == "Point"
    assert metadata["feature_count"] == 1
    assert metadata["metadata"]["attribute_schema"] == {"name": "object", "count": "int32"}
    feature = store.calls[2][1][0]
    assert isinstance(feature["geometry"], bytes)
    assert feature["properties"]["name"] == "Jakarta"


def test_unknown_crs_is_rejected_before_store_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "unknown.geojson"
    _geojson(source, crs=None)
    store = RecordingStore()

    import esdc.spatial_layers.importer as importer_module

    original = importer_module._read_features

    def no_crs(path, layer):
        info, rows = original(path, layer)
        info["crs"] = None
        return info, rows

    monkeypatch.setattr(importer_module, "_read_features", no_crs)
    with pytest.raises(SpatialLayerImportError, match="unknown CRS"):
        import_layer(source, "unknown", None, "overlay", "detailed", store=store)
    assert store.calls == []


def test_local_overview_is_rejected_without_curated_role(tmp_path: Path) -> None:
    source = tmp_path / "local.geojson"
    _geojson(source)

    with pytest.raises(SpatialLayerImportError, match="curated source role"):
        import_layer(source, "local", None, "basemap", "overview", store=RecordingStore())


def test_write_failure_rolls_back_layer_and_version(tmp_path: Path) -> None:
    source = tmp_path / "failed.geojson"
    _geojson(source)
    store = FailingStore()
    with pytest.raises(RuntimeError, match="writer failed"):
        import_layer(source, "failed", None, "overlay", "detailed", store=store)
    assert store.calls == []


@pytest.mark.parametrize("missing", [".shx", ".dbf", ".prj"])
def test_shapefile_missing_sidecar_rejected_before_writes(
    tmp_path: Path, missing: str
) -> None:
    source = tmp_path / "roads.shp"
    source.write_bytes(b"not empty")
    for suffix in (".shx", ".dbf", ".prj"):
        if suffix != missing:
            source.with_suffix(suffix).write_bytes(b"x")
    store = RecordingStore()
    with pytest.raises(SpatialLayerImportError, match="missing"):
        import_layer(source, "roads", None, "overlay", "detailed", store=store)
    assert store.calls == []


def test_multiple_layers_require_explicit_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "multi.gpkg"
    source.write_bytes(b"x")
    monkeypatch.setattr(
        "esdc.spatial_layers.importer.pyogrio.list_layers",
        lambda _source: [["one", "Point"], ["two", "Point"]],
    )
    with pytest.raises(SpatialLayerImportError, match="multiple layers"):
        import_layer(source, "multi", None, "overlay", "detailed", store=RecordingStore())


def test_mixed_geometry_rejected_before_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "mixed.geojson"
    _geojson(source)
    import esdc.spatial_layers.importer as importer_module

    # Two distinct geometry classes exercise the homogeneous-geometry guard.
    monkeypatch.setattr(importer_module, "_read_features", lambda *_: (
        {"crs": "EPSG:4326", "fields": [], "dtypes": []},
        [(Point(0, 0), {}), (Polygon([(0, 0), (1, 0), (1, 1)]), {})],
    ))
    with pytest.raises(SpatialLayerImportError, match="mixed geometry"):
        import_layer(source, "mixed", None, "overlay", "detailed", store=RecordingStore())


def test_empty_vector_rejected_before_writes(tmp_path: Path) -> None:
    source = tmp_path / "empty.geojson"
    source.touch()
    with pytest.raises(SpatialLayerImportError, match="empty"):
        import_layer(source, "empty", None, "overlay", "detailed", store=RecordingStore())


def _bowtie_geojson(path: Path) -> None:
    payload = {
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
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_self_intersecting_polygon_rejected_before_writes(tmp_path: Path) -> None:
    source = tmp_path / "bowtie.geojson"
    _bowtie_geojson(source)
    store = RecordingStore()

    with pytest.raises(SpatialLayerImportError, match=r"feature 0.*invalid"):
        import_layer(source, "bowtie", None, "overlay", "detailed", store=store)

    assert store.calls == []


def test_non_finite_transformed_coordinates_rejected_before_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "infinite.geojson"
    _geojson(source)
    import esdc.spatial_layers.importer as importer_module

    monkeypatch.setattr(
        importer_module,
        "_read_features",
        lambda *_: (
            {"crs": "EPSG:3857", "fields": [], "dtypes": []},
            [(Point(1.0, 2.0), {})],
        ),
    )
    monkeypatch.setattr(
        importer_module, "transform", lambda *_: Point(float("inf"), 0.0)
    )
    store = RecordingStore()

    with pytest.raises(SpatialLayerImportError, match=r"feature 0.*non-finite"):
        import_layer(source, "infinite", None, "overlay", "detailed", store=store)

    assert store.calls == []


@pytest.mark.parametrize("bad_z", [float("inf"), float("-inf"), float("nan")])
def test_non_finite_z_coordinate_rejected_before_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bad_z: float
) -> None:
    source = tmp_path / "z.geojson"
    _geojson(source)
    import esdc.spatial_layers.importer as importer_module

    monkeypatch.setattr(
        importer_module,
        "_read_features",
        lambda *_: (
            {"crs": "EPSG:4326", "fields": [], "dtypes": []},
            [(Point(106.8, -6.2, bad_z), {})],
        ),
    )
    store = RecordingStore()

    with pytest.raises(SpatialLayerImportError, match=r"feature 0.*non-finite"):
        import_layer(source, "z", None, "overlay", "detailed", store=store)

    assert store.calls == []


def test_geopackage_selected_layer_is_imported(tmp_path: Path) -> None:
    source = tmp_path / "layers.gpkg"
    geometry = np.array([wkb.dumps(Point(106.8, -6.2))], dtype=object)
    pyogrio.raw.write(
        str(source), geometry, [np.array(["first"])], ["name"],
        layer="first", driver="GPKG", geometry_type="Point", crs="EPSG:4326",
    )
    pyogrio.raw.write(
        str(source), geometry, [np.array(["second"])], ["name"],
        layer="second", driver="GPKG", geometry_type="Point", crs="EPSG:4326",
        append=True,
    )
    store = RecordingStore()
    import_layer(source, "selected", "second", "overlay", "detailed", store=store)

    features = store.calls[2][1]
    assert features[0]["properties"]["name"] == "second"
