"""Import and canonicalize local vector sources into the spatial layer store."""

from __future__ import annotations

import hashlib
import math
from contextlib import AbstractContextManager, nullcontext
from pathlib import Path
from typing import Any, Protocol, cast

import pyogrio
import pyogrio.raw
from pyproj import CRS, Transformer
from shapely import get_coordinates, is_valid_reason, wkb
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform

from .models import LayerDetail, LayerKind, SpatialLayerVersion

MAX_OVERVIEW_BYTES = 25 * 1024 * 1024
_SUPPORTED_GEOMETRIES = {
    "Point", "MultiPoint", "LineString", "MultiLineString", "Polygon", "MultiPolygon"
}


class SpatialLayerImportError(ValueError):
    """Raised when a local vector source cannot be safely imported."""


class _StoreProtocol(Protocol):
    def transaction(self) -> Any: ...
    def create_layer(self, name: str, kind: LayerKind) -> Any: ...
    def create_version(
        self, layer_id: str, source: Path, **kwargs: Any
    ) -> Any: ...
    def insert_features(
        self, version_id: str, features: list[dict[str, Any]]
    ) -> int: ...


def _source_layer(source: Path, requested: str | None) -> str | None:
    if source.suffix.lower() == ".shp":
        for sidecar in (source.with_suffix(".shx"), source.with_suffix(".dbf")):
            if not sidecar.exists():
                raise SpatialLayerImportError(
                    f"shapefile is missing required sibling {sidecar.name}"
                )
        if not source.with_suffix(".prj").exists():
            raise SpatialLayerImportError(
                "shapefile is missing required CRS sidecar .prj"
            )
    if not source.exists() or source.stat().st_size == 0:
        raise SpatialLayerImportError("source does not exist or is empty")
    try:
        layers = pyogrio.list_layers(source)
    except Exception as exc:
        raise SpatialLayerImportError(
            f"unable to inspect vector source: {exc}"
        ) from exc
    names = [str(row[0]) for row in layers]
    if requested is None:
        if len(names) != 1:
            raise SpatialLayerImportError(
                "source contains multiple layers; source_layer is required"
            )
        return names[0] if names else None
    if requested not in names:
        raise SpatialLayerImportError(f"source layer not found: {requested}")
    return requested


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "item"):
        try:
            return _json_value(value.item())
        except (TypeError, ValueError):
            pass
    return str(value)


def _read_features(
    source: Path, layer: str | None
) -> tuple[dict[str, Any], list[tuple[BaseGeometry, dict[str, Any]]]]:
    """Read without requiring optional geopandas (pyogrio's dataframe API does)."""
    info, _fids, geometries, columns = pyogrio.raw.read(
        str(source), layer=layer, read_geometry=True
    )
    fields = [str(field) for field in info.get("fields", [])]
    rows: list[tuple[BaseGeometry, dict[str, Any]]] = []
    if geometries is None:
        return info, rows
    for index, raw_geometry in enumerate(geometries):
        geometry = wkb.loads(bytes(raw_geometry))
        properties = {
            field: _json_value(columns[column_index][index])
            for column_index, field in enumerate(fields)
        }
        rows.append((geometry, properties))
    return info, rows


def import_layer(
    source: Path,
    name: str,
    source_layer: str | None,
    kind: LayerKind,
    detail: LayerDetail,
    *,
    store: _StoreProtocol,
) -> SpatialLayerVersion:
    """Import one local vector source, canonicalized to EPSG:4326.

    Validation and CRS conversion happen before opening the store transaction.
    ``overview`` is reserved for curated sources; arbitrary local files are
    deliberately rejected because their administrative role cannot be inferred.
    """
    source = Path(source)
    if detail not in ("detailed", "overview"):
        raise ValueError("detail must be 'detailed' or 'overview'")
    if kind not in ("overlay", "basemap"):
        raise ValueError("kind must be 'overlay' or 'basemap'")
    if detail == "overview":
        raise SpatialLayerImportError(
            "local overview imports require curated source role metadata"
        )
    selected = _source_layer(source, source_layer)
    try:
        info, source_rows = _read_features(source, selected)
    except Exception as exc:
        raise SpatialLayerImportError(f"unable to read vector source: {exc}") from exc
    crs_value = info.get("crs")
    if crs_value is None:
        raise SpatialLayerImportError("source has unknown CRS")
    try:
        source_crs = CRS.from_user_input(crs_value)
    except Exception as exc:
        raise SpatialLayerImportError("source has unknown CRS") from exc
    if source_crs == CRS.from_epsg(4326):
        converter = None
    else:
        converter = Transformer.from_crs(
            source_crs, CRS.from_epsg(4326), always_xy=True
        ).transform

    if not source_rows:
        raise SpatialLayerImportError("source contains no geometry features")
    geometries: list[BaseGeometry] = []
    properties: list[dict[str, Any]] = []
    geometry_types: set[str] = set()
    for index, (geometry, row_properties) in enumerate(source_rows):
        if not isinstance(geometry, BaseGeometry) or geometry.is_empty:
            raise SpatialLayerImportError(
                f"feature {index} is missing or empty geometry"
            )
        geometry = transform(converter, geometry) if converter else geometry
        coordinates = get_coordinates(geometry, include_z=geometry.has_z)
        if not all(math.isfinite(value) for value in coordinates.flat):
            raise SpatialLayerImportError(
                f"feature {index} has non-finite coordinates after CRS transformation"
            )
        if not geometry.is_valid:
            raise SpatialLayerImportError(
                f"feature {index} has invalid geometry: {is_valid_reason(geometry)}"
            )
        geometry_type = geometry.geom_type
        if geometry_type not in _SUPPORTED_GEOMETRIES:
            raise SpatialLayerImportError(f"unsupported geometry type: {geometry_type}")
        geometry_types.add(geometry_type)
        geometries.append(geometry)
        properties.append(row_properties)
    if len(geometry_types) != 1:
        raise SpatialLayerImportError("mixed geometry types are not supported")
    min_x = min(geometry.bounds[0] for geometry in geometries)
    min_y = min(geometry.bounds[1] for geometry in geometries)
    max_x = max(geometry.bounds[2] for geometry in geometries)
    max_y = max(geometry.bounds[3] for geometry in geometries)
    attribute_schema = {
        str(column): str(dtype) for column, dtype in zip(
            info.get("fields", []), info.get("dtypes", []), strict=True
        )
    }
    metadata = {"detail": detail, "attribute_schema": attribute_schema}
    source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
    records = [
        {
            "geometry": wkb.dumps(geometry),
            "bounds": geometry.bounds,
            "properties": row_properties,
        }
        for geometry, row_properties in zip(geometries, properties, strict=True)
    ]

    transaction = getattr(store, "transaction", None)
    context = cast(
        AbstractContextManager[Any],
        transaction() if callable(transaction) else nullcontext(store),
    )
    with context:
        layer = store.create_layer(name, kind)
        version = store.create_version(
            layer.id, source, source_crs=source_crs.to_string(),
            geometry_type=next(iter(geometry_types)), feature_count=len(records),
            bounds=(min_x, min_y, max_x, max_y), metadata=metadata,
        )
        store.insert_features(version.id, records)
    # Keep the explicit hash computation above as an input-to-output validation;
    # the authoritative value is populated by SpatialLayerStore.create_version.
    if version.source_sha256 != source_sha256:
        raise RuntimeError("source hash changed while importing")
    return version


__all__ = ["SpatialLayerImportError", "import_layer"]
