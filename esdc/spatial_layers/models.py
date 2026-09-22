"""Typed records returned by the managed spatial-layer store."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

LayerKind = Literal["overlay", "basemap"]
LayerDetail = Literal["detailed", "overview"]


@dataclass(frozen=True, slots=True)
class SpatialLayer:
    """A named layer in the local spatial catalog."""

    id: str
    name: str
    kind: LayerKind
    archived_at: str | None = None
    created_at: str | None = None

    @property
    def active(self) -> bool:
        return self.archived_at is None


@dataclass(frozen=True, slots=True)
class SpatialLayerVersion:
    """An immutable imported version and its provenance metadata."""

    id: str
    layer_id: str
    version: int
    source_path: str
    source_sha256: str
    source_crs: str | None = None
    geometry_type: str | None = None
    feature_count: int = 0
    bounds: tuple[float, float, float, float] | None = None
    metadata_json: str = "{}"
    created_at: str | None = None

    @property
    def imported_at(self) -> str | None:
        """Compatibility name used by catalog consumers."""
        return self.created_at


def utc_timestamp() -> str:
    """Return a stable, timezone-aware ISO-8601 timestamp."""
    return datetime.now().astimezone().isoformat(timespec="seconds")


__all__ = ["LayerDetail", "LayerKind", "SpatialLayer", "SpatialLayerVersion"]
