"""Built-in basemap source metadata.

Source records are deliberately metadata-only.  Deferred sources do not carry
download URLs or acquisition clients; a later task may add those details once
the acquisition recipe has been verified.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

__all__ = ["BasemapSource", "get_basemap_source", "list_basemap_sources"]


@dataclass(frozen=True, slots=True)
class BasemapSource:
    """Immutable metadata describing a built-in basemap source."""

    id: str
    provider: str
    state: Literal["deferred", "verified"]
    attribution: str
    unavailable_reason: str


_BASEMAP_SOURCES: Mapping[str, BasemapSource] = MappingProxyType(
    {
        "osm-admin-indonesia": BasemapSource(
            id="osm-admin-indonesia",
            provider="OpenStreetMap",
            state="deferred",
            attribution="© OpenStreetMap contributors",
            unavailable_reason=(
                "An automatic Indonesia-wide administrative-vector recipe has not been "
                "verified."
            ),
        )
    }
)


def get_basemap_source(source_id: str) -> BasemapSource:
    """Return the registered basemap source for ``source_id``.

    Raises:
        KeyError: If ``source_id`` is not a built-in source ID.
    """
    return _BASEMAP_SOURCES[source_id]


def list_basemap_sources() -> tuple[BasemapSource, ...]:
    """Return all registered basemap source records in stable ID order."""
    return tuple(_BASEMAP_SOURCES[key] for key in sorted(_BASEMAP_SOURCES))
