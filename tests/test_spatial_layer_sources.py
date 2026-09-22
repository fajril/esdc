"""Tests for the deferred basemap source registry."""

from dataclasses import FrozenInstanceError, replace

import pytest

from esdc.spatial_layers.sources import BasemapSource, get_basemap_source


def test_osm_indonesia_source_is_registered() -> None:
    source = get_basemap_source("osm-admin-indonesia")

    assert isinstance(source, BasemapSource)
    assert source.id == "osm-admin-indonesia"
    assert source.provider == "OpenStreetMap"


def test_unknown_source_id_raises_key_error() -> None:
    with pytest.raises(KeyError, match="does-not-exist"):
        get_basemap_source("does-not-exist")


def test_osm_indonesia_source_is_deferred_with_attribution_and_reason() -> None:
    source = get_basemap_source("osm-admin-indonesia")

    assert source.state == "deferred"
    assert "OpenStreetMap" in source.attribution
    assert (
        source.unavailable_reason
        == "An automatic Indonesia-wide administrative-vector recipe has not been verified."
    )
    assert source.unavailable_reason


def test_basemap_source_is_immutable() -> None:
    source = get_basemap_source("osm-admin-indonesia")

    with pytest.raises(FrozenInstanceError):
        source.state = "verified"  # type: ignore[misc]

    assert replace(source) == source
