"""Command-line management for user-provided spatial layers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer

from esdc.configs import Config
from esdc.pod_registry.store import get_esdc_sqlite_path
from esdc.spatial_layers.importer import SpatialLayerImportError, import_layer
from esdc.spatial_layers.models import LayerDetail, LayerKind, SpatialLayerVersion
from esdc.spatial_layers.publisher import publish_spatial_layers
from esdc.spatial_layers.sources import (
    BasemapSource,
    get_basemap_source,
    list_basemap_sources,
)
from esdc.spatial_layers.store import SpatialLayerStore

spatial_app = typer.Typer(no_args_is_help=True)
basemap_app = typer.Typer(invoke_without_command=True, no_args_is_help=False)
spatial_app.add_typer(basemap_app, name="basemap")


def _paths() -> tuple[Path, Path]:
    return Config.get_db_file(), get_esdc_sqlite_path()


def _store() -> SpatialLayerStore:
    return SpatialLayerStore(_paths()[1])


def _publish() -> None:
    duckdb_path, sqlite_path = _paths()
    publish_spatial_layers(duckdb_path, sqlite_path)
    from esdc.api_projection import _invalidate_analytical_caches

    _invalidate_analytical_caches()


def _version(store: SpatialLayerStore, layer_id: str) -> SpatialLayerVersion | None:
    versions = store.versions(layer_id)
    return versions[-1] if versions else None


class _ImporterStore:
    """Typing adapter for the importer's deliberately small store protocol."""

    def __init__(self, store: SpatialLayerStore) -> None:
        self._store = store

    def transaction(self) -> Any:
        return self._store.transaction()

    def create_layer(self, name: str, kind: LayerKind) -> Any:
        return self._store.create_layer(name, kind)

    def create_version(self, layer_id: str, source: Path, **kwargs: Any) -> Any:
        return self._store.create_version(layer_id, source, **kwargs)

    def insert_features(self, version_id: str, features: list[dict[str, Any]]) -> int:
        return self._store.insert_features(version_id, features)


@spatial_app.command(name="load")
def load(
    source: Annotated[Path, typer.Argument(exists=True, readable=True)],
    name: Annotated[str, typer.Option("--name", "-n")],
    layer: Annotated[str | None, typer.Option("--layer")] = None,
    kind: Annotated[LayerKind, typer.Option("--kind")] = "overlay",
    detail: Annotated[LayerDetail, typer.Option("--detail")] = "detailed",
) -> None:
    """Import a local vector source into the authoritative spatial store."""
    store = _store()
    try:
        version = import_layer(
            source, name, layer, kind, detail, store=_ImporterStore(store)
        )
    except (SpatialLayerImportError, ValueError, RuntimeError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from None
    finally:
        store.close()
    try:
        _publish()
    except Exception as exc:
        typer.echo(
            f"Error: spatial layer imported but publication failed: {exc}", err=True
        )
        raise typer.Exit(1) from None
    typer.echo(
        f"Loaded {name!r} ({kind}, {detail}) version {version.version}; "
        f"source hash {version.source_sha256}"
    )


@spatial_app.command(name="list")
def list_layers(
    include_archived: Annotated[
        bool, typer.Option("--all", help="Include archived layers.")
    ] = False,
) -> None:
    """List managed spatial layers and their latest imported versions."""
    store = _store()
    try:
        layers = store.list_layers(include_archived=include_archived)
        rows = [(layer, _version(store, layer.id)) for layer in layers]
    finally:
        store.close()
    typer.echo("ID\tNAME\tSTATE\tVERSION\tCRS\tGEOMETRY\tFEATURES\tIMPORTED")
    for layer, version in rows:
        typer.echo(
            "\t".join(
                str(value)
                for value in (
                    layer.id,
                    layer.name,
                    "active" if layer.active else "archived",
                    version.version if version else "-",
                    version.source_crs if version else "-",
                    version.geometry_type if version else "-",
                    version.feature_count if version else "-",
                    version.imported_at if version else "-",
                )
            )
        )


@spatial_app.command(name="describe")
def describe(
    identifier: Annotated[str | None, typer.Option("--id")] = None,
    name: Annotated[str | None, typer.Option("--name")] = None,
) -> None:
    """Show layer metadata, provenance, and version details."""
    if (identifier is None) == (name is None):
        raise typer.BadParameter("provide exactly one of --id or --name")
    target = identifier or name
    assert target is not None
    store = _store()
    try:
        layer = store.resolve(target)
        if layer is None:
            typer.echo(f"Error: active spatial layer not found: {target}", err=True)
            raise typer.Exit(1)
        version = _version(store, layer.id)
    finally:
        store.close()
    typer.echo(f"ID: {layer.id}")
    typer.echo(f"Name: {layer.name}")
    typer.echo(f"Kind: {layer.kind}")
    if version is not None:
        typer.echo(f"Version: {version.version}")
        typer.echo(f"CRS: {version.source_crs or '-'}")
        typer.echo(f"Geometry: {version.geometry_type or '-'}")
        typer.echo(f"Features: {version.feature_count}")
        typer.echo(f"Source: {version.source_path}")
        typer.echo(f"Source hash: {version.source_sha256}")
        metadata = json.loads(version.metadata_json or "{}")
        if metadata.get("attribution"):
            typer.echo(f"Attribution: {metadata['attribution']}")


@spatial_app.command(name="remove")
def remove(
    identifier: Annotated[str | None, typer.Option("--id")] = None,
    name: Annotated[str | None, typer.Option("--name")] = None,
    purge: Annotated[bool, typer.Option("--purge")] = False,
) -> None:
    """Archive a layer, or permanently purge its stored feature history."""
    if (identifier is None) == (name is None):
        raise typer.BadParameter("provide exactly one of --id or --name")
    target = identifier or name
    assert target is not None
    store = _store()
    try:
        layer = store.resolve(target)
        if layer is None:
            typer.echo(f"Error: active spatial layer not found: {target}", err=True)
            raise typer.Exit(1)
        if purge:
            typer.confirm(
                f"Permanently purge spatial layer {layer.name!r} ({layer.id}) "
                "and its versions?",
                abort=True,
            )
            store.purge_metadata(layer.id)
            action = "Purged"
        else:
            store.archive(target)
            action = "Archived"
    finally:
        store.close()
    _publish()
    typer.echo(f"{action} spatial layer {layer.name!r} ({layer.id})")


def _source_line(source: BasemapSource) -> str:
    return (
        f"{source.id}\t{source.provider}\t{source.state}\t"
        f"{source.unavailable_reason}"
    )


@basemap_app.command(name="sources")
def basemap_sources() -> None:
    """List built-in basemap source metadata."""
    typer.echo("ID\tPROVIDER\tSTATE\tUNAVAILABLE REASON")
    for source in list_basemap_sources():
        typer.echo(_source_line(source))


@basemap_app.command(name="install")
def basemap_install(
    source: Annotated[str, typer.Option("--source")],
) -> None:
    """Install a registered basemap source when an acquisition recipe exists."""
    try:
        record = get_basemap_source(source)
    except KeyError:
        typer.echo(f"Error: unknown basemap source: {source}", err=True)
        raise typer.Exit(1) from None
    if record.state == "deferred":
        typer.echo(
            f"Basemap source {record.id!r} is unavailable: {record.unavailable_reason}",
            err=True,
        )
        raise typer.Exit(1)


@basemap_app.callback()
def basemap_main(ctx: typer.Context) -> None:
    """Show active basemap status when no subcommand is provided."""
    if ctx.invoked_subcommand is not None:
        return
    store = _store()
    try:
        layers = [layer for layer in store.list_layers() if layer.kind == "basemap"]
        if not layers:
            typer.echo("No active basemap installed")
            typer.echo("Hint: run `esdc spatial basemap sources` to inspect sources.")
            typer.echo(
                "Hint: run `esdc spatial load --kind basemap ...` for a local file."
            )
            return
        layer = layers[0]
        version = _version(store, layer.id)
    finally:
        store.close()
    metadata = json.loads(version.metadata_json or "{}") if version else {}
    typer.echo(f"Name: {layer.name}")
    typer.echo(f"ID: {layer.id}")
    typer.echo(f"Version: {version.version if version else '-'}")
    typer.echo(f"Source: {metadata.get('source', 'local')}")
    typer.echo(f"CRS: {version.source_crs if version else '-'}")
    typer.echo(f"Attribution: {metadata.get('attribution', 'Not provided')}")
