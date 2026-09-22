"""CLI contract tests for spatial-layer commands."""

from types import SimpleNamespace

from typer.testing import CliRunner

from esdc.esdc import app
from esdc.spatial_layers.models import SpatialLayer, SpatialLayerVersion

runner = CliRunner()


def test_basemap_sources_lists_deferred_osm_source() -> None:
    result = runner.invoke(app, ["spatial", "basemap", "sources"])

    assert result.exit_code == 0
    assert "osm-admin-indonesia" in result.stdout
    assert "deferred" in result.stdout
    assert "recipe has not been verified" in result.stdout


def test_deferred_basemap_install_rejects_without_store_or_network(
    monkeypatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        "esdc.commands.spatial._store", lambda: calls.append("store")
    )
    monkeypatch.setattr(
        "esdc.commands.spatial._publish", lambda: calls.append("publish")
    )

    result = runner.invoke(
        app,
        ["spatial", "basemap", "install", "--source", "osm-admin-indonesia"],
    )

    assert result.exit_code == 1
    assert "unavailable" in result.stderr
    assert calls == []


def test_basemap_sources_does_not_require_native_extension(monkeypatch) -> None:
    def fail_store() -> None:
        raise AssertionError("sources must not open SQLite")

    monkeypatch.setattr("esdc.commands.spatial._store", fail_store)
    result = runner.invoke(app, ["spatial", "basemap", "sources"])

    assert result.exit_code == 0
    assert "osm-admin-indonesia" in result.stdout


def test_load_requires_a_source_path() -> None:
    result = runner.invoke(
        app,
        ["spatial", "load", "--name", "roads", "/does/not/exist.geojson"],
    )

    assert result.exit_code != 0


def test_remove_requires_exactly_one_identifier() -> None:
    result = runner.invoke(app, ["spatial", "remove"])

    assert result.exit_code != 0
    assert "exactly one" in result.stderr


def test_describe_requires_exactly_one_selector() -> None:
    missing = runner.invoke(app, ["spatial", "describe"])
    both = runner.invoke(app, ["spatial", "describe", "--id", "x", "--name", "y"])

    assert missing.exit_code != 0
    assert "exactly one" in missing.stderr
    assert both.exit_code != 0
    assert "exactly one" in both.stderr


def test_describe_accepts_id_or_name_selector(monkeypatch) -> None:
    layer = SpatialLayer("layer-id", "Indonesia", "basemap")
    fake_store = SimpleNamespace(
        resolve=lambda value: layer if value in {"layer-id", "Indonesia"} else None,
        versions=lambda _id: [],
        close=lambda: None,
    )
    monkeypatch.setattr("esdc.commands.spatial._store", lambda: fake_store)

    by_id = runner.invoke(app, ["spatial", "describe", "--id", "layer-id"])
    by_name = runner.invoke(app, ["spatial", "describe", "--name", "Indonesia"])

    assert by_id.exit_code == 0 and "ID: layer-id" in by_id.stdout
    assert by_name.exit_code == 0 and "Name: Indonesia" in by_name.stdout


def test_bare_basemap_without_active_layer_shows_hints(monkeypatch) -> None:
    fake_store = SimpleNamespace(list_layers=lambda: [], close=lambda: None)
    monkeypatch.setattr("esdc.commands.spatial._store", lambda: fake_store)

    result = runner.invoke(app, ["spatial", "basemap"])

    assert result.exit_code == 0
    assert "No active basemap installed" in result.stdout
    assert "basemap sources" in result.stdout
    assert "spatial load --kind basemap" in result.stdout


def test_bare_basemap_active_status(monkeypatch) -> None:
    layer = SpatialLayer("layer-id", "Indonesia", "basemap")
    version = SpatialLayerVersion(
        "version-id", "layer-id", 3, "/tmp/admin.gpkg", "sha", "EPSG:4326",
        "Polygon", 12, metadata_json='{"source":"local-admin","attribution":"© data"}',
    )
    fake_store = SimpleNamespace(
        list_layers=lambda: [layer],
        versions=lambda _id: [version],
        close=lambda: None,
    )
    monkeypatch.setattr("esdc.commands.spatial._store", lambda: fake_store)

    result = runner.invoke(app, ["spatial", "basemap"])

    assert result.exit_code == 0
    for text in (
        "Indonesia", "layer-id", "Version: 3", "local-admin", "EPSG:4326", "© data"
    ):
        assert text in result.stdout


def test_load_does_not_claim_success_when_publication_fails(monkeypatch, tmp_path) -> None:
    source = tmp_path / "source.geojson"
    source.write_text("placeholder")
    version = SpatialLayerVersion("v", "l", 1, str(source), "sha")
    fake_store = SimpleNamespace(close=lambda: None)
    monkeypatch.setattr("esdc.commands.spatial._store", lambda: fake_store)
    monkeypatch.setattr(
        "esdc.commands.spatial.import_layer", lambda *args, **kwargs: version
    )
    monkeypatch.setattr(
        "esdc.commands.spatial._publish",
        lambda: (_ for _ in ()).throw(RuntimeError("duckdb unavailable")),
    )

    result = runner.invoke(app, ["spatial", "load", str(source), "--name", "roads"])

    assert result.exit_code == 1
    assert "publication failed" in result.stderr
    assert "Loaded" not in result.stdout


def test_remove_purge_by_name_passes_exact_resolved_id(monkeypatch) -> None:
    layer = SpatialLayer("active-id", "roads", "overlay")
    calls: list[tuple[str, str]] = []
    fake_store = SimpleNamespace(
        resolve=lambda value: layer if value == "roads" else None,
        purge_metadata=lambda value: calls.append(("purge", value)),
        archive=lambda value: calls.append(("archive", value)),
        close=lambda: None,
    )
    monkeypatch.setattr("esdc.commands.spatial._store", lambda: fake_store)
    monkeypatch.setattr("esdc.commands.spatial._publish", lambda: None)

    result = runner.invoke(
        app, ["spatial", "remove", "--name", "roads", "--purge"], input="y\n"
    )

    assert result.exit_code == 0
    assert calls == [("purge", "active-id")]
    assert "active-id" in result.stdout


def test_remove_purge_cancellation_does_not_purge(monkeypatch) -> None:
    layer = SpatialLayer("active-id", "roads", "overlay")
    calls: list[tuple[str, str]] = []
    fake_store = SimpleNamespace(
        resolve=lambda value: layer,
        purge_metadata=lambda value: calls.append(("purge", value)),
        archive=lambda value: calls.append(("archive", value)),
        close=lambda: None,
    )
    monkeypatch.setattr("esdc.commands.spatial._store", lambda: fake_store)
    monkeypatch.setattr("esdc.commands.spatial._publish", lambda: None)

    result = runner.invoke(
        app, ["spatial", "remove", "--id", "active-id", "--purge"], input="n\n"
    )

    assert result.exit_code != 0
    assert calls == []


def test_remove_purge_unknown_identifier_does_not_purge(monkeypatch) -> None:
    calls: list[tuple[str, str]] = []
    fake_store = SimpleNamespace(
        resolve=lambda value: None,
        purge_metadata=lambda value: calls.append(("purge", value)),
        archive=lambda value: calls.append(("archive", value)),
        close=lambda: None,
    )
    monkeypatch.setattr("esdc.commands.spatial._store", lambda: fake_store)
    monkeypatch.setattr("esdc.commands.spatial._publish", lambda: None)

    result = runner.invoke(app, ["spatial", "remove", "--id", "missing", "--purge"])

    assert result.exit_code == 1
    assert "not found" in result.stderr
    assert calls == []


def test_spatial_publish_invalidates_analytical_caches(monkeypatch, tmp_path) -> None:
    """A successful spatial publication must not leave stale cached answers."""
    from diskcache import Cache

    from esdc.commands.spatial import _publish
    from esdc.configs import Config

    monkeypatch.setenv("ESDC_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setattr("esdc.chat.tools._sql_cache", None)
    monkeypatch.setattr("esdc.chat.tools._tool_cache", None)

    sql_observer = Cache(str(Config.get_cache_dir() / "sql_results"))
    tool_observer = Cache(str(Config.get_cache_dir() / "tool_results"))
    try:
        sql_observer["query"] = "stale"
        tool_observer["tool"] = "stale"

        monkeypatch.setattr(
            "esdc.commands.spatial.publish_spatial_layers", lambda *_: None
        )
        monkeypatch.setattr(
            "esdc.commands.spatial._paths",
            lambda: (tmp_path / "db.duckdb", tmp_path / "source.sqlite"),
        )

        _publish()

        assert "query" not in sql_observer
        assert "tool" not in tool_observer
    finally:
        sql_observer.close()
        tool_observer.close()
