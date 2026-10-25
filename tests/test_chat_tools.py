"""Tests for read-only SQL and spatial catalog geometry guidance."""

from pathlib import Path

import duckdb
from diskcache import Cache

from esdc.chat.tools import (
    _execute_sql_sync,
    execute_sql,
    invalidate_tool_cache,
    reset_sql_cache,
)
from esdc.dbmanager import invalidate_sql_cache


def _build_spatial_fixture(path: Path) -> None:
    """Build a purpose-specific spatial table independent of project_resources."""
    conn = duckdb.connect(str(path))
    try:
        conn.execute("INSTALL spatial")
        conn.execute("LOAD spatial")
        conn.execute(
            "CREATE TABLE synthetic_spatial_layer ("
            "project_name VARCHAR, geom GEOMETRY)"
        )
        conn.execute(
            "INSERT INTO synthetic_spatial_layer VALUES (?, ST_GeomFromText(?))",
            ["Duri Project", "POINT (101.123 -0.456)"],
        )
        conn.execute(
            "INSERT INTO synthetic_spatial_layer VALUES (?, ?)",
            ["Missing Geometry", None],
        )
    finally:
        conn.close()


def test_execute_sql_formats_geometry_from_spatial_table(
    tmp_path: Path, monkeypatch
) -> None:
    """Spatial SELECTs expose the derived geometry as WKT to chat callers."""
    db_path = tmp_path / "projection.duckdb"
    _build_spatial_fixture(db_path)
    monkeypatch.setenv("ESDC_CONFIG_DIR", str(tmp_path / "config"))
    reset_sql_cache()

    result = _execute_sql_sync(
        "SELECT ST_AsText(geom) AS geom FROM synthetic_spatial_layer",
        str(db_path),
    )

    assert "geom" in result
    assert "POINT (101.123 -0.456)" in result


def test_sql_guidance_uses_resolver_and_catalog_for_geometry(
    tmp_path: Path, monkeypatch
) -> None:
    """Spatial SQL relies on resolved/catalogued tables, not resource columns."""
    assert "Spatial Resolver" in execute_sql.description
    assert "spatial_layer_catalog" in execute_sql.description
    assert "FROM project_resources" not in execute_sql.description

    db_path = tmp_path / "projection.duckdb"
    _build_spatial_fixture(db_path)
    monkeypatch.setenv("ESDC_CONFIG_DIR", str(tmp_path / "config"))
    reset_sql_cache()
    result = _execute_sql_sync(
        "SELECT project_name, ST_AsText(geom) AS field_wkt "
        "FROM synthetic_spatial_layer "
        "WHERE ST_AsText(geom) IS NOT NULL",
        str(db_path),
    )

    assert "Duri Project" in result
    assert "Missing Geometry" not in result
    assert "(1 rows returned)" in result


def test_execute_sql_rejects_non_select_before_opening_database(tmp_path: Path) -> None:
    """The SQL tool rejects writes before attempting to open a database."""
    result = _execute_sql_sync(
        "UPDATE project_resources SET field_geom = 'POINT (0 0)'",
        str(tmp_path / "missing.duckdb"),
    )

    assert result.startswith("Error: Only SELECT queries are allowed.")


def test_execute_sql_reports_missing_spatial_extension(
    tmp_path: Path, monkeypatch
) -> None:
    """An unavailable Spatial extension is returned as an observable SQL error."""
    db_path = tmp_path / "projection.duckdb"
    _build_spatial_fixture(db_path)
    monkeypatch.setenv("ESDC_CONFIG_DIR", str(tmp_path / "config"))
    reset_sql_cache()

    def unavailable_connection(_: str | None = None) -> duckdb.DuckDBPyConnection:
        raise duckdb.Error("Spatial extension unavailable")

    monkeypatch.setattr("esdc.chat.tools.get_db_connection", unavailable_connection)
    result = _execute_sql_sync(
        "SELECT ST_AsText(geom) AS geom FROM synthetic_spatial_layer",
        str(db_path),
    )

    assert result == "SQL Error: Spatial extension unavailable"


def test_independent_sql_cache_handle_observes_invalidation(
    tmp_path: Path, monkeypatch
) -> None:
    from esdc.configs import Config

    monkeypatch.setenv("ESDC_CONFIG_DIR", str(tmp_path / "config"))
    cache_dir = Config.get_cache_dir() / "sql_results"
    first = Cache(str(cache_dir))
    second = Cache(str(cache_dir))
    try:
        first["query"] = "Before"
        assert second["query"] == "Before"

        invalidate_sql_cache()

        assert "query" not in second
    finally:
        first.close()
        second.close()


def test_independent_tool_cache_handle_observes_invalidation(
    tmp_path: Path, monkeypatch
) -> None:
    from esdc.configs import Config

    monkeypatch.setenv("ESDC_CONFIG_DIR", str(tmp_path / "config"))
    cache_dir = Config.get_cache_dir() / "tool_results"
    first = Cache(str(cache_dir))
    second = Cache(str(cache_dir))
    try:
        first["tool"] = "Before"
        assert second["tool"] == "Before"

        invalidate_tool_cache()

        assert "tool" not in second
    finally:
        first.close()
        second.close()
