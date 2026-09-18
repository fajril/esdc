"""Tests for the read-only SQL chat tool and Spatial geometry contract."""

from pathlib import Path

import duckdb

from esdc.chat.tools import _execute_sql_sync, execute_sql, reset_sql_cache


def _build_projection_fixture(path: Path) -> None:
    """Build a minimal project-resources projection with derived geometry."""
    conn = duckdb.connect(str(path))
    try:
        conn.execute("INSTALL spatial")
        conn.execute("LOAD spatial")
        conn.execute(
            "CREATE TABLE project_resources ("
            "project_name VARCHAR, field_geom TEXT, field_geometry GEOMETRY)"
        )
        conn.execute(
            "INSERT INTO project_resources VALUES (?, ?, ST_GeomFromText(?))",
            ["Duri Project", "POINT (101.123 -0.456)", "POINT (101.123 -0.456)"],
        )
        conn.execute(
            "INSERT INTO project_resources VALUES (?, ?, ?)",
            ["Missing Geometry", None, None],
        )
    finally:
        conn.close()


def test_execute_sql_formats_derived_geometry_wkt(
    tmp_path: Path, monkeypatch
) -> None:
    """Spatial SELECTs expose the derived geometry as WKT to chat callers."""
    db_path = tmp_path / "projection.duckdb"
    _build_projection_fixture(db_path)
    monkeypatch.setenv("ESDC_CONFIG_DIR", str(tmp_path / "config"))
    reset_sql_cache()

    result = _execute_sql_sync(
        "SELECT ST_AsText(field_geometry) AS geom FROM project_resources",
        str(db_path),
    )

    assert "geom" in result
    assert "POINT (101.123 -0.456)" in result


def test_documented_spatial_predicate_excludes_null_derived_geometry(
    tmp_path: Path, monkeypatch
) -> None:
    """The documented WKT predicate excludes rows with null derived geometry."""
    assert "WHERE ST_AsText(field_geometry) IS NOT NULL" in execute_sql.description

    db_path = tmp_path / "projection.duckdb"
    _build_projection_fixture(db_path)
    monkeypatch.setenv("ESDC_CONFIG_DIR", str(tmp_path / "config"))
    reset_sql_cache()
    result = _execute_sql_sync(
        "SELECT project_name, ST_AsText(field_geometry) AS field_wkt "
        "FROM project_resources "
        "WHERE ST_AsText(field_geometry) IS NOT NULL",
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
    _build_projection_fixture(db_path)
    monkeypatch.setenv("ESDC_CONFIG_DIR", str(tmp_path / "config"))
    reset_sql_cache()

    def unavailable_connection(_: str | None = None) -> duckdb.DuckDBPyConnection:
        raise duckdb.Error("Spatial extension unavailable")

    monkeypatch.setattr("esdc.chat.tools.get_db_connection", unavailable_connection)
    result = _execute_sql_sync(
        "SELECT ST_AsText(field_geometry) AS geom FROM project_resources",
        str(db_path),
    )

    assert result == "SQL Error: Spatial extension unavailable"
