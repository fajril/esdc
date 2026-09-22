"""Tests for publishing managed spatial layers into DuckDB."""

from pathlib import Path

import duckdb

from esdc.spatial_layers import publisher


class _Cursor:
    def __init__(self, rows):
        self._rows = rows

    def fetchall(self):
        return self._rows


class _Reader:
    def __init__(self, layers, features):
        self.layers = layers
        self.features = features
        self.queries = []

    def execute(self, sql, parameters=()):
        self.queries.append((sql, parameters))
        if "FROM spatial_layer AS" in sql:
            return _Cursor(self.layers)
        return _Cursor(self.features.get(parameters[0], []))

    def close(self):
        pass


def _writer(path: Path, read_only: bool = False):
    del read_only
    connection = duckdb.connect(str(path))
    connection.execute("LOAD spatial")
    return connection


def test_publishes_active_layers_with_opaque_names_and_catalog(tmp_path, monkeypatch):
    layer_id = "3f0d2f73-0f72-4b2d-8c0b-2a3c4d5e6f70"
    reader = _Reader(
        [(layer_id, "Name; DROP TABLE project_resources", 2, "EPSG:4326", "v2")],
        {
            "v2": [
                (1, bytes.fromhex("0101000000000000000000f03f0000000000000040"), "{}")
            ]
        },
    )
    monkeypatch.setattr(publisher, "_open_source_reader", lambda _: reader)
    monkeypatch.setattr(publisher, "get_duckdb_connection", _writer)
    path = tmp_path / "db.duckdb"
    connection = _writer(path)
    connection.execute("CREATE TABLE project_resources (uuid VARCHAR)")
    connection.execute("CREATE TABLE user_spatial_old (feature_id BIGINT)")
    connection.close()

    result = publisher.publish_spatial_layers(path, tmp_path / "source.sqlite")
    assert result.layer_count == 1
    assert result.feature_count == 1
    assert result.table_names == (f"user_spatial_{layer_id.replace('-', '')}",)

    connection = _writer(path)
    table = result.table_names[0]
    assert connection.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone() == (1,)
    assert connection.execute(
        "SELECT name, source_crs FROM spatial_layer_catalog"
    ).fetchone() == (
        "Name; DROP TABLE project_resources",
        "EPSG:4326",
    )
    assert connection.execute("SELECT COUNT(*) FROM project_resources").fetchone() == (
        0,
    )
    assert not connection.execute(
        "SELECT 1 FROM information_schema.tables WHERE table_name='user_spatial_old'"
    ).fetchone()
    connection.close()


def test_failure_rolls_back_spatial_changes_and_preserves_api_tables(
    tmp_path, monkeypatch
):
    layer_id = "11111111-1111-4111-8111-111111111111"
    reader = _Reader(
        [(layer_id, "broken", 1, None, "v1")],
        {"v1": [(1, b"not-wkb", "{}")]},
    )
    monkeypatch.setattr(publisher, "_open_source_reader", lambda _: reader)
    monkeypatch.setattr(publisher, "get_duckdb_connection", _writer)
    path = tmp_path / "db.duckdb"
    connection = _writer(path)
    connection.execute("CREATE TABLE project_resources (uuid VARCHAR)")
    connection.execute("INSERT INTO project_resources VALUES ('keep')")
    connection.execute("CREATE TABLE user_spatial_existing (feature_id BIGINT)")
    connection.execute("INSERT INTO user_spatial_existing VALUES (9)")
    connection.close()

    try:
        publisher.publish_spatial_layers(path, tmp_path / "source.sqlite")
    except Exception:
        pass
    else:
        raise AssertionError("invalid geometry should fail publication")

    connection = _writer(path)
    assert connection.execute("SELECT * FROM project_resources").fetchall() == [
        ("keep",)
    ]
    assert connection.execute("SELECT * FROM user_spatial_existing").fetchall() == [
        (9,)
    ]
    connection.close()
