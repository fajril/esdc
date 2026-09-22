"""Tests for transactional DuckDB project-resource publication."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import duckdb
import pytest

import esdc.api_projection
from esdc.api_projection import publish_project_resources
from esdc.api_snapshot import (
    ApiProjectResourceRow,
    ApiSnapshot,
    get_project_resources_snapshot,
    replace_project_resources_snapshot,
)
from esdc.chat.tools import (
    _execute_sql_sync,
    invalidate_tool_cache,
    reset_sql_cache,
)
from esdc.pod_registry.store import get_sqlite_connection


@pytest.fixture(autouse=True)
def isolated_cache_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ESDC_CONFIG_DIR", str(tmp_path / "config"))


def _seed(
    path: Path,
    *,
    project_id: str = "P-1",
    project_name: str = "Project",
    field_geom: str | None = "POINT (1 2)",
    revision: str = "incoming",
) -> str:
    conn = get_sqlite_connection(path)
    replace_project_resources_snapshot(
        conn,
        [
            ApiProjectResourceRow(
                project_id=project_id,
                report_year=2025,
                project_name=project_name,
                field_name="Field",
                wk_name="Working Area",
                field_geom=field_geom,
                groovy_report="Current situation:\nStable",
            )
        ],
        ApiSnapshot("project_resources", revision, 1, "source-hash", "now", "committed"),
    )
    snapshot = get_project_resources_snapshot(conn)
    assert snapshot is not None
    return snapshot.revision


def _seed_rows(path: Path, rows: list[ApiProjectResourceRow]) -> str:
    conn = get_sqlite_connection(path)
    replace_project_resources_snapshot(
        conn,
        rows,
        ApiSnapshot("project_resources", "incoming", len(rows), "source-hash", "now", "committed"),
    )
    snapshot = get_project_resources_snapshot(conn)
    assert snapshot is not None
    return snapshot.revision


def _project_count(path: Path) -> int:
    conn = duckdb.connect(str(path), read_only=True)
    try:
        row = conn.execute("SELECT COUNT(*) FROM project_resources").fetchone()
        assert row is not None
        return row[0]
    finally:
        conn.close()


def _project_names(path: Path) -> list[str]:
    conn = duckdb.connect(str(path), read_only=True)
    try:
        return [
            row[0]
            for row in conn.execute(
                "SELECT project_name FROM project_resources ORDER BY project_name"
            ).fetchall()
        ]
    finally:
        conn.close()


class _FailingSqlCache:
    """Minimal stand-in for a diskcache handle whose clear fails."""

    def __init__(self) -> None:
        self.clear_calls = 0

    def clear(self) -> None:
        self.clear_calls += 1
        raise OSError("cache clear failed")

    def close(self) -> None:  # pragma: no cover - clear raises first
        pass


def test_publish_preserves_wkt_and_builds_analytics(tmp_path: Path) -> None:
    sqlite_path = tmp_path / "esdc.sqlite"
    duckdb_path = tmp_path / "esdc.duckdb"
    revision = _seed(sqlite_path)

    result = publish_project_resources(duckdb_path, sqlite_path, revision)

    assert result.revision == revision
    assert result.row_count == 1
    conn = duckdb.connect(str(duckdb_path), read_only=True)
    try:
        row = conn.execute(
            "SELECT field_geom, ST_AsText(field_geometry) FROM project_resources"
        ).fetchone()
        assert row == ("POINT (1 2)", "POINT (1 2)")
        row = conn.execute(
            "SELECT COUNT(*) FROM project_resources "
            "WHERE fts_main_project_resources.match_bm25(uuid, ?) IS NOT NULL",
            ["stable"],
        ).fetchone()
        assert row is not None
        assert row[0] == 1
        for view in ("field_resources", "wa_resources", "nkri_resources"):
            conn.execute(f"SELECT * FROM {view} LIMIT 0")
    finally:
        conn.close()


def test_mixed_wkt_publishes_raw_invalid_geometry_and_records_count(tmp_path: Path) -> None:
    sqlite_path = tmp_path / "esdc.sqlite"
    duckdb_path = tmp_path / "esdc.duckdb"
    revision = _seed_rows(
        sqlite_path,
        [
            ApiProjectResourceRow(project_id="P-1", field_geom="POINT (1 2)"),
            ApiProjectResourceRow(project_id="P-2", field_geom="NOT WKT"),
        ],
    )

    result = publish_project_resources(duckdb_path, sqlite_path, revision)

    assert result.invalid_geometry_count == 1
    conn = duckdb.connect(str(duckdb_path), read_only=True)
    try:
        rows = conn.execute(
            "SELECT project_id, field_geom, ST_AsText(field_geometry) "
            "FROM project_resources ORDER BY project_id"
        ).fetchall()
        assert rows == [("P-1", "POINT (1 2)", "POINT (1 2)"), ("P-2", "NOT WKT", None)]
    finally:
        conn.close()
    conn = sqlite3.connect(sqlite_path)
    try:
        row = conn.execute(
            "SELECT publication_state, invalid_geometry_count FROM api_snapshot "
            "WHERE source_table = ?",
            ("project_resources",),
        ).fetchone()
        assert row == ("published", 1)
    finally:
        conn.close()


def test_checkpoint_failure_leaves_new_projection_readable_and_stale(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sqlite_path = tmp_path / "esdc.sqlite"
    duckdb_path = tmp_path / "esdc.duckdb"
    revision = _seed(sqlite_path)

    original_checkpoint = esdc.api_projection._checkpoint

    def fail_checkpoint(_: object) -> None:
        raise RuntimeError("checkpoint failed")

    monkeypatch.setattr("esdc.api_projection._checkpoint", fail_checkpoint)
    with pytest.raises(RuntimeError, match="analytical state requires rebuild"):
        publish_project_resources(duckdb_path, sqlite_path, revision)

    assert _project_count(duckdb_path) == 1
    conn = sqlite3.connect(sqlite_path)
    try:
        assert conn.execute(
            "SELECT publication_state FROM api_snapshot WHERE source_table = ?",
            ("project_resources",),
        ).fetchone()[0] == "analytics_stale"
    finally:
        conn.close()

    monkeypatch.setattr("esdc.api_projection._checkpoint", original_checkpoint)
    result = publish_project_resources(duckdb_path, sqlite_path, revision)
    assert result.revision == revision
    conn = sqlite3.connect(sqlite_path)
    try:
        assert conn.execute(
            "SELECT publication_state FROM api_snapshot WHERE source_table = ?",
            ("project_resources",),
        ).fetchone()[0] == "published"
    finally:
        conn.close()


def test_cache_clear_failure_propagates_and_marks_snapshot_stale(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ESDC_CONFIG_DIR", str(tmp_path / "config"))
    reset_sql_cache()
    invalidate_tool_cache()
    sqlite_path = tmp_path / "esdc.sqlite"
    duckdb_path = tmp_path / "esdc.duckdb"

    before_revision = _seed(sqlite_path, project_name="Before")
    publish_project_resources(duckdb_path, sqlite_path, before_revision)

    failing = _FailingSqlCache()
    monkeypatch.setattr("esdc.chat.tools._sql_cache", failing)

    with pytest.raises(OSError, match="cache clear failed"):
        reset_sql_cache()
    assert failing.clear_calls == 1

    after_revision = _seed(sqlite_path, project_name="After")
    with pytest.raises(RuntimeError, match="cache invalidation failure"):
        publish_project_resources(duckdb_path, sqlite_path, after_revision)

    conn = sqlite3.connect(sqlite_path)
    try:
        assert conn.execute(
            "SELECT publication_state FROM api_snapshot WHERE source_table = ?",
            ("project_resources",),
        ).fetchone()[0] == "analytics_stale"
    finally:
        conn.close()


def test_post_replacement_precommit_failure_rolls_back_mirror_and_keeps_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ESDC_CONFIG_DIR", str(tmp_path / "config"))
    reset_sql_cache()
    invalidate_tool_cache()
    sqlite_path = tmp_path / "esdc.sqlite"
    duckdb_path = tmp_path / "esdc.duckdb"

    before_revision = _seed(sqlite_path, project_name="Before")
    publish_project_resources(duckdb_path, sqlite_path, before_revision)
    query = "SELECT project_name FROM project_resources"
    assert "Before" in _execute_sql_sync(query, str(duckdb_path))

    after_revision = _seed(sqlite_path, project_name="After")

    invalidations: list[str] = []
    monkeypatch.setattr(
        "esdc.api_projection._invalidate_analytical_caches",
        lambda: invalidations.append("invalidate"),
    )

    def fail_rebuild(_: object) -> None:
        raise RuntimeError("rebuild failed after replacement")

    monkeypatch.setattr(
        "esdc.api_projection._rebuild_project_resources", fail_rebuild
    )
    with pytest.raises(RuntimeError, match="rebuild failed after replacement"):
        publish_project_resources(duckdb_path, sqlite_path, after_revision)

    assert invalidations == []
    assert _project_names(duckdb_path) == ["Before"]

    conn = sqlite3.connect(sqlite_path)
    try:
        row = conn.execute(
            "SELECT revision, publication_state FROM api_snapshot "
            "WHERE source_table = ?",
            ("project_resources",),
        ).fetchone()
    finally:
        conn.close()
    assert row == (after_revision, "analytics_stale")


def test_publish_invalidates_cached_sql_after_successful_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ESDC_CONFIG_DIR", str(tmp_path / "config"))
    reset_sql_cache()
    invalidate_tool_cache()
    sqlite_path = tmp_path / "esdc.sqlite"
    duckdb_path = tmp_path / "esdc.duckdb"

    before_revision = _seed(sqlite_path, project_name="Before")
    publish_project_resources(duckdb_path, sqlite_path, before_revision)
    query = "SELECT project_name FROM project_resources"
    assert "Before" in _execute_sql_sync(query, str(duckdb_path))

    after_revision = _seed(sqlite_path, project_name="After")
    publish_project_resources(duckdb_path, sqlite_path, after_revision)

    assert "After" in _execute_sql_sync(query, str(duckdb_path))
    assert "Before" not in _execute_sql_sync(query, str(duckdb_path))


def test_checkpoint_failure_does_not_leave_precommit_sql_cache_value(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ESDC_CONFIG_DIR", str(tmp_path / "config"))
    reset_sql_cache()
    invalidate_tool_cache()
    sqlite_path = tmp_path / "esdc.sqlite"
    duckdb_path = tmp_path / "esdc.duckdb"

    before_revision = _seed(sqlite_path, project_name="Before")
    publish_project_resources(duckdb_path, sqlite_path, before_revision)
    query = "SELECT project_name FROM project_resources"
    assert "Before" in _execute_sql_sync(query, str(duckdb_path))

    after_revision = _seed(sqlite_path, project_name="After")

    def fail_checkpoint(_: object) -> None:
        raise RuntimeError("checkpoint failed")

    monkeypatch.setattr("esdc.api_projection._checkpoint", fail_checkpoint)
    with pytest.raises(RuntimeError, match="analytical state requires rebuild"):
        publish_project_resources(duckdb_path, sqlite_path, after_revision)

    assert "After" in _execute_sql_sync(query, str(duckdb_path))


def test_precommit_failure_leaves_cached_sql_value_valid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ESDC_CONFIG_DIR", str(tmp_path / "config"))
    reset_sql_cache()
    invalidate_tool_cache()
    sqlite_path = tmp_path / "esdc.sqlite"
    duckdb_path = tmp_path / "esdc.duckdb"

    before_revision = _seed(sqlite_path, project_name="Before")
    publish_project_resources(duckdb_path, sqlite_path, before_revision)
    query = "SELECT project_name FROM project_resources"
    assert "Before" in _execute_sql_sync(query, str(duckdb_path))

    after_revision = _seed(sqlite_path, project_name="After")

    def fail_staging(_: object, __: str) -> None:
        raise RuntimeError("staging failed")

    monkeypatch.setattr("esdc.api_projection._staging_schema", fail_staging)
    with pytest.raises(RuntimeError, match="staging failed"):
        publish_project_resources(duckdb_path, sqlite_path, after_revision)

    assert "Before" in _execute_sql_sync(query, str(duckdb_path))


def test_duckdb_connection_failure_marks_snapshot_stale(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sqlite_path = tmp_path / "esdc.sqlite"
    duckdb_path = tmp_path / "esdc.duckdb"
    revision = _seed(sqlite_path)

    def fail_connection(*_: object, **__: object) -> object:
        raise RuntimeError("connection failed")

    monkeypatch.setattr("esdc.api_projection.get_duckdb_connection", fail_connection)
    with pytest.raises(RuntimeError, match="connection failed"):
        publish_project_resources(duckdb_path, sqlite_path, revision)

    conn = sqlite3.connect(sqlite_path)
    try:
        assert conn.execute(
            "SELECT publication_state FROM api_snapshot WHERE source_table = ?",
            ("project_resources",),
        ).fetchone()[0] == "analytics_stale"
    finally:
        conn.close()
