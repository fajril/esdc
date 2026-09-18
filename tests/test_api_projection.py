"""Tests for transactional DuckDB project-resource publication."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import duckdb
import pytest

from esdc.api_projection import publish_project_resources
from esdc.api_snapshot import (
    ApiProjectResourceRow,
    ApiSnapshot,
    get_project_resources_snapshot,
    replace_project_resources_snapshot,
)
from esdc.pod_registry.store import get_sqlite_connection


def _seed(
    path: Path,
    *,
    project_id: str = "P-1",
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
                project_name="Project",
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

    monkeypatch.undo()
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
