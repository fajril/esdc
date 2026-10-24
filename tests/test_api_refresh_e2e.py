"""End-to-end tests for project-resource API refresh and recovery."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import duckdb
import pytest

import esdc.api_projection
import esdc.fetch_publication
from esdc.esdc import (
    rebuild_project_resources_projection,
    refresh_project_resources_from_api,
)


def _payload() -> list[dict[str, object]]:
    first = {
        "project_id": "P-1",
        "report_year": 2025,
        "project_name": "One",
        "field_geom": "POINT (1 2)",
    }
    second = {
        "field_geom": "NOT WKT",
        "project_name": "Two",
        "report_year": 2025,
        "project_id": "P-2",
    }
    return [first, second]


def _snapshot(path: Path) -> tuple[str, int, str, str]:
    conn = sqlite3.connect(path)
    try:
        row = conn.execute(
            "SELECT revision, row_count, publication_state, source_sha256 "
            "FROM api_snapshot "
            "WHERE source_table = 'project_resources'"
        ).fetchone()
        assert row is not None
        return row
    finally:
        conn.close()


def test_refresh_and_rebuild_recover_without_refetch(
    isolated_config, monkeypatch: pytest.MonkeyPatch
) -> None:
    from esdc.configs import Config

    payload = _payload()
    calls: list[str] = []

    def download(*_: object) -> bytes:
        calls.append("download")
        return json.dumps(payload).encode()

    monkeypatch.setattr("esdc.esdc.esdc_downloader", download)
    monkeypatch.setattr("esdc.esdc.load_data_to_db", lambda *_: calls.append("duck-loader"))

    def fail_checkpoint(_: object) -> None:
        raise RuntimeError("post-commit failure")

    original_checkpoint = esdc.fetch_publication._checkpoint
    monkeypatch.setattr("esdc.fetch_publication._checkpoint", fail_checkpoint)
    with pytest.raises(RuntimeError, match="rebuild-project-resources"):
        refresh_project_resources_from_api(
            username="user", password="password", report_year=None
        )

    sqlite_path = Config.get_db_dir() / "esdc.sqlite"
    revision, row_count, state, _ = _snapshot(sqlite_path)
    assert revision
    assert row_count == 2
    assert state == "analytics_stale"
    assert calls == ["download"]

    monkeypatch.setattr("esdc.fetch_publication._checkpoint", original_checkpoint)

    def fail_download(*_: object) -> bytes:
        raise AssertionError("rebuild must not fetch")

    monkeypatch.setattr("esdc.esdc.esdc_downloader", fail_download)
    publication = rebuild_project_resources_projection()

    assert publication.revision == revision
    assert publication.row_count == 2
    assert publication.invalid_geometry_count == 1
    assert _snapshot(sqlite_path)[2] == "published"
    assert calls == ["download"]


@pytest.mark.parametrize("starting_state", ["committed", "analytics_stale", "published"])
def test_rebuild_published_or_stale_snapshot_recreates_missing_mirror_without_fetch(
    isolated_config, monkeypatch: pytest.MonkeyPatch, starting_state: str
) -> None:
    from esdc.configs import Config

    calls: list[str] = []

    def download(*_: object) -> bytes:
        calls.append("download")
        return json.dumps(_payload()).encode()

    monkeypatch.setattr("esdc.esdc.esdc_downloader", download)
    monkeypatch.setattr("esdc.esdc.load_data_to_db", lambda *_: calls.append("duck-loader"))
    refresh_project_resources_from_api(
        username="user", password="password", report_year=None
    )

    sqlite_path = Config.get_db_dir() / "esdc.sqlite"
    duckdb_path = Config.get_db_file()
    revision, row_count, _, source_sha256 = _snapshot(sqlite_path)
    conn = sqlite3.connect(sqlite_path)
    try:
        conn.execute(
            "UPDATE api_snapshot SET publication_state = ? "
            "WHERE source_table = 'project_resources'",
            (starting_state,),
        )
        conn.commit()
    finally:
        conn.close()
    duckdb_path.unlink()

    def fail_download(*_: object) -> bytes:
        raise AssertionError("rebuild must not fetch")

    monkeypatch.setattr("esdc.esdc.esdc_downloader", fail_download)
    publication = rebuild_project_resources_projection()

    assert publication.revision == revision
    assert publication.row_count == row_count
    assert publication.source_sha256 == source_sha256
    assert calls == ["download"]
    assert _snapshot(sqlite_path) == (revision, row_count, "published", source_sha256)
    with duckdb.connect(str(duckdb_path), read_only=True) as conn:
        assert conn.execute("SELECT COUNT(*) FROM project_resources").fetchone() == (
            row_count,
        )

    repeat = rebuild_project_resources_projection()
    assert repeat.revision == revision
    assert repeat.source_sha256 == source_sha256
    assert _snapshot(sqlite_path) == (revision, row_count, "published", source_sha256)


def test_rebuild_refreshes_cached_sql_and_derived_view_values(
    isolated_config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Recovery rebuild must serve the new SQLite truth and derived views."""
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
    from esdc.configs import Config
    from esdc.pod_registry.store import get_sqlite_connection

    reset_sql_cache()
    invalidate_tool_cache()

    before_payload = [
        {
            "project_id": "P-1",
            "report_year": 2025,
            "project_name": "Before",
            "field_name": "F1",
            "wk_name": "W1",
            "rec_oil": 10.0,
            "field_geom": "POINT (1 2)",
        }
    ]
    calls: list[str] = []

    def download_before(*_: object) -> bytes:
        calls.append("download")
        return json.dumps(before_payload).encode()

    monkeypatch.setattr("esdc.esdc.esdc_downloader", download_before)
    monkeypatch.setattr(
        "esdc.esdc.load_data_to_db", lambda *_: calls.append("duck-loader")
    )
    refresh_project_resources_from_api(
        username="user", password="password", report_year=None
    )

    sqlite_path = Config.get_db_dir() / "esdc.sqlite"
    duckdb_path = Config.get_db_file()
    before_revision, _, before_state, _ = _snapshot(sqlite_path)
    assert before_state == "published"

    query = "SELECT project_name FROM project_resources"
    assert "Before" in _execute_sql_sync(query, str(duckdb_path))

    after_rows = [
        ApiProjectResourceRow(
            project_id="P-1",
            report_year=2025,
            project_name="After-1",
            field_name="F1",
            wk_name="W1",
            rec_oil=20.0,
            field_geom="POINT (1 2)",
        ),
        ApiProjectResourceRow(
            project_id="P-2",
            report_year=2025,
            project_name="After-2",
            field_name="F1",
            wk_name="W1",
            rec_oil=30.0,
            field_geom=None,
        ),
    ]
    conn = get_sqlite_connection(sqlite_path)
    try:
        replace_project_resources_snapshot(
            conn,
            after_rows,
            ApiSnapshot(
                "project_resources", "incoming", 2, "after-hash", "now", "committed"
            ),
        )
        snapshot = get_project_resources_snapshot(conn)
        assert snapshot is not None
        after_revision = snapshot.revision
    finally:
        conn.close()
    assert after_revision != before_revision
    assert _snapshot(sqlite_path)[2] == "committed"

    def fail_download(*_: object) -> bytes:
        raise AssertionError("rebuild must not fetch")

    monkeypatch.setattr("esdc.esdc.esdc_downloader", fail_download)

    publication = rebuild_project_resources_projection()
    assert publication.revision == after_revision
    assert publication.row_count == 2
    assert _snapshot(sqlite_path)[2] == "published"
    assert calls == ["download"]

    assert "After" in _execute_sql_sync(query, str(duckdb_path))
    assert "Before" not in _execute_sql_sync(query, str(duckdb_path))

    def _view_values() -> tuple[list[tuple], list[tuple]]:
        with duckdb.connect(str(duckdb_path), read_only=True) as view_conn:
            resources = view_conn.execute(
                "SELECT project_id, project_name, rec_oil FROM project_resources "
                "ORDER BY project_id"
            ).fetchall()
            fields = view_conn.execute(
                "SELECT field_name, project_count, rec_oil FROM field_resources "
                "ORDER BY field_name"
            ).fetchall()
        return resources, fields

    first = _view_values()
    assert first == (
        [("P-1", "After-1", 20.0), ("P-2", "After-2", 30.0)],
        [("F1", 2, 50.0)],
    )
    repeat = rebuild_project_resources_projection()
    assert repeat.revision == after_revision
    assert repeat.source_sha256 == publication.source_sha256
    assert _snapshot(sqlite_path)[:3] == (after_revision, 2, "published")
    assert _view_values() == first
