"""Focused tests for the SQLite API snapshot repository."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from esdc.api_snapshot import (
    ApiProjectResourceRow,
    ApiSnapshot,
    get_project_resources_snapshot,
    mark_project_resources_published,
    replace_project_resources_snapshot,
)
from esdc.pod_registry.store import get_sqlite_connection


def _row(project_id: str = "P-1") -> ApiProjectResourceRow:
    return ApiProjectResourceRow(
        project_id=project_id,
        project_name="Project",
        report_year=2025,
        uncert_level="1P",
        field_geom="POINT (1 2)",
        groovy_report="Current situation:\nStable",
    )


def _snapshot(revision: str = "old") -> ApiSnapshot:
    return ApiSnapshot(
        source_table="project_resources",
        revision=revision,
        row_count=1,
        source_sha256="abc",
        fetched_at="2026-09-18T00:00:00Z",
        publication_state="committed",
    )


def _project_ids(conn: sqlite3.Connection) -> list[str]:
    return [row[0] for row in conn.execute("SELECT project_id FROM api_project_resources")]


def test_replace_writes_rows_and_committed_snapshot(tmp_path: Path) -> None:
    conn = get_sqlite_connection(tmp_path / "esdc.sqlite")
    rows = [_row("P-1"), _row("P-2")]
    snapshot = ApiSnapshot(
        source_table="project_resources",
        revision="incoming",
        row_count=2,
        source_sha256="hash",
        fetched_at="2026-09-18T00:00:00Z",
        publication_state="committed",
    )

    replace_project_resources_snapshot(conn, rows, snapshot)

    assert _project_ids(conn) == ["P-1", "P-2"]
    stored = get_project_resources_snapshot(conn)
    assert stored is not None
    assert stored.source_table == "project_resources"
    assert stored.row_count == 2
    assert stored.source_sha256 == "hash"
    assert stored.publication_state == "committed"
    assert stored.revision != "incoming"


def test_replace_rolls_back_when_any_row_is_invalid(tmp_path: Path) -> None:
    conn = get_sqlite_connection(tmp_path / "esdc.sqlite")
    old_snapshot = _snapshot()
    replace_project_resources_snapshot(conn, [_row()], old_snapshot)
    prior = get_project_resources_snapshot(conn)
    assert prior is not None

    invalid = _row("")
    new_snapshot = ApiSnapshot("project_resources", "new", 2, "abc", "now", "committed")
    with pytest.raises(sqlite3.IntegrityError):
        replace_project_resources_snapshot(conn, [_row(), invalid], new_snapshot)

    assert _project_ids(conn) == ["P-1"]
    assert get_project_resources_snapshot(conn) == prior


def test_mark_published_requires_matching_revision(tmp_path: Path) -> None:
    conn = get_sqlite_connection(tmp_path / "esdc.sqlite")
    replace_project_resources_snapshot(conn, [_row()], _snapshot())
    stored = get_project_resources_snapshot(conn)
    assert stored is not None

    mark_project_resources_published(conn, stored.revision)
    published = get_project_resources_snapshot(conn)
    assert published is not None
    assert published.publication_state == "published"

    with pytest.raises(ValueError):
        mark_project_resources_published(conn, "missing")


def test_rows_are_unique_by_project_year_and_uncertainty(tmp_path: Path) -> None:
    conn = get_sqlite_connection(tmp_path / "esdc.sqlite")
    first = _row("P-1")
    second = ApiProjectResourceRow(
        project_id="P-1",
        project_name="Project",
        report_year=2025,
        uncert_level="2P",
    )
    replace_project_resources_snapshot(
        conn,
        [first, second],
        ApiSnapshot("project_resources", "incoming", 2, "hash", "now", "committed"),
    )
    assert conn.execute("SELECT COUNT(*) FROM api_project_resources").fetchone()[0] == 2
    duplicate = ApiProjectResourceRow(
        project_id="P-1", project_name="Other", report_year=2025, uncert_level="1P"
    )
    with pytest.raises(sqlite3.IntegrityError):
        replace_project_resources_snapshot(
            conn,
            [first, duplicate],
            ApiSnapshot("project_resources", "incoming-2", 2, "hash2", "now", "committed"),
        )


def test_row_count_mismatch_rejected_before_transaction(tmp_path: Path) -> None:
    conn = get_sqlite_connection(tmp_path / "esdc.sqlite")
    replace_project_resources_snapshot(conn, [_row()], _snapshot())
    prior_rows = _project_ids(conn)
    prior_snapshot = get_project_resources_snapshot(conn)

    with pytest.raises(ValueError, match="row_count"):
        replace_project_resources_snapshot(
            conn,
            [_row("P-2")],
            ApiSnapshot("project_resources", "new", 99, "new-hash", "new-time", "committed"),
        )

    assert _project_ids(conn) == prior_rows
    assert get_project_resources_snapshot(conn) == prior_snapshot


def test_json_scalars_round_trip(tmp_path: Path) -> None:
    conn = get_sqlite_connection(tmp_path / "esdc.sqlite")
    row = ApiProjectResourceRow(
        project_id="P-scalar",
        report_year=2025,
        project_name="Text",
        project_isactive=True,
        onstream_year=2030.5,
    )
    replace_project_resources_snapshot(
        conn,
        [row],
        ApiSnapshot("project_resources", "incoming", 1, "hash", "now", "committed"),
    )
    stored = conn.execute(
        "SELECT project_name, project_isactive, onstream_year "
        "FROM api_project_resources WHERE project_id = ?",
        ("P-scalar",),
    ).fetchone()
    assert tuple(stored) == ("Text", 1, 2030.5)


def test_invalid_geometry_count_round_trips_with_default(tmp_path: Path) -> None:
    conn = get_sqlite_connection(tmp_path / "esdc.sqlite")
    replace_project_resources_snapshot(
        conn,
        [_row()],
        ApiSnapshot("project_resources", "incoming", 1, "hash", "now", "committed", 18),
    )
    stored = get_project_resources_snapshot(conn)
    assert stored is not None
    assert stored.invalid_geometry_count == 18
