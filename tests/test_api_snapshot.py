"""Focused tests for the SQLite API snapshot repository."""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

import pytest

from esdc.api_snapshot import (
    ApiProjectResourceRow,
    ApiSnapshot,
    commit_project_resources_update,
    get_project_resources_snapshot,
    iter_project_resources_rows,
    mark_project_resources_published,
    replace_project_resources_snapshot,
    replace_project_resources_years,
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
    return [
        row[0] for row in conn.execute("SELECT project_id FROM api_project_resources")
    ]


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
    assert len(stored.source_sha256) == 64
    assert stored.publication_state == "committed"
    assert stored.revision != "incoming"


def test_replace_selected_years_retains_null_and_other_years(tmp_path: Path) -> None:
    conn = get_sqlite_connection(tmp_path / "esdc.sqlite")
    initial = [_row("P-null"), _row("P-2024"), _row("P-2025")]
    initial[0] = ApiProjectResourceRow(project_id="P-null")
    initial[1] = ApiProjectResourceRow(project_id="P-2024", report_year=2024)
    initial[2] = ApiProjectResourceRow(project_id="P-2025", report_year=2025)
    replace_project_resources_snapshot(
        conn, initial, ApiSnapshot("project_resources", "x", 3, "x", "now", "committed")
    )
    replace_project_resources_years(
        conn,
        [ApiProjectResourceRow(project_id="P-new", report_year=2025)],
        ApiSnapshot("project_resources", "x", 0, "x", "now", "committed"),
        [2025],
    )
    assert sorted(_project_ids(conn)) == ["P-2024", "P-new", "P-null"]
    stored = get_project_resources_snapshot(conn)
    assert stored is not None and stored.row_count == 3


def test_persisted_hash_ignores_order_but_preserves_duplicates(tmp_path: Path) -> None:
    def write(path: Path, rows: list[ApiProjectResourceRow]) -> str:
        conn = get_sqlite_connection(path)
        replace_project_resources_snapshot(
            conn,
            rows,
            ApiSnapshot("project_resources", "x", len(rows), "x", "now", "committed"),
        )
        snapshot = get_project_resources_snapshot(conn)
        assert snapshot is not None
        return snapshot.source_sha256

    one = _row("P-1")
    two = _row("P-2")
    assert write(tmp_path / "a.sqlite", [one, two]) == write(
        tmp_path / "b.sqlite", [two, one]
    )
    duplicate = ApiProjectResourceRow(project_id="DUP")
    duplicate_hash = write(tmp_path / "c.sqlite", [duplicate, duplicate])
    single_hash = write(tmp_path / "d.sqlite", [duplicate])
    assert duplicate_hash != single_hash


def test_partial_and_full_replacement_have_same_final_hash(tmp_path: Path) -> None:
    conn = get_sqlite_connection(tmp_path / "esdc.sqlite")
    rows = [
        ApiProjectResourceRow(project_id="P-24", report_year=2024),
        ApiProjectResourceRow(project_id="P-25", report_year=2025),
    ]
    replace_project_resources_snapshot(
        conn, rows, ApiSnapshot("project_resources", "x", 2, "x", "now", "committed")
    )
    full_snapshot = get_project_resources_snapshot(conn)
    assert full_snapshot is not None
    full_hash = full_snapshot.source_sha256
    full_revision = full_snapshot.revision
    replace_project_resources_years(
        conn,
        [rows[1]],
        ApiSnapshot("project_resources", "x", 0, "x", "now", "committed"),
        [2025],
    )
    partial_snapshot = get_project_resources_snapshot(conn)
    assert partial_snapshot is not None
    assert partial_snapshot.source_sha256 == full_hash
    assert partial_snapshot.revision != full_revision


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


def test_mark_published_rolls_back_when_commit_fails(tmp_path: Path) -> None:
    path = tmp_path / "esdc.sqlite"
    conn = get_sqlite_connection(path)
    replace_project_resources_snapshot(conn, [_row()], _snapshot())
    stored = get_project_resources_snapshot(conn)
    assert stored is not None
    conn.close()

    class CommitFailingConnection(sqlite3.Connection):
        def commit(self) -> None:
            raise sqlite3.OperationalError("injected commit failure")

    failing = sqlite3.connect(path, factory=CommitFailingConnection)
    with pytest.raises(sqlite3.OperationalError, match="injected commit failure"):
        mark_project_resources_published(failing, stored.revision)
    assert not failing.in_transaction
    assert get_project_resources_snapshot(failing) == stored


def test_guarded_update_checks_snapshot_and_returns_complete_hash(
    tmp_path: Path,
) -> None:
    conn = get_sqlite_connection(tmp_path / "esdc.sqlite")
    rows = [
        ApiProjectResourceRow(project_id="P-null"),
        ApiProjectResourceRow(project_id="P-24", report_year=2024),
        ApiProjectResourceRow(project_id="P-25", report_year=2025),
    ]
    replace_project_resources_snapshot(
        conn, rows, ApiSnapshot("project_resources", "x", 3, "x", "now", "committed")
    )
    before = get_project_resources_snapshot(conn)
    assert before is not None
    updated = commit_project_resources_update(
        conn,
        [ApiProjectResourceRow(project_id="P-new", report_year=2025)],
        years=[2025],
        expected_snapshot=before,
    )
    assert updated.row_count == 3
    assert [row.project_id for row in iter_project_resources_rows(conn)] == [
        "P-null",
        "P-24",
        "P-new",
    ]
    assert len(updated.source_sha256) == 64
    assert updated.source_sha256 != before.source_sha256
    assert datetime.fromisoformat(updated.fetched_at)


def test_guarded_update_rejects_changed_expected_snapshot_before_delete(
    tmp_path: Path,
) -> None:
    conn = get_sqlite_connection(tmp_path / "esdc.sqlite")
    replace_project_resources_snapshot(conn, [_row()], _snapshot())
    expected = get_project_resources_snapshot(conn)
    assert expected is not None
    replace_project_resources_snapshot(
        conn,
        [_row("P-new")],
        ApiSnapshot("project_resources", "x", 1, "x", "now", "committed"),
    )
    current = get_project_resources_snapshot(conn)
    with pytest.raises(ValueError, match="snapshot changed"):
        commit_project_resources_update(
            conn, [_row("P-delete")], years=None, expected_snapshot=expected
        )
    assert _project_ids(conn) == ["P-new"]
    assert get_project_resources_snapshot(conn) == current


def test_guarded_update_rolls_back_rows_and_snapshot_on_insert_failure(
    tmp_path: Path,
) -> None:
    conn = get_sqlite_connection(tmp_path / "esdc.sqlite")
    replace_project_resources_snapshot(conn, [_row()], _snapshot())
    before = get_project_resources_snapshot(conn)
    with pytest.raises(sqlite3.IntegrityError):
        commit_project_resources_update(
            conn,
            [_row("P-new"), _row("")],
            years=None,
            expected_snapshot=before,
        )
    assert _project_ids(conn) == ["P-1"]
    assert get_project_resources_snapshot(conn) == before


def test_full_guarded_refresh_can_recover_orphaned_canonical_rows(
    tmp_path: Path,
) -> None:
    conn = get_sqlite_connection(tmp_path / "esdc.sqlite")
    conn.execute("INSERT INTO api_project_resources (project_id) VALUES ('orphaned')")
    conn.commit()

    snapshot = commit_project_resources_update(
        conn,
        [_row("P-refresh")],
        years=None,
        expected_snapshot=None,
    )

    assert snapshot.row_count == 1
    assert _project_ids(conn) == ["P-refresh"]


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
            ApiSnapshot(
                "project_resources", "incoming-2", 2, "hash2", "now", "committed"
            ),
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
            ApiSnapshot(
                "project_resources", "new", 99, "new-hash", "new-time", "committed"
            ),
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
