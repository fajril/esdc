# tests/pod_registry/test_publish.py
"""Tests for POD publication through the single canonical publisher."""

from __future__ import annotations

import sqlite3
from datetime import date
from pathlib import Path
from uuid import uuid4

import duckdb

import esdc.configs as configs
from esdc.loaders import _METADATA_TABLE
from esdc.pod_registry.identity import ApprovalKey, issue_identity
from esdc.pod_registry.migrations import initialize_registry_v2
from esdc.pod_registry.publication import PodPublicationReport
from esdc.pod_registry.publish import publish_pod_registry

# Minimal old-shaped fixture SQL. It is test material only: the store no longer
# carries a superseded-schema constant, so the refusal fixtures declare the shape
# they want directly.
_OLD_SHAPED_FIXTURE_SQL = """
CREATE TABLE r_institution (
    code INTEGER PRIMARY KEY,
    institution TEXT NOT NULL UNIQUE,
    description TEXT
);
CREATE TABLE r_pod_type (
    code INTEGER PRIMARY KEY,
    pod_type TEXT NOT NULL UNIQUE,
    description TEXT
);
CREATE TABLE m_pod (
    id INTEGER PRIMARY KEY,
    pod_id TEXT NOT NULL UNIQUE,
    pod_name TEXT NOT NULL,
    pod_letter_num TEXT,
    approval_date TEXT NOT NULL,
    institution_code INTEGER NOT NULL REFERENCES r_institution(code),
    pod_type_code INTEGER NOT NULL REFERENCES r_pod_type(code),
    rev_num INTEGER NOT NULL DEFAULT 0,
    approval_seq INTEGER NOT NULL UNIQUE
);
CREATE TABLE project_pod (
    pod_id INTEGER NOT NULL REFERENCES m_pod(id),
    project_id TEXT NOT NULL,
    UNIQUE (pod_id, project_id)
);
CREATE TABLE pod_document (
    pod_id INTEGER NOT NULL REFERENCES m_pod(id),
    doc_id TEXT NOT NULL,
    UNIQUE (pod_id, doc_id)
);
CREATE TABLE pod_revision (
    successor_id TEXT NOT NULL,
    predecessor_id TEXT NOT NULL,
    UNIQUE (successor_id, predecessor_id)
);
"""

_CANONICAL_TABLES = (
    "m_project",
    "m_pod",
    "pod_project",
    "pod_monitoring",
    "pod_forecast",
    "pod_monitoring_api",
    "pod_forecast_api",
)
_RETIRED_OBJECTS = (
    "pod_economics",
    "pod_plan",
    "pod_project_economics",
    "pod_revision",
    "pod_value_case",
    "project_pod",
)


def _patch_dirs(monkeypatch, tmp_path):
    monkeypatch.setattr(configs.Config, "get_db_dir", classmethod(lambda cls: tmp_path))
    monkeypatch.setattr(
        configs.Config, "get_db_file", classmethod(lambda cls: tmp_path / "esdc.duckdb")
    )


def _latest_source(path: Path) -> Path:
    """Seed a minimal valid latest source: Jan 1 approval, order 1, generation 3."""
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA foreign_keys = ON")
    initialize_registry_v2(conn)
    conn.execute(
        "INSERT INTO m_project (project_id, project_name, is_active) "
        "VALUES ('P-0000001-01', 'Project One', 1)"
    )
    conn.commit()
    conn.execute("BEGIN IMMEDIATE")
    identity = issue_identity(
        conn,
        key=ApprovalKey(date(2024, 1, 1), 1, 1, 1, 0),
        actor="FA:test-publication",
    )
    conn.execute(
        "INSERT INTO m_pod (pod_id_seq, pod_uuid, approval_date, institution_id, "
        "pod_type_id, rev_num, pod_order_no, pod_id, pod_name, pod_letter_num, "
        "is_latest, capital_drilling_pod) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            identity.pod_id_seq,
            str(identity.pod_uuid),
            "2024-01-01",
            1,
            1,
            0,
            identity.pod_order_no,
            identity.pod_id,
            "Project One POD",
            "LETTER-1",
            1,
            "123.450",
        ),
    )
    conn.execute(
        "UPDATE pod_registry_state SET source_generation = 3 WHERE singleton_id = 1"
    )
    conn.commit()
    conn.close()
    return path


def _old_shaped_source(path: Path) -> Path:
    """Seed a source shaped by the retired POD schema (no current state marker)."""
    conn = sqlite3.connect(path)
    conn.executescript(_OLD_SHAPED_FIXTURE_SQL)
    conn.execute("INSERT INTO r_institution (code, institution) VALUES (1, 'SKK Migas')")
    conn.execute("INSERT INTO r_pod_type (code, pod_type) VALUES (1, 'POD I')")
    conn.execute(
        "INSERT INTO m_pod (id, pod_id, pod_name, approval_date, "
        "institution_code, pod_type_code, rev_num, approval_seq) "
        "VALUES (1, 'PL-2024-0001-1-1-0', 'POD Alpha', '2024-01-15', 1, 1, 0, 1)"
    )
    conn.commit()
    conn.close()
    return path


def _objects(conn: duckdb.DuckDBPyConnection) -> dict[str, str]:
    """Return projected object names with their DuckDB object type."""
    return {
        row[0]: row[1]
        for row in conn.execute(
            "SELECT table_name, table_type FROM information_schema.tables "
            "WHERE table_schema = 'main'"
        ).fetchall()
    }


def _metadata_names(conn: duckdb.DuckDBPyConnection) -> set[str]:
    return {
        row[0] for row in conn.execute(f"SELECT table_name FROM {_METADATA_TABLE}").fetchall()
    }


def test_publisher_facade_projects_only_the_current_canonical_contract(
    monkeypatch, tmp_path: Path
) -> None:
    """One publisher path: the canonical latest projection, no legacy fallback.

    A fresh latest source publishes the canonical tables through the public
    facade. An old-shaped source must be refused before DuckDB is written,
    leaving the existing mirror and unrelated tables intact, instead of being
    served through the retired projection path.
    """
    import esdc.pod_registry.publish as publish_module

    _patch_dirs(monkeypatch, tmp_path)
    target = tmp_path / "esdc.duckdb"

    report = publish_pod_registry(_latest_source(tmp_path / "latest.sqlite"), target)
    assert isinstance(report, PodPublicationReport)
    assert report.projection_error is None
    assert report.published_generation == 3

    conn = duckdb.connect(str(target), read_only=True)
    try:
        objects = _objects(conn)
        for name in _CANONICAL_TABLES:
            assert objects[name] == "BASE TABLE"
        assert objects["pod_registry"] == "VIEW"
        assert not set(_RETIRED_OBJECTS) & set(objects)
        metadata_names = _metadata_names(conn)
        assert {"m_pod", "pod_registry"} <= metadata_names
        assert not set(_RETIRED_OBJECTS) & metadata_names
        expected_rows = conn.execute(
            "SELECT pod_id, pod_name FROM pod_registry ORDER BY pod_id"
        ).fetchall()
    finally:
        conn.close()

    # No retired publisher path may remain reachable from the public facade.
    assert not hasattr(publish_module, "_publish_registry_tables")

    keeper = duckdb.connect(str(target))
    keeper.execute("CREATE TABLE unrelated_duck_table (value INTEGER)")
    keeper.execute("INSERT INTO unrelated_duck_table VALUES (42)")
    keeper.close()

    rejected = publish_pod_registry(
        _old_shaped_source(tmp_path / "old-shaped.sqlite"), target
    )

    assert isinstance(rejected, PodPublicationReport)
    error = rejected.projection_error
    assert error is not None
    assert "missing tables" in error

    after = duckdb.connect(str(target), read_only=True)
    try:
        assert after.execute(
            "SELECT pod_id, pod_name FROM pod_registry ORDER BY pod_id"
        ).fetchall() == expected_rows
        assert after.execute("SELECT value FROM unrelated_duck_table").fetchall() == [
            (42,)
        ]
        after_names = {row[0] for row in after.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'main'"
        ).fetchall()}
        assert not set(_RETIRED_OBJECTS) & after_names
    finally:
        after.close()


def test_fully_initialized_empty_source_publishes_an_empty_projection(
    tmp_path: Path
) -> None:
    """An initialized source with no business rows publishes nothing invented."""
    source = tmp_path / "empty-v2.sqlite"
    target = tmp_path / "esdc.duckdb"
    conn = sqlite3.connect(source)
    conn.execute("PRAGMA foreign_keys = ON")
    initialize_registry_v2(conn)
    conn.commit()
    conn.close()

    report = publish_pod_registry(source, target)

    assert report.projection_error is None
    counts = {result.table_name: result.row_count for result in report}
    for table in _CANONICAL_TABLES:
        assert counts[table] == 0
    mirror = duckdb.connect(str(target), read_only=True)
    try:
        assert mirror.execute("SELECT COUNT(*) FROM m_pod").fetchone() == (0,)
        assert mirror.execute(
            "SELECT source_generation FROM pod_registry_publication"
        ).fetchone() == (0,)
        assert not set(_RETIRED_OBJECTS) & set(_objects(mirror))
    finally:
        mirror.close()


def test_publish_carries_source_rows_and_links_through_the_projection(
    tmp_path: Path
) -> None:
    """Source truth reaches the projection: references, links, and documents."""
    source = _latest_source(tmp_path / "latest.sqlite")
    target = tmp_path / "esdc.duckdb"

    writer = sqlite3.connect(source)
    writer.execute("PRAGMA foreign_keys = ON")
    pod = writer.execute(
        "SELECT pod_id_seq, pod_id FROM m_pod WHERE is_latest = 1"
    ).fetchone()
    writer.execute(
        "INSERT INTO pod_project (pod_project_uuid, pod_id_seq, project_id, "
        "is_main_project, is_auto_inferred) VALUES (?, ?, ?, ?, ?)",
        (str(uuid4()), pod[0], "P-0000001-01", 1, 0),
    )
    writer.execute(
        "INSERT INTO pod_document (pod_id_seq, doc_id) VALUES (?, ?)",
        (pod[0], "ccbd4f3f27635c76"),
    )
    writer.commit()
    writer.close()

    report = publish_pod_registry(source, target)
    assert report.projection_error is None

    conn = duckdb.connect(str(target), read_only=True)
    try:
        assert conn.execute(
            "SELECT pod_id_seq, project_id FROM pod_project"
        ).fetchall() == [(pod[0], "P-0000001-01")]
        assert conn.execute(
            "SELECT pod_id_seq, doc_id FROM pod_document"
        ).fetchall() == [(pod[0], "ccbd4f3f27635c76")]
        row = conn.execute(
            "SELECT r.institution, t.pod_type, p.pod_id, p.pod_name "
            "FROM pod_registry p "
            "JOIN r_institution r ON r.id = p.institution_id "
            "JOIN r_pod_type t ON t.id = p.pod_type_id"
        ).fetchone()
        assert row == (
            "MESDM",
            "POD I",
            "PL-2024-0001-1-1-0",
            "Project One POD",
        )
    finally:
        conn.close()


def test_publish_is_idempotent(tmp_path: Path) -> None:
    """Republishing replaces the projection; it never duplicates rows."""
    source = _latest_source(tmp_path / "latest.sqlite")
    target = tmp_path / "esdc.duckdb"

    first = publish_pod_registry(source, target)
    second = publish_pod_registry(source, target)

    assert first.projection_error is None
    assert second.projection_error is None
    conn = duckdb.connect(str(target), read_only=True)
    try:
        assert conn.execute("SELECT COUNT(*) FROM m_pod").fetchone() == (1,)
        assert conn.execute("SELECT COUNT(*) FROM pod_project").fetchone() == (0,)
        assert conn.execute(
            "SELECT COUNT(*) FROM pod_registry_publication"
        ).fetchone() == (1,)
    finally:
        conn.close()


def test_refresh_does_not_recreate_retired_projection_objects(tmp_path: Path) -> None:
    """Stale retired objects are dropped and never recreated on refresh."""
    source = _latest_source(tmp_path / "latest.sqlite")
    target = tmp_path / "esdc.duckdb"

    stale = duckdb.connect(str(target))
    for name in _RETIRED_OBJECTS:
        stale.execute(f'CREATE TABLE "{name}" (pod_id VARCHAR)')
    stale.execute("CREATE VIEW pod_plan_view AS SELECT 1 AS x")
    stale.close()

    report = publish_pod_registry(source, target)
    assert report.projection_error is None

    conn = duckdb.connect(str(target), read_only=True)
    try:
        objects = _objects(conn)
        assert not set(_RETIRED_OBJECTS) & set(objects)
        assert objects["pod_registry"] == "VIEW"
        assert "pod_plan_view" in objects
        assert not set(_RETIRED_OBJECTS) & _metadata_names(conn)
    finally:
        conn.close()


def test_time_series_projection_tracks_source_changes(tmp_path: Path) -> None:
    """Canonical time-series tables follow the source, including deletions."""
    source = _latest_source(tmp_path / "latest.sqlite")
    target = tmp_path / "esdc.duckdb"

    writer = sqlite3.connect(source)
    writer.execute("PRAGMA foreign_keys = ON")
    pod = writer.execute(
        "SELECT pod_id_seq, pod_id FROM m_pod WHERE is_latest = 1"
    ).fetchone()
    writer.execute(
        "INSERT INTO pod_monitoring (monpod_uuid, pod_id_seq, pod_id, "
        "reported_date, is_completed, pod_stage_id, scenario_otlk) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (str(uuid4()), pod[0], pod[1], "2024-09-01", 1, 1, "outlook"),
    )
    writer.commit()
    writer.close()

    report = publish_pod_registry(source, target)
    assert report.projection_error is None
    mirror = duckdb.connect(str(target), read_only=True)
    try:
        rows = mirror.execute(
            "SELECT pod_id, reported_date, scenario_otlk FROM pod_monitoring"
        ).fetchall()
        assert rows == [(pod[1], date(2024, 9, 1), "outlook")]
        assert mirror.execute(
            "SELECT data_type FROM information_schema.columns "
            "WHERE table_name = 'pod_monitoring' AND column_name = 'reported_date'"
        ).fetchone() == ("DATE",)
    finally:
        mirror.close()

    writer = sqlite3.connect(source)
    writer.execute("DELETE FROM pod_monitoring WHERE pod_id_seq = ?", (pod[0],))
    writer.commit()
    writer.close()

    report = publish_pod_registry(source, target)
    assert report.projection_error is None
    mirror = duckdb.connect(str(target), read_only=True)
    try:
        assert mirror.execute("SELECT COUNT(*) FROM pod_monitoring").fetchone() == (0,)
    finally:
        mirror.close()


def test_contract_table_with_wrong_columns_is_refused_before_writing(
    tmp_path: Path
) -> None:
    """A contract table shaped differently must fail, not be silently read."""
    source = _latest_source(tmp_path / "latest.sqlite")
    target = tmp_path / "esdc.duckdb"

    good = publish_pod_registry(source, target)
    assert good.projection_error is None

    writer = sqlite3.connect(source)
    writer.execute("DROP TABLE pod_monitoring")
    writer.execute("CREATE TABLE pod_monitoring (foo TEXT)")
    writer.commit()
    writer.close()

    report = publish_pod_registry(source, target)
    error = report.projection_error
    assert error is not None
    assert "pod_monitoring" in error
    # The refusal now comes from the shared structural guard used by the publisher.
    assert "does not match the ordered POD contract columns" in error
    assert report.source_committed is False

    after = duckdb.connect(str(target), read_only=True)
    try:
        assert after.execute("SELECT pod_id FROM pod_registry").fetchall()
        assert after.execute(
            "SELECT source_generation FROM pod_registry_publication"
        ).fetchone() == (3,)
    finally:
        after.close()


def test_caches_are_invalidated_only_after_a_successful_publication(
    monkeypatch, tmp_path: Path
) -> None:
    """A refused source must not clear caches for a stale projection."""
    from esdc.chat import tools as chat_tools

    _patch_dirs(monkeypatch, tmp_path)
    source = _latest_source(tmp_path / "latest.sqlite")
    target = tmp_path / "esdc.duckdb"

    calls: list[str] = []
    monkeypatch.setattr(chat_tools, "reset_sql_cache", lambda: calls.append("sql"))
    monkeypatch.setattr(chat_tools, "invalidate_tool_cache", lambda: calls.append("tool"))

    report = publish_pod_registry(source, target)
    assert report.projection_error is None
    assert calls == ["sql", "tool"]

    calls.clear()
    rejected = publish_pod_registry(
        _old_shaped_source(tmp_path / "old-shaped.sqlite"), target
    )
    assert rejected.projection_error is not None
    assert calls == []


def test_preflight_checks_the_current_schema_shape_not_only_table_names(
    tmp_path: Path,
) -> None:
    """Keeping every required table name is not the same as satisfying the contract.

    A source that re-creates one contract table with a wrong column can never publish,
    so the preflight has to refuse it as well; otherwise a caller that mutates DuckDB
    before publishing destroys the projection that this source can no longer rebuild.
    The same read-only check reports a missing state row and an incompatible marker,
    names the tables an old-shaped source cannot satisfy, passes a current source, and
    represents a source it cannot open as a preflight failure.
    """
    from esdc.pod_registry.publish import source_preflight_error

    current = _latest_source(tmp_path / "current.sqlite")
    assert source_preflight_error(current) is None

    writer = sqlite3.connect(current)
    writer.execute("DROP TABLE m_project")
    writer.execute("CREATE TABLE m_project (wrong TEXT)")
    writer.commit()
    writer.close()

    error = source_preflight_error(current)
    assert error is not None
    assert "m_project" in error
    assert "ordered POD contract columns" in error

    no_state_row = _latest_source(tmp_path / "no-state-row.sqlite")
    writer = sqlite3.connect(no_state_row)
    writer.execute("DELETE FROM pod_registry_state")
    writer.commit()
    writer.close()
    error = source_preflight_error(no_state_row)
    assert error is not None
    assert "incompatible" in error

    wrong_marker = _latest_source(tmp_path / "wrong-marker.sqlite")
    writer = sqlite3.connect(wrong_marker)
    writer.execute(
        "UPDATE pod_registry_state SET schema_version = 99 WHERE singleton_id = 1"
    )
    writer.commit()
    writer.close()
    error = source_preflight_error(wrong_marker)
    assert error is not None
    assert "incompatible" in error

    # A source that is not a v2 registry is still named table by table, because the
    # preflight is what a mutating caller sees first.
    error = source_preflight_error(_old_shaped_source(tmp_path / "old-shaped.sqlite"))
    assert error is not None
    assert "missing tables" in error
    assert "pod_project" in error

    corrupt = tmp_path / "corrupt.sqlite"
    corrupt.write_bytes(b"not a database")
    error = source_preflight_error(corrupt)
    assert error is not None
    assert error.startswith("source preflight failed")

    missing = tmp_path / "missing.sqlite"
    assert source_preflight_error(missing) == "SQLite source is unavailable"
    assert not missing.exists()


def test_preflight_read_snapshot_is_ended_and_the_source_is_left_untouched(
    tmp_path: Path,
) -> None:
    """The refusal path ends its read snapshot and does not change the source.

    Preflight runs on readers that must never create, initialize, or move a source to a
    WAL writer, and it must end the snapshot on the refusal path too: a leaked read
    transaction would block the writer that legitimately repairs the source afterwards.
    Both are checked on refusal paths — an old-shaped source that was never initialized,
    and a current source whose shape cannot satisfy the contract.
    """
    from esdc.pod_registry.publish import source_preflight_error

    def probe(path: Path) -> tuple[str, tuple[str, ...], int]:
        conn = sqlite3.connect(path)
        try:
            return (
                conn.execute("PRAGMA journal_mode").fetchone()[0],
                tuple(row[1] for row in conn.execute("PRAGMA table_info(m_project)").fetchall()),
                conn.execute(
                    "SELECT schema_version FROM pod_registry_state WHERE singleton_id = 1"
                ).fetchone()[0],
            )
        finally:
            conn.close()

    legacy = _old_shaped_source(tmp_path / "old-shaped.sqlite")
    assert source_preflight_error(legacy) is not None
    check = sqlite3.connect(legacy)
    try:
        # Nothing was initialized here, and the snapshot is ended so a writer can lock.
        assert check.execute("PRAGMA journal_mode").fetchone()[0] != "wal"
        check.execute("BEGIN IMMEDIATE")
        check.commit()
    finally:
        check.close()

    source = _latest_source(tmp_path / "shape-check.sqlite")
    writer = sqlite3.connect(source)
    writer.execute("DROP TABLE m_project")
    writer.execute("CREATE TABLE m_project (wrong TEXT)")
    writer.commit()
    writer.close()
    before = probe(source)

    assert source_preflight_error(source) is not None

    assert probe(source) == before
    check = sqlite3.connect(source)
    try:
        # The snapshot was ended, so a writer can still take its lock.
        check.execute("BEGIN IMMEDIATE")
        check.commit()
    finally:
        check.close()
