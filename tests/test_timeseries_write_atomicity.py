"""Atomicity contracts for DuckDB timeseries and resource writers."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import duckdb
import pytest

from esdc.api_snapshot import JSONScalar
from esdc.dbmanager import (
    _apply_timeseries_stage,
    _ensure_duckdb_database,
    _stage_timeseries_slices,
    load_data_to_db,
)
from esdc.fetching import FetchFailure, PreparedTimeseriesSlice


def _slice(
    directory: Path,
    year: int,
    header: tuple[str, ...],
    rows: list[tuple[object, ...]],
) -> PreparedTimeseriesSlice:
    path = directory / f"{year}.jsonl"
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return PreparedTimeseriesSlice(year, header, path, len(rows))


def _create_timeseries_target(conn: duckdb.DuckDBPyConnection) -> None:
    from esdc.db_security import _load_sql_script

    for statement in _load_sql_script("create_table_project_timeseries.sql").split(";"):
        if statement.strip():
            conn.execute(statement)
    conn.execute("ALTER TABLE project_timeseries ADD COLUMN uuid VARCHAR")


def _seed_views_and_metadata(conn: duckdb.DuckDBPyConnection) -> None:
    from esdc.db_security import _load_sql_script

    for statement in _load_sql_script("create_timeseries_views.sql").split(";"):
        if statement.strip():
            conn.execute(statement)
    conn.execute("CREATE TABLE _metadata (key VARCHAR PRIMARY KEY, value VARCHAR)")
    conn.execute("INSERT INTO _metadata VALUES ('sentinel', 'preserve-me')")


def _assert_view_and_metadata_state(conn: duckdb.DuckDBPyConnection) -> None:
    assert conn.execute(
        "SELECT value FROM _metadata WHERE key='sentinel'"
    ).fetchone() == ("preserve-me",)
    for view_name in ("field_timeseries", "wa_timeseries", "nkri_timeseries"):
        assert conn.execute(
            "SELECT COUNT(*) FROM information_schema.views WHERE table_name=?",
            [view_name],
        ).fetchone() == (1,)
        conn.execute(f"SELECT * FROM {view_name} LIMIT 0")


def _publish_timeseries_baseline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    rows: list[list[JSONScalar]],
) -> tuple[Path, Path]:
    db_path = tmp_path / "esdc.duckdb"
    sqlite_path = tmp_path / "esdc.sqlite"
    monkeypatch.setenv("ESDC_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setenv("ESDC_DB_DIR", str(tmp_path))
    monkeypatch.setenv("ESDC_DB_FILE", str(db_path))
    monkeypatch.setattr("esdc.dbmanager.Config.get_db_file", lambda: db_path)
    monkeypatch.setattr("esdc.dbmanager.Config.get_db_dir", lambda: tmp_path)
    from esdc.fetch_publication import ingest_timeseries_rows

    ingest_timeseries_rows(
        rows,
        ["report_year", "project_id", "project_name", "cprd_grs_oil"],
        duckdb_path=db_path,
        sqlite_path=sqlite_path,
    )
    with duckdb.connect(str(db_path)) as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS _metadata "
            "(key VARCHAR PRIMARY KEY, value VARCHAR)"
        )
        conn.execute("INSERT OR REPLACE INTO _metadata VALUES ('sentinel', 'preserve-me')")
    return db_path, sqlite_path


def _timeseries_analytics_rows(db_path: Path) -> list[tuple[object, ...]]:
    with duckdb.connect(str(db_path), read_only=True) as conn:
        return conn.execute(
            "SELECT id, report_year, project_id, project_name, cprd_grs_oil, uuid "
            "FROM project_timeseries ORDER BY report_year NULLS FIRST"
        ).fetchall()


def _publish_bootstrapped_timeseries_baseline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path]:
    from esdc.api_snapshot import ApiProjectResourceRow
    from esdc.fetch_publication import publish_prepared_fetch
    from esdc.fetching import PreparedFetch, spool_timeseries
    from esdc.timeseries_bootstrap import bootstrap_timeseries_source

    db_path = tmp_path / "legacy.duckdb"
    sqlite_path = tmp_path / "esdc.sqlite"
    monkeypatch.setenv("ESDC_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setenv("ESDC_DB_DIR", str(tmp_path))
    monkeypatch.setenv("ESDC_DB_FILE", str(db_path))
    monkeypatch.setattr("esdc.dbmanager.Config.get_db_file", lambda: db_path)
    monkeypatch.setattr("esdc.dbmanager.Config.get_db_dir", lambda: tmp_path)
    with duckdb.connect(str(db_path)) as conn:
        _create_timeseries_target(conn)
        _seed_views_and_metadata(conn)
        conn.execute(
            "INSERT INTO project_timeseries "
            "(id, report_year, project_id, project_name, cprd_grs_oil, uuid) VALUES "
            "(201, 2024, 'P-24', 'keep', 24, 'keep-24'), "
            "(202, NULL, 'P-null', 'keep null', 99, NULL), "
            "(203, 2025, 'P-25-old', 'replace', 25, 'replace-25')"
        )
    bootstrap_timeseries_source(
        duckdb_path=db_path, sqlite_path=sqlite_path, apply=True
    )
    prepared_slice = spool_timeseries(
        ("report_year", "project_id", "project_name", "cprd_grs_oil"),
        [(2025, "P-25", "published", 25)],
        report_year=2025,
        directory=tmp_path,
    )
    assert prepared_slice is not None
    outcome = publish_prepared_fetch(
        PreparedFetch(
            (ApiProjectResourceRow(project_id="P-25", report_year=2025),),
            None,
            (prepared_slice,),
            (),
            (),
            timeseries_observed_years=(2025,),
            timeseries_complete_observation=True,
        ),
        duckdb_path=db_path,
        sqlite_path=sqlite_path,
        reindex=False,
    )
    assert outcome.timeseries_revision is not None
    return db_path, sqlite_path


def test_timeseries_stage_apply_rolls_back_as_part_of_callers_transaction(
    tmp_path: Path,
) -> None:
    conn = duckdb.connect(str(tmp_path / "rollback.duckdb"))
    source = _slice(
        tmp_path,
        2025,
        ("report_year", "project_id", "cprd_grs_oil"),
        [(2025, "P-1", 12.5)],
    )
    try:
        conn.execute("BEGIN")
        stage = _stage_timeseries_slices(conn, [source])
        assert stage is not None
        _apply_timeseries_stage(conn, stage, [source])
        with pytest.raises(duckdb.CatalogException):
            conn.execute("SELECT * FROM intentionally_missing_table")
        conn.execute("ROLLBACK")
        assert conn.execute(
            "SELECT COUNT(*) FROM information_schema.tables "
            "WHERE table_name = 'project_timeseries'"
        ).fetchone() == (0,)
    finally:
        conn.close()


def test_timeseries_stage_uses_existing_target_numeric_precision(tmp_path: Path) -> None:
    conn = duckdb.connect(str(tmp_path / "target-types.duckdb"))
    conn.execute(
        "CREATE TABLE project_timeseries (report_year INTEGER, project_id VARCHAR, "
        "cprd_grs_oil DECIMAL(20, 10))"
    )
    value = 12.3456789012
    source = _slice(
        tmp_path,
        2025,
        ("report_year", "project_id", "cprd_grs_oil"),
        [(2025, "P-1", value)],
    )
    try:
        conn.execute("BEGIN")
        stage = _stage_timeseries_slices(conn, [source])
        assert stage is not None
        staged = conn.execute(
            f"SELECT CAST(cprd_grs_oil AS VARCHAR) FROM {stage}"
        ).fetchone()
        conn.execute("ROLLBACK")
    finally:
        conn.close()
    assert staged == ("12.3456789012",)


def test_timeseries_subset_apply_writes_omitted_union_columns_as_null(
    tmp_path: Path,
) -> None:
    conn = duckdb.connect(str(tmp_path / "subset-defaults.duckdb"))
    _create_timeseries_target(conn)
    conn.execute(
        "ALTER TABLE project_timeseries "
        "ALTER COLUMN cprd_grs_con SET DEFAULT 99"
    )
    first = _slice(
        tmp_path,
        2025,
        ("report_year", "project_id", "cprd_grs_oil"),
        [(2025, "P-25", 10.0)],
    )
    second = _slice(
        tmp_path,
        2026,
        ("report_year", "project_id"),
        [(2026, "P-26")],
    )
    try:
        conn.execute("INSTALL fts")
        conn.execute("LOAD fts")
        conn.execute("BEGIN")
        stage = _stage_timeseries_slices(conn, [first, second])
        assert stage is not None
        _apply_timeseries_stage(conn, stage, [first, second])
        conn.execute("COMMIT")
        assert conn.execute(
            "SELECT report_year, cprd_grs_oil, cprd_grs_con "
            "FROM project_timeseries "
            "ORDER BY report_year"
        ).fetchall() == [(2025, 10.0, None), (2026, None, None)]
    finally:
        conn.close()


def test_timeseries_apply_replaces_only_nonempty_year_and_keeps_untouched_ids(
    tmp_path: Path,
) -> None:
    conn = duckdb.connect(str(tmp_path / "slices.duckdb"))
    _create_timeseries_target(conn)
    conn.execute(
        "INSERT INTO project_timeseries (id, report_year, project_id, project_name, uuid) "
        "VALUES (101, 2024, 'P-24', 'old 24', 'keep-24'), "
        "(102, 2025, 'P-25-old', 'old 25', 'replace-25'), "
        "(103, NULL, 'P-null', 'null year', 'keep-null')"
    )
    conn.execute("CREATE TABLE _metadata (key VARCHAR PRIMARY KEY, value VARCHAR)")
    source = _slice(
        tmp_path,
        2025,
        ("report_year", "project_id", "cprd_grs_oil"),
        [(2025, "P-25-new", 0)],
    )
    try:
        conn.execute("BEGIN")
        stage = _stage_timeseries_slices(conn, [source])
        assert stage is not None
        _apply_timeseries_stage(conn, stage, [source])
        conn.execute("COMMIT")
        rows = conn.execute(
            "SELECT id, report_year, project_id, project_name, cprd_grs_oil, uuid "
            "FROM project_timeseries ORDER BY report_year NULLS FIRST"
        ).fetchall()
    finally:
        conn.close()
    assert rows == [
        (103, None, "P-null", "null year", None, "keep-null"),
        (101, 2024, "P-24", "old 24", None, "keep-24"),
        (rows[2][0], 2025, "P-25-new", None, 0.0, rows[2][5]),
    ]
    assert rows[2][0] not in {101, 102, 103}
    assert rows[2][5] and rows[2][5] != "replace-25"


def test_full_timeseries_loader_rolls_back_failed_later_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path, sqlite_path = _publish_timeseries_baseline(
        tmp_path,
        monkeypatch,
        [[2024, "P-24", "protected", 10]],
    )
    before = _timeseries_analytics_rows(db_path)
    invalidations: list[str] = []
    monkeypatch.setattr(
        "esdc.dbmanager.invalidate_sql_cache",
        lambda: invalidations.append("sql"),
    )
    monkeypatch.setattr(
        "esdc.chat.tools.invalidate_tool_cache",
        lambda: invalidations.append("tool"),
    )

    with pytest.raises(FetchFailure) as failure:
        load_data_to_db(
            [["2025", "P-valid", "1.5"], ["2025", "P-bad", "not-a-number"]],
            ["report_year", "project_id", "cprd_grs_oil"],
            "project_timeseries",
        )
    assert failure.value.phase == "publication"
    assert failure.value.sqlite_committed is True

    assert _timeseries_analytics_rows(db_path) == before
    conn = duckdb.connect(str(db_path), read_only=True)
    try:
        _assert_view_and_metadata_state(conn)
    finally:
        conn.close()
    from esdc.timeseries_snapshot import get_timeseries_snapshot

    sqlite_conn = sqlite3.connect(sqlite_path)
    try:
        snapshot = get_timeseries_snapshot(sqlite_conn)
        assert snapshot is not None
        assert snapshot.publication_state == "analytics_stale"
        projects = [
            json.loads(row[0])["project_id"]
            for row in sqlite_conn.execute(
                "SELECT row_json FROM api_project_timeseries ORDER BY row_json"
            )
        ]
        assert set(projects) == {"P-bad", "P-valid"}
    finally:
        sqlite_conn.close()
    assert invalidations == []


def test_per_year_append_rolls_back_failed_insert(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from esdc.esdc import _append_to_table

    db_path, sqlite_path = _publish_bootstrapped_timeseries_baseline(
        tmp_path, monkeypatch
    )
    before = _timeseries_analytics_rows(db_path)
    invalidations: list[str] = []
    monkeypatch.setattr(
        "esdc.dbmanager.invalidate_sql_cache",
        lambda: invalidations.append("sql"),
    )
    monkeypatch.setattr(
        "esdc.chat.tools.invalidate_tool_cache",
        lambda: invalidations.append("tool"),
    )

    with pytest.raises(FetchFailure) as failure:
        _append_to_table(
            "project_timeseries",
            ["report_year", "project_id", "cprd_grs_oil"],
            [["2025", "P-valid", "1.5"], ["2025", "P-bad", "not-a-number"]],
            [2025],
        )
    assert failure.value.phase == "publication"
    assert failure.value.sqlite_committed is True

    assert _timeseries_analytics_rows(db_path) == before
    conn = duckdb.connect(str(db_path), read_only=True)
    try:
        _assert_view_and_metadata_state(conn)
    finally:
        conn.close()
    from esdc.timeseries_snapshot import get_timeseries_snapshot

    sqlite_conn = sqlite3.connect(sqlite_path)
    try:
        snapshot = get_timeseries_snapshot(sqlite_conn)
        assert snapshot is not None
        assert snapshot.publication_state == "analytics_stale"
        projects = [
            json.loads(row[0])["project_id"]
            for row in sqlite_conn.execute(
                "SELECT row_json FROM api_project_timeseries ORDER BY row_json"
            )
        ]
        assert set(projects) == {"P-24", "P-bad", "P-null", "P-valid"}
    finally:
        sqlite_conn.close()
    assert invalidations == []


def test_per_year_append_keeps_unselected_and_null_year_identifiers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from esdc.esdc import _append_to_table

    db_path, _ = _publish_bootstrapped_timeseries_baseline(tmp_path, monkeypatch)
    before = _timeseries_analytics_rows(db_path)
    untouched = {row[1]: row for row in before}

    _append_to_table(
        "project_timeseries",
        ["report_year", "project_id", "project_name"],
        [["2025", "P-25-new", "new"]],
        [2025],
    )

    rows = _timeseries_analytics_rows(db_path)
    assert len(rows) == 3
    by_project = {row[2]: row for row in rows}
    after_by_year = {row[1]: row for row in rows}
    assert after_by_year[None] == untouched[None]
    assert after_by_year[2024] == untouched[2024]
    assert by_project["P-25-new"][1:3] == (2025, "P-25-new")
    assert by_project["P-25-new"][0] != untouched[2025][0]
    assert by_project["P-25-new"][5]


def test_per_year_append_rolls_back_after_delete_when_required_index_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from esdc.esdc import _append_to_table

    db_path, sqlite_path = _publish_timeseries_baseline(
        tmp_path,
        monkeypatch,
        [[2025, "P-25", "before", 25]],
    )
    before = _timeseries_analytics_rows(db_path)
    invalidations: list[str] = []
    monkeypatch.setattr(
        "esdc.dbmanager._create_fts_indexes",
        lambda *args, **kwargs: {"project_timeseries": "required index failed"},
    )
    monkeypatch.setattr(
        "esdc.dbmanager.invalidate_sql_cache",
        lambda: invalidations.append("sql"),
    )
    monkeypatch.setattr(
        "esdc.chat.tools.invalidate_tool_cache",
        lambda: invalidations.append("tool"),
    )

    with pytest.raises(FetchFailure, match="required index work failed"):
        _append_to_table(
            "project_timeseries",
            ["report_year", "project_id", "project_name"],
            [["2025", "P-25-new", "after"]],
            [2025],
        )

    assert _timeseries_analytics_rows(db_path) == before
    conn = duckdb.connect(str(db_path), read_only=True)
    try:
        _assert_view_and_metadata_state(conn)
    finally:
        conn.close()
    from esdc.timeseries_snapshot import get_timeseries_snapshot

    sqlite_conn = sqlite3.connect(sqlite_path)
    try:
        snapshot = get_timeseries_snapshot(sqlite_conn)
        assert snapshot is not None
        assert snapshot.publication_state == "analytics_stale"
        assert json.loads(
            sqlite_conn.execute(
                "SELECT row_json FROM api_project_timeseries"
            ).fetchone()[0]
        )["project_id"] == "P-25-new"
    finally:
        sqlite_conn.close()
    assert invalidations == []


def test_resource_low_level_loader_refuses_before_creating_database(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = tmp_path / "must-not-exist.duckdb"
    monkeypatch.setattr("esdc.dbmanager.Config.get_db_file", lambda: db_path)
    monkeypatch.setattr("esdc.dbmanager.Config.get_db_dir", lambda: tmp_path)

    with pytest.raises(
        ValueError,
        match="^project_resources must be ingested through the canonical SQLite resource ingestion path$",
    ):
        load_data_to_db([], [], "project_resources")

    assert not db_path.exists()


def test_resource_per_year_wrapper_refuses_before_creating_database(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from esdc.esdc import _append_to_table

    db_path = tmp_path / "must-not-exist-append.duckdb"
    monkeypatch.setattr("esdc.esdc.Config.get_db_file", lambda: db_path)
    monkeypatch.setattr("esdc.esdc.Config.get_db_dir", lambda: tmp_path)

    with pytest.raises(
        ValueError,
        match="^project_resources must be ingested through the canonical SQLite resource ingestion path$",
    ):
        _append_to_table("project_resources", [], [], [])

    assert not db_path.exists()


def test_sqlite_format_database_is_preserved_and_refused(tmp_path: Path) -> None:
    db_path = tmp_path / "sqlite-format.duckdb"
    sqlite_conn = sqlite3.connect(db_path)
    sqlite_conn.execute("CREATE TABLE sentinel (value TEXT)")
    sqlite_conn.execute("INSERT INTO sentinel VALUES ('untouched')")
    sqlite_conn.commit()
    sqlite_conn.close()
    original_bytes = db_path.read_bytes()

    with pytest.raises(ValueError, match="SQLite-format"):
        _ensure_duckdb_database(db_path)

    assert db_path.read_bytes() == original_bytes
    sqlite_conn = sqlite3.connect(db_path)
    try:
        assert sqlite_conn.execute("SELECT value FROM sentinel").fetchone() == (
            "untouched",
        )
    finally:
        sqlite_conn.close()


def test_timeseries_wrappers_skip_empty_content_without_opening_database(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from esdc.esdc import _append_to_table

    full_path = tmp_path / "empty-full.duckdb"
    append_path = tmp_path / "empty-append.duckdb"
    monkeypatch.setattr("esdc.dbmanager.Config.get_db_file", lambda: full_path)
    monkeypatch.setattr("esdc.dbmanager.Config.get_db_dir", lambda: tmp_path)
    monkeypatch.setattr("esdc.esdc.Config.get_db_file", lambda: append_path)
    monkeypatch.setattr("esdc.esdc.Config.get_db_dir", lambda: tmp_path)
    monkeypatch.setattr(
        "esdc.dbmanager.get_duckdb_connection",
        lambda *args, **kwargs: pytest.fail("empty full load opened a writer"),
    )
    monkeypatch.setattr(
        "esdc.esdc.get_duckdb_connection",
        lambda *args, **kwargs: pytest.fail("empty append opened a writer"),
    )

    load_data_to_db([], ["report_year", "project_id"], "project_timeseries")
    _append_to_table(
        "project_timeseries", ["report_year", "project_id"], [], [2025]
    )

    assert not full_path.exists()
    assert not append_path.exists()


def test_empty_timeseries_apply_is_noop_without_stage(tmp_path: Path) -> None:
    conn = duckdb.connect(str(tmp_path / "empty.duckdb"))
    try:
        conn.execute("BEGIN")
        stage = _stage_timeseries_slices(conn, [])
        assert stage is None
        _apply_timeseries_stage(conn, "unused", [])
        conn.execute("COMMIT")
        assert conn.execute(
            "SELECT COUNT(*) FROM information_schema.tables "
            "WHERE table_name = 'project_timeseries'"
        ).fetchone() == (0,)
    finally:
        conn.close()


def test_programmatic_timeseries_ingestion_publishes_sqlite_and_duckdb_pair(
    tmp_path: Path,
) -> None:
    from esdc.fetch_publication import ingest_timeseries_rows

    duckdb_path = tmp_path / "programmatic.duckdb"
    sqlite_path = tmp_path / "programmatic.sqlite"
    outcome = ingest_timeseries_rows(
        [(2025, "P-1", 2025, 12.3456789012)],
        ("report_year", "project_id", "year", "cprd_grs_oil"),
        duckdb_path=duckdb_path,
        sqlite_path=sqlite_path,
    )

    assert outcome.timeseries_rows_written == 1
    assert outcome.updated_timeseries_years == (2025,)
    with sqlite3.connect(sqlite_path) as conn:
        canonical = conn.execute(
            "SELECT row_json FROM api_project_timeseries"
        ).fetchone()
    assert canonical is not None
    assert json.loads(canonical[0])["cprd_grs_oil"] == 12.3456789012
    with duckdb.connect(str(duckdb_path), read_only=True) as conn:
        assert conn.execute(
            "SELECT report_year, project_id, year, cprd_grs_oil "
            "FROM project_timeseries"
        ).fetchall()[0][:3] == (2025, "P-1", 2025)
        value = conn.execute(
            "SELECT cprd_grs_oil FROM project_timeseries"
        ).fetchone()
        assert value is not None
        assert value[0] == pytest.approx(12.3456789012)
