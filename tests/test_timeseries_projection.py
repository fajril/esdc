"""Regression tests for the SQLite to DuckDB timeseries projection seam."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import duckdb
import pytest

from esdc.pod_registry.store import get_sqlite_connection
from esdc.timeseries_projection import (
    TimeseriesStage,
    apply_timeseries_stage,
    stage_canonical_timeseries,
)
from esdc.timeseries_snapshot import (
    ensure_timeseries_tables,
    replace_timeseries_in_transaction,
)


def _source(path: Path, rows: list[tuple[int, str, int, float]]) -> sqlite3.Connection:
    conn = get_sqlite_connection(path)
    ensure_timeseries_tables(conn)
    conn.execute("BEGIN")
    replace_timeseries_in_transaction(
        conn,
        (
            __import__(
                "esdc.timeseries_snapshot", fromlist=["x"]
            ).normalize_timeseries_row(
                ("report_year", "project_id", "year", "cprd_grs_oil"),
                row,
                origin="api_json",
            )
            for row in rows
        ),
        years=None,
        revision="source-r1",
        fetched_at="2026-10-23T00:00:00+00:00",
    )
    conn.commit()
    return conn


def _skip_population_fts(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, ...]]:
    calls: list[tuple[str, ...]] = []

    def create_indexes(conn, *, tables, extensions_loaded):
        assert extensions_loaded is True
        calls.append(tuple(tables))
        return dict.fromkeys(tables, "ok")

    monkeypatch.setattr("esdc.dbmanager._create_fts_indexes", create_indexes)
    return calls


def test_full_projection_preserves_duplicate_rows_and_builds_views(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fts_calls = _skip_population_fts(monkeypatch)
    source = _source(
        tmp_path / "source.sqlite",
        [(2025, "P-1", 2025, 1.25), (2025, "P-1", 2025, 1.25)],
    )
    conn = duckdb.connect(":memory:")
    stage = stage_canonical_timeseries(conn, source, years=None)
    conn.execute("BEGIN")
    apply_timeseries_stage(conn, stage)
    conn.execute("COMMIT")
    values = conn.execute(
        "SELECT project_id, report_year, cprd_grs_oil FROM project_timeseries "
        "ORDER BY id"
    ).fetchall()
    assert values == [("P-1", 2025, 1.25), ("P-1", 2025, 1.25)]
    uuids = conn.execute("SELECT uuid FROM project_timeseries").fetchall()
    assert all(value[0] for value in uuids)
    assert len({value[0] for value in uuids}) == 2
    assert conn.execute(
        "SELECT COUNT(*) FROM information_schema.views WHERE table_name='field_timeseries'"
    ).fetchone() == (1,)
    assert fts_calls == [("project_timeseries",)]
    conn.close()
    source.close()


def test_year_projection_preserves_other_ids_null_uuids_and_numeric_type(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fts_calls = _skip_population_fts(monkeypatch)
    source = _source(
        tmp_path / "source.sqlite",
        [(2025, "P-new", 2025, 1.234567890123)],
    )
    conn = duckdb.connect(":memory:")
    from esdc.dbmanager import _execute_sql_script

    _execute_sql_script(conn, "create_table_project_timeseries.sql")
    conn.execute("ALTER TABLE project_timeseries ADD COLUMN uuid VARCHAR")
    conn.execute("ALTER TABLE project_timeseries ALTER COLUMN cprd_grs_oil TYPE DOUBLE")
    conn.execute(
        "INSERT INTO project_timeseries(project_id, report_year, year, cprd_grs_oil, uuid) "
        "VALUES ('P-keep', 2024, 2024, 8.5, NULL), ('P-old', 2025, 2025, 3.5, 'old-uuid')"
    )
    untouched = conn.execute(
        "SELECT id, uuid FROM project_timeseries WHERE project_id='P-keep'"
    ).fetchone()
    stage = stage_canonical_timeseries(conn, source, years=(2025,))
    conn.execute("BEGIN")
    apply_timeseries_stage(conn, stage)
    conn.execute("COMMIT")
    assert (
        conn.execute(
            "SELECT id, uuid FROM project_timeseries WHERE project_id='P-keep'"
        ).fetchone()
        == untouched
    )
    assert conn.execute(
        "SELECT project_id, cprd_grs_oil FROM project_timeseries WHERE report_year=2025"
    ).fetchall() == [("P-new", 1.234567890123)]
    assert conn.execute(
        "SELECT data_type FROM information_schema.columns "
        "WHERE table_name='project_timeseries' AND column_name='cprd_grs_oil'"
    ).fetchone() == ("DOUBLE",)
    uuid_row = conn.execute(
        "SELECT uuid FROM project_timeseries WHERE project_id='P-new'"
    ).fetchone()
    assert uuid_row is not None and uuid_row[0]
    assert fts_calls == [("project_timeseries",)]
    conn.close()
    source.close()


def test_stage_identifier_is_validated_and_rollback_keeps_stage_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _skip_population_fts(monkeypatch)
    source = _source(tmp_path / "source.sqlite", [(2025, "P-1", 2025, 1.0)])
    conn = duckdb.connect(":memory:")
    stage = stage_canonical_timeseries(conn, source, years=(2025,))
    with pytest.raises(ValueError, match="unsafe timeseries stage"):
        apply_timeseries_stage(
            conn, TimeseriesStage("project_timeseries; DROP TABLE x", 1, (2025,))
        )
    conn.execute("BEGIN")
    apply_timeseries_stage(conn, stage)
    conn.execute("ROLLBACK")
    assert conn.execute(
        "SELECT COUNT(*) FROM information_schema.tables WHERE table_name=?",
        [stage.table_name],
    ).fetchone() == (1,)
    conn.close()
    source.close()
