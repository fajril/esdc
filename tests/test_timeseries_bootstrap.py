"""Regressions for legacy DuckDB timeseries bootstrap safety and fidelity."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from decimal import Decimal
from pathlib import Path
from typing import cast

import duckdb
import pytest
from typer.testing import CliRunner

from esdc.timeseries_bootstrap import bootstrap_timeseries_source
from esdc.timeseries_snapshot import (
    TIMESERIES_SOURCE_COLUMNS,
    TIMESERIES_SOURCE_TYPES,
    ensure_timeseries_tables,
    get_timeseries_snapshot,
    iter_timeseries_rows,
)


def _create_duckdb(path: Path, *, missing: str | None = None) -> duckdb.DuckDBPyConnection:
    conn = duckdb.connect(str(path))
    definitions = []
    for column in TIMESERIES_SOURCE_COLUMNS:
        if column == missing:
            continue
        if column == "tpf_oc":
            sql_type = "DECIMAL(24,8)"
        elif column == "tpf_oil":
            sql_type = "DOUBLE"
        else:
            sql_type = {
                "TEXT": "VARCHAR",
                "INTEGER": "INTEGER",
                "REAL": "DOUBLE",
            }[TIMESERIES_SOURCE_TYPES[column]]
        definitions.append(f'"{column}" {sql_type}')
    conn.execute(
        "CREATE TABLE project_timeseries ("
        + ", ".join(definitions)
        + ', id BIGINT, uuid UUID)'
    )
    return conn


def _values(*, year: int | None, decimal_value: str = "1234567890123456.12345678") -> dict[str, object]:
    values = cast(
        dict[str, object],
        dict.fromkeys(TIMESERIES_SOURCE_COLUMNS),
    )
    values["report_year"] = year
    values["project_id"] = "legacy-project"
    values["tpf_oc"] = Decimal(decimal_value)
    values["tpf_oil"] = 1.2345678901234567
    return values


def _insert(conn: duckdb.DuckDBPyConnection, row: dict[str, object], *, row_id: int) -> None:
    columns = (*TIMESERIES_SOURCE_COLUMNS, "id", "uuid")
    values = [row.get(column) for column in TIMESERIES_SOURCE_COLUMNS]
    values.extend([row_id, f"00000000-0000-0000-0000-{row_id:012d}"])
    placeholders = ",".join("?" for _ in columns)
    quoted = ",".join(f'"{column}"' for column in columns)
    conn.execute(f"INSERT INTO project_timeseries ({quoted}) VALUES ({placeholders})", values)


def test_dry_run_creates_nothing(tmp_path: Path) -> None:
    source = tmp_path / "legacy.duckdb"
    sqlite_path = tmp_path / "nested" / "esdc.sqlite"
    conn = _create_duckdb(source)
    _insert(conn, _values(year=2024), row_id=1)
    conn.close()
    original_hash = hashlib.sha256(source.read_bytes()).hexdigest()

    report = bootstrap_timeseries_source(duckdb_path=source, sqlite_path=sqlite_path)

    assert report.applied is False
    assert report.row_count == 1
    assert report.source_sha256 == original_hash
    assert sqlite_path.exists() is False
    assert sqlite_path.parent.exists() is False
    assert hashlib.sha256(source.read_bytes()).hexdigest() == original_hash


def test_apply_preserves_legacy_years_nulls_duplicates_and_decimal(tmp_path: Path) -> None:
    source = tmp_path / "legacy.duckdb"
    sqlite_path = tmp_path / "esdc.sqlite"
    conn = _create_duckdb(source)
    _insert(conn, _values(year=2010), row_id=1)
    _insert(conn, _values(year=None), row_id=2)
    _insert(conn, _values(year=2010), row_id=3)
    conn.close()

    report = bootstrap_timeseries_source(
        duckdb_path=source, sqlite_path=sqlite_path, apply=True
    )

    assert report.applied is True
    assert report.row_count == 3
    with sqlite3.connect(sqlite_path) as sqlite_conn:
        rows = list(iter_timeseries_rows(sqlite_conn))
        snapshot = get_timeseries_snapshot(sqlite_conn)
    assert [row.report_year for row in rows].count(2010) == 2
    assert [row.report_year for row in rows].count(None) == 1
    decoded = [json.loads(row.row_json) for row in rows]
    assert {row["tpf_oc"] for row in decoded} == {"1234567890123456.12345678"}
    assert {row["tpf_oil"] for row in decoded} == {1.2345678901234567}
    assert all(row.origin == "legacy_duckdb" for row in rows)
    assert snapshot is not None and snapshot.publication_state == "committed"


def test_apply_never_changes_duckdb_or_nonapi_sqlite_tables(tmp_path: Path) -> None:
    source = tmp_path / "legacy.duckdb"
    sqlite_path = tmp_path / "esdc.sqlite"
    conn = _create_duckdb(source)
    _insert(conn, _values(year=2022), row_id=77)
    conn.close()
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    with sqlite3.connect(sqlite_path) as sqlite_conn:
        sqlite_conn.execute("CREATE TABLE sentinel (id INTEGER, value TEXT)")
        sqlite_conn.execute("INSERT INTO sentinel VALUES (41, 'keep')")

    bootstrap_timeseries_source(duckdb_path=source, sqlite_path=sqlite_path, apply=True)

    assert hashlib.sha256(source.read_bytes()).hexdigest() == source_hash
    readonly = duckdb.connect(str(source), read_only=True)
    try:
        assert [
            (row_id, str(row_uuid))
            for row_id, row_uuid in readonly.execute(
                "SELECT id, uuid FROM project_timeseries"
            ).fetchall()
        ] == [(77, "00000000-0000-0000-0000-000000000077")]
    finally:
        readonly.close()
    with sqlite3.connect(sqlite_path) as sqlite_conn:
        assert sqlite_conn.execute("SELECT * FROM sentinel").fetchall() == [(41, "keep")]


def test_existing_canonical_refused(tmp_path: Path) -> None:
    source = tmp_path / "legacy.duckdb"
    sqlite_path = tmp_path / "esdc.sqlite"
    conn = _create_duckdb(source)
    _insert(conn, _values(year=2022), row_id=1)
    conn.close()
    with sqlite3.connect(sqlite_path) as sqlite_conn:
        ensure_timeseries_tables(sqlite_conn)
        sqlite_conn.execute(
            "INSERT INTO api_project_timeseries "
            "(report_year, row_json, row_sha256, origin, schema_version) "
            "VALUES (2020, '{}', 'x', 'api_json', 1)"
        )
        sqlite_conn.commit()
    before = sqlite_path.read_bytes()

    with pytest.raises(ValueError, match="canonical timeseries"):
        bootstrap_timeseries_source(duckdb_path=source, sqlite_path=sqlite_path, apply=True)

    assert sqlite_path.read_bytes() == before


@pytest.mark.parametrize("case", ["missing_table", "missing_column", "unsafe_type"])
def test_corrupt_or_incomplete_duckdb_refused(tmp_path: Path, case: str) -> None:
    source = tmp_path / "legacy.duckdb"
    if case == "missing_table":
        conn = duckdb.connect(str(source))
        conn.execute("CREATE TABLE unrelated (value INTEGER)")
    else:
        conn = _create_duckdb(source, missing="project_id" if case == "missing_column" else None)
        if case == "unsafe_type":
            conn.execute("ALTER TABLE project_timeseries ALTER COLUMN report_year TYPE BLOB")
    conn.close()

    with pytest.raises((ValueError, RuntimeError)):
        bootstrap_timeseries_source(duckdb_path=source, sqlite_path=tmp_path / "esdc.sqlite")


def test_source_changes_before_commit_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "legacy.duckdb"
    sqlite_path = tmp_path / "esdc.sqlite"
    conn = _create_duckdb(source)
    _insert(conn, _values(year=2021), row_id=1)
    conn.close()
    import esdc.timeseries_bootstrap as bootstrap

    changed = False

    original_fingerprint = bootstrap._file_fingerprint

    def change_source(path: Path) -> tuple[str, int, int]:
        nonlocal changed
        fingerprint = original_fingerprint(path)
        if not changed:
            changed = True
            with path.open("ab") as source_file:
                source_file.write(b"changed-after-inspection")
        return fingerprint

    monkeypatch.setattr(bootstrap, "_file_fingerprint", change_source)
    with pytest.raises(RuntimeError, match="source changed"):
        bootstrap_timeseries_source(duckdb_path=source, sqlite_path=sqlite_path, apply=True)
    if sqlite_path.exists():
        with sqlite3.connect(sqlite_path) as sqlite_conn:
            assert not _table_exists(sqlite_conn, "api_project_timeseries")
            assert not _table_exists(sqlite_conn, "api_snapshot")


def test_bootstrap_provenance_is_not_api_provenance(tmp_path: Path) -> None:
    source = tmp_path / "legacy.duckdb"
    sqlite_path = tmp_path / "esdc.sqlite"
    conn = _create_duckdb(source)
    _insert(conn, _values(year=2020), row_id=1)
    conn.close()

    report = bootstrap_timeseries_source(
        duckdb_path=source, sqlite_path=sqlite_path, apply=True
    )

    with sqlite3.connect(sqlite_path) as sqlite_conn:
        provenance = sqlite_conn.execute(
            "SELECT revision, origin, details_json FROM api_snapshot_provenance "
            "WHERE source_table = 'project_timeseries'"
        ).fetchone()
    assert provenance is not None
    revision, origin, details_json = provenance
    details = json.loads(details_json)
    assert revision
    assert origin == "legacy_duckdb"
    assert details["source_sha256"] == report.source_sha256
    assert details["source_types"]["tpf_oc"] == "DECIMAL(24,8)"
    assert set(details["source_types"]) == set(TIMESERIES_SOURCE_COLUMNS)
    assert "id" not in details["source_types"]
    assert "uuid" not in details["source_types"]
    assert details["normalized_source_sha256"]
    with sqlite3.connect(sqlite_path) as sqlite_conn:
        snapshot = get_timeseries_snapshot(sqlite_conn)
    assert snapshot is not None and snapshot.revision == revision
    assert details.get("api", False) is False


def test_cli_dry_run_apply_and_refusal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import esdc.commands.analytics as analytics
    import esdc.esdc as cli

    source = tmp_path / "legacy.duckdb"
    sqlite_path = tmp_path / "esdc.sqlite"
    conn = _create_duckdb(source)
    _insert(conn, _values(year=2020), row_id=1)
    conn.close()
    with sqlite3.connect(sqlite_path) as sqlite_conn:
        sqlite_conn.execute("CREATE TABLE sentinel (value TEXT)")
        sqlite_conn.execute("INSERT INTO sentinel VALUES ('unchanged')")
    before_dry_run = sqlite_path.read_bytes()

    monkeypatch.setattr(cli.Config, "init_config", lambda: None)
    monkeypatch.setattr(cli.Config, "get_db_file", lambda: source)
    monkeypatch.setattr(analytics, "get_esdc_sqlite_path", lambda: sqlite_path)
    runner = CliRunner()

    dry_result = runner.invoke(cli.app, ["bootstrap-timeseries-source"])
    assert dry_result.exit_code == 0, dry_result.output
    assert "Dry run: 1 legacy timeseries rows" in dry_result.output
    assert sqlite_path.read_bytes() == before_dry_run

    apply_result = runner.invoke(cli.app, ["bootstrap-timeseries-source", "--apply"])
    assert apply_result.exit_code == 0, apply_result.output
    assert "Applied: 1 legacy timeseries rows" in apply_result.output
    with sqlite3.connect(sqlite_path) as sqlite_conn:
        assert sqlite_conn.execute("SELECT * FROM sentinel").fetchall() == [
            ("unchanged",)
        ]
    committed = sqlite_path.read_bytes()

    refused_result = runner.invoke(cli.app, ["bootstrap-timeseries-source", "--apply"])
    assert refused_result.exit_code == 1
    assert "Bootstrap refused: canonical timeseries" in refused_result.output
    assert sqlite_path.read_bytes() == committed


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone() is not None
