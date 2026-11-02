"""Current-source recovery, read-only paths, and truthful CLI outcomes."""

import sqlite3
from pathlib import Path

import pytest
from typer.testing import CliRunner

import esdc.esdc as cli
from esdc.configs import Config
from esdc.fetching import FetchFailure
from esdc.loaders import generate_pod_workbook_template
from esdc.pod_registry import publish
from esdc.pod_registry.store import ensure_raw_source_tables, get_sqlite_connection
from esdc.source_publication import (
    RECOVERY_COMMAND,
    committed_pod_source_generation,
    get_fetch_publication_status,
)


def test_selected_fetch_refuses_unowned_timeseries_before_source_commit(
    owned_paths, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    import duckdb

    from esdc.fetch_publication import publish_prepared_fetch
    from tests.pod_registry.test_fetch_prepare_v2 import _pod_transport
    from tests.pod_registry.test_fetch_v2 import _prepare

    target, cache = owned_paths
    transport = _pod_transport(monkeypatch)
    prepared = _prepare(cli, tmp_path / "spool", target, (2024,))
    assert transport.calls
    connection = duckdb.connect(str(cache))
    try:
        connection.execute(
            "CREATE TABLE project_timeseries (project_id VARCHAR, report_year INTEGER)"
        )
        connection.execute(
            "INSERT INTO project_timeseries VALUES ('unobserved-2019', 2019)"
        )
    finally:
        connection.close()
    with pytest.raises(
        FetchFailure, match="canonical project_timeseries baseline is missing"
    ) as raised:
        publish_prepared_fetch(
            prepared, sqlite_path=target, duckdb_path=cache, reindex=False
        )
    assert raised.value.phase == "preflight"
    assert raised.value.table == "project_timeseries"
    assert not raised.value.sqlite_committed
    assert not target.exists()
    connection = duckdb.connect(str(cache), read_only=True)
    try:
        assert connection.execute("SELECT * FROM project_timeseries").fetchall() == [
            ("unobserved-2019", 2019)
        ]
    finally:
        connection.close()


@pytest.fixture()
def owned_paths(isolated_config: Path, monkeypatch: pytest.MonkeyPatch):
    db_dir = isolated_config / "db"
    cache = isolated_config / "cache.duckdb"
    monkeypatch.setattr(Config, "get_db_dir", classmethod(lambda cls: db_dir))
    monkeypatch.setattr(Config, "get_db_file", classmethod(lambda cls: cache))
    monkeypatch.setattr(
        Config, "get_cache_dir", classmethod(lambda cls: isolated_config / "cache")
    )
    monkeypatch.setattr(
        Config, "get_credentials", classmethod(lambda cls: ("mock", "mock"))
    )
    return db_dir / "esdc.sqlite", cache


def test_generation_reader_escapes_special_path_without_creating_files(owned_paths):
    target, _ = owned_paths
    target = target.with_name("POD # é.sqlite")
    conn = get_sqlite_connection(target)
    conn.close()
    before = {path.name for path in target.parent.iterdir()}
    before_data = target.read_bytes()
    assert committed_pod_source_generation(target) == 0
    # WAL readers may create coordination sidecars, but no other database file.
    allowed = before | {target.name + "-wal", target.name + "-shm"}
    assert {path.name for path in target.parent.iterdir()} <= allowed
    assert target.read_bytes() == before_data


@pytest.mark.parametrize("raw_only", [False, True])
def test_uninitialized_pod_is_absent_without_fabricated_recovery(owned_paths, raw_only):
    target, cache = owned_paths
    if raw_only:
        target.parent.mkdir(parents=True)
        conn = sqlite3.connect(target)
        try:
            ensure_raw_source_tables(conn)
            conn.commit()
        finally:
            conn.close()
    before = target.read_bytes() if target.exists() else None
    report = get_fetch_publication_status(sqlite_path=target, duckdb_path=cache)
    components = {part.name: part for part in report.components}
    assert components["pod_projection"].status == "absent"
    assert not report.fresh
    assert report.recovery_command is None
    assert (target.read_bytes() if target.exists() else None) == before
    assert not cache.exists()


def test_fetch_prints_recovery_when_raw_pair_committed_but_pod_pending(
    owned_paths, monkeypatch: pytest.MonkeyPatch
):
    target, _ = owned_paths
    conn = get_sqlite_connection(target)
    conn.close()

    def fail(**kwargs):
        raise FetchFailure(
            "POD publication pending after pair commit",
            phase="post_commit",
            sqlite_committed=True,
            duckdb_committed=True,
            recovery_command=RECOVERY_COMMAND,
        )

    monkeypatch.setattr(cli, "load_esdc_data", fail)
    result = CliRunner().invoke(cli.app, ["fetch"])
    assert result.exit_code == 1
    assert "SQLite source committed at generation 0" in result.output
    assert RECOVERY_COMMAND in result.output


def test_load_projection_failure_prints_supported_sqlite_only_recovery(
    owned_paths, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    target, _ = owned_paths
    workbook = tmp_path / "source.xlsx"
    generate_pod_workbook_template(output_path=workbook)

    def fail(*args, **kwargs):
        raise RuntimeError("injected projection failure")

    monkeypatch.setattr(publish, "publish_pod_registry", fail)
    result = CliRunner().invoke(
        cli.app, ["load", "--from-excel", str(workbook), "--schema-pod"]
    )
    assert result.exit_code == 1
    assert RECOVERY_COMMAND in result.output
    conn = sqlite3.connect(target.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        assert conn.execute(
            "SELECT source_generation FROM pod_registry_state"
        ).fetchone() == (1,)
    finally:
        conn.close()
