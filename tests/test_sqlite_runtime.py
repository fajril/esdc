"""Runtime policy, WAL, and read-only SQLite regressions."""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path
from typing import Any, cast

import pytest

import esdc.sqlite_runtime as sqlite_runtime
from esdc.pod_registry import store as pod_store
from esdc.pod_registry.store import (
    get_sqlite_connection,
    get_sqlite_readonly_connection,
)


@pytest.mark.parametrize(
    "version",
    [
        (3, 44, 6),
        (3, 44, 9),
        (3, 50, 7),
        (3, 51, 3),
        (3, 54, 1),
        (4, 0, 0),
    ],
)
def test_supported_sqlite_patch_boundaries(version: tuple[int, int, int]) -> None:
    assert sqlite_runtime.is_wal_runtime_supported(version)


@pytest.mark.parametrize(
    "version",
    [
        (3, 44, 5),
        (3, 45, 0),
        (3, 49, 2),
        (3, 50, 4),
        (3, 50, 6),
        (3, 51, 2),
    ],
)
def test_unsupported_sqlite_patch_boundaries(version: tuple[int, int, int]) -> None:
    assert not sqlite_runtime.is_wal_runtime_supported(version)


def test_writer_gate_rejects_before_directory_or_connect(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sqlite_runtime.sqlite3, "sqlite_version_info", (3, 50, 6))
    connect_calls: list[object] = []

    def record_connect(*args: object, **kwargs: object) -> sqlite3.Connection:
        connect_calls.append((args, kwargs))
        raise AssertionError("SQLite connect must not run for an unsupported version")

    monkeypatch.setattr(pod_store.sqlite3, "connect", record_connect)
    db_path = tmp_path / "not-created" / "esdc.sqlite"

    with pytest.raises(RuntimeError, match="3.50.6"):
        get_sqlite_connection(db_path)

    assert connect_calls == []
    assert not db_path.parent.exists()
    assert not db_path.exists()


def test_actual_runtime_gate_matches_its_sqlite_library(tmp_path: Path) -> None:
    db_path = tmp_path / "nested" / "esdc.sqlite"
    supported = sqlite_runtime.is_wal_runtime_supported(sqlite3.sqlite_version_info)

    if supported:
        connection = get_sqlite_connection(db_path)
        connection.close()
        assert db_path.is_file()
    else:
        with pytest.raises(RuntimeError, match="SQLite"):
            get_sqlite_connection(db_path)
        assert not db_path.parent.exists()
        assert not db_path.exists()


def test_configure_writer_sets_and_checks_connection_policy(tmp_path: Path) -> None:
    connection = sqlite3.connect(tmp_path / "writer.sqlite")
    try:
        sqlite_runtime.configure_writer(connection)

        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert connection.execute("PRAGMA busy_timeout").fetchone()[0] == 5000
        assert connection.execute("PRAGMA synchronous").fetchone()[0] == 2
    finally:
        connection.close()


def test_configure_writer_rejects_an_active_transaction_without_committing(
    tmp_path: Path,
) -> None:
    connection = sqlite3.connect(tmp_path / "transaction.sqlite")
    try:
        connection.execute("CREATE TABLE sample (value TEXT)")
        connection.execute("BEGIN")
        connection.execute("INSERT INTO sample VALUES ('uncommitted')")

        with pytest.raises(RuntimeError, match="transaction"):
            sqlite_runtime.configure_writer(connection)

        assert connection.in_transaction
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "delete"
        connection.rollback()
        assert connection.execute("SELECT COUNT(*) FROM sample").fetchone()[0] == 0
    finally:
        connection.close()


def test_configure_writer_rejects_a_database_that_refuses_wal() -> None:
    connection = sqlite3.connect(":memory:")
    try:
        with pytest.raises(RuntimeError, match="WAL"):
            sqlite_runtime.configure_writer(connection)
    finally:
        connection.close()


def test_wal_reader_keeps_snapshot_while_second_connection_commits(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "snapshot.sqlite"
    first_writer = get_sqlite_connection(db_path)
    reader: sqlite3.Connection | None = None
    second_writer: sqlite3.Connection | None = None
    try:
        first_writer.execute("CREATE TABLE sample (value TEXT NOT NULL)")
        first_writer.execute("INSERT INTO sample VALUES ('old')")
        first_writer.commit()

        reader = sqlite3.connect(
            f"{db_path.resolve().as_uri()}?mode=ro", uri=True, isolation_level=None
        )
        reader.execute("BEGIN")
        assert reader.execute("SELECT value FROM sample").fetchone()[0] == "old"

        second_writer = get_sqlite_connection(db_path)
        second_writer.execute("BEGIN IMMEDIATE")
        second_writer.execute("UPDATE sample SET value='new'")
        second_writer.commit()

        assert reader.execute("SELECT value FROM sample").fetchone()[0] == "old"
        reader.commit()
        assert reader.execute("SELECT value FROM sample").fetchone()[0] == "new"
    finally:
        if second_writer is not None:
            if second_writer.in_transaction:
                second_writer.rollback()
            second_writer.close()
        if reader is not None:
            if reader.in_transaction:
                reader.rollback()
            reader.close()
        first_writer.close()


def test_second_writer_obeys_busy_timeout(tmp_path: Path) -> None:
    db_path = tmp_path / "busy.sqlite"
    first = get_sqlite_connection(db_path)
    second: sqlite3.Connection | None = None
    try:
        first.execute("CREATE TABLE sample (value INTEGER)")
        first.commit()
        second = get_sqlite_connection(db_path)
        assert first.execute("PRAGMA busy_timeout").fetchone()[0] == 5000
        assert second.execute("PRAGMA busy_timeout").fetchone()[0] == 5000

        first.execute("BEGIN IMMEDIATE")
        second.execute("PRAGMA busy_timeout = 25")
        with pytest.raises(sqlite3.OperationalError, match="locked"):
            second.execute("BEGIN IMMEDIATE")
        first.rollback()
    finally:
        if second is not None:
            if second.in_transaction:
                second.rollback()
            second.close()
        first.close()


def test_spatial_writer_gate_runs_before_connect(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from esdc.spatial_layers import store as spatial_store

    monkeypatch.setattr(sqlite_runtime.sqlite3, "sqlite_version_info", (3, 50, 6))
    connect_calls: list[object] = []

    def record_connect(*args: object, **kwargs: object) -> sqlite3.Connection:
        connect_calls.append((args, kwargs))
        raise AssertionError("SQLite connect must not run for an unsupported version")

    monkeypatch.setattr(spatial_store.sqlite3, "connect", record_connect)
    db_path = tmp_path / "spatial.sqlite"

    with pytest.raises(RuntimeError, match="3.50.6"):
        spatial_store.get_sqlite_connection(db_path)

    assert connect_calls == []
    assert not db_path.exists()


def test_analytics_writer_gate_runs_before_connect(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from esdc import analytics_rebuild

    monkeypatch.setattr(sqlite_runtime.sqlite3, "sqlite_version_info", (3, 50, 6))
    connect_calls: list[object] = []

    def record_connect(*args: object, **kwargs: object) -> sqlite3.Connection:
        connect_calls.append((args, kwargs))
        raise AssertionError("SQLite connect must not run for an unsupported version")

    monkeypatch.setattr(analytics_rebuild.sqlite3, "connect", record_connect)
    db_path = tmp_path / "nested" / "esdc.sqlite"

    with pytest.raises(RuntimeError, match="3.50.6"):
        analytics_rebuild._mark_sources(db_path, cast(Any, object()), "published")

    assert connect_calls == []
    assert not db_path.parent.exists()
    assert not db_path.exists()


def _create_legacy_duckdb_source(path: Path) -> None:
    import duckdb

    from esdc.timeseries_snapshot import (
        TIMESERIES_SOURCE_COLUMNS,
        TIMESERIES_SOURCE_TYPES,
    )

    definitions = []
    for column in TIMESERIES_SOURCE_COLUMNS:
        if column == "tpf_oc":
            sql_type = "DECIMAL(24,8)"
        else:
            sql_type = {"TEXT": "VARCHAR", "INTEGER": "INTEGER", "REAL": "DOUBLE"}[
                TIMESERIES_SOURCE_TYPES[column]
            ]
        definitions.append(f'"{column}" {sql_type}')
    columns = (*TIMESERIES_SOURCE_COLUMNS, "id", "uuid")
    definitions.extend(("id BIGINT", "uuid UUID"))
    connection = duckdb.connect(str(path))
    try:
        connection.execute(
            "CREATE TABLE project_timeseries (" + ", ".join(definitions) + ")"
        )
        values = cast(dict[str, object], dict.fromkeys(TIMESERIES_SOURCE_COLUMNS))
        values.update(
            {"report_year": 2024, "project_id": "legacy-project", "tpf_oc": 1}
        )
        parameters = [values[column] for column in TIMESERIES_SOURCE_COLUMNS]
        parameters.extend((1, "00000000-0000-0000-0000-000000000001"))
        placeholders = ",".join("?" for _ in columns)
        quoted = ",".join(f'"{column}"' for column in columns)
        connection.execute(
            f"INSERT INTO project_timeseries ({quoted}) VALUES ({placeholders})",
            parameters,
        )
    finally:
        connection.close()


def test_timeseries_apply_gate_runs_before_output_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from esdc.timeseries_bootstrap import bootstrap_timeseries_source

    source_path = tmp_path / "legacy.duckdb"
    _create_legacy_duckdb_source(source_path)
    sqlite_path = tmp_path / "not-created" / "esdc.sqlite"
    monkeypatch.setattr(sqlite_runtime.sqlite3, "sqlite_version_info", (3, 50, 6))

    with pytest.raises(RuntimeError, match="3.50.6"):
        bootstrap_timeseries_source(
            duckdb_path=source_path, sqlite_path=sqlite_path, apply=True
        )

    assert not sqlite_path.parent.exists()
    assert not sqlite_path.exists()


def test_readonly_connection_preserves_file_and_rejects_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = tmp_path / "readonly.sqlite"
    with sqlite3.connect(db_path) as setup:
        setup.execute("CREATE TABLE sample (value TEXT)")
        setup.execute("INSERT INTO sample VALUES ('stable')")
        setup.execute("PRAGMA journal_mode = DELETE")
    before = hashlib.sha256(db_path.read_bytes()).hexdigest()
    monkeypatch.setattr(sqlite_runtime.sqlite3, "sqlite_version_info", (3, 50, 4))

    connection = get_sqlite_readonly_connection(db_path)

    assert connection is not None
    try:
        assert connection.execute("SELECT value FROM sample").fetchone()[0] == "stable"
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "delete"
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            connection.execute("CREATE TABLE forbidden (value INTEGER)")
    finally:
        connection.close()
    assert hashlib.sha256(db_path.read_bytes()).hexdigest() == before
    assert not Path(f"{db_path}-wal").exists()


def test_readonly_missing_database_returns_none_without_creating_path(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "missing" / "esdc.sqlite"

    assert get_sqlite_readonly_connection(db_path) is None

    assert not db_path.parent.exists()
    assert not db_path.exists()


def test_readonly_portal_and_status_paths_do_not_initialize_sqlite(
    tmp_path, monkeypatch
):
    db_path = tmp_path / "missing" / "esdc.sqlite"
    from esdc.portal import data

    monkeypatch.setattr(data, "get_esdc_sqlite_path", lambda: db_path)
    assert data.read_table_page("m_pod")["rows"] == []
    assert all(not rows for rows in data.read_reference_options().values())
    assert not db_path.parent.exists()


def test_readonly_portal_paths_leave_existing_empty_database_unchanged(tmp_path):
    from esdc.portal.data import read_reference_options, read_table_page

    db_path = tmp_path / "empty.sqlite"
    sqlite3.connect(db_path).close()
    before = db_path.read_bytes()
    for reader in (
        lambda: read_table_page("m_pod", sqlite_path=db_path),
        lambda: read_reference_options(sqlite_path=db_path),
    ):
        with pytest.raises(ValueError, match="Current POD registry"):
            reader()
    assert db_path.read_bytes() == before


def test_portal_queries_reject_unknown_columns_before_opening_a_database(tmp_path):
    from esdc.portal.data import read_table_page

    db_path = tmp_path / "missing" / "esdc.sqlite"
    with pytest.raises(ValueError):
        read_table_page("m_pod", sqlite_path=db_path, sort="not_a_column")
    assert not db_path.parent.exists()


def test_publisher_preflight_refuses_a_missing_source_table_instead_of_swallowing_it(
    tmp_path: Path,
) -> None:
    """A source that cannot satisfy the contract is refused, not read as empty.

    The retired reader treated `no such table` as an empty frame, so a publish
    could go ahead and write an empty projection over a good one. The single
    current path names the tables the source cannot satisfy, and it does that
    read-only: the source is not created, initialized, or upgraded to a WAL
    writer, and a query for a table it does not have still raises.
    """
    from esdc.pod_registry.publish import source_preflight_error

    path = tmp_path / "partial.sqlite"
    source = sqlite3.connect(path)
    source.execute("CREATE TABLE present_table (value TEXT)")
    source.commit()
    source.close()

    error = source_preflight_error(path)
    assert error is not None
    assert "missing tables" in error
    assert "m_pod" in error

    check = sqlite3.connect(path)
    try:
        assert [
            row[0]
            for row in check.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        ] == ["present_table"]
        with pytest.raises(sqlite3.OperationalError, match="no such table"):
            check.execute("SELECT value FROM absent_table")
        with pytest.raises(sqlite3.OperationalError, match="no such column"):
            check.execute("SELECT missing FROM present_table")
    finally:
        check.close()


def test_additional_readers_leave_missing_and_empty_sources_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import duckdb

    from esdc.chat import tools as chat_tools
    from esdc.esdc import proposals
    from esdc.pod_registry.publish import source_preflight_error
    from esdc.spatial_layers import publisher as spatial_publisher

    monkeypatch.setattr(sqlite_runtime.sqlite3, "sqlite_version_info", (3, 50, 4))
    missing_path = tmp_path / "missing" / "esdc.sqlite"
    empty_path = tmp_path / "empty" / "esdc.sqlite"
    empty_path.parent.mkdir()
    sqlite3.connect(empty_path).close()
    empty_hash = hashlib.sha256(empty_path.read_bytes()).hexdigest()

    class _DossierConnection:
        def close(self) -> None:
            pass

    monkeypatch.setattr(pod_store, "get_esdc_sqlite_path", lambda: missing_path)
    monkeypatch.setattr(chat_tools, "get_db_connection", _DossierConnection)
    monkeypatch.setattr("esdc.knowledge.dossier.get_dossier", lambda *_: None)

    def duckdb_writer(path: Path, read_only: bool = False) -> duckdb.DuckDBPyConnection:
        del read_only
        return duckdb.connect(str(path))

    monkeypatch.setattr(spatial_publisher, "get_duckdb_connection", duckdb_writer)

    for source_path in (missing_path, empty_path):
        monkeypatch.setattr(
            pod_store,
            "get_esdc_sqlite_path",
            lambda source_path=source_path: source_path,
        )
        projection = duckdb.connect(":memory:")
        try:
            projection.execute("CREATE TABLE preserved (value INTEGER)")
            projection.execute("INSERT INTO preserved VALUES (42)")
            # The one current path refuses a source that cannot satisfy the
            # contract before it mutates anything, so a caller's existing
            # projection survives an unusable source.
            assert source_preflight_error(source_path) is not None
            assert projection.execute("SELECT value FROM preserved").fetchall() == [
                (42,)
            ]
        finally:
            projection.close()

        spatial_result = spatial_publisher.publish_spatial_layers(
            tmp_path / "spatial.duckdb", source_path
        )
        assert spatial_result.layer_count == 0
        assert spatial_result.feature_count == 0

        assert chat_tools._get_knowledge_context("Project", "P-1") == (None, [])
        proposals()
        assert "No pending proposals." in capsys.readouterr().out

    assert not missing_path.parent.exists()
    assert not missing_path.exists()
    assert hashlib.sha256(empty_path.read_bytes()).hexdigest() == empty_hash
    check = sqlite3.connect(f"{empty_path.resolve().as_uri()}?mode=ro", uri=True)
    try:
        tables = check.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    finally:
        check.close()
    assert tables == []
