"""Latest-only default bootstrap regressions for the operational store."""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path
from typing import Any

import pytest

import esdc.configs as configs
import esdc.sqlite_runtime as sqlite_runtime
from esdc.pod_registry import store
from esdc.pod_registry.contract import ColumnContract, load_pod_contract
from esdc.pod_registry.migrations import (
    MigrationRefusedError,
    _validate_existing_v2_registry,
)

RETIRED_POD_TABLES = frozenset({"project_pod", "pod_revision", "pod_value_case"})
RAW_SOURCE_TABLES = frozenset(
    {
        "api_project_resources",
        "api_snapshot",
        "api_project_timeseries",
        "api_snapshot_provenance",
    }
)
# Minimal old-shaped fixture SQL. This is test material only, never a production
# compatibility constant: the store no longer knows the superseded schema.
_OLD_SHAPED_FIXTURE_SQL = """
CREATE TABLE m_pod (
    id INTEGER PRIMARY KEY,
    pod_id TEXT NOT NULL UNIQUE,
    pod_name TEXT,
    approval_date TEXT,
    institution_code INTEGER,
    pod_type_code INTEGER,
    rev_num INTEGER,
    approval_seq INTEGER
);
CREATE TABLE project_pod (pod_id INTEGER, project_id TEXT);
CREATE TABLE pod_revision (successor_id TEXT, predecessor_id TEXT);
"""


def _patch_dirs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        configs.Config, "get_db_dir", classmethod(lambda cls: tmp_path)
    )


def _table_names(conn: sqlite3.Connection) -> set[str]:
    return {
        str(row[0])
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
    }


def _physical_shape(conn: sqlite3.Connection) -> list[tuple[str, str, str, str | None]]:
    return [
        (str(row[0]), str(row[1]), str(row[2]), row[3])
        for row in conn.execute(
            "SELECT type, name, tbl_name, sql FROM sqlite_master "
            "ORDER BY type, name"
        )
    ]


def _configured_connection(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    sqlite_runtime.configure_writer(conn)
    conn.execute("CREATE TABLE sentinel (value INTEGER)")
    conn.commit()
    return conn


def test_default_bootstrap_creates_only_the_current_schema(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A new operational database gets the single current schema, nothing else."""
    _patch_dirs(monkeypatch, tmp_path)
    conn = store.get_sqlite_connection()
    try:
        names = _table_names(conn)
        assert "m_project" in names
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        first_column = str(conn.execute("PRAGMA table_info(m_pod)").fetchone()[1])
        assert first_column == "pod_id_seq"
        assert not RETIRED_POD_TABLES & names
    finally:
        conn.close()


@pytest.mark.parametrize(
    "entry",
    [store.get_sqlite_connection, store.get_sqlite_writer_connection],
)
def test_default_bootstrap_follows_the_canonical_contract_and_sql_seed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, entry: Any
) -> None:
    """Ordered contract columns, three physical results, SQL seed, no mock rows."""
    _patch_dirs(monkeypatch, tmp_path)
    conn = entry()
    try:
        contract = load_pod_contract()
        for name, table in contract.tables.items():
            actual = tuple(
                str(row[1]) for row in conn.execute(f"PRAGMA table_info({name})")
            )
            assert actual == tuple(column.name for column in table.columns)
            # Default bootstrap creates state only; business rows come from an
            # explicit validated import, never from bootstrap samples.
            assert conn.execute(f"SELECT count(*) FROM {name}").fetchone()[0] == 0
        assert {
            name for name, table in contract.tables.items() if table.role == "output"
        } <= _table_names(conn)
        assert _table_names(conn) >= RAW_SOURCE_TABLES
        seeds = sorted(
            (int(row[0]), str(row[1]))
            for row in conn.execute("SELECT id, institution FROM r_institution")
        )
        assert seeds == [
            (1, "MESDM"),
            (2, "BPPKA"),
            (3, "BP Migas"),
            (4, "SKK Migas"),
            (5, "BPMA"),
        ]
        assert conn.execute("SELECT count(*) FROM r_pod_type").fetchone()[0] == 4
        assert conn.execute("SELECT count(*) FROM r_pod_stage").fetchone()[0] == 4
        state = conn.execute(
            "SELECT schema_version, reference_seed_version, source_generation "
            "FROM pod_registry_state WHERE singleton_id = 1"
        ).fetchone()
        assert (int(state[0]), int(state[1]), int(state[2])) == (2, 20261028, 0)
        before = _physical_shape(conn)
    finally:
        conn.close()

    reopened = entry()
    try:
        assert _physical_shape(reopened) == before
    finally:
        reopened.close()


def test_shared_raw_initializer_is_explicit_and_has_no_old_pod_ddl(tmp_path: Path) -> None:
    """Raw preparation uses explicit shared DDL, not parsing of a superseded one."""
    assert not hasattr(store, "_DDL")
    source = Path(store.__file__).read_text(encoding="utf-8")
    assert "executescript" not in source
    for constant in (store._PROJECT_RESOURCES_DDL, store._SNAPSHOT_DDL):
        assert constant.strip().startswith("CREATE TABLE IF NOT EXISTS api_")

    conn = sqlite3.connect(tmp_path / "raw-only.sqlite")
    try:
        store.ensure_snapshot_table(conn)
        conn.commit()
        assert _table_names(conn) == {"api_snapshot"}
        columns = tuple(
            str(row[1]) for row in conn.execute("PRAGMA table_info(api_snapshot)")
        )
        assert columns == (
            "source_table",
            "revision",
            "row_count",
            "source_sha256",
            "fetched_at",
            "publication_state",
            "invalid_geometry_count",
        )
    finally:
        conn.close()


def test_raw_source_initializer_runs_statementwise_without_committing(
    tmp_path: Path,
) -> None:
    """Raw tables join the caller transaction; nothing is committed by itself."""
    conn = _configured_connection(tmp_path / "raw-join.sqlite")
    conn.execute("BEGIN IMMEDIATE")
    try:
        store.ensure_raw_source_tables(conn)
        assert _table_names(conn) >= RAW_SOURCE_TABLES
        assert conn.in_transaction
        conn.rollback()
        assert _table_names(conn) == {"sentinel"}
    finally:
        conn.close()


@pytest.mark.parametrize(
    "entry",
    [store.get_sqlite_connection, store.get_sqlite_writer_connection],
)
def test_old_shaped_source_is_refused_without_touching_the_file(
    tmp_path: Path, entry: Any
) -> None:
    """An incompatible existing database is refused before any writer mutation."""
    path = tmp_path / "old-shaped.sqlite"
    fixture = sqlite3.connect(path)
    fixture.executescript(_OLD_SHAPED_FIXTURE_SQL)
    fixture.execute("INSERT INTO m_pod (id, pod_id, approval_seq) VALUES (41, 'old', 41)")
    fixture.commit()
    fixture.close()

    before_bytes = hashlib.sha256(path.read_bytes()).hexdigest()
    before_mtime = path.stat().st_mtime_ns

    with pytest.raises(MigrationRefusedError, match="replacement"):
        entry(path)

    assert hashlib.sha256(path.read_bytes()).hexdigest() == before_bytes
    assert path.stat().st_mtime_ns == before_mtime
    assert not Path(f"{path}-wal").exists()
    check = sqlite3.connect(path)
    try:
        assert _table_names(check) <= {"m_pod", "project_pod", "pod_revision"}
        assert check.execute("SELECT id, pod_id FROM m_pod").fetchall() == [(41, "old")]
    finally:
        check.close()


def test_unsupported_runtime_refuses_before_mkdir_or_connect(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sqlite_runtime.sqlite3, "sqlite_version_info", (3, 50, 6))
    connect_calls: list[object] = []

    def record_connect(*args: object, **kwargs: object) -> sqlite3.Connection:
        connect_calls.append((args, kwargs))
        raise AssertionError("SQLite connect must not run for an unsupported version")

    monkeypatch.setattr(store.sqlite3, "connect", record_connect)
    db_path = tmp_path / "not-created" / "esdc.sqlite"

    with pytest.raises(RuntimeError, match="3.50.6"):
        store.get_sqlite_connection(db_path)

    assert connect_calls == []
    assert not db_path.parent.exists()
    assert not db_path.exists()


@pytest.mark.parametrize("nested", [False, True])
def test_full_bootstrap_rolls_back_as_one_unit(
    tmp_path: Path, nested: bool
) -> None:
    """An injected failure in any phase leaves no partial POD/raw/TS objects."""
    conn = _configured_connection(tmp_path / "atomic.sqlite")
    before = _physical_shape(conn)

    def authorizer(
        action: int,
        arg1: str | None,
        arg2: str | None,
        dbname: str | None,
        origin: str | None,
    ) -> int:
        del arg2, dbname, origin
        if action == sqlite3.SQLITE_CREATE_TABLE and arg1 == "api_project_timeseries":
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK

    if nested:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute("INSERT INTO sentinel VALUES (9)")

    conn.set_authorizer(authorizer)
    try:
        with pytest.raises(sqlite3.DatabaseError):
            store.ensure_tables(conn)
    finally:
        conn.set_authorizer(None)

    try:
        assert conn.in_transaction is nested
        assert _physical_shape(conn) == before
        assert conn.execute("SELECT value FROM sentinel").fetchall() == (
            [(9,)] if nested else []
        )
        conn.rollback()
        assert conn.execute("SELECT count(*) FROM sentinel").fetchone()[0] == 0
    finally:
        conn.close()


def test_successful_bootstrap_joins_caller_transaction_without_committing(
    tmp_path: Path,
) -> None:
    """The caller owns the transaction; bootstrap work stays rollbackable."""
    conn = _configured_connection(tmp_path / "nested.sqlite")
    before = _physical_shape(conn)
    conn.execute("BEGIN IMMEDIATE")
    conn.execute("INSERT INTO sentinel VALUES (17)")
    try:
        store.ensure_tables(conn)
        assert conn.in_transaction
        assert "m_pod" in _table_names(conn)
        conn.rollback()
        assert _physical_shape(conn) == before
        assert conn.execute("SELECT count(*) FROM sentinel").fetchone()[0] == 0
    finally:
        conn.close()


def test_owned_bootstrap_commits_once(tmp_path: Path) -> None:
    """When bootstrap owns the transaction it commits exactly once and closes it."""
    conn = _configured_connection(tmp_path / "commit-once.sqlite")
    store.ensure_tables(conn)
    assert not conn.in_transaction
    reader = sqlite3.connect(
        f"{(tmp_path / 'commit-once.sqlite').resolve().as_uri()}?mode=ro", uri=True
    )
    try:
        assert "m_pod" in _table_names(reader)
    finally:
        reader.close()
    conn.close()


def test_readonly_rejects_an_existing_directory_without_creating_one(
    tmp_path: Path,
) -> None:
    path = tmp_path / "directory"
    path.mkdir()
    with pytest.raises((OSError, ValueError)):
        store.get_sqlite_readonly_connection(path)
    assert list(path.iterdir()) == []


def test_readonly_keeps_uri_escaping_and_never_creates_a_missing_file(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing?name with space" / "esdc.sqlite"
    assert store.get_sqlite_readonly_connection(missing) is None
    assert not missing.parent.exists()
    assert not missing.exists()

    escaped = tmp_path / "odd?name" / "esdc.sqlite"
    escaped.parent.mkdir()
    setup = sqlite3.connect(escaped)
    setup.execute("CREATE TABLE sample (value TEXT)")
    setup.execute("INSERT INTO sample VALUES ('stable')")
    setup.commit()
    setup.close()
    before = hashlib.sha256(escaped.read_bytes()).hexdigest()

    connection = store.get_sqlite_readonly_connection(escaped)
    assert connection is not None
    try:
        assert connection.execute("SELECT value FROM sample").fetchone()[0] == "stable"
    finally:
        connection.close()
    assert hashlib.sha256(escaped.read_bytes()).hexdigest() == before


def _canonical_storage_type(column: ColumnContract) -> str:
    """Canonical declared storage type derived from the packaged contract."""
    if column.data_type == "DOUBLE PRECISION":
        return "REAL"
    if column.data_type in {"INTEGER", "BIGINT", "SMALLINT", "BOOLEAN"}:
        return "INTEGER"
    return "TEXT"


def _declared_types(conn: sqlite3.Connection, table: str) -> dict[str, str]:
    return {
        str(row[1]): str(row[2]) for row in conn.execute(f"PRAGMA table_info({table})")
    }


def _drift_declared_type(path: Path, table: str, column: str, drifted_type: str) -> None:
    """Emulate a stored schema whose declared storage type drifted.

    This mutates only an owned disposable fixture; production code never repairs or
    accepts a drifted physical declaration.
    """
    conn = sqlite3.connect(path)
    try:
        current = _declared_types(conn, table)[column]
        stored = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?",
            (table,),
        ).fetchone()[0]
        conn.execute("PRAGMA writable_schema = ON")
        conn.execute(
            "UPDATE sqlite_master SET sql = ? WHERE type = 'table' AND name = ?",
            (stored.replace(f"{column} {current}", f"{column} {drifted_type}", 1), table),
        )
        conn.execute("PRAGMA writable_schema = OFF")
        conn.commit()
    finally:
        conn.close()


_DRIFT_CASES = [
    ("m_pod", "capital_drilling_pod", "NUMERIC", "TEXT"),
    ("pod_monitoring_api", "res_oil_2p_pod", "TEXT", "REAL"),
    ("m_pod", "institution_id", "TEXT", "INTEGER"),
    ("m_pod", "approval_date", "INTEGER", "TEXT"),
]


def test_canonical_storage_types_match_the_contract_for_all_seven_tables(
    tmp_path: Path,
) -> None:
    conn = store.get_sqlite_connection(tmp_path / "canonical.sqlite")
    try:
        for name, table in load_pod_contract().tables.items():
            declared = _declared_types(conn, name)
            for column in table.columns:
                assert declared[column.name] == _canonical_storage_type(column), (
                    f"{name}.{column.name} declares {declared[column.name]!r}"
                )
    finally:
        conn.close()


def test_exact_decimal_precision_survives_the_canonical_storage_type(
    tmp_path: Path,
) -> None:
    conn = store.get_sqlite_connection(tmp_path / "canonical.sqlite")
    try:
        conn.execute(
            "INSERT INTO _pod_identity_ledger (approval_year, approval_date, daily_slot, "
            "pod_order_no, pod_id_seq, pod_id, pod_uuid, institution_id, pod_type_id, "
            "rev_num, actor) VALUES (2026, '2026-01-01', 1, 1, 1, "
            "'PL-2026-0001-1-1-0', '00000000-0000-4000-8000-000000000001', 1, 1, 0, "
            "'FA:regression')"
        )
        conn.execute(
            "INSERT INTO m_pod (pod_id_seq, pod_uuid, approval_date, institution_id, "
            "pod_type_id, rev_num, pod_order_no, pod_id, pod_name, pod_letter_num, "
            "is_latest, capital_drilling_pod) VALUES (1, "
            "'00000000-0000-4000-8000-000000000001', '2026-01-01', 1, 1, 0, 1, "
            "'PL-2026-0001-1-1-0', 'Regression POD', 'R', 1, '12345678901234567.123')"
        )
        row = conn.execute(
            "SELECT typeof(capital_drilling_pod), capital_drilling_pod FROM m_pod"
        ).fetchone()
        assert row[0] == "text"
        assert row[1] == "12345678901234567.123"
    finally:
        conn.close()


@pytest.mark.parametrize(
    "table,column,drifted_type,expected_type",
    _DRIFT_CASES,
)
def test_declared_storage_drift_is_refused_with_context(
    tmp_path: Path, table: str, column: str, drifted_type: str, expected_type: str
) -> None:
    path = tmp_path / "drift.sqlite"
    conn = store.get_sqlite_connection(path)
    conn.close()
    _drift_declared_type(path, table, column, drifted_type)
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    before_mtime = path.stat().st_mtime_ns

    probe = store.get_sqlite_readonly_connection(path)
    assert probe is not None
    try:
        with pytest.raises(MigrationRefusedError) as excinfo:
            _validate_existing_v2_registry(probe)
        message = str(excinfo.value)
        assert table in message and column in message
        assert repr(expected_type) in message and repr(drifted_type) in message
    finally:
        probe.close()

    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
    assert path.stat().st_mtime_ns == before_mtime


def test_storage_drift_is_refused_by_the_default_writer_without_writer_transition(
    tmp_path: Path,
) -> None:
    path = tmp_path / "drift.sqlite"
    conn = store.get_sqlite_connection(path)
    conn.close()
    _drift_declared_type(path, "m_pod", "capital_drilling_pod", "NUMERIC")
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    before_mtime = path.stat().st_mtime_ns

    with pytest.raises(MigrationRefusedError, match="capital_drilling_pod"):
        store.get_sqlite_connection(path)

    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
    assert path.stat().st_mtime_ns == before_mtime


def test_direct_ensure_tables_refuses_before_writer_configuration(
    tmp_path: Path,
) -> None:
    """Read-only classification must precede the WAL writer transition."""
    path = tmp_path / "old-shaped-delete.sqlite"
    conn = sqlite3.connect(path)
    try:
        conn.executescript(_OLD_SHAPED_FIXTURE_SQL)
        conn.execute("INSERT INTO m_pod (id, pod_id, approval_seq) VALUES (7, 'old', 7)")
        conn.commit()
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "delete"

        before = hashlib.sha256(path.read_bytes()).hexdigest()
        before_mtime = path.stat().st_mtime_ns

        with pytest.raises(MigrationRefusedError, match="replacement"):
            store.ensure_tables(conn)

        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "delete"
        assert conn.execute("SELECT id, pod_id FROM m_pod").fetchall() == [(7, "old")]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == before
        assert path.stat().st_mtime_ns == before_mtime
    finally:
        conn.close()
