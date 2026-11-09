"""Explicit, value-preserving upgrade of the POD economics column names.

The historical v2 schema fixture was frozen from the parent of the commit that
renamed the economics columns, so these tests exercise the real contract-1
layout instead of rebuilding an old schema from the current contract text.
"""

from __future__ import annotations

import importlib
import re
import sqlite3
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from tests.pod_registry.economics_fixture import (
    RECOVERY,
    build_registry,
    columns,
    connect,
    dump,
    renames,
    sentinels,
)

_build_registry = build_registry
_columns = columns
_connect = connect
_dump = dump
_renames = renames
_sentinels = sentinels


def _api() -> tuple[Any, Any, Any, Any]:
    try:
        upgrade = importlib.import_module("esdc.pod_registry.economics_upgrade")
        migrations = importlib.import_module("esdc.pod_registry.migrations")
        contract = importlib.import_module("esdc.pod_registry.contract")
        return (
            upgrade.inspect_economics_upgrade,
            upgrade.upgrade_economics_columns,
            migrations.MigrationRefusedError,
            contract.load_pod_contract,
        )
    except (ImportError, AttributeError) as exc:
        pytest.fail(f"POD economics upgrade API is unavailable: {exc}", pytrace=False)


def _master_identity(path: Path) -> list[tuple[Any, ...]]:
    conn = _connect(path)
    try:
        return [
            tuple(row)
            for row in conn.execute(
                "SELECT pod_id_seq, pod_uuid, pod_id, approval_date, superseded_by, "
                "institution_id, pod_type_id, rev_num, pod_order_no, is_latest "
                "FROM m_pod ORDER BY pod_id_seq"
            )
        ]
    finally:
        conn.close()


def _upgrade(path: Path) -> Any:
    _inspect, apply, _refused, _load = _api()
    conn = _connect(path)
    try:
        report = apply(conn)
        conn.commit()
    finally:
        conn.close()
    return report


def test_upgrade_preserves_all_24_economics_values(tmp_path: Path) -> None:
    inspect, apply, _refused, load_contract = _api()
    path = tmp_path / "old.sqlite"
    values = _build_registry(path)

    conn = _connect(path)
    try:
        report = inspect(conn)
        assert report.status == "upgrade_required"
        assert len(report.renames) == 24
        assert {tuple(item) for item in report.renames} == set(_renames())

        applied = apply(conn)
        assert applied.status == "current"
        conn.commit()

        assert inspect(conn).status == "current"
        contract = load_contract()
        for table, old, new in _renames():
            actual = _columns(conn, table)
            assert old not in actual
            assert actual == tuple(
                column.name for column in contract.tables[table].columns
            )
            row = conn.execute(
                f'SELECT typeof("{new}"), "{new}" FROM "{table}" WHERE rowid = 1'
            ).fetchone()
            expected = values[(table, new)]
            if expected is None:
                assert tuple(row) == ("null", None)
            else:
                assert tuple(row) == ("text", expected)
    finally:
        conn.close()


def test_upgrade_preserves_identities_links_and_generations(tmp_path: Path) -> None:
    path = tmp_path / "old.sqlite"
    _build_registry(path)
    identity_before = _master_identity(path)
    preserved_tables = (
        "m_project",
        "pod_project",
        "pod_document",
        "_pod_identity_ledger",
        "pod_read_model_snapshot",
        "pod_knowledge_state",
        "pod_read_model_diagnostic",
        "pod_registry_legacy_identity",
        "application_notes",
    )
    before = _dump(path, tables=preserved_tables)

    _upgrade(path)

    assert _master_identity(path) == identity_before
    assert _dump(path, tables=preserved_tables) == before
    conn = _connect(path)
    try:
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
        assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        state = conn.execute(
            "SELECT source_generation, knowledge_generation, contract_version, "
            "updated_by FROM pod_registry_state WHERE singleton_id = 1"
        ).fetchone()
        assert tuple(state) == (7, 3, 2, "FA:pod-economics-upgrade")
    finally:
        conn.close()


def test_upgrade_is_explicit_and_read_only_inspection_changes_nothing(
    tmp_path: Path,
) -> None:
    inspect, _apply, refused, _load = _api()
    path = tmp_path / "old.sqlite"
    _build_registry(path)
    before_bytes = path.read_bytes()

    conn = _connect(path)
    try:
        assert inspect(conn).status == "upgrade_required"
        assert "npv_pf_ctr_pod" in _columns(conn, "m_pod")
        assert conn.execute(
            "SELECT contract_version FROM pod_registry_state WHERE singleton_id = 1"
        ).fetchone()[0] == 1
    finally:
        conn.close()
    assert path.read_bytes() == before_bytes

    migrations = importlib.import_module("esdc.pod_registry.migrations")
    with pytest.raises(refused, match=re.escape(RECOVERY)):
        migrations.open_registry_v2_writer(path)

    assert path.read_bytes() == before_bytes
    conn = _connect(path)
    try:
        assert "npv_pf_ctr_pod" in _columns(conn, "m_pod")
        assert conn.execute(
            "SELECT contract_version FROM pod_registry_state WHERE singleton_id = 1"
        ).fetchone()[0] == 1
    finally:
        conn.close()


def test_already_renamed_contract_one_is_stamped_without_data_changes(
    tmp_path: Path,
) -> None:
    inspect, apply, _refused, _load = _api()
    path = tmp_path / "renamed.sqlite"
    values = _build_registry(path, renamed=True)
    before = _dump(path)

    conn = _connect(path)
    try:
        report = inspect(conn)
        assert report.status == "upgrade_required"
        assert report.renames == ()
        applied = apply(conn)
        assert applied.status == "current"
        assert applied.renames == ()
        conn.commit()
    finally:
        conn.close()

    after = _dump(path)
    assert set(after) == set(before)
    for table in ("m_pod", "pod_monitoring", "pod_monitoring_api"):
        assert len(after[table]) == len(before[table])
    conn = _connect(path)
    try:
        state = conn.execute(
            "SELECT contract_version, updated_by FROM pod_registry_state "
            "WHERE singleton_id = 1"
        ).fetchone()
        assert tuple(state) == (2, "FA:pod-economics-upgrade")
        for (table, new), expected in values.items():
            row = conn.execute(
                f'SELECT "{new}" FROM "{table}" WHERE rowid = 1'
            ).fetchone()
            assert row[0] == expected
    finally:
        conn.close()


def test_current_contract_two_upgrade_is_noop(tmp_path: Path) -> None:
    _inspect, apply, _refused, _load = _api()
    path = tmp_path / "current.sqlite"
    _build_registry(path)
    _upgrade(path)
    before = _dump(path)

    conn = _connect(path)
    try:
        report = apply(conn)
        assert report.status == "current"
        assert report.renames == ()
        conn.commit()
    finally:
        conn.close()

    assert _dump(path) == before


def _rename_one_column(conn: sqlite3.Connection) -> None:
    conn.execute("ALTER TABLE m_pod RENAME COLUMN npv_pf_ctr_pod TO npv_ctr_pod")


def _add_new_alongside_old(conn: sqlite3.Connection) -> None:
    conn.execute("ALTER TABLE m_pod ADD COLUMN npv_ctr_pod TEXT")


def _drop_expected_column(conn: sqlite3.Connection) -> None:
    conn.execute("ALTER TABLE m_pod DROP COLUMN pi_pf_ctr_pod")


def _add_extra_contract_column(conn: sqlite3.Connection) -> None:
    conn.execute("ALTER TABLE m_pod ADD COLUMN zzz_unexpected TEXT")


def _unsupported_contract_version(conn: sqlite3.Connection) -> None:
    conn.execute("UPDATE pod_registry_state SET contract_version = 99")


def _non_numeric_contract_version(conn: sqlite3.Connection) -> None:
    conn.execute("UPDATE pod_registry_state SET contract_version = 'not-a-number'")


def _wrong_dictionary(conn: sqlite3.Connection) -> None:
    conn.execute("UPDATE pod_registry_state SET dictionary_version = '9.9.9'")


def _missing_metadata(conn: sqlite3.Connection) -> None:
    conn.execute("DELETE FROM pod_registry_state")


def _missing_identity_guard(conn: sqlite3.Connection) -> None:
    conn.execute("DROP TRIGGER trg_m_pod_identity_immutable")


@pytest.mark.parametrize(
    "mutate",
    [
        _rename_one_column,
        _add_new_alongside_old,
        _drop_expected_column,
        _add_extra_contract_column,
        _unsupported_contract_version,
        _non_numeric_contract_version,
        _wrong_dictionary,
        _missing_metadata,
        _missing_identity_guard,
    ],
    ids=[
        "partially-renamed-table",
        "both-old-and-new-columns",
        "missing-column",
        "extra-contract-column",
        "unsupported-contract-version",
        "non-numeric-contract-version",
        "wrong-dictionary",
        "missing-metadata",
        "missing-identity-guard",
    ],
)
def test_mixed_or_unknown_shape_is_refused_atomically(
    tmp_path: Path, mutate: Callable[[sqlite3.Connection], None]
) -> None:
    inspect, apply, refused, _load = _api()
    path = tmp_path / "broken.sqlite"
    _build_registry(path)
    conn = _connect(path)
    try:
        mutate(conn)
        conn.commit()
        before = _dump(path)
        report = inspect(conn)
        assert report.status == "blocked"
        assert report.blockers
        with pytest.raises(refused):
            apply(conn)
        assert not conn.in_transaction
        assert _dump(path) == before
    finally:
        conn.close()


class _AlterFailingProxy:
    """Connection proxy that fails right after the first real rename."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        self.alters = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self._conn, name)

    @property
    def in_transaction(self) -> bool:
        return self._conn.in_transaction

    def execute(self, sql: str, parameters: tuple[Any, ...] = ()) -> Any:
        cursor = self._conn.execute(sql, parameters)
        if " RENAME COLUMN " in sql.upper():
            self.alters += 1
            if self.alters == 1:
                raise RuntimeError("injected rename failure")
        return cursor


def test_failure_after_one_rename_rolls_back_everything(tmp_path: Path) -> None:
    _inspect, apply, _refused, _load = _api()
    path = tmp_path / "old.sqlite"
    _build_registry(path)
    before = _dump(path)
    conn = _connect(path)
    proxy = _AlterFailingProxy(conn)
    try:
        with pytest.raises(RuntimeError, match="injected rename failure"):
            apply(proxy)
        assert proxy.alters == 1
        assert not conn.in_transaction
        assert _dump(path) == before
        assert "npv_pf_ctr_pod" in _columns(conn, "m_pod")
        assert conn.execute(
            "SELECT contract_version FROM pod_registry_state WHERE singleton_id = 1"
        ).fetchone()[0] == 1
    finally:
        conn.close()


def test_upgrade_joins_outer_transaction(tmp_path: Path) -> None:
    _inspect, apply, _refused, _load = _api()
    path = tmp_path / "old.sqlite"
    _build_registry(path)
    sqlite_runtime = importlib.import_module("esdc.sqlite_runtime")
    conn = sqlite3.connect(path)
    try:
        sqlite_runtime.configure_writer(conn)
        conn.execute("BEGIN IMMEDIATE")
        report = apply(conn)
        assert report.status == "current"
        assert conn.in_transaction
        assert "npv_ctr_pod" in _columns(conn, "m_pod")
        assert conn.execute(
            "SELECT contract_version FROM pod_registry_state WHERE singleton_id = 1"
        ).fetchone()[0] == 2

        conn.rollback()
        assert not conn.in_transaction
        assert "npv_pf_ctr_pod" in _columns(conn, "m_pod")
        assert "npv_ctr_pod" not in _columns(conn, "m_pod")
        assert conn.execute(
            "SELECT contract_version FROM pod_registry_state WHERE singleton_id = 1"
        ).fetchone()[0] == 1
    finally:
        conn.close()


def test_upgraded_registry_opens_and_serves_new_economics_names(
    tmp_path: Path,
) -> None:
    inspect, _apply, _refused, _load = _api()
    path = tmp_path / "old.sqlite"
    _build_registry(path)
    _upgrade(path)

    migrations = importlib.import_module("esdc.pod_registry.migrations")
    read_models = importlib.import_module("esdc.pod_registry.read_models")
    conn = migrations.open_registry_v2_writer(path)
    try:
        assert inspect(conn).status == "current"
        conn.execute("BEGIN IMMEDIATE")
        read_models.rebuild_pod_read_models(conn)
        conn.commit()
        row = conn.execute(
            "SELECT pod_id_seq, irr_ctr_pod, pv_ratio_ctr_pod, pi_ctr_pod "
            "FROM pod_monitoring_api WHERE pod_id_seq = 1"
        ).fetchone()
        assert tuple(row) == (1, "0.123456", None, "-2.500")
        assert conn.execute(
            "SELECT irr_ctr_pod FROM m_pod WHERE pod_id_seq = 1"
        ).fetchone()[0] == "0.123456"
    finally:
        conn.close()


def test_inspect_migration_reports_contract_one_upgrade_read_only(tmp_path: Path) -> None:
    path = tmp_path / "old.sqlite"
    _build_registry(path)
    before = path.read_bytes()
    migrations = importlib.import_module("esdc.pod_registry.migrations")
    conn = _connect(path)
    try:
        report = migrations.inspect_migration(conn)
    finally:
        conn.close()

    assert report.status == "blocked"
    assert report.economics_upgrade_required is True
    assert any(RECOVERY in blocker for blocker in report.blockers)
    assert path.read_bytes() == before


def test_operational_store_and_fetch_name_the_economics_upgrade(
    tmp_path: Path,
) -> None:
    path = tmp_path / "old.sqlite"
    _build_registry(path)
    migrations = importlib.import_module("esdc.pod_registry.migrations")
    store = importlib.import_module("esdc.pod_registry.store")
    fetch = importlib.import_module("esdc.fetch_publication")
    fetching = importlib.import_module("esdc.fetching")

    with pytest.raises(migrations.MigrationRefusedError) as store_error:
        store.get_sqlite_connection(path)
    assert "pod-upgrade-economics" in str(store_error.value)
    assert "explicit validated replacement" not in str(store_error.value)

    conn = _connect(path)
    try:
        with pytest.raises(fetching.FetchFailure) as fetch_error:
            fetch._pod_target_shape(conn)
    finally:
        conn.close()
    assert "pod-upgrade-economics" in str(fetch_error.value)
    assert "crosswalk" not in str(fetch_error.value).lower()
