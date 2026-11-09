"""Explicit, value-preserving upgrade of the POD economics column names.

The historical v2 schema fixture was frozen from the parent of the commit that
renamed the economics columns, so these tests exercise the real contract-1
layout instead of rebuilding an old schema from the current contract text.
"""

from __future__ import annotations

import importlib
import importlib.resources
import re
import sqlite3
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

_FIXTURE = Path(__file__).with_name("fixtures") / "pre_economics_rename_v2.sql"
_PREFIXES = (
    ("npv_pf_ctr_", "npv_ctr_"),
    ("irr_pf_ctr_", "irr_ctr_"),
    ("pv_ratio_pf_ctr_", "pv_ratio_ctr_"),
    ("pi_pf_ctr_", "pi_ctr_"),
)
_SUFFIXES = {
    "m_pod": ("pod",),
    "pod_monitoring": ("real", "otlk"),
    "pod_monitoring_api": ("pod", "real", "otlk"),
}
RECOVERY = (
    "Run 'esdc pod-upgrade-economics --database <path>' "
    "to inspect the required upgrade."
)


def _renames() -> tuple[tuple[str, str, str], ...]:
    return tuple(
        (table, old_prefix + suffix, new_prefix + suffix)
        for table, suffixes in _SUFFIXES.items()
        for suffix in suffixes
        for old_prefix, new_prefix in _PREFIXES
    )


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


def _schema_sql() -> str:
    return _FIXTURE.read_text(encoding="utf-8")


def _seed_sql() -> str:
    return (
        importlib.resources.files("esdc")
        .joinpath("sql")
        .joinpath("pod_reference_seed.sql")
        .read_text(encoding="utf-8")
    )


def _quote(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _old_names() -> dict[tuple[str, str], str]:
    return {(table, new): old for table, old, new in _renames()}


def _sentinels() -> dict[tuple[str, str], str | None]:
    values: dict[tuple[str, str], str | None] = {}
    for index, (table, _old, new) in enumerate(_renames(), start=1):
        values[(table, new)] = f"{100 + index}.{index:03d}"
    values[("m_pod", "npv_ctr_pod")] = "123.456"
    values[("m_pod", "irr_ctr_pod")] = "0.123456"
    values[("m_pod", "pv_ratio_ctr_pod")] = None
    values[("m_pod", "pi_ctr_pod")] = "-2.500"
    values[("pod_monitoring", "irr_ctr_real")] = "0.000"
    values[("pod_monitoring", "pv_ratio_ctr_real")] = None
    values[("pod_monitoring_api", "npv_ctr_otlk")] = "0"
    return values


def _columns(conn: sqlite3.Connection, table: str) -> tuple[str, ...]:
    return tuple(
        str(row[1]) for row in conn.execute(f"PRAGMA table_info({_quote(table)})")
    )


def _connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _insert(
    conn: sqlite3.Connection,
    table: str,
    values: dict[str, Any],
    *,
    renamed: bool,
) -> None:
    names = dict(values)
    if not renamed:
        mapping = _old_names()
        for (owner, new), old in mapping.items():
            if owner == table and new in names:
                names[old] = names.pop(new)
    columns = ", ".join(_quote(name) for name in names)
    placeholders = ", ".join("?" for _ in names)
    conn.execute(
        f"INSERT INTO {_quote(table)} ({columns}) VALUES ({placeholders})",
        tuple(names.values()),
    )


def _build_registry(
    path: Path,
    *,
    contract_version: int = 1,
    renamed: bool = False,
    metadata: dict[str, Any] | None = None,
) -> dict[tuple[str, str], str | None]:
    """Create a complete, valid contract-1 registry in either known layout."""
    sentinels = _sentinels()
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(_schema_sql())
    if renamed:
        for table, old, new in _renames():
            conn.execute(
                f"ALTER TABLE {_quote(table)} "
                f"RENAME COLUMN {_quote(old)} TO {_quote(new)}"
            )
    conn.executescript(_seed_sql())

    pod_uuid = "11111111-1111-4111-8111-111111111111"
    pod_id = "PL-2024-0001-1-1-0"
    conn.execute(
        "INSERT INTO _pod_identity_ledger (approval_year, approval_date, daily_slot, "
        "pod_order_no, pod_id_seq, pod_id, pod_uuid, institution_id, pod_type_id, "
        "rev_num, actor, issued_at) VALUES (2024, '2024-01-01', 1, 1, 1, ?, ?, 1, 1, "
        "0, 'FA:test-ledger', '2026-10-27T08:00:00.000Z')",
        (pod_id, pod_uuid),
    )
    master: dict[str, Any] = {
        "pod_id_seq": 1,
        "pod_uuid": pod_uuid,
        "approval_date": "2024-01-01",
        "superseded_by": None,
        "institution_id": 1,
        "pod_type_id": 1,
        "rev_num": 0,
        "pod_order_no": 1,
        "pod_id": pod_id,
        "pod_name": "Synthetic POD",
        "pod_letter_num": "Letter-1",
        "is_latest": 1,
    }
    master.update(
        {new: value for (table, new), value in sentinels.items() if table == "m_pod"}
    )
    _insert(conn, "m_pod", master, renamed=renamed)

    monitoring: dict[str, Any] = {
        "monpod_uuid": "22222222-2222-4222-8222-222222222222",
        "pod_id_seq": 1,
        "pod_id": pod_id,
        "reported_date": "2024-06-30",
        "is_completed": 0,
        "pod_stage_id": 1,
    }
    monitoring.update(
        {
            new: value
            for (table, new), value in sentinels.items()
            if table == "pod_monitoring"
        }
    )
    _insert(conn, "pod_monitoring", monitoring, renamed=renamed)

    api: dict[str, Any] = {"pod_id_seq": 1, "reported_date": "2024-06-30"}
    api.update(
        {
            new: value
            for (table, new), value in sentinels.items()
            if table == "pod_monitoring_api"
        }
    )
    _insert(conn, "pod_monitoring_api", api, renamed=renamed)

    conn.execute(
        "INSERT INTO m_project (project_id, project_name, is_active) "
        "VALUES ('P-1', 'Synthetic project', 1)"
    )
    conn.execute(
        "INSERT INTO pod_project (pod_project_uuid, pod_id_seq, project_id, "
        "is_main_project, is_auto_inferred) VALUES (?, 1, 'P-1', 1, 0)",
        ("33333333-3333-4333-8333-333333333333",),
    )
    conn.execute("INSERT INTO pod_document (pod_id_seq, doc_id) VALUES (1, 'DOC-001')")
    conn.execute(
        "INSERT INTO pod_registry_legacy_identity (legacy_pod_id, legacy_pod_id_value, "
        "target_pod_id_seq, target_pod_id, target_pod_uuid, reconciled_by) "
        "VALUES (900, 'LEGACY-900', 1, ?, ?, 'FA:test')",
        (pod_id, pod_uuid),
    )
    conn.execute(
        "INSERT INTO pod_read_model_snapshot (singleton_id, source_generation, "
        "resource_revision, resource_sha256, timeseries_revision, timeseries_sha256) "
        "VALUES (1, 7, 'resource-r1', 'resource-sha', 'timeseries-r1', 'timeseries-sha')"
    )
    conn.execute(
        "INSERT INTO pod_knowledge_state (singleton_id, basis_sha256, source_generation) "
        "VALUES (1, 'basis-sha', 7)"
    )
    conn.execute(
        "INSERT INTO pod_read_model_diagnostic (diagnostic_id, source_generation, code, "
        "message) VALUES (1, 7, 'synthetic', 'kept')"
    )
    conn.execute("CREATE TABLE application_notes (id INTEGER PRIMARY KEY, note TEXT)")
    conn.execute("INSERT INTO application_notes (id, note) VALUES (1, 'keep me')")
    state = {
        "schema_version": 2,
        "contract_id": "esdc.pod",
        "contract_version": contract_version,
        "dictionary_version": "0.4.0",
        "reference_seed_version": 20261028,
        "source_generation": 7,
        "knowledge_generation": 3,
        "updated_by": "FA:test-initialize",
    }
    if metadata is not None:
        state.update(metadata)
    conn.execute(
        "INSERT INTO pod_registry_state (singleton_id, schema_version, contract_id, "
        "contract_version, dictionary_version, reference_seed_version, "
        "source_generation, knowledge_generation, updated_by) "
        "VALUES (1, :schema_version, :contract_id, :contract_version, "
        ":dictionary_version, :reference_seed_version, :source_generation, "
        ":knowledge_generation, :updated_by)",
        state,
    )
    conn.commit()
    conn.close()
    return sentinels


def _dump(path: Path, *, tables: tuple[str, ...] | None = None) -> dict[str, Any]:
    conn = _connect(path)
    try:
        names = tables or tuple(
            str(row[0])
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' "
                "AND name NOT LIKE 'sqlite_%' ORDER BY name"
            )
        )
        return {
            table: tuple(
                sorted(
                    (tuple(row) for row in conn.execute(f"SELECT * FROM {_quote(table)}")),
                    key=repr,
                )
            )
            for table in names
        }
    finally:
        conn.close()


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
    inspect, apply, _refused, _load = _api()
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
    sentinels = _build_registry(path)

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
            columns = _columns(conn, table)
            assert old not in columns
            assert columns == tuple(
                column.name for column in contract.tables[table].columns
            )
            row = conn.execute(
                f"SELECT typeof({_quote(new)}), {_quote(new)} FROM {_quote(table)} "
                "WHERE rowid = 1"
            ).fetchone()
            expected = sentinels[(table, new)]
            if expected is None:
                assert tuple(row) == ("null", None)
            else:
                assert tuple(row) == ("text", expected)
    finally:
        conn.close()


def test_upgrade_preserves_identities_links_and_generations(tmp_path: Path) -> None:
    _inspect, _apply, _refused, _load = _api()
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
        assert _columns(conn, "m_pod")[0] == "pod_id_seq"
        assert "npv_pf_ctr_pod" in _columns(conn, "m_pod")
        state = conn.execute(
            "SELECT contract_version FROM pod_registry_state WHERE singleton_id = 1"
        ).fetchone()[0]
        assert state == 1
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
    sentinels = _build_registry(path, renamed=True)
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
        for (table, new), expected in sentinels.items():
            row = conn.execute(
                f"SELECT {_quote(new)} FROM {_quote(table)} WHERE rowid = 1"
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
