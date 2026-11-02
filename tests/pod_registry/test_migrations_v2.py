"""Legacy-to-v2 migration, audit, FK, and rollback regressions."""

from __future__ import annotations

import importlib
import sqlite3
from dataclasses import replace
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest


def _api() -> tuple[Any, Any, Any, Any]:
    try:
        migrations = importlib.import_module("esdc.pod_registry.migrations")
        contract = importlib.import_module("esdc.pod_registry.contract")
        models = importlib.import_module("esdc.pod_registry.models")
        return (
            migrations.inspect_migration,
            migrations.migrate_registry,
            contract.load_pod_contract,
            models.PreparedPodBatch,
        )
    except (ImportError, AttributeError) as exc:
        pytest.fail(f"v2 migration API is unavailable: {exc}", pytrace=False)


def _value(column, row_number: int):
    if column.logical_type == "integer":
        return row_number
    if column.logical_type == "boolean":
        return True
    if column.logical_type == "decimal":
        return Decimal("123456789012.123")
    if column.logical_type == "date":
        return date(2024, 1, 1)
    if column.logical_type == "datetime":
        return datetime(2024, 1, 1)
    if column.logical_type == "uuid":
        return uuid4()
    return f"{column.name}-{row_number}"


def _row(contract, table_name: str, row_number: int, overrides: dict[str, Any]):
    row = {
        column.name: None if column.nullable else _value(column, row_number)
        for column in contract.tables[table_name].columns
    }
    row.update(overrides)
    return row


def _batch(load_pod_contract, prepared_pod_batch):
    contract = load_pod_contract()
    headers = {
        name: tuple(column.name for column in table.columns)
        for name, table in contract.tables.items()
        if table.role == "input"
    }
    rows = {name: [] for name in headers}
    rows["m_project"] = [
        _row(
            contract,
            "m_project",
            1,
            {"project_id": "P-0000001-01", "project_name": "Synthetic project"},
        )
    ]
    masters = []
    for number, approval_day, order_no, predecessor in (
        (1, date(2024, 1, 1), 1, 3),
        (2, date(2024, 1, 1), 2, 3),
        (3, date(2024, 1, 2), 21, None),
    ):
        rev_num = 0 if number < 3 else 1
        masters.append(
            _row(
                contract,
                "m_pod",
                number,
                {
                    "pod_id_seq": number,
                    "pod_uuid": uuid4(),
                    "approval_date": approval_day,
                    "superseded_by": predecessor,
                    "institution_id": 1,
                    "pod_type_id": 1,
                    "rev_num": rev_num,
                    "pod_order_no": order_no,
                    "pod_id": f"PL-2024-{order_no:04d}-1-1-{rev_num}",
                    "pod_name": f"Synthetic approval {number}",
                    "pod_letter_num": f"Letter-{number}",
                    "is_latest": number == 3,
                    "ncf_ctr_pod": Decimal("123456789012.123"),
                },
            )
        )
    placeholder = _row(
        contract,
        "m_pod",
        856,
        {
            "pod_id_seq": 856,
            "pod_uuid": uuid4(),
            "approval_date": date(1997, 6, 1),
            "superseded_by": None,
            "institution_id": 2,
            "pod_type_id": 2,
            "rev_num": 0,
            "pod_order_no": None,
            "pod_id": None,
            "pod_name": "Pengembangan Lapangan Semoga",
            "pod_letter_num": "Semoga placeholder",
            "is_latest": True,
        },
    )
    masters.append(placeholder)
    rows["m_pod"] = masters
    rows["pod_project"] = [
        _row(
            contract,
            "pod_project",
            1,
            {
                "id": 1,
                "pod_project_uuid": uuid4(),
                "pod_id_seq": 3,
                "project_id": "P-0000001-01",
                "is_main_project": True,
                "is_auto_inferred": False,
            },
        )
    ]
    rows["pod_monitoring"] = [
        _row(
            contract,
            "pod_monitoring",
            1,
            {
                "monpod_id": 1,
                "monpod_uuid": uuid4(),
                "pod_id_seq": 3,
                "pod_id": "PL-2024-0021-1-1-1",
                "reported_date": date(2024, 2, 1),
                "is_completed": True,
                "pod_stage_id": 1,
            },
        )
    ]
    rows["pod_forecast"] = [
        _row(
            contract,
            "pod_forecast",
            1,
            {
                "frcast_id": 1,
                "frcast_uuid": uuid4(),
                "pod_id_seq": 3,
                "year_frcast": 2025,
            },
        )
    ]
    return prepared_pod_batch(
        source="synthetic-test",
        contract_id=contract.contract_id,
        contract_version=contract.contract_version,
        dictionary_version=contract.dictionary_version,
        headers=headers,
        rows=rows,
    )


def _migrated_synthetic_connection() -> sqlite3.Connection:
    _, migrate_registry, load_pod_contract, prepared_pod_batch = _api()
    conn = _legacy_connection()
    migrate_registry(
        conn,
        prepared_batch=_batch(load_pod_contract, prepared_pod_batch),
        approved_crosswalk={900: 1, 901: 2},
    )
    return conn


def _source_identity_ledger():
    return [
        {
            "approval_year": 2024,
            "approval_date": date(2024, 1, 1),
            "daily_slot": 1,
            "pod_order_no": 1,
            "pod_id_seq": 1,
            "pod_id": "PL-2024-0001-1-1-0",
            "issued_at": "2026-10-27T08:10:26.426Z",
            "issued_by": "FA:initial-pod-identity-allocation",
        },
        {
            "approval_year": 2024,
            "approval_date": date(2024, 1, 1),
            "daily_slot": 2,
            "pod_order_no": 2,
            "pod_id_seq": 2,
            "pod_id": "PL-2024-0002-1-1-0",
            "issued_at": "2026-10-27T08:10:26.427Z",
            "issued_by": "FA:initial-pod-identity-allocation",
        },
        {
            "approval_year": 2024,
            "approval_date": date(2024, 1, 2),
            "daily_slot": 1,
            "pod_order_no": 21,
            "pod_id_seq": 3,
            "pod_id": "PL-2024-0021-1-1-1",
            "issued_at": "2026-10-27T08:10:26.428Z",
            "issued_by": "FA:initial-pod-identity-allocation",
        },
    ]


def _legacy_connection(database: str | Path = ":memory:") -> sqlite3.Connection:
    conn = sqlite3.connect(database)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    if str(database) != ":memory:":
        sqlite_runtime = importlib.import_module("esdc.sqlite_runtime")
        sqlite_runtime.configure_writer(conn)
    conn.executescript(
        """
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
            successor_id TEXT NOT NULL REFERENCES m_pod(pod_id),
            predecessor_id TEXT NOT NULL REFERENCES m_pod(pod_id),
            UNIQUE (successor_id, predecessor_id),
            CHECK (successor_id <> predecessor_id)
        );
        CREATE TABLE documents (doc_id TEXT PRIMARY KEY, content TEXT NOT NULL);
        """
    )
    conn.executemany(
        "INSERT INTO r_institution (code, institution) VALUES (?, ?)",
        [(1, "Legacy Institution")],
    )
    conn.executemany(
        "INSERT INTO r_pod_type (code, pod_type) VALUES (?, ?)",
        [(1, "Legacy Type")],
    )
    conn.executemany(
        "INSERT INTO m_pod (id, pod_id, pod_name, approval_date, institution_code, "
        "pod_type_code, rev_num, approval_seq) VALUES (?, ?, ?, ?, 1, 1, 0, ?)",
        [
            (900, "PL-1990-0001-1-1-0", "Legacy A", "1990-01-01", 1),
            (901, "PL-1991-0001-1-1-0", "Legacy B", "1991-01-01", 2),
        ],
    )
    conn.execute(
        "INSERT INTO pod_revision (successor_id, predecessor_id) "
        "VALUES (?, ?)",
        ("PL-1991-0001-1-1-0", "PL-1990-0001-1-1-0"),
    )
    conn.executemany(
        "INSERT INTO pod_document (pod_id, doc_id) VALUES (?, ?)",
        [(900, "DOC-001"), (901, "DOC-002")],
    )
    conn.executemany(
        "INSERT INTO documents (doc_id, content) VALUES (?, ?)",
        [("DOC-001", "retained source one"), ("DOC-002", "retained source two")],
    )
    conn.commit()
    return conn


def test_inspection_is_read_only_and_migration_preserves_links_and_stales_knowledge():
    inspect_migration, migrate_registry, load_pod_contract, prepared_pod_batch = _api()
    conn = _legacy_connection()
    try:
        conn.execute(
            "CREATE TRIGGER trg_pod_project_reference_insert AFTER INSERT ON project_pod "
            "BEGIN SELECT 1; END"
        )
        conn.commit()
        before_tables = tuple(
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name"
            )
        )
        before_changes = conn.total_changes
        report = inspect_migration(conn)
        assert report.status == "legacy"
        assert conn.total_changes == before_changes
        assert tuple(
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name"
            )
        ) == before_tables

        batch = _batch(load_pod_contract, prepared_pod_batch)
        original_uuids = {
            row["pod_id_seq"]: str(row["pod_uuid"]) for row in batch.rows["m_pod"]
        }
        report = migrate_registry(
            conn,
            prepared_batch=batch,
            approved_crosswalk={900: 1, 901: 2},
        )
        assert report.status == "v2"
        assert report.document_links_preserved == 2
        assert {
            mapping.legacy_pod_id: mapping.pod_id_seq
            for mapping in report.identity_reconciliation
        } == {900: 1, 901: 2}
        assert {
            row[0]: row[1]
            for row in conn.execute("SELECT pod_id_seq, pod_uuid FROM m_pod")
        } == original_uuids
        assert {
            tuple(row)
            for row in conn.execute(
                "SELECT pod_id_seq, superseded_by, is_latest FROM m_pod "
                "WHERE pod_id_seq IN (1, 2, 3)"
            )
        } == {(1, 3, 0), (2, 3, 0), (3, None, 1)}
        assert conn.execute(
            "SELECT COUNT(*) FROM _pod_identity_ledger"
        ).fetchone()[0] == 3
        assert conn.execute(
            "SELECT pod_uuid FROM m_pod WHERE pod_id_seq = 856"
        ).fetchone()[0] == original_uuids[856]
        assert tuple(conn.execute(
            "SELECT typeof(ncf_ctr_pod), ncf_ctr_pod FROM m_pod WHERE pod_id_seq = 1"
        ).fetchone()) == ("text", "123456789012.123")
        assert [tuple(row) for row in conn.execute(
            "SELECT pod_id_seq, doc_id FROM pod_document ORDER BY pod_id_seq"
        ).fetchall()] == [(1, "DOC-001"), (2, "DOC-002")]
        assert [tuple(row) for row in conn.execute(
            "SELECT doc_id, content FROM documents ORDER BY doc_id"
        )] == [
            ("DOC-001", "retained source one"),
            ("DOC-002", "retained source two"),
        ]
        state = tuple(conn.execute(
            "SELECT source_generation, knowledge_generation "
            "FROM pod_registry_state WHERE singleton_id = 1"
        ).fetchone())
        assert state == (1, None)
        assert conn.execute("PRAGMA legacy_alter_table").fetchone()[0] == 0
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
        assert conn.execute("PRAGMA foreign_key_list(pod_document)").fetchone()[2:5] == (
            "m_pod",
            "pod_id_seq",
            "pod_id_seq",
        )
        names = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        assert "project_pod" not in names
        assert "pod_revision" not in names
        assert "pod_registry_legacy_identity" in names
        store = importlib.import_module("esdc.pod_registry.store")
        # Reopening the migrated latest schema is idempotent: it validates the canonical
        # shape and adds only the shared raw-source tables, never a retired POD table.
        store.ensure_tables(conn)
        after = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        assert names <= after
        assert {name for name in after if name.startswith("api_")} == {
            "api_project_resources",
            "api_snapshot",
            "api_project_timeseries",
            "api_snapshot_provenance",
        }
        assert "project_pod" not in after
        assert "pod_revision" not in after
        assert conn.execute(
            "SELECT pod_uuid FROM m_pod WHERE pod_id_seq = 856"
        ).fetchone()[0] == original_uuids[856]
        assert tuple(conn.execute(
            "SELECT source_generation, knowledge_generation "
            "FROM pod_registry_state WHERE singleton_id = 1"
        ).fetchone()) == state
        conn.execute("PRAGMA foreign_keys = OFF")
        with pytest.raises(sqlite3.IntegrityError, match="pod_project reference"):
            conn.execute(
                "INSERT INTO pod_project (pod_project_uuid, pod_id_seq, project_id, "
                "is_main_project, is_auto_inferred) "
                "VALUES ('00000000-0000-4000-8000-000000000009', 3, "
                "'missing-project', 0, 0)"
            )
    finally:
        conn.close()


def test_unknown_foreign_key_dependency_blocks_migration_without_changes():
    inspect_migration, migrate_registry, load_pod_contract, prepared_pod_batch = _api()
    conn = _legacy_connection()
    try:
        conn.execute(
            "CREATE TABLE external_pod_cache (pod_id INTEGER REFERENCES m_pod(id))"
        )
        conn.execute(
            "CREATE TABLE external_project_cache ("
            "project_id TEXT REFERENCES project_pod(project_id))"
        )
        conn.commit()
        report = inspect_migration(conn)
        assert report.blockers
        assert any("external_project_cache" in blocker for blocker in report.blockers)
        before = conn.total_changes
        with pytest.raises(RuntimeError):
            migrate_registry(
                conn,
                prepared_batch=_batch(load_pod_contract, prepared_pod_batch),
                approved_crosswalk={900: 1, 901: 2},
            )
        assert conn.total_changes == before
        assert conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' "
            "AND name = 'pod_registry_state'"
        ).fetchone() is None
    finally:
        conn.close()


def test_rebuild_failure_rolls_back_all_schema_and_data_changes():
    _, migrate_registry, load_pod_contract, prepared_pod_batch = _api()
    conn = _legacy_connection()
    before_master = conn.execute("SELECT COUNT(*) FROM m_pod").fetchone()[0]
    before_links = conn.execute("SELECT COUNT(*) FROM pod_document").fetchone()[0]
    batch = _batch(load_pod_contract, prepared_pod_batch)

    def deny_monitoring_create(action, arg1, _arg2, _database, _trigger):
        if action == sqlite3.SQLITE_CREATE_TABLE and arg1 == "pod_monitoring":
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK

    try:
        conn.set_authorizer(deny_monitoring_create)
        with pytest.raises(sqlite3.DatabaseError):
            migrate_registry(
                conn,
                prepared_batch=batch,
                approved_crosswalk={900: 1, 901: 2},
            )
        conn.set_authorizer(None)
        assert not conn.in_transaction
        assert conn.execute("SELECT COUNT(*) FROM m_pod").fetchone()[0] == before_master
        assert conn.execute("SELECT COUNT(*) FROM pod_document").fetchone()[0] == before_links
        assert conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' "
            "AND name = 'pod_registry_state'"
        ).fetchone() is None
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
    finally:
        conn.set_authorizer(None)
        conn.close()


def test_migration_joins_outer_transaction_and_outer_rollback_restores_legacy(
    tmp_path: Path,
):
    _, migrate_registry, load_pod_contract, prepared_pod_batch = _api()
    conn = _legacy_connection(tmp_path / "outer-transaction.sqlite")
    try:
        conn.execute("BEGIN IMMEDIATE")
        report = migrate_registry(
            conn,
            prepared_batch=_batch(load_pod_contract, prepared_pod_batch),
            approved_crosswalk={900: 1, 901: 2},
        )
        assert report.status == "v2"
        assert conn.in_transaction
        conn.execute("CREATE TABLE caller_owned_work (value TEXT NOT NULL)")
        conn.execute("INSERT INTO caller_owned_work VALUES ('later task work')")
        assert conn.execute("SELECT COUNT(*) FROM m_pod").fetchone()[0] == 4
        conn.rollback()

        assert not conn.in_transaction
        assert conn.execute("SELECT COUNT(*) FROM m_pod").fetchone()[0] == 2
        assert conn.execute("SELECT COUNT(*) FROM pod_document").fetchone()[0] == 2
        assert conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' "
            "AND name = 'pod_registry_state'"
        ).fetchone() is None
        assert conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' "
            "AND name = 'caller_owned_work'"
        ).fetchone() is None
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
    finally:
        conn.close()


def test_migration_preserves_provided_identity_ledger_audit_fields():
    migrations = importlib.import_module("esdc.pod_registry.migrations")
    _, migrate_registry, load_pod_contract, prepared_pod_batch = _api()
    batch = replace(
        _batch(load_pod_contract, prepared_pod_batch),
        identity_ledger=_source_identity_ledger(),
    )
    conn = _legacy_connection()
    try:
        migrate_registry(
            conn,
            prepared_batch=batch,
            approved_crosswalk={900: 1, 901: 2},
        )

        masters = {
            int(row["pod_id_seq"]): str(row["pod_uuid"])
            for row in batch.rows["m_pod"]
        }
        assert batch.identity_ledger is not None
        for entry in batch.identity_ledger:
            issued = conn.execute(
                "SELECT pod_uuid, actor, issued_at FROM _pod_identity_ledger "
                "WHERE pod_id_seq = ?",
                (entry["pod_id_seq"],),
            ).fetchone()
            assert tuple(issued) == (
                masters[int(entry["pod_id_seq"])],
                entry["issued_by"],
                entry["issued_at"],
            )
        before_reimport = [
            tuple(row)
            for row in conn.execute(
                "SELECT approval_year, approval_date, daily_slot, pod_order_no, "
                "pod_id_seq, pod_id, pod_uuid, institution_id, pod_type_id, "
                "rev_num, actor, issued_at FROM _pod_identity_ledger "
                "ORDER BY pod_id_seq"
            )
        ]
        migrations._insert_identity_ledger(
            conn, migrations._identity_ledger_reservations(batch)
        )
        after_reimport = [
            tuple(row)
            for row in conn.execute(
                "SELECT approval_year, approval_date, daily_slot, pod_order_no, "
                "pod_id_seq, pod_id, pod_uuid, institution_id, pod_type_id, "
                "rev_num, actor, issued_at FROM _pod_identity_ledger "
                "ORDER BY pod_id_seq"
            )
        ]
        assert after_reimport == before_reimport
    finally:
        conn.close()


def test_migration_rejects_conflicting_provided_identity_ledger_atomically():
    migrations = importlib.import_module("esdc.pod_registry.migrations")
    _, migrate_registry, load_pod_contract, prepared_pod_batch = _api()
    identity_ledger = _source_identity_ledger()
    identity_ledger[1] = {**identity_ledger[1], "pod_id": "PL-2024-0002-1-1-9"}
    batch = replace(
        _batch(load_pod_contract, prepared_pod_batch),
        identity_ledger=identity_ledger,
    )
    conn = _legacy_connection()
    try:
        with pytest.raises(migrations.MigrationRefusedError, match="ledger.*conflicts"):
            migrate_registry(
                conn,
                prepared_batch=batch,
                approved_crosswalk={900: 1, 901: 2},
            )
        assert conn.execute("SELECT COUNT(*) FROM m_pod").fetchone()[0] == 2
        assert conn.execute("SELECT COUNT(*) FROM pod_document").fetchone()[0] == 2
        assert conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' "
            "AND name = 'pod_registry_state'"
        ).fetchone() is None
    finally:
        conn.close()


def test_empty_provided_identity_ledger_does_not_mask_issued_masters():
    migrations = importlib.import_module("esdc.pod_registry.migrations")
    _, migrate_registry, load_pod_contract, prepared_pod_batch = _api()
    batch = replace(
        _batch(load_pod_contract, prepared_pod_batch),
        identity_ledger=[],
    )
    conn = _legacy_connection()
    try:
        with pytest.raises(migrations.MigrationRefusedError, match="missing issued masters"):
            migrate_registry(
                conn,
                prepared_batch=batch,
                approved_crosswalk={900: 1, 901: 2},
            )
        assert conn.execute("SELECT COUNT(*) FROM m_pod").fetchone()[0] == 2
        assert conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' "
            "AND name = 'pod_registry_state'"
        ).fetchone() is None
    finally:
        conn.close()


def test_migration_rejects_orphan_identity_ledger_without_full_identity_atomically():
    migrations = importlib.import_module("esdc.pod_registry.migrations")
    _, migrate_registry, load_pod_contract, prepared_pod_batch = _api()
    identity_ledger = _source_identity_ledger()
    identity_ledger.append(
        {
            "approval_year": 2023,
            "approval_date": date(2023, 1, 1),
            "daily_slot": 1,
            "pod_order_no": 1,
            "pod_id_seq": 9000,
            "pod_id": "PL-2023-0001-1-1-0",
            "issued_at": "2026-10-27T08:10:26.429Z",
            "issued_by": "FA:initial-pod-identity-allocation",
        }
    )
    batch = replace(
        _batch(load_pod_contract, prepared_pod_batch),
        identity_ledger=identity_ledger,
    )
    conn = _legacy_connection()
    try:
        with pytest.raises(migrations.MigrationRefusedError, match="orphan.*identity"):
            migrate_registry(
                conn,
                prepared_batch=batch,
                approved_crosswalk={900: 1, 901: 2},
            )
        assert conn.execute("SELECT COUNT(*) FROM m_pod").fetchone()[0] == 2
        assert conn.execute("SELECT COUNT(*) FROM pod_document").fetchone()[0] == 2
        assert conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' "
            "AND name = 'pod_registry_state'"
        ).fetchone() is None
    finally:
        conn.close()


def test_migration_preserves_fully_described_orphan_identity_reservation():
    _, migrate_registry, load_pod_contract, prepared_pod_batch = _api()
    orphan_uuid = uuid4()
    identity_ledger = [
        *_source_identity_ledger(),
        {
            "approval_year": 2024,
            "approval_date": date(2024, 1, 3),
            "daily_slot": 1,
            "pod_order_no": 41,
            "pod_id_seq": 9000,
            "pod_id": "PL-2024-0041-1-1-0",
            "pod_uuid": orphan_uuid,
            "institution_id": 1,
            "pod_type_id": 1,
            "rev_num": 0,
            "issued_at": "2026-10-27T08:10:26.430Z",
            "issued_by": "FA:initial-pod-identity-allocation",
        },
    ]
    batch = replace(
        _batch(load_pod_contract, prepared_pod_batch),
        identity_ledger=identity_ledger,
    )
    conn = _legacy_connection()
    try:
        migrate_registry(
            conn,
            prepared_batch=batch,
            approved_crosswalk={900: 1, 901: 2},
        )
        assert tuple(
            conn.execute(
                "SELECT pod_uuid, actor, issued_at FROM _pod_identity_ledger "
                "WHERE pod_id_seq = 9000"
            ).fetchone()
        ) == (
            str(orphan_uuid),
            "FA:initial-pod-identity-allocation",
            "2026-10-27T08:10:26.430Z",
        )
        assert conn.execute("SELECT COUNT(*) FROM _pod_identity_ledger").fetchone()[0] == 4
    finally:
        conn.close()


def test_migrated_issued_identity_is_immutable_and_ledger_bound_after_delete():
    conn = _migrated_synthetic_connection()
    try:
        conn.execute("PRAGMA foreign_keys = OFF")
        assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 0

        with pytest.raises(sqlite3.IntegrityError, match="identity"):
            conn.execute(
                "UPDATE m_pod SET pod_uuid = ? WHERE pod_id_seq = 3",
                (str(uuid4()),),
            )
        with pytest.raises(sqlite3.IntegrityError, match="identity"):
            conn.execute(
                "UPDATE m_pod SET approval_date = '2023-01-01', pod_order_no = 2, "
                "pod_id = 'PL-2023-0002-1-1-0' WHERE pod_id_seq = 2"
            )

        conn.execute("UPDATE m_pod SET pod_name = 'Allowed title edit' WHERE pod_id_seq = 2")
        assert conn.execute(
            "SELECT pod_name FROM m_pod WHERE pod_id_seq = 2"
        ).fetchone()[0] == "Allowed title edit"

        conn.execute("DELETE FROM pod_document WHERE pod_id_seq = 2")
        conn.execute("DELETE FROM m_pod WHERE pod_id_seq = 2")
        with pytest.raises(sqlite3.IntegrityError, match="ledger"):
            conn.execute(
                "INSERT INTO m_pod (pod_id_seq, pod_uuid, approval_date, "
                "superseded_by, institution_id, pod_type_id, rev_num, pod_order_no, "
                "pod_id, pod_name, pod_letter_num, is_latest) VALUES "
                "(2, ?, '2024-01-01', 3, 1, 1, 0, 2, 'PL-2024-0002-1-1-0', "
                "'Synthetic approval 2', 'Letter-2', 0)",
                (str(uuid4()),),
            )
    finally:
        conn.close()


def test_identity_ledger_rejects_conflicting_insert_or_replace():
    conn = _migrated_synthetic_connection()
    columns = (
        "approval_year, approval_date, daily_slot, pod_order_no, pod_id_seq, "
        "pod_id, pod_uuid, institution_id, pod_type_id, rev_num, actor"
    )
    try:
        row = conn.execute(
            f"SELECT {columns} FROM _pod_identity_ledger WHERE pod_id_seq = 2"
        ).fetchone()
        replacement = list(row)
        replacement[5] = "PL-2024-0002-1-1-9"
        replacement[6] = str(uuid4())

        with pytest.raises(sqlite3.IntegrityError, match="ledger"):
            conn.execute(
                f"INSERT OR REPLACE INTO _pod_identity_ledger ({columns}) "
                f"VALUES ({', '.join('?' for _ in replacement)})",
                replacement,
            )

        assert tuple(
            conn.execute(
                "SELECT pod_id, pod_uuid FROM _pod_identity_ledger "
                "WHERE pod_id_seq = 2"
            ).fetchone()
        ) == (row[5], row[6])
    finally:
        conn.close()


def test_migrated_supersession_and_latest_state_are_guarded_with_foreign_keys_off():
    conn = _migrated_synthetic_connection()
    try:
        conn.execute("PRAGMA foreign_keys = OFF")
        assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 0

        with pytest.raises(sqlite3.IntegrityError, match="supersession cycle"):
            conn.execute(
                "UPDATE m_pod SET superseded_by = 1, is_latest = 0 "
                "WHERE pod_id_seq = 3"
            )
        with pytest.raises(sqlite3.IntegrityError, match="later POD"):
            conn.execute(
                "UPDATE m_pod SET superseded_by = 1, is_latest = 0 "
                "WHERE pod_id_seq = 2"
            )
        with pytest.raises(sqlite3.IntegrityError, match="later POD"):
            conn.execute("UPDATE m_pod SET superseded_by = 2 WHERE pod_id_seq = 1")
        with pytest.raises(sqlite3.IntegrityError, match="CHECK constraint"):
            conn.execute("UPDATE m_pod SET is_latest = 0 WHERE pod_id_seq = 3")
        assert [tuple(row) for row in conn.execute(
            "SELECT pod_id_seq, superseded_by, is_latest FROM m_pod "
            "WHERE pod_id_seq IN (1, 2, 3) ORDER BY pod_id_seq"
        )] == [(1, 3, 0), (2, 3, 0), (3, None, 1)]
    finally:
        conn.close()


def test_unissued_placeholder_can_point_to_later_pod_without_daily_order():
    conn = _migrated_synthetic_connection()
    try:
        conn.execute(
            "UPDATE m_pod SET superseded_by = 3, is_latest = 0 "
            "WHERE pod_id_seq = 856"
        )

        assert tuple(
            conn.execute(
                "SELECT superseded_by, is_latest FROM m_pod WHERE pod_id_seq = 856"
            ).fetchone()
        ) == (3, 0)
    finally:
        conn.close()
