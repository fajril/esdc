"""Physical-v2 schema, seed, and legacy-routing regressions."""

from __future__ import annotations

import importlib
import sqlite3
from datetime import date
from pathlib import Path
from typing import Any

import pytest


def _api() -> tuple[Any, Any]:
    try:
        migrations = importlib.import_module("esdc.pod_registry.migrations")
        contract = importlib.import_module("esdc.pod_registry.contract")
        return migrations.initialize_registry_v2, contract.load_pod_contract
    except (ImportError, AttributeError) as exc:
        pytest.fail(f"v2 schema initializer is unavailable: {exc}", pytrace=False)


def _connection() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _columns(conn: sqlite3.Connection, table: str) -> tuple[str, ...]:
    return tuple(row[1] for row in conn.execute(f'PRAGMA table_info("{table}")'))


def _primary_key(conn: sqlite3.Connection, table: str) -> tuple[str, ...]:
    rows = conn.execute(f'PRAGMA table_info("{table}")').fetchall()
    return tuple(row[1] for row in sorted((r for r in rows if r[5]), key=lambda r: r[5]))


def test_explicit_v2_initializer_matches_contract_and_seeds_references_idempotently():
    initialize_registry_v2, load_pod_contract = _api()
    conn = _connection()
    try:
        initialize_registry_v2(conn)
        contract = load_pod_contract()
        expected_tables = (
            "m_project",
            "m_pod",
            "pod_project",
            "pod_monitoring",
            "pod_forecast",
            "pod_monitoring_api",
            "pod_forecast_api",
        )
        assert tuple(contract.tables) == expected_tables
        for name in expected_tables:
            expected = tuple(column.name for column in contract.tables[name].columns)
            assert _columns(conn, name) == expected

        assert conn.execute(
            "SELECT id, institution FROM r_institution ORDER BY id"
        ).fetchall() == [
            (1, "MESDM"),
            (2, "BPPKA"),
            (3, "BP Migas"),
            (4, "SKK Migas"),
            (5, "BPMA"),
        ]
        assert conn.execute("SELECT COUNT(*) FROM r_pod_type").fetchone()[0] == 4
        assert conn.execute("SELECT COUNT(*) FROM r_pod_stage").fetchone()[0] == 4

        conn.execute(
            "INSERT INTO r_institution (id, institution, description, created_by, updated_by) "
            "VALUES (9, 'Local extension', 'kept across seed refresh', 'FA:test', 'FA:test')"
        )
        initialize_registry_v2(conn)
        assert conn.execute(
            "SELECT institution FROM r_institution WHERE id = 9"
        ).fetchone() == ("Local extension",)

        state = conn.execute(
            "SELECT schema_version, source_generation, knowledge_generation "
            "FROM pod_registry_state WHERE singleton_id = 1"
        ).fetchone()
        assert state == (2, 0, None)
        assert _primary_key(conn, "pod_monitoring_api") == (
            "pod_id_seq",
            "reported_date",
        )
        assert _primary_key(conn, "pod_forecast_api") == (
            "pod_id_seq",
            "reported_date",
            "year_frcast",
        )
        for table, key in {
            "pod_monitoring_api": ("pod_id_seq", "reported_date"),
            "pod_forecast_api": ("pod_id_seq", "reported_date", "year_frcast"),
        }.items():
            rows = conn.execute(f'PRAGMA table_info("{table}")').fetchall()
            assert all(next(row[3] for row in rows if row[1] == column) for column in key)
        assert {
            (row[2], row[3], row[4])
            for row in conn.execute('PRAGMA foreign_key_list("pod_project")')
        } == {
            ("m_pod", "pod_id_seq", "pod_id_seq"),
            ("m_project", "project_id", "project_id"),
        }
        assert any(
            (row[2], row[3], row[4]) == ("m_pod", "superseded_by", "pod_id_seq")
            for row in conn.execute('PRAGMA foreign_key_list("m_pod")')
        )
        names = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        assert "pod_document" in names
        assert "pod_revision" not in names
        assert "project_pod" not in names
    finally:
        conn.close()


def test_v2_initializer_stays_inside_a_caller_transaction():
    initialize_registry_v2, _ = _api()
    conn = _connection()
    try:
        conn.execute("BEGIN IMMEDIATE")
        initialize_registry_v2(conn)
        assert conn.in_transaction
        conn.rollback()
        assert conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' "
            "AND name = 'pod_registry_state'"
        ).fetchone() is None
    finally:
        conn.close()


def test_current_store_accepts_the_canonical_database_idempotently():
    initialize_registry_v2, _ = _api()
    store = importlib.import_module("esdc.pod_registry.store")
    conn = _connection()
    try:
        initialize_registry_v2(conn)
        state_before = conn.execute(
            "SELECT schema_version, contract_id, contract_version, "
            "dictionary_version, source_generation FROM pod_registry_state "
            "WHERE singleton_id = 1"
        ).fetchone()
        identities_before = conn.execute(
            "SELECT pod_id_seq, pod_uuid FROM m_pod ORDER BY pod_id_seq"
        ).fetchall()
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "memory"

        # Current policy: a canonical latest-schema database is accepted as it is,
        # and re-initializing it is idempotent rather than a refusal.
        store.ensure_tables(conn)

        names = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        assert "project_pod" not in names
        assert "pod_revision" not in names
        assert conn.execute(
            "SELECT schema_version, contract_id, contract_version, "
            "dictionary_version, source_generation FROM pod_registry_state "
            "WHERE singleton_id = 1"
        ).fetchone() == state_before
        assert conn.execute(
            "SELECT pod_id_seq, pod_uuid FROM m_pod ORDER BY pod_id_seq"
        ).fetchall() == identities_before
    finally:
        conn.close()


def test_fixed_scale_decimal_contract_columns_use_exact_text_storage():
    initialize_registry_v2, load_pod_contract = _api()
    conn = _connection()
    try:
        initialize_registry_v2(conn)
        contract = load_pod_contract()
        fixed_decimals = [
            (table.name, column.name)
            for table in contract.tables.values()
            for column in table.columns
            if column.logical_type == "decimal" and column.scale is not None
        ]
        assert fixed_decimals
        for table, column in fixed_decimals:
            declared_type = conn.execute(
                f'PRAGMA table_info("{table}")'
            ).fetchall()
            sqlite_type = next(row[2] for row in declared_type if row[1] == column)
            assert sqlite_type.upper() == "TEXT"
        pod_columns = conn.execute('PRAGMA table_info("m_pod")').fetchall()
        assert next(row[2] for row in pod_columns if row[1] == "res_oil_2p_pod") == "REAL"
    finally:
        conn.close()


def test_file_initializer_enables_shared_writer_wal_before_schema(tmp_path: Path):
    initialize_registry_v2, _ = _api()
    conn = sqlite3.connect(tmp_path / "registry.sqlite")
    try:
        initialize_registry_v2(conn)
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert conn.execute(
            "SELECT schema_version FROM pod_registry_state WHERE singleton_id = 1"
        ).fetchone()[0] == 2
    finally:
        conn.close()


def test_file_initializer_rejects_unconfigured_outer_transaction_before_ddl(
    tmp_path: Path,
):
    initialize_registry_v2, _ = _api()
    conn = sqlite3.connect(tmp_path / "unconfigured.sqlite")
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        conn.execute("BEGIN IMMEDIATE")
        with pytest.raises(RuntimeError, match="Configure file-backed"):
            initialize_registry_v2(conn)
        assert conn.in_transaction
        assert conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' "
            "AND name = 'pod_registry_state'"
        ).fetchone() is None
    finally:
        conn.rollback()
        conn.close()


def test_file_initializer_rejects_unsupported_sqlite_before_ddl(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    initialize_registry_v2, _ = _api()
    sqlite_runtime = importlib.import_module("esdc.sqlite_runtime")
    monkeypatch.setattr(sqlite_runtime.sqlite3, "sqlite_version_info", (3, 50, 4))
    conn = sqlite3.connect(tmp_path / "unsupported.sqlite")
    try:
        with pytest.raises(RuntimeError, match="unsupported for WAL writers"):
            initialize_registry_v2(conn)
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "delete"
        assert conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' "
            "AND name = 'pod_registry_state'"
        ).fetchone() is None
    finally:
        conn.close()


def test_pod_project_references_remain_guarded_when_connection_foreign_keys_are_off():
    initialize_registry_v2, _ = _api()
    conn = _connection()
    try:
        initialize_registry_v2(conn)
        conn.execute(
            "INSERT INTO m_project (project_id, project_name, is_active) "
            "VALUES ('P-1', 'Project one', 1)"
        )
        identity_api = importlib.import_module("esdc.pod_registry.identity")
        issued = identity_api.issue_identity(
            conn,
            key=identity_api.ApprovalKey(
                approval_date=date(2024, 1, 1),
                daily_slot=1,
                institution_id=1,
                pod_type_id=1,
                rev_num=0,
            ),
            actor="FA:test-schema-reference-guard",
        )
        conn.execute(
            "INSERT INTO m_pod (pod_id_seq, pod_uuid, approval_date, institution_id, "
            "pod_type_id, rev_num, pod_order_no, pod_id, pod_name, pod_letter_num, "
            "is_latest) VALUES (?, ?, ?, 1, 1, 0, ?, ?, 'POD one', 'Letter one', 1)",
            (
                issued.pod_id_seq,
                str(issued.pod_uuid),
                issued.approval_date.isoformat(),
                issued.pod_order_no,
                issued.pod_id,
            ),
        )
        conn.commit()
        conn.execute("PRAGMA foreign_keys = OFF")
        assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 0

        with pytest.raises(sqlite3.IntegrityError, match="pod_project reference"):
            conn.execute(
                "INSERT INTO pod_project (pod_project_uuid, pod_id_seq, project_id, "
                "is_main_project, is_auto_inferred) "
                "VALUES ('00000000-0000-4000-8000-000000000002', 1, "
                "'unknown-project', 0, 0)"
            )
        with pytest.raises(sqlite3.IntegrityError, match="pod_project reference"):
            conn.execute(
                "INSERT INTO pod_project (pod_project_uuid, pod_id_seq, project_id, "
                "is_main_project, is_auto_inferred) "
                "VALUES ('00000000-0000-4000-8000-000000000003', 99, 'P-1', 0, 0)"
            )

        conn.execute(
            "INSERT INTO pod_project (pod_project_uuid, pod_id_seq, project_id, "
            "is_main_project, is_auto_inferred) "
            "VALUES ('00000000-0000-4000-8000-000000000004', 1, 'P-1', 0, 0)"
        )
        with pytest.raises(sqlite3.IntegrityError, match="pod_project reference"):
            conn.execute("UPDATE pod_project SET project_id = 'unknown-project'")
        with pytest.raises(sqlite3.IntegrityError, match="pod_project reference"):
            conn.execute("UPDATE pod_project SET pod_id_seq = 99")
        with pytest.raises(sqlite3.IntegrityError, match="referenced by pod_project"):
            conn.execute("DELETE FROM m_project WHERE project_id = 'P-1'")
        with pytest.raises(sqlite3.IntegrityError, match="referenced by pod_project"):
            conn.execute("DELETE FROM m_pod WHERE pod_id_seq = 1")
        with pytest.raises(sqlite3.IntegrityError, match="referenced by pod_project"):
            conn.execute("UPDATE m_project SET project_id = 'P-2' WHERE project_id = 'P-1'")
        with pytest.raises(sqlite3.IntegrityError, match="referenced by pod_project"):
            conn.execute("UPDATE m_pod SET pod_id_seq = 2 WHERE pod_id_seq = 1")
        assert conn.execute("SELECT COUNT(*) FROM pod_project").fetchone()[0] == 1
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
    finally:
        conn.close()
