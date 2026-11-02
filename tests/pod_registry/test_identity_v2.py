"""POD identity allocation and permanent-ledger regressions."""

from __future__ import annotations

import importlib
import sqlite3
from datetime import date
from typing import Any, cast
from uuid import UUID

import pytest


def _api() -> tuple[Any, Any, Any, Any, Any, Any]:
    try:
        migrations = importlib.import_module("esdc.pod_registry.migrations")
        identity = importlib.import_module("esdc.pod_registry.identity")
        return (
            migrations.initialize_registry_v2,
            identity.ApprovalKey,
            identity.issue_identity,
            identity.explicit_revision_from_name,
            identity.resolve_revision_number,
            identity.next_available_daily_slot,
        )
    except (ImportError, AttributeError) as exc:
        pytest.fail(f"v2 identity API is unavailable: {exc}", pytrace=False)


def _new_identity(conn, issue_identity, approval_key, day: date, slot: int):
    return issue_identity(
        conn,
        key=approval_key(
            approval_date=day,
            daily_slot=slot,
            institution_id=5,
            pod_type_id=1,
            rev_num=0,
        ),
        actor="FA:manual-approval",
    )


def test_identity_uses_date_block_slots_and_rejects_the_twenty_first_atomically():
    (
        initialize_registry_v2,
        approval_key,
        issue_identity,
        _,
        _,
        next_available_daily_slot,
    ) = _api()
    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        initialize_registry_v2(conn)
        before_changes = conn.total_changes
        assert next_available_daily_slot(conn, date(2024, 1, 1)) == 1
        assert conn.total_changes == before_changes
        issued = [
            _new_identity(conn, issue_identity, approval_key, date(2024, 1, 1), slot)
            for slot in range(1, 21)
        ]
        assert [identity.pod_order_no for identity in issued] == list(range(1, 21))
        assert [identity.pod_id_seq for identity in issued] == list(range(1, 21))
        with pytest.raises(ValueError):
            _new_identity(conn, issue_identity, approval_key, date(2024, 1, 1), 21)
        with pytest.raises(TypeError):
            _new_identity(
                conn,
                issue_identity,
                approval_key,
                date(2024, 1, 1),
                cast(int, 1.5),
            )
        with pytest.raises(ValueError, match="all 20"):
            next_available_daily_slot(conn, date(2024, 1, 1))
        assert conn.execute(
            "SELECT COUNT(*) FROM _pod_identity_ledger"
        ).fetchone()[0] == 20
        assert issued[0].pod_id == "PL-2024-0001-5-1-0"
        assert issued[0].actor == "FA:manual-approval"
        assert UUID(str(issued[0].pod_uuid)).version == 4
    finally:
        conn.close()


def test_identity_backfill_does_not_shift_slots_and_leap_year_uses_day_366():
    initialize_registry_v2, approval_key, issue_identity, _, _, _ = _api()
    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        initialize_registry_v2(conn)
        fifth = _new_identity(conn, issue_identity, approval_key, date(2023, 1, 2), 5)
        first = _new_identity(conn, issue_identity, approval_key, date(2023, 1, 2), 1)
        leap_end = _new_identity(
            conn, issue_identity, approval_key, date(2024, 12, 31), 20
        )
        common_end = _new_identity(
            conn, issue_identity, approval_key, date(2023, 12, 31), 20
        )
        assert (fifth.pod_order_no, first.pod_order_no) == (25, 21)
        assert leap_end.pod_order_no == 7320
        assert common_end.pod_order_no == 7300
    finally:
        conn.close()


def test_permanent_ledger_prevents_reusing_a_deleted_master_identity():
    (
        initialize_registry_v2,
        approval_key,
        issue_identity,
        _,
        _,
        next_available_daily_slot,
    ) = _api()
    contract_module = importlib.import_module("esdc.pod_registry.contract")
    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        initialize_registry_v2(conn)
        for slot in range(1, 4):
            _new_identity(conn, issue_identity, approval_key, date(2024, 2, 3), slot)
        issued = _new_identity(
            conn, issue_identity, approval_key, date(2024, 2, 3), 4
        )
        contract = contract_module.load_pod_contract()
        columns = contract.tables["m_pod"].columns
        values = {}
        for column in columns:
            if column.nullable:
                values[column.name] = None
            elif column.name == "pod_id_seq":
                values[column.name] = issued.pod_id_seq
            elif column.name == "pod_uuid":
                values[column.name] = str(issued.pod_uuid)
            elif column.name == "approval_date":
                values[column.name] = "2024-02-03"
            elif column.name == "institution_id":
                values[column.name] = 5
            elif column.name == "pod_type_id":
                values[column.name] = 1
            elif column.name == "rev_num":
                values[column.name] = 0
            elif column.name == "pod_order_no":
                values[column.name] = issued.pod_order_no
            elif column.name == "pod_id":
                values[column.name] = issued.pod_id
            elif column.name == "pod_name":
                values[column.name] = "Identity test"
            elif column.name == "pod_letter_num":
                values[column.name] = "Letter identity test"
            elif column.name == "is_latest":
                values[column.name] = True
            elif column.logical_type == "text":
                values[column.name] = column.name
            elif column.logical_type == "integer":
                values[column.name] = 1
            elif column.logical_type == "boolean":
                values[column.name] = True
            elif column.logical_type == "date":
                values[column.name] = "2024-02-03"
            elif column.logical_type == "datetime":
                values[column.name] = "2024-02-03T00:00:00"
            elif column.logical_type == "uuid":
                values[column.name] = "00000000-0000-4000-8000-000000000001"
            else:
                values[column.name] = "1"
        names = tuple(values)
        conn.execute(
            f"INSERT INTO m_pod ({', '.join(names)}) VALUES "
            f"({', '.join('?' for _ in names)})",
            tuple(values[name] for name in names),
        )
        conn.execute("DELETE FROM m_pod WHERE pod_id_seq = ?", (issued.pod_id_seq,))
        assert conn.execute(
            "SELECT COUNT(*) FROM _pod_identity_ledger"
        ).fetchone()[0] == 4
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute("DELETE FROM _pod_identity_ledger")
        with pytest.raises(ValueError):
            _new_identity(
                conn, issue_identity, approval_key, date(2024, 2, 3), 4
            )
        assert next_available_daily_slot(conn, date(2024, 2, 3)) == 5
    finally:
        conn.close()


def test_identity_sequence_skips_unissued_placeholder_master_id():
    initialize_registry_v2, approval_key, issue_identity, _, _, _ = _api()
    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        initialize_registry_v2(conn)
        conn.execute(
            "INSERT INTO m_pod (pod_id_seq, pod_uuid, approval_date, institution_id, "
            "pod_type_id, rev_num, pod_order_no, pod_id, pod_name, pod_letter_num, "
            "is_latest) VALUES (856, '00000000-0000-4000-8000-000000000856', "
            "'1997-06-01', 2, 2, 0, NULL, NULL, 'Pengembangan Lapangan Semoga', "
            "'Semoga placeholder', 1)"
        )

        issued = _new_identity(
            conn, issue_identity, approval_key, date(2024, 3, 1), 1
        )

        assert issued.pod_id_seq == 857
        assert conn.execute(
            "SELECT pod_id, pod_order_no FROM m_pod WHERE pod_id_seq = 856"
        ).fetchone() == (None, None)
        assert conn.execute(
            "SELECT pod_id_seq FROM _pod_identity_ledger WHERE pod_id_seq = 856"
        ).fetchone() is None
    finally:
        conn.close()


def test_identity_sequence_keeps_high_water_after_deleted_unissued_master():
    initialize_registry_v2, approval_key, issue_identity, _, _, _ = _api()
    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        initialize_registry_v2(conn)
        conn.execute(
            "INSERT INTO m_pod (pod_id_seq, pod_uuid, approval_date, institution_id, "
            "pod_type_id, rev_num, pod_order_no, pod_id, pod_name, pod_letter_num, "
            "is_latest) VALUES (856, '00000000-0000-4000-8000-000000000856', "
            "'1997-06-01', 2, 2, 0, NULL, NULL, 'Pengembangan Lapangan Semoga', "
            "'Semoga placeholder', 1)"
        )
        conn.execute("DELETE FROM m_pod WHERE pod_id_seq = 856")

        issued = _new_identity(
            conn, issue_identity, approval_key, date(2024, 3, 1), 1
        )

        assert issued.pod_id_seq == 857
        assert conn.execute(
            "SELECT pod_id_seq FROM _pod_identity_ledger WHERE pod_id_seq = 856"
        ).fetchone() is None
    finally:
        conn.close()


def test_identity_issuance_joins_outer_transaction_and_rolls_back_with_it():
    initialize_registry_v2, approval_key, issue_identity, _, _, _ = _api()
    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        initialize_registry_v2(conn)
        conn.execute("BEGIN IMMEDIATE")
        _new_identity(conn, issue_identity, approval_key, date(2024, 3, 1), 1)
        assert conn.in_transaction
        conn.rollback()
        assert conn.execute(
            "SELECT COUNT(*) FROM _pod_identity_ledger"
        ).fetchone()[0] == 0
    finally:
        conn.close()


def test_revision_resolution_uses_explicit_name_then_predecessor_chain():
    _, _, _, explicit_revision_from_name, resolve_revision_number, _ = _api()
    assert explicit_revision_from_name("Revisi-2 POD Bekasap Waterflood") == 2
    assert explicit_revision_from_name(
        "Revisi 2 POD I Lapangan Merakes dan Merakes East"
    ) == 2
    assert explicit_revision_from_name("POD Revisi (ketiga) Lapangan Kaji Semoga") == 3
    assert explicit_revision_from_name("Revisi Kedua POD Pertama Lapangan Mengoepeh") == 2
    assert explicit_revision_from_name("POD Sisi Nubi Revisi 2") == 2
    assert explicit_revision_from_name("Revisi II POD I Lapangan") == 2
    assert explicit_revision_from_name("Revisi IV Lapangan") == 4
    assert explicit_revision_from_name("Revisi ke-2 POD A") == 2
    assert explicit_revision_from_name("Revisi Ke 2 POD A") == 2
    assert explicit_revision_from_name("Revisi POP Sumur Tiung Biru 2-ST") is None
    assert explicit_revision_from_name("Revisi POD I Lapangan Kepodang") is None

    assert resolve_revision_number([]) == 0
    assert resolve_revision_number([2, 7]) == 8
    assert resolve_revision_number([2, 7], pod_name="POD Revisi (ketiga)") == 3
    assert resolve_revision_number([2, 7], explicit_rev_num=1) == 1
    with pytest.raises(ValueError, match="conflicts"):
        resolve_revision_number(
            [2, 7], explicit_rev_num=1, pod_name="POD Revisi 3"
        )
    with pytest.raises(ValueError, match="conflicting"):
        explicit_revision_from_name("Revisi 2 Revisi 3 POD I")
    with pytest.raises(ValueError, match="unsupported explicit"):
        explicit_revision_from_name("Revisi XII POD I")
    with pytest.raises(ValueError, match="between 0 and 9"):
        explicit_revision_from_name("Revisi 10 POD I")
    with pytest.raises(ValueError, match="one-digit"):
        resolve_revision_number([9])
