"""DB-layer constraint tests for the current m_pod / pod_document schema."""

import contextlib
import sqlite3
from pathlib import Path

import pytest

from esdc.pod_registry.changesets import apply_changeset
from esdc.pod_registry.store import get_sqlite_connection


def _bootstrap(tmp_path: Path) -> Path:
    """Initialize the store and create one POD; return the database path."""
    path = tmp_path / "source.sqlite"
    get_sqlite_connection(path).close()

    result = apply_changeset(
        "m_pod",
        {
            "source_generation": 0,
            "inserts": [
                {
                    "approval_date": "2024-01-01",
                    "institution_id": 1,
                    "pod_type_id": 1,
                    "pod_name": "POD Example",
                    "pod_letter_num": "001",
                }
            ],
        },
        path,
    )
    assert result.ok
    assert result.errors == []
    return path


def _row(conn: sqlite3.Connection, sql: str, params: tuple = ()) -> tuple:
    row = conn.execute(sql, params).fetchone()
    assert row is not None
    return tuple(row)


def test_pod_document_foreign_key_and_composite_pk(tmp_path):
    """pod_document must reference a real m_pod row and keep its composite PK unique."""
    path = _bootstrap(tmp_path)
    with contextlib.closing(get_sqlite_connection(path)) as conn:
        assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1

        # Unknown pod_id_seq violates the FK to m_pod.
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO pod_document (pod_id_seq, doc_id) VALUES (?, ?)",
                (99999, "D1"),
            )
        conn.rollback()

        # A link to the actual issued POD row commits cleanly.
        pod = _row(conn, "SELECT pod_id_seq FROM m_pod WHERE is_latest = 1")
        conn.execute(
            "INSERT INTO pod_document (pod_id_seq, doc_id) VALUES (?, ?)",
            (pod[0], "D1"),
        )
        conn.commit()

        # Repeating the composite PK (pod_id_seq, doc_id) is rejected.
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO pod_document (pod_id_seq, doc_id) VALUES (?, ?)",
                (pod[0], "D1"),
            )
        conn.rollback()

        assert _row(conn, "SELECT pod_id_seq, doc_id FROM pod_document") == (
            pod[0],
            "D1",
        )


def test_issued_pod_identity_is_immutable(tmp_path):
    """pod_id and pod_uuid of an issued POD cannot be changed."""
    path = _bootstrap(tmp_path)
    with contextlib.closing(get_sqlite_connection(path)) as conn:
        before = _row(conn, "SELECT pod_id_seq, pod_id, pod_uuid FROM m_pod")
        ledger_before = [
            tuple(row)
            for row in conn.execute(
                "SELECT * FROM _pod_identity_ledger ORDER BY pod_id_seq"
            )
        ]

        for sql in (
            "UPDATE m_pod SET pod_id = 'X9' WHERE pod_id_seq = ?",
            "UPDATE m_pod SET pod_uuid = 'deadbeef' WHERE pod_id_seq = ?",
        ):
            with pytest.raises(sqlite3.IntegrityError):
                conn.execute(sql, (before[0],))
            conn.rollback()

        assert [
            tuple(row)
            for row in conn.execute(
                "SELECT * FROM _pod_identity_ledger ORDER BY pod_id_seq"
            )
        ] == ledger_before
        assert _row(conn, "SELECT pod_id_seq, pod_id, pod_uuid FROM m_pod") == before
        assert conn.execute("SELECT COUNT(*) FROM m_pod").fetchone()[0] == 1


def test_referenced_pod_cannot_be_deleted(tmp_path):
    """A POD still referenced by pod_document cannot be removed."""
    path = _bootstrap(tmp_path)
    with contextlib.closing(get_sqlite_connection(path)) as conn:
        pod = _row(conn, "SELECT pod_id_seq FROM m_pod")
        conn.execute(
            "INSERT INTO pod_document (pod_id_seq, doc_id) VALUES (?, ?)",
            (pod[0], "D1"),
        )
        conn.commit()

        with pytest.raises(sqlite3.IntegrityError):
            conn.execute("DELETE FROM m_pod WHERE pod_id_seq = ?", (pod[0],))
        conn.rollback()

        assert _row(conn, "SELECT pod_id_seq FROM m_pod") == pod
        assert _row(
            conn, "SELECT doc_id FROM pod_document WHERE pod_id_seq = ?", pod
        ) == ("D1",)
