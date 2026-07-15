# tests/pod_registry/test_store.py
import sqlite3

import esdc.configs as configs
from esdc.pod_registry.store import get_pod_sqlite_path, get_sqlite_connection


def _patch_dirs(monkeypatch, tmp_path):
    monkeypatch.setattr(configs.Config, "get_db_dir", classmethod(lambda cls: tmp_path))


def test_path_under_db_dir(monkeypatch, tmp_path):
    _patch_dirs(monkeypatch, tmp_path)
    assert get_pod_sqlite_path() == tmp_path / "pod.sqlite"


def test_connection_creates_tables_and_enforces_fk(monkeypatch, tmp_path):
    _patch_dirs(monkeypatch, tmp_path)
    conn = get_sqlite_connection()
    try:
        tables = {
            r["name"]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        assert {"m_pod", "project_pod", "pod_revision", "r_institution", "r_pod_type"} <= tables
        # FK enforced: project_pod row without m_pod parent must fail
        try:
            conn.execute("INSERT INTO project_pod (pod_id, project_id) VALUES (999, 'P-X')")
            assert False, "expected IntegrityError"
        except sqlite3.IntegrityError:
            pass
    finally:
        conn.close()


def test_m_pod_unique_pod_id(monkeypatch, tmp_path):
    _patch_dirs(monkeypatch, tmp_path)
    conn = get_sqlite_connection()
    try:
        conn.execute("INSERT INTO r_institution (code, institution) VALUES (3, 'BP Migas')")
        conn.execute("INSERT INTO r_pod_type (code, pod_type) VALUES (2, 'POD/Waterflood/EOR')")
        row = (1, "PL-2003-0001-3-2-0", "POD A", "L1", "2003-03-14", 3, 2, 0, 1)
        conn.execute(
            "INSERT INTO m_pod (id, pod_id, pod_name, pod_letter_num, approval_date,"
            " institution_code, pod_type_code, rev_num, approval_seq)"
            " VALUES (?,?,?,?,?,?,?,?,?)",
            row,
        )
        try:
            conn.execute(
                "INSERT INTO m_pod (id, pod_id, pod_name, pod_letter_num, approval_date,"
                " institution_code, pod_type_code, rev_num, approval_seq)"
                " VALUES (2, 'PL-2003-0001-3-2-0', 'POD B', NULL, '2003-04-01', 3, 2, 0, 2)"
            )
            assert False, "expected IntegrityError"
        except sqlite3.IntegrityError:
            pass
    finally:
        conn.close()


from esdc.pod_registry.store import allocate_pod_id


def _seed_refs(conn):
    conn.execute("INSERT INTO r_institution (code, institution) VALUES (4, 'SKK Migas')")
    conn.execute("INSERT INTO r_pod_type (code, pod_type) VALUES (3, 'POFD/OPL/OPLL')")


def test_allocate_first_id(monkeypatch, tmp_path):
    _patch_dirs(monkeypatch, tmp_path)
    conn = get_sqlite_connection()
    try:
        pod_id, seq = allocate_pod_id(conn, "2026-07-15", 4, 3, 0)
        assert pod_id == "PL-2026-0001-4-3-0"
        assert seq == 1
    finally:
        conn.close()


def test_allocate_increments_from_max_seq_ignoring_gaps(monkeypatch, tmp_path):
    _patch_dirs(monkeypatch, tmp_path)
    conn = get_sqlite_connection()
    try:
        _seed_refs(conn)
        conn.execute(
            "INSERT INTO m_pod (id, pod_id, pod_name, approval_date,"
            " institution_code, pod_type_code, rev_num, approval_seq)"
            " VALUES (10, 'PL-2020-0741-4-3-0', 'X', '2020-01-01', 4, 3, 0, 741)"
        )
        pod_id, seq = allocate_pod_id(conn, "2026-07-15", 4, 3, 1)
        assert seq == 742
        assert pod_id == "PL-2026-0742-4-3-1"
    finally:
        conn.close()
