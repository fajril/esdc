"""Tests for the POD registry store."""

# tests/pod_registry/test_store.py
import sqlite3

import pytest

import esdc.configs as configs
from esdc.pod_registry.store import (
    allocate_pod_id,
    ensure_tables,
    get_esdc_sqlite_path,
    get_sqlite_connection,
    validate_revision_effect,
)


def _patch_dirs(monkeypatch, tmp_path):
    monkeypatch.setattr(configs.Config, "get_db_dir", classmethod(lambda cls: tmp_path))


def test_path_under_db_dir(monkeypatch, tmp_path):
    _patch_dirs(monkeypatch, tmp_path)
    assert get_esdc_sqlite_path() == tmp_path / "esdc.sqlite"


def test_connection_creates_tables_and_enforces_fk(monkeypatch, tmp_path):
    _patch_dirs(monkeypatch, tmp_path)
    conn = get_sqlite_connection()
    try:
        tables = {
            r["name"]
            for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        assert {
            "m_pod",
            "project_pod",
            "pod_revision",
            "r_institution",
            "r_pod_type",
            "pod_document",
        } <= tables
        # FK enforced: project_pod row without m_pod parent must fail
        try:
            conn.execute(
                "INSERT INTO project_pod (pod_id, project_id) VALUES (999, 'P-X')"
            )
            raise AssertionError("expected IntegrityError")
        except sqlite3.IntegrityError:
            pass
    finally:
        conn.close()


def test_pod_document_fk_and_unique(monkeypatch, tmp_path):
    _patch_dirs(monkeypatch, tmp_path)
    conn = get_sqlite_connection()
    try:
        # FK: doc link without m_pod parent must fail
        try:
            conn.execute(
                "INSERT INTO pod_document (pod_id, doc_id) VALUES (999, 'abc123')"
            )
            raise AssertionError("expected IntegrityError")
        except sqlite3.IntegrityError:
            pass
        conn.execute(
            "INSERT INTO r_institution (code, institution) VALUES (3, 'BP Migas')"
        )
        conn.execute(
            "INSERT INTO r_pod_type (code, pod_type) VALUES (2, 'POD/Waterflood/EOR')"
        )
        conn.execute(
            "INSERT INTO m_pod (id, pod_id, pod_name, approval_date,"
            " institution_code, pod_type_code, rev_num, approval_seq)"
            " VALUES (1, 'PL-2003-0001-3-2-0', 'POD A', '2003-03-14', 3, 2, 0, 1)"
        )
        conn.execute("INSERT INTO pod_document (pod_id, doc_id) VALUES (1, 'abc123')")
        # UNIQUE: same pair twice must fail
        try:
            conn.execute(
                "INSERT INTO pod_document (pod_id, doc_id) VALUES (1, 'abc123')"
            )
            raise AssertionError("expected IntegrityError")
        except sqlite3.IntegrityError:
            pass
        # different doc for same pod is fine (many-to-many)
        conn.execute("INSERT INTO pod_document (pod_id, doc_id) VALUES (1, 'def456')")
    finally:
        conn.close()


def test_m_pod_unique_pod_id(monkeypatch, tmp_path):
    _patch_dirs(monkeypatch, tmp_path)
    conn = get_sqlite_connection()
    try:
        conn.execute(
            "INSERT INTO r_institution (code, institution) VALUES (3, 'BP Migas')"
        )
        conn.execute(
            "INSERT INTO r_pod_type (code, pod_type) VALUES (2, 'POD/Waterflood/EOR')"
        )
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
            raise AssertionError("expected IntegrityError")
        except sqlite3.IntegrityError:
            pass
    finally:
        conn.close()


def _seed_refs(conn):
    conn.execute(
        "INSERT INTO r_institution (code, institution) VALUES (4, 'SKK Migas')"
    )
    conn.execute("INSERT INTO r_pod_type (code, pod_type) VALUES (3, 'POFD/OPL/OPLL')")


def _create_legacy_pod_revision_db(tmp_path):
    """Build a db containing only the old two-column pod_revision schema."""
    db_path = tmp_path / "legacy.sqlite"
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE pod_revision (
            successor_id   TEXT NOT NULL REFERENCES m_pod(pod_id),
            predecessor_id TEXT NOT NULL REFERENCES m_pod(pod_id),
            UNIQUE (successor_id, predecessor_id),
            CHECK (successor_id <> predecessor_id)
        );
        """
    )
    conn.execute(
        "INSERT INTO pod_revision (successor_id, predecessor_id)"
        " VALUES ('PL-2020-0001-1-1-0', 'PL-2020-0001-1-1-1')"
    )
    conn.commit()
    conn.close()
    return db_path


def _assert_pod_revision_check_behavior(conn):
    """Behavioral proof that effect/boolean CHECK constraints are enforced.

    Accepts allowed effect values and valid (null/0/1) booleans; rejects
    out-of-range effect and boolean values on INSERT and UPDATE.
    """
    _seed_refs(conn)
    for i in range(12):
        conn.execute(
            "INSERT INTO m_pod (id, pod_id, pod_name, approval_date,"
            " institution_code, pod_type_code, rev_num, approval_seq)"
            " VALUES (?,?,?,?,?,?,?,?)",
            (i + 1, f"PL-2020-0001-4-3-{i}", f"POD {i}", "2020-01-01", 4, 3, 0, i + 1),
        )

    for n, (effect, flag) in enumerate(
        (("unknown", None), ("partial_amendment", 1), ("full_replacement", 0)),
        start=1,
    ):
        conn.execute(
            "INSERT INTO pod_revision (successor_id, predecessor_id,"
            " revision_effect, previous_remains_valid) VALUES (?,?,?,?)",
            (f"PL-2020-0001-4-3-{n}", f"PL-2020-0001-4-3-{n - 1}", effect, flag),
        )

    for n, (effect, flag) in enumerate(
        (("replace", None), ("unknown", 2)),
        start=10,
    ):
        try:
            conn.execute(
                "INSERT INTO pod_revision (successor_id, predecessor_id,"
                " revision_effect, previous_remains_valid) VALUES (?,?,?,?)",
                (
                    f"PL-2020-0001-4-3-{n}",
                    f"PL-2020-0001-4-3-{n - 1}",
                    effect,
                    flag,
                ),
            )
            raise AssertionError(f"expected IntegrityError for ({effect!r}, {flag!r})")
        except sqlite3.IntegrityError:
            pass

    for effect, flag in (("replace", None), ("partial_amendment", 2)):
        try:
            conn.execute(
                "UPDATE pod_revision SET revision_effect=?, previous_remains_valid=?"
                " WHERE revision_effect='unknown'",
                (effect, flag),
            )
            raise AssertionError(
                f"expected IntegrityError updating to ({effect!r}, {flag!r})"
            )
        except sqlite3.IntegrityError:
            pass


def test_migrate_legacy_pod_revision_gains_temporal_columns(tmp_path):
    db_path = _create_legacy_pod_revision_db(tmp_path)
    conn = sqlite3.connect(db_path)
    try:
        ensure_tables(conn)
        cols = {row[1]: row for row in conn.execute("PRAGMA table_info(pod_revision)")}
        assert cols["revision_effect"][2] == "TEXT"
        assert cols["revision_effect"][3] == 1
        assert cols["revision_effect"][4] == "'unknown'"
        assert cols["effective_date"][2] == "TEXT"
        assert cols["amended_scope"][2] == "TEXT"
        assert cols["previous_remains_valid"][2] == "INTEGER"
        # existing row preserved as unknown, no supersession inferred
        row = conn.execute(
            "SELECT revision_effect, effective_date, amended_scope,"
            " previous_remains_valid FROM pod_revision"
        ).fetchone()
        assert tuple(row) == ("unknown", None, None, None)
    finally:
        conn.close()


def test_migrate_legacy_pod_revision_is_idempotent(tmp_path):
    db_path = _create_legacy_pod_revision_db(tmp_path)
    conn = sqlite3.connect(db_path)
    try:
        ensure_tables(conn)
        ensure_tables(conn)
        cols = {row[1] for row in conn.execute("PRAGMA table_info(pod_revision)")}
        assert {
            "revision_effect",
            "effective_date",
            "amended_scope",
            "previous_remains_valid",
        } <= cols
    finally:
        conn.close()


def test_fresh_pod_revision_has_temporal_columns(monkeypatch, tmp_path):
    _patch_dirs(monkeypatch, tmp_path)
    conn = get_sqlite_connection()
    try:
        cols = {row[1]: row for row in conn.execute("PRAGMA table_info(pod_revision)")}
        for name, typ, notnull, default in [
            ("revision_effect", "TEXT", 1, "'unknown'"),
            ("effective_date", "TEXT", 0, None),
            ("amended_scope", "TEXT", 0, None),
            ("previous_remains_valid", "INTEGER", 0, None),
        ]:
            assert name in cols
            assert cols[name][2] == typ
            assert cols[name][3] == notnull
            assert cols[name][4] == default
    finally:
        conn.close()


def test_fresh_pod_revision_enforces_effect_and_boolean_checks(monkeypatch, tmp_path):
    _patch_dirs(monkeypatch, tmp_path)
    conn = get_sqlite_connection()
    try:
        _assert_pod_revision_check_behavior(conn)
    finally:
        conn.close()


def test_migrated_pod_revision_enforces_effect_and_boolean_checks(tmp_path):
    db_path = _create_legacy_pod_revision_db(tmp_path)
    conn = sqlite3.connect(db_path)
    try:
        ensure_tables(conn)
        _assert_pod_revision_check_behavior(conn)
    finally:
        conn.close()


def test_validate_revision_effect_accepts_valid_combinations():
    for effect, flag in (
        ("unknown", None),
        ("full_replacement", False),
        ("partial_amendment", True),
    ):
        validate_revision_effect(effect, flag)


@pytest.mark.parametrize(
    ("effect", "flag"),
    [
        ("unknown", True),
        ("unknown", False),
        ("full_replacement", True),
        ("full_replacement", None),
        ("partial_amendment", False),
        ("partial_amendment", None),
        ("replace", None),
        ("replace", True),
    ],
)
def test_validate_revision_effect_rejects_invalid_combinations(effect, flag):
    with pytest.raises(ValueError):
        validate_revision_effect(effect, flag)


def test_partial_amendment_with_previous_valid_is_accepted(monkeypatch, tmp_path):
    _patch_dirs(monkeypatch, tmp_path)
    conn = get_sqlite_connection()
    try:
        _seed_refs(conn)
        for row in (
            (1, "PL-2020-0001-4-3-0", "POD A", "2020-01-01", 4, 3, 0, 1),
            (2, "PL-2020-0001-4-3-1", "POD B", "2021-01-01", 4, 3, 1, 2),
        ):
            conn.execute(
                "INSERT INTO m_pod (id, pod_id, pod_name, approval_date,"
                " institution_code, pod_type_code, rev_num, approval_seq)"
                " VALUES (?,?,?,?,?,?,?,?)",
                row,
            )
        validate_revision_effect("partial_amendment", True)
        conn.execute(
            "INSERT INTO pod_revision (successor_id, predecessor_id,"
            " revision_effect, previous_remains_valid)"
            " VALUES ('PL-2020-0001-4-3-1', 'PL-2020-0001-4-3-0',"
            " 'partial_amendment', 1)"
        )
        row = conn.execute(
            "SELECT revision_effect, previous_remains_valid FROM pod_revision"
        ).fetchone()
        assert row["revision_effect"] == "partial_amendment"
        assert row["previous_remains_valid"] == 1
    finally:
        conn.close()


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
