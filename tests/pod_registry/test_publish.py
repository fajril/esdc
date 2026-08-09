"""Tests for POD publication."""

# tests/pod_registry/test_publish.py
from datetime import date

import duckdb
import pandas as pd
import pytest

import esdc.configs as configs
from esdc.dbmanager import get_duckdb_connection
from esdc.loaders import (
    _METADATA_TABLE,
    POD_SCHEMA_PATH,
    load_schema_from_yaml,
    lookup_loaded_schema,
)
from esdc.pod_registry.publish import publish_pod_registry
from esdc.pod_registry.store import get_sqlite_connection
from esdc.pod_registry.value_cases import replace_pod_value_cases


def _patch_dirs(monkeypatch, tmp_path):
    monkeypatch.setattr(configs.Config, "get_db_dir", classmethod(lambda cls: tmp_path))
    monkeypatch.setattr(
        configs.Config, "get_db_file", classmethod(lambda cls: tmp_path / "esdc.duckdb")
    )


def _seed(monkeypatch, tmp_path):
    _patch_dirs(monkeypatch, tmp_path)
    conn = get_sqlite_connection()
    conn.execute("INSERT INTO r_institution (code, institution) VALUES (3, 'BP Migas')")
    conn.execute(
        "INSERT INTO r_pod_type (code, pod_type) VALUES (2, 'POD/Waterflood/EOR')"
    )
    conn.executemany(
        "INSERT INTO m_pod (id, pod_id, pod_name, pod_letter_num, approval_date,"
        " institution_code, pod_type_code, rev_num, approval_seq)"
        " VALUES (?,?,?,?,?,?,?,?,?)",
        [
            (
                645,
                "PL-2003-0005-3-2-0",
                "POD Mengoepeh",
                "294/BP",
                "2003-11-21",
                3,
                2,
                0,
                5,
            ),
            (
                700,
                "PL-2005-0051-3-2-1",
                "POD Mengoepeh Rev",
                "51/BP",
                "2005-06-01",
                3,
                2,
                1,
                51,
            ),
        ],
    )
    conn.execute(
        "INSERT INTO project_pod (pod_id, project_id) VALUES (645, 'P-2403431-01')"
    )
    conn.execute(
        "INSERT INTO pod_document (pod_id, doc_id) VALUES (645, 'ccbd4f3f27635c76')"
    )
    conn.execute(
        "INSERT INTO pod_revision (successor_id, predecessor_id, revision_effect,"
        " effective_date, amended_scope, previous_remains_valid)"
        " VALUES ('PL-2005-0051-3-2-1', 'PL-2003-0005-3-2-0',"
        " 'full_replacement', '2005-06-01', 'Full scope', 0)"
    )
    conn.commit()
    conn.close()


def test_publish_creates_denormalized_tables(monkeypatch, tmp_path):
    _seed(monkeypatch, tmp_path)
    results = publish_pod_registry()
    assert {r.table_name for r in results} == {
        "pod_registry",
        "pod_project",
        "pod_document",
        "pod_revision",
        "pod_value_case",
    }

    conn = get_duckdb_connection(tmp_path / "esdc.duckdb", read_only=True)
    try:
        rows = conn.execute(
            "SELECT pod_id, institution, pod_type, preceded_by, revised_by,"
            " superseded_by FROM pod_registry ORDER BY approval_seq"
        ).fetchall()
        assert rows[0][1] == "BP Migas"
        assert rows[0][4] == "PL-2005-0051-3-2-1"  # original revised by rev
        assert rows[0][5] == "PL-2005-0051-3-2-1"  # full_replacement supersedes
        assert rows[1][3] == "PL-2003-0005-3-2-0"  # rev preceded by original
        assert rows[0][3] is None and rows[1][4] is None and rows[1][5] is None

        link = conn.execute("SELECT pod_id, project_id FROM pod_project").fetchall()
        assert link == [("PL-2003-0005-3-2-0", "P-2403431-01")]

        # canonical PL-... pod_id in the doc snapshot, not the surrogate int
        docs = conn.execute("SELECT pod_id, doc_id FROM pod_document").fetchall()
        assert docs == [("PL-2003-0005-3-2-0", "ccbd4f3f27635c76")]

        meta = conn.execute(
            f"SELECT table_name FROM {_METADATA_TABLE} ORDER BY table_name"
        ).fetchall()
        assert ("pod_registry",) in meta and ("pod_project",) in meta
        assert ("pod_document",) in meta
    finally:
        conn.close()


def test_publish_is_idempotent(monkeypatch, tmp_path):
    _seed(monkeypatch, tmp_path)
    publish_pod_registry()
    publish_pod_registry()  # replaces, no duplicates
    conn = get_duckdb_connection(tmp_path / "esdc.duckdb", read_only=True)
    try:
        count_row = conn.execute("SELECT COUNT(*) FROM pod_registry").fetchone()
        assert count_row is not None
        assert count_row[0] == 2
    finally:
        conn.close()


def test_publish_does_not_treat_legacy_revision_as_supersession(monkeypatch, tmp_path):
    """A legacy pod_revision row must not auto-populate superseded_by.

    The two-column pod_revision row only records that a successor exists;
    it carries no explicit full-replacement evidence. Publishing must not
    turn that record into a supersession claim. Today `_REGISTRY_SQL`
    eagerly joins pod_revision into superseded_by, so this fails on the
    current behavior -- not because a column is absent.
    """
    _patch_dirs(monkeypatch, tmp_path)
    conn = get_sqlite_connection()
    conn.execute(
        "INSERT INTO r_institution (code, institution) VALUES (1, 'SKK Migas')"
    )
    conn.execute("INSERT INTO r_pod_type (code, pod_type) VALUES (1, 'POD I')")
    conn.executemany(
        "INSERT INTO m_pod (id, pod_id, pod_name, pod_letter_num, approval_date,"
        " institution_code, pod_type_code, rev_num, approval_seq)"
        " VALUES (?, ?, ?, ?, ?, 1, 1, ?, ?)",
        [
            (1, "PL-2024-0001-1-1-0", "POD Alpha", "L-001", "2024-01-15", 0, 1),
            (
                2,
                "PL-2024-0002-1-1-1",
                "POD Alpha Revisi",
                "L-002",
                "2024-06-01",
                1,
                2,
            ),
        ],
    )
    conn.execute(
        "INSERT INTO pod_revision (successor_id, predecessor_id)"
        " VALUES ('PL-2024-0002-1-1-1', 'PL-2024-0001-1-1-0')"
    )
    conn.commit()
    conn.close()

    publish_pod_registry()

    conn = get_duckdb_connection(tmp_path / "esdc.duckdb", read_only=True)
    try:
        rows = conn.execute(
            "SELECT pod_id, superseded_by FROM pod_registry ORDER BY pod_id"
        ).fetchall()
    finally:
        conn.close()
    assert all(superseded_by is None for _, superseded_by in rows)


def _seed_revision(
    monkeypatch, tmp_path, *, effect, remains_valid, effective_date="2024-06-01"
):
    _patch_dirs(monkeypatch, tmp_path)
    conn = get_sqlite_connection()
    conn.execute(
        "INSERT INTO r_institution (code, institution) VALUES (1, 'SKK Migas')"
    )
    conn.execute("INSERT INTO r_pod_type (code, pod_type) VALUES (1, 'POD I')")
    conn.executemany(
        "INSERT INTO m_pod (id, pod_id, pod_name, pod_letter_num, approval_date,"
        " institution_code, pod_type_code, rev_num, approval_seq)"
        " VALUES (?, ?, ?, ?, ?, 1, 1, ?, ?)",
        [
            (1, "PL-2024-0001-1-1-0", "POD Alpha", "L-001", "2024-01-15", 0, 1),
            (
                2,
                "PL-2024-0002-1-1-1",
                "POD Alpha Revisi",
                "L-002",
                "2024-06-01",
                1,
                2,
            ),
        ],
    )
    conn.execute(
        "INSERT INTO pod_revision (successor_id, predecessor_id, revision_effect,"
        " effective_date, amended_scope, previous_remains_valid)"
        " VALUES (?,?,?,?,?,?)",
        (
            "PL-2024-0002-1-1-1",
            "PL-2024-0001-1-1-0",
            effect,
            effective_date,
            "Amended scope B",
            remains_valid,
        ),
    )
    conn.commit()
    conn.close()


def _registry_row(monkeypatch, tmp_path, pod_id, column):
    _patch_dirs(monkeypatch, tmp_path)
    conn = get_duckdb_connection(tmp_path / "esdc.duckdb", read_only=True)
    try:
        row = conn.execute(
            f"SELECT {column} FROM pod_registry WHERE pod_id = ?", [pod_id]
        ).fetchone()
    finally:
        conn.close()
    assert row is not None
    return row[0]


def test_publish_full_replacement_populates_revised_by_and_superseded_by(
    monkeypatch, tmp_path
):
    _seed_revision(monkeypatch, tmp_path, effect="full_replacement", remains_valid=0)
    publish_pod_registry()
    assert _registry_row(monkeypatch, tmp_path, "PL-2024-0001-1-1-0", "revised_by") == (
        "PL-2024-0002-1-1-1"
    )
    assert (
        _registry_row(monkeypatch, tmp_path, "PL-2024-0001-1-1-0", "superseded_by")
        == "PL-2024-0002-1-1-1"
    )
    assert _registry_row(
        monkeypatch, tmp_path, "PL-2024-0002-1-1-1", "preceded_by"
    ) == ("PL-2024-0001-1-1-0")
    assert (
        _registry_row(monkeypatch, tmp_path, "PL-2024-0002-1-1-1", "revised_by") is None
    )
    assert (
        _registry_row(monkeypatch, tmp_path, "PL-2024-0002-1-1-1", "superseded_by")
        is None
    )


def test_publish_partial_amendment_revises_but_does_not_supersede(
    monkeypatch, tmp_path
):
    _seed_revision(monkeypatch, tmp_path, effect="partial_amendment", remains_valid=1)
    publish_pod_registry()
    assert _registry_row(monkeypatch, tmp_path, "PL-2024-0001-1-1-0", "revised_by") == (
        "PL-2024-0002-1-1-1"
    )
    assert (
        _registry_row(monkeypatch, tmp_path, "PL-2024-0001-1-1-0", "superseded_by")
        is None
    )


def test_publish_creates_pod_revision_projection(monkeypatch, tmp_path):
    _seed_revision(monkeypatch, tmp_path, effect="partial_amendment", remains_valid=1)
    publish_pod_registry()
    conn = get_duckdb_connection(tmp_path / "esdc.duckdb", read_only=True)
    try:
        rows = conn.execute(
            "SELECT successor_id, predecessor_id, revision_effect, effective_date,"
            " amended_scope, previous_remains_valid FROM pod_revision"
        ).fetchall()
    finally:
        conn.close()
    assert rows == [
        (
            "PL-2024-0002-1-1-1",
            "PL-2024-0001-1-1-0",
            "partial_amendment",
            date(2024, 6, 1),
            "Amended scope B",
            1,
        )
    ]


def test_publish_projects_revision_validity_as_boolean(monkeypatch, tmp_path):
    """Catch revision metadata claiming boolean while DuckDB stores INTEGER."""
    _seed_revision(monkeypatch, tmp_path, effect="partial_amendment", remains_valid=1)
    publish_pod_registry()
    conn = get_duckdb_connection(tmp_path / "esdc.duckdb", read_only=True)
    try:
        row = conn.execute(
            "SELECT data_type FROM information_schema.columns"
            " WHERE table_name = 'pod_revision'"
            " AND column_name = 'previous_remains_valid'"
        ).fetchone()
    finally:
        conn.close()
    assert row == ("BOOLEAN",)


def _seed_value_cases(monkeypatch, tmp_path):
    _seed(monkeypatch, tmp_path)
    schema = load_schema_from_yaml(POD_SCHEMA_PATH)
    plan_df = pd.DataFrame(
        [
            {
                "pod_id": "PL-2003-0005-3-2-0",
                "report_date": "2024-06-15",
                "as_of_date": "2024-12-31",
                "pod_scope": "Development",
                "lifting_oil": 1.0,
            }
        ]
    )
    monitoring_df = pd.DataFrame(
        [
            {
                "pod_id": "PL-2003-0005-3-2-0",
                "case_type": "outlook",
                "report_date": "2024-09-01",
                "as_of_date": "2024-12-31",
                "lifting_oil": 2.0,
            }
        ]
    )
    replace_pod_value_cases(None, plan_df, monitoring_df, schema)


def _view_names(conn):
    return {
        r[0]
        for r in conn.execute(
            "SELECT view_name FROM duckdb_views()"
            " WHERE database_name = current_database()"
        ).fetchall()
    }


def test_publish_creates_economics_projection(monkeypatch, tmp_path):
    _seed_value_cases(monkeypatch, tmp_path)
    publish_pod_registry()
    conn = get_duckdb_connection(tmp_path / "esdc.duckdb", read_only=True)
    try:
        vc = conn.execute(
            "SELECT pod_id, case_type, as_of_date, lifting_oil FROM pod_value_case"
            " ORDER BY case_type"
        ).fetchall()
        assert vc == [
            ("PL-2003-0005-3-2-0", "outlook", date(2024, 12, 31), 2.0),
            ("PL-2003-0005-3-2-0", "plan", date(2024, 12, 31), 1.0),
        ]
        eco = conn.execute(
            "SELECT case_type, pod_id, pod_name, pod_scope, lifting_oil, as_of_date"
            " FROM pod_economics ORDER BY case_type"
        ).fetchall()
        assert eco[0][0] == "outlook"
        assert eco[0][2] == "POD Mengoepeh"  # registry metadata joined in
        assert eco[0][3] is None
        assert eco[0][4] == 2.0
        assert eco[1][0] == "plan"
        assert eco[1][1] == "PL-2003-0005-3-2-0"
        assert eco[1][3] == "Development"
        assert eco[1][5] == date(2024, 12, 31)
    finally:
        conn.close()


def test_publish_creates_economics_views(monkeypatch, tmp_path):
    _seed_value_cases(monkeypatch, tmp_path)
    publish_pod_registry()
    conn = get_duckdb_connection(tmp_path / "esdc.duckdb", read_only=True)
    try:
        views = _view_names(conn)
        assert {
            "pod_economics",
            "pod_plan",
            "pod_monitoring",
            "pod_project_economics",
        } <= views
        assert conn.execute("SELECT case_type FROM pod_plan").fetchall() == [("plan",)]
        assert conn.execute("SELECT case_type FROM pod_monitoring").fetchall() == [
            ("outlook",)
        ]
        assert sorted(
            conn.execute("SELECT project_id FROM pod_project_economics").fetchall()
        ) == [
            ("P-2403431-01",),
            ("P-2403431-01",),
        ]
    finally:
        conn.close()


def test_publish_refreshes_economics_schema_metadata(monkeypatch, tmp_path):
    _seed_value_cases(monkeypatch, tmp_path)
    publish_pod_registry()
    conn = get_duckdb_connection(tmp_path / "esdc.duckdb", read_only=True)
    try:
        names = {
            r[0]
            for r in conn.execute(
                f"SELECT table_name FROM {_METADATA_TABLE}"
            ).fetchall()
        }
        assert {
            "pod_economics",
            "pod_plan",
            "pod_monitoring",
            "pod_project_economics",
        } <= names
    finally:
        conn.close()


def test_publish_value_case_metadata_includes_metrics(monkeypatch, tmp_path):
    """Catch pod_value_case metadata omitting columns present in its table."""
    _seed_value_cases(monkeypatch, tmp_path)
    publish_pod_registry()
    schema_text = lookup_loaded_schema("pod_value_case")
    assert schema_text is not None
    assert "lifting_oil" in schema_text


def test_publish_drops_stale_legacy_pod_base_tables(monkeypatch, tmp_path):
    _seed_value_cases(monkeypatch, tmp_path)
    conn = get_duckdb_connection(tmp_path / "esdc.duckdb", read_only=False)
    conn.execute("CREATE TABLE pod_plan (pod_id VARCHAR)")
    conn.execute("CREATE TABLE pod_monitoring (pod_id VARCHAR)")
    conn.close()

    publish_pod_registry()

    conn = get_duckdb_connection(tmp_path / "esdc.duckdb", read_only=True)
    try:
        tables = {
            r[0]
            for r in conn.execute(
                "SELECT table_name FROM duckdb_tables()"
                " WHERE database_name = current_database()"
            ).fetchall()
        }
        assert "pod_plan" not in tables
        assert "pod_monitoring" not in tables
        assert "pod_economics" in _view_names(conn)
    finally:
        conn.close()


def _sqlite_value_cases():
    conn = get_sqlite_connection()
    try:
        return sorted(
            tuple(row)
            for row in conn.execute(
                "SELECT pod_id, case_type, as_of_date FROM pod_value_case"
            ).fetchall()
        )
    finally:
        conn.close()


def _duckdb_value_cases(tmp_path):
    conn = get_duckdb_connection(tmp_path / "esdc.duckdb", read_only=True)
    try:
        return sorted(
            conn.execute(
                "SELECT pod_id, case_type, CAST(as_of_date AS VARCHAR)"
                " FROM pod_value_case"
            ).fetchall()
        )
    finally:
        conn.close()


def test_publish_projection_tracks_sqlite_value_case_changes(monkeypatch, tmp_path):
    _seed_value_cases(monkeypatch, tmp_path)
    publish_pod_registry()
    assert _duckdb_value_cases(tmp_path) == _sqlite_value_cases()

    conn = get_sqlite_connection()
    conn.execute(
        "INSERT INTO pod_value_case (pod_id, case_type, report_date, as_of_date,"
        " pod_scope, lifting_oil) VALUES (?,?,?,?,?,?)",
        (
            "PL-2003-0005-3-2-0",
            "actual",
            "2025-03-01",
            "2024-12-31",
            "Development",
            3.0,
        ),
    )
    conn.commit()
    conn.close()

    publish_pod_registry()
    assert _duckdb_value_cases(tmp_path) == _sqlite_value_cases()
    assert ("PL-2003-0005-3-2-0", "actual", "2024-12-31") in _duckdb_value_cases(
        tmp_path
    )


def test_publish_removes_deleted_sqlite_value_case(monkeypatch, tmp_path):
    _seed_value_cases(monkeypatch, tmp_path)
    publish_pod_registry()
    assert len(_duckdb_value_cases(tmp_path)) == 2

    conn = get_sqlite_connection()
    conn.execute("DELETE FROM pod_value_case WHERE case_type = 'plan'")
    conn.commit()
    conn.close()

    publish_pod_registry()
    rows = _duckdb_value_cases(tmp_path)
    assert rows == _sqlite_value_cases()
    assert ("PL-2003-0005-3-2-0", "plan", "2024-12-31") not in rows


def test_publish_value_case_projection_is_idempotent(monkeypatch, tmp_path):
    _seed_value_cases(monkeypatch, tmp_path)
    publish_pod_registry()
    first = _duckdb_value_cases(tmp_path)
    publish_pod_registry()
    assert _duckdb_value_cases(tmp_path) == first


def test_publish_raises_on_value_case_table_with_wrong_schema(monkeypatch, tmp_path):
    """A pre-existing pod_value_case lacking the identifier columns must fail.

    Registry seeded normally but pod_value_case pre-created with an
    unrelated column: publish reads it, and the economics views fail to
    bind (no pod_id). The read path must not silently swallow the error.
    """
    _seed(monkeypatch, tmp_path)
    conn = get_sqlite_connection()
    conn.execute("CREATE TABLE pod_value_case (foo TEXT)")
    conn.commit()
    conn.close()

    with pytest.raises(duckdb.Error):
        publish_pod_registry()
