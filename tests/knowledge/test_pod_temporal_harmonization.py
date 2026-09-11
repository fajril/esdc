"""End-to-end temporal POD harmonization: SQLite truth == DuckDB/Ladybug projections.

One small POD world -- original, partial amendment, full replacement -- with
plan/actual/outlook value cases and project/document links, asserted
identically from SQLite (source of truth), DuckDB (read projection), and the
disposable Ladybug instance graph. Then both projections are torn down and
rebuilt from the same SQLite file and must restore the same answers.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd
import pytest

from esdc.chat.domain_knowledge.instance_graph import InstanceGraphManager
from esdc.knowledge.linker import run_deterministic_linking
from esdc.knowledge.store import KnowledgeStore
from esdc.loaders import POD_SCHEMA_PATH, load_schema_from_yaml
from esdc.pod_registry.publish import publish_pod_registry
from esdc.pod_registry.store import get_sqlite_connection
from esdc.pod_registry.value_cases import replace_pod_value_cases

ORIG = "PL-2019-0001-2-2-0"
AMEND = "PL-2020-0003-2-2-0"
FINAL = "PL-2021-0004-2-2-2"
PROJECT = "PRJ-001"
DOC = "DOC-ORIG"


@pytest.fixture(autouse=True)
def _reset_instance_graph_singleton():
    InstanceGraphManager.reset_for_tests()
    yield
    InstanceGraphManager.reset_for_tests()


def _seed_sqlite(sqlite_path: Path) -> None:
    conn = get_sqlite_connection(sqlite_path)
    conn.execute(
        "INSERT INTO r_institution (code, institution) VALUES (2, 'SKK Migas')"
    )
    conn.execute("INSERT INTO r_pod_type (code, pod_type) VALUES (2, 'POD I')")
    conn.executemany(
        "INSERT INTO m_pod (id, pod_id, pod_name, pod_letter_num, approval_date,"
        " institution_code, pod_type_code, rev_num, approval_seq)"
        " VALUES (?,?,?,?,?,2,2,?,?)",
        [
            (1, ORIG, "POD Duri", "SRT-2019/ORIG", "2019-05-01", 0, 1),
            (2, AMEND, "POD Duri Revisi", "SRT-2020/AMEND", "2020-06-01", 1, 2),
            (3, FINAL, "POD Duri Final", "SRT-2021/FINAL", "2021-07-01", 2, 3),
        ],
    )
    conn.execute(
        "INSERT INTO project_pod (pod_id, project_id) VALUES (1, ?)", [PROJECT]
    )
    conn.executemany(
        "INSERT INTO pod_revision (successor_id, predecessor_id, revision_effect,"
        " effective_date, amended_scope, previous_remains_valid)"
        " VALUES (?,?,?,?,?,?)",
        [
            (AMEND, ORIG, "partial_amendment", "2020-06-01", "sec. 3", 1),
            (FINAL, AMEND, "full_replacement", "2021-07-01", "whole pod", 0),
        ],
    )
    conn.execute(
        """
        CREATE TABLE documents (
            doc_id TEXT PRIMARY KEY, file_name TEXT, file_hash TEXT,
            doc_type TEXT, doc_topic TEXT, doc_number TEXT, doc_date TEXT,
            subject TEXT, wk_name TEXT, field_name TEXT, project_name TEXT,
            pod_name TEXT, suggested_pod_ids TEXT, markdown TEXT,
            raw_entities TEXT, metadata TEXT, ingested_at TEXT
        )
        """
    )
    conn.execute(
        "INSERT INTO documents (doc_id, file_name, doc_type, doc_number,"
        " doc_date, subject) VALUES (?, 'orig.pdf', 'letter', 'SRT-2019/ORIG',"
        " '2019-05-01', 'Persetujuan POD Duri')",
        [DOC],
    )
    conn.commit()
    conn.close()


def _seed_duckdb(duckdb_path: Path) -> None:
    dconn = duckdb.connect(str(duckdb_path))
    dconn.execute(
        "CREATE TABLE project_resources (project_id VARCHAR, project_name VARCHAR,"
        " field_name VARCHAR, wk_name VARCHAR, report_year INTEGER,"
        " project_remarks VARCHAR)"
    )
    dconn.execute(
        "INSERT INTO project_resources VALUES (?, 'Duri Steamflood', 'Duri',"
        " 'Rokan', 2024, NULL)",
        [PROJECT],
    )
    dconn.close()


def _replace_value_cases(sqlite_path: Path) -> None:
    schema = load_schema_from_yaml(POD_SCHEMA_PATH)
    plan_df = pd.DataFrame(
        [
            {
                "pod_id": ORIG,
                "report_date": "2019-04-20",
                "as_of_date": "2019-12-31",
                "pod_scope": "Development",
                "lifting_oil": 1.0,
            }
        ]
    )
    monitoring_df = pd.DataFrame(
        [
            {
                "pod_id": ORIG,
                "case_type": "actual",
                "report_date": "2020-02-15",
                "as_of_date": "2019-12-31",
                "lifting_oil": 1.1,
            },
            {
                "pod_id": ORIG,
                "case_type": "outlook",
                "report_date": "2020-09-30",
                "as_of_date": "2020-12-31",
                "lifting_oil": 1.5,
            },
        ]
    )
    replace_pod_value_cases(sqlite_path, plan_df, monitoring_df, schema)


def _link(sqlite_path: Path, duckdb_path: Path) -> None:
    conn = get_sqlite_connection(sqlite_path)
    dconn = duckdb.connect(str(duckdb_path))
    try:
        store = KnowledgeStore(conn)
        store.ensure_tables()
        run_deterministic_linking(conn, dconn, store)
    finally:
        dconn.close()
        conn.close()


def _assert_sqlite(sqlite_path: Path) -> None:
    conn = get_sqlite_connection(sqlite_path)
    try:
        revisions = conn.execute(
            "SELECT successor_id, predecessor_id, revision_effect,"
            " effective_date, previous_remains_valid"
            " FROM pod_revision ORDER BY predecessor_id"
        ).fetchall()
        assert [tuple(r) for r in revisions] == [
            (AMEND, ORIG, "partial_amendment", "2020-06-01", 1),
            (FINAL, AMEND, "full_replacement", "2021-07-01", 0),
        ]
        value_cases = conn.execute(
            "SELECT pod_id, case_type, report_date, as_of_date"
            " FROM pod_value_case ORDER BY case_type"
        ).fetchall()
        assert [tuple(r) for r in value_cases] == [
            (ORIG, "actual", "2020-02-15", "2019-12-31"),
            (ORIG, "outlook", "2020-09-30", "2020-12-31"),
            (ORIG, "plan", "2019-04-20", "2019-12-31"),
        ]
        assert [
            tuple(r)
            for r in conn.execute(
                "SELECT pod_id, project_id FROM project_pod"
            ).fetchall()
        ] == [(1, PROJECT)]
        assert [
            tuple(r)
            for r in conn.execute("SELECT pod_id, doc_id FROM pod_document").fetchall()
        ] == [(1, DOC)]
    finally:
        conn.close()


def _assert_duckdb(duckdb_path: Path) -> None:
    conn = duckdb.connect(str(duckdb_path), read_only=True)
    try:
        registry = conn.execute(
            "SELECT pod_id, pod_name, preceded_by, revised_by, superseded_by"
            " FROM pod_registry ORDER BY pod_id"
        ).fetchall()
        assert registry == [
            (ORIG, "POD Duri", None, AMEND, None),
            (AMEND, "POD Duri Revisi", ORIG, FINAL, FINAL),
            (FINAL, "POD Duri Final", AMEND, None, None),
        ]
        project = conn.execute("SELECT pod_id, project_id FROM pod_project").fetchall()
        assert project == [(ORIG, PROJECT)]
        docs = conn.execute("SELECT pod_id, doc_id FROM pod_document").fetchall()
        assert docs == [(ORIG, DOC)]
        value_cases = conn.execute(
            "SELECT pod_id, case_type, CAST(report_date AS VARCHAR),"
            " CAST(as_of_date AS VARCHAR)"
            " FROM pod_value_case ORDER BY case_type"
        ).fetchall()
        assert value_cases == [
            (ORIG, "actual", "2020-02-15", "2019-12-31"),
            (ORIG, "outlook", "2020-09-30", "2020-12-31"),
            (ORIG, "plan", "2019-04-20", "2019-12-31"),
        ]
        economics = conn.execute(
            "SELECT case_type, pod_name, lifting_oil,"
            " CAST(as_of_date AS VARCHAR)"
            " FROM pod_economics ORDER BY case_type"
        ).fetchall()
        assert economics == [
            ("actual", "POD Duri", 1.1, "2019-12-31"),
            ("outlook", "POD Duri", 1.5, "2020-12-31"),
            ("plan", "POD Duri", 1.0, "2019-12-31"),
        ]
    finally:
        conn.close()


def _revision_edges(mgr: InstanceGraphManager):
    return mgr.query(
        "MATCH (a:POD)-[r:REVISES]->(b:POD) "
        "RETURN a.pod_id, b.pod_id, r.valid_from"
        " ORDER BY b.pod_id, a.pod_id"
    )


def _supersede_edges(mgr: InstanceGraphManager):
    return mgr.query(
        "MATCH (a:POD)-[r:SUPERSEDES]->(b:POD) "
        "RETURN a.pod_id, b.pod_id, r.valid_from"
        " ORDER BY b.pod_id, a.pod_id"
    )


def _assert_graph(mgr: InstanceGraphManager) -> None:
    assert _revision_edges(mgr) == [
        {"a.pod_id": AMEND, "b.pod_id": ORIG, "r.valid_from": "2020-06-01"},
        {"a.pod_id": FINAL, "b.pod_id": AMEND, "r.valid_from": "2021-07-01"},
    ]
    assert _supersede_edges(mgr) == [
        {"a.pod_id": FINAL, "b.pod_id": AMEND, "r.valid_from": "2021-07-01"},
    ]
    props = mgr.query(
        "MATCH (a:POD)-[r:SUPERSEDES]->(b:POD) "
        "RETURN r.properties_json ORDER BY b.pod_id"
    )
    assert props == [
        {
            "r.properties_json": (
                '{"amended_scope": "whole pod", "previous_remains_valid": false,'
                ' "revision_effect": "full_replacement"}'
            )
        }
    ]
    assert mgr.query(
        "MATCH (a:POD)-[:HAS_PROJECT]->(b:Project) "
        "RETURN a.pod_id, b.project_id ORDER BY a.pod_id"
    ) == [{"a.pod_id": ORIG, "b.project_id": PROJECT}]
    assert mgr.query(
        "MATCH (d:Document)-[:ABOUT_POD]->(p:POD) "
        "RETURN d.doc_id, p.pod_id ORDER BY d.doc_id"
    ) == [{"d.doc_id": DOC, "p.pod_id": ORIG}]


def test_sqlite_truth_matches_duckdb_and_ladybug(tmp_path: Path) -> None:
    sqlite_path = tmp_path / "esdc.sqlite"
    duckdb_path = tmp_path / "esdc.duckdb"
    _seed_sqlite(sqlite_path)
    _seed_duckdb(duckdb_path)
    _replace_value_cases(sqlite_path)
    _link(sqlite_path, duckdb_path)

    _assert_sqlite(sqlite_path)

    publish_pod_registry(sqlite_path=sqlite_path, db_path=duckdb_path)
    _assert_duckdb(duckdb_path)

    mgr = InstanceGraphManager(sqlite_path=sqlite_path, duckdb_path=duckdb_path)
    _assert_graph(mgr)

    # Tear down both projections and rebuild from the same SQLite truth.
    duckdb_path.unlink()
    publish_pod_registry(sqlite_path=sqlite_path, db_path=duckdb_path)
    _assert_duckdb(duckdb_path)

    mgr.close()
    InstanceGraphManager.reset_for_tests()
    rebuilt = InstanceGraphManager(sqlite_path=sqlite_path, duckdb_path=duckdb_path)
    _assert_graph(rebuilt)
