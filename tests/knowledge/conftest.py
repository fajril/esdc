"""Shared fixtures for knowledge graph tests."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from datetime import date
from pathlib import Path
from uuid import uuid4

import duckdb
import pytest

from esdc.knowledge.freshness import compute_knowledge_basis, mark_knowledge_current
from esdc.pod_registry.identity import ApprovalKey, issue_identity
from esdc.pod_registry.store import get_sqlite_connection

ACTOR = "FA:test-knowledge"
INSTITUTION_ID = 4
POD_TYPE_ID = 1

PROJECT_DURI = "P-TEST001-01"
PROJECT_KAMPUNG = "P-TEST002-01"

POD_DURI = "PL-2019-0001-4-1-0"
POD_DURI_REV1 = "PL-2022-0002-4-1-1"
POD_KAMPUNG = "PL-2020-0003-4-1-0"

_DOCUMENTS_DDL = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id TEXT PRIMARY KEY,
    file_name TEXT, file_hash TEXT, doc_type TEXT, doc_topic TEXT,
    doc_number TEXT, doc_date TEXT, subject TEXT,
    wk_name TEXT, field_name TEXT, project_name TEXT,
    pod_name TEXT, suggested_pod_ids TEXT, markdown TEXT,
    raw_entities TEXT, metadata TEXT, ingested_at TEXT
)
"""


@pytest.fixture(autouse=True)
def _isolate_user_guideline(tmp_path_factory, monkeypatch):
    """Point Config.get_db_dir at an empty dir so a developer's real.

    ~/.esdc/guideline.yaml can never leak into tests (load_guideline falls
    back to the packaged default).
    """
    from esdc.configs import Config

    empty = tmp_path_factory.mktemp("no-user-guideline")
    monkeypatch.setattr(Config, "get_db_dir", classmethod(lambda cls: empty))


@pytest.fixture(autouse=True)
def _isolate_tool_cache_dir(tmp_path_factory, monkeypatch):
    """Point Config.get_cache_dir at a tmp dir for every knowledge test.

    Without this, code that resolves Config.get_cache_dir() at call time
    (e.g. invalidate_tool_cache(), which shutil.rmtree()s the cache dir)
    would delete the developer's real ~/.esdc/cache. This fixture must stay
    generic to tests/knowledge/ as a whole -- it is the single source of truth
    for cache-dir isolation, so individual test modules (test_learn.py,
    test_explore_entity.py, etc.) do not need to patch Config.get_cache_dir
    themselves.
    """
    from esdc.configs import Config

    cache_dir = tmp_path_factory.mktemp("tool-cache")
    monkeypatch.setattr(Config, "get_cache_dir", classmethod(lambda cls: cache_dir))


@pytest.fixture(autouse=True)
def _isolate_instance_graph_duckdb(tmp_path_factory, monkeypatch):
    """Point Config.get_db_file at a tmp path that does NOT exist.

    Isolation guard for anything in tests/knowledge that resolves the default
    Duck path at call time (learn pipeline, chat tool layer): without it those
    calls would read the developer's real ~1GB ~/.esdc/esdc.duckdb. It also
    keeps the real implementation away from a test run, because that getter is
    not pure -- it can rename a legacy default database and rewrite config.yaml.
    InstanceGraphManager itself no longer resolves any Duck path (it reads SQLite
    only), so this fixture is not what protects the graph; the constructor
    regression in test_instance_graph.py re-reads the real implementation against
    its own tmp paths and asserts construction leaves nothing changed.
    """
    from esdc.configs import Config

    missing = tmp_path_factory.mktemp("no-duckdb") / "esdc.duckdb"
    monkeypatch.setattr(Config, "get_db_file", classmethod(lambda cls: missing))


def _issue_pod(
    conn: sqlite3.Connection,
    *,
    approval_date: date,
    daily_slot: int,
    rev_num: int,
    pod_name: str,
    pod_letter_num: str,
) -> int:
    """Issue the permanent identity and write the matching current m_pod row."""
    issued = issue_identity(
        conn,
        key=ApprovalKey(
            approval_date=approval_date,
            daily_slot=daily_slot,
            institution_id=INSTITUTION_ID,
            pod_type_id=POD_TYPE_ID,
            rev_num=rev_num,
        ),
        actor=ACTOR,
    )
    conn.execute(
        "INSERT INTO m_pod (pod_id_seq, pod_uuid, approval_date, institution_id, "
        "pod_type_id, rev_num, pod_order_no, pod_id, pod_name, pod_letter_num, "
        "is_latest, created_by, updated_by) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?)",
        (
            issued.pod_id_seq,
            str(issued.pod_uuid),
            issued.approval_date.isoformat(),
            INSTITUTION_ID,
            POD_TYPE_ID,
            rev_num,
            issued.pod_order_no,
            issued.pod_id,
            pod_name,
            pod_letter_num,
            ACTOR,
            ACTOR,
        ),
    )
    return issued.pod_id_seq


def _seed_knowledge_world(conn: sqlite3.Connection) -> None:
    """Seed the tiny POD world on the canonical single schema.

    Reference labels come from the SQL seed (institution 4 = SKK Migas, POD
    type 1 = POD I); they are never re-inserted or rewritten here. Identities
    come from the permanent identity service, so every pod_id is the canonical
    formula for its approval date and daily slot.
    """
    conn.execute(_DOCUMENTS_DDL)

    conn.executemany(
        "INSERT INTO m_project (project_id, project_name, is_active, created_by, updated_by) "
        "VALUES (?, ?, 1, ?, ?)",
        [
            (PROJECT_DURI, "Duri Steamflood", ACTOR, ACTOR),
            (PROJECT_KAMPUNG, "Kampung Baru Dev", ACTOR, ACTOR),
        ],
    )

    first_seq = _issue_pod(
        conn,
        approval_date=date(2019, 1, 1),
        daily_slot=1,
        rev_num=0,
        pod_name="POD I Duri",
        pod_letter_num="SRT-0368/SKKMA0000/2019/S1",
    )
    latest_seq = _issue_pod(
        conn,
        approval_date=date(2022, 1, 1),
        daily_slot=2,
        rev_num=1,
        pod_name="POD I Duri Revisi 1",
        pod_letter_num="SRT-0100/SKKMA0000/2022/S1",
    )
    kampung_seq = _issue_pod(
        conn,
        approval_date=date(2020, 1, 1),
        daily_slot=3,
        rev_num=0,
        pod_name="POD Kampung Baru",
        pod_letter_num="SRT-0200/SKKMA0000/2020/S1",
    )

    # The latest-only pointer is the current revision contract: a POD is current
    # only while it has no successor, so both flags change in one statement.
    conn.execute(
        "UPDATE m_pod SET superseded_by=?, is_latest=0 WHERE pod_id_seq=?",
        (latest_seq, first_seq),
    )

    conn.executemany(
        "INSERT INTO pod_project (pod_project_uuid, pod_id_seq, project_id, "
        "is_main_project, is_auto_inferred, created_by, updated_by) "
        "VALUES (?, ?, ?, 1, 0, ?, ?)",
        [
            (str(uuid4()), first_seq, PROJECT_DURI, ACTOR, ACTOR),
            (str(uuid4()), latest_seq, PROJECT_DURI, ACTOR, ACTOR),
            (str(uuid4()), kampung_seq, PROJECT_KAMPUNG, ACTOR, ACTOR),
        ],
    )

    conn.executemany(
        "INSERT INTO api_project_resources (project_id, report_year, project_name, "
        "field_name, wk_name, uncert_level, project_remarks) "
        "VALUES (?, ?, ?, ?, ?, '1P', ?)",
        [
            (
                PROJECT_DURI,
                2024,
                "Duri Steamflood",
                "Duri",
                "Rokan",
                "Water handling constraint ongoing",
            ),
            (
                PROJECT_DURI,
                2023,
                "Duri Steamflood",
                "Duri",
                "Rokan",
                "Older remark",
            ),
            (
                PROJECT_KAMPUNG,
                2024,
                "Kampung Baru Dev",
                "Kampung Baru",
                "Pertamina EP",
                "Waiting rig availability",
            ),
        ],
    )

    conn.executemany(
        """
        INSERT INTO documents (doc_id, file_name, file_hash, doc_type, doc_topic,
            doc_number, doc_date, subject, wk_name, field_name, project_name,
            pod_name, suggested_pod_ids, markdown)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            # letter number exactly matches the current identity in 2022
            (
                "DOC-A",
                "a.pdf",
                "hash-a",
                "letter",
                '["pod"]',
                "SRT-0100/SKKMA0000/2022/S1",
                "2022-03-01",
                "Persetujuan POD I Duri Revisi 1",
                "Rokan",
                "Duri",
                None,
                "POD I Duri Revisi 1",
                None,
                "# Approval\nEconomics: NPV 100 MUSD.",
            ),
            # no letter match, but suggested_pod_ids present (review-only)
            (
                "DOC-B",
                "b.pdf",
                "hash-b",
                "mom",
                '["pod"]',
                "MOM-77/2023",
                "2023-04-01",
                "MoM Monitoring POD I Duri",
                "Rokan",
                "Duri",
                None,
                "POD I Duri",
                json.dumps([POD_DURI]),
                "# MoM\nDiscussed drilling delay.",
            ),
            # regulation: no entities at all
            (
                "DOC-C",
                "c.pdf",
                "hash-c",
                "permen",
                '["others"]',
                "199.K/HK.02/MEM.M/2021",
                "2021-06-01",
                "Pedoman Insentif Hulu Migas",
                None,
                None,
                None,
                None,
                None,
                "# Regulation text",
            ),
        ],
    )

    conn.commit()

    # Knowledge is marked current only at the end-of-seed boundary, measured
    # against the same snapshot the readers use.
    conn.execute("BEGIN IMMEDIATE")
    basis = compute_knowledge_basis(conn)
    assert mark_knowledge_current(conn, basis)
    conn.commit()


@pytest.fixture
def sqlite_conn(tmp_path: Path) -> Iterator[sqlite3.Connection]:
    """Canonical registry plus a minimal documents table, seeded with a tiny POD world."""
    conn = get_sqlite_connection(tmp_path / "esdc.sqlite")
    try:
        _seed_knowledge_world(conn)
        yield conn
    finally:
        conn.close()


@pytest.fixture
def duck_conn(tmp_path: Path) -> Iterator[duckdb.DuckDBPyConnection]:
    """DuckDB connection for the disposable corpus mirror and dossier cache.

    Knowledge readers take project, field, working-area and remark values from
    the canonical SQLite raw resource rows seeded in ``sqlite_conn``. Since S2b2a
    the instance graph never opens Duck (it reads the SQLite snapshot only), so
    the mirror rows below serve the learn pipeline's mirror indexes and dossier
    storage, not graph name resolution.
    """
    conn = duckdb.connect(str(tmp_path / "esdc.duckdb"))
    conn.execute(
        """
        CREATE TABLE project_resources (
            project_id VARCHAR, project_name VARCHAR, field_name VARCHAR,
            wk_name VARCHAR, report_year INTEGER, project_remarks VARCHAR
        )
        """
    )
    conn.execute(
        """
        INSERT INTO project_resources VALUES
        ('P-TEST001-01', 'Duri Steamflood', 'Duri', 'Rokan', 2024,
         'Water handling constraint ongoing'),
        ('P-TEST001-01', 'Duri Steamflood', 'Duri', 'Rokan', 2023, 'Older remark'),
        ('P-TEST002-01', 'Kampung Baru Dev', 'Kampung Baru', 'Pertamina EP', 2024,
         'Waiting rig availability')
        """
    )
    yield conn
    conn.close()
