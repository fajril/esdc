from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import duckdb
import pytest

from esdc.pod_registry.store import get_sqlite_connection


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
    (e.g. invalidate_tool_cache(), which shutil.rmtree()s the cache dir and
    rewrites .last_invalidated) would operate on the developer's real
    ~/.esdc/cache. This fixture must stay generic to tests/knowledge/ as a
    whole -- it is the single source of truth for cache-dir isolation, so
    individual test modules (test_learn.py, test_explore_entity.py, etc.)
    do not need to patch Config.get_cache_dir themselves.
    """
    from esdc.configs import Config

    cache_dir = tmp_path_factory.mktemp("tool-cache")
    monkeypatch.setattr(Config, "get_cache_dir", classmethod(lambda cls: cache_dir))


@pytest.fixture(autouse=True)
def _isolate_instance_graph_duckdb(tmp_path_factory, monkeypatch):
    """Point Config.get_db_file at a tmp path that does NOT exist.

    InstanceGraphManager.__init__ defaults duckdb_path to Config.get_db_file()
    when the caller omits it (e.g. every pre-existing test in
    test_instance_graph.py that only passes sqlite_path=..., plus the no-arg
    singleton InstanceGraphManager() built by test_end_to_end.py). Without
    this fixture those tests would silently open the developer's real,
    ~1GB ~/.esdc/esdc.duckdb read-only -- a test-isolation violation.
    InstanceGraphManager degrades gracefully when the duckdb file is absent
    (falls back to project_id as the project name, is_available() stays
    True), so pointing at a path that does not exist is safe and requires no
    other test changes. This fixture must stay generic to tests/knowledge/ as
    a whole -- it is the single source of truth for duckdb-path isolation.
    Explicit duckdb_path=... arguments passed by individual tests still win
    over this default and are unaffected.
    """
    from esdc.configs import Config

    missing = tmp_path_factory.mktemp("no-duckdb") / "esdc.duckdb"
    monkeypatch.setattr(Config, "get_db_file", classmethod(lambda cls: missing))


@pytest.fixture
def sqlite_conn(tmp_path: Path) -> sqlite3.Connection:
    """Registry + minimal documents table, seeded with a tiny POD world."""
    conn = get_sqlite_connection(tmp_path / "esdc.sqlite")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS documents (
            doc_id TEXT PRIMARY KEY,
            file_name TEXT, file_hash TEXT, doc_type TEXT, doc_topic TEXT,
            doc_number TEXT, doc_date TEXT, subject TEXT,
            wk_name TEXT, field_name TEXT, project_name TEXT,
            pod_name TEXT, suggested_pod_ids TEXT, markdown TEXT,
            raw_entities TEXT, metadata TEXT, ingested_at TEXT
        )
        """
    )
    conn.execute(
        "INSERT INTO r_institution (code, institution) VALUES (2, 'SKK Migas')"
    )
    conn.execute("INSERT INTO r_pod_type (code, pod_type) VALUES (2, 'POD I')")
    conn.executemany(
        """
        INSERT INTO m_pod (id, pod_id, pod_name, pod_letter_num, approval_date,
                           institution_code, pod_type_code, rev_num, approval_seq)
        VALUES (?, ?, ?, ?, ?, 2, 2, ?, ?)
        """,
        [
            (
                1,
                "PL-2019-0001-2-2-0",
                "POD I Duri",
                "SRT-0368/SKKMA0000/2019/S1",
                "2019-05-01",
                0,
                1,
            ),
            (
                2,
                "PL-2022-0002-2-2-1",
                "POD I Duri Revisi 1",
                "SRT-0100/SKKMA0000/2022/S1",
                "2022-03-01",
                1,
                2,
            ),
            (
                3,
                "PL-2020-0003-2-2-0",
                "POD Kampung Baru",
                "SRT-0200/SKKMA0000/2020/S1",
                "2020-01-01",
                0,
                3,
            ),
        ],
    )
    conn.executemany(
        "INSERT INTO project_pod (pod_id, project_id) VALUES (?, ?)",
        [(1, "PRJ-001"), (2, "PRJ-001"), (3, "PRJ-002")],
    )
    conn.execute(
        "INSERT INTO pod_revision (successor_id, predecessor_id) VALUES (?, ?)",
        ("PL-2022-0002-2-2-1", "PL-2019-0001-2-2-0"),
    )
    conn.executemany(
        """
        INSERT INTO documents (doc_id, file_name, file_hash, doc_type, doc_topic,
            doc_number, doc_date, subject, wk_name, field_name, project_name,
            pod_name, suggested_pod_ids, markdown)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            # letter number exactly matches m_pod id=2
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
            # no letter match, but suggested_pod_ids present
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
                json.dumps(["PL-2019-0001-2-2-0"]),
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
    yield conn
    conn.close()


@pytest.fixture
def duck_conn(tmp_path: Path) -> duckdb.DuckDBPyConnection:
    """Minimal project_resources for canonical names + remarks."""
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
        ('PRJ-001', 'Duri Steamflood', 'Duri', 'Rokan', 2024,
         'Water handling constraint ongoing'),
        ('PRJ-001', 'Duri Steamflood', 'Duri', 'Rokan', 2023, 'Older remark'),
        ('PRJ-002', 'Kampung Baru Dev', 'Kampung Baru', 'Pertamina EP', 2024,
         'Waiting rig availability')
        """
    )
    yield conn
    conn.close()
