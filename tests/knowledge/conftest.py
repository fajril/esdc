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
            pod_name TEXT, suggested_pod_ids TEXT, markdown TEXT
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
