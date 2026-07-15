# esdc/pod_registry/store.py
"""SQLite store for the POD master registry (source of truth)."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from esdc.configs import Config

POD_SQLITE_FILENAME = "pod.sqlite"

_DDL = """
CREATE TABLE IF NOT EXISTS r_institution (
    code        INTEGER PRIMARY KEY,
    institution TEXT NOT NULL UNIQUE,
    description TEXT
);
CREATE TABLE IF NOT EXISTS r_pod_type (
    code        INTEGER PRIMARY KEY,
    pod_type    TEXT NOT NULL UNIQUE,
    description TEXT
);
CREATE TABLE IF NOT EXISTS m_pod (
    id               INTEGER PRIMARY KEY,
    pod_id           TEXT NOT NULL UNIQUE,
    pod_name         TEXT NOT NULL,
    pod_letter_num   TEXT,
    approval_date    TEXT NOT NULL,
    institution_code INTEGER NOT NULL REFERENCES r_institution(code),
    pod_type_code    INTEGER NOT NULL REFERENCES r_pod_type(code),
    rev_num          INTEGER NOT NULL DEFAULT 0,
    approval_seq     INTEGER NOT NULL UNIQUE
);
CREATE TABLE IF NOT EXISTS project_pod (
    pod_id     INTEGER NOT NULL REFERENCES m_pod(id),
    project_id TEXT NOT NULL,
    UNIQUE (pod_id, project_id)
);
CREATE TABLE IF NOT EXISTS pod_revision (
    successor_id   TEXT NOT NULL REFERENCES m_pod(pod_id),
    predecessor_id TEXT NOT NULL REFERENCES m_pod(pod_id),
    UNIQUE (successor_id, predecessor_id),
    CHECK (successor_id <> predecessor_id)
);
"""


def get_pod_sqlite_path() -> Path:
    """Location of the registry database inside the esdc db directory."""
    return Config.get_db_dir() / POD_SQLITE_FILENAME


def ensure_tables(conn: sqlite3.Connection) -> None:
    conn.executescript(_DDL)


def get_sqlite_connection(path: Path | None = None) -> sqlite3.Connection:
    """Open (creating if needed) the registry db with FK enforcement on."""
    db_path = path or get_pod_sqlite_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    ensure_tables(conn)
    return conn
