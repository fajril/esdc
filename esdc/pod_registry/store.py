# esdc/pod_registry/store.py
"""SQLite store for esdc operational data (source of truth).

Currently holds the POD master registry; intended to grow into the
app-wide operational database (auth, entity relationships, ...), hence
the generic esdc.sqlite filename.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from esdc.configs import Config

ESDC_SQLITE_FILENAME = "esdc.sqlite"
_LEGACY_SQLITE_FILENAME = "pod.sqlite"  # pre-rename registry database

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


def get_esdc_sqlite_path() -> Path:
    """Location of the operational database inside the esdc db directory."""
    return Config.get_db_dir() / ESDC_SQLITE_FILENAME


def _migrate_legacy_pod_sqlite(db_path: Path) -> None:
    """One-time rename of the pre-existing pod.sqlite to esdc.sqlite."""
    legacy = db_path.parent / _LEGACY_SQLITE_FILENAME
    if not db_path.exists() and legacy.exists():
        legacy.rename(db_path)


def ensure_tables(conn: sqlite3.Connection) -> None:
    conn.executescript(_DDL)


def get_sqlite_connection(path: Path | None = None) -> sqlite3.Connection:
    """Open (creating if needed) the operational db with FK enforcement on."""
    db_path = path or get_esdc_sqlite_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if path is None:
        _migrate_legacy_pod_sqlite(db_path)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    ensure_tables(conn)
    return conn


def allocate_pod_id(
    conn: sqlite3.Connection,
    approval_date: str,
    institution_code: int,
    pod_type_code: int,
    rev_num: int,
) -> tuple[str, int]:
    """Compute the next pod_id. Caller must insert within the same transaction.

    pod_id is immutable once issued and approval_seq never gets reused, so the
    next sequence is always max(approval_seq) + 1 (gaps from deletes are fine).
    """
    year = approval_date[:4]
    row = conn.execute("SELECT COALESCE(MAX(approval_seq), 0) FROM m_pod").fetchone()
    seq = int(row[0]) + 1
    pod_id = f"PL-{year}-{seq:04d}-{institution_code}-{pod_type_code}-{rev_num}"
    return pod_id, seq
