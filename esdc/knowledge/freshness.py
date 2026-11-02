"""Dependency-basis freshness for POD knowledge derived from SQLite truth."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from typing import Any

from esdc.knowledge.guideline import load_guideline

_STATE_TABLE = "pod_knowledge_state"
_DEPENDENCIES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "m_project",
        ("project_id", "project_name", "field_name", "wk_name", "is_active"),
    ),
    (
        "m_pod",
        (
            "pod_id_seq",
            "pod_id",
            "pod_uuid",
            "approval_date",
            "institution_id",
            "pod_type_id",
            "rev_num",
            "is_latest",
            "superseded_by",
            "pod_name",
            "pod_letter_num",
            "pod_stage_id",
        ),
    ),
    ("pod_project", ("pod_id_seq", "project_id")),
    ("pod_document", ("pod_id_seq", "doc_id")),
    (
        "api_project_resources",
        (
            "project_id",
            "project_name",
            "report_year",
            "project_remarks",
            "field_name",
            "wk_name",
        ),
    ),
    (
        "documents",
        (
            "doc_id",
            "file_hash",
            "doc_type",
            "doc_date",
            "subject",
            "sender",
            "recipient",
            "doc_level",
            "doc_number",
            "field_name",
            "wk_name",
            "project_name",
            "pod_name",
            "markdown",
        ),
    ),
)


@dataclass(frozen=True, slots=True)
class KnowledgeFreshness:
    """Current knowledge-generation state compared with live source inputs."""

    fresh: bool
    source_generation: int | None
    knowledge_generation: int | None
    basis_sha256: str | None
    stored_basis_sha256: str | None


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    return (
        conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", [table]
        ).fetchone()
        is not None
    )


def _available_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {
        str(row[1])
        for row in conn.execute(f'PRAGMA table_info("{table}")').fetchall()
    }


def compute_knowledge_basis(conn: sqlite3.Connection) -> str:
    """Hash only real inputs to entity resolution, extraction, and dossiers.

    Monitoring/forecast values, generated edges/claims/suggestions, learn state,
    and dossiers are deliberately excluded so numeric-only changes do not cause
    another LLM pass and generated output cannot invalidate its own basis.
    """
    inputs: list[Any] = [["guideline", load_guideline().content_hash]]
    for table, desired_columns in _DEPENDENCIES:
        if not _table_exists(conn, table):
            inputs.append([table, None])
            continue
        columns = _available_columns(conn, table)
        selected = tuple(column for column in desired_columns if column in columns)
        if not selected:
            inputs.append([table, (), ()])
            continue
        projection = ", ".join(f'"{column}"' for column in selected)
        selected_order = tuple(
            name
            for name in (
                "project_id",
                "report_year",
                "pod_id_seq",
                "doc_id",
                "project_name",
                "project_remarks",
                "field_name",
                "wk_name",
            )
            if name in selected
        )
        order_columns = selected_order + tuple(
            name for name in selected if name not in selected_order
        )
        order_sql = ", ".join(f'"{name}"' for name in order_columns)
        distinct = "DISTINCT " if table == "api_project_resources" else ""
        rows = [
            tuple(row)
            for row in conn.execute(
                f'SELECT {distinct}{projection} FROM "{table}" '
                f"ORDER BY {order_sql}"
            ).fetchall()
        ]
        inputs.append([table, selected, rows])
    payload = json.dumps(inputs, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def ensure_knowledge_state_table(conn: sqlite3.Connection) -> None:
    """Create the technical basis marker inside the caller-owned transaction."""
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {_STATE_TABLE} (
            singleton_id INTEGER PRIMARY KEY CHECK (singleton_id = 1),
            basis_sha256 TEXT NOT NULL,
            source_generation INTEGER NOT NULL,
            updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
        )
        """
    )


def knowledge_freshness(conn: sqlite3.Connection) -> KnowledgeFreshness:
    """Read generation and basis state against the current registry schema."""
    if not _table_exists(conn, "pod_registry_state"):
        return KnowledgeFreshness(False, None, None, None, None)
    state = conn.execute(
        "SELECT source_generation, knowledge_generation FROM pod_registry_state "
        "WHERE singleton_id=1"
    ).fetchone()
    if state is None:
        return KnowledgeFreshness(False, None, None, None, None)
    source_generation = int(state[0])
    knowledge_generation = None if state[1] is None else int(state[1])
    current_basis = compute_knowledge_basis(conn)
    if not _table_exists(conn, _STATE_TABLE):
        return KnowledgeFreshness(
            False, source_generation, knowledge_generation, current_basis, None
        )
    stored = conn.execute(
        f"SELECT basis_sha256 FROM {_STATE_TABLE} WHERE singleton_id=1"
    ).fetchone()
    stored_basis = None if stored is None else str(stored[0])
    fresh = (
        knowledge_generation == source_generation and stored_basis == current_basis
    )
    return KnowledgeFreshness(
        fresh,
        source_generation,
        knowledge_generation,
        current_basis,
        stored_basis,
    )


def knowledge_freshness_markers(conn: sqlite3.Connection) -> KnowledgeFreshness:
    """Read compact committed markers without hashing corpus source rows."""
    if not _table_exists(conn, "pod_registry_state"):
        return KnowledgeFreshness(False, None, None, None, None)
    state = conn.execute(
        "SELECT source_generation, knowledge_generation "
        "FROM pod_registry_state WHERE singleton_id=1"
    ).fetchone()
    if state is None:
        return KnowledgeFreshness(False, None, None, None, None)
    source_generation = int(state[0])
    knowledge_generation = None if state[1] is None else int(state[1])
    if not _table_exists(conn, _STATE_TABLE):
        return KnowledgeFreshness(
            False, source_generation, knowledge_generation, None, None
        )
    marker = conn.execute(
        f"SELECT basis_sha256, source_generation FROM {_STATE_TABLE} "
        "WHERE singleton_id=1"
    ).fetchone()
    if marker is None:
        return KnowledgeFreshness(
            False, source_generation, knowledge_generation, None, None
        )
    stored_basis = str(marker[0])
    marker_generation = int(marker[1])
    fresh = (
        knowledge_generation == source_generation
        and marker_generation == source_generation
    )
    return KnowledgeFreshness(
        fresh,
        source_generation,
        knowledge_generation,
        None,
        stored_basis,
    )


def committed_marker_metadata(conn: sqlite3.Connection) -> tuple[Any, ...] | None:
    """Read the committed state markers as stored, without hashing corpus rows.

    The registry schema, contract and reference-seed versions with the state
    updated marker, plus the knowledge state updated marker. A disposable reader
    compares these against the marker fingerprint it built from, so a committed
    transition is observed without re-hashing the corpus. Returns ``None`` when
    either marker row is absent; the caller owns the read transaction.
    """
    if not _table_exists(conn, "pod_registry_state"):
        return None
    state = conn.execute(
        "SELECT schema_version, contract_id, contract_version, dictionary_version, "
        "reference_seed_version, updated_at FROM pod_registry_state "
        "WHERE singleton_id=1"
    ).fetchone()
    if state is None or not _table_exists(conn, _STATE_TABLE):
        return None
    knowledge = conn.execute(
        f"SELECT updated_at FROM {_STATE_TABLE} WHERE singleton_id=1"
    ).fetchone()
    if knowledge is None:
        return None
    marker = tuple(value if value is None else str(value) for value in state)
    return marker + (str(knowledge[0]),)


def reconcile_knowledge_after_pod_change(
    conn: sqlite3.Connection,
    *,
    previous_generation: int,
    previous_knowledge_generation: int | None,
    source_generation: int,
    actor: str,
) -> bool:
    """Invalidate derived knowledge atomically when its true inputs changed.

    Returns True when the basis changed. Call inside the active transaction
    that owns the source change; this function never commits.
    """
    if not conn.in_transaction:
        raise RuntimeError("knowledge invalidation requires the source transaction")
    if not actor.strip():
        raise ValueError("knowledge invalidation actor cannot be empty")
    ensure_knowledge_state_table(conn)
    basis = compute_knowledge_basis(conn)
    row = conn.execute(
        f"SELECT basis_sha256 FROM {_STATE_TABLE} WHERE singleton_id=1"
    ).fetchone()
    previous_basis = None if row is None else str(row[0])
    changed = previous_basis != basis
    if changed:
        from esdc.knowledge.store import KnowledgeStore

        KnowledgeStore(conn).ensure_tables(commit=False)
        for table in ("kg_edge", "kg_claim", "kg_proposal", "learn_state"):
            if _table_exists(conn, table):
                conn.execute(f'DELETE FROM "{table}"')
        if (
            _table_exists(conn, "documents")
            and "suggested_pod_ids" in _available_columns(conn, "documents")
        ):
            conn.execute("UPDATE documents SET suggested_pod_ids=NULL")
        conn.execute(
            "UPDATE pod_registry_state SET knowledge_generation=NULL "
            "WHERE singleton_id=1"
        )
    elif (
        previous_knowledge_generation is not None
        and previous_knowledge_generation == previous_generation
    ):
        conn.execute(
            "UPDATE pod_registry_state SET knowledge_generation=? WHERE singleton_id=1",
            [source_generation],
        )
    conn.execute(
        f"""
        INSERT INTO {_STATE_TABLE} (singleton_id, basis_sha256, source_generation)
        VALUES (1, ?, ?)
        ON CONFLICT (singleton_id) DO UPDATE SET
            basis_sha256=excluded.basis_sha256,
            source_generation=excluded.source_generation,
            updated_at=strftime('%Y-%m-%dT%H:%M:%fZ','now')
        """,
        [basis, source_generation],
    )
    return changed


def invalidate_knowledge_after_dependency_change(
    conn: sqlite3.Connection, *, actor: str, force: bool = False
) -> bool:
    """Invalidate derived knowledge after a corpus input changes.

    Call from the same SQLite transaction as a document insert/update/delete.
    Corpus changes do not advance the POD registry generation, but they still
    invalidate every derived edge, claim, proposal, and learn marker whose
    basis included the prior corpus.
    """
    if not conn.in_transaction:
        raise RuntimeError("knowledge invalidation requires the source transaction")
    if not actor.strip():
        raise ValueError("knowledge invalidation actor cannot be empty")
    if not _table_exists(conn, "pod_registry_state"):
        return False

    state = conn.execute(
        "SELECT source_generation FROM pod_registry_state WHERE singleton_id=1"
    ).fetchone()
    if state is None:
        return False
    generation = int(state[0])
    ensure_knowledge_state_table(conn)
    basis = compute_knowledge_basis(conn)
    row = conn.execute(
        f"SELECT basis_sha256 FROM {_STATE_TABLE} WHERE singleton_id=1"
    ).fetchone()
    previous_basis = None if row is None else str(row[0])
    changed = previous_basis != basis

    if changed or force:
        from esdc.knowledge.store import KnowledgeStore

        KnowledgeStore(conn, allow_stale=True).ensure_tables(commit=False)
        for table in ("kg_edge", "kg_claim", "kg_proposal", "learn_state"):
            if _table_exists(conn, table):
                conn.execute(f'DELETE FROM "{table}"')
        if (
            _table_exists(conn, "documents")
            and "suggested_pod_ids" in _available_columns(conn, "documents")
        ):
            conn.execute("UPDATE documents SET suggested_pod_ids=NULL")
        conn.execute(
            "UPDATE pod_registry_state SET knowledge_generation=NULL "
            "WHERE singleton_id=1"
        )

    conn.execute(
        f"""
        INSERT INTO {_STATE_TABLE} (singleton_id, basis_sha256, source_generation)
        VALUES (1, ?, ?)
        ON CONFLICT (singleton_id) DO UPDATE SET
            basis_sha256=excluded.basis_sha256,
            source_generation=excluded.source_generation,
            updated_at=strftime('%Y-%m-%dT%H:%M:%fZ','now')
        """,
        [basis, generation],
    )
    return changed


def mark_knowledge_current(conn: sqlite3.Connection, expected_basis: str) -> bool:
    """Mark knowledge fresh in the caller transaction if its inputs still match."""
    if not conn.in_transaction:
        raise RuntimeError("knowledge freshness update requires an active transaction")
    current_basis = compute_knowledge_basis(conn)
    if current_basis != expected_basis:
        return False
    state = conn.execute(
        "SELECT source_generation FROM pod_registry_state WHERE singleton_id=1"
    ).fetchone()
    if state is None:
        return False
    generation = int(state[0])
    ensure_knowledge_state_table(conn)
    conn.execute(
        f"""
        INSERT INTO {_STATE_TABLE} (singleton_id, basis_sha256, source_generation)
        VALUES (1, ?, ?)
        ON CONFLICT (singleton_id) DO UPDATE SET
            basis_sha256=excluded.basis_sha256,
            source_generation=excluded.source_generation,
            updated_at=strftime('%Y-%m-%dT%H:%M:%fZ','now')
        """,
        [current_basis, generation],
    )
    conn.execute(
        "UPDATE pod_registry_state SET knowledge_generation=? WHERE singleton_id=1",
        [generation],
    )
    return True
