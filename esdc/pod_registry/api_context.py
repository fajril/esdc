"""Read-only POD reconciliation context for API preparation (Task5C1).

The context is one snapshot taken inside a single ``mode=ro`` transaction: the
five canonical input tables, the permanent identity ledger, the existing
document links, and the generation/protected-state basis that a later apply must
still be based on. The reader never creates a path, initializes a registry,
migrates a legacy target, or writes; it closes its connection before returning.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from esdc.pod_registry.api_import import INPUT_TABLES
from esdc.pod_registry.contract import PodContract, load_pod_contract
from esdc.pod_registry.migrations import (
    MigrationRefusedError,
    MigrationReport,
    _validate_existing_v2_registry,
    inspect_migration,
)
from esdc.pod_registry.models import ImportTargetBasis, PreparedRow, PreparedScalar
from esdc.pod_registry.normalization import (
    PodNormalizationError,
    normalize_source_row,
)
from esdc.pod_registry.store import _refuse_incompatible_shape

FRESH_PROTECTED_STATE_SHA256 = hashlib.sha256(b"fresh").hexdigest()
NON_POD_TARGET_PROTECTED_STATE_SHA256 = hashlib.sha256(b"non-pod-target").hexdigest()

# Task4 source audit semantics are unchanged: a source ledger is read against these
# fields, with the issuer recorded as ``issued_by``. The operational v2 ledger has a
# different shape, so the reader supplies the canonical DDL columns explicitly.
_SOURCE_LEDGER_FIELDS = frozenset(
    {
        "approval_year",
        "approval_date",
        "daily_slot",
        "pod_order_no",
        "pod_id_seq",
        "pod_id",
        "issued_at",
        "issued_by",
    }
)
_CANONICAL_V2_LEDGER_FIELDS = frozenset(
    {
        "approval_year",
        "approval_date",
        "daily_slot",
        "pod_order_no",
        "pod_id_seq",
        "pod_id",
        "pod_uuid",
        "institution_id",
        "pod_type_id",
        "rev_num",
        "actor",
        "issued_at",
    }
)
_FETCH_BATCH = 1000


class PodImportError(ValueError):
    """One or more source values or structures do not satisfy the POD contract."""

    def __init__(self, errors: Sequence[str]) -> None:
        self.errors = tuple(errors)
        super().__init__("\n".join(self.errors))


@dataclass(frozen=True, slots=True)
class PodApiContext:
    """Read-only reconciliation facts a prepared POD batch must be based on."""

    target_basis: ImportTargetBasis
    existing_rows: Mapping[str, Sequence[PreparedRow]]
    reserved_pod_identities: Sequence[PreparedRow]
    linked_pod_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class TargetSnapshot:
    """One read-only transaction's worth of target facts."""

    basis: ImportTargetBasis
    existing_rows: Mapping[str, Sequence[PreparedRow]]
    reserved_pod_identities: Sequence[PreparedRow] | None
    linked_pod_ids: tuple[int, ...]


def quote_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def object_type(conn: sqlite3.Connection, name: str) -> str | None:
    row = conn.execute(
        "SELECT type FROM sqlite_master WHERE name = ?", (name,)
    ).fetchone()
    return None if row is None else str(row[0])


def column_names(conn: sqlite3.Connection, name: str) -> list[str]:
    cursor = conn.execute(f"PRAGMA table_info({quote_identifier(name)})")
    return [str(row[1]) for row in cursor.fetchall()]


def read_only_connection(path: Path) -> sqlite3.Connection:
    """Open a read-only snapshot connection and close it if read setup fails."""
    uri = f"{path.resolve().as_uri()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    try:
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA query_only = ON")
        conn.execute("BEGIN")
    except BaseException:
        # A connection this helper created is always its own to close, for any
        # exception including a cancellation; the exception itself is not wrapped,
        # so callers keep their contextual SQL error and cancellation propagates.
        conn.close()
        raise
    return conn


def target_protected_state_digest(
    conn: sqlite3.Connection,
    *,
    need_preserved_inputs: bool = False,
    contract: PodContract | None = None,
) -> str:
    """Hash preserved facts in stable key order using bounded cursor reads."""
    digest = hashlib.sha256()
    tables = ["pod_document"]
    if need_preserved_inputs:
        tables.extend(("pod_forecast", "pod_monitoring"))
    active = contract if contract is not None else load_pod_contract()
    for table_name in sorted(tables):
        if table_name == "pod_document":
            query = (
                "SELECT pod_id_seq, doc_id FROM pod_document "
                "ORDER BY pod_id_seq, doc_id"
            )
        else:
            columns = tuple(column.name for column in active.tables[table_name].columns)
            key = active.tables[table_name].primary_key
            query = (
                f"SELECT {', '.join(quote_identifier(column) for column in columns)} "
                f"FROM {quote_identifier(table_name)} "
                f"ORDER BY {', '.join(quote_identifier(column) for column in key)}"
            )
        cursor = conn.execute(query)
        while batch := cursor.fetchmany(_FETCH_BATCH):
            for row in batch:
                digest.update(
                    json.dumps(
                        [table_name, *tuple(row)],
                        ensure_ascii=False,
                        separators=(",", ":"),
                        default=str,
                    ).encode("utf-8")
                )
                digest.update(b"\n")
    return digest.hexdigest()


def read_identity_ledger_rows(
    conn: sqlite3.Connection,
    *,
    required_fields: frozenset[str] = _SOURCE_LEDGER_FIELDS,
) -> Sequence[PreparedRow] | None:
    """Read permanent identity reservations, or ``None`` when the ledger is absent."""
    ledger_type = object_type(conn, "_pod_identity_ledger")
    if ledger_type is None:
        return None
    if ledger_type != "table":
        raise PodImportError(["SQLite source _pod_identity_ledger must be a table"])
    headers = column_names(conn, "_pod_identity_ledger")
    missing = sorted(required_fields - set(headers))
    if missing:
        raise PodImportError(
            [f"SQLite source identity ledger is missing fields: {', '.join(missing)}"]
        )
    cursor = conn.execute(f"SELECT * FROM {quote_identifier('_pod_identity_ledger')}")
    rows: list[PreparedRow] = []
    while batch := cursor.fetchmany(_FETCH_BATCH):
        for row in batch:
            rows.append(
                {
                    name: cast(PreparedScalar, value)
                    for name, value in zip(headers, row, strict=True)
                }
            )
    return tuple(rows)


def read_source_identity_ledger_rows(
    conn: sqlite3.Connection,
) -> Sequence[PreparedRow] | None:
    """Read a source ledger under the profile its own declared shape selects.

    A source without POD registry metadata keeps Task4's supplied ePOD profile. A source
    that declares the canonical current registry is validated against the canonical
    contract and read on the canonical fields; only the technical representation of the
    issuer is normalized, ``actor`` becoming the prepared ``issued_by``. Identity,
    reservation, audit and metadata values are neither dropped nor reinterpreted, and a
    malformed canonical shape still refuses before any writer opens.
    """
    state_type = object_type(conn, "pod_registry_state")
    if state_type is None:
        return read_identity_ledger_rows(conn)
    if state_type != "table":
        raise PodImportError(
            ["Canonical SQLite source pod_registry_state must be a table"]
        )
    try:
        _validate_existing_v2_registry(conn)
    except (sqlite3.Error, ValueError, RuntimeError) as exc:
        raise PodImportError(
            [f"Canonical SQLite source preflight failed: {exc}"]
        ) from exc
    headers = frozenset(column_names(conn, "_pod_identity_ledger"))
    if headers != _CANONICAL_V2_LEDGER_FIELDS:
        missing = sorted(_CANONICAL_V2_LEDGER_FIELDS - headers)
        unexpected = sorted(headers - _CANONICAL_V2_LEDGER_FIELDS)
        raise PodImportError(
            [
                "Canonical SQLite source ledger fields mismatch: "
                f"missing={missing}, unexpected={unexpected}"
            ]
        )
    rows = read_identity_ledger_rows(conn, required_fields=_CANONICAL_V2_LEDGER_FIELDS)
    if rows is None:
        raise PodImportError(["Canonical SQLite source identity ledger is missing"])
    normalized: list[PreparedRow] = []
    for row in rows:
        item = dict(row)
        item["issued_by"] = item.pop("actor")
        normalized.append(item)
    return tuple(normalized)


def refuse_target_shape(conn: sqlite3.Connection) -> MigrationReport:
    """Classify one target with the accepted store shape policy, without writing.

    The reader and the writer must see the same boundary, so both call this one policy:
    documents, raw-source and unrelated application objects alone are an additive
    no-POD target, while a legacy or partial POD footprint, including a reserved
    POD-shaped view, is refused before any writer or DDL runs.
    """
    try:
        report = inspect_migration(conn)
        _refuse_incompatible_shape(report)
    except MigrationRefusedError as exc:
        raise PodImportError([str(exc)]) from exc
    return report


def read_target_snapshot(
    conn: sqlite3.Connection,
    path: Path,
    contract: PodContract,
    *,
    materialize_rows: bool,
    need_preserved_inputs: bool = False,
) -> TargetSnapshot:
    """Classify and read one target snapshot on a caller-owned read-only connection."""
    state_type = object_type(conn, "pod_registry_state")
    if state_type is None:
        # The shared classifier decides the shape; nothing here infers a POD identity.
        # A no-POD target has no preserved input context, so its inputs stay empty and
        # no absent POD table is read.
        refuse_target_shape(conn)
        return TargetSnapshot(
            ImportTargetBasis(
                True,
                None,
                NON_POD_TARGET_PROTECTED_STATE_SHA256,
                path.resolve(strict=False),
            ),
            {},
            None,
            (),
        )
    if state_type != "table":
        raise PodImportError(["Operational pod_registry_state must be a table"])

    try:
        _validate_existing_v2_registry(conn)
    except (sqlite3.Error, ValueError, RuntimeError) as exc:
        raise PodImportError(
            [f"Operational v2 registry preflight failed: {exc}"]
        ) from exc
    state = conn.execute(
        "SELECT source_generation FROM pod_registry_state WHERE singleton_id=1"
    ).fetchone()
    if state is None:
        raise PodImportError(["Operational v2 registry state row is missing"])

    # The basis, the rows, the ledger, and the document links must all describe the
    # same snapshot, so they are read on this one connection and transaction.
    rows = read_input_rows(conn, contract) if materialize_rows else {}
    ledger = (
        read_identity_ledger_rows(conn, required_fields=_CANONICAL_V2_LEDGER_FIELDS)
        if materialize_rows
        else None
    )
    linked_pod_ids = tuple(
        sorted(
            {
                int(row[0])
                for row in conn.execute("SELECT DISTINCT pod_id_seq FROM pod_document")
            }
        )
    )
    basis = ImportTargetBasis(
        True,
        int(state[0]),
        target_protected_state_digest(
            conn, need_preserved_inputs=need_preserved_inputs, contract=contract
        ),
        path.resolve(strict=False),
    )
    return TargetSnapshot(basis, rows, ledger, linked_pod_ids)


def read_input_rows(
    conn: sqlite3.Connection, contract: PodContract
) -> dict[str, tuple[PreparedRow, ...]]:
    """Read the five canonical inputs as typed rows in stable key order."""
    rows: dict[str, tuple[PreparedRow, ...]] = {}
    for table_name in INPUT_TABLES:
        table = contract.tables[table_name]
        names = tuple(column.name for column in table.columns)
        key = table.primary_key or names[:1]
        query = (
            f"SELECT {', '.join(quote_identifier(name) for name in names)} "
            f"FROM {quote_identifier(table_name)} "
            f"ORDER BY {', '.join(quote_identifier(column) for column in key)}"
        )
        cursor = conn.execute(query)
        materialized: list[PreparedRow] = []
        number = 0
        while batch := cursor.fetchmany(_FETCH_BATCH):
            for values in batch:
                number += 1
                try:
                    _, normalized = normalize_source_row(
                        list(names),
                        list(values),
                        table,
                        source="sqlite",
                        row_number=number,
                    )
                except PodNormalizationError as exc:
                    raise PodImportError([f"{table_name}: {exc}"]) from exc
                materialized.append(normalized)
        rows[table_name] = tuple(materialized)
    return rows


def read_pod_target(
    path: Path,
    contract: PodContract,
    *,
    materialize_rows: bool,
    need_preserved_inputs: bool = False,
) -> TargetSnapshot:
    """Read one target snapshot without creating, initializing, or writing anything."""
    if not path.is_file():
        return TargetSnapshot(
            ImportTargetBasis(
                False,
                None,
                FRESH_PROTECTED_STATE_SHA256,
                path.resolve(strict=False),
            ),
            {},
            None,
            (),
        )
    conn: sqlite3.Connection | None = None
    try:
        conn = read_only_connection(path)
        return read_target_snapshot(
            conn,
            path,
            contract,
            materialize_rows=materialize_rows,
            need_preserved_inputs=need_preserved_inputs,
        )
    except sqlite3.Error as exc:
        raise PodImportError([f"Could not read POD target {path}: {exc}"]) from exc
    finally:
        if conn is not None:
            conn.close()


def read_pod_api_context(
    target_path: Path | str,
    contract: PodContract | None = None,
) -> PodApiContext:
    """Read the read-only reconciliation context for one POD target."""
    path = Path(target_path)
    active = contract if contract is not None else load_pod_contract()
    snapshot = read_pod_target(path, active, materialize_rows=True)
    return PodApiContext(
        snapshot.basis,
        snapshot.existing_rows,
        snapshot.reserved_pod_identities or (),
        snapshot.linked_pod_ids,
    )
