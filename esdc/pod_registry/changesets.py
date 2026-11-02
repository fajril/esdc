"""Controlled current-contract portal edits over the canonical POD source."""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from uuid import uuid4

from esdc.pod_registry.api_context import (
    PodImportError,
    read_pod_api_context,
    read_source_identity_ledger_rows,
    target_protected_state_digest,
)
from esdc.pod_registry.contract import (
    ColumnContract,
    load_pod_contract,
    validate_pod_batch,
)
from esdc.pod_registry.identity import (
    issue_identity,
    next_available_daily_slot,
    resolve_revision_number,
)
from esdc.pod_registry.identity_key import ApprovalKey
from esdc.pod_registry.migrations import _atomic_writer, open_registry_v2_writer
from esdc.pod_registry.models import PreparedPodBatch, PreparedScalar
from esdc.pod_registry.normalization import normalize_pod_value
from esdc.pod_registry.read_models import refresh_pod_after_source_change
from esdc.pod_registry.sqlite_import import INPUT_TABLES, apply_pod_batch_in_transaction
from esdc.pod_registry.store import get_esdc_sqlite_path

_ACTOR = "FA:portal"
_IDENTITY = frozenset({"approval_date", "institution_id", "pod_type_id", "rev_num"})


@dataclass
class RowError:
    kind: str
    index: int
    message: str
    field: str | None = None


@dataclass
class ChangesetResult:
    ok: bool
    applied: dict[str, int] = field(default_factory=dict)
    generated: list[dict] = field(default_factory=list)
    errors: list[RowError] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    source_generation: int | None = None
    source_committed: bool = False
    conflict: bool = False


class _RejectedError(ValueError):
    def __init__(
        self,
        kind: str,
        index: int,
        message: str,
        column: str | None = None,
        *,
        errors: list[RowError] | None = None,
    ):
        super().__init__(message)
        self.errors = errors or [RowError(kind, index, message, column)]
        self.issue = self.errors[0]


def _value(value: object, column: ColumnContract) -> PreparedScalar:
    # Form editors return text; imports keep their own stricter source adapters.
    if isinstance(value, str):
        if column.logical_type == "integer":
            if not re.fullmatch(r"[+-]?[0-9]+", value):
                raise ValueError(f"{column.name} must be an integer")
            value = int(value)
        elif column.logical_type == "boolean":
            if value not in {"true", "false", "0", "1"}:
                raise ValueError(f"{column.name} must be true, false, 0, or 1")
            value = value in {"true", "1"}
    return normalize_pod_value(value, column, source="api")


def _audit(row: dict[str, PreparedScalar], *, created: bool) -> None:
    now = datetime.now(timezone.utc)
    for key, value in (("updated_at", now), ("updated_by", _ACTOR)):
        if key in row:
            row[key] = value
    if created:
        for key, value in (("created_at", now), ("created_by", _ACTOR)):
            if key in row:
                row[key] = value


def _shape(changes: dict) -> tuple[int, dict[str, list[dict]]]:
    if not isinstance(changes, dict) or set(changes) - {
        "source_generation",
        "inserts",
        "updates",
        "deletes",
    }:
        raise _RejectedError("batch", -1, "Unknown changeset fields")
    generation = changes.get("source_generation")
    if (
        isinstance(generation, bool)
        or not isinstance(generation, int)
        or generation < 0
    ):
        raise _RejectedError(
            "batch", -1, "A current source_generation is required; reload the table"
        )
    groups = {}
    for kind in ("inserts", "updates", "deletes"):
        group = changes.get(kind, [])
        if not isinstance(group, list) or any(
            not isinstance(row, dict) for row in group
        ):
            raise _RejectedError("batch", -1, f"{kind} must be an array of row objects")
        groups[kind] = group
    if sum(map(len, groups.values())) > 500:
        raise _RejectedError("batch", -1, "Save at most 500 changed rows at once")
    return generation, groups


def _new_row(
    conn: sqlite3.Connection, table: str, frame: dict, supplied: dict
) -> dict[str, PreparedScalar]:
    contract = load_pod_contract().tables[table]
    result: dict[str, PreparedScalar] = {
        column.name: None for column in contract.columns
    }
    result.update(supplied)
    for column in contract.columns:
        if column.logical_type == "uuid":
            result[column.name] = uuid4()
    if table == "m_pod":
        predecessors = supplied.pop("predecessors", [])
        result.pop("predecessors", None)
        if not isinstance(predecessors, list):
            raise ValueError("predecessors must be an array of POD sequence integers")
        sequence_column = load_pod_contract().tables["m_pod"].columns[0]
        predecessors = [_value(value, sequence_column) for value in predecessors]
        if any(not isinstance(pk, int) for pk in predecessors):
            raise ValueError("predecessor POD sequences must be integers")
        if len(set(predecessors)) != len(predecessors):
            raise ValueError("predecessors contains duplicate PODs")
        prior = {row["pod_id_seq"]: row for row in frame["m_pod"]}
        if any(pk not in prior for pk in predecessors):
            raise ValueError("a predecessor POD does not exist")
        if any(prior[pk]["superseded_by"] is not None for pk in predecessors):
            raise ValueError("a predecessor is already superseded")
        revision = resolve_revision_number(
            [int(str(prior[pk]["rev_num"])) for pk in predecessors],
            explicit_rev_num=supplied.get("rev_num"),
            pod_name=supplied.get("pod_name"),
        )
        approved = supplied.get("approval_date")
        if not isinstance(approved, date) or isinstance(approved, datetime):
            raise ValueError("approval_date is required as an ISO calendar date")
        institution = supplied.get("institution_id")
        pod_type = supplied.get("pod_type_id")
        if isinstance(institution, bool) or not isinstance(institution, int):
            raise ValueError("institution_id is required as an integer")
        if isinstance(pod_type, bool) or not isinstance(pod_type, int):
            raise ValueError("pod_type_id is required as an integer")
        issued = issue_identity(
            conn,
            key=ApprovalKey(
                approved,
                next_available_daily_slot(conn, approved),
                institution,
                pod_type,
                revision,
            ),
            actor=_ACTOR,
        )
        result.update(
            pod_id_seq=issued.pod_id_seq,
            pod_uuid=issued.pod_uuid,
            approval_date=approved,
            pod_order_no=issued.pod_order_no,
            pod_id=issued.pod_id,
            rev_num=revision,
        )
        for pk in predecessors:
            prior[pk].update(superseded_by=issued.pod_id_seq, is_latest=False)
            _audit(prior[pk], created=False)
    elif table != "m_project":
        pk = contract.primary_key[0]
        sequence = conn.execute(
            "SELECT seq FROM sqlite_sequence WHERE name=?", (table,)
        ).fetchone()
        result[pk] = (
            max(
                [int(sequence[0]) if sequence else 0]
                + [int(str(row[pk])) for row in frame[table]]
            )
            + 1
        )
    if table == "pod_project":
        result["is_auto_inferred"] = False
    if table == "pod_monitoring":
        master = next(
            (
                row
                for row in frame["m_pod"]
                if row["pod_id_seq"] == result["pod_id_seq"]
            ),
            None,
        )
        result["pod_id"] = None if master is None else master["pod_id"]
        result["is_completed"] = False
    _audit(result, created=True)
    return result


def _edit_inputs(
    conn: sqlite3.Connection,
    table: str,
    frame: dict,
    groups: dict,
    affected: dict[tuple, tuple[str, int]],
) -> list[dict]:
    spec = load_pod_contract().tables[table]
    columns = {col.name: col for col in spec.columns}
    keys = spec.primary_key
    targets = {tuple(row[key] for key in keys): row for row in frame[table]}
    seen: set[tuple] = set()
    generated = []
    for operation in ("deletes", "updates", "inserts"):
        for index, raw in enumerate(groups[operation]):
            try:
                special = (
                    {"predecessors"}
                    if table == "m_pod" and operation == "inserts"
                    else set()
                )
                allowed = (
                    set(keys)
                    if operation == "deletes"
                    else {c.name for c in spec.columns if c.ownership == "input"}
                    | set(keys)
                    | special
                )
                if set(raw) - allowed:
                    raise ValueError(
                        "Unknown or read-only fields: "
                        + ", ".join(sorted(set(raw) - allowed))
                    )
                key: tuple[PreparedScalar, ...] = ()
                if operation != "inserts":
                    if set(keys) - set(raw):
                        raise ValueError("All primary-key fields are required")
                    key = tuple(_value(raw[name], columns[name]) for name in keys)
                    if key in seen or key not in targets:
                        raise ValueError(
                            "Row is missing or changed more than once in this save"
                        )
                    seen.add(key)
                else:
                    generated_keys = {
                        name for name in keys if columns[name].ownership != "input"
                    }
                    if generated_keys & set(raw):
                        raise ValueError("Primary keys are assigned by the server")
                if operation == "deletes":
                    del targets[key]
                    frame[table].remove(
                        next(
                            row
                            for row in frame[table]
                            if tuple(row[k] for k in keys) == key
                        )
                    )
                    continue
                mutable = {}
                for name, value in raw.items():
                    if name in special:
                        mutable[name] = value
                    elif operation == "updates" and name in keys:
                        continue
                    elif (
                        operation == "updates"
                        and table == "m_pod"
                        and name in _IDENTITY
                    ):
                        raise ValueError(
                            f"Issued identity component {name} cannot be edited"
                        )
                    else:
                        mutable[name] = _value(value, columns[name])
                if operation == "updates":
                    row = targets[key]
                    if "pod_name" in mutable and table == "m_pod":
                        resolve_revision_number(
                            [],
                            explicit_rev_num=row["rev_num"],
                            pod_name=mutable["pod_name"],
                        )
                    row.update(mutable)
                    _audit(row, created=False)
                else:
                    row = _new_row(conn, table, frame, mutable)
                    key = tuple(row[k] for k in keys)
                    if key in targets or key in seen:
                        raise ValueError(
                            "Duplicate primary key in the final input state"
                        )
                    seen.add(key)
                    targets[key] = row
                    frame[table].append(row)
                    generated.append(
                        {
                            name: str(value)
                            if name.endswith("uuid")
                            or (isinstance(value, int) and abs(value) > 2**53 - 1)
                            else value
                            for name, value in row.items()
                            if name in keys or name in {"pod_id", "pod_uuid"}
                        }
                    )
                affected[key] = (operation.removesuffix("s"), index)
            except (ValueError, TypeError) as exc:
                raise _RejectedError(
                    operation.removesuffix("s"), index, str(exc)
                ) from exc
    for row in frame["m_pod"]:
        row["is_latest"] = row["superseded_by"] is None
    pod_ids = {row["pod_id_seq"]: row["pod_id"] for row in frame["m_pod"]}
    for row in frame["pod_monitoring"]:
        row["pod_id"] = pod_ids.get(row["pod_id_seq"])
    return generated


def _edit_links(conn: sqlite3.Connection, groups: dict) -> None:
    if groups["updates"]:
        raise _RejectedError(
            "update", 0, "Document link keys are immutable; delete and add the link"
        )
    seen = set()
    for operation in ("deletes", "inserts"):
        for index, row in enumerate(groups[operation]):
            if set(row) != {"pod_id_seq", "doc_id"} or not isinstance(
                row["doc_id"], str
            ):
                raise _RejectedError(
                    operation.removesuffix("s"),
                    index,
                    "A POD sequence integer and document ID are required",
                )
            column = load_pod_contract().tables["m_pod"].columns[0]
            pod_sequence = _value(row["pod_id_seq"], column)
            if (
                not isinstance(pod_sequence, int)
                or not -(2**63) <= pod_sequence < 2**63
            ):
                raise _RejectedError(
                    "validation", index, "POD sequence is outside BIGINT range"
                )
            key = (pod_sequence, row["doc_id"])
            if key in seen:
                raise _RejectedError(
                    "batch", index, "A document link is changed more than once"
                )
            seen.add(key)
            if operation == "inserts":
                exists = conn.execute(
                    "SELECT 1 FROM sqlite_master "
                    "WHERE type='table' AND name='documents'"
                ).fetchone()
                if (
                    exists is None
                    or conn.execute(
                        "SELECT 1 FROM documents WHERE doc_id=?", (key[1],)
                    ).fetchone()
                    is None
                ):
                    raise _RejectedError(
                        "insert", index, "The selected document does not exist"
                    )
                conn.execute(
                    "INSERT INTO pod_document(pod_id_seq,doc_id) VALUES (?,?)", key
                )
            else:
                deleted = conn.execute(
                    "DELETE FROM pod_document WHERE pod_id_seq=? AND doc_id=?", key
                )
                if deleted.rowcount != 1:
                    raise _RejectedError(
                        "delete", index, "The selected document link no longer exists"
                    )


def apply_changeset(
    table: str, changes: dict, sqlite_path: Path | None = None
) -> ChangesetResult:
    """Commit one validated source generation; publish it separately."""
    if table not in {*INPUT_TABLES, "pod_document"}:
        raise ValueError(f"Table {table!r} is not an editable POD input")
    try:
        expected, groups = _shape(changes)
        path = Path(sqlite_path) if sqlite_path is not None else get_esdc_sqlite_path()
        context = read_pod_api_context(path)
        if context.target_basis.source_generation is None:
            raise _RejectedError(
                "source",
                -1,
                "POD data is not initialized; fetch or load the current schema first",
            )
        if expected != context.target_basis.source_generation:
            return ChangesetResult(
                False,
                errors=[
                    RowError(
                        "conflict",
                        -1,
                        "POD source changed; keep this draft and reload current data",
                    )
                ],
                source_generation=context.target_basis.source_generation,
                conflict=True,
            )
        if not any(groups.values()):
            return ChangesetResult(
                True, source_generation=expected, applied=dict.fromkeys(groups, 0)
            )
        frame = {
            name: [dict(row) for row in context.existing_rows[name]]
            for name in INPUT_TABLES
        }
        conn = open_registry_v2_writer(path)
        try:
            with _atomic_writer(conn):
                current = conn.execute(
                    "SELECT source_generation FROM pod_registry_state "
                    "WHERE singleton_id=1"
                ).fetchone()
                if (
                    current is None
                    or int(current[0]) != expected
                    or target_protected_state_digest(conn)
                    != context.target_basis.protected_state_sha256
                ):
                    raise _RejectedError(
                        "conflict",
                        -1,
                        "POD source or protected links changed; "
                        "keep this draft and reload",
                    )
                generated = []
                if table == "pod_document":
                    _edit_links(conn, groups)
                    refresh_pod_after_source_change(conn, actor=_ACTOR)
                    state = conn.execute(
                        "SELECT source_generation FROM pod_registry_state "
                        "WHERE singleton_id=1"
                    ).fetchone()
                    if state is None:
                        raise RuntimeError("POD registry state disappeared during save")
                    generation = int(state[0])
                else:
                    affected: dict[tuple, tuple[str, int]] = {}
                    generated = _edit_inputs(conn, table, frame, groups, affected)
                    contract = load_pod_contract()
                    batch = PreparedPodBatch(
                        source="portal",
                        contract_id=contract.contract_id,
                        contract_version=contract.contract_version,
                        dictionary_version=contract.dictionary_version,
                        headers={
                            name: tuple(c.name for c in contract.tables[name].columns)
                            for name in INPUT_TABLES
                        },
                        rows=frame,
                        identity_ledger=read_source_identity_ledger_rows(conn),
                        target_basis=context.target_basis,
                    )
                    issues = validate_pod_batch(batch, contract)
                    if issues:
                        errors = []
                        for issue in issues:
                            kind, index = "validation", -1
                            location = issue.table
                            if issue.row and issue.table in frame:
                                record = frame[issue.table][issue.row - 1]
                                keys = contract.tables[issue.table].primary_key
                                key = tuple(record[name] for name in keys)
                                location += (
                                    " ("
                                    + ", ".join(
                                        f"{name}={record[name]}" for name in keys
                                    )
                                    + ")"
                                )
                                if issue.table == table:
                                    kind, index = affected.get(key, (kind, index))
                            errors.append(
                                RowError(
                                    kind,
                                    index,
                                    f"{location}.{issue.column or '*'}: "
                                    f"{issue.message}",
                                    issue.column,
                                )
                            )
                        raise _RejectedError(
                            "validation",
                            -1,
                            "POD contract validation failed",
                            errors=errors,
                        )
                    generation = apply_pod_batch_in_transaction(
                        conn, batch, _ACTOR
                    ).source_generation
        finally:
            conn.close()
        return ChangesetResult(
            True,
            applied={kind: len(group) for kind, group in groups.items()},
            generated=generated,
            source_generation=generation,
            source_committed=True,
        )
    except _RejectedError as exc:
        return ChangesetResult(
            False, errors=exc.errors, conflict=exc.issue.kind == "conflict"
        )
    except (PodImportError, sqlite3.IntegrityError, ValueError) as exc:
        return ChangesetResult(False, errors=[RowError("validation", -1, str(exc))])
