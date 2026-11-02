"""Task5B: POD API projection, normalization, and prepared batch assembly."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timezone
from importlib.resources import files
from typing import Any, TypeAlias
from uuid import NAMESPACE_DNS, UUID

from esdc.pod_registry.api_payload import parse_pod_api_payload
from esdc.pod_registry.contract import (
    ColumnContract,
    PodContract,
    TableContract,
    load_pod_contract,
    validate_pod_batch,
)
from esdc.pod_registry.models import PreparedPodBatch, PreparedRow, PreparedScalar
from esdc.pod_registry.normalization import (
    PodNormalizationError,
    canonical_column_map,
    normalize_pod_value,
    normalize_source_row,
)

INPUT_TABLES = (
    "m_project",
    "m_pod",
    "pod_project",
    "pod_monitoring",
    "pod_forecast",
)
NATURAL_KEYS: dict[str, tuple[str, ...]] = {
    "m_project": ("project_id",),
    "m_pod": ("pod_id",),
    "pod_project": ("pod_id_seq", "project_id"),
    "pod_monitoring": ("pod_id_seq", "reported_date"),
    "pod_forecast": ("pod_id_seq", "year_frcast"),
}
# Placeholder POD master rows carry no canonical issued identity, so the
# surrogate sequence is the natural key for them.
PLACEHOLDER_POD_ID_SEQ = 856


class PodApiImportError(ValueError):
    """One projection, ownership, or reconciliation failure before any write."""

    def __init__(self, table_name: str, reason: str, row: int | None = None) -> None:
        self.table_name = table_name
        self.reason = reason
        self.row = row
        location = f"record {row}" if row is not None else "payload"
        super().__init__(f"pod api {table_name} {location}: {reason}")


def pod_stage_labels(contract: PodContract) -> dict[str, int]:
    """Resolve stage labels from the packaged reference seed, not local literals."""
    seed = files("esdc").joinpath("sql").joinpath("pod_reference_seed.sql")
    text = seed.read_text(encoding="utf-8")
    block = re.search(
        r"INSERT INTO r_pod_stage[^;]*?VALUES(.+?)(?:ON CONFLICT|;)", text, re.S
    )
    if block is None:
        raise PodApiImportError("r_pod_stage", "reference seed has no stage rows")
    allowed = set(contract.reference_values.get("r_pod_stage.id", ()))
    labels: dict[str, int] = {}
    for line in block.group(1).split("\n"):
        match = re.match(r"\s*\((\d+),'([^']*)'", line.strip())
        if match is None:
            continue
        stage_id = int(match.group(1))
        if allowed and stage_id not in allowed:
            raise PodApiImportError(
                "r_pod_stage", f"seed stage id {stage_id} is outside the contract"
            )
        labels[match.group(2)] = stage_id
    return labels


def business_natural_key(
    contract: PodContract, table_name: str, row: Mapping[str, Any]
) -> tuple[str, ...]:
    """Return the documented business natural key for one canonical row."""
    columns = NATURAL_KEYS.get(table_name)
    if columns is None:
        raise PodApiImportError(table_name, "has no documented natural key")
    key: list[str] = []
    for name in columns:
        if name not in row:
            raise PodApiImportError(
                table_name, f"natural key column {name!r} is missing"
            )
        value = row[name]
        if table_name == "m_pod" and name == "pod_id" and value in (None, ""):
            key.append(str(row["pod_id_seq"]))
            continue
        if value is None:
            raise PodApiImportError(table_name, f"natural key column {name!r} is blank")
        key.append(
            value.isoformat() if isinstance(value, (datetime, date)) else str(value)
        )
    return tuple(key)


def natural_key_uuid(key: str | Sequence[str], *, version: int = 5) -> UUID:
    """Deterministic natural-key UUID; the version nibble follows the contract."""
    parts = key if isinstance(key, str) else "|".join(str(part) for part in key)
    digest = bytearray(
        hashlib.sha1(NAMESPACE_DNS.bytes + parts.encode("utf-8")).digest()[:16]
    )
    digest[6] = (version << 4) | (digest[6] & 0x0F)
    digest[8] = 0x80 | (digest[8] & 0x3F)
    return UUID(bytes=bytes(digest))


def _input_columns(table: TableContract) -> dict[str, ColumnContract]:
    return {
        column.name: column for column in table.columns if column.ownership == "input"
    }


def _view_headers(
    contract: PodContract, table_name: str
) -> Mapping[str, ColumnContract]:
    view = contract.tables.get(f"{table_name}_api")
    return canonical_column_map(view) if view is not None else {}


# Columns the physical API view does not carry: local secondary identifiers,
# UUIDs, generated audit, and the stage reference resolved from its label.
PROTOCOL_OMISSIONS = frozenset(
    {
        "id",
        "monpod_id",
        "monpod_uuid",
        "frcast_id",
        "frcast_uuid",
        "pod_stage_id",
        "created_at",
        "created_by",
        "updated_at",
        "updated_by",
    }
)


IDENTITY_COPIES = frozenset({"pod_id", "pod_uuid"})
# Duplicate keys reject; only these documented contexts may dedupe identical values.
DEDUPE_EXCEPTIONS = frozenset({"m_project", "pod_forecast"})
# Aliases the projection accepts, recorded under their canonical column name.
EXPLICIT_ALIASES: dict[str, dict[str, str]] = {"m_project": {"name": "project_name"}}
MutableRow: TypeAlias = dict[str, PreparedScalar]


def _mutable(row: PreparedRow) -> MutableRow:
    """Copy the supplied context row so the caller's rows are never mutated."""
    return dict(row)


def _bodies(payload: bytes | Sequence[bytes], table_name: str) -> list[bytes]:
    if isinstance(payload, (bytes, bytearray)):
        return [bytes(payload)]
    if isinstance(payload, str):
        raise PodApiImportError(table_name, "payload body must be raw bytes, not text")
    bodies = [bytes(body) for body in payload]
    if not bodies:
        raise PodApiImportError(table_name, "payload body is empty")
    if table_name != "pod_monitoring" and len(bodies) > 1:
        raise PodApiImportError(table_name, "expects one payload body")
    return bodies


def _canonical_value(
    column: ColumnContract, raw: object, table_name: str, row: int
) -> PreparedScalar:
    try:
        return normalize_pod_value(raw, column, source="api")
    except PodNormalizationError as error:
        raise PodApiImportError(table_name, str(error), row) from error


def _project_record(
    contract: PodContract, table_name: str, record: Mapping[str, object], row: int
) -> tuple[dict[str, PreparedScalar], dict[str, PreparedScalar], object, object]:
    table = contract.tables[table_name]
    input_columns = _input_columns(table)
    aliases = canonical_column_map(table)
    for alias, target in EXPLICIT_ALIASES.get(table_name, {}).items():
        if target in aliases:
            aliases[alias] = aliases[target]
    view = _view_headers(contract, table_name)
    reconciled = frozenset(_reconciled_columns(contract, table_name))
    headers: list[str] = []
    raw_values: list[object] = []
    copies: dict[str, PreparedScalar] = {}
    seen: dict[str, tuple[str, object]] = {}
    stage_label: object = None
    rejected: object = None
    for header, raw in record.items():
        name = str(header)
        # Explicit protocol exceptions: the physical view exposes the stage label
        # and the optional rejection flag instead of the canonical columns.
        if table_name == "pod_monitoring" and name == "pod_stage":
            stage_label = raw
            continue
        if table_name == "m_project" and name == "is_rejected":
            if raw is None:
                continue
            if not isinstance(raw, bool):
                raise PodApiImportError(table_name, "is_rejected must be boolean", row)
            rejected = raw
            continue
        column = aliases.get(name)
        if column is None:
            copy = view.get(name)
            if copy is None:
                raise PodApiImportError(
                    table_name, f"column {name!r} is not declared by the contract", row
                )
            # Read-only copies of master or derived view fields are dropped and are
            # never compared as rounded inequalities; only business identity copies
            # must agree with the authoritative master row.
            if name in IDENTITY_COPIES:
                copies[name] = _canonical_value(copy, raw, table_name, row)
            continue
        first = seen.get(column.name)
        if first is None:
            seen[column.name] = (name, raw)
            # A header resolving through the alias map keeps its canonical name.
            headers.append(column.name)
            raw_values.append(raw)
            continue
        stored_value = _canonical_value(column, first[1], table_name, row)
        value = _canonical_value(column, raw, table_name, row)
        if stored_value == value:
            continue
        if column.name in reconciled and (stored_value is None or value is None):
            # Only locally reconciled ID/UUID/audit metadata merges to the value stated.
            index = headers.index(column.name)
            raw_values[index] = first[1] if stored_value is not None else raw
            continue
        # A business unknown is not merged with a supplied value in either order:
        # two headers resolving to one business column must agree semantically.
        raise PodApiImportError(
            table_name,
            f"alias {name!r} conflicts with column {column.name!r}",
            row,
        )
    try:
        _, projected = normalize_source_row(
            headers, raw_values, table, source="api", row_number=row
        )
    except PodNormalizationError as error:
        raise PodApiImportError(table_name, str(error), row) from error
    for name in input_columns:
        if name in PROTOCOL_OMISSIONS:
            continue
        if name not in projected:
            raise PodApiImportError(
                table_name, f"missing declared business header {name!r}", row
            )
    return projected, copies, stage_label, rejected


AUDIT_COLUMNS = frozenset({"created_at", "created_by", "updated_at", "updated_by"})
UPDATE_AUDIT_COLUMNS = frozenset({"updated_at", "updated_by"})
_IDENTITY_ONLY: dict[str, set[str]] = {
    "m_project": set(),
    "m_pod": set(),
    "pod_project": set(),
    "pod_monitoring": {"pod_id_seq", "pod_id", "reported_date"},
    "pod_forecast": {"pod_id_seq", "year_frcast"},
}


def _is_real(table_name: str, row: Mapping[str, PreparedScalar]) -> bool:
    """A synthetic projection row has no record unless the contract says it does."""
    if table_name in ("m_project", "m_pod", "pod_project"):
        return True
    for name, value in row.items():
        if name in _IDENTITY_ONLY[table_name] or name in AUDIT_COLUMNS:
            continue
        if value is not None:
            return True
    return False


def _resolve_stage_label(
    contract: PodContract,
    row: MutableRow,
    label: object,
    row_number: int,
    labels: Mapping[str, int],
) -> None:
    """Validate a provided label before a synthetic projection row can be ignored."""
    if label is None:
        return
    if not isinstance(label, str) or not label.strip():
        raise PodApiImportError("pod_monitoring", "stage label is blank", row_number)
    resolved = labels.get(label.strip())
    if resolved is None:
        raise PodApiImportError(
            "pod_monitoring",
            f"stage label {label!r} is not a reference stage",
            row_number,
        )
    declared = row.get("pod_stage_id")
    if declared is not None and declared != resolved:
        raise PodApiImportError(
            "pod_monitoring",
            f"stage label {label!r} conflicts with stage id {declared}",
            row_number,
        )
    row["pod_stage_id"] = resolved


def _require_stage(contract: PodContract, row: MutableRow, row_number: int) -> None:
    """A real submission keeps the canonical non-nullable stage."""
    allowed = set(contract.reference_values.get("r_pod_stage.id", ()))
    if row.get("pod_stage_id") is None:
        raise PodApiImportError(
            "pod_monitoring", "stage is blank and was not invented", row_number
        )
    if row["pod_stage_id"] not in allowed:
        raise PodApiImportError(
            "pod_monitoring",
            f"stage id {row['pod_stage_id']} is outside the reference domain",
            row_number,
        )


UUID_COLUMN = {
    "m_project": None,
    "m_pod": "pod_uuid",
    "pod_project": "pod_project_uuid",
    "pod_monitoring": "monpod_uuid",
    "pod_forecast": "frcast_uuid",
}


def _conflicts(
    left: Mapping[str, PreparedScalar],
    right: Mapping[str, PreparedScalar],
    mergeable: frozenset[str] = frozenset(),
    uuid_column: str | None = None,
) -> set[str]:
    """Values the two records state differently for the same canonical row.

    Owned business values must agree, including NULL against 0: NULL is unknown, not
    a blank permission to coalesce. Only locally reconciled ID/UUID/audit metadata
    merges when one side is blank, and two supplied metadata values still conflict.
    """
    conflicts: set[str] = set()
    for name in set(left) | set(right):
        left_value, right_value = left.get(name), right.get(name)
        if uuid_column is not None and name == uuid_column:
            left_value, right_value = _as_uuid(left_value), _as_uuid(right_value)
        if left_value == right_value:
            continue
        if name in mergeable and (left_value is None or right_value is None):
            continue
        conflicts.add(name)
    return conflicts


def _reconciled_columns(contract: PodContract, table_name: str) -> set[str]:
    """Names that may be blank in a payload because they reconcile with prior state."""
    names = set(AUDIT_COLUMNS)
    identifier = _surrogate_id(contract.tables[table_name])
    for name in (UUID_COLUMN.get(table_name), identifier):
        if name is not None:
            names.add(name)
    return names


def _as_uuid(value: object) -> UUID | None:
    """The raw SQLite context carries UUIDs as text; read them semantically."""
    if isinstance(value, UUID):
        return value
    if isinstance(value, str):
        try:
            return UUID(value.strip())
        except ValueError:
            return None
    return None


def _permanent_uuid(
    table: str,
    natural: tuple[str, ...],
    row: Mapping[str, PreparedScalar],
    reserved: Sequence[Mapping[str, PreparedScalar]],
) -> UUID | None:
    """Restore an orphan permanent-ledger UUID without remapping a provided identity."""
    column = UUID_COLUMN.get(table)
    if column is None:
        return None
    pod_id = str(row.get("pod_id") or "")
    matches: list[UUID] = []
    for entry in reserved:
        value = _as_uuid(entry.get(column))
        if value is None:
            continue
        entry_id = entry.get("pod_id")
        entry_seq = entry.get("pod_id_seq")
        if natural[0] not in (
            str(entry_id or ""),
            str(entry_seq or ""),
        ) and pod_id != str(entry_id or ""):
            continue
        if entry_id is not None and pod_id and str(entry_id) != pod_id:
            raise PodApiImportError(
                table, f"permanent identity {value} is not congruent with {natural}"
            )
        if (
            entry_seq is not None
            and row.get("pod_id_seq") is not None
            and str(entry_seq) != str(row.get("pod_id_seq"))
        ):
            raise PodApiImportError(
                table, f"permanent identity {value} is not congruent with {natural}"
            )
        matches.append(value)
    return matches[0] if matches else None


def _check_permanent_identity(
    contract: PodContract,
    table: str,
    natural: tuple[str, ...],
    row: MutableRow,
    reserved: Sequence[Mapping[str, PreparedScalar]],
) -> None:
    """The permanent m_pod identity is immutable; a provided one is never remapped."""
    column = UUID_COLUMN.get(table)
    if column is None:
        return
    permanent = _permanent_uuid(table, natural, row, reserved)
    supplied = _as_uuid(row.get(column))
    if supplied is not None and permanent is not None and supplied != permanent:
        raise PodApiImportError(table, f"conflicting UUID for natural key {natural}")


def _reconciled_uuid(
    contract: PodContract,
    table: str,
    natural: tuple[str, ...],
    row: MutableRow,
    prior: Mapping[str, PreparedScalar] | None,
    reserved: Sequence[Mapping[str, PreparedScalar]],
) -> UUID | None:
    column = UUID_COLUMN.get(table)
    if column is None:
        return None
    supplied = _as_uuid(row.get(column))
    current = _as_uuid(prior.get(column)) if prior is not None else None
    permanent = _permanent_uuid(table, natural, row, reserved)
    if supplied is not None:
        # Only the permanent m_pod identity is immutable; a supplied secondary UUID is a
        # legitimate update even when it replaces a different prior record.
        if table == "m_pod" and permanent is not None and supplied != permanent:
            raise PodApiImportError(
                table, f"conflicting UUID for natural key {natural}"
            )
        return supplied
    if current is not None:
        return current
    if permanent is not None:
        return permanent
    version = next(
        (
            spec.uuid_version or 5
            for spec in contract.tables[table].columns
            if spec.name == column
        ),
        5,
    )
    return natural_key_uuid(natural, version=version)


@dataclass
class _ProjectedEntry:
    natural: tuple[str, ...]
    row: MutableRow
    copies: MutableRow
    number: int


def _merge_entry(
    contract: PodContract,
    table: str,
    stored: _ProjectedEntry,
    incoming: _ProjectedEntry,
) -> _ProjectedEntry:
    """Only the documented report-context exceptions dedupe a repeated key."""
    mergeable = frozenset(_reconciled_columns(contract, table))
    conflicts = _conflicts(stored.row, incoming.row, mergeable, UUID_COLUMN.get(table))
    conflicts |= _conflicts(stored.copies, incoming.copies, frozenset(), "pod_uuid")
    winner, loser = stored, incoming
    if table == "m_project" and bool(stored.row.get("is_active")) != bool(
        incoming.row.get("is_active")
    ):
        # Active priority is the documented exception to the agreement rule.
        winner = incoming if incoming.row.get("is_active") else stored
        loser = stored if winner is incoming else incoming
    elif conflicts:
        raise PodApiImportError(
            table,
            f"conflicting values for natural key {stored.natural}",
            incoming.number,
        )
    elif table not in DEDUPE_EXCEPTIONS:
        # A repeated key rejects. Only the documented report-context exceptions may
        # dedupe identical values; the other tables keep one record per key.
        raise PodApiImportError(
            table,
            f"duplicate key for natural key {stored.natural}",
            incoming.number,
        )
    for name, value in loser.row.items():
        if winner.row.get(name) is None and value is not None:
            winner.row[name] = value
    for name, value in loser.copies.items():
        if winner.copies.get(name) is None and value is not None:
            winner.copies[name] = value
    return winner


def _authoritative_master(
    table: str,
    row: Mapping[str, PreparedScalar],
    masters: Sequence[Mapping[str, PreparedScalar]],
) -> Mapping[str, PreparedScalar] | None:
    pod_seq = str(row.get("pod_id_seq") or "")
    pod_id = str(row.get("pod_id") or "")
    for master in masters:
        if pod_seq and str(master.get("pod_id_seq") or "") == pod_seq:
            return master
    for master in masters:
        if pod_id and str(master.get("pod_id") or "") == pod_id:
            return master
    return None


def _reconcile_copies(
    table: str,
    row: MutableRow,
    copies: MutableRow,
    master: Mapping[str, PreparedScalar],
    number: int,
) -> None:
    """Fill a blank declared copy from the master; never invent a stated one."""
    for name in sorted(IDENTITY_COPIES | {"pod_id_seq"}):
        authoritative = (
            _as_uuid(master.get(name)) if name == "pod_uuid" else master.get(name)
        )
        stated = row.get(name) if name in row else copies.get(name)
        if stated is None:
            if name in row and authoritative is not None:
                row[name] = authoritative
            continue
        stated = _as_uuid(stated) if name == "pod_uuid" else stated
        if authoritative is not None and stated != authoritative:
            raise PodApiImportError(
                table, f"identity copy {name} disagrees with m_pod", number
            )


def _surrogate_id(table: TableContract) -> str | None:
    first = table.primary_key[0] if table.primary_key else None
    return None if first in (None, "project_id") else first


def prepare_pod_api_batch(
    payloads: Mapping[str, bytes | Sequence[bytes]],
    *,
    existing_rows: Mapping[str, Sequence[PreparedRow]] | None = None,
    reserved_pod_identities: Sequence[PreparedRow] = (),
    monitoring_years: Sequence[int] | None = None,
    actor: str = "FA:fetch",
    observed_at: datetime | None = None,
    contract: PodContract | None = None,
) -> PreparedPodBatch:
    """Project API payloads into the five canonical prepared tables, without writing."""
    active = contract if contract is not None else load_pod_contract()
    observed = observed_at if observed_at is not None else datetime.now(timezone.utc)
    labels = pod_stage_labels(active)

    for key in payloads:
        if key not in INPUT_TABLES:
            raise PodApiImportError(
                str(key), f"payload key {str(key)!r} is not a canonical POD input table"
            )
    for table in INPUT_TABLES:
        if table not in payloads:
            raise PodApiImportError(
                table, f"payload key {table!r} is missing; all five tables are required"
            )

    prior_state: dict[str, dict[tuple[str, ...], MutableRow]] = {}
    for table in INPUT_TABLES:
        rows: dict[tuple[str, ...], MutableRow] = {}
        for row in (existing_rows or {}).get(table, ()):
            natural = business_natural_key(active, table, row)
            if natural in rows and _conflicts(
                rows[natural],
                row,
                frozenset(_reconciled_columns(active, table)),
                UUID_COLUMN.get(table),
            ):
                raise PodApiImportError(
                    table, f"conflicting existing rows for natural key {natural}"
                )
            rows[natural] = _mutable(row)
        prior_state[table] = rows

    selected_years = set(monitoring_years) if monitoring_years is not None else None
    output: dict[str, dict[tuple[str, ...], _ProjectedEntry]] = {
        table: {} for table in INPUT_TABLES
    }

    # Prior state is reconciliation context, not an output seed: the canonical
    # tables carry only incoming records, and monitoring additionally keeps prior
    # facts reported outside the selected years.
    for table in INPUT_TABLES:
        number = 0
        for body in _bodies(payloads[table], table):
            for record in parse_pod_api_payload(
                body, table_name=table, contract=active
            ):
                number += 1
                row, copies, stage_label, rejected = _project_record(
                    active, table, record, number
                )
                if table == "pod_monitoring":
                    _resolve_stage_label(active, row, stage_label, number, labels)
                if rejected is True:
                    continue
                if not _is_real(table, row):
                    continue
                if table == "pod_monitoring":
                    _require_stage(active, row, number)
                    reported = row.get("reported_date")
                    if (
                        selected_years is not None
                        and isinstance(reported, date)
                        and reported.year not in selected_years
                    ):
                        continue
                masters = [entry.row for entry in output["m_pod"].values()]
                masters.extend(prior_state["m_pod"].values())
                master = _authoritative_master(table, row, masters)
                if master is not None:
                    _reconcile_copies(table, row, copies, master, number)
                natural = business_natural_key(active, table, row)
                if table == "m_pod":
                    _check_permanent_identity(
                        active, table, natural, row, reserved_pod_identities
                    )
                entry = _ProjectedEntry(natural, row, copies, number)
                stored = output[table].get(natural)
                if stored is None:
                    output[table][natural] = entry
                else:
                    output[table][natural] = _merge_entry(active, table, stored, entry)

    for natural, row in sorted(prior_state["pod_monitoring"].items()):
        if selected_years is None or natural in output["pod_monitoring"]:
            continue
        reported = row.get("reported_date")
        if isinstance(reported, date) and reported.year in selected_years:
            continue
        output["pod_monitoring"][natural] = _ProjectedEntry(natural, row, {}, 0)

    # Identifiers and creation history reconcile with the preserved record: a supplied
    # value is preserved even when it replaces a different prior record, and a blank
    # one reuses the prior value by natural key. Update audit is not inherited, it is
    # stamped by this preparation. Only the permanent m_pod identity is immutable, and
    # that congruence is enforced against the authoritative master row.
    for table in INPUT_TABLES:
        column = UUID_COLUMN.get(table)
        for natural, entry in sorted(output[table].items()):
            prior = prior_state[table].get(natural)
            if prior is None:
                continue
            for name in sorted(_reconciled_columns(active, table)):
                if name in UPDATE_AUDIT_COLUMNS:
                    continue
                old = prior.get(name)
                if entry.row.get(name) is None and old is not None:
                    entry.row[name] = _as_uuid(old) if name == column else old

    # Deterministic identifiers for the records a payload leaves blank.
    for table in INPUT_TABLES:
        entries = output[table]
        identifier = _surrogate_id(active.tables[table])
        used: list[int] = []
        if identifier is not None:
            sources = list(prior_state[table].values()) + [
                entry.row for entry in entries.values()
            ]
            for row in sources:
                value = row.get(identifier)
                if isinstance(value, (int, str)):
                    used.append(int(value))
        next_id = max(used or [0])
        for natural in sorted(entries):
            row = entries[natural].row
            column = UUID_COLUMN.get(table)
            if column is not None:
                value = _reconciled_uuid(
                    active,
                    table,
                    natural,
                    row,
                    prior_state[table].get(natural),
                    reserved_pod_identities,
                )
                if value is not None:
                    row[column] = value
            if identifier is not None and row.get(identifier) is None:
                next_id += 1
                row[identifier] = next_id
            for name in sorted(AUDIT_COLUMNS):
                if row.get(name) is None:
                    row[name] = actor if name.endswith("by") else observed

    # Copied master identity is also checked after the master itself is reconciled.
    final_masters = {
        str(entry.row.get("pod_id_seq")): entry.row
        for entry in output["m_pod"].values()
    }
    for table in ("pod_project", "pod_monitoring", "pod_forecast"):
        for entry in sorted(output[table].values(), key=lambda item: item.natural):
            master = final_masters.get(str(entry.row.get("pod_id_seq")))
            if master is None:
                continue
            for name in sorted(IDENTITY_COPIES):
                copy = entry.row.get(name, entry.copies.get(name))
                authoritative = master.get(name)
                if name == "pod_uuid":
                    copy, authoritative = _as_uuid(copy), _as_uuid(authoritative)
                if (
                    copy is not None
                    and authoritative is not None
                    and copy != authoritative
                ):
                    raise PodApiImportError(
                        table,
                        f"identity copy {name} disagrees with m_pod",
                        entry.number,
                    )

    batch = PreparedPodBatch(
        source="api",
        contract_id=active.contract_id,
        contract_version=active.contract_version,
        dictionary_version=active.dictionary_version,
        headers={
            table: tuple(column.name for column in active.tables[table].columns)
            for table in INPUT_TABLES
        },
        rows={
            table: tuple(
                entry
                if all(name in entry for name in column_names(active.tables[table]))
                else {
                    name: entry.get(name) for name in column_names(active.tables[table])
                }
                for entry in (
                    output[table][natural].row for natural in sorted(output[table])
                )
            )
            for table in INPUT_TABLES
        },
        profile="full",
    )
    issues = validate_pod_batch(batch, active)
    if issues:
        issue = issues[0]
        raise PodApiImportError(
            issue.table, f"validation failed: {issue.message}", issue.row
        )
    return batch


def column_names(table: TableContract) -> tuple[str, ...]:
    return tuple(column.name for column in table.columns)
