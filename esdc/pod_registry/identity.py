"""Permanent POD business-identity allocation for the v2 registry."""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime
from uuid import UUID, uuid4

from esdc.pod_registry.identity_key import (
    ApprovalKey,
)
from esdc.pod_registry.identity_key import (
    validate_key as _validate_key,
)
from esdc.pod_registry.migrations import _ensure_registry_writer

_REVISION_LABEL = re.compile(
    r"\brevisi\b(?:\s*[-:]\s*|\s+)\(?\s*"
    r"(?P<label>\d+|ke(?:-\s*|\s+)\d+|pertama|kedua|ketiga|keempat|kelima|keenam|"
    r"ketujuh|kedelapan|kesembilan|kesepuluh|[ivxlcdm]+)\b\s*\)?",
    re.IGNORECASE,
)
_REVISION_ORDINALS = {
    "pertama": 1,
    "kedua": 2,
    "ketiga": 3,
    "keempat": 4,
    "kelima": 5,
    "keenam": 6,
    "ketujuh": 7,
    "kedelapan": 8,
    "kesembilan": 9,
    "kesepuluh": 10,
}
_REVISION_ROMAN = {
    "i": 1,
    "ii": 2,
    "iii": 3,
    "iv": 4,
    "v": 5,
    "vi": 6,
    "vii": 7,
    "viii": 8,
    "ix": 9,
}


@dataclass(frozen=True, slots=True)
class IssuedIdentity:
    """Identity reservation written to the append-only ledger."""

    pod_id_seq: int
    pod_uuid: UUID
    approval_date: date
    daily_slot: int
    pod_order_no: int
    pod_id: str
    actor: str


def explicit_revision_from_name(pod_name: str) -> int | None:
    """Read a strongly adjacent numeric or Indonesian ordinal revision label.

    Free-standing numbers elsewhere in the name and Roman revision labels
    outside I–IX are not inferred. Conflicting explicit labels raise so callers
    can review them.
    """
    labels = [
        match.group("label").casefold() for match in _REVISION_LABEL.finditer(pod_name)
    ]
    if not labels:
        return None

    revisions: set[int] = set()
    for label in labels:
        if label in _REVISION_ORDINALS:
            revision = _REVISION_ORDINALS[label]
        elif label in _REVISION_ROMAN:
            revision = _REVISION_ROMAN[label]
        elif label.startswith("ke"):
            number_match = re.search(r"\d+$", label)
            if number_match is None:
                raise ValueError(f"unsupported explicit revision label: {label!r}")
            revision = int(number_match.group())
        elif label.isdecimal():
            revision = int(label)
        else:
            raise ValueError(f"unsupported explicit revision label: {label!r}")
        if not 0 <= revision <= 9:
            raise ValueError("explicit revision must be between 0 and 9")
        revisions.add(revision)
    if len(revisions) > 1:
        raise ValueError("pod_name contains conflicting explicit revision labels")
    return revisions.pop()


def resolve_revision_number(
    predecessor_rev_nums: Sequence[int],
    *,
    explicit_rev_num: int | None = None,
    pod_name: str | None = None,
) -> int:
    """Resolve a numeric revision label before allocating a new POD identity.

    A strong numeric label in ``pod_name`` or ``explicit_rev_num`` takes
    precedence. Otherwise, use zero when there are no predecessors or one more
    than the highest predecessor revision. Numeric and Indonesian ordinal
    labels plus Roman I–IX are accepted adjacent to ``Revisi``; phases and
    free-standing numbers elsewhere in the name are not inferred.
    """
    revisions = (*predecessor_rev_nums,)
    for revision in revisions:
        if isinstance(revision, bool) or not isinstance(revision, int):
            raise TypeError("predecessor revisions must be integers")
        if not 0 <= revision <= 9:
            raise ValueError("predecessor revisions must be between 0 and 9")

    name_revision = (
        explicit_revision_from_name(pod_name) if pod_name is not None else None
    )
    if explicit_rev_num is not None:
        if isinstance(explicit_rev_num, bool) or not isinstance(explicit_rev_num, int):
            raise TypeError("explicit revision must be an integer")
        if not 0 <= explicit_rev_num <= 9:
            raise ValueError("explicit revision must be between 0 and 9")
    if (
        name_revision is not None
        and explicit_rev_num is not None
        and name_revision != explicit_rev_num
    ):
        raise ValueError("explicit revision conflicts with the revision in pod_name")
    if name_revision is not None:
        explicit_rev_num = name_revision
    if explicit_rev_num is not None:
        if not 0 <= explicit_rev_num <= 9:
            raise ValueError("explicit revision must be between 0 and 9")
        return explicit_rev_num

    resolved = max(revisions, default=-1) + 1
    if resolved > 9:
        raise ValueError("resolved revision exceeds the one-digit identity range")
    return resolved


def next_available_daily_slot(conn: sqlite3.Connection, approval_date: date) -> int:
    """Return the first ledger-free slot for a date, without reserving it.

    Call ``issue_identity`` inside the caller's writer transaction to reserve
    the returned slot; a slot remains unavailable after its master row is
    deleted because reservations live in the permanent ledger.
    """
    if not isinstance(approval_date, date) or isinstance(approval_date, datetime):
        raise TypeError("approval_date must be a datetime.date")
    if conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' "
        "AND name = '_pod_identity_ledger'"
    ).fetchone() is None:
        raise RuntimeError("initialize_registry_v2 must run before slot selection")

    reserved = {
        int(row[0])
        for row in conn.execute(
            "SELECT daily_slot FROM _pod_identity_ledger WHERE approval_date = ?",
            (approval_date.isoformat(),),
        )
    }
    for daily_slot in range(1, 21):
        if daily_slot not in reserved:
            return daily_slot
    raise ValueError("all 20 approval slots for this date have been issued")


@contextmanager
def _identity_transaction(conn: sqlite3.Connection) -> Iterator[None]:
    owns_transaction = not conn.in_transaction
    savepoint = "pod_identity_issue"
    if owns_transaction:
        conn.execute("BEGIN IMMEDIATE")
    else:
        conn.execute(f"SAVEPOINT {savepoint}")
    try:
        yield
    except BaseException:
        if owns_transaction:
            conn.rollback()
        else:
            conn.execute(f"ROLLBACK TO {savepoint}")
            conn.execute(f"RELEASE {savepoint}")
        raise
    else:
        if owns_transaction:
            conn.commit()
        else:
            conn.execute(f"RELEASE {savepoint}")


def _require_v2_registry(conn: sqlite3.Connection) -> None:
    state_table = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' "
        "AND name = 'pod_registry_state'"
    ).fetchone()
    if state_table is None:
        raise RuntimeError(
            "initialize_registry_v2 must run before issuing POD identities"
        )
    state = conn.execute(
        "SELECT schema_version FROM pod_registry_state WHERE singleton_id = 1"
    ).fetchone()
    if state is None or state[0] != 2:
        raise RuntimeError("POD identity issuance requires registry schema version 2")


def _next_pod_id_seq(conn: sqlite3.Connection) -> int:
    master = conn.execute("SELECT COALESCE(MAX(pod_id_seq), 0) FROM m_pod").fetchone()
    ledger = conn.execute(
        "SELECT COALESCE(MAX(pod_id_seq), 0) FROM _pod_identity_ledger"
    ).fetchone()
    sequence = conn.execute(
        "SELECT seq FROM sqlite_sequence WHERE name = 'm_pod'"
    ).fetchone()
    return max(int(master[0]), int(ledger[0]), int(sequence[0]) if sequence else 0) + 1


def issue_identity(
    conn: sqlite3.Connection,
    *,
    key: ApprovalKey,
    actor: str,
) -> IssuedIdentity:
    """Reserve an immutable identity inside the caller's transaction when present.

    A call made outside a transaction owns and commits its reservation. Callers
    that also insert ``m_pod`` should begin their writer transaction first so
    both writes commit or roll back together.
    """
    _ensure_registry_writer(conn)
    year, order_no, pod_id = _validate_key(key)
    if not actor.startswith("FA:") or not actor[3:].strip():
        raise ValueError("actor must preserve an FA:<action> audit suffix")
    if conn.execute("PRAGMA foreign_keys").fetchone()[0] != 1:
        raise RuntimeError("POD identity issuance requires foreign_keys=ON")

    with _identity_transaction(conn):
        _require_v2_registry(conn)
        if conn.execute(
            "SELECT 1 FROM r_institution WHERE id = ?", (key.institution_id,)
        ).fetchone() is None:
            raise ValueError(f"unknown institution_id: {key.institution_id}")
        if conn.execute(
            "SELECT 1 FROM r_pod_type WHERE id = ?", (key.pod_type_id,)
        ).fetchone() is None:
            raise ValueError(f"unknown pod_type_id: {key.pod_type_id}")
        if conn.execute(
            "SELECT 1 FROM _pod_identity_ledger "
            "WHERE approval_date = ? AND daily_slot = ?",
            (key.approval_date.isoformat(), key.daily_slot),
        ).fetchone() is not None:
            raise ValueError("approval date slot has already been issued")

        identity = IssuedIdentity(
            pod_id_seq=_next_pod_id_seq(conn),
            pod_uuid=uuid4(),
            approval_date=key.approval_date,
            daily_slot=key.daily_slot,
            pod_order_no=order_no,
            pod_id=pod_id,
            actor=actor,
        )
        conn.execute(
            "INSERT INTO _pod_identity_ledger ("
            "approval_year, approval_date, daily_slot, pod_order_no, pod_id_seq, "
            "pod_id, pod_uuid, institution_id, pod_type_id, rev_num, actor"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                year,
                key.approval_date.isoformat(),
                key.daily_slot,
                order_no,
                identity.pod_id_seq,
                identity.pod_id,
                str(identity.pod_uuid),
                key.institution_id,
                key.pod_type_id,
                key.rev_num,
                actor,
            ),
        )
    return identity
