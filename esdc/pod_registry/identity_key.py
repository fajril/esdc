"""Pure POD approval-key inputs and the canonical business-identity formula.

This module has no package imports so the shared batch validator can use the
approved identity rule without importing the writer-facing issuance module, which
imports migrations and would close an import cycle.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

DAILY_SLOTS_PER_DAY = 20


@dataclass(frozen=True, slots=True)
class ApprovalKey:
    """Inputs that determine one permanent date-scoped approval identity."""

    approval_date: date
    daily_slot: int
    institution_id: int
    pod_type_id: int
    rev_num: int


def validate_key(key: ApprovalKey) -> tuple[int, int, str]:
    """Return the canonical approval year, order number, and POD ID for a key.

    Raises ``TypeError`` or ``ValueError`` for inputs outside the approved
    identity ranges; callers validating untrusted rows report those as issues.
    """
    if not isinstance(key.approval_date, date) or isinstance(
        key.approval_date, datetime
    ):
        raise TypeError("approval_date must be a datetime.date")
    integer_fields = (
        (
            "daily_slot",
            key.daily_slot,
            1,
            DAILY_SLOTS_PER_DAY,
            f"daily_slot must be between 1 and {DAILY_SLOTS_PER_DAY}",
        ),
        (
            "institution_id",
            key.institution_id,
            1,
            9,
            "institution_id must be a one-digit code between 1 and 9",
        ),
        (
            "pod_type_id",
            key.pod_type_id,
            1,
            9,
            "pod_type_id must be a one-digit code between 1 and 9",
        ),
        (
            "rev_num",
            key.rev_num,
            0,
            9,
            "rev_num must be a one-digit revision from 0 to 9",
        ),
    )
    for name, value, minimum, maximum, message in integer_fields:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
        if not minimum <= value <= maximum:
            raise ValueError(message)
    order_no = (
        (key.approval_date.timetuple().tm_yday - 1) * DAILY_SLOTS_PER_DAY
        + key.daily_slot
    )
    pod_id = (
        f"PL-{key.approval_date.year:04d}-{order_no:04d}-"
        f"{key.institution_id}-{key.pod_type_id}-{key.rev_num}"
    )
    return key.approval_date.year, order_no, pod_id
