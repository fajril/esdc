"""Shared neutral models for prepared POD batches and validation issues."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal, TypeAlias
from uuid import UUID

PreparedScalar: TypeAlias = (
    str | int | float | bool | Decimal | date | datetime | UUID | None
)
PreparedRow: TypeAlias = Mapping[str, PreparedScalar]


@dataclass(slots=True)
class ProjectSnapshotStage:
    """Owned, bounded staging files for optional large project snapshots."""

    resources_path: Path | None = None
    timeseries_path: Path | None = None
    resource_row_count: int = 0
    timeseries_row_count: int = 0
    resources_sha256: str | None = None
    _temporary_directory: TemporaryDirectory[str] | None = field(
        default=None, repr=False
    )

    def cleanup(self) -> None:
        if self._temporary_directory is not None:
            self._temporary_directory.cleanup()
            self._temporary_directory = None


@dataclass(frozen=True, slots=True)
class ImportTargetBasis:
    """Read-only target facts that a prepared import must still be based on."""

    existed: bool
    source_generation: int | None
    protected_state_sha256: str
    target_path: Path | None = None


@dataclass(frozen=True, slots=True)
class PodReplacementPlan:
    """Immutable, one-time approval for replacing a recognized old POD dataset."""

    target_sha256: str
    approved_crosswalk: tuple[tuple[int, int], ...]


@dataclass(frozen=True, slots=True)
class PreparedPodBatch:
    """Normalized source tables and their declared headers before any write."""

    source: str
    contract_id: str
    contract_version: int
    dictionary_version: str
    headers: Mapping[str, Sequence[str]]
    rows: Mapping[str, Sequence[PreparedRow]]
    # Source ledger audit rows are optional because the YAML contract excludes them.
    identity_ledger: Sequence[PreparedRow] | None = None
    profile: Literal["full", "master-only"] = "full"
    source_path: Path | None = None
    target_basis: ImportTargetBasis | None = None
    staged_snapshots: ProjectSnapshotStage | None = None
    replacement_plan: PodReplacementPlan | None = None

    def cleanup_staging(self) -> None:
        if self.staged_snapshots is not None:
            self.staged_snapshots.cleanup()


@dataclass(frozen=True, slots=True)
class ImportResult:
    """Committed POD source counts, before the separate DuckDB publication."""

    table_counts: Mapping[str, int]
    source_generation: int
    source_committed: bool
    resource_row_count: int = 0
    timeseries_row_count: int = 0


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    """One actionable contract or row validation issue."""

    source: str
    table: str
    row: int | None
    column: str | None
    code: str
    message: str
