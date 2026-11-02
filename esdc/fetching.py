"""Validation and bounded temporary storage for complete API fetch batches."""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from esdc.api_snapshot import ApiProjectResourceRow, JSONScalar
from esdc.pod_registry.models import PreparedPodBatch
from esdc.timeseries_snapshot import (
    TIMESERIES_COLUMNS,
    TIMESERIES_SOURCE_TYPES,
    parse_report_year,
)
from esdc.timeseries_snapshot import (
    TIMESERIES_SOURCE_COLUMNS as TIMESERIES_SOURCE_COLUMNS,
)

FetchPhase = Literal[
    "download", "parse", "validation", "preflight", "publication", "post_commit"
]


class FetchFailure(RuntimeError):  # noqa: N818 - fixed public plan interface
    """A fetch error with enough phase and source context for callers."""

    def __init__(
        self,
        message: str,
        *,
        phase: FetchPhase,
        table: str | None = None,
        report_year: int | None = None,
        sqlite_committed: bool = False,
        duckdb_committed: bool = False,
        recovery_command: str | None = None,
    ) -> None:
        super().__init__(message)
        self.phase = phase
        self.table = table
        self.report_year = report_year
        self.sqlite_committed = sqlite_committed
        self.duckdb_committed = duckdb_committed
        self.recovery_command = recovery_command


@dataclass(frozen=True)
class PreparedTimeseriesSlice:
    report_year: int
    header: tuple[str, ...]
    rows_path: Path
    row_count: int
    origin: Literal["api_json", "api_csv"] = "api_json"


@dataclass(frozen=True)
class PreparedFetch:
    resource_rows: tuple[ApiProjectResourceRow, ...]
    resource_years: tuple[int, ...] | None
    timeseries: tuple[PreparedTimeseriesSlice, ...]
    skipped_resource_years: tuple[int, ...]
    skipped_timeseries_years: tuple[int, ...]
    timeseries_mode: Literal["replace_years", "replace_all"] = "replace_years"
    timeseries_observed_years: tuple[int, ...] = ()
    timeseries_complete_observation: bool = False
    # A later publication must distinguish a scope that was never requested from a
    # requested scope that answered valid-empty, so both stay recorded here.
    pod_batch: PreparedPodBatch | None = None
    resources_observed: bool = False


@dataclass(frozen=True)
class FetchOutcome:
    resource_revision: str | None
    resource_source_sha256: str | None
    resource_row_count: int
    resource_invalid_geometry_count: int | None
    updated_resource_years: tuple[int, ...]
    timeseries_rows_written: int
    updated_timeseries_years: tuple[int, ...]
    skipped_resource_years: tuple[int, ...]
    skipped_timeseries_years: tuple[int, ...]
    timeseries_revision: str | None = None
    timeseries_source_sha256: str | None = None
    timeseries_row_count: int = 0
    # Committed source truth only. A publication is a separate concern, so these
    # defaults stay unset for callers that did not commit a POD source generation.
    pod_source_generation: int | None = None
    sqlite_committed: bool = False


def save_payload_atomically(path: Path, payload: bytes) -> None:
    """Replace one raw payload only after the complete bytes are available."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
            temporary_path = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def validate_timeseries_header(header: Sequence[str]) -> tuple[str, ...]:
    names = tuple(header)
    if len(names) != len(set(names)):
        raise ValueError("project_timeseries contains duplicate headers")
    if "report_year" not in names:
        raise ValueError("project_timeseries is missing required report_year column")
    unknown = sorted(set(names) - TIMESERIES_COLUMNS)
    if unknown:
        raise ValueError(
            "project_timeseries contains unknown headers: " + ", ".join(unknown)
        )
    return names


def spool_timeseries(
    header: Sequence[str],
    rows: Iterable[Sequence[JSONScalar]],
    *,
    report_year: int,
    directory: Path,
    origin: Literal["api_json", "api_csv"] = "api_json",
) -> PreparedTimeseriesSlice | None:
    """Validate and write matching rows as UTF-8 JSON lines."""
    names = validate_timeseries_header(header)
    year_index = names.index("report_year")
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"project_timeseries_{report_year}.jsonl"
    row_count = 0
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="", dir=directory, delete=False
        ) as stream:
            temporary_path = Path(stream.name)
            for index, raw_row in enumerate(rows):
                values = tuple(raw_row)
                if len(values) != len(names):
                    raise ValueError(
                        f"project_timeseries record {index} has wrong width"
                    )
                if any(
                    value is not None and not isinstance(value, (str, int, float, bool))
                    for value in values
                ):
                    raise ValueError(
                        f"project_timeseries record {index} contains a non-scalar value"
                    )
                for column, value in zip(names, values, strict=True):
                    if isinstance(value, float) and not math.isfinite(value):
                        raise ValueError(
                            f"project_timeseries record {index} column {column} "
                            "must be finite"
                        )
                    if (
                        TIMESERIES_SOURCE_TYPES[column] != "TEXT"
                        and isinstance(value, str)
                        and value.strip().casefold().lstrip("+-")
                        in {"nan", "inf", "infinity"}
                    ):
                        raise ValueError(
                            f"project_timeseries record {index} column {column} "
                            "must be finite"
                        )
                try:
                    actual_year = parse_report_year(values[year_index])
                except ValueError as exc:
                    raise ValueError(
                        f"project_timeseries record {index} column report_year: {exc}"
                    ) from exc
                # Validate every field before scoping away rows returned by an
                # endpoint that ignored its report-year query parameter.
                encoded_values = list(values)
                encoded_values[year_index] = actual_year
                if actual_year != report_year:
                    continue
                stream.write(
                    json.dumps(
                        encoded_values, ensure_ascii=False, separators=(",", ":")
                    )
                )
                stream.write("\n")
                row_count += 1
        if row_count == 0:
            temporary_path.unlink()
            return None
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return PreparedTimeseriesSlice(report_year, names, path, row_count, origin)


def iter_spooled_rows(
    prepared: PreparedTimeseriesSlice,
) -> Iterator[tuple[JSONScalar, ...]]:
    """Yield rows from a spool without collecting the complete slice."""
    with prepared.rows_path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise FetchFailure(
                    f"project_timeseries spool line {line_number} is malformed",
                    phase="validation",
                    table="project_timeseries",
                    report_year=prepared.report_year,
                ) from exc
            if not isinstance(row, list) or len(row) != len(prepared.header):
                raise FetchFailure(
                    f"project_timeseries spool line {line_number} has wrong width",
                    phase="validation",
                    table="project_timeseries",
                    report_year=prepared.report_year,
                )
            yield tuple(row)
