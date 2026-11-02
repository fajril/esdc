"""Contract-driven SQLite/XLSX POD import preparation and transactional apply."""

from __future__ import annotations

import gzip
import hashlib
import heapq
import json
import os
import re
import sqlite3
import uuid
import zipfile
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal, cast

from openpyxl import Workbook, load_workbook
from openpyxl.comments import Comment
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

from esdc.api_project_resources import normalize_project_resources
from esdc.api_snapshot import ApiProjectResourceRow, ApiSnapshot, JSONScalar
from esdc.configs import Config
from esdc.pod_registry.api_context import (
    PodImportError,
    refuse_target_shape,
)
from esdc.pod_registry.api_context import (
    column_names as _column_names,
)
from esdc.pod_registry.api_context import (
    object_type as _object_type,
)
from esdc.pod_registry.api_context import (
    quote_identifier as _quote,
)
from esdc.pod_registry.api_context import (
    read_only_connection as _read_only_connection,
)
from esdc.pod_registry.api_context import (
    read_pod_target as _read_pod_target,
)
from esdc.pod_registry.api_context import (
    read_source_identity_ledger_rows as _read_source_identity_ledger,
)
from esdc.pod_registry.api_context import (
    target_protected_state_digest as _target_protected_state_digest,
)
from esdc.pod_registry.contract import (
    PodContract,
    TableContract,
    load_pod_contract,
    validate_pod_batch,
)
from esdc.pod_registry.migrations import (
    MigrationRefusedError,
    _atomic_writer,
    _identity_ledger_reservations,
    _IdentityLedgerReservation,
    _insert_identity_ledger,
    _sqlite_value,
    _validate_existing_v2_registry,
    initialize_registry_v2,
    open_registry_v2_writer,
)
from esdc.pod_registry.models import (
    ImportResult,
    ImportTargetBasis,
    PreparedPodBatch,
    PreparedRow,
    ProjectSnapshotStage,
)
from esdc.pod_registry.normalization import (
    PodNormalizationError,
    normalize_project_id,
    normalize_source_row,
)
from esdc.pod_registry.read_models import refresh_pod_after_source_change
from esdc.pod_registry.replacement import (
    capture_replacement_plan,
    validate_replacement_batch,
    verify_replacement_target,
)
from esdc.pod_registry.store import ensure_project_resources_table
from esdc.timeseries_snapshot import (
    CanonicalTimeseriesRow,
    ensure_timeseries_tables,
    record_observed_empty_timeseries,
    replace_timeseries_in_transaction,
)

POD_METADATA_SHEET = "__pod_metadata"
POD_GUIDE_SHEET = "README"
POD_REFERENCE_TABLES = ("r_institution", "r_pod_type", "r_pod_stage")
POD_BASIS_SHEET = "__pod_basis"
POD_LEDGER_SHEET = "__pod_identity_ledger"
POD_LEDGER_HEADERS = (
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
    "issued_by",
    "issued_at",
)

OUTPUT_TABLES = ("pod_monitoring_api", "pod_forecast_api")
INPUT_TABLES = (
    "m_project",
    "m_pod",
    "pod_project",
    "pod_monitoring",
    "pod_forecast",
)
MASTER_TABLES = ("m_project", "m_pod", "pod_project")
_RESOURCE_SOURCE_EXTRAS = frozenset(
    {"id", "uuid", "project_uuid", "project_id_canonical"}
)
_TIMESERIES_SOURCE_EXTRAS = frozenset({"id", "uuid", "project_id_canonical"})
_GUIDE_SHEETS = frozenset(
    {
        POD_GUIDE_SHEET,
        POD_METADATA_SHEET,
        POD_BASIS_SHEET,
        POD_LEDGER_SHEET,
        *POD_REFERENCE_TABLES,
    }
)


class PodImportProjectionError(RuntimeError):
    """SQLite committed but the DuckDB POD projection did not follow it."""

    source_committed = True


def _same_file(left: Path, right: Path) -> bool:
    if left.resolve(strict=False) == right.resolve(strict=False):
        return True
    if left.exists() and right.exists():
        try:
            return os.path.samefile(left, right)
        except OSError:
            return False
    return False


def _canonical_headers(headers: Sequence[str], table: TableContract) -> list[str]:
    aliases = {column.name: column.name for column in table.columns}
    for column in table.columns:
        aliases.update(dict.fromkeys(column.aliases, column.name))
    return [aliases.get(header, header) for header in headers]


def _validate_batch(batch: PreparedPodBatch, contract: PodContract) -> None:
    issues = validate_pod_batch(batch, contract)
    if issues:
        raise PodImportError(
            [
                f"{issue.table}"
                + (f" row {issue.row}" if issue.row is not None else "")
                + (f" column {issue.column}" if issue.column is not None else "")
                + f" [{issue.code}]: {issue.message}"
                for issue in issues
            ]
        )


def _validate_metadata(metadata: Mapping[str, object], contract: PodContract) -> None:
    if not metadata:
        return
    expected = {
        "contract_id": contract.contract_id,
        "contract_version": str(contract.contract_version),
        "dictionary_version": contract.dictionary_version,
    }
    actual = {key: str(value).strip() for key, value in metadata.items()}
    errors = [
        f"workbook metadata {key}={actual.get(key)!r} is unsupported; "
        f"expected {value!r}"
        for key, value in expected.items()
        if actual.get(key) != value
    ]
    unknown = sorted(set(actual) - set(expected))
    if unknown:
        errors.append(f"workbook metadata has unknown keys: {', '.join(unknown)}")
    missing = sorted(set(expected) - set(actual))
    if missing:
        errors.append(f"workbook metadata is missing keys: {', '.join(missing)}")
    if errors:
        raise PodImportError(errors)


def _workbook_metadata(workbook) -> dict[str, object]:
    if POD_METADATA_SHEET not in workbook.sheetnames:
        return {}
    sheet = workbook[POD_METADATA_SHEET]
    values = list(sheet.iter_rows(values_only=True))
    if not values or tuple(values[0][:2]) != ("key", "value"):
        raise PodImportError(
            [f"{POD_METADATA_SHEET} must start with key,value headers"]
        )
    result: dict[str, object] = {}
    for row in values[1:]:
        if not row or len(row) < 2 or row[0] is None and row[1] is None:
            continue
        if not isinstance(row[0], str) or not row[0].strip():
            raise PodImportError(
                [f"{POD_METADATA_SHEET} contains a blank metadata key"]
            )
        key = row[0].strip()
        if key in result:
            raise PodImportError(
                [f"{POD_METADATA_SHEET} duplicates metadata key {key!r}"]
            )
        result[key] = row[1]
    return result


def _read_xlsx_ledger(workbook) -> Sequence[PreparedRow] | None:
    if POD_LEDGER_SHEET not in workbook.sheetnames:
        return None
    iterator = workbook[POD_LEDGER_SHEET].iter_rows(values_only=True)
    headers = next(iterator, ())
    if len(headers) != len(POD_LEDGER_HEADERS) or set(headers) != set(
        POD_LEDGER_HEADERS
    ):
        raise PodImportError(
            [
                f"{POD_LEDGER_SHEET}: expected exactly the canonical "
                f"identity ledger columns"
            ]
        )
    integer_fields = {
        "approval_year",
        "daily_slot",
        "pod_order_no",
        "pod_id_seq",
        "institution_id",
        "pod_type_id",
        "rev_num",
    }
    rows: list[PreparedRow] = []
    for row_number, values in enumerate(iterator, 2):
        if all(value is None for value in values):
            continue
        if len(values) != len(headers):
            raise PodImportError(
                [f"{POD_LEDGER_SHEET} row {row_number}: unexpected cell count"]
            )
        row = dict(zip(cast(tuple[str, ...], headers), values, strict=True))
        for field in integer_fields:
            value = row[field]
            if isinstance(value, str):
                if not re.fullmatch(r"[+-]?[0-9]+", value, re.ASCII):
                    raise PodImportError(
                        [
                            f"{POD_LEDGER_SHEET} row {row_number}: "
                            f"{field} must be an exact integer"
                        ]
                    )
                value = int(value)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not -(2**63) <= value < 2**63
            ):
                raise PodImportError(
                    [f"{POD_LEDGER_SHEET} row {row_number}: invalid BIGINT {field}"]
                )
            row[field] = value
        rows.append(row)
    return tuple(rows)


def _read_xlsx_tables(
    path: Path,
    contract: PodContract,
    profile: Literal["full", "master-only"],
) -> tuple[
    dict[str, Sequence[str]],
    dict[str, Sequence[PreparedRow]],
    Sequence[PreparedRow] | None,
]:
    if not path.is_file():
        raise PodImportError([f"Excel file not found: {path}"])
    if path.suffix.lower() != ".xlsx":
        raise PodImportError(["Only .xlsx workbooks are supported."])
    workbook = None
    try:
        workbook = load_workbook(path, read_only=True, data_only=True)
        _validate_metadata(_workbook_metadata(workbook), contract)
        selected = INPUT_TABLES if profile == "full" else MASTER_TABLES
        known = set(INPUT_TABLES) | set(OUTPUT_TABLES) | _GUIDE_SHEETS
        unknown_sheets = sorted(set(workbook.sheetnames) - known)
        if unknown_sheets:
            raise PodImportError(
                [f"unknown POD workbook sheet(s): {', '.join(unknown_sheets)}"]
            )
        headers_by_table: dict[str, Sequence[str]] = {}
        rows_by_table: dict[str, Sequence[PreparedRow]] = {}
        for table_name in selected:
            if table_name not in workbook.sheetnames:
                raise PodImportError([f"missing required POD sheet: {table_name}"])
            table = contract.tables[table_name]
            sheet = workbook[table_name]
            iterator = sheet.iter_rows(values_only=True)
            raw_header = next(iterator, None)
            if raw_header is None:
                headers: list[str] = []
            else:
                header_values = list(raw_header)
                while header_values and header_values[-1] is None:
                    header_values.pop()
                headers = [
                    "" if value is None else str(value) for value in header_values
                ]
            prepared_headers = _canonical_headers(headers, table)
            prepared_rows: list[PreparedRow] = []
            for row_number, raw_values in enumerate(iterator, start=2):
                values = cast(list[object], list(raw_values))
                if all(value is None for value in values):
                    continue
                normalized_headers, row = normalize_source_row(
                    headers,
                    values,
                    table,
                    source="xlsx",
                    row_number=row_number,
                )
                prepared_headers = normalized_headers
                prepared_rows.append(row)
            headers_by_table[table_name] = tuple(prepared_headers)
            rows_by_table[table_name] = tuple(prepared_rows)
        return headers_by_table, rows_by_table, _read_xlsx_ledger(workbook)
    except PodImportError:
        raise
    except (
        OSError,
        ValueError,
        TypeError,
        PodNormalizationError,
        zipfile.BadZipFile,
    ) as exc:
        raise PodImportError(
            [f"Failed to prepare Excel workbook {path}: {exc}"]
        ) from exc
    finally:
        if workbook is not None:
            workbook.close()


def _source_contract_metadata(conn: sqlite3.Connection, contract: PodContract) -> None:
    if _object_type(conn, "pod_registry_state") is None:
        return
    if _object_type(conn, "pod_registry_state") != "table":
        raise PodImportError(["SQLite pod_registry_state must be a table"])
    row = conn.execute(
        "SELECT contract_id, contract_version, dictionary_version "
        "FROM pod_registry_state WHERE singleton_id=1"
    ).fetchone()
    if row is None or tuple(row) != (
        contract.contract_id,
        contract.contract_version,
        contract.dictionary_version,
    ):
        raise PodImportError(["SQLite source POD contract metadata is unsupported"])


def _read_sqlite_table(
    conn: sqlite3.Connection,
    table_name: str,
    table: TableContract,
    *,
    required: bool,
) -> tuple[Sequence[str], Sequence[PreparedRow]] | None:
    object_type = _object_type(conn, table_name)
    if object_type is None:
        if required:
            raise PodImportError(
                [f"SQLite source is missing required table {table_name!r}"]
            )
        return None
    if object_type not in {"table", "view"}:
        raise PodImportError(
            [f"SQLite source object {table_name!r} is not a table or view"]
        )
    headers = _column_names(conn, table_name)
    prepared_headers = _canonical_headers(headers, table)
    prepared_rows: list[PreparedRow] = []
    try:
        cursor = conn.execute(f"SELECT * FROM {_quote(table_name)}")
        for row_number, source_row in enumerate(cursor, start=1):
            values = list(source_row)
            _row_headers, row = normalize_source_row(
                headers,
                values,
                table,
                source="sqlite",
                row_number=row_number,
            )
            prepared_rows.append(row)
    except (sqlite3.Error, PodNormalizationError) as exc:
        raise PodImportError(
            [f"Failed to read SQLite source table {table_name!r}: {exc}"]
        ) from exc
    return tuple(prepared_headers), tuple(prepared_rows)


def _read_target_state(
    target_path: Path,
    contract: PodContract,
    *,
    need_preserved_inputs: bool,
) -> tuple[
    ImportTargetBasis,
    dict[str, Sequence[str]],
    dict[str, Sequence[PreparedRow]],
    tuple[int, ...],
]:
    """Read a target snapshot for validation without opening a writer."""
    snapshot = _read_pod_target(
        target_path,
        contract,
        materialize_rows=False,
        need_preserved_inputs=need_preserved_inputs,
    )
    if snapshot.basis.source_generation is None:
        # A fresh or non-POD target has no preserved input context.
        return snapshot.basis, {}, {}, snapshot.linked_pod_ids

    preserved_headers: dict[str, Sequence[str]] = {}
    preserved_rows: dict[str, Sequence[PreparedRow]] = {}
    if need_preserved_inputs:
        for table_name in ("pod_monitoring", "pod_forecast"):
            table = contract.tables[table_name]
            columns = tuple(column.name for column in table.columns)
            preserved_headers[table_name] = columns
            # These rows are preserved byte-for-byte by a master-only load.
            # Validate their final references separately in SQL, so a large
            # annual forecast history never becomes a wide in-memory copy.
            preserved_rows[table_name] = ()

    return snapshot.basis, preserved_headers, preserved_rows, snapshot.linked_pod_ids


def _current_protected_state_digest(
    conn: sqlite3.Connection, *, need_preserved_inputs: bool
) -> str:
    return _target_protected_state_digest(
        conn, need_preserved_inputs=need_preserved_inputs
    )


def _validate_preserved_references(
    target_path: Path, selected_pod_ids: set[int]
) -> None:
    if not target_path.is_file():
        return
    conn = _read_only_connection(target_path)
    try:
        if _object_type(conn, "pod_registry_state") != "table":
            return
        for table_name in ("pod_monitoring", "pod_forecast"):
            bad = [
                int(row[0])
                for row in conn.execute(
                    f"SELECT DISTINCT pod_id_seq FROM {_quote(table_name)}"
                )
                if int(row[0]) not in selected_pod_ids
            ]
            if bad:
                raise PodImportError(
                    [
                        f"Selected master source would orphan preserved {table_name} "
                        "facts for POD sequences: "
                        + ", ".join(map(str, sorted(bad)[:20]))
                    ]
                )
        inconsistent = conn.execute(
            "SELECT mon.monpod_id FROM pod_monitoring AS mon "
            "JOIN m_pod AS pod ON pod.pod_id_seq=mon.pod_id_seq "
            "WHERE mon.pod_id IS NOT pod.pod_id LIMIT 1"
        ).fetchone()
        if inconsistent is not None:
            raise PodImportError(
                [
                    "Operational monitoring facts contain a POD id that disagrees "
                    "with their preserved POD sequence."
                ]
            )
    finally:
        conn.close()


def _known_v04_mockup_without_forecast(conn: sqlite3.Connection) -> bool:
    """Recognize the reviewed v0.4 SQLite mockup with derived forecast only."""
    if _object_type(conn, "pod_registry_state") is not None:
        return False
    if int(conn.execute("PRAGMA user_version").fetchone()[0]) != 1:
        return False
    required_tables = {
        "m_project",
        "m_pod",
        "pod_project",
        "pod_monitoring",
        "_pod_identity_ledger",
    }
    if any(_object_type(conn, name) != "table" for name in required_tables):
        return False
    return _object_type(conn, "pod_forecast_api") == "view"


def _read_pod_source(
    conn: sqlite3.Connection,
    contract: PodContract,
    profile: Literal["full", "master-only"],
) -> tuple[
    dict[str, Sequence[str]],
    dict[str, Sequence[PreparedRow]],
    Sequence[PreparedRow] | None,
]:
    _source_contract_metadata(conn, contract)
    selected = INPUT_TABLES if profile == "full" else MASTER_TABLES
    headers: dict[str, Sequence[str]] = {}
    rows: dict[str, Sequence[PreparedRow]] = {}
    for table_name in selected:
        table = contract.tables[table_name]
        read = _read_sqlite_table(
            conn,
            table_name,
            table,
            required=(
                table_name != "pod_forecast"
                or profile == "master-only"
                or _object_type(conn, table_name) is not None
            ),
        )
        if read is None:
            if table_name != "pod_forecast" or not _known_v04_mockup_without_forecast(
                conn
            ):
                raise PodImportError(
                    [f"SQLite source is missing required table {table_name!r}"]
                )
            headers[table_name] = tuple(column.name for column in table.columns)
            rows[table_name] = ()
        else:
            headers[table_name], rows[table_name] = read
    ledger = _read_source_identity_ledger(conn)
    return headers, rows, ledger


def _serialize_resource_row(row: ApiProjectResourceRow) -> str:
    values = tuple(
        getattr(row, field_name)
        for field_name in ApiProjectResourceRow.__dataclass_fields__
    )
    return json.dumps(
        values, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    )


def _resource_row_from_line(line: str) -> ApiProjectResourceRow:
    values = json.loads(line)
    fields_by_name = tuple(ApiProjectResourceRow.__dataclass_fields__)
    if not isinstance(values, list) or len(values) != len(fields_by_name):
        raise PodImportError(["Staged project-resource row has an invalid shape"])
    return ApiProjectResourceRow(**dict(zip(fields_by_name, values, strict=True)))


def _sorted_resource_hash(path: Path, temporary_directory: Path) -> tuple[int, str]:
    """Hash persisted resource rows with bounded external sorting."""
    runs: list[Path] = []
    count = 0
    with path.open("r", encoding="utf-8") as source:
        while batch := source.readlines(1000 * 4096):
            rows = [line.rstrip("\n") for line in batch]
            count += len(rows)
            rows.sort()
            run_path = temporary_directory / f"resource-hash-{len(runs):04d}.jsonl"
            run_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
            runs.append(run_path)
    digest = hashlib.sha256()
    digest.update(b"[")
    files = [run.open("r", encoding="utf-8") for run in runs]
    try:
        merged = heapq.merge(*(iter(file) for file in files))
        first = True
        for line in merged:
            if not first:
                digest.update(b",")
            digest.update(line.rstrip("\n").encode("utf-8"))
            first = False
    finally:
        for file in files:
            file.close()
    digest.update(b"]")
    for run_path in runs:
        run_path.unlink()
    return count, digest.hexdigest()


def _stage_project_snapshots(
    conn: sqlite3.Connection,
    stage: ProjectSnapshotStage,
    temporary_directory: Path,
) -> None:
    resource_type = _object_type(conn, "project_resources")
    if resource_type is not None:
        if resource_type not in {"table", "view"}:
            raise PodImportError(["SQLite project_resources must be a table or view"])
        headers = _column_names(conn, "project_resources")
        allowed = set(ApiProjectResourceRow.__dataclass_fields__) - {"groovy_report"}
        allowed.update(_RESOURCE_SOURCE_EXTRAS)
        allowed.update(
            {
                "grov_current_situation",
                "grov_executive_summary",
                "grov_key_challenges",
                "grov_solution_proposal",
                "grov_status",
            }
        )
        unknown = sorted(set(headers) - allowed)
        if unknown:
            raise PodImportError(
                ["SQLite project_resources has unknown columns: " + ", ".join(unknown)]
            )
        if "project_id" not in headers:
            raise PodImportError(["SQLite project_resources is missing project_id"])
        stage.resources_path = temporary_directory / "project-resources.jsonl"
        with stage.resources_path.open("w", encoding="utf-8") as output:
            cursor = conn.execute("SELECT * FROM project_resources")
            while batch := cursor.fetchmany(100):
                for source_row in batch:
                    raw = dict(zip(headers, source_row, strict=True))
                    project_id = normalize_project_id(raw.get("project_id"))
                    alias = raw.get("project_id_canonical")
                    if alias is not None and normalize_project_id(alias) != project_id:
                        raise PodImportError(
                            [
                                "project_resources project_id_canonical does not "
                                "match project_id"
                            ]
                        )
                    source_values = {
                        key: value
                        for key, value in raw.items()
                        if key in allowed - _RESOURCE_SOURCE_EXTRAS
                        and key
                        not in {
                            "basin86",
                            "basin86_id",
                            "groovy_report",
                            "project_id_canonical",
                        }
                    }
                    source_values["project_id"] = project_id
                    try:
                        normalized = normalize_project_resources((source_values,))
                    except (TypeError, ValueError) as exc:
                        raise PodImportError(
                            [
                                f"project_resources row "
                                f"{stage.resource_row_count + 1}: {exc}"
                            ]
                        ) from exc
                    row = replace(
                        normalized.rows[0],
                        basin86=cast(JSONScalar, raw.get("basin86")),
                        basin86_id=cast(JSONScalar, raw.get("basin86_id")),
                    )
                    output.write(_serialize_resource_row(row) + "\n")
                    stage.resource_row_count += 1
        _count, stage.resources_sha256 = _sorted_resource_hash(
            stage.resources_path, temporary_directory
        )
    timeseries_type = _object_type(conn, "project_timeseries")
    if timeseries_type is not None:
        if timeseries_type not in {"table", "view"}:
            raise PodImportError(["SQLite project_timeseries must be a table or view"])
        from esdc.timeseries_snapshot import (
            TIMESERIES_SOURCE_COLUMNS,
            normalize_timeseries_row,
        )

        headers = _column_names(conn, "project_timeseries")
        allowed = set(TIMESERIES_SOURCE_COLUMNS) | set(_TIMESERIES_SOURCE_EXTRAS)
        unknown = sorted(set(headers) - allowed)
        if unknown:
            raise PodImportError(
                ["SQLite project_timeseries has unknown columns: " + ", ".join(unknown)]
            )
        required = set(TIMESERIES_SOURCE_COLUMNS) - {"basin128", "is_offshore"}
        missing = sorted(required - set(headers))
        if missing:
            raise PodImportError(
                [
                    "SQLite project_timeseries is missing canonical columns: "
                    + ", ".join(missing)
                ]
            )
        if "project_id" not in headers:
            raise PodImportError(["SQLite project_timeseries is missing project_id"])
        stage.timeseries_path = temporary_directory / "project-timeseries.jsonl.gz"
        with gzip.open(stage.timeseries_path, "wt", encoding="utf-8") as output:
            cursor = conn.execute("SELECT * FROM project_timeseries")
            while batch := cursor.fetchmany(1000):
                for source_row in batch:
                    raw = dict(zip(headers, source_row, strict=True))
                    project_id = normalize_project_id(raw.get("project_id"))
                    alias = raw.get("project_id_canonical")
                    if alias is not None and normalize_project_id(alias) != project_id:
                        raise PodImportError(
                            [
                                "project_timeseries project_id_canonical does not "
                                "match project_id"
                            ]
                        )
                    selected_headers = [
                        name for name in headers if name in TIMESERIES_SOURCE_COLUMNS
                    ]
                    values = [raw[name] for name in selected_headers]
                    values[selected_headers.index("project_id")] = project_id
                    try:
                        normalized = normalize_timeseries_row(
                            selected_headers,
                            values,
                            origin="api_json",
                        )
                    except (TypeError, ValueError) as exc:
                        raise PodImportError(
                            [
                                f"project_timeseries row "
                                f"{stage.timeseries_row_count + 1}: {exc}"
                            ]
                        ) from exc
                    output.write(
                        json.dumps(
                            [
                                normalized.report_year,
                                normalized.row_json,
                                normalized.row_sha256,
                            ],
                            ensure_ascii=False,
                            separators=(",", ":"),
                        )
                        + "\n"
                    )
                    stage.timeseries_row_count += 1


def _merged_for_validation(
    contract: PodContract,
    selected_headers: Mapping[str, Sequence[str]],
    selected_rows: Mapping[str, Sequence[PreparedRow]],
    preserved_headers: Mapping[str, Sequence[str]],
    preserved_rows: Mapping[str, Sequence[PreparedRow]],
    profile: Literal["full", "master-only"],
    source: str,
    source_path: Path,
    target_basis: ImportTargetBasis,
    ledger: Sequence[PreparedRow] | None,
) -> PreparedPodBatch:
    headers = dict(selected_headers)
    rows = dict(selected_rows)
    if profile == "master-only":
        for table_name in ("pod_monitoring", "pod_forecast"):
            headers[table_name] = preserved_headers.get(
                table_name,
                tuple(column.name for column in contract.tables[table_name].columns),
            )
            rows[table_name] = preserved_rows.get(table_name, ())
    return PreparedPodBatch(
        source=source,
        contract_id=contract.contract_id,
        contract_version=contract.contract_version,
        dictionary_version=contract.dictionary_version,
        headers=headers,
        rows=rows,
        identity_ledger=ledger,
        profile=profile,
        source_path=source_path,
        target_basis=target_basis,
    )


def prepare_pod_import(
    path: Path | str,
    source: Literal["sqlite", "xlsx"],
    contract: PodContract | None = None,
    *,
    profile: Literal["full", "master-only"] = "full",
    target_path: Path | str | None = None,
    stage_snapshots: bool = True,
    approved_crosswalk: Mapping[int, int] | None = None,
) -> PreparedPodBatch:
    """Read, normalize, and validate a POD source without opening a writer."""
    source_path = Path(path)
    if source not in {"sqlite", "xlsx"}:
        raise ValueError("source must be 'sqlite' or 'xlsx'")
    if profile not in {"full", "master-only"}:
        raise ValueError("profile must be 'full' or 'master-only'")
    if contract is None:
        contract = load_pod_contract()
    if source_path.suffix.lower() != (".sqlite" if source == "sqlite" else ".xlsx"):
        raise PodImportError(
            [f"Source extension does not match {source}: {source_path}"]
        )
    from esdc.pod_registry.store import get_esdc_sqlite_path

    operational_path = (
        Path(target_path) if target_path is not None else get_esdc_sqlite_path()
    )
    duck_path = Config.get_db_file()
    if source == "sqlite" and (
        _same_file(source_path, operational_path) or _same_file(source_path, duck_path)
    ):
        raise PodImportError(
            [
                "SQLite source must not be the operational target or "
                "DuckDB projection file"
            ]
        )

    replacement_plan = capture_replacement_plan(
        operational_path, approved_crosswalk, profile=profile
    )
    if replacement_plan is None:
        basis, preserved_headers, preserved_rows, linked_pods = _read_target_state(
            operational_path, contract, need_preserved_inputs=(profile == "master-only")
        )
    else:
        basis = ImportTargetBasis(
            True,
            None,
            replacement_plan.target_sha256,
            operational_path.resolve(strict=False),
        )
        preserved_headers, preserved_rows, linked_pods = {}, {}, ()
    stage: ProjectSnapshotStage | None = None
    temporary_directory: TemporaryDirectory[str] | None = None
    if source == "sqlite" and stage_snapshots and profile == "full":
        temporary_directory = TemporaryDirectory(
            prefix="esdc-pod-import-", dir="/private/tmp"
        )
        stage = ProjectSnapshotStage(_temporary_directory=temporary_directory)
    if source == "sqlite":
        if not source_path.is_file():
            if stage is not None:
                stage.cleanup()
            raise PodImportError([f"SQLite file not found: {source_path}"])
        try:
            source_conn = _read_only_connection(source_path)
            try:
                headers, rows, ledger = _read_pod_source(source_conn, contract, profile)
                if stage is not None:
                    assert temporary_directory is not None
                    _stage_project_snapshots(
                        source_conn, stage, Path(temporary_directory.name)
                    )
            finally:
                source_conn.close()
        except sqlite3.Error as exc:
            if stage is not None:
                stage.cleanup()
            raise PodImportError(
                [f"Could not read SQLite source {source_path}: {exc}"]
            ) from exc
        except BaseException:
            if stage is not None:
                stage.cleanup()
            raise
    else:
        headers, rows, ledger = _read_xlsx_tables(source_path, contract, profile)

    try:
        selected_pod_ids: set[int] = set()
        for row in rows.get("m_pod", ()):
            pod_id_seq = row.get("pod_id_seq")
            if isinstance(pod_id_seq, int) and not isinstance(pod_id_seq, bool):
                selected_pod_ids.add(pod_id_seq)
        missing_document_targets = sorted(set(linked_pods) - selected_pod_ids)
        if missing_document_targets:
            raise PodImportError(
                [
                    "Selected master source omits POD identities referenced by "
                    "existing "
                    "document links: "
                    + ", ".join(map(str, missing_document_targets[:20]))
                    + ". Use the approved explicit crosswalk migration path "
                    "to preserve existing document links before importing."
                ]
            )

        batch = _merged_for_validation(
            contract,
            headers,
            rows,
            preserved_headers,
            preserved_rows,
            profile,
            source,
            source_path,
            basis,
            ledger,
        )
        _validate_batch(batch, contract)
        try:
            _identity_ledger_reservations(batch)
        except MigrationRefusedError as exc:
            raise PodImportError([str(exc)]) from exc
        batch = replace(batch, replacement_plan=replacement_plan)
        validate_replacement_batch(batch)
        if profile == "master-only":
            _validate_preserved_references(operational_path, selected_pod_ids)
    except BaseException:
        if stage is not None:
            stage.cleanup()
        raise

    return replace(batch, staged_snapshots=stage)


def _iter_staged_timeseries(path: Path) -> Iterator[CanonicalTimeseriesRow]:
    with gzip.open(path, "rt", encoding="utf-8") as source:
        for line in source:
            report_year, row_json, row_sha256 = json.loads(line)
            yield CanonicalTimeseriesRow(report_year, row_json, row_sha256, "api_json")


def _ensure_snapshot_metadata_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """CREATE TABLE IF NOT EXISTS api_snapshot (
            source_table TEXT PRIMARY KEY,
            revision TEXT NOT NULL,
            row_count INTEGER NOT NULL,
            source_sha256 TEXT NOT NULL,
            fetched_at TEXT NOT NULL,
            publication_state TEXT NOT NULL CHECK (
                publication_state IN ('committed', 'published', 'analytics_stale')
            ),
            invalid_geometry_count INTEGER NOT NULL DEFAULT 0
                CHECK (invalid_geometry_count >= 0)
        )"""
    )


def _write_resource_stage(
    conn: sqlite3.Connection, stage: ProjectSnapshotStage
) -> None:
    if stage.resources_path is None:
        return
    ensure_project_resources_table(conn)
    _ensure_snapshot_metadata_table(conn)
    from esdc.api_snapshot import _ROW_COLUMNS, _ROW_INSERT_SQL

    conn.execute("DELETE FROM api_project_resources")
    sql = _ROW_INSERT_SQL
    values: list[tuple[object, ...]] = []
    with stage.resources_path.open("r", encoding="utf-8") as source:
        for line in source:
            row = _resource_row_from_line(line)
            values.append(tuple(getattr(row, column) for column in _ROW_COLUMNS))
            if len(values) == 500:
                conn.executemany(sql, values)
                values.clear()
    if values:
        conn.executemany(sql, values)
    if stage.resources_sha256 is None:
        raise PodImportError(["Project-resource stage is missing its validated hash"])
    snapshot = ApiSnapshot(
        "project_resources",
        str(uuid.uuid4()),
        stage.resource_row_count,
        stage.resources_sha256,
        datetime.now(timezone.utc).isoformat(),
        "committed",
        0,
    )
    conn.execute(
        """INSERT INTO api_snapshot
           (source_table, revision, row_count, source_sha256, fetched_at,
            publication_state, invalid_geometry_count)
           VALUES (?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(source_table) DO UPDATE SET revision=excluded.revision,
             row_count=excluded.row_count, source_sha256=excluded.source_sha256,
             fetched_at=excluded.fetched_at,
             publication_state=excluded.publication_state,
             invalid_geometry_count=excluded.invalid_geometry_count""",
        (
            snapshot.source_table,
            snapshot.revision,
            snapshot.row_count,
            snapshot.source_sha256,
            snapshot.fetched_at,
            snapshot.publication_state,
            snapshot.invalid_geometry_count,
        ),
    )


def _write_timeseries_stage(
    conn: sqlite3.Connection, stage: ProjectSnapshotStage
) -> None:
    if stage.timeseries_path is None:
        return
    _ensure_snapshot_metadata_table(conn)
    ensure_timeseries_tables(conn)
    if stage.timeseries_row_count == 0:
        conn.execute("DELETE FROM api_project_timeseries")
        snapshot = record_observed_empty_timeseries(
            conn,
            revision=str(uuid.uuid4()),
            fetched_at=datetime.now(timezone.utc).isoformat(),
            origin="api_json",
            refresh_read_models=False,
        )
    else:
        snapshot = replace_timeseries_in_transaction(
            conn,
            _iter_staged_timeseries(stage.timeseries_path),
            years=None,
            revision=str(uuid.uuid4()),
            fetched_at=datetime.now(timezone.utc).isoformat(),
            refresh_read_models=False,
        )
    from esdc.timeseries_snapshot import record_timeseries_provenance

    provenance = conn.execute(
        "SELECT origin, details_json FROM api_snapshot_provenance "
        "WHERE source_table='project_timeseries' AND revision=?",
        (snapshot.revision,),
    ).fetchone()
    if provenance is None:
        raise PodImportError(
            ["Staged timeseries snapshot is missing canonical provenance"]
        )
    details = json.loads(provenance[1])
    details.update(import_source="sqlite", source_path_name="project_timeseries")
    record_timeseries_provenance(
        conn,
        revision=snapshot.revision,
        origin=provenance[0],
        details=details,
    )


def _insert_rows(
    conn: sqlite3.Connection,
    table_name: str,
    rows: Sequence[PreparedRow],
    contract: PodContract,
    *,
    replace_supersession: bool = False,
) -> None:
    table = contract.tables[table_name]
    columns = tuple(column.name for column in table.columns)
    keys = table.primary_key
    identity_columns = {
        "pod_id_seq",
        "pod_uuid",
        "approval_date",
        "institution_id",
        "pod_type_id",
        "rev_num",
        "pod_order_no",
        "pod_id",
    }
    for row in rows:
        bind_values = dict(row)
        if replace_supersession:
            bind_values["superseded_by"] = None
            bind_values["is_latest"] = True
        insert_sql = (
            f"INSERT INTO {_quote(table_name)} "
            f"({', '.join(_quote(column) for column in columns)}) "
            f"VALUES ({', '.join('?' for _ in columns)}) "
            f"ON CONFLICT ({', '.join(_quote(key) for key in keys)}) DO UPDATE SET "
            + ", ".join(
                f"{_quote(column)}=excluded.{_quote(column)}"
                for column in columns
                if column not in keys and column not in identity_columns
            )
        )
        if table_name == "pod_project":
            # Source surrogate ids are stable identities. If a target has the same
            # business pair under another surrogate, reject rather than re-key it.
            conflicting = conn.execute(
                "SELECT id FROM pod_project WHERE pod_id_seq=? AND project_id=?",
                (row.get("pod_id_seq"), row.get("project_id")),
            ).fetchone()
            row_id = row.get("id")
            if isinstance(row_id, bool) or not isinstance(row_id, int):
                raise PodImportError(["pod_project id must be an integer"])
            if conflicting is not None and int(conflicting[0]) != row_id:
                raise PodImportError(
                    [
                        "pod_project source id conflicts with an existing business "
                        "mapping; explicit identity reconciliation is required"
                    ]
                )
        bind_values = {key: _sqlite_value(value) for key, value in bind_values.items()}
        conn.execute(insert_sql, tuple(bind_values.get(column) for column in columns))

    if table_name == "m_pod" and replace_supersession:
        for row in rows:
            successor = row.get("superseded_by")
            if successor is not None:
                conn.execute(
                    "UPDATE m_pod SET superseded_by=?, is_latest=0 WHERE pod_id_seq=?",
                    (_sqlite_value(successor), _sqlite_value(row.get("pod_id_seq"))),
                )


def _delete_obsolete_rows(
    conn: sqlite3.Connection,
    table_name: str,
    rows: Sequence[PreparedRow],
    contract: PodContract,
) -> None:
    table = contract.tables[table_name]
    key = table.primary_key
    selected = {tuple(row.get(name) for name in key) for row in rows}
    existing = conn.execute(
        f"SELECT {', '.join(_quote(name) for name in key)} FROM {_quote(table_name)}"
    )
    stale = [tuple(row) for row in existing if tuple(row) not in selected]
    if not stale:
        return
    where = " AND ".join(f"{_quote(name)}=?" for name in key)
    conn.executemany(f"DELETE FROM {_quote(table_name)} WHERE {where}", stale)


_LEDGER_BUSINESS_FIELDS = (
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
)


def _reconcile_ledger_reservations(
    conn: sqlite3.Connection,
    reservations: Sequence[_IdentityLedgerReservation],
) -> tuple[_IdentityLedgerReservation, ...]:
    """Keep only the identities that still need the strict migration insert.

    A committed permanent identity is append-only and its actor and issue time are
    its own provenance. When a prepared reservation names an already committed
    business identity, the committed audit is retained exactly and the reservation is
    not reinserted. A committed identity whose business fields differ is a conflict,
    and only unknown identities reach the strict insert, which keeps enforcing the
    cross-sequence uniqueness guards.
    """
    pending: list[_IdentityLedgerReservation] = []
    for reservation in reservations:
        committed = conn.execute(
            "SELECT approval_year, approval_date, daily_slot, pod_order_no, "
            "pod_id_seq, pod_id, pod_uuid, institution_id, pod_type_id, rev_num "
            "FROM _pod_identity_ledger WHERE pod_id_seq = ?",
            (reservation.pod_id_seq,),
        ).fetchone()
        if committed is None:
            pending.append(reservation)
            continue
        supplied = tuple(
            getattr(reservation, field) for field in _LEDGER_BUSINESS_FIELDS
        )
        if tuple(committed) == supplied:
            continue
        conflicts = [
            field
            for field, committed_value, supplied_value in zip(
                _LEDGER_BUSINESS_FIELDS,
                committed,
                supplied,
                strict=True,
            )
            if committed_value != supplied_value
        ]
        raise PodImportError(
            [
                "identity ledger reservation conflicts with the committed permanent "
                f"identity for pod_id_seq {reservation.pod_id_seq} on "
                f"{', '.join(conflicts)}"
            ]
        )
    return tuple(pending)


def apply_staged_project_snapshots(
    conn: sqlite3.Connection, stage: ProjectSnapshotStage | None
) -> None:
    """Apply validated raw stages inside the caller's source transaction."""
    if not conn.in_transaction:
        raise RuntimeError("Project snapshots require the caller's source transaction")
    if stage is not None:
        _write_resource_stage(conn, stage)
        _write_timeseries_stage(conn, stage)


def apply_pod_batch_in_transaction(
    conn: sqlite3.Connection,
    batch: PreparedPodBatch,
    actor: str,
    *,
    contract: PodContract | None = None,
    refresh: bool = True,
) -> ImportResult:
    """Compose source and staged-snapshot writes in the caller transaction.

    Pass ``refresh=False`` when a caller has additional POD/snapshot source writes;
    that caller must invoke ``refresh_pod_after_source_change`` exactly once before
    committing. This function never commits or publishes DuckDB.
    """
    if not conn.in_transaction:
        raise RuntimeError("POD batch apply requires an active caller transaction")
    if not actor.strip():
        raise ValueError("POD import actor cannot be empty")
    contract = contract or load_pod_contract()
    _validate_batch(batch, contract)
    if batch.profile == "master-only" and batch.staged_snapshots is not None:
        raise PodImportError(["master-only profile cannot apply project snapshots"])
    if batch.target_basis is not None:
        current = conn.execute(
            "SELECT source_generation FROM pod_registry_state WHERE singleton_id=1"
        ).fetchone()
        actual_generation = None if current is None else int(current[0])
        if batch.target_basis.source_generation is None:
            # Preparation captured an absent target, so the writer lock must still
            # observe a freshly initialized registry. A target created and advanced
            # by another writer after preparation is no longer the prepared target.
            if actual_generation is not None and actual_generation > 0:
                raise PodImportError(
                    [
                        "Operational POD registry changed after import preparation; "
                        "prepare again"
                    ]
                )
        elif actual_generation != batch.target_basis.source_generation:
            raise PodImportError(
                [
                    "Operational POD registry changed after import preparation; "
                    "prepare again"
                ]
            )
        if batch.target_basis.source_generation is not None:
            actual_digest = _current_protected_state_digest(
                conn, need_preserved_inputs=(batch.profile == "master-only")
            )
            if actual_digest != batch.target_basis.protected_state_sha256:
                raise PodImportError(
                    [
                        "Protected operational POD facts changed after import "
                        "preparation; prepare again"
                    ]
                )
    reservations = _identity_ledger_reservations(batch)
    _insert_identity_ledger(conn, _reconcile_ledger_reservations(conn, reservations))

    # Full profile owns all five contract inputs. The master-only profile merges
    # target monitoring/forecast rows for validation and leaves those two tables
    # untouched while replacing the three global master/mapping inputs.
    if batch.profile == "full":
        for table_name in ("pod_monitoring", "pod_forecast"):
            conn.execute(f"DELETE FROM {_quote(table_name)}")

    _insert_rows(conn, "m_project", batch.rows["m_project"], contract)

    # Clear supersession edges first so replacements can remove stale successors
    # without violating the immutable identity ledger or self-reference guard.
    conn.execute(
        "UPDATE m_pod SET superseded_by=NULL, is_latest=1 "
        "WHERE superseded_by IS NOT NULL"
    )
    _insert_rows(
        conn,
        "m_pod",
        batch.rows["m_pod"],
        contract,
        replace_supersession=True,
    )
    _insert_rows(conn, "pod_project", batch.rows["pod_project"], contract)
    _delete_obsolete_rows(conn, "pod_project", batch.rows["pod_project"], contract)

    if batch.profile == "full":
        for table_name in ("pod_monitoring", "pod_forecast"):
            _insert_rows(conn, table_name, batch.rows[table_name], contract)

    for table_name in ("m_pod", "m_project"):
        _delete_obsolete_rows(conn, table_name, batch.rows[table_name], contract)

    stage = batch.staged_snapshots
    apply_staged_project_snapshots(conn, stage)

    violations = conn.execute("PRAGMA foreign_key_check").fetchall()
    if violations:
        raise PodImportError(
            [f"POD import would violate foreign keys: {violations[:5]!r}"]
        )
    if refresh:
        refresh_pod_after_source_change(conn, actor=actor)
    state = conn.execute(
        "SELECT source_generation FROM pod_registry_state WHERE singleton_id=1"
    ).fetchone()
    if state is None:
        raise PodImportError(
            ["Operational POD registry state row is missing after apply"]
        )
    return ImportResult(
        table_counts={
            table_name: len(batch.rows.get(table_name, ()))
            for table_name in INPUT_TABLES
            if batch.profile == "full" or table_name in MASTER_TABLES
        },
        source_generation=int(state[0]),
        source_committed=False,
        resource_row_count=0 if stage is None else stage.resource_row_count,
        timeseries_row_count=0 if stage is None else stage.timeseries_row_count,
    )


def apply_pod_batch(
    conn: sqlite3.Connection,
    batch: PreparedPodBatch,
    actor: str,
    *,
    contract: PodContract | None = None,
    refresh: bool = True,
) -> ImportResult:
    """Atomically apply a prepared batch, respecting caller-owned transactions."""
    owns_transaction = not conn.in_transaction
    with _atomic_writer(conn):
        result = apply_pod_batch_in_transaction(
            conn, batch, actor, contract=contract, refresh=refresh
        )
    return replace(result, source_committed=owns_transaction)


def load_prepared_pod_batch(
    batch: PreparedPodBatch,
    *,
    sqlite_path: Path | str,
    duck_path: Path | str,
    actor: str = "FA:pod-registry-import",
    contract: PodContract | None = None,
    publish: bool = True,
) -> tuple[ImportResult, object | None]:
    """Open or initialize a v2 source only after preparation, then publish once."""
    from esdc.source_publication import publish_committed_pod_sources
    from esdc.sqlite_runtime import configure_writer, require_wal_runtime

    # The runtime safety gate precedes every target side effect: no directory is
    # created, no writer connection is opened, and no journal-mode setup runs on an
    # unsupported SQLite library.
    require_wal_runtime()
    if not actor.strip():
        raise ValueError("source-change actor cannot be empty")
    source_path = Path(sqlite_path)
    target_path = Path(duck_path)
    if (
        batch.target_basis is not None
        and batch.target_basis.target_path is not None
        and batch.target_basis.target_path != source_path.resolve(strict=False)
    ):
        raise PodImportError(
            [
                "Prepared batch was validated against a different operational SQLite "
                "path; prepare again for the selected target."
            ]
        )
    if _same_file(batch.source_path or Path(), source_path) or _same_file(
        source_path, target_path
    ):
        raise PodImportError(
            ["POD SQLite target must be distinct from source and DuckDB paths"]
        )
    if (
        batch.target_basis is not None
        and source_path.exists() != batch.target_basis.existed
    ):
        raise PodImportError(
            [
                "Operational SQLite target existence changed after import preparation; "
                "prepare again."
            ]
        )
    # A prepared no-POD target is one whose basis recorded no POD registry. The
    # derivation must precede both writer-selection branches, and it only compares that
    # preparation; a batch prepared against a current registry keeps its prior behavior.
    prepared_no_pod_target = (
        batch.target_basis is not None and batch.target_basis.source_generation is None
    )
    replacement_plan = batch.replacement_plan
    if replacement_plan is not None:
        _validate_batch(batch, contract or load_pod_contract())
        _identity_ledger_reservations(batch)
        validate_replacement_batch(batch)
        try:
            probe = _read_only_connection(source_path)
            try:
                verify_replacement_target(probe, replacement_plan)
            finally:
                probe.close()
            conn = sqlite3.connect(
                source_path.resolve().as_uri() + "?mode=rw", uri=True
            )
            try:
                configure_writer(conn)
            except BaseException:
                conn.close()
                raise
        except BaseException:
            batch.cleanup_staging()
            raise
    elif (
        batch.target_basis
        and batch.target_basis.source_generation is not None
        and source_path.exists()
    ):
        conn = open_registry_v2_writer(source_path)
    else:
        if source_path.exists():
            probe = _read_only_connection(source_path)
            try:
                if _object_type(probe, "pod_registry_state") is not None:
                    conn = open_registry_v2_writer(source_path)
                else:
                    # One shared shape policy decides this: application, document,
                    # raw-source and unrelated objects alone are an additive target,
                    # while a legacy or partial POD footprint, including a reserved
                    # POD-shaped view, still refuses before any writer or DDL runs.
                    refuse_target_shape(probe)
                    conn = None
            finally:
                probe.close()
        else:
            conn = None
        if conn is None:
            source_path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(source_path)
            configure_writer(conn)
    initialized = False
    try:
        with _atomic_writer(conn):
            # Reclassify under the write boundary, before initialization. A prepared
            # no-POD target that a concurrent writer has already turned into a current
            # registry is a changed target, even at generation 0; the later apply guard
            # alone would only refuse an advanced registry.
            if replacement_plan is not None:
                from esdc.pod_registry.migrations import migrate_registry

                verify_replacement_target(conn, replacement_plan)
                migration = migrate_registry(
                    conn,
                    prepared_batch=batch,
                    approved_crosswalk=dict(replacement_plan.approved_crosswalk),
                )
                if migration.source_generation is None:
                    raise RuntimeError("POD replacement committed no source generation")
                stage = batch.staged_snapshots
                result = ImportResult(
                    table_counts={name: len(batch.rows[name]) for name in INPUT_TABLES},
                    source_generation=migration.source_generation,
                    source_committed=False,
                    resource_row_count=0 if stage is None else stage.resource_row_count,
                    timeseries_row_count=0
                    if stage is None
                    else stage.timeseries_row_count,
                )
            else:
                report = refuse_target_shape(conn)
                if prepared_no_pod_target and report.status != "fresh":
                    raise PodImportError(
                        [
                            "Operational POD registry changed after import "
                            "preparation; prepare again"
                        ]
                    )
                if report.status == "fresh":
                    initialize_registry_v2(conn)
                    initialized = True
                else:
                    _validate_existing_v2_registry(conn)
                result = apply_pod_batch_in_transaction(
                    conn,
                    batch,
                    actor,
                    contract=contract,
                    refresh=True,
                )
        result = replace(result, source_committed=True)
    except BaseException:
        if initialized and conn.in_transaction:
            conn.rollback()
        raise
    finally:
        conn.close()
        batch.cleanup_staging()

    publication = None
    if publish:
        try:
            report = publish_committed_pod_sources(
                source_path, target_path, expected_generation=result.source_generation
            )
        except Exception as exc:
            raise PodImportProjectionError(
                f"SQLite source committed, but DuckDB publication raised: {exc}"
            ) from exc
        projection_error = getattr(report, "projection_error", None)
        if projection_error is not None:
            raise PodImportProjectionError(
                "SQLite source committed, but DuckDB publication failed: "
                + str(projection_error)
            )
        publication = report
    return result, publication


def generate_pod_import_template(
    output_path: Path | str,
    *,
    overwrite: bool = False,
    contract: PodContract | None = None,
) -> Path:
    """Write the current 5-input/2-output POD workbook template."""
    destination = Path(output_path)
    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"Output schema already exists: {destination}. "
            "Use --overwrite to replace it."
        )
    contract = contract or load_pod_contract()
    workbook = Workbook()
    active_sheet = workbook.active
    if active_sheet is not None:
        workbook.remove(active_sheet)
    for table_name in (*INPUT_TABLES, *OUTPUT_TABLES):
        table = contract.tables[table_name]
        sheet = workbook.create_sheet(table_name)
        sheet.freeze_panes = "A2"
        sheet.append([column.name for column in table.columns])
        for cell, column in zip(sheet[1], table.columns, strict=True):
            cell.font = Font(bold=True, color="FFFFFF")
            cell.fill = PatternFill("solid", fgColor="174A6E")
            cell.alignment = Alignment(wrap_text=True)
            cell.comment = (
                None
                if column.description is None
                else Comment(column.description, "ESDC")
            )
        sheet.auto_filter.ref = f"A1:{get_column_letter(len(table.columns))}1"
        if table_name in OUTPUT_TABLES:
            sheet.protection.sheet = True
            sheet.protection.selectLockedCells = True
            sheet.protection.selectUnlockedCells = False
    metadata = workbook.create_sheet(POD_METADATA_SHEET)
    metadata.append(["key", "value"])
    metadata.append(["contract_id", contract.contract_id])
    metadata.append(["contract_version", contract.contract_version])
    metadata.append(["dictionary_version", contract.dictionary_version])
    guide = workbook.create_sheet(POD_GUIDE_SHEET, 0)
    guide.append(["ESDC POD Registry Workbook", "Guidance"])
    guide.append(
        ["Source of truth", "The five input sheets are the only editable POD inputs."]
    )
    guide.append(["Input sheets", ", ".join(INPUT_TABLES)])
    guide.append(
        ["Output sheets", ", ".join(OUTPUT_TABLES) + " are generated and read-only."]
    )
    guide.append(
        [
            "Decimals",
            "Enter exact decimal strings as text when Excel numeric precision "
            "is insufficient; values are never rounded to fit the contract.",
        ]
    )
    guide.append(
        [
            "Dates",
            "Use ISO dates and timestamps. Date-only fields must not contain "
            "a time component.",
        ]
    )
    guide.append(
        [
            "Project IDs",
            "Use canonical P-XXXXXXX-XX IDs; compact PXXXXXXXXX IDs are "
            "normalized on import.",
        ]
    )
    guide.append(
        [
            "Metadata",
            f"Keep {POD_METADATA_SHEET} unchanged so import can validate the "
            "supported contract version.",
        ]
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(destination)
    return destination
