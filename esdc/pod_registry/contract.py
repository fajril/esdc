"""Static POD contract loader and source-batch validation."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, time
from decimal import Decimal, InvalidOperation
from importlib.resources import files
from numbers import Integral, Real
from typing import Any
from uuid import UUID

import yaml

from esdc.pod_registry.identity_key import (
    DAILY_SLOTS_PER_DAY,
    ApprovalKey,
    validate_key,
)
from esdc.pod_registry.models import PreparedPodBatch, PreparedRow, ValidationIssue

_EXPECTED_COLUMN_COUNTS = {
    "m_project": 7,
    "m_pod": 82,
    "pod_project": 10,
    "pod_monitoring": 92,
    "pod_forecast": 13,
    "pod_monitoring_api": 176,
    "pod_forecast_api": 24,
}
_EXPECTED_CONTRACT = ("esdc.pod", 1, "0.4.0")
_INTEGER_LIMITS = {
    "BIGINT": (-(2**63), 2**63 - 1),
    "INTEGER": (-(2**63), 2**63 - 1),
    "INT": (-(2**63), 2**63 - 1),
    "SMALLINT": (-(2**15), 2**15 - 1),
}
_BETWEEN_RE = re.compile(
    r"\bBETWEEN\s+(-?\d+(?:\.\d+)?)\s+AND\s+(-?\d+(?:\.\d+)?)", re.IGNORECASE
)
_COMPARE_RE = re.compile(
    r"\b([A-Za-z_][A-Za-z0-9_]*)\s*(>=|>|<=|<|=|<>)\s*"
    r"([A-Za-z_][A-Za-z0-9_]*|-?\d+(?:\.\d+)?)"
)


@dataclass(frozen=True, slots=True)
class ColumnContract:
    name: str
    formal_name: str
    logical_type: str
    data_type: str
    nullable: bool
    declared_nullable: bool | None
    ownership: str
    unit: str | None
    precision: int | None
    scale: int | None
    max_length: int | None
    constraint: str | None
    description: str | None
    group: str | None
    source: str | None
    aliases: tuple[str, ...]
    uuid_version: int | None = None
    pattern: str | None = None
    minimum: int | float | None = None
    maximum: int | float | None = None


@dataclass(frozen=True, slots=True)
class ReferenceContract:
    column: str
    target_table: str
    target_column: str
    nullable: bool
    allowed_values: tuple[object, ...]
    consistent_with: str | None = None


@dataclass(frozen=True, slots=True)
class ConditionalUnique:
    columns: tuple[str, ...]
    when_column: str
    equals: object


@dataclass(frozen=True, slots=True)
class TableContract:
    name: str
    role: str
    required: bool
    source_names: tuple[str, ...]
    primary_key: tuple[str, ...]
    unique_keys: tuple[tuple[str, ...], ...]
    references: tuple[ReferenceContract, ...]
    conditional_unique: tuple[ConditionalUnique, ...]
    columns: tuple[ColumnContract, ...]


@dataclass(frozen=True, slots=True)
class PodContract:
    contract_id: str
    contract_version: int
    dictionary_version: str
    source_of_truth: str
    notes: tuple[str, ...]
    reference_values: Mapping[str, tuple[object, ...]]
    exceptions: tuple[Mapping[str, Any], ...]
    tables: Mapping[str, TableContract]


def _optional_str(value: object) -> str | None:
    return None if value is None else str(value)


def _column_from_data(data: Mapping[str, Any]) -> ColumnContract:
    nullable_value = data.get("declared_nullable")
    return ColumnContract(
        name=str(data["name"]),
        formal_name=str(data.get("formal_name") or data["name"]),
        logical_type=str(data["logical_type"]),
        data_type=str(data.get("data_type") or ""),
        nullable=bool(data.get("nullable", False)),
        declared_nullable=(None if nullable_value is None else bool(nullable_value)),
        ownership=str(data.get("ownership") or "input"),
        unit=_optional_str(data.get("unit")),
        precision=_optional_int(data.get("precision")),
        scale=_optional_int(data.get("scale")),
        max_length=_optional_int(data.get("max_length")),
        constraint=_optional_str(data.get("constraint")),
        description=_optional_str(data.get("description")),
        group=_optional_str(data.get("group")),
        source=_optional_str(data.get("source")),
        aliases=tuple(str(alias) for alias in data.get("aliases", ())),
        uuid_version=_optional_int(data.get("uuid_version")),
        pattern=_optional_str(data.get("pattern")),
        minimum=_optional_number(data.get("minimum")),
        maximum=_optional_number(data.get("maximum")),
    )


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("boolean is not valid integer metadata")
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        return int(value)
    raise ValueError(f"invalid integer metadata: {value!r}")


def _optional_number(value: object) -> int | float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("boolean is not valid numeric metadata")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value
    if isinstance(value, str):
        return float(value)
    raise ValueError(f"invalid numeric metadata: {value!r}")


def _table_from_data(data: Mapping[str, Any]) -> TableContract:
    columns = tuple(_column_from_data(column) for column in data["columns"])
    references = tuple(
        ReferenceContract(
            column=str(item["column"]),
            target_table=str(item["target_table"]),
            target_column=str(item["target_column"]),
            nullable=bool(item.get("nullable", False)),
            allowed_values=tuple(item.get("allowed_values", ())),
            consistent_with=_optional_str(item.get("consistent_with")),
        )
        for item in data.get("references", ())
    )
    conditional_unique = tuple(
        ConditionalUnique(
            columns=tuple(str(name) for name in item["columns"]),
            when_column=str(item["when"]["column"]),
            equals=item["when"]["equals"],
        )
        for item in data.get("conditional_unique", ())
    )
    return TableContract(
        name=str(data["name"]),
        role=str(data["role"]),
        required=bool(data.get("required", False)),
        source_names=tuple(str(name) for name in data.get("source_names", ())),
        primary_key=tuple(str(name) for name in data.get("primary_key", ())),
        unique_keys=tuple(
            tuple(str(name) for name in key) for key in data.get("unique_keys", ())
        ),
        references=references,
        conditional_unique=conditional_unique,
        columns=columns,
    )


def load_pod_contract() -> PodContract:
    """Load the packaged, versioned contract without consulting local references."""
    schema_path = files("esdc").joinpath("schemas").joinpath("pod.yaml")
    payload = yaml.safe_load(schema_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("POD contract root must be a mapping")
    metadata = (
        str(payload.get("contract_id")),
        int(payload.get("contract_version", 0)),
        str(payload.get("dictionary_version")),
    )
    if metadata != _EXPECTED_CONTRACT:
        raise ValueError(f"unsupported packaged POD contract metadata: {metadata!r}")
    raw_tables = payload.get("tables")
    if not isinstance(raw_tables, Sequence):
        raise ValueError("POD contract tables must be a sequence")
    tables_list = [_table_from_data(table) for table in raw_tables]
    tables = {table.name: table for table in tables_list}
    if (
        len(tables) != len(tables_list)
        or {name: len(table.columns) for name, table in tables.items()}
        != _EXPECTED_COLUMN_COUNTS
    ):
        raise ValueError(
            "POD contract table names or ordered column counts are invalid"
        )
    for table in tables.values():
        if len({column.name for column in table.columns}) != len(table.columns):
            raise ValueError(f"duplicate column definition in {table.name}")
        if any(
            column.logical_type
            not in {"text", "integer", "decimal", "boolean", "date", "datetime", "uuid"}
            for column in table.columns
        ):
            raise ValueError(f"unsupported logical type in {table.name}")
    ref_values = payload.get("reference_values", {})
    return PodContract(
        contract_id=metadata[0],
        contract_version=metadata[1],
        dictionary_version=metadata[2],
        source_of_truth=str(payload.get("source_of_truth") or ""),
        notes=tuple(str(note) for note in payload.get("notes", ())),
        reference_values={
            str(name): tuple(values) for name, values in ref_values.items()
        },
        exceptions=tuple(payload.get("exceptions", ())),
        tables=tables,
    )


def _issue(
    batch: PreparedPodBatch,
    table: str,
    code: str,
    message: str,
    *,
    row: int | None = None,
    column: str | None = None,
) -> ValidationIssue:
    return ValidationIssue(
        source=batch.source,
        table=table,
        row=row,
        column=column,
        code=code,
        message=message,
    )


def _table_aliases(contract: PodContract) -> dict[str, TableContract]:
    aliases: dict[str, TableContract] = {}
    for table in contract.tables.values():
        if table.role != "input":
            continue
        for name in (table.name, *table.source_names):
            existing = aliases.get(name)
            if existing is not None and existing.name != table.name:
                raise ValueError(f"ambiguous source table alias {name!r}")
            aliases[name] = table
    return aliases


def _column_aliases(table: TableContract) -> dict[str, ColumnContract]:
    aliases: dict[str, ColumnContract] = {}
    for column in table.columns:
        for name in (column.name, *column.aliases):
            existing = aliases.get(name)
            if existing is not None and existing.name != column.name:
                raise ValueError(f"ambiguous column alias {name!r} in {table.name}")
            aliases[name] = column
    return aliases


def validate_pod_batch(
    batch: PreparedPodBatch, contract: PodContract
) -> tuple[ValidationIssue, ...]:
    """Validate headers, logical values, keys, and references before any write."""
    issues: list[ValidationIssue] = []
    if (
        batch.contract_id != contract.contract_id
        or batch.contract_version != contract.contract_version
        or batch.dictionary_version != contract.dictionary_version
    ):
        issues.append(
            _issue(
                batch,
                "batch",
                "unsupported_version",
                "Batch contract metadata does not match the supported POD contract.",
            )
        )

    table_aliases = _table_aliases(contract)
    supplied_headers: dict[str, tuple[str, Sequence[str]]] = {}
    for source_name, headers in batch.headers.items():
        table = table_aliases.get(source_name)
        if table is None:
            issues.append(
                _issue(
                    batch,
                    source_name,
                    "unexpected_table",
                    f"Table or sheet {source_name!r} is not a declared POD input.",
                )
            )
            continue
        if table.name in supplied_headers:
            issues.append(
                _issue(
                    batch,
                    table.name,
                    "duplicate_table",
                    f"Table {table.name!r} was supplied more than once "
                    "through aliases.",
                )
            )
            continue
        supplied_headers[table.name] = (source_name, headers)

    for table in contract.tables.values():
        if (
            table.role == "input"
            and table.required
            and table.name not in supplied_headers
        ):
            issues.append(
                _issue(
                    batch,
                    table.name,
                    "missing_table",
                    f"Required input table or sheet {table.name!r} is missing.",
                )
            )

    row_sources: dict[str, Sequence[PreparedRow]] = {}
    for source_name, rows in batch.rows.items():
        table = table_aliases.get(source_name)
        if table is None:
            if source_name not in batch.headers:
                issues.append(
                    _issue(
                        batch,
                        source_name,
                        "missing_table",
                        "Rows were supplied without a corresponding declared header.",
                    )
                )
            continue
        if table.name not in supplied_headers:
            issues.append(
                _issue(
                    batch,
                    table.name,
                    "missing_table",
                    "Rows were supplied for a table with no declared headers.",
                )
            )
            continue
        if table.name in row_sources:
            issues.append(
                _issue(
                    batch,
                    table.name,
                    "duplicate_row_source",
                    f"Rows for {table.name!r} were supplied through "
                    "more than one table name.",
                )
            )
            continue
        row_sources[table.name] = rows

    canonical_rows: dict[str, list[dict[str, object]]] = {}
    for table_name, (_source_name, headers) in supplied_headers.items():
        table = contract.tables[table_name]
        column_aliases = _column_aliases(table)
        header_columns: dict[str, str] = {}
        for header in headers:
            column_name = str(header)
            column = column_aliases.get(column_name)
            if column is None:
                issues.append(
                    _issue(
                        batch,
                        table_name,
                        "unexpected_column",
                        f"Column {column_name!r} is not declared for {table_name}.",
                        column=column_name,
                    )
                )
                continue
            if column.name in header_columns:
                issues.append(
                    _issue(
                        batch,
                        table_name,
                        "duplicate_column",
                        f"Column {column.name!r} appears more than once.",
                        column=column.name,
                    )
                )
                continue
            header_columns[column.name] = column_name
        for column in table.columns:
            if column.name not in header_columns:
                issues.append(
                    _issue(
                        batch,
                        table_name,
                        "missing_column",
                        f"Required contract column {column.name!r} is absent.",
                        column=column.name,
                    )
                )

        table_rows = row_sources.get(table_name, ())
        canonical_rows[table_name] = []
        for row_index, source_row in enumerate(table_rows, start=1):
            row_number = row_index
            if not isinstance(source_row, Mapping):
                issues.append(
                    _issue(
                        batch,
                        table_name,
                        "type_mismatch",
                        "Prepared source row must map column names to scalar values.",
                        row=row_number,
                    )
                )
                continue
            normalized_row: dict[str, object] = {}
            for source_column, value in source_row.items():
                source_column_name = str(source_column)
                column = column_aliases.get(source_column_name)
                if column is None:
                    issues.append(
                        _issue(
                            batch,
                            table_name,
                            "unexpected_column",
                            f"Row contains undeclared column {source_column_name!r}.",
                            row=row_number,
                            column=source_column_name,
                        )
                    )
                    continue
                if column.name in normalized_row:
                    issues.append(
                        _issue(
                            batch,
                            table_name,
                            "duplicate_column",
                            f"Row maps more than one source column to {column.name!r}.",
                            row=row_number,
                            column=column.name,
                        )
                    )
                    continue
                normalized_row[column.name] = value
            for column in table.columns:
                value = normalized_row.get(column.name)
                _validate_value(
                    batch,
                    contract,
                    table,
                    column,
                    value,
                    normalized_row,
                    row_number,
                    issues,
                )
            _validate_row_comparisons(batch, table, normalized_row, row_number, issues)
            canonical_rows[table_name].append(normalized_row)

    _validate_keys(batch, contract, canonical_rows, issues)
    _validate_references(batch, contract, canonical_rows, issues)
    _validate_supersession(batch, contract, canonical_rows, issues)
    _validate_conditional_keys(batch, contract, canonical_rows, issues)
    _validate_identity_coherence(batch, contract, canonical_rows, issues)
    return tuple(issues)


def _semoga_exception(
    contract: PodContract,
    table_name: str,
    column_name: str,
    row: Mapping[str, object],
) -> bool:
    if table_name != "m_pod":
        return False
    for exception in contract.exceptions:
        if exception.get("table") != table_name:
            continue
        exception_columns = exception.get("columns", (exception.get("column"),))
        if isinstance(exception_columns, str) or not isinstance(
            exception_columns, Sequence
        ) or column_name not in exception_columns:
            continue
        identity = exception.get("identity", {})
        if not isinstance(identity, Mapping):
            continue
        matches = True
        for name, expected in identity.items():
            actual = row.get(str(name))
            if name == "approval_date":
                actual_date = _date_value(actual)
                matches = actual_date is not None and actual_date.isoformat() == str(
                    expected
                )
            else:
                matches = actual == expected
            if not matches:
                break
        if matches:
            return True
    return False


def _canonical_key_value(column: ColumnContract, value: object) -> object:
    if column.logical_type == "uuid":
        if isinstance(value, UUID):
            return value
        if isinstance(value, str):
            try:
                return UUID(value)
            except ValueError:
                return value
    if column.logical_type == "date":
        normalized_date = _date_value(value)
        if normalized_date is not None:
            return normalized_date
    return value


def _is_hashable(value: object) -> bool:
    try:
        hash(value)
    except TypeError:
        return False
    return True


def _date_value(value: object) -> date | None:
    if isinstance(value, datetime):
        return value.date() if value.time().replace(tzinfo=None) == time.min else None
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value)
        except ValueError:
            return None
    return None


def _decimal_value(value: object) -> Decimal | None:
    if isinstance(value, (bool, str)):
        return None
    if isinstance(value, Decimal):
        return value
    if isinstance(value, (Integral, Real)):
        try:
            return Decimal(str(value))
        except InvalidOperation:
            return None
    return None


def _validate_value(
    batch: PreparedPodBatch,
    contract: PodContract,
    table: TableContract,
    column: ColumnContract,
    value: object,
    row: Mapping[str, object],
    row_number: int,
    issues: list[ValidationIssue],
) -> None:
    def add(code: str, message: str) -> None:
        issues.append(
            _issue(
                batch,
                table.name,
                code,
                message,
                row=row_number,
                column=column.name,
            )
        )

    if value is None:
        allowed_exception = _semoga_exception(
            contract, table.name, column.name, row
        )
        if not column.nullable and not allowed_exception:
            add("missing_value", f"{column.name} is required.")
        return

    logical_type = column.logical_type
    if logical_type == "text":
        if not isinstance(value, str):
            add("type_mismatch", f"{column.name} must be text.")
            return
        if column.max_length is not None and len(value) > column.max_length:
            add(
                "invalid_value",
                f"{column.name} exceeds {column.max_length} characters.",
            )
        if column.pattern is not None and re.fullmatch(column.pattern, value) is None:
            add("invalid_value", f"{column.name} does not match its declared format.")
        if (
            column.constraint
            and re.search(r"btrim\s*\(", column.constraint, re.I)
            and not value.strip()
        ):
            add("invalid_value", f"{column.name} must not be blank.")
        return

    if logical_type == "integer":
        integer: int | None = None
        if isinstance(value, bool):
            add(
                "type_mismatch",
                f"{column.name} must be an integer identifier or count.",
            )
            return
        if isinstance(value, Integral):
            integer = int(value)
        elif isinstance(value, Decimal) and value.is_finite():
            if value == value.to_integral_value():
                integer = int(value)
        elif isinstance(value, Real):
            numeric_value = float(value)
            if math.isfinite(numeric_value) and numeric_value.is_integer():
                integer = int(numeric_value)
        if integer is None:
            add(
                "type_mismatch",
                f"{column.name} must be an integer; numeric text is not coerced.",
            )
            return
        limits = _INTEGER_LIMITS.get(column.data_type.upper())
        minimum = (
            column.minimum
            if column.minimum is not None
            else (limits[0] if limits else None)
        )
        maximum = (
            column.maximum
            if column.maximum is not None
            else (limits[1] if limits else None)
        )
        if minimum is not None and integer < minimum:
            add("invalid_value", f"{column.name} is below its minimum of {minimum}.")
        if maximum is not None and integer > maximum:
            add("invalid_value", f"{column.name} is above its maximum of {maximum}.")
        _validate_scalar_checks(
            batch, table, column, Decimal(integer), row_number, issues
        )
        return

    if logical_type == "decimal":
        number = _decimal_value(value)
        if number is None:
            add(
                "type_mismatch",
                f"{column.name} must be numeric; numeric text is not coerced.",
            )
            return
        if not number.is_finite():
            add("non_finite", f"{column.name} must be finite.")
            return
        if column.precision is not None and column.scale is not None:
            quantum = Decimal(1).scaleb(-column.scale)
            try:
                if number.quantize(quantum) != number:
                    add(
                        "precision_exceeded",
                        f"{column.name} has more than "
                        f"{column.scale} fractional digits.",
                    )
                    return
            except InvalidOperation:
                add(
                    "precision_exceeded",
                    f"{column.name} exceeds its declared precision.",
                )
                return
            integer_digits = column.precision - column.scale
            if abs(number) >= Decimal(10) ** integer_digits:
                add(
                    "precision_exceeded",
                    f"{column.name} exceeds precision {column.precision}.",
                )
                return
        _validate_scalar_checks(batch, table, column, number, row_number, issues)
        return

    if logical_type == "boolean":
        if isinstance(value, bool):
            return
        if isinstance(value, Integral) and int(value) in (0, 1):
            return
        if (
            isinstance(value, Real)
            and math.isfinite(float(value))
            and float(value) in (0.0, 1.0)
        ):
            return
        add(
            "type_mismatch",
            f"{column.name} must be boolean or the logical value 0 or 1.",
        )
        return

    if logical_type == "date":
        if isinstance(value, datetime):
            if value.time().replace(tzinfo=None) != time.min:
                add("invalid_date", f"{column.name} must not contain a time component.")
            return
        if isinstance(value, date):
            return
        if isinstance(value, str):
            try:
                date.fromisoformat(value)
            except ValueError:
                add("invalid_date", f"{column.name} is not a valid ISO date.")
            return
        add("type_mismatch", f"{column.name} must be a date.")
        return

    if logical_type == "datetime":
        if isinstance(value, datetime):
            return
        if isinstance(value, str):
            try:
                datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError:
                add("invalid_date", f"{column.name} is not a valid ISO datetime.")
            return
        add("type_mismatch", f"{column.name} must be a datetime.")
        return

    if logical_type == "uuid":
        if isinstance(value, UUID):
            parsed_uuid = value
        elif isinstance(value, str):
            try:
                parsed_uuid = UUID(value)
            except (ValueError, AttributeError):
                add("invalid_uuid", f"{column.name} is not a valid UUID.")
                return
        else:
            add("type_mismatch", f"{column.name} must be a UUID or UUID string.")
            return
        if (
            column.uuid_version is not None
            and parsed_uuid.version != column.uuid_version
        ):
            add(
                "invalid_uuid",
                f"{column.name} must be a UUID version {column.uuid_version}.",
            )
        return

    add(
        "type_mismatch",
        f"{column.name} uses unsupported logical type {logical_type!r}.",
    )


def _validate_scalar_checks(
    batch: PreparedPodBatch,
    table: TableContract,
    column: ColumnContract,
    value: Decimal,
    row_number: int,
    issues: list[ValidationIssue],
) -> None:
    constraint = column.constraint or ""
    rules: list[tuple[str, Decimal]] = []
    if column.minimum is not None:
        rules.append((">=", Decimal(str(column.minimum))))
    if column.maximum is not None:
        rules.append(("<=", Decimal(str(column.maximum))))
    for match in _BETWEEN_RE.finditer(constraint):
        rules.append((">=", Decimal(match.group(1))))
        rules.append(("<=", Decimal(match.group(2))))
    for match in _COMPARE_RE.finditer(constraint):
        left, operator, right = match.groups()
        if left.casefold() != column.name.casefold():
            continue
        try:
            rules.append((operator, Decimal(right)))
        except InvalidOperation:
            continue
    for operator, bound in rules:
        valid = {
            ">=": value >= bound,
            ">": value > bound,
            "<=": value <= bound,
            "<": value < bound,
            "=": value == bound,
            "<>": value != bound,
        }[operator]
        if not valid:
            issues.append(
                _issue(
                    batch,
                    table.name,
                    "invalid_value",
                    f"{column.name} violates its declared numeric check.",
                    row=row_number,
                    column=column.name,
                )
            )
            break


def _validate_row_comparisons(
    batch: PreparedPodBatch,
    table: TableContract,
    row: Mapping[str, object],
    row_number: int,
    issues: list[ValidationIssue],
) -> None:
    for column in table.columns:
        if not column.constraint:
            continue
        for left_name, operator, right_name in _COMPARE_RE.findall(column.constraint):
            left_value = row.get(left_name)
            if isinstance(left_value, bool) or left_value is None:
                continue
            if right_name in row:
                right_value = row.get(right_name)
                if isinstance(right_value, bool) or right_value is None:
                    continue
                left_number = _decimal_value(left_value)
                right_number = _decimal_value(right_value)
                if (
                    left_number is None
                    or right_number is None
                    or not left_number.is_finite()
                    or not right_number.is_finite()
                ):
                    continue
            else:
                continue
            valid = {
                ">=": left_number >= right_number,
                ">": left_number > right_number,
                "<=": left_number <= right_number,
                "<": left_number < right_number,
                "=": left_number == right_number,
                "<>": left_number != right_number,
            }[operator]
            if not valid:
                issues.append(
                    _issue(
                        batch,
                        table.name,
                        "invalid_value",
                        f"{left_name} violates its declared relationship "
                        f"to {right_name}.",
                        row=row_number,
                        column=left_name,
                    )
                )


def _validate_keys(
    batch: PreparedPodBatch,
    contract: PodContract,
    rows: Mapping[str, Sequence[Mapping[str, object]]],
    issues: list[ValidationIssue],
) -> None:
    for table_name, table_rows in rows.items():
        table = contract.tables[table_name]
        keys = ([table.primary_key] if table.primary_key else []) + list(
            table.unique_keys
        )
        columns_by_name = {column.name: column for column in table.columns}
        for key in keys:
            seen: dict[tuple[object, ...], int] = {}
            for row_number, row in enumerate(table_rows, start=1):
                raw_values = tuple(row.get(column) for column in key)
                if any(value is None for value in raw_values):
                    continue
                values = tuple(
                    _canonical_key_value(columns_by_name[column], value)
                    for column, value in zip(key, raw_values, strict=True)
                )
                if not all(_is_hashable(value) for value in values):
                    continue
                if values in seen:
                    issues.append(
                        _issue(
                            batch,
                            table_name,
                            "duplicate_key",
                            f"Duplicate key {key!r} first seen on row {seen[values]}.",
                            row=row_number,
                            column=key[0] if key else None,
                        )
                    )
                else:
                    seen[values] = row_number


def _identity_integer_value(value: object) -> int | None:
    """Canonicalize an integer using the representations the validator accepts."""
    if isinstance(value, bool):
        return None
    if isinstance(value, Integral):
        return int(value)
    if (
        isinstance(value, Decimal)
        and value.is_finite()
        and value == value.to_integral_value()
    ):
        return int(value)
    if isinstance(value, Real):
        numeric = float(value)
        if math.isfinite(numeric) and numeric.is_integer():
            return int(numeric)
    return None


def _validate_identity_coherence(
    batch: PreparedPodBatch,
    contract: PodContract,
    rows: Mapping[str, Sequence[Mapping[str, object]]],
    issues: list[ValidationIssue],
) -> None:
    """Compare each declared business identity with the approved approval-key rule.

    Values that are already the wrong type or outside an approved range are reported
    by the column and reference checks, so this pass only compares a fully typed key
    with the canonical identity and never raises for them. The declared exception is
    limited to its exact unissued pair: a supplied pair on that row is validated like
    any other identity, and a half pair is reported as an incomplete identity claim.
    """
    table_name = "m_pod"
    for row_number, row in enumerate(rows.get(table_name, ()), start=1):
        if _semoga_exception(contract, table_name, "pod_id", row):
            # The declared exception covers the exact unissued pair only. A supplied
            # pair on that row is still an identity claim compared with the canonical
            # formula, and one half alone is not the placeholder the exception names.
            has_code = row.get("pod_id") is not None
            has_order = row.get("pod_order_no") is not None
            if has_code != has_order:
                missing = "pod_id" if not has_code else "pod_order_no"
                issues.append(
                    _issue(
                        batch,
                        table_name,
                        "invalid_value",
                        "the declared exception permits the unissued identity pair "
                        f"only; {missing} is missing while the other half is supplied.",
                        row=row_number,
                        column=missing,
                    )
                )
                continue
            if not has_code and not has_order:
                continue
        approval_date = _date_value(row.get("approval_date"))
        order_no = _identity_integer_value(row.get("pod_order_no"))
        if approval_date is None or order_no is None:
            continue
        day_offset = (approval_date.timetuple().tm_yday - 1) * DAILY_SLOTS_PER_DAY
        daily_slot = order_no - day_offset
        if not 1 <= daily_slot <= DAILY_SLOTS_PER_DAY:
            issues.append(
                _issue(
                    batch,
                    table_name,
                    "invalid_value",
                    f"pod_order_no {order_no} is not one of the {DAILY_SLOTS_PER_DAY} "
                    "order numbers for approval_date "
                    f"{approval_date.isoformat()}; expected "
                    f"{day_offset + 1}..{day_offset + DAILY_SLOTS_PER_DAY}.",
                    row=row_number,
                    column="pod_order_no",
                )
            )
            continue
        institution_id = _identity_integer_value(row.get("institution_id"))
        pod_type_id = _identity_integer_value(row.get("pod_type_id"))
        rev_num = _identity_integer_value(row.get("rev_num"))
        if institution_id is None or pod_type_id is None or rev_num is None:
            continue
        try:
            _, _, canonical_pod_id = validate_key(
                ApprovalKey(
                    approval_date=approval_date,
                    daily_slot=daily_slot,
                    institution_id=institution_id,
                    pod_type_id=pod_type_id,
                    rev_num=rev_num,
                )
            )
        except (TypeError, ValueError):
            continue
        pod_id = row.get("pod_id")
        if isinstance(pod_id, str) and pod_id != canonical_pod_id:
            issues.append(
                _issue(
                    batch,
                    table_name,
                    "invalid_value",
                    f"pod_id {pod_id} is not the approved identity for this approval "
                    f"key; expected {canonical_pod_id}.",
                    row=row_number,
                    column="pod_id",
                )
            )


def _validate_references(
    batch: PreparedPodBatch,
    contract: PodContract,
    rows: Mapping[str, Sequence[Mapping[str, object]]],
    issues: list[ValidationIssue],
) -> None:
    indexes: dict[tuple[str, str], set[object]] = {}
    rows_by_key: dict[tuple[str, str], dict[object, Mapping[str, object]]] = {}
    for table_name, table_rows in rows.items():
        table = contract.tables[table_name]
        indexed_columns = {table.primary_key[0]} if table.primary_key else set()
        indexed_columns.update(
            reference.target_column
            for source in contract.tables.values()
            for reference in source.references
            if reference.target_table == table_name
        )
        for column_name in indexed_columns:
            values: set[object] = set()
            mapped: dict[object, Mapping[str, object]] = {}
            for row in table_rows:
                value = row.get(column_name)
                if value is not None and _is_hashable(value):
                    values.add(value)
                    mapped[value] = row
            indexes[(table_name, column_name)] = values
            rows_by_key[(table_name, column_name)] = mapped

    for table_name, table_rows in rows.items():
        table = contract.tables[table_name]
        for reference in table.references:
            for row_number, row in enumerate(table_rows, start=1):
                value = row.get(reference.column)
                if value is None or not _is_hashable(value):
                    continue
                allowed = set(reference.allowed_values)
                if not allowed:
                    allowed = set(
                        contract.reference_values.get(
                            f"{reference.target_table}.{reference.target_column}", ()
                        )
                    )
                target_values = indexes.get(
                    (reference.target_table, reference.target_column), set()
                )
                if allowed and value not in allowed:
                    issues.append(
                        _issue(
                            batch,
                            table_name,
                            "invalid_reference",
                            f"{reference.column} references an unknown "
                            f"{reference.target_table}.{reference.target_column}.",
                            row=row_number,
                            column=reference.column,
                        )
                    )
                    continue
                if (
                    reference.target_table in contract.tables
                    and (target_values or reference.target_table in rows)
                    and value not in target_values
                ):
                    issues.append(
                        _issue(
                            batch,
                            table_name,
                            "invalid_reference",
                            f"{reference.column} references a missing "
                            f"{reference.target_table} row.",
                            row=row_number,
                            column=reference.column,
                        )
                    )
                    continue
                if reference.consistent_with:
                    join_value = row.get(reference.consistent_with)
                    target_rows = rows_by_key.get(
                        (reference.target_table, reference.consistent_with), {}
                    )
                    target_row = (
                        target_rows.get(join_value)
                        if _is_hashable(join_value)
                        else None
                    )
                    if (
                        target_row is not None
                        and target_row.get(reference.target_column) != value
                    ):
                        issues.append(
                            _issue(
                                batch,
                                table_name,
                                "invalid_reference",
                                f"{reference.column} does not match its "
                                f"{reference.consistent_with} master row.",
                                row=row_number,
                                column=reference.column,
                            )
                        )


def _validate_supersession(
    batch: PreparedPodBatch,
    contract: PodContract,
    rows: Mapping[str, Sequence[Mapping[str, object]]],
    issues: list[ValidationIssue],
) -> None:
    master_rows = rows.get("m_pod", ())
    master: dict[object, tuple[int, Mapping[str, object]]] = {}
    for index, row in enumerate(master_rows, start=1):
        pod_id_seq = row.get("pod_id_seq")
        if pod_id_seq is not None and _is_hashable(pod_id_seq):
            master[pod_id_seq] = (index, row)
    for row_number, row in enumerate(master_rows, start=1):
        replacement_id = row.get("superseded_by")
        if replacement_id is None:
            if row.get("is_latest") is False or row.get("is_latest") == 0:
                issues.append(
                    _issue(
                        batch,
                        "m_pod",
                        "invalid_value",
                        "is_latest must be true when superseded_by is NULL.",
                        row=row_number,
                        column="is_latest",
                    )
                )
            continue
        if not _is_hashable(replacement_id):
            continue
        if replacement_id == row.get("pod_id_seq"):
            issues.append(
                _issue(
                    batch,
                    "m_pod",
                    "invalid_reference",
                    "A POD cannot supersede itself.",
                    row=row_number,
                    column="superseded_by",
                )
            )
            continue
        target = master.get(replacement_id)
        if target is not None:
            target_date = _date_value(target[1].get("approval_date"))
            source_date = _date_value(row.get("approval_date"))
            if (
                target_date is not None
                and source_date is not None
                and target_date <= source_date
            ):
                issues.append(
                    _issue(
                        batch,
                        "m_pod",
                        "invalid_reference",
                        "A superseding POD must have a later approval date.",
                        row=row_number,
                        column="superseded_by",
                    )
                )
            if row.get("is_latest") is True or row.get("is_latest") == 1:
                issues.append(
                    _issue(
                        batch,
                        "m_pod",
                        "invalid_value",
                        "is_latest must be false when superseded_by is set.",
                        row=row_number,
                        column="is_latest",
                    )
                )

    reported_cycles: set[object] = set()
    for start_id, (row_number, _) in master.items():
        seen: set[object] = set()
        current = start_id
        while current in master:
            if current in seen:
                if current not in reported_cycles:
                    reported_cycles.add(current)
                    issues.append(
                        _issue(
                            batch,
                            "m_pod",
                            "invalid_reference",
                            "POD supersession relationships must be acyclic.",
                            row=row_number,
                            column="superseded_by",
                        )
                    )
                break
            seen.add(current)
            current = master[current][1].get("superseded_by")
            if current is None or not _is_hashable(current):
                break


def _validate_conditional_keys(
    batch: PreparedPodBatch,
    contract: PodContract,
    rows: Mapping[str, Sequence[Mapping[str, object]]],
    issues: list[ValidationIssue],
) -> None:
    for table_name, table_rows in rows.items():
        table = contract.tables[table_name]
        for rule in table.conditional_unique:
            seen: dict[tuple[object, ...], int] = {}
            for row_number, row in enumerate(table_rows, start=1):
                if row.get(rule.when_column) != rule.equals:
                    continue
                values = tuple(row.get(column) for column in rule.columns)
                if any(value is None for value in values) or not all(
                    _is_hashable(value) for value in values
                ):
                    continue
                if values in seen:
                    issues.append(
                        _issue(
                            batch,
                            table_name,
                            "duplicate_key",
                            f"Duplicate conditional key {rule.columns!r}.",
                            row=row_number,
                            column=rule.columns[0] if rule.columns else None,
                        )
                    )
                else:
                    seen[values] = row_number
