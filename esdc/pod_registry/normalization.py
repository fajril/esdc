"""Source normalization shared by POD workbook, SQLite, and API adapters."""

from __future__ import annotations

import math
import re
from datetime import date, datetime, time
from decimal import Decimal, InvalidOperation
from numbers import Integral, Real
from uuid import UUID

from esdc.pod_registry.contract import ColumnContract, TableContract
from esdc.pod_registry.models import PreparedScalar

_CANONICAL_PROJECT_ID = re.compile(r"P-[A-Z0-9]{7}-[A-Z0-9]{2}\Z", re.ASCII)
_COMPACT_PROJECT_ID = re.compile(r"P[A-Z0-9]{9}\Z", re.ASCII)


class PodNormalizationError(ValueError):
    """A source cell cannot be represented without losing its meaning."""


def normalize_project_id(value: object) -> str:
    """Return the canonical alphanumeric project identifier used by ESDC."""
    if not isinstance(value, str):
        raise ValueError("canonical project ID must be text")
    normalized = value.strip().upper()
    if _CANONICAL_PROJECT_ID.fullmatch(normalized):
        return normalized
    if _COMPACT_PROJECT_ID.fullmatch(normalized):
        return f"{normalized[:1]}-{normalized[1:8]}-{normalized[8:10]}"
    raise ValueError("value is not a canonical project ID")


def _decimal(value: object, *, allow_text: bool) -> Decimal:
    if isinstance(value, bool):
        raise PodNormalizationError("boolean is not a numeric value")
    if isinstance(value, Decimal):
        result = value
    elif isinstance(value, Integral):
        result = Decimal(int(value))
    elif isinstance(value, Real):
        numeric = float(value)
        if not math.isfinite(numeric):
            raise PodNormalizationError("numeric value must be finite")
        result = Decimal(str(numeric))
    elif allow_text and isinstance(value, str):
        try:
            result = Decimal(value.strip())
        except InvalidOperation as exc:
            raise PodNormalizationError("numeric text is invalid") from exc
    else:
        raise PodNormalizationError("value must be numeric")
    if not result.is_finite():
        raise PodNormalizationError("numeric value must be finite")
    return result


def _as_date(value: object) -> date:
    if isinstance(value, datetime):
        if value.timetz().replace(tzinfo=None) != time.min:
            raise PodNormalizationError("date must not contain a time component")
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value.strip())
        except ValueError:
            try:
                parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
            except ValueError as exc:
                raise PodNormalizationError("date is not valid ISO date text") from exc
            if parsed.timetz().replace(tzinfo=None) != time.min:
                raise PodNormalizationError(
                    "date must not contain a time component"
                ) from None
            return parsed.date()
    raise PodNormalizationError("value must be a date")


def _as_datetime(value: object) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        except ValueError as exc:
            raise PodNormalizationError(
                "timestamp is not valid ISO datetime text"
            ) from exc
    raise PodNormalizationError("value must be a timestamp")


def normalize_pod_value(
    value: object,
    column: ColumnContract,
    *,
    source: str,
) -> PreparedScalar:
    """Normalize a cell by its YAML logical type without rounding or guessing."""
    if value is None:
        return None
    logical_type = column.logical_type
    if logical_type == "text":
        if not isinstance(value, str):
            raise PodNormalizationError("value must be text")
        return normalize_project_id(value) if column.name == "project_id" else value
    if logical_type == "integer":
        # XLSX stores long identifiers as literal text to avoid Excel rounding.
        # API payloads retain their strict numeric integer contract.
        if (
            source == "xlsx"
            and isinstance(value, str)
            and re.fullmatch(r"[+-]?[0-9]+", value, re.ASCII)
        ):
            return int(value)
        if isinstance(value, bool):
            raise PodNormalizationError("boolean is not an integer identifier")
        if isinstance(value, Integral):
            return int(value)
        if isinstance(value, Decimal) and value.is_finite():
            if value == value.to_integral_value():
                return int(value)
        elif isinstance(value, Real):
            numeric = float(value)
            if math.isfinite(numeric) and numeric.is_integer():
                return int(numeric)
        raise PodNormalizationError("value must be an integral number")
    if logical_type == "decimal":
        number = _decimal(value, allow_text=source in {"sqlite", "xlsx", "api"})
        storage_type = column.data_type.upper()
        if any(marker in storage_type for marker in ("DOUBLE", "REAL", "FLOAT")):
            converted = float(number)
            if not math.isfinite(converted):
                raise PodNormalizationError(
                    "numeric value is outside the finite SQL DOUBLE range"
                )
            return converted
        return number
    if logical_type == "boolean":
        if isinstance(value, bool):
            return value
        if isinstance(value, Integral) and int(value) in (0, 1):
            return int(value)
        if (
            isinstance(value, Real)
            and math.isfinite(float(value))
            and float(value) in (0.0, 1.0)
        ):
            return int(float(value))
        raise PodNormalizationError("boolean value must be true, false, 0, or 1")
    if logical_type == "date":
        return _as_date(value)
    if logical_type == "datetime":
        return _as_datetime(value)
    if logical_type == "uuid":
        if isinstance(value, UUID):
            return value
        if isinstance(value, str):
            try:
                return UUID(value.strip())
            except ValueError as exc:
                raise PodNormalizationError("value is not a valid UUID") from exc
        raise PodNormalizationError("value must be a UUID")
    raise PodNormalizationError(f"unsupported logical type {logical_type!r}")


def canonical_column_map(table: TableContract) -> dict[str, ColumnContract]:
    """Map declared headers and aliases to their canonical YAML columns."""
    result: dict[str, ColumnContract] = {}
    for column in table.columns:
        result[column.name] = column
        result.update(dict.fromkeys(column.aliases, column))
    return result


def normalize_source_row(
    headers: list[str],
    values: list[object],
    table: TableContract,
    *,
    source: str,
    row_number: int,
) -> tuple[list[str], dict[str, PreparedScalar]]:
    """Map aliases and normalize one row while retaining unknown columns."""
    by_header = canonical_column_map(table)
    canonical_headers: list[str] = []
    normalized: dict[str, PreparedScalar] = {}
    width = max(len(headers), len(values))
    for index in range(width):
        header = (
            headers[index] if index < len(headers) else f"__extra_column_{index + 1}"
        )
        raw = values[index] if index < len(values) else None
        column = by_header.get(header)
        if column is None:
            canonical_headers.append(header)
            normalized[header] = raw  # type: ignore[assignment]
            continue
        canonical_headers.append(column.name)
        try:
            normalized[column.name] = normalize_pod_value(raw, column, source=source)
        except PodNormalizationError as exc:
            raise PodNormalizationError(
                f"{table.name} row {row_number}, column {column.name}: {exc}"
            ) from exc
    return canonical_headers, normalized
