"""SQLite truth for POD value cases (plan, actual, outlook).

Canonical POD identity is ``m_pod.pod_id``; ``project_pod`` stays the
separate canonical linkage and is never written here. All rows are
validated before the replacement transaction opens.
"""

from __future__ import annotations

import sqlite3
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from esdc.loaders import LoadSchema
from esdc.pod_registry.store import get_sqlite_connection

_VALUE_CASE_IDENTIFIERS = (
    "pod_id",
    "case_type",
    "report_date",
    "as_of_date",
    "pod_scope",
)
_MONITORING_CASE_TYPES = ("actual", "outlook")


def replace_pod_value_cases(
    sqlite_path: Path | None,
    plan_df: pd.DataFrame,
    monitoring_df: pd.DataFrame,
    schema: LoadSchema,
) -> int:
    """Validate and replace every POD value case in one SQLite transaction.

    Returns the number of value-case rows written. Any validation error
    (unknown ``pod_id``, duplicate key, invalid case type or date) aborts
    before any existing truth rows are replaced.
    """
    _require_identifier_columns(schema)
    metric_columns = _metric_columns(schema)
    rows = _rows_from_frame(
        plan_df, metric_columns, source="plan", case_type="plan"
    ) + _rows_from_frame(
        monitoring_df, metric_columns, source="monitoring", case_type=None
    )
    _reject_duplicate_keys(rows)

    conn = get_sqlite_connection(sqlite_path)
    try:
        _reject_unknown_pod_ids(conn, rows)
        with conn:
            conn.execute(_create_table_sql(metric_columns))
            conn.execute("DELETE FROM pod_value_case")
            conn.executemany(
                _insert_sql(metric_columns),
                [_row_tuple(row, metric_columns) for row in rows],
            )
        return len(rows)
    finally:
        conn.close()


def _require_identifier_columns(schema: LoadSchema) -> None:
    names = {column.name for column in schema.columns}
    missing = [name for name in _VALUE_CASE_IDENTIFIERS if name not in names]
    if missing:
        raise ValueError(
            "value-case schema is missing required identifier columns: "
            + ", ".join(missing)
        )


def _metric_columns(schema: LoadSchema) -> tuple[str, ...]:
    return tuple(
        column.name
        for column in schema.columns
        if column.name not in _VALUE_CASE_IDENTIFIERS
    )


def _is_missing(value: Any) -> bool:
    return value is None or bool(pd.isna(value))


def _required_pod_id(value: Any, row: int) -> str:
    if _is_missing(value):
        raise ValueError(f"value-case row {row}: pod_id is required")
    pod_id = str(value).strip()
    if not pod_id:
        raise ValueError(f"value-case row {row}: pod_id is required")
    return pod_id


def _required_iso_date(value: Any, column: str, row: int) -> str:
    if _is_missing(value):
        raise ValueError(f"value-case row {row}: {column} is required")
    if isinstance(value, pd.Timestamp):
        return value.date().isoformat()
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, str):
        try:
            return date.fromisoformat(value.strip()).isoformat()
        except ValueError as e:
            raise ValueError(
                f"value-case row {row}: {column} must be an ISO date"
            ) from e
    raise ValueError(f"value-case row {row}: {column} must be an ISO date")


def _optional_text(value: Any, row: int) -> str | None:
    if _is_missing(value):
        return None
    text = str(value).strip()
    return text or None


def _monitoring_case_type(value: Any, row: int) -> str:
    if _is_missing(value):
        raise ValueError(f"monitoring row {row}: case_type is required")
    case_type = str(value).strip().lower()
    if case_type not in _MONITORING_CASE_TYPES:
        raise ValueError(
            f"monitoring row {row}: case_type {value!r} must be one of "
            + ", ".join(_MONITORING_CASE_TYPES)
        )
    return case_type


def _to_real(value: Any, column: str, row: int) -> float | None:
    if _is_missing(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as e:
        raise ValueError(f"value-case row {row}: {column} must be numeric") from e


def _rows_from_frame(
    df: pd.DataFrame,
    metric_columns: tuple[str, ...],
    *,
    source: str,
    case_type: str | None,
) -> list[dict[str, Any]]:
    if len(df) == 0:
        return []
    required = ("pod_id", "report_date", "as_of_date")
    if case_type is None:
        required = required + ("case_type",)
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"{source} is missing required columns: {', '.join(missing)}")

    rows: list[dict[str, Any]] = []
    for index, record in enumerate(df.to_dict("records"), start=1):
        row: dict[str, Any] = {
            "pod_id": _required_pod_id(record.get("pod_id"), index),
            "case_type": (
                case_type
                if case_type is not None
                else _monitoring_case_type(record.get("case_type"), index)
            ),
            "report_date": _required_iso_date(
                record.get("report_date"), "report_date", index
            ),
            "as_of_date": _required_iso_date(
                record.get("as_of_date"), "as_of_date", index
            ),
            "pod_scope": _optional_text(record.get("pod_scope"), index),
        }
        for column in metric_columns:
            row[column] = _to_real(record.get(column), column, index)
        rows.append(row)
    return rows


def _reject_duplicate_keys(rows: list[dict[str, Any]]) -> None:
    seen: set[tuple[str, str, str]] = set()
    for row in rows:
        key = (row["pod_id"], row["case_type"], row["as_of_date"])
        if key in seen:
            raise ValueError(
                f"duplicate value-case key (pod_id, case_type, as_of_date): {key!r}"
            )
        seen.add(key)


def _reject_unknown_pod_ids(
    conn: sqlite3.Connection, rows: list[dict[str, Any]]
) -> None:
    known = {str(value) for (value,) in conn.execute("SELECT pod_id FROM m_pod")}
    unknown = sorted({row["pod_id"] for row in rows} - known)
    if unknown:
        raise ValueError(
            "unknown pod_id in value cases (not in canonical m_pod): "
            + ", ".join(unknown)
        )


def _create_table_sql(metric_columns: tuple[str, ...]) -> str:
    metric_sql = "\n".join(f"    {column} REAL," for column in metric_columns)
    return (
        "CREATE TABLE IF NOT EXISTS pod_value_case (\n"
        "    pod_id TEXT NOT NULL REFERENCES m_pod(pod_id)"
        " DEFERRABLE INITIALLY DEFERRED,\n"
        "    case_type TEXT NOT NULL"
        " CHECK (case_type IN ('plan', 'actual', 'outlook')),\n"
        "    report_date TEXT NOT NULL,\n"
        "    as_of_date TEXT NOT NULL,\n"
        "    pod_scope TEXT,\n"
        f"{metric_sql}\n"
        "    UNIQUE (pod_id, case_type, as_of_date)\n"
        ")"
    )


def _insert_sql(metric_columns: tuple[str, ...]) -> str:
    columns = _VALUE_CASE_IDENTIFIERS + metric_columns
    placeholders = ", ".join("?" for _ in columns)
    return f"INSERT INTO pod_value_case ({', '.join(columns)}) VALUES ({placeholders})"


def _row_tuple(row: dict[str, Any], metric_columns: tuple[str, ...]) -> tuple[Any, ...]:
    return tuple(row[column] for column in _VALUE_CASE_IDENTIFIERS + metric_columns)
