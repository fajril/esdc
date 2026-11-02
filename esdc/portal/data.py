"""Read-only bounded current-registry portal queries."""

from __future__ import annotations

import json
import math
import re
import sqlite3
from collections.abc import Callable
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation
from math import ceil
from pathlib import Path
from typing import Any

from esdc.pod_registry.api_context import read_only_connection
from esdc.pod_registry.migrations import (
    MigrationRefusedError,
    _validate_existing_v2_registry,
)
from esdc.pod_registry.store import get_esdc_sqlite_path
from esdc.portal.tables import TABLE_CONFIGS


def validate_page_size(page: object, size: object) -> tuple[int, int]:
    if isinstance(page, bool) or not isinstance(page, int):
        raise ValueError("page must be an integer")
    if isinstance(size, bool) or not isinstance(size, int):
        raise ValueError("size must be an integer")
    if page < 1:
        raise ValueError("page must be >= 1")
    if size < 1 or size > 200:
        raise ValueError("size must be between 1 and 200")
    return (page, size)


def finite_decimal(value: object) -> Decimal:
    if isinstance(value, bool):
        raise ValueError("value must not be a boolean")
    if isinstance(value, float):
        raise ValueError("value must not be a float")
    if isinstance(value, str):
        if value == "":
            raise ValueError("value must not be an empty string")
        try:
            d = Decimal(value)
        except InvalidOperation as exc:
            raise ValueError(f"invalid decimal string: {value}") from exc
    elif isinstance(value, int):
        d = Decimal(value)
    elif isinstance(value, Decimal):
        d = value
    else:
        raise ValueError("value must be str, int, or Decimal")
    if not d.is_finite():
        raise ValueError("decimal value must be finite")
    return d


def compare_decimal(left: str, right: str) -> int:
    dl = finite_decimal(left)
    dr = finite_decimal(right)
    if dl < dr:
        return -1
    if dl > dr:
        return 1
    return 0


def iso_date(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("value must be a string")
    pattern = r"^\d{4}-\d{2}-\d{2}$"
    if not re.fullmatch(pattern, value):
        raise ValueError(f"invalid ISO date format: {value}")
    try:
        date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"invalid calendar date: {value}") from exc
    return value


def literal_contains(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("value must be a string")
    escaped = value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{escaped}%"


def strict_integer(value: object) -> int:
    if isinstance(value, bool):
        raise ValueError("value must not be a boolean")
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        pattern = r"^[+-]?[0-9]+$"
        if not re.fullmatch(pattern, value):
            raise ValueError(f"invalid integer string: {value}")
        return int(value)
    raise ValueError("value must be an int or a valid integer string")


def strict_boolean(value: object) -> int:
    if isinstance(value, bool):
        return 1 if value else 0
    if isinstance(value, int):
        if value == 0:
            return 0
        if value == 1:
            return 1
        raise ValueError("integer must be 0 or 1")
    if isinstance(value, str):
        lowered = value.lower()
        if lowered in ("true", "1"):
            return 1
        if lowered in ("false", "0"):
            return 0
        raise ValueError(f"invalid boolean string: {value}")
    raise ValueError("value must be bool, int (0/1), or string")


def strict_string(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("value must be text")
    return value


def normalize_timestamp(value: object) -> str:
    """Normalize a strict ISO datetime string to UTC.

    Accepts YYYY-MM-DD[T or space]HH:MM:SS with optional .fraction (1-6 digits),
    optional Z or +HH:MM/-HH:MM offset. Naive timestamps are interpreted as UTC
    for comparison purposes only, following SQLite timestamp convention.
    Raises ValueError on invalid input, bool, non-string, empty, trailing newline,
    NaN, Infinity, unsupported trailing data, or normalization overflow.
    """
    if isinstance(value, bool) or not isinstance(value, str):
        raise ValueError("Invalid value")

    pattern = (
        r"[0-9]{4}-[0-9]{2}-[0-9]{2}[T ][0-9]{2}:[0-9]{2}:[0-9]{2}"
        r"(?:\.[0-9]{1,6})?(?:Z|[+-](?:[01][0-9]|2[0-3]):[0-5][0-9])?"
    )
    if not re.fullmatch(pattern, value):
        raise ValueError("Invalid format")

    try:
        dt = datetime.fromisoformat(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        return dt.isoformat(timespec="microseconds")
    except (ValueError, OverflowError) as exc:
        raise ValueError("Invalid ISO timestamp or UTC normalization overflow") from exc


def compare_timestamp(left: str, right: str) -> int:
    """Compare two timestamps as normalized UTC strings.

    Uses normalize_timestamp to convert inputs to UTC ISO format with microseconds.
    Returns -1 if left < right, 0 if equal, 1 if left > right.
    Equivalent offsets compare as 0. Raises ValueError on invalid values or overflow.
    """
    norm_left = normalize_timestamp(left)
    norm_right = normalize_timestamp(right)

    dt_left = datetime.fromisoformat(norm_left)
    dt_right = datetime.fromisoformat(norm_right)

    if dt_left < dt_right:
        return -1
    elif dt_left > dt_right:
        return 1
    else:
        return 0


def build_query(
    table: str,
    metadata: dict[str, Any],
    filters: list[dict[str, Any]] | None,
    sort: str | None,
    direction: str,
) -> tuple[str, str, list[Any]]:
    def q(identifier: str) -> str:
        return '"' + identifier.replace('"', '""') + '"'

    if direction not in ("asc", "desc"):
        raise ValueError("direction must be asc or desc")

    columns = metadata.get("columns", [])
    key_fields = metadata.get("key_fields", [])
    if not key_fields:
        raise ValueError("key_fields must be nonempty")

    column_map = {}
    for col in columns:
        name = col["name"]
        kind = col["kind"]
        storage_type = col["storage_type"]
        column_map[name] = {"kind": kind, "storage_type": storage_type}

    if sort is not None:
        if not isinstance(sort, str):
            raise ValueError("sort must be a field name")
        if sort == "":
            raise ValueError("sort cannot be empty string")
        if sort not in column_map:
            raise ValueError(f"unknown sort field: {sort}")

    ops_map = {"eq": "=", "ne": "!=", "gt": ">", "gte": ">=", "lt": "<", "lte": "<="}

    clauses = []
    params = []

    if filters is not None:
        if not isinstance(filters, list):
            raise ValueError("filters must be None or list")
        for f in filters:
            if not isinstance(f, dict):
                raise ValueError("each filter must be a dict")
            keys = set(f.keys())
            allowed_keys = {"field", "op", "value"}
            if not keys.issubset(allowed_keys):
                raise ValueError("filter keys must be subset of field/op/value")

            field = f.get("field")
            op = f.get("op")
            value = f.get("value")

            if not isinstance(field, str) or not isinstance(op, str):
                raise ValueError("field and op must be strings")

            if field not in column_map:
                raise ValueError(f"unknown filter field: {field}")

            col_info = column_map[field]
            kind = col_info["kind"]
            storage_type = col_info["storage_type"]

            if op == "is_null":
                val = strict_boolean(value)
                clause = f"{q(field)} IS {'NULL' if val else 'NOT NULL'}"
                clauses.append(clause)
                continue

            if op == "contains":
                if kind not in ("text", "uuid"):
                    raise ValueError("contains requires text")
                clauses.append(f"{q(field)} LIKE ? ESCAPE '\\'")
                params.append(literal_contains(value))
                continue

            if op not in ops_map:
                raise ValueError(f"unknown op: {op}")

            sql_op = ops_map[op]

            if value is None:
                if op == "eq":
                    clause = f"{q(field)} IS NULL"
                    clauses.append(clause)
                    continue
                elif op == "ne":
                    clause = f"{q(field)} IS NOT NULL"
                    clauses.append(clause)
                    continue
                else:
                    raise ValueError(
                        "comparison with None not allowed for non-eq/ne ops"
                    )

            if kind == "decimal" and storage_type.startswith("NUMERIC"):
                val = str(finite_decimal(value))
                expression = f"decimal_compare({q(field)}, ?)"
                clause = f"{expression} {sql_op} 0"
                params.append(val)
                null_guard = f"{q(field)} IS NOT NULL"
                clauses.append(null_guard)
                clauses.append(clause)
            elif kind == "integer":
                val = strict_integer(value)
                clause = f"{q(field)} {sql_op} ?"
                params.append(val)
                clauses.append(clause)
            elif kind == "boolean":
                val = strict_boolean(value)
                clause = f"{q(field)} {sql_op} ?"
                params.append(val)
                clauses.append(clause)
            elif kind == "date":
                val = iso_date(value)
                clause = f"{q(field)} {sql_op} ?"
                params.append(val)
                clauses.append(clause)
            elif kind == "datetime":
                val = normalize_timestamp(value)
                clause = f"timestamp_compare({q(field)}, ?) {sql_op} 0"
                params.append(val)
                clauses.append(clause)
            elif kind in ("text", "uuid"):
                if op == "contains":
                    val = literal_contains(value)
                    clause = f"{q(field)} LIKE ? ESCAPE '\\'"
                    params.append(val)
                    clauses.append(clause)
                else:
                    val = strict_string(value)
                    clause = f"{q(field)} {sql_op} ?"
                    params.append(val)
                    clauses.append(clause)
            elif kind == "decimal":
                if isinstance(value, bool):
                    raise ValueError("boolean not allowed for decimal")
                val = float(value)
                if not math.isfinite(val):
                    raise ValueError("non-finite decimal value")
                clause = f"{q(field)} {sql_op} ?"
                params.append(val)
                clauses.append(clause)
            else:
                raise ValueError(f"unknown kind: {kind}")

    where_clause = " WHERE " + " AND ".join(clauses) if clauses else ""

    order_fields = []
    seen = set()
    if sort is not None and sort not in seen:
        order_fields.append(sort)
        seen.add(sort)
    for kf in key_fields:
        if kf not in seen:
            order_fields.append(kf)
            seen.add(kf)

    order_parts = []
    for i, field in enumerate(order_fields):
        col_info = column_map[field]
        kind = col_info["kind"]
        storage_type = col_info["storage_type"]
        dir_str = direction if i == 0 and sort is not None else "ASC"
        if kind == "decimal" and storage_type.startswith("NUMERIC"):
            order_parts.append(f"{q(field)} COLLATE exact_decimal {dir_str}")
        elif kind == "datetime":
            order_parts.append(f"{q(field)} COLLATE utc_timestamp {dir_str}")
        else:
            order_parts.append(f"{q(field)} {dir_str}")

    order_clause = " ORDER BY " + ", ".join(order_parts)

    select_cols = ", ".join(q(col["name"]) for col in columns)
    select_query = f"SELECT {select_cols} FROM {q(table)}{where_clause}{order_clause}"
    count_query = f"SELECT COUNT(*) FROM {q(table)}{where_clause}"

    return select_query, count_query, params


def register_decimal(conn: sqlite3.Connection) -> None:
    def scalar_compare(a: Any, b: Any) -> int | None:
        if a is None or b is None:
            return None
        return compare_decimal(str(a), str(b))

    conn.create_function("decimal_compare", 2, scalar_compare)
    conn.create_collation("exact_decimal", compare_decimal)

    def scalar_timestamp(a: Any, b: Any) -> int | None:
        if a is None or b is None:
            return None
        return compare_timestamp(str(a), str(b))

    conn.create_function("timestamp_compare", 2, scalar_timestamp)
    conn.create_collation("utc_timestamp", compare_timestamp)


def serialize_row(row: sqlite3.Row, metadata: dict) -> dict:
    result = {}

    for col in metadata["columns"]:
        name = col["name"]
        kind = col["kind"]
        storage_type = col.get("storage_type", "")
        value = row[name]

        if kind == "decimal" and storage_type.startswith("NUMERIC"):
            result[name] = str(value) if value is not None else None
        elif kind == "boolean":
            result[name] = bool(value) if value is not None else None
        elif kind == "integer":
            if value is not None:
                iv = int(value)
                if abs(iv) > 2**53 - 1:
                    result[name] = str(iv)
                else:
                    result[name] = iv
            else:
                result[name] = None
        elif kind in ("date", "datetime", "text", "uuid"):
            result[name] = str(value) if value is not None else None
        elif kind == "decimal" and storage_type in (
            "REAL",
            "DOUBLE PRECISION",
            "FLOAT",
        ):
            result[name] = float(value) if value is not None else None
        else:
            result[name] = value

    result["_row_key"] = json.dumps(
        [row[key] for key in metadata["key_fields"]],
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return result


def read_page(
    table: str,
    *,
    path: Path,
    page: int = 1,
    size: int = 100,
    filters: list[dict] | None = None,
    sort: str | None = None,
    direction: str = "asc",
    configs: dict[str, Any],
    open_snapshot: Callable[[Path], sqlite3.Connection],
    validate_snapshot: Callable[[sqlite3.Connection], None],
    generation: Callable[[sqlite3.Connection], Any],
) -> dict[str, Any]:
    if not isinstance(table, str) or table not in configs:
        raise ValueError(f"table {table!r} is not allowlisted")

    validate_page_size(page, size)

    metadata = configs[table]
    select_sql, count_sql, params = build_query(
        table, metadata, filters, sort, direction
    )

    if not path.exists():
        return {
            "rows": [],
            "total": 0,
            "page": page,
            "size": size,
            "last_page": 1,
            "source_generation": None,
            "initialized": False,
        }

    if not path.is_file():
        raise ValueError(f"path {path!r} is not a file")

    conn = open_snapshot(path)
    try:
        validate_snapshot(conn)
        register_decimal(conn)

        total = conn.execute(count_sql, params).fetchone()[0]

        select_sql += " LIMIT ? OFFSET ?"
        offset = (page - 1) * size
        rows_raw = conn.execute(select_sql, params + [size, offset]).fetchall()

        source_gen = generation(conn)
    finally:
        conn.close()

    last_page = max(1, ceil(total / size)) if total else 1

    return {
        "rows": [serialize_row(r, metadata) for r in rows_raw],
        "total": total,
        "page": page,
        "size": size,
        "last_page": last_page,
        "source_generation": source_gen,
        "initialized": True,
    }


def _metadata(config: dict) -> dict:
    return {
        "columns": [
            {
                "name": c["field"],
                "kind": c["logical_type"],
                "storage_type": c["data_type"],
            }
            for c in config["columns"]
        ],
        "key_fields": config["pk"],
    }


def _validate(conn: sqlite3.Connection) -> None:
    try:
        _validate_existing_v2_registry(conn)
    except (MigrationRefusedError, sqlite3.Error) as exc:
        raise ValueError(
            "Current POD registry required; load a validated current dataset "
            f"before browsing: {exc}"
        ) from exc


def _generation(conn: sqlite3.Connection) -> int:
    row = conn.execute(
        "SELECT source_generation FROM pod_registry_state WHERE singleton_id=1"
    ).fetchone()
    if row is None:
        raise ValueError(
            "POD registry state is missing; load a validated current dataset"
        )
    return int(row[0])


def read_table_page(
    table: str,
    *,
    sqlite_path: Path | None = None,
    page: int = 1,
    size: int = 100,
    filters: list[dict] | None = None,
    sort: str | None = None,
    direction: str = "asc",
) -> dict:
    return read_page(
        table,
        path=sqlite_path if sqlite_path is not None else get_esdc_sqlite_path(),
        page=page,
        size=size,
        filters=filters,
        sort=sort,
        direction=direction,
        configs={name: _metadata(config) for name, config in TABLE_CONFIGS.items()},
        open_snapshot=read_only_connection,
        validate_snapshot=_validate,
        generation=_generation,
    )


def lookup_projects(
    *, sqlite_path: Path | None = None, q: str = "", limit: int = 20
) -> list[dict]:
    validate_page_size(1, limit)
    pattern = literal_contains(q)
    path = sqlite_path if sqlite_path is not None else get_esdc_sqlite_path()
    if not path.exists():
        return []
    conn = read_only_connection(path)
    try:
        _validate(conn)
        rows = conn.execute(
            "SELECT project_id, project_name, is_active FROM m_project "
            "WHERE project_id LIKE ? ESCAPE '\\' "
            "OR project_name LIKE ? ESCAPE '\\' ORDER BY project_id LIMIT ?",
            (pattern, pattern, limit),
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def _reference_row(row: sqlite3.Row, table: str) -> dict:
    metadata = _metadata(TABLE_CONFIGS[table])
    selected_columns = set(row.keys())
    metadata["columns"] = [
        column for column in metadata["columns"] if column["name"] in selected_columns
    ]
    result = serialize_row(row, metadata)
    result.pop("_row_key")
    return result


def read_reference_options(*, sqlite_path: Path | None = None) -> dict:
    path = sqlite_path if sqlite_path is not None else get_esdc_sqlite_path()
    result = {
        name: [] for name in ("pods", "r_institution", "r_pod_type", "r_pod_stage")
    }
    if not path.exists():
        return result
    conn = read_only_connection(path)
    try:
        _validate(conn)
        result["pods"] = [
            _reference_row(row, "m_pod")
            for row in conn.execute(
                "SELECT pod_id_seq, pod_id, pod_name, approval_date, pod_letter_num "
                "FROM m_pod ORDER BY approval_date, pod_id_seq"
            )
        ]
        for name in ("r_institution", "r_pod_type", "r_pod_stage"):
            columns = TABLE_CONFIGS[name]["columns"]
            actual = {row[1] for row in conn.execute(f'PRAGMA table_info("{name}")')}
            if actual != {column["field"] for column in columns}:
                raise ValueError(f"Reference {name} has invalid columns")
            sql, _, params = build_query(
                name, _metadata(TABLE_CONFIGS[name]), None, None, "asc"
            )
            result[name] = [
                _reference_row(row, name) for row in conn.execute(sql, params)
            ]
            if not result[name] or any(
                set(row) != {column["field"] for column in columns}
                for row in result[name]
            ):
                raise ValueError(
                    f"Reference {name} has invalid shape; "
                    "load a validated current dataset"
                )
        return result
    finally:
        conn.close()
