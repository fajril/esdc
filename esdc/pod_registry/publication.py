"""Consistent, typed publication of the canonical POD registry snapshot."""

from __future__ import annotations

import contextlib
import hashlib
import json
import re
import sqlite3
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Literal, overload
from uuid import UUID

import duckdb
import yaml

from esdc.loaders import _METADATA_TABLE, LoadResult, _create_metadata_table
from esdc.pod_registry.contract import (
    PodContract,
    TableContract,
    _validate_value,
    load_pod_contract,
)
from esdc.pod_registry.migrations import SCHEMA_VERSION, _validate_existing_v2_registry
from esdc.pod_registry.models import PreparedPodBatch, ValidationIssue
from esdc.pod_registry.ownership import (
    PUBLICATION_TABLE,
    REGISTRY_VIEW,
    RETIRED_OBJECTS,
    SUPPORT_TABLES,
)
from esdc.pod_registry.store import ESDC_SQLITE_FILENAME, get_sqlite_readonly_connection


@dataclass(frozen=True, slots=True)
class PodPublicationReport(Sequence[LoadResult]):
    """Publication outcome, with sequence behavior for older callers."""

    results: tuple[LoadResult, ...] = ()
    source_committed: bool = False
    source_generation: int | None = None
    published_generation: int | None = None
    source_sha256: str | None = None
    projection_error: str | None = None

    def __len__(self) -> int:
        """Return the number of table results."""
        return len(self.results)

    @overload
    def __getitem__(self, index: int) -> LoadResult: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[LoadResult, ...]: ...

    def __getitem__(
        self, index: int | slice
    ) -> LoadResult | tuple[LoadResult, ...]:
        """Return one result or a result slice."""
        return self.results[index]

    def __iter__(self) -> Iterator[LoadResult]:
        """Iterate over published table results."""
        return iter(self.results)


@dataclass(frozen=True, slots=True)
class PodRegistryStatus:
    """Read-only comparison of fast source markers and DuckDB publication."""

    state: Literal["fresh", "stale", "unpublished", "unavailable"]
    source_generation: int | None = None
    published_generation: int | None = None
    source_sha256: str | None = None
    published_sha256: str | None = None
    message: str | None = None


def has_v2_state(sqlite_path: Path | None = None) -> bool:
    """Return whether SQLite has the v2 source marker, without writing."""
    conn = get_sqlite_readonly_connection(sqlite_path)
    if conn is None:
        return False
    try:
        return (
            conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' "
                "AND name='pod_registry_state'"
            ).fetchone()
            is not None
        )
    finally:
        conn.close()


def _table_columns(conn: sqlite3.Connection, table: str) -> tuple[str, ...]:
    return tuple(str(row[1]) for row in conn.execute(f'PRAGMA table_info("{table}")'))


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    return (
        conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", [table]
        ).fetchone()
        is not None
    )


def _order_clause(
    conn: sqlite3.Connection, table: str, columns: tuple[str, ...]
) -> str:
    info = conn.execute(f'PRAGMA table_info("{table}")').fetchall()
    primary = tuple(
        str(row[1])
        for row in sorted(
            (row for row in info if int(row[5]) > 0),
            key=lambda row: int(row[5]),
        )
    )
    order = primary or ("rowid",)
    return ", ".join(f'"{column}"' for column in order)


def _contract_value(value: object, logical_type: str) -> object:
    """Decode SQLite text-backed logical numerics without rounding."""
    if logical_type != "decimal" or not isinstance(value, str):
        return value
    try:
        return Decimal(value)
    except InvalidOperation:
        return value


def _hash_rows(
    digest: Any,
    conn: sqlite3.Connection,
    table: str,
    columns: tuple[str, ...],
    *,
    contract: PodContract | None = None,
    table_contract: TableContract | None = None,
) -> None:
    projection = ", ".join(f'"{column}"' for column in columns)
    digest.update(
        json.dumps([table, columns], ensure_ascii=False, default=str).encode("utf-8")
        + b"\n"
    )
    cursor = conn.execute(
        f'SELECT {projection} FROM "{table}" '
        f'ORDER BY {_order_clause(conn, table, columns)}'
    )
    row_number = 0
    validation_batch = (
        None
        if contract is None or table_contract is None
        else PreparedPodBatch(
            source="SQLite POD publication",
            contract_id=contract.contract_id,
            contract_version=contract.contract_version,
            dictionary_version=contract.dictionary_version,
            headers={},
            rows={},
        )
    )
    while batch := cursor.fetchmany(2048):
        for raw_row in batch:
            row_number += 1
            row = tuple(raw_row)
            if validation_batch is not None and table_contract is not None:
                assert contract is not None
                row_data = dict(zip(columns, row, strict=True))
                issues: list[ValidationIssue] = []
                for column, value in zip(
                    table_contract.columns, row, strict=True
                ):
                    _validate_value(
                        validation_batch,
                        contract,
                        table_contract,
                        column,
                        _contract_value(value, column.logical_type),
                        row_data,
                        row_number,
                        issues,
                    )
                if issues:
                    issue = issues[0]
                    raise ValueError(
                        f"{table}.{issue.column} row {row_number}: {issue.message}"
                    )
            digest.update(
                json.dumps(row, ensure_ascii=False, default=str).encode("utf-8")
                + b"\n"
            )


def _validate_source(
    conn: sqlite3.Connection, contract: PodContract
) -> tuple[
    int,
    str,
    dict[str, tuple[tuple[str, ...], dict[str, str]]],
    str,
    tuple[Any, ...] | None,
]:
    """Validate schema/integrity and hash rows incrementally in one read view."""
    integrity = [
        tuple(row) for row in conn.execute("PRAGMA integrity_check").fetchall()
    ]
    if integrity != [("ok",)]:
        raise ValueError(f"SQLite integrity check failed: {integrity[:5]!r}")
    foreign_key_errors = conn.execute("PRAGMA foreign_key_check").fetchall()
    if foreign_key_errors:
        raise ValueError(f"SQLite foreign-key check failed: {foreign_key_errors[:5]!r}")

    expected_tables = (*contract.tables, *SUPPORT_TABLES)
    missing = [table for table in expected_tables if not _table_exists(conn, table)]
    if missing:
        raise ValueError(f"v2 SQLite source is missing tables: {', '.join(missing)}")

    state = conn.execute(
        "SELECT schema_version, contract_id, contract_version, dictionary_version, "
        "source_generation, updated_at FROM pod_registry_state WHERE singleton_id=1"
    ).fetchone()
    if state is None:
        raise ValueError("v2 SQLite source has no singleton registry state")
    (
        schema_version,
        contract_id,
        contract_version,
        dictionary_version,
        generation,
        updated_at,
    ) = state
    if schema_version != SCHEMA_VERSION:
        raise ValueError(f"unsupported POD registry schema version: {schema_version}")
    if (
        contract_id != contract.contract_id
        or contract_version != contract.contract_version
        or dictionary_version != contract.dictionary_version
    ):
        raise ValueError(
            "POD registry contract or dictionary marker does not match "
            "the packaged contract"
        )

    # One shared storage policy: the same structural validator used by the
    # operational writer boundary must also gate the canonical publisher, so a
    # source whose declared physical types drifted from the contract is refused
    # before row hashing, the DuckDB target is opened, or anything is written.
    _validate_existing_v2_registry(conn)

    source_tables: dict[str, tuple[tuple[str, ...], dict[str, str]]] = {}
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            [schema_version, contract_id, contract_version, dictionary_version,
             int(generation), str(updated_at)],
            ensure_ascii=False,
            default=str,
        ).encode("utf-8")
        + b"\n"
    )
    for table_name, table_contract in contract.tables.items():
        columns = tuple(column.name for column in table_contract.columns)
        actual_columns = _table_columns(conn, table_name)
        if actual_columns != columns:
            raise ValueError(
                f"SQLite {table_name} columns do not match "
                f"the v{contract_version} contract"
            )
        source_tables[table_name] = (columns, {})
        _hash_rows(
            digest,
            conn,
            table_name,
            columns,
            contract=contract,
            table_contract=table_contract,
        )

    for table_name in SUPPORT_TABLES:
        info = conn.execute(f'PRAGMA table_info("{table_name}")').fetchall()
        columns = tuple(str(row[1]) for row in info)
        declared_types: dict[str, str] = {
            str(row[1]): str(row[2] or "") for row in info
        }
        source_tables[table_name] = (columns, declared_types)
        _hash_rows(digest, conn, table_name, columns)

    read_model_marker = conn.execute(
        "SELECT source_generation, resource_revision, resource_sha256, "
        "timeseries_revision, timeseries_sha256, updated_at "
        "FROM pod_read_model_snapshot WHERE singleton_id=1"
    ).fetchone()
    return (
        int(generation),
        digest.hexdigest(),
        source_tables,
        str(updated_at),
        None if read_model_marker is None else tuple(read_model_marker),
    )

def _duck_type(logical_type: str, precision: int | None, scale: int | None) -> str:
    if logical_type == "text":
        return "VARCHAR"
    if logical_type == "integer":
        return "BIGINT"
    if logical_type == "decimal":
        if precision is None:
            raise ValueError("decimal contract column has no precision")
        if scale is None:
            return "DOUBLE"
        return f"DECIMAL({precision}, {scale})"
    if logical_type == "boolean":
        return "BOOLEAN"
    if logical_type == "date":
        return "DATE"
    if logical_type == "datetime":
        return "TIMESTAMP"
    if logical_type == "uuid":
        return "UUID"
    raise ValueError(f"unsupported POD logical type {logical_type!r}")


def _special_duck_type(sqlite_type: str) -> str:
    normalized = sqlite_type.upper().strip()
    decimal_match = re.fullmatch(
        r"(?:DECIMAL|NUMERIC)\s*\((\d+)\s*,\s*(\d+)\)", normalized
    )
    if decimal_match:
        return f"DECIMAL({decimal_match.group(1)}, {decimal_match.group(2)})"
    if "INT" in normalized:
        return "BIGINT"
    if "BOOL" in normalized:
        return "BOOLEAN"
    if "DATE" in normalized and "TIME" not in normalized:
        return "DATE"
    if "TIME" in normalized:
        return "TIMESTAMP"
    if any(token in normalized for token in ("REAL", "FLOAT", "DOUBLE")):
        return "DOUBLE"
    if "BLOB" in normalized:
        return "BLOB"
    return "VARCHAR"


def _special_logical_type(sqlite_type: str) -> str | None:
    normalized = sqlite_type.upper()
    if "INT" in normalized:
        return "integer"
    if "BOOL" in normalized:
        return "boolean"
    if "DATE" in normalized and "TIME" not in normalized:
        return "date"
    if "TIME" in normalized:
        return "datetime"
    if any(token in normalized for token in ("REAL", "FLOAT", "DOUBLE")):
        return "float"
    if "DECIMAL" in normalized or "NUMERIC" in normalized:
        return "decimal"
    if "BLOB" in normalized:
        return "blob"
    return "text"


def _typed_value(value: Any, logical_type: str | None) -> Any:
    if value is None or logical_type is None:
        return value
    if logical_type == "text":
        return str(value)
    if logical_type == "integer":
        return int(value)
    if logical_type == "decimal":
        return Decimal(str(value))
    if logical_type == "boolean":
        if isinstance(value, str):
            if value not in ("0", "1"):
                raise ValueError(f"invalid boolean source value {value!r}")
            return value == "1"
        return bool(value)
    if logical_type == "date":
        return value if isinstance(value, date) else date.fromisoformat(str(value))
    if logical_type == "datetime":
        if isinstance(value, datetime):
            return value
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).replace(
            tzinfo=None
        )
    if logical_type == "uuid":
        return value if isinstance(value, UUID) else UUID(str(value))
    if logical_type == "float":
        return float(value)
    if logical_type in {"blob", "decimal"}:
        return value if logical_type == "blob" else Decimal(str(value))
    return value


def _drop_object(conn: duckdb.DuckDBPyConnection, name: str) -> None:
    row = conn.execute(
        "SELECT table_type FROM information_schema.tables "
        "WHERE table_schema='main' AND table_name=?",
        [name],
    ).fetchone()
    if row is not None:
        kind = "VIEW" if row[0] == "VIEW" else "TABLE"
        conn.execute(f'DROP {kind} IF EXISTS "{name}"')


def _create_table_from_sqlite(
    target: duckdb.DuckDBPyConnection,
    source: sqlite3.Connection,
    table: str,
    columns: tuple[str, ...],
    logical_types: dict[str, str | None],
    declared_types: dict[str, str],
    contract_columns: tuple[Any, ...] | None = None,
) -> int:
    if contract_columns is None:
        definitions = [
            f'"{column}" {_special_duck_type(declared_types.get(column, ""))}'
            for column in columns
        ]
    else:
        definitions = [
            f'"{column.name}" '
            f'{_duck_type(column.logical_type, column.precision, column.scale)}'
            for column in contract_columns
        ]
    target.execute(f'CREATE TABLE "{table}" ({", ".join(definitions)})')
    projection = ", ".join(f'"{column}"' for column in columns)
    cursor = source.execute(
        f'SELECT {projection} FROM "{table}" '
        f'ORDER BY {_order_clause(source, table, columns)}'
    )
    placeholders = ", ".join("?" for _ in columns)
    column_sql = ", ".join(f'"{column}"' for column in columns)
    insert_sql = f'INSERT INTO "{table}" ({column_sql}) VALUES ({placeholders})'
    count = 0
    while batch := cursor.fetchmany(2048):
        typed_rows = [
            tuple(
                _typed_value(value, logical_types.get(column))
                for column, value in zip(columns, row, strict=True)
            )
            for row in batch
        ]
        target.executemany(insert_sql, typed_rows)
        count += len(batch)
    return count


def _table_metadata(
    table_name: str,
    description: str,
    columns: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "table_name": table_name,
        "description": description,
        "sheet_name": ESDC_SQLITE_FILENAME,
        "columns": columns,
    }


def _write_metadata(
    conn: duckdb.DuckDBPyConnection,
    table_name: str,
    entry: dict[str, Any],
) -> None:
    loaded_at = datetime.now(timezone.utc).isoformat()
    conn.execute(f"DELETE FROM {_METADATA_TABLE} WHERE table_name=?", [table_name])
    conn.execute(
        f"INSERT INTO {_METADATA_TABLE} "
        "(table_name, description, sheet_name, schema_yaml, loaded_at) "
        "VALUES (?, ?, ?, ?, ?)",
        [
            table_name,
            entry["description"],
            entry["sheet_name"],
            yaml.safe_dump(entry, sort_keys=False, allow_unicode=True),
            loaded_at,
        ],
    )


def _publish_snapshot(
    conn: duckdb.DuckDBPyConnection,
    source: sqlite3.Connection,
    contract: PodContract,
    generation: int,
    source_sha256: str,
    source_tables: dict[str, tuple[tuple[str, ...], dict[str, str]]],
    source_updated_at: str,
    read_model_marker: tuple[Any, ...] | None,
    db_path: Path,
) -> tuple[LoadResult, ...]:
    conn.execute("BEGIN TRANSACTION")
    try:
        for name in (
            *RETIRED_OBJECTS,
            PUBLICATION_TABLE,
            REGISTRY_VIEW,
            *source_tables,
        ):
            _drop_object(conn, name)
        _create_metadata_table(conn)
        stale_metadata = (
            *RETIRED_OBJECTS,
            "pod_registry",
            *source_tables,
        )
        if stale_metadata:
            placeholders = ", ".join("?" for _ in stale_metadata)
            conn.execute(
                f"DELETE FROM {_METADATA_TABLE} WHERE table_name IN ({placeholders})",
                list(stale_metadata),
            )

        results: list[LoadResult] = []
        metadata_entries: dict[str, dict[str, Any]] = {}
        for table_name, table_contract in contract.tables.items():
            columns, declared_types = source_tables[table_name]
            contract_logical_types: dict[str, str | None] = {
                column.name: (
                    "float"
                    if column.logical_type == "decimal"
                    and column.scale is None
                    and "DOUBLE" in column.data_type.upper()
                    else column.logical_type
                )
                for column in table_contract.columns
            }
            count = _create_table_from_sqlite(
                conn,
                source,
                table_name,
                columns,
                contract_logical_types,
                declared_types,
                table_contract.columns,
            )
            schema_columns = [
                {
                    "name": column.name,
                    "formal_name": column.formal_name,
                    "data_type": column.data_type,
                    "logical_type": column.logical_type,
                    "unit": column.unit,
                    "nullable": column.nullable,
                    "description": column.description,
                }
                for column in table_contract.columns
            ]
            entry = _table_metadata(
                table_name,
                f"Canonical POD v2 source table ({table_contract.role}).",
                schema_columns,
            )
            metadata_entries[table_name] = entry
            results.append(
                LoadResult(
                    table_name=table_name,
                    row_count=count,
                    column_count=len(columns),
                    db_path=db_path,
                    link_warnings=(),
                )
            )

        for table_name in SUPPORT_TABLES:
            columns, declared_types = source_tables[table_name]
            support_logical_types: dict[str, str | None]
            if table_name == "pod_document":
                support_logical_types = {
                    "pod_id_seq": "integer",
                    "doc_id": "text",
                }
            else:
                support_logical_types = {
                    column: _special_logical_type(declared_types[column])
                    for column in columns
                }
            count = _create_table_from_sqlite(
                conn,
                source,
                table_name,
                columns,
                support_logical_types,
                declared_types,
            )
            entry = _table_metadata(
                table_name,
                "Technical POD registry state and read-model diagnostics.",
                [
                    {
                        "name": column,
                        "data_type": declared_types.get(column) or "TEXT",
                        "description": None,
                    }
                    for column in columns
                ],
            )
            metadata_entries[table_name] = entry
            results.append(
                LoadResult(
                    table_name=table_name,
                    row_count=count,
                    column_count=len(columns),
                    db_path=db_path,
                    link_warnings=(),
                )
            )

        conn.execute('CREATE VIEW "pod_registry" AS SELECT * FROM "m_pod"')
        metadata_entries["pod_registry"] = _table_metadata(
            "pod_registry",
            "Compatibility view of the canonical m_pod registry.",
            metadata_entries["m_pod"]["columns"],
        )
        for table_name, entry in metadata_entries.items():
            _write_metadata(conn, table_name, entry)

        conn.execute(
            f"CREATE TABLE {PUBLICATION_TABLE} ("
            "source_generation BIGINT NOT NULL, source_sha256 VARCHAR NOT NULL, "
            "contract_id VARCHAR NOT NULL, contract_version INTEGER NOT NULL, "
            "dictionary_version VARCHAR NOT NULL, source_updated_at VARCHAR NOT NULL, "
            "read_model_source_generation BIGINT, resource_revision VARCHAR, "
            "resource_sha256 VARCHAR, timeseries_revision VARCHAR, "
            "timeseries_sha256 VARCHAR, read_model_updated_at VARCHAR, "
            "published_at TIMESTAMP NOT NULL DEFAULT current_timestamp)"
        )
        conn.execute(
            f"INSERT INTO {PUBLICATION_TABLE} "
            "(source_generation, source_sha256, contract_id, contract_version, "
            "dictionary_version, source_updated_at, read_model_source_generation, "
            "resource_revision, resource_sha256, timeseries_revision, "
            "timeseries_sha256, read_model_updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                generation,
                source_sha256,
                contract.contract_id,
                contract.contract_version,
                contract.dictionary_version,
                source_updated_at,
                *(read_model_marker or (None, None, None, None, None, None)),
            ],
        )
        conn.execute("COMMIT")
        return tuple(results)
    except Exception:
        with contextlib.suppress(duckdb.Error):
            conn.execute("ROLLBACK")
        raise

def publish_v2_registry(
    sqlite_path: Path,
    db_path: Path,
    *,
    sqlite_conn: sqlite3.Connection | None = None,
    duck_conn: duckdb.DuckDBPyConnection | None = None,
) -> PodPublicationReport:
    """Publish one validated v2 SQLite snapshot in a single DuckDB transaction."""
    owned_sqlite = sqlite_conn is None
    source = (
        sqlite_conn
        if sqlite_conn is not None
        else get_sqlite_readonly_connection(sqlite_path)
    )
    if source is None:
        return PodPublicationReport(projection_error="SQLite source is unavailable")
    if not owned_sqlite and source.in_transaction:
        return PodPublicationReport(
            projection_error="caller SQLite connection has an active transaction"
        )

    source_transaction_started = False
    own_duck = duck_conn is None
    target = duck_conn
    generation: int | None = None
    source_hash: str | None = None
    source_updated_at: str | None = None
    contract: PodContract | None = None
    try:
        source.execute("BEGIN")
        source_transaction_started = True
        try:
            contract = load_pod_contract()
            (
                generation,
                source_hash,
                source_tables,
                source_updated_at,
                read_model_marker,
            ) = _validate_source(source, contract)
        except Exception as exc:
            return PodPublicationReport(
                source_committed=False,
                projection_error=f"source preflight failed: {exc}",
            )

        try:
            if own_duck:
                db_path.parent.mkdir(parents=True, exist_ok=True)
                target = duckdb.connect(
                    str(db_path), config={"autoinstall_known_extensions": "false"}
                )
            assert target is not None and contract is not None
            results = _publish_snapshot(
                target,
                source,
                contract,
                generation,
                source_hash,
                source_tables,
                source_updated_at,
                read_model_marker,
                db_path,
            )
        except Exception as exc:
            return PodPublicationReport(
                source_committed=True,
                source_generation=generation,
                source_sha256=source_hash,
                projection_error=f"DuckDB publication failed: {exc}",
            )
        return PodPublicationReport(
            results=results,
            source_committed=True,
            source_generation=generation,
            published_generation=generation,
            source_sha256=source_hash,
        )
    except Exception as exc:
        return PodPublicationReport(
            source_committed=generation is not None,
            source_generation=generation,
            source_sha256=source_hash,
            projection_error=f"publication setup failed: {exc}",
        )
    finally:
        if own_duck and target is not None:
            target.close()
        if source_transaction_started and source.in_transaction:
            with contextlib.suppress(sqlite3.Error):
                source.execute("ROLLBACK")
        if owned_sqlite:
            source.close()

def get_pod_registry_status(
    sqlite_path: Path,
    db_path: Path,
) -> PodRegistryStatus:
    """Read fast source and publication markers without scanning source rows."""
    source = get_sqlite_readonly_connection(sqlite_path)
    if source is None:
        return PodRegistryStatus("unavailable", message="SQLite source is unavailable")
    try:
        state_exists = _table_exists(source, "pod_registry_state")
        if not state_exists:
            return PodRegistryStatus(
                "unavailable", message="SQLite source has no v2 registry state"
            )
        marker = source.execute(
            "SELECT schema_version, contract_id, contract_version, dictionary_version, "
            "source_generation, updated_at FROM pod_registry_state WHERE singleton_id=1"
        ).fetchone()
        if marker is None:
            return PodRegistryStatus(
                "unavailable", message="SQLite registry marker is missing"
            )
        contract = load_pod_contract()
        (
            schema_version,
            contract_id,
            contract_version,
            dictionary_version,
            generation,
            updated_at,
        ) = marker
        if (
            schema_version != SCHEMA_VERSION
            or contract_id != contract.contract_id
            or contract_version != contract.contract_version
            or dictionary_version != contract.dictionary_version
        ):
            return PodRegistryStatus(
                "unavailable",
                message="SQLite registry marker does not match packaged contract",
            )
        generation = int(generation)
        source_updated_at = str(updated_at)
        read_model_marker = source.execute(
            "SELECT source_generation, resource_revision, resource_sha256, "
            "timeseries_revision, timeseries_sha256, updated_at "
            "FROM pod_read_model_snapshot WHERE singleton_id=1"
        ).fetchone()
        read_model_marker = (
            None if read_model_marker is None else tuple(read_model_marker)
        )
    except Exception as exc:
        return PodRegistryStatus("unavailable", message=f"SQLite marker invalid: {exc}")
    finally:
        source.close()

    try:
        target = duckdb.connect(
            str(db_path), read_only=True,
            config={"autoinstall_known_extensions": "false"},
        )
    except duckdb.Error:
        return PodRegistryStatus("unpublished", source_generation=generation)
    try:
        row = target.execute(
            f"SELECT source_generation, source_sha256, contract_id, "
            f"contract_version, dictionary_version, source_updated_at, "
            f"read_model_source_generation, resource_revision, resource_sha256, "
            f"timeseries_revision, timeseries_sha256, read_model_updated_at "
            f"FROM {PUBLICATION_TABLE}"
        ).fetchone()
    except duckdb.Error:
        row = None
    finally:
        target.close()
    if row is None:
        return PodRegistryStatus("unpublished", source_generation=generation)
    (
        published_generation,
        published_hash,
        pub_contract,
        pub_version,
        pub_dictionary,
        pub_updated_at,
        pub_read_generation,
        pub_resource_revision,
        pub_resource_sha256,
        pub_timeseries_revision,
        pub_timeseries_sha256,
        pub_read_updated_at,
    ) = row
    expected_read_model_marker = read_model_marker or (
        None,
        None,
        None,
        None,
        None,
        None,
    )
    published_read_model_marker = (
        pub_read_generation,
        pub_resource_revision,
        pub_resource_sha256,
        pub_timeseries_revision,
        pub_timeseries_sha256,
        pub_read_updated_at,
    )
    fresh = (
        int(published_generation) == generation
        and str(pub_contract) == contract.contract_id
        and int(pub_version) == contract.contract_version
        and str(pub_dictionary) == contract.dictionary_version
        and str(pub_updated_at) == source_updated_at
        and published_read_model_marker == expected_read_model_marker
    )
    return PodRegistryStatus(
        "fresh" if fresh else "stale",
        source_generation=generation,
        published_generation=int(published_generation),
        published_sha256=str(published_hash),
        message=(
            None
            if fresh
            else "published generation or read-model source markers differ from SQLite"
        ),
    )

def read_pod_read_model_diagnostics(
    sqlite_path: Path, *, pod_id_seqs: Sequence[int] | None = None
) -> tuple[dict[str, Any], ...]:
    """Read only diagnostics tied to the current source generation and snapshot."""
    conn = get_sqlite_readonly_connection(sqlite_path)
    if conn is None:
        return ()
    try:
        conn.execute("BEGIN")
        state = conn.execute(
            "SELECT source_generation FROM pod_registry_state WHERE singleton_id=1"
        ).fetchone()
        if state is None:
            return ()
        generation = int(state[0])
        snapshot = conn.execute(
            "SELECT source_generation FROM pod_read_model_snapshot WHERE singleton_id=1"
        ).fetchone()
        if snapshot is None or int(snapshot[0]) != generation:
            return ()
        params: list[Any] = [generation]
        scope_sql = ""
        if pod_id_seqs:
            placeholders = ", ".join("?" for _ in pod_id_seqs)
            scope_sql = f" AND (pod_id_seq IS NULL OR pod_id_seq IN ({placeholders}))"
            params.extend(int(value) for value in pod_id_seqs)
        rows = conn.execute(
            "SELECT source_generation, resource_revision, resource_sha256, "
            "timeseries_revision, timeseries_sha256, code, message, pod_id_seq, "
            "reported_date, report_year, year_frcast, project_id, component "
            "FROM pod_read_model_diagnostic WHERE source_generation=?"
            + scope_sql
            + " ORDER BY diagnostic_id",
            params,
        ).fetchall()
        keys = (
            "source_generation",
            "resource_revision",
            "resource_sha256",
            "timeseries_revision",
            "timeseries_sha256",
            "code",
            "message",
            "pod_id_seq",
            "reported_date",
            "report_year",
            "year_frcast",
            "project_id",
            "component",
        )
        return tuple(dict(zip(keys, row, strict=True)) for row in rows)
    finally:
        conn.close()
