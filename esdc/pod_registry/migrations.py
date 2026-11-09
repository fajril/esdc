"""Explicit, transactional creation and migration of the POD registry v2 schema."""

from __future__ import annotations

import importlib.resources
import re
import sqlite3
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal
from uuid import UUID

from esdc.pod_registry.contract import (
    ColumnContract,
    load_pod_contract,
    validate_pod_batch,
)
from esdc.pod_registry.models import PreparedPodBatch, PreparedScalar
from esdc.sqlite_runtime import configure_writer, require_wal_runtime

SCHEMA_VERSION = 2
REFERENCE_SEED_VERSION = 20261028
MIGRATION_ACTOR = "FA:migrate-pod-registry-v2"
INITIALIZE_ACTOR = "FA:pod-registry-initialize"
ECONOMICS_UPGRADE_RECOVERY = (
    "Run 'esdc pod-upgrade-economics --database <path>' "
    "to inspect the required upgrade."
)

_LEGACY_OBJECTS = (
    "m_pod",
    "project_pod",
    "pod_document",
    "pod_revision",
    "r_institution",
    "r_pod_type",
)
_KNOWN_LEGACY_CHILDREN = {
    "project_pod": {("pod_id", "id")},
    "pod_document": {("pod_id", "id")},
    "pod_revision": {
        ("successor_id", "pod_id"),
        ("predecessor_id", "pod_id"),
    },
}
_V2_TABLES = {
    "m_project",
    "m_pod",
    "pod_project",
    "pod_monitoring",
    "pod_forecast",
    "pod_monitoring_api",
    "pod_forecast_api",
    "r_institution",
    "r_pod_type",
    "r_pod_stage",
    "pod_document",
    "pod_registry_state",
    "pod_read_model_snapshot",
    "pod_read_model_diagnostic",
    "pod_knowledge_state",
    "_pod_identity_ledger",
    "pod_registry_legacy_identity",
}
_DEPENDENCY_NAME = re.compile(
    r"\b(?:m_pod|project_pod|pod_document|pod_revision)\b", re.I
)
# Names that may only be occupied by the current physical registry objects. An
# unrelated application table or view is never a POD object; a view that takes one of
# these names is a partial footprint, not the canonical table.
_POD_RESERVED_NAMES = frozenset(_LEGACY_OBJECTS) | frozenset(_V2_TABLES)


class MigrationRefusedError(RuntimeError):
    """Raised when inspection finds an unsafe or unsupported schema boundary."""


@dataclass(frozen=True, slots=True)
class LegacyIdentityMapping:
    """Audit-only mapping from a legacy row identity to the current master."""

    legacy_pod_id: int
    legacy_pod_id_value: str
    pod_id_seq: int
    pod_id: str
    pod_uuid: str


@dataclass(frozen=True, slots=True)
class MigrationReport:
    """Read-only inspection or completed migration evidence."""

    status: Literal["fresh", "legacy", "v2", "blocked"]
    blockers: tuple[str, ...] = ()
    legacy_tables: tuple[str, ...] = ()
    source_generation: int | None = None
    knowledge_generation: int | None = None
    document_links_preserved: int = 0
    documents_table_present: bool = False
    identity_reconciliation: tuple[LegacyIdentityMapping, ...] = ()
    unmapped_legacy_pod_ids: tuple[int, ...] = ()
    row_counts: tuple[tuple[str, int], ...] = ()


@dataclass(frozen=True, slots=True)
class _IdentityLedgerReservation:
    approval_year: int
    approval_date: str
    daily_slot: int
    pod_order_no: int
    pod_id_seq: int
    pod_id: str
    pod_uuid: str
    institution_id: int
    pod_type_id: int
    rev_num: int
    actor: str
    issued_at: str | None


@contextmanager
def _atomic_writer(conn: sqlite3.Connection) -> Iterator[None]:
    owns_transaction = not conn.in_transaction
    savepoint = "pod_registry_v2_operation"
    if owns_transaction:
        conn.execute("BEGIN IMMEDIATE")
    else:
        conn.execute(f"SAVEPOINT {savepoint}")
    try:
        yield
        if owns_transaction:
            conn.commit()
        else:
            conn.execute(f"RELEASE {savepoint}")
    except BaseException:
        if owns_transaction:
            conn.rollback()
        else:
            conn.execute(f"ROLLBACK TO {savepoint}")
            conn.execute(f"RELEASE {savepoint}")
        raise


def _quote(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _table_names(conn: sqlite3.Connection) -> set[str]:
    return {
        str(row[0])
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' "
            "AND name NOT LIKE 'sqlite_%'"
        )
    }


def _reserved_pod_views(conn: sqlite3.Connection) -> set[str]:
    """View names that occupy a reserved POD registry slot, without writing.

    SQLite object names are compared case-insensitively, so a view spelled ``M_POD``
    still occupies the canonical ``m_pod`` slot.
    """
    return {
        str(row[0])
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'view'")
        if str(row[0]).casefold() in _POD_RESERVED_NAMES
    }


def _column_names(conn: sqlite3.Connection, table: str) -> tuple[str, ...]:
    return tuple(
        str(row[1]) for row in conn.execute(f"PRAGMA table_info({_quote(table)})")
    )


def _primary_key(conn: sqlite3.Connection, table: str) -> tuple[str, ...]:
    rows = conn.execute(f"PRAGMA table_info({_quote(table)})").fetchall()
    ordered = sorted((row for row in rows if row[5]), key=lambda row: row[5])
    return tuple(str(row[1]) for row in ordered)


def _ensure_registry_writer(conn: sqlite3.Connection) -> None:
    """Apply shared WAL policy to files, or verify it before caller-owned work."""
    main_database = next(
        (row for row in conn.execute("PRAGMA database_list") if row[1] == "main"),
        None,
    )
    if main_database is None or not str(main_database[2]):
        return

    if not conn.in_transaction:
        configure_writer(conn)
        return

    require_wal_runtime()
    journal_mode = conn.execute("PRAGMA journal_mode").fetchone()
    foreign_keys = conn.execute("PRAGMA foreign_keys").fetchone()
    busy_timeout = conn.execute("PRAGMA busy_timeout").fetchone()
    synchronous = conn.execute("PRAGMA synchronous").fetchone()
    if (
        journal_mode is None
        or str(journal_mode[0]).lower() != "wal"
        or foreign_keys is None
        or foreign_keys[0] != 1
        or busy_timeout is None
        or busy_timeout[0] != 5000
        or synchronous is None
        or synchronous[0] != 2
    ):
        raise RuntimeError(
            "Configure file-backed registry connections for WAL before BEGIN."
        )


def _execute_sql_script(conn: sqlite3.Connection, script: str) -> None:
    """Execute one complete SQL statement at a time without implicit commits."""
    statement = ""
    for line in script.splitlines(keepends=True):
        statement += line
        if sqlite3.complete_statement(statement):
            sql = statement.strip()
            if sql:
                conn.execute(sql)
            statement = ""
    if statement.strip():
        raise ValueError("SQL resource ends with an incomplete statement")


def _canonical_storage_type(column: ColumnContract) -> str:
    """Declared SQLite storage type required by the canonical contract.

    Fixed-scale decimals store as TEXT so exact decimal text keeps its precision;
    DOUBLE PRECISION stores as REAL; integers and booleans store as INTEGER; text,
    UUID, date and datetime storage stays TEXT.
    """
    if column.data_type == "DOUBLE PRECISION":
        return "REAL"
    if column.data_type in {"INTEGER", "BIGINT", "SMALLINT", "BOOLEAN"}:
        return "INTEGER"
    return "TEXT"


def _verify_v2_contract(
    conn: sqlite3.Connection, *, legacy_economics: bool = False
) -> None:
    contract = load_pod_contract()
    legacy_map: dict[tuple[str, str], str] = {}
    if legacy_economics:
        from esdc.pod_registry.economics_upgrade import ECONOMICS_RENAMES

        legacy_map = {(owner, new): old for owner, old, new in ECONOMICS_RENAMES}
    for name, table in contract.tables.items():
        rows = conn.execute(f"PRAGMA table_info({_quote(name)})").fetchall()
        expected_columns = tuple(
            legacy_map.get((name, column.name), column.name)
            for column in table.columns
        )
        if tuple(str(row[1]) for row in rows) != expected_columns:
            raise MigrationRefusedError(
                f"v2 table {name!r} does not match the ordered POD contract columns"
            )
        for column, row in zip(table.columns, rows, strict=True):
            actual = str(row[2]).upper()
            expected = _canonical_storage_type(column)
            if actual != expected:
                raise MigrationRefusedError(
                    f"v2 table {name!r} column {column.name!r} declares {actual!r} "
                    f"storage; the canonical contract storage type is {expected!r}"
                )
    expected_keys = {
        "m_project": ("project_id",),
        "m_pod": ("pod_id_seq",),
        "pod_project": ("id",),
        "pod_monitoring": ("monpod_id",),
        "pod_forecast": ("frcast_id",),
        "pod_monitoring_api": ("pod_id_seq", "reported_date"),
        "pod_forecast_api": ("pod_id_seq", "reported_date", "year_frcast"),
    }
    for name, expected in expected_keys.items():
        if _primary_key(conn, name) != expected:
            raise MigrationRefusedError(
                f"v2 table {name!r} has an unexpected physical key"
            )
    expected_triggers = {
        "trg_pod_identity_ledger_no_update": "_pod_identity_ledger",
        "trg_pod_identity_ledger_no_delete": "_pod_identity_ledger",
        "trg_pod_identity_ledger_no_conflicting_insert": "_pod_identity_ledger",
        "trg_m_pod_ledger_binding_insert": "m_pod",
        "trg_m_pod_identity_immutable": "m_pod",
        "trg_m_pod_supersession_guard_insert": "m_pod",
        "trg_m_pod_supersession_guard_update": "m_pod",
        "trg_pod_project_reference_insert": "pod_project",
        "trg_pod_project_reference_update": "pod_project",
        "trg_m_pod_pod_project_no_delete": "m_pod",
        "trg_m_project_pod_project_no_delete": "m_project",
        "trg_m_pod_pod_project_no_key_update": "m_pod",
        "trg_m_project_pod_project_no_key_update": "m_project",
    }
    actual_triggers = {
        str(row[0]): str(row[1])
        for row in conn.execute(
            "SELECT name, tbl_name FROM sqlite_master WHERE type = 'trigger'"
        )
    }
    if any(
        actual_triggers.get(name) != table for name, table in expected_triggers.items()
    ):
        raise MigrationRefusedError("v2 registry is missing a required guard trigger")


def open_registry_v2_writer(path: Path) -> sqlite3.Connection:
    """Open an existing v2 registry with writer policy; never initialize schema."""
    if not path.is_file():
        raise FileNotFoundError(f"v2 POD registry does not exist: {path}")
    require_wal_runtime()
    uri = f"{path.resolve().as_uri()}?mode=ro"
    probe = sqlite3.connect(uri, uri=True)
    try:
        _validate_existing_v2_registry(probe)
    finally:
        probe.close()

    conn = sqlite3.connect(path)
    try:
        _ensure_registry_writer(conn)
        _validate_existing_v2_registry(conn)
        conn.row_factory = sqlite3.Row
        return conn
    except BaseException:
        conn.close()
        raise


def _validate_existing_v2_registry(conn: sqlite3.Connection) -> None:
    names = _table_names(conn)
    if "pod_registry_state" not in names:
        raise MigrationRefusedError("database is not an initialized v2 POD registry")
    missing = sorted(_V2_TABLES - names)
    if missing:
        raise MigrationRefusedError(
            "v2 POD registry is missing required tables: " + ", ".join(missing)
        )
    state = conn.execute(
        "SELECT schema_version, contract_id, contract_version, dictionary_version "
        "FROM pod_registry_state WHERE singleton_id=1"
    ).fetchone()
    contract = load_pod_contract()
    expected = (
        SCHEMA_VERSION,
        contract.contract_id,
        contract.contract_version,
        contract.dictionary_version,
    )
    if state is not None and tuple(state) == expected:
        _verify_v2_contract(conn)
        return
    if (
        state is not None
        and int(state[0]) == SCHEMA_VERSION
        and str(state[1]) == contract.contract_id
        and str(state[3]) == contract.dictionary_version
    ):
        from esdc.pod_registry.economics_upgrade import inspect_economics_upgrade

        report = inspect_economics_upgrade(conn)
        if report.status == "upgrade_required":
            raise MigrationRefusedError(ECONOMICS_UPGRADE_RECOVERY)
    raise MigrationRefusedError("existing POD registry metadata is incompatible")


def initialize_registry_v2(conn: sqlite3.Connection) -> None:
    """Create the v2 schema explicitly; this never migrates a legacy registry."""
    _ensure_registry_writer(conn)
    if conn.execute("PRAGMA foreign_keys").fetchone()[0] != 1:
        raise RuntimeError("POD registry initialization requires foreign_keys=ON")
    names = _table_names(conn)
    contract = load_pod_contract()
    expected = (
        SCHEMA_VERSION,
        contract.contract_id,
        contract.contract_version,
        contract.dictionary_version,
    )
    state_exists = "pod_registry_state" in names
    if state_exists:
        state = conn.execute(
            "SELECT schema_version, contract_id, contract_version, dictionary_version "
            "FROM pod_registry_state WHERE singleton_id = 1"
        ).fetchone()
        if state is not None and tuple(state) != expected:
            raise MigrationRefusedError(
                "existing POD registry metadata is incompatible"
            )
    elif names & _V2_TABLES:
        raise MigrationRefusedError(
            "POD tables already exist without v2 metadata; use inspect_migration"
        )

    with _atomic_writer(conn):
        conn.execute("PRAGMA legacy_alter_table = OFF")
        schema = (
            importlib.resources.files("esdc")
            .joinpath("sql")
            .joinpath("pod_registry_v2.sql")
            .read_text(encoding="utf-8")
        )
        seed = (
            importlib.resources.files("esdc")
            .joinpath("sql")
            .joinpath("pod_reference_seed.sql")
            .read_text(encoding="utf-8")
        )
        _execute_sql_script(conn, schema)
        _execute_sql_script(conn, seed)
        conn.execute(
            "INSERT OR IGNORE INTO pod_registry_state ("
            "singleton_id, schema_version, contract_id, contract_version, "
            "dictionary_version, reference_seed_version, source_generation, "
            "knowledge_generation, updated_by"
            ") VALUES (1, ?, ?, ?, ?, ?, 0, NULL, ?)",
            (
                SCHEMA_VERSION,
                contract.contract_id,
                contract.contract_version,
                contract.dictionary_version,
                REFERENCE_SEED_VERSION,
                INITIALIZE_ACTOR,
            ),
        )
        conn.execute(
            "UPDATE pod_registry_state SET reference_seed_version = ? "
            "WHERE singleton_id = 1 AND reference_seed_version <> ?",
            (REFERENCE_SEED_VERSION, REFERENCE_SEED_VERSION),
        )
        _verify_v2_contract(conn)


def _inspect_dependencies(conn: sqlite3.Connection, names: set[str]) -> list[str]:
    blockers: list[str] = []
    for table in sorted(names):
        if table.startswith("sqlite_"):
            continue
        foreign_keys = conn.execute(
            f"PRAGMA foreign_key_list({_quote(table)})"
        ).fetchall()
        dependent = [row for row in foreign_keys if str(row[2]).casefold() == "m_pod"]
        if dependent:
            if table not in _KNOWN_LEGACY_CHILDREN:
                blockers.append(f"unrecognized foreign-key dependent: {table}")
                continue
            actual = {(str(row[3]), str(row[4])) for row in dependent}
            if actual != _KNOWN_LEGACY_CHILDREN[table]:
                blockers.append(f"unexpected m_pod foreign keys on {table}")
        if table in _KNOWN_LEGACY_CHILDREN:
            extra = [row for row in foreign_keys if str(row[2]).casefold() != "m_pod"]
            if extra:
                blockers.append(f"unhandled foreign-key dependency on {table}")

    legacy_targets = {name.casefold() for name in names & set(_LEGACY_OBJECTS)}
    for table in sorted(names - set(_LEGACY_OBJECTS)):
        if table.startswith("sqlite_"):
            continue
        foreign_keys = conn.execute(
            f"PRAGMA foreign_key_list({_quote(table)})"
        ).fetchall()
        for foreign_key in foreign_keys:
            target = str(foreign_key[2]).casefold()
            if target in legacy_targets and target != "m_pod":
                blockers.append(
                    f"unrecognized foreign-key dependency: {table} -> {target}"
                )

    objects = conn.execute(
        "SELECT type, name, tbl_name, sql FROM sqlite_master "
        "WHERE type IN ('view', 'trigger') AND sql IS NOT NULL"
    ).fetchall()
    for kind, name, table, sql in objects:
        if kind == "trigger" and str(table) in _LEGACY_OBJECTS:
            continue
        if _DEPENDENCY_NAME.search(str(sql)):
            blockers.append(f"unrecognized {kind} dependency: {name}")
    return blockers


def inspect_migration(conn: sqlite3.Connection) -> MigrationReport:
    """Inspect the current registry and its dependencies without writing."""
    tables = _table_names(conn)
    reserved_views = _reserved_pod_views(conn)
    names = tables | reserved_views
    legacy_tables = tuple(name for name in _LEGACY_OBJECTS if name in names)
    # A reserved registry slot may only hold the canonical table with the canonical
    # spelling. SQLite names are case-insensitive, so a differently spelled object still
    # occupies that slot; an unrelated application name is never a POD object.
    shape_blockers = [
        f"POD registry object is a view, not the canonical table: {name}"
        for name in sorted(reserved_views)
    ]
    shape_blockers.extend(
        f"POD registry object is not the canonical table: {name}"
        for name in sorted(
            name
            for name in tables
            if name.casefold() in _POD_RESERVED_NAMES
            and name not in _POD_RESERVED_NAMES
        )
    )
    if shape_blockers:
        return MigrationReport(
            status="blocked",
            blockers=tuple(shape_blockers),
            legacy_tables=legacy_tables,
            documents_table_present="documents" in names,
        )
    if "pod_registry_state" in names:
        state = conn.execute(
            "SELECT schema_version, source_generation, knowledge_generation "
            "FROM pod_registry_state WHERE singleton_id = 1"
        ).fetchone()
        if state is not None and state[0] == SCHEMA_VERSION:
            from esdc.pod_registry.economics_upgrade import (
                inspect_economics_upgrade,
            )

            economics = inspect_economics_upgrade(conn)
            if economics.status != "current":
                blockers = economics.blockers or (ECONOMICS_UPGRADE_RECOVERY,)
                return MigrationReport(
                    status="blocked",
                    blockers=blockers,
                    legacy_tables=legacy_tables,
                    documents_table_present="documents" in names,
                )
            _verify_v2_contract(conn)
            return MigrationReport(
                status="v2",
                source_generation=int(state[1]),
                knowledge_generation=(int(state[2]) if state[2] is not None else None),
                legacy_tables=legacy_tables,
                documents_table_present="documents" in names,
            )
        return MigrationReport(
            status="blocked",
            blockers=(
                "POD registry metadata exists but is incomplete or incompatible",
            ),
            documents_table_present="documents" in names,
        )

    if "m_pod" not in names:
        blockers = [
            f"legacy child table has no m_pod parent: {name}"
            for name in ("project_pod", "pod_document", "pod_revision")
            if name in names
        ]
        blockers.extend(
            f"v2 target table exists without registry metadata: {name}"
            for name in sorted((names & _V2_TABLES) - set(_KNOWN_LEGACY_CHILDREN))
        )
        return MigrationReport(
            status="blocked" if blockers else "fresh",
            blockers=tuple(blockers),
            legacy_tables=legacy_tables,
            documents_table_present="documents" in names,
        )

    blockers = _inspect_dependencies(conn, names)
    expected_m_pod_columns = {"id", "pod_id"}
    actual_m_pod_columns = set(_column_names(conn, "m_pod"))
    if not expected_m_pod_columns <= actual_m_pod_columns:
        blockers.append("legacy m_pod lacks the expected id and pod_id columns")
    for target in sorted((names & _V2_TABLES) - set(_LEGACY_OBJECTS)):
        blockers.append(f"v2 target table exists without registry metadata: {target}")
    for child, required in {
        "pod_document": {"pod_id", "doc_id"},
        "project_pod": {"pod_id", "project_id"},
        "pod_revision": {"successor_id", "predecessor_id"},
    }.items():
        if child in names and not required <= set(_column_names(conn, child)):
            blockers.append(f"legacy table {child!r} has an unsupported shape")
    return MigrationReport(
        status="blocked" if blockers else "legacy",
        blockers=tuple(blockers),
        legacy_tables=legacy_tables,
        documents_table_present="documents" in names,
    )


def _normalize_date(value: PreparedScalar) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        return date.fromisoformat(value[:10])
    raise ValueError(f"invalid approval_date in prepared master row: {value!r}")


def _sqlite_value(value: PreparedScalar) -> int | float | str | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, Decimal):
        return format(value, "f")
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, UUID):
        return str(value)
    return value


def _validate_prepared_batch(batch: PreparedPodBatch) -> Any:
    contract = load_pod_contract()
    issues = validate_pod_batch(batch, contract)
    if issues:
        summary = "; ".join(
            f"{issue.table}[{issue.row}].{issue.column}: {issue.message}"
            for issue in issues[:5]
        )
        raise MigrationRefusedError(
            f"prepared POD batch failed contract validation: {summary}"
        )
    input_tables = {
        name for name, table in contract.tables.items() if table.role == "input"
    }
    if set(batch.rows) != input_tables:
        raise MigrationRefusedError(
            "prepared POD batch must contain all five input tables"
        )
    return contract


def _integer_value(value: PreparedScalar, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise MigrationRefusedError(f"prepared {field} must be an integer")
    return value


def _identity_ledger_reservations(
    batch: PreparedPodBatch,
) -> tuple[_IdentityLedgerReservation, ...]:
    masters: dict[int, Mapping[str, PreparedScalar]] = {}
    issued_masters: dict[int, Mapping[str, PreparedScalar]] = {}
    for row in batch.rows["m_pod"]:
        pod_id_seq = _integer_value(row.get("pod_id_seq"), field="pod_id_seq")
        masters[pod_id_seq] = row
        order_value = row.get("pod_order_no")
        pod_id_value = row.get("pod_id")
        if order_value is None and pod_id_value is None:
            if pod_id_seq != 856:
                raise MigrationRefusedError(
                    "only pod_id_seq 856 may be imported without an issued identity"
                )
            continue
        if order_value is None or pod_id_value is None:
            raise MigrationRefusedError(
                "prepared master identity must include both pod_order_no and pod_id"
            )
        issued_masters[pod_id_seq] = row

    if batch.identity_ledger is None:
        reservations: list[_IdentityLedgerReservation] = []
        for pod_id_seq, row in issued_masters.items():
            approval_date = _normalize_date(row.get("approval_date"))
            order_no = _integer_value(row.get("pod_order_no"), field="pod_order_no")
            daily_slot = order_no - (approval_date.timetuple().tm_yday - 1) * 20
            reservations.append(
                _master_identity_reservation(
                    row,
                    pod_id_seq=pod_id_seq,
                    approval_date=approval_date,
                    daily_slot=daily_slot,
                    order_no=order_no,
                    actor=MIGRATION_ACTOR,
                    issued_at=None,
                )
            )
        return tuple(reservations)

    identity_ledger = batch.identity_ledger
    if isinstance(identity_ledger, (str, bytes)):
        raise MigrationRefusedError("prepared identity ledger must be a row sequence")

    reservations = []
    seen_sequences: set[int] = set()
    seen_slots: set[tuple[str, int]] = set()
    seen_order_numbers: set[tuple[int, int]] = set()
    seen_pod_ids: set[str] = set()
    seen_pod_uuids: set[str] = set()
    for row in identity_ledger:
        if not isinstance(row, Mapping):
            raise MigrationRefusedError(
                "prepared identity ledger rows must be mappings"
            )
        required = {
            "approval_year",
            "approval_date",
            "daily_slot",
            "pod_order_no",
            "pod_id_seq",
            "pod_id",
            "issued_at",
            "issued_by",
        }
        missing = required - set(row)
        if missing:
            raise MigrationRefusedError(
                f"prepared identity ledger row is missing fields: {sorted(missing)}"
            )
        try:
            approval_date = _normalize_date(row.get("approval_date"))
        except (TypeError, ValueError) as exc:
            raise MigrationRefusedError(
                "prepared identity ledger has an invalid approval_date"
            ) from exc
        approval_year = _integer_value(row.get("approval_year"), field="approval_year")
        daily_slot = _integer_value(row.get("daily_slot"), field="daily_slot")
        order_no = _integer_value(row.get("pod_order_no"), field="pod_order_no")
        pod_id_seq = _integer_value(row.get("pod_id_seq"), field="pod_id_seq")
        pod_id_value = row.get("pod_id")
        actor_value = row.get("issued_by")
        issued_at_value = row.get("issued_at")
        if not isinstance(pod_id_value, str) or not pod_id_value:
            raise MigrationRefusedError("prepared identity ledger pod_id must be text")
        if not isinstance(actor_value, str) or not actor_value.startswith("FA:"):
            raise MigrationRefusedError(
                "prepared identity ledger issued_by must preserve an FA:<action> actor"
            )
        if not actor_value[3:].strip():
            raise MigrationRefusedError(
                "prepared identity ledger issued_by must preserve an FA:<action> actor"
            )
        if not isinstance(issued_at_value, str) or not issued_at_value:
            raise MigrationRefusedError(
                "prepared identity ledger issued_at must be non-empty text"
            )
        expected_daily_slot = order_no - (approval_date.timetuple().tm_yday - 1) * 20
        if (
            approval_year != approval_date.year
            or not 1 <= daily_slot <= 20
            or daily_slot != expected_daily_slot
        ):
            raise MigrationRefusedError(
                "prepared identity ledger has an invalid date, year, or daily slot"
            )

        master = issued_masters.get(pod_id_seq)
        if master is None:
            if pod_id_seq in masters:
                raise MigrationRefusedError(
                    "identity ledger conflicts with unissued master sequence"
                )
            full_identity_fields = {
                "pod_uuid",
                "institution_id",
                "pod_type_id",
                "rev_num",
            }
            if not full_identity_fields <= set(row) or any(
                row.get(field) is None for field in full_identity_fields
            ):
                raise MigrationRefusedError(
                    "orphan identity ledger row lacks full identity metadata"
                )
            pod_uuid_value = row.get("pod_uuid")
            institution_id = _integer_value(
                row.get("institution_id"), field="institution_id"
            )
            pod_type_id = _integer_value(row.get("pod_type_id"), field="pod_type_id")
            rev_num = _integer_value(row.get("rev_num"), field="rev_num")
            try:
                pod_uuid = _validated_uuid(pod_uuid_value, field="pod_uuid")
            except MigrationRefusedError as exc:
                raise MigrationRefusedError(
                    "orphan identity ledger row has invalid full identity metadata"
                ) from exc
            reservation = _ledger_identity_reservation(
                approval_year=approval_year,
                approval_date=approval_date,
                daily_slot=daily_slot,
                order_no=order_no,
                pod_id_seq=pod_id_seq,
                pod_id=pod_id_value,
                pod_uuid=pod_uuid,
                institution_id=institution_id,
                pod_type_id=pod_type_id,
                rev_num=rev_num,
                actor=actor_value,
                issued_at=issued_at_value,
            )
        else:
            master_date = _normalize_date(master.get("approval_date"))
            master_order = _integer_value(
                master.get("pod_order_no"), field="pod_order_no"
            )
            master_pod_id = master.get("pod_id")
            if (
                approval_date != master_date
                or order_no != master_order
                or pod_id_value != master_pod_id
            ):
                raise MigrationRefusedError(
                    "provided identity ledger conflicts with prepared master identity"
                )
            pod_uuid = _validated_uuid(master.get("pod_uuid"), field="pod_uuid")
            institution_id = _integer_value(
                master.get("institution_id"), field="institution_id"
            )
            pod_type_id = _integer_value(master.get("pod_type_id"), field="pod_type_id")
            rev_num = _integer_value(master.get("rev_num"), field="rev_num")
            for field, expected in (
                ("pod_uuid", pod_uuid),
                ("institution_id", institution_id),
                ("pod_type_id", pod_type_id),
                ("rev_num", rev_num),
            ):
                provided = row.get(field)
                if provided is None:
                    continue
                if field == "pod_uuid":
                    matches = _validated_uuid(provided, field=field) == expected
                else:
                    matches = _integer_value(provided, field=field) == expected
                if not matches:
                    raise MigrationRefusedError(
                        "provided identity ledger conflicts with prepared master "
                        "identity"
                    )
            reservation = _ledger_identity_reservation(
                approval_year=approval_year,
                approval_date=approval_date,
                daily_slot=daily_slot,
                order_no=order_no,
                pod_id_seq=pod_id_seq,
                pod_id=pod_id_value,
                pod_uuid=pod_uuid,
                institution_id=institution_id,
                pod_type_id=pod_type_id,
                rev_num=rev_num,
                actor=actor_value,
                issued_at=issued_at_value,
            )
        if pod_id_seq in seen_sequences:
            raise MigrationRefusedError(
                "prepared identity ledger has duplicate sequences"
            )
        slot_key = (reservation.approval_date, reservation.daily_slot)
        order_key = (reservation.approval_year, reservation.pod_order_no)
        if (
            slot_key in seen_slots
            or order_key in seen_order_numbers
            or reservation.pod_id in seen_pod_ids
            or reservation.pod_uuid in seen_pod_uuids
        ):
            raise MigrationRefusedError(
                "prepared identity ledger has duplicate identities"
            )
        seen_sequences.add(pod_id_seq)
        seen_slots.add(slot_key)
        seen_order_numbers.add(order_key)
        seen_pod_ids.add(reservation.pod_id)
        seen_pod_uuids.add(reservation.pod_uuid)
        reservations.append(reservation)

    missing_sequences = set(issued_masters) - seen_sequences
    if missing_sequences:
        raise MigrationRefusedError(
            "provided identity ledger is missing issued masters: "
            f"{sorted(missing_sequences)}"
        )
    return tuple(reservations)


def _validated_uuid(value: PreparedScalar, *, field: str) -> str:
    try:
        parsed = value if isinstance(value, UUID) else UUID(str(value))
    except (TypeError, ValueError, AttributeError) as exc:
        raise MigrationRefusedError(f"prepared {field} must be a UUIDv4") from exc
    if parsed.version != 4:
        raise MigrationRefusedError(f"prepared {field} must be a UUIDv4")
    return str(parsed)


def _master_identity_reservation(
    row: Mapping[str, PreparedScalar],
    *,
    pod_id_seq: int,
    approval_date: date,
    daily_slot: int,
    order_no: int,
    actor: str,
    issued_at: str | None,
) -> _IdentityLedgerReservation:
    return _ledger_identity_reservation(
        approval_year=approval_date.year,
        approval_date=approval_date,
        daily_slot=daily_slot,
        order_no=order_no,
        pod_id_seq=pod_id_seq,
        pod_id=row.get("pod_id"),
        pod_uuid=_validated_uuid(row.get("pod_uuid"), field="pod_uuid"),
        institution_id=_integer_value(
            row.get("institution_id"), field="institution_id"
        ),
        pod_type_id=_integer_value(row.get("pod_type_id"), field="pod_type_id"),
        rev_num=_integer_value(row.get("rev_num"), field="rev_num"),
        actor=actor,
        issued_at=issued_at,
    )


def _ledger_identity_reservation(
    *,
    approval_year: int,
    approval_date: date,
    daily_slot: int,
    order_no: int,
    pod_id_seq: int,
    pod_id: PreparedScalar,
    pod_uuid: str,
    institution_id: int,
    pod_type_id: int,
    rev_num: int,
    actor: str,
    issued_at: str | None,
) -> _IdentityLedgerReservation:
    expected_pod_id = (
        f"PL-{approval_year}-{order_no:04d}-{institution_id}-{pod_type_id}-{rev_num}"
    )
    if (
        pod_id_seq < 1
        or not 1 <= daily_slot <= 20
        or not 1 <= order_no <= 7320
        or institution_id not in range(1, 6)
        or pod_type_id not in range(1, 5)
        or not 0 <= rev_num <= 9
        or not isinstance(pod_id, str)
        or pod_id != expected_pod_id
    ):
        raise MigrationRefusedError("prepared identity ledger has an invalid identity")
    if order_no != (approval_date.timetuple().tm_yday - 1) * 20 + daily_slot:
        raise MigrationRefusedError(
            "prepared identity ledger order does not match its date slot"
        )
    return _IdentityLedgerReservation(
        approval_year=approval_year,
        approval_date=approval_date.isoformat(),
        daily_slot=daily_slot,
        pod_order_no=order_no,
        pod_id_seq=pod_id_seq,
        pod_id=pod_id,
        pod_uuid=pod_uuid,
        institution_id=institution_id,
        pod_type_id=pod_type_id,
        rev_num=rev_num,
        actor=actor,
        issued_at=issued_at,
    )


def _insert_prepared_batch(
    conn: sqlite3.Connection,
    batch: PreparedPodBatch,
    contract: Any,
) -> None:
    for table_name, table in contract.tables.items():
        if table.role != "input":
            continue
        columns = tuple(column.name for column in table.columns)
        if not batch.rows.get(table_name):
            continue
        sql = (
            f"INSERT INTO {_quote(table_name)} "
            f"({', '.join(_quote(column) for column in columns)}) "
            f"VALUES ({', '.join('?' for _ in columns)})"
        )
        insert_rows = batch.rows[table_name]
        if table_name == "m_pod":
            insert_rows = [
                {**row, "superseded_by": None, "is_latest": True} for row in insert_rows
            ]
        values = [
            tuple(_sqlite_value(row.get(column)) for column in columns)
            for row in insert_rows
        ]
        conn.executemany(sql, values)
        if table_name == "m_pod":
            for row in batch.rows[table_name]:
                predecessor = row.get("superseded_by")
                if predecessor is None:
                    continue
                conn.execute(
                    "UPDATE m_pod SET superseded_by = ?, is_latest = 0 "
                    "WHERE pod_id_seq = ?",
                    (
                        _sqlite_value(predecessor),
                        _sqlite_value(row.get("pod_id_seq")),
                    ),
                )


def _legacy_document_links(
    conn: sqlite3.Connection, names: set[str]
) -> tuple[tuple[int, str], ...]:
    if "pod_document" not in names:
        return ()
    return tuple(
        (int(row[0]), str(row[1]))
        for row in conn.execute("SELECT pod_id, doc_id FROM pod_document")
    )


def _legacy_master_rows(conn: sqlite3.Connection, names: set[str]) -> dict[int, str]:
    if "m_pod" not in names:
        return {}
    return {
        int(row[0]): str(row[1]) for row in conn.execute("SELECT id, pod_id FROM m_pod")
    }


def _insert_identity_ledger(
    conn: sqlite3.Connection,
    reservations: tuple[_IdentityLedgerReservation, ...],
) -> None:
    for reservation in reservations:
        identity_values = (
            reservation.approval_year,
            reservation.approval_date,
            reservation.daily_slot,
            reservation.pod_order_no,
            reservation.pod_id_seq,
            reservation.pod_id,
            reservation.pod_uuid,
            reservation.institution_id,
            reservation.pod_type_id,
            reservation.rev_num,
            reservation.actor,
        )
        existing = conn.execute(
            "SELECT approval_year, approval_date, daily_slot, pod_order_no, "
            "pod_id_seq, pod_id, pod_uuid, institution_id, pod_type_id, rev_num, "
            "actor, issued_at FROM _pod_identity_ledger WHERE pod_id_seq = ?",
            (reservation.pod_id_seq,),
        ).fetchone()
        if existing is not None:
            if tuple(existing[:11]) != identity_values or (
                reservation.issued_at is not None
                and existing[11] != reservation.issued_at
            ):
                raise MigrationRefusedError(
                    "identity ledger reservation conflicts with prepared identity"
                )
            continue
        columns = (
            "approval_year, approval_date, daily_slot, pod_order_no, pod_id_seq, "
            "pod_id, pod_uuid, institution_id, pod_type_id, rev_num, actor"
        )
        values = identity_values
        if reservation.issued_at is not None:
            columns += ", issued_at"
            values = (*values, reservation.issued_at)
        conn.execute(
            f"INSERT INTO _pod_identity_ledger ({columns}) "
            f"VALUES ({', '.join('?' for _ in values)})",
            values,
        )


def _rename_legacy_objects(conn: sqlite3.Connection, names: set[str]) -> dict[str, str]:
    renamed: dict[str, str] = {}
    for name in (
        "m_pod",
        "r_institution",
        "r_pod_type",
        "project_pod",
        "pod_document",
        "pod_revision",
    ):
        if name not in names:
            continue
        temporary = f"__pod_registry_legacy_{name}"
        conn.execute(f"ALTER TABLE {_quote(name)} RENAME TO {_quote(temporary)}")
        renamed[name] = temporary
    return renamed


def _drop_legacy_objects(conn: sqlite3.Connection, renamed: Mapping[str, str]) -> None:
    for name in (
        "project_pod",
        "pod_document",
        "pod_revision",
        "m_pod",
        "r_institution",
        "r_pod_type",
    ):
        temporary = renamed.get(name)
        if temporary is not None:
            conn.execute(f"DROP TABLE {_quote(temporary)}")


def migrate_registry(
    conn: sqlite3.Connection,
    *,
    prepared_batch: PreparedPodBatch,
    approved_crosswalk: dict[int, int],
) -> MigrationReport:
    """Replace an inspected legacy POD schema using an explicit validated snapshot."""
    if prepared_batch.profile != "full":
        raise MigrationRefusedError("POD replacement requires a full current dataset")
    contract = _validate_prepared_batch(prepared_batch)
    ledger_reservations = _identity_ledger_reservations(prepared_batch)
    _ensure_registry_writer(conn)
    if conn.execute("PRAGMA foreign_keys").fetchone()[0] != 1:
        raise RuntimeError("POD migration requires foreign_keys=ON")
    if any(
        isinstance(old, bool)
        or not isinstance(old, int)
        or isinstance(new, bool)
        or not isinstance(new, int)
        for old, new in approved_crosswalk.items()
    ):
        raise TypeError(
            "approved_crosswalk must map integer legacy IDs to integer "
            "pod_id_seq values"
        )

    current_ids = {
        _integer_value(row["pod_id_seq"], field="pod_id_seq")
        for row in prepared_batch.rows["m_pod"]
    }
    missing_targets = set(approved_crosswalk.values()) - current_ids
    if missing_targets:
        raise MigrationRefusedError(
            "crosswalk targets are absent from the prepared master: "
            f"{sorted(missing_targets)}"
        )

    row_counts: dict[str, int] = {}
    reconciliation: list[LegacyIdentityMapping] = []
    renamed: dict[str, str] = {}
    legacy_rows: dict[int, str] = {}
    mapped_links: list[tuple[int, str]] = []
    with _atomic_writer(conn):
        conn.execute("PRAGMA legacy_alter_table = OFF")
        inspection = inspect_migration(conn)
        if inspection.status != "legacy" or inspection.blockers:
            details = "; ".join(inspection.blockers) or inspection.status
            raise MigrationRefusedError(f"POD migration is not safe to run: {details}")
        names = _table_names(conn)
        legacy_rows = _legacy_master_rows(conn, names)
        legacy_links = _legacy_document_links(conn, names)
        linked_ids = {legacy_id for legacy_id, _ in legacy_links}
        missing_links = linked_ids - set(approved_crosswalk)
        if missing_links:
            raise MigrationRefusedError(
                f"document crosswalk is missing legacy POD IDs: {sorted(missing_links)}"
            )
        unknown_crosswalk = set(approved_crosswalk) - set(legacy_rows)
        if unknown_crosswalk:
            raise MigrationRefusedError(
                "crosswalk contains unknown legacy POD IDs: "
                f"{sorted(unknown_crosswalk)}"
            )
        mapped_links = [
            (approved_crosswalk[legacy_id], doc_id)
            for legacy_id, doc_id in legacy_links
        ]
        if len(mapped_links) != len(set(mapped_links)):
            raise MigrationRefusedError(
                "approved crosswalk would collapse duplicate document links"
            )
        conn.execute("PRAGMA defer_foreign_keys = ON")
        renamed = _rename_legacy_objects(conn, names)
        _drop_legacy_objects(conn, renamed)
        initialize_registry_v2(conn)
        _insert_identity_ledger(conn, ledger_reservations)
        _insert_prepared_batch(conn, prepared_batch, contract)

        target_rows = {
            _integer_value(row["pod_id_seq"], field="pod_id_seq"): row
            for row in prepared_batch.rows["m_pod"]
        }
        for legacy_id, target_id in sorted(approved_crosswalk.items()):
            target = target_rows[target_id]
            mapping = LegacyIdentityMapping(
                legacy_pod_id=legacy_id,
                legacy_pod_id_value=legacy_rows[legacy_id],
                pod_id_seq=target_id,
                pod_id=str(target["pod_id"]) if target["pod_id"] is not None else "",
                pod_uuid=str(target["pod_uuid"]),
            )
            reconciliation.append(mapping)
            conn.execute(
                "INSERT INTO pod_registry_legacy_identity ("
                "legacy_pod_id, legacy_pod_id_value, target_pod_id_seq, "
                "target_pod_id, target_pod_uuid, reconciled_by"
                ") VALUES (?, ?, ?, ?, ?, ?)",
                (
                    mapping.legacy_pod_id,
                    mapping.legacy_pod_id_value,
                    mapping.pod_id_seq,
                    mapping.pod_id or None,
                    mapping.pod_uuid,
                    MIGRATION_ACTOR,
                ),
            )
        conn.executemany(
            "INSERT INTO pod_document (pod_id_seq, doc_id) VALUES (?, ?)",
            mapped_links,
        )
        from esdc.pod_registry.read_models import refresh_pod_after_source_change
        from esdc.pod_registry.sqlite_import import apply_staged_project_snapshots
        from esdc.pod_registry.store import ensure_raw_source_tables

        ensure_raw_source_tables(conn)
        apply_staged_project_snapshots(conn, prepared_batch.staged_snapshots)
        refresh_pod_after_source_change(conn, actor=MIGRATION_ACTOR)
        violations = conn.execute("PRAGMA foreign_key_check").fetchall()
        if violations:
            raise MigrationRefusedError(
                f"native foreign-key check failed after rebuild: {violations[:5]!r}"
            )
        _verify_v2_contract(conn)
        for table_name in sorted(_V2_TABLES):
            if table_name in _table_names(conn):
                row_counts[table_name] = int(
                    conn.execute(
                        f"SELECT COUNT(*) FROM {_quote(table_name)}"
                    ).fetchone()[0]
                )

    state = conn.execute(
        "SELECT source_generation, knowledge_generation FROM pod_registry_state "
        "WHERE singleton_id = 1"
    ).fetchone()
    return MigrationReport(
        status="v2",
        legacy_tables=tuple(renamed),
        source_generation=int(state[0]),
        knowledge_generation=(int(state[1]) if state[1] is not None else None),
        document_links_preserved=len(mapped_links),
        documents_table_present="documents" in _table_names(conn),
        identity_reconciliation=tuple(reconciliation),
        unmapped_legacy_pod_ids=tuple(
            sorted(set(legacy_rows) - set(approved_crosswalk))
        ),
        row_counts=tuple(sorted(row_counts.items())),
    )
