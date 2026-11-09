"""Explicit, transactional upgrade of the POD economics column names.

The economics columns dropped their historical ``pf`` segment. A database created
before that rename keeps the old physical names while advertising contract 1, so
opening it through the ordinary writer is refused until this upgrade runs.
Inspection is read-only; applying the upgrade is a name-only, transactional
operation that joins the caller's transaction when one is already open.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Literal, cast

from esdc.pod_registry.contract import load_pod_contract
from esdc.pod_registry.migrations import (
    SCHEMA_VERSION,
    MigrationRefusedError,
    _atomic_writer,
    _column_names,
    _ensure_registry_writer,
    _quote,
    _table_names,
    _verify_v2_contract,
)

UPGRADE_ACTOR = "FA:pod-economics-upgrade"

_PREFIXES = (
    ("npv_pf_ctr_", "npv_ctr_"),
    ("irr_pf_ctr_", "irr_ctr_"),
    ("pv_ratio_pf_ctr_", "pv_ratio_ctr_"),
    ("pi_pf_ctr_", "pi_ctr_"),
)
_SUFFIXES = {
    "m_pod": ("pod",),
    "pod_monitoring": ("real", "otlk"),
    "pod_monitoring_api": ("pod", "real", "otlk"),
}

#: The complete rename set: 24 columns across the three economics tables.
ECONOMICS_RENAMES: tuple[tuple[str, str, str], ...] = tuple(
    (table, old_prefix + suffix, new_prefix + suffix)
    for table, suffixes in _SUFFIXES.items()
    for suffix in suffixes
    for old_prefix, new_prefix in _PREFIXES
)
_ECONOMICS_TABLES = frozenset(_SUFFIXES)


@dataclass(frozen=True, slots=True)
class EconomicsUpgradeReport:
    """Read-only inspection or completed upgrade evidence."""

    status: Literal["upgrade_required", "current", "blocked"]
    renames: tuple[tuple[str, str, str], ...] = ()
    blockers: tuple[str, ...] = ()


def _metadata(conn: sqlite3.Connection) -> tuple[object, ...] | None:
    if "pod_registry_state" not in _table_names(conn):
        return None
    row = conn.execute(
        "SELECT schema_version, contract_id, contract_version, dictionary_version "
        "FROM pod_registry_state WHERE singleton_id = 1"
    ).fetchone()
    return None if row is None else tuple(row)


def _legacy_columns(contract, table: str) -> tuple[str, ...]:
    renames: dict[str, str] = {}
    for owner, old, new in ECONOMICS_RENAMES:
        if owner == table:
            renames[new] = old
    expected: list[str] = []
    for column in contract.tables[table].columns:
        name = str(column.name)
        expected.append(cast(str, renames.get(name, name)))
    return tuple(expected)


def inspect_economics_upgrade(
    conn: sqlite3.Connection,
) -> EconomicsUpgradeReport:
    """Classify the registry layout without writing or changing PRAGMAs."""
    contract = load_pod_contract()
    tables = _table_names(conn)
    blockers: list[str] = []
    missing = sorted(set(contract.tables) - tables)
    if missing:
        blockers.append(
            "POD registry is missing required tables: " + ", ".join(missing)
        )

    metadata = _metadata(conn)
    if metadata is None:
        blockers.append("POD registry metadata is missing")
    else:
        schema_version, contract_id, contract_version, dictionary_version = metadata
        if int(str(schema_version)) != SCHEMA_VERSION:
            blockers.append("POD registry schema version is unsupported")
        if str(contract_id) != contract.contract_id:
            blockers.append("POD registry contract id is unsupported")
        if str(dictionary_version) != contract.dictionary_version:
            blockers.append("POD registry dictionary version is unsupported")
        if int(str(contract_version)) not in (1, contract.contract_version):
            blockers.append("POD registry contract version is unsupported")

    if blockers:
        return EconomicsUpgradeReport(status="blocked", blockers=tuple(blockers))

    layouts: dict[str, str] = {}
    for name, table in contract.tables.items():
        actual = _column_names(conn, name)
        expected = tuple(column.name for column in table.columns)
        if actual == expected:
            layouts[name] = "current"
        elif actual == _legacy_columns(contract, name):
            layouts[name] = "legacy"
        else:
            blockers.append(
                f"v2 table {name!r} has a mixed or unknown column shape"
            )

    if blockers:
        return EconomicsUpgradeReport(status="blocked", blockers=tuple(blockers))

    legacy_tables = {
        name for name in _ECONOMICS_TABLES if layouts[name] == "legacy"
    }
    if legacy_tables and legacy_tables != set(_ECONOMICS_TABLES):
        blockers.append("POD economics tables use mixed column layouts")
        return EconomicsUpgradeReport(status="blocked", blockers=tuple(blockers))
    legacy = bool(legacy_tables)

    try:
        _verify_v2_contract(conn, legacy_economics=legacy)
    except MigrationRefusedError as exc:
        blockers.append(str(exc))

    if blockers:
        return EconomicsUpgradeReport(status="blocked", blockers=tuple(blockers))

    metadata = _metadata(conn)
    assert metadata is not None
    contract_version = int(str(metadata[2]))
    if contract_version == contract.contract_version:
        if legacy:
            return EconomicsUpgradeReport(
                status="blocked",
                blockers=(
                    "contract 2 registry still uses the legacy economics column names",
                ),
            )
        return EconomicsUpgradeReport(status="current")
    if contract_version == 1:
        pending = ECONOMICS_RENAMES if legacy else ()
        return EconomicsUpgradeReport(
            status="upgrade_required", renames=tuple(pending)
        )
    return EconomicsUpgradeReport(
        status="blocked",
        blockers=("POD registry contract version is unsupported",),
    )


def upgrade_economics_columns(
    conn: sqlite3.Connection,
) -> EconomicsUpgradeReport:
    """Apply the explicit economics rename inside the caller's transaction."""
    _ensure_registry_writer(conn)
    if conn.execute("PRAGMA foreign_keys").fetchone()[0] != 1:
        raise RuntimeError("POD economics upgrade requires foreign_keys=ON")
    contract = load_pod_contract()

    with _atomic_writer(conn):
        report = inspect_economics_upgrade(conn)
        if report.status == "blocked":
            raise MigrationRefusedError("; ".join(report.blockers))
        if report.status == "current":
            return report

        for table, old, new in report.renames:
            conn.execute(
                f"ALTER TABLE {_quote(table)} "
                f"RENAME COLUMN {_quote(old)} TO {_quote(new)}"
            )
        conn.execute(
            "UPDATE pod_registry_state SET contract_version = ?, updated_by = ? "
            "WHERE singleton_id = 1",
            (contract.contract_version, UPGRADE_ACTOR),
        )
        _verify_v2_contract(conn)
        if conn.execute("PRAGMA foreign_key_check").fetchall():
            raise MigrationRefusedError(
                "POD economics upgrade produced foreign key violations"
            )
        return EconomicsUpgradeReport(status="current", renames=report.renames)
