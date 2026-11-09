"""Shared synthetic contract-1 POD registry builder for economics upgrade tests.

The historical schema is loaded from the frozen SQL fixture; no current schema is
rewritten to imitate it. Rows are intentionally synthetic and never read from a
live database.
"""

from __future__ import annotations

import importlib.resources
import sqlite3
from pathlib import Path
from typing import Any

FIXTURE = Path(__file__).with_name("fixtures") / "pre_economics_rename_v2.sql"
PREFIXES = (
    ("npv_pf_ctr_", "npv_ctr_"),
    ("irr_pf_ctr_", "irr_ctr_"),
    ("pv_ratio_pf_ctr_", "pv_ratio_ctr_"),
    ("pi_pf_ctr_", "pi_ctr_"),
)
SUFFIXES = {
    "m_pod": ("pod",),
    "pod_monitoring": ("real", "otlk"),
    "pod_monitoring_api": ("pod", "real", "otlk"),
}
RECOVERY = (
    "Run 'esdc pod-upgrade-economics --database <path>' "
    "to inspect the required upgrade."
)


def renames() -> tuple[tuple[str, str, str], ...]:
    return tuple(
        (table, old_prefix + suffix, new_prefix + suffix)
        for table, suffixes in SUFFIXES.items()
        for suffix in suffixes
        for old_prefix, new_prefix in PREFIXES
    )


def schema_sql() -> str:
    return FIXTURE.read_text(encoding="utf-8")


def seed_sql() -> str:
    return (
        importlib.resources.files("esdc")
        .joinpath("sql")
        .joinpath("pod_reference_seed.sql")
        .read_text(encoding="utf-8")
    )


def quote(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def old_names() -> dict[tuple[str, str], str]:
    return {(table, new): old for table, old, new in renames()}


def sentinels() -> dict[tuple[str, str], str | None]:
    values: dict[tuple[str, str], str | None] = {}
    for index, (table, _old, new) in enumerate(renames(), start=1):
        values[(table, new)] = f"{100 + index}.{index:03d}"
    values[("m_pod", "npv_ctr_pod")] = "123.456"
    values[("m_pod", "irr_ctr_pod")] = "0.123456"
    values[("m_pod", "pv_ratio_ctr_pod")] = None
    values[("m_pod", "pi_ctr_pod")] = "-2.500"
    values[("pod_monitoring", "irr_ctr_real")] = "0.000"
    values[("pod_monitoring", "pv_ratio_ctr_real")] = None
    values[("pod_monitoring_api", "npv_ctr_otlk")] = "0"
    return values


def columns(conn: sqlite3.Connection, table: str) -> tuple[str, ...]:
    return tuple(
        str(row[1]) for row in conn.execute(f"PRAGMA table_info({quote(table)})")
    )


def connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _insert(
    conn: sqlite3.Connection,
    table: str,
    values: dict[str, Any],
    *,
    renamed: bool,
) -> None:
    names = dict(values)
    if not renamed:
        mapping = old_names()
        for (owner, new), old in mapping.items():
            if owner == table and new in names:
                names[old] = names.pop(new)
    column_list = ", ".join(quote(name) for name in names)
    placeholders = ", ".join("?" for _ in names)
    conn.execute(
        f"INSERT INTO {quote(table)} ({column_list}) VALUES ({placeholders})",
        tuple(names.values()),
    )


def build_registry(
    path: Path,
    *,
    contract_version: int = 1,
    renamed: bool = False,
    metadata: dict[str, Any] | None = None,
) -> dict[tuple[str, str], str | None]:
    """Create a complete, valid contract-1 registry in either known layout."""
    values = sentinels()
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(schema_sql())
    if renamed:
        for table, old, new in renames():
            conn.execute(
                f"ALTER TABLE {quote(table)} "
                f"RENAME COLUMN {quote(old)} TO {quote(new)}"
            )
    conn.executescript(seed_sql())

    pod_uuid = "11111111-1111-4111-8111-111111111111"
    pod_id = "PL-2024-0001-1-1-0"
    conn.execute(
        "INSERT INTO _pod_identity_ledger (approval_year, approval_date, daily_slot, "
        "pod_order_no, pod_id_seq, pod_id, pod_uuid, institution_id, pod_type_id, "
        "rev_num, actor, issued_at) VALUES (2024, '2024-01-01', 1, 1, 1, ?, ?, 1, 1, "
        "0, 'FA:test-ledger', '2026-10-27T08:00:00.000Z')",
        (pod_id, pod_uuid),
    )
    master: dict[str, Any] = {
        "pod_id_seq": 1,
        "pod_uuid": pod_uuid,
        "approval_date": "2024-01-01",
        "superseded_by": None,
        "institution_id": 1,
        "pod_type_id": 1,
        "rev_num": 0,
        "pod_order_no": 1,
        "pod_id": pod_id,
        "pod_name": "Synthetic POD",
        "pod_letter_num": "Letter-1",
        "is_latest": 1,
    }
    master.update(
        {new: value for (table, new), value in values.items() if table == "m_pod"}
    )
    _insert(conn, "m_pod", master, renamed=renamed)

    monitoring: dict[str, Any] = {
        "monpod_uuid": "22222222-2222-4222-8222-222222222222",
        "pod_id_seq": 1,
        "pod_id": pod_id,
        "reported_date": "2024-06-30",
        "is_completed": 0,
        "pod_stage_id": 1,
    }
    monitoring.update(
        {
            new: value
            for (table, new), value in values.items()
            if table == "pod_monitoring"
        }
    )
    _insert(conn, "pod_monitoring", monitoring, renamed=renamed)

    api: dict[str, Any] = {"pod_id_seq": 1, "reported_date": "2024-06-30"}
    api.update(
        {
            new: value
            for (table, new), value in values.items()
            if table == "pod_monitoring_api"
        }
    )
    _insert(conn, "pod_monitoring_api", api, renamed=renamed)

    conn.execute(
        "INSERT INTO m_project (project_id, project_name, is_active) "
        "VALUES ('P-1', 'Synthetic project', 1)"
    )
    conn.execute(
        "INSERT INTO pod_project (pod_project_uuid, pod_id_seq, project_id, "
        "is_main_project, is_auto_inferred) VALUES (?, 1, 'P-1', 1, 0)",
        ("33333333-3333-4333-8333-333333333333",),
    )
    conn.execute("INSERT INTO pod_document (pod_id_seq, doc_id) VALUES (1, 'DOC-001')")
    conn.execute(
        "INSERT INTO pod_registry_legacy_identity (legacy_pod_id, legacy_pod_id_value, "
        "target_pod_id_seq, target_pod_id, target_pod_uuid, reconciled_by) "
        "VALUES (900, 'LEGACY-900', 1, ?, ?, 'FA:test')",
        (pod_id, pod_uuid),
    )
    conn.execute(
        "INSERT INTO pod_read_model_snapshot (singleton_id, source_generation, "
        "resource_revision, resource_sha256, timeseries_revision, timeseries_sha256) "
        "VALUES (1, 7, 'resource-r1', 'resource-sha', 'timeseries-r1', 'timeseries-sha')"
    )
    conn.execute(
        "INSERT INTO pod_knowledge_state (singleton_id, basis_sha256, source_generation) "
        "VALUES (1, 'basis-sha', 7)"
    )
    conn.execute(
        "INSERT INTO pod_read_model_diagnostic (diagnostic_id, source_generation, code, "
        "message) VALUES (1, 7, 'synthetic', 'kept')"
    )
    conn.execute("CREATE TABLE application_notes (id INTEGER PRIMARY KEY, note TEXT)")
    conn.execute("INSERT INTO application_notes (id, note) VALUES (1, 'keep me')")
    state = {
        "schema_version": 2,
        "contract_id": "esdc.pod",
        "contract_version": contract_version,
        "dictionary_version": "0.4.0",
        "reference_seed_version": 20261028,
        "source_generation": 7,
        "knowledge_generation": 3,
        "updated_by": "FA:test-initialize",
    }
    if metadata is not None:
        state.update(metadata)
    conn.execute(
        "INSERT INTO pod_registry_state (singleton_id, schema_version, contract_id, "
        "contract_version, dictionary_version, reference_seed_version, "
        "source_generation, knowledge_generation, updated_by) "
        "VALUES (1, :schema_version, :contract_id, :contract_version, "
        ":dictionary_version, :reference_seed_version, :source_generation, "
        ":knowledge_generation, :updated_by)",
        state,
    )
    conn.commit()
    conn.close()
    return values


def dump(path: Path, *, tables: tuple[str, ...] | None = None) -> dict[str, Any]:
    conn = connect(path)
    try:
        names = tables or tuple(
            str(row[0])
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' "
                "AND name NOT LIKE 'sqlite_%' ORDER BY name"
            )
        )
        return {
            table: tuple(
                sorted(
                    (
                        tuple(row)
                        for row in conn.execute(f"SELECT * FROM {quote(table)}")
                    ),
                    key=repr,
                )
            )
            for table in names
        }
    finally:
        conn.close()
