"""Task5D1: shared fetch-guard regressions for common apply and WAL gate order.

Fixtures live under pytest's temporary task directory; the active operational
database is never opened. These regressions exercise the real SQLite adapter.
"""

from __future__ import annotations

import sqlite3
from dataclasses import replace
from pathlib import Path

import pytest

from esdc.pod_registry import sqlite_import
from esdc.pod_registry.contract import PodContract, load_pod_contract
from esdc.pod_registry.migrations import initialize_registry_v2
from esdc.pod_registry.models import (
    ImportTargetBasis,
    PreparedPodBatch,
    PreparedRow,
    PreparedScalar,
)
from esdc.sqlite_runtime import configure_writer, require_wal_runtime

CONTRACT = load_pod_contract()

INPUT_TABLES = ("m_project", "m_pod", "pod_project", "pod_monitoring", "pod_forecast")

TEST_ACTOR = "FA:fetch-guard-test"
TEST_TIMESTAMP = "2024-01-01T00:00:00"


def _canonical_headers(contract: PodContract) -> dict[str, tuple[str, ...]]:
    return {
        table_name: tuple(
            column.name for column in contract.tables[table_name].columns
        )
        for table_name in INPUT_TABLES
    }


def _project_row(contract: PodContract, project_id: str) -> PreparedRow:
    row: dict[str, PreparedScalar] = {}
    for column in contract.tables["m_project"].columns:
        if column.name == "project_id":
            row[column.name] = project_id
        elif column.name == "project_name":
            row[column.name] = "Race project"
        elif column.group == "audit":
            row[column.name] = TEST_TIMESTAMP if column.logical_type == "datetime" else TEST_ACTOR
        elif column.logical_type == "boolean":
            row[column.name] = 1
        else:
            row[column.name] = None
    return row


def _project_only_batch(
    contract: PodContract,
    project_id: str,
    target_basis: ImportTargetBasis | None,
) -> PreparedPodBatch:
    rows: dict[str, tuple[PreparedRow, ...]] = dict.fromkeys(INPUT_TABLES, ())
    rows["m_project"] = (_project_row(contract, project_id),)
    return PreparedPodBatch(
        source="sqlite",
        contract_id=contract.contract_id,
        contract_version=contract.contract_version,
        dictionary_version=contract.dictionary_version,
        headers=_canonical_headers(contract),
        rows=rows,
        identity_ledger=None,
        profile="full",
        source_path=None,
        target_basis=target_basis,
        staged_snapshots=None,
    )


def test_fresh_target_basis_rejects_a_generation_created_after_preparation(
    tmp_path: Path,
) -> None:
    """A batch prepared as a fresh target must not overwrite a raced registry.

    Preparation captured an absent target. Before apply, another writer created
    the target and committed generation 1; applying the prepared batch would
    replace that committed project and advance the generation again.
    """
    target = tmp_path / "raced-target.sqlite"
    prepared_fresh_basis = ImportTargetBasis(
        existed=False,
        source_generation=None,
        protected_state_sha256="",
        target_path=target.resolve(strict=False),
    )

    conn = sqlite3.connect(target)
    try:
        require_wal_runtime()
        configure_writer(conn)
        initialize_registry_v2(conn)

        # Another writer initializes the target and commits its own project.
        first_result = sqlite_import.apply_pod_batch(
            conn,
            _project_only_batch(CONTRACT, "P-0000001-01", None),
            "FA:other-writer",
        )
        assert first_result.source_generation == 1
        committed_ledger_count = conn.execute(
            "SELECT count(*) FROM _pod_identity_ledger"
        ).fetchone()[0]

        raced_batch = _project_only_batch(
            CONTRACT, "P-0000002-01", prepared_fresh_basis
        )
        with pytest.raises(
            sqlite_import.PodImportError,
            match="changed after import preparation",
        ):
            sqlite_import.apply_pod_batch(conn, raced_batch, TEST_ACTOR)

        # The raced apply must not have replaced the committed writer data.
        assert conn.execute(
            "SELECT project_id FROM m_project"
        ).fetchall() == [("P-0000001-01",)]
        assert conn.execute(
            "SELECT source_generation FROM pod_registry_state WHERE singleton_id=1"
        ).fetchone() == (1,)
        assert conn.execute(
            "SELECT count(*) FROM _pod_identity_ledger"
        ).fetchone()[0] == committed_ledger_count
    finally:
        conn.close()


def test_fresh_target_basis_still_applies_an_unchanged_generation_zero_target(
    tmp_path: Path,
) -> None:
    """The guard rejects only a raced target, not a genuinely fresh one."""
    target = tmp_path / "fresh-target.sqlite"
    prepared_fresh_basis = ImportTargetBasis(
        existed=False,
        source_generation=None,
        protected_state_sha256="",
        target_path=target.resolve(strict=False),
    )

    conn = sqlite3.connect(target)
    try:
        require_wal_runtime()
        configure_writer(conn)
        initialize_registry_v2(conn)

        result = sqlite_import.apply_pod_batch(
            conn,
            _project_only_batch(CONTRACT, "P-0000001-01", prepared_fresh_basis),
            TEST_ACTOR,
        )

        assert result.source_generation == 1
        assert conn.execute(
            "SELECT project_id FROM m_project"
        ).fetchall() == [("P-0000001-01",)]
    finally:
        conn.close()


def test_unsupported_wal_runtime_rejects_before_target_side_effects(
    tmp_path: Path,
) -> None:
    """The runtime gate is the first check; it must not create the target parent."""
    target_dir = tmp_path / "target-parent"
    target = target_dir / "guarded-target.sqlite"
    duck = tmp_path / "guarded-target.duckdb"

    batch = _project_only_batch(
        CONTRACT,
        "P-0000001-01",
        ImportTargetBasis(
            existed=False,
            source_generation=None,
            protected_state_sha256="",
            target_path=target.resolve(strict=False),
        ),
    )
    source_path = tmp_path / "prepared-source.jsonl"
    source_path.write_text("", encoding="utf-8")
    batch = replace(batch, source_path=source_path)

    original_version = sqlite3.sqlite_version_info
    try:
        sqlite3.sqlite_version_info = (3, 50, 4)
        with pytest.raises(RuntimeError, match="unsupported for WAL writers"):
            sqlite_import.load_prepared_pod_batch(
                batch,
                sqlite_path=target,
                duck_path=duck,
                actor=TEST_ACTOR,
                publish=False,
            )
    finally:
        sqlite3.sqlite_version_info = original_version

    assert not target_dir.exists(), "WAL gate must run before creating the target parent"
    assert not target.exists()
