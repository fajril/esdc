"""Current POD staged import path: prepare, apply, and publish without a legacy loader."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest
from openpyxl import Workbook, load_workbook
from openpyxl.cell import Cell

from esdc.pod_registry.contract import load_pod_contract
from esdc.pod_registry.sqlite_import import (
    PodImportError,
    generate_pod_import_template,
    load_prepared_pod_batch,
    prepare_pod_import,
)
from esdc.pod_registry.store import get_sqlite_readonly_connection

ACTOR = "FA:test-importer"
INPUT_TABLES: tuple[str, ...] = (
    "m_project",
    "m_pod",
    "pod_project",
    "pod_monitoring",
    "pod_forecast",
)
METADATA_SHEET = "__pod_metadata"


def _value_for(column, row_number: int) -> Any:
    """One contract-valid value per logical type, on the canonical identity for row 1."""
    name = column.name
    if name == "project_id":
        return f"P-{row_number:07d}-01"
    if name == "pod_id":
        # The business identity is derived from the values this builder assigns.
        return f"PL-2024-{row_number:04d}-{row_number}-{row_number}-0"
    if name in {"institution_id", "pod_type_id", "pod_stage_id", "pod_order_no"}:
        return row_number
    if name in {"pod_id_seq", "monpod_id", "frcast_id"}:
        return row_number
    if name.startswith("year"):
        return 2024
    if name == "rev_num":
        return 0
    if column.logical_type == "integer":
        return 1
    if column.logical_type == "decimal":
        return Decimal("1")
    if column.logical_type == "boolean":
        return True
    if column.logical_type == "date":
        return date(2024, 1, 1)
    if column.logical_type == "datetime":
        return "2024-01-01T00:00:00"
    if column.logical_type == "uuid":
        return str(uuid4())
    return "example"


def _row(contract, table_name: str, row_number: int = 1) -> dict[str, Any]:
    return {
        column.name: None if column.nullable else _value_for(column, row_number)
        for column in contract.tables[table_name].columns
    }


def _write_workbook(
    path: Path,
    contract,
    rows_by_table: dict[str, list[dict[str, Any]]],
    *,
    extra_header: tuple[str, str] | None = None,
) -> Path:
    """Write a contract-shaped workbook; header-only sheets carry no data rows."""
    workbook = Workbook()
    active = workbook.active
    if active is not None:
        workbook.remove(active)
    for table_name, rows in rows_by_table.items():
        sheet = workbook.create_sheet(table_name)
        columns = contract.tables[table_name].columns
        header = [column.name for column in columns]
        if extra_header and extra_header[0] == table_name:
            header.append(extra_header[1])
        sheet.append(header)
        for row in rows:
            values = [row[column.name] for column in columns]
            if extra_header and extra_header[0] == table_name:
                values.append("not-declared")
            sheet.append(values)
    workbook.save(path)
    return path


def _counts(path: Path) -> dict[str, int]:
    conn = get_sqlite_readonly_connection(path)
    assert conn is not None
    try:
        return {
            table: conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
            for table in INPUT_TABLES
        }
    finally:
        conn.close()


def test_full_import_commits_and_publishes_from_the_current_template(
    tmp_path: Path,
) -> None:
    contract = load_pod_contract()
    rows = {table: [_row(contract, table)] for table in INPUT_TABLES}
    workbook = _write_workbook(tmp_path / "pod.xlsx", contract, rows)
    target = tmp_path / "target.sqlite"
    duck_path = tmp_path / "projection.duckdb"

    batch = prepare_pod_import(workbook, "xlsx", contract, target_path=target)
    assert not target.exists()

    result, publication = load_prepared_pod_batch(
        batch, sqlite_path=target, duck_path=duck_path, actor=ACTOR
    )
    assert result.source_committed
    assert getattr(publication, "projection_error", None) is None
    assert result.source_generation == 1
    assert _counts(target) == dict.fromkeys(INPUT_TABLES, 1)
    assert result.table_counts == dict.fromkeys(INPUT_TABLES, 1)


def test_master_only_profile_leaves_observed_target_rows_untouched(
    tmp_path: Path,
) -> None:
    contract = load_pod_contract()
    rows = {table: [_row(contract, table)] for table in INPUT_TABLES}
    target = tmp_path / "target.sqlite"
    duck_path = tmp_path / "projection.duckdb"

    batch = prepare_pod_import(
        _write_workbook(tmp_path / "pod.xlsx", contract, rows),
        "xlsx",
        contract,
        target_path=target,
    )
    load_prepared_pod_batch(
        batch, sqlite_path=target, duck_path=duck_path, actor=ACTOR, publish=False
    )

    master_only = {
        table: rows[table] for table in ("m_project", "m_pod", "pod_project")
    }
    batch = prepare_pod_import(
        _write_workbook(tmp_path / "master.xlsx", contract, master_only),
        "xlsx",
        contract,
        profile="master-only",
        target_path=target,
        stage_snapshots=False,
    )
    result = load_prepared_pod_batch(
        batch, sqlite_path=target, duck_path=duck_path, actor=ACTOR, publish=False
    )[0]

    assert batch.profile == "master-only"
    assert sum(result.table_counts.values()) == 3
    assert _counts(target) == dict.fromkeys(INPUT_TABLES, 1)


def test_template_metadata_that_disagrees_with_the_contract_is_refused(
    tmp_path: Path,
) -> None:
    contract = load_pod_contract()
    template = generate_pod_import_template(tmp_path / "pod.xlsx", contract=contract)
    workbook = load_workbook(template)
    metadata_sheet = workbook[METADATA_SHEET]
    for row in metadata_sheet.iter_rows(min_row=2, max_col=1):
        key_cell = row[0]
        if key_cell.value == "dictionary_version":
            version_cell = metadata_sheet.cell(row=key_cell.row, column=2)
            assert isinstance(version_cell, Cell)
            version_cell.value = "0.3.9"
    workbook.save(template)
    target = tmp_path / "target.sqlite"

    with pytest.raises(PodImportError, match="dictionary_version"):
        prepare_pod_import(template, "xlsx", contract, target_path=target)

    assert not target.exists()
