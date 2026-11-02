"""Current POD value semantics on the public path; the retired value-case model is gone."""

from __future__ import annotations

import sqlite3
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from esdc.pod_registry.contract import load_pod_contract
from esdc.pod_registry.sqlite_import import (
    PodImportError,
    load_prepared_pod_batch,
    prepare_pod_import,
)
from esdc.pod_registry.store import get_sqlite_readonly_connection
from tests.pod_registry.test_importer import (
    ACTOR,
    INPUT_TABLES,
    _counts,
    _row,
    _write_workbook,
)

RETIRED_VOCABULARY = frozenset(
    {"case_type", "report_date", "as_of_date", "effective_date", "revision_effect"}
)


def _import(path: Path, contract, target: Path, duck_path: Path):
    """Run the public import path without publishing, returning the import result."""
    batch = prepare_pod_import(path, "xlsx", contract, target_path=target)
    return load_prepared_pod_batch(
        batch, sqlite_path=target, duck_path=duck_path, actor=ACTOR, publish=False
    )[0]


def test_canonical_contract_has_no_value_case_table_or_retired_vocabulary() -> None:
    contract = load_pod_contract()
    assert "pod_value_case" not in contract.tables
    for table in contract.tables.values():
        assert not RETIRED_VOCABULARY & {column.name for column in table.columns}
    pod_columns = {column.name for column in contract.tables["m_pod"].columns}
    assert {"approval_date", "is_latest", "superseded_by"} <= pod_columns
    assert "pod_value_case" not in " ".join(contract.notes)


def test_incompatible_workbook_is_refused_before_any_write(tmp_path: Path) -> None:
    contract = load_pod_contract()
    workbook = _write_workbook(
        tmp_path / "pod.xlsx",
        contract,
        {table: [_row(contract, table)] for table in INPUT_TABLES},
        extra_header=("m_pod", "not_declared"),
    )
    target = tmp_path / "target.sqlite"

    with pytest.raises(PodImportError, match="not declared for m_pod"):
        prepare_pod_import(workbook, "xlsx", contract, target_path=target)

    assert not target.exists()


def test_unknown_domain_value_is_refused_before_any_write(tmp_path: Path) -> None:
    contract = load_pod_contract()
    rows = {table: [_row(contract, table)] for table in INPUT_TABLES}
    rows["m_pod"][0]["institution_id"] = 7
    workbook = _write_workbook(tmp_path / "pod.xlsx", contract, rows)
    target = tmp_path / "target.sqlite"

    with pytest.raises(PodImportError, match="institution_id") as exc:
        prepare_pod_import(workbook, "xlsx", contract, target_path=target)

    assert any("reference" in message for message in exc.value.errors)
    assert not target.exists()


def test_full_import_then_valid_empty_input_clears_observed_tables(
    tmp_path: Path,
) -> None:
    contract = load_pod_contract()
    rows = {table: [_row(contract, table)] for table in INPUT_TABLES}
    target = tmp_path / "target.sqlite"
    duck_path = tmp_path / "projection.duckdb"

    _import(
        _write_workbook(tmp_path / "pod.xlsx", contract, rows),
        contract,
        target,
        duck_path,
    )
    assert _counts(target) == dict.fromkeys(INPUT_TABLES, 1)

    empty = dict.fromkeys(INPUT_TABLES, [])
    empty["m_pod"] = rows["m_pod"]
    empty["m_project"] = rows["m_project"]
    result = _import(
        _write_workbook(tmp_path / "empty.xlsx", contract, empty),
        contract,
        target,
        duck_path,
    )

    counts = _counts(target)
    assert counts["pod_monitoring"] == 0
    assert counts["pod_forecast"] == 0
    assert counts["m_pod"] == 1
    assert sum(result.table_counts.values()) == 2


def test_refused_batch_after_a_good_import_preserves_committed_rows(
    tmp_path: Path,
) -> None:
    contract = load_pod_contract()
    rows = {table: [_row(contract, table)] for table in INPUT_TABLES}
    target = tmp_path / "target.sqlite"
    duck_path = tmp_path / "projection.duckdb"

    generation = _import(
        _write_workbook(tmp_path / "good.xlsx", contract, rows),
        contract,
        target,
        duck_path,
    ).source_generation

    stale = {table: [dict(row) for row in rows[table]] for table in INPUT_TABLES}
    stale["m_pod"][0]["institution_id"] = 7
    with pytest.raises(PodImportError):
        prepare_pod_import(
            _write_workbook(tmp_path / "bad.xlsx", contract, stale),
            "xlsx",
            contract,
            target_path=target,
        )

    conn = get_sqlite_readonly_connection(target)
    assert conn is not None
    try:
        state = tuple(
            conn.execute(
                "SELECT source_generation FROM pod_registry_state WHERE singleton_id=1"
            ).fetchone()
        )
        pod_name = tuple(
            conn.execute(
                "SELECT pod_name FROM m_pod WHERE pod_id = ?",
                ("PL-2024-0001-1-1-0",),
            ).fetchone()
        )
    finally:
        conn.close()
    assert state == (generation,)
    assert pod_name == ("example",)


def test_fixed_scale_decimal_round_trips_exactly_through_the_public_path(
    tmp_path: Path,
) -> None:
    contract = load_pod_contract()
    table_name = next(
        name
        for name, table in contract.tables.items()
        if any(column.scale for column in table.columns)
    )
    column = next(
        column for column in contract.tables[table_name].columns if column.scale
    )
    scale = column.scale
    assert scale is not None
    digits = "34567890123456"[:scale]
    expected = Decimal(f"12.{digits}")
    source = tmp_path / "source.sqlite"
    target = tmp_path / "target.sqlite"

    conn = sqlite3.connect(source)
    try:
        for name in INPUT_TABLES:
            names = [c.name for c in contract.tables[name].columns]
            conn.execute(
                f'CREATE TABLE "{name}" (' + ", ".join(f'"{n}"' for n in names) + ")"
            )
        for name in INPUT_TABLES:
            names = [c.name for c in contract.tables[name].columns]
            row: dict[str, Any] = _row(contract, name)
            if name == table_name:
                row[column.name] = expected
            conn.execute(
                f'INSERT INTO "{name}" VALUES (' + ",".join("?" for _ in names) + ")",
                [
                    format(value, "f")
                    if isinstance(value, Decimal)
                    else value.isoformat()
                    if isinstance(value, date)
                    else value
                    for value in (row[n] for n in names)
                ],
            )
        conn.commit()
    finally:
        conn.close()

    batch = prepare_pod_import(
        source, "sqlite", contract, target_path=target, stage_snapshots=False
    )
    load_prepared_pod_batch(
        batch,
        sqlite_path=target,
        duck_path=tmp_path / "projection.duckdb",
        actor=ACTOR,
        publish=False,
    )

    conn = get_sqlite_readonly_connection(target)
    assert conn is not None
    try:
        stored = conn.execute(f'SELECT "{column.name}" FROM "{table_name}"').fetchone()[
            0
        ]
    finally:
        conn.close()
    assert Decimal(str(stored)) == expected
    # The fixed scale is preserved exactly: a binary float would leave residue here.
    assert Decimal(str(stored)) * (10**scale) == Decimal(f"12{digits}")


def test_date_columns_store_date_only_and_refuse_a_time_component(
    tmp_path: Path,
) -> None:
    contract = load_pod_contract()
    rows = {table: [_row(contract, table)] for table in INPUT_TABLES}
    target = tmp_path / "target.sqlite"
    duck_path = tmp_path / "projection.duckdb"

    midnight = {table: [dict(row) for row in rows[table]] for table in INPUT_TABLES}
    midnight["m_pod"][0]["approval_date"] = datetime(2024, 1, 1, 0, 0)
    _write_workbook(tmp_path / "midnight.xlsx", contract, midnight)
    _import(tmp_path / "midnight.xlsx", contract, target, duck_path)

    conn = get_sqlite_readonly_connection(target)
    assert conn is not None
    try:
        stored = conn.execute(
            "SELECT approval_date FROM m_pod WHERE pod_id = ?",
            ("PL-2024-0001-1-1-0",),
        ).fetchone()[0]
    finally:
        conn.close()
    assert stored == "2024-01-01"

    timed = {table: [dict(row) for row in rows[table]] for table in INPUT_TABLES}
    timed["m_pod"][0]["approval_date"] = datetime(2024, 1, 1, 9, 30)
    before = tmp_path / "timed.xlsx"
    _write_workbook(before, contract, timed)
    refused_target = tmp_path / "refused.sqlite"

    with pytest.raises(PodImportError, match="time component"):
        prepare_pod_import(before, "xlsx", contract, target_path=refused_target)

    assert not refused_target.exists()
