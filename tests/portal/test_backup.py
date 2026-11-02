"""Actual XLSX round-trip preserves exact facts and orphan identity reservations."""

import sqlite3
from contextlib import closing

import pytest
from openpyxl import load_workbook

from esdc.pod_registry.changesets import apply_changeset
from esdc.pod_registry.sqlite_import import (
    POD_LEDGER_SHEET,
    PodImportError,
    load_prepared_pod_batch,
    prepare_pod_import,
)
from esdc.pod_registry.store import get_sqlite_connection
from esdc.portal.backup import export_pod_backup


def records(path, table):
    with closing(sqlite3.connect(path)) as conn:
        conn.row_factory = sqlite3.Row
        return [dict(r) for r in conn.execute(f'SELECT * FROM "{table}" ORDER BY 1')]


@pytest.fixture()
def source(tmp_path):
    path = tmp_path / "source.sqlite"
    get_sqlite_connection(path).close()
    approval = {
        "approval_date": "2024-01-01",
        "institution_id": 1,
        "pod_type_id": 1,
        "pod_name": "=SUM(1,2)",
        "pod_letter_num": "POD/1",
        "capital_drilling_pod": "12345678901234567.123",
        "remarks_pod": "",
    }
    result = apply_changeset(
        "m_pod", {"source_generation": 0, "inserts": [approval]}, path
    )
    assert result.ok, result.errors
    first = records(path, "m_pod")[0]
    result = apply_changeset(
        "m_pod",
        {"source_generation": 1, "deletes": [{"pod_id_seq": first["pod_id_seq"]}]},
        path,
    )
    assert result.ok, result.errors
    # Reserve a high surrogate before issuance, respecting the immutable-ID trigger.
    with closing(sqlite3.connect(path)) as conn:
        conn.execute(
            "UPDATE sqlite_sequence SET seq=? WHERE name='m_pod'", (9007199254740992,)
        )
        conn.commit()
    result = apply_changeset(
        "m_pod", {"source_generation": 2, "inserts": [approval]}, path
    )
    assert result.ok, result.errors
    return path


def test_business_workbook_roundtrip_retains_exact_values_and_orphan_ledger(
    source, tmp_path
):
    output = export_pod_backup(source, tmp_path / "backup.xlsx")
    ledger = records(source, "_pod_identity_ledger")
    before = records(source, "m_pod")[0]
    with closing(load_workbook(output, read_only=True, data_only=False)) as workbook:
        sheet = workbook["m_pod"]
        headers = next(sheet.iter_rows(values_only=True))
        values = next(sheet.iter_rows(min_row=2))
        data = dict(zip(headers, values, strict=True))
        assert data["pod_id_seq"].value == "9007199254740993"
        assert (
            data["pod_name"].value == "=SUM(1,2)" and data["pod_name"].data_type == "s"
        )
        assert data["capital_drilling_pod"].value == "12345678901234567.123"
        assert data["remarks_pod"].value == "" and data["onstream_date"].value is None
        assert len(list(workbook[POD_LEDGER_SHEET].values)) == 3
    target = tmp_path / "restore.sqlite"
    prepared = prepare_pod_import(output, "xlsx", target_path=target)
    assert not target.exists()
    assert len(prepared.identity_ledger or []) == 2
    result, _ = load_prepared_pod_batch(
        prepared, sqlite_path=target, duck_path=tmp_path / "cache.duckdb", publish=False
    )
    assert result.source_committed
    after = records(target, "m_pod")[0]
    for field in (
        "pod_id_seq",
        "pod_uuid",
        "pod_id",
        "capital_drilling_pod",
        "remarks_pod",
        "onstream_date",
    ):
        assert after[field] == before[field], field
    assert records(target, "_pod_identity_ledger") == ledger
    added = apply_changeset(
        "m_pod",
        {
            "source_generation": result.source_generation,
            "inserts": [
                {
                    "approval_date": "2024-01-01",
                    "institution_id": 1,
                    "pod_type_id": 1,
                    "pod_name": "Next",
                    "pod_letter_num": "3",
                }
            ],
        },
        target,
    )
    assert added.ok, added.errors
    assert records(target, "_pod_identity_ledger")[-1]["daily_slot"] == 3
    assert records(source, "m_pod")[0] == before


def test_failed_export_preserves_destination_and_cleans_owned_temps(
    source, tmp_path, monkeypatch
):
    from esdc.portal import backup

    output = tmp_path / "existing.xlsx"
    output.write_bytes(b"preserve me")
    monkeypatch.setattr(backup, "_MAX_ROWS", 1)
    with pytest.raises(ValueError, match="row limit"):
        export_pod_backup(source, output, overwrite=True)
    assert output.read_bytes() == b"preserve me"
    assert not list(tmp_path.glob(".esdc-pod-*.xlsx"))
    with pytest.raises(FileExistsError):
        export_pod_backup(source, output)
    with pytest.raises(ValueError, match="distinct"):
        export_pod_backup(source, source, overwrite=True)


def test_malformed_ledger_or_unknown_sheet_refuses_before_target_write(
    source, tmp_path
):
    output = export_pod_backup(source, tmp_path / "backup.xlsx")
    with closing(load_workbook(output)) as workbook:
        workbook[POD_LEDGER_SHEET].cell(1, 1, "unexpected")
        workbook.save(output)
    target = tmp_path / "missing.sqlite"
    with pytest.raises(PodImportError, match="canonical identity ledger columns"):
        prepare_pod_import(output, "xlsx", target_path=target)
    assert not target.exists()
    export_pod_backup(source, output, overwrite=True)
    with closing(load_workbook(output)) as workbook:
        workbook.create_sheet("arbitrary")
        workbook.save(output)
    with pytest.raises(PodImportError, match="unknown POD workbook"):
        prepare_pod_import(output, "xlsx", target_path=target)
    assert not target.exists()
