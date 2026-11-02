"""Lossless POD business workbook from one read-only SQLite snapshot."""

from __future__ import annotations

import os
from collections.abc import Iterable, Sequence
from contextlib import closing, suppress
from pathlib import Path
from tempfile import mkstemp
from typing import Any

from openpyxl import Workbook
from openpyxl.cell import WriteOnlyCell
from openpyxl.cell.rich_text import CellRichText

from esdc.pod_registry.api_context import (
    column_names,
    object_type,
    quote_identifier,
    read_only_connection,
)
from esdc.pod_registry.contract import load_pod_contract
from esdc.pod_registry.migrations import _validate_existing_v2_registry
from esdc.pod_registry.sqlite_import import (
    INPUT_TABLES,
    OUTPUT_TABLES,
    POD_BASIS_SHEET,
    POD_GUIDE_SHEET,
    POD_LEDGER_HEADERS,
    POD_LEDGER_SHEET,
    POD_METADATA_SHEET,
    POD_REFERENCE_TABLES,
    _same_file,
)

_MAX_ROWS = 1048576
_MAX_TEXT = 32767
_GRAINS = {
    "pod_monitoring_api": ("pod_id_seq", "reported_date"),
    "pod_forecast_api": ("pod_id_seq", "reported_date", "year_frcast"),
}


def _append(sheet, values: Sequence[Any], headers: Sequence[str]) -> None:
    index = sheet._esdc_row_number + 1
    if index > _MAX_ROWS:
        raise ValueError(f"{sheet.title}: Excel row limit exceeded")
    cells = []
    for header, value in zip(headers, values, strict=True):
        if (
            isinstance(value, int)
            and not isinstance(value, bool)
            and len(str(abs(value))) >= 16
        ):
            value = str(value)
        if isinstance(value, str) and len(value) > _MAX_TEXT:
            raise ValueError(
                f"{sheet.title} row {index} column {header}: text "
                f"exceeds {_MAX_TEXT} characters"
            )
        # A rich empty string survives read_only/data_only reload as '', not NULL.
        cell = WriteOnlyCell(sheet, value=value)
        if value == "":
            cell.value = CellRichText([""])
        if isinstance(value, str):
            cell.data_type = "s"
        cells.append(cell)
    sheet.append(cells)
    sheet._esdc_row_number = index


def _sheet(
    workbook,
    name: str,
    headers: Sequence[str],
    rows: Iterable[Sequence[Any]],
    *,
    readonly: bool,
):
    sheet = workbook.create_sheet(name)
    sheet._esdc_row_number = 0
    sheet.freeze_panes = "A2"
    sheet.protection.sheet = readonly
    _append(sheet, headers, headers)
    for values in rows:
        _append(sheet, values, headers)
    return sheet


def _cleanup_workbook(workbook) -> None:
    # openpyxl only cleans streaming XML automatically after save succeeds.
    # These writers and their temporary paths belong exclusively to this workbook.
    for sheet in workbook.worksheets:
        writer = getattr(sheet, "_writer", None)
        if writer is not None:
            with suppress(Exception):
                if not sheet.closed:
                    sheet.close()
                writer.cleanup()
    workbook.close()


def export_pod_backup(
    sqlite_path: Path | str, output_path: Path | str, *, overwrite: bool = False
) -> Path:
    """Export POD facts, derived results, refs and permanent identity reservations.

    Raw project resources/timeseries and corpus documents are excluded. This is
    a POD business workbook, not a full database backup. Result sheets provide
    the captured basis; loading regenerates outputs from available local sources.
    """
    source, output = Path(sqlite_path), Path(output_path)
    if not source.is_file():
        raise ValueError(f"SQLite source does not exist: {source}")
    if _same_file(source, output):
        raise ValueError("Source and workbook destination must be distinct")
    if output.suffix.lower() != ".xlsx":
        raise ValueError("POD workbook destination must end in .xlsx")
    if output.exists() and not overwrite:
        raise FileExistsError(output)
    contract = load_pod_contract()
    workbook = Workbook(write_only=True)
    temp: Path | None = None
    try:
        with closing(read_only_connection(source)) as conn:
            _validate_existing_v2_registry(conn)
            for name in (*INPUT_TABLES, *OUTPUT_TABLES):
                table = contract.tables[name]
                headers = tuple(c.name for c in table.columns)
                order = _GRAINS.get(name, table.primary_key)
                projection = ",".join(map(quote_identifier, headers))
                ordering = ",".join(map(quote_identifier, order))
                cursor = conn.execute(
                    f"SELECT {projection} FROM {quote_identifier(name)} "
                    f"ORDER BY {ordering}"
                )
                boolean_indexes = {
                    i
                    for i, c in enumerate(table.columns)
                    if c.logical_type == "boolean"
                }
                _sheet(
                    workbook,
                    name,
                    headers,
                    (
                        tuple(
                            bool(v) if i in boolean_indexes and v is not None else v
                            for i, v in enumerate(row)
                        )
                        for row in cursor
                    ),
                    readonly=name in OUTPUT_TABLES,
                )
            for name in POD_REFERENCE_TABLES:
                headers = tuple(column_names(conn, name))
                _sheet(
                    workbook,
                    name,
                    headers,
                    conn.execute(f"SELECT * FROM {quote_identifier(name)} ORDER BY id"),
                    readonly=True,
                )
            projection = ",".join(
                "actor AS issued_by" if h == "issued_by" else quote_identifier(h)
                for h in POD_LEDGER_HEADERS
            )
            _sheet(
                workbook,
                POD_LEDGER_SHEET,
                POD_LEDGER_HEADERS,
                conn.execute(
                    f"SELECT {projection} FROM _pod_identity_ledger "
                    f"ORDER BY approval_year,pod_order_no"
                ),
                readonly=True,
            )
            basis = workbook.create_sheet(POD_BASIS_SHEET)
            basis._esdc_row_number = 0
            basis.protection.sheet = True
            _append(basis, ("key", "value"), ("key", "value"))
            for name in (
                "pod_registry_state",
                "pod_read_model_snapshot",
                "api_snapshot",
                "api_timeseries_snapshot",
            ):
                if object_type(conn, name) != "table":
                    continue
                headers = column_names(conn, name)
                ordering = ",".join(map(quote_identifier, headers))
                for index, row in enumerate(
                    conn.execute(
                        f"SELECT * FROM {quote_identifier(name)} ORDER BY {ordering}"
                    )
                ):
                    for header, value in zip(headers, row, strict=True):
                        _append(
                            basis, (f"{name}.{index}.{header}", value), ("key", "value")
                        )
            _sheet(
                workbook,
                POD_METADATA_SHEET,
                ("key", "value"),
                (
                    ("contract_id", contract.contract_id),
                    ("contract_version", str(contract.contract_version)),
                    ("dictionary_version", contract.dictionary_version),
                ),
                readonly=True,
            )
            _sheet(
                workbook,
                POD_GUIDE_SHEET,
                ("topic", "guidance"),
                (
                    ("Inputs", ", ".join(INPUT_TABLES)),
                    (
                        "Generated fields",
                        "Keep issued identifiers, UUIDs and audit fields unchanged.",
                    ),
                    (
                        "Outputs",
                        "Derived output sheets are read-only snapshots; "
                        "import regenerates them from local sources.",
                    ),
                    (
                        "References",
                        "Reference sheets are read-only; SQL seeds in "
                        "ESDC define their values.",
                    ),
                    (
                        "Identity ledger",
                        "Keep the technical ledger unchanged, including "
                        "reservations for deleted approvals.",
                    ),
                    (
                        "Precision",
                        "Decimals and long identifiers are text. "
                        "Preserve their exact values.",
                    ),
                    ("NULL", "An empty string differs from a missing cell (NULL)."),
                    (
                        "Dates and IRR",
                        "Use ISO dates. IRR values are fractions: 0.15 represents 15%.",
                    ),
                    (
                        "Excluded",
                        "Raw project resources, timeseries and corpus "
                        "documents are excluded. Fetch raw sources when needed.",
                    ),
                    (
                        "Import",
                        "esdc load --schema-pod restores POD inputs; "
                        "API fetch replaces imported facts.",
                    ),
                ),
                readonly=True,
            )
            output.parent.mkdir(parents=True, exist_ok=True)
            fd, filename = mkstemp(
                prefix=".esdc-pod-", suffix=".xlsx", dir=output.parent
            )
            os.close(fd)
            temp = Path(filename)
            workbook.save(temp)
            if overwrite:
                os.replace(temp, output)
            else:
                os.link(temp, output)
            return output
    finally:
        _cleanup_workbook(workbook)
        if temp is not None:
            temp.unlink(missing_ok=True)
