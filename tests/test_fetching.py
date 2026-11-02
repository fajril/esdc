"""Tests for strict fetch preparation and operation-owned timeseries spools."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from esdc.fetching import (
    FetchFailure,
    iter_spooled_rows,
    parse_report_year,
    spool_timeseries,
)
from esdc.selection import FileType, PodApiTable, TableName


def _pod_payload(table: TableName | PodApiTable) -> tuple[bytes, str]:
    """A POD answer is JSON; an empty array keeps these preparation tests scoped."""
    return b"[]", f"fixture://{table.value}"


@pytest.mark.parametrize(
    ("value", "expected"),
    [(2025, 2025), ("2025", 2025), (" 2025 ", 2025)],
)
def test_parse_report_year_normalizes_supported_values(value, expected) -> None:
    assert parse_report_year(value) == expected


@pytest.mark.parametrize(
    "value",
    [True, False, 2025.0, None, "", "  ", "999", "0999", "0000", "10000", "20.5", "-25", "٢٠٢٥", [], {}],
)
def test_parse_report_year_rejects_invalid_values(value) -> None:
    with pytest.raises(ValueError):
        parse_report_year(value)


def test_spooled_rows_keep_json_scalar_fidelity_and_scope_by_actual_year(
    tmp_path: Path,
) -> None:
    header = ("report_year", "project_name", "project_id", "rate_oil", "report_status")
    rows = [
        ("2024", "other year", "P0", 0, None),
        ("2025", "Minyak\nbumi – Indonesia", "001", 0, ""),
    ]

    prepared = spool_timeseries(header, rows, report_year=2025, directory=tmp_path)

    assert prepared is not None
    assert prepared.header == header
    assert prepared.report_year == 2025
    assert prepared.row_count == 1
    assert list(iter_spooled_rows(prepared)) == [
        (2025, "Minyak\nbumi – Indonesia", "001", 0, "")
    ]
    assert prepared.rows_path.read_bytes().endswith(b"\n")


@pytest.mark.parametrize(
    ("header", "rows", "message"),
    [
        (("project_id",), [("P1",)], "report_year"),
        (("report_year", "project_id", "project_id"), [(2025, "P1", "P1")], "duplicate"),
        (("report_year", "not_a_column"), [(2025, "x")], "unknown"),
        (("report_year", "project_id"), [(2025,)], "width"),
        (("report_year", "project_id"), [(2025, ["not", "scalar"])], "scalar"),
        (("report_year", "project_id"), [(2025.0, "P1")], "report_year"),
    ],
)
def test_spool_rejects_invalid_schema_and_rows(tmp_path, header, rows, message) -> None:
    with pytest.raises((ValueError, FetchFailure), match=message):
        spool_timeseries(header, rows, report_year=2025, directory=tmp_path)


def test_nonfinite_out_of_requested_year_rejected_before_filtering(tmp_path: Path) -> None:
    header = ("report_year", "project_remarks", "cprd_grs_oil")
    for invalid in (float("nan"), float("inf"), float("-inf"), "NaN", " +INF ", "-Infinity"):
        with pytest.raises(ValueError, match="cprd_grs_oil"):
            spool_timeseries(
                header,
                [(2024, "ordinary note", invalid)],
                report_year=2025,
                directory=tmp_path,
            )

    narrative = spool_timeseries(
        header,
        [(2025, "NaN and Infinity are narrative words", 1.0)],
        report_year=2025,
        directory=tmp_path,
    )
    assert narrative is not None


def test_header_only_timeseries_is_a_valid_empty_slice(tmp_path: Path) -> None:
    assert spool_timeseries(("report_year", "project_id"), [], report_year=2025, directory=tmp_path) is None


def test_prepare_fetch_uses_named_json_fields_and_deduplicates_years(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import esdc.esdc as esdc

    calls: list[tuple[TableName, int | None]] = []

    def download(table, filetype, username, password, report_year):
        calls.append((table, report_year))
        if isinstance(table, PodApiTable):
            return _pod_payload(table)
        if table == TableName.PROJECT_RESOURCES:
            return json.dumps(
                [
                    {"project_id": "000", "report_year": "2024", "project_name": "Other"},
                    {"project_id": "001", "report_year": " 2025 ", "project_name": "Test"},
                ]
            ).encode(), "fixture://resources"
        return json.dumps(
            [
                {"report_year": "2025", "project_id": "001", "cprd_grs_oil": 0},
                {"cprd_grs_oil": 0, "project_id": "002", "report_year": 2025},
            ]
        ).encode(), "fixture://timeseries"

    monkeypatch.setattr(esdc, "_download_table_bytes", download)
    monkeypatch.setattr(esdc, "load_data_to_db", lambda *args, **kwargs: pytest.fail("DB mutation"))
    monkeypatch.setattr(esdc, "_commit_normalized_project_resources", lambda *args, **kwargs: pytest.fail("DB mutation"))

    prepared = esdc.prepare_fetch(
        filetype=FileType.JSON,
        to_file=False,
        username="fake",
        password="fake",
        years=[2025, 2025],
        directory=tmp_path,
        sqlite_path=tmp_path / "pod-target.sqlite",
    )

    assert [call for call in calls if not isinstance(call[0], PodApiTable)] == [
        (TableName.PROJECT_RESOURCES, 2025),
        (TableName.PROJECT_TIMESERIES, 2025),
    ]
    # The masters and the approved forecast stay global; monitoring follows the year.
    assert [call for call in calls if isinstance(call[0], PodApiTable)] == [
        (PodApiTable.MASTER_DATA_PROJECT, None),
        (PodApiTable.MASTER_DATA_POD, None),
        (PodApiTable.POD_PROJECT, None),
        (PodApiTable.POD_MONITORING, 2025),
        (PodApiTable.POD_FORECAST, None),
    ]
    assert prepared.resource_rows[0].report_year == 2025
    assert prepared.resource_rows[0].project_id == "001"
    assert len(prepared.resource_rows) == 1
    # A selected year that is requested and validated counts as observed.
    assert prepared.timeseries_observed_years == (2025,)
    assert not prepared.timeseries_complete_observation
    assert list(iter_spooled_rows(prepared.timeseries[0])) == [
        (2025, "001", 0),
        (2025, "002", 0),
    ]


def test_prepare_fetch_csv_matches_json_year_and_row_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import esdc.esdc as esdc

    def download(table, filetype, username, password, report_year):
        if isinstance(table, PodApiTable):
            return _pod_payload(table)
        if table == TableName.PROJECT_RESOURCES:
            return b"report_year;project_id;project_name\n2025;001;Test\n", "fixture://resources"
        return b"report_year;project_id\n2025;001\n", "fixture://timeseries"

    monkeypatch.setattr(esdc, "_download_table_bytes", download)

    prepared = esdc.prepare_fetch(
        filetype=FileType.CSV,
        to_file=False,
        username="fake",
        password="fake",
        years=[2025],
        directory=tmp_path,
        sqlite_path=tmp_path / "pod-target.sqlite",
    )

    assert prepared.resource_rows[0].report_year == 2025
    assert list(iter_spooled_rows(prepared.timeseries[0])) == [(2025, "001")]
    assert prepared.timeseries[0].origin == "api_csv"
    # The POD batch is part of the prepared fetch even for a CSV snapshot run.
    assert prepared.pod_batch is not None


def test_unfiltered_fetch_records_complete_timeseries_observation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import esdc.esdc as esdc

    def download(table, filetype, username, password, report_year):
        if table == TableName.PROJECT_RESOURCES:
            return b'[{"project_id":"P","report_year":2025}]', "fixture://resources"
        return b'[]', "fixture://timeseries"

    monkeypatch.setattr(esdc, "_download_table_bytes", download)

    prepared = esdc.prepare_fetch(
        filetype=FileType.JSON,
        to_file=False,
        username="fake",
        password="fake",
        years=None,
        directory=tmp_path,
        sqlite_path=tmp_path / "pod-target.sqlite",
    )

    assert prepared.timeseries_complete_observation
    assert prepared.timeseries_observed_years == (2025,)
    assert prepared.skipped_timeseries_years == (2025,)


def test_unfiltered_fetch_without_timeseries_requests_is_not_an_observation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import esdc.esdc as esdc

    calls: list[TableName | PodApiTable] = []

    def download(table, filetype, username, password, report_year):
        calls.append(table)
        if isinstance(table, PodApiTable):
            return _pod_payload(table)
        return b"[]", "fixture://resources"

    monkeypatch.setattr(esdc, "_download_table_bytes", download)

    prepared = esdc.prepare_fetch(
        filetype=FileType.JSON,
        to_file=False,
        username="fake",
        password="fake",
        years=None,
        directory=tmp_path,
        sqlite_path=tmp_path / "pod-target.sqlite",
    )

    # Nothing observed means no timeseries request, but the POD endpoints still run.
    assert [table for table in calls if not isinstance(table, PodApiTable)] == [
        TableName.PROJECT_RESOURCES
    ]
    assert [table for table in calls if isinstance(table, PodApiTable)] == [
        PodApiTable.MASTER_DATA_PROJECT,
        PodApiTable.MASTER_DATA_POD,
        PodApiTable.POD_PROJECT,
        PodApiTable.POD_MONITORING,
        PodApiTable.POD_FORECAST,
    ]
    assert prepared.timeseries == ()
    assert prepared.timeseries_observed_years == ()
    assert not prepared.timeseries_complete_observation
    assert prepared.pod_batch is not None


def test_prepare_fetch_saves_only_after_validating_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import esdc.esdc as esdc

    save_path = tmp_path / "project_resources.json"
    save_path.write_bytes(b"previous valid payload")
    monkeypatch.setattr(esdc.Config, "get_db_dir", staticmethod(lambda: tmp_path))
    monkeypatch.setattr(
        esdc,
        "_download_table_bytes",
        lambda *args: (
            b'[{"project_id":"P","report_year":"20x5"}]',
            "fixture://resources",
        ),
    )

    with pytest.raises(FetchFailure, match="report_year"):
        esdc.prepare_fetch(
            filetype=FileType.JSON,
            to_file=True,
            username="fake",
            password="fake",
            years=None,
            directory=tmp_path / "spool",
            sqlite_path=tmp_path / "pod-target.sqlite",
        )

    assert save_path.read_bytes() == b"previous valid payload"


def test_full_empty_resources_skip_timeseries_but_selected_empty_does_not(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import esdc.esdc as esdc

    calls: list[tuple[TableName | PodApiTable, int | None]] = []

    def download(table, filetype, username, password, report_year):
        calls.append((table, report_year))
        if isinstance(table, PodApiTable):
            return _pod_payload(table)
        if table == TableName.PROJECT_RESOURCES:
            return b"[]", "fixture://resources"
        return b'[{"report_year":"2025","project_id":"P"}]', "fixture://timeseries"

    monkeypatch.setattr(esdc, "_download_table_bytes", download)
    full = esdc.prepare_fetch(
        filetype=FileType.JSON,
        to_file=False,
        username="fake",
        password="fake",
        years=None,
        directory=tmp_path / "full",
        sqlite_path=tmp_path / "pod-target.sqlite",
    )
    assert full.resource_rows == ()
    assert full.timeseries == ()
    assert [call for call in calls if not isinstance(call[0], PodApiTable)] == [
        (TableName.PROJECT_RESOURCES, None)
    ]
    # Empty resources no longer skip the five POD endpoints.
    assert len([call for call in calls if isinstance(call[0], PodApiTable)]) == 5
    assert not full.timeseries_complete_observation

    calls.clear()
    selected = esdc.prepare_fetch(
        filetype=FileType.JSON,
        to_file=False,
        username="fake",
        password="fake",
        years=[2025],
        directory=tmp_path / "selected",
        sqlite_path=tmp_path / "pod-target.sqlite",
    )
    assert selected.resource_rows == ()
    # A valid-empty selected year is recorded as requested, not as never requested.
    assert selected.resource_years == (2025,)
    assert selected.skipped_resource_years == (2025,)
    assert len(selected.timeseries) == 1
    assert [call for call in calls if not isinstance(call[0], PodApiTable)] == [
        (TableName.PROJECT_RESOURCES, 2025),
        (TableName.PROJECT_TIMESERIES, 2025),
    ]
    assert [call for call in calls if isinstance(call[0], PodApiTable)] == [
        (PodApiTable.MASTER_DATA_PROJECT, None),
        (PodApiTable.MASTER_DATA_POD, None),
        (PodApiTable.POD_PROJECT, None),
        (PodApiTable.POD_MONITORING, 2025),
        (PodApiTable.POD_FORECAST, None),
    ]


def test_empty_json_timeseries_is_a_skipped_slice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import esdc.esdc as esdc

    def download(table, filetype, username, password, report_year):
        if table == TableName.PROJECT_RESOURCES:
            return b'[{"project_id":"P","report_year":2025}]', "fixture://resources"
        return b"[]", "fixture://timeseries"

    monkeypatch.setattr(esdc, "_download_table_bytes", download)

    prepared = esdc.prepare_fetch(
        filetype=FileType.JSON,
        to_file=False,
        username="fake",
        password="fake",
        years=None,
        directory=tmp_path,
        sqlite_path=tmp_path / "pod-target.sqlite",
    )

    assert prepared.timeseries == ()
    assert prepared.skipped_timeseries_years == (2025,)


def test_fetch_table_none_is_contextual_for_either_table(monkeypatch: pytest.MonkeyPatch) -> None:
    import esdc.esdc as esdc

    monkeypatch.setattr(esdc, "esdc_url_builder", lambda **kwargs: "fixture://missing")
    monkeypatch.setattr(esdc, "esdc_downloader", lambda *args: None)
    for table in (TableName.PROJECT_RESOURCES, TableName.PROJECT_TIMESERIES):
        with pytest.raises(FetchFailure) as error:
            esdc._fetch_and_parse_table(
                table, FileType.JSON, False, "fake", "fake", report_year=2025
            )
        assert error.value.phase == "download"
        assert error.value.table == table.value
        assert error.value.report_year == 2025


@pytest.mark.parametrize(
    ("filetype", "payload"),
    [(FileType.JSON, b"[]"), (FileType.CSV, b"report_year;project_id\n")],
)
def test_legacy_table_fetch_returns_none_for_valid_empty_timeseries(
    monkeypatch: pytest.MonkeyPatch, filetype: FileType, payload: bytes
) -> None:
    import esdc.esdc as esdc

    monkeypatch.setattr(esdc, "esdc_url_builder", lambda **kwargs: "fixture://empty")
    monkeypatch.setattr(esdc, "esdc_downloader", lambda *args: payload)

    assert (
        esdc._fetch_and_parse_table(
            TableName.PROJECT_TIMESERIES,
            filetype,
            False,
            "fake",
            "fake",
            report_year=2025,
        )
        is None
    )


def test_prepare_fetch_aborts_before_returning_batch_on_second_year_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import esdc.esdc as esdc

    calls: list[tuple[TableName, int | None]] = []

    def download(table, filetype, username, password, report_year):
        calls.append((table, report_year))
        if table == TableName.PROJECT_TIMESERIES and report_year == 2025:
            raise FetchFailure(
                "project_timeseries report year 2025: download failed (no response payload)",
                phase="download",
                table=table.value,
                report_year=report_year,
            )
        if table == TableName.PROJECT_RESOURCES:
            return json.dumps([{"project_id": "P", "report_year": report_year}]).encode(), "fixture://resources"
        return json.dumps([{"report_year": report_year, "project_id": "P"}]).encode(), "fixture://timeseries"

    monkeypatch.setattr(esdc, "_download_table_bytes", download)

    with pytest.raises(FetchFailure, match="report year 2025") as error:
        esdc.prepare_fetch(
            filetype=FileType.JSON,
            to_file=False,
            username="fake",
            password="fake",
            years=[2024, 2025],
            directory=tmp_path,
            sqlite_path=tmp_path / "pod-target.sqlite",
        )

    assert error.value.phase == "download"
    assert error.value.table == TableName.PROJECT_TIMESERIES.value
    assert error.value.report_year == 2025
    assert calls[-2:] == [
        (TableName.PROJECT_TIMESERIES, 2024),
        (TableName.PROJECT_TIMESERIES, 2025),
    ]
