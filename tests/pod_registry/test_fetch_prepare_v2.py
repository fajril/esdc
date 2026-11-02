"""Task5C2: POD endpoint transport and PreparedFetch integration regressions.

The application network is mocked; the accepted 5A payload parser and 5B batch
adapter are exercised for real. Every SQLite fixture lives under /private/tmp and
the active operational database is never opened.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest

from esdc.configs import KEY_DESCRIPTIONS, Config
from esdc.fetching import FetchFailure
from esdc.pod_registry import identity
from esdc.pod_registry.api_context import read_pod_api_context
from esdc.pod_registry.api_import import PROTOCOL_OMISSIONS
from esdc.pod_registry.contract import load_pod_contract
from esdc.pod_registry.migrations import initialize_registry_v2
from esdc.selection import ApiVer, FileType, PodApiTable, TableName

CONTRACT = load_pod_contract()

POD_TABLES = (
    PodApiTable.MASTER_DATA_PROJECT,
    PodApiTable.MASTER_DATA_POD,
    PodApiTable.POD_PROJECT,
    PodApiTable.POD_MONITORING,
    PodApiTable.POD_FORECAST,
)
POD_NAMES = {table.value for table in POD_TABLES}
PROJECT_ID = "P-001459F-01"
POD_ID = "PL-1997-0001-4-1-0"
POD_SEQ = 1
POD_UUID = "4aac3a88-10f4-4c74-87d5-12246ce43ca2"
STAGE_LABEL = "Stage III"
STAGE_ID = 3
ACTOR = "FA:fetch"
OBSERVED_AT = datetime(2026, 11, 1, tzinfo=timezone.utc)

_WRITE_VERBS = ("INSERT", "UPDATE", "DELETE", "CREATE", "DROP", "ALTER")


# The canonical minimal values verified against the accepted 5A/5B contract.
CANONICAL_RECORD: dict[str, dict[str, Any]] = {
    "m_project": {
        "project_id": PROJECT_ID,
        "project_name": "Test Project",
        "is_active": 1,
    },
    "m_pod": {
        "pod_id_seq": POD_SEQ,
        "pod_uuid": POD_UUID,
        "approval_date": date(1997, 1, 1),
        "institution_id": 4,
        "pod_type_id": 1,
        "rev_num": 0,
        "pod_order_no": 1,
        "pod_id": POD_ID,
        "pod_name": "POD Test",
        "pod_letter_num": "A",
        "is_latest": 1,
    },
    "pod_project": {
        "pod_id_seq": POD_SEQ,
        "project_id": PROJECT_ID,
        "is_main_project": 1,
        "is_auto_inferred": 0,
    },
    "pod_monitoring": {
        "pod_id_seq": POD_SEQ,
        "pod_id": POD_ID,
        "reported_date": date(1997, 1, 1),
        "is_completed": 0,
        "pod_stage_id": STAGE_ID,
    },
    "pod_forecast": {
        "pod_id_seq": POD_SEQ,
        "year_frcast": 2027,
    },
}

DEFAULT_VALUE = {
    "text": "x",
    "integer": 1,
    "decimal": Decimal("1.5"),
    "boolean": 1,
    "date": date(1997, 1, 1),
    "datetime": OBSERVED_AT,
    "uuid": UUID("00000000-0000-4000-8000-000000000000"),
}
# Surrogate identifiers and the supersession pointer stay blank so the adapter can
# issue or reconcile them; a populated superseded_by must match its master row.
BLANK_FIELDS = frozenset(
    {
        "id",
        "monpod_id",
        "frcast_id",
        "pod_project_uuid",
        "monpod_uuid",
        "frcast_uuid",
        "pod_uuid",
        "superseded_by",
    }
)


def _record(table: str, **overrides: object) -> dict[str, Any]:
    """One canonical input record carrying every header the contract declares."""
    record = {
        column.name: DEFAULT_VALUE[column.logical_type]
        for column in CONTRACT.tables[table].columns
    }
    for name in BLANK_FIELDS:
        if name in record:
            record[name] = None
    record.update(CANONICAL_RECORD[table])
    record.update(overrides)
    return record


def _payload(records: object, table: str) -> bytes:
    """One response body, in the plain array shape or the envelope shape."""
    if isinstance(records, dict):
        return json.dumps(
            {
                "contract_id": CONTRACT.contract_id,
                "contract_version": CONTRACT.contract_version,
                "dictionary_version": CONTRACT.dictionary_version,
                "data": records,
            },
            default=str,
        ).encode()
    return json.dumps(records, default=str).encode()


class Transport:
    """Answers prepare_fetch requests with fixtures and records what it asked for."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import esdc.esdc as esdc

        self.esdc = esdc
        self.payloads: dict[tuple[str, int | None], bytes] = {}
        self.resource_rows: list[dict[str, object]] = [
            {"project_id": "P-001459F-01", "report_year": 2024, "project_name": "Old"},
            {"project_id": "P001459F01", "report_year": 2025, "project_name": "New"},
        ]
        self.timeseries_rows: dict[int | None, list[dict[str, object]]] = {}
        self.config: dict[str, object] = {}
        self.calls: list[tuple[str, FileType, int | None]] = []
        self.urls: list[str] = []
        self.fail: tuple[str, int | None] | None = None
        monkeypatch.setattr(esdc, "_download_table_bytes", self._download)
        monkeypatch.setattr(
            Config, "get_api_url", lambda: "https://esdc.skkmigas.go.id/"
        )
        monkeypatch.setattr(Config, "_load_config", lambda: self.config)
        self._patch_writers(monkeypatch)

    def _patch_writers(self, monkeypatch: pytest.MonkeyPatch) -> None:
        esdc = self.esdc

        def forbidden(name: str):
            def fail(*args, **kwargs):
                pytest.fail(f"{name} must not run during prepare")

            return fail

        for name in (
            "load_data_to_db",
            "run_query",
            "_commit_normalized_project_resources",
            "publish_prepared_fetch",
        ):
            monkeypatch.setattr(esdc, name, forbidden(name))
        for name in ("prepare_pod_import", "load_prepared_pod_batch"):
            monkeypatch.setattr(esdc, name, forbidden(name))
        monkeypatch.setattr(
            "esdc.pod_registry.migrations.initialize_registry_v2",
            lambda *args, **kwargs: pytest.fail(
                "prepare must not initialize a registry"
            ),
        )

    def _download(
        self,
        table: TableName | PodApiTable,
        filetype: FileType,
        username: str,
        password: str,
        report_year: int | None,
    ) -> tuple[bytes, str]:
        name = table.value
        self.calls.append((name, filetype, report_year))
        self.urls.append(
            self.esdc.esdc_url_builder(table_name=table, file_type=filetype)
        )
        if self.fail == (name, report_year):
            raise self.esdc._fetch_failure(
                "download failed (no response payload)",
                table,
                report_year,
                phase="download",
            )
        if name == TableName.PROJECT_RESOURCES.value:
            rows = self.resource_rows
            if filetype == FileType.CSV:
                columns = ["report_year", "project_id", "project_name"]
                body = (
                    ";".join(columns)
                    + "\n"
                    + "".join(
                        ";".join(str(row[column]) for column in columns) + "\n"
                        for row in rows
                    )
                )
                return body.encode(), "fixture://resources"
            return json.dumps(rows).encode(), "fixture://resources"
        if name == TableName.PROJECT_TIMESERIES.value:
            rows = self.timeseries_rows.get(report_year, [])
            if filetype == FileType.CSV:
                body = "report_year;project_id\n" + "".join(
                    f"{row['report_year']};{row['project_id']}\n" for row in rows
                )
                return body.encode(), "fixture://timeseries"
            return json.dumps(rows).encode(), "fixture://timeseries"
        return self.payloads[(name, report_year)], f"fixture://{name}"


def _pod_transport(
    monkeypatch: pytest.MonkeyPatch,
    years: tuple[int, ...] = (2024, 2025),
    master: dict[str, Any] | None = None,
) -> Transport:
    transport = Transport(monkeypatch)
    for table in POD_TABLES:
        record = (
            master
            if master is not None and table == PodApiTable.MASTER_DATA_POD
            else _record(table.value)
        )
        transport.payloads[(table.value, None)] = _payload([record], table.value)
    for year in years:
        transport.payloads[(PodApiTable.POD_MONITORING.value, year)] = _payload(
            [
                _record(
                    PodApiTable.POD_MONITORING.value,
                    reported_date=date(year, 1, 1),
                )
            ],
            PodApiTable.POD_MONITORING.value,
        )
    return transport


def test_prepare_fetch_prepares_all_five_pod_endpoints_and_counts_requests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import esdc.esdc as esdc

    transport = _pod_transport(monkeypatch)
    transport.timeseries_rows = {
        2024: [{"report_year": 2024, "project_id": PROJECT_ID}]
    }
    prepared = esdc.prepare_fetch(
        filetype=FileType.CSV,
        to_file=False,
        username="fake-user",
        password="fake-secret",
        years=[2025, 2025, 2024],
        directory=tmp_path / "spool",
        sqlite_path=tmp_path / "absent.sqlite",
    )

    batch = prepared.pod_batch
    assert batch is not None
    assert prepared.resources_observed is True
    # The masters and the approved forecast are fetched once globally; monitoring
    # follows the deduplicated, sorted selection.
    assert [call for call in transport.calls if call[0].startswith(("m_", "pod_"))] == [
        (PodApiTable.MASTER_DATA_PROJECT.value, FileType.JSON, None),
        (PodApiTable.MASTER_DATA_POD.value, FileType.JSON, None),
        (PodApiTable.POD_PROJECT.value, FileType.JSON, None),
        (PodApiTable.POD_MONITORING.value, FileType.JSON, 2024),
        (PodApiTable.POD_MONITORING.value, FileType.JSON, 2025),
        (PodApiTable.POD_FORECAST.value, FileType.JSON, None),
    ]
    # The snapshot resource and timeseries protocol is unchanged by the POD requests.
    assert [
        call
        for call in transport.calls
        if call[0] in ("project_resources", "project_timeseries")
    ] == [
        ("project_resources", FileType.CSV, 2024),
        ("project_resources", FileType.CSV, 2025),
        ("project_timeseries", FileType.CSV, 2024),
        ("project_timeseries", FileType.CSV, 2025),
    ]
    assert batch.source == "api"
    assert set(batch.rows) == {table.value for table in POD_TABLES}
    assert len(batch.rows[PodApiTable.MASTER_DATA_POD.value]) == 1
    master = batch.rows[PodApiTable.MASTER_DATA_POD.value][0]
    assert master["pod_id"] == POD_ID
    assert master["pod_id_seq"] == POD_SEQ
    assert len(batch.rows[PodApiTable.POD_MONITORING.value]) == 2
    # An absent target is a fresh basis, and the batch must still carry it.
    basis = batch.target_basis
    assert basis is not None
    assert basis.existed is False
    assert basis.target_path == (tmp_path / "absent.sqlite").resolve()
    assert prepared.resource_years == (2024, 2025)
    assert len(prepared.timeseries) == 1
    assert prepared.timeseries[0].report_year == 2024
    assert prepared.skipped_timeseries_years == (2025,)
    # 2025 answered valid-empty and is still recorded as observed.
    assert prepared.timeseries_observed_years == (2024, 2025)


def test_pod_urls_use_api_v1_json_and_configurable_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import esdc.esdc as esdc

    transport = Transport(monkeypatch)
    base = "https://esdc.skkmigas.go.id"

    # A POD table requests JSON even when the snapshot format is CSV.
    assert esdc.esdc_url_builder(
        PodApiTable.MASTER_DATA_POD, file_type=FileType.CSV
    ) == (f"{base}/api/v1/master-data-pod?verbose=3&output=json")
    # Query order is unchanged, and the documented defaults still apply.
    assert (
        esdc.esdc_url_builder(
            PodApiTable.POD_MONITORING, report_year=2024, file_type=FileType.CSV
        )
        == f"{base}/api/v1/pod-monitoring?verbose=3&report-year=2024&output=json"
    )
    assert esdc.esdc_url_builder(PodApiTable.MASTER_DATA_PROJECT) == (
        f"{base}/api/v1/master-data-project?verbose=3&output=json"
    )
    # Legacy snapshot tables keep V2 and the requested format.
    assert (
        esdc.esdc_url_builder(
            TableName.PROJECT_TIMESERIES, report_year=2019, file_type=FileType.CSV
        )
        == f"{base}/api/v2/project-timeseries?verbose=3&report-year=2019&output=csv"
    )
    # An explicit version is honored, and only the monitoring/forecast slugs are
    # configurable.
    assert esdc.esdc_url_builder(PodApiTable.POD_FORECAST, api_ver=ApiVer.V1) == (
        f"{base}/api/v1/pod-forecast?verbose=3&output=json"
    )
    transport.config = {"api": {"pod_monitoring_path": "custom-monitoring"}}
    assert (
        Config.get_pod_endpoint(PodApiTable.POD_MONITORING.value) == "custom-monitoring"
    )
    assert Config.get_pod_endpoint(PodApiTable.POD_FORECAST.value) == "pod-forecast"
    assert (
        Config.get_pod_endpoint(PodApiTable.MASTER_DATA_POD.value) == "master-data-pod"
    )
    # The configured override is what the request actually uses.
    assert esdc.esdc_url_builder(PodApiTable.POD_MONITORING) == (
        f"{base}/api/v1/custom-monitoring?verbose=3&output=json"
    )
    # Only the two configurable slugs are described for config.yaml.
    assert [key for key in KEY_DESCRIPTIONS if key.startswith("api.pod_")] == [
        "api.pod_monitoring_path",
        "api.pod_forecast_path",
    ]


def test_monitoring_follows_selection_and_forecast_year_is_not_filtered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import esdc.esdc as esdc

    transport = _pod_transport(monkeypatch, years=(2024,))
    prepared = esdc.prepare_fetch(
        filetype=FileType.JSON,
        to_file=False,
        username="fake-user",
        password="fake-secret",
        years=[2024],
        directory=tmp_path / "spool",
        sqlite_path=tmp_path / "absent.sqlite",
    )
    batch = prepared.pod_batch
    assert batch is not None
    assert [
        call
        for call in transport.calls
        if call[0] in {table.value for table in POD_TABLES}
    ] == [
        (PodApiTable.MASTER_DATA_PROJECT.value, FileType.JSON, None),
        (PodApiTable.MASTER_DATA_POD.value, FileType.JSON, None),
        (PodApiTable.POD_PROJECT.value, FileType.JSON, None),
        (PodApiTable.POD_MONITORING.value, FileType.JSON, 2024),
        (PodApiTable.POD_FORECAST.value, FileType.JSON, None),
    ]
    # Selecting a year does not restrict which forecast year the answer carries.
    assert batch.rows[PodApiTable.POD_FORECAST.value][0]["year_frcast"] == 2027


def test_unfiltered_monitoring_is_requested_once_full(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import esdc.esdc as esdc

    transport = _pod_transport(monkeypatch)
    prepared = esdc.prepare_fetch(
        filetype=FileType.JSON,
        to_file=False,
        username="fake-user",
        password="fake-secret",
        years=None,
        directory=tmp_path / "spool",
        sqlite_path=tmp_path / "absent.sqlite",
    )
    assert [
        call for call in transport.calls if call[0] == PodApiTable.POD_MONITORING.value
    ] == [(PodApiTable.POD_MONITORING.value, FileType.JSON, None)]
    assert prepared.pod_batch is not None
    assert prepared.timeseries_complete_observation is True


def test_no_year_timeseries_keeps_years_below_the_removed_2020_cutoff(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import esdc.esdc as esdc

    transport = _pod_transport(monkeypatch)
    transport.resource_rows = [{"project_id": PROJECT_ID, "report_year": 2019}]
    transport.timeseries_rows = {
        2019: [{"report_year": 2019, "project_id": PROJECT_ID}]
    }

    prepared = esdc.prepare_fetch(
        filetype=FileType.JSON,
        to_file=False,
        username="fake-user",
        password="fake-secret",
        years=None,
        directory=tmp_path / "spool",
        sqlite_path=tmp_path / "absent.sqlite",
    )

    # 2019 is now a positive request instead of being silently skipped.
    assert [call for call in transport.calls if call[0] == "project_timeseries"] == [
        ("project_timeseries", FileType.JSON, 2019)
    ]
    assert prepared.skipped_timeseries_years == ()
    assert prepared.timeseries_observed_years == (2019,)
    assert prepared.pod_batch is not None


def test_save_writes_pod_bodies_as_exact_bytes_once_per_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import esdc.esdc as esdc

    transport = _pod_transport(monkeypatch, years=(2024, 2025))
    # A 17-integer-digit + 3-fraction token for NUMERIC(20,3), and a 20-digit token for
    # a DOUBLE PRECISION column. Both stay in the body as the raw source wrote them.
    numeric = _record(
        "m_pod",
        capital_drilling_pod="@@NUMERIC@@",
        res_oil_1p_pod="@@DOUBLE@@",
        res_oil_2p_pod=None,
    )
    transport.payloads[(PodApiTable.MASTER_DATA_POD.value, None)] = (
        _payload([numeric], PodApiTable.MASTER_DATA_POD.value)
        .replace(b'"@@NUMERIC@@"', b"12345678901234567.123")
        .replace(b'"@@DOUBLE@@"', b"12345678901234567890.5")
    )

    saved = tmp_path / "db"
    saved.mkdir()
    monkeypatch.setattr(Config, "get_db_dir", lambda: saved)

    prepared = esdc.prepare_fetch(
        filetype=FileType.CSV,
        to_file=True,
        username="fake-user",
        password="fake-secret",
        years=[2024, 2025],
        directory=tmp_path / "spool",
        sqlite_path=tmp_path / "absent.sqlite",
    )

    # A POD body keeps its exact bytes, including raw numeric tokens, and the global
    # scopes are saved once while each monitoring year stays distinct.
    master = saved / "m_pod.json"
    assert (
        master.read_bytes()
        == transport.payloads[(PodApiTable.MASTER_DATA_POD.value, None)]
    )
    assert b"12345678901234567.123" in master.read_bytes()
    assert b"12345678901234567890.5" in master.read_bytes()
    assert [
        path.name for path in sorted(saved.iterdir()) if path.suffix == ".json"
    ] == [
        "m_pod.json",
        "m_project.json",
        "pod_forecast.json",
        "pod_monitoring_2024.json",
        "pod_monitoring_2025.json",
        "pod_project.json",
    ]
    assert (
        saved.joinpath("pod_monitoring_2024.json").read_bytes()
        == transport.payloads[(PodApiTable.POD_MONITORING.value, 2024)]
    )
    # The snapshot formats are unchanged by the POD request format.
    assert [path.name for path in sorted(saved.iterdir()) if path.suffix == ".csv"] == [
        "project_resources_2024.csv",
        "project_resources_2025.csv",
        "project_timeseries_2024.csv",
        "project_timeseries_2025.csv",
    ]
    assert prepared.pod_batch is not None
    # Exact bytes are the durable guarantee for both tokens. 5D must consume this
    # validated PreparedPodBatch and apply the original bytes, not re-encode a value
    # decoded through a float.
    assert b'"capital_drilling_pod": 12345678901234567.123' in master.read_bytes()
    assert b'"res_oil_1p_pod": 12345678901234567890.5' in master.read_bytes()


def test_numeric_column_precision_stays_exact_and_double_uses_float_semantics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import esdc.esdc as esdc

    transport = _pod_transport(monkeypatch, years=(2024,))
    numeric = _record(
        "m_pod",
        capital_drilling_pod="@@NUMERIC@@",
        res_oil_1p_pod="@@DOUBLE@@",
        res_oil_2p_pod=None,
    )
    transport.payloads[(PodApiTable.MASTER_DATA_POD.value, None)] = (
        _payload([numeric], PodApiTable.MASTER_DATA_POD.value)
        .replace(b'"@@NUMERIC@@"', b"12345678901234567.123")
        .replace(b'"@@DOUBLE@@"', b"12345678901234567890.5")
    )

    prepared = esdc.prepare_fetch(
        filetype=FileType.JSON,
        to_file=False,
        username="fake-user",
        password="fake-secret",
        years=[2024],
        directory=tmp_path / "spool",
        sqlite_path=tmp_path / "absent.sqlite",
    )

    assert prepared.pod_batch is not None
    row = prepared.pod_batch.rows[PodApiTable.MASTER_DATA_POD.value][0]
    # NUMERIC(20,3) keeps its declared precision and scale: the validated value is a
    # Decimal holding all 20 digits, exactly as the source token wrote it.
    exact = row["capital_drilling_pod"]
    assert isinstance(exact, Decimal)
    assert exact == Decimal("12345678901234567.123")
    assert exact.as_tuple().exponent == -3
    # DOUBLE PRECISION has float semantics by contract, so the full 20-digit precision
    # lives in the exact saved bytes, not in the normalized value.
    approximate = row["res_oil_1p_pod"]
    assert isinstance(approximate, float)
    assert approximate == float("12345678901234567890.5")


def test_invalid_pod_response_does_not_overwrite_a_prior_good_save(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import esdc.esdc as esdc

    transport = _pod_transport(monkeypatch, years=(2024,))
    saved = tmp_path / "db"
    saved.mkdir()
    monkeypatch.setattr(Config, "get_db_dir", lambda: saved)
    prior = saved / "m_pod.json"
    prior.write_bytes(b"[{}]")
    transport.payloads[(PodApiTable.MASTER_DATA_POD.value, None)] = (
        b'{"error": "unavailable"}'
    )

    with pytest.raises(FetchFailure) as error:
        esdc.prepare_fetch(
            filetype=FileType.JSON,
            to_file=True,
            username="fake-user",
            password="fake-secret",
            years=[2024],
            directory=tmp_path / "spool",
            sqlite_path=tmp_path / "absent.sqlite",
        )

    assert prior.read_bytes() == b"[{}]"
    assert error.value.phase == "parse"
    assert error.value.table == PodApiTable.MASTER_DATA_POD.value
    # A rejected wrapper body never reaches the writer or the saved POD scope.
    assert [
        path.name for path in saved.iterdir() if path.name.startswith(("m_", "pod_"))
    ] == ["m_pod.json"]


def test_failure_context_names_table_year_and_phase_without_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import esdc.esdc as esdc

    transport = _pod_transport(monkeypatch, years=(2025,))
    transport.fail = (PodApiTable.POD_MONITORING.value, 2025)

    with pytest.raises(FetchFailure) as error:
        esdc.prepare_fetch(
            filetype=FileType.JSON,
            to_file=False,
            username="fake-user",
            password="fake-secret",
            years=[2025],
            directory=tmp_path / "spool",
            sqlite_path=tmp_path / "absent.sqlite",
        )

    failure = error.value
    assert failure.phase == "download"
    assert failure.table == PodApiTable.POD_MONITORING.value
    assert failure.report_year == 2025
    assert "fake-secret" not in str(failure) and "fake-user" not in str(failure)
    # The batch aborts at the last failed endpoint; nothing downstream was produced.
    assert not any(
        call[0] == PodApiTable.POD_FORECAST.value for call in transport.calls
    )


def test_single_year_validation_failure_keeps_the_record_index_and_request_year(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import esdc.esdc as esdc

    transport = _pod_transport(monkeypatch, years=(2025,))
    record = _record(PodApiTable.POD_MONITORING.value, reported_date="2025-12-31")
    record.pop("lpg_yield")
    transport.payloads[(PodApiTable.POD_MONITORING.value, 2025)] = _payload(
        [record], PodApiTable.POD_MONITORING.value
    )

    with pytest.raises(FetchFailure) as error:
        esdc.prepare_fetch(
            filetype=FileType.JSON,
            to_file=False,
            username="fake-user",
            password="fake-secret",
            years=[2025],
            directory=tmp_path / "spool",
            sqlite_path=tmp_path / "absent.sqlite",
        )

    failure = error.value
    assert failure.phase == "validation"
    assert failure.table == PodApiTable.POD_MONITORING.value
    # One selected year means the record's request scope is known, so it is reported.
    assert failure.report_year == 2025
    message = str(failure)
    assert "record 1" in message and "lpg_yield" in message
    assert "fake-secret" not in message and "fake-user" not in message


def test_multi_year_validation_failure_names_the_selected_years_and_combined_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import esdc.esdc as esdc

    transport = _pod_transport(monkeypatch, years=(2024, 2025))
    bad = _record(PodApiTable.POD_MONITORING.value, reported_date="2025-12-31")
    bad.pop("lpg_yield")
    transport.payloads[(PodApiTable.POD_MONITORING.value, 2025)] = _payload(
        [bad], PodApiTable.POD_MONITORING.value
    )

    with pytest.raises(FetchFailure) as error:
        esdc.prepare_fetch(
            filetype=FileType.JSON,
            to_file=False,
            username="fake-user",
            password="fake-secret",
            years=[2024, 2025],
            directory=tmp_path / "spool",
            sqlite_path=tmp_path / "absent.sqlite",
        )

    failure = error.value
    assert failure.phase == "validation"
    assert failure.table == PodApiTable.POD_MONITORING.value
    # The combined record index is the one the adapter validated against, and the exact
    # year is not reliably assignable, so every selected year is named instead of a
    # silent None.
    message = str(failure)
    assert failure.report_year is None
    assert "record 2" in message
    assert "2024" in message and "2025" in message
    assert "lpg_yield" in message


def test_global_master_validation_failure_keeps_the_year_scope_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import esdc.esdc as esdc

    transport = _pod_transport(monkeypatch, years=(2024,))
    record = _record(PodApiTable.MASTER_DATA_POD.value)
    record.pop("pod_name")
    transport.payloads[(PodApiTable.MASTER_DATA_POD.value, None)] = _payload(
        [record], PodApiTable.MASTER_DATA_POD.value
    )

    with pytest.raises(FetchFailure) as error:
        esdc.prepare_fetch(
            filetype=FileType.JSON,
            to_file=False,
            username="fake-user",
            password="fake-secret",
            years=[2024],
            directory=tmp_path / "spool",
            sqlite_path=tmp_path / "absent.sqlite",
        )

    failure = error.value
    assert failure.phase == "validation"
    assert failure.table == PodApiTable.MASTER_DATA_POD.value
    # A global master has no year scope, and nothing is invented for it.
    assert failure.report_year is None
    assert "record 1" in str(failure) and "pod_name" in str(failure)


def test_download_boundary_builds_the_pod_url_and_keeps_credentials_out_of_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import esdc.esdc as esdc

    requested: list[tuple[str, str, str]] = []

    def downloader(url: str, username: str, password: str) -> bytes | None:
        requested.append((url, username, password))
        return None if "approved-forecast" in url else b"[]"

    monkeypatch.setattr(esdc, "esdc_downloader", downloader)
    monkeypatch.setattr(Config, "get_api_url", lambda: "https://esdc.skkmigas.go.id/")
    monkeypatch.setattr(
        Config,
        "_load_config",
        lambda: {"api": {"pod_forecast_path": "approved-forecast"}},
    )

    data, url = esdc._download_table_bytes(
        PodApiTable.POD_MONITORING, FileType.CSV, "fake-user", "fake-secret", 2024
    )
    assert data == b"[]"
    assert url == (
        "https://esdc.skkmigas.go.id/api/v1/pod-monitoring?verbose=3&report-year=2024&output=json"
    )
    # The existing base/auth contract is reused unchanged, and auth is never in the URL.
    assert requested[0] == (url, "fake-user", "fake-secret")

    with pytest.raises(FetchFailure) as error:
        esdc._download_table_bytes(
            PodApiTable.POD_FORECAST, FileType.CSV, "fake-user", "fake-secret", None
        )

    failure = error.value
    assert failure.phase == "download"
    assert failure.table == PodApiTable.POD_FORECAST.value
    # The configured override is what the request used, and the message carries no
    # credentials.
    assert "approved-forecast" in requested[1][0]
    assert "fake-user" not in str(failure) and "fake-secret" not in str(failure)


def test_absent_target_and_absent_parent_create_no_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import esdc.esdc as esdc

    _pod_transport(monkeypatch)
    initialized: list[int] = []
    monkeypatch.setattr(
        "esdc.pod_registry.migrations.initialize_registry_v2",
        lambda *args, **kwargs: initialized.append(1),
    )

    target = tmp_path / "absent" / "esdc.sqlite"
    prepared = esdc.prepare_fetch(
        filetype=FileType.JSON,
        to_file=False,
        username="fake-user",
        password="fake-secret",
        years=[2024],
        directory=tmp_path / "spool",
        sqlite_path=target,
    )

    assert not target.parent.exists()
    assert not target.exists()
    assert initialized == []
    assert prepared.pod_batch is not None
    basis = prepared.pod_batch.target_basis
    assert basis is not None
    assert basis.existed is False


def test_corrupt_target_is_refused_before_any_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import esdc.esdc as esdc

    transport = _pod_transport(monkeypatch)
    target = tmp_path / "corrupt.sqlite"
    target.write_bytes(b"not a sqlite database")

    with pytest.raises(FetchFailure) as error:
        esdc.prepare_fetch(
            filetype=FileType.JSON,
            to_file=False,
            username="fake-user",
            password="fake-secret",
            years=[2024],
            directory=tmp_path / "spool",
            sqlite_path=target,
        )

    assert error.value.phase == "preflight"
    assert "Could not read POD target" in str(error.value)
    assert transport.calls == []


def test_legacy_registry_target_is_refused_before_any_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import esdc.esdc as esdc

    transport = _pod_transport(monkeypatch)
    target = tmp_path / "legacy.sqlite"
    conn = sqlite3.connect(target)
    conn.execute("CREATE TABLE m_pod (pod_id TEXT PRIMARY KEY)")
    conn.commit()
    conn.close()

    with pytest.raises(FetchFailure) as error:
        esdc.prepare_fetch(
            filetype=FileType.JSON,
            to_file=False,
            username="fake-user",
            password="fake-secret",
            years=[2024],
            directory=tmp_path / "spool",
            sqlite_path=target,
        )

    assert error.value.phase == "preflight"
    assert "explicit validated replacement" in str(error.value)
    # The current store refuses an old-shaped target before any request or write.
    assert "operational store serves only the current POD schema" in str(error.value)
    assert transport.calls == []


def _sql_value(value: object) -> object:
    return str(value) if isinstance(value, (Decimal, UUID, date, datetime)) else value


def _insert(conn: sqlite3.Connection, table: str, row: dict[str, object]) -> None:
    names = [
        column.name for column in CONTRACT.tables[table].columns if column.name in row
    ]
    conn.execute(
        f"INSERT INTO {table} ({', '.join(names)}) VALUES "
        f"({', '.join('?' for _ in names)})",
        tuple(_sql_value(row[name]) for name in names),
    )


def _v2_target(path: Path) -> tuple[sqlite3.Connection, dict[str, Any]]:
    """Test setup only: a v2 registry under /private/tmp, never the active database."""
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA foreign_keys = ON")
    initialize_registry_v2(conn)
    issued = identity.issue_identity(
        conn,
        key=identity.ApprovalKey(
            approval_date=date(1997, 1, 1),
            daily_slot=1,
            institution_id=4,
            pod_type_id=1,
            rev_num=0,
        ),
        actor=ACTOR,
    )
    # The stored master and the payload that reconciles it must state the same
    # permanent identity the ledger issued.
    master = _record(
        "m_pod",
        pod_id_seq=issued.pod_id_seq,
        pod_uuid=str(issued.pod_uuid),
        pod_order_no=issued.pod_order_no,
        pod_id=issued.pod_id,
    )
    _insert(conn, "m_pod", master)
    _insert(conn, "m_project", _record("m_project"))
    conn.commit()
    return conn, master


def test_existing_target_keeps_its_basis_and_preserves_unselected_years(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import esdc.esdc as esdc

    target = tmp_path / "esdc.sqlite"
    conn, master = _v2_target(target)
    _insert(
        conn,
        "pod_monitoring",
        _record(
            PodApiTable.POD_MONITORING.value,
            monpod_id=11,
            monpod_uuid="00000011-1111-4111-8111-111111111111",
            reported_date=date(2023, 6, 1),
        ),
    )
    _insert(
        conn,
        "pod_monitoring",
        _record(
            PodApiTable.POD_MONITORING.value,
            monpod_id=12,
            monpod_uuid="00000012-1112-4112-8112-111111111112",
            reported_date=date(2024, 6, 1),
        ),
    )
    conn.commit()
    conn.close()

    transport = _pod_transport(monkeypatch, years=(2024,), master=master)
    # 2024 answers valid-empty, so the prior 2024 row is cleared while 2023 is kept.
    transport.payloads[(PodApiTable.POD_MONITORING.value, 2024)] = b"[]"
    transport.resource_rows = [{"project_id": PROJECT_ID, "report_year": 2023}]

    prepared = esdc.prepare_fetch(
        filetype=FileType.JSON,
        to_file=False,
        username="fake-user",
        password="fake-secret",
        years=[2024],
        directory=tmp_path / "spool",
        sqlite_path=target,
    )

    batch = prepared.pod_batch
    assert batch is not None
    basis = batch.target_basis
    assert basis is not None
    assert basis.existed is True
    assert basis.target_path == target.resolve()
    assert prepared.resource_years == (2024,)
    assert prepared.skipped_resource_years == (2024,)
    monitoring = batch.rows[PodApiTable.POD_MONITORING.value]
    assert [row["reported_date"] for row in monitoring] == [date(2023, 6, 1)]
    # 2024 is recorded as requested-and-empty, not as never requested.
    assert prepared.timeseries_observed_years == (2024,)
    assert prepared.skipped_timeseries_years == (2024,)
    # The batch is built from the same read snapshot the reader captured.
    assert batch.target_basis == read_pod_api_context(target, CONTRACT).target_basis


def _linked_target(tmp_path: Path) -> tuple[Path, dict[str, Any]]:
    """A valid v2 target: one stored master and one document link that satisfies it today."""
    target = tmp_path / "esdc.sqlite"
    conn, master = _v2_target(target)
    conn.execute(
        "INSERT INTO pod_document (pod_id_seq, doc_id) VALUES (?, 'DOC-1')", (POD_SEQ,)
    )
    # A stored observation in the year that will be selected, so the only thing the
    # selected scope clears is this year's row.
    _insert(
        conn,
        "pod_monitoring",
        _record(
            PodApiTable.POD_MONITORING.value,
            monpod_id=11,
            monpod_uuid="00000011-1111-4111-8111-111111111111",
            reported_date=date(2025, 6, 1),
        ),
    )
    conn.commit()
    conn.close()
    return target, master


def test_stored_master_does_not_rescue_a_link_the_new_source_omits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import esdc.esdc as esdc

    target, master = _linked_target(tmp_path)
    before = hashlib.sha256(target.read_bytes()).digest()

    transport = _pod_transport(monkeypatch, years=(2025,), master=master)
    transport.resource_rows = []
    # Every endpoint answers valid-empty for the requested scope, which is valid on its
    # own; only the protected link check can fail here.
    for table in POD_TABLES:
        transport.payloads[(table.value, None)] = b"[]"
    transport.payloads[(PodApiTable.POD_MONITORING.value, 2025)] = b"[]"

    saved = tmp_path / "db"
    saved.mkdir()
    monkeypatch.setattr(Config, "get_db_dir", lambda: saved)
    prior = saved / "m_pod.json"
    prior.write_bytes(b"[{}]")

    with pytest.raises(FetchFailure) as error:
        esdc.prepare_fetch(
            filetype=FileType.JSON,
            to_file=True,
            username="fake-user",
            password="fake-secret",
            years=[2025],
            directory=tmp_path / "spool",
            sqlite_path=target,
        )

    failure = error.value
    assert failure.phase == "preflight", str(failure)
    assert failure.table == PodApiTable.MASTER_DATA_POD.value
    assert failure.report_year is None
    assert str(POD_SEQ) in str(failure)
    assert "crosswalk" in str(failure) and "never inferred" in str(failure)
    # The abort is a preflight after every request, not a skipped request: the batch
    # itself is never produced.
    assert [call[0] for call in transport.calls if call[0] in POD_NAMES] == [
        table.value for table in POD_TABLES
    ]
    # Nothing was written: the target is byte-for-byte as found, and the prior good save
    # survives. A stored master row is not a rescue, and no numeric crosswalk is invented.
    assert hashlib.sha256(target.read_bytes()).digest() == before
    assert prior.read_bytes() == b"[{}]"
    assert [
        path.name for path in saved.iterdir() if path.name.startswith(("m_", "pod_"))
    ] == ["m_pod.json"]
    check = sqlite3.connect(target)
    try:
        assert check.execute("SELECT count(*) FROM m_pod").fetchone()[0] == 1
        assert check.execute("SELECT count(*) FROM pod_document").fetchone()[0] == 1
    finally:
        check.close()


def test_link_check_passes_when_the_source_retains_the_referenced_pod(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import esdc.esdc as esdc

    target, master = _linked_target(tmp_path)
    before = hashlib.sha256(target.read_bytes()).digest()

    # The same permanent identity the ledger issued is stated by the source, so the link
    # stays covered by the prepared masters alone.
    transport = _pod_transport(monkeypatch, years=(2025,), master=master)

    prepared = esdc.prepare_fetch(
        filetype=FileType.JSON,
        to_file=False,
        username="fake-user",
        password="fake-secret",
        years=[2025],
        directory=tmp_path / "spool",
        sqlite_path=target,
    )

    batch = prepared.pod_batch
    assert batch is not None
    assert [
        row["pod_id_seq"] for row in batch.rows[PodApiTable.MASTER_DATA_POD.value]
    ] == [POD_SEQ]
    # The retained master is what covers the link, and the link check itself ran after
    # all five requests.
    assert [call[0] for call in transport.calls if call[0] in POD_NAMES] == [
        table.value for table in POD_TABLES
    ]
    assert hashlib.sha256(target.read_bytes()).digest() == before


def test_stage_label_projection_and_compact_project_id_canonicalization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import esdc.esdc as esdc

    transport = _pod_transport(monkeypatch, years=(2024,))
    # The physical API view shape carries every declared business header but none of
    # the local protocol columns; the stage arrives as its label only.
    view_row = {
        name: value
        for name, value in _record(
            PodApiTable.POD_MONITORING.value, reported_date="2024-12-31"
        ).items()
        if name not in PROTOCOL_OMISSIONS
    }
    view_row["pod_stage"] = STAGE_LABEL
    transport.payloads[(PodApiTable.POD_MONITORING.value, 2024)] = _payload(
        [view_row],
        PodApiTable.POD_MONITORING.value,
    )
    transport.resource_rows = [
        {"project_id": "P001459F01", "report_year": 2024, "project_name": "Compact"},
        {"project_id": "001", "report_year": 2024, "project_name": "Plain"},
    ]

    prepared = esdc.prepare_fetch(
        filetype=FileType.JSON,
        to_file=False,
        username="fake-user",
        password="fake-secret",
        years=[2024],
        directory=tmp_path / "spool",
        sqlite_path=tmp_path / "absent.sqlite",
    )

    assert [row.project_id for row in prepared.resource_rows] == [PROJECT_ID, "001"]
    batch = prepared.pod_batch
    assert batch is not None
    assert batch.rows[PodApiTable.POD_MONITORING.value][0]["pod_stage_id"] == STAGE_ID


def test_no_year_run_requests_every_observed_resource_year_in_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import esdc.esdc as esdc

    transport = _pod_transport(monkeypatch)
    transport.resource_rows = [
        {"project_id": PROJECT_ID, "report_year": 2025, "project_name": "Latest"},
        {"project_id": PROJECT_ID, "report_year": 2019, "project_name": "Earliest"},
    ]

    prepared = esdc.prepare_fetch(
        filetype=FileType.JSON,
        to_file=False,
        username="fake-user",
        password="fake-secret",
        years=None,
        directory=tmp_path / "spool",
        sqlite_path=tmp_path / "absent.sqlite",
    )

    # Both observed years are requested, sorted, with no lower cutoff applied.
    assert [call[2] for call in transport.calls if call[0] == "project_timeseries"] == [
        2019,
        2025,
    ]
    assert prepared.timeseries_observed_years == (2019, 2025)
    assert prepared.timeseries_complete_observation
    # An unfiltered POD monitoring scope is requested once, not per observed year.
    assert [
        call for call in transport.calls if call[0] == PodApiTable.POD_MONITORING.value
    ] == [(PodApiTable.POD_MONITORING.value, FileType.JSON, None)]


def test_prepare_leaves_the_target_file_exactly_as_found(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import esdc.esdc as esdc

    target = tmp_path / "esdc.sqlite"
    conn, master = _v2_target(target)
    conn.close()
    before = hashlib.sha256(target.read_bytes()).digest()
    check = sqlite3.connect(target)
    journal_mode = check.execute("PRAGMA journal_mode").fetchone()[0]
    check.close()

    transport = _pod_transport(monkeypatch, years=(2024,), master=master)
    prepared = esdc.prepare_fetch(
        filetype=FileType.JSON,
        to_file=False,
        username="fake-user",
        password="fake-secret",
        years=[2024],
        directory=tmp_path / "spool",
        sqlite_path=target,
    )

    assert prepared.pod_batch is not None
    # Preparation reads only: the main file and its journal mode are unchanged, and the
    # WAL index a read connection opens carries no frames.
    assert hashlib.sha256(target.read_bytes()).digest() == before
    wal = Path(f"{target}-wal")
    assert not wal.exists() or wal.stat().st_size == 0
    check = sqlite3.connect(target)
    try:
        assert check.execute("PRAGMA journal_mode").fetchone()[0] == journal_mode
    finally:
        check.close()
    # The read snapshot still produced the batch, so the guard is not vacuous.
    assert [call[0] for call in transport.calls if call[0] in POD_NAMES] == [
        table.value for table in POD_TABLES
    ]
