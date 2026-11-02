"""Task5B POD API projection, normalization, and prepared batch regressions."""

from __future__ import annotations

import json
import sqlite3
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_DNS, UUID, uuid5

import pytest

from esdc.pod_registry import api_import
from esdc.pod_registry.api_import import (
    PodApiImportError,
    business_natural_key,
    natural_key_uuid,
    pod_stage_labels,
    prepare_pod_api_batch,
)
from esdc.pod_registry.api_payload import PodApiPayloadError
from esdc.pod_registry.contract import load_pod_contract
from esdc.pod_registry.models import PreparedPodBatch

CONTRACT = load_pod_contract()
INPUT_TABLES = ("m_project", "m_pod", "pod_project", "pod_monitoring", "pod_forecast")
POD_ID = "PL-1997-0001-4-1-0"
POD_UUID = UUID("4aac3a88-10f4-4c74-87d5-12246ce43ca2")
MON_UUID = UUID("11111111-5111-4111-8111-111111111111")
PROJ_UUID = UUID("22222222-5222-4222-8222-222222222222")
FRC_UUID = UUID("33333333-5333-4333-8333-333333333333")
LEDGER_UUID = UUID("44444444-5444-4444-8444-444444444444")
PROJECT_ID = "P-2403431-01"
ACTOR = "FA:fetch"
OBSERVED_AT = datetime(2026, 10, 31, 12, 0, 0, tzinfo=timezone.utc)
PRIOR_ACTOR = "Manual:editor"
PRIOR_OBSERVED_AT = datetime(2026, 10, 1, tzinfo=timezone.utc)

_DEFAULTS: dict[str, dict[str, Any]] = {
    "m_project": {
        "project_id": PROJECT_ID,
        "project_name": "Test Project",
        "is_active": 1,
    },
    "m_pod": {
        "pod_id_seq": 1,
        "pod_uuid": str(POD_UUID),
        "approval_date": "1997-01-01",
        "institution_id": 4,
        "pod_type_id": 1,
        "rev_num": 0,
        "pod_order_no": 1,
        "pod_id": POD_ID,
        "pod_name": "POD Test",
        "pod_letter_num": "A",
        "is_latest": True,
    },
    "pod_project": {
        "id": 1,
        "pod_project_uuid": str(PROJ_UUID),
        "pod_id_seq": 1,
        "project_id": PROJECT_ID,
        "is_main_project": 1,
        "is_auto_inferred": 0,
    },
    "pod_monitoring": {
        "monpod_id": 1,
        "monpod_uuid": str(MON_UUID),
        "pod_id_seq": 1,
        "pod_id": POD_ID,
        "reported_date": "2025-12-31",
        "is_completed": False,
        "pod_stage_id": 3,
        "pod_stage": "Stage III",
    },
    "pod_forecast": {
        "frcast_id": 1,
        "frcast_uuid": str(FRC_UUID),
        "pod_id_seq": 1,
        "year_frcast": 2025,
    },
}


def _record(table: str, **overrides: Any) -> dict[str, Any]:
    """A record shaped like the canonical input contract."""
    names = {column.name for column in CONTRACT.tables[table].columns}
    record = dict.fromkeys(names)
    record.update(
        {name: value for name, value in _DEFAULTS[table].items() if name in names}
    )
    record.update(overrides)
    return record


def _view_record(table: str, **overrides: Any) -> dict[str, Any]:
    """A record shaped like the physical API view with its protocol omissions."""
    names = {column.name for column in CONTRACT.tables[f"{table}_api"].columns}
    record = dict.fromkeys(names)
    record.update(
        {name: value for name, value in _DEFAULTS[table].items() if name in names}
    )
    record.update(overrides)
    return record


def _payloads(**tables: Any) -> dict[str, Any]:
    payloads: dict[str, Any] = {}
    for name in INPUT_TABLES:
        value = tables.get(name)
        if value is None:
            payloads[name] = json.dumps([_record(name)]).encode()
        elif isinstance(value, bytes):
            payloads[name] = value
        elif value and isinstance(value[0], list):
            payloads[name] = [json.dumps(body).encode() for body in value]
        else:
            payloads[name] = json.dumps(value).encode()
    return payloads


def _prepare(payloads: dict[str, Any] | None = None, **kwargs: Any) -> PreparedPodBatch:
    settings: dict[str, Any] = {"actor": ACTOR, "observed_at": OBSERVED_AT}
    settings.update(kwargs)
    return prepare_pod_api_batch(
        payloads if payloads is not None else _payloads(), **settings
    )


def _rows(batch: PreparedPodBatch, table: str) -> list[dict[str, Any]]:
    return [dict(row) for row in batch.rows[table]]


def test_prepare_returns_the_five_canonical_headers_and_full_profile():
    batch = _prepare(_payloads())
    assert batch.source == "api"
    assert batch.profile == "full"
    assert batch.identity_ledger is None
    assert batch.source_path is None
    assert batch.target_basis is None
    assert batch.staged_snapshots is None
    for table in INPUT_TABLES:
        declared = tuple(column.name for column in CONTRACT.tables[table].columns)
        assert tuple(batch.headers[table]) == declared
        assert len(batch.rows[table]) == 1
        assert all(key in declared for key in _rows(batch, table)[0])


def test_payload_keys_are_exactly_the_five_canonical_input_tables():
    payloads = _payloads()
    payloads.pop("m_pod")
    with pytest.raises(PodApiImportError) as exc_info:
        _prepare(payloads)
    assert "payload key 'm_pod'" in str(exc_info.value)
    payloads = _payloads()
    payloads["pod_monitoring_api"] = b"[]"
    with pytest.raises(PodApiImportError) as exc_info:
        _prepare(payloads)
    assert "not a canonical POD input table" in str(exc_info.value)


def test_global_tables_take_one_body_and_monitoring_accepts_many():
    payloads = _payloads(
        pod_monitoring=[
            [_record("pod_monitoring")],
            [
                _record(
                    "pod_monitoring",
                    monpod_id=2,
                    monpod_uuid=str(LEDGER_UUID),
                    reported_date="2025-06-30",
                )
            ],
        ]
    )
    batch = _prepare(payloads)
    assert len(_rows(batch, "pod_monitoring")) == 2
    payloads = _payloads(m_pod=[_record("m_pod"), _record("m_pod", pod_id_seq=2)])
    payloads["m_pod"] = [payloads["m_pod"], payloads["m_pod"]]
    with pytest.raises(PodApiImportError) as exc_info:
        _prepare(payloads)
    assert "one payload body" in str(exc_info.value)


def test_non_bytes_payload_body_rejects():
    payloads = _payloads()
    payloads["m_pod"] = "[]"
    with pytest.raises(PodApiImportError) as exc_info:
        _prepare(payloads)
    assert "raw bytes" in str(exc_info.value)


def test_parser_rejects_before_projection():
    payloads = _payloads(m_pod=b"[NaN]")
    with pytest.raises(PodApiPayloadError) as exc_info:
        _prepare(payloads)
    assert exc_info.value.table_name == "m_pod"
    assert "non-finite" in str(exc_info.value)


def test_unknown_column_rejects_for_every_table():
    record = _record("m_project")
    record["unsupported"] = "extra"
    with pytest.raises(PodApiImportError) as exc_info:
        _prepare(_payloads(m_project=[record]))
    assert "not declared" in str(exc_info.value)
    record = _view_record("pod_monitoring")
    record["unknown_view_column"] = 1
    with pytest.raises(PodApiImportError) as exc_info:
        _prepare(_payloads(pod_monitoring=[record]))
    assert "not declared" in str(exc_info.value)


def test_missing_business_header_rejects_even_when_nullable():
    record = _view_record("pod_monitoring")
    del record["opex_real"]
    with pytest.raises(PodApiImportError) as exc_info:
        _prepare(_payloads(pod_monitoring=[record]))
    assert "missing declared" in str(exc_info.value)
    assert "opex_real" in str(exc_info.value)


def test_view_projection_keeps_owned_input_columns_and_drops_copied_master_fields():
    master = _record("m_pod", opex_pod="1.235")
    view = _view_record("pod_monitoring", opex_pod="1.24", opex_real="1.240")
    batch = _prepare(_payloads(m_pod=[master], pod_monitoring=[view]))
    row = _rows(batch, "pod_monitoring")[0]
    assert "opex_pod" not in row
    assert row["opex_real"] == Decimal("1.240")
    assert _rows(batch, "m_pod")[0]["opex_pod"] == Decimal("1.235")


def test_rounded_master_copy_is_never_compared_as_an_inequality():
    master = _record("m_pod", wap_oil_pod="1.235000")
    view = _view_record("pod_monitoring", wap_oil_pod="1.24", wap_oil_real="1.240000")
    batch = _prepare(_payloads(m_pod=[master], pod_monitoring=[view]))
    row = _rows(batch, "pod_monitoring")[0]
    assert row["wap_oil_real"] == Decimal("1.240000")
    assert _rows(batch, "m_pod")[0]["wap_oil_pod"] == Decimal("1.235000")
    assert "wap_oil_pod" not in row


def test_stage_labels_resolve_from_the_packaged_reference_seed():
    labels = pod_stage_labels(CONTRACT)
    assert labels == {"Stage I": 1, "Stage II": 2, "Stage III": 3, "Stage IV": 4}
    source = Path(api_import.__file__).read_text(encoding="utf-8")
    assert "pod_reference_seed.sql" in source
    for label in ("Stage I", "Stage II", "Stage III", "Stage IV"):
        assert f"'{label}'" not in source


def test_stage_label_projection_and_canonical_stage_id_both_work():
    batch = _prepare(
        _payloads(pod_monitoring=[_view_record("pod_monitoring", pod_stage="Stage II")])
    )
    assert _rows(batch, "pod_monitoring")[0]["pod_stage_id"] == 2
    batch = _prepare(
        _payloads(pod_monitoring=[_record("pod_monitoring", pod_stage_id=1)])
    )
    assert _rows(batch, "pod_monitoring")[0]["pod_stage_id"] == 1
    batch = _prepare(
        _payloads(
            pod_monitoring=[
                _record("pod_monitoring", pod_stage_id=3, pod_stage="Stage III")
            ]
        )
    )
    assert _rows(batch, "pod_monitoring")[0]["pod_stage_id"] == 3


def test_conflicting_stage_label_and_stage_id_reject():
    record = _record("pod_monitoring", pod_stage_id=2, pod_stage="Stage III")
    with pytest.raises(PodApiImportError) as exc_info:
        _prepare(_payloads(pod_monitoring=[record]))
    assert "stage" in str(exc_info.value)


def test_unknown_stage_label_rejects_and_blank_stage_is_not_invented():
    record = _view_record("pod_monitoring", pod_stage="Stage V", opex_real="1.000")
    with pytest.raises(PodApiImportError) as exc_info:
        _prepare(_payloads(pod_monitoring=[record]))
    assert "stage" in str(exc_info.value)
    record = _view_record("pod_monitoring", pod_stage=None, opex_real="1.000")
    with pytest.raises(PodApiImportError) as exc_info:
        _prepare(_payloads(pod_monitoring=[record]))
    assert "stage" in str(exc_info.value)


def test_null_synthetic_monitor_projection_creates_no_input():
    synthetic = _view_record(
        "pod_monitoring",
        monpod_id=None,
        monpod_uuid=None,
        pod_stage=None,
        pod_stage_id=None,
        is_completed=None,
    )
    batch = _prepare(
        _payloads(pod_monitoring=[synthetic, _view_record("pod_monitoring")])
    )
    rows = _rows(batch, "pod_monitoring")
    assert len(rows) == 1
    assert rows[0]["monpod_id"] == 1


def test_real_monitor_id_with_all_unknown_values_is_retained():
    record = _view_record(
        "pod_monitoring", monpod_id=404, monpod_uuid=None, pod_stage="Stage II"
    )
    batch = _prepare(_payloads(pod_monitoring=[record]))
    row = _rows(batch, "pod_monitoring")[0]
    assert row["monpod_id"] == 404
    assert row["opex_real"] is None
    assert row["sow_real"] is None
    assert row["pod_stage_id"] == 2


def test_false_and_zero_are_preserved_as_values_not_missing():
    record = _view_record(
        "pod_monitoring",
        is_completed=False,
        well_total_real=0,
        opex_real="0.000",
        pod_stage="Stage I",
    )
    batch = _prepare(_payloads(pod_monitoring=[record]))
    row = _rows(batch, "pod_monitoring")[0]
    assert row["is_completed"] is False
    assert row["well_total_real"] == 0
    assert row["opex_real"] == Decimal("0.000")


def test_outlook_only_forecast_rows_create_no_approved_input():
    outlook = _view_record("pod_forecast", frcast_id=None, usf_oil_otlk="330.008")
    batch = _prepare(_payloads(pod_forecast=[outlook, _view_record("pod_forecast")]))
    rows = _rows(batch, "pod_forecast")
    assert len(rows) == 1
    assert rows[0]["frcast_id"] == 1


def test_real_forecast_id_with_all_unknown_values_is_retained():
    record = _view_record(
        "pod_forecast", frcast_id=77, frcast_uuid=None, year_frcast=2026
    )
    batch = _prepare(_payloads(pod_forecast=[record]))
    row = _rows(batch, "pod_forecast")[0]
    assert row["frcast_id"] == 77
    assert row["usf_oil_pod"] is None
    assert row["frcast_remarks"] is None


def test_combined_derived_forecast_columns_do_not_establish_a_real_record():
    record = _view_record(
        "pod_forecast",
        frcast_id=None,
        usf_oil_otlk="1.000",
        usf_oc_otlk="2.000",
        usf_an_otlk="3.000",
        pod_name="POD Test",
        reported_date="2025-12-31",
        monpod_id=9,
    )
    batch = _prepare(_payloads(pod_forecast=[record]))
    assert _rows(batch, "pod_forecast") == []


def test_identical_records_across_contexts_dedupe_to_one_canonical_row():
    # The documented exceptions are the m_project same-priority dedup and the
    # pod_forecast repeated report-context dedup.
    batch = _prepare(_payloads(m_project=[dict(_record("m_project"))] * 2))
    assert len(_rows(batch, "m_project")) == 1

    forecast = _view_record(
        "pod_forecast",
        frcast_id=3,
        frcast_uuid=str(FRC_UUID),
        pod_id_seq=1,
        year_frcast=2025,
        usf_oil_pod="0.500000",
    )
    batch = _prepare(_payloads(pod_forecast=[forecast, dict(forecast)]))
    assert len(_rows(batch, "pod_forecast")) == 1

    # A repeated monitoring key is a duplicate key, not a dedupe, even when the two
    # report contexts state identical values.
    monitor = _view_record(
        "pod_monitoring",
        monpod_id=7,
        monpod_uuid=str(MON_UUID),
        pod_id_seq=1,
        reported_date="2025-01-31",
        pod_stage="Stage II",
        opex_real="1.000000",
    )
    with pytest.raises(PodApiImportError, match="duplicate key") as exc_info:
        _prepare(_payloads(pod_monitoring=[monitor, dict(monitor)]))
    message = str(exc_info.value)
    assert "pod_monitoring" in message and "record 2" in message


@pytest.mark.parametrize("table", ("m_pod", "pod_project", "pod_monitoring"))
def test_repeated_keys_reject_even_for_identical_records_in_the_three_tables(
    table: str,
):
    record = _record(table)
    with pytest.raises(PodApiImportError, match="duplicate key") as exc_info:
        _prepare(_payloads(**{table: [record, dict(record)]}))
    message = str(exc_info.value)
    # The error context names the table, the record, and the key.
    assert (
        table in message
        and "record 2" in message
        and table in exc_info.value.table_name
    )


def test_conflicting_values_for_the_same_canonical_row_reject_before_return():
    first = _view_record(
        "pod_forecast",
        frcast_id=3,
        frcast_uuid=str(FRC_UUID),
        pod_id_seq=1,
        year_frcast=2025,
        usf_oil_pod="0.500000",
    )
    second = dict(first, usf_oil_pod="0.600000")
    with pytest.raises(PodApiImportError, match="conflicting"):
        _prepare(_payloads(pod_forecast=[first, second]))

    other = dict(first, frcast_id=4, frcast_uuid=str(LEDGER_UUID), year_frcast=2026)
    batch = _prepare(_payloads(pod_forecast=[first, other]))
    assert len(_rows(batch, "pod_forecast")) == 2


def test_business_natural_keys_follow_the_documented_business_columns():
    batch = _prepare(_payloads())

    master = batch.rows["m_pod"][0]
    assert business_natural_key(CONTRACT, "m_pod", master) == (POD_ID,)
    mapping = batch.rows["pod_project"][0]
    assert business_natural_key(CONTRACT, "pod_project", mapping) == ("1", PROJECT_ID)
    monitor = batch.rows["pod_monitoring"][0]
    assert business_natural_key(CONTRACT, "pod_monitoring", monitor) == (
        "1",
        "2025-12-31",
    )
    forecast = batch.rows["pod_forecast"][0]
    assert business_natural_key(CONTRACT, "pod_forecast", forecast) == ("1", "2025")
    project = batch.rows["m_project"][0]
    assert business_natural_key(CONTRACT, "m_project", project) == (PROJECT_ID,)


def test_natural_key_uuid_is_deterministic_and_follows_the_declared_uuid_version():
    value = natural_key_uuid(("1", "2025-12-31"))

    assert isinstance(value, UUID)
    assert value.version == 5
    assert value == natural_key_uuid(("1", "2025-12-31"))
    assert value == uuid5(NAMESPACE_DNS, "1|2025-12-31")
    assert value != natural_key_uuid(("2", "2025-12-31"))
    assert natural_key_uuid(("1", "2025-12-31"), version=4).version == 4
    assert natural_key_uuid(("1", "2025-12-31"), version=4) == natural_key_uuid(
        ("1", "2025-12-31"), version=4
    )


def test_generated_audit_metadata_uses_the_supplied_actor_and_observed_at():
    batch = _prepare(_payloads(), actor="FA:fetch", observed_at=OBSERVED_AT)

    for table in ("m_project", "m_pod", "pod_monitoring"):
        row = batch.rows[table][0]
        assert row["created_by"] == "FA:fetch"
        assert row["updated_by"] == "FA:fetch"
        assert row["created_at"] == OBSERVED_AT
        assert row["updated_at"] == OBSERVED_AT
    provided = dict(
        _record("pod_monitoring"),
        created_by="source",
        created_at="2020-01-01T00:00:00+00:00",
    )
    row = _rows(_prepare(_payloads(pod_monitoring=[provided])), "pod_monitoring")[0]
    assert row["created_by"] == "source"
    assert row["created_at"] == datetime(2020, 1, 1, tzinfo=timezone.utc)
    assert row["updated_by"] == "FA:fetch"
    assert batch.identity_ledger is None


def test_prior_state_is_reconciliation_context_not_an_output_seed():
    first = _prepare(_payloads(), actor="PriorActor")
    existing = {table: list(first.rows[table]) for table in INPUT_TABLES}

    # An empty payload replaces the old state; nothing is seeded from prior rows.
    batch = prepare_pod_api_batch(
        dict.fromkeys(INPUT_TABLES, b"[]"),
        existing_rows=existing,
        actor=ACTOR,
        observed_at=OBSERVED_AT,
    )
    for table in INPUT_TABLES:
        assert _rows(batch, table) == []

    # Existing mutable values legitimately change: the incoming record wins.
    before = dict(existing["m_project"][0])
    renamed = dict(_record("m_project"), project_name="Renamed Project")
    batch = prepare_pod_api_batch(
        _payloads(m_project=[renamed]),
        existing_rows=existing,
        actor=ACTOR,
        observed_at=OBSERVED_AT,
    )
    assert existing["m_project"][0] == before
    row = _rows(batch, "m_project")[0]
    assert row["project_name"] == "Renamed Project"
    # Missing immutable identifiers and metadata are reconciled from prior state.
    assert row["created_by"] == "PriorActor"
    assert row["created_at"] == OBSERVED_AT

    # Monitoring keeps prior facts outside the selected years only.
    prior_fact = dict(
        _record("pod_monitoring"), monpod_id=5, reported_date="2024-12-31"
    )
    prior = _prepare(_payloads(pod_monitoring=[prior_fact]), actor="PriorActor")
    existing = {table: [dict(r) for r in prior.rows[table]] for table in INPUT_TABLES}
    before = dict(existing["pod_monitoring"][0])
    # The selected-year fact is a different record, so it does not claim the preserved
    # fact's identity; the preserved fact keeps its own.
    incoming = dict(_record("pod_monitoring"), monpod_id=None, monpod_uuid=None)
    batch = _prepare(
        _payloads(pod_monitoring=[incoming]),
        existing_rows=existing,
        monitoring_years=[2025],
    )
    assert existing["pod_monitoring"][0] == before
    rows = _rows(batch, "pod_monitoring")
    assert [row["reported_date"] for row in rows] == [
        date(2024, 12, 31),
        date(2025, 12, 31),
    ]
    assert rows[0]["monpod_id"] == 5
    assert rows[0]["monpod_uuid"] == MON_UUID
    assert rows[1]["monpod_id"] == 6
    assert rows[1]["monpod_uuid"] == natural_key_uuid(("1", "2025-12-31"), version=4)


def test_empty_selected_snapshot_deletes_old_selected_facts():
    prior = _prepare(
        _payloads(
            pod_monitoring=[dict(_record("pod_monitoring"), reported_date="2024-12-31")]
        )
    )
    existing = {table: [dict(r) for r in prior.rows[table]] for table in INPUT_TABLES}
    batch = _prepare(
        _payloads(pod_monitoring=[]), existing_rows=existing, monitoring_years=[2024]
    )
    assert _rows(batch, "pod_monitoring") == []

    batch = _prepare(_payloads(pod_monitoring=[]), existing_rows=existing)
    assert _rows(batch, "pod_monitoring") == []


def test_orphan_preserved_monitoring_rejects_against_the_new_masters():
    prior = _prepare(
        _payloads(
            pod_monitoring=[dict(_record("pod_monitoring"), reported_date="2024-12-31")]
        )
    )
    existing = {table: [dict(r) for r in prior.rows[table]] for table in INPUT_TABLES}
    payloads: dict[str, Any] = dict.fromkeys(INPUT_TABLES, b"[]")
    payloads["pod_monitoring"] = json.dumps([]).encode()
    with pytest.raises(PodApiImportError, match="missing m_pod"):
        _prepare(payloads, existing_rows=existing, monitoring_years=[2025])


def test_active_wins_only_within_incoming_collisions_not_against_old_state():
    prior = _prepare(_payloads())
    existing = {table: [dict(r) for r in prior.rows[table]] for table in INPUT_TABLES}
    batch = _prepare(
        _payloads(m_project=[dict(_record("m_project"), is_active=0)]),
        existing_rows=existing,
    )
    row = _rows(batch, "m_project")[0]
    assert row["is_active"] == 0
    assert row["project_name"] == "Test Project"


def test_project_name_accepts_the_explicit_name_alias():
    record = {
        "project_id": PROJECT_ID,
        "is_active": 1,
        "name": "Alias Project",
    }
    batch = _prepare(_payloads(m_project=[record]))
    row = _rows(batch, "m_project")[0]
    assert row["project_name"] == "Alias Project"

    both = dict(_record("m_project"), project_name="Canonical", name="Alias Project")
    with pytest.raises(PodApiImportError, match="conflicts"):
        _prepare(_payloads(m_project=[both]))

    unknown = dict(_record("m_project"), nick="extra")
    with pytest.raises(PodApiImportError, match="not declared"):
        _prepare(_payloads(m_project=[unknown]))


def test_business_aliases_resolving_to_one_column_must_agree_in_either_order():
    def with_alias(
        canonical: object, alias: object, alias_first: bool
    ) -> dict[str, Any]:
        base = dict(_record("m_project"))
        base["project_name"] = canonical
        if alias_first:
            return {"name": alias, **base}
        return {**base, "name": alias}

    # A business unknown is not merged with a supplied alias value, in either order.
    for alias_first in (False, True):
        with pytest.raises(PodApiImportError, match="conflicts") as exc_info:
            _prepare(
                _payloads(m_project=[with_alias(None, "Alias Project", alias_first)])
            )
        assert "record 1" in str(exc_info.value)

    # Two equivalent normalized values from both headers are accepted.
    row = _rows(
        _prepare(
            _payloads(m_project=[with_alias("Same Project", "Same Project", True)])
        ),
        "m_project",
    )[0]
    assert row["project_name"] == "Same Project"

    # Two different supplied values reject regardless of which header arrives first.
    for alias_first in (False, True):
        with pytest.raises(PodApiImportError, match="conflicts"):
            _prepare(
                _payloads(
                    m_project=[
                        with_alias("Canonical Project", "Alias Project", alias_first)
                    ]
                )
            )


def test_supplied_duplicate_identities_must_agree_and_audit_conflicts_reject():
    first = _view_record(
        "pod_forecast",
        frcast_id=11,
        frcast_uuid=str(FRC_UUID),
        pod_id_seq=1,
        year_frcast=2025,
        usf_oil_pod="0.500000",
    )
    with pytest.raises(PodApiImportError, match="conflicting"):
        _prepare(_payloads(pod_forecast=[first, dict(first, frcast_id=12)]))

    # One missing and one supplied identity merge to the supplied value before
    # any local generation.
    blank = dict(first, frcast_id=None, frcast_uuid=None)
    batch = _prepare(_payloads(pod_forecast=[blank, first]))
    row = _rows(batch, "pod_forecast")[0]
    assert row["frcast_id"] == 11
    assert row["frcast_uuid"] == FRC_UUID

    audit = dict(_record("pod_forecast"), created_by="source")
    with pytest.raises(PodApiImportError, match="conflicting"):
        _prepare(_payloads(pod_forecast=[audit, dict(audit, created_by="other")]))

    # Owned business values must agree including NULL against 0: NULL is unknown, not
    # a blank permission to coalesce an unknown component into a reported zero.
    zeroed = dict(first, usf_oil_pod="0.000000")
    unknown = dict(zeroed, usf_oil_pod=None)
    with pytest.raises(PodApiImportError, match="conflicting"):
        _prepare(_payloads(pod_forecast=[zeroed, unknown]))

    monitor = dict(_record("pod_monitoring"), monpod_uuid=str(MON_UUID))
    with pytest.raises(PodApiImportError, match="conflicting"):
        _prepare(
            _payloads(
                pod_monitoring=[monitor, dict(monitor, monpod_uuid=str(LEDGER_UUID))]
            )
        )


def test_prior_audit_metadata_is_mutable_snapshot_state_not_an_immutable_lock():
    first = _prepare(_payloads(), actor="PriorActor")
    existing = {table: [dict(r) for r in first.rows[table]] for table in INPUT_TABLES}

    # Supplied audit metadata is preserved over the prior snapshot; an API update
    # that changes updated_by is a legitimate change, not a conflict with old state.
    supplied = dict(_record("pod_forecast"), created_by="source", updated_by="another")
    row = _rows(
        _prepare(_payloads(pod_forecast=[supplied]), existing_rows=existing),
        "pod_forecast",
    )[0]
    assert row["created_by"] == "source"
    assert row["updated_by"] == "another"

    # A blank audit field may still reuse the sensible prior creation metadata.
    blank = dict(_record("pod_forecast"))
    row = _rows(
        _prepare(_payloads(pod_forecast=[blank]), existing_rows=existing),
        "pod_forecast",
    )[0]
    assert row["created_by"] == "PriorActor"

    # Two supplied audit values inside one incoming duplicate context still reject.
    with pytest.raises(PodApiImportError, match="conflicting"):
        _prepare(
            _payloads(pod_forecast=[supplied, dict(supplied, created_by="other")]),
            existing_rows=existing,
        )


def test_audit_fill_uses_the_canonical_four_columns_for_every_table():
    batch = _prepare(_payloads(), actor="FA:fetch", observed_at=OBSERVED_AT)
    for table in INPUT_TABLES:
        row = _rows(batch, table)[0]
        assert row["created_by"] == "FA:fetch"
        assert row["updated_by"] == "FA:fetch"
        assert row["created_at"] == OBSERVED_AT
        assert row["updated_at"] == OBSERVED_AT
    assert batch.identity_ledger is None


def test_string_uuid_contexts_are_normalised_and_orphan_ledger_uuid_restored():
    batch = _prepare(
        _payloads(m_pod=[dict(_record("m_pod"), pod_uuid=None)]),
        reserved_pod_identities=[{"pod_id": POD_ID, "pod_uuid": str(POD_UUID)}],
    )
    assert _rows(batch, "m_pod")[0]["pod_uuid"] == POD_UUID

    prior = _prepare(_payloads())
    existing = {table: [dict(r) for r in prior.rows[table]] for table in INPUT_TABLES}
    existing["m_pod"] = [dict(existing["m_pod"][0], pod_uuid=str(POD_UUID))]
    batch = _prepare(
        _payloads(m_pod=[dict(_record("m_pod"), pod_uuid=None)]),
        existing_rows=existing,
    )
    assert _rows(batch, "m_pod")[0]["pod_uuid"] == POD_UUID

    monitor = dict(_record("pod_monitoring"), monpod_uuid=None)
    batch = _prepare(_payloads(pod_monitoring=[monitor]), existing_rows=existing)
    assert _rows(batch, "pod_monitoring")[0]["monpod_uuid"] == MON_UUID


def test_seq_pod_id_uuid_congruence_rejects_without_remapping_provided_identities():
    prior = _prepare(_payloads())
    existing = {table: [dict(r) for r in prior.rows[table]] for table in INPUT_TABLES}
    with pytest.raises(PodApiImportError, match="disagrees with m_pod"):
        _prepare(
            _payloads(m_pod=[dict(_record("m_pod"), pod_id_seq=2)]),
            existing_rows=existing,
        )

    batch = _prepare(
        _payloads(m_pod=[dict(_record("m_pod"), pod_uuid=None)]),
        reserved_pod_identities=[
            {"pod_id": POD_ID, "pod_id_seq": 1, "pod_uuid": str(POD_UUID)}
        ],
    )
    assert _rows(batch, "m_pod")[0]["pod_id_seq"] == 1
    assert _rows(batch, "m_pod")[0]["pod_uuid"] == POD_UUID

    with pytest.raises(PodApiImportError, match="congruent"):
        _prepare(
            _payloads(m_pod=[dict(_record("m_pod"), pod_id_seq=777, pod_uuid=None)]),
            reserved_pod_identities=[
                {"pod_id": POD_ID, "pod_id_seq": 1, "pod_uuid": str(POD_UUID)}
            ],
        )

    # Only the permanent m_pod identity is immutable: a supplied secondary record ID or
    # UUID is a legitimate update even when the prior record it replaces differs.
    updated = dict(_record("pod_forecast"), frcast_id=42, frcast_uuid=str(LEDGER_UUID))
    row = _rows(
        _prepare(_payloads(pod_forecast=[updated]), existing_rows=existing),
        "pod_forecast",
    )[0]
    assert row["frcast_id"] == 42
    assert row["frcast_uuid"] == LEDGER_UUID


def test_copied_pod_uuid_is_checked_after_master_reconciliation():
    master = dict(_record("m_pod"), pod_uuid=None)
    reserved = [{"pod_id": POD_ID, "pod_uuid": str(POD_UUID)}]
    disagreeing = dict(_view_record("pod_monitoring"), pod_uuid=str(LEDGER_UUID))
    with pytest.raises(PodApiImportError, match="disagrees with m_pod"):
        _prepare(
            _payloads(m_pod=[master], pod_monitoring=[disagreeing]),
            reserved_pod_identities=reserved,
        )

    # The pod_monitoring view is the API shape that carries the pod_uuid copy, so the
    # copy is checked after the master itself is reconciled with the reserved ledger.
    agreeing = dict(_view_record("pod_monitoring"), pod_uuid=str(POD_UUID))
    batch = _prepare(
        _payloads(m_pod=[master], pod_monitoring=[agreeing]),
        reserved_pod_identities=reserved,
    )
    assert _rows(batch, "m_pod")[0]["pod_uuid"] == POD_UUID
    assert len(_rows(batch, "pod_monitoring")) == 1


def test_stage_label_alone_is_a_real_owned_submission():
    record = dict.fromkeys(
        {column.name for column in CONTRACT.tables["pod_monitoring"].columns}
    )
    record["pod_id_seq"] = 1
    record["reported_date"] = "2025-12-31"
    record["pod_stage"] = "Stage II"
    # Real evidence reaches validation, and the contract's required is_completed still
    # binds: nothing is invented for it and the row is not silently skipped.
    with pytest.raises(PodApiImportError, match="is_completed is required"):
        _prepare(_payloads(pod_monitoring=[record]))

    # A truly all-null projection has no record and still skips.
    synthetic = dict.fromkeys(
        {column.name for column in CONTRACT.tables["pod_monitoring"].columns}
    )
    synthetic["pod_id_seq"] = 1
    synthetic["reported_date"] = "2025-12-31"
    batch = _prepare(_payloads(pod_monitoring=[synthetic]))
    assert _rows(batch, "pod_monitoring") == []

    # A supplied label is validated as reference evidence even with null IDs.
    invalid = dict.fromkeys(
        {column.name for column in CONTRACT.tables["pod_monitoring"].columns}
    )
    invalid["pod_stage"] = "NOT A STAGE"
    with pytest.raises(PodApiImportError, match="reference stage"):
        _prepare(_payloads(pod_monitoring=[invalid]))

    # A real submission that carries the required flag with the stage succeeds.
    valid = dict(synthetic, is_completed=False, pod_stage="Stage II")
    batch = _prepare(_payloads(pod_monitoring=[valid]))
    rows = _rows(batch, "pod_monitoring")
    assert len(rows) == 1
    assert rows[0]["pod_stage_id"] == 2
    assert rows[0]["is_completed"] is False


def test_api_decimal_text_is_canonicalized_by_the_contract_scale():
    record = _view_record(
        "pod_monitoring",
        monpod_id=7,
        monpod_uuid=str(MON_UUID),
        pod_id_seq=1,
        reported_date="2025-01-31",
        pod_stage="Stage I",
        wap_oil_real="1.234567",
        well_total_real=0,
    )
    row = _rows(_prepare(_payloads(pod_monitoring=[record])), "pod_monitoring")[0]

    assert row["wap_oil_real"] == Decimal("1.234567")
    assert row["well_total_real"] == 0

    over_scale = dict(record, wap_oil_real="1.2345678")
    with pytest.raises(PodApiImportError, match="fractional"):
        _prepare(_payloads(pod_monitoring=[over_scale]))


def test_monitoring_years_filter_keeps_selected_years_and_existing_facts():
    prior = _prepare(
        _payloads(
            pod_monitoring=[dict(_record("pod_monitoring"), reported_date="2024-12-31")]
        )
    )
    existing = {
        table: [dict(row) for row in prior.rows[table]] for table in INPUT_TABLES
    }
    selected = dict(
        _record("pod_monitoring"),
        monpod_id=2,
        monpod_uuid=str(LEDGER_UUID),
        reported_date="2025-01-31",
    )
    outside = dict(
        _record("pod_monitoring"),
        monpod_id=3,
        monpod_uuid=str(PROJ_UUID),
        reported_date="2026-01-31",
    )
    batch = _prepare(
        _payloads(pod_monitoring=[selected, outside]),
        existing_rows=existing,
        monitoring_years=[2025],
    )

    assert [row["reported_date"] for row in _rows(batch, "pod_monitoring")] == [
        date(2024, 12, 31),
        date(2025, 1, 31),
    ]


def test_rejected_project_rows_are_skipped_and_non_boolean_flags_reject():
    retained = _record("m_project")
    skipped = dict(_record("m_project"), project_id="P-2403431-02", is_rejected=True)
    batch = _prepare(_payloads(m_project=[retained, skipped]))
    rows = _rows(batch, "m_project")

    assert len(rows) == 1
    assert rows[0]["project_id"] == PROJECT_ID

    with pytest.raises(PodApiImportError, match="is_rejected"):
        _prepare(_payloads(m_project=[dict(_record("m_project"), is_rejected="true")]))


def test_missing_uuids_reuse_the_prior_or_reserved_identity():
    first = _prepare(_payloads())
    existing = {
        table: [dict(row) for row in first.rows[table]] for table in INPUT_TABLES
    }
    same_fact = dict(_view_record("pod_monitoring"), monpod_id=None, monpod_uuid=None)
    batch = _prepare(_payloads(pod_monitoring=[same_fact]), existing_rows=existing)
    assert _rows(batch, "pod_monitoring")[0]["monpod_uuid"] == MON_UUID

    master = dict(_record("m_pod"), pod_uuid=None)
    batch = _prepare(
        _payloads(m_pod=[master]),
        reserved_pod_identities=[{"pod_id": POD_ID, "pod_uuid": POD_UUID}],
    )
    assert _rows(batch, "m_pod")[0]["pod_uuid"] == POD_UUID

    conflicting = dict(master, pod_uuid=str(LEDGER_UUID))
    with pytest.raises(PodApiImportError, match="conflicting UUID"):
        _prepare(
            _payloads(m_pod=[conflicting]),
            reserved_pod_identities=[{"pod_id": POD_ID, "pod_uuid": POD_UUID}],
        )


def test_secondary_identifiers_allocate_stably_in_sorted_natural_key_order():
    prior = _prepare(
        _payloads(
            pod_monitoring=[
                dict(_record("pod_monitoring"), monpod_id=5, reported_date="2024-12-31")
            ]
        )
    )
    existing = {
        table: [dict(row) for row in prior.rows[table]] for table in INPUT_TABLES
    }
    later = dict(
        _record("pod_monitoring"),
        monpod_id=None,
        monpod_uuid=None,
        reported_date="2025-06-30",
        pod_stage_id=1,
        opex_real="1.000",
    )
    earlier = dict(
        _record("pod_monitoring"),
        monpod_id=None,
        monpod_uuid=None,
        reported_date="2025-01-31",
        pod_stage_id=1,
        opex_real="2.000",
    )
    batch = _prepare(
        _payloads(pod_monitoring=[later, earlier]),
        existing_rows=existing,
        monitoring_years=[2025],
    )
    rows = sorted(_rows(batch, "pod_monitoring"), key=lambda row: row["reported_date"])

    # Allocation counts the existing snapshot IDs even though the preserved row
    # is reconciliation output, not an incoming record.
    assert [row["monpod_id"] for row in rows] == [5, 6, 7]
    assert [row["reported_date"] for row in rows] == [
        date(2024, 12, 31),
        date(2025, 1, 31),
        date(2025, 6, 30),
    ]
    assert all(row["monpod_uuid"] != MON_UUID for row in rows[1:])


def test_conflicting_monitoring_rows_for_the_same_id_reject_before_return():
    first = _view_record(
        "pod_monitoring",
        monpod_id=7,
        monpod_uuid=str(MON_UUID),
        pod_id_seq=1,
        reported_date="2025-01-31",
        pod_stage="Stage II",
        opex_real="1.000000",
    )
    with pytest.raises(PodApiImportError, match="conflicting"):
        _prepare(_payloads(pod_monitoring=[first, dict(first, opex_real="2.000000")]))


def test_identity_copy_must_agree_with_the_authoritative_master_row():
    disagreeing = dict(_view_record("pod_monitoring"), pod_id="PL-1997-0001-4-1-9")
    with pytest.raises(PodApiImportError, match="disagrees with m_pod"):
        _prepare(_payloads(pod_monitoring=[disagreeing]))

    disagreeing_uuid = dict(_view_record("pod_monitoring"), pod_uuid=str(LEDGER_UUID))
    with pytest.raises(PodApiImportError, match="disagrees with m_pod"):
        _prepare(_payloads(pod_monitoring=[disagreeing_uuid]))

    agreeing = dict(_view_record("pod_forecast"), pod_id=POD_ID)
    batch = _prepare(_payloads(pod_forecast=[agreeing]))
    assert _rows(batch, "pod_forecast")[0]["pod_id_seq"] == 1


def test_active_wins_priority_collisions_and_inactive_only_rows_are_retained():
    active = dict(_record("m_project"), project_name="Active Project", is_active=1)
    inactive = dict(_record("m_project"), project_name="Inactive Project", is_active=0)
    batch = _prepare(_payloads(m_project=[inactive, active]))
    rows = _rows(batch, "m_project")
    assert len(rows) == 1
    assert rows[0]["project_name"] == "Active Project"

    with pytest.raises(PodApiImportError, match="conflicting"):
        _prepare(_payloads(m_project=[active, dict(active, project_name="Third")]))

    batch = _prepare(_payloads(m_project=[inactive]))
    assert _rows(batch, "m_project")[0]["is_active"] == 0


def test_global_approved_forecast_is_not_filtered_by_monitoring_years():
    approved = dict(
        _view_record("pod_forecast"),
        frcast_id=9,
        frcast_uuid=str(LEDGER_UUID),
        year_frcast=2027,
        usf_oil_pod="1.000000",
    )
    batch = _prepare(_payloads(pod_forecast=[approved]), monitoring_years=[2025])
    rows = _rows(batch, "pod_forecast")
    assert len(rows) == 1
    assert rows[0]["year_frcast"] == 2027


def test_new_global_master_omitting_a_pod_referenced_by_facts_rejects():
    payloads = _payloads()
    payloads["m_pod"] = json.dumps(
        [dict(_record("m_pod"), pod_id_seq=999, pod_id="PL-1997-0001-4-1-9")]
    ).encode()
    with pytest.raises(PodApiImportError, match="missing m_pod"):
        _prepare(payloads)


def test_supplied_secondary_id_conflicting_with_preserved_facts_rejects():
    prior = _prepare(
        _payloads(
            pod_monitoring=[
                dict(_record("pod_monitoring"), monpod_id=5, reported_date="2024-12-31")
            ]
        )
    )
    existing = {
        table: [dict(row) for row in prior.rows[table]] for table in INPUT_TABLES
    }
    conflict = dict(
        _record("pod_monitoring"),
        monpod_id=5,
        monpod_uuid=str(LEDGER_UUID),
        reported_date="2025-01-31",
        pod_stage_id=1,
        opex_real="1.000",
    )
    with pytest.raises(PodApiImportError, match="Duplicate key"):
        _prepare(
            _payloads(pod_monitoring=[conflict]),
            existing_rows=existing,
            monitoring_years=[2025],
        )


def test_prepare_never_opens_a_database_or_writes_to_disk(
    monkeypatch, tmp_path
) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("prepare touched a database connection or the filesystem")

    monkeypatch.setattr(sqlite3, "connect", forbidden)
    monkeypatch.setattr(Path, "mkdir", forbidden)
    batch = _prepare(_payloads())

    assert batch.identity_ledger is None
    assert batch.source_path is None
    assert batch.target_basis is None
    assert batch.staged_snapshots is None
    assert list(tmp_path.iterdir()) == []


def test_http_numeric_text_is_exact_and_rejects_beyond_the_declared_precision():
    exact = dict(
        _view_record("pod_monitoring"),
        monpod_id=7,
        monpod_uuid=str(MON_UUID),
        reported_date="2025-01-31",
        pod_stage="Stage I",
        opex_real="12345678901234567890.123",
    )
    with pytest.raises(PodApiImportError, match="precision"):
        _prepare(_payloads(pod_monitoring=[exact]))

    within = dict(exact, opex_real="1234567890123456.123")
    row = _rows(_prepare(_payloads(pod_monitoring=[within])), "pod_monitoring")[0]
    assert row["opex_real"] == Decimal("1234567890123456.123")


def test_normalization_failures_report_the_table_and_record_number():
    boolean_value = dict(_view_record("pod_monitoring"), opex_real=True)
    pattern = r"record 1: pod_monitoring row 1, column opex_real: boolean is not"
    with pytest.raises(PodApiImportError, match=pattern):
        _prepare(_payloads(pod_monitoring=[boolean_value]))


def test_blank_update_audit_comes_from_the_fetch_not_prior_editor_state():
    """A fetch stamps its own update audit; only creation history reconciles with prior."""
    prior = _prepare(_payloads(), actor=PRIOR_ACTOR, observed_at=PRIOR_OBSERVED_AT)
    existing = {table: [dict(row) for row in prior.rows[table]] for table in INPUT_TABLES}

    batch = _prepare(_payloads(), existing_rows=existing)
    for table in INPUT_TABLES:
        row = _rows(batch, table)[0]
        assert row["created_by"] == PRIOR_ACTOR
        assert row["created_at"] == PRIOR_OBSERVED_AT
        assert row["updated_by"] == ACTOR
        assert row["updated_at"] == OBSERVED_AT


def test_explicit_upstream_update_audit_is_preserved_and_not_reattributed():
    supplied = dict(
        _record("pod_forecast"),
        created_by="Upstream:source",
        created_at="2020-01-01T00:00:00+00:00",
        updated_by="Upstream:api",
        updated_at="2020-02-02T00:00:00+00:00",
    )
    prior = _prepare(_payloads(pod_forecast=[supplied]), actor=PRIOR_ACTOR)
    existing = {table: [dict(row) for row in prior.rows[table]] for table in INPUT_TABLES}

    row = _rows(_prepare(_payloads(pod_forecast=[supplied]), existing_rows=existing), "pod_forecast")[0]
    assert row["created_by"] == "Upstream:source"
    assert row["created_at"] == datetime(2020, 1, 1, tzinfo=timezone.utc)
    assert row["updated_by"] == "Upstream:api"
    assert row["updated_at"] == datetime(2020, 2, 2, tzinfo=timezone.utc)


def test_unselected_preserved_monitoring_fact_keeps_its_prior_audit_exactly():
    prior = _prepare(
        _payloads(
            pod_monitoring=[dict(_record("pod_monitoring"), reported_date="2024-12-31")]
        ),
        actor=PRIOR_ACTOR,
        observed_at=PRIOR_OBSERVED_AT,
    )
    existing = {table: [dict(row) for row in prior.rows[table]] for table in INPUT_TABLES}
    before = dict(existing["pod_monitoring"][0])

    batch = _prepare(
        _payloads(pod_monitoring=[]), existing_rows=existing, monitoring_years=[2025]
    )
    row = _rows(batch, "pod_monitoring")[0]

    assert row == before
    assert row["created_by"] == PRIOR_ACTOR
    assert row["updated_by"] == PRIOR_ACTOR
    assert row["created_at"] == PRIOR_OBSERVED_AT
    assert row["updated_at"] == PRIOR_OBSERVED_AT
    assert existing["pod_monitoring"][0] == before
