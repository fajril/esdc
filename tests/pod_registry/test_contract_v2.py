"""Regression coverage for the shared POD contract and batch validator."""

from __future__ import annotations

import importlib
import sqlite3
from datetime import date, datetime
from decimal import Decimal
from typing import Any, cast
from uuid import UUID, uuid1, uuid4

import pytest

INPUT_COLUMN_COUNTS = {
    "m_project": 7,
    "m_pod": 82,
    "pod_project": 10,
    "pod_monitoring": 92,
    "pod_forecast": 13,
}
ALL_COLUMN_COUNTS = {
    **INPUT_COLUMN_COUNTS,
    "pod_monitoring_api": 176,
    "pod_forecast_api": 24,
}


def _api():
    try:
        contract_module = importlib.import_module("esdc.pod_registry.contract")
        models_module = importlib.import_module("esdc.pod_registry.models")
    except ImportError as exc:
        pytest.fail(f"shared POD contract API is unavailable: {exc}", pytrace=False)
    return (
        contract_module.load_pod_contract,
        contract_module.validate_pod_batch,
        models_module.PreparedPodBatch,
    )


def _empty_batch(contract, batch_type, *, omit=(), contract_version=1):
    headers = {}
    rows = {}
    for name, table in contract.tables.items():
        if table.role != "input" or name in omit:
            continue
        headers[name] = tuple(column.name for column in table.columns)
        rows[name] = []
    return batch_type(
        source="test-fixture",
        contract_id=contract.contract_id,
        contract_version=contract_version,
        dictionary_version=contract.dictionary_version,
        headers=headers,
        rows=rows,
    )


def _value_for(column, row_number: int):
    name = column.name
    if name == "project_id":
        return f"P-{row_number:07d}-01"
    if name == "project_name":
        return f"Project {row_number}"
    if name == "pod_id":
        return f"PL-2024-{row_number:04d}-1-1-0"
    if name == "pod_name":
        return f"Example POD {row_number}"
    if name == "pod_letter_num":
        return f"SKK/2024/{row_number:04d}"
    if name.endswith("_uuid") or name == "pod_uuid":
        return str(uuid4())
    if name in {"institution_id", "pod_type_id", "pod_stage_id"}:
        return 1
    if name in {"pod_id_seq", "monpod_id", "frcast_id", "id"}:
        return row_number
    if name == "year_frcast":
        return 2024
    if name == "pod_order_no":
        return row_number
    if name == "rev_num":
        return 0
    if name in {
        "sow_pod",
        "scenario_pod",
        "sow_real",
        "scenario_real",
        "sow_otlk",
        "scenario_otlk",
    }:
        return "example"
    if column.logical_type == "integer":
        return 1
    if column.logical_type == "decimal":
        return Decimal("1")
    if column.logical_type == "boolean":
        return True
    if column.logical_type == "date":
        return date(2024, 1, 1)
    if column.logical_type == "datetime":
        return datetime(2024, 1, 1)
    if column.logical_type == "uuid":
        return str(uuid4())
    if column.logical_type == "text":
        return "example"
    raise AssertionError(
        f"test fixture needs a value for {name}: {column.logical_type}"
    )


def _row(contract, table_name, row_number=1):
    return {
        column.name: None if column.nullable else _value_for(column, row_number)
        for column in contract.tables[table_name].columns
    }


def _batch_with_rows(contract, batch_type, replacements=None, *, contract_version=1):
    batch = _empty_batch(contract, batch_type, contract_version=contract_version)
    rows = dict(batch.rows)
    if replacements:
        rows.update(replacements)
    return batch_type(
        source=batch.source,
        contract_id=batch.contract_id,
        contract_version=batch.contract_version,
        dictionary_version=batch.dictionary_version,
        headers=batch.headers,
        rows=rows,
    )


def test_contract_declares_complete_input_and_api_columns_in_order():
    load_pod_contract, _, _ = _api()
    contract = load_pod_contract()

    assert contract.contract_id == "esdc.pod"
    assert contract.contract_version == 1
    assert contract.dictionary_version == "0.4.0"
    assert {
        name: len(table.columns) for name, table in contract.tables.items()
    } == ALL_COLUMN_COUNTS
    assert tuple(column.name for column in contract.tables["m_pod"].columns)[0] == (
        "pod_id_seq"
    )
    assert all(
        column.logical_type
        for table in contract.tables.values()
        for column in table.columns
    )
    assert all(
        column.ownership in {"input", "derived"}
        for table in contract.tables.values()
        for column in table.columns
    )


def test_missing_required_table_is_distinct_from_present_empty_table():
    load_pod_contract, validate, batch_type = _api()
    contract = load_pod_contract()
    missing = _empty_batch(contract, batch_type, omit={"pod_forecast"})
    empty_forecast = _empty_batch(contract, batch_type)

    missing_issues = validate(missing, contract)
    empty_issues = validate(empty_forecast, contract)

    assert any(
        issue.table == "pod_forecast" and issue.code == "missing_table"
        for issue in missing_issues
    )
    assert not any(issue.table == "pod_forecast" for issue in empty_issues)


@pytest.mark.parametrize(
    ("headers", "expected_code", "expected_column"),
    [
        (("project_id", "project_id"), "duplicate_column", "project_id"),
        (("project_id", "unknown_extra"), "unexpected_column", "unknown_extra"),
    ],
)
def test_headers_reject_duplicate_and_undeclared_columns(
    headers, expected_code, expected_column
):
    load_pod_contract, validate, batch_type = _api()
    contract = load_pod_contract()
    batch = _empty_batch(contract, batch_type)
    changed_headers = dict(batch.headers)
    changed_headers["m_project"] = headers
    changed = batch_type(
        source=batch.source,
        contract_id=batch.contract_id,
        contract_version=batch.contract_version,
        dictionary_version=batch.dictionary_version,
        headers=changed_headers,
        rows=batch.rows,
    )

    issues = validate(changed, contract)

    assert any(
        issue.table == "m_project"
        and issue.code == expected_code
        and issue.column == expected_column
        for issue in issues
    )


def test_unsupported_contract_version_is_reported():
    load_pod_contract, validate, batch_type = _api()
    contract = load_pod_contract()
    batch = _empty_batch(contract, batch_type, contract_version=99)

    issues = validate(batch, contract)

    assert any(issue.code == "unsupported_version" for issue in issues)


@pytest.mark.parametrize(
    ("column", "value", "expected_code"),
    [
        ("pod_id_seq", "1", "type_mismatch"),
        ("approval_date", "2024-02-30", "invalid_date"),
        ("pod_uuid", "not-a-uuid", "invalid_uuid"),
        ("res_oil_1p_pod", float("inf"), "non_finite"),
        ("is_latest", 2, "type_mismatch"),
        ("res_oil_2p_risked_pod", "12.4", "type_mismatch"),
    ],
)
def test_invalid_native_values_are_reported_without_coercion(
    column, value, expected_code
):
    load_pod_contract, validate, batch_type = _api()
    contract = load_pod_contract()
    row = _row(contract, "m_pod")
    row[column] = value
    batch = _batch_with_rows(contract, batch_type, {"m_pod": [row]})

    issues = validate(batch, contract)

    assert any(
        issue.table == "m_pod"
        and issue.column == column
        and issue.code == expected_code
        for issue in issues
    )


@pytest.mark.parametrize(
    "value",
    [Decimal("NaN"), Decimal("Infinity"), Decimal("-Infinity"), Decimal("sNaN")],
    ids=["nan", "positive-infinity", "negative-infinity", "signaling-nan"],
)
def test_non_finite_decimal_comparison_reports_without_raising(value):
    load_pod_contract, validate, batch_type = _api()
    contract = load_pod_contract()
    row = _row(contract, "m_pod")
    row["res_oil_1p_pod"] = value
    row["res_oil_2p_pod"] = Decimal("1")
    batch = _batch_with_rows(contract, batch_type, {"m_pod": [row]})

    issues = validate(batch, contract)

    assert any(
        issue.table == "m_pod"
        and issue.column == "res_oil_1p_pod"
        and issue.code == "non_finite"
        for issue in issues
    )


def test_decimal_precision_is_checked_before_rounding():
    load_pod_contract, validate, batch_type = _api()
    contract = load_pod_contract()
    row = _row(contract, "m_pod")
    row["capital_drilling_pod"] = Decimal("1.2345")
    batch = _batch_with_rows(contract, batch_type, {"m_pod": [row]})

    issues = validate(batch, contract)

    assert any(
        issue.table == "m_pod"
        and issue.column == "capital_drilling_pod"
        and issue.code == "precision_exceeded"
        for issue in issues
    )


def test_zero_irr_and_null_risked_reserve_are_valid_values():
    load_pod_contract, validate, batch_type = _api()
    contract = load_pod_contract()
    row = _row(contract, "m_pod")
    row["irr_pf_ctr_pod"] = Decimal("0")
    row["res_oil_2p_risked_pod"] = None
    batch = _batch_with_rows(contract, batch_type, {"m_pod": [row]})

    issues = validate(batch, contract)

    assert not any(issue.table == "m_pod" for issue in issues)


def test_duplicate_primary_key_and_inconsistent_mapping_are_reported():
    load_pod_contract, validate, batch_type = _api()
    contract = load_pod_contract()
    first = _row(contract, "m_pod", 1)
    second = _row(contract, "m_pod", 2)
    second["pod_id_seq"] = first["pod_id_seq"]
    batch = _batch_with_rows(contract, batch_type, {"m_pod": [first, second]})
    rows = dict(batch.rows)
    rows["m_project"] = [_row(contract, "m_project", 1)]
    mapping = _row(contract, "pod_project", 2)
    mapping.update({"pod_id_seq": 999, "project_id": "P-9999999-01"})
    rows["pod_project"] = [mapping]
    batch = batch_type(
        source=batch.source,
        contract_id=batch.contract_id,
        contract_version=batch.contract_version,
        dictionary_version=batch.dictionary_version,
        headers=batch.headers,
        rows=rows,
    )

    issues = validate(batch, contract)

    assert any(
        issue.table == "m_pod" and issue.code == "duplicate_key" for issue in issues
    )
    assert any(
        issue.table == "pod_project" and issue.code == "invalid_reference"
        for issue in issues
    )


def test_superseded_master_reference_must_exist_and_be_acyclic():
    load_pod_contract, validate, batch_type = _api()
    contract = load_pod_contract()
    row = _row(contract, "m_pod")
    row["superseded_by"] = 999
    batch = _batch_with_rows(contract, batch_type, {"m_pod": [row]})

    issues = validate(batch, contract)

    assert any(
        issue.table == "m_pod"
        and issue.column == "superseded_by"
        and issue.code == "invalid_reference"
        for issue in issues
    )


def test_only_semoga_placeholder_may_have_unissued_pod_identity_and_order():
    load_pod_contract, validate, batch_type = _api()
    contract = load_pod_contract()
    row = _row(contract, "m_pod")
    row.update(
        {
            "pod_id_seq": 856,
            "approval_date": date(1997, 6, 1),
            "institution_id": 2,
            "pod_type_id": 2,
            "pod_id": None,
            "pod_order_no": None,
            "pod_name": "Pengembangan Lapangan Semoga",
        }
    )
    batch = _batch_with_rows(contract, batch_type, {"m_pod": [row]})

    issues = validate(batch, contract)

    assert not any(
        issue.table == "m_pod"
        and issue.column in {"pod_id", "pod_order_no"}
        for issue in issues
    )


def test_other_pods_still_require_an_order_number():
    load_pod_contract, validate, batch_type = _api()
    contract = load_pod_contract()
    row = _row(contract, "m_pod")
    row.update({"pod_id_seq": 857, "pod_order_no": None})
    batch = _batch_with_rows(contract, batch_type, {"m_pod": [row]})

    issues = validate(batch, contract)

    assert any(
        issue.table == "m_pod"
        and issue.column == "pod_order_no"
        and issue.code == "missing_value"
        for issue in issues
    )


def test_uuid_object_and_string_are_the_same_key():
    load_pod_contract, validate, batch_type = _api()
    contract = load_pod_contract()
    first = _row(contract, "m_pod", 1)
    second = _row(contract, "m_pod", 2)
    second["pod_uuid"] = cast(Any, UUID(cast(str, first["pod_uuid"])))
    batch = _batch_with_rows(contract, batch_type, {"m_pod": [first, second]})

    issues = validate(batch, contract)

    assert any(
        issue.table == "m_pod"
        and issue.column == "pod_uuid"
        and issue.code == "duplicate_key"
        for issue in issues
    )


def test_date_and_iso_string_are_the_same_composite_key():
    load_pod_contract, validate, batch_type = _api()
    contract = load_pod_contract()
    first = _row(contract, "pod_monitoring", 1)
    second = _row(contract, "pod_monitoring", 2)
    first["monpod_id"] = 1
    second["monpod_id"] = 2
    first["pod_id_seq"] = 1
    second["pod_id_seq"] = 1
    first["reported_date"] = date(2024, 1, 1)
    second["reported_date"] = "2024-01-01"
    batch = _batch_with_rows(
        contract, batch_type, {"pod_monitoring": [first, second]}
    )

    issues = validate(batch, contract)

    assert any(
        issue.table == "pod_monitoring"
        and issue.column == "pod_id_seq"
        and issue.code == "duplicate_key"
        for issue in issues
    )


def test_invalid_batch_validation_does_not_write_a_target_database(tmp_path):
    load_pod_contract, validate, batch_type = _api()
    contract = load_pod_contract()
    target = tmp_path / "target.sqlite"
    with sqlite3.connect(target) as connection:
        connection.execute("CREATE TABLE sentinel (value TEXT)")
        connection.execute("INSERT INTO sentinel VALUES ('unchanged')")
    before = target.read_bytes()
    batch = _empty_batch(contract, batch_type, omit={"pod_forecast"})

    validate(batch, contract)

    assert target.read_bytes() == before


@pytest.mark.parametrize(
    ("table_name", "column"),
    [
        ("m_pod", "pod_uuid"),
        ("pod_monitoring", "monpod_uuid"),
        ("pod_forecast", "frcast_uuid"),
    ],
)
def test_generated_source_uuid_fields_require_version_four(table_name, column):
    load_pod_contract, validate, batch_type = _api()
    contract = load_pod_contract()
    row = _row(contract, table_name)
    row[column] = str(uuid1())
    batch = _batch_with_rows(contract, batch_type, {table_name: [row]})

    issues = validate(batch, contract)

    assert any(
        issue.table == table_name
        and issue.column == column
        and issue.code == "invalid_uuid"
        for issue in issues
    )


@pytest.mark.parametrize(
    ("table_name", "column"),
    [("m_pod", "pod_id_seq"), ("pod_project", "project_id")],
)
def test_non_scalar_keys_and_references_return_issues(table_name, column):
    load_pod_contract, validate, batch_type = _api()
    contract = load_pod_contract()
    row = _row(contract, table_name)
    row[column] = cast(Any, [])
    batch = _batch_with_rows(contract, batch_type, {table_name: [row]})

    issues = validate(batch, contract)

    assert any(
        issue.table == table_name
        and issue.column == column
        and issue.code == "type_mismatch"
        for issue in issues
    )


def test_duplicate_row_sources_for_a_declared_table_alias_are_rejected():
    load_pod_contract, validate, batch_type = _api()
    contract = load_pod_contract()
    batch = _empty_batch(contract, batch_type)
    rows = dict(batch.rows)
    rows["project_pod"] = [_row(contract, "pod_project", 2)]
    changed = batch_type(
        source=batch.source,
        contract_id=batch.contract_id,
        contract_version=batch.contract_version,
        dictionary_version=batch.dictionary_version,
        headers=batch.headers,
        rows=rows,
    )

    issues = validate(changed, contract)

    assert any(
        issue.table == "pod_project" and issue.code == "duplicate_row_source"
        for issue in issues
    )


def test_business_identity_code_must_be_coherent_with_its_approval_key():
    load_pod_contract, validate, batch_type = _api()
    contract = load_pod_contract()
    row = _row(contract, "m_pod")
    row.update(
        {
            "approval_date": date(1997, 1, 1),
            "pod_order_no": 1,
            "pod_id": "PL-1997-0001-2-3-4",
            "institution_id": 2,
            "pod_type_id": 3,
            "rev_num": 4,
        }
    )
    batch = _batch_with_rows(contract, batch_type, {"m_pod": [row]})

    issues = validate(batch, contract)

    assert not any(
        issue.table == "m_pod"
        and issue.column in {"approval_date", "pod_order_no", "pod_id"}
        for issue in issues
    )


def test_order_number_must_be_inside_the_approval_date_order_window():
    load_pod_contract, validate, batch_type = _api()
    contract = load_pod_contract()
    row = _row(contract, "m_pod")
    row.update(
        {
            "approval_date": date(1997, 6, 1),
            "pod_order_no": 1,
            "pod_id": "PL-1997-0001-4-1-0",
            "institution_id": 4,
        }
    )
    batch = _batch_with_rows(contract, batch_type, {"m_pod": [row]})

    issues = validate(batch, contract)

    assert any(
        issue.table == "m_pod"
        and issue.column == "pod_order_no"
        and issue.code == "invalid_value"
        for issue in issues
    )


def test_identity_code_year_must_match_the_approval_date():
    load_pod_contract, validate, batch_type = _api()
    contract = load_pod_contract()
    row = _row(contract, "m_pod")
    row.update({"pod_id": "PL-2025-0001-1-1-0"})
    batch = _batch_with_rows(contract, batch_type, {"m_pod": [row]})

    issues = validate(batch, contract)

    assert any(
        issue.table == "m_pod"
        and issue.column == "pod_id"
        and issue.code == "invalid_value"
        for issue in issues
    )


def test_identity_code_order_part_must_match_the_declared_order_number():
    load_pod_contract, validate, batch_type = _api()
    contract = load_pod_contract()
    row = _row(contract, "m_pod", 2)
    row.update({"pod_id": "PL-2024-0001-1-1-0"})
    batch = _batch_with_rows(contract, batch_type, {"m_pod": [row]})

    issues = validate(batch, contract)

    assert any(
        issue.table == "m_pod"
        and issue.column == "pod_id"
        and issue.code == "invalid_value"
        for issue in issues
    )


def test_identity_code_suffix_must_match_institution_type_and_revision():
    load_pod_contract, validate, batch_type = _api()
    contract = load_pod_contract()
    row = _row(contract, "m_pod")
    row.update({"institution_id": 2, "pod_type_id": 3, "rev_num": 4})
    batch = _batch_with_rows(contract, batch_type, {"m_pod": [row]})

    issues = validate(batch, contract)

    assert any(
        issue.table == "m_pod"
        and issue.column == "pod_id"
        and issue.code == "invalid_value"
        for issue in issues
    )


def test_identity_coherence_compares_an_iso_date_string_row_too():
    load_pod_contract, validate, batch_type = _api()
    contract = load_pod_contract()

    def issues_for(row):
        return validate(_batch_with_rows(contract, batch_type, {"m_pod": [row]}), contract)

    def reports(issues, column, code):
        return any(
            issue.table == "m_pod" and issue.column == column and issue.code == code
            for issue in issues
        )

    row = _row(contract, "m_pod")
    row.update(
        {
            "approval_date": "1997-06-01",
            "pod_order_no": 1,
            "pod_id": "PL-1997-0001-4-1-0",
            "institution_id": 4,
        }
    )
    assert reports(issues_for(row), "pod_order_no", "invalid_value")

    # A midnight datetime is an accepted date representation for the same validator,
    # so the same incoherent pair must be reported for it too.
    midnight = dict(row, approval_date=datetime(1997, 6, 1, 0, 0))
    assert reports(issues_for(midnight), "pod_order_no", "invalid_value")

    # A value with a time of day is not a date: the existing column error stands and
    # the coherence pass does not invent an identity for it.
    noon = dict(row, approval_date=datetime(1997, 6, 1, 12, 0))
    issues = issues_for(noon)
    assert reports(issues, "approval_date", "invalid_date")
    assert not reports(issues, "pod_order_no", "invalid_value")


def test_identity_coherence_uses_the_declared_integer_representations():
    load_pod_contract, validate, batch_type = _api()
    contract = load_pod_contract()

    def issues_for(row):
        return validate(_batch_with_rows(contract, batch_type, {"m_pod": [row]}), contract)

    def reports(issues, column, code):
        return any(
            issue.table == "m_pod" and issue.column == column and issue.code == code
            for issue in issues
        )

    row = _row(contract, "m_pod")
    row.update(
        {
            "approval_date": date(1997, 6, 1),
            "pod_order_no": Decimal(1),
            "pod_id": "PL-1997-0001-4-1-0",
            "institution_id": 4,
        }
    )
    assert reports(issues_for(row), "pod_order_no", "invalid_value")

    coherent = _row(contract, "m_pod")
    coherent.update(
        {
            "approval_date": date(1997, 1, 1),
            "pod_order_no": Decimal(1),
            "pod_id": "PL-1997-0001-4-1-0",
            "institution_id": 4,
            "pod_type_id": Decimal(1),
            "rev_num": Decimal(0),
        }
    )
    assert issues_for(coherent) == ()

    # The same key expressed with native integers is accepted, so the reported
    # representation is not an accident of Decimal handling.
    native = dict(coherent, pod_order_no=1, institution_id=4, pod_type_id=1, rev_num=0)
    assert issues_for(native) == ()

    # A suffix expressed with Decimal is compared with the same canonical integer.
    suffix = dict(coherent, institution_id=Decimal(2))
    assert reports(issues_for(suffix), "pod_id", "invalid_value")

    # Boolean and fractional values are not accepted integer representations: the
    # declared column error stands and the coherence pass invents no key for them.
    for value in (True, Decimal("1.5")):
        issues = issues_for(dict(coherent, pod_order_no=value))
        assert reports(issues, "pod_order_no", "type_mismatch")
        assert not reports(issues, "pod_order_no", "invalid_value")


def test_semoga_exception_permits_only_the_exact_unissued_pair():
    load_pod_contract, validate, batch_type = _api()
    contract = load_pod_contract()

    def issues_for(row):
        return validate(_batch_with_rows(contract, batch_type, {"m_pod": [row]}), contract)

    def reports(issues, column, code):
        return any(
            issue.table == "m_pod" and issue.column == column and issue.code == code
            for issue in issues
        )

    def semoga(**overrides):
        row = _row(contract, "m_pod")
        row.update(
            {
                "pod_id_seq": 856,
                "approval_date": date(1997, 6, 1),
                "pod_name": "Pengembangan Lapangan Semoga",
                "institution_id": 4,
                "pod_type_id": 1,
                "rev_num": 0,
            }
        )
        row.update(overrides)
        return row

    # The accepted placeholder is the whole unissued pair.
    assert not any(
        issue.column in {"pod_id", "pod_order_no"}
        for issue in issues_for(semoga(pod_id=None, pod_order_no=None))
    )

    # A supplied pair on the exception row is still compared with the canonical
    # formula for its approval date, so an incoherent pair rejects.
    assert reports(
        issues_for(semoga(pod_id="PL-1997-0001-4-1-0", pod_order_no=1)),
        "pod_order_no",
        "invalid_value",
    )

    # Each half on its own is not the exception; the pair must be issued together.
    assert reports(
        issues_for(semoga(pod_id=None, pod_order_no=3021)),
        "pod_id",
        "invalid_value",
    )
    assert reports(
        issues_for(semoga(pod_id="PL-1997-3021-4-1-0", pod_order_no=None)),
        "pod_order_no",
        "invalid_value",
    )

    # An explicitly resolved pair that matches the canonical formula stays valid.
    assert issues_for(semoga(pod_id="PL-1997-3021-4-1-0", pod_order_no=3021)) == ()


def test_identity_coherence_reports_issues_instead_of_crashing_on_invalid_types():
    load_pod_contract, validate, batch_type = _api()
    contract = load_pod_contract()
    row = _row(contract, "m_pod")
    row.update(
        {
            "approval_date": date(1997, 6, 1),
            "pod_order_no": "1",
            "institution_id": "4",
            "pod_id": 1,
        }
    )
    batch = _batch_with_rows(contract, batch_type, {"m_pod": [row]})

    issues = validate(batch, contract)

    assert any(
        issue.table == "m_pod"
        and issue.column == "pod_order_no"
        and issue.code == "type_mismatch"
        for issue in issues
    )
    assert any(
        issue.table == "m_pod"
        and issue.column == "pod_id"
        and issue.code == "type_mismatch"
        for issue in issues
    )
