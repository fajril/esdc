"""Task5A strict POD API JSON payload decoder regressions."""

from __future__ import annotations

import decimal
import json
import sys
from dataclasses import replace
from decimal import Decimal

import pytest

from esdc.pod_registry.api_payload import PodApiPayloadError, parse_pod_api_payload
from esdc.pod_registry.contract import load_pod_contract

ALLOWED_TABLES = (
    "m_project",
    "m_pod",
    "pod_project",
    "pod_monitoring",
    "pod_forecast",
)
SECRET = "TOP-SECRET-PROJECT-TOKEN"


def _parse(payload: bytes, *, table_name: str = "m_pod") -> list[dict[str, object]]:
    return parse_pod_api_payload(payload, table_name=table_name)


def _metadata_envelope() -> bytes:
    contract = load_pod_contract()
    return (
        f'{{"data": [], "contract_id": "{contract.contract_id}", '
        f'"contract_version": {contract.contract_version}, '
        f'"dictionary_version": "{contract.dictionary_version}"}}'
    ).encode()


def _deep_document(depth: int) -> object:
    """Build an over-deep decoded document without recursing to build it."""
    value: object = []
    for _ in range(depth):
        value = [value]
    return value


def test_payload_error_is_a_value_error_with_the_table_name():
    assert issubclass(PodApiPayloadError, ValueError)
    with pytest.raises(PodApiPayloadError) as exc_info:
        _parse(b"42", table_name="pod_forecast")
    assert exc_info.value.table_name == "pod_forecast"
    assert str(exc_info.value).startswith("pod_forecast:")
    assert exc_info.value.reason


def test_bare_array_layout_returns_record_objects():
    records = _parse(b'[{"pod_id_seq": 7, "pod_id": "PL-2024-0007-1-1-0"}]')
    assert records == [{"pod_id_seq": 7, "pod_id": "PL-2024-0007-1-1-0"}]
    assert isinstance(records, list)
    assert isinstance(records[0], dict)


def test_envelope_layout_returns_data_records():
    records = _parse(b'{"data": [{"monpod_id": 12}]}')
    assert records == [{"monpod_id": 12}]


def test_envelope_metadata_is_optional_but_must_be_complete():
    assert _parse(b"[]") == []
    assert _parse(b'{"data": []}') == []
    assert _parse(_metadata_envelope()) == []
    with pytest.raises(PodApiPayloadError) as exc_info:
        _parse(b'{"data": [], "contract_id": "esdc.pod"}')
    assert "metadata is incomplete" in str(exc_info.value)


def test_metadata_must_match_the_contract_supplied_to_the_parser():
    contract = load_pod_contract()
    payload = _metadata_envelope()
    assert parse_pod_api_payload(payload, table_name="m_pod", contract=contract) == []
    mismatched = replace(contract, contract_version=contract.contract_version + 1)
    with pytest.raises(PodApiPayloadError) as exc_info:
        parse_pod_api_payload(payload, table_name="m_pod", contract=mismatched)
    assert "metadata field 'contract_version' does not match" in str(exc_info.value)
    with pytest.raises(PodApiPayloadError) as exc_info:
        _parse(
            b'{"data": [], "contract_id": "other", "contract_version": 1, '
            b'"dictionary_version": "0.4.0"}'
        )
    assert "metadata field 'contract_id' does not match" in str(exc_info.value)


def test_exact_fractional_token_is_preserved_as_decimal_not_float():
    records = _parse(b'[{"measured": 12345678901234567890.123456789}]')
    value = records[0]["measured"]
    assert isinstance(value, Decimal)
    assert str(value) == "12345678901234567890.123456789"
    assert value != Decimal(str(float("12345678901234567890.123456789")))


def test_exponent_token_remains_decimal_even_for_whole_values():
    records = _parse(b'[{"irr": 1.5e3, "big": 2E+300}]')
    assert isinstance(records[0]["irr"], Decimal)
    assert records[0]["irr"] == Decimal("1.5E+3")
    assert isinstance(records[0]["big"], Decimal)
    assert str(records[0]["big"]) == "2E+300"


def test_integer_and_bool_null_zero_and_string_are_kept_distinctly():
    records = _parse(b'[{"i": 42, "z": 0, "t": true, "f": false, "s": "42", "n": null}]')
    row = records[0]
    assert row["i"] == 42 and type(row["i"]) is int
    assert row["z"] == 0 and type(row["z"]) is int
    assert row["t"] is True
    assert row["f"] is False
    assert row["s"] == "42" and type(row["s"]) is str
    assert row["n"] is None


def test_quoted_numeric_text_is_never_coerced_to_a_number():
    records = _parse(b'[{"value": "12.50", "flag": "false", "empty": ""}]')
    assert records[0]["value"] == "12.50"
    assert not isinstance(records[0]["value"], Decimal)
    assert records[0]["flag"] == "false"
    assert records[0]["empty"] == ""


def test_nested_containers_stay_raw_for_the_normalizer():
    records = _parse(b'[{"attrs": {"nested": [1, 2.5, null], "flag": false}}]')
    nested = records[0]["attrs"]
    assert isinstance(nested, dict)
    assert isinstance(nested["nested"], list)
    assert nested["nested"][1] == Decimal("2.5")
    assert nested["flag"] is False
    assert nested["nested"][2] is None


@pytest.mark.parametrize(
    "payload",
    [
        b'[{"v": NaN}]',
        b'{"data": [{"v": Infinity}]}',
        b'[{"attrs": {"v": -Infinity}}]',
        b"[NaN]",
    ],
)
def test_non_finite_numeric_literals_are_rejected_everywhere(payload: bytes):
    with pytest.raises(PodApiPayloadError) as exc_info:
        _parse(payload)
    assert "non-finite" in str(exc_info.value)


@pytest.mark.parametrize(
    "payload",
    [
        b'{"data": [], "data": []}',
        b'[{"pod_id_seq": 1, "pod_id_seq": 2}]',
        b'[{"pod_id_seq": 1, "pod_id_seq": 1}]',
        b'[{"attrs": {"flag": 1, "flag": 2}}]',
        b'{"data": [{"a": [{"b": 1, "b": 2}]}]}',
    ],
)
def test_duplicate_object_keys_are_rejected_at_every_level(payload: bytes):
    with pytest.raises(PodApiPayloadError) as exc_info:
        _parse(payload)
    assert "duplicate object key" in str(exc_info.value)


@pytest.mark.parametrize(
    "payload",
    [
        b'[{"s": "\xff"}]',
        b'[{"s": "\xc3"}]',
        b'[{"s": "\xe0\x80\x80"}]',
        b'{"data": [{"s": "ok"}], "note": "\xc2"}',
    ],
)
def test_malformed_utf8_is_rejected_before_parsing(payload: bytes):
    with pytest.raises(PodApiPayloadError) as exc_info:
        _parse(payload)
    assert "UTF-8" in str(exc_info.value)


@pytest.mark.parametrize(
    "payload",
    [
        b"",
        b"   ",
        b'[{"a": 1},]',
        b"[{'a': 1}]",
        b'[{"a": 1}',
        b'[{"a": 1}extra]',
        b'{"data": []} {"data": []}',
        b'[{"a": "\x01"}]',
    ],
)
def test_malformed_json_is_rejected(payload: bytes):
    with pytest.raises(PodApiPayloadError) as exc_info:
        _parse(payload)
    assert "malformed JSON" in str(exc_info.value)


@pytest.mark.parametrize(
    ("payload", "reason"),
    [
        (b'"just a string"', "root"),
        (b"42", "root"),
        (b"true", "root"),
        (b"null", "root"),
        (b"[[1], 2]", "record"),
        (b'[{"a": 1}, []]', "record"),
        (b'{"data": null}', "data"),
        (b'{"data": {"a": 1}}', "data"),
        (b'{"data": 0}', "data"),
        (b'{"data": [1]}', "record"),
    ],
)
def test_invalid_containers_and_records_are_rejected(payload: bytes, reason: str):
    with pytest.raises(PodApiPayloadError) as exc_info:
        _parse(payload)
    assert reason in str(exc_info.value)


def test_error_shaped_http200_body_is_not_read_as_empty_data():
    for payload in (
        b'{"error": "not found"}',
        b'{"success": true, "status": "ok"}',
        b'{"data": [], "error": "partial"}',
        b'{"data": null, "status": "error"}',
    ):
        with pytest.raises(PodApiPayloadError) as exc_info:
            _parse(payload, table_name="pod_project")
        assert "envelope" in str(exc_info.value)
        assert exc_info.value.table_name == "pod_project"


@pytest.mark.parametrize(
    "payload",
    [
        b'{"data": [], "meta": {}}',
        b'{"data": [], "count": 0}',
        b'{"data": [], "table_name": "m_pod"}',
        b'{"data": [], "complete": true}',
        b'{"data": [], "contract_id": "esdc.pod", "contract_version": 1, '
        b'"dictionary_version": "0.4.0", "extra": 1}',
    ],
)
def test_unknown_envelope_fields_are_rejected(payload: bytes):
    with pytest.raises(PodApiPayloadError) as exc_info:
        _parse(payload)
    assert "envelope field" in str(exc_info.value)


@pytest.mark.parametrize(
    "payload",
    [
        b'{"data": [], "contract_id": true, "contract_version": 1, '
        b'"dictionary_version": "0.4.0"}',
        b'{"data": [], "contract_id": "esdc.pod", "contract_version": true, '
        b'"dictionary_version": "0.4.0"}',
        b'{"data": [], "contract_id": "esdc.pod", "contract_version": false, '
        b'"dictionary_version": "0.4.0"}',
        b'{"data": [], "contract_id": "esdc.pod", "contract_version": 1.0, '
        b'"dictionary_version": "0.4.0"}',
        b'{"data": [], "contract_id": "esdc.pod", "contract_version": "1", '
        b'"dictionary_version": 4}',
    ],
)
def test_metadata_types_are_strict_and_bool_never_counts_as_an_integer(
    payload: bytes,
):
    with pytest.raises(PodApiPayloadError) as exc_info:
        _parse(payload)
    assert "metadata field" in str(exc_info.value)


@pytest.mark.parametrize("table_name", ALLOWED_TABLES)
def test_only_the_declared_api_tables_are_accepted(table_name: str):
    assert _parse(b'[{"id": 1}]', table_name=table_name) == [{"id": 1}]
    for unknown in ("pod_monitoring_api", "r_pod_type", "m_pod ", ""):
        with pytest.raises(PodApiPayloadError) as exc_info:
            _parse(b'[{"id": 1}]', table_name=unknown)
        assert exc_info.value.table_name == unknown
        assert "allowed POD API table" in str(exc_info.value)


def test_error_never_echoes_the_raw_payload_or_a_secret():
    with pytest.raises(PodApiPayloadError) as exc_info:
        _parse(f'{{"data": ["{SECRET}"]}}'.encode(), table_name="m_project")
    message = str(exc_info.value)
    assert SECRET not in message
    assert "record" in message
    assert exc_info.value.table_name == "m_project"


def test_parser_is_pure_and_never_creates_or_opens_a_database(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("ESDC_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setenv("ESDC_DB_DIR", str(tmp_path / "db"))
    monkeypatch.setenv("ESDC_DB_FILE", str(tmp_path / "db" / "esdc.duckdb"))
    assert _parse(b"[]", table_name="pod_monitoring") == []
    assert [path for path in tmp_path.rglob("*") if path.is_file()] == []


def test_huge_integer_token_rejects_at_the_decoder_boundary_without_echoing_it():
    token = b"1" * 5000
    payload = b'[{"x":' + token + b"}]"
    with pytest.raises(PodApiPayloadError) as exc_info:
        _parse(payload)
    message = str(exc_info.value)
    assert exc_info.value.table_name == "m_pod"
    assert "decoder numeric boundary" in message
    assert token.decode() not in message
    assert len(message) < 200
    assert exc_info.value.__cause__ is None


def test_exponent_overflow_token_rejects_at_the_decoder_boundary_without_echoing_it():
    token = b"1e" + b"9" * 33
    payload = b'[{"x":' + token + b"}]"
    with pytest.raises(PodApiPayloadError) as exc_info:
        _parse(payload)
    message = str(exc_info.value)
    assert exc_info.value.table_name == "m_pod"
    assert "decoder numeric boundary" in message
    assert token.decode() not in message
    assert len(message) < 200
    assert exc_info.value.__cause__ is None


def test_deep_nesting_rejects_at_the_normalization_boundary_with_the_table_name(
    monkeypatch: pytest.MonkeyPatch,
):
    """Let the decoder accept the depth so the normalization walk owns the rejection."""
    depth = sys.getrecursionlimit() + 100
    document = _deep_document(depth)

    def accept_depth(self: json.JSONDecoder, text: str, *args: object) -> object:
        return document

    monkeypatch.setattr(json.JSONDecoder, "decode", accept_depth)
    with pytest.raises(PodApiPayloadError) as exc_info:
        _parse(b'{"data": [{"x": []}]}')
    message = str(exc_info.value)
    assert exc_info.value.table_name == "m_pod"
    assert "normalization boundary" in message
    assert "decoder boundary" not in message
    assert "[" * depth not in message
    assert len(message) < 200
    assert exc_info.value.__cause__ is None


def test_real_deep_json_input_rejects_at_a_boundary_and_keeps_the_table_name():
    """A real over-deep document can reject at either stage and must stay contextual."""
    depth = sys.getrecursionlimit() + 200
    payload = b'[{"x":' + b"[" * depth + b"]" * depth + b"}]"
    with pytest.raises(PodApiPayloadError) as exc_info:
        _parse(payload, table_name="pod_project")
    message = str(exc_info.value)
    assert exc_info.value.table_name == "pod_project"
    assert "boundary" in message
    assert "[" * depth not in message
    assert len(message) < 200
    assert exc_info.value.__cause__ is None


def test_decoder_recursion_failure_is_contextual_even_when_the_scanner_raises_it(
    monkeypatch: pytest.MonkeyPatch,
):
    def explode(self: json.JSONDecoder, text: str, *args: object) -> object:
        raise RecursionError("maximum recursion depth exceeded")

    monkeypatch.setattr(json.JSONDecoder, "decode", explode)
    with pytest.raises(PodApiPayloadError) as exc_info:
        _parse(b'{"data": [{"x": []}]}', table_name="pod_forecast")
    message = str(exc_info.value)
    assert exc_info.value.table_name == "pod_forecast"
    assert "decoder boundary" in message
    assert "normalization boundary" not in message
    assert len(message) < 200
    assert exc_info.value.__cause__ is None


def test_parser_never_changes_global_numeric_or_recursion_settings():
    limit_before = sys.getrecursionlimit()
    digits_before = sys.get_int_max_str_digits()
    emax_before = decimal.getcontext().Emax
    depth = limit_before + 100
    for payload in (
        b'[{"x":' + b"1" * 5000 + b"}]",
        b'[{"x":1e' + b"9" * 33 + b"}]",
        b'[{"x":' + b"[" * depth + b"]" * depth + b"}]",
    ):
        with pytest.raises(PodApiPayloadError):
            _parse(payload)
    assert sys.getrecursionlimit() == limit_before
    assert sys.get_int_max_str_digits() == digits_before
    assert decimal.getcontext().Emax == emax_before
