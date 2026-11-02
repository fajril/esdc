"""Strict decoding of POD API JSON payloads (Task5A parser slice).

The parser is pure: it turns one raw response body into record mappings without
consulting a database, projecting contract columns, or validating business
values. Logical typing, UUID reconciliation, and row validation remain in the
later Task5B/C slices.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal
from typing import NoReturn

from esdc.pod_registry.contract import PodContract, load_pod_contract

ALLOWED_TABLE_NAMES = frozenset(
    {"m_project", "m_pod", "pod_project", "pod_monitoring", "pod_forecast"}
)
DATA_FIELD = "data"
METADATA_FIELDS = ("contract_id", "contract_version", "dictionary_version")
ENVELOPE_FIELDS = frozenset((*METADATA_FIELDS, DATA_FIELD))
WRAPPER_FIELDS = frozenset({"error", "success", "status", "message", "code"})


class PodApiPayloadError(ValueError):
    """Rejected payload reason naming the table, never the raw payload."""

    def __init__(self, table_name: str, reason: str) -> None:
        self.table_name = table_name
        self.reason = reason
        super().__init__(f"{table_name}: {reason}")


@dataclass(frozen=True, slots=True)
class _ParsedObject:
    """A JSON object kept with its original pairs so duplicates stay visible."""

    pairs: tuple[tuple[str, object], ...]


def parse_pod_api_payload(
    payload: bytes,
    *,
    table_name: str,
    contract: PodContract | None = None,
) -> list[dict[str, object]]:
    """Decode one strict JSON body into raw record mappings for the table."""
    if table_name not in ALLOWED_TABLE_NAMES:
        raise PodApiPayloadError(
            table_name,
            f"table name {table_name!r} is not an allowed POD API table",
        )
    active_contract = contract if contract is not None else load_pod_contract()
    text = _decode_text(payload, table_name)
    document = _normalize(_decode_json(text, table_name), "$", table_name)
    if isinstance(document, list):
        return [
            _record(value, f"$[{index}]", table_name)
            for index, value in enumerate(document)
        ]
    if isinstance(document, dict):
        return _envelope_records(document, table_name, active_contract)
    raise PodApiPayloadError(
        table_name,
        "root must be a record array or an envelope object",
    )


def _decode_text(payload: bytes, table_name: str) -> str:
    if not isinstance(payload, (bytes, bytearray)):
        raise PodApiPayloadError(table_name, "payload must be raw bytes")
    try:
        return bytes(payload).decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PodApiPayloadError(
            table_name,
            f"payload is not valid UTF-8 at byte offset {exc.start}",
        ) from None


def _decode_json(text: str, table_name: str) -> object:
    try:
        return _decoder(table_name).decode(text)
    except RecursionError:
        raise PodApiPayloadError(
            table_name,
            "nesting depth exceeds the decoder boundary",
        ) from None
    except json.JSONDecodeError as exc:
        raise PodApiPayloadError(
            table_name,
            f"malformed JSON: {exc.msg} at offset {exc.pos}",
        ) from None


def _decoder(table_name: str) -> json.JSONDecoder:
    return json.JSONDecoder(
        object_pairs_hook=lambda pairs: _ParsedObject(tuple(pairs)),
        parse_float=_parse_decimal(table_name),
        parse_int=_parse_int(table_name),
        parse_constant=_reject_constant(table_name),
    )


def _parse_int(table_name: str) -> Callable[[str], int]:
    def parse(token: str) -> int:
        try:
            return int(token)
        except ValueError:
            raise PodApiPayloadError(
                table_name,
                "integer token exceeds the decoder numeric boundary",
            ) from None

    return parse


def _parse_decimal(table_name: str) -> Callable[[str], Decimal]:
    def parse(token: str) -> Decimal:
        try:
            return Decimal(token)
        except ArithmeticError:
            raise PodApiPayloadError(
                table_name,
                "numeric token exceeds the decoder numeric boundary",
            ) from None

    return parse


def _reject_constant(table_name: str) -> Callable[[str], NoReturn]:
    def reject(token: str) -> NoReturn:
        raise PodApiPayloadError(
            table_name,
            f"non-finite numeric literal {token!r} is not valid JSON",
        )

    return reject


def _normalize(value: object, path: str, table_name: str) -> object:
    """Guard the recursive walk so a deep container stays a contextual error."""
    try:
        return _walk(value, path, table_name)
    except RecursionError:
        raise PodApiPayloadError(
            table_name,
            "nesting depth exceeds the normalization boundary",
        ) from None


def _walk(value: object, path: str, table_name: str) -> object:
    """Rebuild decoded containers pre-order so key order and duplicates survive."""
    if isinstance(value, _ParsedObject):
        mapping: dict[str, object] = {}
        for key, item in value.pairs:
            if key in mapping:
                raise PodApiPayloadError(
                    table_name,
                    f"duplicate object key {key!r} at {path}",
                )
            mapping[key] = _walk(item, f"{path}.{key}", table_name)
        return mapping
    if isinstance(value, list):
        return [
            _walk(item, f"{path}[{index}]", table_name)
            for index, item in enumerate(value)
        ]
    return value


def _envelope_records(
    envelope: dict[str, object],
    table_name: str,
    contract: PodContract,
) -> list[dict[str, object]]:
    if DATA_FIELD not in envelope:
        if any(field in WRAPPER_FIELDS for field in envelope):
            raise PodApiPayloadError(
                table_name,
                "envelope is an error or status wrapper body, not a data payload",
            )
        raise PodApiPayloadError(table_name, "envelope must contain a data array")
    unknown = [field for field in envelope if field not in ENVELOPE_FIELDS]
    if unknown:
        raise PodApiPayloadError(
            table_name,
            f"envelope field {unknown[0]!r} is not allowed",
        )
    data = envelope[DATA_FIELD]
    if not isinstance(data, list):
        raise PodApiPayloadError(
            table_name,
            f"envelope field {DATA_FIELD!r} must be an array of records",
        )
    records = [
        _record(value, f"$.{DATA_FIELD}[{index}]", table_name)
        for index, value in enumerate(data)
    ]
    _check_metadata(envelope, table_name, contract)
    return records


def _record(value: object, path: str, table_name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise PodApiPayloadError(table_name, f"record at {path} must be a JSON object")
    return value


def _check_metadata(
    envelope: dict[str, object],
    table_name: str,
    contract: PodContract,
) -> None:
    supplied = [field for field in METADATA_FIELDS if field in envelope]
    if not supplied:
        return
    if len(supplied) != len(METADATA_FIELDS):
        raise PodApiPayloadError(
            table_name,
            f"metadata is incomplete: {supplied[0]!r} was supplied without all of "
            f"{', '.join(METADATA_FIELDS)}",
        )
    expected = {
        "contract_id": contract.contract_id,
        "contract_version": contract.contract_version,
        "dictionary_version": contract.dictionary_version,
    }
    for field in METADATA_FIELDS:
        value = envelope[field]
        if field == "contract_version":
            if isinstance(value, bool) or not isinstance(value, int):
                raise PodApiPayloadError(
                    table_name,
                    f"metadata field {field!r} must be an integer, not a boolean "
                    "or a fraction",
                )
        elif not isinstance(value, str):
            raise PodApiPayloadError(
                table_name,
                f"metadata field {field!r} must be a string",
            )
        if value != expected[field]:
            raise PodApiPayloadError(
                table_name,
                f"metadata field {field!r} does not match the current contract",
            )
