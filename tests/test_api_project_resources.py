"""Focused tests for strict project-resource normalization."""

from __future__ import annotations

import hashlib
import json

import pytest

from esdc.api_project_resources import _SOURCE_FIELDS, normalize_project_resources
from esdc.api_snapshot import ApiProjectResourceRow, JSONScalar


def base_row(**overrides: JSONScalar) -> dict[str, JSONScalar]:
    row: dict[str, JSONScalar] = {
        "project_id": "P-1",
        "project_name": "Example",
        "report_year": 2025,
        "field_geom": "POINT (1 2)",
        "grov_current_situation": None,
        "grov_executive_summary": None,
        "grov_key_challenges": None,
        "grov_solution_proposal": None,
        "grov_status": None,
    }
    row.update(overrides)
    return row


def test_normalize_labels_nonempty_grov_sections_in_stable_order() -> None:
    row = base_row(
        grov_current_situation="Current",
        grov_executive_summary=" ",
        grov_key_challenges="Challenge",
        grov_solution_proposal="Solution",
        grov_status="aprv",
    )

    result = normalize_project_resources([row])

    assert result.rows[0].groovy_report == (
        "Current situation:\nCurrent\n\n"
        "Key challenges:\nChallenge\n\n"
        "Solution proposal:\nSolution"
    )
    assert "grov_" not in result.rows[0].__dict__


def test_normalize_all_empty_narratives_to_none() -> None:
    result = normalize_project_resources(
        [
            base_row(
                grov_current_situation=" \n",
                grov_executive_summary="",
                grov_key_challenges=None,
                grov_solution_proposal="\t",
            )
        ]
    )

    assert result.rows[0].groovy_report is None


def test_missing_historical_grov_keys_are_accepted() -> None:
    row = base_row()
    for key in (
        "grov_current_situation",
        "grov_executive_summary",
        "grov_key_challenges",
        "grov_solution_proposal",
        "grov_status",
    ):
        row.pop(key)

    result = normalize_project_resources([row])

    assert result.rows[0].groovy_report is None


def test_unknown_key_rejected_with_record_index_and_only_unknown_names() -> None:
    with pytest.raises(ValueError, match=r"record 0.*unexpected_field") as error:
        normalize_project_resources([base_row(unexpected_field="nope")])

    message = str(error.value)
    assert "project_id" not in message
    assert "grov_status" not in message


def test_basin86_keys_are_not_accepted_as_source_inputs() -> None:
    with pytest.raises(ValueError, match=r"basin86(?:_id)?"):
        normalize_project_resources([base_row(basin86_id="B-86")])


def test_ragged_key_order_has_stable_hash_and_named_values() -> None:
    first = base_row(project_id="first", project_name="First")
    second = dict(reversed(list(first.items())))

    result_one = normalize_project_resources([first])
    result_two = normalize_project_resources([second])

    assert result_one.source_sha256 == result_two.source_sha256
    assert result_one.rows == result_two.rows
    canonical = json.dumps(
        [first], ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    assert (
        result_one.source_sha256
        == hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    )


def test_field_geom_wkt_text_is_preserved_exactly() -> None:
    wkt = "  MULTIPOLYGON (((1 2, 3 4, 1 2)))  "

    result = normalize_project_resources([base_row(field_geom=wkt)])

    assert result.rows[0].field_geom == wkt


def test_named_numeric_and_boolean_values_are_retained() -> None:
    result = normalize_project_resources(
        [base_row(rec_oil=12.5, is_discovered=True, report_year=2025)]
    )

    assert result.rows[0].rec_oil == 12.5
    assert result.rows[0].is_discovered is True
    assert result.rows[0].report_year == 2025


def test_model_schema_excludes_raw_grov_fields() -> None:
    model_fields = set(ApiProjectResourceRow.__dataclass_fields__)
    expected = set(_SOURCE_FIELDS) | {"field_geom", "groovy_report"}
    assert expected == model_fields - {"basin86_id", "basin86"}
    assert model_fields - expected == {"basin86_id", "basin86"}
    assert not any(field.startswith("grov_") for field in model_fields)
