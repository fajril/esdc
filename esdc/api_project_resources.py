"""Strict normalization for project-resource API rows."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from esdc.api_snapshot import ApiProjectResourceRow, JSONScalar

_SOURCE_FIELDS: tuple[str, ...] = (
    "project_id",
    "report_year",
    "project_name",
    "report_date",
    "field_id",
    "field_name",
    "wk_id",
    "wk_name",
    "operator_name",
    "province",
    "project_stage",
    "project_class",
    "project_level",
    "uncert_level",
    "project_remarks",
    "is_discovered",
    "is_ltp",
    "is_pse_approved",
    "is_pod_approved",
    "is_report_accepted",
    "is_unitization",
    "project_eol",
    "project_isactive",
    "groovy_isactive",
    "fusion_isactive",
    "onstream_year",
    "onstream_actual",
    "rec_oil",
    "rec_con",
    "rec_ga",
    "rec_gn",
    "rec_oc",
    "rec_an",
    "rec_mboe",
    "wk_lat",
    "wk_long",
    "wk_area",
    "operator_group",
    "basin128_id",
    "basin128",
    "prov_id",
    "field_lat",
    "field_long",
    "field_area",
    "pod_letter_num",
    "pod_name",
    "pod_date",
    "project_name_previous",
    "project_level_previous",
    "prod_stage",
    "wk_subgroup",
    "psc_eff_start",
    "psc_eff_end",
    "is_offshore",
    "report_status",
    "wk_area_ori",
    "wk_area_latest",
    "wk_area_perwakilan_skkmigas",
    "wk_regionisasi_ngi",
    "field_name_previous",
    "vol_remarks",
    "gcf_srock",
    "gcf_res",
    "gcf_ts",
    "gcf_dyn",
    "gcf_total",
    "pd",
    "rec_oil_risked",
    "rec_con_risked",
    "rec_ga_risked",
    "rec_gn_risked",
    "rec_oc_risked",
    "rec_an_risked",
    "rec_mboe_risked",
    "res_oil",
    "res_con",
    "res_ga",
    "res_gn",
    "res_oc",
    "res_an",
    "eur_rec_oil",
    "eur_rec_con",
    "eur_rec_ga",
    "eur_rec_gn",
    "eur_rec_oc",
    "eur_rec_an",
    "eur_res_oil",
    "eur_res_con",
    "eur_res_ga",
    "eur_res_gn",
    "eur_res_oc",
    "eur_res_an",
    "rf_oil",
    "rf_gas",
    "urf_rec_oil",
    "urf_rec_gas",
    "urf_res_oil",
    "urf_res_gas",
    "prj_ioip",
    "prj_igip",
    "rate_grs_oil",
    "rate_grs_con",
    "rate_grs_ga",
    "rate_grs_gn",
    "rate_grs_oc",
    "rate_grs_an",
    "rate_sls_oil",
    "rate_sls_con",
    "rate_sls_ga",
    "rate_sls_gn",
    "rate_sls_oc",
    "rate_sls_an",
    "cprd_grs_oil",
    "cprd_grs_con",
    "cprd_grs_ga",
    "cprd_grs_gn",
    "cprd_grs_oc",
    "cprd_grs_an",
    "cprd_sls_oil",
    "cprd_sls_con",
    "cprd_sls_ga",
    "cprd_sls_gn",
    "cprd_sls_oc",
    "cprd_sls_an",
    "dcpy_um_oil",
    "dcpy_um_con",
    "dcpy_um_ga",
    "dcpy_um_gn",
    "dcpy_um_oc",
    "dcpy_um_an",
    "dcpy_ppa_oil",
    "dcpy_ppa_con",
    "dcpy_ppa_ga",
    "dcpy_ppa_gn",
    "dcpy_ppa_oc",
    "dcpy_ppa_an",
    "dcpy_wi_oil",
    "dcpy_wi_con",
    "dcpy_wi_ga",
    "dcpy_wi_gn",
    "dcpy_wi_oc",
    "dcpy_wi_an",
    "dcpy_cio_oil",
    "dcpy_cio_con",
    "dcpy_cio_ga",
    "dcpy_cio_gn",
    "dcpy_cio_oc",
    "dcpy_cio_an",
    "dcpy_uc_oil",
    "dcpy_uc_con",
    "dcpy_uc_ga",
    "dcpy_uc_gn",
    "dcpy_uc_oc",
    "dcpy_uc_an",
    "dcpy_gtr_oil",
    "dcpy_gtr_con",
    "dcpy_gtr_ga",
    "dcpy_gtr_gn",
    "dcpy_gtr_oc",
    "dcpy_gtr_an",
    "ghv_avg",
)
_GROV_FIELDS: tuple[str, ...] = (
    "grov_current_situation",
    "grov_executive_summary",
    "grov_key_challenges",
    "grov_solution_proposal",
    "grov_status",
)
_ALLOWED_SOURCE_FIELDS = frozenset((*_SOURCE_FIELDS, "field_geom", *_GROV_FIELDS))


@dataclass(frozen=True)
class NormalizedProjectResources:
    rows: tuple[ApiProjectResourceRow, ...]
    source_sha256: str


def _narrative(value: JSONScalar, field: str, index: int) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"record {index} field {field!r} must be a string or null")
    text = value.strip()
    return text or None


def _groovy_report(row: Mapping[str, JSONScalar], index: int) -> str | None:
    sections: list[str] = []
    labels = (
        ("grov_current_situation", "Current situation"),
        ("grov_executive_summary", "Executive summary"),
        ("grov_key_challenges", "Key challenges"),
        ("grov_solution_proposal", "Solution proposal"),
    )
    for field, label in labels:
        text = _narrative(row.get(field), field, index)
        if text is not None:
            sections.append(f"{label}:\n{text}")
    return "\n\n".join(sections) or None


def _project_id(value: JSONScalar) -> str | None:
    if value is None or isinstance(value, str):
        return value
    raise ValueError("project_id must be a string or null")


def _row_from_source(
    row: Mapping[str, JSONScalar], report: str | None
) -> ApiProjectResourceRow:
    values: dict[str, JSONScalar] = {
        field: row.get(field) for field in _SOURCE_FIELDS if field != "project_id"
    }
    values["field_geom"] = row.get("field_geom")
    values["groovy_report"] = report
    return ApiProjectResourceRow(
        project_id=_project_id(row.get("project_id")), **values
    )


def normalize_project_resources(
    payload: Sequence[Mapping[str, JSONScalar]],
) -> NormalizedProjectResources:
    for index, row in enumerate(payload):
        unknown = sorted(set(row) - _ALLOWED_SOURCE_FIELDS)
        if unknown:
            names = ", ".join(unknown)
            raise ValueError(f"record {index} contains unknown keys: {names}")

    canonical = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    source_sha256 = hashlib.sha256(canonical).hexdigest()
    rows = tuple(
        _row_from_source(row, _groovy_report(row, index))
        for index, row in enumerate(payload)
    )
    return NormalizedProjectResources(rows=rows, source_sha256=source_sha256)
