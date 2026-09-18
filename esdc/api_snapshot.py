"""Transactional SQLite repository for normalized project-resource snapshots."""

from __future__ import annotations

import sqlite3
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

JSONScalar = str | int | float | bool | None


@dataclass(frozen=True)
class ApiProjectResourceRow:
    """Normalized API row persisted in ``api_project_resources``."""

    project_id: str | None = None
    report_year: JSONScalar = None
    project_name: JSONScalar = None
    report_date: JSONScalar = None
    field_geom: JSONScalar = None
    groovy_report: JSONScalar = None
    field_id: JSONScalar = None
    field_name: JSONScalar = None
    wk_id: JSONScalar = None
    wk_name: JSONScalar = None
    operator_name: JSONScalar = None
    province: JSONScalar = None
    project_stage: JSONScalar = None
    project_class: JSONScalar = None
    project_level: JSONScalar = None
    uncert_level: JSONScalar = None
    project_remarks: JSONScalar = None
    is_discovered: JSONScalar = None
    is_ltp: JSONScalar = None
    is_pse_approved: JSONScalar = None
    is_pod_approved: JSONScalar = None
    is_report_accepted: JSONScalar = None
    is_unitization: JSONScalar = None
    project_eol: JSONScalar = None
    project_isactive: JSONScalar = None
    groovy_isactive: JSONScalar = None
    fusion_isactive: JSONScalar = None
    onstream_year: JSONScalar = None
    onstream_actual: JSONScalar = None
    rec_oil: JSONScalar = None
    rec_con: JSONScalar = None
    rec_ga: JSONScalar = None
    rec_gn: JSONScalar = None
    rec_oc: JSONScalar = None
    rec_an: JSONScalar = None
    rec_mboe: JSONScalar = None
    wk_lat: JSONScalar = None
    wk_long: JSONScalar = None
    wk_area: JSONScalar = None
    operator_group: JSONScalar = None
    basin86_id: JSONScalar = None
    basin86: JSONScalar = None
    basin128_id: JSONScalar = None
    basin128: JSONScalar = None
    prov_id: JSONScalar = None
    field_lat: JSONScalar = None
    field_long: JSONScalar = None
    field_area: JSONScalar = None
    pod_letter_num: JSONScalar = None
    pod_name: JSONScalar = None
    pod_date: JSONScalar = None
    project_name_previous: JSONScalar = None
    project_level_previous: JSONScalar = None
    prod_stage: JSONScalar = None
    wk_subgroup: JSONScalar = None
    psc_eff_start: JSONScalar = None
    psc_eff_end: JSONScalar = None
    is_offshore: JSONScalar = None
    report_status: JSONScalar = None
    wk_area_ori: JSONScalar = None
    wk_area_latest: JSONScalar = None
    wk_area_perwakilan_skkmigas: JSONScalar = None
    wk_regionisasi_ngi: JSONScalar = None
    field_name_previous: JSONScalar = None
    vol_remarks: JSONScalar = None
    gcf_srock: JSONScalar = None
    gcf_res: JSONScalar = None
    gcf_ts: JSONScalar = None
    gcf_dyn: JSONScalar = None
    gcf_total: JSONScalar = None
    pd: JSONScalar = None
    rec_oil_risked: JSONScalar = None
    rec_con_risked: JSONScalar = None
    rec_ga_risked: JSONScalar = None
    rec_gn_risked: JSONScalar = None
    rec_oc_risked: JSONScalar = None
    rec_an_risked: JSONScalar = None
    rec_mboe_risked: JSONScalar = None
    res_oil: JSONScalar = None
    res_con: JSONScalar = None
    res_ga: JSONScalar = None
    res_gn: JSONScalar = None
    res_oc: JSONScalar = None
    res_an: JSONScalar = None
    eur_rec_oil: JSONScalar = None
    eur_rec_con: JSONScalar = None
    eur_rec_ga: JSONScalar = None
    eur_rec_gn: JSONScalar = None
    eur_rec_oc: JSONScalar = None
    eur_rec_an: JSONScalar = None
    eur_res_oil: JSONScalar = None
    eur_res_con: JSONScalar = None
    eur_res_ga: JSONScalar = None
    eur_res_gn: JSONScalar = None
    eur_res_oc: JSONScalar = None
    eur_res_an: JSONScalar = None
    rf_oil: JSONScalar = None
    rf_gas: JSONScalar = None
    urf_rec_oil: JSONScalar = None
    urf_rec_gas: JSONScalar = None
    urf_res_oil: JSONScalar = None
    urf_res_gas: JSONScalar = None
    prj_ioip: JSONScalar = None
    prj_igip: JSONScalar = None
    rate_grs_oil: JSONScalar = None
    rate_grs_con: JSONScalar = None
    rate_grs_ga: JSONScalar = None
    rate_grs_gn: JSONScalar = None
    rate_grs_oc: JSONScalar = None
    rate_grs_an: JSONScalar = None
    rate_sls_oil: JSONScalar = None
    rate_sls_con: JSONScalar = None
    rate_sls_ga: JSONScalar = None
    rate_sls_gn: JSONScalar = None
    rate_sls_oc: JSONScalar = None
    rate_sls_an: JSONScalar = None
    cprd_grs_oil: JSONScalar = None
    cprd_grs_con: JSONScalar = None
    cprd_grs_ga: JSONScalar = None
    cprd_grs_gn: JSONScalar = None
    cprd_grs_oc: JSONScalar = None
    cprd_grs_an: JSONScalar = None
    cprd_sls_oil: JSONScalar = None
    cprd_sls_con: JSONScalar = None
    cprd_sls_ga: JSONScalar = None
    cprd_sls_gn: JSONScalar = None
    cprd_sls_oc: JSONScalar = None
    cprd_sls_an: JSONScalar = None
    dcpy_um_oil: JSONScalar = None
    dcpy_um_con: JSONScalar = None
    dcpy_um_ga: JSONScalar = None
    dcpy_um_gn: JSONScalar = None
    dcpy_um_oc: JSONScalar = None
    dcpy_um_an: JSONScalar = None
    dcpy_ppa_oil: JSONScalar = None
    dcpy_ppa_con: JSONScalar = None
    dcpy_ppa_ga: JSONScalar = None
    dcpy_ppa_gn: JSONScalar = None
    dcpy_ppa_oc: JSONScalar = None
    dcpy_ppa_an: JSONScalar = None
    dcpy_wi_oil: JSONScalar = None
    dcpy_wi_con: JSONScalar = None
    dcpy_wi_ga: JSONScalar = None
    dcpy_wi_gn: JSONScalar = None
    dcpy_wi_oc: JSONScalar = None
    dcpy_wi_an: JSONScalar = None
    dcpy_cio_oil: JSONScalar = None
    dcpy_cio_con: JSONScalar = None
    dcpy_cio_ga: JSONScalar = None
    dcpy_cio_gn: JSONScalar = None
    dcpy_cio_oc: JSONScalar = None
    dcpy_cio_an: JSONScalar = None
    dcpy_uc_oil: JSONScalar = None
    dcpy_uc_con: JSONScalar = None
    dcpy_uc_ga: JSONScalar = None
    dcpy_uc_gn: JSONScalar = None
    dcpy_uc_oc: JSONScalar = None
    dcpy_uc_an: JSONScalar = None
    dcpy_gtr_oil: JSONScalar = None
    dcpy_gtr_con: JSONScalar = None
    dcpy_gtr_ga: JSONScalar = None
    dcpy_gtr_gn: JSONScalar = None
    dcpy_gtr_oc: JSONScalar = None
    dcpy_gtr_an: JSONScalar = None
    ghv_avg: JSONScalar = None


@dataclass(frozen=True)
class ApiSnapshot:
    source_table: str
    revision: str
    row_count: int
    source_sha256: str
    fetched_at: str
    publication_state: Literal["committed", "published", "analytics_stale"]
    invalid_geometry_count: int = 0


_ROW_COLUMNS = tuple(ApiProjectResourceRow.__dataclass_fields__)
_ROW_INSERT_SQL = (
    "INSERT INTO api_project_resources ("
    + ", ".join(_ROW_COLUMNS)
    + ") VALUES ("
    + ", ".join("?" for _ in _ROW_COLUMNS)
    + ")"
)


def replace_project_resources_snapshot(
    conn: sqlite3.Connection,
    rows: Sequence[ApiProjectResourceRow],
    snapshot: ApiSnapshot,
) -> None:
    """Atomically replace rows and metadata, preserving the prior state on error."""
    if snapshot.source_table != "project_resources":
        raise ValueError("snapshot source_table must be 'project_resources'")
    if snapshot.row_count != len(rows):
        raise ValueError(
            f"snapshot row_count {snapshot.row_count} does not match {len(rows)} rows"
        )
    revision = str(uuid.uuid4())
    committed = ApiSnapshot(
        source_table=snapshot.source_table,
        revision=revision,
        row_count=snapshot.row_count,
        source_sha256=snapshot.source_sha256,
        fetched_at=snapshot.fetched_at,
        publication_state="committed",
        invalid_geometry_count=snapshot.invalid_geometry_count,
    )
    try:
        conn.execute("BEGIN")
        conn.execute("DELETE FROM api_project_resources")
        conn.executemany(
            _ROW_INSERT_SQL,
            [tuple(getattr(row, column) for column in _ROW_COLUMNS) for row in rows],
        )
        conn.execute(
            """INSERT INTO api_snapshot
               (source_table, revision, row_count, source_sha256,
                fetched_at, publication_state, invalid_geometry_count)
               VALUES (?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(source_table) DO UPDATE SET
                   revision = excluded.revision,
                   row_count = excluded.row_count,
                   source_sha256 = excluded.source_sha256,
                   fetched_at = excluded.fetched_at,
                   publication_state = excluded.publication_state,
                   invalid_geometry_count = excluded.invalid_geometry_count""",
            (
                committed.source_table,
                committed.revision,
                committed.row_count,
                committed.source_sha256,
                committed.fetched_at,
                committed.publication_state,
                committed.invalid_geometry_count,
            ),
        )
        conn.commit()
    except BaseException:
        conn.rollback()
        raise


def get_project_resources_snapshot(conn: sqlite3.Connection) -> ApiSnapshot | None:
    row = conn.execute(
        """SELECT source_table, revision, row_count, source_sha256,
                  fetched_at, publication_state, invalid_geometry_count
           FROM api_snapshot WHERE source_table = ?""",
        ("project_resources",),
    ).fetchone()
    if row is None:
        return None
    return ApiSnapshot(
        source_table=row[0],
        revision=row[1],
        row_count=row[2],
        source_sha256=row[3],
        fetched_at=row[4],
        publication_state=row[5],
        invalid_geometry_count=row[6],
    )


def mark_project_resources_published(conn: sqlite3.Connection, revision: str) -> None:
    cursor = conn.execute(
        """UPDATE api_snapshot SET publication_state = 'published'
           WHERE source_table = ? AND revision = ?""",
        ("project_resources", revision),
    )
    if cursor.rowcount != 1:
        conn.rollback()
        raise ValueError(f"no project_resources snapshot matches revision {revision!r}")
    conn.commit()
