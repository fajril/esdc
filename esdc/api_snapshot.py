"""Transactional SQLite repository for normalized project-resource snapshots."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal

JSONScalar = str | int | float | bool | None


@dataclass(frozen=True)
class ApiProjectResourceRow:
    """Normalized API row persisted in ``api_project_resources``."""

    project_id: str | None = None
    report_year: JSONScalar = None
    project_name: JSONScalar = None
    report_date: JSONScalar = None
    # Legacy source WKT remains canonical for replay and snapshot hashes.
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


def _persisted_hash(conn: sqlite3.Connection) -> tuple[int, str]:
    rows = conn.execute(
        "SELECT " + ", ".join(_ROW_COLUMNS) + " FROM api_project_resources"
    ).fetchall()
    encoded = sorted(
        json.dumps(tuple(row), ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        )
        for row in rows
    )
    return len(rows), hashlib.sha256(b"[" + b",".join(encoded) + b"]").hexdigest()


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
    _replace_project_resources(conn, rows, snapshot, None)


def replace_project_resources_years(
    conn: sqlite3.Connection,
    rows: Sequence[ApiProjectResourceRow],
    snapshot: ApiSnapshot,
    years: Sequence[int],
) -> None:
    """Replace selected report-year slices while retaining all other rows."""
    if snapshot.source_table != "project_resources":
        raise ValueError("snapshot source_table must be 'project_resources'")
    selected = tuple(dict.fromkeys(years))
    if not selected:
        raise ValueError("at least one report year is required")
    if snapshot.row_count < 0:
        raise ValueError("snapshot row_count must be non-negative")
    if any(row.report_year not in selected for row in rows):
        raise ValueError("incoming project-resource row is outside selected years")
    _replace_project_resources(conn, rows, snapshot, selected)


def _replace_project_resources(
    conn: sqlite3.Connection,
    rows: Sequence[ApiProjectResourceRow],
    snapshot: ApiSnapshot,
    years: Sequence[int] | None,
) -> None:
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
        _replace_project_resources_in_transaction(conn, rows, committed, years)
        conn.commit()
    except BaseException:
        conn.rollback()
        raise


def _replace_project_resources_in_transaction(
    conn: sqlite3.Connection,
    rows: Sequence[ApiProjectResourceRow],
    committed: ApiSnapshot,
    years: Sequence[int] | None,
    *,
    refresh_read_models: bool = True,
) -> ApiSnapshot:
    if years is None:
        conn.execute("DELETE FROM api_project_resources")
    else:
        placeholders = ", ".join("?" for _ in years)
        conn.execute(
            f"DELETE FROM api_project_resources WHERE report_year IN ({placeholders})",
            tuple(years),
        )
    conn.executemany(
        _ROW_INSERT_SQL,
        [tuple(getattr(row, column) for column in _ROW_COLUMNS) for row in rows],
    )
    row_count, source_sha256 = _persisted_hash(conn)
    conn.execute(
        """INSERT INTO api_snapshot
           (source_table, revision, row_count, source_sha256,
            fetched_at, publication_state, invalid_geometry_count)
           VALUES (?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(source_table) DO UPDATE SET
               revision = excluded.revision, row_count = excluded.row_count,
               source_sha256 = excluded.source_sha256,
               fetched_at = excluded.fetched_at,
               publication_state = excluded.publication_state,
               invalid_geometry_count = excluded.invalid_geometry_count""",
        (
            committed.source_table,
            committed.revision,
            row_count,
            source_sha256,
            committed.fetched_at,
            committed.publication_state,
            committed.invalid_geometry_count,
        ),
    )
    result = ApiSnapshot(
        committed.source_table,
        committed.revision,
        row_count,
        source_sha256,
        committed.fetched_at,
        committed.publication_state,
        committed.invalid_geometry_count,
    )
    if refresh_read_models:
        from esdc.pod_registry.read_models import refresh_pod_after_source_change

        refresh_pod_after_source_change(conn, actor="FA:project_resources_snapshot")
    return result


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


def iter_project_resources_rows(
    conn: sqlite3.Connection,
) -> Iterator[ApiProjectResourceRow]:
    """Yield canonical resource rows in their stored column order."""
    cursor = conn.execute(
        "SELECT " + ", ".join(_ROW_COLUMNS) + " FROM api_project_resources"
    )
    for values in cursor:
        yield ApiProjectResourceRow(**dict(zip(_ROW_COLUMNS, values, strict=True)))


def commit_project_resources_update(
    conn: sqlite3.Connection,
    rows: Sequence[ApiProjectResourceRow],
    *,
    years: Sequence[int] | None,
    expected_snapshot: ApiSnapshot | None,
) -> ApiSnapshot:
    """Commit a full or selected-year update against an optimistic baseline."""
    if years is not None:
        years = tuple(dict.fromkeys(years))
    revision = str(uuid.uuid4())
    try:
        conn.execute("BEGIN IMMEDIATE")
        validate_resource_update(
            conn, rows, years=years, expected_snapshot=expected_snapshot
        )
        committed = ApiSnapshot(
            "project_resources",
            revision,
            0,
            "",
            datetime.now(timezone.utc).isoformat(),
            "committed",
        )
        result = _replace_project_resources_in_transaction(conn, rows, committed, years)
        conn.commit()
        return result
    except BaseException:
        conn.rollback()
        raise


def validate_resource_update(
    conn: sqlite3.Connection,
    rows: Sequence[ApiProjectResourceRow],
    *,
    years: Sequence[int] | None,
    expected_snapshot: ApiSnapshot | None,
) -> None:
    """Validate a resource update inside the caller's active transaction."""
    if not conn.in_transaction:
        raise RuntimeError("resource validation requires an active transaction")
    if (
        expected_snapshot is not None
        and expected_snapshot.source_table != "project_resources"
    ):
        raise ValueError("expected snapshot source_table must be 'project_resources'")
    if years is not None:
        selected = tuple(dict.fromkeys(years))
        if not selected:
            raise ValueError("at least one report year is required")
        if any(row.report_year not in selected for row in rows):
            raise ValueError("incoming project-resource row is outside selected years")
        years = selected
    current = get_project_resources_snapshot(conn)
    if current != expected_snapshot:
        raise ValueError("project_resources snapshot changed during preparation")
    if (
        years is not None
        and current is None
        and conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' "
            "AND name = 'api_project_resources'"
        ).fetchone()
        is not None
        and conn.execute("SELECT 1 FROM api_project_resources LIMIT 1").fetchone()
    ):
        raise ValueError("project_resources rows exist without a snapshot")


def mark_project_resources_published(
    conn: sqlite3.Connection,
    revision: str,
    *,
    invalid_geometry_count: int | None = None,
) -> None:
    try:
        cursor = conn.execute(
            """UPDATE api_snapshot SET publication_state = 'published',
                   invalid_geometry_count = COALESCE(?, invalid_geometry_count)
               WHERE source_table = ? AND revision = ?""",
            (invalid_geometry_count, "project_resources", revision),
        )
        if cursor.rowcount != 1:
            raise ValueError(
                f"no project_resources snapshot matches revision {revision!r}"
            )
        conn.commit()
    except BaseException:
        conn.rollback()
        raise
