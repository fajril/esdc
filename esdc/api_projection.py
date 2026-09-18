"""Publish committed API project resources into the DuckDB analytical mirror."""

from __future__ import annotations

import contextlib
import re
import uuid
from dataclasses import dataclass
from pathlib import Path

import duckdb

from esdc.api_snapshot import get_project_resources_snapshot
from esdc.db_security import _load_sql_script
from esdc.dbmanager import get_duckdb_connection
from esdc.pod_registry.store import get_sqlite_connection


@dataclass(frozen=True)
class ProjectPublication:
    revision: str
    row_count: int
    source_sha256: str
    invalid_geometry_count: int


_SOURCE_COLUMNS = (
    "project_id, report_year, project_name, report_date, field_geom, groovy_report, "
    "field_id, field_name, wk_id, wk_name, operator_name, province, project_stage, "
    "project_class, project_level, uncert_level, project_remarks, is_discovered, "
    "is_ltp, is_pse_approved, is_pod_approved, is_report_accepted, is_unitization, "
    "project_eol, project_isactive, groovy_isactive, fusion_isactive, onstream_year, "
    "onstream_actual, rec_oil, rec_con, rec_ga, rec_gn, rec_oc, rec_an, rec_mboe, "
    "wk_lat, wk_long, wk_area, operator_group, basin86_id, basin86, basin128_id, "
    "basin128, prov_id, field_lat, field_long, field_area, pod_letter_num, pod_name, "
    "pod_date, project_name_previous, project_level_previous, prod_stage, wk_subgroup, "
    "psc_eff_start, psc_eff_end, is_offshore, report_status, wk_area_ori, "
    "wk_area_latest, wk_area_perwakilan_skkmigas, wk_regionisasi_ngi, "
    "field_name_previous, "
    "vol_remarks, gcf_srock, gcf_res, gcf_ts, gcf_dyn, gcf_total, pd, rec_oil_risked, "
    "rec_con_risked, rec_ga_risked, rec_gn_risked, rec_oc_risked, rec_an_risked, "
    "rec_mboe_risked, res_oil, res_con, res_ga, res_gn, res_oc, res_an, eur_rec_oil, "
    "eur_rec_con, eur_rec_ga, eur_rec_gn, eur_rec_oc, eur_rec_an, eur_res_oil, "
    "eur_res_con, eur_res_ga, eur_res_gn, eur_res_oc, eur_res_an, rf_oil, rf_gas, "
    "urf_rec_oil, urf_rec_gas, urf_res_oil, urf_res_gas, prj_ioip, prj_igip, "
    "rate_grs_oil, rate_grs_con, rate_grs_ga, rate_grs_gn, rate_grs_oc, rate_grs_an, "
    "rate_sls_oil, rate_sls_con, rate_sls_ga, rate_sls_gn, rate_sls_oc, rate_sls_an, "
    "cprd_grs_oil, cprd_grs_con, cprd_grs_ga, cprd_grs_gn, cprd_grs_oc, cprd_grs_an, "
    "cprd_sls_oil, cprd_sls_con, cprd_sls_ga, cprd_sls_gn, cprd_sls_oc, cprd_sls_an, "
    "dcpy_um_oil, dcpy_um_con, dcpy_um_ga, dcpy_um_gn, dcpy_um_oc, dcpy_um_an, "
    "dcpy_ppa_oil, dcpy_ppa_con, dcpy_ppa_ga, dcpy_ppa_gn, dcpy_ppa_oc, dcpy_ppa_an, "
    "dcpy_wi_oil, dcpy_wi_con, dcpy_wi_ga, dcpy_wi_gn, dcpy_wi_oc, dcpy_wi_an, "
    "dcpy_cio_oil, dcpy_cio_con, dcpy_cio_ga, dcpy_cio_gn, dcpy_cio_oc, dcpy_cio_an, "
    "dcpy_uc_oil, dcpy_uc_con, dcpy_uc_ga, dcpy_uc_gn, dcpy_uc_oc, dcpy_uc_an, "
    "dcpy_gtr_oil, dcpy_gtr_con, dcpy_gtr_ga, dcpy_gtr_gn, dcpy_gtr_oc, dcpy_gtr_an, "
    "ghv_avg"
)


def _execute_script(
    conn: duckdb.DuckDBPyConnection,
    filename: str,
    replacements: dict[str, str] | None = None,
) -> None:
    source = _load_sql_script(filename)
    for placeholder, value in (replacements or {}).items():
        source = source.replace(placeholder, value)
    for statement in source.split(";"):
        statement = statement.strip()
        if statement:
            conn.execute(statement)


def _staging_schema(conn: duckdb.DuckDBPyConnection, table: str) -> None:
    source = _load_sql_script("create_table_project_resources.sql")
    source = re.sub(r"^DROP TABLE IF EXISTS project_resources;\s*", "", source)
    source = re.sub(r"^DROP SEQUENCE IF EXISTS seq_project_resources;\s*", "", source)
    source = source.replace("seq_project_resources", f"seq_{table}")
    source = source.replace("project_resources", table)
    for statement in source.split(";"):
        statement = statement.strip()
        if statement:
            conn.execute(statement)


def _mark_stale(sqlite_path: Path, revision: str) -> None:
    conn = get_sqlite_connection(sqlite_path)
    try:
        conn.execute(
            "UPDATE api_snapshot SET publication_state = 'analytics_stale' "
            "WHERE source_table = ? AND revision = ?",
            ("project_resources", revision),
        )
        conn.commit()
    finally:
        conn.close()


def _validate_geometry(conn: duckdb.DuckDBPyConnection, table: str) -> int:
    rows = conn.execute(
        f"SELECT project_id, field_geom FROM {table} "
        "WHERE field_geom IS NOT NULL AND length(trim(field_geom)) > 0"
    ).fetchall()
    invalid_count = 0
    for project_id, wkt in rows:
        valid = conn.execute(
            "SELECT ST_IsValid(TRY(ST_GeomFromText(?)))", [wkt]
        ).fetchone()
        if valid is None or not valid[0]:
            invalid_count += 1
            continue
        conn.execute(
            f"UPDATE {table} SET field_geometry = ST_GeomFromText(?) "
            "WHERE project_id = ? AND field_geom = ?",
            [wkt, project_id, wkt],
        )
    return invalid_count


def _rebuild_project_resources(conn: duckdb.DuckDBPyConnection) -> None:
    for name in ("idx_project_resources_report_year", "idx_project_resources_agg"):
        with contextlib.suppress(duckdb.Error):
            conn.execute(f"DROP INDEX {name}")
    _execute_script(conn, "create_esdc_view.sql")
    conn.execute("INSTALL fts")
    conn.execute("LOAD fts")
    conn.execute(
        "PRAGMA create_fts_index('project_resources', 'uuid', 'project_name', "
        "'field_name', 'wk_name', 'province', 'basin128', 'operator_name', "
        "'project_remarks', 'vol_remarks', 'groovy_report', lower=1, "
        "strip_accents=1, overwrite=1)"
    )


def _checkpoint(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute("CHECKPOINT")
    conn.execute(
        "CREATE INDEX idx_project_resources_report_year "
        "ON project_resources(report_year DESC)"
    )
    conn.execute(
        "CREATE INDEX idx_project_resources_agg ON project_resources("
        "report_year, wk_id, field_id, project_class, project_level)"
    )


def publish_project_resources(
    duckdb_path: Path, sqlite_path: Path, revision: str
) -> ProjectPublication:
    """Atomically publish one committed SQLite revision into DuckDB."""
    sqlite_conn = get_sqlite_connection(sqlite_path)
    try:
        snapshot = get_project_resources_snapshot(sqlite_conn)
    finally:
        sqlite_conn.close()
    if snapshot is None or snapshot.revision != revision:
        raise ValueError(f"no committed project_resources revision {revision!r}")
    if snapshot.publication_state not in ("committed", "analytics_stale"):
        raise ValueError(f"revision {revision!r} is not publishable")

    conn = get_duckdb_connection(duckdb_path, read_only=False)
    stage = f"project_resources_stage_{uuid.uuid4().hex}"
    committed = False
    try:
        conn.execute("INSTALL spatial")
        conn.execute("LOAD spatial")
        conn.execute("BEGIN")
        _staging_schema(conn, stage)
        conn.execute(
            f"INSERT INTO {stage} ({_SOURCE_COLUMNS}) "
            f"SELECT {_SOURCE_COLUMNS} FROM sqlite_scan(?, 'api_project_resources')",
            [str(sqlite_path)],
        )
        invalid_geometry_count = _validate_geometry(conn, stage)
        _execute_script(conn, "create_column_uuid.sql", {"{table_name}": stage})
        _execute_script(
            conn, "create_project_resources_uuid.sql", {"project_resources": stage}
        )
        _execute_script(
            conn,
            "create_project_resources_is_discovered.sql",
            {"project_resources": stage},
        )
        _execute_script(
            conn,
            "create_project_resources_project_stage.sql",
            {"project_resources": stage},
        )
        conn.execute("DROP TABLE IF EXISTS project_resources")
        conn.execute(f"ALTER TABLE {stage} RENAME TO project_resources")
        _rebuild_project_resources(conn)
        conn.execute("COMMIT")
        committed = True
        try:
            _checkpoint(conn)
        except Exception as exc:
            _mark_stale(sqlite_path, revision)
            raise RuntimeError(
                "analytical state requires rebuild after post-commit failure"
            ) from exc
    except Exception:
        if not committed:
            with contextlib.suppress(duckdb.Error):
                conn.execute("ROLLBACK")
            _mark_stale(sqlite_path, revision)
        raise
    finally:
        conn.close()

    sqlite_conn = get_sqlite_connection(sqlite_path)
    try:
        sqlite_conn.execute(
            "UPDATE api_snapshot SET publication_state = 'published', "
            "invalid_geometry_count = ? "
            "WHERE source_table = ? AND revision = ?",
            (invalid_geometry_count, "project_resources", revision),
        )
        sqlite_conn.commit()
    finally:
        sqlite_conn.close()
    return ProjectPublication(
        revision, snapshot.row_count, snapshot.source_sha256, invalid_geometry_count
    )
