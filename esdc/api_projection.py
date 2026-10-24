"""Publish committed API project resources into the DuckDB analytical mirror."""

from __future__ import annotations

import contextlib
import itertools
import logging
import re
import uuid
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

import duckdb

from esdc.api_snapshot import (
    ApiProjectResourceRow,
    ApiSnapshot,
    _persisted_hash,
    get_project_resources_snapshot,
    iter_project_resources_rows,
    mark_project_resources_published,
)
from esdc.db_security import _load_sql_script
from esdc.dbmanager import get_duckdb_connection
from esdc.pod_registry.store import get_sqlite_connection


@dataclass(frozen=True)
class ProjectPublication:
    revision: str
    row_count: int
    source_sha256: str
    invalid_geometry_count: int


@dataclass(frozen=True)
class ResourcePublicationRecord:
    source_table: str
    revision: str
    source_row_count: int
    source_sha256: str
    invalid_geometry_count: int
    published_at: datetime


def get_resource_publication(
    conn: duckdb.DuckDBPyConnection,
) -> ResourcePublicationRecord | None:
    """Read the resource marker without creating lineage schema."""
    return get_api_publication(conn, "project_resources")


def get_api_publication(
    conn: duckdb.DuckDBPyConnection,
    source_table: Literal["project_resources", "project_timeseries"],
) -> ResourcePublicationRecord | None:
    """Read one API source marker without creating lineage schema."""
    exists = conn.execute(
        "SELECT 1 FROM information_schema.tables WHERE table_name = 'api_publication'"
    ).fetchone()
    if exists is None:
        return None
    row = conn.execute(
        "SELECT source_table, revision, source_row_count, source_sha256, "
        "invalid_geometry_count, CAST(published_at AS VARCHAR) FROM api_publication "
        "WHERE source_table = ?",
        [source_table],
    ).fetchone()
    if row is None:
        return None
    return ResourcePublicationRecord(
        source_table=row[0],
        revision=row[1],
        source_row_count=row[2],
        source_sha256=row[3],
        invalid_geometry_count=row[4],
        published_at=datetime.fromisoformat(row[5].replace(" ", "T", 1)),
    )


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


def _rebuild_project_resources(
    conn: duckdb.DuckDBPyConnection, *, allow_install: bool = True
) -> None:
    for name in ("idx_project_resources_report_year", "idx_project_resources_agg"):
        with contextlib.suppress(duckdb.Error):
            conn.execute(f"DROP INDEX {name}")
    _execute_script(conn, "create_esdc_view.sql")
    conn.execute("LOAD fts")
    conn.execute(
        "PRAGMA create_fts_index('project_resources', 'uuid', 'project_name', "
        "'field_name', 'wk_name', 'province', 'basin128', 'operator_name', "
        "'project_remarks', 'vol_remarks', 'groovy_report', lower=1, "
        "strip_accents=1, overwrite=1)"
    )


def _checkpoint(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute("CHECKPOINT")


def _invalidate_analytical_caches() -> None:
    """Invalidate all analytical result caches after DuckDB changes."""
    failures: list[BaseException] = []
    from esdc.chat.tools import invalidate_tool_cache
    from esdc.dbmanager import invalidate_sql_cache

    for invalidate in (invalidate_sql_cache, invalidate_tool_cache):
        try:
            invalidate()
        except BaseException as exc:
            failures.append(exc)
    if failures:
        interruption = next(
            (
                error
                for error in failures
                if isinstance(error, (KeyboardInterrupt, SystemExit))
            ),
            None,
        )
        if interruption is not None:
            interruption.args = (
                *interruption.args,
                "analytical cache invalidation also reported: "
                + "; ".join(
                    str(error) for error in failures if error is not interruption
                ),
            )
            raise interruption
        raise RuntimeError(
            "one or more analytical cache invalidations failed: "
            + "; ".join(str(error) for error in failures)
        ) from failures[0]


@dataclass(frozen=True)
class ResourceStage:
    table_name: str
    row_count: int
    invalid_geometry_count: int


def _stage_project_resources(
    conn: duckdb.DuckDBPyConnection,
    rows: Iterable[ApiProjectResourceRow],
) -> ResourceStage:
    """Create a typed resource stage from a stream of committed source rows."""
    table = f"project_resources_stage_{uuid.uuid4().hex}"
    _staging_schema(conn, table)
    columns = tuple(part.strip() for part in _SOURCE_COLUMNS.split(","))
    insert_sql = (
        f"INSERT INTO {table} ({_SOURCE_COLUMNS}) VALUES "
        f"({', '.join('?' for _ in columns)})"
    )
    row_count = 0
    iterator = iter(rows)
    while batch := list(itertools.islice(iterator, 1000)):
        values = [tuple(getattr(row, column) for column in columns) for row in batch]
        conn.executemany(insert_sql, values)
        row_count += len(values)
    invalid_count = _validate_geometry(conn, table)
    _execute_script(conn, "create_column_uuid.sql", {"{table_name}": table})
    _execute_script(
        conn, "create_project_resources_uuid.sql", {"project_resources": table}
    )
    _execute_script(
        conn, "create_project_resources_is_discovered.sql", {"project_resources": table}
    )
    _execute_script(
        conn, "create_project_resources_project_stage.sql", {"project_resources": table}
    )
    return ResourceStage(table, row_count, invalid_count)


def _swap_project_resources(
    conn: duckdb.DuckDBPyConnection,
    stage: ResourceStage,
    *,
    allow_install: bool = True,
) -> None:
    """Replace the resource projection inside the caller's transaction."""
    conn.execute("DROP TABLE IF EXISTS project_resources")
    conn.execute(f"ALTER TABLE {stage.table_name} RENAME TO project_resources")
    if allow_install:
        _rebuild_project_resources(conn)
    else:
        _rebuild_project_resources(conn, allow_install=False)
    conn.execute(
        "CREATE INDEX idx_project_resources_report_year "
        "ON project_resources(report_year DESC)"
    )
    conn.execute(
        "CREATE INDEX idx_project_resources_agg ON project_resources("
        "report_year, wk_id, field_id, project_class, project_level)"
    )


def _record_resource_publication(
    conn: duckdb.DuckDBPyConnection,
    snapshot: ApiSnapshot,
    *,
    invalid_geometry_count: int,
) -> None:
    _record_api_publication(
        conn, snapshot, invalid_geometry_count=invalid_geometry_count
    )


def _record_api_publication(
    conn: duckdb.DuckDBPyConnection,
    snapshot: ApiSnapshot,
    *,
    invalid_geometry_count: int = 0,
) -> None:
    if snapshot.source_table not in {"project_resources", "project_timeseries"}:
        raise ValueError(f"unsupported API publication source: {snapshot.source_table}")
    conn.execute(
        "CREATE TABLE IF NOT EXISTS api_publication ("
        "source_table VARCHAR PRIMARY KEY, revision VARCHAR NOT NULL, "
        "source_row_count BIGINT NOT NULL, source_sha256 VARCHAR NOT NULL, "
        "invalid_geometry_count BIGINT NOT NULL, "
        "published_at TIMESTAMPTZ NOT NULL)"
    )
    conn.execute(
        "INSERT OR REPLACE INTO api_publication "
        "(source_table, revision, source_row_count, source_sha256, "
        "invalid_geometry_count, published_at) "
        "VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)",
        (
            snapshot.source_table,
            snapshot.revision,
            snapshot.row_count,
            snapshot.source_sha256,
            invalid_geometry_count,
        ),
    )


def _finish_resource_publication(
    conn: duckdb.DuckDBPyConnection,
    sqlite_path: Path,
    snapshot: ApiSnapshot,
    *,
    invalid_geometry_count: int,
) -> ProjectPublication:
    """Run post-commit cache/checkpoint work without marking SQLite published."""
    failures: list[Exception] = []
    for finish in (_invalidate_analytical_caches, lambda: _checkpoint(conn)):
        try:
            finish()
        except Exception as exc:
            failures.append(exc)
    if failures:
        try:
            _mark_stale(sqlite_path, snapshot.revision)
        except Exception as state_error:
            failures.append(state_error)
        raise RuntimeError(
            "analytical state requires rebuild after cache invalidation failure or "
            "post-commit maintenance failure: "
            + "; ".join(str(error) for error in failures)
        ) from failures[0]
    return ProjectPublication(
        snapshot.revision,
        snapshot.row_count,
        snapshot.source_sha256,
        invalid_geometry_count,
    )


def publish_project_resources(
    duckdb_path: Path,
    sqlite_path: Path,
    revision: str,
    *,
    allow_install: bool = True,
) -> ProjectPublication:
    """Atomically publish one committed SQLite revision into DuckDB."""
    sqlite_conn = get_sqlite_connection(sqlite_path)
    try:
        snapshot = get_project_resources_snapshot(sqlite_conn)
        actual_count, actual_hash = _persisted_hash(sqlite_conn)
    finally:
        sqlite_conn.close()
    if snapshot is None or snapshot.revision != revision:
        raise ValueError(f"no committed project_resources revision {revision!r}")
    if snapshot.publication_state not in ("committed", "analytics_stale", "published"):
        raise ValueError(f"revision {revision!r} is not publishable")
    if (
        actual_count <= 0
        or actual_count != snapshot.row_count
        or actual_hash != snapshot.source_sha256
    ):
        raise ValueError(
            "project_resources snapshot is empty or its count/hash does not match"
        )

    conn: duckdb.DuckDBPyConnection | None = None
    committed = False
    try:
        if not allow_install:
            from esdc.analytics_rebuild import _preflight_extensions

            _preflight_extensions()
        if allow_install:
            conn = get_duckdb_connection(duckdb_path, read_only=False)
        else:
            conn = get_duckdb_connection(
                duckdb_path, read_only=False, allow_install=False
            )
        if allow_install:
            conn.execute("INSTALL fts")
        conn.execute("LOAD fts")
        if allow_install:
            conn.execute("INSTALL spatial")
        conn.execute("LOAD spatial")
        conn.execute("BEGIN")
        sqlite_conn = get_sqlite_connection(sqlite_path)
        try:
            sqlite_conn.execute("BEGIN")
            current = get_project_resources_snapshot(sqlite_conn)
            if current != snapshot:
                raise ValueError(
                    "project_resources snapshot changed during publication"
                )
            stage = _stage_project_resources(
                conn, iter_project_resources_rows(sqlite_conn)
            )
        finally:
            sqlite_conn.close()
        if stage.row_count != snapshot.row_count:
            raise ValueError("staged project_resources count does not match snapshot")
        invalid_geometry_count = stage.invalid_geometry_count
        _swap_project_resources(conn, stage, allow_install=allow_install)
        _record_resource_publication(
            conn, snapshot, invalid_geometry_count=invalid_geometry_count
        )
        check_conn = get_sqlite_connection(sqlite_path)
        try:
            latest_snapshot = get_project_resources_snapshot(check_conn)
            latest_count, latest_hash = _persisted_hash(check_conn)
        finally:
            check_conn.close()
        if (
            latest_snapshot != snapshot
            or latest_count != snapshot.row_count
            or latest_hash != snapshot.source_sha256
        ):
            raise ValueError(
                "project_resources revision was superseded before publication"
            )
        conn.execute("COMMIT")
        committed = True
        result = _finish_resource_publication(
            conn,
            sqlite_path,
            snapshot,
            invalid_geometry_count=invalid_geometry_count,
        )
    except BaseException as exc:
        if not committed:
            with contextlib.suppress(duckdb.Error):
                if conn is not None:
                    conn.execute("ROLLBACK")
        try:
            _mark_stale(sqlite_path, revision)
        except BaseException as state_error:
            logging.error(
                "Unable to record stale resource revision %s after %r: %r",
                revision,
                exc,
                state_error,
            )
            exc.args = (
                *exc.args,
                f"Additionally, recording analytics_stale failed: {state_error}",
            )
        raise
    finally:
        if conn is not None:
            conn.close()

    try:
        sqlite_conn = get_sqlite_connection(sqlite_path)
        try:
            mark_project_resources_published(
                sqlite_conn, revision, invalid_geometry_count=invalid_geometry_count
            )
        finally:
            sqlite_conn.close()
    except BaseException as exc:
        try:
            _mark_stale(sqlite_path, revision)
        except BaseException as state_error:
            logging.error(
                "Unable to record stale resource revision %s after final mark "
                "error %r: %r",
                revision,
                exc,
                state_error,
            )
            exc.args = (
                *exc.args,
                f"Additionally, recording analytics_stale failed: {state_error}",
            )
        raise
    return result
