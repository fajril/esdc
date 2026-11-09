# esdc/pod_registry/store.py
"""SQLite store for esdc operational data (source of truth).

Currently holds the POD master registry; intended to grow into the
app-wide operational database (auth, entity relationships, ...), hence
the generic esdc.sqlite filename.

The store serves one current schema: the canonical POD registry SQL plus the
generic raw API source tables. The superseded POD tables (numeric ``m_pod`` id,
``project_pod``, ``pod_revision``, ``pod_value_case``) are no longer created or
repaired here; replacing an old-shaped dataset is a dedicated validated
operation, not a runtime fallback.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from esdc.configs import Config
from esdc.pod_registry.migrations import (
    ECONOMICS_UPGRADE_RECOVERY,
    MigrationRefusedError,
    MigrationReport,
    _atomic_writer,
    _ensure_registry_writer,
    initialize_registry_v2,
    inspect_migration,
)
from esdc.sqlite_runtime import configure_writer, require_wal_runtime

ESDC_SQLITE_FILENAME = "esdc.sqlite"

# Canonical raw API source tables. These are current generic capabilities
# (source corpus storage, geometry audit, provenance), not the retired POD
# schema, so they stay explicit here instead of being parsed out of an old DDL.
_PROJECT_RESOURCES_DDL = """
CREATE TABLE IF NOT EXISTS api_project_resources (
    project_id TEXT NOT NULL CHECK (length(trim(project_id)) > 0),
    report_year INTEGER,
    project_name TEXT,
    report_date TEXT,
    field_geom TEXT,
    groovy_report TEXT,
    field_id TEXT,
    field_name TEXT,
    wk_id TEXT,
    wk_name TEXT,
    operator_name TEXT,
    province TEXT,
    project_stage TEXT,
    project_class TEXT,
    project_level TEXT,
    uncert_level TEXT,
    project_remarks TEXT,
    is_discovered INTEGER,
    is_ltp INTEGER,
    is_pse_approved INTEGER,
    is_pod_approved INTEGER,
    is_report_accepted INTEGER,
    is_unitization INTEGER,
    project_eol INTEGER,
    project_isactive INTEGER,
    groovy_isactive INTEGER,
    fusion_isactive INTEGER,
    onstream_year INTEGER,
    onstream_actual TEXT,
    rec_oil REAL,
    rec_con REAL,
    rec_ga REAL,
    rec_gn REAL,
    rec_oc REAL,
    rec_an REAL,
    rec_mboe REAL,
    wk_lat REAL,
    wk_long REAL,
    wk_area REAL,
    operator_group TEXT,
    basin86_id TEXT,
    basin86 TEXT,
    basin128_id TEXT,
    basin128 TEXT,
    prov_id TEXT,
    field_lat REAL,
    field_long REAL,
    field_area REAL,
    pod_letter_num TEXT,
    pod_name TEXT,
    pod_date TEXT,
    project_name_previous TEXT,
    project_level_previous TEXT,
    prod_stage TEXT,
    wk_subgroup TEXT,
    psc_eff_start TEXT,
    psc_eff_end TEXT,
    is_offshore INTEGER,
    report_status TEXT,
    wk_area_ori TEXT,
    wk_area_latest REAL,
    wk_area_perwakilan_skkmigas TEXT,
    wk_regionisasi_ngi TEXT,
    field_name_previous TEXT,
    vol_remarks TEXT,
    gcf_srock REAL,
    gcf_res REAL,
    gcf_ts REAL,
    gcf_dyn REAL,
    gcf_total REAL,
    pd INTEGER,
    rec_oil_risked REAL,
    rec_con_risked REAL,
    rec_ga_risked REAL,
    rec_gn_risked REAL,
    rec_oc_risked REAL,
    rec_an_risked REAL,
    rec_mboe_risked REAL,
    res_oil REAL,
    res_con REAL,
    res_ga REAL,
    res_gn REAL,
    res_oc REAL,
    res_an REAL,
    eur_rec_oil REAL,
    eur_rec_con REAL,
    eur_rec_ga REAL,
    eur_rec_gn REAL,
    eur_rec_oc REAL,
    eur_rec_an REAL,
    eur_res_oil REAL,
    eur_res_con REAL,
    eur_res_ga REAL,
    eur_res_gn REAL,
    eur_res_oc REAL,
    eur_res_an REAL,
    rf_oil REAL,
    rf_gas REAL,
    urf_rec_oil REAL,
    urf_rec_gas REAL,
    urf_res_oil REAL,
    urf_res_gas REAL,
    prj_ioip REAL,
    prj_igip REAL,
    rate_grs_oil REAL,
    rate_grs_con REAL,
    rate_grs_ga REAL,
    rate_grs_gn REAL,
    rate_grs_oc REAL,
    rate_grs_an REAL,
    rate_sls_oil REAL,
    rate_sls_con REAL,
    rate_sls_ga REAL,
    rate_sls_gn REAL,
    rate_sls_oc REAL,
    rate_sls_an REAL,
    cprd_grs_oil REAL,
    cprd_grs_con REAL,
    cprd_grs_ga REAL,
    cprd_grs_gn REAL,
    cprd_grs_oc REAL,
    cprd_grs_an REAL,
    cprd_sls_oil REAL,
    cprd_sls_con REAL,
    cprd_sls_ga REAL,
    cprd_sls_gn REAL,
    cprd_sls_oc REAL,
    cprd_sls_an REAL,
    dcpy_um_oil REAL,
    dcpy_um_con REAL,
    dcpy_um_ga REAL,
    dcpy_um_gn REAL,
    dcpy_um_oc REAL,
    dcpy_um_an REAL,
    dcpy_ppa_oil REAL,
    dcpy_ppa_con REAL,
    dcpy_ppa_ga REAL,
    dcpy_ppa_gn REAL,
    dcpy_ppa_oc REAL,
    dcpy_ppa_an REAL,
    dcpy_wi_oil REAL,
    dcpy_wi_con REAL,
    dcpy_wi_ga REAL,
    dcpy_wi_gn REAL,
    dcpy_wi_oc REAL,
    dcpy_wi_an REAL,
    dcpy_cio_oil REAL,
    dcpy_cio_con REAL,
    dcpy_cio_ga REAL,
    dcpy_cio_gn REAL,
    dcpy_cio_oc REAL,
    dcpy_cio_an REAL,
    dcpy_uc_oil REAL,
    dcpy_uc_con REAL,
    dcpy_uc_ga REAL,
    dcpy_uc_gn REAL,
    dcpy_uc_oc REAL,
    dcpy_uc_an REAL,
    dcpy_gtr_oil REAL,
    dcpy_gtr_con REAL,
    dcpy_gtr_ga REAL,
    dcpy_gtr_gn REAL,
    dcpy_gtr_oc REAL,
    dcpy_gtr_an REAL,
    ghv_avg REAL,
    PRIMARY KEY (project_id, report_year, uncert_level)
)
"""

_SNAPSHOT_DDL = """
CREATE TABLE IF NOT EXISTS api_snapshot (
    source_table TEXT PRIMARY KEY,
    revision TEXT NOT NULL,
    row_count INTEGER NOT NULL,
    source_sha256 TEXT NOT NULL,
    fetched_at TEXT NOT NULL,
    publication_state TEXT NOT NULL CHECK (
        publication_state IN ('committed', 'published', 'analytics_stale')
    ),
    invalid_geometry_count INTEGER NOT NULL DEFAULT 0
        CHECK (invalid_geometry_count >= 0)
)
"""


def get_esdc_sqlite_path() -> Path:
    """Location of the operational database inside the esdc db directory."""
    return Config.get_db_dir() / ESDC_SQLITE_FILENAME


def ensure_project_resources_table(conn: sqlite3.Connection) -> None:
    """Ensure only the canonical API resource input table exists.

    This additive helper is also safe for an explicitly initialized current POD
    registry, and it never commits: it runs inside the caller's transaction.
    """
    conn.execute(_PROJECT_RESOURCES_DDL.strip())

    from esdc.api_snapshot import ApiProjectResourceRow

    expected = tuple(ApiProjectResourceRow.__dataclass_fields__)
    actual = tuple(
        str(row[1]) for row in conn.execute("PRAGMA table_info(api_project_resources)")
    )
    if actual != expected:
        raise RuntimeError(
            "api_project_resources has unsupported columns or column order"
        )


def ensure_snapshot_table(conn: sqlite3.Connection) -> None:
    """Ensure the shared raw snapshot marker, inside the caller's transaction."""
    conn.execute(_SNAPSHOT_DDL.strip())

    # Generic raw metadata repair only: an older raw snapshot gains the geometry
    # audit column, which is a current capability rather than old POD migration.
    columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(api_snapshot)")}
    if "invalid_geometry_count" not in columns:
        conn.execute(
            "ALTER TABLE api_snapshot ADD COLUMN invalid_geometry_count "
            "INTEGER NOT NULL DEFAULT 0"
        )


def ensure_raw_source_tables(conn: sqlite3.Connection) -> None:
    """Create the shared raw API source tables statement-wise, never committing."""
    ensure_project_resources_table(conn)
    ensure_snapshot_table(conn)

    from esdc.timeseries_snapshot import ensure_timeseries_tables

    ensure_timeseries_tables(conn)


def ensure_tables(conn: sqlite3.Connection) -> None:
    """Create or validate the single current schema as one atomic unit.

    Read-only classification runs before any writer transition, so a source with a
    retired or incompatible shape is refused while its journal mode, schema, data,
    bytes and mtime stay exactly as the caller left them. Every phase runs
    statement-wise inside a transaction owned here, or inside a savepoint when the
    caller already owns the transaction, so an injected failure leaves no partial
    registry, raw-source, or timeseries objects and the caller's preexisting rows and
    transaction ownership stay correct. The initialization commits exactly once when it
    owns the transaction.
    """
    _refuse_incompatible_shape(inspect_migration(conn))
    _ensure_registry_writer(conn)

    with _atomic_writer(conn):
        initialize_registry_v2(conn)
        ensure_raw_source_tables(conn)


def get_sqlite_readonly_connection(
    path: Path | None = None,
) -> sqlite3.Connection | None:
    """Open an existing operational database read-only, or return ``None``."""
    db_path = path or get_esdc_sqlite_path()
    if not db_path.exists():
        return None
    if not db_path.is_file():
        raise ValueError(f"operational database path is not a regular file: {db_path}")
    uri = db_path.resolve().as_uri() + "?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def is_missing_table_error(error: sqlite3.OperationalError) -> bool:
    """Return whether SQLite reports an absent table rather than another failure."""
    return str(error).startswith("no such table:")


def _refuse_incompatible_shape(report: MigrationReport) -> None:
    """Refuse a source that cannot serve the single current schema, without writing."""
    if report.status not in {"legacy", "blocked"}:
        return
    detail = "; ".join(report.blockers) or ", ".join(report.legacy_tables)
    if report.economics_upgrade_required:
        raise MigrationRefusedError(
            "The operational SQLite database needs the explicit POD economics "
            f"upgrade. {ECONOMICS_UPGRADE_RECOVERY} This store never renames "
            "columns automatically."
        )
    message = (
        "the operational store serves only the current POD schema; this "
        f"database needs the explicit validated replacement ({detail})"
    )
    if report.status == "legacy":
        # The crosswalk route is part of this policy: an application-only or raw-only
        # target stays additive, while a legacy or partial POD footprint is never
        # reconciled by inference.
        message = (
            "The operational SQLite database has a legacy POD registry. "
            + message
            + " Replace it once using 'esdc load --from-sqlite SOURCE' or "
            "'esdc load --from-excel SOURCE --schema-pod'. Protected document links "
            "require a reviewed --pod-crosswalk JSON mapping; numeric POD IDs "
            "are never inferred."
        )
    raise MigrationRefusedError(message)


def get_sqlite_connection(path: Path | None = None) -> sqlite3.Connection:
    """Open (creating if needed) the operational db with writer policy enabled."""
    db_path = path or get_esdc_sqlite_path()
    require_wal_runtime()
    probe = get_sqlite_readonly_connection(db_path)
    if probe is not None:
        try:
            _refuse_incompatible_shape(inspect_migration(probe))
        finally:
            probe.close()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        configure_writer(conn)
        conn.row_factory = sqlite3.Row
        ensure_tables(conn)
    except BaseException:
        conn.close()
        raise
    return conn


def get_sqlite_writer_connection(path: Path | None = None) -> sqlite3.Connection:
    """Open the operational db through the single latest-only writer policy."""
    return get_sqlite_connection(path)
