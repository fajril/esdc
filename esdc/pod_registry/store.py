# esdc/pod_registry/store.py
"""SQLite store for esdc operational data (source of truth).

Currently holds the POD master registry; intended to grow into the
app-wide operational database (auth, entity relationships, ...), hence
the generic esdc.sqlite filename.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from esdc.configs import Config

ESDC_SQLITE_FILENAME = "esdc.sqlite"

_DDL = """
CREATE TABLE IF NOT EXISTS r_institution (
    code        INTEGER PRIMARY KEY,
    institution TEXT NOT NULL UNIQUE,
    description TEXT
);
CREATE TABLE IF NOT EXISTS r_pod_type (
    code        INTEGER PRIMARY KEY,
    pod_type    TEXT NOT NULL UNIQUE,
    description TEXT
);
CREATE TABLE IF NOT EXISTS m_pod (
    id               INTEGER PRIMARY KEY,
    pod_id           TEXT NOT NULL UNIQUE,
    pod_name         TEXT NOT NULL,
    pod_letter_num   TEXT,
    approval_date    TEXT NOT NULL,
    institution_code INTEGER NOT NULL REFERENCES r_institution(code),
    pod_type_code    INTEGER NOT NULL REFERENCES r_pod_type(code),
    rev_num          INTEGER NOT NULL DEFAULT 0,
    approval_seq     INTEGER NOT NULL UNIQUE
);
CREATE TABLE IF NOT EXISTS project_pod (
    pod_id     INTEGER NOT NULL REFERENCES m_pod(id),
    project_id TEXT NOT NULL,
    UNIQUE (pod_id, project_id)
);
CREATE TABLE IF NOT EXISTS pod_document (
    pod_id INTEGER NOT NULL REFERENCES m_pod(id),
    doc_id TEXT NOT NULL,
    UNIQUE (pod_id, doc_id)
);
CREATE TABLE IF NOT EXISTS pod_revision (
    successor_id   TEXT NOT NULL REFERENCES m_pod(pod_id),
    predecessor_id TEXT NOT NULL REFERENCES m_pod(pod_id),
    revision_effect TEXT NOT NULL DEFAULT 'unknown'
        CHECK (revision_effect IN ('unknown', 'partial_amendment', 'full_replacement')),
    effective_date TEXT,
    amended_scope TEXT,
    previous_remains_valid INTEGER
        CHECK (previous_remains_valid IN (0, 1)),
    UNIQUE (successor_id, predecessor_id),
    CHECK (successor_id <> predecessor_id)
);
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
);
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
);
"""

_POD_REVISION_TEMPORAL_COLUMNS = (
    (
        "revision_effect",
        "TEXT NOT NULL DEFAULT 'unknown'"
        " CHECK (revision_effect IN ('unknown', 'partial_amendment',"
        " 'full_replacement'))",
    ),
    ("effective_date", "TEXT"),
    ("amended_scope", "TEXT"),
    ("previous_remains_valid", "INTEGER CHECK (previous_remains_valid IN (0, 1))"),
)


def get_esdc_sqlite_path() -> Path:
    """Location of the operational database inside the esdc db directory."""
    return Config.get_db_dir() / ESDC_SQLITE_FILENAME


def ensure_tables(conn: sqlite3.Connection) -> None:
    conn.executescript(_DDL)
    # Migrate pre-temporal pod_revision tables (two-column) in place: add only
    # absent columns. Existing rows keep revision_effect='unknown'; no
    # supersession is inferred. ADD COLUMN with NOT NULL DEFAULT backfills rows.
    existing = {row[1] for row in conn.execute("PRAGMA table_info(pod_revision)")}
    for name, definition in _POD_REVISION_TEMPORAL_COLUMNS:
        if name not in existing:
            conn.execute(f"ALTER TABLE pod_revision ADD COLUMN {name} {definition}")
    snapshot_columns = {
        row[1] for row in conn.execute("PRAGMA table_info(api_snapshot)")
    }
    if "invalid_geometry_count" not in snapshot_columns:
        conn.execute(
            "ALTER TABLE api_snapshot ADD COLUMN invalid_geometry_count "
            "INTEGER NOT NULL DEFAULT 0"
        )


def validate_revision_effect(
    effect: str,
    previous_remains_valid: bool | None,
) -> None:
    """Validate a revision_effect against its previous_remains_valid flag."""
    if effect == "partial_amendment":
        if previous_remains_valid is not True:
            raise ValueError("partial_amendment requires previous_remains_valid=True")
    elif effect == "full_replacement":
        if previous_remains_valid is not False:
            raise ValueError("full_replacement requires previous_remains_valid=False")
    elif effect == "unknown":
        if previous_remains_valid is not None:
            raise ValueError("unknown permits previous_remains_valid=None only")
    else:
        raise ValueError(f"unknown revision_effect: {effect!r}")


def get_sqlite_connection(path: Path | None = None) -> sqlite3.Connection:
    """Open (creating if needed) the operational db with FK enforcement on."""
    db_path = path or get_esdc_sqlite_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    ensure_tables(conn)
    return conn


def allocate_pod_id(
    conn: sqlite3.Connection,
    approval_date: str,
    institution_code: int,
    pod_type_code: int,
    rev_num: int,
) -> tuple[str, int]:
    """Compute the next pod_id. Caller must insert within the same transaction.

    pod_id is immutable once issued and approval_seq never gets reused, so the
    next sequence is always max(approval_seq) + 1 (gaps from deletes are fine).
    """
    year = approval_date[:4]
    row = conn.execute("SELECT COALESCE(MAX(approval_seq), 0) FROM m_pod").fetchone()
    seq = int(row[0]) + 1
    pod_id = f"PL-{year}-{seq:04d}-{institution_code}-{pod_type_code}-{rev_num}"
    return pod_id, seq
