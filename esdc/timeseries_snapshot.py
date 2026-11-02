"""SQLite source snapshots for project timeseries rows."""

from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import Literal, cast

from esdc.api_snapshot import ApiSnapshot, JSONScalar

TimeseriesOrigin = Literal["api_json", "api_csv", "legacy_duckdb"]

# Ordered source schema copied from create_table_project_timeseries.sql. Types
# are kept beside the names because spooling must distinguish numeric fields
# from narrative text before discarding rows outside the requested year.
TIMESERIES_SOURCE_TYPES: dict[str, str] = {
    "report_date": "TEXT",
    "report_year": "INTEGER",
    "is_offshore": "INTEGER",
    "basin86_id": "TEXT",
    "basin128": "TEXT",
    "province": "TEXT",
    "operator_group": "TEXT",
    "report_status": "TEXT",
    "wk_id": "TEXT",
    "wk_name": "TEXT",
    "wk_area": "REAL",
    "wk_regionisasi_ngi": "TEXT",
    "wk_area_perwakilan_skkmigas": "TEXT",
    "psc_eff_start": "TEXT",
    "psc_eff_end": "TEXT",
    "wk_lat": "TEXT",
    "wk_long": "TEXT",
    "field_lat": "TEXT",
    "field_long": "TEXT",
    "wk_subgroup": "TEXT",
    "operator_name": "TEXT",
    "field_id": "TEXT",
    "field_name": "TEXT",
    "project_id": "TEXT",
    "project_name": "TEXT",
    "project_isactive": "INTEGER",
    "project_remarks": "TEXT",
    "vol_remarks": "TEXT",
    "frcast_remarks": "TEXT",
    "pod_letter_num": "TEXT",
    "pod_name": "TEXT",
    "onstream_year": "INTEGER",
    "onstream_actual": "TEXT",
    "project_class": "TEXT",
    "project_level": "TEXT",
    "prod_stage": "TEXT",
    "year": "INTEGER",
    "hist_year": "INTEGER",
    "cprd_grs_oil": "REAL",
    "cprd_grs_con": "REAL",
    "cprd_grs_ga": "REAL",
    "cprd_grs_gn": "REAL",
    "cprd_grs_oc": "REAL",
    "cprd_grs_an": "REAL",
    "cprd_sls_oil": "REAL",
    "cprd_sls_con": "REAL",
    "cprd_sls_ga": "REAL",
    "cprd_sls_gn": "REAL",
    "cprd_sls_oc": "REAL",
    "cprd_sls_an": "REAL",
    "rate_oil": "REAL",
    "rate_con": "REAL",
    "rate_ga": "REAL",
    "rate_gn": "REAL",
    "rate_oc": "REAL",
    "rate_an": "REAL",
    "tpf_oil": "REAL",
    "tpf_con": "REAL",
    "tpf_ga": "REAL",
    "tpf_gn": "REAL",
    "tpf_oc": "REAL",
    "tpf_an": "REAL",
    "tpf_risked_oil": "REAL",
    "tpf_risked_con": "REAL",
    "tpf_risked_ga": "REAL",
    "tpf_risked_gn": "REAL",
    "tpf_risked_oc": "REAL",
    "tpf_risked_an": "REAL",
    "slf_oil": "REAL",
    "slf_con": "REAL",
    "slf_ga": "REAL",
    "slf_gn": "REAL",
    "slf_oc": "REAL",
    "slf_an": "REAL",
    "spf_oil": "REAL",
    "spf_con": "REAL",
    "spf_ga": "REAL",
    "spf_gn": "REAL",
    "spf_oc": "REAL",
    "spf_an": "REAL",
    "crf_oil": "REAL",
    "crf_con": "REAL",
    "crf_ga": "REAL",
    "crf_gn": "REAL",
    "crf_oc": "REAL",
    "crf_an": "REAL",
    "prf_oil": "REAL",
    "prf_con": "REAL",
    "prf_ga": "REAL",
    "prf_gn": "REAL",
    "prf_oc": "REAL",
    "prf_an": "REAL",
    "ciof_oil": "REAL",
    "ciof_con": "REAL",
    "ciof_ga": "REAL",
    "ciof_gn": "REAL",
    "ciof_oc": "REAL",
    "ciof_an": "REAL",
    "lossf_oil": "REAL",
    "lossf_con": "REAL",
    "lossf_ga": "REAL",
    "lossf_gn": "REAL",
    "lossf_oc": "REAL",
    "lossf_an": "REAL",
    "ghv": "REAL",
}
TIMESERIES_SOURCE_COLUMNS = tuple(TIMESERIES_SOURCE_TYPES)
TIMESERIES_COLUMNS = frozenset(TIMESERIES_SOURCE_COLUMNS)
_YEAR_PATTERN = re.compile(r"[0-9]{4}\Z", re.ASCII)
_NONFINITE_TEXT = re.compile(r"[+-]?(?:nan|inf|infinity)\Z", re.IGNORECASE)


@dataclass(frozen=True)
class CanonicalTimeseriesRow:
    report_year: int | None
    row_json: str
    row_sha256: str
    origin: TimeseriesOrigin


def parse_report_year(value: JSONScalar) -> int:
    """Validate and normalize a report year from JSON or CSV."""
    if isinstance(value, bool):
        raise ValueError("report_year must be an integer or four-digit string")
    if isinstance(value, int):
        year = value
    elif isinstance(value, str):
        stripped = value.strip()
        if not _YEAR_PATTERN.fullmatch(stripped):
            raise ValueError("report_year must be exactly four ASCII digits")
        year = int(stripped)
    else:
        raise ValueError("report_year must be an integer or four-digit string")
    if not 1000 <= year <= 9999:
        raise ValueError("report_year must be between 1000 and 9999")
    return year


def _canonical_json(value: dict[str, JSONScalar]) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"project_timeseries contains a non-JSON scalar: {exc}"
        ) from exc


def normalize_timeseries_row(
    header: Sequence[str],
    values: Sequence[JSONScalar],
    *,
    origin: TimeseriesOrigin,
) -> CanonicalTimeseriesRow:
    """Normalize one positional source row and fill omitted fields with null."""
    if len(header) != len(values):
        raise ValueError("project_timeseries record has wrong width")
    if len(header) != len(set(header)):
        raise ValueError("project_timeseries contains duplicate columns")
    unknown = sorted(set(header) - TIMESERIES_COLUMNS)
    if unknown:
        raise ValueError(
            f"project_timeseries contains unknown columns: {', '.join(unknown)}"
        )
    supplied = dict(zip(header, values, strict=True))
    report_year_value = supplied.get("report_year")
    if report_year_value is None:
        if origin != "legacy_duckdb":
            raise ValueError(
                "project_timeseries column report_year is required for API input"
            )
        report_year: int | None = None
    else:
        try:
            report_year = parse_report_year(report_year_value)
        except ValueError as exc:
            raise ValueError(f"project_timeseries column report_year: {exc}") from exc
    supplied["report_year"] = report_year

    for column, value in supplied.items():
        if value is not None and not isinstance(value, (str, int, float, bool)):
            raise ValueError(f"project_timeseries column {column} is not a JSON scalar")
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"project_timeseries column {column} must be finite")
        if (
            TIMESERIES_SOURCE_TYPES[column] != "TEXT"
            and isinstance(value, str)
            and _NONFINITE_TEXT.fullmatch(value.strip())
        ):
            raise ValueError(f"project_timeseries column {column} must be finite")

    normalized = {column: supplied.get(column) for column in TIMESERIES_SOURCE_COLUMNS}
    row_json = _canonical_json(normalized)
    return CanonicalTimeseriesRow(
        report_year=report_year,
        row_json=row_json,
        row_sha256=hashlib.sha256(row_json.encode("utf-8")).hexdigest(),
        origin=origin,
    )


def ensure_timeseries_tables(conn: sqlite3.Connection) -> None:
    """Create additive timeseries snapshot schema inside the current transaction."""
    statements = (
        """CREATE TABLE IF NOT EXISTS api_project_timeseries (
            storage_id INTEGER PRIMARY KEY,
            report_year INTEGER,
            row_json TEXT NOT NULL,
            row_sha256 TEXT NOT NULL,
            origin TEXT NOT NULL CHECK (
                origin IN ('api_json','api_csv','legacy_duckdb')
            ),
            schema_version INTEGER NOT NULL CHECK (schema_version = 1)
        )""",
        "CREATE INDEX IF NOT EXISTS idx_api_project_timeseries_year "
        "ON api_project_timeseries(report_year)",
        "CREATE INDEX IF NOT EXISTS idx_api_project_timeseries_hash "
        "ON api_project_timeseries(row_sha256)",
        """CREATE TABLE IF NOT EXISTS api_snapshot_provenance (
            source_table TEXT NOT NULL,
            revision TEXT NOT NULL,
            origin TEXT NOT NULL,
            details_json TEXT NOT NULL,
            PRIMARY KEY(source_table, revision)
        )""",
    )
    for statement in statements:
        conn.execute(statement)


def _iter_content_hash(conn: sqlite3.Connection) -> tuple[int, str]:
    digest = hashlib.sha256()
    digest.update(b"[")
    count = 0
    cursor = conn.execute(
        "SELECT report_year, row_json, row_sha256, origin FROM api_project_timeseries "
        "ORDER BY row_json COLLATE BINARY"
    )
    while batch := cursor.fetchmany(1000):
        for report_year, row_json, row_sha256, origin in batch:
            _validate_canonical_row(
                CanonicalTimeseriesRow(report_year, row_json, row_sha256, origin)
            )
            if count:
                digest.update(b",")
            digest.update(row_json.encode("utf-8"))
            count += 1
    digest.update(b"]")
    return count, digest.hexdigest()


def timeseries_fingerprint(conn: sqlite3.Connection) -> tuple[int, str]:
    """Return count and duplicate-sensitive canonical content hash."""
    return _iter_content_hash(conn)


def get_timeseries_snapshot(conn: sqlite3.Connection) -> ApiSnapshot | None:
    row = conn.execute(
        """SELECT revision, row_count, source_sha256, fetched_at,
                  publication_state, invalid_geometry_count
           FROM api_snapshot WHERE source_table = 'project_timeseries'"""
    ).fetchone()
    if row is None:
        return None
    return ApiSnapshot("project_timeseries", *row)


def iter_timeseries_rows(
    conn: sqlite3.Connection, *, years: Sequence[int] | None = None
) -> Iterator[CanonicalTimeseriesRow]:
    """Yield canonical rows in deterministic order, including null years by default."""
    if years is None:
        cursor = conn.execute(
            "SELECT report_year, row_json, row_sha256, origin "
            "FROM api_project_timeseries "
            "ORDER BY row_json COLLATE BINARY, storage_id"
        )
    else:
        selected = tuple(dict.fromkeys(years))
        if not selected:
            return
        placeholders = ",".join("?" for _ in selected)
        cursor = conn.execute(
            "SELECT report_year, row_json, row_sha256, origin "
            "FROM api_project_timeseries "
            f"WHERE report_year IN ({placeholders}) "
            "ORDER BY row_json COLLATE BINARY, storage_id",
            selected,
        )
    while batch := cursor.fetchmany(1000):
        for report_year, row_json, row_sha256, origin in batch:
            item = CanonicalTimeseriesRow(
                report_year, row_json, row_sha256, cast(TimeseriesOrigin, origin)
            )
            _validate_canonical_row(item)
            yield item


def _validate_canonical_row(row: CanonicalTimeseriesRow) -> None:
    parsed = json.loads(row.row_json)
    if not isinstance(parsed, dict) or set(parsed) != TIMESERIES_COLUMNS:
        raise ValueError(
            "project_timeseries canonical row has an invalid source-column set"
        )
    if parsed.get("report_year") != row.report_year:
        raise ValueError(
            "project_timeseries indexed and JSON report_year values disagree"
        )
    if row.origin not in {"api_json", "api_csv", "legacy_duckdb"}:
        raise ValueError("project_timeseries canonical row has an invalid origin")
    normalized = normalize_timeseries_row(
        tuple(parsed), tuple(parsed.values()), origin=row.origin
    )
    if normalized.row_json != row.row_json:
        raise ValueError("project_timeseries canonical row is invalid or noncanonical")
    if row.row_sha256 != normalized.row_sha256:
        raise ValueError(
            "project_timeseries canonical row hash does not match row_json"
        )


def _record_provenance(
    conn: sqlite3.Connection, revision: str, origin: str, details: dict[str, object]
) -> None:
    conn.execute(
        """INSERT INTO api_snapshot_provenance
           (source_table, revision, origin, details_json) VALUES (?, ?, ?, ?)
           ON CONFLICT(source_table, revision) DO UPDATE SET
               origin=excluded.origin, details_json=excluded.details_json""",
        (
            "project_timeseries",
            revision,
            origin,
            json.dumps(
                details, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ),
        ),
    )


def record_timeseries_provenance(
    conn: sqlite3.Connection,
    *,
    revision: str,
    origin: str,
    details: dict[str, object],
) -> None:
    """Record operation lineage such as a bootstrap file hash and type map."""
    if not conn.in_transaction:
        raise RuntimeError(
            "timeseries provenance requires an active caller transaction"
        )
    ensure_timeseries_tables(conn)
    _record_provenance(conn, revision, origin, details)


def record_observed_empty_timeseries(
    conn: sqlite3.Connection,
    *,
    revision: str,
    fetched_at: str,
    origin: Literal["api_json", "api_csv"],
    refresh_read_models: bool = True,
) -> ApiSnapshot:
    """Record an explicit upstream empty observation and its empty fingerprint."""
    if not conn.in_transaction:
        raise RuntimeError(
            "timeseries observation requires an active caller transaction"
        )
    ensure_timeseries_tables(conn)
    row_count, content_hash = timeseries_fingerprint(conn)
    snapshot = ApiSnapshot(
        "project_timeseries",
        revision,
        row_count,
        content_hash,
        fetched_at,
        "committed",
        0,
    )
    conn.execute(
        """INSERT INTO api_snapshot
           (source_table, revision, row_count, source_sha256, fetched_at,
            publication_state, invalid_geometry_count)
           VALUES (?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(source_table) DO UPDATE SET revision=excluded.revision,
             row_count=excluded.row_count, source_sha256=excluded.source_sha256,
           fetched_at=excluded.fetched_at,
           publication_state=excluded.publication_state,
             invalid_geometry_count=excluded.invalid_geometry_count""",
        (
            snapshot.source_table,
            snapshot.revision,
            snapshot.row_count,
            snapshot.source_sha256,
            snapshot.fetched_at,
            snapshot.publication_state,
            snapshot.invalid_geometry_count,
        ),
    )
    record_timeseries_provenance(
        conn,
        revision=revision,
        origin=origin,
        details={"observed_empty": True, "input_origins": [origin]},
    )
    if refresh_read_models:
        from esdc.pod_registry.read_models import refresh_pod_after_source_change

        refresh_pod_after_source_change(conn, actor="FA:project_timeseries_snapshot")
    return snapshot


def replace_timeseries_in_transaction(
    conn: sqlite3.Connection,
    rows: Iterable[CanonicalTimeseriesRow],
    *,
    years: Sequence[int] | None,
    revision: str,
    fetched_at: str,
    refresh_read_models: bool = True,
) -> ApiSnapshot:
    """Replace all or selected year slices without taking transaction ownership."""
    if not conn.in_transaction:
        raise RuntimeError(
            "timeseries replacement requires an active caller transaction"
        )
    ensure_timeseries_tables(conn)
    selected_years = None if years is None else tuple(dict.fromkeys(years))
    if selected_years == ():
        raise ValueError("selected timeseries replacement requires at least one year")
    if selected_years is None:
        conn.execute("DELETE FROM api_project_timeseries")
    else:
        placeholders = ",".join("?" for _ in selected_years)
        conn.execute(
            f"DELETE FROM api_project_timeseries WHERE report_year IN ({placeholders})",
            selected_years,
        )

    insert_sql = """INSERT INTO api_project_timeseries
        (report_year, row_json, row_sha256, origin, schema_version)
        VALUES (?, ?, ?, ?, 1)"""
    batch: list[tuple[int | None, str, str, str]] = []
    origins: set[str] = set()
    for row in rows:
        _validate_canonical_row(row)
        if selected_years is not None and row.report_year not in selected_years:
            raise ValueError(
                "project_timeseries replacement row is outside selected years"
            )
        batch.append((row.report_year, row.row_json, row.row_sha256, row.origin))
        origins.add(row.origin)
        if len(batch) == 1000:
            conn.executemany(insert_sql, batch)
            batch.clear()
    if batch:
        conn.executemany(insert_sql, batch)
    if not origins:
        raise ValueError(
            "empty timeseries operations require explicit observed-empty provenance"
        )

    row_count, content_hash = timeseries_fingerprint(conn)
    snapshot = ApiSnapshot(
        "project_timeseries",
        revision,
        row_count,
        content_hash,
        fetched_at,
        "committed",
        0,
    )
    conn.execute(
        """INSERT INTO api_snapshot
           (source_table, revision, row_count, source_sha256, fetched_at,
            publication_state, invalid_geometry_count)
           VALUES (?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(source_table) DO UPDATE SET revision=excluded.revision,
             row_count=excluded.row_count, source_sha256=excluded.source_sha256,
           fetched_at=excluded.fetched_at,
           publication_state=excluded.publication_state,
             invalid_geometry_count=excluded.invalid_geometry_count""",
        (
            snapshot.source_table,
            snapshot.revision,
            snapshot.row_count,
            snapshot.source_sha256,
            snapshot.fetched_at,
            snapshot.publication_state,
            snapshot.invalid_geometry_count,
        ),
    )
    _record_provenance(
        conn,
        revision,
        "+".join(sorted(origins)),
        {
            "input_origins": sorted(origins),
            "observed_empty": False,
            "years": selected_years,
        },
    )
    if refresh_read_models:
        from esdc.pod_registry.read_models import refresh_pod_after_source_change

        refresh_pod_after_source_change(conn, actor="FA:project_timeseries_snapshot")
    return snapshot
