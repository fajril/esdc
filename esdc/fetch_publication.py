"""Fetch preflight and SQLite-first joint analytical publication."""

from __future__ import annotations

import json
import logging
import sqlite3
import tempfile
import uuid
from collections import Counter
from collections.abc import Iterator, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
from urllib.parse import quote

import duckdb

from esdc.api_projection import (
    ResourcePublicationRecord,
    _checkpoint,
    _record_api_publication,
    _stage_project_resources,
    _swap_project_resources,
    get_api_publication,
    get_resource_publication,
)
from esdc.api_snapshot import (
    ApiProjectResourceRow,
    ApiSnapshot,
    JSONScalar,
    _persisted_hash,
    _replace_project_resources_in_transaction,
    get_project_resources_snapshot,
    iter_project_resources_rows,
    mark_project_resources_published,
    validate_resource_update,
)
from esdc.dbmanager import (
    _reindex_fts_for_path,
    get_duckdb_connection,
)
from esdc.fetching import (
    FetchFailure,
    FetchOutcome,
    FetchPhase,
    PreparedFetch,
    parse_report_year,
)
from esdc.pod_registry.store import get_sqlite_connection
from esdc.timeseries_projection import (
    TimeseriesStage,
    _safe_type,
    apply_timeseries_stage,
    stage_canonical_timeseries,
)
from esdc.timeseries_projection import (
    _legacy_type_overrides as _verified_legacy_type_overrides,
)
from esdc.timeseries_snapshot import (
    TIMESERIES_SOURCE_COLUMNS,
    CanonicalTimeseriesRow,
    ensure_timeseries_tables,
    get_timeseries_snapshot,
    iter_timeseries_rows,
    normalize_timeseries_row,
    record_observed_empty_timeseries,
    record_timeseries_provenance,
    replace_timeseries_in_transaction,
    timeseries_fingerprint,
)

_SELECTIVE_REFUSAL = (
    "Cannot update selected resource years: the canonical SQLite snapshot is missing, "
    "stale, incomplete, or not verified against DuckDB. Existing data was retained."
)
ResourceIdentity = tuple[str | None, int | None, str | None]


def _readonly_sqlite(path: Path) -> sqlite3.Connection | None:
    if not path.is_file():
        return None
    uri = f"file:{quote(str(path.absolute()))}?mode=ro"
    try:
        return sqlite3.connect(uri, uri=True)
    except sqlite3.Error as exc:
        raise ValueError(
            "Cannot inspect the existing canonical SQLite database"
        ) from exc


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    return (
        conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (table,)
        ).fetchone()
        is not None
    )


def _canonical_identity(row: ApiProjectResourceRow) -> ResourceIdentity:
    year = _normalize_identity_year(row.report_year)
    uncertainty = None if row.uncert_level is None else str(row.uncert_level)
    return row.project_id, year, uncertainty


def _normalize_identity_year(value: object) -> int | None:
    if value is None:
        return None
    if not isinstance(value, (int, str)) or isinstance(value, bool):
        raise ValueError("invalid report year in resource identity")
    try:
        return parse_report_year(value)
    except ValueError as exc:
        raise ValueError("invalid report year in resource identity") from exc


def _duckdb_identity(row: tuple[object, object, object]) -> ResourceIdentity:
    project_id, report_year, uncert_level = row
    year = _normalize_identity_year(report_year)
    return (
        None if project_id is None else str(project_id),
        year,
        None if uncert_level is None else str(uncert_level),
    )


def _read_canonical(
    path: Path,
) -> tuple[ApiSnapshot | None, list[ApiProjectResourceRow], int, str] | None:
    conn = _readonly_sqlite(path)
    if conn is None:
        return None
    try:
        conn.execute("BEGIN")
        if not _table_exists(conn, "api_project_resources"):
            return None
        snapshot = (
            get_project_resources_snapshot(conn)
            if _table_exists(conn, "api_snapshot")
            else None
        )
        rows = list(iter_project_resources_rows(conn))
        count, digest = _persisted_hash(conn)
        return snapshot, rows, count, digest
    except sqlite3.Error as exc:
        raise ValueError(
            "Cannot inspect the existing canonical SQLite database"
        ) from exc
    finally:
        conn.close()


def _read_mirror(
    path: Path,
    *,
    allow_install: bool = True,
) -> (
    tuple[
        int,
        Counter[tuple[str | None, int | None, str | None]],
        ResourcePublicationRecord | None,
    ]
    | None
):
    if not path.is_file():
        return None
    try:
        if allow_install:
            conn = duckdb.connect(str(path), read_only=True)
        else:
            conn = duckdb.connect(
                str(path),
                read_only=True,
                config={"autoinstall_known_extensions": "false"},
            )
    except duckdb.Error as exc:
        raise ValueError("Cannot inspect the existing DuckDB database") from exc
    try:
        has_resources = conn.execute(
            "SELECT 1 FROM information_schema.tables "
            "WHERE table_name = 'project_resources'"
        ).fetchone()
        if has_resources is None:
            return 0, Counter(), None
        identities = Counter(
            _duckdb_identity(row)
            for row in conn.execute(
                "SELECT project_id, report_year, uncert_level FROM project_resources"
            ).fetchall()
        )
        return sum(identities.values()), identities, get_resource_publication(conn)
    except duckdb.Error as exc:
        raise ValueError("Cannot inspect the existing DuckDB resource schema") from exc
    finally:
        conn.close()


def ensure_selective_resource_baseline(
    duckdb_path: Path, sqlite_path: Path, *, allow_install: bool = True
) -> ApiSnapshot | None:
    """Verify canonical and analytical resource state without opening writers."""
    try:
        canonical = _read_canonical(sqlite_path)
    except ValueError as exc:
        raise ValueError(
            f"{_SELECTIVE_REFUSAL} Run an unfiltered 'esdc fetch' before using --year."
        ) from exc
    try:
        mirror = _read_mirror(duckdb_path, allow_install=allow_install)
    except ValueError as exc:
        guidance = (
            "Run 'esdc rebuild-project-resources' before using --year."
            if canonical is not None and canonical[2] > 0
            else "Run an unfiltered 'esdc fetch' before using --year."
        )
        raise ValueError(f"{_SELECTIVE_REFUSAL} {guidance}") from exc
    canonical_snapshot, canonical_rows, canonical_count, canonical_hash = (
        canonical if canonical is not None else (None, [], 0, "")
    )
    mirror_count, mirror_identities, lineage = (
        mirror if mirror is not None else (0, Counter(), None)
    )

    if canonical_count == 0 and mirror_count == 0:
        return canonical_snapshot
    if canonical_count == 0 or canonical_snapshot is None:
        guidance = "Run an unfiltered 'esdc fetch' before using --year."
        raise ValueError(f"{_SELECTIVE_REFUSAL} {guidance}")
    guidance = "Run 'esdc rebuild-project-resources' before using --year."
    if (
        canonical_snapshot.publication_state != "published"
        or canonical_snapshot.row_count != canonical_count
        or canonical_snapshot.source_sha256 != canonical_hash
        or mirror is None
        or mirror_count == 0
        or lineage is None
        or lineage.revision != canonical_snapshot.revision
        or lineage.source_row_count != canonical_count
        or lineage.source_sha256 != canonical_hash
    ):
        raise ValueError(f"{_SELECTIVE_REFUSAL} {guidance}")
    canonical_identities = Counter(_canonical_identity(row) for row in canonical_rows)
    if canonical_identities != mirror_identities:
        raise ValueError(f"{_SELECTIVE_REFUSAL} {guidance}")
    return canonical_snapshot


def ensure_timeseries_baseline(
    duckdb_path: Path,
    sqlite_path: Path,
    *,
    allow_bootstrap: bool = False,
    allow_install: bool = True,
) -> ApiSnapshot | None:
    """Verify the complete canonical timeseries source against its DuckDB marker."""
    sqlite_conn = _readonly_sqlite(sqlite_path)
    target: duckdb.DuckDBPyConnection | None = None
    comparison: duckdb.DuckDBPyConnection | None = None
    source_provenance: tuple[str, str] | None = None
    provenance_details: dict[str, object] | None = None

    def open_target(path: Path, *, read_only: bool) -> duckdb.DuckDBPyConnection:
        if allow_install:
            return duckdb.connect(str(path), read_only=read_only)
        return duckdb.connect(
            str(path),
            read_only=read_only,
            config={"autoinstall_known_extensions": "false"},
        )

    try:
        if sqlite_conn is None:
            if not duckdb_path.is_file():
                return None
            target = open_target(duckdb_path, read_only=True)
            lineage = get_api_publication(target, "project_timeseries")
            if not _target_timeseries_exists(target) and lineage is None:
                return None
            count = _target_timeseries_count(target)
            if count != 0 or lineage is not None:
                raise ValueError("canonical project_timeseries baseline is missing")
            _validate_target_timeseries_schema(target)
            return None
        sqlite_conn.execute("BEGIN")
        if not _table_exists(sqlite_conn, "api_project_timeseries"):
            if (
                _table_exists(sqlite_conn, "api_snapshot")
                and get_timeseries_snapshot(sqlite_conn) is not None
            ):
                raise ValueError(
                    "canonical project_timeseries metadata exists without "
                    "its source table"
                )
            if not duckdb_path.is_file():
                return None
            target = open_target(duckdb_path, read_only=True)
            lineage = get_api_publication(target, "project_timeseries")
            if not _target_timeseries_exists(target) and lineage is None:
                return None
            if _target_timeseries_count(target) == 0 and lineage is None:
                _validate_target_timeseries_schema(target)
                return None
            raise ValueError("canonical project_timeseries baseline is missing")
        snapshot = (
            get_timeseries_snapshot(sqlite_conn)
            if _table_exists(sqlite_conn, "api_snapshot")
            else None
        )
        count, digest = timeseries_fingerprint(sqlite_conn)
        if count == 0 and snapshot is None:
            if not duckdb_path.is_file():
                return None
            target = open_target(duckdb_path, read_only=True)
            lineage = get_api_publication(target, "project_timeseries")
            if not _target_timeseries_exists(target) and lineage is None:
                return None
            if _target_timeseries_count(target) == 0 and lineage is None:
                _validate_target_timeseries_schema(target)
                return None
            raise ValueError("canonical project_timeseries metadata is missing")
        if (
            snapshot is None
            or snapshot.row_count != count
            or snapshot.source_sha256 != digest
        ):
            raise ValueError(
                "canonical project_timeseries metadata is missing or stale"
            )
        if _table_exists(sqlite_conn, "api_snapshot_provenance"):
            source_provenance = sqlite_conn.execute(
                "SELECT origin, details_json FROM api_snapshot_provenance "
                "WHERE source_table='project_timeseries' AND revision=?",
                (snapshot.revision,),
            ).fetchone()
        if source_provenance is None:
            raise ValueError("canonical project_timeseries provenance is missing")
        if source_provenance[0] not in {"api_json", "api_csv", "legacy_duckdb"}:
            raise ValueError(
                "canonical project_timeseries provenance origin is invalid"
            )
        try:
            loaded_details = json.loads(source_provenance[1])
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError(
                "canonical project_timeseries provenance is malformed"
            ) from exc
        if not isinstance(loaded_details, dict):
            raise ValueError("canonical project_timeseries provenance is malformed")
        provenance_details = loaded_details
        if snapshot.publication_state == "analytics_stale":
            raise ValueError(
                "project_timeseries baseline is not published "
                "(canonical analytics_stale); "
                "run 'esdc rebuild-api-analytics' before another preserving fetch"
            )
        has_legacy_rows = (
            _table_exists(sqlite_conn, "api_project_timeseries")
            and sqlite_conn.execute(
                "SELECT 1 FROM api_project_timeseries "
                "WHERE origin='legacy_duckdb' LIMIT 1"
            ).fetchone()
            is not None
        )
        verified_legacy_types = (
            _verified_legacy_type_overrides(sqlite_conn) if has_legacy_rows else {}
        )
        if not duckdb_path.is_file():
            raise ValueError("analytical project_timeseries baseline is missing")
        target = open_target(duckdb_path, read_only=True)
        target_count = _target_timeseries_count(target)
        if target_count != count:
            raise ValueError(
                "canonical and analytical project_timeseries counts differ"
            )
        if count == 0:
            if source_provenance is None:
                raise ValueError("empty project_timeseries lacks current provenance")
            if source_provenance[0] not in {"api_json", "api_csv"} or (
                provenance_details.get("observed_empty") is not True
            ):
                raise ValueError("empty project_timeseries lacks observed-empty proof")
            _validate_target_timeseries_schema(target)
        else:
            types = {
                column: _safe_type(kind)
                for column, kind in target.execute(
                    "SELECT column_name, data_type FROM information_schema.columns "
                    "WHERE table_schema='main' AND table_name='project_timeseries'"
                ).fetchall()
            }
            if set(TIMESERIES_SOURCE_COLUMNS) - set(types) or any(
                types[column] is None for column in TIMESERIES_SOURCE_COLUMNS
            ):
                raise ValueError("analytical project_timeseries schema is incomplete")
            if allow_install:
                comparison = duckdb.connect(":memory:")
            else:
                comparison = duckdb.connect(
                    ":memory:", config={"autoinstall_known_extensions": "false"}
                )
            definitions = ", ".join(
                f"{column} {types[column]}" for column in TIMESERIES_SOURCE_COLUMNS
            )
            comparison.execute(f"CREATE TABLE canonical_values ({definitions})")
            insert = (
                "INSERT INTO canonical_values VALUES ("
                + ",".join("?" for _ in TIMESERIES_SOURCE_COLUMNS)
                + ")"
            )
            batch: list[tuple[object, ...]] = []
            for row in iter_timeseries_rows(sqlite_conn):
                values = json.loads(row.row_json)
                batch.append(
                    tuple(values[column] for column in TIMESERIES_SOURCE_COLUMNS)
                )
                if len(batch) == 1000:
                    comparison.executemany(insert, batch)
                    batch.clear()
            if batch:
                comparison.executemany(insert, batch)
            columns = ", ".join(TIMESERIES_SOURCE_COLUMNS)
            source_cursor = comparison.execute(
                f"SELECT {columns} FROM canonical_values ORDER BY {columns}"
            )
            target_cursor = target.execute(
                f"SELECT {columns} FROM project_timeseries ORDER BY {columns}"
            )
            while True:
                source_batch = source_cursor.fetchmany(1000)
                target_batch = target_cursor.fetchmany(1000)
                if source_batch != target_batch:
                    raise ValueError(
                        "canonical and analytical project_timeseries values differ"
                    )
                if not source_batch:
                    break
        lineage = get_api_publication(target, "project_timeseries")
        bootstrap_provenance = (
            source_provenance[1] if source_provenance[0] == "legacy_duckdb" else None
        )
        bootstrap_exception = False
        if snapshot.publication_state != "published":
            if (
                allow_bootstrap
                and snapshot.publication_state == "committed"
                and bootstrap_provenance is not None
            ):
                try:
                    bootstrap_details = json.loads(bootstrap_provenance)
                except (TypeError, json.JSONDecodeError):
                    bootstrap_details = None
                has_complete_map = (
                    isinstance(bootstrap_details, dict)
                    and isinstance(bootstrap_details.get("source_types"), dict)
                    and set(bootstrap_details["source_types"])
                    == set(TIMESERIES_SOURCE_COLUMNS)
                )
                if has_complete_map:
                    bootstrap_exception = bool(verified_legacy_types)
            if not bootstrap_exception:
                raise ValueError(
                    "canonical project_timeseries baseline is not published"
                )
        if lineage is None:
            if not bootstrap_exception:
                raise ValueError("analytical project_timeseries lineage is missing")
        elif (
            lineage.revision != snapshot.revision
            or lineage.source_row_count != count
            or lineage.source_sha256 != digest
        ):
            raise ValueError("analytical project_timeseries lineage is stale")
        if count == 0 and lineage is None:
            raise ValueError("empty analytical project_timeseries lineage is missing")
        return snapshot
    except (sqlite3.Error, duckdb.Error, KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, ValueError) and str(exc).startswith(
            "canonical and analytical"
        ):
            raise
        raise ValueError(
            f"project_timeseries baseline verification failed: {exc}"
        ) from exc
    finally:
        if comparison is not None:
            comparison.close()
        if target is not None:
            target.close()
        if sqlite_conn is not None:
            sqlite_conn.close()


def _target_timeseries_count(conn: duckdb.DuckDBPyConnection) -> int:
    if not _target_timeseries_exists(conn):
        return 0
    row = conn.execute("SELECT COUNT(*) FROM project_timeseries").fetchone()
    return 0 if row is None else int(row[0])


def _target_timeseries_exists(conn: duckdb.DuckDBPyConnection) -> bool:
    exists = conn.execute(
        "SELECT 1 FROM information_schema.tables WHERE table_schema='main' "
        "AND table_name='project_timeseries'"
    ).fetchone()
    return exists is not None


def _validate_target_timeseries_schema(
    conn: duckdb.DuckDBPyConnection,
) -> None:
    types = {
        column: _safe_type(kind)
        for column, kind in conn.execute(
            "SELECT column_name, data_type FROM information_schema.columns "
            "WHERE table_schema='main' AND table_name='project_timeseries'"
        ).fetchall()
    }
    if set(TIMESERIES_SOURCE_COLUMNS) - set(types) or any(
        types[column] is None for column in TIMESERIES_SOURCE_COLUMNS
    ):
        raise ValueError("analytical project_timeseries schema is incomplete")


def _record_stale_after_failure(
    sqlite_path: Path,
    revision: str,
    source_table: str = "project_resources",
) -> str | None:
    try:
        conn = get_sqlite_connection(sqlite_path)
        try:
            conn.execute(
                "UPDATE api_snapshot SET publication_state='analytics_stale' "
                "WHERE source_table=? AND revision=?",
                (source_table, revision),
            )
            conn.commit()
        finally:
            conn.close()
    except BaseException as exc:
        logging.exception("Unable to record analytics_stale for revision %s", revision)
        return f"; recording analytics_stale also failed: {exc}"
    return None


def _failure(
    message: str,
    *,
    phase: FetchPhase,
    sqlite_committed: bool,
    duckdb_committed: bool = False,
    resource_revision: str | None = None,
    timeseries_revision: str | None = None,
    table: str | None = None,
    report_year: int | None = None,
) -> FetchFailure:
    recovery = "esdc rebuild-project-resources" if resource_revision else None
    suffix = ""
    if (resource_revision or timeseries_revision) and not duckdb_committed:
        suffix = (
            " Canonical SQLite data was committed; DuckDB retains its prior analytical "
            "state. Pair recovery is required before another preserving fetch."
        )
        if resource_revision:
            suffix += " Run 'esdc rebuild-project-resources' to recover resources."
        if timeseries_revision:
            suffix += " Use the analytics pair-recovery command for timeseries."
    elif (resource_revision or timeseries_revision) and duckdb_committed:
        suffix = (
            " New DuckDB data is committed and remains readable; canonical publication "
            "state needs recovery."
        )
        if resource_revision:
            suffix += " Run 'esdc rebuild-project-resources' to rebuild resources."
        if timeseries_revision:
            suffix += " Use the analytics pair-recovery command for timeseries."
    elif duckdb_committed:
        suffix = " New DuckDB data is committed and remains readable."
    return FetchFailure(
        message + suffix,
        phase=phase,
        table=table,
        report_year=report_year,
        sqlite_committed=sqlite_committed,
        duckdb_committed=duckdb_committed,
        recovery_command=recovery,
    )


def _prepared_canonical_rows(
    prepared: PreparedFetch,
) -> Iterator[CanonicalTimeseriesRow]:
    from esdc.fetching import iter_spooled_rows, validate_timeseries_header

    for item in prepared.timeseries:
        header = validate_timeseries_header(item.header)
        if item.row_count <= 0:
            raise ValueError("timeseries preparation contains an empty slice")
        year_index = header.index("report_year")
        count = 0
        for values in iter_spooled_rows(item):
            if values[year_index] != item.report_year:
                raise ValueError(
                    "project_timeseries row does not match report year "
                    f"{item.report_year}"
                )
            yield normalize_timeseries_row(header, values, origin=item.origin)
            count += 1
        if count != item.row_count:
            raise ValueError(
                f"project_timeseries slice {item.report_year} contains {count} rows; "
                f"expected {item.row_count}"
            )


def _validate_prepared_timeseries(prepared: PreparedFetch) -> None:
    seen: set[int] = set()
    for item in prepared.timeseries:
        if item.report_year in seen:
            raise ValueError(f"duplicate prepared timeseries year {item.report_year}")
        seen.add(item.report_year)
    for _ in _prepared_canonical_rows(prepared):
        pass


def _read_timeseries_snapshot(path: Path) -> ApiSnapshot | None:
    conn = _readonly_sqlite(path)
    if conn is None:
        return None
    try:
        conn.execute("BEGIN")
        if not _table_exists(conn, "api_snapshot"):
            return None
        return get_timeseries_snapshot(conn)
    finally:
        conn.close()


def _timeseries_source_state(
    conn: sqlite3.Connection,
) -> tuple[ApiSnapshot | None, int, str | None]:
    snapshot = (
        get_timeseries_snapshot(conn) if _table_exists(conn, "api_snapshot") else None
    )
    if not _table_exists(conn, "api_project_timeseries"):
        return snapshot, 0, None
    count, digest = timeseries_fingerprint(conn)
    return snapshot, count, digest


def _legacy_type_overrides(
    conn: sqlite3.Connection,
) -> dict[str, str]:
    return _verified_legacy_type_overrides(conn)


def _mark_source_published(
    sqlite_path: Path, source_table: str, snapshot: ApiSnapshot
) -> None:
    if source_table == "project_resources":
        conn = get_sqlite_connection(sqlite_path)
        try:
            mark_project_resources_published(
                conn,
                snapshot.revision,
                invalid_geometry_count=snapshot.invalid_geometry_count,
            )
        finally:
            conn.close()
        return
    conn = get_sqlite_connection(sqlite_path)
    try:
        conn.execute("BEGIN IMMEDIATE")
        cursor = conn.execute(
            "UPDATE api_snapshot SET publication_state='published' "
            "WHERE source_table=? AND revision=?",
            (source_table, snapshot.revision),
        )
        if cursor.rowcount != 1:
            raise ValueError(f"{source_table} revision changed before publish marking")
        conn.commit()
    except BaseException:
        conn.rollback()
        raise
    finally:
        conn.close()


def publish_prepared_fetch(
    prepared: PreparedFetch,
    *,
    duckdb_path: Path,
    sqlite_path: Path,
    reindex: bool,
) -> FetchOutcome:
    """Commit canonical datasets together, then publish their captured pair."""
    if not prepared.resource_rows and not prepared.timeseries:
        return FetchOutcome(
            None,
            None,
            0,
            None,
            (),
            0,
            (),
            prepared.skipped_resource_years,
            prepared.skipped_timeseries_years,
        )
    try:
        _validate_prepared_timeseries(prepared)
    except Exception as exc:
        raise FetchFailure(
            f"project_timeseries input validation failed: {exc}",
            phase="validation",
            table="project_timeseries",
        ) from exc

    resources_changed = bool(prepared.resource_rows)
    expected_resource = None
    if resources_changed:
        try:
            if prepared.resource_years is None:
                existing = _read_canonical(sqlite_path)
                expected_resource = None if existing is None else existing[0]
            else:
                expected_resource = ensure_selective_resource_baseline(
                    duckdb_path, sqlite_path
                )
        except Exception as exc:
            raise FetchFailure(
                str(exc), phase="preflight", table="project_resources"
            ) from exc

    expected_timeseries = None
    complete_empty_observation = (
        prepared.timeseries_complete_observation
        and not prepared.timeseries
        and resources_changed
    )
    replace_all = (
        bool(prepared.timeseries) and prepared.timeseries_mode == "replace_all"
    )
    try:
        can_bootstrap = (
            prepared.timeseries_complete_observation and prepared.resource_years is None
        )
        if not replace_all:
            expected_timeseries = ensure_timeseries_baseline(
                duckdb_path, sqlite_path, allow_bootstrap=can_bootstrap
            )
        elif prepared.timeseries:
            expected_timeseries = _read_timeseries_snapshot(sqlite_path)
    except Exception as exc:
        raise FetchFailure(
            str(exc), phase="preflight", table="project_timeseries"
        ) from exc

    timeseries_changed = bool(prepared.timeseries) or complete_empty_observation
    if complete_empty_observation and expected_timeseries is not None:
        # A verified empty snapshot is still a stable source revision. Repeated
        # empty observations must not manufacture a new revision or lineage.
        timeseries_changed = False
    if not resources_changed and not timeseries_changed:
        retained_timeseries = expected_timeseries
        return FetchOutcome(
            None,
            None,
            0,
            None,
            (),
            0,
            (),
            prepared.skipped_resource_years,
            prepared.skipped_timeseries_years,
            None,
            None,
            0 if retained_timeseries is None else retained_timeseries.row_count,
        )

    batch_revision = str(uuid.uuid4())
    fetched_at = datetime.now(timezone.utc).isoformat()
    committed_resource: ApiSnapshot | None = None
    committed_timeseries: ApiSnapshot | None = None
    retained_timeseries: ApiSnapshot | None = None
    sqlite_committed = False
    c1_active_table: str | None = None
    c1_primary_error: BaseException | None = None
    resource_years = prepared.resource_years
    timeseries_years: tuple[int, ...] | None = None
    if prepared.timeseries:
        timeseries_years = (
            None
            if prepared.timeseries_mode == "replace_all"
            else tuple(item.report_year for item in prepared.timeseries)
        )
    try:
        sqlite_conn = get_sqlite_connection(sqlite_path)
        try:
            sqlite_conn.execute("BEGIN IMMEDIATE")
            if timeseries_changed:
                ensure_timeseries_tables(sqlite_conn)
            if resources_changed:
                c1_active_table = "project_resources"
                validate_resource_update(
                    sqlite_conn,
                    prepared.resource_rows,
                    years=resource_years,
                    expected_snapshot=expected_resource,
                )
            if not replace_all:
                c1_active_table = "project_timeseries"
                current_timeseries, current_count, current_hash = (
                    _timeseries_source_state(sqlite_conn)
                )
                if expected_timeseries is None and current_count != 0:
                    raise ValueError(
                        "project_timeseries source appeared during preparation"
                    )
                if current_timeseries != expected_timeseries:
                    raise ValueError(
                        "project_timeseries snapshot changed during preparation"
                    )
                if expected_timeseries is not None and (
                    current_count != expected_timeseries.row_count
                    or current_hash != expected_timeseries.source_sha256
                ):
                    raise ValueError(
                        "project_timeseries content changed during preparation"
                    )
            elif (
                timeseries_changed
                and get_timeseries_snapshot(sqlite_conn) != expected_timeseries
            ):
                c1_active_table = "project_timeseries"
                raise ValueError(
                    "project_timeseries snapshot changed during preparation"
                )
            if resources_changed:
                c1_active_table = "project_resources"
                committed_resource = _replace_project_resources_in_transaction(
                    sqlite_conn,
                    prepared.resource_rows,
                    ApiSnapshot(
                        "project_resources",
                        batch_revision,
                        0,
                        "",
                        fetched_at,
                        "committed",
                    ),
                    resource_years,
                )
            if timeseries_changed:
                c1_active_table = "project_timeseries"
                if prepared.timeseries:
                    # Empty selected slices are a no-op by preparation contract.
                    if timeseries_years is not None and not timeseries_years:
                        raise ValueError("selected timeseries replacement has no years")
                    inherited = _legacy_type_overrides(sqlite_conn)
                    committed_timeseries = replace_timeseries_in_transaction(
                        sqlite_conn,
                        _prepared_canonical_rows(prepared),
                        years=timeseries_years,
                        revision=batch_revision,
                        fetched_at=fetched_at,
                    )
                    retains_legacy = (
                        sqlite_conn.execute(
                            "SELECT 1 FROM api_project_timeseries "
                            "WHERE origin='legacy_duckdb' LIMIT 1"
                        ).fetchone()
                        is not None
                    )
                    details: dict[str, object] = {
                        "input_origins": sorted(
                            {item.origin for item in prepared.timeseries}
                        ),
                        "observed_years": list(prepared.timeseries_observed_years),
                    }
                    if inherited and retains_legacy:
                        details["legacy_type_overrides"] = inherited
                    origins = {item.origin for item in prepared.timeseries}
                    provenance_origin = (
                        "api_csv" if origins == {"api_csv"} else "api_json"
                    )
                    record_timeseries_provenance(
                        sqlite_conn,
                        revision=batch_revision,
                        origin=provenance_origin,
                        details=details,
                    )
                else:
                    committed_timeseries = record_observed_empty_timeseries(
                        sqlite_conn,
                        revision=batch_revision,
                        fetched_at=fetched_at,
                        origin="api_json",
                    )
                    record_timeseries_provenance(
                        sqlite_conn,
                        revision=batch_revision,
                        origin="api_json",
                        details={
                            "observed_empty": True,
                            "observed_years": list(prepared.timeseries_observed_years),
                            "input_origins": ["api_json"],
                        },
                    )
            retained_timeseries = get_timeseries_snapshot(sqlite_conn)
            c1_active_table = None
            sqlite_conn.commit()
            sqlite_committed = True
        except BaseException as exc:
            c1_primary_error = exc
            try:
                sqlite_conn.rollback()
            except BaseException as cleanup:
                if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                    exc.args = (*exc.args, f"SQLite rollback also failed: {cleanup}")
                else:
                    raise RuntimeError(
                        f"{exc}; SQLite rollback failed: {cleanup}"
                    ) from exc
            raise
        finally:
            try:
                sqlite_conn.close()
            except BaseException as cleanup:
                if isinstance(c1_primary_error, (KeyboardInterrupt, SystemExit)):
                    c1_primary_error.args = (
                        *c1_primary_error.args,
                        f"SQLite close also failed: {cleanup}",
                    )
                elif c1_primary_error is None:
                    raise
    except BaseException as exc:
        # A failed close after COMMIT is still a committed C1 and needs stale marking.
        if sqlite_committed:
            for table, snapshot in (
                ("project_resources", committed_resource),
                ("project_timeseries", committed_timeseries),
            ):
                if snapshot is not None:
                    _record_stale_after_failure(sqlite_path, snapshot.revision, table)
        failure = _failure(
            f"Canonical SQLite batch commit failed: {exc}",
            phase="publication",
            sqlite_committed=sqlite_committed,
            table=c1_active_table,
        )
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            exc.args = (*exc.args, str(failure))
            raise
        raise failure from exc

    changed: list[tuple[str, ApiSnapshot]] = []
    if committed_resource is not None:
        changed.append(("project_resources", committed_resource))
    if committed_timeseries is not None:
        changed.append(("project_timeseries", committed_timeseries))
    writer: duckdb.DuckDBPyConnection | None = None
    source_conn: sqlite3.Connection | None = None
    transaction_open = False
    duckdb_committed = False
    invalid_geometry_count: int | None = None
    resource_snapshot: ApiSnapshot | None = None
    c2_error: BaseException | None = None
    active_table: str | None = None
    try:
        writer = get_duckdb_connection(duckdb_path, read_only=False)
        writer.execute("INSTALL fts")
        writer.execute("LOAD fts")
        writer.execute("INSTALL spatial")
        writer.execute("LOAD spatial")
        writer.execute("BEGIN")
        transaction_open = True
        source_conn = _readonly_sqlite(sqlite_path)
        if source_conn is None:
            raise ValueError("canonical SQLite database disappeared before C2")
        source_conn.row_factory = sqlite3.Row
        source_conn.execute("BEGIN")

        guard_preserved_timeseries = committed_timeseries is None and not replace_all
        preserved_timeseries = expected_timeseries
        if guard_preserved_timeseries:
            captured_preserved, preserved_count, preserved_hash = (
                _timeseries_source_state(source_conn)
            )
            if preserved_timeseries is None and preserved_count != 0:
                raise ValueError(
                    "preserved project_timeseries source appeared before C2"
                )
            if captured_preserved != preserved_timeseries:
                raise ValueError(
                    "preserved project_timeseries baseline changed before C2"
                )
            if preserved_timeseries is not None and (
                preserved_count != preserved_timeseries.row_count
                or preserved_hash != preserved_timeseries.source_sha256
            ):
                raise ValueError(
                    "preserved project_timeseries baseline changed before C2"
                )

        resource_stage = None
        if committed_resource is not None:
            active_table = "project_resources"
            captured = get_project_resources_snapshot(source_conn)
            count, digest = _persisted_hash(source_conn)
            if (
                captured != committed_resource
                or count != committed_resource.row_count
                or digest != committed_resource.source_sha256
            ):
                raise ValueError("project_resources revision changed before C2 staging")
            resource_stage = _stage_project_resources(
                writer, iter_project_resources_rows(source_conn)
            )
            if resource_stage.row_count != committed_resource.row_count:
                raise ValueError("staged project_resources count does not match C1")
            invalid_geometry_count = resource_stage.invalid_geometry_count
            _swap_project_resources(writer, resource_stage)
            resource_snapshot = ApiSnapshot(
                "project_resources",
                committed_resource.revision,
                committed_resource.row_count,
                committed_resource.source_sha256,
                committed_resource.fetched_at,
                committed_resource.publication_state,
                invalid_geometry_count,
            )
            _record_api_publication(
                writer, resource_snapshot, invalid_geometry_count=invalid_geometry_count
            )

        timeseries_stage = None
        if committed_timeseries is not None:
            active_table = "project_timeseries"
            captured_ts = get_timeseries_snapshot(source_conn)
            ts_count, ts_digest = timeseries_fingerprint(source_conn)
            if (
                captured_ts != committed_timeseries
                or ts_count != committed_timeseries.row_count
                or ts_digest != committed_timeseries.source_sha256
            ):
                raise ValueError(
                    "project_timeseries revision changed before C2 staging"
                )
            if timeseries_years is not None:
                year_stages: list[TimeseriesStage] = []
                for year in timeseries_years:
                    try:
                        timeseries_stage = stage_canonical_timeseries(
                            writer, source_conn, years=(year,)
                        )
                        expected_slice_count = sum(
                            1
                            for row in iter_timeseries_rows(source_conn, years=(year,))
                        )
                        if timeseries_stage.row_count != expected_slice_count:
                            raise ValueError(
                                "staged project_timeseries slice count "
                                "does not match C1"
                            )
                        year_stages.append(timeseries_stage)
                    except Exception as exc:
                        raise ValueError(f"year={year}: {exc}") from exc
                if len(year_stages) == 1:
                    timeseries_stage = year_stages[0]
                else:
                    combined_table = f"project_timeseries_stage_{uuid.uuid4().hex}"
                    union = " UNION ALL ".join(
                        f"SELECT * FROM {stage.table_name}" for stage in year_stages
                    )
                    writer.execute(f"CREATE TEMP TABLE {combined_table} AS {union}")
                    for stage in year_stages:
                        writer.execute(f"DROP TABLE {stage.table_name}")
                    timeseries_stage = TimeseriesStage(
                        combined_table,
                        sum(stage.row_count for stage in year_stages),
                        timeseries_years,
                    )
                apply_timeseries_stage(writer, timeseries_stage)
            else:
                timeseries_stage = stage_canonical_timeseries(
                    writer, source_conn, years=None
                )
                if timeseries_stage.row_count != ts_count:
                    raise ValueError(
                        "staged project_timeseries count does not match C1"
                    )
                apply_timeseries_stage(writer, timeseries_stage)
            _record_api_publication(writer, committed_timeseries)

        # A fresh read observes concurrent C1 changes, while the pinned source
        # transaction above remains the exact row pair used for these markers.
        check_conn = _readonly_sqlite(sqlite_path)
        if check_conn is None:
            raise ValueError("canonical SQLite database disappeared before C2 commit")
        try:
            check_conn.execute("BEGIN")
            for source_table, snapshot in changed:
                active_table = source_table
                latest = (
                    get_project_resources_snapshot(check_conn)
                    if source_table == "project_resources"
                    else get_timeseries_snapshot(check_conn)
                )
                if latest != snapshot:
                    raise ValueError(
                        f"{source_table} revision was superseded before C2 commit"
                    )
                if source_table == "project_resources":
                    latest_count, latest_hash = _persisted_hash(check_conn)
                else:
                    latest_count, latest_hash = timeseries_fingerprint(check_conn)
                if (
                    latest_count != snapshot.row_count
                    or latest_hash != snapshot.source_sha256
                ):
                    raise ValueError(f"{source_table} content changed before C2 commit")
            if guard_preserved_timeseries:
                active_table = "project_timeseries"
                latest_preserved, latest_count, latest_hash = _timeseries_source_state(
                    check_conn
                )
                if preserved_timeseries is None and latest_count != 0:
                    raise ValueError(
                        "preserved project_timeseries source appeared before C2 commit"
                    )
                if latest_preserved != preserved_timeseries:
                    raise ValueError(
                        "preserved project_timeseries baseline changed before C2 commit"
                    )
                if preserved_timeseries is not None and (
                    latest_count != preserved_timeseries.row_count
                    or latest_hash != preserved_timeseries.source_sha256
                ):
                    raise ValueError(
                        "preserved project_timeseries baseline changed before C2 commit"
                    )
        finally:
            check_conn.close()
        active_table = None
        writer.execute("COMMIT")
        transaction_open = False
        duckdb_committed = True
    except BaseException as exc:
        c2_error = exc
        if writer is not None and transaction_open:
            try:
                writer.execute("ROLLBACK")
                transaction_open = False
            except BaseException as cleanup:
                if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                    exc.args = (*exc.args, f"DuckDB rollback also failed: {cleanup}")
                    c2_error = exc
                else:
                    c2_error = RuntimeError(f"{exc}; DuckDB rollback failed: {cleanup}")
    finally:
        if source_conn is not None:
            try:
                source_conn.close()
            except BaseException as close_error:
                if isinstance(close_error, (KeyboardInterrupt, SystemExit)):
                    if c2_error is not None:
                        close_error.args = (
                            *close_error.args,
                            f"Earlier C2 failure: {c2_error}",
                        )
                    c2_error = close_error
                elif c2_error is None:
                    c2_error = close_error
            source_conn = None
        if writer is not None:
            try:
                writer.close()
            except BaseException as close_error:
                if isinstance(close_error, (KeyboardInterrupt, SystemExit)):
                    if c2_error is not None:
                        close_error.args = (
                            *close_error.args,
                            f"Earlier C2 failure: {c2_error}",
                        )
                    c2_error = close_error
                elif c2_error is None:
                    c2_error = close_error
            writer = None
    if c2_error is not None:
        stale_errors = [
            _record_stale_after_failure(sqlite_path, snapshot.revision, table)
            for table, snapshot in changed
        ]
        suffix = "; ".join(error for error in stale_errors if error)
        failure = _failure(
            f"Analytical publication failed during C2: {c2_error}"
            + (f"; {suffix}" if suffix else ""),
            phase="publication",
            sqlite_committed=True,
            duckdb_committed=duckdb_committed,
            resource_revision=(
                None if committed_resource is None else committed_resource.revision
            ),
            timeseries_revision=(
                None if committed_timeseries is None else committed_timeseries.revision
            ),
            table=active_table
            or ("project_timeseries" if committed_timeseries else "project_resources"),
        )
        if isinstance(c2_error, (KeyboardInterrupt, SystemExit)):
            c2_error.args = (*c2_error.args, str(failure))
            raise c2_error
        raise failure from c2_error

    # C3 remains independent: attempt every maintenance action after the C2
    # writer has closed, then mark only captured revisions as published.
    maintenance: list[BaseException] = []
    try:
        from esdc.api_projection import _invalidate_analytical_caches

        _invalidate_analytical_caches()
    except BaseException as exc:
        maintenance.append(exc)
    checkpoint_conn: duckdb.DuckDBPyConnection | None = None
    try:
        checkpoint_conn = get_duckdb_connection(duckdb_path, read_only=False)
        _checkpoint(checkpoint_conn)
    except BaseException as exc:
        maintenance.append(exc)
    finally:
        if checkpoint_conn is not None:
            try:
                checkpoint_conn.close()
            except BaseException as exc:
                maintenance.append(exc)
    if reindex:
        try:
            _reindex_fts_for_path(duckdb_path, strict=True)
        except BaseException as exc:
            maintenance.append(exc)
    for table, snapshot in changed:
        try:
            marking_snapshot = (
                resource_snapshot
                if table == "project_resources" and resource_snapshot is not None
                else snapshot
            )
            _mark_source_published(sqlite_path, table, marking_snapshot)
        except BaseException as exc:
            maintenance.append(exc)
            _record_stale_after_failure(sqlite_path, snapshot.revision, table)
    if maintenance:
        for table, snapshot in changed:
            _record_stale_after_failure(sqlite_path, snapshot.revision, table)
        failure = _failure(
            "Post-commit maintenance or publication-state update failed: "
            + "; ".join(str(exc) for exc in maintenance),
            phase="post_commit",
            sqlite_committed=True,
            duckdb_committed=duckdb_committed,
            resource_revision=(
                None if committed_resource is None else committed_resource.revision
            ),
            timeseries_revision=(
                None if committed_timeseries is None else committed_timeseries.revision
            ),
            table="project_timeseries" if committed_timeseries else "project_resources",
        )
        interruption = next(
            (
                exc
                for exc in maintenance
                if isinstance(exc, (KeyboardInterrupt, SystemExit))
            ),
            None,
        )
        if interruption is not None:
            interruption.args = (*interruption.args, str(failure))
            raise interruption
        raise failure from maintenance[0]

    updated_resource_years = resource_years
    if updated_resource_years is None and resources_changed:
        updated_resource_years = tuple(
            sorted(
                {
                    parse_report_year(row.report_year)
                    for row in prepared.resource_rows
                    if row.report_year is not None
                }
            )
        )
    updated_timeseries_years = tuple(item.report_year for item in prepared.timeseries)
    published_timeseries = committed_timeseries or retained_timeseries
    return FetchOutcome(
        None if committed_resource is None else committed_resource.revision,
        None if committed_resource is None else committed_resource.source_sha256,
        0 if committed_resource is None else committed_resource.row_count,
        invalid_geometry_count,
        () if updated_resource_years is None else tuple(updated_resource_years),
        sum(item.row_count for item in prepared.timeseries),
        updated_timeseries_years,
        prepared.skipped_resource_years,
        prepared.skipped_timeseries_years,
        None if published_timeseries is None else published_timeseries.revision,
        None if published_timeseries is None else published_timeseries.source_sha256,
        0 if published_timeseries is None else published_timeseries.row_count,
    )


def ingest_timeseries_rows(
    content: Sequence[Sequence[JSONScalar]],
    header: Sequence[str],
    *,
    duckdb_path: Path,
    sqlite_path: Path,
    replace_all: bool = True,
    reindex: bool = True,
) -> FetchOutcome:
    """Ingest programmatic rows through the canonical SQLite coordinator."""
    return _ingest_timeseries_rows(
        content,
        header,
        duckdb_path=duckdb_path,
        sqlite_path=sqlite_path,
        replace_all=replace_all,
        reindex=reindex,
        origin="api_json",
    )


def _ingest_timeseries_rows(
    content: Sequence[Sequence[JSONScalar]],
    header: Sequence[str],
    *,
    duckdb_path: Path,
    sqlite_path: Path,
    replace_all: bool,
    reindex: bool,
    origin: Literal["api_json", "api_csv"],
) -> FetchOutcome:
    from esdc.fetching import (
        PreparedFetch,
        spool_timeseries,
        validate_timeseries_header,
    )
    from esdc.fetching import (
        parse_report_year as parse_year,
    )

    if not content:
        return FetchOutcome(None, None, 0, None, (), 0, (), (), ())
    names = validate_timeseries_header(header)
    year_index = names.index("report_year")
    rows_by_year: dict[int, list[Sequence[JSONScalar]]] = {}
    for index, row in enumerate(content):
        if len(row) != len(names):
            raise ValueError(f"project_timeseries record {index} has wrong width")
        try:
            year = parse_year(row[year_index])
        except ValueError as exc:
            raise ValueError(
                f"project_timeseries record {index} column report_year: {exc}"
            ) from exc
        normalized = list(row)
        normalized[year_index] = year
        rows_by_year.setdefault(year, []).append(normalized)

    with tempfile.TemporaryDirectory(prefix="esdc-timeseries-ingest-") as directory:
        slices = []
        for year in sorted(rows_by_year):
            prepared_slice = spool_timeseries(
                names,
                rows_by_year[year],
                report_year=year,
                directory=Path(directory),
                origin=origin,
            )
            if prepared_slice is not None:
                slices.append(prepared_slice)
        prepared = PreparedFetch(
            (),
            None,
            tuple(slices),
            (),
            (),
            timeseries_mode="replace_all" if replace_all else "replace_years",
        )
        return publish_prepared_fetch(
            prepared,
            duckdb_path=duckdb_path,
            sqlite_path=sqlite_path,
            reindex=reindex,
        )
