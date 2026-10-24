"""Offline recovery of the committed SQLite API pair into DuckDB."""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypedDict

import duckdb

from esdc.api_projection import (
    ResourcePublicationRecord,
    _invalidate_analytical_caches,
    _record_api_publication,
    _stage_project_resources,
    _swap_project_resources,
    get_api_publication,
)
from esdc.api_snapshot import (
    ApiSnapshot,
    _persisted_hash,
    get_project_resources_snapshot,
    iter_project_resources_rows,
)
from esdc.dbmanager import _reindex_fts_for_path, get_duckdb_connection
from esdc.timeseries_projection import (
    _legacy_type_overrides,
    apply_timeseries_stage,
    stage_canonical_timeseries,
)
from esdc.timeseries_snapshot import (
    get_timeseries_snapshot,
    iter_timeseries_rows,
    timeseries_fingerprint,
)


@dataclass(frozen=True)
class AnalyticsRebuildReport:
    """Outcome and source markers for an API analytics reconstruction."""

    output_path: Path
    source_revisions: dict[str, str]
    source_hashes: dict[str, str]
    subsystem_status: dict[str, Literal["restored", "absent"]]
    object_counts: dict[str, int]
    requires_regeneration: tuple[str, ...]


class SnapshotStatus(TypedDict):
    revision: str
    row_count: int
    source_sha256: str
    publication_state: str


class LineageStatus(TypedDict):
    revision: str
    source_row_count: int
    source_sha256: str
    invalid_geometry_count: int


class ApiSourceStatus(TypedDict):
    status: str
    detail: str
    snapshot: SnapshotStatus | None
    lineage: LineageStatus | None


@dataclass(frozen=True)
class _ApiSources:
    resources: ApiSnapshot
    timeseries: ApiSnapshot
    resource_count: int
    resource_hash: str
    timeseries_count: int
    timeseries_hash: str


def _sqlite_readonly(path: Path) -> sqlite3.Connection:
    if not path.is_file():
        raise FileNotFoundError(f"canonical SQLite database does not exist: {path}")
    connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("BEGIN")
    return connection


def _table_exists(connection: sqlite3.Connection, table: str) -> bool:
    return (
        connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
        ).fetchone()
        is not None
    )


def _validate_sources(connection: sqlite3.Connection) -> _ApiSources:
    required = {"api_snapshot", "api_project_resources", "api_project_timeseries"}
    missing = sorted(
        table for table in required if not _table_exists(connection, table)
    )
    if missing:
        raise ValueError(
            "canonical API source is missing table(s): " + ", ".join(missing)
        )

    resources = get_project_resources_snapshot(connection)
    timeseries = get_timeseries_snapshot(connection)
    if resources is None or timeseries is None:
        raise ValueError("both canonical API snapshots are required for pair recovery")
    if resources.publication_state not in {"committed", "published", "analytics_stale"}:
        raise ValueError("project_resources snapshot state is invalid")
    if timeseries.publication_state not in {
        "committed",
        "published",
        "analytics_stale",
    }:
        raise ValueError("project_timeseries snapshot state is invalid")

    resource_count, resource_hash = _persisted_hash(connection)
    if (
        resources.row_count != resource_count
        or resources.source_sha256 != resource_hash
    ):
        raise ValueError(
            "project_resources snapshot count/hash does not match canonical rows"
        )
    if resource_count == 0:
        raise ValueError(
            "project_resources recovery requires a verified nonempty source"
        )

    timeseries_count, timeseries_hash = timeseries_fingerprint(connection)
    if (
        timeseries.row_count != timeseries_count
        or timeseries.source_sha256 != timeseries_hash
    ):
        raise ValueError(
            "project_timeseries snapshot count/hash does not match canonical rows"
        )
    if not _table_exists(connection, "api_snapshot_provenance"):
        raise ValueError("project_timeseries current provenance is missing")
    provenance = connection.execute(
        "SELECT origin, details_json FROM api_snapshot_provenance "
        "WHERE source_table='project_timeseries' AND revision=?",
        (timeseries.revision,),
    ).fetchone()
    if provenance is None or provenance[0] not in {
        "api_json",
        "api_csv",
        "legacy_duckdb",
    }:
        raise ValueError("project_timeseries current provenance is missing or invalid")
    try:
        details = json.loads(provenance[1])
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("project_timeseries current provenance is malformed") from exc
    if not isinstance(details, dict):
        raise ValueError("project_timeseries current provenance is malformed")
    source_origins = {
        row[0]
        for row in connection.execute(
            "SELECT DISTINCT origin FROM api_project_timeseries"
        ).fetchall()
    }
    allowed_origins = {"api_json", "api_csv", "legacy_duckdb"}
    if not source_origins <= allowed_origins:
        raise ValueError("project_timeseries rows contain an invalid source origin")
    input_origins = details.get("input_origins")
    if (
        (timeseries_count == 0 or provenance[0] != "legacy_duckdb")
        and (
            not isinstance(input_origins, list)
            or not input_origins
            or any(
                not isinstance(origin, str) or origin not in {"api_json", "api_csv"}
                for origin in input_origins
            )
        )
    ):
        raise ValueError(
            "project_timeseries current provenance input_origins is invalid"
        )
    if timeseries_count == 0:
        if (
            provenance[0] not in {"api_json", "api_csv"}
            or details.get("observed_empty") is not True
            or provenance[0] not in input_origins
        ):
            raise ValueError(
                "empty project_timeseries lacks observed-empty current provenance"
            )
    elif provenance[0] == "legacy_duckdb":
        if "source_types" not in details:
            raise ValueError("legacy project_timeseries provenance lacks source types")
    else:
        if (
            provenance[0] not in input_origins
        ):
            raise ValueError(
                "project_timeseries current provenance origins do not match rows"
            )

    # Force every source row through its canonical parser while the read view is pinned.
    _legacy_type_overrides(connection)
    for _ in iter_project_resources_rows(connection):
        pass
    for _ in iter_timeseries_rows(connection):
        pass
    return _ApiSources(
        resources,
        timeseries,
        resource_count,
        resource_hash,
        timeseries_count,
        timeseries_hash,
    )


def _preflight_extensions() -> None:
    connection = duckdb.connect(
        ":memory:", config={"autoinstall_known_extensions": "false"}
    )
    failure: BaseException | None = None
    try:
        for extension in ("vss", "spatial", "fts"):
            try:
                connection.execute(f"LOAD {extension}")
            except duckdb.Error as exc:
                failure = RuntimeError(
                    f"required DuckDB extension {extension!r} is not cached; "
                    "install it separately before offline analytics recovery"
                )
                raise failure from exc
    except BaseException as exc:
        failure = exc
        raise
    finally:
        try:
            connection.close()
        except BaseException as cleanup:
            if failure is None:
                raise
            if isinstance(failure, (KeyboardInterrupt, SystemExit)):
                failure.args = (
                    *failure.args,
                    f"closing extension preflight also failed: {cleanup}",
                )
            else:
                logging.error(
                    "Closing extension preflight failed after %r: %s", failure, cleanup
                )


def _same_file(first: Path, second: Path) -> bool:
    if first.resolve() == second.resolve():
        return True
    try:
        return first.samefile(second)
    except FileNotFoundError:
        return False


def _mark_sources(
    sqlite_path: Path,
    sources: _ApiSources,
    state: Literal["published", "analytics_stale"],
    *,
    invalid_geometry_count: int | None = None,
) -> None:
    connection = sqlite3.connect(sqlite_path)
    failure: BaseException | None = None
    try:
        connection.execute("BEGIN IMMEDIATE")
        for table, snapshot in (
            ("project_resources", sources.resources),
            ("project_timeseries", sources.timeseries),
        ):
            if table == "project_resources":
                cursor = connection.execute(
                    "UPDATE api_snapshot SET publication_state=?, "
                    "invalid_geometry_count=COALESCE(?, invalid_geometry_count) "
                    "WHERE source_table=? AND revision=?",
                    (state, invalid_geometry_count, table, snapshot.revision),
                )
            else:
                cursor = connection.execute(
                    "UPDATE api_snapshot SET publication_state=? "
                    "WHERE source_table=? AND revision=?",
                    (state, table, snapshot.revision),
                )
            if cursor.rowcount != 1:
                raise ValueError(f"{table} revision changed before publication marking")
        connection.commit()
    except BaseException as exc:
        failure = exc
        try:
            connection.rollback()
        except BaseException as cleanup:
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                exc.args = (*exc.args, f"SQLite rollback also failed: {cleanup}")
            else:
                logging.error(
                    "SQLite rollback failed after publication marker error: %s", cleanup
                )
        raise
    finally:
        try:
            connection.close()
        except BaseException as cleanup:
            if failure is None:
                raise
            if isinstance(failure, (KeyboardInterrupt, SystemExit)):
                failure.args = (
                    *failure.args,
                    f"closing SQLite marker writer also failed: {cleanup}",
                )
            else:
                logging.error(
                    "Closing SQLite marker writer failed after %r: %s", failure, cleanup
                )


def _recheck_sources(sqlite_path: Path, expected: _ApiSources) -> None:
    connection = _sqlite_readonly(sqlite_path)
    try:
        current = _validate_sources(connection)
        if current != expected:
            raise ValueError("canonical API source pair changed before DuckDB commit")
    finally:
        connection.close()


def _resource_projection_matches(sqlite_path: Path, duckdb_path: Path) -> bool:
    """Compare the typed/derived projection against the canonical source."""
    from esdc.resource_audit import compare_resource_projection

    target = duckdb.connect(
        str(duckdb_path),
        read_only=True,
        config={"autoinstall_known_extensions": "false"},
    )
    try:
        report = compare_resource_projection(target, sqlite_path)
        if any(
            "could not" in issue.lower() or "cannot inspect" in issue.lower()
            for issue in report.issues
        ):
            raise RuntimeError("resource projection parity read could not be completed")
        return report.ok
    finally:
        target.close()


def _record_stale_after_failure(
    sqlite_path: Path, sources: _ApiSources, error: BaseException
) -> None:
    try:
        _mark_sources(sqlite_path, sources, "analytics_stale")
    except BaseException as cleanup:
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            error.args = (
                *error.args,
                f"recording analytics_stale also failed: {cleanup}",
            )
        else:
            logging.exception(
                "Could not mark API source pair stale after recovery failure"
            )


def rebuild_api_analytics(
    *, duckdb_path: Path, sqlite_path: Path, reindex: bool = True
) -> AnalyticsRebuildReport:
    """Rebuild both API projections from committed SQLite snapshots offline."""
    duckdb_path = Path(duckdb_path)
    sqlite_path = Path(sqlite_path)
    if _same_file(duckdb_path, sqlite_path):
        raise ValueError("SQLite source and DuckDB target must be different files")

    source: sqlite3.Connection | None = _sqlite_readonly(sqlite_path)
    writer: duckdb.DuckDBPyConnection | None = None
    sources: _ApiSources | None = None
    c2_committed = False
    post_commit_errors: list[BaseException] = []
    resource_count = timeseries_count = invalid_geometry_count = 0
    hnsw_present = False
    try:
        assert source is not None
        sources = _validate_sources(source)
        # A verified observed-empty source may only initialize an absent/empty target.
        if sources.timeseries_count == 0 and duckdb_path.is_file():
            with duckdb.connect(
                str(duckdb_path),
                read_only=True,
                config={"autoinstall_known_extensions": "false"},
            ) as inspection:
                table_exists = (
                    inspection.execute(
                        "SELECT 1 FROM information_schema.tables "
                        "WHERE table_schema='main' "
                        "AND table_name='project_timeseries'"
                    ).fetchone()
                    is not None
                )
                target_count = (
                    inspection.execute(
                        "SELECT COUNT(*) FROM project_timeseries"
                    ).fetchone()
                    if table_exists
                    else None
                )
                if table_exists and (target_count is None or target_count[0] != 0):
                    raise ValueError(
                        "verified empty project_timeseries cannot replace "
                        "a populated target"
                    )
        _preflight_extensions()

        writer = get_duckdb_connection(
            duckdb_path, read_only=False, allow_install=False
        )
        writer.execute("LOAD fts")
        hnsw_present = (
            writer.execute(
                "SELECT 1 FROM duckdb_indexes() WHERE schema_name='main' "
                "AND index_name='idx_hnsw_embeddings'"
            ).fetchone()
            is not None
        )
        writer.execute("BEGIN TRANSACTION")
        try:
            assert source is not None
            if sources.timeseries_count == 0:
                target_timeseries = writer.execute(
                    "SELECT 1 FROM information_schema.tables "
                    "WHERE table_schema='main' AND table_name='project_timeseries'"
                ).fetchone()
                target_count = (
                    writer.execute(
                        "SELECT COUNT(*) FROM project_timeseries"
                    ).fetchone()
                    if target_timeseries is not None
                    else None
                )
                if target_timeseries is not None and (
                    target_count is None or target_count[0] != 0
                ):
                    raise ValueError(
                        "verified empty project_timeseries cannot replace "
                        "a populated target"
                    )
            resource_stage = _stage_project_resources(
                writer, iter_project_resources_rows(source)
            )
            timeseries_stage = stage_canonical_timeseries(writer, source, years=None)
            if resource_stage.row_count != sources.resource_count:
                raise ValueError("staged project_resources count differs from SQLite")
            if timeseries_stage.row_count != sources.timeseries_count:
                raise ValueError("staged project_timeseries count differs from SQLite")
            _swap_project_resources(writer, resource_stage, allow_install=False)
            apply_timeseries_stage(writer, timeseries_stage)
            _record_api_publication(
                writer,
                sources.resources,
                invalid_geometry_count=resource_stage.invalid_geometry_count,
            )
            _record_api_publication(writer, sources.timeseries)
            resource_count = resource_stage.row_count
            timeseries_count = timeseries_stage.row_count
            invalid_geometry_count = resource_stage.invalid_geometry_count
            _recheck_sources(sqlite_path, sources)
            writer.execute("COMMIT")
            c2_committed = True
        except BaseException as exc:
            try:
                writer.execute("ROLLBACK")
            except BaseException as cleanup:
                if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                    exc.args = (*exc.args, f"DuckDB rollback also failed: {cleanup}")
                else:
                    logging.error(
                        "DuckDB rollback failed after API recovery error: %s", cleanup
                    )
            try:
                writer.close()
            except BaseException as cleanup:
                if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                    exc.args = (
                        *exc.args,
                        f"closing DuckDB writer also failed: {cleanup}",
                    )
                else:
                    logging.error(
                        "Closing DuckDB writer failed after API recovery error: %s",
                        cleanup,
                    )
            writer = None
            if source is not None:
                try:
                    source.close()
                except BaseException as cleanup:
                    if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                        exc.args = (
                            *exc.args,
                            f"closing SQLite source also failed: {cleanup}",
                        )
                source = None
            _record_stale_after_failure(sqlite_path, sources, exc)
            raise
        else:
            try:
                writer.close()
            except BaseException as cleanup:
                post_commit_errors.append(cleanup)
            writer = None
            if source is not None:
                try:
                    source.close()
                except BaseException as cleanup:
                    post_commit_errors.append(cleanup)
                source = None
    except BaseException as exc:
        if writer is not None:
            try:
                writer.close()
            except BaseException as cleanup:
                if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                    exc.args = (
                        *exc.args,
                        f"closing DuckDB writer also failed: {cleanup}",
                    )
                else:
                    logging.error(
                        "Closing DuckDB writer failed after API recovery error: %s",
                        cleanup,
                    )
            writer = None
        if source is not None:
            try:
                source.close()
            except BaseException as cleanup:
                if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                    exc.args = (
                        *exc.args,
                        f"closing SQLite source also failed: {cleanup}",
                    )
                else:
                    logging.error(
                        "Closing SQLite source failed after API recovery error: %s",
                        cleanup,
                    )
            source = None
        if c2_committed and sources is not None:
            _record_stale_after_failure(sqlite_path, sources, exc)
        raise
    finally:
        if source is not None:
            source.close()

    assert sources is not None
    maintenance_errors = post_commit_errors
    for maintenance in (
        _invalidate_analytical_caches,
        lambda: _checkpoint_path(duckdb_path),
    ):
        try:
            maintenance()
        except BaseException as exc:
            maintenance_errors.append(exc)
    if reindex:
        try:
            _reindex_fts_for_path(duckdb_path, strict=True, allow_install=False)
        except BaseException as exc:
            maintenance_errors.append(exc)
    if maintenance_errors:
        primary = next(
            (
                error
                for error in maintenance_errors
                if isinstance(error, (KeyboardInterrupt, SystemExit))
            ),
            maintenance_errors[0],
        )
        _record_stale_after_failure(sqlite_path, sources, primary)
        for extra in maintenance_errors:
            if extra is not primary:
                primary.args = (*primary.args, str(extra))
        if isinstance(primary, (KeyboardInterrupt, SystemExit)):
            raise primary
        raise RuntimeError(
            "API analytics committed but post-commit maintenance failed: "
            + "; ".join(str(error) for error in maintenance_errors)
            + ". Retry with 'esdc rebuild-api-analytics'."
        ) from primary

    try:
        _mark_sources(
            sqlite_path,
            sources,
            "published",
            invalid_geometry_count=invalid_geometry_count,
        )
    except BaseException as exc:
        _record_stale_after_failure(sqlite_path, sources, exc)
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            exc.args = (
                *exc.args,
                "DuckDB API analytics committed; publication marking failed; "
                "canonical publication remains pending or stale. "
                f"Cause: {exc}. Run 'esdc rebuild-api-analytics'.",
            )
            raise
        raise RuntimeError(
            "DuckDB API analytics committed; publication marking failed; "
            "canonical publication remains pending or stale. "
            f"Cause: {exc}. Run 'esdc rebuild-api-analytics'."
        ) from exc
    return AnalyticsRebuildReport(
        output_path=duckdb_path,
        source_revisions={
            "project_resources": sources.resources.revision,
            "project_timeseries": sources.timeseries.revision,
        },
        source_hashes={
            "project_resources": sources.resource_hash,
            "project_timeseries": sources.timeseries_hash,
        },
        subsystem_status={"api": "restored"},
        object_counts={
            "project_resources": resource_count,
            "project_timeseries": timeseries_count,
            "invalid_resource_geometry": invalid_geometry_count,
        },
        requires_regeneration=("embeddings",) if hnsw_present and reindex else (),
    )


def _checkpoint_path(path: Path) -> None:
    connection = get_duckdb_connection(path, read_only=False, allow_install=False)
    failure: BaseException | None = None
    try:
        connection.execute("CHECKPOINT")
    except BaseException as exc:
        failure = exc
        raise
    finally:
        try:
            connection.close()
        except BaseException as cleanup:
            if failure is None:
                raise
            if isinstance(failure, (KeyboardInterrupt, SystemExit)):
                failure.args = (
                    *failure.args,
                    f"closing checkpoint connection also failed: {cleanup}",
                )
            else:
                logging.error(
                    "Closing checkpoint connection failed after %r: %s",
                    failure,
                    cleanup,
                )


def get_api_analytics_status(
    *, duckdb_path: Path, sqlite_path: Path
) -> dict[str, ApiSourceStatus]:
    """Read API source markers and verify whether each DuckDB projection is current."""
    from esdc.fetch_publication import (
        ensure_selective_resource_baseline,
        ensure_timeseries_baseline,
    )

    result: dict[str, ApiSourceStatus] = {}
    source_path_exists = sqlite_path.exists()
    source_exists = sqlite_path.is_file()
    snapshots: dict[str, ApiSnapshot | None] = {
        "project_resources": None,
        "project_timeseries": None,
    }
    source_errors: dict[str, str] = {}
    if source_path_exists and not source_exists:
        source_errors = {
            "project_resources": "canonical SQLite path is not a regular file",
            "project_timeseries": "canonical SQLite path is not a regular file",
        }
    if source_exists:
        source: sqlite3.Connection | None = None
        try:
            source = _sqlite_readonly(sqlite_path)
            for table, source_table in (
                ("api_project_resources", "project_resources"),
                ("api_project_timeseries", "project_timeseries"),
            ):
                if not _table_exists(source, table) or not _table_exists(
                    source, "api_snapshot"
                ):
                    source_errors[source_table] = (
                        "canonical source table or snapshot metadata is missing"
                    )
                    continue
                snapshot = (
                    get_project_resources_snapshot(source)
                    if source_table == "project_resources"
                    else get_timeseries_snapshot(source)
                )
                snapshots[source_table] = snapshot
                if snapshot is None:
                    source_errors[source_table] = "canonical snapshot is missing"
                elif source_table == "project_resources":
                    count, digest = _persisted_hash(source)
                    if count != snapshot.row_count or digest != snapshot.source_sha256:
                        source_errors[source_table] = (
                            "canonical snapshot count/hash is stale"
                        )
                else:
                    count, digest = timeseries_fingerprint(source)
                    if count != snapshot.row_count or digest != snapshot.source_sha256:
                        source_errors[source_table] = (
                            "canonical snapshot count/hash is stale"
                        )
        except (OSError, sqlite3.Error, ValueError, TypeError) as exc:
            source_errors = {
                "project_resources": str(exc),
                "project_timeseries": str(exc),
            }
        finally:
            if source is not None:
                try:
                    source.close()
                except Exception as exc:
                    detail = f"closing SQLite status read failed: {exc}"
                    source_errors = {
                        "project_resources": detail,
                        "project_timeseries": detail,
                    }

    lineage_rows: dict[str, ResourcePublicationRecord | None] = {}
    target_tables: set[str] = set()
    target_error: str | None = None
    target_path_exists = duckdb_path.exists()
    if target_path_exists and not duckdb_path.is_file():
        target_error = "DuckDB target path is not a regular file"
    elif duckdb_path.is_file():
        target: duckdb.DuckDBPyConnection | None = None
        try:
            target = duckdb.connect(
                str(duckdb_path),
                read_only=True,
                config={"autoinstall_known_extensions": "false"},
            )
            target_tables = {
                row[0]
                for row in target.execute(
                    "SELECT table_name FROM information_schema.tables "
                    "WHERE table_schema='main'"
                ).fetchall()
            }
            lineage_rows = {
                table: get_api_publication(target, table)
                for table in ("project_resources", "project_timeseries")
            }
        except (OSError, duckdb.Error, ValueError) as exc:
            target_error = str(exc)
        finally:
            if target is not None:
                try:
                    target.close()
                except Exception as exc:
                    target_error = f"closing DuckDB status read failed: {exc}"

    for table in ("project_resources", "project_timeseries"):
        snapshot = snapshots[table]
        lineage = lineage_rows.get(table)
        if target_error is not None:
            state = "error"
            detail = target_error
        elif table in source_errors:
            state = "error"
            detail = source_errors[table]
        elif not source_exists:
            state = "absent"
            detail = "canonical SQLite database is absent"
        elif snapshot is None:
            state = "missing"
            detail = "canonical snapshot is missing"
        elif not duckdb_path.is_file() or table not in target_tables or lineage is None:
            state = "missing"
            detail = "DuckDB projection or lineage is missing"
        elif (
            lineage.revision != snapshot.revision
            or lineage.source_row_count != snapshot.row_count
            or lineage.source_sha256 != snapshot.source_sha256
        ):
            state = "stale"
            detail = "DuckDB lineage does not match the current SQLite snapshot"
        elif snapshot.publication_state != "published":
            state = "pending" if snapshot.publication_state == "committed" else "stale"
            detail = f"SQLite publication state is {snapshot.publication_state}"
        else:
            try:
                if table == "project_resources":
                    ensure_selective_resource_baseline(
                        duckdb_path, sqlite_path, allow_install=False
                    )
                    if not _resource_projection_matches(sqlite_path, duckdb_path):
                        raise ValueError(
                            "DuckDB project_resources values differ from "
                            "canonical SQLite rows"
                        )
                else:
                    ensure_timeseries_baseline(
                        duckdb_path, sqlite_path, allow_install=False
                    )
                state = "published"
                detail = "SQLite rows, snapshot, and DuckDB projection agree"
            except (OSError, sqlite3.Error, duckdb.Error, RuntimeError) as exc:
                state = "error"
                detail = str(exc)
            except ValueError as exc:
                detail = str(exc)
                state = (
                    "error"
                    if any(
                        marker in detail.lower()
                        for marker in ("could not", "cannot inspect", "read failed")
                    )
                    else "stale"
                )
        result[table] = {
            "status": state,
            "detail": detail,
            "snapshot": None
            if snapshot is None
            else {
                "revision": snapshot.revision,
                "row_count": snapshot.row_count,
                "source_sha256": snapshot.source_sha256,
                "publication_state": snapshot.publication_state,
            },
            "lineage": None
            if lineage is None
            else {
                "revision": lineage.revision,
                "source_row_count": lineage.source_row_count,
                "source_sha256": lineage.source_sha256,
                "invalid_geometry_count": lineage.invalid_geometry_count,
            },
        }
    return result
