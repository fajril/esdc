"""Shared SQLite-only publication of committed canonical source truth (Task5D2b).

The publisher consumes committed SQLite truth only: no HTTP body, no spool, no prepared
fetch object, no source re-apply, no new source generation and no new revision UUID. It
projects the canonical project pair first and only then the POD projection, so a
whole-fetch freshness claim is made when both parts agree with the captured generation.
"""

from __future__ import annotations

import dataclasses
import sqlite3
from pathlib import Path
from typing import cast

import duckdb

from esdc import analytics_rebuild
from esdc.analytics_rebuild import (
    AnalyticsRebuildReport,
    ApiSourceStatus,
    ComponentPublication,
    ComponentStatus,
)
from esdc.fetching import FetchFailure

RECOVERY_COMMAND = "esdc rebuild-api-analytics"


def committed_pod_source_generation(sqlite_path: Path) -> int | None:
    """Read the committed POD source generation from SQLite without writing anywhere."""
    path = Path(sqlite_path)
    if not path.is_file():
        return None
    connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        connection.execute("BEGIN")
        if (
            connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' "
                "AND name='pod_registry_state'"
            ).fetchone()
            is None
        ):
            return None
        row = connection.execute(
            "SELECT source_generation FROM pod_registry_state WHERE singleton_id=1"
        ).fetchone()
    finally:
        connection.close()
    return None if row is None else int(row[0])


def _pod_source_absent(path: Path) -> bool:
    """Distinguish uninitialized POD from an invalid or incompatible registry."""
    from esdc.pod_registry.migrations import inspect_migration

    if not path.is_file():
        return True
    connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        connection.execute("BEGIN")
        return inspect_migration(connection).status == "fresh"
    except (sqlite3.Error, ValueError, RuntimeError):
        return False
    finally:
        connection.close()


def _projection_counts(duckdb_path: Path) -> dict[str, int]:
    """Read the current mirror projection row counts without writing anywhere."""
    if not duckdb_path.is_file():
        return {}
    connection = duckdb.connect(
        str(duckdb_path),
        read_only=True,
        config={"autoinstall_known_extensions": "false"},
    )
    try:
        counts: dict[str, int] = {}
        for table in ("project_resources", "project_timeseries"):
            exists = connection.execute(
                "SELECT 1 FROM information_schema.tables "
                "WHERE table_schema='main' AND table_name=?",
                (table,),
            ).fetchone()
            if exists is not None:
                row = connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
                if row is not None:
                    counts[table] = int(row[0])
        return counts
    finally:
        connection.close()


def _pair_component(
    table: str,
    entry: ApiSourceStatus,
    projected_rows: int,
) -> ComponentPublication:
    """Report one canonical pair part from committed SQLite truth, never from cache."""
    snapshot = entry["snapshot"]
    status: ComponentStatus
    if snapshot is None:
        # A partner that was never observed is absent; unowned cached mirror data stays
        # preserved and is not claimed as the current projection.
        status = "preserved" if projected_rows > 0 else "absent"
    elif entry["status"] == "published":
        status = "published"
    elif snapshot["publication_state"] == "committed":
        status = "pending"
    elif entry["status"] in {"stale", "error", "absent", "pending", "preserved"}:
        status = cast(ComponentStatus, entry["status"])
    else:
        status = "preserved" if projected_rows > 0 else "absent"
    return ComponentPublication(
        name=table,
        status=status,
        rows_written=projected_rows,
        revision=None if snapshot is None else snapshot["revision"],
        error=(
            None if status in {"published", "absent", "preserved"} else entry["detail"]
        ),
    )


def get_fetch_publication_status(
    *, sqlite_path: Path, duckdb_path: Path
) -> AnalyticsRebuildReport:
    """Read the combined project-pair plus POD gate from committed SQLite truth only.

    POD freshness alone never proves the whole fetch is fresh: the pair and the POD
    projection must both agree with the current committed generation before the gate
    reports freshness.
    """
    from esdc.pod_registry.publication import get_pod_registry_status

    sqlite_path = Path(sqlite_path)
    duckdb_path = Path(duckdb_path)
    # Bracket the committed generation so a change between the two reads cannot combine
    # a pair read from one generation with a POD read from another.
    generation_before = committed_pod_source_generation(sqlite_path)
    api_status = analytics_rebuild.get_api_analytics_status(
        duckdb_path=duckdb_path, sqlite_path=sqlite_path
    )
    registry = get_pod_registry_status(sqlite_path, duckdb_path)
    generation_after = committed_pod_source_generation(sqlite_path)
    counts = _projection_counts(duckdb_path)
    components = [
        _pair_component(table, api_status[table], counts.get(table, 0))
        for table in ("project_resources", "project_timeseries")
    ]
    if generation_before is None and _pod_source_absent(sqlite_path):
        pod_status: ComponentStatus = "absent"
    elif registry.state == "fresh":
        pod_status = "published"
    elif registry.state == "stale":
        pod_status = "stale"
    elif registry.state == "unpublished":
        pod_status = "pending"
    else:
        pod_status = "error"
    if pod_status == "published" and registry.published_generation != generation_before:
        # A mirror marker from another generation is not this fetch's truth.
        pod_status = "stale"
    components.append(
        ComponentPublication(
            name="pod_projection",
            status=pod_status,
            rows_written=0,
            revision=None,
            generation=generation_before,
            error=None if pod_status in {"published", "absent"} else registry.message,
        )
    )
    if generation_after != generation_before:
        detail = (
            f"generation changed from {generation_before} to {generation_after} during "
            "the gate read; the components were measured on different bases"
        )
        components = [
            dataclasses.replace(component, status="stale", error=detail)
            for component in components
        ]
        pod_status = "stale"
        fresh = False
    else:
        fresh = all(component.status == "published" for component in components)
    return AnalyticsRebuildReport(
        output_path=duckdb_path,
        source_revisions={
            component.name: component.revision for component in components
        },
        source_hashes={},
        subsystem_status={
            "api": (
                "restored"
                if all(component.status == "published" for component in components[:2])
                else "absent"
            ),
            "pod_registry": "restored" if pod_status == "published" else "absent",
        },
        object_counts={
            component.name: component.rows_written for component in components
        },
        requires_regeneration=(),
        components=tuple(components),
        fresh=fresh,
        recovery_command=(
            RECOVERY_COMMAND
            if any(part.status in {"pending", "stale"} for part in components)
            or any(
                api_status[table]["snapshot"] is not None
                and components[index].status != "published"
                for index, table in enumerate(
                    ("project_resources", "project_timeseries")
                )
            )
            else None
        ),
    )


def _has_project_observation(sqlite_path: Path) -> bool:
    """Distinguish a POD workbook source from an observed raw pair, including empty."""
    connection = sqlite3.connect(sqlite_path.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        connection.execute("BEGIN")
        if (
            connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='api_snapshot'"
            ).fetchone()
            is None
        ):
            return False
        return (
            connection.execute(
                "SELECT 1 FROM api_snapshot WHERE source_table IN "
                "('project_resources', 'project_timeseries') LIMIT 1"
            ).fetchone()
            is not None
        )
    finally:
        connection.close()


def _publish_pod_only(
    source_path: Path, target_path: Path, generation: int
) -> AnalyticsRebuildReport:
    """Recover a workbook-only source without manufacturing raw observations."""
    from esdc.pod_registry.publication import get_pod_registry_status
    from esdc.pod_registry.publish import publish_pod_registry

    report = publish_pod_registry(source_path, target_path)
    if report.projection_error is not None:
        raise RuntimeError(report.projection_error)
    try:
        current = committed_pod_source_generation(source_path)
        status = get_pod_registry_status(source_path, target_path)
        if (
            current != generation
            or status.state != "fresh"
            or status.published_generation != generation
        ):
            raise RuntimeError(
                "POD source changed during publication; retry committed source"
            )
    except Exception as exc:
        raise FetchFailure(
            f"POD mirror committed, but publication verification failed: {exc}",
            phase="post_commit",
            table="pod_registry",
            sqlite_committed=True,
            duckdb_committed=True,
            recovery_command=RECOVERY_COMMAND,
        ) from exc
    counts = {result.table_name: result.row_count for result in report}
    return AnalyticsRebuildReport(
        output_path=target_path,
        source_revisions={},
        source_hashes={"pod_projection": report.source_sha256},
        subsystem_status={"api": "absent", "pod_registry": "restored"},
        object_counts=counts,
        requires_regeneration=(),
        components=(
            ComponentPublication(
                "pod_projection",
                "published",
                sum(counts.values()),
                generation=generation,
            ),
        ),
        fresh=True,
    )


def publish_committed_pod_sources(
    sqlite_path: Path,
    duckdb_path: Path,
    *,
    expected_generation: int | None = None,
    reindex: bool = True,
) -> AnalyticsRebuildReport:
    """Publish committed raw observations and POD, or a workbook-only POD source.

    ``expected_generation`` is the committed generation the caller believes exists; a
    mismatch is refused before any write, so a retry can only republish what SQLite
    already committed.
    """
    source_path = Path(sqlite_path)
    target_path = Path(duckdb_path)
    committed = committed_pod_source_generation(source_path)
    if expected_generation is not None and committed != expected_generation:
        raise FetchFailure(
            f"POD source generation {committed} does not match the expected committed "
            f"generation {expected_generation}; nothing was published",
            phase="preflight",
            table="pod_registry",
            sqlite_committed=True,
            recovery_command=RECOVERY_COMMAND,
        )

    try:
        if committed is not None and not _has_project_observation(source_path):
            return _publish_pod_only(source_path, target_path, committed)
        return analytics_rebuild.rebuild_api_analytics(
            duckdb_path=target_path,
            sqlite_path=source_path,
            reindex=reindex,
            expected_generation=committed,
        )
    except FetchFailure:
        # The publisher already reported accurate phase flags for its own work.
        raise
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException as exc:
        raise FetchFailure(
            f"POD publication failed: {exc}",
            phase="publication",
            table="pod_registry",
            sqlite_committed=True,
            duckdb_committed=False,
            recovery_command=RECOVERY_COMMAND,
        ) from exc
