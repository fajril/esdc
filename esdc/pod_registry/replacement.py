"""Read-only approval and optimistic checks for one-time POD replacement."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Mapping
from pathlib import Path

from esdc.pod_registry.api_context import (
    PodImportError,
    object_type,
    quote_identifier,
    read_only_connection,
)
from esdc.pod_registry.migrations import (
    _LEGACY_OBJECTS,
    _legacy_document_links,
    _legacy_master_rows,
    _table_names,
    inspect_migration,
)
from esdc.pod_registry.models import PodReplacementPlan, PreparedPodBatch

_RAW_BASIS_TABLES = (
    "api_project_resources",
    "api_project_timeseries",
    "api_snapshot",
    "api_snapshot_provenance",
)


def replacement_fingerprint(conn: sqlite3.Connection) -> str:
    """Hash old schema, POD facts, references, protected links and raw source basis."""
    digest = hashlib.sha256()

    def add(values: tuple[object, ...]) -> None:
        typed = [
            [type(value).__name__, value.hex() if isinstance(value, bytes) else value]
            for value in values
        ]
        digest.update(
            json.dumps(typed, ensure_ascii=False, separators=(",", ":")).encode()
            + b"\n"
        )

    for row in conn.execute(
        "SELECT type,name,tbl_name,sql FROM sqlite_master ORDER BY type,name"
    ):
        add(tuple(row))
    names = _table_names(conn)
    for name in sorted(names & {*_LEGACY_OBJECTS, *_RAW_BASIS_TABLES}):
        columns = [
            str(row[1])
            for row in conn.execute(f"PRAGMA table_info({quote_identifier(name)})")
        ]
        order = ",".join(quote_identifier(column) for column in columns)
        cursor = conn.execute(
            f"SELECT * FROM {quote_identifier(name)} ORDER BY {order}"
        )
        while rows := cursor.fetchmany(512):
            for row in rows:
                add((name, *tuple(row)))
    return digest.hexdigest()


def capture_replacement_plan(
    path: Path,
    approved_crosswalk: Mapping[int, int] | None,
    *,
    profile: str,
) -> PodReplacementPlan | None:
    """Inspect full-import replacement; ordinary readers remain current-only."""
    if not path.is_file():
        if approved_crosswalk is not None:
            raise PodImportError(
                ["--pod-crosswalk requires an existing old POD target"]
            )
        return None
    conn = read_only_connection(path)
    try:
        if (
            object_type(conn, "pod_registry_state") is not None
            or object_type(conn, "m_pod") != "table"
        ):
            if approved_crosswalk is not None:
                raise PodImportError(
                    ["--pod-crosswalk applies only to old POD replacement"]
                )
            return None
        report = inspect_migration(conn)
        if report.status != "legacy":
            if approved_crosswalk is not None:
                detail = "; ".join(report.blockers) or report.status
                raise PodImportError([f"POD replacement refused: {detail}"])
            return None
        if profile != "full":
            raise PodImportError(
                [
                    "Old POD data requires a full --from-sqlite or --schema-pod load; "
                    "master-only and scoped fetch cannot replace it."
                ]
            )
        mapping = dict(approved_crosswalk or {})
        if any(
            isinstance(old, bool)
            or not isinstance(old, int)
            or isinstance(new, bool)
            or not isinstance(new, int)
            for old, new in mapping.items()
        ):
            raise PodImportError(["POD crosswalk must map integer IDs to integer IDs"])
        names = _table_names(conn)
        masters = _legacy_master_rows(conn, names)
        links = _legacy_document_links(conn, names)
        missing = {old for old, _ in links} - mapping.keys()
        if missing:
            raise PodImportError(
                [
                    "Protected document links require --pod-crosswalk; "
                    "missing old POD IDs: " + ", ".join(map(str, sorted(missing)))
                ]
            )
        unknown = mapping.keys() - masters.keys()
        if unknown:
            raise PodImportError(
                [f"Crosswalk contains unknown old POD IDs: {sorted(unknown)}"]
            )
        mapped = [(mapping[old], doc_id) for old, doc_id in links]
        if len(mapped) != len(set(mapped)):
            raise PodImportError(["Crosswalk would collapse duplicate document links"])
        return PodReplacementPlan(
            replacement_fingerprint(conn), tuple(sorted(mapping.items()))
        )
    except sqlite3.Error as exc:
        raise PodImportError(
            [f"Could not inspect POD replacement target: {exc}"]
        ) from exc
    finally:
        conn.close()


def validate_replacement_batch(batch: PreparedPodBatch) -> None:
    """Validate every approved target against the incoming current master."""
    plan = batch.replacement_plan
    if plan is None:
        return
    if batch.profile != "full":
        raise PodImportError(["POD replacement requires the full import profile"])
    # Incoming rows are typed as PreparedScalar, so narrow to the integer IDs the
    # crosswalk can target. A non-int scalar can never equal an int target, so this
    # changes no behaviour — it only makes the two sets the same type.
    targets = {
        pod_id
        for row in batch.rows["m_pod"]
        if isinstance(pod_id := row["pod_id_seq"], int)
    }
    missing = {new for _, new in plan.approved_crosswalk} - targets
    if missing:
        raise PodImportError(
            [f"Crosswalk targets are absent from incoming m_pod: {sorted(missing)}"]
        )


def verify_replacement_target(
    conn: sqlite3.Connection, plan: PodReplacementPlan
) -> None:
    """Recheck the captured target under the same write lock that owns replacement."""
    if replacement_fingerprint(conn) != plan.target_sha256:
        raise PodImportError(
            ["Operational POD target changed after import preparation; prepare again"]
        )
    report = inspect_migration(conn)
    if report.status != "legacy":
        raise PodImportError(
            ["Operational POD target is no longer the prepared old schema"]
        )
