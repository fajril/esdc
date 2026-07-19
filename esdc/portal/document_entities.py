"""Validate-then-apply changesets for the documents entity-column grid.

Covers the `documents.wk_name/field_name/project_name` columns edited in
the portal. Unlike `esdc.pod_registry.changesets` (which owns the POD registry
tables), `documents` is populated by the corpus ingest pipeline, not the
portal grid — so inserts/deletes stay rejected here and only the three
entity columns are editable. SQLite (`esdc.pod_registry.store`) is the
source of truth; the DuckDB `documents` mirror (`esdc.corpus.store`,
same file as `Config.get_db_file()`) is best-effort — a mirror failure
is reported as a warning, never rolls back the SQLite commit.

Name validation mirrors `_validate_entity_overrides` in
`esdc.corpus.pipeline`: each cell is `;`-separated raw names, each name
is resolved against `EntityResolver.resolve_name`, and 0-match names are
rejected with `suggest_names` candidates.
"""

from __future__ import annotations

import contextlib
import json
import sqlite3
from pathlib import Path
from typing import Any

from esdc.pod_registry.changesets import ChangesetResult, RowError
from esdc.pod_registry.store import get_sqlite_connection

ENTITY_FIELDS: tuple[str, ...] = ("wk_name", "field_name", "project_name")


def apply_document_entity_changeset(
    changes: dict,
    sqlite_path: Path | None = None,
    db_path: Path | None = None,
    resolver: Any | None = None,
) -> ChangesetResult:
    """Validate then apply entity-column edits to `documents`.

    `changes` is the grid payload `{"updates": [...], "inserts": [...],
    "deletes": [...]}`. Only `updates` are ever allowed; `inserts` and
    `deletes` are rejected outright since `documents` is ingest-only.
    Each update row may carry `doc_id` plus any subset of
    `ENTITY_FIELDS`; cell values are `;`-separated raw names, resolved
    against `resolver` and stored as a JSON array (or NULL when empty).

    Validation runs entirely before any SQLite write — a resolver that
    can't be reached, an unknown doc_id, or an unresolvable name all
    reject the whole changeset with no partial writes. Once the SQLite
    transaction commits, mirroring the same UPDATEs into the DuckDB
    `documents` table is best-effort: any failure is appended to
    `result.warnings` rather than raised.
    """
    inserts = changes.get("inserts") or []
    updates = changes.get("updates") or []
    deletes = changes.get("deletes") or []

    errors: list[RowError] = []
    for i, _row in enumerate(inserts):
        errors.append(
            RowError(
                "insert", i, "documents is ingest-only; inserts are not allowed here"
            )
        )
    for i, _row in enumerate(deletes):
        errors.append(
            RowError(
                "delete", i, "documents is ingest-only; deletes are not allowed here"
            )
        )

    # Structural validation (required/unknown fields) happens before any
    # resolver or DB access — a malformed payload should never touch them.
    parsed_updates: list[dict[str, Any]] = []
    for i, row in enumerate(updates):
        if "doc_id" not in row:
            errors.append(RowError("update", i, "missing required field: doc_id"))
            continue
        extra = set(row.keys()) - {"doc_id"} - set(ENTITY_FIELDS)
        if extra:
            errors.append(
                RowError(
                    "update", i, f"unknown field(s): {', '.join(sorted(extra))}"
                )
            )
            continue
        fields = {k: v for k, v in row.items() if k in ENTITY_FIELDS}
        parsed_updates.append({"index": i, "doc_id": row["doc_id"], "fields": fields})

    if errors:
        return ChangesetResult(ok=False, errors=errors)

    if not parsed_updates:
        return ChangesetResult(
            ok=True, applied={"inserts": 0, "updates": 0, "deletes": 0}
        )

    # The default resolver holds its own read-only DuckDB connection to the
    # SAME file the mirror later opens read-write; DuckDB refuses to open a
    # file with two different configurations at once, so the resolver
    # connection MUST be closed (see the finally below) before
    # _mirror_updates runs.
    resolver_conn = None
    if resolver is None:
        resolver, resolver_conn = _build_resolver(db_path)
        if resolver is None:
            return ChangesetResult(
                ok=False,
                errors=[
                    RowError(
                        "resolver",
                        -1,
                        "entity resolver unavailable — corpus DuckDB is locked "
                        "or missing; cannot validate names",
                    )
                ],
            )

    sconn = get_sqlite_connection(sqlite_path)
    try:
        resolved_rows: list[dict[str, Any]] = []
        try:
            for item in parsed_updates:
                i = item["index"]
                doc_id = item["doc_id"]
                existing = sconn.execute(
                    "SELECT 1 FROM documents WHERE doc_id = ?", (doc_id,)
                ).fetchone()
                if existing is None:
                    errors.append(
                        RowError("update", i, f"doc_id {doc_id!r} does not exist")
                    )
                    continue

                resolved_fields: dict[str, list[str] | None] = {}
                for field, raw_value in item["fields"].items():
                    resolved_fields[field] = _resolve_field(
                        resolver, field, raw_value, i, errors
                    )
                if resolved_fields:
                    resolved_rows.append({"doc_id": doc_id, "fields": resolved_fields})
        finally:
            if resolver_conn is not None:
                with contextlib.suppress(Exception):
                    resolver_conn.close()

        if errors:
            return ChangesetResult(ok=False, errors=errors)

        with sconn:
            for row in resolved_rows:
                _apply_row_update(sconn, row["doc_id"], row["fields"])
    finally:
        sconn.close()

    warnings = _mirror_updates(db_path, resolved_rows) if resolved_rows else []

    return ChangesetResult(
        ok=True,
        applied={"inserts": 0, "updates": len(resolved_rows), "deletes": 0},
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Cell parsing + name resolution
# ---------------------------------------------------------------------------


def _parse_names(raw: Any) -> list[str]:
    """Split a grid cell into stripped, non-empty names. None/"" -> []."""
    if raw is None:
        return []
    return [part.strip() for part in str(raw).split(";") if part.strip()]


def _resolve_field(
    resolver: Any,
    field: str,
    raw_value: Any,
    row_index: int,
    errors: list[RowError],
) -> list[str] | None:
    """Resolve one cell's `;`-separated names to canonical form, or None.

    Appends a RowError per unresolvable name instead of raising, so a
    single bad name doesn't stop the rest of the changeset from being
    validated (all errors surface together).
    """
    names = _parse_names(raw_value)
    if not names:
        return None

    resolved: list[str] = []
    for name in names:
        matches = resolver.resolve_name(name, field)
        if not matches:
            suggestions = resolver.suggest_names(name, field)
            if suggestions:
                listed = ", ".join(f"'{s}'" for s in suggestions)
                message = (
                    f"{field} '{name}' not found in database; "
                    f"closest matches: {listed}"
                )
            else:
                message = f"{field} '{name}' not found in database and no close matches"
            errors.append(RowError("update", row_index, message))
        elif len(matches) == 1:
            resolved.append(matches[0]["name"])
        else:
            # Ambiguous (>1 match) — keep the user's raw value untouched.
            resolved.append(name)
    return resolved


def _build_resolver(db_path: Path | None) -> tuple[Any | None, Any | None]:
    """Lazily construct an EntityResolver plus its owned DuckDB connection.

    Returns (resolver, connection) — the caller MUST close the connection
    once validation is done (before the read-write mirror open), or
    (None, None) when the corpus DuckDB can't be opened.
    """
    from esdc.chat.domain_knowledge.entity_resolver_lib import EntityResolver
    from esdc.configs import Config
    from esdc.dbmanager import get_duckdb_connection

    path = db_path or Config.get_db_file()
    try:
        conn = get_duckdb_connection(path, read_only=True)
    except Exception:
        return None, None
    return EntityResolver(conn), conn


# ---------------------------------------------------------------------------
# Apply + mirror
# ---------------------------------------------------------------------------


def _apply_row_update(
    sconn: sqlite3.Connection, doc_id: str, fields: dict[str, list[str] | None]
) -> None:
    set_clause = ", ".join(f"{k} = ?" for k in fields)
    values = [json.dumps(v) if v is not None else None for v in fields.values()]
    sconn.execute(
        f"UPDATE documents SET {set_clause} WHERE doc_id = ?", (*values, doc_id)
    )


def _mirror_updates(
    db_path: Path | None, resolved_rows: list[dict[str, Any]]
) -> list[str]:
    """Best-effort mirror of the same UPDATEs into the DuckDB documents table.

    SQLite already committed by the time this runs, so any failure here
    (missing table, locked file, ...) is reported as a warning, never
    raised — the save itself already succeeded.
    """
    from esdc.configs import Config
    from esdc.dbmanager import get_duckdb_connection

    path = db_path or Config.get_db_file()
    try:
        conn = get_duckdb_connection(path, read_only=False)
    except Exception as exc:
        return [f"DuckDB mirror unavailable: {exc}"]

    warnings: list[str] = []
    try:
        for row in resolved_rows:
            fields = row["fields"]
            set_clause = ", ".join(f"{k} = ?" for k in fields)
            values = [json.dumps(v) if v is not None else None for v in fields.values()]
            try:
                conn.execute(
                    f"UPDATE documents SET {set_clause} WHERE doc_id = ?",
                    (*values, row["doc_id"]),
                )
            except Exception as exc:
                warnings.append(
                    f"DuckDB mirror failed for doc_id {row['doc_id']!r}: {exc}"
                )
    finally:
        conn.close()
    return warnings
