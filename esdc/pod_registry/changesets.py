"""Validate-then-apply changesets against the POD registry SQLite db."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from esdc.pod_registry.store import allocate_pod_id, get_sqlite_connection

_TABLES = ("m_pod", "project_pod", "pod_revision", "r_institution", "r_pod_type")
_M_POD_REQUIRED = (
    "id",
    "pod_name",
    "approval_date",
    "institution_code",
    "pod_type_code",
    "rev_num",
)
_M_POD_UPDATABLE = {"pod_name", "pod_letter_num"}
_M_POD_OPTIONAL = {"pod_letter_num"}


@dataclass
class RowError:
    kind: str
    index: int
    message: str


@dataclass
class ChangesetResult:
    ok: bool
    applied: dict[str, int] = field(default_factory=dict)
    generated: list[dict] = field(default_factory=list)
    errors: list[RowError] = field(default_factory=list)


def apply_changeset(
    table: str,
    changes: dict,
    sqlite_path: Path | None = None,
    known_project_ids: set[str] | None = None,
) -> ChangesetResult:
    if table not in _TABLES:
        raise ValueError(f"unknown table: {table}")
    inserts = changes.get("inserts") or []
    updates = changes.get("updates") or []
    deletes = changes.get("deletes") or []

    conn = get_sqlite_connection(sqlite_path)
    try:
        errors = _validate(conn, table, inserts, updates, deletes, known_project_ids)
        if errors:
            return ChangesetResult(ok=False, errors=errors)
        generated: list[dict] = []
        try:
            with conn:
                _apply_deletes(conn, table, deletes)
                _apply_updates(conn, table, updates)
                generated = _apply_inserts(conn, table, inserts)
        except sqlite3.IntegrityError as exc:
            # Second line of defense: the with-block above has already rolled
            # back, so map the constraint violation to a row error.
            return ChangesetResult(
                ok=False,
                errors=[RowError("apply", -1, f"constraint violation: {exc}")],
            )
        return ChangesetResult(
            ok=True,
            applied={
                "inserts": len(inserts),
                "updates": len(updates),
                "deletes": len(deletes),
            },
            generated=generated,
        )
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Validation dispatch
# ---------------------------------------------------------------------------


def _validate(
    conn: sqlite3.Connection,
    table: str,
    inserts: list[dict],
    updates: list[dict],
    deletes: list[dict],
    known_project_ids: set[str] | None,
) -> list[RowError]:
    if table == "m_pod":
        return _validate_m_pod(conn, inserts, updates, deletes)
    if table == "project_pod":
        return _validate_project_pod(conn, inserts, updates, deletes, known_project_ids)
    if table == "pod_revision":
        return _validate_pod_revision(conn, inserts, updates, deletes)
    if table in ("r_institution", "r_pod_type"):
        return _validate_r_table(conn, table, inserts, updates, deletes)
    raise ValueError(f"unknown table: {table}")


# ---------------------------------------------------------------------------
# m_pod
# ---------------------------------------------------------------------------


def _validate_m_pod(
    conn: sqlite3.Connection,
    inserts: list[dict],
    updates: list[dict],
    deletes: list[dict],
) -> list[RowError]:
    errors: list[RowError] = []
    seen_ids_in_changeset: set[int] = set()

    for i, row in enumerate(inserts):
        missing = [k for k in _M_POD_REQUIRED if k not in row]
        if missing:
            errors.append(
                RowError(
                    "insert", i, f"missing required field(s): {', '.join(missing)}"
                )
            )
            continue
        extra = set(row.keys()) - set(_M_POD_REQUIRED) - _M_POD_OPTIONAL
        if extra:
            errors.append(
                RowError("insert", i, f"unknown field(s): {', '.join(sorted(extra))}")
            )

        row_id = row["id"]
        if not isinstance(row_id, int):
            errors.append(RowError("insert", i, "id must be an integer"))
        else:
            if row_id in seen_ids_in_changeset:
                errors.append(
                    RowError("insert", i, f"duplicate id {row_id} within changeset")
                )
            seen_ids_in_changeset.add(row_id)
            existing = conn.execute(
                "SELECT 1 FROM m_pod WHERE id = ?", (row_id,)
            ).fetchone()
            if existing is not None:
                errors.append(RowError("insert", i, f"id {row_id} already exists"))

        inst_code = row.get("institution_code")
        if (
            conn.execute(
                "SELECT 1 FROM r_institution WHERE code = ?", (inst_code,)
            ).fetchone()
            is None
        ):
            errors.append(
                RowError("insert", i, f"institution_code {inst_code} does not exist")
            )

        type_code = row.get("pod_type_code")
        if (
            conn.execute(
                "SELECT 1 FROM r_pod_type WHERE code = ?", (type_code,)
            ).fetchone()
            is None
        ):
            errors.append(
                RowError("insert", i, f"pod_type_code {type_code} does not exist")
            )

        rev_num = row.get("rev_num")
        if not isinstance(rev_num, int) or isinstance(rev_num, bool) or rev_num < 0:
            errors.append(RowError("insert", i, "rev_num must be an integer >= 0"))

        approval_date = row.get("approval_date")
        try:
            date.fromisoformat(str(approval_date))
        except (ValueError, TypeError):
            errors.append(
                RowError(
                    "insert",
                    i,
                    f"approval_date {approval_date!r} is not a valid ISO date",
                )
            )

    for i, row in enumerate(updates):
        if "id" not in row:
            errors.append(RowError("update", i, "missing required field: id"))
            continue
        row_id = row["id"]
        existing = conn.execute(
            "SELECT 1 FROM m_pod WHERE id = ?", (row_id,)
        ).fetchone()
        if existing is None:
            errors.append(RowError("update", i, f"id {row_id} does not exist"))
            continue
        extra = set(row.keys()) - {"id"} - _M_POD_UPDATABLE
        if extra:
            errors.append(
                RowError(
                    "update",
                    i,
                    "structural fields are locked once pod_id is issued: "
                    f"{', '.join(sorted(extra))}",
                )
            )

    for i, row in enumerate(deletes):
        if "id" not in row:
            errors.append(RowError("delete", i, "missing required field: id"))
            continue
        row_id = row["id"]
        existing = conn.execute(
            "SELECT 1 FROM m_pod WHERE id = ?", (row_id,)
        ).fetchone()
        if existing is None:
            errors.append(RowError("delete", i, f"id {row_id} does not exist"))
            continue
        ref_count = conn.execute(
            "SELECT COUNT(*) FROM project_pod WHERE pod_id = ?", (row_id,)
        ).fetchone()[0]
        if ref_count:
            errors.append(
                RowError("delete", i, f"id {row_id} is referenced by project_pod")
            )
            continue
        pod_id_row = conn.execute(
            "SELECT pod_id FROM m_pod WHERE id = ?", (row_id,)
        ).fetchone()
        pod_id = pod_id_row[0] if pod_id_row else None
        if pod_id is not None:
            rev_count = conn.execute(
                "SELECT COUNT(*) FROM pod_revision"
                " WHERE successor_id = ? OR predecessor_id = ?",
                (pod_id, pod_id),
            ).fetchone()[0]
            if rev_count:
                errors.append(
                    RowError("delete", i, f"id {row_id} is referenced by pod_revision")
                )

    return errors


def _apply_inserts(
    conn: sqlite3.Connection, table: str, inserts: list[dict]
) -> list[dict]:
    if table == "m_pod":
        return _apply_m_pod_inserts(conn, inserts)
    if table == "project_pod":
        for row in inserts:
            conn.execute(
                "INSERT INTO project_pod (pod_id, project_id) VALUES (?, ?)",
                (row["pod_id"], row["project_id"]),
            )
        return []
    if table == "pod_revision":
        for row in inserts:
            conn.execute(
                "INSERT INTO pod_revision (successor_id, predecessor_id) VALUES (?, ?)",
                (row["successor_id"], row["predecessor_id"]),
            )
        return []
    if table in ("r_institution", "r_pod_type"):
        col = "institution" if table == "r_institution" else "pod_type"
        for row in inserts:
            conn.execute(
                f"INSERT INTO {table} (code, {col}, description) VALUES (?, ?, ?)",
                (row["code"], row["name"], row.get("description")),
            )
        return []
    return []


def _apply_m_pod_inserts(conn: sqlite3.Connection, inserts: list[dict]) -> list[dict]:
    generated: list[dict] = []
    for i, row in enumerate(inserts):
        pod_id, approval_seq = allocate_pod_id(
            conn,
            row["approval_date"],
            row["institution_code"],
            row["pod_type_code"],
            row["rev_num"],
        )
        conn.execute(
            "INSERT INTO m_pod (id, pod_id, pod_name, pod_letter_num, approval_date,"
            " institution_code, pod_type_code, rev_num, approval_seq)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                row["id"],
                pod_id,
                row["pod_name"],
                row.get("pod_letter_num"),
                row["approval_date"],
                row["institution_code"],
                row["pod_type_code"],
                row["rev_num"],
                approval_seq,
            ),
        )
        generated.append(
            {
                "index": i,
                "id": row["id"],
                "pod_id": pod_id,
                "approval_seq": approval_seq,
            }
        )
    return generated


def _apply_updates(conn: sqlite3.Connection, table: str, updates: list[dict]) -> None:
    if table == "m_pod":
        for row in updates:
            fields = {k: v for k, v in row.items() if k != "id"}
            if not fields:
                continue
            set_clause = ", ".join(f"{k} = ?" for k in fields)
            conn.execute(
                f"UPDATE m_pod SET {set_clause} WHERE id = ?",
                (*fields.values(), row["id"]),
            )
    elif table in ("r_institution", "r_pod_type"):
        col = "institution" if table == "r_institution" else "pod_type"
        for row in updates:
            fields = {}
            if "name" in row:
                fields[col] = row["name"]
            if "description" in row:
                fields["description"] = row["description"]
            if not fields:
                continue
            set_clause = ", ".join(f"{k} = ?" for k in fields)
            conn.execute(
                f"UPDATE {table} SET {set_clause} WHERE code = ?",
                (*fields.values(), row["code"]),
            )
    # project_pod and pod_revision have no updates (validated to be empty).


def _apply_deletes(conn: sqlite3.Connection, table: str, deletes: list[dict]) -> None:
    if table == "m_pod":
        for row in deletes:
            conn.execute("DELETE FROM m_pod WHERE id = ?", (row["id"],))
    elif table == "project_pod":
        for row in deletes:
            conn.execute(
                "DELETE FROM project_pod WHERE pod_id = ? AND project_id = ?",
                (row["pod_id"], row["project_id"]),
            )
    elif table == "pod_revision":
        for row in deletes:
            conn.execute(
                "DELETE FROM pod_revision"
                " WHERE successor_id = ? AND predecessor_id = ?",
                (row["successor_id"], row["predecessor_id"]),
            )
    elif table in ("r_institution", "r_pod_type"):
        for row in deletes:
            conn.execute(f"DELETE FROM {table} WHERE code = ?", (row["code"],))


# ---------------------------------------------------------------------------
# project_pod
# ---------------------------------------------------------------------------


def _validate_project_pod(
    conn: sqlite3.Connection,
    inserts: list[dict],
    updates: list[dict],
    deletes: list[dict],
    known_project_ids: set[str] | None,
) -> list[RowError]:
    errors: list[RowError] = []

    if updates:
        for i, _row in enumerate(updates):
            errors.append(
                RowError(
                    "update",
                    i,
                    "project_pod does not support updates; delete and insert instead",
                )
            )

    seen_pairs_in_changeset: set[tuple[int, str]] = set()
    for i, row in enumerate(inserts):
        pod_id = row.get("pod_id")
        project_id = row.get("project_id")
        if not isinstance(pod_id, int):
            errors.append(RowError("insert", i, "pod_id must be an integer"))
        else:
            exists = conn.execute(
                "SELECT 1 FROM m_pod WHERE id = ?", (pod_id,)
            ).fetchone()
            if exists is None:
                errors.append(
                    RowError("insert", i, f"pod_id {pod_id} does not exist in m_pod")
                )
        if not isinstance(project_id, str) or not project_id.strip():
            errors.append(
                RowError("insert", i, "project_id must be a non-empty string")
            )
        else:
            if known_project_ids is not None and project_id not in known_project_ids:
                errors.append(
                    RowError(
                        "insert",
                        i,
                        f"project_id {project_id!r} is not a known project id",
                    )
                )

        if isinstance(pod_id, int) and isinstance(project_id, str):
            pair = (pod_id, project_id)
            if pair in seen_pairs_in_changeset:
                errors.append(
                    RowError(
                        "insert",
                        i,
                        f"duplicate pair ({pod_id}, {project_id}) within changeset",
                    )
                )
            seen_pairs_in_changeset.add(pair)
            existing = conn.execute(
                "SELECT 1 FROM project_pod WHERE pod_id = ? AND project_id = ?",
                (pod_id, project_id),
            ).fetchone()
            if existing is not None:
                errors.append(
                    RowError(
                        "insert", i, f"pair ({pod_id}, {project_id}) already exists"
                    )
                )

    for i, row in enumerate(deletes):
        pod_id = row.get("pod_id")
        project_id = row.get("project_id")
        existing = conn.execute(
            "SELECT 1 FROM project_pod WHERE pod_id = ? AND project_id = ?",
            (pod_id, project_id),
        ).fetchone()
        if existing is None:
            errors.append(
                RowError("delete", i, f"pair ({pod_id}, {project_id}) does not exist")
            )

    return errors


# ---------------------------------------------------------------------------
# pod_revision
# ---------------------------------------------------------------------------


def _validate_pod_revision(
    conn: sqlite3.Connection,
    inserts: list[dict],
    updates: list[dict],
    deletes: list[dict],
) -> list[RowError]:
    errors: list[RowError] = []

    if updates:
        for i, _row in enumerate(updates):
            errors.append(
                RowError(
                    "update",
                    i,
                    "pod_revision does not support updates; delete and insert instead",
                )
            )

    seen_pairs_in_changeset: set[tuple[str, str]] = set()
    for i, row in enumerate(inserts):
        successor_id = row.get("successor_id")
        predecessor_id = row.get("predecessor_id")

        if successor_id is not None:
            exists = conn.execute(
                "SELECT 1 FROM m_pod WHERE pod_id = ?", (successor_id,)
            ).fetchone()
            if exists is None:
                errors.append(
                    RowError(
                        "insert",
                        i,
                        f"successor_id {successor_id} does not exist in m_pod",
                    )
                )
        else:
            errors.append(RowError("insert", i, "missing required field: successor_id"))

        if predecessor_id is not None:
            exists = conn.execute(
                "SELECT 1 FROM m_pod WHERE pod_id = ?", (predecessor_id,)
            ).fetchone()
            if exists is None:
                errors.append(
                    RowError(
                        "insert",
                        i,
                        f"predecessor_id {predecessor_id} does not exist in m_pod",
                    )
                )
        else:
            errors.append(
                RowError("insert", i, "missing required field: predecessor_id")
            )

        if (
            successor_id is not None
            and predecessor_id is not None
            and successor_id == predecessor_id
        ):
            errors.append(
                RowError(
                    "insert", i, "successor_id and predecessor_id must not be equal"
                )
            )

        if successor_id is not None and predecessor_id is not None:
            pair = (successor_id, predecessor_id)
            if pair in seen_pairs_in_changeset:
                errors.append(
                    RowError("insert", i, f"duplicate pair {pair} within changeset")
                )
            seen_pairs_in_changeset.add(pair)
            existing = conn.execute(
                "SELECT 1 FROM pod_revision"
                " WHERE successor_id = ? AND predecessor_id = ?",
                (successor_id, predecessor_id),
            ).fetchone()
            if existing is not None:
                errors.append(RowError("insert", i, f"pair {pair} already exists"))

    for i, row in enumerate(deletes):
        successor_id = row.get("successor_id")
        predecessor_id = row.get("predecessor_id")
        existing = conn.execute(
            "SELECT 1 FROM pod_revision WHERE successor_id = ? AND predecessor_id = ?",
            (successor_id, predecessor_id),
        ).fetchone()
        if existing is None:
            errors.append(
                RowError(
                    "delete",
                    i,
                    f"pair ({successor_id}, {predecessor_id}) does not exist",
                )
            )

    return errors


# ---------------------------------------------------------------------------
# r_institution / r_pod_type
# ---------------------------------------------------------------------------


def _validate_r_table(
    conn: sqlite3.Connection,
    table: str,
    inserts: list[dict],
    updates: list[dict],
    deletes: list[dict],
) -> list[RowError]:
    errors: list[RowError] = []

    seen_codes_in_changeset: set[int] = set()
    for i, row in enumerate(inserts):
        code = row.get("code")
        name = row.get("name")
        if not isinstance(code, int):
            errors.append(RowError("insert", i, "code must be an integer"))
        else:
            if code in seen_codes_in_changeset:
                errors.append(
                    RowError("insert", i, f"duplicate code {code} within changeset")
                )
            seen_codes_in_changeset.add(code)
            existing = conn.execute(
                f"SELECT 1 FROM {table} WHERE code = ?", (code,)
            ).fetchone()
            if existing is not None:
                errors.append(RowError("insert", i, f"code {code} already exists"))
        if not isinstance(name, str) or not name.strip():
            errors.append(RowError("insert", i, "name must be a non-empty string"))

    for i, row in enumerate(updates):
        code = row.get("code")
        existing = conn.execute(
            f"SELECT 1 FROM {table} WHERE code = ?", (code,)
        ).fetchone()
        if existing is None:
            errors.append(RowError("update", i, f"code {code} does not exist"))
            continue
        if "name" in row and (
            not isinstance(row["name"], str) or not row["name"].strip()
        ):
            errors.append(RowError("update", i, "name must be a non-empty string"))

    for i, row in enumerate(deletes):
        code = row.get("code")
        existing = conn.execute(
            f"SELECT 1 FROM {table} WHERE code = ?", (code,)
        ).fetchone()
        if existing is None:
            errors.append(RowError("delete", i, f"code {code} does not exist"))
            continue
        fk_col = "institution_code" if table == "r_institution" else "pod_type_code"
        ref_count = conn.execute(
            f"SELECT COUNT(*) FROM m_pod WHERE {fk_col} = ?", (code,)
        ).fetchone()[0]
        if ref_count:
            errors.append(RowError("delete", i, f"code {code} is referenced by m_pod"))

    return errors
