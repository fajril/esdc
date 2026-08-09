# esdc/pod_registry/publish.py
"""Snapshot the SQLite POD registry into DuckDB for iris text-to-SQL."""

from __future__ import annotations

import contextlib
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pandas as pd
import yaml

from esdc.configs import Config
from esdc.dbmanager import get_duckdb_connection
from esdc.loaders import (
    _METADATA_TABLE,
    LoadResult,
    _create_metadata_table,
    _load_pod_domain_schema,
)
from esdc.pod_registry.store import ESDC_SQLITE_FILENAME, get_sqlite_connection
from esdc.pod_registry.value_cases import _VALUE_CASE_IDENTIFIERS

_REGISTRY_SQL = """
SELECT
    m.id,
    m.pod_id,
    m.pod_name,
    m.pod_letter_num,
    m.approval_date,
    ri.institution,
    rt.pod_type,
    m.rev_num,
    m.approval_seq,
    (SELECT group_concat(predecessor_id, '; ') FROM pod_revision
      WHERE successor_id = m.pod_id) AS preceded_by,
    (SELECT group_concat(successor_id, '; ') FROM pod_revision
      WHERE predecessor_id = m.pod_id) AS revised_by,
    (SELECT group_concat(successor_id, '; ') FROM pod_revision
      WHERE predecessor_id = m.pod_id
        AND revision_effect = 'full_replacement') AS superseded_by
FROM m_pod m
JOIN r_institution ri ON ri.code = m.institution_code
JOIN r_pod_type rt ON rt.code = m.pod_type_code
"""

_PROJECT_SQL = """
SELECT m.pod_id, pp.project_id
FROM project_pod pp
JOIN m_pod m ON m.id = pp.pod_id
"""

_DOCUMENT_SQL = """
SELECT m.pod_id, pd.doc_id
FROM pod_document pd
JOIN m_pod m ON m.id = pd.pod_id
"""

_REVISION_SQL = """
SELECT
    m.pod_id AS successor_id,
    p.pod_id AS predecessor_id,
    r.revision_effect,
    r.effective_date,
    r.amended_scope,
    r.previous_remains_valid
FROM pod_revision r
JOIN m_pod m ON m.pod_id = r.successor_id
JOIN m_pod p ON p.pod_id = r.predecessor_id
"""

_VALUE_CASE_SQL = """
SELECT * FROM pod_value_case
"""

# Registry metadata columns joined onto pod_economics from pod_registry.
_ECONOMICS_META_COLUMNS = (
    "pod_name",
    "pod_letter_num",
    "institution",
    "pod_type",
    "approval_date",
    "rev_num",
    "approval_seq",
    "preceded_by",
    "revised_by",
    "superseded_by",
)


def _read_value_cases(sconn: sqlite3.Connection) -> pd.DataFrame:
    """Read pod_value_case; a truth never seeded (no table) yields an empty frame.

    When the table exists the read is performed directly, without a broad
    try/except, so a malformed table fails loudly instead of being masked
    as an empty payload.
    """
    exists = sconn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='pod_value_case'"
    ).fetchone()
    if exists is None:
        schema = _load_pod_domain_schema()
        return pd.DataFrame(columns=[column.name for column in schema.columns])
    df = pd.read_sql_query(_VALUE_CASE_SQL, sconn)
    metric_columns = [
        column.name
        for column in _load_pod_domain_schema().columns
        if column.name not in _VALUE_CASE_IDENTIFIERS
    ]
    for column in metric_columns:
        if column not in df.columns:
            df[column] = None
    return df


def _pod_view_schema_entries() -> list[tuple[str, str, dict]]:
    """Build `_loaded_table_schemas` entries for the economics views.

    Columns come from the harmonized contract: the value-case payload
    (pod_schema.yaml) followed by the registry metadata (pod_registry_schema.yaml).
    """
    vc_schema = _load_pod_domain_schema()
    vc_by_name = {column["name"]: dict(column) for column in vc_schema.raw["columns"]}
    schemas = _load_schema_entries()
    reg_by_name = {
        column["name"]: dict(column) for column in schemas["pod_registry"]["columns"]
    }
    project_id_col = next(
        column
        for column in schemas["pod_project"]["columns"]
        if column["name"] == "project_id"
    )

    metric_names = tuple(
        column.name
        for column in vc_schema.columns
        if column.name not in _VALUE_CASE_IDENTIFIERS
    )
    ordered = _VALUE_CASE_IDENTIFIERS + metric_names + _ECONOMICS_META_COLUMNS
    economics_columns = [
        reg_by_name[name] if name in reg_by_name else vc_by_name[name]
        for name in ordered
    ]

    def entry(name: str, description: str, columns: list[dict]) -> dict:
        return {
            "table_name": name,
            "description": description,
            "sheet_name": ESDC_SQLITE_FILENAME,
            "columns": columns,
        }

    economics = entry(
        "pod_economics",
        "Unified POD value cases (plan, actual, outlook) joined with canonical"
        " POD registry metadata: identity, approval info, and revision lineage.",
        economics_columns,
    )
    plan = entry(
        "pod_plan",
        "POD plan value cases (case_type='plan').",
        economics_columns,
    )
    monitoring = entry(
        "pod_monitoring",
        "POD monitoring value cases (case_type IN ('actual','outlook')).",
        economics_columns,
    )
    project_economics = entry(
        "pod_project_economics",
        "POD value cases joined through canonical pod_project to expose project_id.",
        economics_columns + [dict(project_id_col)],
    )
    return [
        (item["table_name"], item["description"], item)
        for item in (economics, plan, monitoring, project_economics)
    ]


def _refresh_view_schema_metadata(conn: duckdb.DuckDBPyConnection) -> None:
    """Refresh `_loaded_table_schemas` rows for the economics views."""
    loaded_at = datetime.now(timezone.utc).isoformat()
    for table_name, description, entry in _pod_view_schema_entries():
        conn.execute(
            f"DELETE FROM {_METADATA_TABLE} WHERE table_name = ?", [table_name]
        )
        conn.execute(
            f"""
            INSERT INTO {_METADATA_TABLE}
                (table_name, description, sheet_name, schema_yaml, loaded_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            [
                table_name,
                description,
                ESDC_SQLITE_FILENAME,
                yaml.safe_dump(entry, sort_keys=False, allow_unicode=True),
                loaded_at,
            ],
        )


def _create_pod_views(conn: duckdb.DuckDBPyConnection) -> None:
    """Rebuild the POD economics compatibility views from the published tables.

    `pod_economics` joins the authoritative `pod_value_case` rows with
    `pod_registry` metadata; `pod_plan` / `pod_monitoring` are thin
    case_type filters over it; `pod_project_economics` joins through the
    canonical `pod_project`. Stale legacy base tables left by the old
    workbook loader (`pod_plan`/`pod_monitoring` as tables) are removed
    first so the view names do not collide with them. `as_of_date` is the
    only period column; the ambiguous analytical `effective_date` is not
    reintroduced.
    """
    for table in ("pod_plan", "pod_monitoring"):
        # A stale legacy base table OR an economics view from a previous
        # publish can occupy the name. DuckDB's DROP ... IF EXISTS does not
        # suppress a type mismatch (dropping a table when a view exists, and
        # vice versa, still raises), so branch on the actual catalog object.
        row = conn.execute(
            "SELECT COUNT(*) FROM duckdb_views()"
            " WHERE view_name = ? AND database_name = current_database()",
            [table],
        ).fetchone()
        is_view = bool(row and row[0] > 0)
        if is_view:
            conn.execute(f"DROP VIEW IF EXISTS {table}")
        else:
            conn.execute(f"DROP TABLE IF EXISTS {table}")

    vc_schema = _load_pod_domain_schema()
    metric_names = tuple(
        column.name
        for column in vc_schema.columns
        if column.name not in _VALUE_CASE_IDENTIFIERS
    )
    select_columns = (
        [f"vc.{name}" for name in _VALUE_CASE_IDENTIFIERS]
        + [f"vc.{name}" for name in metric_names]
        + [f"r.{name}" for name in _ECONOMICS_META_COLUMNS]
    )
    columns_sql = ",\n    ".join(select_columns)
    conn.execute(f"""
        CREATE OR REPLACE VIEW pod_economics AS
        SELECT
            {columns_sql}
        FROM pod_value_case vc
        JOIN pod_registry r ON r.pod_id = vc.pod_id
    """)
    conn.execute(
        "CREATE OR REPLACE VIEW pod_plan AS"
        " SELECT * FROM pod_economics WHERE case_type = 'plan'"
    )
    conn.execute(
        "CREATE OR REPLACE VIEW pod_monitoring AS"
        " SELECT * FROM pod_economics"
        " WHERE case_type IN ('actual', 'outlook')"
    )
    conn.execute("""
        CREATE OR REPLACE VIEW pod_project_economics AS
        SELECT pe.*, pp.project_id
        FROM pod_economics pe
        JOIN pod_project pp ON pp.pod_id = pe.pod_id
    """)
    _refresh_view_schema_metadata(conn)


def _load_schema_entries() -> dict[str, dict]:
    schema_path = (
        Path(__file__).resolve().parent.parent
        / "chat"
        / "domain_knowledge"
        / "pod_registry_schema.yaml"
    )
    data = yaml.safe_load(schema_path.read_text(encoding="utf-8"))
    entries = {entry["table_name"]: entry for entry in data["tables"]}
    value_case = dict(_load_pod_domain_schema().raw)
    value_case["sheet_name"] = ESDC_SQLITE_FILENAME
    entries["pod_value_case"] = value_case
    return entries


def _publish_registry_tables(
    conn: duckdb.DuckDBPyConnection,
    sqlite_path: Path | None,
    db_path: Path | None = None,
) -> tuple[LoadResult, ...]:
    """Write the POD read projections onto an existing DuckDB conn.

    Emits pod_registry/pod_project/pod_document/pod_revision/pod_value_case
    plus the POD economics views.
    """
    sconn = get_sqlite_connection(sqlite_path)
    try:
        registry_df = pd.read_sql_query(_REGISTRY_SQL, sconn)
        project_df = pd.read_sql_query(_PROJECT_SQL, sconn)
        document_df = pd.read_sql_query(_DOCUMENT_SQL, sconn)
        revision_df = pd.read_sql_query(_REVISION_SQL, sconn)
        value_case_df = _read_value_cases(sconn)
    finally:
        sconn.close()

    if db_path is None:
        row = conn.execute(
            "SELECT path FROM duckdb_databases() "
            "WHERE database_name = current_database()"
        ).fetchone()
        db_path = Path(row[0]) if row and row[0] else Path()

    schemas = _load_schema_entries()
    results: list[LoadResult] = []
    conn.execute("BEGIN")
    try:
        _create_metadata_table(conn)
        for table_name, df, date_cols, boolean_cols in (
            ("pod_registry", registry_df, ("approval_date",), ()),
            ("pod_project", project_df, (), ()),
            ("pod_document", document_df, (), ()),
            (
                "pod_revision",
                revision_df,
                ("effective_date",),
                ("previous_remains_valid",),
            ),
            (
                "pod_value_case",
                value_case_df,
                ("report_date", "as_of_date"),
                (),
            ),
        ):
            conn.register("_pod_pub_df", df)
            select_cols = ", ".join(
                (
                    f"CAST({c} AS DATE) AS {c}"
                    if c in date_cols
                    else f"CAST({c} AS BOOLEAN) AS {c}"
                    if c in boolean_cols
                    else c
                )
                for c in df.columns
            )
            conn.execute(f"DROP TABLE IF EXISTS {table_name}")
            conn.execute(
                f"CREATE TABLE {table_name} AS SELECT {select_cols} FROM _pod_pub_df"
            )
            with contextlib.suppress(Exception):
                conn.unregister("_pod_pub_df")
            entry = schemas[table_name]
            conn.execute(
                f"DELETE FROM {_METADATA_TABLE} WHERE table_name = ?", [table_name]
            )
            conn.execute(
                f"""
                INSERT INTO {_METADATA_TABLE}
                    (table_name, description, sheet_name, schema_yaml, loaded_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                [
                    table_name,
                    entry["description"],
                    ESDC_SQLITE_FILENAME,
                    yaml.safe_dump(entry, sort_keys=False, allow_unicode=True),
                    datetime.now(timezone.utc).isoformat(),
                ],
            )
            results.append(
                LoadResult(
                    table_name=table_name,
                    row_count=len(df),
                    column_count=len(df.columns),
                    db_path=db_path,
                    link_warnings=(),
                )
            )
        _create_pod_views(conn)
        conn.execute("COMMIT")
    except Exception:
        with contextlib.suppress(Exception):
            conn.execute("ROLLBACK")
        raise
    return tuple(results)


def publish_pod_registry(
    sqlite_path: Path | None = None, db_path: Path | None = None
) -> tuple[LoadResult, ...]:
    """Replace the DuckDB POD read projections with the current SQLite state.

    Rebuilds pod_registry/pod_project/pod_revision/pod_value_case and the
    POD economics views in one replace-mode transaction.
    """
    target = db_path or Config.get_db_file()
    Config.get_db_dir().mkdir(parents=True, exist_ok=True)
    conn = get_duckdb_connection(target, read_only=False)
    try:
        results = _publish_registry_tables(conn, sqlite_path, target)
        conn.execute("CHECKPOINT")
    finally:
        conn.close()

    from esdc.chat.tools import invalidate_tool_cache, reset_sql_cache

    reset_sql_cache()
    invalidate_tool_cache()
    return results
