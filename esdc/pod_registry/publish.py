# esdc/pod_registry/publish.py
"""Snapshot the SQLite POD registry into DuckDB for iris text-to-SQL."""

from __future__ import annotations

import contextlib
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

from esdc.configs import Config
from esdc.dbmanager import get_duckdb_connection
from esdc.loaders import _METADATA_TABLE, LoadResult, _create_metadata_table
from esdc.pod_registry.store import ESDC_SQLITE_FILENAME, get_sqlite_connection

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
      WHERE predecessor_id = m.pod_id) AS superseded_by
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


def _load_schema_entries() -> dict[str, dict]:
    schema_path = (
        Path(__file__).resolve().parent.parent
        / "chat" / "domain_knowledge" / "pod_registry_schema.yaml"
    )
    data = yaml.safe_load(schema_path.read_text(encoding="utf-8"))
    return {entry["table_name"]: entry for entry in data["tables"]}


def publish_pod_registry(
    sqlite_path: Path | None = None, db_path: Path | None = None
) -> tuple[LoadResult, ...]:
    """Replace DuckDB pod_registry/pod_project with the current SQLite state."""
    sconn = get_sqlite_connection(sqlite_path)
    try:
        registry_df = pd.read_sql_query(_REGISTRY_SQL, sconn)
        project_df = pd.read_sql_query(_PROJECT_SQL, sconn)
        document_df = pd.read_sql_query(_DOCUMENT_SQL, sconn)
    finally:
        sconn.close()

    schemas = _load_schema_entries()
    target = db_path or Config.get_db_file()
    Config.get_db_dir().mkdir(parents=True, exist_ok=True)
    conn = get_duckdb_connection(target, read_only=False)
    results: list[LoadResult] = []
    try:
        conn.execute("BEGIN")
        _create_metadata_table(conn)
        for table_name, df, date_cols in (
            ("pod_registry", registry_df, ("approval_date",)),
            ("pod_project", project_df, ()),
            ("pod_document", document_df, ()),
        ):
            conn.register("_pod_pub_df", df)
            select_cols = ", ".join(
                f"CAST({c} AS DATE) AS {c}" if c in date_cols else c
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
                    db_path=target,
                    link_warnings=(),
                )
            )
        conn.execute("COMMIT")
        conn.execute("CHECKPOINT")
    except Exception:
        with contextlib.suppress(Exception):
            conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()

    from esdc.chat.tools import invalidate_tool_cache, reset_sql_cache

    reset_sql_cache()
    invalidate_tool_cache()
    return tuple(results)
