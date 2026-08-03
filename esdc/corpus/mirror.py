"""Wholesale SQLite -> DuckDB mirror refresh.

The DuckDB copies of `documents` and the POD registry are derived data:
they are rebuilt from the SQLite truth in one statement rather than
maintained row-by-row. A rebuild either ran or it did not, so there is no
partially-diverged state to detect or repair.

`document_chunks` is NOT derived from SQLite — its embeddings exist only
in DuckDB — so refresh never rebuilds it and may only delete orphans.

SQLite is dynamically typed. Copying with a bare `SELECT *` silently
downgrades `doc_date` to VARCHAR, which breaks `EXTRACT(year FROM ...)`
in CorpusStore._build_filter_clause. Every type-bearing column is cast
explicitly below; do not simplify this away.

The `doc_date`/`ingested_at` casts use TRY_CAST rather than CAST: one
malformed value (a hand-typed, non-ISO `doc_date` anywhere in the truth
table) must not raise `ConversionException` and abort the whole
`CREATE OR REPLACE TABLE`, which would leave callers treating a hard
refresh failure as a mere warning and the mirror silently stale.
TRY_CAST converts the bad value to NULL and lets the rebuild succeed;
`refresh_documents` then counts and logs any such rows so the bad data
stays visible instead of silently disappearing.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Iterator
from pathlib import Path

import duckdb

logger = logging.getLogger(__name__)

DOC_TABLE = "documents"
CHUNK_TABLE = "document_chunks"

# documents columns holding JSON arrays/objects in the DuckDB mirror.
_JSON_COLUMNS = (
    "doc_topic",
    "wk_name",
    "field_name",
    "project_name",
    "pod_name",
    "suggested_pod_ids",
    "raw_entities",
    "metadata",
)


def _json_cast(col: str) -> str:
    """Cast a SQLite TEXT column to JSON, wrapping legacy bare strings.

    Pre-JSON rows stored entity names as plain text ('Rokan'). A direct
    CAST would produce invalid JSON and break json_each(), so those are
    wrapped into a one-element array — the same repair
    CorpusStore._migrate_legacy_entity_columns performs in place.
    """
    return (
        f"CASE WHEN {col} IS NULL THEN NULL "
        f"WHEN json_valid({col}) THEN CAST({col} AS JSON) "
        f"ELSE CAST(to_json([{col}]) AS JSON) END AS {col}"
    )


DOC_CAST_REPLACE = ",\n    ".join(
    [
        "TRY_CAST(doc_date AS DATE) AS doc_date",
        "TRY_CAST(ingested_at AS TIMESTAMP) AS ingested_at",
        *[_json_cast(c) for c in _JSON_COLUMNS],
    ]
)

_DOC_INDEXES = (
    ("idx_doc_doc_type", "doc_type"),
    ("idx_doc_doc_topic", "doc_topic"),
    ("idx_doc_doc_level", "doc_level"),
    ("idx_doc_wk_name", "wk_name"),
    ("idx_doc_field_name", "field_name"),
    ("idx_doc_project_name", "project_name"),
    ("idx_doc_doc_date", "doc_date"),
)


@contextlib.contextmanager
def attached_truth(
    conn: duckdb.DuckDBPyConnection, sqlite_path: Path, alias: str = "truth"
) -> Iterator[str]:
    """ATTACH the SQLite truth read-only for the duration of the block."""
    with contextlib.suppress(Exception):
        conn.execute("INSTALL sqlite")
    conn.execute("LOAD sqlite")
    conn.execute(f"ATTACH '{sqlite_path}' AS {alias} (TYPE sqlite, READ_ONLY)")
    try:
        yield alias
    finally:
        with contextlib.suppress(Exception):
            conn.execute(f"DETACH {alias}")


def refresh_documents(
    conn: duckdb.DuckDBPyConnection, sqlite_path: Path
) -> int:
    """Rebuild the DuckDB `documents` mirror from the SQLite truth.

    Returns the number of rows copied. CREATE OR REPLACE drops the
    table's B-tree indexes, so they are recreated afterwards.
    """
    with attached_truth(conn, sqlite_path) as truth:
        conn.execute(
            f"CREATE OR REPLACE TABLE {DOC_TABLE} AS "
            f"SELECT * REPLACE (\n    {DOC_CAST_REPLACE}\n) "
            f"FROM {truth}.{DOC_TABLE}"
        )
        # TRY_CAST turns a malformed doc_date/ingested_at into NULL instead
        # of failing the whole rebuild (see module docstring). That must
        # not go unnoticed, so count — in one query, joined on doc_id —
        # rows where the source had a value but the cast produced NULL.
        bad_doc_date, bad_ingested_at = conn.execute(
            f"SELECT "
            f"SUM(CASE WHEN t.doc_date IS NOT NULL AND d.doc_date IS NULL "
            f"THEN 1 ELSE 0 END), "
            f"SUM(CASE WHEN t.ingested_at IS NOT NULL AND d.ingested_at IS NULL "
            f"THEN 1 ELSE 0 END) "
            f"FROM {truth}.{DOC_TABLE} t JOIN {DOC_TABLE} d ON d.doc_id = t.doc_id"
        ).fetchone()
    for column, bad_count in (
        ("doc_date", bad_doc_date),
        ("ingested_at", bad_ingested_at),
    ):
        if bad_count:
            logger.warning(
                "[Mirror] %s failed TRY_CAST and was set NULL | count=%d",
                column,
                bad_count,
            )
    for idx_name, column in _DOC_INDEXES:
        try:
            conn.execute(
                f"CREATE INDEX IF NOT EXISTS {idx_name} ON {DOC_TABLE}({column})"
            )
        except Exception as e:  # index failure must not fail the refresh
            logger.warning("[Mirror] index %s failed: %s", idx_name, e)
    count = conn.execute(f"SELECT COUNT(*) FROM {DOC_TABLE}").fetchone()[0]
    logger.info("[Mirror] documents refreshed | rows=%d", count)
    return count


def sweep_orphan_chunks(conn: duckdb.DuckDBPyConnection) -> int:
    """Delete chunks whose document no longer exists in the mirror.

    `document_chunks` holds embeddings that exist only in DuckDB and are
    never rebuilt from SQLite. When a document is deleted from the truth
    it vanishes from `documents` at the next refresh, but its chunks
    would remain searchable — phantom hits pointing at a doc_id that can
    no longer be read. This sweep deletes exactly those.
    """
    try:
        deleted = conn.execute(
            f"DELETE FROM {CHUNK_TABLE} WHERE doc_id NOT IN "
            f"(SELECT doc_id FROM {DOC_TABLE}) RETURNING chunk_id"
        ).fetchall()
    except duckdb.CatalogException:
        return 0  # chunks table not created yet (fresh install)
    if deleted:
        logger.info("[Mirror] orphan chunks removed | count=%d", len(deleted))
    return len(deleted)
