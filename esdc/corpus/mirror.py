"""Wholesale SQLite -> DuckDB mirror refresh.

The DuckDB copies of `documents` and the POD registry are derived data:
they are rebuilt from the SQLite truth in one statement rather than
maintained row-by-row. A rebuild either ran or it did not, so there is no
partially-diverged state to detect or repair.

Beyond `documents`, this module also mirrors the POD registry and
knowledge-graph tables (see `REGISTRY_TABLES`) so that `execute_sql`,
which runs against DuckDB, can reach POD/project/institution lookups and
the learned `kg_edge`/`kg_claim` facts without a cross-database query.
`refresh_registry` copies each of those tables that exists in the truth
and skips those that don't; see its own docstring for which tables are
deliberately excluded from mirroring altogether.

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

    Uses a correlated NOT EXISTS rather than NOT IN: SQL's NOT IN against
    a subquery containing any NULL evaluates to UNKNOWN for every row, so
    the DELETE would silently match nothing. SQLite permits NULL in a
    non-INTEGER PRIMARY KEY column, so `documents.doc_id` being declared
    TEXT PRIMARY KEY does not rule this out. NOT EXISTS has no such
    landmine.
    """
    try:
        deleted = conn.execute(
            f"DELETE FROM {CHUNK_TABLE} c WHERE NOT EXISTS "
            f"(SELECT 1 FROM {DOC_TABLE} d WHERE d.doc_id = c.doc_id) "
            f"RETURNING c.chunk_id"
        ).fetchall()
    except duckdb.CatalogException:
        return 0  # chunks table not created yet (fresh install)
    if deleted:
        logger.info("[Mirror] orphan chunks removed | count=%d", len(deleted))
    return len(deleted)


# Relational tables that live only in SQLite. Mirroring them verbatim is
# what lets execute_sql (which runs on DuckDB) reach the POD registry and
# the learned knowledge graph at all. They are small — hundreds to low
# thousands of rows — so a wholesale copy is cheaper than any sync.
REGISTRY_TABLES = (
    "r_institution",
    "r_pod_type",
    "m_pod",
    "project_pod",
    "pod_document",
    "pod_revision",
    "kg_edge",
    "kg_claim",
)


def refresh_registry(
    conn: duckdb.DuckDBPyConnection, sqlite_path: Path
) -> dict[str, int]:
    """Copy each registry/knowledge table that exists in the truth.

    Mirrors the POD registry (`r_institution`, `r_pod_type`, `m_pod`,
    `project_pod`, `pod_document`, `pod_revision`) and the learned
    knowledge-graph facts (`kg_edge`, `kg_claim`) so `execute_sql` can
    query them from DuckDB. `kg_proposal` (the human curation queue
    behind `esdc corpus proposals`) and `learn_state` (per-document
    idempotency bookkeeping for `esdc corpus learn`) are deliberately
    left out of `REGISTRY_TABLES` — neither is domain knowledge the chat
    agent should query, so their absence here is intentional, not an
    oversight.

    kg_edge and kg_claim only exist after `esdc corpus learn` has run, so
    an absent table is skipped rather than raising.
    """
    copied: dict[str, int] = {}
    with attached_truth(conn, sqlite_path) as truth:
        present = {
            r[0]
            for r in conn.execute(
                "SELECT table_name FROM duckdb_tables() WHERE database_name = ?",
                [truth],
            ).fetchall()
        }
        for table in REGISTRY_TABLES:
            if table not in present:
                logger.debug("[Mirror] skipping absent table | table=%s", table)
                continue
            conn.execute(
                f"CREATE OR REPLACE TABLE {table} AS SELECT * FROM {truth}.{table}"
            )
            copied[table] = conn.execute(
                f"SELECT COUNT(*) FROM {table}"
            ).fetchone()[0]
    logger.info("[Mirror] registry refreshed | tables=%d", len(copied))
    return copied


def create_views(conn: duckdb.DuckDBPyConnection) -> list[str]:
    """Create the denormalized document/POD views.

    Two grains, both named explicitly:

    - `v_doc_pod_link` — one row per (document, POD). Convenient for
      joins, but COUNT(*) over it counts LINKS, not documents.
    - `v_document` — one row per document, POD links aggregated into
      `linked_pod_ids`. This is the view to count.

    Returns the view names created. If the registry tables are absent
    (learn never ran) only `v_document` is created, with an empty
    `linked_pod_ids`.
    """
    created: list[str] = []
    has_links = bool(
        conn.execute(
            "SELECT COUNT(*) FROM duckdb_tables() "
            "WHERE table_name IN ('pod_document', 'm_pod') "
            "AND database_name = current_database()"
        ).fetchone()[0]
        == 2
    )

    if has_links:
        conn.execute("""
            CREATE OR REPLACE VIEW v_doc_pod_link AS
            SELECT d.doc_id, d.file_name, d.doc_type, d.doc_date, d.subject,
                   p.pod_id, p.pod_name
            FROM documents d
            JOIN pod_document pd ON pd.doc_id = d.doc_id
            JOIN m_pod p ON p.id = pd.pod_id
        """)
        created.append("v_doc_pod_link")
        conn.execute("""
            CREATE OR REPLACE VIEW v_document AS
            SELECT d.*, COALESCE(l.linked_pod_ids, []) AS linked_pod_ids
            FROM documents d
            LEFT JOIN (
                SELECT doc_id, list(pod_id) AS linked_pod_ids
                FROM v_doc_pod_link GROUP BY doc_id
            ) l ON l.doc_id = d.doc_id
        """)
    else:
        conn.execute("""
            CREATE OR REPLACE VIEW v_document AS
            SELECT d.*, CAST([] AS VARCHAR[]) AS linked_pod_ids FROM documents d
        """)
    created.append("v_document")
    return created
