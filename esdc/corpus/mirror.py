"""Wholesale SQLite -> DuckDB mirror refresh.

The DuckDB copies of `documents` and the POD registry are derived data:
they are rebuilt from the SQLite truth in one statement rather than
maintained row-by-row. A rebuild either ran or it did not, so there is no
partially-diverged state to detect or repair.

Beyond `documents`, this module also mirrors the learned knowledge-graph
tables (see `REGISTRY_TABLES`) so that `execute_sql`, which runs against
DuckDB, can reach `kg_edge`/`kg_claim` without a cross-database query.
`refresh_registry` copies each of those tables that exists in the truth.
A table absent from the truth is not merely skipped: its DuckDB mirror,
if one exists from a previous refresh, is DROPped, so a table that
disappears from the truth (backup restore, truth file swap, KG state
reset) cannot leave stale rows mirrored in DuckDB forever — the
no-partially-diverged-state invariant above holds for the registry too.
See `refresh_registry`'s own docstring for which tables are deliberately
excluded from mirroring altogether.

The POD registry is deliberately NOT raw-mirrored here. `m_pod`,
`r_institution`, `r_pod_type`, `project_pod`, `pod_document`, and
`pod_revision` are normalized SQLite operational tables — `pod_id` on
the SQLite `pod_document` is `m_pod.id`, a surrogate BIGINT foreign key.
`esdc/pod_registry/publish.py` owns the read-side POD shapes: it
denormalizes them into `pod_registry` / `pod_project` / `pod_document`,
dissolves `r_institution`/`r_pod_type` into plain text columns, and
replaces every `pod_id` with the canonical string identifier
(`PL-YYYY-XXXX-A-B-R`) that `esdc/chat/domain_knowledge/
pod_registry_schema.yaml` documents to the chat agent. A raw copy of the
SQLite tables under those same names would silently overwrite the
published, agent-documented shapes with the internal surrogate-keyed
ones — `refresh_all` converges the published tables instead (see its
docstring) precisely to avoid that collision.

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
from dataclasses import dataclass, field
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
    # Load-bearing escape: ATTACH cannot take a `?` bind parameter for the
    # file path, so sqlite_path is the ONLY non-bound value in this
    # statement. It derives from Config.get_db_dir(), i.e. from a config
    # file, and an ordinary home directory can contain an apostrophe (e.g.
    # /Users/O'Brien/.esdc). Doubling single quotes is what keeps it a safe
    # SQL string literal — do not remove.
    escaped_path = str(sqlite_path).replace("'", "''")
    conn.execute(f"ATTACH '{escaped_path}' AS {alias} (TYPE sqlite, READ_ONLY)")
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
# what lets execute_sql (which runs on DuckDB) reach the learned
# knowledge graph. They are small — hundreds to low thousands of rows —
# so a wholesale copy is cheaper than any sync.
REGISTRY_TABLES = (
    "kg_edge",
    "kg_claim",
)

# POD tables an earlier build of this module raw-mirrored under the old,
# larger REGISTRY_TABLES. `esdc/pod_registry/publish.py` now owns all six
# shapes (see module docstring), so none of them belong here any more —
# but a DuckDB file refreshed by that older code can still be carrying
# them, most importantly a `pod_document` with a BIGINT `pod_id`
# clobbering the published, agent-documented VARCHAR one. `refresh_registry`
# retires any of these it finds using the same DROP TABLE IF EXISTS
# mechanism it already uses for a table that vanished from the truth, so
# the very next `esdc corpus sync` (or any commit/learn batch) repairs a
# database left in that state.
_RETIRED_REGISTRY_TABLES = (
    "r_institution",
    "r_pod_type",
    "m_pod",
    "project_pod",
    "pod_document",
    "pod_revision",
)


def refresh_registry(
    conn: duckdb.DuckDBPyConnection, sqlite_path: Path
) -> dict[str, int]:
    """Copy each knowledge-graph table that exists in the truth.

    Mirrors the learned knowledge-graph facts (`kg_edge`, `kg_claim`) so
    `execute_sql` can query them from DuckDB. `kg_proposal` (the human
    curation queue behind `esdc corpus proposals`) and `learn_state`
    (per-document idempotency bookkeeping for `esdc corpus learn`) are
    deliberately left out of `REGISTRY_TABLES` — neither is domain
    knowledge the chat agent should query, so their absence here is
    intentional, not an oversight. The POD registry tables are excluded
    for a different reason — see the module docstring and
    `_RETIRED_REGISTRY_TABLES` — and are actively retired below rather
    than just never copied.

    kg_edge and kg_claim only exist after `esdc corpus learn` has run, so
    an absent table does not raise. But absent is not the same as never
    mirrored: if a table that was previously copied later disappears from
    the truth (an older `esdc.sqlite` restored from backup, the truth
    file swapped, KG state reset), its DuckDB copy is DROPped, not left
    in place. Leaving it would mean `execute_sql` — which runs against
    DuckDB — keeps serving rows the truth no longer has, with nothing
    anywhere to warn that they are stale. The returned dict still omits
    the table's key either way: dropped and never-mirrored look the same
    to callers checking `table in copied`.

    The drop is only logged when a mirrored copy actually existed to
    remove. On most installs `kg_edge`/`kg_claim` have never been mirrored
    at all — `esdc corpus learn` has never run — and that steady state
    must not log anything above debug on every `corpus commit`/`learn`/
    `sync`; only a genuine state change (a table that DID exist in DuckDB
    losing its truth) is worth an info line. The same silence-unless-
    something-changed rule applies to retiring `_RETIRED_REGISTRY_TABLES`:
    once a database has been cleaned up once, every subsequent refresh is
    a silent no-op.
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
        # What DuckDB itself already has mirrored, so "absent from the
        # truth" can be split into two very different cases below: a stale
        # copy that must be dropped (and is worth logging) versus a table
        # that has simply never existed on either side (e.g. kg_edge/
        # kg_claim before `esdc corpus learn` has ever run), which is the
        # steady state on most installs and must stay silent. Same filter
        # form as create_views uses for the main catalog. Also doubles as
        # the retirement check below, since it reflects DuckDB's actual
        # catalog regardless of REGISTRY_TABLES membership.
        mirrored = {
            r[0]
            for r in conn.execute(
                "SELECT table_name FROM duckdb_tables() "
                "WHERE database_name = current_database()"
            ).fetchall()
        }
        for table in REGISTRY_TABLES:
            if table not in present:
                # Table names come only from the hardcoded REGISTRY_TABLES
                # tuple, never from caller input, so f-string interpolation
                # is safe here — same reasoning as the CREATE OR REPLACE
                # TABLE below. DROP (not skip) is what stops a table that
                # has vanished from the truth (backup restore, truth file
                # swap, KG state reset) from leaving stale rows mirrored in
                # DuckDB forever. Only do it — and only log it — when a
                # mirrored copy actually exists; otherwise this is just the
                # ordinary "never mirrored" case and produces no output.
                if table in mirrored:
                    conn.execute(f"DROP TABLE IF EXISTS {table}")
                    logger.info(
                        "[Mirror] dropped absent registry table | table=%s", table
                    )
                continue
            conn.execute(
                f"CREATE OR REPLACE TABLE {table} AS SELECT * FROM {truth}.{table}"
            )
            copied[table] = conn.execute(
                f"SELECT COUNT(*) FROM {table}"
            ).fetchone()[0]
        # Retire any POD table a pre-fix build left raw-mirrored here.
        # Unconditional on truth presence (unlike the loop above): these
        # tables are real SQLite operational tables and normally ARE
        # present in the truth, so "present in truth" cannot be the
        # signal to drop them — the signal is simply that this module no
        # longer wants to see them mirrored at all, regardless of truth
        # state. `publish_pod_registry`/`refresh_all` owns recreating
        # `pod_document` correctly immediately afterward.
        for table in _RETIRED_REGISTRY_TABLES:
            if table in mirrored:
                conn.execute(f"DROP TABLE IF EXISTS {table}")
                logger.info(
                    "[Mirror] dropped retired registry table | table=%s", table
                )
    logger.info("[Mirror] registry refreshed | tables=%d", len(copied))
    return copied


def create_views(conn: duckdb.DuckDBPyConnection) -> list[str]:
    """Create the denormalized document/POD views.

    Two grains, both named explicitly:

    - `v_doc_pod_link` — one row per (document, POD). Convenient for
      joins, but COUNT(*) over it counts LINKS, not documents.
    - `v_document` — one row per document, POD links aggregated into
      `linked_pod_ids`. This is the view to count.

    Returns the view names created. If the published POD tables are
    absent (`publish_pod_registry`/`refresh_all` has never run, or the
    truth has no POD data yet) only `v_document` is created, with an
    empty `linked_pod_ids`; any `v_doc_pod_link` left over from an
    earlier refresh is dropped so it cannot dangle and reference a table
    that no longer exists.
    """
    created: list[str] = []
    has_links = bool(
        conn.execute(
            "SELECT COUNT(*) FROM duckdb_tables() "
            "WHERE table_name IN ('pod_document', 'pod_registry') "
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
            JOIN pod_registry p ON p.pod_id = pd.pod_id
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
        # A registry table that backed v_doc_pod_link in a previous refresh
        # may have just been dropped (see refresh_registry). Leaving the
        # view in place would make it reference a table that no longer
        # exists, turning any query against it into a CatalogException
        # instead of returning stale rows.
        conn.execute("DROP VIEW IF EXISTS v_doc_pod_link")
        conn.execute("""
            CREATE OR REPLACE VIEW v_document AS
            SELECT d.*, CAST([] AS VARCHAR[]) AS linked_pod_ids FROM documents d
        """)
    created.append("v_document")
    return created


@dataclass
class MirrorReport:
    documents: int = 0
    orphan_chunks: int = 0
    # Every table this refresh actually converged in DuckDB, by row count:
    # kg_edge/kg_claim (raw-mirrored, see REGISTRY_TABLES) plus
    # pod_registry/pod_project/pod_document (published, see
    # esdc.pod_registry.publish). One flat dict rather than a separate
    # field per source — `esdc corpus sync` just enumerates it — so the
    # CLI output stays honest about everything refresh_all touched
    # without the report shape caring which module produced which table.
    registry: dict[str, int] = field(default_factory=dict)
    views: list[str] = field(default_factory=list)


def refresh_all(
    conn: duckdb.DuckDBPyConnection, sqlite_path: Path
) -> MirrorReport:
    """Rebuild every derived table/view in DuckDB from the SQLite truth.

    Converges both raw mirrors (`documents`, `kg_edge`/`kg_claim`) and
    the published POD registry (`pod_registry`, `pod_project`,
    `pod_document`) so `esdc corpus sync` — the only place a user runs
    this by hand — repairs the whole read side in one call, including
    the collision `refresh_registry`/`_RETIRED_REGISTRY_TABLES` retires
    (see their docstrings): a pre-fix build's raw `pod_document` mirror
    clobbering the published, agent-documented one.

    The POD publish runs on this same `conn` rather than opening a
    second write connection — see `_publish_registry_tables`'s docstring
    in `esdc/pod_registry/publish.py` for why a second connection is
    unsafe for the in-memory DuckDB handles this module's own tests use,
    even though it happens to be safe for on-disk ones.
    """
    from esdc.pod_registry.publish import _publish_registry_tables

    report = MirrorReport()
    report.documents = refresh_documents(conn, sqlite_path)
    report.orphan_chunks = sweep_orphan_chunks(conn)
    report.registry = refresh_registry(conn, sqlite_path)
    pod_results = _publish_registry_tables(conn, sqlite_path)
    report.registry.update({r.table_name: r.row_count for r in pod_results})
    report.views = create_views(conn)
    return report
