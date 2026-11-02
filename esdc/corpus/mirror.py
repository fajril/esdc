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

The POD registry is deliberately NOT raw-mirrored here. The SQLite
operational tables are source truth; their DuckDB shapes are produced by the
canonical publisher reached through `esdc/pod_registry/publish.py`, which
publishes the current contract tables (`m_project`, `m_pod`, `pod_project`,
`pod_monitoring`, `pod_forecast` and the two API output tables) plus the
r_refs seed and technical state tables, and exposes `pod_registry` as the
documents-facing view of `m_pod` holding the canonical string identifier
(`PL-YYYY-XXXX-A-B-R`) documented to the chat agent. Published shapes and
their documented columns come from `esdc/schemas/pod.yaml`; the document
link table `pod_document` keys its join on `pod_id_seq`, the surrogate
identity key that contract defines. A raw copy of the SQLite tables under
those same names would silently overwrite the published, agent-documented
shapes with internal source shapes, so `refresh_registry` never copies them.
Retirement and replacement of POD-owned objects is the publisher's, inside
its own transaction. `refresh_all` checks the source against the contract
read-only before it mutates anything in DuckDB, so a missing, empty, corrupt,
or shape-incompatible source is refused while the projection the mirror
already had is still intact.
Objects from the superseded POD model (`pod_economics`, `pod_plan`,
`pod_project_economics`, the revision registry, and the value-case table)
are retired by the publisher and are not recreated on refresh.

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


def refresh_documents(conn: duckdb.DuckDBPyConnection, sqlite_path: Path) -> int:
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
        row = conn.execute(
            f"SELECT "
            f"SUM(CASE WHEN t.doc_date IS NOT NULL AND d.doc_date IS NULL "
            f"THEN 1 ELSE 0 END), "
            f"SUM(CASE WHEN t.ingested_at IS NOT NULL AND d.ingested_at IS NULL "
            f"THEN 1 ELSE 0 END) "
            f"FROM {truth}.{DOC_TABLE} t JOIN {DOC_TABLE} d ON d.doc_id = t.doc_id"
        ).fetchone()
        if row is None:
            raise RuntimeError("mirror query returned no row")
        bad_doc_date, bad_ingested_at = row[0], row[1]
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
    count_row = conn.execute(f"SELECT COUNT(*) FROM {DOC_TABLE}").fetchone()
    if count_row is None:
        raise RuntimeError("mirror query returned no row")
    count = count_row[0]
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

# SQLite operational tables this module must never raw-mirror: the canonical
# publisher owns their DuckDB shapes (see the module docstring), so a raw copy
# under the same names would clobber the published, agent-documented ones. They
# are not copied here and they are not dropped here either — retirement is the
# publisher's inside its own transaction, which is what keeps a source the
# publisher cannot validate from destroying a projection it can no longer
# rebuild. `pod_document`, `m_pod`, `r_institution`, `r_pod_type` and the
# superseded-model names are listed in `esdc.pod_registry.publication` as the
# objects that publisher replaces.


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
    intentional, not an oversight. The POD registry tables are excluded for a
    different reason — see the module docstring — and they are neither copied
    nor dropped here: retiring and replacing them is the canonical publisher's
    job inside its own transaction, so a source the publisher cannot validate
    cannot cost the mirror a projection it already had.

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
    losing its truth) is worth an info line.
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
        # form as create_views uses for the main catalog.
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
            copied_row = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
            if copied_row is None:
                raise RuntimeError("mirror query returned no row")
            copied[table] = copied_row[0]
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

    Presence is measured in the catalog being written to: an attached
    database cannot satisfy the link join here, because the view's
    unqualified names resolve in the current catalog. A pair that exists
    only there, or one half here and one half attached, must not count.
    """
    created: list[str] = []
    has_links_row = conn.execute(
        "SELECT COUNT(DISTINCT table_name) FROM information_schema.tables "
        "WHERE table_catalog = current_database() AND table_schema='main' "
        "AND table_name IN ('pod_document', 'pod_registry')"
    ).fetchone()
    if has_links_row is None:
        raise RuntimeError("mirror query returned no row")
    # Both published POD tables must exist; a lone pod_document without
    # pod_registry (or vice versa) would make the link view's JOIN dangle.
    has_links = has_links_row[0] == 2

    if has_links:
        # The published shapes only: `pod_document` links on the contract's
        # `pod_id_seq`, and `pod_registry` is the view that carries the
        # canonical string `pod_id`. There is no alternate join for an older
        # column shape — a half-published or superseded-shape registry is
        # refused by the publisher, so it cannot reach here.
        conn.execute("""
            CREATE OR REPLACE VIEW v_doc_pod_link AS
            SELECT d.doc_id, d.file_name, d.doc_type, d.doc_date, d.subject,
                   p.pod_id, p.pod_name
            FROM documents d
            JOIN pod_document pd ON pd.doc_id = d.doc_id
            JOIN pod_registry p ON p.pod_id_seq = pd.pod_id_seq
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
        # may have just been retired (the publisher drops and recreates the
        # POD objects inside its own transaction). Leaving the view in place
        # would make it reference a table that no longer exists, turning any
        # query against it into a CatalogException instead of returning stale
        # rows.
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
    # kg_edge/kg_claim (raw-mirrored, see REGISTRY_TABLES) plus the canonical
    # POD publication (contract tables, r_refs seed, and technical state, see
    # esdc.pod_registry.publish). One flat dict rather than a separate field
    # per source — `esdc corpus sync` just enumerates it — so the CLI output
    # stays honest about everything refresh_all touched without the report
    # shape caring which module produced which table.
    registry: dict[str, int] = field(default_factory=dict)
    views: list[str] = field(default_factory=list)


def refresh_all(conn: duckdb.DuckDBPyConnection, sqlite_path: Path) -> MirrorReport:
    """Rebuild every derived table/view in DuckDB from the SQLite truth.

    Converges the raw mirrors (`documents`, `kg_edge`/`kg_claim`) and the
    canonical POD publication — the contract tables, the r_refs seed, and the
    technical state tables that `publish_pod_registry` produces, including the
    `pod_registry` view and the `pod_document` link keyed on `pod_id_seq`.
    Objects from the superseded POD model are retired by the publisher and are
    not recreated here. One `esdc corpus sync` therefore repairs the whole read
    side, including a database left raw-mirrored by an older build: the
    publisher replaces those raw copies with the current shapes.

    The source is checked against the contract read-only before anything in
    DuckDB is mutated. A missing, empty, corrupt, or shape-incompatible source
    is refused at that point, so the documents mirror, the knowledge-graph
    mirror, and the POD projection a previous successful refresh published all
    stay exactly as they were. Without that ordering a refused source could
    destroy a projection it can no longer rebuild, because the publisher's own
    transaction is the only place POD objects are dropped and replaced.

    The POD publish runs on this same `conn` rather than opening a second write
    connection: DuckDB allows only one writer per file, and an in-memory handle
    passed in by the caller cannot be re-opened as a writer at all. Publish is
    the only POD write path here.

    A publication failure is checked on `projection_error`, not on report
    truthiness: the report is a sequence and an empty one is falsy.
    """
    from esdc.pod_registry.publication import publish_v2_registry
    from esdc.pod_registry.publish import source_preflight_error

    preflight_error = source_preflight_error(sqlite_path)
    if preflight_error is not None:
        raise RuntimeError(preflight_error)

    report = MirrorReport()
    report.documents = refresh_documents(conn, sqlite_path)
    report.orphan_chunks = sweep_orphan_chunks(conn)
    report.registry = refresh_registry(conn, sqlite_path)

    db_row = conn.execute(
        "SELECT path FROM duckdb_databases() "
        "WHERE database_name = current_database()"
    ).fetchone()
    db_path = Path(db_row[0]) if db_row and db_row[0] else Path(":memory:")
    publication = publish_v2_registry(sqlite_path, db_path, duck_conn=conn)
    if publication.projection_error is not None:
        raise RuntimeError(publication.projection_error)
    report.registry.update(
        {result.table_name: result.row_count for result in publication.results}
    )

    from esdc.knowledge.freshness import knowledge_freshness
    from esdc.pod_registry.store import get_sqlite_readonly_connection

    source = get_sqlite_readonly_connection(sqlite_path)
    if source is not None:
        try:
            stale_knowledge = not knowledge_freshness(source).fresh
        finally:
            source.close()
        if stale_knowledge:
            mirrored = {
                row[0]
                for row in conn.execute(
                    "SELECT table_name FROM duckdb_tables() "
                    "WHERE database_name = current_database()"
                ).fetchall()
            }
            for table in ("kg_edge", "kg_claim"):
                if table in mirrored:
                    conn.execute(f'DELETE FROM "{table}"')
    report.views = create_views(conn)
    return report
