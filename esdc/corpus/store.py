"""Corpus store: SQLite source of truth + DuckDB derived mirror.

``documents`` (one row per source file, full markdown + metadata) lives
in the operational SQLite db (``esdc.sqlite``) — the source of truth for
everything a human writes or corrects. DuckDB holds derived data only:
``document_chunks`` (chunk + embedding, HNSW/FTS indexed), the 1-row
``corpus_meta``, and mirrors of ``documents`` plus the POD registry,
rebuilt wholesale by ``refresh_mirror()``.

Read-path routing rule — split by purpose, not by read/write:

* **Serving reads** (answering a user or agent: search, get_document,
  list_documents, find_doc_ids) run against DuckDB. They tolerate the
  refresh window.
* **Deciding reads** (whose result determines a mutation: document_exists
  for ingest dedupe, fingerprint_rows, get_document_by_hash) run against
  SQLite. A stale answer here would re-ingest or double-delete.
* **Truth-backed reads** (not deciding a mutation, but writing to disk
  what a user diffs: get_document_by_id, used by export's content
  fetch) also run against SQLite -- a stale mirror read here would
  overwrite a sidecar with wrong content.

Mutations write the SQLite truth; the DuckDB side is rebuilt by
``refresh_mirror()`` at the end of each batch (commit, learn, portal
save, ``esdc corpus sync``). There is no row-by-row mirroring and so no
drift to reconcile.

Incremental by design: dedupe on ``file_hash``, DELETE never DROP on
user data — the one exception is ``set_meta`` recreating
``document_chunks`` when the embedding dimension changes, since chunks
are derived data reproducible from ``documents.markdown``.

Follows the DuckDB VSS/FTS patterns in ``esdc.search.semantic_resolver``
(HNSW index, manual cosine via ``list_dot_product``, BM25 FTS, RRF
merge) rather than inventing new syntax.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING, Any

import duckdb

from esdc.configs import Config
from esdc.corpus.chunker import Chunk

if TYPE_CHECKING:
    from esdc.corpus.mirror import MirrorReport

logger = logging.getLogger(__name__)

_SQLITE_DOC_DDL = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id TEXT PRIMARY KEY,
    file_name TEXT NOT NULL,
    file_path TEXT NOT NULL,
    file_hash TEXT NOT NULL UNIQUE,
    doc_type TEXT,
    doc_topic TEXT,
    doc_number TEXT,
    doc_date TEXT,
    subject TEXT,
    sender TEXT,
    recipient TEXT,
    doc_level TEXT,
    wk_name TEXT,
    field_name TEXT,
    project_name TEXT,
    pod_name TEXT,
    suggested_pod_ids TEXT,
    raw_entities TEXT,
    metadata TEXT,
    markdown TEXT NOT NULL,
    extraction_method TEXT NOT NULL,
    embedding_model TEXT NOT NULL,
    page_count INTEGER,
    ingested_at TEXT DEFAULT (datetime('now'))
)
"""


def _to_json(val: Any) -> str | None:
    """Serialize a value to JSON string, or None if val is None."""
    return json.dumps(val) if val is not None else None


def _parse_json_fields(doc: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    """Parse JSON-string fields in place; leave unparseable values as-is."""
    for key in keys:
        if isinstance(doc.get(key), str):
            with contextlib.suppress(json.JSONDecodeError, TypeError):
                doc[key] = json.loads(doc[key])
    return doc


def _header_matches(embed_text: str, prefix: str) -> bool:
    """True when embed_text's header segment was built from this prefix.

    build_embed_text joins prefix and section with ' | ' and separates the
    header from the body with a blank line, so the header is either the
    prefix exactly or the prefix followed by ' | <section>'.

    Comparing the whole segment rather than embed_text.startswith(prefix)
    is what catches a name SHORTENED to a prefix of itself: renaming a
    field 'Duri' -> 'Dur' leaves the old text still starting with the new
    prefix, and a startswith check would report that document as current
    forever.
    """
    header = embed_text.split("\n\n", 1)[0]
    return header == prefix or header.startswith(prefix + " | ")


def _trim_snippet(
    text: str | None, query: str | None, width: int = 300
) -> str | None:
    """Match-centred excerpt of a chunk, trimmed to ``width`` chars.

    _aggregate_keyword/_aggregate_semantic pick the snippet via
    ``arg_max(chunk_text, ...)``, which hands back the ENTIRE matching
    chunk (up to the chunker's ~3000-char cap) — measured at 89% of a
    50-document `list` payload on the live corpus. Trimming here, at the
    single point aggregate() assembles `documents`, is one place to get
    it right rather than duplicating the logic in both aggregate
    functions, and it only runs over the page actually returned (<=
    `limit` documents) rather than every match in the corpus.

    Centers the excerpt on the first occurrence of the query's first
    term so the returned text actually shows why the document matched,
    instead of an arbitrary chunk-start slice. `query=None` (the
    metadata-only path) has no term to center on, so a leading excerpt is
    used instead. Ellipses mark whichever side was cut, so the excerpt is
    never mistaken for the full chunk.
    """
    if text is None:
        return None
    if len(text) <= width:
        return text

    idx = 0
    if query:
        terms = [t for t in query.split() if t]
        if terms:
            pos = text.lower().find(terms[0].lower())
            if pos != -1:
                idx = pos

    half = width // 2
    start = max(0, idx - half)
    end = min(len(text), start + width)
    start = max(0, end - width)

    excerpt = text[start:end]
    if start > 0:
        excerpt = "..." + excerpt
    if end < len(text):
        excerpt = excerpt + "..."
    return excerpt


# Columns on `documents` that may be filtered by exact match in search().
_EXACT_FILTER_COLUMNS = ("doc_type", "doc_level")
# Columns on `documents` that store JSON arrays; filtered via case-insensitive
# substring match over each array element (json_each + ILIKE).
_JSON_ARRAY_FILTER_COLUMNS = (
    "wk_name", "field_name", "project_name", "doc_topic", "pod_name",
)
# Free-text columns filtered by case-insensitive substring. These hold long
# institutional strings -- a sender reads
# "PERTAMINA BADAN PEMBINAAN PENGUSAHAAN KONTRAKTOR ASING..." -- so exact
# match would never hit and only a substring is usable. Interpolated into
# SQL by name, so this tuple is the allowlist: never add a caller-supplied
# column here.
_TEXT_FILTER_COLUMNS = ("sender", "recipient", "subject", "doc_number")

# Full row shape for `documents`, shared by get_document (DuckDB mirror),
# get_document_by_id, and get_document_by_hash (both SQLite truth) so the
# three never drift apart. Order matters only for the DuckDB reader's
# zip-based dict construction; the SQLite readers key off column name via
# sqlite3.Row and would tolerate reordering, but keep them in lockstep.
_DOC_COLUMNS = (
    "doc_id", "file_name", "file_path", "file_hash", "doc_type",
    "doc_topic", "doc_number", "doc_date", "subject", "sender",
    "recipient", "doc_level", "wk_name", "field_name", "project_name",
    "pod_name", "suggested_pod_ids", "raw_entities", "metadata",
    "markdown", "extraction_method", "embedding_model", "page_count",
    "ingested_at",
)
# Columns within _DOC_COLUMNS that hold JSON-encoded values and must be
# parsed back to Python lists/dicts before a row is returned to a caller.
_DOC_JSON_FIELDS = (
    "doc_topic", "wk_name", "field_name", "project_name",
    "pod_name", "suggested_pod_ids", "raw_entities", "metadata",
)


class CorpusStore:
    """DuckDB store for ingested corpus documents and chunk embeddings."""

    DOC_TABLE = "documents"
    CHUNK_TABLE = "document_chunks"
    META_TABLE = "corpus_meta"

    def __init__(
        self,
        db_path: Path | None = None,
        embedder: Any | None = None,
        sqlite_path: Path | None = None,
    ) -> None:
        """Initialize with lazy DuckDB/SQLite connections and an embedder.

        Args:
            db_path: DuckDB file path. Defaults to Config.get_db_file().
            embedder: Object with generate_embedding/generate_embeddings_batch
                and a `.model` attribute. Defaults to the internal llama.cpp
                Qwen3 embedder (esdc.corpus.embedder.InternalEmbedder) — no
                Ollama daemon needed for corpus commit/search.
            sqlite_path: Operational SQLite db holding the documents
                source of truth. Defaults to the shared esdc.sqlite.
        """
        if db_path is None:
            db_path = Config.get_db_file()
        self._db_path = Path(db_path)
        self._conn: duckdb.DuckDBPyConnection | None = None
        self._sqlite_path = Path(sqlite_path) if sqlite_path is not None else None
        self._sconn: sqlite3.Connection | None = None

        if embedder is None:
            from esdc.corpus.embedder import InternalEmbedder

            embedder = InternalEmbedder()
        self._embedder = embedder

    def _get_connection(self) -> duckdb.DuckDBPyConnection:
        """Get or create the DuckDB connection with VSS + FTS extensions loaded."""
        if self._conn is not None:
            try:
                self._conn.execute("SELECT 1")
            except Exception:
                logger.info("[Corpus] stale connection detected, reconnecting")
                with contextlib.suppress(Exception):
                    self._conn.close()
                self._conn = None

        if self._conn is None:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = duckdb.connect(str(self._db_path))
            with contextlib.suppress(Exception):
                self._conn.execute("INSTALL vss")
            self._conn.execute("LOAD vss")
            with contextlib.suppress(Exception):
                self._conn.execute("INSTALL fts")
            self._conn.execute("LOAD fts")
            self._conn.execute("SET hnsw_enable_experimental_persistence = true")
            logger.debug("[Corpus] DuckDB connection established with VSS/FTS")
        return self._conn

    def _get_sqlite(self) -> sqlite3.Connection:
        """Get or create the SQLite connection holding the documents table."""
        if self._sconn is not None:
            try:
                self._sconn.execute("SELECT 1")
            except Exception:
                with contextlib.suppress(Exception):
                    self._sconn.close()
                self._sconn = None
        if self._sconn is None:
            from esdc.pod_registry.store import get_sqlite_connection

            self._sconn = get_sqlite_connection(self._sqlite_path)
            self._sconn.execute(_SQLITE_DOC_DDL)
            for col in (
                "pod_name",
                "suggested_pod_ids",
                "raw_entities",
                "metadata",
            ):
                with contextlib.suppress(sqlite3.OperationalError):
                    self._sconn.execute(
                        f"ALTER TABLE documents ADD COLUMN {col} TEXT"
                    )
            # ingested_at's DDL default (`DEFAULT (datetime('now'))`) is a
            # non-constant expression -- SQLite's ADD COLUMN only accepts a
            # constant default, so self-heal without one rather than
            # raising. Rows added before this migration simply have a NULL
            # ingested_at, which TRY_CAST in the mirror already tolerates.
            with contextlib.suppress(sqlite3.OperationalError):
                self._sconn.execute("ALTER TABLE documents ADD COLUMN ingested_at TEXT")
            self._sconn.commit()
        return self._sconn

    def ensure_tables(self, validate_model: bool = False) -> None:
        """Create documents/document_chunks/corpus_meta tables if missing.

        Detects the embedding dimension from the configured embedder and
        pins it (with the model name) in corpus_meta. If corpus_meta
        already holds a different model/dim, raises ValueError instructing
        the user to run `esdc corpus reembed`.

        When ``validate_model=False`` (default), skips the embedder probe
        if tables already exist so read-only commands like ``corpus list``
        work without loading the embedding model. Pass ``validate_model=True`` from
        write paths (commit, reembed) to detect model mismatches.
        """
        conn = self._get_connection()

        # Check if all tables already exist
        row = conn.execute(
            "SELECT COUNT(*) FROM information_schema.tables "
            "WHERE table_name IN (?, ?, ?)",
            [self.DOC_TABLE, self.CHUNK_TABLE, self.META_TABLE],
        ).fetchone()
        tables_exist = row is not None and row[0] == 3

        if tables_exist:
            existing = conn.execute(
                f"SELECT embedding_model, dim FROM {self.META_TABLE} LIMIT 1"
            ).fetchone()
            if existing is None:
                # Tables exist but meta is empty — need embedder probe
                tables_exist = False

        if tables_exist and not validate_model:
            model, dim = existing  # type: ignore[misc]
            logger.debug("[Corpus] tables exist, dim=%d from corpus_meta", dim)
        else:
            logger.info("[Corpus] detecting embedding dimension from model")
            test_embedding = self._embedder.generate_embedding("test")
            dim = len(test_embedding)
            model = self._embedder.model

        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.DOC_TABLE} (
                doc_id VARCHAR PRIMARY KEY,
                file_name VARCHAR NOT NULL,
                file_path VARCHAR NOT NULL,
                file_hash VARCHAR NOT NULL UNIQUE,
                doc_type VARCHAR,
                doc_topic JSON,
                doc_number VARCHAR,
                doc_date DATE,
                subject TEXT,
                sender VARCHAR,
                recipient VARCHAR,
                doc_level VARCHAR,
                wk_name JSON,
                field_name JSON,
                project_name JSON,
                pod_name JSON,
                suggested_pod_ids JSON,
                raw_entities JSON,
                metadata JSON,
                markdown TEXT NOT NULL,
                extraction_method VARCHAR NOT NULL,
                embedding_model VARCHAR NOT NULL,
                page_count INTEGER,
                ingested_at TIMESTAMP DEFAULT current_timestamp
            )
        """)
        # documents predating the doc_topic split (Task 4) won't have this
        # column — CREATE TABLE IF NOT EXISTS above is a no-op for them, so
        # add it explicitly. Idempotent: a no-op once the column exists.
        conn.execute(
            f"ALTER TABLE {self.DOC_TABLE} ADD COLUMN IF NOT EXISTS doc_topic JSON"
        )
        for col in ("pod_name", "suggested_pod_ids"):
            conn.execute(
                f"ALTER TABLE {self.DOC_TABLE} ADD COLUMN IF NOT EXISTS {col} JSON"
            )
        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.CHUNK_TABLE} (
                chunk_id VARCHAR PRIMARY KEY,
                doc_id VARCHAR NOT NULL,
                chunk_index INTEGER NOT NULL,
                section VARCHAR,
                chunk_text TEXT NOT NULL,
                embed_text TEXT,
                embedding FLOAT[{dim}]
            )
        """)
        conn.execute(
            f"ALTER TABLE {self.CHUNK_TABLE} ADD COLUMN IF NOT EXISTS embed_text TEXT"
        )
        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.META_TABLE} (
                embedding_model VARCHAR NOT NULL,
                dim INTEGER NOT NULL
            )
        """)
        # probe_vec added after corpus_meta shipped; ALTER keeps existing
        # corpora intact (same approach as embed_text / doc_topic).
        conn.execute(
            f"ALTER TABLE {self.META_TABLE} ADD COLUMN IF NOT EXISTS probe_vec JSON"
        )

        if tables_exist:
            if validate_model:
                # Check for model mismatch
                existing = conn.execute(
                    f"SELECT embedding_model, dim FROM {self.META_TABLE} LIMIT 1"
                ).fetchone()
                if existing is not None and (
                    existing[0] != model or existing[1] != dim
                ):
                    raise ValueError(
                        f"[Corpus] embedding model changed "
                        f"(was {existing[0]} dim={existing[1]}, "
                        f"now {model} dim={dim}). "
                        f"Existing chunk embeddings are no longer comparable. "
                        f"Run `esdc corpus reembed` to rebuild them."
                    )
        else:
            # First run — seed corpus_meta
            existing = conn.execute(
                f"SELECT embedding_model, dim FROM {self.META_TABLE} LIMIT 1"
            ).fetchone()
            if existing is None:
                conn.execute(
                    f"INSERT INTO {self.META_TABLE} "
                    "(embedding_model, dim) VALUES (?, ?)",
                    [model, dim],
                )
            elif existing[0] != model or existing[1] != dim:
                raise ValueError(
                    f"[Corpus] embedding model changed "
                    f"(was {existing[0]} dim={existing[1]}, now {model} dim={dim}). "
                    f"Existing chunk embeddings are no longer comparable. "
                    f"Run `esdc corpus reembed` to rebuild them."
                )

        if validate_model:
            # Write paths only: proves this embedder shares a cosine space
            # with whatever produced the stored vectors. Seeds on first run.
            from esdc.embedders import check_or_seed_probe

            check_or_seed_probe(conn, self.META_TABLE, self._embedder)

        self._migrate_legacy_entity_columns()
        self._create_document_indexes()
        self._get_sqlite()  # documents source-of-truth table

    def _migrate_legacy_entity_columns(self) -> None:
        """Wrap pre-JSON plain-text entity values into JSON arrays.

        Older schemas stored wk_name/field_name/project_name as bare
        VARCHAR ('Rokan'). json_each()/json_contains() raise on non-JSON
        input, so one legacy row would break every filtered search.
        Idempotent: rows already holding valid JSON are untouched.
        """
        conn = self._get_connection()
        for col in _JSON_ARRAY_FILTER_COLUMNS:
            conn.execute(
                f"""
                UPDATE {self.DOC_TABLE}
                SET {col} = to_json([{col}])
                WHERE {col} IS NOT NULL AND NOT json_valid({col})
                """
            )

    def _create_document_indexes(self) -> None:
        """Create B-tree indexes on documents columns used for filtering."""
        conn = self._get_connection()
        doc_indexes = [
            ("idx_doc_doc_type", "doc_type"),
            ("idx_doc_doc_topic", "doc_topic"),
            ("idx_doc_doc_level", "doc_level"),
            ("idx_doc_wk_name", "wk_name"),
            ("idx_doc_field_name", "field_name"),
            ("idx_doc_project_name", "project_name"),
            ("idx_doc_doc_date", "doc_date"),
        ]
        for idx_name, column in doc_indexes:
            try:
                conn.execute(
                    f"CREATE INDEX IF NOT EXISTS {idx_name} "
                    f"ON {self.DOC_TABLE}({column})"
                )
                logger.debug("[Corpus] B-tree index created: %s", idx_name)
            except Exception as e:
                logger.warning("[Corpus] failed to create index %s: %s", idx_name, e)

    def _count(self, table: str) -> int:
        """Return COUNT(*) for a table (0 if the table/row set is empty)."""
        conn = self._get_connection()
        result = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
        return result[0] if result else 0

    def document_exists(self, file_hash: str) -> bool:
        """Return True if a document with this file_hash is already stored."""
        sconn = self._get_sqlite()
        result = sconn.execute(
            f"SELECT 1 FROM {self.DOC_TABLE} WHERE file_hash = ? LIMIT 1",
            [file_hash],
        ).fetchone()
        return result is not None

    def fill_blank_entities(
        self, doc_id: str, entities: dict[str, list[str] | None]
    ) -> list[str]:
        """Fill NULL/empty entity columns on an already-committed doc.

        For each column in ``entities`` (``wk_name``/``field_name``/
        ``project_name``): if the stored SQLite value is NULL or an empty
        array AND the given sidecar value is non-empty, write the sidecar
        value (JSON array) to SQLite (truth). The DuckDB mirror picks up
        the change at the next ``refresh_mirror()``, same as any other
        truth write. A stored value that already holds names — including
        one edited through the portal — is left untouched. Returns the
        field names actually filled; doc_id not found -> [].
        """
        if not entities:
            return []
        sconn = self._get_sqlite()
        cols = list(entities.keys())
        row = sconn.execute(
            f"SELECT {', '.join(cols)} FROM {self.DOC_TABLE} WHERE doc_id = ?",
            [doc_id],
        ).fetchone()
        if row is None:
            return []

        updates: dict[str, list[str]] = {}
        for col, current_raw in zip(cols, row, strict=True):
            sidecar_value = entities.get(col)
            if not sidecar_value:
                continue
            current = json.loads(current_raw) if current_raw else None
            if current:
                continue
            updates[col] = sidecar_value

        if not updates:
            return []

        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = [_to_json(v) for v in updates.values()]
        with sconn:
            sconn.execute(
                f"UPDATE {self.DOC_TABLE} SET {set_clause} WHERE doc_id = ?",
                (*values, doc_id),
            )

        return list(updates.keys())

    def _doc_row_values(self, doc: dict[str, Any]) -> list[Any]:
        return [
            doc["doc_id"],
            doc["file_name"],
            doc["file_path"],
            doc["file_hash"],
            doc.get("doc_type"),
            _to_json(doc.get("doc_topic")),
            doc.get("doc_number"),
            doc.get("doc_date"),
            doc.get("subject"),
            doc.get("sender"),
            doc.get("recipient"),
            doc.get("doc_level"),
            _to_json(doc.get("wk_name")),
            _to_json(doc.get("field_name")),
            _to_json(doc.get("project_name")),
            _to_json(doc.get("pod_name")),
            _to_json(doc.get("suggested_pod_ids")),
            doc.get("raw_entities"),
            doc.get("metadata"),
            doc["markdown"],
            doc["extraction_method"],
            self._embedder.model,
            doc.get("page_count"),
        ]

    _DOC_INSERT_SQL_TEMPLATE = """
        INSERT INTO {table} (
            doc_id, file_name, file_path, file_hash, doc_type, doc_topic,
            doc_number, doc_date, subject, sender, recipient,
            doc_level, wk_name, field_name, project_name,
            pod_name, suggested_pod_ids,
            raw_entities, metadata, markdown, extraction_method,
            embedding_model, page_count
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """

    def insert_document(self, doc: dict[str, Any], chunks: list[Chunk]) -> None:
        """Insert a document (SQLite truth + DuckDB chunks).

        DuckDB holds derived data only: the chunks here, and the
        ``documents`` mirror via ``refresh_mirror()`` at the end of the
        batch — this call never writes it. No cross-db transaction
        exists, so the SQLite row is written LAST as the commit marker.
        """
        conn = self._get_connection()
        sconn = self._get_sqlite()
        from esdc.corpus.context import build_context_prefix, build_embed_text

        prefix = build_context_prefix(doc)
        embed_texts = [build_embed_text(prefix, c.section, c.text) for c in chunks]
        embeddings = (
            self._embedder.generate_embeddings_batch(embed_texts)
            if embed_texts
            else []
        )
        values = self._doc_row_values(doc)

        # DuckDB holds derived data only: chunks here, the `documents`
        # mirror via refresh_mirror() at the end of the batch. The SQLite
        # row stays the commit marker, so it is written last.
        conn.execute("BEGIN TRANSACTION")
        try:
            conn.execute(
                f"DELETE FROM {self.CHUNK_TABLE} WHERE doc_id = ?", [doc["doc_id"]]
            )
            if chunks:
                conn.executemany(
                    f"""
                    INSERT INTO {self.CHUNK_TABLE} (
                        chunk_id, doc_id, chunk_index, section, chunk_text,
                        embed_text, embedding
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        [
                            f"{doc['doc_id']}:{chunk.index:04d}",
                            doc["doc_id"],
                            chunk.index,
                            chunk.section,
                            chunk.text,
                            embed_text,
                            embedding,
                        ]
                        for chunk, embed_text, embedding in zip(
                            chunks, embed_texts, embeddings, strict=True
                        )
                    ],
                )
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise

        try:
            with sconn:
                sconn.execute(
                    self._DOC_INSERT_SQL_TEMPLATE.format(table=self.DOC_TABLE),
                    values,
                )
        except Exception:
            # Truth write failed: drop the chunks we just wrote so no
            # orphaned embeddings survive. (A later refresh would sweep
            # them anyway; this keeps the failure local.)
            with contextlib.suppress(Exception):
                conn.execute(
                    f"DELETE FROM {self.CHUNK_TABLE} WHERE doc_id = ?",
                    [doc["doc_id"]],
                )
            raise

    def delete_document(self, doc_id: str) -> None:
        """Delete a document: SQLite truth row + DuckDB chunks and mirror row.

        get_document/list_documents/find_doc_ids are serving reads that
        answer from the DuckDB mirror (see the module docstring's
        read-path routing rule), so a delete must remove the mirror's
        `documents` row too, or those calls keep surfacing a deleted
        document until the next refresh_mirror(). The chunk delete and
        mirror-row delete run in one DuckDB transaction; this is not a
        return to the dual-write compensation Task 7 removed — there is
        no write to compensate, only a delete that the next refresh
        would perform anyway, so it is idempotent and needs no rollback
        logic of its own beyond the transaction already here.
        """
        conn = self._get_connection()
        conn.execute("BEGIN TRANSACTION")
        try:
            conn.execute(
                f"DELETE FROM {self.CHUNK_TABLE} WHERE doc_id = ?", [doc_id]
            )
            conn.execute(f"DELETE FROM {self.DOC_TABLE} WHERE doc_id = ?", [doc_id])
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
        sconn = self._get_sqlite()
        with sconn:
            sconn.execute(f"DELETE FROM {self.DOC_TABLE} WHERE doc_id = ?", [doc_id])

    def list_documents(self) -> list[dict[str, Any]]:
        """List all documents from the DuckDB mirror with chunk counts, newest first.

        Serving read: answers a user/agent, so it uses the mirror. See
        the module docstring's read-path routing rule. Both `documents`
        and `document_chunks` live in DuckDB, so the chunk count is a
        single joined query rather than a second cross-store lookup.
        """
        conn = self._get_connection()
        cols = [
            "doc_id", "file_name", "doc_type", "doc_topic", "doc_date",
            "subject", "doc_level", "wk_name", "field_name", "project_name",
            "pod_name", "suggested_pod_ids",
            "extraction_method", "page_count", "ingested_at", "n_chunks",
        ]
        rows = conn.execute(f"""
            SELECT d.doc_id, d.file_name, d.doc_type, d.doc_topic, d.doc_date,
                   d.subject, d.doc_level, d.wk_name, d.field_name,
                   d.project_name, d.pod_name, d.suggested_pod_ids,
                   d.extraction_method, d.page_count, d.ingested_at,
                   COUNT(c.chunk_id) AS n_chunks
            FROM {self.DOC_TABLE} d
            LEFT JOIN {self.CHUNK_TABLE} c ON c.doc_id = d.doc_id
            GROUP BY ALL
            ORDER BY d.ingested_at DESC, d.doc_id
        """).fetchall()

        docs = []
        for row in rows:
            doc = dict(zip(cols, row, strict=True))
            _parse_json_fields(
                doc,
                (
                    "doc_topic",
                    "wk_name",
                    "field_name",
                    "project_name",
                    "pod_name",
                    "suggested_pod_ids",
                ),
            )
            docs.append(doc)
        return docs

    def fingerprint_rows(self) -> list[tuple[str, str]]:
        """(doc_id, file_hash) for every document — input to corpus_fingerprint."""
        sconn = self._get_sqlite()
        rows = sconn.execute(
            f"SELECT doc_id, file_hash FROM {self.DOC_TABLE}"
        ).fetchall()
        return [(r[0], r[1]) for r in rows]

    def document_bodies(self) -> list[tuple[str, str, str]]:
        """(doc_id, doc_number, markdown) for every document.

        Build-time read for citation scanning: markdown is only in the
        SQLite truth table, and the scan needs whole bodies rather than
        chunks.
        """
        sconn = self._get_sqlite()
        rows = sconn.execute(
            f"SELECT doc_id, COALESCE(doc_number, ''), COALESCE(markdown, '') "
            f"FROM {self.DOC_TABLE}"
        ).fetchall()
        return [(r[0], r[1], r[2]) for r in rows]

    def sample_content(
        self, doc_id: str, chunk_seed: str | None = None
    ) -> dict[str, Any] | None:
        """doc_type + subject + one chunk's text for one doc, for query synthesis.

        chunk_seed=None returns the first chunk. Given a seed, one chunk
        is picked deterministically from the whole document: the first
        chunk of a letter is the letterhead, which yields a benchmark
        query about the header block rather than about the substance.
        """
        sconn = self._get_sqlite()
        row = sconn.execute(
            f"SELECT doc_id, doc_type, subject, file_hash FROM {self.DOC_TABLE} "
            f"WHERE doc_id = ?",
            (doc_id,),
        ).fetchone()
        if row is None:
            return None
        doc = dict(row)
        rows = (
            self._get_connection()
            .execute(
                f"SELECT chunk_text FROM {self.CHUNK_TABLE} "
                f"WHERE doc_id = ? ORDER BY chunk_index",
                [doc_id],
            )
            .fetchall()
        )
        if not rows:
            doc["chunk_text"] = ""
        elif chunk_seed is None:
            doc["chunk_text"] = rows[0][0]
        else:
            digest = hashlib.sha1(chunk_seed.encode("utf-8")).hexdigest()
            doc["chunk_text"] = rows[int(digest, 16) % len(rows)][0]
        doc["subject"] = doc.get("subject") or ""
        return doc

    def stale_embed_docs(self) -> list[str]:
        """doc_ids whose chunks carry an out-of-date context prefix.

        embed_text is derived: build_context_prefix(doc) + section +
        chunk_text. Editing a document's entities changes the correct
        prefix but leaves the chunks untouched, so search keeps matching
        the old names. Rather than tracking a dirty flag, staleness is
        recomputed here by comparing what each chunk stores against what
        the current document row implies.

        Reads the DuckDB `documents` mirror, so it only sees entity edits
        that have already been through refresh_mirror(). That is the
        intended contract: an unrefreshed mirror is the mirror's problem,
        not a staleness signal.
        """
        from esdc.corpus.context import build_context_prefix

        conn = self._get_connection()
        rows = conn.execute(f"""
            SELECT d.doc_id, d.doc_type, d.doc_topic, d.subject, d.wk_name,
                   d.field_name, d.project_name, d.pod_name,
                   ANY_VALUE(c.embed_text) AS sample_embed_text
            FROM {self.DOC_TABLE} d
            JOIN {self.CHUNK_TABLE} c ON c.doc_id = d.doc_id
            GROUP BY ALL
        """).fetchall()

        cols = (
            "doc_id", "doc_type", "doc_topic", "subject", "wk_name",
            "field_name", "project_name", "pod_name", "sample_embed_text",
        )
        stale: list[str] = []
        for row in rows:
            doc = dict(zip(cols, row, strict=True))
            embed_text = doc.pop("sample_embed_text") or ""
            _parse_json_fields(
                doc, ("doc_topic", "wk_name", "field_name", "project_name", "pod_name")
            )
            expected = build_context_prefix(doc)
            if expected and not _header_matches(embed_text, expected):
                stale.append(doc["doc_id"])
        return sorted(stale)

    def find_doc_ids(self, filters: dict[str, Any]) -> list[tuple[str, str]]:
        """(doc_id, file_name) pairs matching documents-column filters.

        Serving read: answers a user/agent, so it uses the mirror. See
        the module docstring's read-path routing rule. Same allowlisted
        filter semantics as search(); empty filters match everything
        (callers gate destructive use).
        """
        conn = self._get_connection()
        clause, params = self._build_filter_clause(filters, "d")
        rows = conn.execute(
            f"SELECT d.doc_id, d.file_name FROM {self.DOC_TABLE} d "
            f"WHERE 1=1{clause} ORDER BY d.file_name",
            params,
        ).fetchall()
        return [(r[0], r[1]) for r in rows]

    def clear(self) -> dict[str, int]:
        """Delete all documents (both stores) and chunks; corpus_meta is preserved."""
        conn = self._get_connection()
        sconn = self._get_sqlite()
        counts = self.counts()

        with sconn:
            sconn.execute(f"DELETE FROM {self.DOC_TABLE}")
        conn.execute("BEGIN TRANSACTION")
        try:
            conn.execute(f"DELETE FROM {self.CHUNK_TABLE}")
            conn.execute(f"DELETE FROM {self.DOC_TABLE}")
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise

        return counts

    def replace_chunks(self, doc: dict[str, Any], chunks: list[Chunk]) -> None:
        """Replace all chunks for a document (used by `corpus reembed`)."""
        from esdc.corpus.context import build_context_prefix, build_embed_text

        doc_id = doc["doc_id"]
        conn = self._get_connection()
        prefix = build_context_prefix(doc)
        embed_texts = [build_embed_text(prefix, c.section, c.text) for c in chunks]
        embeddings = (
            self._embedder.generate_embeddings_batch(embed_texts)
            if embed_texts
            else []
        )

        conn.execute("BEGIN TRANSACTION")
        try:
            conn.execute(
                f"DELETE FROM {self.CHUNK_TABLE} WHERE doc_id = ?", [doc_id]
            )
            for chunk, embed_text, embedding in zip(
                chunks, embed_texts, embeddings, strict=True
            ):
                chunk_id = f"{doc_id}:{chunk.index:04d}"
                conn.execute(
                    f"""
                    INSERT INTO {self.CHUNK_TABLE} (
                        chunk_id, doc_id, chunk_index, section, chunk_text,
                        embed_text, embedding
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        chunk_id,
                        doc_id,
                        chunk.index,
                        chunk.section,
                        chunk.text,
                        embed_text,
                        embedding,
                    ],
                )
            conn.execute(
                f"UPDATE {self.DOC_TABLE} SET embedding_model = ? WHERE doc_id = ?",
                [self._embedder.model, doc_id],
            )
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
        sconn = self._get_sqlite()
        with sconn:
            sconn.execute(
                f"UPDATE {self.DOC_TABLE} SET embedding_model = ? WHERE doc_id = ?",
                [self._embedder.model, doc_id],
            )

    def set_meta(self, embedding_model: str, dim: int) -> None:
        """Overwrite corpus_meta; recreate document_chunks if dim changed.

        document_chunks is the ONLY table this store ever drops: chunks
        are derived data reproducible from documents.markdown via
        `corpus reembed`, unlike documents which holds the source of truth.
        """
        conn = self._get_connection()
        existing = conn.execute(
            f"SELECT embedding_model, dim FROM {self.META_TABLE} LIMIT 1"
        ).fetchone()

        dim_changed = existing is not None and existing[1] != dim

        from esdc.embedders import PROBE_TEXT

        probe = json.dumps(self._embedder.generate_embedding(PROBE_TEXT))

        conn.execute("BEGIN TRANSACTION")
        try:
            conn.execute(f"DELETE FROM {self.META_TABLE}")
            conn.execute(
                f"INSERT INTO {self.META_TABLE} "
                "(embedding_model, dim, probe_vec) VALUES (?, ?, ?)",
                [embedding_model, dim, probe],
            )
            if dim_changed:
                logger.info(
                    "[Corpus] embedding dim changed %s -> %d, recreating %s",
                    existing[1] if existing else None,
                    dim,
                    self.CHUNK_TABLE,
                )
                conn.execute(f"DROP TABLE IF EXISTS {self.CHUNK_TABLE}")
                conn.execute(f"""
                    CREATE TABLE {self.CHUNK_TABLE} (
                        chunk_id VARCHAR PRIMARY KEY,
                        doc_id VARCHAR NOT NULL,
                        chunk_index INTEGER NOT NULL,
                        section VARCHAR,
                        chunk_text TEXT NOT NULL,
                        embed_text TEXT,
                        embedding FLOAT[{dim}]
                    )
                """)
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise

    def rebuild_indexes(self) -> None:
        """Rebuild the HNSW vector index and FTS index on document_chunks.

        Call once after a batch of inserts/updates, not per-document.
        """
        conn = self._get_connection()
        conn.execute("SET hnsw_enable_experimental_persistence = true")

        try:
            conn.execute("DROP INDEX IF EXISTS idx_hnsw_chunks")
            conn.execute(f"""
                CREATE INDEX idx_hnsw_chunks
                ON {self.CHUNK_TABLE} USING HNSW (embedding)
                WITH (metric = 'cosine')
            """)
            logger.info("[Corpus] HNSW index created")
        except Exception as e:
            logger.error("[Corpus] HNSW index failed | error=%s", e)

        try:
            conn.execute(
                f"PRAGMA create_fts_index("
                f"'{self.CHUNK_TABLE}', 'chunk_id', 'embed_text', overwrite=1)"
            )
            logger.info("[Corpus] FTS index created")
        except Exception as e:
            logger.error("[Corpus] FTS index failed | error=%s", e)

    def refresh_mirror(self) -> MirrorReport:
        """Rebuild the DuckDB derived tables from the SQLite truth.

        The mirror is derived data: this replaces it wholesale rather
        than reconciling it, so drift is not possible. Cheap — a full
        rebuild of ~1k documents measures ~0.03s. Call it at the end of
        any batch that mutated the truth.
        """
        from esdc.corpus.mirror import refresh_all

        return refresh_all(self._get_connection(), self._resolved_sqlite_path())

    def _resolved_sqlite_path(self) -> Path:
        """Path of the SQLite truth, defaulting to the shared esdc.sqlite."""
        if self._sqlite_path is not None:
            return self._sqlite_path
        from esdc.pod_registry.store import get_esdc_sqlite_path

        return get_esdc_sqlite_path()

    def counts(self) -> dict[str, int]:
        """Return current row counts for documents (SQLite) and chunks (DuckDB)."""
        sconn = self._get_sqlite()
        n_docs = sconn.execute(f"SELECT COUNT(*) FROM {self.DOC_TABLE}").fetchone()[0]
        return {
            "documents": n_docs,
            "chunks": self._count(self.CHUNK_TABLE),
        }

    def _build_filter_clause(
        self,
        filters: dict[str, Any] | None,
        table_alias: str,
        dialect: str = "duckdb",
    ) -> tuple[str, list[Any]]:
        """Build a parameterized WHERE clause for documents-column filters.

        Column names are validated against a hardcoded allowlist before
        being interpolated; values are always bound via `?`. The clause
        works against either backend's `documents` table: the default
        `duckdb` dialect serves the DuckDB mirror (search, find_doc_ids
        both run on DuckDB now), while the `sqlite` dialect is kept for
        any caller filtering the SQLite truth table directly; only the
        case-insensitive LIKE and the year extraction differ per dialect
        (SQLite LIKE is already case-insensitive for ASCII).
        """
        conditions: list[str] = []
        params: list[Any] = []
        if not filters:
            return "", params

        like = "ILIKE" if dialect == "duckdb" else "LIKE"
        for col in _EXACT_FILTER_COLUMNS:
            if filters.get(col):
                conditions.append(f"{table_alias}.{col} = ?")
                params.append(filters[col])
        for col in _JSON_ARRAY_FILTER_COLUMNS:
            if filters.get(col):
                # Case-insensitive substring over each array element.
                # json_each stringifies elements as '"Name"'; the %...%
                # pattern makes the surrounding quotes irrelevant.
                conditions.append(
                    f"EXISTS (SELECT 1 FROM json_each({table_alias}.{col}) "
                    f"WHERE CAST(value AS VARCHAR) {like} '%' || ? || '%')"
                )
                params.append(str(filters[col]))
        for col in _TEXT_FILTER_COLUMNS:
            if filters.get(col):
                conditions.append(f"{table_alias}.{col} {like} '%' || ? || '%'")
                params.append(str(filters[col]))
        if filters.get("year"):
            if dialect == "duckdb":
                conditions.append(f"EXTRACT(year FROM {table_alias}.doc_date) = ?")
            else:
                conditions.append(
                    f"CAST(strftime('%Y', {table_alias}.doc_date) AS INTEGER) = ?"
                )
            params.append(filters["year"])

        clause = (" AND " + " AND ".join(conditions)) if conditions else ""
        return clause, params

    def _vector_search(
        self, query_embedding: list[float], limit: int, filters: dict[str, Any] | None
    ) -> list[dict[str, Any]]:
        conn = self._get_connection()
        filter_clause, filter_params = self._build_filter_clause(filters, "d")
        # dim comes from len() — always an int, safe to interpolate.
        # DuckDB VSS only rewrites `array_cosine_distance(...) AS dist
        # ... ORDER BY dist LIMIT n` over a plain table scan into
        # HNSW_INDEX_SCAN (EXPLAIN-verified); a JOIN defeats the rewrite,
        # so the filtered path below is a deliberate sequential scan.
        dim = len(query_embedding)
        distance = f"array_cosine_distance(c.embedding, ?::FLOAT[{dim}]) AS dist"

        if filter_clause:
            sql = f"""
                SELECT c.chunk_id, c.doc_id, c.section, c.chunk_text, c.embed_text,
                    {distance}
                FROM {self.CHUNK_TABLE} c
                JOIN {self.DOC_TABLE} d ON d.doc_id = c.doc_id
                WHERE 1=1{filter_clause}
                ORDER BY dist ASC
                LIMIT ?
            """
        else:
            sql = f"""
                SELECT c.chunk_id, c.doc_id, c.section, c.chunk_text, c.embed_text,
                    {distance}
                FROM {self.CHUNK_TABLE} c
                ORDER BY dist ASC
                LIMIT ?
            """
        params = [query_embedding, *filter_params, limit]
        rows = conn.execute(sql, params).fetchall()
        return [
            {
                "chunk_id": row[0],
                "doc_id": row[1],
                "section": row[2],
                "chunk_text": row[3],
                "embed_text": row[4],
                "similarity": 1 - row[5],
            }
            for row in rows
        ]

    def _bm25_predicate(
        self,
        query: str,
        filters: dict[str, Any] | None,
        alias: str = "c",
    ) -> tuple[str, str, list[Any]]:
        """Build the FTS match expression plus the documents filter clause.

        Load-bearing escape: the FTS match_bm25 macro cannot take a `?`
        bind parameter, so the query text is the ONLY non-bound value in
        this module. Doubling single quotes is what keeps it a safe SQL
        string literal — do not remove, and do not reimplement this
        anywhere else.
        """
        escaped_query = query.replace("'", "''")
        match_expr = (
            f"fts_main_{self.CHUNK_TABLE}.match_bm25("
            f"{alias}.chunk_id, '{escaped_query}')"
        )
        filter_clause, filter_params = self._build_filter_clause(filters, "d")
        return match_expr, filter_clause, filter_params

    def _keyword_search(
        self, query: str, limit: int, filters: dict[str, Any] | None
    ) -> list[dict[str, Any]]:
        conn = self._get_connection()
        match_expr, filter_clause, filter_params = self._bm25_predicate(query, filters)

        sql = f"""
            SELECT
                c.chunk_id, c.doc_id, c.section, c.chunk_text, c.embed_text,
                {match_expr} AS bm25_score
            FROM {self.CHUNK_TABLE} c
            JOIN {self.DOC_TABLE} d ON d.doc_id = c.doc_id
            WHERE {match_expr} IS NOT NULL{filter_clause}
            ORDER BY bm25_score DESC
            LIMIT ?
        """
        params = [*filter_params, limit]
        rows = conn.execute(sql, params).fetchall()
        return [
            {
                "chunk_id": row[0],
                "doc_id": row[1],
                "section": row[2],
                "chunk_text": row[3],
                "embed_text": row[4],
                "bm25_score": round(float(row[5] or 0), 4),
            }
            for row in rows
        ]

    @staticmethod
    def _merge_rrf(
        vector_results: list[dict[str, Any]],
        keyword_results: list[dict[str, Any]],
        k: int = 60,
    ) -> list[dict[str, Any]]:
        """Merge vector and keyword results using Reciprocal Rank Fusion."""
        if not vector_results and not keyword_results:
            return []

        scored: dict[str, dict[str, Any]] = {}
        for rank, result in enumerate(vector_results):
            key = result["chunk_id"]
            if key not in scored:
                scored[key] = dict(result)
                scored[key]["score"] = 0.0
            scored[key]["score"] += 1 / (k + rank + 1)

        for rank, result in enumerate(keyword_results):
            key = result["chunk_id"]
            if key not in scored:
                scored[key] = dict(result)
                scored[key]["score"] = 0.0
            scored[key]["score"] += 1 / (k + rank + 1)

        return sorted(scored.values(), key=lambda x: x["score"], reverse=True)

    def _maybe_rerank(
        self,
        query: str,
        merged: list[dict[str, Any]],
        rerank: bool | None,
    ) -> list[dict[str, Any]]:
        """Reorder the top of the RRF list with the local cross-encoder.

        rerank=None reads corpus.rerank from config; an explicit bool
        overrides it (the eval harness compares both modes). Any failure
        keeps RRF order — rerank never breaks search.
        """
        cfg = Config.get_corpus_config()
        enabled = cfg.get("rerank", False) if rerank is None else rerank
        if not enabled or len(merged) <= 1:
            return merged

        from esdc.corpus.reranker import Reranker

        rr = Reranker.get()
        if rr is None:
            return merged

        pool = min(int(cfg.get("rerank_pool", 30)), len(merged))
        top = merged[:pool]
        try:
            scores = rr.rerank(
                query, [r.get("embed_text") or r["chunk_text"] for r in top]
            )
        except Exception as e:
            logger.warning(
                "[Corpus] rerank failed, keeping RRF order | error=%s", e
            )
            return merged
        for r, s in zip(top, scores, strict=True):
            r["rerank_score"] = s
        top.sort(key=lambda r: r["rerank_score"], reverse=True)
        return top + merged[pool:]

    @staticmethod
    def _cap_per_doc(
        merged: list[dict[str, Any]], cap: int
    ) -> list[dict[str, Any]]:
        """Keep at most `cap` entries per doc_id, preserving incoming order.

        cap <= 0 disables the cap (identity) — that is what the A/B eval
        run uses to reproduce the uncapped baseline.
        """
        if cap <= 0:
            return merged
        counts: dict[str, int] = {}
        kept = []
        for r in merged:
            doc_id = r["doc_id"]
            n = counts.get(doc_id, 0)
            if n < cap:
                kept.append(r)
                counts[doc_id] = n + 1
        return kept

    _DOC_META_COLUMNS = (
        "doc_id", "file_name", "doc_type", "doc_topic", "doc_date", "subject",
        "wk_name", "field_name", "project_name",
    )

    def _hydrate_docs(self, doc_ids: list[str]) -> dict[str, dict[str, Any]]:
        """doc_id -> document metadata, JSON array columns parsed."""
        if not doc_ids:
            return {}
        conn = self._get_connection()
        placeholders = ", ".join("?" for _ in doc_ids)
        rows = conn.execute(
            f"SELECT {', '.join(self._DOC_META_COLUMNS)} FROM {self.DOC_TABLE} "
            f"WHERE doc_id IN ({placeholders})",
            doc_ids,
        ).fetchall()
        docs: dict[str, dict[str, Any]] = {}
        for row in rows:
            doc = dict(zip(self._DOC_META_COLUMNS, row, strict=True))
            _parse_json_fields(
                doc, ("doc_topic", "wk_name", "field_name", "project_name")
            )
            docs[doc["doc_id"]] = doc
        return docs

    def _corpus_unavailable(self) -> dict[str, Any] | None:
        """not_available payload when there is nothing to search, else None."""
        try:
            n_chunks = self._count(self.CHUNK_TABLE)
        except Exception:
            return {
                "status": "not_available",
                "message": "Corpus not initialized. Run `esdc corpus commit` first.",
                "results": [],
                "count": 0,
            }
        if n_chunks == 0:
            return {
                "status": "not_available",
                "message": "No documents in corpus yet.",
                "results": [],
                "count": 0,
            }
        return None

    def search(
        self,
        query: str,
        limit: int = 10,
        filters: dict[str, Any] | None = None,
        rerank: bool | None = None,
    ) -> dict[str, Any]:
        """Hybrid (vector + BM25) search over document_chunks.

        Never raises: missing tables/indexes are reported as
        status="not_available"; other failures as status="error".
        """
        unavailable = self._corpus_unavailable()
        if unavailable is not None:
            return unavailable

        try:
            # Over-retrieve before RRF: a wider pool costs little here and
            # feeds both the fusion and the optional reranker.
            pool = max(limit * 2, 50)
            query_embedding = self._embedder.generate_embedding(query)
            vector_results = self._vector_search(query_embedding, pool, filters)

            try:
                keyword_results = self._keyword_search(query, pool, filters)
            except Exception as e:
                logger.warning(
                    "[Corpus] keyword search failed, using vector-only | error=%s", e
                )
                keyword_results = []

            merged = self._merge_rrf(vector_results, keyword_results)
            merged = self._maybe_rerank(query, merged, rerank)
            # Cap AFTER rerank, BEFORE the limit slice: capping first would
            # keep whichever chunk RRF happened to rank first per document;
            # capping after rerank keeps the best-scoring chunk instead.
            cap = Config.get_corpus_config().get("max_chunks_per_doc", 0)
            merged = self._cap_per_doc(merged, cap)[:limit]

            if not merged:
                return {"status": "no_results", "results": [], "count": 0}

            doc_ids = list({r["doc_id"] for r in merged})
            docs_by_id = self._hydrate_docs(doc_ids)

            results = []
            for r in merged:
                doc = docs_by_id.get(r["doc_id"], {})
                results.append(
                    {
                        "doc_id": r["doc_id"],
                        "file_name": doc.get("file_name"),
                        "doc_type": doc.get("doc_type"),
                        "doc_topic": doc.get("doc_topic"),
                        "doc_date": doc.get("doc_date"),
                        "subject": doc.get("subject"),
                        "wk_name": doc.get("wk_name"),
                        "field_name": doc.get("field_name"),
                        "project_name": doc.get("project_name"),
                        "section": r["section"],
                        "chunk_text": r["chunk_text"],
                        "score": round(r["score"], 6),
                    }
                )
                # Only present when the cross-encoder actually ran. It is
                # the one calibrated relevance number search produces —
                # P("yes") from RANK pooling — so the eval harness uses it
                # to decide whether a query found anything at all.
                if "rerank_score" in r:
                    results[-1]["rerank_score"] = r["rerank_score"]

            return {"status": "success", "results": results, "count": len(results)}

        except Exception as e:
            logger.error("[Corpus] search failed | error=%s", e)
            return {"status": "error", "message": str(e), "results": [], "count": 0}

    @staticmethod
    def _truncation_note(field: str, total: int, returned: int) -> str:
        """Note shown whenever a payload list is a partial page of the total.

        Shared wording for list-mode `documents` and count-mode `doc_ids`
        so an agent reading either payload gets the same instruction: the
        `count` field is exhaustive regardless of how many items are in
        the array, and raising `limit` (or narrowing with `filters`) is
        how to see the rest. Never present the array itself as exhaustive.
        """
        return (
            f"count ({total}) is the complete, exhaustive total. The "
            f"'{field}' array below holds only {returned} of them — a "
            f"partial page, not the full set. Raise `limit` or narrow with "
            f"`filters` to see more; do not report {returned} as the answer."
        )

    def aggregate(
        self,
        query: str | None,
        mode: str = "count",
        match: str = "hybrid",
        group_by: str | None = None,
        semantic_candidates: int = 20,
        limit: int = 50,
        filters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Exhaustive, document-level counting/listing over the corpus.

        Differs from search() in exactly two ways: no top-K cap, and
        results are deduped to doc_id. Never raises — an empty corpus is
        reported as not_available, other failures as error.

        match="keyword" is literal and conjunctive over document body
        text: BM25 selects candidates, then every query term must also
        appear in chunk_text of the SAME chunk. That confinement matters
        because the FTS index is built on embed_text, which prepends the
        document's type, subject and entity names to EVERY chunk --
        without it, a term appearing only in a subject line would count
        as a body mention.

        match="hybrid" (default) keeps that exact count as `count` and
        additionally returns the top semantically-similar documents the
        keyword pass MISSED, as `semantic_candidates` with scores, plus a
        `provenance` breakdown. The headline number stays reproducible;
        the semantic tail is offered for review, never folded in.

        match="semantic" returns only the ranking. There `count` means
        "candidates returned", not a total, and approximate is True.

        The conjunction is evaluated within ONE chunk, so a multi-term
        query counts documents that discuss the terms together, not
        documents that merely contain all of them somewhere. Deliberate —
        "dokumen yang menyebutkan X Y" asks for documents where X and Y
        are discussed together, not ones that happen to contain both
        tokens forty pages apart. Doc-level conjunction (one EXISTS per
        term) is the alternative and was not chosen.
        """
        unavailable = self._corpus_unavailable()
        if unavailable is not None:
            return unavailable

        semantic_rows: list[dict[str, Any]] = []
        try:
            if query is None:
                doc_ids, snippets = self._aggregate_metadata(filters)
                match_used, approximate = "metadata", False
            elif match == "semantic":
                semantic_rows = self._aggregate_semantic(
                    query, semantic_candidates, filters
                )
                doc_ids = [r["doc_id"] for r in semantic_rows]
                snippets = {r["doc_id"]: r["snippet"] for r in semantic_rows}
                match_used, approximate = "semantic", True
            else:
                doc_ids, snippets = self._aggregate_keyword(query, filters)
                match_used, approximate = "keyword", False
                if match == "hybrid":
                    # The tail is what keyword missed. Ask for the overlap
                    # too (candidates + len(doc_ids)) so that after
                    # subtracting the keyword hits there are still up to
                    # `semantic_candidates` genuinely new documents left,
                    # rather than a tail silently shortened by however many
                    # of the top-ranked ones keyword already found.
                    ranked = self._aggregate_semantic(
                        query, semantic_candidates + len(doc_ids), filters
                    )
                    kw_set = set(doc_ids)
                    semantic_rows = [
                        r for r in ranked if r["doc_id"] not in kw_set
                    ][:semantic_candidates]
                    match_used = "hybrid"
        except Exception as e:
            logger.error("[Corpus] aggregate failed | error=%s", e)
            return {"status": "error", "message": str(e), "count": 0}

        result: dict[str, Any] = {
            "status": "success" if doc_ids else "no_results",
            "mode": mode,
            "match": match_used,
            "approximate": approximate,
            "count": len(doc_ids),
        }

        if mode == "list":
            page = doc_ids[:limit]
            hydrated = self._hydrate_docs(page)
            result["documents"] = [
                {**hydrated.get(doc_id, {"doc_id": doc_id}),
                 "matched_snippet": _trim_snippet(snippets.get(doc_id), query)}
                for doc_id in page
            ]
            result["returned"] = len(page)
            result["truncated"] = len(doc_ids) > limit
            if result["truncated"]:
                result["note"] = self._truncation_note(
                    "documents", len(doc_ids), len(page)
                )
        else:
            # count/facets stay exhaustive: `count` above is len(doc_ids),
            # the FULL match set, computed before this branch and never
            # touched by `limit`. Only the doc_ids array we hand back is
            # paged — some callers use it, so it isn't dropped outright,
            # just bounded the same way list mode's `documents` is.
            page = doc_ids[:limit]
            result["doc_ids"] = page
            result["returned"] = len(page)
            result["truncated"] = len(doc_ids) > limit
            if result["truncated"]:
                result["note"] = self._truncation_note(
                    "doc_ids", len(doc_ids), len(page)
                )

        if match_used == "hybrid":
            # Only two numbers here are properties of the corpus rather
            # than of the caller's parameters, so only two are reported.
            #
            # `exact_total` is a real count: every document whose body
            # literally contains the terms. `semantic_extra` is the SIZE
            # OF A RANKING the caller asked for -- request 200 candidates
            # and it returns 200, which says nothing about how many
            # documents are "semantically related". It is labelled as a
            # ranking so the model cannot report it as a total.
            #
            # Deliberately NOT reported: a keyword_only/both split. The
            # overlap between the exact hits and the semantic top-N moves
            # with semantic_candidates (measured: 29/5 at N=5 versus 7/27
            # at N=200 for the same query and the same count of 34), so
            # it is an artifact of the knob, not a finding -- the same
            # trap the removed similarity_threshold represented.
            result["provenance"] = {
                "exact_total": len(doc_ids),
                "semantic_extra": len(semantic_rows),
                "semantic_extra_is_a_ranking": True,
            }
        if semantic_rows:
            result["semantic_candidates"] = [
                {
                    "doc_id": r["doc_id"],
                    "similarity": r["similarity"],
                    "matched_snippet": _trim_snippet(r["snippet"], query),
                }
                for r in semantic_rows
            ]

        if group_by:
            try:
                result["facets"] = self._facets(doc_ids, group_by)
            except ValueError as e:
                result["facets_error"] = str(e)

        return result

    def _aggregate_metadata(
        self, filters: dict[str, Any] | None
    ) -> tuple[list[str], dict[str, str]]:
        """Filter-only aggregation over the documents mirror."""
        conn = self._get_connection()
        clause, params = self._build_filter_clause(filters, "d")
        rows = conn.execute(
            f"SELECT d.doc_id FROM {self.DOC_TABLE} d WHERE 1=1{clause} "
            f"ORDER BY d.doc_id",
            params,
        ).fetchall()
        return [r[0] for r in rows], {}

    def _aggregate_keyword(
        self, query: str, filters: dict[str, Any] | None
    ) -> tuple[list[str], dict[str, str]]:
        """Exhaustive literal, conjunctive, body-text-only doc matching.

        The ILIKE terms are ANDed within a single chunk row, so all terms
        must co-occur in one passage. GROUP BY doc_id then dedups, which
        is what makes the count a document count rather than a hit count.
        """
        conn = self._get_connection()
        match_expr, filter_clause, filter_params = self._bm25_predicate(query, filters)

        terms = [t for t in query.split() if t]
        literal_clause = "".join(
            " AND c.chunk_text ILIKE '%' || ? || '%'" for _ in terms
        )

        sql = f"""
            SELECT c.doc_id,
                   arg_max(c.chunk_text, {match_expr}) AS snippet
            FROM {self.CHUNK_TABLE} c
            JOIN {self.DOC_TABLE} d ON d.doc_id = c.doc_id
            WHERE {match_expr} IS NOT NULL{filter_clause}{literal_clause}
            GROUP BY c.doc_id
            ORDER BY c.doc_id
        """
        rows = conn.execute(sql, [*filter_params, *terms]).fetchall()
        return [r[0] for r in rows], {r[0]: r[1] for r in rows}

    def _aggregate_semantic(
        self,
        query: str,
        candidates: int,
        filters: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        """Top-`candidates` documents by cosine similarity, best first.

        Returns a RANKING, never a set, and deliberately takes no
        similarity threshold. Measured over ten real queries on the live
        corpus, per-document max similarity tops out between 0.530
        ("separator") and 0.706 ("sumur eksplorasi"), and a single 0.5
        cutoff selects 6 documents for the first and 342 for the second.
        The absolute score tracks the query's own embedding scale, not
        document relevance, so no fixed threshold means the same thing
        twice and any count derived from one is an artifact. Rank within
        a single query IS stable, so rank is what callers get, with the
        score attached so they can say how close each match was.

        Scores every chunk (no top-K pool applied before scoring — see
        the pool-cap regression fixed in commit ad92fb8), dedups to the
        best chunk per document, then truncates the ranking.
        """
        conn = self._get_connection()
        filter_clause, filter_params = self._build_filter_clause(filters, "d")
        query_embedding = self._embedder.generate_embedding(query)
        dim = len(query_embedding)
        similarity = f"(1 - array_cosine_distance(c.embedding, ?::FLOAT[{dim}]))"

        sql = f"""
            WITH scored AS (
                SELECT c.doc_id, c.chunk_text, {similarity} AS similarity
                FROM {self.CHUNK_TABLE} c
                JOIN {self.DOC_TABLE} d ON d.doc_id = c.doc_id
                WHERE 1=1{filter_clause}
            )
            SELECT doc_id,
                   max(similarity) AS similarity,
                   arg_max(chunk_text, similarity) AS snippet
            FROM scored
            GROUP BY doc_id
            ORDER BY similarity DESC, doc_id
            LIMIT ?
        """
        params = [query_embedding, *filter_params, candidates]
        rows = conn.execute(sql, params).fetchall()
        return [
            {"doc_id": r[0], "similarity": round(float(r[1]), 4), "snippet": r[2]}
            for r in rows
        ]

    # group_by dimensions and how they aggregate. JSON-array columns expand
    # via json_each, so a document contributes to several buckets and the
    # buckets do NOT sum to the document count.
    _SCALAR_FACETS = ("doc_type", "doc_level")
    _JSON_FACETS = ("doc_topic", "wk_name", "field_name", "project_name")

    def _facets(self, doc_ids: list[str], group_by: str) -> dict[str, Any]:
        """Group the matched document set by one dimension."""
        if not doc_ids:
            return {"dimension": group_by, "multi_valued": False, "values": {}}
        conn = self._get_connection()
        placeholders = ", ".join("?" for _ in doc_ids)

        if group_by == "year":
            sql = (
                f"SELECT CAST(EXTRACT(year FROM d.doc_date) AS VARCHAR), COUNT(*) "
                f"FROM {self.DOC_TABLE} d WHERE d.doc_id IN ({placeholders}) "
                f"AND d.doc_date IS NOT NULL GROUP BY 1 ORDER BY 1"
            )
            multi = False
        elif group_by in self._SCALAR_FACETS:
            sql = (
                f"SELECT CAST(d.{group_by} AS VARCHAR), COUNT(*) "
                f"FROM {self.DOC_TABLE} d WHERE d.doc_id IN ({placeholders}) "
                f"GROUP BY 1 ORDER BY 1"
            )
            multi = False
        elif group_by in self._JSON_FACETS:
            sql = (
                f"SELECT json_extract_string(j.value, '$'), "
                f"COUNT(DISTINCT d.doc_id) "
                f"FROM {self.DOC_TABLE} d, json_each(d.{group_by}) j "
                f"WHERE d.doc_id IN ({placeholders}) GROUP BY 1 ORDER BY 1"
            )
            multi = True
        else:
            raise ValueError(
                f"unsupported group_by '{group_by}'; use one of: year, "
                f"{', '.join(self._SCALAR_FACETS + self._JSON_FACETS)}"
            )

        rows = conn.execute(sql, doc_ids).fetchall()
        return {
            "dimension": group_by,
            "multi_valued": multi,
            "values": {str(r[0]): r[1] for r in rows if r[0] is not None},
        }

    def get_document(self, doc_id: str) -> dict[str, Any] | None:
        """Fetch a single document's full row from the DuckDB mirror.

        Serving read: answers a user/agent, so it uses the mirror. See
        the module docstring's read-path routing rule.
        """
        conn = self._get_connection()
        row = conn.execute(
            f"SELECT {', '.join(_DOC_COLUMNS)} FROM {self.DOC_TABLE} WHERE doc_id = ?",
            [doc_id],
        ).fetchone()
        if row is None:
            return None
        # DuckDB returns plain tuples (no column names attached), unlike
        # sqlite3.Row below -- zip against the shared column tuple to
        # rebuild the dict.
        doc = dict(zip(_DOC_COLUMNS, row, strict=True))
        return _parse_json_fields(doc, _DOC_JSON_FIELDS)

    def _get_document_by(self, column: str, value: Any) -> dict[str, Any] | None:
        """Fetch a `documents` row from the SQLite truth by one exact-match column.

        Shared by get_document_by_id and get_document_by_hash, which are
        otherwise identical apart from their WHERE column. `column` is
        always one of our own hardcoded literals, never caller input.
        """
        sconn = self._get_sqlite()
        row = sconn.execute(
            f"""
            SELECT {', '.join(_DOC_COLUMNS)}
            FROM {self.DOC_TABLE}
            WHERE {column} = ?
            LIMIT 1
            """,
            [value],
        ).fetchone()
        if row is None:
            return None

        # sqlite3.Row maps by the SQL result's column names (here, the
        # same _DOC_COLUMNS used to build the SELECT), so dict(row) already
        # matches the DuckDB reader's zip-based dict above.
        doc = dict(row)
        return _parse_json_fields(doc, _DOC_JSON_FIELDS)

    def get_document_by_id(self, doc_id: str) -> dict[str, Any] | None:
        """Fetch a stored document's full row by doc_id (SQLite truth), or None.

        Truth-backed read for callers that need guaranteed-fresh
        content rather than the mirror's refresh window -- e.g. export,
        which rewrites files a user diffs. See the module docstring's
        read-path routing rule. Returns the same shape as get_document.
        """
        return self._get_document_by("doc_id", doc_id)

    def get_document_by_hash(self, file_hash: str) -> dict[str, Any] | None:
        """Fetch a stored document's full row by file_hash (SQLite truth), or None."""
        return self._get_document_by("file_hash", file_hash)

    def close(self) -> None:
        """Close the database connections."""
        if self._conn:
            self._conn.close()
            self._conn = None
        if self._sconn:
            with contextlib.suppress(Exception):
                self._sconn.close()
            self._sconn = None
