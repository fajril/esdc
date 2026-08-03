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


# Columns on `documents` that may be filtered by exact match in search().
_EXACT_FILTER_COLUMNS = ("doc_type", "doc_level")
# Columns on `documents` that store JSON arrays; filtered via case-insensitive
# substring match over each array element (json_each + ILIKE).
_JSON_ARRAY_FILTER_COLUMNS = ("wk_name", "field_name", "project_name", "doc_topic")


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
        value (JSON array) to SQLite (truth) and the DuckDB mirror
        (best-effort). A stored value that already holds names — including
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

        try:
            conn = self._get_connection()
            conn.execute(
                f"UPDATE {self.DOC_TABLE} SET {set_clause} WHERE doc_id = ?",
                (*values, doc_id),
            )
        except Exception as e:
            logger.warning(
                "[Corpus] fill_blank_entities DuckDB mirror failed | "
                "doc_id=%s error=%s",
                doc_id,
                e,
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
        """Delete a document: SQLite truth row + DuckDB chunks.

        The DuckDB `documents` mirror is derived, so it is not deleted
        here — the next refresh_mirror() rebuilds it without this row,
        and sweep_orphan_chunks() would clear any chunks this missed.
        Chunks are deleted eagerly so a delete takes effect on search
        immediately rather than at the next refresh.
        """
        conn = self._get_connection()
        conn.execute(f"DELETE FROM {self.CHUNK_TABLE} WHERE doc_id = ?", [doc_id])
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

    def sample_content(self, doc_id: str) -> dict[str, Any] | None:
        """doc_type + subject + first chunk text for one doc, for query synthesis."""
        sconn = self._get_sqlite()
        row = sconn.execute(
            f"SELECT doc_id, doc_type, subject, file_hash FROM {self.DOC_TABLE} "
            f"WHERE doc_id = ?",
            (doc_id,),
        ).fetchone()
        if row is None:
            return None
        doc = dict(row)
        chunk = (
            self._get_connection()
            .execute(
                f"SELECT chunk_text FROM {self.CHUNK_TABLE} "
                f"WHERE doc_id = ? ORDER BY chunk_index LIMIT 1",
                [doc_id],
            )
            .fetchone()
        )
        doc["chunk_text"] = chunk[0] if chunk else ""
        doc["subject"] = doc.get("subject") or ""
        return doc

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

    def refresh_mirror(self) -> "MirrorReport":
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
        being interpolated; values are always bound via `?`. The same
        filters work on the DuckDB mirror (search paths) and the SQLite
        truth table (find_doc_ids); only the case-insensitive LIKE and
        the year extraction differ per dialect (SQLite LIKE is already
        case-insensitive for ASCII).
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

    def _keyword_search(
        self, query: str, limit: int, filters: dict[str, Any] | None
    ) -> list[dict[str, Any]]:
        conn = self._get_connection()
        # Load-bearing escape: the FTS match_bm25 macro cannot take a `?`
        # bind parameter, so the query text is the ONLY non-bound value in
        # this module. Doubling single quotes is what keeps it a safe SQL
        # string literal — do not remove.
        escaped_query = query.replace("'", "''")
        filter_clause, filter_params = self._build_filter_clause(filters, "d")

        sql = f"""
            SELECT
                c.chunk_id, c.doc_id, c.section, c.chunk_text, c.embed_text,
                fts_main_{self.CHUNK_TABLE}.match_bm25(
                    c.chunk_id, '{escaped_query}'
                ) AS bm25_score
            FROM {self.CHUNK_TABLE} c
            JOIN {self.DOC_TABLE} d ON d.doc_id = c.doc_id
            WHERE fts_main_{self.CHUNK_TABLE}.match_bm25(
                c.chunk_id, '{escaped_query}'
            ) IS NOT NULL{filter_clause}
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
        conn = self._get_connection()

        try:
            n_chunks = self._count(self.CHUNK_TABLE)
        except Exception:
            return {
                "status": "not_available",
                "message": "Corpus not initialized. Run `esdc corpus commit` first.",
                "results": [],
                "count": 0,
            }

        try:
            if n_chunks == 0:
                return {
                    "status": "not_available",
                    "message": "No documents in corpus yet.",
                    "results": [],
                    "count": 0,
                }

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
            merged = self._maybe_rerank(query, merged, rerank)[:limit]

            if not merged:
                return {"status": "no_results", "results": [], "count": 0}

            doc_ids = list({r["doc_id"] for r in merged})
            placeholders = ", ".join("?" for _ in doc_ids)
            doc_rows = conn.execute(
                f"""
                SELECT doc_id, file_name, doc_type, doc_topic, doc_date, subject,
                       wk_name, field_name, project_name
                FROM {self.DOC_TABLE}
                WHERE doc_id IN ({placeholders})
                """,
                doc_ids,
            ).fetchall()
            doc_cols = [
                "doc_id", "file_name", "doc_type", "doc_topic", "doc_date", "subject",
                "wk_name", "field_name", "project_name",
            ]
            docs_by_id = {}
            for row in doc_rows:
                doc = dict(zip(doc_cols, row, strict=True))
                _parse_json_fields(
                    doc, ("doc_topic", "wk_name", "field_name", "project_name")
                )
                docs_by_id[row[0]] = doc

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

            return {"status": "success", "results": results, "count": len(results)}

        except Exception as e:
            logger.error("[Corpus] search failed | error=%s", e)
            return {"status": "error", "message": str(e), "results": [], "count": 0}

    def get_document(self, doc_id: str) -> dict[str, Any] | None:
        """Fetch a single document's full row from the DuckDB mirror.

        Serving read: answers a user/agent, so it uses the mirror. See
        the module docstring's read-path routing rule.
        """
        conn = self._get_connection()
        cols = [
            "doc_id", "file_name", "file_path", "file_hash", "doc_type",
            "doc_topic", "doc_number", "doc_date", "subject", "sender",
            "recipient", "doc_level", "wk_name", "field_name", "project_name",
            "pod_name", "suggested_pod_ids", "raw_entities", "metadata",
            "markdown", "extraction_method", "embedding_model", "page_count",
            "ingested_at",
        ]
        row = conn.execute(
            f"SELECT {', '.join(cols)} FROM {self.DOC_TABLE} WHERE doc_id = ?",
            [doc_id],
        ).fetchone()
        if row is None:
            return None
        doc = dict(zip(cols, row, strict=True))
        return _parse_json_fields(
            doc,
            (
                "doc_topic",
                "wk_name",
                "field_name",
                "project_name",
                "pod_name",
                "suggested_pod_ids",
                "raw_entities",
                "metadata",
            ),
        )

    def get_document_by_hash(self, file_hash: str) -> dict[str, Any] | None:
        """Fetch a stored document's full row by file_hash (SQLite truth), or None."""
        sconn = self._get_sqlite()
        row = sconn.execute(
            f"""
            SELECT doc_id, file_name, file_path, file_hash, doc_type, doc_topic,
                   doc_number, doc_date, subject, sender, recipient,
                   doc_level, wk_name, field_name, project_name,
                   pod_name, suggested_pod_ids,
                   raw_entities, metadata, markdown, extraction_method,
                   embedding_model, page_count, ingested_at
            FROM {self.DOC_TABLE}
            WHERE file_hash = ?
            LIMIT 1
            """,
            [file_hash],
        ).fetchone()
        if row is None:
            return None

        doc = dict(row)
        _parse_json_fields(
            doc,
            (
                "doc_topic",
                "wk_name",
                "field_name",
                "project_name",
                "pod_name",
                "suggested_pod_ids",
                "raw_entities",
                "metadata",
            ),
        )
        return doc

    def close(self) -> None:
        """Close the database connections."""
        if self._conn:
            self._conn.close()
            self._conn = None
        if self._sconn:
            with contextlib.suppress(Exception):
                self._sconn.close()
            self._sconn = None
