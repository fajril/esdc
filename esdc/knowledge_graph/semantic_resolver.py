"""Semantic search resolver for ESDC using DuckDB VSS extension.

Provides semantic search capabilities for project_remarks
using vector embeddings and HNSW index for fast similarity search.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import duckdb

from esdc.configs import Config
from esdc.knowledge_graph.embedding_manager import EmbeddingManager

logger = logging.getLogger(__name__)


class SemanticResolver:
    """Resolve semantic queries using DuckDB VSS extension.

    Provides:
    - Semantic search by text (auto-generates query embedding)
    - Semantic search by embedding (for pre-computed queries)
    - Embedding storage and retrieval from DuckDB
    - HNSW index for fast similarity search
    """

    EMBEDDING_TABLE = "project_embeddings"
    DEFAULT_LIMIT = 10

    def __init__(
        self,
        db_path: Path | str | None = None,
        model: str | None = None,
    ) -> None:
        """Initialize with DuckDB connection and EmbeddingManager."""
        if db_path is None:
            db_path = Config.get_db_file()
        self._db_path = Path(db_path)
        self._conn: duckdb.DuckDBPyConnection | None = None
        self._embedding_manager = EmbeddingManager(model=model)

    def _get_connection(self) -> duckdb.DuckDBPyConnection:
        """Get or create DuckDB connection with VSS extension loaded."""
        if self._conn is None:
            self._conn = duckdb.connect(str(self._db_path))
            # Load VSS extension for vector similarity search
            try:
                self._conn.execute("INSTALL vss")
            except Exception:
                pass  # Already installed
            self._conn.execute("LOAD vss")
            logger.debug("[Semantic] DuckDB connection established with VSS extension")
        return self._conn

    def build_embeddings_table(self) -> bool:
        """Create embeddings table and HNSW index.

        Creates table to store pre-computed embeddings for
        project_remarks.

        Returns:
            True if successful
        """
        conn = self._get_connection()

        try:
            # Create table for embeddings with dynamic dimensions
            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.EMBEDDING_TABLE} (
                    uuid VARCHAR PRIMARY KEY,
                    table_name VARCHAR NOT NULL,
                    field_name VARCHAR,
                    project_name VARCHAR,
                    source_text TEXT,
                    embedding FLOAT[]
                )
            """)

            logger.info("[Semantic] embeddings table created")
            return True

        except Exception as e:
            logger.error("[Semantic] failed to create table | error=%s", e)
            return False

    def generate_and_store_embeddings(
        self,
        table_name: str = "project_resources",
        batch_size: int = 100,
    ) -> dict[str, Any]:
        """Generate and store embeddings for all documents.

        Args:
            table_name: Source table name
            batch_size: Number of documents to process per batch

        Returns:
            Dict with status and count
        """
        conn = self._get_connection()

        try:
            # Fetch documents with remarks
            result = conn.execute(f"""
                SELECT 
                    uuid,
                    field_name,
                    project_name,
                    COALESCE(project_remarks, '') as source_text
                FROM {table_name}
                WHERE project_remarks IS NOT NULL
                    AND LENGTH(project_remarks) > 0
            """).fetchall()

            total = len(result)
            if total == 0:
                logger.info("[Semantic] no documents with project_remarks found")
                return {"status": "success", "count": 0, "table": table_name}

            logger.info("[Semantic] generating embeddings | count=%d", total)

            # Process in batches
            for i in range(0, total, batch_size):
                batch = result[i : i + batch_size]
                texts = [row[3] for row in batch]

                # Generate embeddings
                embeddings = self._embedding_manager.generate_embeddings_batch(texts)

                # Store in DuckDB
                for j, (row, embedding) in enumerate(zip(batch, embeddings)):
                    conn.execute(
                        f"""
                        INSERT OR REPLACE INTO {self.EMBEDDING_TABLE}
                        (uuid, table_name, field_name, project_name, source_text, embedding)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """,
                        [row[0], table_name, row[1], row[2], row[3], embedding],
                    )

                logger.debug(
                    "[Semantic] batch %d/%d processed",
                    i // batch_size + 1,
                    (total + batch_size - 1) // batch_size,
                )

            # Create HNSW index
            self._create_hnsw_index()

            return {
                "status": "success",
                "count": total,
                "table": table_name,
            }

        except Exception as e:
            logger.error("[Semantic] generation failed | error=%s", e)
            return {"status": "error", "message": str(e)}

    def _create_hnsw_index(self) -> None:
        """Create HNSW index for fast similarity search."""
        conn = self._get_connection()

        try:
            # Drop existing index if any
            conn.execute(f"DROP INDEX IF EXISTS idx_hnsw_embeddings")

            # Create HNSW index
            conn.execute(f"""
                CREATE INDEX idx_hnsw_embeddings
                ON {self.EMBEDDING_TABLE} USING HNSW (embedding)
                WITH (metric = 'cosine')
            """)
            logger.info("[Semantic] HNSW index created")
        except Exception as e:
            logger.error("[Semantic] HNSW index failed | error=%s", e)

    def search_by_text(
        self,
        query: str,
        limit: int = DEFAULT_LIMIT,
        table_name: str | None = None,
    ) -> dict[str, Any]:
        """Search by text query (auto-generates embedding).

        Args:
            query: User query text
            limit: Max results
            table_name: Filter by table (optional)

        Returns:
            Dict with status and results
        """
        # Generate query embedding
        query_embedding = self._embedding_manager.generate_embedding(query)

        return self.search_by_embedding(query_embedding, limit, table_name)

    def search_by_embedding(
        self,
        query_embedding: list[float],
        limit: int = DEFAULT_LIMIT,
        table_name: str | None = None,
    ) -> dict[str, Any]:
        """Search by pre-computed embedding vector.

        Args:
            query_embedding: Query embedding vector
            limit: Max results
            table_name: Filter by table (optional)

        Returns:
            Dict with status and results
        """
        conn = self._get_connection()

        try:
            # Check if embeddings table exists and has data
            check = conn.execute(f"""
                SELECT COUNT(*) FROM {self.EMBEDDING_TABLE}
            """).fetchone()

            if check[0] == 0:
                return {
                    "status": "not_available",
                    "message": "No embeddings found. Run 'esdc reload' to generate embeddings.",
                    "results": [],
                }

            # Build query
            if table_name:
                sql = f"""
                    SELECT 
                        uuid,
                        field_name,
                        project_name,
                        source_text,
                        array_cosine_similarity(embedding, ?::FLOAT[]) as similarity
                    FROM {self.EMBEDDING_TABLE}
                    WHERE table_name = ?
                    ORDER BY similarity DESC
                    LIMIT ?
                """
                params = [query_embedding, table_name, limit]
            else:
                sql = f"""
                    SELECT 
                        uuid,
                        field_name,
                        project_name,
                        source_text,
                        array_cosine_similarity(embedding, ?::FLOAT[]) as similarity
                    FROM {self.EMBEDDING_TABLE}
                    ORDER BY similarity DESC
                    LIMIT ?
                """
                params = [query_embedding, limit]

            result = conn.execute(sql, params).fetchall()

            if not result:
                return {
                    "status": "no_results",
                    "query": "embedding search",
                    "results": [],
                }

            results = [
                {
                    "uuid": row[0],
                    "field_name": row[1],
                    "project_name": row[2],
                    "source_text": row[3][:200] + "..."
                    if len(row[3]) > 200
                    else row[3],
                    "similarity": round(row[4], 4),
                }
                for row in result
            ]

            return {
                "status": "success",
                "count": len(results),
                "results": results,
            }

        except Exception as e:
            logger.error("[Semantic] search failed | error=%s", e)
            return {
                "status": "error",
                "message": str(e),
                "results": [],
            }

    def close(self) -> None:
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
