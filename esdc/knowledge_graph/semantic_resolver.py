"""Semantic search resolver for ESDC using DuckDB VSS extension.

Provides semantic search capabilities for project_remarks
using vector embeddings and HNSW index for fast similarity search.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Callable
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
        """Get or create DuckDB connection with VSS extension loaded.

        Includes health check to detect stale connections after DB file
        replacement (e.g. during esdc fetch).
        """
        if self._conn is not None:
            try:
                self._conn.execute("SELECT 1")
            except Exception:
                logger.info("[Semantic] Stale connection detected, reconnecting")
                with contextlib.suppress(Exception):
                    self._conn.close()
                self._conn = None

        if self._conn is None:
            self._conn = duckdb.connect(str(self._db_path))
            with contextlib.suppress(Exception):
                self._conn.execute("INSTALL vss")
            self._conn.execute("LOAD vss")
            logger.debug("[Semantic] DuckDB connection established with VSS extension")
        return self._conn

    def build_embeddings_table(self) -> bool:
        """Create embeddings table and HNSW index.

        Creates table to store pre-computed embeddings for
        project_remarks with contextual columns for filtering.
        Detects embedding dimension dynamically from the model.

        Returns:
            True if successful
        """
        conn = self._get_connection()

        try:
            # Check if old table exists with uuid column (migration from old schema)
            try:
                table_info = conn.execute(f"""
                    SELECT column_name
                    FROM information_schema.columns
                    WHERE table_name = '{self.EMBEDDING_TABLE}'
                """).fetchall()

                columns = [col[0] for col in table_info]

                # If table exists but has old schema (uuid column), drop it
                if "uuid" in columns and "project_id" not in columns:
                    logger.info(
                        "[Semantic] dropping old embeddings table for schema migration"
                    )
                    conn.execute(f"DROP TABLE IF EXISTS {self.EMBEDDING_TABLE}")
                    # Also drop HNSW index if exists
                    conn.execute("DROP INDEX IF EXISTS idx_hnsw_embeddings")
            except Exception:
                # Table doesn't exist or other error, proceed normally
                pass

            # Detect embedding dimension by generating a test embedding
            logger.info("[Semantic] detecting embedding dimension from model")
            test_embedding = self._embedding_manager.generate_embedding("test")
            embedding_dim = len(test_embedding)
            logger.info(f"[Semantic] detected embedding dimension: {embedding_dim}")

            # Create table for embeddings with contextual columns
            # Primary key is (project_id, report_year) - one embedding per project per year  # noqa: E501
            # embedding column uses fixed-size array FLOAT[N] required by HNSW index
            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.EMBEDDING_TABLE} (
                    project_id VARCHAR NOT NULL,
                    report_year INTEGER NOT NULL,
                    table_name VARCHAR NOT NULL,
                    field_name VARCHAR,
                    project_name VARCHAR,
                    pod_name VARCHAR,
                    wk_name VARCHAR,
                    province VARCHAR,
                    basin128 VARCHAR,
                    project_class VARCHAR,
                    project_stage VARCHAR,
                    project_level VARCHAR,
                    operator_name VARCHAR,
                    operator_group VARCHAR,
                    wk_subgroup VARCHAR,
                    wk_regionisasi_ngi VARCHAR,
                    wk_area_perwakilan_skkmigas VARCHAR,
                    project_remarks TEXT,
                    embedding FLOAT[{embedding_dim}],
                    PRIMARY KEY (project_id, report_year)
                )
            """)

            logger.info(
                "[Semantic] embeddings table created with dimension %d", embedding_dim
            )
            return True

        except Exception as e:
            logger.error("[Semantic] failed to create table | error=%s", e)
            return False

    def count_documents_with_remarks(
        self,
        table_name: str = "project_resources",
    ) -> int:
        """Count distinct documents with non-empty project_remarks.

        Uses DISTINCT to match what generate_and_store_embeddings will actually
        process (one embedding per project per year, deduplicated across uncert_levels).

        Args:
            table_name: Source table name

        Returns:
            Number of distinct documents with project_remarks
        """
        conn = self._get_connection()
        result = conn.execute(f"""
            SELECT COUNT(DISTINCT project_id || '-' || CAST(report_year AS VARCHAR))
            FROM {table_name}
            WHERE project_remarks IS NOT NULL
                AND LENGTH(project_remarks) > 0
        """).fetchone()
        return result[0] if result else 0

    def generate_and_store_embeddings(
        self,
        table_name: str = "project_resources",
        batch_size: int = 100,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> dict[str, Any]:
        """Generate and store embeddings for all documents.

        Args:
            table_name: Source table name
            batch_size: Number of documents to process per batch
            progress_callback: Optional callback function(progress: int, total: int)

        Returns:
            Dict with status and count
        """
        conn = self._get_connection()

        try:
            # Fetch documents with remarks and contextual data
            # DISTINCT to handle cases where same project appears in multiple rows (e.g., different uncert_levels)  # noqa: E501
            result = conn.execute(f"""
                SELECT DISTINCT
                    project_id,
                    report_year,
                    field_name,
                    project_name,
                    pod_name,
                    wk_name,
                    province,
                    basin128,
                    project_class,
                    project_stage,
                    project_level,
                    operator_name,
                    operator_group,
                    wk_subgroup,
                    wk_regionisasi_ngi,
                    wk_area_perwakilan_skkmigas,
                    COALESCE(project_remarks, '') as project_remarks
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
                # row[16] is project_remarks (the text to embed)
                texts = [row[16] for row in batch]

                # Generate embeddings
                embeddings = self._embedding_manager.generate_embeddings_batch(texts)

                # Store in DuckDB with all contextual columns
                for j, (row, embedding) in enumerate(zip(batch, embeddings)):
                    conn.execute(
                        f"""
                        INSERT OR REPLACE INTO {self.EMBEDDING_TABLE}
                        (project_id, report_year, table_name, field_name, project_name,
                         pod_name, wk_name, province, basin128, project_class,
                         project_stage,
                         project_level, operator_name, operator_group, wk_subgroup,
                         wk_regionisasi_ngi, wk_area_perwakilan_skkmigas,
                         project_remarks, embedding)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                        [
                            row[0],  # project_id
                            row[1],  # report_year
                            table_name,
                            row[2],  # field_name
                            row[3],  # project_name
                            row[4],  # pod_name
                            row[5],  # wk_name
                            row[6],  # province
                            row[7],  # basin128
                            row[8],  # project_class
                            row[9],  # project_stage
                            row[10],  # project_level
                            row[11],  # operator_name
                            row[12],  # operator_group
                            row[13],  # wk_subgroup
                            row[14],  # wk_regionisasi_ngi
                            row[15],  # wk_area_perwakilan_skkmigas
                            row[16],  # project_remarks
                            embedding,
                        ],
                    )

                logger.debug(
                    "[Semantic] batch %d/%d processed",
                    i // batch_size + 1,
                    (total + batch_size - 1) // batch_size,
                )

                # Call progress callback if provided
                if progress_callback:
                    progress_callback(min(i + batch_size, total), total)

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
            # Enable experimental persistence for HNSW on disk-based databases
            conn.execute("SET hnsw_enable_experimental_persistence = true")

            # Drop existing index if any
            conn.execute("DROP INDEX IF EXISTS idx_hnsw_embeddings")

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
        filters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Search by text query (auto-generates embedding).

        Args:
            query: User query text
            limit: Max results
            filters: Optional filters for contextual columns:
                - report_year: Filter by year (int)
                - field_name: Filter by field (ILIKE pattern)
                - pod_name: Filter by POD name (ILIKE pattern)
                - wk_name: Filter by working area (ILIKE pattern)
                - province: Filter by province (ILIKE pattern)
                - basin128: Filter by basin (ILIKE pattern)
                - project_class: Filter by class (ILIKE pattern)
                - project_stage: Filter by stage (ILIKE pattern)
                - project_level: Filter by level (ILIKE pattern)
                - operator_name: Filter by operator (ILIKE pattern)
                - operator_group: Filter by operator group (ILIKE pattern)
                - wk_subgroup: Filter by working area subgroup (ILIKE pattern)
                - wk_regionisasi_ngi: Filter by NGI region (ILIKE pattern)
                -
                wk_area_perwakilan_skkmigas: Filter by SKK Migas region (ILIKE pattern)

        Returns:
            Dict with status and results including all contextual columns
        """
        # Generate query embedding
        query_embedding = self._embedding_manager.generate_embedding(query)

        return self.search_by_embedding(query_embedding, limit, filters)

    def search_by_embedding(
        self,
        query_embedding: list[float],
        limit: int = DEFAULT_LIMIT,
        filters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Search by pre-computed embedding vector.

        Args:
            query_embedding: Query embedding vector
            limit: Max results
            filters: Optional filters for contextual columns:
                - report_year: Filter by year (int)
                - field_name: Filter by field (ILIKE pattern)
                - pod_name: Filter by POD name (ILIKE pattern)
                - wk_name: Filter by working area (ILIKE pattern)
                - province: Filter by province (ILIKE pattern)
                - basin128: Filter by basin (ILIKE pattern)
                - project_class: Filter by class (ILIKE pattern)
                - project_stage: Filter by stage (ILIKE pattern)
                - project_level: Filter by level (ILIKE pattern)
                - operator_name: Filter by operator (ILIKE pattern)
                - operator_group: Filter by operator group (ILIKE pattern)
                - wk_subgroup: Filter by working area subgroup (ILIKE pattern)
                - wk_regionisasi_ngi: Filter by NGI region (ILIKE pattern)
                -
                wk_area_perwakilan_skkmigas: Filter by SKK Migas region (ILIKE pattern)

        Returns:
            Dict with status and results including all contextual columns
        """
        conn = self._get_connection()

        try:
            # Check if embeddings table exists and has data
            check = conn.execute(f"""
                SELECT COUNT(*) FROM {self.EMBEDDING_TABLE}
            """).fetchone()

            if check is None or check[0] == 0:
                return {
                    "status": "not_available",
                    "message": "No embeddings found. Run 'esdc reload' to generate embeddings.",  # noqa: E501
                    "results": [],
                }

            # Build WHERE clause from filters
            where_conditions = ["table_name = 'project_resources'"]
            where_params: list[Any] = []

            if filters:
                if "report_year" in filters:
                    where_conditions.append("report_year = ?")
                    where_params.append(filters["report_year"])
                if "field_name" in filters:
                    where_conditions.append("field_name ILIKE ?")
                    where_params.append(f"%{filters['field_name']}%")
                if "pod_name" in filters:
                    where_conditions.append("pod_name ILIKE ?")
                    where_params.append(f"%{filters['pod_name']}%")
                if "wk_name" in filters:
                    where_conditions.append("wk_name ILIKE ?")
                    where_params.append(f"%{filters['wk_name']}%")
                if "province" in filters:
                    where_conditions.append("province ILIKE ?")
                    where_params.append(f"%{filters['province']}%")
                if "basin128" in filters:
                    where_conditions.append("basin128 ILIKE ?")
                    where_params.append(f"%{filters['basin128']}%")
                if "project_class" in filters:
                    where_conditions.append("project_class ILIKE ?")
                    where_params.append(f"%{filters['project_class']}%")
                if "project_stage" in filters:
                    where_conditions.append("project_stage ILIKE ?")
                    where_params.append(f"%{filters['project_stage']}%")
                if "project_level" in filters:
                    where_conditions.append("project_level ILIKE ?")
                    where_params.append(f"%{filters['project_level']}%")
                if "operator_name" in filters:
                    where_conditions.append("operator_name ILIKE ?")
                    where_params.append(f"%{filters['operator_name']}%")
                if "operator_group" in filters:
                    where_conditions.append("operator_group ILIKE ?")
                    where_params.append(f"%{filters['operator_group']}%")
                if "wk_subgroup" in filters:
                    where_conditions.append("wk_subgroup ILIKE ?")
                    where_params.append(f"%{filters['wk_subgroup']}%")
                if "wk_regionisasi_ngi" in filters:
                    where_conditions.append("wk_regionisasi_ngi ILIKE ?")
                    where_params.append(f"%{filters['wk_regionisasi_ngi']}%")
                if "wk_area_perwakilan_skkmigas" in filters:
                    where_conditions.append("wk_area_perwakilan_skkmigas ILIKE ?")
                    where_params.append(f"%{filters['wk_area_perwakilan_skkmigas']}%")

            where_clause = " AND ".join(where_conditions)

            # Build query using manual cosine similarity calculation
            # DuckDB's array_cosine_similarity requires fixed-size arrays
            # We compute it manually: dot_product / (norm_a * norm_b)
            sql = f"""
                SELECT
                    project_id,
                    report_year,
                    field_name,
                    project_name,
                    pod_name,
                    wk_name,
                    province,
                    basin128,
                    project_class,
                    project_stage,
                    project_level,
                    operator_name,
                    operator_group,
                    wk_subgroup,
                    wk_regionisasi_ngi,
                    wk_area_perwakilan_skkmigas,
                    project_remarks,
                    list_dot_product(embedding, ?::FLOAT[]) /
                    (sqrt(list_dot_product(embedding,
                    embedding))
                    * sqrt(list_dot_product(?::FLOAT[], ?::FLOAT[]))
                    as similarity
                FROM {self.EMBEDDING_TABLE}
                WHERE {where_clause}
                ORDER BY similarity DESC
                LIMIT ?
            """
            params = [
                query_embedding,
                query_embedding,
                query_embedding,
                *where_params,
                limit,
            ]

            result = conn.execute(sql, params).fetchall()

            if not result:
                return {
                    "status": "no_results",
                    "query": "embedding search",
                    "results": [],
                }

            results = [
                {
                    "project_id": row[0],
                    "report_year": row[1],
                    "field_name": row[2],
                    "project_name": row[3],
                    "pod_name": row[4],
                    "wk_name": row[5],
                    "province": row[6],
                    "basin128": row[7],
                    "project_class": row[8],
                    "project_stage": row[9],
                    "project_level": row[10],
                    "operator_name": row[11],
                    "operator_group": row[12],
                    "wk_subgroup": row[13],
                    "wk_regionisasi_ngi": row[14],
                    "wk_area_perwakilan_skkmigas": row[15],
                    "project_remarks": row[16][:200] + "..."
                    if len(row[16]) > 200
                    else row[16],
                    "similarity": round(row[17], 4),
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
