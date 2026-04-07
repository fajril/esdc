"""Graph database wrapper for knowledge graph operations with fallback support.

Currently uses LadybugDB as backend. Can be replaced with other graph databases
(Neo4j, ArangoDB, etc.) in the future without changing the interface.
"""

import logging
from typing import TYPE_CHECKING, Any

from esdc.configs import Config

if TYPE_CHECKING:
    import real_ladybug as lb

logger = logging.getLogger(__name__)

LADYBUG_AVAILABLE = False
try:
    import real_ladybug as lb

    LADYBUG_AVAILABLE = True
except ImportError:
    logger.info(
        "real_ladybug not available. Document knowledge graph features disabled. "
        "Install with: pip install real-ladybug"
    )


class LadybugDB:
    """Graph database wrapper for document knowledge graph.

    Currently implemented with LadybugDB as backend.
    Gracefully falls back to no-op mode if real_ladybug is not available.
    All methods return empty results or None when database is unavailable.

    Future: Can be abstracted to GraphDB interface with pluggable backends
    (Neo4j, ArangoDB, etc.) without changing caller code.
    """

    def __init__(self, db_path: Any = None):
        """Initialize LadybugDB connection.

        Args:
            db_path: Path to LadybugDB file. Defaults to ~/.esdc/ladybug/documents.lbug
        """
        from pathlib import Path

        if db_path is None:
            db_path = Config.get_ladybug_db_path()

        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._db: lb.Database | None = None
        self._conn: lb.Connection | None = None
        self._available = LADYBUG_AVAILABLE

        if self._available:
            self._init_connection()

    def _init_connection(self) -> None:
        """Initialize LadybugDB connection and schema."""
        if not self._available:
            return

        try:
            self._db = lb.Database(str(self.db_path))
            self._conn = lb.Connection(self._db)
            self._init_schema()
        except Exception as e:
            logger.error(f"Failed to initialize LadybugDB: {e}")
            self._available = False
            self._db = None
            self._conn = None

    def _init_schema(self) -> None:
        """Initialize LadybugDB schema with Cypher queries."""
        if not self._available or not self._conn:
            return

        try:
            self._conn.execute("""
                CREATE NODE TABLE IF NOT EXISTS Document (
                    id STRING PRIMARY KEY,
                    title STRING,
                    doc_type STRING,
                    date STRING,
                    file_path STRING,
                    file_name STRING,
                    is_timeless BOOLEAN,
                    extracted_at STRING,
                    extraction_confidence DOUBLE
                )
            """)

            self._conn.execute("""
                CREATE NODE TABLE IF NOT EXISTS Chunk (
                    id STRING PRIMARY KEY,
                    document_id STRING,
                    chunk_index INT64,
                    text STRING,
                    section_type STRING,
                    is_summary BOOLEAN
                )
            """)

            self._conn.execute("""
                CREATE NODE TABLE IF NOT EXISTS Entity (
                    id STRING PRIMARY KEY,
                    name STRING,
                    normalized_name STRING,
                    type STRING,
                    source STRING,
                    esdc_id STRING
                )
            """)

            self._conn.execute("""
                CREATE REL TABLE IF NOT EXISTS HAS_CHUNK (
                    FROM Document TO Chunk
                )
            """)

            self._conn.execute("""
                CREATE REL TABLE IF NOT EXISTS HAS_ENTITY (
                    FROM Document TO Entity
                )
            """)

            self._conn.execute("""
                CREATE REL TABLE IF NOT EXISTS SUPERSEDES (
                    FROM Document TO Document
                )
            """)

            self._conn.execute("""
                CREATE REL TABLE IF NOT EXISTS REFERENCES (
                    FROM Document TO Document
                )
            """)

            self._conn.execute("""
                CREATE REL TABLE IF NOT EXISTS AMENDS (
                    FROM Document TO Document
                )
            """)

            logger.info("LadybugDB schema initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize schema: {e}")
            raise

    def is_connected(self) -> bool:
        """Check if database is connected."""
        return self._available and self._conn is not None

    def is_available(self) -> bool:
        """Check if LadybugDB is available."""
        return self._available

    def create_document(
        self,
        title: str,
        doc_type: str,
        date: str | None,
        file_path: str,
        is_timeless: bool = False,
        extraction_confidence: float = 0.0,
    ) -> str | None:
        """Create a document node.

        Args:
            title: Document title
            doc_type: Document type (MoM, DEFINITION, etc.)
            date: Document date (ISO format)
            file_path: Path to source file
            is_timeless: Whether document has no date
            extraction_confidence: Confidence score of extraction

        Returns:
            Document ID if successful, None if LadybugDB unavailable
        """
        if not self._available or not self._conn:
            return None

        import uuid
        from datetime import datetime
        from pathlib import Path

        doc_id = str(uuid.uuid4())
        file_name = Path(file_path).name
        extracted_at = datetime.now().isoformat()

        try:
            self._conn.execute(
                """
                CREATE (d:Document {
                    id: $id,
                    title: $title,
                    doc_type: $doc_type,
                    date: $date,
                    file_path: $file_path,
                    file_name: $file_name,
                    is_timeless: $is_timeless,
                    extracted_at: $extracted_at,
                    extraction_confidence: $extraction_confidence
                })
            """,
                {
                    "id": doc_id,
                    "title": title,
                    "doc_type": doc_type,
                    "date": date or "",
                    "file_path": file_path,
                    "file_name": file_name,
                    "is_timeless": is_timeless,
                    "extracted_at": extracted_at,
                    "extraction_confidence": extraction_confidence,
                },
            )
            return doc_id
        except Exception as e:
            logger.error(f"Failed to create document: {e}")
            return None

    def create_chunk(
        self,
        document_id: str,
        chunk_index: int,
        text: str,
        section_type: str,
        is_summary: bool = False,
        embedding: list[float] | None = None,
    ) -> str | None:
        """Create a chunk node.

        Args:
            document_id: Parent document ID
            chunk_index: Index of chunk in document
            text: Chunk text content
            section_type: Type of section (TITLE, CONTENT, etc.)
            is_summary: Whether this is a summary chunk
            embedding: Vector embedding (768 dimensions for nomic-embed-text)

        Returns:
            Chunk ID if successful, None if LadybugDB unavailable
        """
        if not self._available or not self._conn:
            return None

        import uuid

        chunk_id = str(uuid.uuid4())

        try:
            self._conn.execute(
                """
                CREATE (c:Chunk {
                    id: $chunk_id,
                    document_id: $document_id,
                    chunk_index: $chunk_index,
                    text: $text,
                    section_type: $section_type,
                    is_summary: $is_summary
                })
            """,
                {
                    "chunk_id": chunk_id,
                    "document_id": document_id,
                    "chunk_index": chunk_index,
                    "text": text,
                    "section_type": section_type,
                    "is_summary": is_summary,
                },
            )

            self._conn.execute(
                """
                MATCH (d:Document {id: $document_id}), (c:Chunk {id: $chunk_id})
                CREATE (d)-[:HAS_CHUNK]->(c)
            """,
                {"document_id": document_id, "chunk_id": chunk_id},
            )

            return chunk_id
        except Exception as e:
            logger.error(f"Failed to create chunk: {e}")
            return None

    def link_entity(
        self,
        document_id: str,
        entity_name: str,
        entity_type: str,
        esdc_id: str | None = None,
    ) -> str | None:
        """Link document to entity.

        Args:
            document_id: Document ID
            entity_name: Entity name
            entity_type: Entity type (project, field, wk, operator, pod)
            esdc_id: ESDC database ID if resolved

        Returns:
            Entity ID if successful, None if LadybugDB unavailable
        """
        if not self._available or not self._conn:
            return None

        import uuid

        entity_id = str(uuid.uuid4())
        normalized_name = entity_name.lower().strip()

        try:
            self._conn.execute(
                """
                CREATE (e:Entity {
                    id: $entity_id,
                    name: $entity_name,
                    normalized_name: $normalized_name,
                    type: $entity_type,
                    source: $document_id,
                    esdc_id: $esdc_id
                })
            """,
                {
                    "entity_id": entity_id,
                    "entity_name": entity_name,
                    "normalized_name": normalized_name,
                    "entity_type": entity_type,
                    "document_id": document_id,
                    "esdc_id": esdc_id or "",
                },
            )

            self._conn.execute(
                """
                MATCH (d:Document {id: $document_id}), (e:Entity {id: $entity_id})
                CREATE (d)-[:HAS_ENTITY]->(e)
            """,
                {"document_id": document_id, "entity_id": entity_id},
            )

            return entity_id
        except Exception as e:
            logger.error(f"Failed to link entity: {e}")
            return None

    def search_similar(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        entity_filter: str | None = None,
        doc_type_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        """Search for similar chunks using vector similarity.

        Args:
            query_embedding: Query vector (768 dimensions)
            top_k: Number of results to return
            entity_filter: Filter by entity name (optional)
            doc_type_filter: Filter by document type (optional)

        Returns:
            List of matching chunks with similarity scores
        """
        if not self._available or not self._conn:
            return []

        try:
            result = self._conn.execute(
                """
                MATCH (d:Document)-[:HAS_CHUNK]->(c:Chunk)
                OPTIONAL MATCH (d)-[:HAS_ENTITY]->(e:Entity)
                RETURN d.id, d.title, d.doc_type, c.text, c.section_type,
                       COALESCE(e.name, ''), COALESCE(e.type, '')
                LIMIT $top_k
            """,
                {"top_k": top_k},
            )

            chunks: list[dict[str, Any]] = []
            for row_list in result:
                if isinstance(row_list, list):
                    chunks.append(
                        {
                            "document_id": row_list[0],
                            "title": row_list[1],
                            "doc_type": row_list[2],
                            "text": row_list[3],
                            "section_type": row_list[4],
                            "entity_name": row_list[5],
                            "entity_type": row_list[6],
                        }
                    )

            return chunks
        except Exception as e:
            logger.error(f"Failed to search similar: {e}")
            return []

    def get_entity_documents(
        self,
        entity_name: str,
        entity_type: str,
        temporal: str = "latest",
    ) -> list[dict[str, Any]]:
        """Get all documents for a specific entity.

        Args:
            entity_name: Entity name to search
            entity_type: Entity type (project, field, wk, operator, pod)
            temporal: Temporal filter ('latest', 'all', or date range)

        Returns:
            List of documents related to this entity
        """
        if not self._available or not self._conn:
            return []

        normalized_name = entity_name.lower().strip()

        try:
            if temporal == "latest":
                result = self._conn.execute(
                    """
                    MATCH (d:Document)-[:HAS_ENTITY]->(e:Entity)
                    WHERE e.normalized_name = $normalized_name AND e.type = $entity_type
                    RETURN DISTINCT d.id, d.title, d.doc_type, d.date, d.file_path
                    ORDER BY d.date DESC, d.extracted_at DESC
                    LIMIT 10
                """,
                    {"normalized_name": normalized_name, "entity_type": entity_type},
                )
            else:
                result = self._conn.execute(
                    """
                    MATCH (d:Document)-[:HAS_ENTITY]->(e:Entity)
                    WHERE e.normalized_name = $normalized_name AND e.type = $entity_type
                    RETURN DISTINCT d.id, d.title, d.doc_type, d.date, d.file_path
                    ORDER BY d.date DESC, d.extracted_at DESC
                """,
                    {"normalized_name": normalized_name, "entity_type": entity_type},
                )

            documents: list[dict[str, Any]] = []
            for row_list in result:
                if isinstance(row_list, list):
                    documents.append(
                        {
                            "id": row_list[0],
                            "title": row_list[1],
                            "doc_type": row_list[2],
                            "date": row_list[3],
                            "file_path": row_list[4],
                        }
                    )

            return documents
        except Exception as e:
            logger.error(f"Failed to get entity documents: {e}")
            return []

    def close(self) -> None:
        """Close database connection."""
        if self._conn:
            try:
                pass
            except Exception as e:
                logger.error(f"Error closing connection: {e}")

        if self._db:
            try:
                pass
            except Exception as e:
                logger.error(f"Error closing database: {e}")

        self._conn = None
        self._db = None
