"""Mock LadybugDB for unit testing without real_ladybug dependency."""

import uuid
from pathlib import Path


class MockLadybugDB:
    """Mock LadybugDB for testing without real_ladybug dependency."""

    def __init__(self, db_path: Path | None = None):
        """Initialize mock database.

        Args:
            db_path: Path to database file (ignored in mock)
        """
        self.db_path = Path(db_path) if db_path else Path("/tmp/mock.lbug")
        self._connected = True
        self._documents: dict[str, dict] = {}
        self._chunks: dict[str, dict] = {}
        self._entities: dict[str, dict] = {}
        self._schema_initialized = False
        self._init_schema()

    def _init_schema(self):
        """Mock schema initialization."""
        self._schema_initialized = True

    def is_connected(self) -> bool:
        """Check if database is connected."""
        return self._connected

    def is_available(self) -> bool:
        """Check if LadybugDB is available (mock always True)."""
        return True

    def create_document(
        self,
        title: str,
        doc_type: str,
        date: str | None,
        file_path: str,
        is_timeless: bool = False,
        extraction_confidence: float = 0.0,
    ) -> str:
        """Create a document node.

        Args:
            title: Document title
            doc_type: Document type (MoM, DEFINITION, etc.)
            date: Document date
            file_path: Path to source file
            is_timeless: Whether document has no date
            extraction_confidence: Confidence score of extraction

        Returns:
            Document ID
        """
        doc_id = str(uuid.uuid4())
        self._documents[doc_id] = {
            "id": doc_id,
            "title": title,
            "doc_type": doc_type,
            "date": date,
            "file_path": file_path,
            "is_timeless": is_timeless,
            "extraction_confidence": extraction_confidence,
        }
        return doc_id

    def create_chunk(
        self,
        document_id: str,
        chunk_index: int,
        text: str,
        section_type: str,
        is_summary: bool = False,
        embedding: list[float] | None = None,
    ) -> str:
        """Create a chunk node.

        Args:
            document_id: Parent document ID
            chunk_index: Index of chunk in document
            text: Chunk text content
            section_type: Type of section (TITLE, CONTENT, etc.)
            is_summary: Whether this is a summary chunk
            embedding: Vector embedding (optional for mock)

        Returns:
            Chunk ID
        """
        chunk_id = str(uuid.uuid4())
        self._chunks[chunk_id] = {
            "id": chunk_id,
            "document_id": document_id,
            "chunk_index": chunk_index,
            "text": text,
            "section_type": section_type,
            "is_summary": is_summary,
            "embedding": embedding,
        }
        return chunk_id

    def link_entity(
        self,
        document_id: str,
        entity_name: str,
        entity_type: str,
        esdc_id: str | None = None,
    ) -> str:
        """Link document to entity.

        Args:
            document_id: Document ID
            entity_name: Entity name
            entity_type: Entity type (project, field, wk, operator, pod)
            esdc_id: ESDC database ID if resolved

        Returns:
            Entity ID
        """
        entity_id = str(uuid.uuid4())
        self._entities[entity_id] = {
            "id": entity_id,
            "document_id": document_id,
            "name": entity_name,
            "type": entity_type,
            "esdc_id": esdc_id,
        }
        return entity_id

    def search_similar(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        entity_filter: str | None = None,
        doc_type_filter: str | None = None,
    ) -> list[dict]:
        """Search for similar chunks (mock returns empty).

        Args:
            query_embedding: Query vector
            top_k: Number of results
            entity_filter: Filter by entity name
            doc_type_filter: Filter by document type

        Returns:
            List of matching chunks (empty in mock)
        """
        return []

    def get_entity_documents(
        self,
        entity_name: str,
        entity_type: str,
        temporal: str = "latest",
    ) -> list[dict]:
        """Get documents for entity (mock returns empty).

        Args:
            entity_name: Entity name
            entity_type: Entity type
            temporal: Temporal filter ('latest', 'all')

        Returns:
            List of documents (empty in mock)
        """
        return []

    def clear(self):
        """Clear all data in mock database."""
        self._documents.clear()
        self._chunks.clear()
        self._entities.clear()
