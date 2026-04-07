import hashlib
import logging
from pathlib import Path
from typing import Any

from esdc.configs import Config
from esdc.knowledge_graph.chunking import DocumentChunker
from esdc.knowledge_graph.document_types import DocumentTypeDetector
from esdc.knowledge_graph.embeddings import EmbeddingGenerator
from esdc.knowledge_graph.entity_resolver import EntityResolver
from esdc.knowledge_graph.hybrid_extractor import HybridEntityExtractor
from esdc.knowledge_graph.llm_extraction import LLMExtractor
from esdc.knowledge_graph.master_entities import MasterEntityLoader
from esdc.knowledge_graph.semantic_matcher import SemanticEntityMatcher

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """Orchestrates the complete document ingestion pipeline."""

    def __init__(
        self,
        db=None,
        llm=None,
        detector: DocumentTypeDetector | None = None,
        resolver: EntityResolver | None = None,
        chunker: DocumentChunker | None = None,
        use_hybrid_extraction: bool = True,
    ):
        """Initialize the pipeline with optional dependencies.

        Args:
            db: Database connection (for future use)
            llm: LangChain chat model for LLM extraction
            detector: DocumentTypeDetector instance
            resolver: EntityResolver instance
            chunker: DocumentChunker instance
            use_hybrid_extraction: Use hybrid semantic+LLM entity extraction
        """
        self.db = db
        self.llm = llm
        self.detector = detector or DocumentTypeDetector(llm=llm)
        self.resolver = resolver or EntityResolver(db_connection=db)
        self.chunker = chunker or DocumentChunker()
        self.use_hybrid_extraction = use_hybrid_extraction

        # Initialize hybrid extraction components
        if use_hybrid_extraction:
            self.embedding_gen = EmbeddingGenerator()
            self.entity_loader = MasterEntityLoader(db_connection=db)
            self.semantic_matcher = SemanticEntityMatcher(
                embedding_generator=self.embedding_gen,
                threshold=Config.get_entity_match_threshold(),
            )
            self.hybrid_extractor = HybridEntityExtractor(
                semantic_matcher=self.semantic_matcher,
                llm=llm,
                top_k=20,
            )
        else:
            self.embedding_gen = None
            self.entity_loader = None
            self.semantic_matcher = None
            self.hybrid_extractor = None

    def ingest(
        self,
        file_path: Path,
        doc_type_override: str | None = None,
        interactive: bool = False,
        analyze_only: bool = False,
    ) -> dict[str, Any]:
        """Ingest a document through the pipeline.

        Args:
            file_path: Path to the document file
            doc_type_override: Override document type detection
            interactive: Enable interactive mode (future use)
            analyze_only: Only analyze, don't store

        Returns:
            Dictionary with:
            - success: Whether ingestion succeeded
            - doc_id: Generated document ID (if stored)
            - mode: "ingest" or "analyze"
            - analysis: Analysis results (if analyze_only=True)
            - error: Error message (if failed)
        """
        try:
            content = self._load_document(file_path)
            doc_id = self._generate_doc_id(file_path)

            structure = self._extract_structure(
                content, file_path.name, doc_type_override=doc_type_override
            )

            entities = self._extract_entities(content, structure)

            if analyze_only:
                return {
                    "success": True,
                    "mode": "analyze",
                    "analysis": {
                        "doc_id": doc_id,
                        "doc_type": structure.get("doc_type"),
                        "is_timeless": structure.get("is_timeless", False),
                        "entities": entities,
                        "sections": list(structure.get("sections", {}).keys()),
                    },
                }

            doc_structure = {
                "id": doc_id,
                "title": structure.get("title", file_path.stem),
                "sections": structure.get("sections", {}),
            }

            chunks = self._chunk_document(doc_structure)

            stored_doc_id = self._store_in_ladybug(
                doc_id=doc_id,
                structure=structure,
                entities=entities,
                chunks=chunks,
            )

            return {
                "success": True,
                "doc_id": stored_doc_id,
                "mode": "ingest",
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
            }

    def _load_document(self, file_path: Path) -> str:
        """Load document content from file.

        Args:
            file_path: Path to document file

        Returns:
            Document content as string

        Raises:
            FileNotFoundError: If file doesn't exist
        """
        if not file_path.exists():
            raise FileNotFoundError(f"Document not found: {file_path}")
        return file_path.read_text(encoding="utf-8")

    def _generate_doc_id(self, file_path: Path) -> str:
        """Generate document ID from file content.

        Args:
            file_path: Path to document file

        Returns:
            MD5 hash of file content
        """
        content = file_path.read_text(encoding="utf-8")
        return hashlib.md5(content.encode()).hexdigest()

    def _extract_structure(
        self,
        content: str,
        filename: str,
        doc_type_override: str | None = None,
    ) -> dict[str, Any]:
        """Extract document structure using type detector.

        Args:
            content: Document content
            filename: Document filename
            doc_type_override: Override detected type

        Returns:
            Document structure dictionary
        """
        if doc_type_override:
            return {
                "doc_type": doc_type_override,
                "confidence": 1.0,
                "sections": {},
                "is_timeless": False,
            }

        result = self.detector.detect(content, filename)

        if self.llm:
            try:
                extractor = LLMExtractor(llm=self.llm)
                llm_result = extractor.extract_structure(content, filename)

                if llm_result.get("confidence", 0) > result.get("confidence", 0):
                    result = llm_result
            except (ValueError, KeyError):
                pass

        return result

    def _extract_entities(
        self, content: str, structure: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Extract entities from document.

        Uses hybrid extraction (semantic + LLM) if enabled, otherwise falls back
        to structure-based extraction.

        Args:
            content: Full document content
            structure: Document structure dictionary

        Returns:
            List of resolved entities
        """
        if self.use_hybrid_extraction and self.entity_loader and self.hybrid_extractor:
            try:
                # Load master entities with embeddings (cached)
                master_entities, master_embeddings = (
                    self.entity_loader.load_all_with_embeddings()
                )

                logger.info(
                    f"Loaded {sum(len(v) for v in master_entities.values())} "
                    "master entities"
                )

                # Hybrid extraction: semantic + LLM verification
                entities = self.hybrid_extractor.extract(
                    document_text=content,
                    master_entities=master_entities,
                    master_embeddings=master_embeddings,
                )

                logger.info(f"Extracted {len(entities)} entities via hybrid method")
                return entities

            except Exception as e:
                logger.error(
                    f"Hybrid extraction failed: {e}, falling back to structure-based"
                )

        # Fallback: structure-based entity extraction
        entities = []

        text_entities = structure.get("entities", [])
        for entity_name in text_entities:
            entity = self.resolver.resolve(entity_name)
            entities.append(entity)

        return entities

    def _chunk_document(self, doc_structure: dict[str, Any]) -> list[dict[str, Any]]:
        """Chunk document using hybrid strategy.

        Args:
            doc_structure: Document structure with id, title, and sections

        Returns:
            List of document chunks
        """
        return self.chunker.chunk_hybrid(doc_structure)

    def _store_in_ladybug(
        self,
        doc_id: str,
        structure: dict[str, Any],
        entities: list[dict[str, Any]],
        chunks: list[dict[str, Any]],
    ) -> str:
        """Store document in LadybugDB (placeholder).

        Args:
            doc_id: Document ID
            structure: Document structure
            entities: Extracted entities
            chunks: Document chunks

        Returns:
            Document ID
        """
        return doc_id
