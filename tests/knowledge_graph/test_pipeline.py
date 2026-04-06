from pathlib import Path
from unittest.mock import MagicMock

from esdc.knowledge_graph.pipeline import IngestionPipeline


def test_ingest_document_basic():
    """Test basic ingestion pipeline."""
    pipeline = IngestionPipeline()

    # Mock all dependencies
    pipeline._load_document = MagicMock(return_value="content")
    pipeline._generate_doc_id = MagicMock(return_value="doc-001")
    pipeline._extract_structure = MagicMock(
        return_value={
            "doc_type": "MoM",
            "sections": {"title": {"text": "Test"}},
            "is_timeless": False,
        }
    )
    pipeline._extract_entities = MagicMock(return_value=[])
    pipeline._chunk_document = MagicMock(return_value=[])
    pipeline._store_in_ladybug = MagicMock(return_value="doc-001")

    result = pipeline.ingest(Path("test.md"))

    assert result["success"] is True
    assert result["doc_id"] == "doc-001"


def test_ingest_analyze_mode():
    """Test analyze mode doesn't store."""
    pipeline = IngestionPipeline()

    pipeline._load_document = MagicMock(return_value="content")
    pipeline._generate_doc_id = MagicMock(return_value="doc-001")
    pipeline._extract_structure = MagicMock(
        return_value={
            "doc_type": "MoM",
            "sections": {},
            "is_timeless": False,
        }
    )

    result = pipeline.ingest(Path("test.md"), analyze_only=True)

    assert result["success"] is True
    assert result["mode"] == "analyze"
    assert "analysis" in result


def test_generate_doc_id():
    """Test document ID generation."""
    pipeline = IngestionPipeline()

    # Create temp file
    temp_file = Path("/tmp/test_doc_123.md")
    temp_file.write_text("test content")

    doc_id = pipeline._generate_doc_id(temp_file)

    assert len(doc_id) == 32  # MD5 hash
    assert isinstance(doc_id, str)

    # Cleanup
    temp_file.unlink()
