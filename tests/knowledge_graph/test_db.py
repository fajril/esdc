"""Tests for LadybugDB connection and basic operations."""

from esdc.knowledge_graph.db import LadybugDB


def test_ladybug_db_connection():
    """Test basic connection to LadybugDB."""
    db = LadybugDB()
    assert db.is_connected() is True


def test_create_document_node():
    """Test creating a document node."""
    db = LadybugDB()
    doc_id = db.create_document(
        title="Test Document",
        doc_type="MoM",
        date="2024-01-15",
        file_path="/test/doc.md",
        is_timeless=False,
    )
    assert doc_id is not None
    assert len(doc_id) > 0
