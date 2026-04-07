"""Tests for GraphDB connection and basic operations."""

from pathlib import Path

import pytest

from esdc.knowledge_graph.graph_db import LadybugDB
from tests.knowledge_graph.mock_ladybug import MockLadybugDB


class TestMockLadybugDB:
    """Tests using MockLadybugDB (no real_ladybug dependency)."""

    def test_mock_connection(self):
        """Test mock database connection."""
        db = MockLadybugDB()
        assert db.is_connected() is True
        assert db.is_available() is True

    def test_mock_create_document(self):
        """Test creating a document in mock database."""
        db = MockLadybugDB()
        doc_id = db.create_document(
            title="Test Document",
            doc_type="MoM",
            date="2024-01-15",
            file_path="/test/doc.md",
            is_timeless=False,
        )
        assert doc_id is not None
        assert len(doc_id) > 0
        assert doc_id in db._documents

    def test_mock_create_chunk(self):
        """Test creating a chunk in mock database."""
        db = MockLadybugDB()
        doc_id = db.create_document(
            title="Test",
            doc_type="MoM",
            date="2024-01-15",
            file_path="/test/doc.md",
        )
        chunk_id = db.create_chunk(
            document_id=doc_id,
            chunk_index=0,
            text="Test content",
            section_type="CONTENT",
            is_summary=False,
        )
        assert chunk_id is not None
        assert chunk_id in db._chunks

    def test_mock_link_entity(self):
        """Test linking document to entity."""
        db = MockLadybugDB()
        doc_id = db.create_document(
            title="Test",
            doc_type="MoM",
            date="2024-01-15",
            file_path="/test/doc.md",
        )
        entity_id = db.link_entity(
            document_id=doc_id,
            entity_name="Abadi",
            entity_type="project",
            esdc_id="123",
        )
        assert entity_id is not None
        assert entity_id in db._entities

    def test_mock_search_returns_empty(self):
        """Test that mock search returns empty results."""
        db = MockLadybugDB()
        results = db.search_similar(query_embedding=[0.1] * 768, top_k=5)
        assert results == []

    def test_mock_get_entity_documents_returns_empty(self):
        """Test that mock get_entity_documents returns empty."""
        db = MockLadybugDB()
        results = db.get_entity_documents("Abadi", "project")
        assert results == []

    def test_mock_clear(self):
        """Test clearing mock database."""
        db = MockLadybugDB()
        db.create_document("Test", "MoM", "2024-01-15", "/test.md")
        assert len(db._documents) > 0
        db.clear()
        assert len(db._documents) == 0

    def test_mock_custom_path(self):
        """Test mock database with custom path."""
        custom_path = Path("/custom/mock.lbug")
        db = MockLadybugDB(db_path=custom_path)
        assert db.db_path == custom_path


class TestLadybugDBWithFallback:
    """Tests for LadybugDB with fallback behavior."""

    def test_ladybug_db_initialization(self):
        """Test LadybugDB initialization (uses mock when real_ladybug not available)."""
        db = LadybugDB()
        assert db is not None

    def test_ladybug_db_is_connected(self):
        """Test is_connected method."""
        db = LadybugDB()
        result = db.is_connected()
        assert isinstance(result, bool)

    def test_ladybug_db_create_document(self):
        """Test create_document method with fallback."""
        db = LadybugDB()
        doc_id = db.create_document(
            title="Test Document",
            doc_type="MoM",
            date="2024-01-15",
            file_path="/test/doc.md",
            is_timeless=False,
        )
        if db.is_available():
            assert doc_id is not None
        else:
            assert doc_id is None or doc_id == ""

    def test_ladybug_db_create_chunk(self):
        """Test create_chunk method with fallback."""
        db = LadybugDB()
        chunk_id = db.create_chunk(
            document_id="test-doc-id",
            chunk_index=0,
            text="Test content",
            section_type="CONTENT",
            is_summary=False,
        )
        if db.is_available():
            assert chunk_id is not None
        else:
            assert chunk_id is None or chunk_id == ""

    def test_ladybug_db_search_similar(self):
        """Test search_similar method with fallback."""
        db = LadybugDB()
        results = db.search_similar(query_embedding=[0.1] * 768, top_k=5)
        assert isinstance(results, list)

    def test_ladybug_db_get_entity_documents(self):
        """Test get_entity_documents method with fallback."""
        db = LadybugDB()
        results = db.get_entity_documents("Abadi", "project")
        assert isinstance(results, list)

    def test_ladybug_db_is_available(self):
        """Test is_available method."""
        db = LadybugDB()
        result = db.is_available()
        assert isinstance(result, bool)
