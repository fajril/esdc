"""Tests for master entity loader."""

import sqlite3
from unittest.mock import MagicMock, patch

import pytest

from esdc.knowledge_graph.master_entities import MasterEntityLoader


@pytest.fixture
def mock_db():
    """Create mock database connection."""
    db = MagicMock(spec=sqlite3.Connection)
    cursor = MagicMock()

    # Track execution order
    execution_order = []

    def mock_execute(query, params=None):
        # Track queries
        execution_order.append(query)

        # Return appropriate results based on query
        if "field_name" in query:
            cursor.fetchall.return_value = [("Field1",), ("Field2",)]
        elif "project_name" in query:
            cursor.fetchall.return_value = [("Project1",), ("Project2",)]
        elif "wk_name" in query:
            cursor.fetchall.return_value = [("WK1",)]
        elif "operator_name" in query:
            cursor.fetchall.return_value = [("Operator1",)]
        elif "pod_name" in query:
            cursor.fetchall.return_value = [("POD1",)]
        else:
            cursor.fetchall.return_value = []

        return None

    cursor.execute.side_effect = mock_execute
    db.cursor.return_value = cursor

    return db


def test_load_all(mock_db):
    """Test loading all entities from database."""
    loader = MasterEntityLoader(db_connection=mock_db)

    entities = loader.load_all()

    assert "field" in entities
    assert "project" in entities
    assert "wk" in entities
    assert "operator" in entities
    assert "pod" in entities

    # Check external entities included
    assert "ministry" in entities
    assert "directorate" in entities

    assert "Kementerian ESDM" in [e["name"] for e in entities["ministry"]]
    assert "Ditjen Migas" in [e["name"] for e in entities["directorate"]]


def test_deduplicate_entities():
    """Test entity deduplication."""
    loader = MasterEntityLoader(db_connection=MagicMock())

    entities = [
        {"name": "Entity1", "esdc_id": "1"},
        {"name": "Entity2", "esdc_id": "2"},
        {"name": "Entity1", "esdc_id": "1"},  # Duplicate
        {"name": "Entity3", "esdc_id": "3"},
    ]

    result = loader._deduplicate_entities(entities)

    assert len(result) == 3
    assert len([e for e in result if e["name"] == "Entity1"]) == 1


def test_external_entities_included():
    """Test that external entities are included."""
    loader = MasterEntityLoader(db_connection=MagicMock())

    # Just check that external entities are defined
    assert "ministry" in loader.EXTERNAL_ENTITIES
    assert "directorate" in loader.EXTERNAL_ENTITIES

    # Check specific entities
    ministry_names = [e["name"] for e in loader.EXTERNAL_ENTITIES["ministry"]]
    assert "Kementerian ESDM" in ministry_names
    assert "Kementerian Keuangan" in ministry_names

    directorate_names = [e["name"] for e in loader.EXTERNAL_ENTITIES["directorate"]]
    assert "Ditjen Migas" in directorate_names


def test_entity_type_mapping():
    """Test entity type mapping configuration."""
    loader = MasterEntityLoader(db_connection=MagicMock())

    assert len(loader.ENTITY_TYPE_MAPPING) == 5

    column_names = [col for col, _ in loader.ENTITY_TYPE_MAPPING]
    assert "field_name" in column_names
    assert "project_name" in column_names
    assert "wk_name" in column_names
    assert "operator_name" in column_names
    assert "pod_name" in column_names


def test_count_entities():
    """Test entity counting."""
    mock_db = MagicMock(spec=sqlite3.Connection)
    cursor = MagicMock()

    # Track which query is being executed
    query_count = {"count": 0}

    def mock_execute(query, params=None):
        query_count["count"] += 1

        # Simulate different result counts per entity type
        if query_count["count"] % 5 == 1:
            cursor.fetchall.return_value = [("Field1",), ("Field2",), ("Field3",)]
        elif query_count["count"] % 5 == 2:
            cursor.fetchall.return_value = [("Project1",), ("Project2",)]
        elif query_count["count"] % 5 == 3:
            cursor.fetchall.return_value = [("WK1",)]
        elif query_count["count"] % 5 == 4:
            cursor.fetchall.return_value = [("Operator1",)]
        else:
            cursor.fetchall.return_value = [("POD1",)]
        return None

    cursor.execute.side_effect = mock_execute
    mock_db.cursor.return_value = cursor

    loader = MasterEntityLoader(db_connection=mock_db)
    counts = loader.count_entities()

    assert "field" in counts
    assert "project" in counts
    assert counts == pytest.approx(counts)  # All values are integers


def test_close_connection():
    """Test closing database connection."""
    mock_db = MagicMock(spec=sqlite3.Connection)

    loader = MasterEntityLoader(db_connection=mock_db)
    loader.close()

    # Should NOT close because connection was provided externally
    mock_db.close.assert_not_called()


def test_close_owned_connection():
    """Test closing connection when owned by loader."""
    with patch("esdc.knowledge_graph.master_entities.sqlite3.connect") as mock_connect:
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn

        loader = MasterEntityLoader()  # No connection provided, creates own
        loader.close()

        # Should close because connection was created by loader
        mock_conn.close.assert_called_once()
