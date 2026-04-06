# Standard library
import json

# Third-party
import pytest

# Local
from esdc.chat.document_tools import get_entity_document_context, search_documents


def test_search_documents():
    """Test document search tool."""
    result = search_documents.invoke({"query": "latest about Abadi"})

    assert isinstance(result, str)
    # Should return JSON
    data = json.loads(result)
    assert "query" in data
    assert "documents" in data


def test_search_documents_with_filters():
    """Test search with entity and doc_type filters."""
    result = search_documents.invoke(
        {
            "query": "production decline",
            "entity_filter": "Abadi",
            "doc_type_filter": "MoM",
            "temporal": "latest",
        }
    )

    data = json.loads(result)
    assert data["entity_filter"] == "Abadi"
    assert data["doc_type_filter"] == "MoM"


def test_get_entity_document_context():
    """Test entity document context retrieval."""
    result = get_entity_document_context.invoke(
        {"entity_name": "Abadi", "entity_type": "field"}
    )

    data = json.loads(result)
    assert data["entity"] == "Abadi"
    assert data["type"] == "field"
