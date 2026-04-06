import pytest

from esdc.knowledge_graph.relationships import RelationshipExtractor


def test_extract_supersedes_pattern():
    """Test pattern-based SUPERSEDES extraction."""
    extractor = RelationshipExtractor()

    content = "Dengan ini mencabut SK No. 10 Tahun 2023."

    relationships = extractor.extract(content)

    assert len(relationships) >= 1
    assert relationships[0]["type"] == "SUPERSEDES"
    assert "SK No. 10" in relationships[0]["target_doc"]


def test_extract_references_pattern():
    """Test pattern-based REFERENCES extraction."""
    extractor = RelationshipExtractor()

    content = "Sebagaimana dimaksud dalam Peraturan No. 5/2024."

    relationships = extractor.extract(content)

    assert any(r["type"] == "REFERENCES" for r in relationships)
    assert any("Peraturan" in r["target_doc"] for r in relationships)


def test_extract_amends_pattern():
    """Test pattern-based AMENDS extraction."""
    extractor = RelationshipExtractor()

    content = "Perubahan atas Undang-Undang No. 22 Tahun 2001."

    relationships = extractor.extract(content)

    assert any(r["type"] == "AMENDS" for r in relationships)


def test_multiple_relationships():
    """Test extracting multiple relationships."""
    extractor = RelationshipExtractor()

    content = """
    Sebagaimana dimaksud dalam SK No. 5 Tahun 2023.
    Dengan ini mencabut Peraturan No. 10/2022.
    """

    relationships = extractor.extract(content)

    assert len(relationships) >= 2
    types = [r["type"] for r in relationships]
    assert "REFERENCES" in types
    assert "SUPERSEDES" in types


def test_no_relationships():
    """Test document with no relationships."""
    extractor = RelationshipExtractor()

    content = "This is a simple document with no references."

    relationships = extractor.extract(content)

    assert len(relationships) == 0
