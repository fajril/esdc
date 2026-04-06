from unittest.mock import MagicMock

import pytest

from esdc.knowledge_graph.llm_extraction import EXTRACTION_PROMPT, LLMExtractor


def test_extract_document_structure():
    """Test LLM extraction of document structure."""
    mock_llm = MagicMock()
    mock_llm.invoke.return_value.content = """
    {
        "doc_type": "MoM",
        "confidence": 0.95,
        "date": "2024-01-15",
        "date_source": "content",
        "is_timeless": false,
        "sections": {
            "title": {"text": "MoM Field Abadi Review", "confidence": 0.98},
            "kesimpulan": {"text": "Produksi menurun 15%", "confidence": 0.95}
        },
        "title": "MoM Field Abadi Review"
    }
    """

    extractor = LLMExtractor(llm=mock_llm)
    result = extractor.extract_structure(
        content="...",
        filename="MoM-2024-001.md",
    )

    assert result["doc_type"] == "MoM"
    assert "kesimpulan" in result["sections"]
    assert result["is_timeless"] is False


def test_extract_structure_with_fallback():
    """Test extraction with invalid JSON (fallback handling)."""
    mock_llm = MagicMock()
    mock_llm.invoke.return_value.content = "Invalid JSON response"

    extractor = LLMExtractor(llm=mock_llm)

    with pytest.raises(ValueError, match="Failed to parse"):
        extractor.extract_structure(content="test", filename="test.md")


def test_extract_without_llm_raises_error():
    """Test extraction without LLM initialized raises error."""
    extractor = LLMExtractor()

    with pytest.raises(ValueError, match="LLM not initialized"):
        extractor.extract_structure(content="test", filename="test.md")


def test_extract_json_embedded_in_text():
    """Test extraction when JSON is embedded in markdown text."""
    mock_llm = MagicMock()
    mock_llm.invoke.return_value.content = """Here is the extracted structure:

```json
{
    "doc_type": "SK",
    "confidence": 0.88,
    "date": "2024-03-01",
    "date_source": "content",
    "is_timeless": false,
    "sections": {
        "title": {"text": "SK Menteri Energi", "confidence": 0.95}
    },
    "title": "SK Menteri Energi"
}
```

That's the result."""

    extractor = LLMExtractor(llm=mock_llm)
    result = extractor.extract_structure(content="test", filename="test.md")

    assert result["doc_type"] == "SK"
    assert result["confidence"] == 0.88


def test_extract_with_missing_fields():
    """Test normalization when LLM returns partial data."""
    mock_llm = MagicMock()
    mock_llm.invoke.return_value.content = """{
        "doc_type": "DEFINITION"
    }"""

    extractor = LLMExtractor(llm=mock_llm)
    result = extractor.extract_structure(content="test", filename="test.md")

    assert result["doc_type"] == "DEFINITION"
    assert result["confidence"] == 0.5
    assert result["is_timeless"] is False
    assert result["sections"] == {}
    assert result["title"] == ""


def test_extraction_prompt_format():
    """Test that EXTRACTION_PROMPT contains expected elements."""
    assert "doc_type" in EXTRACTION_PROMPT
    assert "confidence" in EXTRACTION_PROMPT
    assert "is_timeless" in EXTRACTION_PROMPT
    assert "sections" in EXTRACTION_PROMPT
    assert "{filename}" in EXTRACTION_PROMPT
    assert "{content}" in EXTRACTION_PROMPT
