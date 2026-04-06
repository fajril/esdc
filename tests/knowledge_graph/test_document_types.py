from esdc.knowledge_graph.document_types import DocumentTypeDetector


def test_detect_mom_document():
    """Test detection of Minutes of Meeting document."""
    detector = DocumentTypeDetector()

    content = """
    # Minutes of Meeting
    # Field Abadi Review

    Date: 15 January 2024

    ## Agenda
    1. Production review
    2. EOR planning

    ## Kesimpulan
    Produksi menurun 15%.
    """

    result = detector.detect(content, filename="MoM-2024-001.md")

    assert result["doc_type"] == "MoM"
    assert result["confidence"] >= 0.8
    assert "kesimpulan" in result["sections"]


def test_detect_definition_document():
    """Test detection of Definition document (timeless)."""
    detector = DocumentTypeDetector()

    content = """
    # Petroleum Resources Management System (PRMS)

    ## Definition
    Petroleum resources are...

    ## Scope
    This document defines...
    """

    result = detector.detect(content, filename="PRMS-definitions.md")

    assert result["doc_type"] == "DEFINITION"
    assert result["is_timeless"] is True


def test_detect_sk_document():
    """Test detection of Surat Keputusan document."""
    detector = DocumentTypeDetector()

    content = """
    # Surat Keputusan Menteri
    No. 15 Tahun 2024

    ## Ketetapan
    Dengan ini menetapkan...

    ## Dasar Hukum
    Berdasarkan...
    """

    result = detector.detect(content, filename="SK-2024-015.md")

    assert result["doc_type"] == "SK"
    assert "ketetapan" in result["sections"]


def test_detect_with_llm_fallback_disabled():
    """Test that LLM fallback can be disabled."""
    from unittest.mock import MagicMock

    mock_llm = MagicMock()
    mock_llm.invoke.return_value.content = '{"doc_type": "MoM", "confidence": 0.7}'

    detector = DocumentTypeDetector(llm=mock_llm)

    content = "Some random content with no keywords"
    result = detector.detect(content, filename="unknown.md", use_llm_fallback=False)

    assert result["doc_type"] == "TECHNICAL_REPORT"
    assert result["confidence"] == 0.3


def test_detect_uses_llm_for_low_confidence():
    """Test that LLM is used when keyword confidence is low."""
    from unittest.mock import MagicMock

    mock_llm = MagicMock()
    mock_llm.invoke.return_value.content = """
    {
        "doc_type": "MoM",
        "confidence": 0.85,
        "is_timeless": false,
        "sections": {"title": {"text": "Test", "confidence": 0.9}}
    }
    """

    detector = DocumentTypeDetector(llm=mock_llm)

    content = "Some random content with no keywords"
    result = detector.detect(content, filename="unknown.md")

    assert result["doc_type"] == "MoM"
    assert result["confidence"] == 0.85
