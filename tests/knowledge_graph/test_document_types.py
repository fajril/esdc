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
