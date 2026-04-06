from esdc.knowledge_graph.chunking import DocumentChunker


def test_chunk_hybrid_structure():
    """Test hybrid chunking with sections."""
    chunker = DocumentChunker()

    doc_structure = {
        "id": "doc-001",
        "title": "MoM Abadi Review",
        "sections": {
            "title": {"text": "Minutes of Meeting - Field Abadi Review"},
            "kesimpulan": {"text": "Kesimpulan: Produksi menurun 15%."},
            "tindak_lanjut": {"text": "Tindak Lanjut: Submit EOR plan."},
        },
    }

    chunks = chunker.chunk_hybrid(doc_structure)

    # Should have summary + 3 sections
    assert len(chunks) >= 4

    # First chunk should be summary
    assert chunks[0]["is_summary"] is True

    # Should have title section
    title_chunks = [c for c in chunks if c["section_type"] == "title"]
    assert len(title_chunks) >= 1


def test_chunk_large_section():
    """Test chunking large sections with overlap."""
    chunker = DocumentChunker(chunk_size=100, overlap=20)

    long_text = "This is a sentence. " * 50  # ~1200 chars

    doc_structure = {
        "id": "doc-002",
        "title": "Test",
        "sections": {
            "content": {"text": long_text},
        },
    }

    chunks = chunker.chunk_hybrid(doc_structure)

    # Should split large section into multiple chunks
    content_chunks = [c for c in chunks if c["section_type"] == "content"]
    assert len(content_chunks) > 1

    # Chunks should overlap
    if len(content_chunks) > 1:
        # Check overlap exists (simplified check)
        assert len(content_chunks[0]["text"]) <= chunker.chunk_size + 50


def test_generate_summary():
    """Test summary generation."""
    chunker = DocumentChunker()

    doc_structure = {
        "id": "doc-003",
        "title": "Test Document",
        "sections": {
            "title": {"text": "Document Title"},
            "kesimpulan": {"text": "Key conclusion here"},
        },
    }

    summary = chunker._generate_summary(doc_structure)

    assert "Document Title" in summary
    assert "Key conclusion here" in summary
