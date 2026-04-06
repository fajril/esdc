class DocumentChunker:
    """Chunk documents using hybrid strategy.

    Hybrid strategy combines summary chunks and section chunks:
    - Summary chunk: Title + priority sections for high-level retrieval
    - Section chunks: Each section as separate chunk for precise retrieval
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_hybrid(self, doc_structure: dict) -> list[dict]:
        """Chunk document using hybrid strategy.

        Args:
            doc_structure: Document with id, title, and sections

        Returns:
            List of chunks with metadata
        """
        chunks = []
        doc_id = doc_structure["id"]

        # Priority sections for summary
        priority_sections = ["title", "kesimpulan", "executive_summary", "definition"]

        # Generate summary chunk first
        summary_text = self._generate_summary(doc_structure, priority_sections)
        chunks.append(
            {
                "id": f"{doc_id}_summary",
                "document_id": doc_id,
                "text": summary_text,
                "section_type": "summary",
                "is_summary": True,
                "chunk_index": 0,
            }
        )

        # Create section chunks
        chunk_index = 1
        for section_name, section_data in doc_structure["sections"].items():
            section_text = section_data["text"]

            # Split large sections
            if len(section_text) > self.chunk_size:
                section_chunks = self._split_with_overlap(section_text)
                for i, chunk_text in enumerate(section_chunks):
                    chunks.append(
                        {
                            "id": f"{doc_id}_{section_name}_{i}",
                            "document_id": doc_id,
                            "text": chunk_text,
                            "section_type": section_name,
                            "is_summary": False,
                            "chunk_index": chunk_index,
                        }
                    )
                    chunk_index += 1
            else:
                chunks.append(
                    {
                        "id": f"{doc_id}_{section_name}",
                        "document_id": doc_id,
                        "text": section_text,
                        "section_type": section_name,
                        "is_summary": False,
                        "chunk_index": chunk_index,
                    }
                )
                chunk_index += 1

        return chunks

    def _generate_summary(
        self, doc_structure: dict, priority_sections: list[str] | None = None
    ) -> str:
        """Generate summary from title and priority sections.

        Args:
            doc_structure: Document structure
            priority_sections: Sections to include in summary

        Returns:
            Summary text
        """
        if priority_sections is None:
            priority_sections = [
                "title",
                "kesimpulan",
                "executive_summary",
                "definition",
            ]

        parts = [doc_structure["title"]]

        for section_name in priority_sections:
            if section_name in doc_structure["sections"]:
                parts.append(doc_structure["sections"][section_name]["text"])

        return "\n\n".join(parts)

    def _split_with_overlap(self, text: str) -> list[str]:
        """Split text into chunks with overlap.

        Args:
            text: Text to split

        Returns:
            List of overlapping chunks
        """
        chunks = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size

            # If not at the end, try to break at sentence boundary
            if end < len(text):
                # Look for sentence boundary within last 50 chars
                boundary = text.rfind(".", start, end)
                if boundary > start + self.chunk_size - 50:
                    end = boundary + 1

            chunk = text[start:end]
            chunks.append(chunk)

            # Move start forward with overlap
            start = end - self.overlap
            if start < 0:
                start = 0
            if start >= len(text):
                break

        return chunks
