"""Document type detection for ESDC knowledge graph."""

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

DOCUMENT_SCHEMAS = {
    "DEFINITION": {
        "sections": ["title", "definition", "scope", "term", "concept", "reference"],
        "priority": "title",
        "is_timeless": True,
        "keywords": ["definition", "glossary", "PRMS", "Petroleum Resources"],
    },
    "MoM": {
        "sections": [
            "title",
            "agenda",
            "attendees",
            "discussion",
            "kesimpulan",
            "tindak_lanjut",
        ],
        "priority": "title",
        "is_timeless": False,
        "keywords": ["minutes of meeting", "notulen", "kesimpulan rapat", "MoM"],
    },
    "SK": {
        "sections": ["title", "ketetapan", "dasar_hukum", "menetapkan", "kewenangan"],
        "priority": "title",
        "is_timeless": False,
        "keywords": ["surat keputusan", "menetapkan", "SK Menteri", "keputusan"],
    },
    "POD": {
        "sections": [
            "title",
            "executive_summary",
            "development_plan",
            "economics",
            "timeline",
        ],
        "priority": "title",
        "is_timeless": False,
        "keywords": [
            "plan of development",
            "POD",
            "pengembangan",
            "plan of development",
        ],
    },
    "PERATURAN": {
        "sections": ["title", "pasal", "bab", "ketentuan", "sanksi"],
        "priority": "title",
        "is_timeless": False,
        "keywords": ["peraturan", "pasal", "undang-undang", "permen"],
    },
    "TECHNICAL_REPORT": {
        "sections": [
            "title",
            "executive_summary",
            "findings",
            "recommendations",
            "appendix",
        ],
        "priority": "title",
        "is_timeless": False,
        "keywords": ["feasibility", "technical report", "study", "laporan"],
    },
}


DOCUMENT_SCHEMAS = {
    "DEFINITION": {
        "sections": ["title", "definition", "scope", "term", "concept", "reference"],
        "priority": "title",
        "is_timeless": True,
        "keywords": ["definition", "glossary", "PRMS", "Petroleum Resources"],
    },
    "MoM": {
        "sections": [
            "title",
            "agenda",
            "attendees",
            "discussion",
            "kesimpulan",
            "tindak_lanjut",
        ],
        "priority": "title",
        "is_timeless": False,
        "keywords": ["minutes of meeting", "notulen", "kesimpulan rapat", "MoM"],
    },
    "SK": {
        "sections": ["title", "ketetapan", "dasar_hukum", "menetapkan", "kewenangan"],
        "priority": "title",
        "is_timeless": False,
        "keywords": ["surat keputusan", "menetapkan", "SK Menteri", "keputusan"],
    },
    "POD": {
        "sections": [
            "title",
            "executive_summary",
            "development_plan",
            "economics",
            "timeline",
        ],
        "priority": "title",
        "is_timeless": False,
        "keywords": [
            "plan of development",
            "POD",
            "pengembangan",
            "plan of development",
        ],
    },
    "PERATURAN": {
        "sections": ["title", "pasal", "bab", "ketentuan", "sanksi"],
        "priority": "title",
        "is_timeless": False,
        "keywords": ["peraturan", "pasal", "undang-undang", "permen"],
    },
    "TECHNICAL_REPORT": {
        "sections": [
            "title",
            "executive_summary",
            "findings",
            "recommendations",
            "appendix",
        ],
        "priority": "title",
        "is_timeless": False,
        "keywords": ["feasibility", "technical report", "study", "laporan"],
    },
}


class DocumentTypeDetector:
    """Detects document type from content and filename."""

    def __init__(self, llm: BaseChatModel | None = None):
        """Initialize the detector.

        Args:
            llm: Optional LangChain chat model for LLM-based extraction.
                 If provided, will be used as fallback when keyword
                 matching yields low confidence.
        """
        self.llm = llm

    def detect(
        self, content: str, filename: str, *, use_llm_fallback: bool = True
    ) -> dict[str, Any]:
        """Detect document type from content and filename.

        Args:
            content: Document content
            filename: Document filename
            use_llm_fallback: Whether to use LLM for low confidence results

        Returns:
            Dictionary with:
            - doc_type: Detected document type
            - confidence: Confidence score (0-1)
            - sections: List of detected sections
            - is_timeless: Whether document is timeless
        """
        keyword_result = self._detect_by_keywords(content, filename)

        if keyword_result["confidence"] >= 0.5 or not use_llm_fallback:
            return keyword_result

        if self.llm is None:
            return keyword_result

        try:
            from esdc.knowledge_graph.llm_extraction import LLMExtractor

            extractor = LLMExtractor(llm=self.llm)
            llm_result = extractor.extract_structure(content, filename)

            if llm_result["confidence"] > keyword_result["confidence"]:
                return {
                    "doc_type": llm_result["doc_type"],
                    "confidence": llm_result["confidence"],
                    "sections": list(llm_result["sections"].keys()),
                    "is_timeless": llm_result["is_timeless"],
                }
        except (ValueError, KeyError):
            pass

        return keyword_result

    def _detect_by_keywords(self, content: str, filename: str) -> dict[str, Any]:
        """Detect document type using keyword matching.

        Args:
            content: Document content
            filename: Document filename

        Returns:
            Dictionary with detection results
        """
        content_lower = content.lower()
        filename_lower = filename.lower()

        scores: dict[str, float] = {}
        detected_keywords: dict[str, list[str]] = {}

        for doc_type, schema in DOCUMENT_SCHEMAS.items():
            score = 0.0
            keywords_found: list[str] = []

            for keyword in schema["keywords"]:
                keyword_lower = keyword.lower()
                in_content = keyword_lower in content_lower
                in_filename = keyword_lower in filename_lower

                if in_content:
                    score += 1.0
                    keywords_found.append(keyword)
                if in_filename:
                    score += 1.5
                    if keyword not in keywords_found:
                        keywords_found.append(keyword)

            if score > 0:
                scores[doc_type] = score
                detected_keywords[doc_type] = keywords_found

        if not scores:
            return {
                "doc_type": "TECHNICAL_REPORT",
                "confidence": 0.3,
                "sections": [],
                "is_timeless": False,
            }

        best_type = max(scores, key=lambda k: scores[k])
        keywords_count = len(detected_keywords[best_type])
        total_keywords = len(DOCUMENT_SCHEMAS[best_type]["keywords"])

        base_confidence = keywords_count / total_keywords

        filename_boost = any(
            kw.lower() in filename_lower
            for kw in DOCUMENT_SCHEMAS[best_type]["keywords"]
        )
        if filename_boost:
            base_confidence = min(base_confidence + 0.3, 1.0)

        confidence = base_confidence

        sections = self._extract_sections(content_lower, best_type)

        return {
            "doc_type": best_type,
            "confidence": confidence,
            "sections": sections,
            "is_timeless": DOCUMENT_SCHEMAS[best_type]["is_timeless"],
        }

    def _extract_sections(self, content_lower: str, doc_type: str) -> list[str]:
        """Extract sections from document content.

        Args:
            content_lower: Lowercase document content
            doc_type: Document type

        Returns:
            List of detected section names
        """
        sections = []
        schema = DOCUMENT_SCHEMAS[doc_type]

        for section in schema["sections"]:
            if section == "title":
                continue

            section_variations = self._get_section_variations(section)

            for variation in section_variations:
                if variation.lower() in content_lower:
                    sections.append(section)
                    break

        return sections

    def _get_section_variations(self, section: str) -> list[str]:
        """Get variations of section names to search for.

        Args:
            section: Section name

        Returns:
            List of variations to search for
        """
        variations = {
            "kesimpulan": ["kesimpulan", "conclusion", "ringkasan"],
            "tindak_lanjut": ["tindak lanjut", "action items", "follow-up"],
            "ketetapan": ["ketetapan", "ketentuan"],
            "dasar_hukum": ["dasar hukum", "legal basis", "landasan hukum"],
            "menetapkan": ["menetapkan", "decides"],
            "kewenangan": ["kewenangan", "authority"],
            "executive_summary": ["executive summary", "ringkasan eksekutif"],
            "development_plan": ["development plan", "rencana pengembangan"],
            "economics": ["economics", "ekonomi"],
            "timeline": ["timeline", "jadwal"],
            "pasal": ["pasal", "article"],
            "bab": ["bab", "chapter"],
            "ketentuan": ["ketentuan", "provisions"],
            "sanksi": ["sanksi", "sanctions", "penalty"],
            "findings": ["findings", "temuan"],
            "recommendations": ["recommendations", "rekomendasi"],
            "appendix": ["appendix", "lampiran"],
            "definition": ["definition", "definisi"],
            "scope": ["scope", "lingkup"],
            "term": ["term", "istilah"],
            "concept": ["concept", "konsep"],
            "reference": ["reference", "referensi"],
            "agenda": ["agenda", "jadwal rapat"],
            "attendees": ["attendees", "peserta", "hadir"],
            "discussion": ["discussion", "pembahasan"],
        }

        return variations.get(section, [section])
