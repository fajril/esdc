import re
from typing import Any

PATTERNS: dict[str, list[str]] = {
    "SUPERSEDES": [
        r"(mencabut\s+(?:SK|Surat\s+Keputusan|Peraturan).+?\d{4}[^\n]*)",
        r"(menghapus\s+(?:ketentuan|pasal).+?\d{4}[^\n]*)",
        r"(supersede.+(?:SK|document).+?\d{4}[^\n]*)",
    ],
    "REFERENCES": [
        r"(sebagaimana\s+dimaksud\s+dalam\s+[^\n]+?\d{4}[^\n]*)",
        r"(merujuk\s+pada\s+[^\n]+)",
        r"(berdasarkan\s+[^\n]+?\d{4}[^\n]*)",
    ],
    "AMENDS": [
        r"(perubahan\s+(?:atas|terhadap)\s+[^\n]+)",
        r"(addendum.+(?:kepada|for)\s+[^\n]+)",
        r"(amandemen\s+[^\n]+)",
    ],
}


class RelationshipExtractor:
    """Extract document relationships using pattern matching."""

    def extract(self, content: str) -> list[dict[str, Any]]:
        """Extract relationships from document content.

        Args:
            content: Document text to analyze.

        Returns:
            List of relationship dicts with type, target_doc, confidence,
            detected_by, and evidence fields.
        """
        relationships = []

        for rel_type, patterns in PATTERNS.items():
            for pattern in patterns:
                matches = re.finditer(pattern, content, re.IGNORECASE)
                for match in matches:
                    target_doc = match.group(1) if match.lastindex else match.group(0)

                    relationships.append(
                        {
                            "type": rel_type,
                            "target_doc": target_doc.strip(),
                            "confidence": 0.85,
                            "detected_by": "pattern",
                            "evidence": match.group(0),
                        }
                    )

        return relationships
