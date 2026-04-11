"""Query pattern matcher for Knowledge Traversal.

Matches natural language queries against KG query patterns using keyword extraction.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from .schema import KGSchema

logger = logging.getLogger(__name__)

_INDONESIAN_KEYWORDS: dict[str, str] = {
    "cadangan": "reserves",
    "sumber": "resources",
    "sumber daya": "resources",
    "profil": "profile",
    "produksi": "production",
    "prediksi": "forecast",
    "proyeksi": "forecast",
    "ramalan": "forecast",
    "forecast": "forecast",
    "lapangan": "field",
    "wilayah kerja": "work_area",
    "wk": "work_area",
    "operator": "operator",
    "isu": "issues",
    "masalah": "issues",
    "catatan": "remarks",
    "tertinggi": "top",
    "terbesar": "top",
    "ranking": "top",
    "perubahan": "change",
    "kenapa turun": "change",
    "terakhir berproduksi": "last_production",
    "status": "status",
    "informasi": "info",
    "recovery factor": "recovery",
    "rf": "recovery",
    "in place": "inplace",
    "ioip": "inplace",
    "igip": "inplace",
}


class QueryPatternMatcher:
    """Match natural language queries against KG query patterns."""

    def __init__(self, schema: KGSchema) -> None:
        self.schema = schema

    def match(self, query: str) -> dict[str, Any] | None:
        """Match a query against known patterns and return best match."""
        keywords = self._extract_keywords(query)
        if not keywords:
            return None

        patterns = self.schema.get_pattern_for_keywords(keywords)
        if not patterns:
            return None

        best_pattern = self._score_patterns(query, patterns)
        if best_pattern:
            pattern_name = best_pattern["name"]
            result = {
                "pattern_name": pattern_name,
                "primary_entity": best_pattern.get("primary_entity"),
                "suggested_table": best_pattern.get("default_table"),
                "suggested_columns": best_pattern.get("suggested_columns", []),
                "requires_entity": best_pattern.get("requires_entity", True),
                "description": best_pattern.get("description", ""),
                "confidence": best_pattern.get("_score", 0.0),
            }
            cypher = best_pattern.get("cypher")
            if cypher:
                result["cypher_template"] = cypher.strip()
            return result

        return None

    def _extract_keywords(self, query: str) -> list[str]:
        normalized = query.lower().strip()
        tokens = re.findall(r"[a-z_]+", normalized)
        keywords: list[str] = []
        bigrams = [f"{tokens[i]} {tokens[i + 1]}" for i in range(len(tokens) - 1)]

        for bigram in bigrams:
            if bigram in _INDONESIAN_KEYWORDS:
                keywords.append(_INDONESIAN_KEYWORDS[bigram])

        for token in tokens:
            if token in _INDONESIAN_KEYWORDS:
                keywords.append(_INDONESIAN_KEYWORDS[token])
            elif len(token) > 2:
                keywords.append(token)

        seen: set[str] = set()
        unique: list[str] = []
        for kw in keywords:
            if kw not in seen:
                seen.add(kw)
                unique.append(kw)

        return unique

    def _score_patterns(
        self, query: str, patterns: list[dict[str, Any]]
    ) -> dict[str, Any]:
        query_lower = query.lower()
        best: dict[str, Any] | None = None
        best_score = 0.0

        for pattern in patterns:
            pattern_keywords = pattern.get("keywords", [])
            score = 0.0
            for kw in pattern_keywords:
                if kw.lower() in query_lower:
                    score += 1.0
                kw_tokens = kw.lower().split()
                if all(t in query_lower for t in kw_tokens):
                    score += 0.5

            specificity = 1.0 / max(len(pattern_keywords), 1)
            score += specificity * 0.1

            pattern["_score"] = score
            if score > best_score:
                best_score = score
                best = pattern

        return best or patterns[0]
