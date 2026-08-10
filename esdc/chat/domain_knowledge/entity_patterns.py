"""Entity pattern matcher for query resolution.

Matches natural language queries against entity patterns using keyword extraction.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from .entity_schema import KGSchema

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
        """Initialize query pattern matcher."""
        self.schema = schema
        self._max_keyword_tokens = max(
            len(keyword.split())
            for pattern in schema.query_patterns.values()
            for keyword in pattern["keywords"]
        )

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
            return result

        return None

    def _extract_keywords(self, query: str) -> list[str]:
        normalized = query.lower().strip()
        tokens = re.findall(r"[a-z_]+", normalized)
        keywords: list[str] = []
        for size in range(self._max_keyword_tokens, 0, -1):
            for start in range(len(tokens) - size + 1):
                phrase = " ".join(tokens[start : start + size])
                keywords.append(phrase)
                alias = _INDONESIAN_KEYWORDS.get(phrase)
                if alias:
                    keywords.append(alias)

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
        best_rank = (0, 0, 0, 0.0)

        for pattern in patterns:
            pattern_keywords = pattern.get("keywords", [])
            score = 0.0
            exact_matches: list[str] = []
            for kw in pattern_keywords:
                if kw.lower() in query_lower:
                    score += 1.0
                    exact_matches.append(kw)
                kw_tokens = kw.lower().split()
                if all(t in query_lower for t in kw_tokens):
                    score += 0.5

            specificity = 1.0 / max(len(pattern_keywords), 1)
            score += specificity * 0.1

            pattern["_score"] = score
            rank = (
                len(exact_matches),
                max((len(keyword.split()) for keyword in exact_matches), default=0),
                int(
                    any(
                        query_lower.startswith(keyword.lower())
                        for keyword in exact_matches
                    )
                ),
                score,
            )
            if rank > best_rank:
                best_rank = rank
                best = pattern

        return best or patterns[0]
