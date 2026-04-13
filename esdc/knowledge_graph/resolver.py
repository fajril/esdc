"""Knowledge Traversal entity resolver.

Resolves entities from the database using pattern matching and FTS.
Returns structured context for single-shot SQL generation.
"""

from __future__ import annotations

import logging
import re
from typing import Any

import duckdb

from .patterns import QueryPatternMatcher
from .schema import KGSchema

logger = logging.getLogger(__name__)

_CONFIDENCE_THRESHOLD = 0.7

_ENTITY_TYPE_QUERIES: dict[str, str] = {
    "Field": """
        SELECT DISTINCT field_id, field_name
        FROM project_resources
        WHERE field_name ILIKE '%' || ? || '%'
           OR field_id ILIKE '%' || ? || '%'
        LIMIT 10
    """,
    "WorkingArea": """
        SELECT DISTINCT wk_id, wk_name
        FROM project_resources
        WHERE wk_name ILIKE '%' || ? || '%'
           OR wk_id ILIKE '%' || ? || '%'
        LIMIT 10
    """,
    "Operator": """
        SELECT DISTINCT operator_name
        FROM project_resources
        WHERE operator_name ILIKE '%' || ? || '%'
        LIMIT 10
    """,
}

_YEAR_PATTERN = re.compile(r"\b(20\d{2})\b")

_UNCERTAINTY_MAP: dict[str, str] = {
    "1p": "1. Low Value",
    "proven": "1. Low Value",
    "terbukti": "1. Low Value",
    "low": "1. Low Value",
    "2p": "2. Middle Value",
    "probable": "2. Middle Value",
    "mungkin": "2. Middle Value",
    "middle": "2. Middle Value",
    "mid": "2. Middle Value",
    "3p": "3. High Value",
    "possible": "3. High Value",
    "harapan": "3. High Value",
    "high": "3. High Value",
    "1c": "1. Low Value",
    "2c": "2. Middle Value",
    "3c": "3. High Value",
}

_CLASS_MAP: dict[str, str] = {
    "reserves": "1. Reserves & GRR",
    "cadangan": "1. Reserves & GRR",
    "grr": "1. Reserves & GRR",
    "contingent": "2. Contingent Resources",
    "kontingen": "2. Contingent Resources",
    "prospective": "3. Prospective Resources",
    "prospektif": "3. Prospective Resources",
    "abandoned": "4. Abandoned",
}

_ENTITY_HINTS: dict[str, str] = {
    "lapangan": "Field",
    "field": "Field",
    "wk": "WorkingArea",
    "wilayah": "WorkingArea",
    "operator": "Operator",
    "perusahaan": "Operator",
}

_STOP_WORDS: set[str] = {
    "di",
    "yang",
    "dan",
    "atau",
    "dengan",
    "untuk",
    "dari",
    "ke",
    "pada",
    "ini",
    "itu",
    "adalah",
    "tidak",
    "sudah",
    "akan",
    "apa",
    "saja",
    "berapa",
    "siapa",
    "dimana",
    "kapan",
    "data",
    "info",
    "informasi",
    "lihat",
    "tampilkan",
    "show",
    "the",
    "of",
    "in",
    "for",
    "at",
    "to",
    "from",
    "with",
    "by",
    "about",
    "how",
    "what",
    "when",
    "where",
    "which",
}


class KnowledgeTraversalResolver:
    """Resolve entities from database for Knowledge Traversal."""

    def __init__(self, db: duckdb.DuckDBPyConnection) -> None:
        self.schema = KGSchema()
        self.pattern_matcher = QueryPatternMatcher(self.schema)
        self.db = db

    def resolve(self, query: str, return_multiple: bool = False) -> dict[str, Any]:
        """Resolve entities and patterns from a natural language query.

        Returns structured context for single-shot SQL generation.
        """
        entities = self._resolve_entities(query, return_multiple)
        pattern_result = self.pattern_matcher.match(query)

        if not entities and not pattern_result:
            return {
                "status": "failed",
                "fallback": "multi_round",
                "message": "Could not resolve any entities or patterns from query",
                "entities": [],
                "pattern": None,
                "suggested_table": None,
                "where_conditions": [],
                "required_columns": [],
                "confidence": 0.0,
            }

        best_confidence = max((e.get("confidence", 0.0) for e in entities), default=0.0)

        if (
            not return_multiple
            and best_confidence < _CONFIDENCE_THRESHOLD
            and len(entities) > 1
        ):
            return {
                "status": "ambiguous",
                "message": (
                    f"Found {len(entities)} possible matches. Please specify which one."
                ),
                "candidates": entities,
                "pattern": pattern_result,
                "suggested_table": pattern_result.get("suggested_table")
                if pattern_result
                else None,
                "confidence": best_confidence,
            }

        where_conditions = self._build_where_conditions(entities)
        suggested_table = self._determine_table(query, entities, pattern_result)

        return {
            "status": "success",
            "entities": entities,
            "pattern": pattern_result,
            "suggested_table": suggested_table,
            "where_conditions": where_conditions,
            "required_columns": pattern_result.get("suggested_columns", [])
            if pattern_result
            else [],
            "confidence": best_confidence,
        }

    def _resolve_entities(
        self, query: str, return_multiple: bool
    ) -> list[dict[str, Any]]:
        entities: list[dict[str, Any]] = []

        year_match = _YEAR_PATTERN.search(query)
        if year_match:
            entities.append(
                {
                    "type": "Year",
                    "value": int(year_match.group(1)),
                    "name": year_match.group(1),
                    "confidence": 1.0,
                }
            )

        uncertainty = self._resolve_uncertainty(query)
        if uncertainty:
            entities.append(uncertainty)

        project_class = self._resolve_class(query)
        if project_class:
            entities.append(project_class)

        entity_type_hint = self._detect_entity_type(query)
        named_entities = self._resolve_named_entities(
            query, entity_type_hint, return_multiple
        )
        entities.extend(named_entities)

        return entities

    def _resolve_uncertainty(self, query: str) -> dict[str, Any] | None:
        query_lower = query.lower()
        for keyword, db_value in _UNCERTAINTY_MAP.items():
            if keyword in query_lower:
                return {
                    "type": "UncertaintyLevel",
                    "name": keyword.upper(),
                    "db_value": db_value,
                    "confidence": 1.0,
                }
        return None

    def _resolve_class(self, query: str) -> dict[str, Any] | None:
        query_lower = query.lower()
        for keyword, db_value in _CLASS_MAP.items():
            if keyword in query_lower:
                return {
                    "type": "ProjectClass",
                    "name": keyword,
                    "db_value": db_value,
                    "confidence": 0.9,
                }
        return None

    def _detect_entity_type(self, query: str) -> str | None:
        query_lower = query.lower()
        for keyword, entity_type in _ENTITY_HINTS.items():
            if keyword in query_lower:
                return entity_type
        return None

    def _resolve_named_entities(
        self, query: str, entity_type_hint: str | None, return_multiple: bool
    ) -> list[dict[str, Any]]:
        type_order = (
            [entity_type_hint]
            if entity_type_hint
            else ["Field", "WorkingArea", "Operator"]
        )
        type_order = [t for t in type_order if t in _ENTITY_TYPE_QUERIES]

        search_term = self._extract_entity_name(query)
        if not search_term:
            return []

        all_results: list[dict[str, Any]] = []
        for etype in type_order:
            results = self._query_entity_type(etype, search_term, return_multiple)
            if results:
                all_results.extend(results)
                if not return_multiple and results[0]["confidence"] >= 1.0:
                    break

        if not return_multiple and all_results:
            all_results.sort(key=lambda x: x["confidence"], reverse=True)
            return [all_results[0]]

        return all_results

    def _extract_entity_name(self, query: str) -> str:
        words = re.findall(r"[a-zA-Z\u00C0-\u024F]+", query)
        skip_words = (
            set(_ENTITY_HINTS.keys())
            | set(_CLASS_MAP.keys())
            | set(_UNCERTAINTY_MAP.keys())
            | _STOP_WORDS
        )
        candidates = [w for w in words if len(w) > 2 and w.lower() not in skip_words]
        pattern_keywords = set()
        for pattern in self.schema.query_patterns.values():
            pattern_keywords.update(kw.lower() for kw in pattern.get("keywords", []))
        candidates = [w for w in candidates if w.lower() not in pattern_keywords]

        return " ".join(candidates) if candidates else ""

    def _query_entity_type(
        self, entity_type: str, search_term: str, return_multiple: bool
    ) -> list[dict[str, Any]]:
        sql = _ENTITY_TYPE_QUERIES.get(entity_type)
        if not sql:
            return []

        limit = 10 if return_multiple else 3
        limited_sql = sql.replace("LIMIT 10", f"LIMIT {limit}")

        try:
            if entity_type == "Operator":
                result = self.db.execute(limited_sql, [search_term]).fetchall()
            else:
                result = self.db.execute(
                    limited_sql, [search_term, search_term]
                ).fetchall()
        except Exception:
            logger.debug(
                "[KG] entity_query_failed | type=%s term=%s", entity_type, search_term
            )
            return []

        if not result:
            return []

        entities: list[dict[str, Any]] = []
        for row in result:
            if entity_type == "Field" or entity_type == "WorkingArea":
                entity_id, entity_name = row[0], row[1]
            elif entity_type == "Operator":
                entity_id, entity_name = None, row[0]
            else:
                continue

            search_lower = search_term.lower()
            name_lower = entity_name.lower() if entity_name else ""
            if name_lower == search_lower:
                confidence = 1.0
            elif name_lower.startswith(search_lower):
                confidence = 0.9
            elif search_lower in name_lower:
                confidence = 0.8
            else:
                confidence = 0.6

            entity: dict[str, Any] = {
                "type": entity_type,
                "id": entity_id,
                "name": entity_name,
                "confidence": confidence,
            }
            if entity_type in ("Field", "WorkingArea"):
                entity["filter_column"] = (
                    "field_name" if entity_type == "Field" else "wk_name"
                )

            entities.append(entity)

        return entities

    def _build_where_conditions(self, entities: list[dict[str, Any]]) -> list[str]:
        conditions: list[str] = []
        for entity in entities:
            etype = entity.get("type")
            if etype == "Year":
                conditions.append(f"report_year = {entity['value']}")
            elif etype == "UncertaintyLevel":
                conditions.append(f"uncert_level = '{entity['db_value']}'")
            elif etype == "ProjectClass":
                conditions.append(f"project_class = '{entity['db_value']}'")
            elif (
                etype in ("Field", "WorkingArea")
                and entity.get("filter_column")
                and entity.get("name")
            ):
                col = entity["filter_column"]
                name = entity["name"].replace("'", "''")
                conditions.append(f"{col} = '{name}'")
            elif etype == "Operator" and entity.get("name"):
                name = entity["name"].replace("'", "''")
                conditions.append(f"operator_name = '{name}'")
        return conditions

    def _determine_table(
        self, query: str, entities: list[dict[str, Any]], pattern: dict[str, Any] | None
    ) -> str:
        if pattern and pattern.get("suggested_table"):
            return pattern["suggested_table"]

        for entity in entities:
            etype = entity.get("type")
            if etype == "Field":
                if any(
                    kw in query.lower()
                    for kw in ("produksi", "forecast", "profil", "produksi", "rate")
                ):
                    return "field_timeseries"
                return "field_resources"
            elif etype == "WorkingArea":
                if any(
                    kw in query.lower()
                    for kw in ("produksi", "forecast", "profil", "rate")
                ):
                    return "wa_timeseries"
                return "wa_resources"
            elif etype == "Operator":
                return "project_resources"

        return "project_resources"
