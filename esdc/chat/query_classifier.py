"""Query classification for optimal tool selection.

Pre-classifies incoming queries to determine the minimal set of tools needed.
This reduces tool calls from 3-4 per query to 1-2 for most cases.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum, auto


class QueryType(Enum):
    """Query classification types."""

    SIMPLE_FACTUAL = auto()  # Direct SQL possible (80% of queries)
    COMPLEX_FACTUAL = auto()  # May need entity resolution
    CONCEPTUAL = auto()  # Semantic search needed
    SPATIAL = auto()  # Spatial queries
    AMBIGUOUS = auto()  # Fallback to full tool set


@dataclass
class QueryClassification:
    """Result of query classification."""

    query_type: QueryType
    confidence: float  # 0.0-1.0
    detected_entities: dict[str, str]  # e.g., {"field_name": "Duri"}
    suggested_table: str | None
    suggested_columns: list[str]
    reason: str  # Explanation for classification


class QueryClassifier:
    """Classify queries to optimize tool selection."""

    # Simple factual patterns (high confidence, direct SQL possible)
    SIMPLE_PATTERNS = {
        "reserves": {
            "patterns": [
                r"berapa\s+cadangan",
                r"jumlah\s+cadangan",
                r"total\s+cadangan",
                r"cadangan\s+(?:minyak|gas|oil|gas)",
                r"cadangan\s+(?:lapangan|field|wk|wilayah)",
            ],
            "columns": ["res_oc", "res_an", "res_oil", "res_ga", "res_gn"],
            "tables": ["field_resources", "wa_resources", "project_resources"],
            "project_class": "Reserves & GRR",
        },
        "resources": {
            "patterns": [
                r"berapa\s+sumber\s+daya",
                r"jumlah\s+sumber\s+daya",
                r"total\s+sumber\s+daya",
                r"sumber\s+daya\s+(?:minyak|gas)",
                r"potensi\s+(?:minyak|gas)",
            ],
            "columns": ["rec_oc", "rec_an", "rec_oil", "rec_ga", "rec_gn"],
            "tables": ["field_resources", "wa_resources", "project_resources"],
            "project_class": None,  # Can be any class
        },
        "production_profile": {
            "patterns": [
                r"profil\s+produksi",
                r"forecast",
                r"proyeksi\s+produksi",
                r"produksi\s+(?:minyak|gas)\s+(?:tahun|tahunan)",
            ],
            "columns": ["tpf_oc", "tpf_an", "slf_oc", "slf_an", "year"],
            "tables": ["field_timeseries", "project_timeseries"],
            "project_class": None,
        },
        "inplace": {
            "patterns": [
                r"(?:initial|original)\s+(?:oil|gas)\s+in\s+place",
                r"ioip|igip",
                r"(?:minyak|gas)\s+in\s+place",
            ],
            "columns": ["prj_ioip", "prj_igip"],
            "tables": ["project_resources"],
            "project_class": None,
        },
    }

    # Entity detection patterns
    ENTITY_PATTERNS = {
        "field_name": [
            r"lapangan\s+(\w+)",
            r"field\s+(\w+)",
            r"di\s+(\w+)\s+(?:tahun|tahun\s+\d{4})",  # Contextual
        ],
        "wk_name": [
            r"wk\s+(\w+)",
            r"wilayah\s+kerja\s+(\w+)",
            r"working\s+area\s+(\w+)",
        ],
        "report_year": [
            r"tahun\s+(\d{4})",
            r"year\s+(\d{4})",
            r"(?:\s)(\d{4})(?:\s|$)",  # Standalone year
        ],
        "province": [
            r"provinsi\s+(\w+)",
            r"di\s+(\w+)\s+(?:provinsi|province)",
        ],
        "basin": [
            r"cekungan\s+(\w+)",
            r"basin\s+(\w+)",
        ],
    }

    # Conceptual patterns (need semantic search)
    CONCEPTUAL_PATTERNS = {
        "economic_issues": [
            r"tidak\s+ekonomis",
            r"economic\s+issue",
            r"low\s+oil\s+price",
            r"tidak\s+layak",
            r"not\s+viable",
            r"suboptimal",
        ],
        "technical_problems": [
            r"kendala\s+teknis",
            r"technical\s+problem",
            r"challenges",
            r"issues",
            r"masalah\s+teknis",
            r"kompleksitas",
            r"complexity",
        ],
        "reservoir_issues": [
            r"ketebalan",
            r"porositas",
            r"permeabilitas",
            r"heterogen",
        ],
    }

    # Spatial patterns
    SPATIAL_PATTERNS = {
        "proximity": [
            r"dekatin",
            r"near",
            r"terdekat",
            r"closest",
            r"sekitar",
        ],
        "distance": [
            r"jarak",
            r"distance",
            r"radius",
            r"km\s+dari",
        ],
    }

    def classify(self, query: str) -> QueryClassification:
        """Classify a query and return optimal tool configuration.

        Args:
            query: The user's query string

        Returns:
            QueryClassification with type, confidence, and suggestions
        """
        query_lower = query.lower()
        detected_entities = self._extract_entities(query_lower)

        # Check for conceptual queries first (highest priority)
        conceptual_match = self._match_patterns(query_lower, self.CONCEPTUAL_PATTERNS)
        if conceptual_match:
            return QueryClassification(
                query_type=QueryType.CONCEPTUAL,
                confidence=0.9,
                detected_entities=detected_entities,
                suggested_table=None,  # Semantic search first
                suggested_columns=[],
                reason=f"Conceptual query detected: {conceptual_match}",
            )

        # Check for spatial queries
        spatial_match = self._match_patterns(query_lower, self.SPATIAL_PATTERNS)
        if spatial_match:
            return QueryClassification(
                query_type=QueryType.SPATIAL,
                confidence=0.85,
                detected_entities=detected_entities,
                suggested_table=None,  # resolve_spatial first
                suggested_columns=[],
                reason=f"Spatial query detected: {spatial_match}",
            )

        # Check for simple factual queries
        for query_category, config in self.SIMPLE_PATTERNS.items():
            for pattern in config["patterns"]:
                if re.search(pattern, query_lower, re.IGNORECASE):
                    # Determine table based on entities
                    suggested_table = self._suggest_table(
                        query_category, detected_entities
                    )

                    return QueryClassification(
                        query_type=QueryType.SIMPLE_FACTUAL,
                        confidence=0.9,
                        detected_entities=detected_entities,
                        suggested_table=suggested_table,
                        suggested_columns=config["columns"],
                        reason=f"Matched '{query_category}' pattern: {pattern}",
                    )

        # Check for complex patterns (has entities but not simple)
        if detected_entities:
            return QueryClassification(
                query_type=QueryType.COMPLEX_FACTUAL,
                confidence=0.7,
                detected_entities=detected_entities,
                suggested_table="project_resources",  # Default
                suggested_columns=["project_name", "project_remarks"],
                reason="Entities detected but no simple pattern match",
            )

        # Fallback
        return QueryClassification(
            query_type=QueryType.AMBIGUOUS,
            confidence=0.5,
            detected_entities=detected_entities,
            suggested_table=None,
            suggested_columns=[],
            reason="No clear pattern match, using full tool set",
        )

    def _extract_entities(self, query: str) -> dict[str, str]:
        """Extract entities from query."""
        entities = {}
        for entity_type, patterns in self.ENTITY_PATTERNS.items():
            for pattern in patterns:
                match = re.search(pattern, query, re.IGNORECASE)
                if match:
                    entities[entity_type] = match.group(1)
                    break
        return entities

    def _match_patterns(self, query: str, pattern_groups: dict) -> str | None:
        """Check if query matches any pattern in groups.

        Returns:
            Matched category name or None
        """
        for category, patterns in pattern_groups.items():
            for pattern in patterns:
                if re.search(pattern, query, re.IGNORECASE):
                    return category
        return None

    def _suggest_table(self, query_category: str, entities: dict[str, str]) -> str:
        """Suggest optimal table based on query type and entities."""
        # Field-level queries
        if "field_name" in entities:
            if query_category in ("reserves", "resources", "inplace"):
                return "field_resources"
            elif query_category == "production_profile":
                return "field_timeseries"

        # WA-level queries
        if "wk_name" in entities:
            if query_category in ("reserves", "resources"):
                return "wa_resources"
            elif query_category == "production_profile":
                return "wa_timeseries"

        # National-level queries
        if query_category in ("reserves", "resources"):
            return "field_resources"  # Most common

        return "project_resources"  # Default fallback


_SCHEMA_TOOLS = ["Schema Inspector", "Table Lister", "Table Selector"]


def get_tools_for_classification(classification: QueryClassification) -> list[str]:
    """Get minimal tool set for query type.

    Returns LangChain tool names (matching @tool decorator strings).
    ALWAYS includes schema tools since these are lightweight and may be
    needed for any query type.

    Args:
        classification: The query classification

    Returns:
        List of LangChain tool names to bind for this query
    """
    base_tools = ["SQL Executor"] + _SCHEMA_TOOLS

    if classification.query_type == QueryType.SIMPLE_FACTUAL:
        return base_tools

    elif classification.query_type == QueryType.CONCEPTUAL:
        return ["Semantic Search"] + base_tools

    elif classification.query_type == QueryType.SPATIAL:
        return ["Spatial Resolver"] + base_tools

    elif classification.query_type == QueryType.COMPLEX_FACTUAL:
        return ["Uncertainty Resolver", "Problem Cluster Search"] + base_tools

    else:  # AMBIGUOUS
        return [
            "Knowledge Traversal",
            "Semantic Search",
            "Spatial Resolver",
            "Uncertainty Resolver",
            "Problem Cluster Search",
        ] + base_tools


def format_classification_for_prompt(classification: QueryClassification) -> str:
    """Format classification as string to inject into system prompt.

    This gives the LLM explicit guidance on how to handle this query.
    """
    lines = [
        "## Query Analysis (Pre-computed)",
        f"- **Query type**: {classification.query_type.name.replace('_', ' ').title()}",
        f"- **Confidence**: {classification.confidence:.0%}",
    ]

    if classification.detected_entities:
        lines.append("- **Detected entities**:")
        for key, value in classification.detected_entities.items():
            lines.append(f"  - {key}: '{value}'")

    if classification.suggested_table:
        lines.append(f"- **Suggested table**: `{classification.suggested_table}`")

    if classification.suggested_columns:
        cols = ", ".join(f"`{c}`" for c in classification.suggested_columns[:4])
        lines.append(f"- **Key columns**: {cols}")

    # Add execution strategy
    lines.append("")
    lines.append("### Execution Strategy")

    if classification.query_type == QueryType.SIMPLE_FACTUAL:
        lines.append("**Write SQL directly using the schema above.**")
        lines.append("- DO NOT call knowledge_traversal")
        lines.append("- DO NOT call get_recommended_table")
        lines.append("- DO NOT call get_resources_columns")
        lines.append("- Use suggested table and columns above")
        lines.append("- Default: report_year = MAX, uncert_level = '2. Middle Value'")

    elif classification.query_type == QueryType.CONCEPTUAL:
        lines.append("**This is a conceptual query about issues/problems.**")
        lines.append("1. Call semantic_search(query, filters) FIRST")
        lines.append("2. WAIT for results")
        lines.append("3. Use returned project_ids to write execute_sql")

    elif classification.query_type == QueryType.SPATIAL:
        lines.append("**This is a spatial query.**")
        lines.append("1. Call resolve_spatial(query_type, target, radius_km)")
        lines.append("2. WAIT for results")
        lines.append("3. Use returned entity IDs in execute_sql")

    elif classification.query_type == QueryType.COMPLEX_FACTUAL:
        lines.append("**Entity resolution may be needed.**")
        lines.append(
            "- If auto-resolved entities above are sufficient → write SQL directly"
        )
        lines.append("- If entities are unclear → call knowledge_traversal")

    lines.append("")
    lines.append(f"*Reason: {classification.reason}*")

    return "\n".join(lines)
