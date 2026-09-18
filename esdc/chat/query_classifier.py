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

    SIMPLE_FACTUAL = auto()
    COMPLEX_FACTUAL = auto()
    CONCEPTUAL = auto()
    SPATIAL = auto()
    YEAR_TRANSITION = auto()
    AMBIGUOUS = auto()
    DOCUMENT = auto()
    KSMI = auto()


@dataclass(frozen=True)
class KnowledgeRequest:
    """A declarative request for a domain-knowledge lookup."""

    topic: str
    entity: str | None = None


@dataclass(frozen=True)
class RoutingDecision:
    """Minimal capabilities and declarative context for one query."""

    classification: QueryClassification
    allowed_tools: tuple[str, ...]
    skill_names: tuple[str, ...]
    required_knowledge: tuple[KnowledgeRequest, ...]


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
                r"cadangan\s+(?:nasional|indonesia)",
                r"\bcadangan\b",
                r"berapa\s+reserves",
                r"reserves\s+(?:of|in)",
            ],
            "columns": ["res_oc", "res_an", "res_oil", "res_ga", "res_gn"],
            "tables": ["field_resources", "wa_resources", "project_resources"],
            "project_class": "Reserves & GRR",
        },
        "contingent": {
            "patterns": [
                r"potensi\s+contingent",
                r"contingent\s+resources?",
                r"contingent\b",
                r"sumber\s+daya\s+contingent",
                r"potensi\s+(?:tidak\s+)?terkategorikan",
            ],
            "columns": ["rec_oc", "rec_an", "rec_oil", "rec_ga", "rec_gn"],
            "tables": ["field_resources", "wa_resources", "project_resources"],
            "project_class": "Contingent",
        },
        "prospective": {
            "patterns": [
                r"potensi\s+prospective",
                r"potensi\s+eksplorasi",
                r"prospective\s+resources?",
                r"prospective\b",
                r"sumber\s+daya\s+prospective",
                r"sumber\s+daya\s+eksplorasi",
            ],
            "columns": ["rec_oc", "rec_an", "rec_oil", "rec_ga", "rec_gn"],
            "tables": ["field_resources", "wa_resources", "project_resources"],
            "project_class": "Prospective",
        },
        "resources": {
            "patterns": [
                r"berapa\s+sumber\s+daya",
                r"jumlah\s+sumber\s+daya",
                r"total\s+sumber\s+daya",
                r"sumber\s+daya\s+(?:minyak|gas)",
                r"potensi\s+(?:minyak|gas)",
                r"\bpotensi\b",
                r"berapa\s+potensi",
                r"jumlah\s+potensi",
                r"total\s+potensi",
                r"potensi\s+(?:lapangan|field|wk|wilayah|kerja)",
                r"apa\s+potensi",
                r"resources\b",
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
        "project_name": [
            r"(?:proyek|project)\s+(.+?)(?=\s+(?:tahun|year|di|pada|oleh|operator|untuk|dengan|dan)\b|$)",
        ],
        "field_name": [
            r"(?:lapangan|field)\s+(.+?)(?=\s+(?:tahun|year|di|pada|oleh|operator|untuk|dengan|dan|wk|wilayah\s+kerja)\b|$)",
            r"di\s+(\w+)\s+(?:tahun|tahun\s+\d{4})",  # Contextual
        ],
        "wk_name": [
            r"(?:wk|wilayah\s+kerja|working\s+area)\s+(.+?)(?=\s+(?:tahun|year|di|pada|oleh|operator|untuk|dengan|dan|lapangan|field)\b|$)",
        ],
        "operator_name": [
            r"(?:operator|perusahaan|oleh)\s+(.+?)(?=\s+(?:tahun|year|di|pada|untuk|dengan|dan|wk|wilayah\s+kerja|lapangan|field)\b|$)",
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

    # Document patterns (surat, MoM, berita acara, official correspondence)
    DOCUMENT_PATTERNS = {
        "document_types": [
            r"surat\s+(?:tentang|perihal|persetujuan|keputusan|rekomendasi)",
            r"\bmom\b",
            r"minutes\s+of\s+meeting",
            r"berita\s+acara",
            r"notulen",
            r"korespondensi",
            r"correspondence",
            r"official\s+letter",
            r"dokumen\s+(?:resmi|persetujuan|terkait)",
        ],
        "document_references": [
            r"pod\s+(?:i{1,3}|iv|\d+|pertama|kedua|ketiga)\s+.*revisi",
            r"revisi\s+\d+",
            r"persetujuan\s+(?:pod|wp&?b|afe|poffd|pofd)",
        ],
        "document_enumeration": [
            r"berapa\s+(?:banyak\s+)?dokumen",
            r"dokumen\s+(?:apa|mana)\s+saja",
            r"how\s+many\s+documents",
            r"(?:list|show)\s+all\s+documents",
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

    # Year transition patterns (comparing data across years)
    YEAR_TRANSITION_PATTERNS = [
        r"tahun\s+\d{4}.*(?:namun|tetapi|tapi|namun\s+di)\s+(?:tahun\s+)?\d{4}",
        r"dari\s+tahun\s+\d{4}.*(?:ke|menjadi)\s+(?:tahun\s+)?\d{4}",
        r"di\s+\d{4}.*(?:menjadi|berubah|berpindah).*\d{4}",
        r"in\s+\d{4}.*(?:became|changed|moved|transitioned).*\d{4}",
        r"\d{4}\s+(?:to|vs|versus)\s+\d{4}",
    ]

    def classify(self, query: str) -> QueryClassification:
        """Classify a query and return optimal tool configuration.

        Args:
            query: The user's query string

        Returns:
            QueryClassification with type, confidence, and suggestions
        """
        query_lower = query.lower()
        detected_entities = self._extract_entities(query_lower)

        # KSMI terminology is deliberately checked before generic conceptual,
        # factual, and fallback classification.  A level is extracted only
        # from an explicit token (for example, ``E3``), never inferred from
        # surrounding prose.
        if self._matches_ksmi(query_lower):
            return QueryClassification(
                query_type=QueryType.KSMI,
                confidence=0.9,
                detected_entities=detected_entities,
                suggested_table=None,
                suggested_columns=[],
                reason="KSMI classification or domain-knowledge query detected",
            )

        # Check for document queries first (highest priority) — surat/MoM/
        # berita acara wording (e.g. "persetujuan") could otherwise drift
        # into CONCEPTUAL, so document intent must be resolved explicitly.
        document_match = self._match_patterns(query_lower, self.DOCUMENT_PATTERNS)
        if document_match:
            return QueryClassification(
                query_type=QueryType.DOCUMENT,
                confidence=0.9,
                detected_entities=detected_entities,
                suggested_table=None,
                suggested_columns=[],
                reason=f"Document query detected: {document_match}",
            )

        # Check for conceptual queries
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
                suggested_table=None,
                suggested_columns=[],
                reason=f"Spatial query detected: {spatial_match}",
            )

        # Check for year transition queries
        if self._matches_year_transition(query_lower):
            return QueryClassification(
                query_type=QueryType.YEAR_TRANSITION,
                confidence=0.85,
                detected_entities=detected_entities,
                suggested_table="project_resources",
                suggested_columns=["project_class", "report_year"],
                reason="Year-over-year transition/comparison query detected",
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
                    value = match.group(1)
                    if entity_type != "report_year":
                        value = self._clean_entity_value(value)
                    if value:
                        entities[entity_type] = value
                    break
        return entities

    def _clean_entity_value(self, value: str) -> str:
        """Normalize a regex-captured entity phrase."""
        cleaned = re.sub(r"\b20\d{2}\b", "", value)
        cleaned = re.sub(r"[^\w\s&./'-]", " ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        first_word = cleaned.split(" ", 1)[0] if cleaned else ""
        if first_word in {"apa", "yang", "siapa", "berapa", "mana"}:
            return ""
        return cleaned

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
        _resource_categories = (
            "reserves",
            "resources",
            "contingent",
            "prospective",
            "inplace",
        )

        if "field_name" in entities:
            if query_category in _resource_categories:
                return "field_resources"
            elif query_category == "production_profile":
                return "field_timeseries"

        if "wk_name" in entities:
            if query_category in _resource_categories:
                return "wa_resources"
            elif query_category == "production_profile":
                return "wa_timeseries"

        if "project_name" in entities or "operator_name" in entities:
            if query_category == "production_profile":
                return "project_timeseries"
            return "project_resources"

        if query_category in _resource_categories:
            return "field_resources"

        return "project_resources"

    def _matches_year_transition(self, query: str) -> bool:
        """Check if query involves year-over-year transition or comparison."""
        for pattern in self.YEAR_TRANSITION_PATTERNS:
            if re.search(pattern, query, re.IGNORECASE):
                return True
        return False

    def _matches_ksmi(self, query: str) -> bool:
        """Check for explicit KSMI vocabulary or level/classification intent."""
        if re.search(r"\bksmi\b", query, re.IGNORECASE):
            return True
        if _KSMI_LEVEL_RE.search(query):
            return True
        return bool(
            re.search(
                r"\b(?:wap|groovy|pse|izin\s+berproduksi|transisi|transition)\b",
                query,
                re.IGNORECASE,
            )
            or re.search(
                r"\b(?:formula|formulasi|rumus|hierarki|hierarchy|klasifikasi|classification)\b"
                r".*\b(?:ksmi|level|proyek|project)\b",
                query,
                re.IGNORECASE,
            )
        )


_KSMI_LEVEL_RE = re.compile(r"\b(?:E[0-8]|X[0-6]|A[12])\b", re.IGNORECASE)
_KSMI_RECORD_TERMS = re.compile(
    r"\b(?:proyek|project|lapangan|field|wk|wilayah\s+kerja|\d{4})\b",
    re.IGNORECASE,
)


def _explicit_ksmi_level(query: str) -> str | None:
    """Return the first explicitly written KSMI level token, if any."""
    match = _KSMI_LEVEL_RE.search(query)
    return match.group(0).upper() if match else None


def _ksmi_entity(query: str) -> str:
    """Return the stable entity label used by declarative KSMI lookups."""
    lowered = query.lower()
    if "groovy" in lowered:
        return "GROOVY"
    if "pse" in lowered:
        return "PSE"
    return "izin berproduksi"


def route_query(query: str, *, conversation_context: str = "") -> RoutingDecision:
    """Classify *query* and select the smallest required capability set."""
    normalized = re.sub(
        r"\s+", " ", f"{conversation_context} {query}".strip().lower()
    )
    classification = QueryClassifier().classify(normalized)

    if classification.query_type == QueryType.KSMI:
        level = _explicit_ksmi_level(normalized)
        is_transition = bool(
            re.search(r"\b(?:transisi|transition|berpindah|pindah)\b", normalized)
            or re.search(r"\bke\s+(?:e[0-8]|x[0-6]|a[12])\b", normalized)
        )
        if is_transition and level:
            tools = ["Knowledge Traversal"]
            if _KSMI_RECORD_TERMS.search(normalized):
                tools.append("SQL Executor")
            return RoutingDecision(
                classification,
                tuple(tools),
                ("ksmi_reasoning",),
                (KnowledgeRequest("transition", level),),
            )

        if re.search(r"\b(?:groovy|pse|izin\s+berproduksi)\b", normalized):
            if "groovy" in normalized and re.search(
                r"\b(?:isi|laporan|report|kendala|narasi|content)\b", normalized
            ):
                return RoutingDecision(
                    classification,
                    ("Semantic Search", "SQL Executor"),
                    ("project_narrative_analysis",),
                    (),
                )
            tools = ["Knowledge Traversal"]
            if _KSMI_RECORD_TERMS.search(normalized):
                tools.append("SQL Executor")
            return RoutingDecision(
                classification,
                tuple(tools),
                ("ksmi_reasoning",),
                (KnowledgeRequest("document", _ksmi_entity(normalized)),),
            )

        if re.search(r"\b(?:formula|formulasi|rumus|perhitungan)\b", normalized):
            topic = "formula"
        elif re.search(
            r"\b(?:hierarki|hierarchy|klasifikasi|classification)\b", normalized
        ):
            topic = "hierarchy"
        else:
            topic = "level" if level else "definition"
        knowledge = (
            [KnowledgeRequest(topic, level)]
            if level
            else [KnowledgeRequest(topic)]
        )
        tools = ["Knowledge Traversal"]
        if _KSMI_RECORD_TERMS.search(normalized) and not is_transition:
            tools.append("SQL Executor")
        return RoutingDecision(
            classification, tuple(tools), ("ksmi_reasoning",), tuple(knowledge)
        )

    if classification.query_type == QueryType.SIMPLE_FACTUAL:
        return RoutingDecision(
            classification, ("Simple Data Query",), ("resource_sql",), ()
        )
    if classification.query_type == QueryType.CONCEPTUAL:
        return RoutingDecision(
            classification,
            ("Semantic Search", "SQL Executor"),
            ("project_narrative_analysis",),
            (),
        )
    if classification.query_type == QueryType.SPATIAL:
        return RoutingDecision(
            classification,
            ("Spatial Resolver", "SQL Executor"),
            ("spatial_analysis",),
            (),
        )
    if classification.query_type == QueryType.DOCUMENT:
        return RoutingDecision(
            classification,
            ("Document Search", "Document Reader", "Document Aggregator"),
            ("document_evidence",),
            (),
        )
    if classification.query_type == QueryType.YEAR_TRANSITION:
        return RoutingDecision(
            classification,
            ("SQL Executor", "Resources Column Guide", "Uncertainty Resolver"),
            ("resource_sql",),
            (),
        )
    if classification.query_type == QueryType.COMPLEX_FACTUAL:
        return RoutingDecision(
            classification,
            (
                "SQL Executor",
                "Entity Resolver",
                "Schema Inspector",
                "Resources Column Guide",
                "Uncertainty Resolver",
            ),
            ("resource_sql",),
            (),
        )
    return RoutingDecision(
        classification,
        ("Entity Resolver", "Schema Inspector", "Table Lister"),
        (),
        (),
    )


def get_tools_for_classification(classification: QueryClassification) -> list[str]:
    """Return a static minimal route for legacy classification callers."""
    routes: dict[QueryType, list[str]] = {
        QueryType.SIMPLE_FACTUAL: ["Simple Data Query"],
        QueryType.COMPLEX_FACTUAL: [
            "SQL Executor",
            "Entity Resolver",
            "Schema Inspector",
            "Resources Column Guide",
            "Uncertainty Resolver",
        ],
        QueryType.CONCEPTUAL: ["Semantic Search", "SQL Executor"],
        QueryType.SPATIAL: ["Spatial Resolver", "SQL Executor"],
        QueryType.YEAR_TRANSITION: [
            "SQL Executor",
            "Resources Column Guide",
            "Uncertainty Resolver",
        ],
        QueryType.DOCUMENT: [
            "Document Search",
            "Document Reader",
            "Document Aggregator",
        ],
        QueryType.KSMI: ["Knowledge Traversal"],
        QueryType.AMBIGUOUS: ["Entity Resolver", "Schema Inspector", "Table Lister"],
    }
    return list(routes[classification.query_type])


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
        lines.append("- DO NOT call entity_resolver")
        lines.append("- DO NOT call get_recommended_table")
        lines.append("- DO NOT call get_resources_columns")
        lines.append("- Use suggested table and columns above")
        lines.append("- Default: report_year = MAX, uncert_level = '2. Middle Value'")

    elif classification.query_type == QueryType.YEAR_TRANSITION:
        lines.append("**This is a year-over-year transition/comparison query.**")
        lines.append("- Write ONE SQL query using CASE WHEN or conditional aggregation")
        lines.append(
            "- Compare both years in a single query (avoid sequential queries)"
        )
        lines.append("- Use `report_year IN (year1, year2)` with GROUP BY report_year")
        lines.append("- DO NOT query one year then another — combine them")
        lines.append(
            "- Example: `SUM(CASE WHEN report_year=2024 THEN rec_oc END)"
            " as y2024, SUM(CASE WHEN report_year=2025 THEN rec_oc END)"
            " as y2025`"
        )

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
        lines.append("- If entities are unclear → call entity_resolver")

    elif classification.query_type == QueryType.DOCUMENT:
        lines.append("**This is a document query (surat/MoM/berita acara).**")
        lines.append(
            "- Call search_documents(query, [doc_type/doc_topic/year filters]) FIRST"
        )
        lines.append("- DO NOT call entity_resolver — it cannot find documents")
        lines.append(
            "- Use read_document(doc_id) for full text when comparing or quoting"
        )
        lines.append(
            "- If no results: tell the user no matching documents are ingested"
        )
        lines.append(
            "- For 'berapa dokumen / dokumen apa saja / dokumen mana saja': "
            "call aggregate_documents, NOT search_documents (which returns "
            "only the top few)"
        )
        lines.append(
            "- aggregate_documents returns an exact `count` plus separate "
            "`semantic_candidates`; report the count as the answer and the "
            "candidates as related documents, never summed"
        )

    lines.append("")
    lines.append(f"*Reason: {classification.reason}*")

    return "\n".join(lines)
