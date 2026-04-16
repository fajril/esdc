"""Tests for query classifier."""

from esdc.chat.query_classifier import (
    QueryClassification,
    QueryClassifier,
    QueryType,
    format_classification_for_prompt,
    get_tools_for_classification,
)


class TestQueryClassifier:
    """Test query classification."""

    def setup_method(self):
        """Set up classifier."""
        self.classifier = QueryClassifier()

    def test_simple_reserves_field_query(self):
        """Test classification of simple reserves query."""
        result = self.classifier.classify("berapa cadangan lapangan Duri tahun 2024?")

        assert result.query_type == QueryType.SIMPLE_FACTUAL
        assert result.confidence == 0.9
        assert result.detected_entities.get("field_name").lower() == "duri"
        assert result.detected_entities.get("report_year") == "2024"
        assert result.suggested_table == "field_resources"
        assert "res_oc" in result.suggested_columns

    def test_simple_reserves_wa_query(self):
        """Test classification of WA-level reserves query."""
        result = self.classifier.classify("berapa cadangan wk rokan tahun 2024?")

        assert result.query_type == QueryType.SIMPLE_FACTUAL
        assert result.detected_entities.get("wk_name") == "rokan"
        assert result.suggested_table == "wa_resources"

    def test_production_profile_query(self):
        """Test classification of production profile query."""
        result = self.classifier.classify("profil produksi lapangan Duri 2024")

        assert result.query_type == QueryType.SIMPLE_FACTUAL
        assert result.suggested_table == "field_timeseries"
        assert "tpf_oc" in result.suggested_columns

    def test_conceptual_query_economic(self):
        """Test classification of economic issues query."""
        result = self.classifier.classify(
            "proyek apa yang tidak ekonomis di tahun 2024?"
        )

        assert result.query_type == QueryType.CONCEPTUAL
        assert result.confidence == 0.9
        assert result.detected_entities.get("report_year") == "2024"
        assert result.suggested_table is None  # Semantic search first

    def test_conceptual_query_technical(self):
        """Test classification of technical problems query."""
        result = self.classifier.classify("kendala teknis di lapangan Duri")

        assert result.query_type == QueryType.CONCEPTUAL

    def test_spatial_query_proximity(self):
        """Test classification of spatial proximity query."""
        result = self.classifier.classify("lapangan terdekat dengan Duri")

        assert result.query_type == QueryType.SPATIAL
        assert result.confidence >= 0.85

    def test_spatial_query_distance(self):
        """Test classification of distance query."""
        result = self.classifier.classify("jarak Duri ke Rokan")

        assert result.query_type == QueryType.SPATIAL

    def test_resources_query(self):
        """Test classification of resources query."""
        result = self.classifier.classify("berapa sumber daya gas lapangan Duri?")

        assert result.query_type == QueryType.SIMPLE_FACTUAL
        assert "rec_an" in result.suggested_columns

    def test_inplace_query(self):
        """Test classification of in-place query."""
        result = self.classifier.classify("berapa IOIP di project Duri Steam?")

        assert result.query_type == QueryType.SIMPLE_FACTUAL
        # IOIP is at project level, so suggested table depends on entity type
        assert result.suggested_columns == ["prj_ioip", "prj_igip"]

    def test_ambiguous_query(self):
        """Test classification of ambiguous query."""
        result = self.classifier.classify("info tentang minyak")  # Very vague

        assert result.query_type == QueryType.AMBIGUOUS

    def test_detect_entities_multiple(self):
        """Test detection of multiple entities."""
        result = self.classifier.classify(
            "berapa cadangan lapangan Duri wk rokan tahun 2024 provinsi Riau?"
        )

        assert result.detected_entities.get("field_name").lower() == "duri"
        assert result.detected_entities.get("wk_name") == "rokan"
        assert result.detected_entities.get("report_year") == "2024"


class TestToolSelection:
    """Test tool selection based on classification."""

    def test_simple_factual_tools(self):
        """Test minimal tools for simple queries."""
        classification = QueryClassification(
            query_type=QueryType.SIMPLE_FACTUAL,
            confidence=0.9,
            detected_entities={},
            suggested_table="field_resources",
            suggested_columns=["res_oc"],
            reason="Test",
        )

        tools = get_tools_for_classification(classification)
        assert "SQL Executor" in tools
        assert "Schema Inspector" in tools
        assert "Table Lister" in tools
        assert "Table Selector" in tools
        assert "Spatial Resolver" not in tools
        assert "Semantic Search" not in tools

    def test_conceptual_tools(self):
        """Test tools for conceptual queries."""
        classification = QueryClassification(
            query_type=QueryType.CONCEPTUAL,
            confidence=0.9,
            detected_entities={},
            suggested_table=None,
            suggested_columns=[],
            reason="Test",
        )

        tools = get_tools_for_classification(classification)
        assert "Semantic Search" in tools
        assert "SQL Executor" in tools
        assert "Spatial Resolver" not in tools

    def test_spatial_tools(self):
        """Test tools for spatial queries."""
        classification = QueryClassification(
            query_type=QueryType.SPATIAL,
            confidence=0.85,
            detected_entities={},
            suggested_table=None,
            suggested_columns=[],
            reason="Test",
        )

        tools = get_tools_for_classification(classification)
        assert "Spatial Resolver" in tools
        assert "SQL Executor" in tools
        assert "Semantic Search" not in tools

    def test_ambiguous_tools(self):
        """Test full tool set for ambiguous queries."""
        classification = QueryClassification(
            query_type=QueryType.AMBIGUOUS,
            confidence=0.5,
            detected_entities={},
            suggested_table=None,
            suggested_columns=[],
            reason="Test",
        )

        tools = get_tools_for_classification(classification)
        assert "Knowledge Traversal" in tools
        assert "SQL Executor" in tools


class TestPromptFormatting:
    """Test formatting classification for prompts."""

    def test_simple_factual_formatting(self):
        """Test formatting of simple factual classification."""
        classification = QueryClassification(
            query_type=QueryType.SIMPLE_FACTUAL,
            confidence=0.9,
            detected_entities={"field_name": "Duri", "report_year": "2024"},
            suggested_table="field_resources",
            suggested_columns=["res_oc", "res_an"],
            reason="Matched reserves pattern",
        )

        formatted = format_classification_for_prompt(classification)

        assert "Query Analysis" in formatted
        assert "field_name: 'Duri'" in formatted
        assert "field_resources" in formatted
        assert "res_oc" in formatted
        assert "DO NOT call knowledge_traversal" in formatted
        assert "DO NOT call get_recommended_table" in formatted

    def test_conceptual_formatting(self):
        """Test formatting of conceptual classification."""
        classification = QueryClassification(
            query_type=QueryType.CONCEPTUAL,
            confidence=0.9,
            detected_entities={"report_year": "2024"},
            suggested_table=None,
            suggested_columns=[],
            reason="Economic issues detected",
        )

        formatted = format_classification_for_prompt(classification)

        assert "semantic_search" in formatted
        assert "WAIT for results" in formatted

    def test_spatial_formatting(self):
        """Test formatting of spatial classification."""
        classification = QueryClassification(
            query_type=QueryType.SPATIAL,
            confidence=0.85,
            detected_entities={"field_name": "Duri"},
            suggested_table=None,
            suggested_columns=[],
            reason="Proximity query detected",
        )

        formatted = format_classification_for_prompt(classification)

        assert "resolve_spatial" in formatted
