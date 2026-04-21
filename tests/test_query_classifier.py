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
        assert result.detected_entities.get("field_name") is not None
        assert result.detected_entities.get("field_name", "").lower() == "duri"
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
        result = self.classifier.classify("info tentang minyak")

        assert result.query_type == QueryType.AMBIGUOUS

    def test_year_transition_indonesian(self):
        """Test classification of year transition query (Indonesian)."""
        result = self.classifier.classify(
            "ada berapa banyak project yang tahun 2024 prospective resources "
            "namun di 2025 menjadi contingent resources atau reserves"
        )

        assert result.query_type == QueryType.YEAR_TRANSITION
        assert result.confidence >= 0.85
        assert (
            result.suggested_table is not None
            and "project_resources" in result.suggested_table
        )

    def test_year_transition_dari_ke(self):
        """Test classification of dari...ke year transition query."""
        result = self.classifier.classify(
            "project yang dari tahun 2023 ke tahun 2025 berubah project class"
        )

        assert result.query_type == QueryType.YEAR_TRANSITION

    def test_year_transition_menjadi(self):
        """Test classification of 'menjadi' year transition query."""
        result = self.classifier.classify(
            "project di 2024 yang menjadi reserves di 2025"
        )

        assert result.query_type == QueryType.YEAR_TRANSITION

    def test_year_transition_english(self):
        """Test classification of English year transition query."""
        result = self.classifier.classify(
            "projects in 2024 that became reserves in 2025"
        )

        assert result.query_type == QueryType.YEAR_TRANSITION

    def test_detect_entities_multiple(self):
        """Test detection of multiple entities."""
        result = self.classifier.classify(
            "berapa cadangan lapangan Duri wk rokan tahun 2024 provinsi Riau?"
        )

        assert result.detected_entities.get("field_name", "").lower() == "duri"
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

    def test_year_transition_tools(self):
        """Test minimal tools for year transition queries."""
        classification = QueryClassification(
            query_type=QueryType.YEAR_TRANSITION,
            confidence=0.85,
            detected_entities={},
            suggested_table="project_resources",
            suggested_columns=["project_class", "report_year"],
            reason="Year transition query",
        )

        tools = get_tools_for_classification(classification)
        assert "SQL Executor" in tools
        assert "Knowledge Traversal" in tools
        assert "Semantic Search" not in tools
        assert "Spatial Resolver" not in tools


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

    def test_year_transition_formatting(self):
        """Test formatting of year transition classification."""
        classification = QueryClassification(
            query_type=QueryType.YEAR_TRANSITION,
            confidence=0.85,
            detected_entities={"report_year": "2024"},
            suggested_table="project_resources",
            suggested_columns=["project_class", "report_year"],
            reason="Year-over-year transition detected",
        )

        formatted = format_classification_for_prompt(classification)

        assert "Year Transition" in formatted
        assert "CASE WHEN" in formatted
        assert "ONE SQL query" in formatted
        assert "project_resources" in formatted


class TestConditionalToolPreservation:
    """Test conditionally-registered tools are preserved in allowed_tools.

    The bug: query_classification_node overrides allowed_tools with only
    classifier-selected tools, dropping conditionally-registered tools
    like Compute Engine, File Processing, View File. Fix: preserve tools
    that exist in all_tools but are missing from classifier output.
    """

    # Tools that the classifier never returns (conditionally registered)
    CONDITIONAL_TOOLS = {"Compute Engine", "File Processing", "View File"}

    # All possible tools = classifier tools + conditional tools
    ALL_TOOLS = {
        "Knowledge Traversal",
        "SQL Executor",
        "Schema Inspector",
        "Table Lister",
        "Table Selector",
        "Semantic Search",
        "Spatial Resolver",
        "Uncertainty Resolver",
        "Problem Cluster Search",
    } | CONDITIONAL_TOOLS

    def test_classifier_never_includes_conditional_tools(self):
        """Verify classifier output never includes conditional tools.

        Confirms the bug exists at the classifier level.
        """
        from esdc.chat.query_classifier import QueryType

        for qtype in QueryType:
            classification = QueryClassification(
                query_type=qtype,
                confidence=0.9,
                detected_entities={},
                suggested_table=None,
                suggested_columns=[],
                reason="Test",
            )
            tools = get_tools_for_classification(classification)
            for ct in self.CONDITIONAL_TOOLS:
                assert ct not in tools, (
                    f"Conditional tool {ct!r} should NOT be in classifier "
                    f"output for {qtype.name}, but it was found"
                )

    def test_preservation_logic_simple_factual(self):
        """Simulate the fix: classifier tools + preserved conditional tools."""
        classification = QueryClassification(
            query_type=QueryType.SIMPLE_FACTUAL,
            confidence=0.9,
            detected_entities={},
            suggested_table="field_resources",
            suggested_columns=["res_oc"],
            reason="Test",
        )

        classifier_tools = get_tools_for_classification(classification)
        classifier_tool_set = set(classifier_tools)

        # Simulate the fix: preserve tools in all_tools not in classifier output
        preserved = self.ALL_TOOLS - classifier_tool_set
        final_tools = list(classifier_tool_set | preserved)

        for ct in self.CONDITIONAL_TOOLS:
            assert ct in final_tools, (
                f"Conditional tool {ct!r} should be preserved in final tools"
            )

    def test_preservation_logic_ambiguous(self):
        """Test preservation with ambiguous query type (most tools)."""
        classification = QueryClassification(
            query_type=QueryType.AMBIGUOUS,
            confidence=0.5,
            detected_entities={},
            suggested_table=None,
            suggested_columns=[],
            reason="Test",
        )

        classifier_tools = get_tools_for_classification(classification)
        classifier_tool_set = set(classifier_tools)

        preserved = self.ALL_TOOLS - classifier_tool_set
        final_tools = list(classifier_tool_set | preserved)

        for ct in self.CONDITIONAL_TOOLS:
            assert ct in final_tools, (
                f"Conditional tool {ct!r} should be preserved in final tools"
            )

    def test_preservation_logic_no_conditional_tools(self):
        """Conditional tools not in all_tools should NOT be preserved."""
        classification = QueryClassification(
            query_type=QueryType.SIMPLE_FACTUAL,
            confidence=0.9,
            detected_entities={},
            suggested_table="field_resources",
            suggested_columns=["res_oc"],
            reason="Test",
        )

        classifier_tools = get_tools_for_classification(classification)
        classifier_tool_set = set(classifier_tools)

        # Simulate all_tools WITHOUT conditional tools (sandbox not configured)
        all_tools_without_sandbox = {
            "Knowledge Traversal",
            "SQL Executor",
            "Schema Inspector",
            "Table Lister",
            "Table Selector",
        }

        preserved = all_tools_without_sandbox - classifier_tool_set
        final_tools = list(classifier_tool_set | preserved)

        for ct in self.CONDITIONAL_TOOLS:
            assert ct not in final_tools, (
                f"Conditional tool {ct!r} should NOT be preserved when not in all_tools"
            )
