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

    def test_simple_reserves_project_query(self):
        """Test classification of project-level reserves query."""
        result = self.classifier.classify("berapa cadangan proyek Abadi LNG 2024?")

        assert result.query_type == QueryType.SIMPLE_FACTUAL
        assert result.detected_entities.get("project_name") == "abadi lng"
        assert result.suggested_table == "project_resources"

    def test_simple_reserves_operator_query(self):
        """Test classification of operator-level reserves query."""
        result = self.classifier.classify(
            "berapa cadangan operator Pertamina Hulu Rokan 2024?"
        )

        assert result.query_type == QueryType.SIMPLE_FACTUAL
        assert result.detected_entities.get("operator_name") == (
            "pertamina hulu rokan"
        )
        assert result.suggested_table == "project_resources"

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

    def test_potensi_work_area(self):
        """Test 'apa potensi wk jabung?' is classified as SIMPLE_FACTUAL."""
        result = self.classifier.classify("apa potensi wk jabung?")

        assert result.query_type == QueryType.SIMPLE_FACTUAL
        assert result.confidence == 0.9
        assert result.detected_entities.get("wk_name") == "jabung"
        assert result.suggested_table == "wa_resources"
        assert "rec_oc" in result.suggested_columns

    def test_potensi_standalone(self):
        """Test standalone 'potensi' without substance qualifier."""
        result = self.classifier.classify("berapa potensi nasional?")

        assert result.query_type == QueryType.SIMPLE_FACTUAL
        assert result.confidence == 0.9
        # Default table for resources with no entity detected
        assert result.suggested_table == "field_resources"

    def test_potensi_field(self):
        """Test 'potensi lapangan Duri'."""
        result = self.classifier.classify("berapa potensi lapangan Duri?")

        assert result.query_type == QueryType.SIMPLE_FACTUAL
        assert result.confidence == 0.9
        assert result.detected_entities.get("field_name", "").lower() == "duri"

    def test_resources_english(self):
        """Test English 'resources' query."""
        result = self.classifier.classify("resources of WK Rokan")

        assert result.query_type == QueryType.SIMPLE_FACTUAL
        assert result.confidence == 0.9
        assert result.detected_entities.get("wk_name") == "rokan"

    def test_reserves_english(self):
        """Test English 'reserves' query."""
        result = self.classifier.classify("how much reserves in Rokan?")

        assert result.query_type == QueryType.SIMPLE_FACTUAL
        assert result.confidence == 0.9

    def test_contingent_query(self):
        """Test 'berapa contingent resources nasional?' is SIMPLE_FACTUAL."""
        result = self.classifier.classify("berapa contingent resources nasional?")
        assert result.query_type == QueryType.SIMPLE_FACTUAL
        assert result.confidence == 0.9

    def test_potensi_contingent_routes_to_contingent(self):
        """Test 'potensi contingent WK Rokan' routes to contingent."""
        result = self.classifier.classify("potensi contingent WK Rokan")
        assert result.query_type == QueryType.SIMPLE_FACTUAL
        assert result.detected_entities.get("wk_name") == "rokan"
        # The reason should mention 'contingent', not 'resources'
        assert "contingent" in result.reason.lower()

    def test_prospective_query(self):
        """Test 'prospective resources di WK Jatibarang' is classified correctly."""
        result = self.classifier.classify("prospective resources di WK Jatibarang")
        assert result.query_type == QueryType.SIMPLE_FACTUAL

    def test_potensi_eksplorasi_routes_to_prospective(self):
        """Test 'potensi eksplorasi lapangan Duri' routes to prospective."""
        result = self.classifier.classify("potensi eksplorasi lapangan Duri")
        assert result.query_type == QueryType.SIMPLE_FACTUAL
        assert "prospective" in result.reason.lower()

    def test_document_query_pod_revisi_comparison(self):
        """Real-world failure query: POD revision comparison -> DOCUMENT."""
        result = self.classifier.classify(
            "apakah beda Surat Persetujuan POD I Lapangan Abadi Revisi 1 dan 2"
        )
        assert result.query_type == QueryType.DOCUMENT
        assert result.confidence == 0.9

    def test_document_query_surat_persetujuan(self):
        """Test 'surat persetujuan POD Duri' is classified as DOCUMENT."""
        result = self.classifier.classify("surat persetujuan POD Duri")
        assert result.query_type == QueryType.DOCUMENT
        assert result.confidence == 0.9

    def test_document_patterns_take_priority_over_conceptual(self):
        """Technical-problem queries without document wording stay CONCEPTUAL."""
        result = self.classifier.classify("kendala teknis di WK Rokan")
        assert result.query_type == QueryType.CONCEPTUAL

    def test_simple_factual_query_unaffected_by_document_patterns(self):
        """Test 'berapa cadangan Duri' is still SIMPLE_FACTUAL (unchanged)."""
        result = self.classifier.classify("berapa cadangan Duri")
        assert result.query_type == QueryType.SIMPLE_FACTUAL

    def test_enumeration_queries_classify_as_document(self):
        """Exhaustive-count/list phrasing routes to DOCUMENT for aggregate_documents."""
        for query in (
            "berapa dokumen yang menyatakan akan onstream di 2026?",
            "dokumen apa saja yang menyebutkan kata separator?",
            "dokumen mana saja yang membahas separator?",
        ):
            assert self.classifier.classify(query).query_type == QueryType.DOCUMENT, (
                query
            )


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
        assert "Entity Resolver" in tools
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
        assert "Entity Resolver" in tools
        assert "Semantic Search" not in tools
        assert "Spatial Resolver" not in tools

    def test_document_tools(self):
        """DOCUMENT queries get Semantic Search plus doc tools from base_tools."""
        classification = QueryClassification(
            query_type=QueryType.DOCUMENT,
            confidence=0.9,
            detected_entities={},
            suggested_table=None,
            suggested_columns=[],
            reason="Test",
        )

        tools = get_tools_for_classification(classification)
        assert "Document Search" in tools
        assert "Document Reader" in tools
        assert "Semantic Search" in tools
        assert "SQL Executor" in tools

    def test_document_tools_always_available(self):
        """Document Search/Reader are in base_tools for every query type."""
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
            assert "Document Search" in tools, f"missing for {qtype.name}"
            assert "Document Reader" in tools, f"missing for {qtype.name}"

    def test_document_aggregator_is_bound_for_every_query_type(self):
        """A tool the classifier does not bind is a tool the model cannot call."""
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
            assert "Document Aggregator" in tools, f"missing for {qtype.name}"


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
        assert "DO NOT call entity_resolver" in formatted
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

    def test_document_formatting(self):
        """Test formatting of DOCUMENT classification."""
        classification = QueryClassification(
            query_type=QueryType.DOCUMENT,
            confidence=0.9,
            detected_entities={},
            suggested_table=None,
            suggested_columns=[],
            reason="Document query detected: document_types",
        )

        formatted = format_classification_for_prompt(classification)

        assert "search_documents" in formatted
        assert "DO NOT call entity_resolver" in formatted
        assert "read_document" in formatted

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
    classifier-selected tools, dropping conditionally-registered tools.
    Fix: preserve tools that exist in all_tools but are missing from
    classifier output.

    Code Interpreter, Shell Executor, Resources Column Guide, and
    Timeseries Column Guide are now in base_tools (not conditional).
    CONDITIONAL_TOOLS is empty since File Processing and View File
    have been removed from the codebase.
    """

    # Tools that the classifier never returns (conditionally registered)
    # Currently empty — all formerly-conditional tools are now in base_tools
    CONDITIONAL_TOOLS: set[str] = set()

    # All possible tools = classifier tools + conditional tools
    ALL_TOOLS = {
        "Entity Resolver",
        "Knowledge Traversal",
        "SQL Executor",
        "Simple Data Query",
        "Code Interpreter",
        "Shell Executor",
        "Resources Column Guide",
        "Timeseries Column Guide",
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

        Since all formerly-conditional tools are now in base_tools,
        there are no conditional tools to exclude. This test serves
        as a regression guard in case new conditional tools are added.
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
        """Verify base_tools are always present in classifier output."""
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

        # Base tools should always be present
        base_tools = {
            "Code Interpreter",
            "Shell Executor",
            "Resources Column Guide",
            "Timeseries Column Guide",
        }
        for bt in base_tools:
            assert bt in classifier_tool_set, (
                f"Base tool {bt!r} should be in classifier output for ALL query types"
            )

    def test_preservation_logic_ambiguous(self):
        """Verify base_tools present even for ambiguous query type."""
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

        # Base tools should always be present
        base_tools = {
            "Code Interpreter",
            "Shell Executor",
            "Resources Column Guide",
            "Timeseries Column Guide",
        }
        for bt in base_tools:
            assert bt in classifier_tool_set, (
                f"Base tool {bt!r} should be in classifier output for AMBIGUOUS"
            )

    def test_preservation_logic_no_conditional_tools(self):
        """Verify all base_tools are in classifier output without preservation.

        Code Interpreter, Shell Executor, Resources Column Guide, and
        Timeseries Column Guide are now in base_tools, so they appear
        in classifier output directly — no preservation mechanism needed.
        """
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

        # Base tools should always be present
        base_tools = {
            "Code Interpreter",
            "Shell Executor",
            "Resources Column Guide",
            "Timeseries Column Guide",
        }
        for bt in base_tools:
            assert bt in classifier_tool_set, (
                f"Base tool {bt!r} should be in classifier output without preservation"
            )
