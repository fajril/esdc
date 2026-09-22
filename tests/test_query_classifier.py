"""Tests for query classifier."""

import pytest

from esdc.chat.query_classifier import (
    QueryClassification,
    QueryClassifier,
    QueryType,
    format_classification_for_prompt,
    get_tools_for_classification,
    route_query,
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

    def test_route_query_is_exposed_for_capability_selection(self):
        decision = route_query("Berapa cadangan minyak Lapangan Duri tahun 2025?")
        assert "Simple Data Query" in decision.allowed_tools

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
        assert result.detected_entities.get("operator_name") == ("pertamina hulu rokan")
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
        assert tools == ["Simple Data Query"]

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
        assert tools == ["Entity Resolver", "Schema Inspector", "Table Lister"]

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
        assert tools == [
            "SQL Executor",
            "Resources Column Guide",
            "Uncertainty Resolver",
        ]
        assert "Semantic Search" not in tools
        assert "Spatial Resolver" not in tools

    def test_document_tools(self):
        """DOCUMENT queries get the document tools only."""
        classification = QueryClassification(
            query_type=QueryType.DOCUMENT,
            confidence=0.9,
            detected_entities={},
            suggested_table=None,
            suggested_columns=[],
            reason="Test",
        )

        tools = get_tools_for_classification(classification)
        assert tools == [
            "Document Search",
            "Document Reader",
            "Document Aggregator",
        ]


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


class TestCurrentIntentAndVisualization:
    """Current-turn intent must win over history; visualization is explicit."""

    def test_current_factual_intent_overrides_prior_ksmi(self) -> None:
        decision = route_query(
            "Berapa cadangan minyak Lapangan Duri tahun 2025?",
            conversation_context="Bolehkah E3 ke E4?",
        )
        assert decision.allowed_tools == ("Simple Data Query",)
        assert decision.required_knowledge == ()

    def test_anaphoric_follow_up_uses_nearest_prior_subject(self) -> None:
        decision = route_query(
            "bagaimana tahun 2024?",
            conversation_context="berapa cadangan minyak Lapangan Duri tahun 2023?",
        )
        assert decision.allowed_tools == ("Simple Data Query",)
        assert (
            decision.classification.detected_entities.get("field_name", "").lower()
            == "duri"
        )

    def test_explicit_new_year_overrides_prior_context_year(self) -> None:
        decision = route_query(
            "bagaimana tahun 2024?",
            conversation_context="berapa cadangan minyak Lapangan Duri tahun 2023?",
        )
        assert decision.classification.detected_entities.get("report_year") == "2024"
        assert (
            decision.classification.detected_entities.get("field_name", "").lower()
            == "duri"
        )

    def test_ambiguous_standalone_question_does_not_inherit_history(self) -> None:
        decision = route_query(
            "jelaskan produksi sumur",
            conversation_context="Bolehkah E3 ke E4?",
        )
        assert decision.classification.query_type == QueryType.AMBIGUOUS
        assert decision.allowed_tools == (
            "Entity Resolver",
            "Schema Inspector",
            "Table Lister",
        )

    def test_definitional_question_does_not_inherit_ksmi_history(self) -> None:
        decision = route_query(
            "apa itu migas?", conversation_context="Bolehkah E3 ke E4?"
        )
        assert decision.classification.query_type == QueryType.AMBIGUOUS

    def test_explicit_new_entity_overrides_prior_context(self) -> None:
        decision = route_query(
            "berapa cadangan minyak Lapangan Rokan tahun 2025?",
            conversation_context="berapa cadangan minyak Lapangan Duri tahun 2023?",
        )
        assert decision.allowed_tools == ("Simple Data Query",)
        assert (
            decision.classification.detected_entities.get("field_name", "").lower()
            == "rokan"
        )

    def test_ksmi_context_does_not_leak_into_current_factual_route(self) -> None:
        decision = route_query(
            "Berapa cadangan minyak Lapangan Duri tahun 2025?",
            conversation_context="Apa formula GROOVY untuk level E3?",
        )
        assert decision.allowed_tools == ("Simple Data Query",)
        assert decision.required_knowledge == ()

    def test_ordinary_query_has_no_visualization_flag(self) -> None:
        decision = route_query("Berapa cadangan minyak Lapangan Duri tahun 2025?")
        assert decision.visualization is False

    def test_explicit_visualization_request_sets_flag(self) -> None:
        decision = route_query("Buat peta sebaran lapangan Duri")
        assert decision.visualization is True
        assert "SQL Executor" in decision.allowed_tools

    def test_negated_visualization_request_clears_flag(self) -> None:
        decision = route_query(
            "Berapa cadangan lapangan Duri tahun 2025? Jangan buat peta."
        )
        assert decision.visualization is False

    def test_negation_filler_verb_does_not_hide_a_map_request(self) -> None:
        decision = route_query("Tolong jangan lupa buat peta sebaran lapangan Duri")
        assert decision.visualization is True

    def test_contrastive_negation_does_not_hide_a_map_request(self) -> None:
        decision = route_query(
            "Tidak hanya peta, tapi juga grafik lapangan Duri tahun 2025"
        )
        assert decision.visualization is True

    def test_morphological_map_request_sets_flag(self) -> None:
        decision = route_query("Buatkan pemetaan sebaran lapangan Duri")
        assert decision.visualization is True

    def test_visualization_flag_is_backward_compatible_default(self) -> None:
        decision = route_query("halo")
        assert decision.visualization is False

    def test_complete_production_question_ignores_old_ksmi(self) -> None:
        query = "Berapa produksi sumur tahun 2025?"
        alone = route_query(query)
        with_history = route_query(query, conversation_context="Bolehkah E3 ke E4?")
        assert with_history == alone
        assert with_history.classification.query_type == QueryType.COMPLEX_FACTUAL
        assert with_history.required_knowledge == ()
        assert "SQL Executor" in with_history.allowed_tools
        assert "Knowledge Traversal" not in with_history.allowed_tools

    @pytest.mark.parametrize(
        "query",
        [
            "Berapa hasil produksi tahun 2025?",
            "Tampilkan data tahun 2025?",
            "Berapa total cadangan minyak nasional tahun 2025?",
        ],
    )
    def test_complete_requests_are_independent_of_unrelated_history(
        self, query: str
    ) -> None:
        alone = route_query(query)
        with_ksmi = route_query(query, conversation_context="Bolehkah E3 ke E4?")
        with_document = route_query(
            query, conversation_context="surat persetujuan POD Duri revisi 1"
        )
        assert with_ksmi == alone
        assert with_document == alone

    @pytest.mark.parametrize("query", ["bagaimana tahun 2024?", "what about 2024?"])
    def test_year_only_follow_up_retains_prior_subject_and_current_year(
        self, query: str
    ) -> None:
        decision = route_query(
            query,
            conversation_context="berapa cadangan minyak Lapangan Duri tahun 2023?",
        )
        assert decision.classification.detected_entities.get("report_year") == "2024"
        assert (
            decision.classification.detected_entities.get("field_name", "").lower()
            == "duri"
        )

    def test_year_only_follow_up_with_unrelated_ksmi_keeps_discovery_route(self) -> None:
        decision = route_query(
            "bagaimana tahun 2024?", conversation_context="Bolehkah E3 ke E4?"
        )
        assert decision.classification.query_type != QueryType.KSMI
        assert decision.required_knowledge == ()
        assert "Knowledge Traversal" not in decision.allowed_tools

    def test_positive_chart_survives_negated_map(self) -> None:
        decision = route_query(
            "Jangan buat peta, tapi tampilkan grafik lapangan Duri"
        )
        assert decision.visualization is True

    @pytest.mark.parametrize(
        "query",
        [
            "No map, but show chart of Duri",
            "Jangan buat gambar, tapi tampilkan peta Duri",
            "Tampilkan grafik Duri, tanpa peta",
            "Tidak hanya peta, tapi juga grafik Duri",
            "Tolong jangan lupa buat peta Duri",
        ],
    )
    def test_affirmative_visualization_survives_local_negation(
        self, query: str
    ) -> None:
        assert route_query(query).visualization is True

    @pytest.mark.parametrize(
        "query",
        [
            "Jangan buat peta Duri",
            "Don't plot Duri",
            "Jangan buat peta atau grafik Duri",
            "Berapa cadangan lapangan Duri?",
        ],
    )
    def test_fully_negated_or_absent_visualization_is_false(self, query: str) -> None:
        assert route_query(query).visualization is False

    @pytest.mark.parametrize(
        "query",
        [
            "Jangan buat peta Duri atau grafik produksi",
            "Jangan buat peta, grafik, atau gambar Duri",
            "Do not show a map or chart of Duri",
            "Jangan buat peta Rokan atau grafik produksi",
            "Jangan buat peta, grafik, atau gambar Abadi LNG",
            "Do not show a map or chart of Rokan",
            "Jangan buat peta, gambar, atau grafik Duri",
            "No chart, plot, or map Rokan",
        ],
    )
    def test_negation_scope_covers_unknown_subjects_and_comma_lists(
        self, query: str
    ) -> None:
        assert route_query(query).visualization is False

    @pytest.mark.parametrize(
        "query",
        [
            "Jangan buat peta, visualisasikan produksi Duri",
            "Jangan buat peta, tolong tampilkan grafik Duri",
            "No map, please show a chart of Duri",
        ],
    )
    def test_affirmative_command_after_comma_resets_negation(
        self, query: str
    ) -> None:
        assert route_query(query).visualization is True


class TestConditionalToolPreservation:
    """Ensure route selection remains deliberately compact."""

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
        """Simple factual routes expose only the direct query tool."""
        classification = QueryClassification(
            query_type=QueryType.SIMPLE_FACTUAL,
            confidence=0.9,
            detected_entities={},
            suggested_table="field_resources",
            suggested_columns=["res_oc"],
            reason="Test",
        )

        classifier_tools = get_tools_for_classification(classification)
        assert classifier_tools == ["Simple Data Query"]

    def test_preservation_logic_ambiguous(self):
        """Ambiguous routes expose discovery tools without universal base tools."""
        classification = QueryClassification(
            query_type=QueryType.AMBIGUOUS,
            confidence=0.5,
            detected_entities={},
            suggested_table=None,
            suggested_columns=[],
            reason="Test",
        )

        classifier_tools = get_tools_for_classification(classification)
        assert classifier_tools == ["Entity Resolver", "Schema Inspector", "Table Lister"]

    def test_preservation_logic_no_conditional_tools(self):
        """No route receives unrelated tools through a hidden base set."""
        classification = QueryClassification(
            query_type=QueryType.SIMPLE_FACTUAL,
            confidence=0.9,
            detected_entities={},
            suggested_table="field_resources",
            suggested_columns=["res_oc"],
            reason="Test",
        )

        classifier_tools = get_tools_for_classification(classification)
        assert classifier_tools == ["Simple Data Query"]
