"""Tests for dynamic tool binding based on query classification."""

from esdc.chat.query_classifier import (
    QueryClassification,
    QueryType,
    get_tools_for_classification,
)


class TestDynamicToolBinding:
    """Test that get_tools_for_classification returns correct tool sets."""

    def test_simple_factual_gets_minimal_tools(self):
        classification = QueryClassification(
            query_type=QueryType.SIMPLE_FACTUAL,
            confidence=0.9,
            detected_entities={"field_name": "Duri"},
            suggested_table="field_resources",
            suggested_columns=["res_oc"],
            reason="Matched reserves pattern",
        )
        tools = get_tools_for_classification(classification)
        assert tools == ["Simple Data Query"]

    def test_spatial_gets_spatial_plus_sql(self):
        classification = QueryClassification(
            query_type=QueryType.SPATIAL,
            confidence=0.85,
            detected_entities={"field_name": "Duri"},
            suggested_table=None,
            suggested_columns=[],
            reason="Proximity query detected",
        )
        tools = get_tools_for_classification(classification)
        assert "Spatial Resolver" in tools
        assert "SQL Executor" in tools
        assert "Semantic Search" not in tools
        assert tools == ["Spatial Resolver", "SQL Executor"]

    def test_conceptual_gets_semantic_plus_sql(self):
        classification = QueryClassification(
            query_type=QueryType.CONCEPTUAL,
            confidence=0.9,
            detected_entities={},
            suggested_table=None,
            suggested_columns=[],
            reason="Economic issues",
        )
        tools = get_tools_for_classification(classification)
        assert tools == ["Semantic Search", "SQL Executor"]

    def test_ambiguous_gets_full_tool_set(self):
        classification = QueryClassification(
            query_type=QueryType.AMBIGUOUS,
            confidence=0.5,
            detected_entities={},
            suggested_table=None,
            suggested_columns=[],
            reason="No clear pattern match",
        )
        tools = get_tools_for_classification(classification)
        assert tools == ["Entity Resolver", "Schema Inspector", "Table Lister"]

    def test_complex_factual_gets_targeted_tools(self):
        classification = QueryClassification(
            query_type=QueryType.COMPLEX_FACTUAL,
            confidence=0.7,
            detected_entities={"field_name": "Duri"},
            suggested_table="project_resources",
            suggested_columns=["project_name", "project_remarks"],
            reason="Entities detected but no simple pattern match",
        )
        tools = get_tools_for_classification(classification)
        assert tools == [
            "SQL Executor",
            "Entity Resolver",
            "Schema Inspector",
            "Resources Column Guide",
            "Uncertainty Resolver",
        ]

    def test_routes_include_only_their_declared_tools(self):
        expected = {
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
            assert tools == expected[qtype]

    def test_tool_names_are_langchain_decorator_names(self):
        """Verify tool names match @tool() decorator strings, not Python names."""
        classification = QueryClassification(
            query_type=QueryType.SIMPLE_FACTUAL,
            confidence=0.9,
            detected_entities={},
            suggested_table="field_resources",
            suggested_columns=["res_oc"],
            reason="Test",
        )
        tools = get_tools_for_classification(classification)
        python_names = {
            "execute_sql",
            "get_schema",
            "list_tables",
            "resolve_spatial",
            "semantic_search",
        }
        for tool_name in tools:
            assert tool_name not in python_names, (
                f"Tool name {tool_name} looks like a Python function name, "
                f"not a LangChain decorator name"
            )


def test_merge_allowed_tools_respects_classifier_restriction():
    from esdc.chat.agent import _merge_allowed_tools

    allowed = _merge_allowed_tools(
        ["execute_sql", "get_schema"],
        {"run_command"},  # conditionally-registered (e.g. OpenTerminal)
    )
    assert "execute_sql" in allowed
    assert "run_command" in allowed
    # The old bug unioned in ALL registered tools; a non-selected,
    # non-conditional tool must NOT be allowed.
    assert "semantic_search" not in allowed
