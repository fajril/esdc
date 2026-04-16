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
        assert "SQL Executor" in tools
        assert "Schema Inspector" in tools
        assert "Table Lister" in tools
        assert "Table Selector" in tools
        assert "Spatial Resolver" not in tools
        assert "Semantic Search" not in tools
        assert "Knowledge Traversal" not in tools

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
        assert "Knowledge Traversal" not in tools

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
        assert "Semantic Search" in tools
        assert "SQL Executor" in tools
        assert "Spatial Resolver" not in tools

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
        assert "Knowledge Traversal" in tools
        assert "SQL Executor" in tools
        assert "Spatial Resolver" in tools
        assert "Semantic Search" in tools

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
        assert "SQL Executor" in tools
        assert "Uncertainty Resolver" in tools
        assert "Problem Cluster Search" in tools
        assert "Spatial Resolver" not in tools
        assert "Semantic Search" not in tools

    def test_always_includes_schema_tools(self):
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
            for schema_tool in ("Schema Inspector", "Table Lister", "Table Selector"):
                assert schema_tool in tools, (
                    f"Schema tool {schema_tool} missing for {qtype}"
                )

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
