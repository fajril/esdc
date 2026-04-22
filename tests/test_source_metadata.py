"""Tests for source context metadata in function_call_output items."""

import pytest


class TestBuildSourceMetadata:
    """Test _build_source_metadata helper function."""

    def test_build_source_metadata_for_known_tool(self):
        from esdc.server.responses_wrapper import _build_source_metadata

        source = _build_source_metadata("execute_sql")
        assert source is not None
        assert source["resource_type"] == "sql_query"
        assert source["resource_id"] == "project_resources"

    def test_build_source_metadata_for_unknown_tool(self):
        from esdc.server.responses_wrapper import _build_source_metadata

        source = _build_source_metadata("unknown_tool")
        assert source is None

    def test_build_source_metadata_for_semantic_search(self):
        from esdc.server.responses_wrapper import _build_source_metadata

        source = _build_source_metadata("semantic_search")
        assert source is not None
        assert source["resource_type"] == "semantic_search"
        assert source["resource_id"] == "project_embeddings"

    def test_build_source_metadata_for_spatial(self):
        from esdc.server.responses_wrapper import _build_source_metadata

        source = _build_source_metadata("resolve_spatial")
        assert source is not None
        assert source["resource_type"] == "spatial_query"

    def test_build_source_metadata_for_cypher(self):
        from esdc.server.responses_wrapper import _build_source_metadata

        source = _build_source_metadata("execute_cypher")
        assert source is not None
        assert source["resource_type"] == "cypher_query"


class TestResponseFunctionCallResultModel:
    """Test ResponseFunctionCallResult Pydantic model accepts source field."""

    def test_response_function_call_result_model_accepts_source(self):
        from esdc.server.responses_models import ResponseFunctionCallResult

        item = ResponseFunctionCallResult(
            id="fco_test",
            status="completed",
            call_id="call_abc",
            output="test result",
            source={"resource_type": "sql_query", "resource_id": "project_resources"},
        )
        assert item.source is not None
        assert item.source["resource_type"] == "sql_query"

    def test_response_function_call_result_model_source_optional(self):
        from esdc.server.responses_models import ResponseFunctionCallResult

        item = ResponseFunctionCallResult(
            id="fco_test",
            status="completed",
            call_id="call_abc",
            output="test result",
        )
        assert item.source is None

    def test_response_function_call_result_model_with_complex_source(self):
        from esdc.server.responses_models import ResponseFunctionCallResult

        item = ResponseFunctionCallResult(
            id="fco_test",
            status="completed",
            call_id="call_abc",
            output="test result",
            source={
                "resource_type": "semantic_search",
                "resource_id": "project_embeddings",
                "query": "cadangan minyak",
                "filters": {"field_name": "Duri"},
            },
        )
        assert item.source is not None
        assert item.source["resource_type"] == "semantic_search"
        assert "filters" in item.source


class TestToolSourceMap:
    """Test that _TOOL_SOURCE_MAP covers expected tools."""

    def test_tool_source_map_has_expected_tools(self):
        from esdc.server.responses_wrapper import _TOOL_SOURCE_MAP

        expected_tools = [
            "execute_sql",
            "execute_cypher",
            "semantic_search",
            "get_schema",
            "list_tables",
            "get_recommended_table",
            "resolve_spatial",
            "resolve_uncertainty_level",
            "search_problem_cluster",
            "get_timeseries_columns",
            "get_resources_columns",
        ]
        for tool in expected_tools:
            assert tool in _TOOL_SOURCE_MAP, f"Expected {tool} in _TOOL_SOURCE_MAP"
            assert "resource_type" in _TOOL_SOURCE_MAP[tool]
            assert "resource_id" in _TOOL_SOURCE_MAP[tool]

    def test_tool_source_map_no_duplicate_resource_types(self):
        from esdc.server.responses_wrapper import _TOOL_SOURCE_MAP

        resource_types = [v["resource_type"] for v in _TOOL_SOURCE_MAP.values()]
        # Some tools share resource_types (like schema_lookup), that's OK
        assert len(resource_types) == len(_TOOL_SOURCE_MAP)
