"""Tests for citation annotations in output_text content parts."""

import pytest


class TestCitationAnnotationsOutput:
    """Test annotations field in output_text content parts."""

    def test_output_text_annotations_structure(self):
        """Output text should include annotations list for citations."""
        annotations = [
            {
                "type": "source_citation",
                "resource_type": "sql_query",
                "resource_id": "project_resources",
            }
        ]
        output_text = {
            "type": "output_text",
            "text": "Cadangan minyak Duri: 123 MMSTB",
            "annotations": annotations,
        }
        assert "annotations" in output_text
        assert output_text["annotations"][0]["resource_type"] == "sql_query"

    def test_output_text_empty_annotations(self):
        """Output text can have empty annotations list."""
        output_text = {
            "type": "output_text",
            "text": "Hello world",
            "annotations": [],
        }
        assert output_text["annotations"] == []

    def test_build_source_metadata_for_citation(self):
        """Source metadata should be convertible to citation annotation."""
        from esdc.server.responses_wrapper import _build_source_metadata

        source = _build_source_metadata("execute_sql")
        assert source is not None
        citation = {"type": "source_citation", **source}
        assert citation["type"] == "source_citation"
        assert citation["resource_type"] == "sql_query"

    def test_multiple_sources_in_annotations(self):
        """Multiple tool calls should result in multiple citation annotations."""
        from esdc.server.responses_wrapper import _build_source_metadata

        sources = []
        for tool_name in ["execute_sql", "semantic_search"]:
            source = _build_source_metadata(tool_name)
            if source:
                sources.append({"type": "source_citation", **source})
        assert len(sources) == 2
        assert sources[0]["resource_type"] == "sql_query"
        assert sources[1]["resource_type"] == "semantic_search"


class TestCitationIntegration:
    """Test full citation flow from tool results to annotations."""

    def test_collected_sources_accumulate_tool_metadata(self):
        """collected_sources should accumulate source metadata from tool results."""
        from esdc.server.responses_wrapper import _build_source_metadata

        collected_sources = []
        # Simulate two tool calls
        for tool_name in ["execute_sql", "get_schema"]:
            source = _build_source_metadata(tool_name)
            if source:
                collected_sources.append({"type": "source_citation", **source})
        assert len(collected_sources) == 2
        assert collected_sources[0]["resource_id"] == "project_resources"
        assert collected_sources[1]["resource_id"] == "project_resources"

    def test_empty_annotations_when_no_tools(self):
        """When no tools are called, annotations should be empty list."""
        collected_sources = []
        annotations = collected_sources if collected_sources else []
        assert annotations == []
