"""Tests for entity resolution WK enrichment."""

from esdc.knowledge_graph.resolver import KnowledgeTraversalResolver


class TestWKEnrichment:
    """Test that entity resolution includes WK context in WHERE conditions."""

    def test_where_conditions_include_wk_for_working_area(self):
        """When both Field and WorkingArea are resolved, WHERE should include wk_name."""
        entities = [
            {
                "type": "Field",
                "filter_column": "field_name",
                "name": "Tambora",
                "confidence": 0.9,
            },
            {
                "type": "WorkingArea",
                "filter_column": "wk_name",
                "name": "Mahakam",
                "confidence": 1.0,
            },
        ]
        resolver = KnowledgeTraversalResolver.__new__(KnowledgeTraversalResolver)
        conditions = resolver._build_where_conditions(entities)

        has_field = any("field_name" in c for c in conditions)
        has_wk = any("wk_name" in c for c in conditions)
        assert has_field, f"Missing field_name condition: {conditions}"
        assert has_wk, f"Missing wk_name condition: {conditions}"

    def test_where_conditions_field_only(self):
        """When only a Field is resolved, no wk_name condition is added."""
        entities = [
            {
                "type": "Field",
                "filter_column": "field_name",
                "name": "Duri",
                "confidence": 0.9,
            },
        ]
        resolver = KnowledgeTraversalResolver.__new__(KnowledgeTraversalResolver)
        conditions = resolver._build_where_conditions(entities)

        has_field = any("field_name" in c for c in conditions)
        has_wk = any("wk_name" in c for c in conditions)
        assert has_field
        assert not has_wk, f"Unexpected wk_name condition: {conditions}"

    def test_where_conditions_year_only(self):
        """Year entity should produce report_year condition."""
        entities = [
            {
                "type": "Year",
                "value": 2024,
                "name": "2024",
                "confidence": 1.0,
            },
        ]
        resolver = KnowledgeTraversalResolver.__new__(KnowledgeTraversalResolver)
        conditions = resolver._build_where_conditions(entities)

        assert any("report_year" in c for c in conditions), (
            f"Missing year: {conditions}"
        )
