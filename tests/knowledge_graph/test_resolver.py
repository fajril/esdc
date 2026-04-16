"""Tests for Knowledge Traversal tool and resolver."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import duckdb
import pytest

from esdc.knowledge_graph.patterns import QueryPatternMatcher
from esdc.knowledge_graph.resolver import KnowledgeTraversalResolver
from esdc.knowledge_graph.schema import KGSchema


@pytest.fixture
def schema() -> KGSchema:
    schema_path = (
        Path(__file__).parent.parent.parent
        / "esdc"
        / "chat"
        / "domain_knowledge"
        / "graph_schema.yaml"
    )
    return KGSchema(schema_path=str(schema_path))


@pytest.fixture
def mock_db() -> duckdb.DuckDBPyConnection:
    conn = duckdb.connect(":memory:")
    conn.execute("""
        CREATE TABLE project_resources (
            uuid TEXT,
            report_year INTEGER,
            field_id TEXT,
            field_name TEXT,
            wk_id TEXT,
            wk_name TEXT,
            operator_name TEXT,
            project_name TEXT,
            project_class TEXT,
            project_stage TEXT,
            uncert_level TEXT,
            project_remarks TEXT,
            vol_remarks TEXT
        )
    """)
    conn.executemany(
        "INSERT INTO project_resources VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (
                "u1",
                2024,
                "DUR",
                "Duri",
                "ROK",
                "WK Rokan",
                "PT Pertamina Hulu Rokan",
                "Duri Phase 1",
                "1. Reserves & GRR",
                "1. Exploitation",
                "2. Middle Value",
                "water cut increasing",
                "volume adjustment",
            ),
            (
                "u2",
                2024,
                "ABD",
                "Abadi",
                "MAS",
                "WK Masela",
                "INPEX Masela",
                "Abadi LNG",
                "1. Reserves & GRR",
                "1. Exploitation",
                "2. Middle Value",
                "on schedule",
                "no change",
            ),
            (
                "u3",
                2024,
                "DUR",
                "Duri Selatan",
                "ROK",
                "WK Rokan",
                "PT Pertamina Hulu Rokan",
                "Duri Selatan",
                "1. Reserves & GRR",
                "1. Exploitation",
                "1. Low Value",
                "minor issues",
                "gas decline",
            ),
        ],
    )
    return conn


class TestKGSchema:
    def test_schema_loads(self, schema: KGSchema):
        assert len(schema.entity_types) > 0
        assert "Project" in schema.entity_types
        assert "Field" in schema.entity_types
        assert "Report" in schema.entity_types

    def test_schema_relationships(self, schema: KGSchema):
        assert "PROJECT_BELONGS_TO_FIELD" in schema.relationships
        assert "FIELD_HAS_RESERVES" in schema.relationships

    def test_schema_query_patterns(self, schema: KGSchema):
        assert "cadangan" in schema.query_patterns
        assert "profil_produksi" in schema.query_patterns
        assert "field_complete_status" in schema.query_patterns

    def test_get_pattern_for_keywords(self, schema: KGSchema):
        patterns = schema.get_pattern_for_keywords(["cadangan", "reserves"])
        assert len(patterns) > 0
        assert any(p["name"] == "cadangan" for p in patterns)

    def test_get_suggested_table(self, schema: KGSchema):
        table = schema.get_suggested_table("cadangan")
        assert table == "field_resources"

    def test_get_suggested_columns(self, schema: KGSchema):
        cols = schema.get_suggested_columns("cadangan")
        assert "res_oc" in cols
        assert "res_an" in cols

    def test_enum_values(self, schema: KGSchema):
        values = schema.get_enum_values("project_class")
        assert "1. Reserves & GRR" in values


class TestQueryPatternMatcher:
    def test_match_cadangan(self, schema: KGSchema):
        matcher = QueryPatternMatcher(schema)
        result = matcher.match("cadangan Duri 2024")
        assert result is not None
        assert result["pattern_name"] == "cadangan"
        assert result["suggested_table"] == "field_resources"

    def test_match_profil_produksi(self, schema: KGSchema):
        matcher = QueryPatternMatcher(schema)
        result = matcher.match("profil produksi Abadi")
        assert result is not None
        assert result["pattern_name"] == "profil_produksi"

    def test_match_top(self, schema: KGSchema):
        matcher = QueryPatternMatcher(schema)
        result = matcher.match("top 5 lapangan terbesar")
        assert result is not None
        assert result["pattern_name"] == "top_n"

    def test_no_match_gibberish(self, schema: KGSchema):
        matcher = QueryPatternMatcher(schema)
        result = matcher.match("xyzzy foobar baz")
        assert result is None


class TestKnowledgeTraversalResolver:
    def test_resolve_field_entity(self, mock_db: duckdb.DuckDBPyConnection):
        resolver = KnowledgeTraversalResolver(db=mock_db)
        result = resolver.resolve("cadangan Duri 2024")
        assert result["status"] == "success"
        assert any(e["type"] == "Field" for e in result["entities"])
        field_entity = next(e for e in result["entities"] if e["type"] == "Field")
        assert "Duri" in field_entity["name"]

    def test_resolve_year(self, mock_db: duckdb.DuckDBPyConnection):
        resolver = KnowledgeTraversalResolver(db=mock_db)
        result = resolver.resolve("cadangan Duri 2024")
        year_entities = [e for e in result["entities"] if e["type"] == "Year"]
        assert len(year_entities) == 1
        assert year_entities[0]["value"] == 2024

    def test_resolve_uncertainty(self, mock_db: duckdb.DuckDBPyConnection):
        resolver = KnowledgeTraversalResolver(db=mock_db)
        result = resolver.resolve("cadangan 2P Duri")
        assert any(e["type"] == "UncertaintyLevel" for e in result["entities"])

    def test_resolve_pattern(self, mock_db: duckdb.DuckDBPyConnection):
        resolver = KnowledgeTraversalResolver(db=mock_db)
        result = resolver.resolve("cadangan Duri 2024")
        assert result["pattern"] is not None
        assert result["pattern"]["pattern_name"] == "cadangan"
        assert result["suggested_table"] == "field_resources"

    def test_resolve_where_conditions(self, mock_db: duckdb.DuckDBPyConnection):
        resolver = KnowledgeTraversalResolver(db=mock_db)
        result = resolver.resolve("cadangan Duri 2024")
        assert "field_name = 'Duri'" in result["where_conditions"]
        assert "report_year = 2024" in result["where_conditions"]

    def test_ambiguous_returns_candidates(self, mock_db: duckdb.DuckDBPyConnection):
        resolver = KnowledgeTraversalResolver(db=mock_db)
        result = resolver.resolve("cadangan Duri", return_multiple=True)
        assert result["status"] == "success"
        assert len(result["entities"]) >= 1

    def test_fallback_on_no_match(self, mock_db: duckdb.DuckDBPyConnection):
        resolver = KnowledgeTraversalResolver(db=mock_db)
        result = resolver.resolve("xyzzy foobar baz 9999")
        assert result["status"] == "failed"
        assert result["fallback"] == "multi_round"

    def test_working_area_entity(self, mock_db: duckdb.DuckDBPyConnection):
        resolver = KnowledgeTraversalResolver(db=mock_db)
        result = resolver.resolve("data di WK Rokan 2024")
        assert any(e["type"] == "WorkingArea" for e in result["entities"])

    def test_determine_table_for_production(self, mock_db: duckdb.DuckDBPyConnection):
        resolver = KnowledgeTraversalResolver(db=mock_db)
        result = resolver.resolve("profil produksi Duri 2024")
        assert result["suggested_table"] == "field_timeseries"

    def test_determine_table_for_reserves(self, mock_db: duckdb.DuckDBPyConnection):
        resolver = KnowledgeTraversalResolver(db=mock_db)
        result = resolver.resolve("cadangan Duri 2024")
        assert result["suggested_table"] == "field_resources"


class TestKnowledgeTraversalTool:
    @patch("esdc.chat.tools.get_db_connection")
    def test_tool_success(self, mock_get_db, mock_db: duckdb.DuckDBPyConnection):
        mock_get_db.return_value = mock_db
        from esdc.chat.tools import knowledge_traversal

        result = knowledge_traversal.invoke({"query": "cadangan Duri 2024"})
        parsed = json.loads(result)
        assert parsed["status"] == "success"
        assert "entities" in parsed

    @patch("esdc.chat.tools.get_db_connection")
    def test_tool_returns_json(self, mock_get_db, mock_db: duckdb.DuckDBPyConnection):
        mock_get_db.return_value = mock_db
        from esdc.chat.tools import knowledge_traversal

        result = knowledge_traversal.invoke({"query": "profil produksi Abadi"})
        parsed = json.loads(result)
        assert "status" in parsed
        assert "entities" in parsed
        assert "pattern" in parsed
        assert "suggested_table" in parsed
        assert "where_conditions" in parsed
