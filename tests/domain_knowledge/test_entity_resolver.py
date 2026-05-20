"""Tests for Entity Resolver tool and resolver."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import duckdb
import pytest

from esdc.chat.domain_knowledge.entity_patterns import QueryPatternMatcher
from esdc.chat.domain_knowledge.entity_resolver_lib import EntityResolver
from esdc.chat.domain_knowledge.entity_schema import KGSchema


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
            (
                "u4",
                2024,
                "WID",
                "Widuri",
                "WID-WK",
                "WK Widuri",
                "PT Widuri",
                "Widuri Base",
                "1. Reserves & GRR",
                "1. Exploitation",
                "2. Middle Value",
                "contains duri as substring",
                "no change",
            ),
            (
                "u5",
                2024,
                "DURNFR",
                "Duri NFR",
                "ROK",
                "WK Rokan",
                "PT Pertamina Hulu Rokan",
                "Duri NFR Base",
                "1. Reserves & GRR",
                "1. Exploitation",
                "2. Middle Value",
                "prefix candidate",
                "no change",
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


class TestEntityResolver:
    def test_resolve_field_entity(self, mock_db: duckdb.DuckDBPyConnection):
        resolver = EntityResolver(db=mock_db)
        result = resolver.resolve("cadangan Duri 2024")
        assert result["status"] == "success"
        assert any(e["type"] == "Field" for e in result["entities"])
        field_entity = next(e for e in result["entities"] if e["type"] == "Field")
        assert "Duri" in field_entity["name"]

    def test_single_best_prefers_exact_field_over_prefix_and_substring(
        self, mock_db: duckdb.DuckDBPyConnection
    ):
        resolver = EntityResolver(db=mock_db)
        result = resolver.resolve("Duri", return_multiple=False)
        assert result["status"] == "success"
        field_entity = next(e for e in result["entities"] if e["type"] == "Field")
        assert field_entity["name"] == "Duri"
        assert field_entity["match_type"] == "exact"

    def test_resolve_year(self, mock_db: duckdb.DuckDBPyConnection):
        resolver = EntityResolver(db=mock_db)
        result = resolver.resolve("cadangan Duri 2024")
        year_entities = [e for e in result["entities"] if e["type"] == "Year"]
        assert len(year_entities) == 1
        assert year_entities[0]["value"] == 2024

    def test_resolve_uncertainty(self, mock_db: duckdb.DuckDBPyConnection):
        resolver = EntityResolver(db=mock_db)
        result = resolver.resolve("cadangan 2P Duri")
        assert any(e["type"] == "UncertaintyLevel" for e in result["entities"])

    def test_resolve_pattern(self, mock_db: duckdb.DuckDBPyConnection):
        resolver = EntityResolver(db=mock_db)
        result = resolver.resolve("cadangan Duri 2024")
        assert result["pattern"] is not None
        assert result["pattern"]["pattern_name"] == "cadangan"
        assert result["suggested_table"] == "field_resources"

    def test_resolve_where_conditions(self, mock_db: duckdb.DuckDBPyConnection):
        resolver = EntityResolver(db=mock_db)
        result = resolver.resolve("cadangan Duri 2024")
        assert "field_name = 'Duri'" in result["where_conditions"]
        assert "report_year = 2024" in result["where_conditions"]

    def test_ambiguous_returns_candidates(self, mock_db: duckdb.DuckDBPyConnection):
        resolver = EntityResolver(db=mock_db)
        result = resolver.resolve("cadangan Duri", return_multiple=True)
        assert result["status"] == "success"
        assert len(result["entities"]) >= 1

    def test_fallback_on_no_match(self, mock_db: duckdb.DuckDBPyConnection):
        resolver = EntityResolver(db=mock_db)
        result = resolver.resolve("xyzzy foobar baz 9999")
        assert result["status"] == "failed"
        assert result["fallback"] == "multi_round"

    def test_working_area_entity(self, mock_db: duckdb.DuckDBPyConnection):
        resolver = EntityResolver(db=mock_db)
        result = resolver.resolve("data di WK Rokan 2024")
        assert any(e["type"] == "WorkingArea" for e in result["entities"])

    def test_working_area_prefers_wa_resources(
        self, mock_db: duckdb.DuckDBPyConnection
    ):
        resolver = EntityResolver(db=mock_db)
        result = resolver.resolve("cadangan WK Rokan 2024")
        assert result["suggested_table"] == "wa_resources"
        assert "wk_name = 'WK Rokan'" in result["where_conditions"]

    def test_project_entity_prefers_project_resources(
        self, mock_db: duckdb.DuckDBPyConnection
    ):
        resolver = EntityResolver(db=mock_db)
        result = resolver.resolve("cadangan proyek Abadi LNG 2024")
        assert result["status"] == "success"
        assert result["suggested_table"] == "project_resources"
        assert any(e.get("entity_type") == "project_name" for e in result["entities"])
        assert "project_name = 'Abadi LNG'" in result["where_conditions"]

    def test_operator_entity_prefers_project_resources(
        self, mock_db: duckdb.DuckDBPyConnection
    ):
        resolver = EntityResolver(db=mock_db)
        result = resolver.resolve("cadangan operator Pertamina Hulu Rokan 2024")
        assert result["status"] == "success"
        assert result["suggested_table"] == "project_resources"
        assert any(e.get("entity_type") == "operator_name" for e in result["entities"])
        assert "operator_name = 'PT Pertamina Hulu Rokan'" in result["where_conditions"]

    def test_multi_entity_working_area_and_operator(
        self, mock_db: duckdb.DuckDBPyConnection
    ):
        resolver = EntityResolver(db=mock_db)
        result = resolver.resolve("top proyek di WK Rokan oleh Pertamina")
        entity_types = {e["entity_type"] for e in result["entities"]}
        assert {"wk_name", "operator_name"}.issubset(entity_types)
        assert "wk_name = 'WK Rokan'" in result["where_conditions"]
        assert "operator_name = 'PT Pertamina Hulu Rokan'" in result["where_conditions"]

    def test_determine_table_for_production(self, mock_db: duckdb.DuckDBPyConnection):
        resolver = EntityResolver(db=mock_db)
        result = resolver.resolve("profil produksi Duri 2024")
        assert result["suggested_table"] == "field_timeseries"

    def test_determine_table_for_reserves(self, mock_db: duckdb.DuckDBPyConnection):
        resolver = EntityResolver(db=mock_db)
        result = resolver.resolve("cadangan Duri 2024")
        assert result["suggested_table"] == "field_resources"


class TestEntityResolverTool:
    @patch("esdc.chat.tools.get_db_connection")
    def test_tool_success(self, mock_get_db, mock_db: duckdb.DuckDBPyConnection):
        mock_get_db.return_value = mock_db
        from esdc.chat.tools import entity_resolver

        result = entity_resolver.invoke({"query": "cadangan Duri 2024"})
        parsed = json.loads(result)
        assert parsed["status"] == "success"
        assert "entities" in parsed

    @patch("esdc.chat.tools.get_db_connection")
    def test_tool_returns_json(self, mock_get_db, mock_db: duckdb.DuckDBPyConnection):
        mock_get_db.return_value = mock_db
        from esdc.chat.tools import entity_resolver

        result = entity_resolver.invoke({"query": "profil produksi Abadi"})
        parsed = json.loads(result)
        assert "status" in parsed
        assert "entities" in parsed
        assert "pattern" in parsed
        assert "suggested_table" in parsed
        assert "where_conditions" in parsed

    @patch("esdc.chat.tools._get_tool_cache")
    @patch("esdc.chat.domain_knowledge.entity_resolver_lib.EntityResolver")
    @patch("esdc.chat.tools.get_db_connection")
    def test_tool_defaults_return_multiple_true(
        self,
        mock_get_db,
        mock_resolver_class,
        mock_get_tool_cache,
        mock_db: duckdb.DuckDBPyConnection,
    ):
        class EmptyCache(dict):
            def set(self, key, value):
                self[key] = value

        mock_get_tool_cache.return_value = EmptyCache()
        mock_get_db.return_value = mock_db
        mock_resolver = mock_resolver_class.return_value
        mock_resolver.resolve.return_value = {
            "status": "success",
            "entities": [],
            "pattern": None,
            "suggested_table": None,
            "where_conditions": [],
            "required_columns": [],
            "confidence": 0.0,
        }
        from esdc.chat.tools import entity_resolver

        query = "__return_multiple_default_test__"
        entity_resolver.invoke({"query": query})

        mock_resolver.resolve.assert_called_once_with(
            query=query, return_multiple=True
        )
