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


class TestResolveName:
    """resolve_name(): a targeted, no-NL-parsing lookup against one spec.

    Returns FINAL confident picks (>= 0.8, word-boundary guarded) -- callers
    no longer re-filter by confidence.
    """

    def test_whole_string_wins_first_single_best_only(
        self, mock_db: duckdb.DuckDBPyConnection
    ):
        """An exact whole-string hit short-circuits the fallback chain and returns only the single best match -- not every row that happens to contain the term (no fan-out from the whole-string step)."""
        resolver = EntityResolver(db=mock_db)
        matches = resolver.resolve_name("Duri", "field_name")
        assert len(matches) == 1
        assert matches[0]["name"] == "Duri"
        assert matches[0]["confidence"] == 1.0

    def test_unknown_entity_type_raises(self, mock_db: duckdb.DuckDBPyConnection):
        resolver = EntityResolver(db=mock_db)
        with pytest.raises(ValueError):
            resolver.resolve_name("Duri", "not_a_real_entity_type")

    def test_domain_word_stripping_resolves_via_normalized_phrase(
        self, mock_db: duckdb.DuckDBPyConnection
    ):
        """Unlike resolve(), there's no NL keyword-candidate extraction, but the fallback chain's step 2 does strip domain boilerplate words (including resolve_name-local noise words like "kerja") so a phrase that doesn't match verbatim can still resolve via its stripped form."""
        resolver = EntityResolver(db=mock_db)
        matches = resolver.resolve_name("wilayah kerja Rokan", "wk_name")
        assert len(matches) == 1
        assert matches[0]["name"] == "WK Rokan"
        # The literal stored value still matches directly (whole-string step).
        assert resolver.resolve_name("WK Rokan", "wk_name")[0]["name"] == "WK Rokan"

    def test_empty_name_returns_empty(self, mock_db: duckdb.DuckDBPyConnection):
        resolver = EntityResolver(db=mock_db)
        assert resolver.resolve_name("", "field_name") == []
        assert resolver.resolve_name("   ", "field_name") == []


class TestResolveNameParentFilter:
    """parent_filter constrains resolution to rows under a resolved parent."""

    def test_resolve_name_with_parent_filter(self, mock_db: duckdb.DuckDBPyConnection):
        """Filtered resolution returns only entities under the parent."""
        resolver = EntityResolver(db=mock_db)
        # Duri is under WK Rokan, Widuri is under WK Widuri
        results = resolver.resolve_name(
            "Duri", "field_name", parent_filter={"wk_name": "WK Rokan"}
        )
        names = [r["name"] for r in results]
        assert "Duri" in names
        assert "Widuri" not in names

    def test_resolve_name_parent_filter_no_match_fallback(
        self, mock_db: duckdb.DuckDBPyConnection
    ):
        """When filter yields nothing, returns empty (caller handles fallback)."""
        resolver = EntityResolver(db=mock_db)
        results = resolver.resolve_name(
            "Widuri", "field_name", parent_filter={"wk_name": "WK Rokan"}
        )
        assert results == []

    def test_resolve_name_parent_filter_none(self, mock_db: duckdb.DuckDBPyConnection):
        """parent_filter=None behaves like before (no filtering)."""
        resolver = EntityResolver(db=mock_db)
        results = resolver.resolve_name("Duri", "field_name", parent_filter=None)
        names = [r["name"] for r in results]
        assert "Duri" in names

    def test_resolve_name_project_with_field_filter(
        self, mock_db: duckdb.DuckDBPyConnection
    ):
        """Project resolution filtered by field_name + wk_name."""
        resolver = EntityResolver(db=mock_db)
        results = resolver.resolve_name(
            "Duri Phase 1",
            "project_name",
            parent_filter={"wk_name": "WK Rokan", "field_name": "Duri"},
        )
        names = [r["name"] for r in results]
        assert "Duri Phase 1" in names

    def test_resolve_name_parent_filter_applies_to_name_branch(
        self, mock_db: duckdb.DuckDBPyConnection
    ):
        """Parent filter must constrain name matches, not just id matches (OR/AND precedence)."""
        resolver = EntityResolver(db=mock_db)
        # "Duri" matches field_name directly; wrong wk must exclude it
        results = resolver.resolve_name(
            "Duri", "field_name", parent_filter={"wk_name": "WK Mahakam"}
        )
        assert results == []


class TestSuggestNames:
    """suggest_names(): fuzzy-ranked canonical candidates for unresolvable names.

    resolve_name's ILIKE lookup finds nothing for a typo like "Durri", so
    this is the copy-paste escape hatch surfaced in unresolved warnings.
    """

    def test_typo_returns_fuzzy_ranked_names(self, mock_db: duckdb.DuckDBPyConnection):
        resolver = EntityResolver(db=mock_db)
        names = resolver.suggest_names("Durri", "field_name")
        assert names[0] == "Duri"  # highest similarity first
        assert "Abadi" not in names  # below the 0.7 similarity floor

    def test_typo_against_wk_name(self, mock_db: duckdb.DuckDBPyConnection):
        resolver = EntityResolver(db=mock_db)
        assert resolver.suggest_names("Rokann", "wk_name") == ["WK Rokan"]

    def test_garbage_returns_empty(self, mock_db: duckdb.DuckDBPyConnection):
        resolver = EntityResolver(db=mock_db)
        assert resolver.suggest_names("zzzzqqq", "field_name") == []

    def test_blank_returns_empty(self, mock_db: duckdb.DuckDBPyConnection):
        resolver = EntityResolver(db=mock_db)
        assert resolver.suggest_names("", "field_name") == []
        assert resolver.suggest_names("   ", "field_name") == []

    def test_unknown_entity_type_raises(self, mock_db: duckdb.DuckDBPyConnection):
        resolver = EntityResolver(db=mock_db)
        with pytest.raises(ValueError):
            resolver.suggest_names("Duri", "not_a_real_entity_type")

    def test_limit_respected(self, mock_db: duckdb.DuckDBPyConnection):
        resolver = EntityResolver(db=mock_db)
        names = resolver.suggest_names("Durri", "field_name", limit=1)
        assert names == ["Duri"]


@pytest.fixture
def multi_entity_db() -> duckdb.DuckDBPyConnection:
    """DB fixture mirroring the real-world Arung/Nowera/Garung/Duri cases."""
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
    rows = [
        ("ARUNG", "ARUNG - BASE", "South Sumatera"),
        ("NOWERA", "NOWERA - BASE", "South Sumatera"),
        ("GARUNG", "GARUNG - BASE", "Garung"),
        ("PEMARUNG", "PEMARUNG BASE", "Garung"),
        ("DURI", "Duri Phase 1", "WK Rokan"),
        ("DURI UTARA", "Duri Utara Phase 1", "WK Rokan"),
    ]
    conn.executemany(
        "INSERT INTO project_resources VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (
                f"u-{field_name}",
                2024,
                field_name,
                field_name,
                wk_name,
                wk_name,
                "Operator",
                project_name,
                "1. Reserves & GRR",
                "1. Exploitation",
                "2. Middle Value",
                "",
                "",
            )
            for field_name, project_name, wk_name in rows
        ],
    )
    return conn


class TestResolveNameFallbackChain:
    """The full whole-string -> normalized-phrase -> segmentation chain."""

    def test_splits_multi_entity_string(
        self, multi_entity_db: duckdb.DuckDBPyConnection
    ):
        resolver = EntityResolver(db=multi_entity_db)
        matches = resolver.resolve_name("Arung Nowera", "field_name")
        names = sorted(m["name"] for m in matches)
        assert names == ["ARUNG", "NOWERA"]

    def test_strips_indonesian_domain_words_then_splits(
        self, multi_entity_db: duckdb.DuckDBPyConnection
    ):
        resolver = EntityResolver(db=multi_entity_db)
        matches = resolver.resolve_name(
            "Proyek Pengembangan Lapangan Arung Nowera", "project_name"
        )
        names = sorted(m["name"] for m in matches)
        assert names == ["ARUNG - BASE", "NOWERA - BASE"]

    def test_boundary_guard_rejects_non_boundary_substring(
        self, multi_entity_db: duckdb.DuckDBPyConnection
    ):
        """Arung must not match GARUNG/PEMARUNG (no word-boundary containment) even though a bare ILIKE substring test would clear the 0.8 threshold."""
        resolver = EntityResolver(db=multi_entity_db)
        matches = resolver.resolve_name("arung", "field_name")
        names = [m["name"] for m in matches]
        assert "GARUNG" not in names
        assert "PEMARUNG" not in names
        assert names == ["ARUNG"]

    def test_boundary_guard_rejects_non_boundary_substring_in_project_name(
        self, multi_entity_db: duckdb.DuckDBPyConnection
    ):
        resolver = EntityResolver(db=multi_entity_db)
        matches = resolver.resolve_name("arung", "project_name")
        names = [m["name"] for m in matches]
        assert "GARUNG - BASE" not in names
        assert "PEMARUNG BASE" not in names
        assert names == ["ARUNG - BASE"]

    def test_no_fan_out_when_full_phrase_matches(
        self, multi_entity_db: duckdb.DuckDBPyConnection
    ):
        """Duri Utara must resolve only to "DURI UTARA", not also "DURI" -- the whole-string step wins before segmentation could split it."""
        resolver = EntityResolver(db=multi_entity_db)
        matches = resolver.resolve_name("Duri Utara", "field_name")
        assert [m["name"] for m in matches] == ["DURI UTARA"]

    def test_stopword_token_skipped_in_segmentation(
        self, multi_entity_db: duckdb.DuckDBPyConnection
    ):
        resolver = EntityResolver(db=multi_entity_db)
        matches = resolver.resolve_name("Arung dan Nowera", "field_name")
        names = sorted(m["name"] for m in matches)
        assert names == ["ARUNG", "NOWERA"]

    def test_whole_string_still_wins_first_for_exact_name(
        self, multi_entity_db: duckdb.DuckDBPyConnection
    ):
        resolver = EntityResolver(db=multi_entity_db)
        matches = resolver.resolve_name("ARUNG - BASE", "project_name")
        assert len(matches) == 1
        assert matches[0]["name"] == "ARUNG - BASE"
        assert matches[0]["confidence"] == 1.0


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

        mock_resolver.resolve.assert_called_once_with(query=query, return_multiple=True)
