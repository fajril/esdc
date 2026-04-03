"""Tests for domain_knowledge module."""

import pytest
from esdc.chat.domain_knowledge import (
    DOMAIN_CONCEPTS,
    SYNONYMS,
    COLUMN_METADATA,
    UNCERTAINTY_MAP,
    UncertaintySpec,
    TABLE_HIERARCHY,
    AGGREGATION_LEVELS,
    resolve_concept,
    get_uncertainty_filter,
    get_uncertainty_spec,
    build_uncertainty_sql,
    get_project_class_filter,
    get_volume_columns,
    get_columns_for_substance,
    get_column_group,
    format_response_value,
    build_sql_pattern,
    get_table_for_query,
    can_use_view_for_calculation,
    get_entity_filter_column,
    build_aggregate_query,
    get_aggregation_table_info,
    get_recommended_table,
)


class TestDomainConcepts:
    """Tests for DOMAIN_CONCEPTS structure."""

    def test_uncertainty_levels_exist(self):
        """Test that uncertainty levels are defined."""
        assert "uncertainty_levels" in DOMAIN_CONCEPTS
        assert "1P" in DOMAIN_CONCEPTS["uncertainty_levels"]
        assert "2P" in DOMAIN_CONCEPTS["uncertainty_levels"]
        assert "3P" in DOMAIN_CONCEPTS["uncertainty_levels"]
        assert "1R" in DOMAIN_CONCEPTS["uncertainty_levels"]
        assert "2R" in DOMAIN_CONCEPTS["uncertainty_levels"]
        assert "3R" in DOMAIN_CONCEPTS["uncertainty_levels"]

    def test_project_classes_exist(self):
        """Test that project classes are defined."""
        assert "project_classes" in DOMAIN_CONCEPTS
        assert "reserves" in DOMAIN_CONCEPTS["project_classes"]
        assert "grr" in DOMAIN_CONCEPTS["project_classes"]
        assert "contingent" in DOMAIN_CONCEPTS["project_classes"]
        assert "prospective" in DOMAIN_CONCEPTS["project_classes"]

    def test_volume_types_exist(self):
        """Test that volume types are defined."""
        assert "volume_types" in DOMAIN_CONCEPTS
        assert "cadangan" in DOMAIN_CONCEPTS["volume_types"]
        assert "sumber_daya" in DOMAIN_CONCEPTS["volume_types"]

    def test_uncertainty_db_values(self):
        """Test that uncertainty levels map to correct DB values."""
        assert DOMAIN_CONCEPTS["uncertainty_levels"]["1P"]["db_value"] == "1. Low Value"
        assert DOMAIN_CONCEPTS["uncertainty_levels"]["2P"]["db_value"] == "2. Middle Value"
        assert DOMAIN_CONCEPTS["uncertainty_levels"]["3P"]["db_value"] == "3. High Value"


class TestSynonyms:
    """Tests for SYNONYMS mapping."""

    def test_indonesian_synonyms(self):
        """Test Indonesian term synonyms."""
        assert SYNONYMS["cadangan"] == "cadangan"
        assert SYNONYMS["sumber daya"] == "sumber_daya"
        assert SYNONYMS["terbukti"] == "proven"
        assert SYNONYMS["mungkin"] == "probable"
        assert SYNONYMS["harapan"] == "possible"

    def test_english_synonyms(self):
        """Test English term synonyms."""
        assert SYNONYMS["reserves"] == "cadangan"
        assert SYNONYMS["resources"] == "sumber_daya"
        assert SYNONYMS["proven"] == "proven"
        assert SYNONYMS["probable"] == "probable"

    def test_abbreviation_synonyms(self):
        """Test abbreviation synonyms."""
        assert SYNONYMS["1p"] == "1P"
        assert SYNONYMS["2p"] == "2P"
        assert SYNONYMS["3p"] == "3P"
        assert SYNONYMS["grr"] == "grr"


class TestColumnMetadata:
    """Tests for COLUMN_METADATA."""

    def test_reserve_columns(self):
        """Test reserves column metadata."""
        assert "res_oc" in COLUMN_METADATA
        assert COLUMN_METADATA["res_oc"].units == "MSTB"
        assert COLUMN_METADATA["res_oc"].description == "Reserves oil + condensate"

    def test_resource_columns(self):
        """Test resources column metadata."""
        assert "rec_oc" in COLUMN_METADATA
        assert COLUMN_METADATA["rec_oc"].units == "MSTB"
        assert "rec_an" in COLUMN_METADATA
        assert COLUMN_METADATA["rec_an"].units == "BSCF"

    def test_risked_resource_columns(self):
        """Test risked resources column metadata."""
        assert "rec_oc_risked" in COLUMN_METADATA
        assert COLUMN_METADATA["rec_oc_risked"].units == "MSTB"


class TestResolveConcept:
    """Tests for resolve_concept function."""

    def test_resolve_volume_type(self):
        """Test resolving volume type concepts."""
        result = resolve_concept("cadangan")
        assert result is not None
        assert result["type"] == "volume_type"
        assert "res_oc" in result["columns"]
        assert "res_an" in result["columns"]

    def test_resolve_uncertainty_level(self):
        """Test resolving uncertainty level concepts."""
        result = resolve_concept("1P")
        assert result is not None
        assert result["type"] == "uncertainty_level"
        assert result["db_value"] == "1. Low Value"

    def test_resolve_project_class(self):
        """Test resolving project class concepts."""
        result = resolve_concept("grr")
        assert result is not None
        assert result["type"] == "project_class"
        assert result["db_value"] == "1. Reserves & GRR"

    def test_resolve_unknown_concept(self):
        """Test resolving unknown concept."""
        result = resolve_concept("unknown_term")
        assert result is None

    def test_resolve_with_synonym(self):
        """Test resolving concept using synonym."""
        result = resolve_concept("sumberdaya")
        assert result is not None
        assert result["type"] == "volume_type"


class TestGetUncertaintyFilter:
    """Tests for get_uncertainty_filter function."""

    def test_1p_filter(self):
        """Test 1P uncertainty filter."""
        assert get_uncertainty_filter("1P") == "1. Low Value"
        assert get_uncertainty_filter("1R") == "1. Low Value"
        assert get_uncertainty_filter("1C") == "1. Low Value"
        assert get_uncertainty_filter("1U") == "1. Low Value"

    def test_2p_filter(self):
        """Test 2P uncertainty filter."""
        assert get_uncertainty_filter("2P") == "2. Middle Value"
        assert get_uncertainty_filter("2R") == "2. Middle Value"
        assert get_uncertainty_filter("2C") == "2. Middle Value"

    def test_3p_filter(self):
        """Test 3P uncertainty filter."""
        assert get_uncertainty_filter("3P") == "3. High Value"
        assert get_uncertainty_filter("3R") == "3. High Value"
        assert get_uncertainty_filter("3U") == "3. High Value"

    def test_proven_filter(self):
        """Test proven filter returns 1. Low Value."""
        assert get_uncertainty_filter("proven") == "1. Low Value"
        assert get_uncertainty_filter("PROVEN") == "1. Low Value"

    def test_probable_filter(self):
        """Test probable filter returns default (middle)."""
        assert get_uncertainty_filter("probable") == "2. Middle Value"

    def test_possible_filter(self):
        """Test possible filter returns default (middle)."""
        assert get_uncertainty_filter("possible") == "2. Middle Value"

    def test_default_filter(self):
        """Test default filter for unknown."""
        assert get_uncertainty_filter("unknown") == "2. Middle Value"


class TestUncertaintySpec:
    """Tests for UncertaintySpec dataclass and UNCERTAINTY_MAP."""

    def test_uncertainty_spec_exists(self):
        """Test that UncertaintySpec is defined."""
        assert UncertaintySpec is not None

    def test_uncertainty_map_has_direct_values(self):
        """Test UNCERTAINTY_MAP has direct value types."""
        assert "1P" in UNCERTAINTY_MAP
        assert "2P" in UNCERTAINTY_MAP
        assert "3P" in UNCERTAINTY_MAP
        assert UNCERTAINTY_MAP["1P"].type == "direct"
        assert UNCERTAINTY_MAP["2P"].type == "direct"

    def test_uncertainty_map_has_calculated_values(self):
        """Test UNCERTAINTY_MAP has calculated types."""
        assert "PROBABLE" in UNCERTAINTY_MAP
        assert "POSSIBLE" in UNCERTAINTY_MAP
        assert UNCERTAINTY_MAP["PROBABLE"].type == "calculated"
        assert UNCERTAINTY_MAP["POSSIBLE"].type == "calculated"

    def test_probable_is_reserves_only(self):
        """Test that probable/possible are reserves_only."""
        assert UNCERTAINTY_MAP["PROBABLE"].reserves_only is True
        assert UNCERTAINTY_MAP["POSSIBLE"].reserves_only is True
        assert UNCERTAINTY_MAP["PROVEN"].reserves_only is True
        assert UNCERTAINTY_MAP["1P"].reserves_only is False
        assert UNCERTAINTY_MAP["2P"].reserves_only is False

    def test_probable_calculation(self):
        """Test probable calculation expression."""
        assert UNCERTAINTY_MAP["PROBABLE"].calculation == "2P - 1P"
        assert UNCERTAINTY_MAP["PROBABLE"].sql_template is not None
        assert "2. Middle Value" in UNCERTAINTY_MAP["PROBABLE"].sql_template
        assert "1. Low Value" in UNCERTAINTY_MAP["PROBABLE"].sql_template

    def test_possible_calculation(self):
        """Test possible calculation expression."""
        assert UNCERTAINTY_MAP["POSSIBLE"].calculation == "3P - 2P"
        assert UNCERTAINTY_MAP["POSSIBLE"].sql_template is not None
        assert "3. High Value" in UNCERTAINTY_MAP["POSSIBLE"].sql_template
        assert "2. Middle Value" in UNCERTAINTY_MAP["POSSIBLE"].sql_template

    def test_indonesian_synonyms_in_map(self):
        """Test Indonesian synonyms are in the map."""
        assert "TERBUKTI" in UNCERTAINTY_MAP
        assert "MUNGKIN" in UNCERTAINTY_MAP
        assert "HARAPAN" in UNCERTAINTY_MAP
        assert UNCERTAINTY_MAP["TERBUKTI"].db_value == "1. Low Value"
        assert UNCERTAINTY_MAP["MUNGKIN"].reserves_only is True


class TestGetUncertaintySpec:
    """Tests for get_uncertainty_spec function."""

    def test_direct_value_spec(self):
        """Test getting spec for direct values."""
        spec = get_uncertainty_spec("2P")
        assert spec is not None
        assert spec.type == "direct"
        assert spec.db_value == "2. Middle Value"
        assert spec.is_cumulative is True

    def test_calculated_value_spec(self):
        """Test getting spec for calculated values."""
        spec = get_uncertainty_spec("probable")
        assert spec is not None
        assert spec.type == "calculated"
        assert spec.calculation == "2P - 1P"
        assert spec.reserves_only is True

    def test_reserves_only_validation_passes(self):
        """Test reserves_only validation passes for reserves."""
        spec = get_uncertainty_spec("probable", volume_type="reserves")
        assert spec is not None
        assert spec.reserves_only is True

        spec = get_uncertainty_spec("probable", volume_type="cadangan")
        assert spec is not None

        spec = get_uncertainty_spec("probable", volume_type="res_oc")
        assert spec is not None

    def test_reserves_only_validation_fails(self):
        """Test reserves_only validation fails for non-reserves."""
        with pytest.raises(ValueError) as exc:
            get_uncertainty_spec("probable", volume_type="contingent")
        assert "only applies to Reserves" in str(exc.value)

        with pytest.raises(ValueError) as exc:
            get_uncertainty_spec("possible", volume_type="prospective")
        assert "only applies to Reserves" in str(exc.value)

    def test_direct_values_work_with_any_volume(self):
        """Test that direct values (1P/2P/3P) work with any volume type."""
        spec = get_uncertainty_spec("2P", volume_type="reserves")
        assert spec is not None
        assert spec.db_value == "2. Middle Value"

        spec = get_uncertainty_spec("2C", volume_type="contingent")
        assert spec is not None
        assert spec.db_value == "2. Middle Value"

        spec = get_uncertainty_spec("2U", volume_type="prospective")
        assert spec is not None
        assert spec.db_value == "2. Middle Value"

    def test_unknown_uncertainty(self):
        """Test unknown uncertainty returns None."""
        spec = get_uncertainty_spec("unknown")
        assert spec is None


class TestBuildUncertaintySql:
    """Tests for build_uncertainty_sql function."""

    def test_direct_value_sql(self):
        """Test SQL for direct values."""
        sql = build_uncertainty_sql("2P", "res_oc")
        assert sql == "pr.res_oc"

    def test_calculated_probable_sql(self):
        """Test SQL for probable calculation."""
        sql = build_uncertainty_sql("probable", "res_oc")
        assert "SUM(CASE WHEN" in sql
        assert "2. Middle Value" in sql
        assert "1. Low Value" in sql
        assert "res_oc" in sql

    def test_calculated_possible_sql(self):
        """Test SQL for possible calculation."""
        sql = build_uncertainty_sql("possible", "res_an")
        assert "SUM(CASE WHEN" in sql
        assert "3. High Value" in sql
        assert "2. Middle Value" in sql
        assert "res_an" in sql

    def test_table_alias_parameter(self):
        """Test custom table alias."""
        sql = build_uncertainty_sql("2P", "res_oc", table_alias="project")
        assert sql == "project.res_oc"

    def test_sql_with_indonesian_terms(self):
        """Test SQL with Indonesian terms."""
        sql_mungkin = build_uncertainty_sql("mungkin", "res_oc")
        sql_probable = build_uncertainty_sql("probable", "res_oc")
        assert sql_mungkin == sql_probable


class TestGetProjectClassFilter:
    """Tests for get_project_class_filter function."""

    def test_grr_filter(self):
        """Test GRR project class filter."""
        assert get_project_class_filter("grr") == "1. Reserves & GRR"
        assert get_project_class_filter("GRR") == "1. Reserves & GRR"

    def test_contingent_filter(self):
        """Test Contingent project class filter."""
        assert get_project_class_filter("contingent") == "2. Contingent Resources"
        assert get_project_class_filter("Contingent Resources") == "2. Contingent Resources"

    def test_prospective_filter(self):
        """Test Prospective project class filter."""
        assert get_project_class_filter("prospective") == "3. Prospective Resources"

    def test_unknown_filter(self):
        """Test unknown project class."""
        assert get_project_class_filter("unknown") is None


class TestGetVolumeColumns:
    """Tests for get_volume_columns function."""

    def test_cadangan_columns(self):
        """Test cadangan volume columns."""
        oc_col, gas_col = get_volume_columns("cadangan")
        assert oc_col == "res_oc"
        assert gas_col == "res_an"

    def test_sumber_daya_columns(self):
        """Test sumber_daya volume columns."""
        oc_col, gas_col = get_volume_columns("sumber_daya")
        assert oc_col == "rec_oc"
        assert gas_col == "rec_an"

    def test_risked_columns(self):
        """Test risked volume columns."""
        oc_col, gas_col = get_volume_columns("sumber_daya", is_risked=True)
        assert oc_col == "rec_oc_risked"
        assert gas_col == "rec_an_risked"

    def test_grr_columns(self):
        """Test GRR volume columns."""
        oc_col, gas_col = get_volume_columns("grr")
        assert oc_col == "rec_oc"
        assert gas_col == "rec_an"


class TestGetColumnsForSubstance:
    """Tests for get_columns_for_substance function."""

    def test_oil_substance(self):
        """Test oil substance columns."""
        cols = get_columns_for_substance("oil")
        assert "_oil" in cols
        assert "_oc" in cols

    def test_gas_substance(self):
        """Test gas substance columns."""
        cols = get_columns_for_substance("gas")
        assert "_an" in cols


class TestGetColumnGroup:
    """Tests for get_column_group function."""

    def test_reserve_column_group(self):
        """Test reserves column group."""
        assert get_column_group("res_oc") == "reserves"
        assert get_column_group("res_an") == "reserves"

    def test_resource_column_group(self):
        """Test resources column group."""
        assert get_column_group("rec_oc") == "resources"
        assert get_column_group("rec_an") == "resources"

    def test_unknown_column_group(self):
        """Test unknown column group."""
        assert get_column_group("unknown_column") is None


class TestFormatResponseValue:
    """Tests for format_response_value function."""

    def test_format_mstb(self):
        """Test MSTB formatting."""
        result = format_response_value(1000.5, "MSTB")
        assert "1,000.50" in result
        assert "MSTB" in result

    def test_format_bscf(self):
        """Test BSCF formatting."""
        result = format_response_value(500.123, "BSCF")
        assert "500.12" in result
        assert "BSCF" in result

    def test_format_none(self):
        """Test None value formatting."""
        result = format_response_value(None, "MSTB")
        assert result == "N/A"


class TestBuildSqlPattern:
    """Tests for build_sql_pattern function."""

    def test_build_cadangan_pattern(self):
        """Test building cadangan SQL pattern."""
        patterns = build_sql_pattern("cadangan")
        assert "base" in patterns
        assert "res_oc" in patterns["base"]
        assert "res_an" in patterns["base"]

    def test_build_pattern_with_uncertainty(self):
        """Test building pattern with uncertainty filter."""
        patterns = build_sql_pattern("cadangan", uncertainty="2P")
        assert "2. Middle Value" in patterns["base"]

    def test_build_pattern_with_project_class(self):
        """Test building pattern with project class filter."""
        patterns = build_sql_pattern("sumber_daya", project_class="grr")
        assert "1. Reserves & GRR" in patterns["base"]


class TestTableHierarchy:
    """Tests for table/view hierarchy constants."""

    def test_table_hierarchy_exists(self):
        """Test TABLE_HIERARCHY is defined."""
        assert TABLE_HIERARCHY is not None
        assert "field" in TABLE_HIERARCHY
        assert "work_area" in TABLE_HIERARCHY
        assert "national" in TABLE_HIERARCHY

    def test_table_hierarchy_mappings(self):
        """Test TABLE_HIERARCHY mappings."""
        assert TABLE_HIERARCHY["project"] == "project_resources"
        assert TABLE_HIERARCHY["field"] == "field_resources"
        assert TABLE_HIERARCHY["work_area"] == "wa_resources"
        assert TABLE_HIERARCHY["wa"] == "wa_resources"
        assert TABLE_HIERARCHY["national"] == "nkri_resources"
        assert TABLE_HIERARCHY["nkri"] == "nkri_resources"

    def test_aggregation_levels_exist(self):
        """Test AGGREGATION_LEVELS is defined."""
        assert AGGREGATION_LEVELS is not None
        assert len(AGGREGATION_LEVELS) == 8


class TestGetTableForQuery:
    """Tests for get_table_for_query function."""

    def test_get_table_for_field(self):
        """Test getting table for field queries."""
        assert get_table_for_query("field") == "field_resources"

    def test_get_table_for_work_area(self):
        """Test getting table for work area queries."""
        assert get_table_for_query("work_area") == "wa_resources"
        assert get_table_for_query("wa") == "wa_resources"

    def test_get_table_for_national(self):
        """Test getting table for national queries."""
        assert get_table_for_query("national") == "nkri_resources"
        assert get_table_for_query("nkri") == "nkri_resources"

    def test_get_table_for_project(self):
        """Test getting table for project queries."""
        assert get_table_for_query("project") == "project_resources"

    def test_get_table_require_detail(self):
        """Test getting project_resources when detail required."""
        assert get_table_for_query("field", require_detail=True) == "project_resources"
        assert get_table_for_query("work_area", require_detail=True) == "project_resources"
        assert get_table_for_query("national", require_detail=True) == "project_resources"

    def test_get_table_unknown_entity(self):
        """Test getting table for unknown entity."""
        assert get_table_for_query("unknown") == "project_resources"
        assert get_table_for_query(None) == "project_resources"


class TestCanUseViewForCalculation:
    """Tests for can_use_view_for_calculation function."""

    def test_direct_values_work_with_views(self):
        """Test that direct values work with views."""
        assert can_use_view_for_calculation("2P", "field_resources") is True
        assert can_use_view_for_calculation("1P", "wa_resources") is True
        assert can_use_view_for_calculation("3P", "nkri_resources") is True

    def test_calculated_values_work_with_views(self):
        """Test that calculated values work with views."""
        assert can_use_view_for_calculation("probable", "field_resources") is True
        assert can_use_view_for_calculation("possible", "wa_resources") is True

    def test_invalid_table_returns_false(self):
        """Test that invalid tables return False."""
        assert can_use_view_for_calculation("2P", "invalid_table") is False


class TestGetEntityFilterColumn:
    """Tests for get_entity_filter_column function."""

    def test_field_filter_column(self):
        """Test getting filter column for field."""
        assert get_entity_filter_column("field", "field_resources") == "field_name"
        assert get_entity_filter_column("field", "project_resources") == "field_name"

    def test_work_area_filter_column(self):
        """Test getting filter column for work area."""
        assert get_entity_filter_column("work_area", "wa_resources") == "wk_name"
        assert get_entity_filter_column("wa", "wa_resources") == "wk_name"

    def test_national_filter_column(self):
        """Test that national has no filter column."""
        assert get_entity_filter_column("national", "nkri_resources") is None
        assert get_entity_filter_column("nkri", "nkri_resources") is None


class TestBuildAggregateQuery:
    """Tests for build_aggregate_query function."""

    def test_build_field_query(self):
        """Test building field-level query."""
        result = build_aggregate_query("field", "Duri", "cadangan", "2P")
        assert result["table"] == "field_resources"
        assert "Duri" in result["sql"]
        assert "res_oc" in result["sql"]

    def test_build_wa_query(self):
        """Test building work area query."""
        result = build_aggregate_query("work_area", "Rokan", "cadangan", "2P")
        assert result["table"] == "wa_resources"
        assert "Rokan" in result["sql"]

    def test_build_national_query(self):
        """Test building national-level query."""
        result = build_aggregate_query("national", None, "cadangan", "2P")
        assert result["table"] == "nkri_resources"

    def test_build_probable_query(self):
        """Test building calculated probable query."""
        result = build_aggregate_query("field", "Duri", "cadangan", "probable")
        assert result["table"] == "field_resources"
        assert "CASE WHEN" in result["sql"]
        assert "2. Middle Value" in result["sql"]
        assert "1. Low Value" in result["sql"]

    def test_build_query_with_project_class(self):
        """Test building query with GRR project class."""
        result = build_aggregate_query("field", "Duri", "sumber_daya", "2P", project_class="grr")
        assert result["table"] == "field_resources"
        assert "1. Reserves & GRR" in result["sql"]

    def test_build_query_use_view_false(self):
        """Test building query with use_view=False returns project_resources."""
        result = build_aggregate_query("field", "Duri", "cadangan", "2P", use_view=False)
        assert result["table"] == "project_resources"


class TestGetAggregationTableInfo:
    """Tests for get_aggregation_table_info function."""

    def test_get_table_info_exists(self):
        """Test table info is returned."""
        info = get_aggregation_table_info()
        assert info is not None
        assert "project_resources" in info
        assert "field_resources" in info
        assert "wa_resources" in info
        assert "nkri_resources" in info

    def test_table_info_levels(self):
        """Test table info has correct levels."""
        info = get_aggregation_table_info()
        assert info["project_resources"]["level"] == 1
        assert info["field_resources"]["level"] == 2
        assert info["wa_resources"]["level"] == 3
        assert info["nkri_resources"]["level"] == 4

    def test_table_info_entity_types(self):
        """Test table info has correct entity types."""
        info = get_aggregation_table_info()
        assert info["project_resources"]["entity_type"] == "project"
        assert info["field_resources"]["entity_type"] == "field"
        assert info["wa_resources"]["entity_type"] == "work_area"
        assert info["nkri_resources"]["entity_type"] == "national"


class TestGetRecommendedTable:
    """Tests for get_recommended_table function."""

    def test_recommended_table_for_field(self):
        """Test recommended table for field queries."""
        assert get_recommended_table("field") == "field_resources"

    def test_recommended_table_for_work_area(self):
        """Test recommended table for work area queries."""
        assert get_recommended_table("work_area") == "wa_resources"
        assert get_recommended_table("wa") == "wa_resources"
        assert get_recommended_table("wilayah_kerja") == "wa_resources"

    def test_recommended_table_for_national(self):
        """Test recommended table for national queries."""
        assert get_recommended_table("national") == "nkri_resources"
        assert get_recommended_table("nkri") == "nkri_resources"

    def test_recommended_table_with_detail_needed(self):
        """Test recommended table when detail is needed."""
        assert get_recommended_table("field", query_needs_detail=True) == "project_resources"
        assert get_recommended_table("work_area", query_needs_detail=True) == "project_resources"

    def test_recommended_table_unknown(self):
        """Test recommended table for unknown entity."""
        assert get_recommended_table("unknown") == "project_resources"
        assert get_recommended_table(None) == "project_resources"
