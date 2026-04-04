"""Tests for domain knowledge tools."""

import json
import pytest
from esdc.chat.tools import get_recommended_table, resolve_uncertainty_level


class TestGetRecommendedTable:
    """Tests for get_recommended_table tool."""

    def test_field_entity_returns_field_resources(self):
        """Test field entity returns field_resources."""
        result = get_recommended_table.invoke({"entity_type": "field"})
        data = json.loads(result)
        assert data["table"] == "field_resources"
        assert "field" in data["explanation"].lower()

    def test_work_area_entity_returns_wa_resources(self):
        """Test work_area entity returns wa_resources."""
        result = get_recommended_table.invoke({"entity_type": "work_area"})
        data = json.loads(result)
        assert data["table"] == "wa_resources"
        assert "work area" in data["explanation"].lower()

    def test_national_entity_returns_nkri_resources(self):
        """Test national entity returns nkri_resources."""
        result = get_recommended_table.invoke({"entity_type": "national"})
        data = json.loads(result)
        assert data["table"] == "nkri_resources"
        assert "national" in data["explanation"].lower()

    def test_project_entity_returns_project_resources(self):
        """Test project entity returns project_resources."""
        result = get_recommended_table.invoke({"entity_type": "project"})
        data = json.loads(result)
        assert data["table"] == "project_resources"
        assert "project" in data["explanation"].lower()

    def test_unknown_entity_defaults_to_project_resources(self):
        """Test unknown entity defaults to project_resources."""
        result = get_recommended_table.invoke({"entity_type": "unknown_entity"})
        data = json.loads(result)
        assert data["table"] == "project_resources"
        assert (
            "aggregat" in data["explanation"].lower()
            or "default" in data["explanation"].lower()
        )

    def test_field_with_project_detail_returns_project_resources(self):
        """Test field with project_detail=True returns project_resources."""
        result = get_recommended_table.invoke(
            {"entity_type": "field", "needs_project_detail": True}
        )
        data = json.loads(result)
        assert data["table"] == "project_resources"
        assert "breakdown" in data["explanation"].lower()

    def test_work_area_with_project_detail_returns_project_resources(self):
        """Test work_area with project_detail=True returns project_resources."""
        result = get_recommended_table.invoke(
            {"entity_type": "work_area", "needs_project_detail": True}
        )
        data = json.loads(result)
        assert data["table"] == "project_resources"

    def test_indonesian_field_term(self):
        """Test Indonesian term 'lapangan' returns field_resources."""
        result = get_recommended_table.invoke({"entity_type": "lapangan"})
        data = json.loads(result)
        assert data["table"] == "field_resources"

    def test_indonesian_work_area_term(self):
        """Test Indonesian term 'wilayah kerja' returns wa_resources."""
        result = get_recommended_table.invoke({"entity_type": "wilayah kerja"})
        data = json.loads(result)
        assert data["table"] == "wa_resources"


class TestResolveUncertaintyLevel:
    """Tests for resolve_uncertainty_level tool."""

    def test_1p_returns_direct_value(self):
        """Test 1P returns direct database value."""
        result = resolve_uncertainty_level.invoke({"level": "1P"})
        data = json.loads(result)
        assert data["db_value"] == "1. Low Value"
        assert data["type"] == "direct"

    def test_2p_returns_direct_value(self):
        """Test 2P returns direct database value."""
        result = resolve_uncertainty_level.invoke({"level": "2P"})
        data = json.loads(result)
        assert data["db_value"] == "2. Middle Value"
        assert data["type"] == "direct"

    def test_3p_returns_direct_value(self):
        """Test 3P returns direct database value."""
        result = resolve_uncertainty_level.invoke({"level": "3P"})
        data = json.loads(result)
        assert data["db_value"] == "3. High Value"
        assert data["type"] == "direct"

    def test_proven_equals_1p(self):
        """Test 'proven' returns same as 1P."""
        result = resolve_uncertainty_level.invoke({"level": "proven"})
        data = json.loads(result)
        assert data["db_value"] == "1. Low Value"
        assert data["type"] == "direct"

    def test_probable_returns_calculated_reserves(self):
        """Test 'probable' returns calculated for reserves."""
        result = resolve_uncertainty_level.invoke(
            {"level": "probable", "volume_type": "reserves"}
        )
        data = json.loads(result)
        assert "sql_template" in data
        assert (
            "CASE WHEN" in data["sql_template"]
            or "Middle Value" in data["sql_template"]
        )

    def test_default_volume_type_is_reserves(self):
        """Test default volume_type is 'reserves'."""
        result = resolve_uncertainty_level.invoke({"level": "probable"})
        data = json.loads(result)
        assert data["type"] == "calculated"

    def test_direct_value_synonyms_resolve(self):
        """Test direct value synonyms resolve to database values."""
        # Test low_value synonym
        result = resolve_uncertainty_level.invoke({"level": "low_value"})
        data = json.loads(result)
        assert data["db_value"] == "1. Low Value"
        assert data["type"] == "direct"
        assert data["filter_value"] == "1. Low Value"

        # Test middle_value synonym
        result = resolve_uncertainty_level.invoke({"level": "middle_value"})
        data = json.loads(result)
        assert data["db_value"] == "2. Middle Value"
        assert data["type"] == "direct"

        # Test high_value synonym
        result = resolve_uncertainty_level.invoke({"level": "high_value"})
        data = json.loads(result)
        assert data["db_value"] == "3. High Value"
        assert data["type"] == "direct"
        assert data["calculation"] is None  # Direct values have no calculation

    def test_possible_returns_calculated_reserves(self):
        """Test 'possible' returns calculated for reserves."""
        result = resolve_uncertainty_level.invoke(
            {"level": "possible", "volume_type": "reserves"}
        )
        data = json.loads(result)
        assert data["type"] == "calculated"
        assert data["calculation"] == "3P - 2P"

    def test_probable_warns_for_resources(self):
        """Test 'probable' warns when used with resources."""
        result = resolve_uncertainty_level.invoke(
            {"level": "probable", "volume_type": "resources"}
        )
        data = json.loads(result)
        assert "error" in data or "warning" in data
        if "error" in data:
            assert "reserves" in data["error"].lower()
        else:
            assert "reserves" in data["warning"].lower()

    def test_possible_warns_for_resources(self):
        """Test 'possible' warns when used with resources."""
        result = resolve_uncertainty_level.invoke(
            {"level": "possible", "volume_type": "resources"}
        )
        data = json.loads(result)
        assert "error" in data or "warning" in data
        if "error" in data:
            assert "reserves" in data["error"].lower()
        else:
            assert "reserves" in data["warning"].lower()

    def test_indonesian_terbukti_equals_proven(self):
        """Test Indonesian 'terbukti' equals 'proven'."""
        result = resolve_uncertainty_level.invoke({"level": "terbukti"})
        data = json.loads(result)
        assert data["db_value"] == "1. Low Value"
        assert data["type"] == "direct"

    def test_indonesian_mungkin_equals_probable(self):
        """Test Indonesian 'mungkin' equals 'probable'."""
        result = resolve_uncertainty_level.invoke(
            {"level": "mungkin", "volume_type": "reserves"}
        )
        data = json.loads(result)
        assert data["type"] == "calculated"
        assert data["calculation"] == "2P - 1P"

    def test_indonesian_harapan_equals_possible(self):
        """Test Indonesian 'harapan' equals 'possible'."""
        result = resolve_uncertainty_level.invoke(
            {"level": "harapan", "volume_type": "reserves"}
        )
        data = json.loads(result)
        assert data["type"] == "calculated"
        assert data["calculation"] == "3P - 2P"

    def test_1c_returns_direct_value(self):
        """Test 1C returns direct value for contingent resources."""
        result = resolve_uncertainty_level.invoke(
            {"level": "1C", "volume_type": "resources"}
        )
        data = json.loads(result)
        assert data["db_value"] == "1. Low Value"
        assert data["type"] == "direct"

    def test_2c_returns_direct_value(self):
        """Test 2C returns direct value for contingent resources."""
        result = resolve_uncertainty_level.invoke(
            {"level": "2C", "volume_type": "resources"}
        )
        data = json.loads(result)
        assert data["db_value"] == "2. Middle Value"

    def test_3c_returns_direct_value(self):
        """Test 3C returns direct value for contingent resources."""
        result = resolve_uncertainty_level.invoke(
            {"level": "3C", "volume_type": "resources"}
        )
        data = json.loads(result)
        assert data["db_value"] == "3. High Value"

    def test_unknown_level_returns_suggestion(self):
        """Test unknown level returns helpful suggestion."""
        result = resolve_uncertainty_level.invoke({"level": "XYZ"})
        data = json.loads(result)
        assert "suggestion" in data or "error" in data

    def test_filter_column_present(self):
        """Test filter_column is present for direct values."""
        result = resolve_uncertainty_level.invoke({"level": "2P"})
        data = json.loads(result)
        assert "filter_column" in data
        assert data["filter_column"] == "uncert_level"

    def test_sql_template_for_calculated_values(self):
        """Test SQL template is provided for calculated values."""
        result = resolve_uncertainty_level.invoke(
            {"level": "probable", "volume_type": "reserves"}
        )
        data = json.loads(result)
        assert "sql_template" in data
        assert (
            "CASE WHEN" in data["sql_template"]
            or "Middle Value" in data["sql_template"]
        )

    def test_default_volume_type_is_reserves(self):
        """Test default volume_type is 'reserves'."""
        result = resolve_uncertainty_level.invoke({"level": "probable"})
        data = json.loads(result)
        assert data["type"] == "calculated"
        assert data["calculation"] == "2P - 1P"
        assert "sql_template" in data

    def test_possible_returns_calculated_reserves(self):
        """Test 'possible' returns calculated for reserves."""
        result = resolve_uncertainty_level.invoke(
            {"level": "possible", "volume_type": "reserves"}
        )
        data = json.loads(result)
        assert data["type"] == "calculated"
        assert data["calculation"] == "3P - 2P"

    def test_probable_warns_for_resources(self):
        """Test 'probable' warns when used with resources."""
        result = resolve_uncertainty_level.invoke(
            {"level": "probable", "volume_type": "resources"}
        )
        data = json.loads(result)
        assert "error" in data or "warning" in data
        if "error" in data:
            assert "reserves" in data["error"].lower()
        else:
            assert "reserves" in data["warning"].lower()

    def test_possible_warns_for_resources(self):
        """Test 'possible' warns when used with resources."""
        result = resolve_uncertainty_level.invoke(
            {"level": "possible", "volume_type": "resources"}
        )
        data = json.loads(result)
        assert "error" in data or "warning" in data
        if "error" in data:
            assert "reserves" in data["error"].lower()
        else:
            assert "reserves" in data["warning"].lower()

    def test_indonesian_terbukti_equals_proven(self):
        """Test Indonesian 'terbukti' equals 'proven'."""
        result = resolve_uncertainty_level.invoke({"level": "terbukti"})
        data = json.loads(result)
        assert data["db_value"] == "1. Low Value"
        assert data["type"] == "direct"

    def test_indonesian_mungkin_equals_probable(self):
        """Test Indonesian 'mungkin' equals 'probable'."""
        result = resolve_uncertainty_level.invoke(
            {"level": "mungkin", "volume_type": "reserves"}
        )
        data = json.loads(result)
        assert data["type"] == "calculated"
        assert data["calculation"] == "2P - 1P"

    def test_indonesian_harapan_equals_possible(self):
        """Test Indonesian 'harapan' equals 'possible'."""
        result = resolve_uncertainty_level.invoke(
            {"level": "harapan", "volume_type": "reserves"}
        )
        data = json.loads(result)
        assert data["type"] == "calculated"
        assert data["calculation"] == "3P - 2P"

    def test_1c_returns_direct_value(self):
        """Test 1C returns direct value for contingent resources."""
        result = resolve_uncertainty_level.invoke(
            {"level": "1C", "volume_type": "resources"}
        )
        data = json.loads(result)
        assert data["db_value"] == "1. Low Value"
        assert data["type"] == "direct"

    def test_2c_returns_direct_value(self):
        """Test 2C returns direct value for contingent resources."""
        result = resolve_uncertainty_level.invoke(
            {"level": "2C", "volume_type": "resources"}
        )
        data = json.loads(result)
        assert data["db_value"] == "2. Middle Value"

    def test_3c_returns_direct_value(self):
        """Test 3C returns direct value for contingent resources."""
        result = resolve_uncertainty_level.invoke(
            {"level": "3C", "volume_type": "resources"}
        )
        data = json.loads(result)
        assert data["db_value"] == "3. High Value"

    def test_unknown_level_returns_suggestion(self):
        """Test unknown level returns helpful suggestion."""
        result = resolve_uncertainty_level.invoke({"level": "XYZ"})
        data = json.loads(result)
        assert "suggestion" in data or "error" in data or "warning" in data

    def test_sql_template_for_direct_values(self):
        """Test filter info is provided for direct values."""
        result = resolve_uncertainty_level.invoke({"level": "2P"})
        data = json.loads(result)
        assert "filter_column" in data
        assert data["filter_column"] == "uncert_level"
        assert "filter_value" in data

    def test_sql_template_for_calculated_values(self):
        """Test SQL template is provided for calculated values."""
        result = resolve_uncertainty_level.invoke(
            {"level": "probable", "volume_type": "reserves"}
        )
        data = json.loads(result)
        assert "sql_template" in data
        assert (
            "CASE WHEN" in data["sql_template"]
            or "Middle Value" in data["sql_template"]
        )

    def test_default_volume_type_is_reserves(self):
        """Test default volume_type is 'reserves'."""
        result = resolve_uncertainty_level.invoke({"level": "probable"})
        data = json.loads(result)
        assert data["type"] == "calculated"
