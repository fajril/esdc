"""Tests for report year fallback functionality."""

import pytest

from esdc.chat.domain_knowledge import (
    build_aggregate_query,
    build_report_year_filter,
    detect_report_year_from_query,
    get_available_report_year,
)


class TestDetectReportYearFromQuery:
    """Tests for detect_report_year_from_query function."""

    def test_detect_year_with_tahun(self):
        """Should detect year with 'tahun' prefix."""
        assert detect_report_year_from_query("berapa cadangan tahun 2024?") == 2024
        assert detect_report_year_from_query("data tahun 2023") == 2023
        assert detect_report_year_from_query("laporan tahun 2025") == 2025

    def test_detect_year_with_year(self):
        """Should detect year with 'year' prefix."""
        assert detect_report_year_from_query("reserves year 2024?") == 2024
        assert detect_report_year_from_query("year 2023 data") == 2023

    def test_detect_standalone_year(self):
        """Should detect standalone year in sentence."""
        assert detect_report_year_from_query("cadangan 2024 di lapangan duri") == 2024
        assert detect_report_year_from_query("data 2023 untuk lapangan X") == 2023

    def test_detect_two_digit_year(self):
        """Should convert 2-digit year to 4-digit."""
        assert detect_report_year_from_query("cadangan tahun 24?") == 2024
        assert detect_report_year_from_query("data tahun 23") == 2023

    def test_no_year_detected(self):
        """Should return None when no year mentioned."""
        assert detect_report_year_from_query("berapa cadangan lapangan duri?") is None
        assert detect_report_year_from_query("data potensi wilayah kerja rokan") is None

    def test_invalid_year_ignored(self):
        """Should ignore years outside valid range."""
        assert detect_report_year_from_query("data tahun 1999") is None
        assert detect_report_year_from_query("cadangan tahun 2101") is None


class TestGetAvailableReportYear:
    """Tests for get_available_report_year function."""

    def test_default_to_current_year(self):
        """Should default to current year if not specified."""
        from datetime import datetime

        result = get_available_report_year("field_resources")
        assert result["requested_year"] == datetime.now().year

    def test_validate_valid_table(self):
        """Should accept valid table names."""
        valid_tables = [
            "field_resources",
            "wa_resources",
            "nkri_resources",
            "project_resources",
            "field_timeseries",
            "wa_timeseries",
            "nkri_timeseries",
            "project_timeseries",
        ]
        for table in valid_tables:
            result = get_available_report_year(table, 2024)
            assert result["requested_year"] == 2024

    def test_reject_invalid_table(self):
        """Should reject invalid table names."""
        with pytest.raises(ValueError, match="Invalid table"):
            get_available_report_year("invalid_table", 2024)

    def test_years_checked_generation(self):
        """Should generate list of years to check."""
        result = get_available_report_year(
            "field_resources", 2024, max_fallback_years=5
        )
        expected_years = [2024, 2023, 2022, 2021, 2020, 2019]
        assert result["years_checked"] == expected_years

    def test_with_entity_filter(self):
        """Should accept entity filter."""
        result = get_available_report_year(
            "field_resources", 2024, entity_filter="field_name LIKE '%Duri%'"
        )
        assert result["requested_year"] == 2024
        assert result["has_data"] is False  # Will be set by caller

    def test_metadata_structure(self):
        """Should return proper metadata structure."""
        result = get_available_report_year("project_resources", 2025)
        assert "requested_year" in result
        assert "actual_year" in result
        assert "years_checked" in result
        assert "has_data" in result
        assert "message" in result
        assert "fallback_needed" in result


class TestBuildReportYearFilter:
    """Tests for build_report_year_filter function."""

    def test_subquery_approach_default(self):
        """Should use subquery by default."""
        sql_clause, metadata = build_report_year_filter("field_resources", 2024)
        assert "SELECT MAX(report_year)" in sql_clause
        assert "report_year <= 2024" in sql_clause
        assert "FROM field_resources" in sql_clause

    def test_with_entity_filter(self):
        """Should include entity filter in subquery."""
        sql_clause, metadata = build_report_year_filter(
            "field_resources", 2024, entity_filter="field_name LIKE '%Duri%'"
        )
        assert "field_name LIKE '%Duri%'" in sql_clause
        assert "AND field_name LIKE '%Duri%'" in sql_clause

    def test_simple_filter_approach(self):
        """Should support simple filter approach."""
        sql_clause, metadata = build_report_year_filter(
            "project_resources", 2024, use_subquery=False
        )
        assert "report_year <= 2024" in sql_clause
        assert "SELECT MAX" not in sql_clause

    def test_returns_metadata(self):
        """Should return metadata along with SQL."""
        sql_clause, metadata = build_report_year_filter("field_resources", 2024)
        assert metadata["requested_year"] == 2024
        assert "years_checked" in metadata

    def test_handles_none_year(self):
        """Should use current year if None provided."""
        from datetime import datetime

        sql_clause, metadata = build_report_year_filter("field_resources", None)
        assert metadata["requested_year"] == datetime.now().year


class TestBuildAggregateQuery:
    """Tests for build_aggregate_query with report_year parameter."""

    def test_default_report_year_subquery(self):
        """Should use subquery for report_year by default."""
        result = build_aggregate_query("field", "Duri", "cadangan", "2P")
        assert "SELECT MAX(report_year)" in result["sql"]
        assert "FROM field_resources" in result["sql"]
        assert "report_year_info" in result

    def test_explicit_report_year_2023(self):
        """Should use specified report year."""
        result = build_aggregate_query(
            "field", "Duri", "cadangan", "2P", report_year=2023
        )
        assert "report_year <= 2023" in result["sql"]
        assert result["report_year_info"]["requested_year"] == 2023

    def test_report_year_with_entity_filter(self):
        """Should include entity filter in subquery."""
        result = build_aggregate_query(
            "field", "Duri", "cadangan", "2P", report_year=2024
        )
        assert "LIKE '%Duri%'" in result["sql"]
        assert "report_year <= 2024" in result["sql"]

    def test_report_year_metadata_returned(self):
        """Should return report year metadata."""
        result = build_aggregate_query(
            "field", "Duri", "cadangan", "2P", report_year=2024
        )
        metadata = result["report_year_info"]
        assert metadata["requested_year"] == 2024
        assert isinstance(metadata["years_checked"], list)

    def test_national_level_query(self):
        """Should work for national level queries."""
        result = build_aggregate_query(
            "national", None, "cadangan", "2P", report_year=2024
        )
        assert "nkri_resources" in result["sql"]
        assert "report_year <= 2024" in result["sql"]

    def test_project_level_query(self):
        """Should work for project level queries."""
        result = build_aggregate_query(
            "project", "X", "sumber_daya", "2P", use_view=False, report_year=2023
        )
        assert "project_resources" in result["sql"]
        assert "report_year <= 2023" in result["sql"]


class TestReportYearIntegration:
    """Integration tests combining year detection with query building."""

    def test_full_workflow_with_year_detection(self):
        """Test complete workflow from query detection to SQL building."""
        query = "berapa cadangan lapangan duri tahun 2024?"

        # Step 1: Detect year from query
        year = detect_report_year_from_query(query)
        assert year == 2024

        # Step 2: Build query with year
        result = build_aggregate_query(
            "field", "Duri", "cadangan", "2P", report_year=year
        )

        # Step 3: Verify SQL
        assert "report_year <= 2024" in result["sql"]
        assert "LIKE '%Duri%'" in result["sql"]
        assert result["report_year_info"]["requested_year"] == 2024

    def test_full_workflow_without_year(self):
        """Test workflow when no year is specified."""
        query = "berapa cadangan lapangan duri?"

        # Step 1: No year detected
        year = detect_report_year_from_query(query)
        assert year is None

        # Step 2: Build query with default (current year)
        result = build_aggregate_query("field", "Duri", "cadangan", "2P")

        # Step 3: Verify SQL uses MAX(report_year)
        assert "SELECT MAX(report_year)" in result["sql"]

    def test_year_fallback_message(self):
        """Test that fallback messages are generated."""
        _, metadata = build_report_year_filter("field_resources", 2025)

        # Metadata tracks requested year
        assert metadata["requested_year"] == 2025

        # Years checked includes fallback range
        assert 2025 in metadata["years_checked"]
        assert 2024 in metadata["years_checked"]

    def test_two_digit_year_workflow(self):
        """Test handling of 2-digit year input."""
        query = "cadangan tahun 24 lapangan duri"

        year = detect_report_year_from_query(query)
        assert year == 2024

        result = build_aggregate_query(
            "field", "Duri", "cadangan", "2P", report_year=year
        )
        assert "report_year <= 2024" in result["sql"]
