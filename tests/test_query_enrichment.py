"""Tests for SQL query enrichment functionality."""

import pytest

from esdc.chat.domain_knowledge import (
    CLASSIFICATION_CONTEXT_COLUMNS,
    REQUIRES_CLASSIFICATION_PREFIXES,
    TABLE_REMARKS_COLUMNS,
    enrich_sql_query,
    extract_selected_columns,
    extract_table_from_sql,
    get_classification_context_columns,
    get_remarks_column,
    requires_classification_columns,
    requires_classification_context,
    should_include_remarks,
)
from esdc.chat.domain_knowledge.functions import _build_enriched_sql, EnrichedQuery


class TestExtractTableFromSql:
    """Tests for extract_table_from_sql function."""

    def test_extract_simple_table(self):
        """Test extracting table from simple query."""
        sql = "SELECT * FROM field_resources WHERE field_name LIKE '%Duri%'"
        assert extract_table_from_sql(sql) == "field_resources"

    def test_extract_table_with_alias(self):
        """Test extracting table with alias."""
        sql = "SELECT * FROM field_resources AS fr WHERE fr.field_name = 'Duri'"
        assert extract_table_from_sql(sql) == "field_resources"

    def test_extract_table_with_short_alias(self):
        """Test extracting table with short alias."""
        sql = "SELECT rec_oc FROM wa_resources pr WHERE pr.wk_name LIKE '%Masela%'"
        assert extract_table_from_sql(sql) == "wa_resources"

    def test_no_table_found(self):
        """Test when no FROM clause exists."""
        sql = "SELECT 1 + 1"
        assert extract_table_from_sql(sql) is None


class TestExtractSelectedColumns:
    """Tests for extract_selected_columns function."""

    def test_extract_simple_columns(self):
        """Test extracting simple column names."""
        sql = "SELECT rec_oc, rec_an FROM field_resources"
        columns = extract_selected_columns(sql)
        assert "rec_oc" in columns
        assert "rec_an" in columns

    def test_extract_with_aggregation(self):
        """Test extracting columns from aggregated query."""
        sql = "SELECT SUM(rec_oc) as total FROM field_resources"
        columns = extract_selected_columns(sql)
        assert "rec_oc" in columns

    def test_extract_with_table_prefix(self):
        """Test extracting columns with table prefix."""
        sql = "SELECT pr.rec_oc, pr.rec_an FROM field_resources pr"
        columns = extract_selected_columns(sql)
        assert "rec_oc" in columns
        assert "rec_an" in columns

    def test_extract_star(self):
        """Test that SELECT * returns empty list."""
        sql = "SELECT * FROM field_resources"
        columns = extract_selected_columns(sql)
        assert columns == []


class TestRequiresClassificationContext:
    """Tests for requires_classification_context function."""

    def test_rec_columns_require_classification(self):
        """Test that rec_* columns require classification."""
        assert requires_classification_context(["rec_oc", "res_oc"]) is True
        assert requires_classification_context(["rec_oil"]) is True
        assert requires_classification_context(["rec_an", "rec_ga"]) is True

    def test_res_columns_no_classification(self):
        """Test that res_* columns don't require classification."""
        assert requires_classification_context(["res_oc", "res_an"]) is False
        assert requires_classification_context(["res_oil"]) is False

    def test_other_columns_no_classification(self):
        """Test that other columns don't require classification."""
        assert requires_classification_context(["tpf_oc", "prj_ioip"]) is False
        assert requires_classification_context(["cprd_grs_oc"]) is False


class TestGetRemarksColumn:
    """Tests for get_remarks_column function."""

    def test_field_resources_has_remarks(self):
        """Test field_resources has field_remarks."""
        assert get_remarks_column("field_resources") == "field_remarks"

    def test_project_resources_has_remarks(self):
        """Test project_resources has project_remarks."""
        assert get_remarks_column("project_resources") == "project_remarks"

    def test_wa_resources_has_remarks(self):
        """Test wa_resources has wa_remarks."""
        assert get_remarks_column("wa_resources") == "wa_remarks"

    def test_nkri_resources_no_remarks(self):
        """Test nkri_resources has no remarks."""
        assert get_remarks_column("nkri_resources") is None

    def test_timeseries_tables(self):
        """Test timeseries tables have appropriate remarks."""
        assert get_remarks_column("field_timeseries") == "field_remarks"
        assert get_remarks_column("project_timeseries") == "project_remarks"
        assert get_remarks_column("nkri_timeseries") is None


class TestEnrichSqlQuery:
    """Tests for enrich_sql_query function."""

    def test_enrich_resources_query_adds_classification(self):
        """Test that rec_* queries get classification columns."""
        sql = "SELECT SUM(rec_oc) FROM field_resources WHERE field_name LIKE '%Abadi%'"
        result = enrich_sql_query(sql)

        assert "project_class" in result.enriched_sql
        assert "project_stage" in result.enriched_sql
        assert "field_remarks" in result.enriched_sql

    def test_enrich_resources_query_adds_group_by(self):
        """Test that rec_* queries get GROUP BY clause."""
        sql = "SELECT SUM(rec_oc) FROM field_resources"
        result = enrich_sql_query(sql)

        assert "GROUP BY" in result.enriched_sql.upper()
        assert "pr.project_class" in result.enriched_sql
        assert "pr.project_stage" in result.enriched_sql

    def test_enrich_reserves_query_no_classification(self):
        """Test that res_* queries don't get classification columns."""
        sql = "SELECT SUM(res_oc) FROM field_resources"
        result = enrich_sql_query(sql)

        assert "project_class" not in result.enriched_sql
        assert "project_stage" not in result.enriched_sql
        assert "field_remarks" in result.enriched_sql  # But remarks still added

    def test_enrich_returns_correct_metadata(self):
        """Test that enrichment returns correct metadata."""
        sql = "SELECT rec_oc, rec_an FROM field_resources"
        result = enrich_sql_query(sql)

        assert result.table == "field_resources"
        assert "project_class" in result.classification_columns
        assert "project_stage" in result.classification_columns
        assert result.remarks_column == "field_remarks"

    def test_enrich_no_table_returns_unchanged(self):
        """Test that queries without table are returned unchanged."""
        sql = "SELECT 1 + 1"
        result = enrich_sql_query(sql)

        assert result.enriched_sql == sql
        assert result.table == ""

    def test_enrich_preserves_where_conditions(self):
        """Test that WHERE conditions are preserved."""
        sql = "SELECT SUM(rec_oc) FROM field_resources WHERE field_name = 'Duri' AND report_year = 2024"
        result = enrich_sql_query(sql)

        assert "field_name = 'Duri'" in result.enriched_sql
        assert "report_year = 2024" in result.enriched_sql


class TestShouldIncludeRemarks:
    """Tests for should_include_remarks function."""

    def test_meaningful_remarks(self):
        """Test that meaningful remarks should be shown."""
        assert should_include_remarks("Field under development planning") is True
        assert should_include_remarks("Exploration ongoing in northern block") is True

    def test_empty_remarks(self):
        """Test that empty remarks should not be shown."""
        assert should_include_remarks("") is False
        assert should_include_remarks(None) is False

    def test_generic_remarks(self):
        """Test that generic/placeholder remarks should not be shown."""
        assert should_include_remarks("-") is False
        assert should_include_remarks("null") is False
        assert should_include_remarks("N/A") is False
        assert should_include_remarks("na") is False


class TestIsAlreadyEnriched:
    """Tests for is_already_enriched function (hybrid approach detector)."""

    def test_detects_model_enriched_full(self):
        """Test detection when model enriched with all required columns."""
        from esdc.chat.domain_knowledge.functions import is_already_enriched

        sql = "SELECT field_remarks, project_class, project_stage, SUM(rec_oc) FROM field_resources"
        is_enriched, source = is_already_enriched(sql)

        assert is_enriched is True
        assert source == "model"

    def test_detects_needs_fallback_for_rec(self):
        """Test detection when model forgot classification for rec_* query."""
        from esdc.chat.domain_knowledge.functions import is_already_enriched

        sql = "SELECT SUM(rec_oc) FROM field_resources"
        is_enriched, source = is_already_enriched(sql)

        assert is_enriched is False
        assert source == "needs_fallback"

    def test_detects_no_classification_needed_for_res(self):
        """Test detection for res_* queries (no classification needed)."""
        from esdc.chat.domain_knowledge.functions import is_already_enriched

        sql = "SELECT res_oc FROM field_resources"
        is_enriched, source = is_already_enriched(sql)

        assert is_enriched is True
        assert source == "no_classification_needed"


class TestHybridEnrichmentBehavior:
    """Tests for hybrid model-guided + code fallback approach."""

    def test_skips_when_model_already_enriched(self):
        """Test enrichment is skipped when model already added context."""
        sql = "SELECT field_remarks, project_class, project_stage, SUM(rec_oc) FROM field_resources"
        result = enrich_sql_query(sql)

        assert result.was_already_enriched is True
        assert result.enrichment_source == "model"
        assert result.added_columns == []
        assert result.enriched_sql == sql

    def test_fallback_when_model_forgot(self):
        """Test fallback enrichment when model forgot context."""
        sql = "SELECT SUM(rec_oc) FROM field_resources"
        result = enrich_sql_query(sql)

        assert result.was_already_enriched is False
        assert result.enrichment_source == "fallback"
        assert "field_remarks" in result.added_columns
        assert "project_class" in result.added_columns


class TestBuildEnrichedSql:
    """Tests for _build_enriched_sql internal function."""

    def test_build_adds_columns(self):
        """Test that columns are added to SELECT."""
        sql = "SELECT SUM(rec_oc) FROM field_resources"
        enriched = _build_enriched_sql(
            original_sql=sql,
            added_columns=["project_class", "project_stage"],
            group_by_columns=["project_class", "project_stage"],
        )

        assert "pr.project_class" in enriched
        assert "pr.project_stage" in enriched


class TestClassificationConstants:
    """Tests for classification-related constants."""

    def test_classification_context_columns(self):
        """Test that classification context columns are defined."""
        columns = get_classification_context_columns()
        assert "project_class" in columns
        assert "project_stage" in columns

    def test_requires_classification_prefixes(self):
        """Test that classification prefixes are defined."""
        assert "rec_" in REQUIRES_CLASSIFICATION_PREFIXES


class TestIntegration:
    """Integration tests combining multiple functions."""

    def test_full_enrichment_pipeline(self):
        """Test complete enrichment pipeline for resources query."""
        original_sql = """
            SELECT SUM(rec_oc) as oil_total, SUM(rec_an) as gas_total
            FROM field_resources
            WHERE field_name LIKE '%Abadi%'
            AND report_year = (SELECT MAX(report_year) FROM field_resources)
        """

        result = enrich_sql_query(original_sql)

        # Verify enrichment happened
        assert result.table == "field_resources"
        assert len(result.added_columns) > 0

        # Verify classification columns added
        assert "project_class" in result.enriched_sql
        assert "project_stage" in result.enriched_sql

        # Verify remarks added
        assert "field_remarks" in result.enriched_sql

        # Verify GROUP BY added
        assert "GROUP BY" in result.enriched_sql.upper()

        # Verify original conditions preserved
        assert "field_name LIKE '%Abadi%'" in result.enriched_sql
        assert "MAX(report_year)" in result.enriched_sql

    def test_enrichment_for_work_area_query(self):
        """Test enrichment for work area resources query."""
        sql = "SELECT SUM(rec_oc) FROM wa_resources WHERE wk_name = 'Masela'"
        result = enrich_sql_query(sql)

        assert result.table == "wa_resources"
        assert "project_class" in result.enriched_sql
        assert "project_stage" in result.enriched_sql
        assert "wa_remarks" in result.enriched_sql
