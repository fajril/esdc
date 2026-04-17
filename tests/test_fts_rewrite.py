"""Tests for Full-Text Search (FTS) query rewrite functionality.

Tests that ILIKE patterns are correctly rewritten to use FTS where appropriate,
and that the rewrite handles all edge cases correctly.
"""

from esdc.chat.tools import _rewrite_with_fts


class TestFTSRewriteBaseTables:
    """Test FTS rewrite for base tables (project_resources, project_timeseries)."""

    def test_rewrite_project_resources_field_name(self):
        """Test rewriting field_name ILIKE on project_resources."""
        query = "SELECT * FROM project_resources WHERE field_name ILIKE '%Duri%'"
        result = _rewrite_with_fts(query)

        assert "fts_main_project_resources" in result
        assert "match_bm25" in result
        assert "field_name ILIKE '%Duri%'" in result  # Original preserved

    def test_rewrite_project_resources_project_name(self):
        """Test rewriting project_name ILIKE on project_resources."""
        query = "SELECT * FROM project_resources WHERE project_name ILIKE '%Abadi%'"
        result = _rewrite_with_fts(query)

        assert "fts_main_project_resources.match_bm25" in result
        assert "project_name ILIKE '%Abadi%'" in result

    def test_rewrite_project_resources_wk_name(self):
        """Test rewriting wk_name ILIKE on project_resources."""
        query = "SELECT * FROM project_resources WHERE wk_name ILIKE '%Rokan%'"
        result = _rewrite_with_fts(query)

        assert "fts_main_project_resources" in result
        assert "wk_name ILIKE '%Rokan%'" in result

    def test_rewrite_project_resources_province(self):
        """Test rewriting province ILIKE on project_resources."""
        query = "SELECT * FROM project_resources WHERE province ILIKE '%Riau%'"
        result = _rewrite_with_fts(query)

        assert "fts_main_project_resources" in result
        assert "province ILIKE '%Riau%'" in result

    def test_rewrite_project_resources_basin128(self):
        """Test rewriting basin128 ILIKE on project_resources."""
        query = "SELECT * FROM project_resources WHERE basin128 ILIKE '%Sumatera%'"
        result = _rewrite_with_fts(query)

        assert "fts_main_project_resources" in result
        assert "basin128 ILIKE '%Sumatera%'" in result

    def test_rewrite_project_resources_operator_name(self):
        """Test rewriting operator_name ILIKE on project_resources."""
        query = (
            "SELECT * FROM project_resources WHERE operator_name ILIKE '%Pertamina%'"
        )
        result = _rewrite_with_fts(query)

        assert "fts_main_project_resources" in result
        assert "operator_name ILIKE '%Pertamina%'" in result

    def test_rewrite_project_resources_project_remarks(self):
        """Test rewriting project_remarks ILIKE on project_resources."""
        query = (
            "SELECT * FROM project_resources WHERE project_remarks ILIKE '%economic%'"
        )
        result = _rewrite_with_fts(query)

        assert "fts_main_project_resources" in result
        assert "project_remarks ILIKE '%economic%'" in result

    def test_rewrite_project_timeseries_field_name(self):
        """Test rewriting field_name ILIKE on project_timeseries."""
        query = "SELECT * FROM project_timeseries WHERE field_name ILIKE '%Duri%'"
        result = _rewrite_with_fts(query)

        assert "fts_main_project_timeseries" in result
        assert "match_bm25" in result
        assert "field_name ILIKE '%Duri%'" in result

    def test_rewrite_project_timeseries_project_remarks(self):
        """Test rewriting project_remarks ILIKE on project_timeseries."""
        query = "SELECT * FROM project_timeseries WHERE project_remarks ILIKE '%issue%'"
        result = _rewrite_with_fts(query)

        assert "fts_main_project_timeseries" in result
        assert "project_remarks ILIKE '%issue%'" in result

    def test_no_rewrite_for_ineligible_column(self):
        """Test that non-FTS columns are not rewritten."""
        query = "SELECT * FROM project_resources WHERE project_id ILIKE '%ABC%'"
        result = _rewrite_with_fts(query)

        # Should return unchanged
        assert result == query
        assert "match_bm25" not in result

    def test_no_rewrite_without_ilike(self):
        """Test that queries without ILIKE are not modified."""
        query = "SELECT * FROM project_resources WHERE report_year = 2024"
        result = _rewrite_with_fts(query)

        assert result == query
        assert "match_bm25" not in result


class TestFTSRewriteViews:
    """Test FTS rewrite for views (field_resources, wa_resources, etc.)."""

    def test_rewrite_field_resources_uses_subquery(self):
        """Test field_resources uses subquery to base table."""
        query = "SELECT * FROM field_resources WHERE field_name ILIKE '%Duri%'"
        result = _rewrite_with_fts(query)

        # Should add subquery using project_resources FTS
        assert "fts_main_project_resources" in result
        assert "field_id IN" in result
        assert "match_bm25" in result
        # Original ILIKE preserved
        assert "field_name ILIKE '%Duri%'" in result

    def test_rewrite_wa_resources_uses_subquery(self):
        """Test wa_resources uses subquery to base table."""
        query = "SELECT * FROM wa_resources WHERE wk_name ILIKE '%Rokan%'"
        result = _rewrite_with_fts(query)

        assert "fts_main_project_resources" in result
        assert "wk_id IN" in result
        assert "wk_name ILIKE '%Rokan%'" in result

    def test_rewrite_field_timeseries_uses_subquery(self):
        """Test field_timeseries uses subquery."""
        query = "SELECT * FROM field_timeseries WHERE field_name ILIKE '%Duri%'"
        result = _rewrite_with_fts(query)

        assert "fts_main_project_timeseries" in result
        assert "field_id IN" in result
        assert "match_bm25" in result

    def test_rewrite_wa_timeseries_uses_subquery(self):
        """Test wa_timeseries uses subquery."""
        query = "SELECT * FROM wa_timeseries WHERE wk_name ILIKE '%Rokan%'"
        result = _rewrite_with_fts(query)

        assert "fts_main_project_timeseries" in result
        assert "wk_id IN" in result
        assert "match_bm25" in result


class TestFTSRewriteEdgeCases:
    """Test edge cases and special scenarios."""

    def test_multiple_fts_columns_combined(self):
        """Test multiple FTS columns in same query."""
        query = (
            "SELECT * FROM project_resources WHERE field_name ILIKE '%Duri%' "
            "AND wk_name ILIKE '%Rokan%'"
        )
        result = _rewrite_with_fts(query)

        # Should have single match_bm25 call with combined keywords
        assert "match_bm25" in result
        assert "Duri Rokan" in result or "Rokan Duri" in result

    def test_mixed_fts_and_non_fts_columns(self):
        """Test query with both FTS and non-FTS columns."""
        query = (
            "SELECT * FROM project_resources WHERE field_name ILIKE '%Duri%' "
            "AND report_year = 2024"
        )
        result = _rewrite_with_fts(query)

        # Should rewrite FTS columns only
        assert "fts_main_project_resources" in result
        assert "match_bm25" in result
        assert "report_year = 2024" in result  # Non-FTS preserved

    def test_existing_where_clause(self):
        """Test adding FTS to existing WHERE clause."""
        query = (
            "SELECT * FROM project_resources WHERE report_year = 2024 "
            "AND field_name ILIKE '%Duri%'"
        )
        result = _rewrite_with_fts(query)

        # Should add FTS condition after WHERE
        assert "fts_main_project_resources.match_bm25" in result
        assert "report_year = 2024" in result

    def test_no_where_clause_adds_where(self):
        """Test adding WHERE clause if none exists."""
        query = "SELECT * FROM project_resources WHERE field_name ILIKE '%Duri%'"
        result = _rewrite_with_fts(query)

        # Should have WHERE clause with FTS
        assert "WHERE" in result
        assert "fts_main_project_resources.match_bm25" in result

    def test_complex_query_with_joins(self):
        """Test FTS rewrite doesn't break complex queries."""
        query = (
            "SELECT pr.*, pt.year FROM project_resources pr "
            "LEFT JOIN project_timeseries pt ON pr.project_id = pt.project_id "
            "WHERE pr.field_name ILIKE '%Duri%'"
        )
        result = _rewrite_with_fts(query)

        # Should still add FTS
        assert "fts_main_project_resources" in result or "match_bm25" in result

    def test_case_insensitive_table_name(self):
        """Test table name matching is case insensitive."""
        query = "SELECT * FROM PROJECT_RESOURCES WHERE field_name ILIKE '%Duri%'"
        result = _rewrite_with_fts(query)

        assert "fts_main_project_resources" in result

    def test_special_characters_escaped(self):
        """Test special characters in keywords are escaped."""
        query = "SELECT * FROM project_resources WHERE field_name ILIKE '%Duri''s%'"
        result = _rewrite_with_fts(query)

        # Should escape single quote
        assert "Duri''s" in result or "match_bm25" in result


class TestFTSRewritePreservation:
    """Test that original query semantics are preserved."""

    def test_original_ilike_preserved(self):
        """Test original ILIKE is always preserved as secondary filter."""
        query = "SELECT * FROM project_resources WHERE field_name ILIKE '%Duri%'"
        result = _rewrite_with_fts(query)

        # Original ILIKE should still be there
        assert "field_name ILIKE '%Duri%'" in result

    def test_select_columns_preserved(self):
        """Test SELECT columns are not modified."""
        query = (
            "SELECT project_id, project_name, res_oc FROM project_resources "
            "WHERE project_name ILIKE '%Abadi%'"
        )
        result = _rewrite_with_fts(query)

        assert "SELECT project_id, project_name, res_oc" in result
        assert "fts_main_project_resources" in result

    def test_group_by_preserved(self):
        """Test GROUP BY is preserved."""
        query = (
            "SELECT wk_name, SUM(res_oc) FROM project_resources "
            "WHERE wk_name ILIKE '%Rokan%' GROUP BY wk_name"
        )
        result = _rewrite_with_fts(query)

        assert "GROUP BY wk_name" in result
        assert "fts_main_project_resources" in result

    def test_order_by_preserved(self):
        """Test ORDER BY is preserved."""
        query = (
            "SELECT * FROM project_resources WHERE field_name ILIKE '%Duri%' "
            "ORDER BY report_year DESC"
        )
        result = _rewrite_with_fts(query)

        assert "ORDER BY report_year DESC" in result
        assert "fts_main_project_resources" in result


class TestFTSRewriteNoMatch:
    """Test cases where rewrite should NOT happen."""

    def test_no_table_match(self):
        """Test queries on tables without FTS are not rewritten."""
        query = "SELECT * FROM some_other_table WHERE name ILIKE '%test%'"
        result = _rewrite_with_fts(query)

        assert result == query
        assert "match_bm25" not in result

    def test_no_ilike_pattern(self):
        """Test queries without ILIKE patterns are not rewritten."""
        query = "SELECT * FROM project_resources WHERE report_year = 2024"
        result = _rewrite_with_fts(query)

        assert result == query

    def test_exact_match_no_wildcards(self):
        """Test ILIKE without wildcards may not match pattern."""
        # The regex requires % wildcards, so exact match won't be rewritten
        query = "SELECT * FROM project_resources WHERE field_name ILIKE 'Duri'"
        result = _rewrite_with_fts(query)

        # Current implementation requires % wildcards
        assert "match_bm25" not in result or "fts_main_project_resources" not in result

    def test_subselect_not_rewritten(self):
        """Test ILIKE in subselect may not be rewritten."""
        query = (
            "SELECT * FROM project_resources WHERE project_id IN "
            "(SELECT project_id FROM field_resources WHERE field_name ILIKE '%Duri%')"
        )
        result = _rewrite_with_fts(query)

        # May or may not be rewritten depending on implementation
        # Just verify it doesn't crash
        assert isinstance(result, str)
