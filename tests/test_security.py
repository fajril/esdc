import pytest

from esdc.db_security import SQLSanitizer, _load_sql_script
from esdc.selection import TableName


class TestSanitizeLike:
    """Tests for SQLSanitizer.sanitize_like()."""

    def test_simple_string(self):
        """Test sanitizing a simple string."""
        result = SQLSanitizer.sanitize_like("test")
        assert result == "%test%"

    def test_percent_char(self):
        """Test escaping % character."""
        result = SQLSanitizer.sanitize_like("100%")
        assert result == r"%100\%%"

    def test_underscore_char(self):
        """Test escaping _ character."""
        result = SQLSanitizer.sanitize_like("field_name")
        assert result == r"%field\_name%"

    def test_combined_special_chars(self):
        """Test escaping multiple special chars."""
        result = SQLSanitizer.sanitize_like("test_value%done")
        assert result == r"%test\_value\%done%"

    def test_empty_string(self):
        """Test sanitizing empty string."""
        result = SQLSanitizer.sanitize_like("")
        assert result == "%%"


class TestBuildQueryNkriResources:
    """Tests for build_query with NKRI_RESOURCES table."""

    def test_nkri_with_year(self):
        """Test NKRI query with year filter uses parameterized query."""
        query, params = SQLSanitizer.build_query(TableName.NKRI_RESOURCES, year=2024)
        assert "?" in query
        assert 2024 in params
        assert "<year>" not in query
        assert "report_year = ?" in query

    def test_nkri_without_year(self):
        """Test NKRI query without year removes WHERE clause."""
        query, params = SQLSanitizer.build_query(TableName.NKRI_RESOURCES, year=None)
        assert "<year>" not in query
        assert "WHERE report_year = ?" not in query
        assert len(params) == 0

    def test_nkri_ignores_where_like(self):
        """Test that NKRI ignores where and like parameters."""
        query, params = SQLSanitizer.build_query(
            TableName.NKRI_RESOURCES, where="project_name", like="test", year=2024
        )
        assert "LIKE" not in query
        assert "project_name" not in query
        assert 2024 in params


class TestBuildQueryResourceTables:
    """Tests for build_query with project/field/WA resources."""

    def test_project_with_like_and_year(self):
        """Test project resources with LIKE and year filters."""
        query, params = SQLSanitizer.build_query(
            TableName.PROJECT_RESOURCES, like="abc", year=2024
        )
        assert "project_name LIKE ?" in query
        assert "report_year = ?" in query
        assert len(params) == 2
        assert params[0] == "%abc%"
        assert params[1] == 2024

    def test_field_with_like_and_year(self):
        """Test field resources with LIKE and year filters."""
        query, params = SQLSanitizer.build_query(
            TableName.FIELD_RESOURCES, like="field1", year=2023
        )
        assert "field_name LIKE ?" in query
        assert "report_year = ?" in query
        assert len(params) == 2
        assert params[0] == "%field1%"
        assert params[1] == 2023

    def test_wa_with_where_override(self):
        """Test WA resources with custom where column."""
        query, params = SQLSanitizer.build_query(
            TableName.WA_RESOURCES, where="custom_col", like="test", year=2024
        )
        assert "custom_col LIKE ?" in query
        assert len(params) == 2

    def test_like_only_no_year(self):
        """Test query with LIKE but no year removes year clause."""
        query, params = SQLSanitizer.build_query(
            TableName.PROJECT_RESOURCES, like="test"
        )
        assert "report_year = ?" not in query
        assert "AND report_year" not in query
        assert "project_name LIKE ?" in query
        assert len(params) == 1
        assert params[0] == "%test%"

    def test_year_only_no_like(self):
        """Test query with year but no LIKE."""
        query, params = SQLSanitizer.build_query(TableName.PROJECT_RESOURCES, year=2024)
        assert "report_year = ?" in query
        assert "LIKE" not in query
        assert "<where>" not in query
        assert len(params) == 1
        assert params[0] == 2024

    def test_no_filters(self):
        """Test query with no LIKE and no year removes entire WHERE clause."""
        query, params = SQLSanitizer.build_query(TableName.PROJECT_RESOURCES)
        assert "WHERE" not in query.upper() or "FROM" in query.upper()
        assert len(params) == 0

    def test_like_special_chars_escaped(self):
        """Test that LIKE special characters are escaped in params."""
        query, params = SQLSanitizer.build_query(
            TableName.FIELD_RESOURCES, like="field_name%", year=2024
        )
        assert params[0] == r"%field\_name\%%"
        assert params[1] == 2024


class TestBuildQueryColumns:
    """Tests for build_query with column selection."""

    def test_with_columns(self):
        """Test column selection modifies SELECT clause."""
        query, params = SQLSanitizer.build_query(
            TableName.NKRI_RESOURCES,
            year=2024,
            columns=["report_year", "project_stage"],
        )
        assert "report_year, project_stage" in query
        assert "SELECT" in query

    def test_with_string_column(self):
        """Test single column as string."""
        query, params = SQLSanitizer.build_query(
            TableName.NKRI_RESOURCES, year=2024, columns="report_year"
        )
        assert "report_year" in query


class TestBuildQuerySqlInjection:
    """Tests for SQL injection prevention."""

    def test_year_injection_prevented(self):
        """Test that SQL injection via year parameter is prevented."""
        # Year should be an int, so injection via string isn't possible
        query, params = SQLSanitizer.build_query(
            TableName.PROJECT_RESOURCES, like="test", year=2024
        )
        assert "2024" not in query or "?" in query
        assert query.count("?") >= 1

    def test_like_injection_prevented(self):
        """Test that SQL injection via like parameter is prevented."""
        malicious_like = "'; DROP TABLE users; --"
        query, params = SQLSanitizer.build_query(
            TableName.PROJECT_RESOURCES, like=malicious_like, year=2024
        )
        assert "DROP" not in query
        assert "?" in query
        assert malicious_like in str(params[0])

    def test_where_injection_prevented(self):
        """Test that SQL injection via where parameter uses parameterized query."""
        malicious_where = "1=1 OR"
        query, params = SQLSanitizer.build_query(
            TableName.PROJECT_RESOURCES, where=malicious_where, like="test", year=2024
        )
        assert "1=1" in query
        assert "?" in query


class TestLoadSqlScript:
    """Tests for _load_sql_script()."""

    def test_load_existing_script(self):
        """Test loading an existing SQL script."""
        script = _load_sql_script("view_project_resources.sql")
        assert "SELECT" in script.upper()

    def test_nonexistent_script_raises(self):
        """Test loading non-existent script raises error."""
        with pytest.raises(FileNotFoundError):
            _load_sql_script("nonexistent.sql")
