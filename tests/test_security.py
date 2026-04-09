import pytest

from esdc.db_security import SQLSanitizer, _load_sql_script
from esdc.selection import TableName


class TestSanitizeLike:
    def test_simple_string(self):
        result = SQLSanitizer.sanitize_like("test")
        assert result == "%test%"

    def test_percent_char(self):
        result = SQLSanitizer.sanitize_like("100%")
        assert result == r"%100\%%"

    def test_underscore_char(self):
        result = SQLSanitizer.sanitize_like("field_name")
        assert result == r"%field\_name%"

    def test_combined_special_chars(self):
        result = SQLSanitizer.sanitize_like("test_value%done")
        assert result == r"%test\_value\%done%"

    def test_empty_string(self):
        result = SQLSanitizer.sanitize_like("")
        assert result == "%%"


class TestBuildQueryNkriResources:
    def test_nkri_with_year(self):
        query, params = SQLSanitizer.build_query(TableName.NKRI_RESOURCES, years=[2024])
        assert "?" in query
        assert 2024 in params
        assert "report_year = ?" in query

    def test_nkri_with_multiple_years(self):
        query, params = SQLSanitizer.build_query(
            TableName.NKRI_RESOURCES, years=[2024, 2025]
        )
        assert "report_year IN (?, ?)" in query
        assert 2024 in params
        assert 2025 in params

    def test_nkri_without_year(self):
        query, params = SQLSanitizer.build_query(TableName.NKRI_RESOURCES)
        assert "WHERE report_year" not in query
        assert len(params) == 0

    def test_nkri_ignores_where_like(self):
        query, params = SQLSanitizer.build_query(
            TableName.NKRI_RESOURCES, where="project_name", like="test", years=[2024]
        )
        assert "ILIKE" not in query
        assert "project_name" not in query
        assert 2024 in params


class TestBuildQueryResourceTables:
    def test_project_with_like_and_year(self):
        query, params = SQLSanitizer.build_query(
            TableName.PROJECT_RESOURCES, like="abc", years=[2024]
        )
        assert "project_name ILIKE ?" in query
        assert "report_year = ?" in query
        assert len(params) == 2
        assert params[0] == "%abc%"
        assert params[1] == 2024

    def test_field_with_like_and_year(self):
        query, params = SQLSanitizer.build_query(
            TableName.FIELD_RESOURCES, like="field1", years=[2023]
        )
        assert "field_name ILIKE ?" in query
        assert "report_year = ?" in query
        assert len(params) == 2
        assert params[0] == "%field1%"
        assert params[1] == 2023

    def test_field_with_multiple_years(self):
        query, params = SQLSanitizer.build_query(
            TableName.FIELD_RESOURCES, like="abadi", years=[2024, 2025]
        )
        assert "report_year IN (?, ?)" in query
        assert params.count(2024) >= 1
        assert params.count(2025) >= 1

    def test_wa_with_where_override(self):
        query, params = SQLSanitizer.build_query(
            TableName.WA_RESOURCES, where="custom_col", like="test", years=[2024]
        )
        assert "custom_col ILIKE ?" in query

    def test_like_only_no_year(self):
        query, params = SQLSanitizer.build_query(
            TableName.PROJECT_RESOURCES, like="test"
        )
        assert "report_year" not in query or "SELECT" in query
        assert "project_name ILIKE ?" in query
        assert len(params) == 1
        assert params[0] == "%test%"

    def test_year_only_no_like(self):
        query, params = SQLSanitizer.build_query(
            TableName.PROJECT_RESOURCES, years=[2024]
        )
        assert "report_year = ?" in query
        assert "ILIKE" not in query
        assert len(params) == 1
        assert params[0] == 2024

    def test_no_filters(self):
        query, params = SQLSanitizer.build_query(TableName.PROJECT_RESOURCES)
        assert "WHERE" not in query.upper() or "FROM" in query.upper()
        assert len(params) == 0

    def test_like_special_chars_escaped(self):
        query, params = SQLSanitizer.build_query(
            TableName.FIELD_RESOURCES, like="field_name%", years=[2024]
        )
        assert params[0] == r"%field\_name\%%"
        assert params[1] == 2024


class TestBuildQueryDetailLevel:
    def test_default_detail_is_resources(self):
        query, params = SQLSanitizer.build_query(
            TableName.FIELD_RESOURCES, years=[2025]
        )
        assert "resources_mstb" in query
        assert "reserves_mstb" in query

    def test_detail_reserves(self):
        query, params = SQLSanitizer.build_query(
            TableName.FIELD_RESOURCES, details=["reserves"], years=[2025]
        )
        assert "reserves_mstb" in query
        assert "resources_mstb" not in query

    def test_detail_inplace_project(self):
        query, params = SQLSanitizer.build_query(
            TableName.PROJECT_RESOURCES, details=["inplace"], years=[2025]
        )
        assert "prj_ioip AS 'ioip'" in query
        assert "prj_igip AS 'igip'" in query

    def test_detail_inplace_field(self):
        query, params = SQLSanitizer.build_query(
            TableName.FIELD_RESOURCES, details=["inplace"], years=[2025]
        )
        assert "ioip AS 'ioip'" in query or "ioip" in query

    def test_detail_all(self):
        query, params = SQLSanitizer.build_query(
            TableName.FIELD_RESOURCES, details=["all"], years=[2025]
        )
        assert "SELECT *" in query

    def test_detail_multiple(self):
        query, params = SQLSanitizer.build_query(
            TableName.FIELD_RESOURCES, details=["reserves", "inplace"], years=[2025]
        )
        assert "reserves_mstb" in query
        assert "ioip" in query

    def test_detail_invalid(self):
        with pytest.raises(ValueError, match="Invalid detail"):
            SQLSanitizer.build_query(
                TableName.FIELD_RESOURCES, details=["invalid_name"]
            )

    def test_detail_cumprod(self):
        query, params = SQLSanitizer.build_query(
            TableName.FIELD_RESOURCES, details=["cumprod"], years=[2025]
        )
        assert "sales_cumprod_mstb" in query
        assert "is_discovered" in query


class TestBuildQueryColumns:
    def test_with_columns(self):
        query, params = SQLSanitizer.build_query(
            TableName.NKRI_RESOURCES,
            years=[2024],
            columns=["report_year", "project_stage"],
        )
        assert "report_year, project_stage" in query

    def test_with_string_column(self):
        query, params = SQLSanitizer.build_query(
            TableName.NKRI_RESOURCES, years=[2024], columns="report_year"
        )
        assert "report_year" in query


class TestBuildQuerySqlInjection:
    def test_year_injection_prevented(self):
        query, params = SQLSanitizer.build_query(
            TableName.PROJECT_RESOURCES, like="test", years=[2024]
        )
        assert query.count("?") >= 1

    def test_like_injection_prevented(self):
        malicious_like = "'; DROP TABLE users; --"
        query, params = SQLSanitizer.build_query(
            TableName.PROJECT_RESOURCES, like=malicious_like, years=[2024]
        )
        assert "DROP" not in query
        assert "?" in query

    def test_where_injection_prevented(self):
        malicious_where = "1=1 OR"
        query, params = SQLSanitizer.build_query(
            TableName.PROJECT_RESOURCES,
            where=malicious_where,
            like="test",
            years=[2024],
        )
        assert "1=1" in query
        assert "?" in query


class TestLoadSqlScript:
    def test_load_existing_script(self):
        script = _load_sql_script("create_esdc_view.sql")
        assert "SELECT" in script.upper()

    def test_nonexistent_script_raises(self):
        with pytest.raises(FileNotFoundError):
            _load_sql_script("nonexistent.sql")
