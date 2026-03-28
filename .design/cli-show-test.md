# CLI Show Command Test Plan

This document outlines a comprehensive test strategy for the `esdc show` command, covering all functionality described in the CLI show interface design document.

---

## Current Test Coverage Assessment

### Existing Test Suite Overview

**Unit Tests** (`tests/unit/test_show_command.py`)
- **Lines of Code**: 843 lines
- **Test Classes**: 15 classes covering:
  - `TestShowCommandArguments` (4 tests) - Table name validation
  - `TestShowCommandOptions` (13 tests) - Command options handling
  - `TestShowCommandFiltering` (8 tests) - Filter combinations and behavior
  - `TestShowCommandOutput` (3 tests) - Console output formatting
  - `TestShowCommandExcelExport` (5 tests) - Excel export functionality
  - `TestShowCommandDetailLevels` (4 tests) - Output level mapping
  - `TestShowCommandSQLConstruction` (7 tests) - SQL query building
  - `TestShowCommandErrorHandling` (8 tests) - Error scenarios
  - `TestShowCommandTableSpecific` (5 tests) - Table-specific behavior

**Integration Tests** (`tests/integration/test_show_integration.py`)
- **Lines of Code**: 306 lines
- **Test Classes**: 1 class covering:
  - Full workflow with real database operations
  - Multi-table querying scenarios
  - CLI invocation via Typer runner
  - Real data filtering and validation
  - Excel export integration

### Current Coverage Strengths

✅ **Well Covered Areas:**
- Table name validation and enumeration
- Command option parsing and validation
- Basic filtering functionality (where, search, year)
- SQL placeholder replacement
- Excel export workflow
- Error handling for missing database
- Basic output formatting

✅ **Test Infrastructure:**
- Comprehensive fixtures for sample data (project, field, WA, NKRI)
- Mock SQL views for all table types
- Temporary database setup
- Rich test data covering multiple years and scenarios

### Identified Coverage Gaps

❌ **Missing Test Categories:**
1. **Performance and Load Testing**
   - Large dataset handling (>10,000 records)
   - Query performance benchmarks
   - Memory usage validation

2. **Edge Cases and Boundary Conditions**
   - Empty search patterns with special characters
   - Very long search terms (>255 characters)
   - Extreme year values (2019, future years)
   - Unicode and special character handling

3. **Data Integrity and Validation**
   - Column aliasing verification
   - Data type preservation through formatting
   - Null/None value handling
   - Numeric precision validation

4. **Concurrent Access Scenarios**
   - Database locking during queries
   - Multiple simultaneous CLI invocations
   - Database modification during query execution

5. **Output Formatting Edge Cases**
   - Very large numbers (trillions, quadrillions)
   - Very small decimal values
   - Negative values in resources
   - Mixed data types in columns

6. **Cross-Platform Compatibility**
   - Path handling on Windows vs Unix
   - Excel file creation permissions
   - Database file locking behavior

---

## Comprehensive Test Plan

### 1. Enhanced Argument Validation Tests

#### 1.1 Table Name Validation
```python
class TestTableNameValidation:
    """Test table name validation with comprehensive edge cases."""

    @pytest.mark.parametrize("valid_name", [
        "project_resources",
        "field_resources",
        "wa_resources",
        "nkri_resources"
    ])
    def test_valid_table_names_lowercase(self, valid_name):
        """Test all valid table names are accepted."""
        result = TableName(valid_name)
        assert result.value == valid_name

    @pytest.mark.parametrize("invalid_name", [
        "PROJECT_RESOURCES",  # Wrong case
        "project-resources",  # Wrong separator
        "project resources",  # Space instead of underscore
        "project_resources_extra",  # Extra suffix
        "project",  # Too short
        "resources",  # Too short
        "",  # Empty string
        "proj_resource",  # Misspelled
    ])
    def test_invalid_table_names_variations(self, invalid_name):
        """Test various invalid table name formats."""
        with pytest.raises(ValueError, match="is not a valid TableName"):
            TableName(invalid_name)

    def test_table_name_with_special_characters(self):
        """Test table names with special characters are rejected."""
        invalid_names = ["project@resources", "project#resources", "project/resources"]
        for name in invalid_names:
            with pytest.raises(ValueError):
                TableName(name)

    def test_table_name_unicode_handling(self):
        """Test table name validation with unicode characters."""
        # Should reject unicode table names
        with pytest.raises(ValueError):
            TableName("project_ресурсы")  # Cyrillic

    def test_table_name_whitespace_handling(self):
        """Test table names with leading/trailing whitespace."""
        with pytest.raises(ValueError):
            TableName("  project_resources  ")
```

#### 1.2 Parameter Validation
```python
class TestParameterValidation:
    """Test CLI parameter validation with boundary conditions."""

    def test_year_minimum_boundary_2019(self):
        """Test year parameter minimum value (2019)."""
        from typer.testing import CliRunner
        runner = CliRunner()

        # Should accept year 2019
        result = runner.invoke(app, [
            "show", "project_resources",
            "--year", "2019"
        ])
        assert result.exit_code == 0 or "Database file does not exist" in result.stdout

    def test_year_below_minimum_rejected(self):
        """Test year values below 2019 are rejected."""
        runner = CliRunner()

        for invalid_year in ["2018", "2000", "0", "-1"]:
            result = runner.invoke(app, [
                "show", "project_resources",
                "--year", invalid_year
            ])
            assert result.exit_code != 0
            assert "Invalid value" in result.stdout or "min" in result.stdout.lower()

    @pytest.mark.parametrize("future_year", [2030, 2050, 2100])
    def test_year_future_values_handling(self, future_year):
        """Test future year values are accepted but may return empty results."""
        runner = CliRunner()

        result = runner.invoke(app, [
            "show", "project_resources",
            "--year", str(future_year)
        ])
        # Should accept the parameter but may find no data
        assert result.exit_code == 0 or "Database file does not exist" in result.stdout

    @pytest.mark.parametrize("output_level", [-1, -5, -10])
    def test_output_level_negative_values(self, output_level):
        """Test negative output levels are handled gracefully."""
        from esdc.esdc import show
        from unittest.mock import patch

        with patch('esdc.esdc.run_query') as mock_run_query:
            mock_run_query.return_value = pd.DataFrame()

            # Should handle negative levels by capping or using default
            show(table="project_resources", output=output_level)

            # Verify the function was called (handled gracefully)
            mock_run_query.assert_called_once()

    @pytest.mark.parametrize("output_level", [100, 1000, 9999])
    def test_output_level_very_large_values(self, output_level):
        """Test very large output levels are capped at maximum."""
        from esdc.esdc import show
        from unittest.mock import patch

        with patch('esdc.esdc.run_query') as mock_run_query:
            mock_run_query.return_value = pd.DataFrame()

            # Large values should be capped at 4
            show(table="project_resources", output=output_level)

            # Verify call with capped value
            call_args = mock_run_query.call_args
            assert call_args[1]['output'] <= 4

    def test_columns_parameter_whitespace_handling(self):
        """Test columns parameter with various whitespace patterns."""
        from esdc.esdc import show

        test_cases = [
            "project_name field_name",  # Normal case
            " project_name field_name ",  # Leading/trailing spaces
            "project_name  field_name",   # Multiple spaces
            "   project_name   field_name   ",  # Excessive spaces
        ]

        for columns_str in test_cases:
            with patch('esdc.esdc.run_query') as mock_run_query:
                mock_run_query.return_value = pd.DataFrame()

                show(table="project_resources", columns=columns_str)

                # Verify columns were split correctly
                call_args = mock_run_query.call_args
                expected_columns = ["project_name", "field_name"]
                assert call_args[1]['columns'] == expected_columns
```

### 2. Advanced Filtering Tests

#### 2.1 Search Pattern Edge Cases
```python
class TestSearchPatternEdgeCases:
    """Test search pattern handling with edge cases and security concerns."""

    def test_empty_search_with_multiple_calls(self, mock_database, mock_sql_view_project):
        """Test empty search string handling consistency."""
        tmp_path, db_path = mock_database

        with patch('esdc.dbmanager.Config.get_db_path', return_value=tmp_path):
            with patch('esdc.dbmanager._load_sql_script', return_value=mock_sql_view_project):
                # Empty search should match all records
                result1 = dbmanager.run_query(
                    table=TableName.PROJECT_RESOURCES,
                    like="",
                    year=2024
                )

                # Should return same results as None
                result2 = dbmanager.run_query(
                    table=TableName.PROJECT_RESOURCES,
                    like=None,
                    year=2024
                )

                assert result1 is not None
                assert result2 is not None
                assert len(result1) == len(result2)

    @pytest.mark.parametrize("sql_injection", [
        "'; DROP TABLE project_resources; --",
        "1' OR '1'='1",
        "project_name' UNION SELECT * FROM field_resources --",
        "'; UPDATE project_resources SET project_name='hacked'; --",
        "1'; EXEC xp_cmdshell('dir'); --"
    ])
    def test_search_with_sql_injection_attempts(self, sql_injection, mock_database, mock_sql_view_project):
        """Test that SQL injection attempts are safely handled."""
        tmp_path, db_path = mock_database

        with patch('esdc.dbmanager.Config.get_db_path', return_value=tmp_path):
            with patch('esdc.dbmanager._load_sql_script', return_value=mock_sql_view_project):
                # SQL injection attempts should either:
                # 1. Return no results (safely escaped)
                # 2. Raise an exception (blocked)
                # 3. Be treated as literal search strings

                try:
                    result = dbmanager.run_query(
                        table=TableName.PROJECT_RESOURCES,
                        like=sql_injection,
                        year=2024
                    )
                    # If successful, should return empty or be treated literally
                    assert result is not None
                    if len(result) > 0:
                        # Results should be legitimate, not hacked data
                        assert all("hacked" not in str(name) for name in result.get('project_name', []))
                except Exception as e:
                    # Should be a safe exception, not database corruption
                    assert "sql" in str(e).lower() or "syntax" in str(e).lower()

    @pytest.mark.parametrize("wildcard_pattern", [
        "%",
        "%%",
        "_",
        "__",
        "%Project%",
        "_as_",
        "G%s",
        "F___d"
    ])
    def test_search_with_wildcard_characters(self, wildcard_pattern, mock_database, mock_sql_view_project):
        """Test SQL wildcard characters in search patterns."""
        tmp_path, db_path = mock_database

        with patch('esdc.dbmanager.Config.get_db_path', return_value=tmp_path):
            with patch('esdc.dbmanager._load_sql_script', return_value=mock_sql_view_project):
                result = dbmanager.run_query(
                    table=TableName.PROJECT_RESOURCES,
                    like=wildcard_pattern,
                    year=2024
                )

                assert result is not None
                # Should handle wildcards according to SQL LIKE semantics

    @pytest.mark.parametrize("unicode_search", [
        "Mähakäm",  # German umlaut
        "Махакам",  # Cyrillic
        "まはかむ",  # Japanese hiragana
        "Mahakām",  # Macron
        "🛢️Project",  # Emoji
        "Mahakam\u200B",  # Zero-width space
    ])
    def test_search_with_unicode_characters(self, unicode_search, mock_database, mock_sql_view_project):
        """Test Unicode character handling in search patterns."""
        tmp_path, db_path = mock_database

        with patch('esdc.dbmanager.Config.get_db_path', return_value=tmp_path):
            with patch('esdc.dbmanager._load_sql_script', return_value=mock_sql_view_project):
                result = dbmanager.run_query(
                    table=TableName.PROJECT_RESOURCES,
                    like=unicode_search,
                    year=2024
                )

                # Should handle Unicode without crashes
                assert result is not None

    def test_search_very_long_strings(self, mock_database, mock_sql_view_project):
        """Test very long search strings."""
        tmp_path, db_path = mock_database

        long_strings = [
            "a" * 100,   # 100 chars
            "a" * 255,   # 255 chars (common limit)
            "a" * 1000,  # 1000 chars (very long)
            "a" * 10000, # 10K chars (extremely long)
        ]

        with patch('esdc.dbmanager.Config.get_db_path', return_value=tmp_path):
            with patch('esdc.dbmanager._load_sql_script', return_value=mock_sql_view_project):
                for long_string in long_strings:
                    result = dbmanager.run_query(
                        table=TableName.PROJECT_RESOURCES,
                        like=long_string,
                        year=2024
                    )

                    # Should handle long strings gracefully (likely no matches)
                    assert result is not None

    @pytest.mark.parametrize("special_char", [
        "'", '"', '`', '\\', '/',
        '[', ']', '(', ')', '{', '}',
        ';', ':', ',', '.', '!', '@', '#', '$', '%', '^', '&', '*', '+', '=', '|', '~', '?'
    ])
    def test_search_with_special_sql_chars(self, special_char, mock_database, mock_sql_view_project):
        """Test special SQL characters in search patterns."""
        tmp_path, db_path = mock_database

        with patch('esdc.dbmanager.Config.get_db_path', return_value=tmp_path):
            with patch('esdc.dbmanager._load_sql_script', return_value=mock_sql_view_project):
                result = dbmanager.run_query(
                    table=TableName.PROJECT_RESOURCES,
                    like=f"Project{special_char}Test",
                    year=2024
                )

                # Should handle special characters safely
                assert result is not None

    def test_search_pattern_case_sensitivity(self, mock_database, sample_project_data, mock_sql_view_project):
        """Test search pattern case sensitivity behavior."""
        tmp_path, db_path = mock_database

        # Add test data with different cases
        test_data = pd.concat([
            sample_project_data,
            pd.DataFrame({
                'report_year': [2024],
                'project_name': ['GAS PROJECT LOWERCASE'],
                'field_name': ['Field Gamma'],
                'wk_name': ['Blok C'],
                'project_stage': ['production'],
                'project_class': ['contingent'],
                'project_level': ['field'],
                'uncert_level': ['P50'],
                'rec_oc_risked': [100.0],
                'rec_an_risked': [500.0],
                'res_oc': [80.0],
                'res_an': [400.0],
                'prj_ioip': [1000.0],
                'prj_igip': [5000.0],
                'cprd_sls_oc': [20.0],
                'cprd_sls_an': [100.0],
            })
        ], ignore_index=True)

        with sqlite3.connect(db_path) as conn:
            test_data.to_sql('project_resources', conn, if_exists='replace', index=False)

        with patch('esdc.dbmanager.Config.get_db_path', return_value=tmp_path):
            with patch('esdc.dbmanager._load_sql_script', return_value=mock_sql_view_project):
                # Test case-sensitive matching
                result_lower = dbmanager.run_query(
                    table=TableName.PROJECT_RESOURCES,
                    like="gas",
                    year=2024
                )

                result_upper = dbmanager.run_query(
                    table=TableName.PROJECT_RESOURCES,
                    like="GAS",
                    year=2024
                )

                result_mixed = dbmanager.run_query(
                    table=TableName.PROJECT_RESOURCES,
                    like="Gas",
                    year=2024
                )

                assert result_lower is not None
                assert result_upper is not None
                assert result_mixed is not None

                # Document case sensitivity behavior (SQLite default is case-sensitive for ASCII, case-insensitive for Unicode)
```

#### 2.2 Complex Filter Combinations
```python
class TestComplexFiltering:
    """Test complex filter combinations and edge case scenarios."""

    def test_multiple_filters_combined(self, mock_database, mock_sql_view_project):
        """Test combining where, search, and year filters."""
        tmp_path, db_path = mock_database

        with patch('esdc.dbmanager.Config.get_db_path', return_value=tmp_path):
            with patch('esdc.dbmanager._load_sql_script', return_value=mock_sql_view_project):
                # Combine all three filters
                result = dbmanager.run_query(
                    table=TableName.PROJECT_RESOURCES,
                    where="field_name",
                    like="Alpha",
                    year=2024
                )

                assert result is not None
                if len(result) > 0:
                    # All records should match all filters
                    assert all("Alpha" in str(field) for field in result['field_name'])
                    assert all(year == 2024 for year in result['report_year'])

    def test_filter_override_behavior(self, mock_database, mock_sql_view_project):
        """Test filter parameter override and precedence."""
        tmp_path, db_path = mock_database

        with patch('esdc.dbmanager.Config.get_db_path', return_value=tmp_path):
            with patch('esdc.dbmanager._load_sql_script', return_value=mock_sql_view_project):
                # Custom where should override default
                result_custom = dbmanager.run_query(
                    table=TableName.PROJECT_RESOURCES,
                    where="field_name",  # Custom where column
                    like="Alpha",
                    year=2024
                )

                # Default where should use project_name
                result_default = dbmanager.run_query(
                    table=TableName.PROJECT_RESOURCES,
                    where=None,  # Use default (project_name)
                    like="Gas",
                    year=2024
                )

                assert result_custom is not None
                assert result_default is not None

    def test_filter_with_null_values(self, mock_database, mock_sql_view_project):
        """Test filtering when data contains null values."""
        tmp_path, db_path = mock_database

        # Add data with null values
        test_data_with_nulls = pd.DataFrame({
            'report_year': [2024, 2024, 2024],
            'project_name': ['Project A', None, 'Project C'],
            'field_name': ['Field X', 'Field Y', None],
            'wk_name': ['Blok A', None, 'Blok C'],
            'project_stage': ['production', 'development', None],
            'project_class': ['contingent', None, 'prospective'],
            'project_level': ['field', 'reservoir', 'field'],
            'uncert_level': ['P50', 'P90', None],
            'rec_oc_risked': [100.0, None, 150.0],
            'rec_an_risked': [500.0, 750.0, None],
            'res_oc': [80.0, 120.0, 100.0],
            'res_an': [400.0, None, 500.0],
            'prj_ioip': [1000.0, 2000.0, 1500.0],
            'prj_igip': [5000.0, None, 6500.0],
            'cprd_sls_oc': [20.0, 30.0, None],
            'cprd_sls_an': [100.0, None, 120.0],
        })

        with sqlite3.connect(db_path) as conn:
            test_data_with_nulls.to_sql('project_resources', conn, if_exists='replace', index=False)

        with patch('esdc.dbmanager.Config.get_db_path', return_value=tmp_path):
            with patch('esdc.dbmanager._load_sql_script', return_value=mock_sql_view_project):
                # Search should handle null values gracefully
                result = dbmanager.run_query(
                    table=TableName.PROJECT_RESOURCES,
                    where="project_name",
                    like="Project",
                    year=2024
                )

                assert result is not None
                # Should not crash on null values
                if len(result) > 0:
                    assert all(pd.notna(result['project_name']))

    def test_filter_with_empty_result_sets(self, mock_database, mock_sql_view_project):
        """Test filtering that returns no results."""
        tmp_path, db_path = mock_database

        with patch('esdc.dbmanager.Config.get_db_path', return_value=tmp_path):
            with patch('esdc.dbmanager._load_sql_script', return_value=mock_sql_view_project):
                # Search for non-existent data
                result = dbmanager.run_query(
                    table=TableName.PROJECT_RESOURCES,
                    where="project_name",
                    like="NonExistentProject12345",
                    year=2024
                )

                # Should return empty DataFrame, not None
                assert result is not None
                assert len(result) == 0
                assert isinstance(result, pd.DataFrame)

    def test_cross_table_filter_consistency(self, mock_database, mock_sql_view_project, mock_sql_view_field):
        """Test that similar filters work consistently across different tables."""
        tmp_path, db_path = mock_database

        with patch('esdc.dbmanager.Config.get_db_path', return_value=tmp_path):
            # Test project_resources with field_name filter
            with patch('esdc.dbmanager._load_sql_script', return_value=mock_sql_view_project):
                project_result = dbmanager.run_query(
                    table=TableName.PROJECT_RESOURCES,
                    where="field_name",
                    like="Alpha",
                    year=2024
                )

            # Test field_resources with field_name filter
            with patch('esdc.dbmanager._load_sql_script', return_value=mock_sql_view_field):
                field_result = dbmanager.run_query(
                    table=TableName.FIELD_RESOURCES,
                    where="field_name",
                    like="Alpha",
                    year=2024
                )

            # Both should work without errors
            assert project_result is not None
            assert field_result is not None
```

### 3. Output Level Enhancement Tests

#### 3.1 Detail Level Verification
```python
class TestDetailLevelVerification:
    def test_level_0_minimal_columns_exact()
    def test_level_1_standard_columns_exact()
    def test_level_2_extended_columns_exact()
    def test_level_3_comprehensive_columns_exact()
    def test_level_4_all_columns_verification()
    def test_column_count_by_level()
    def test_required_columns_present_by_level()
```

#### 3.2 Column Selection Override
```python
class TestColumnSelectionOverride:
    def test_columns_overrides_output_level()
    def test_invalid_column_names_handling()
    def test_duplicate_column_names()
    def test_column_order_preservation()
    def test_column_selection_with_spaces()
    def test_column_selection_empty_list()
```

### 4. Data Integrity and Formatting Tests

#### 4.1 Numeric Precision and Formatting
```python
class TestNumericFormatting:
    def test_large_number_formatting_trillions()
    def test_small_decimal_formatting()
    def test_negative_value_formatting()
    def test_zero_value_display()
    def test_scientific_notation_prevention()
    def test_currency_format_consistency()
    def test_decimal_places_enforcement()
```

#### 4.2 Data Type Preservation
```python
class TestDataTypePreservation:
    def test_integer_columns_remain_integer()
    def test_float_columns_precision_preserved()
    def test_string_columns_unicode_handling()
    def test_boolean_columns_display()
    def test_date_columns_formatting()
    def test_null_value_display_consistency()
```

### 5. Excel Export Enhancement Tests

#### 5.1 Excel File Validation
```python
class TestExcelFileValidation:
    def test_excel_file_creation_permissions()
    def test_excel_file_overwrite_behavior()
    def test_excel_sheet_name_validation()
    def test_excel_column_width_auto_adjustment()
    def test_excel_numeric_format_preservation()
    def test_excel_large_dataset_handling()
    def test_excel_corrupted_file_handling()
```

#### 5.2 Excel Content Verification
```python
class TestExcelContentVerification:
    def test_excel_data_matches_console_output()
    def test_excel_column_headers_preservation()
    def test_excel_data_type_preservation()
    def test_excel_index_column_absence()
    def test_excel_worksheet_naming_convention()
    def test_excel_filename_timestamp_format()
```

### 6. Error Handling and Recovery Tests

#### 6.1 Database Error Scenarios
```python
class TestDatabaseErrorScenarios:
    def test_database_corrupted_file()
    def test_database_permission_denied()
    def test_database_locked_by_another_process()
    def test_database_disk_full()
    def test_database_connection_timeout()
    def test_database_schema_mismatch()
```

#### 6.2 System Resource Errors
```python
class TestSystemResourceErrors:
    def test_memory_insufficient_large_datasets()
    def test_disk_space_insufficient_excel_export()
    def test_file_system_permission_errors()
    def test_concurrent_file_access_conflicts()
    def test_network_drive_database_access()
```

### 7. Performance and Load Tests

#### 7.1 Large Dataset Handling
```python
class TestLargeDatasetHandling:
    """Test performance and resource usage with large datasets."""

    @pytest.fixture
    def large_dataset_10k(self):
        """Generate 10,000 project records for performance testing."""
        import random
        from datetime import datetime

        projects = []
        field_names = [f"Field {chr(65+i)}{j}" for i in range(26) for j in range(1, 10)]
        wk_names = [f"Blok {chr(65+i)}" for i in range(26)]
        stages = ["discovery", "appraisal", "development", "production"]
        classes = ["contingent", "prospective"]
        levels = ["reservoir", "field", "prospect"]
        uncertainties = ["P90", "P50", "P10", "U"]

        for i in range(10000):
            projects.append({
                'report_year': random.choice([2022, 2023, 2024]),
                'project_name': f"Project {i:04d}",
                'field_name': random.choice(field_names),
                'wk_name': random.choice(wk_names),
                'project_stage': random.choice(stages),
                'project_class': random.choice(classes),
                'project_level': random.choice(levels),
                'uncert_level': random.choice(uncertainties),
                'rec_oc_risked': random.uniform(50.0, 5000.0),
                'rec_an_risked': random.uniform(200.0, 20000.0),
                'res_oc': random.uniform(40.0, 4000.0),
                'res_an': random.uniform(150.0, 15000.0),
                'prj_ioip': random.uniform(500.0, 10000.0),
                'prj_igip': random.uniform(2500.0, 50000.0),
                'cprd_sls_oc': random.uniform(10.0, 1000.0),
                'cprd_sls_an': random.uniform(50.0, 5000.0),
            })

        return pd.DataFrame(projects)

    def test_query_performance_10k_records(self, tmp_path, large_dataset_10k):
        """Test query performance with 10,000 records."""
        db_path = tmp_path / "large_test.db"

        # Load large dataset
        with sqlite3.connect(db_path) as conn:
            large_dataset_10k.to_sql('project_resources', conn, if_exists='replace', index=False)

        import time
        start_time = time.time()

        with patch('esdc.dbmanager.Config.get_db_path', return_value=tmp_path):
            with patch('esdc.dbmanager._load_sql_script') as mock_load_sql:
                # Simple query for performance testing
                mock_load_sql.return_value = """
                SELECT report_year, project_name, field_name,
                    rec_oc_risked as 'resources_mstb', res_oc as 'reserves_mstb'
                FROM project_resources
                WHERE report_year = <year>
                ORDER BY project_name;
                """

                result = dbmanager.run_query(
                    table=TableName.PROJECT_RESOURCES,
                    year=2024,
                    output=0
                )

        end_time = time.time()
        query_time = end_time - start_time

        # Performance assertions
        assert result is not None
        assert len(result) > 0
        assert query_time < 5.0, f"Query took {query_time:.2f}s, should be < 5.0s"

    @pytest.mark.slow
    def test_memory_usage_large_result_sets(self, tmp_path, large_dataset_10k):
        """Test memory usage with large result sets."""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        db_path = tmp_path / "memory_test.db"
        with sqlite3.connect(db_path) as conn:
            large_dataset_10k.to_sql('project_resources', conn, if_exists='replace', index=False)

        with patch('esdc.dbmanager.Config.get_db_path', return_value=tmp_path):
            with patch('esdc.dbmanager._load_sql_script') as mock_load_sql:
                # Query that returns all data
                mock_load_sql.return_value = "SELECT * FROM project_resources ORDER BY project_name;"

                result = dbmanager.run_query(
                    table=TableName.PROJECT_RESOURCES,
                    output=4  # All columns
                )

        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = peak_memory - initial_memory

        # Memory usage assertions
        assert result is not None
        assert len(result) == 10000
        assert memory_increase < 500, f"Memory increased by {memory_increase:.1f}MB, should be < 500MB"

    def test_console_output_large_tables(self, tmp_path, large_dataset_10k):
        """Test console output performance with large tables."""
        db_path = tmp_path / "output_test.db"
        with sqlite3.connect(db_path) as conn:
            large_dataset_10k.to_sql('project_resources', conn, if_exists='replace', index=False)

        import time
        from io import StringIO
        import sys

        with patch('esdc.dbmanager.Config.get_db_path', return_value=tmp_path):
            with patch('esdc.dbmanager._load_sql_script') as mock_load_sql:
                mock_load_sql.return_value = "SELECT * FROM project_resources LIMIT 1000;"

                # Capture output and measure time
                start_time = time.time()
                old_stdout = sys.stdout
                sys.stdout = captured_output = StringIO()

                try:
                    result = dbmanager.run_query(
                        table=TableName.PROJECT_RESOURCES,
                        output=4
                    )

                    if result is not None and len(result) > 0:
                        # Simulate tabulate formatting
                        formatted_table = str(result.head(10))  # Just test formatting logic
                finally:
                    sys.stdout = old_stdout

                end_time = time.time()
                output_time = end_time - start_time

        assert output_time < 3.0, f"Output formatting took {output_time:.2f}s, should be < 3.0s"

    def test_excel_export_large_datasets(self, tmp_path, large_dataset_10k):
        """Test Excel export performance with large datasets."""
        db_path = tmp_path / "excel_test.db"
        with sqlite3.connect(db_path) as conn:
            large_dataset_10k.to_sql('project_resources', conn, if_exists='replace', index=False)

        import time
        excel_path = tmp_path / "large_export.xlsx"

        with patch('esdc.dbmanager.Config.get_db_path', return_value=tmp_path):
            with patch('esdc.dbmanager._load_sql_script') as mock_load_sql:
                mock_load_sql.return_value = "SELECT * FROM project_resources LIMIT 5000;"

                result = dbmanager.run_query(
                    table=TableName.PROJECT_RESOURCES,
                    output=4
                )

                if result is not None and len(result) > 0:
                    start_time = time.time()
                    result.to_excel(excel_path, index=False, sheet_name="resources report")
                    end_time = time.time()

                    export_time = end_time - start_time

                    # Performance assertions
                    assert excel_path.exists()
                    assert export_time < 30.0, f"Excel export took {export_time:.2f}s, should be < 30.0s"

                    # Verify file size is reasonable
                    file_size_mb = excel_path.stat().st_size / 1024 / 1024
                    assert file_size_mb < 50, f"Excel file is {file_size_mb:.1f}MB, should be < 50MB"

    def test_filtering_performance_large_data(self, tmp_path, large_dataset_10k):
        """Test filtering performance on large datasets."""
        db_path = tmp_path / "filter_test.db"
        with sqlite3.connect(db_path) as conn:
            large_dataset_10k.to_sql('project_resources', conn, if_exists='replace', index=False)

        import time

        with patch('esdc.dbmanager.Config.get_db_path', return_value=tmp_path):
            with patch('esdc.dbmanager._load_sql_script') as mock_load_sql:
                mock_load_sql.return_value = """
                SELECT * FROM project_resources
                WHERE <where> LIKE '%<like>%' AND report_year = <year>
                ORDER BY project_name;
                """

                # Test various filter combinations
                filter_tests = [
                    {"where": "project_name", "like": "Project 1"},
                    {"where": "field_name", "like": "Field A"},
                    {"where": "wk_name", "like": "Blok A"},
                ]

                for filter_test in filter_tests:
                    start_time = time.time()
                    result = dbmanager.run_query(
                        table=TableName.PROJECT_RESOURCES,
                        where=filter_test["where"],
                        like=filter_test["like"],
                        year=2024
                    )
                    end_time = time.time()

                    filter_time = end_time - start_time

                    # Each filter should be fast
                    assert result is not None
                    assert filter_time < 2.0, f"Filter {filter_test} took {filter_time:.2f}s, should be < 2.0s"
```

#### 7.2 Query Optimization
```python
class TestQueryOptimization:
    def test_index_usage_effectiveness()
    def test_query_plan_optimization()
    def test_cached_query_performance()
    def test_complex_filter_performance()
    def test_sorting_performance_large_sets()
```

### 8. Integration and Workflow Tests

#### 8.1 End-to-End Workflows
```python
class TestEndToEndWorkflows:
    def test_complete_workflow_fetch_reload_show()
    def test_multi_user_concurrent_access()
    def test_database_upgrade_compatibility()
    def test_configuration_change_impact()
    def test_cross_command_data_consistency()
```

#### 8.2 CLI Integration Tests
```python
class TestCLIIntegration:
    def test_command_line_argument_quoting()
    def test_shell_script_integration()
    def test_pipe_redirection_compatibility()
    def test_background_process_execution()
    def test_environment_variable_integration()
```

### 9. Cross-Platform Compatibility Tests

#### 9.1 Operating System Specific Tests
```python
class TestCrossPlatformCompatibility:
    def test_windows_path_handling()
    def test_unix_path_handling()
    def test_macos_path_handling()
    def test_file_permission_differences()
    def test_case_sensitive_filesystems()
    def test_network_path_handling()
```

#### 9.2 Locale and Internationalization
```python
class TestLocaleSupport:
    def test_different_decimal_separators()
    def test_different_date_formats()
    def test_unicode_filename_handling()
    def test_multilingual_content_display()
    def test_timezone_aware_timestamps()
```

### 10. Regression Test Suite

#### 10.1 Known Bug Fixes
```python
class TestRegressionBugs:
    def test_column_selection_sql_syntax_bug()  # Current xfail
    def test_year_filter_special_handling_nkri()
    def test_placeholder_replacement_edge_cases()
    def test_output_level_boundary_conditions()
    def test_excel_export_filename_conflicts()
```

#### 10.2 Security Tests
```python
class TestSecurityScenarios:
    def test_sql_injection_prevention()
    def test_path_traversal_prevention()
    def test_command_injection_prevention()
    def test_file_system_access_limits()
    def test_database_privilege_escalation()
```

---

## Test Implementation Strategy

### Phase 1: Critical Gap Coverage (Week 1-2)
1. **Fix existing xfail tests** - Column selection SQL syntax error
2. **Add missing edge case tests** - Boundary conditions, special characters
3. **Enhance error handling tests** - Database corruption, permissions
4. **Add data integrity tests** - Numeric precision, type preservation

### Phase 2: Performance and Scale (Week 3-4)
1. **Large dataset tests** - 10K+ record performance
2. **Memory usage validation** - Resource consumption limits
3. **Query optimization tests** - Index usage, caching
4. **Excel export scalability** - Large file handling

### Phase 3: Integration and Robustness (Week 5-6)
1. **Cross-platform compatibility** - Windows, Unix, macOS
2. **Concurrent access scenarios** - Multi-user testing
3. **End-to-end workflow tests** - Complete data pipeline
4. **Security vulnerability tests** - Injection prevention

### Phase 4: Advanced Features (Week 7-8)
1. **Locale and internationalization** - Different formats
2. **Configuration integration** - Settings impact
3. **CLI toolchain integration** - Shell compatibility
4. **Performance benchmarking** - Baseline establishment

---

## Test Infrastructure Enhancements

### Required Fixtures and Utilities

#### 1. Large Dataset Generator
```python
@pytest.fixture
def large_project_dataset():
    """Generate large dataset for performance testing."""
    import random
    import string

    def generate_projects(num_records):
        """Generate realistic project data."""
        field_names = [f"Field {chr(65+i)}{j}" for i in range(26) for j in range(1, 20)]
        wk_names = [f"Blok {chr(65+i)}{j}" for i in range(26) for j in range(1, 5)]
        stages = ["discovery", "appraisal", "development", "production"]
        classes = ["contingent", "prospective"]
        levels = ["reservoir", "field", "prospect"]
        uncertainties = ["P90", "P50", "P10", "U"]

        projects = []
        for i in range(num_records):
            base_name = ''.join(random.choices(string.ascii_uppercase, k=3))
            projects.append({
                'report_year': random.choice([2019, 2020, 2021, 2022, 2023, 2024]),
                'project_name': f"{base_name}-{i:04d}",
                'field_name': random.choice(field_names),
                'wk_name': random.choice(wk_names),
                'project_stage': random.choice(stages),
                'project_class': random.choice(classes),
                'project_level': random.choice(levels),
                'uncert_level': random.choice(uncertainties),
                'rec_oc_risked': random.lognormvariate(5, 1.5),  # Log-normal distribution
                'rec_an_risked': random.lognormvariate(6, 1.5),
                'res_oc': random.lognormvariate(4.5, 1.5),
                'res_an': random.lognormvariate(5.5, 1.5),
                'prj_ioip': random.lognormvariate(6, 1.2),
                'prj_igip': random.lognormvariate(7, 1.2),
                'cprd_sls_oc': random.lognormvariate(3, 1.8),
                'cprd_sls_an': random.lognormvariate(4, 1.8),
            })
        return pd.DataFrame(projects)

    return generate_projects

@pytest.fixture
def edge_case_data():
    """Generate data with edge cases for testing."""
    return pd.DataFrame({
        'report_year': [2024, 2024, 2024, 2024, 2024],
        'project_name': [
            'Mähakäm Field',  # Unicode
            '🛢️ Oil Project',  # Emoji
            'Project' * 50,  # Very long name
            'Normal Project',  # Normal case
            ''  # Empty string
        ],
        'field_name': [
            'Field-Alpha',  # Hyphen
            '123 Numeric',  # Numbers
            None,  # Null value
            'Field with spaces',  # Multiple spaces
            'Field_Beta'  # Underscore
        ],
        'wk_name': ['Blok A', 'Blok B', None, 'Blok D', 'Blok E'],
        'project_stage': ['production', 'development', None, 'production', 'appraisal'],
        'project_class': ['contingent', None, 'prospective', 'contingent', 'contingent'],
        'project_level': ['field', 'reservoir', 'field', None, 'prospect'],
        'uncert_level': ['P50', 'P90', 'P10', 'U', None],
        # Very large numbers
        'rec_oc_risked': [1.5e12, 2.8e11, 9.9e15, 0.0, -1.0],  # Trillions, negative
        'rec_an_risked': [5.2e12, 8.1e11, 3.3e16, 0.0, -5.0],
        'res_oc': [1.2e12, 2.1e11, 8.8e15, 0.0, 0.0],
        'res_an': [4.8e12, 7.2e11, 2.9e16, 0.0, 0.0],
        'prj_ioip': [9.9e12, 1.8e12, 6.6e15, 0.0, 100.0],
        'prj_igip': [4.5e13, 8.2e12, 3.1e16, 0.0, 500.0],
        'cprd_sls_oc': [1e9, 2e8, 7.5e11, 0.0, 10.0],
        'cprd_sls_an': [5e9, 1e9, 3.8e12, 0.0, 50.0],
    })

@pytest.fixture
def corrupted_database(tmp_path):
    """Create database with various corruption scenarios."""
    def create_corrupted_db(corruption_type="truncated"):
        db_path = tmp_path / f"corrupted_{corruption_type}.db"

        if corruption_type == "truncated":
            # Create normal database then truncate it
            with sqlite3.connect(db_path) as conn:
                conn.execute("CREATE TABLE project_resources (id INTEGER)")
                conn.execute("INSERT INTO project_resources VALUES (1)")

            # Truncate the file
            with open(db_path, 'r+b') as f:
                f.seek(100)
                f.truncate()

        elif corruption_type == "invalid_header":
            # Create file with invalid SQLite header
            with open(db_path, 'wb') as f:
                f.write(b'INVALID_SQLITE_DB_HEADER')

        elif corruption_type == "permission_denied":
            # Create file then make it read-only
            with sqlite3.connect(db_path) as conn:
                conn.execute("CREATE TABLE project_resources (id INTEGER)")
            db_path.chmod(0o444)  # Read-only

        elif corruption_type == "missing_tables":
            # Create database but no tables
            with sqlite3.connect(db_path) as conn:
                pass  # Empty database

        return db_path

    return create_corrupted_db

@pytest.fixture
def performance_monitor():
    """Monitor performance metrics during test execution."""
    import time
    import psutil
    import os

    class PerformanceMonitor:
        def __init__(self):
            self.process = psutil.Process(os.getpid())
            self.start_time = None
            self.start_memory = None
            self.measurements = []

        def start(self):
            """Start monitoring."""
            self.start_time = time.time()
            self.start_memory = self.process.memory_info().rss / 1024 / 1024
            self.measurements = []

        def measure(self, label=""):
            """Record current performance metrics."""
            current_time = time.time()
            current_memory = self.process.memory_info().rss / 1024 / 1024

            if self.start_time:
                elapsed = current_time - self.start_time
                memory_delta = current_memory - self.start_memory

                self.measurements.append({
                    'label': label,
                    'time': elapsed,
                    'memory_mb': current_memory,
                    'memory_delta_mb': memory_delta
                })

        def get_peak_memory(self):
            """Get peak memory usage."""
            return max(m['memory_mb'] for m in self.measurements) if self.measurements else 0

        def get_total_time(self):
            """Get total elapsed time."""
            if self.measurements:
                return self.measurements[-1]['time']
            return 0

    return PerformanceMonitor()

@pytest.fixture
def cross_platform_paths(tmp_path):
    """Generate cross-platform path scenarios for testing."""
    import platform

    class CrossPlatformPaths:
        def __init__(self, base_path):
            self.base_path = base_path
            self.os_name = platform.system().lower()

        def get_test_paths(self):
            """Get various path scenarios for testing."""
            scenarios = {}

            # Normal path
            scenarios['normal'] = self.base_path / "test.db"

            # Path with spaces
            scenarios['spaces'] = self.base_path / "test path" / "test.db"

            # Path with unicode characters
            scenarios['unicode'] = self.base_path / "тестовая" / "test.db"

            # Very long path
            long_name = "a" * 100
            scenarios['long'] = self.base_path / long_name / "test.db"

            # Path with special characters (OS-specific)
            if self.os_name == 'windows':
                scenarios['special'] = self.base_path / "test-#1.db"
            else:
                scenarios['special'] = self.base_path / "test@#1.db"

            return scenarios

        def create_directories(self):
            """Create directories for path testing."""
            scenarios = self.get_test_paths()
            for path in scenarios.values():
                path.parent.mkdir(parents=True, exist_ok=True)
            return scenarios

    return CrossPlatformPaths(tmp_path)

@pytest.fixture
def concurrent_database(tmp_path, sample_project_data):
    """Setup for testing concurrent database access."""
    import threading
    import queue

    db_path = tmp_path / "concurrent_test.db"
    with sqlite3.connect(db_path) as conn:
        sample_project_data.to_sql('project_resources', conn, if_exists='replace', index=False)

    class ConcurrentTestHelper:
        def __init__(self, db_path):
            self.db_path = db_path
            self.results = queue.Queue()
            self.errors = queue.Queue()

        def worker_query(self, thread_id, query_params):
            """Worker function for concurrent queries."""
            try:
                with patch('esdc.dbmanager.Config.get_db_path', return_value=self.db_path.parent):
                    with patch('esdc.dbmanager._load_sql_script') as mock_load_sql:
                        mock_load_sql.return_value = "SELECT * FROM project_resources WHERE report_year = 2024;"

                        result = dbmanager.run_query(
                            table=TableName.PROJECT_RESOURCES,
                            **query_params
                        )

                        self.results.put((thread_id, result))
            except Exception as e:
                self.errors.put((thread_id, e))

        def run_concurrent_queries(self, num_threads=5, query_params=None):
            """Run multiple concurrent queries."""
            if query_params is None:
                query_params = {'year': 2024, 'output': 0}

            threads = []
            for i in range(num_threads):
                thread = threading.Thread(
                    target=self.worker_query,
                    args=(i, query_params)
                )
                threads.append(thread)
                thread.start()

            # Wait for all threads to complete
            for thread in threads:
                thread.join(timeout=10)  # 10 second timeout

            # Collect results
            results = []
            errors = []

            while not self.results.empty():
                results.append(self.results.get())

            while not self.errors.empty():
                errors.append(self.errors.get())

            return results, errors

    return ConcurrentTestHelper(db_path)
```

### Test Data Enhancements

#### 1. Edge Case Data Fixtures
```python
@pytest.fixture
def numerical_edge_cases():
    """Data with extreme numerical values."""
    return pd.DataFrame({
        'report_year': [2024] * 8,
        'project_name': [f'Extreme Case {i}' for i in range(8)],
        'field_name': ['Test Field'] * 8,
        'wk_name': ['Test Block'] * 8,
        'project_stage': ['production'] * 8,
        'project_class': ['contingent'] * 8,
        'project_level': ['field'] * 8,
        'uncert_level': ['P50'] * 8,
        # Extreme numerical values
        'rec_oc_risked': [
            0.0,  # Zero
            1e-10,  # Very small
            999999999999.99,  # Just under trillion
            1e15,  # Quadrillion
            -1.0,  # Negative
            float('inf'),  # Infinity (if allowed)
            float('nan'),  # NaN (if allowed)
            1.23456789012345  # High precision
        ],
        'rec_an_risked': [v * 5 for v in [
            0.0, 1e-10, 999999999999.99, 1e15, -1.0, float('inf'), float('nan'), 1.23456789012345
        ]],
        'res_oc': [v * 0.8 for v in [
            0.0, 1e-10, 999999999999.99, 1e15, -1.0, float('inf'), float('nan'), 1.23456789012345
        ]],
        'res_an': [v * 4 for v in [
            0.0, 1e-10, 999999999999.99, 1e15, -1.0, float('inf'), float('nan'), 1.23456789012345
        ]],
        'prj_ioip': [1000.0] * 8,
        'prj_igip': [5000.0] * 8,
        'cprd_sls_oc': [100.0] * 8,
        'cprd_sls_an': [500.0] * 8,
    })

@pytest.fixture
def sql_injection_test_data():
    """Data that might trigger SQL injection if not properly handled."""
    return pd.DataFrame({
        'report_year': [2024] * 6,
        'project_name': [
            "'; DROP TABLE project_resources; --",
            "Project' OR '1'='1",
            "Robert'); DROP TABLE project_resources; --",
            "Project' UNION SELECT * FROM field_resources --",
            "'; UPDATE project_resources SET project_name='HACKED'; --",
            "Normal Project"
        ],
        'field_name': ['Test Field'] * 6,
        'wk_name': ['Test Block'] * 6,
        'project_stage': ['production'] * 6,
        'project_class': ['contingent'] * 6,
        'project_level': ['field'] * 6,
        'uncert_level': ['P50'] * 6,
        'rec_oc_risked': [100.0] * 6,
        'rec_an_risked': [500.0] * 6,
        'res_oc': [80.0] * 6,
        'res_an': [400.0] * 6,
        'prj_ioip': [1000.0] * 6,
        'prj_igip': [5000.0] * 6,
        'cprd_sls_oc': [20.0] * 6,
        'cprd_sls_an': [100.0] * 6,
    })
```

#### 2. Performance Test Data
- Scalable dataset generator with configurable size (implemented above)
- Realistic data distribution patterns using log-normal distributions
- Various query selectivity scenarios (different filter combinations)

#### 3. Error Scenario Data
- Database files with specific corruption patterns (truncated, invalid header, missing tables)
- Permission-restricted file scenarios (read-only files)
- Network-based database configurations (path testing)
- Concurrent access scenarios (threading fixture)

---

## Success Metrics

### Coverage Targets
- **Line Coverage**: >95% for esdc.py and dbmanager.py
- **Branch Coverage**: >90% for all conditional logic
- **Function Coverage**: 100% for all public functions

### Performance Benchmarks
- **Query Time**: <2 seconds for 10K records
- **Memory Usage**: <100MB for typical operations
- **Excel Export**: <30 seconds for 50K records

### Quality Gates
- **All tests pass** on Windows, macOS, and Linux
- **No regression failures** in existing functionality
- **Performance no degradation** from baseline

---

## Test Execution Plan

### Continuous Integration
1. **Unit Tests**: Every commit (fast feedback)
2. **Integration Tests**: Every pull request
3. **Performance Tests**: Weekly builds
4. **Cross-Platform Tests**: Release candidates

### Test Categories by Execution Frequency
- **Smoke Tests**: Every commit (basic functionality)
- **Regression Tests**: Every PR (bug fix validation)
- **Performance Tests**: Nightly (benchmark tracking)
- **Full Suite**: Pre-release (comprehensive validation)

---

## Dependencies and Requirements

### Additional Test Dependencies
```toml
# Add to pyproject.toml [test-dependencies] or requirements-dev.txt
[project.optional-dependencies]
test = [
    "pytest >= 7.0.0",
    "pytest-cov >= 4.0.0",      # Coverage reporting
    "pytest-xdist >= 3.0.0",    # Parallel test execution
    "pytest-mock >= 3.10.0",    # Enhanced mocking
    "pytest-benchmark >= 4.0.0", # Performance testing
    "pytest-slow >= 0.3.0",     # Slow test marker
    "memory-profiler >= 0.60.0", # Memory usage tracking
    "psutil >= 5.9.0",           # System resource monitoring
    "hypothesis >= 6.0.0",       # Property-based testing
]

# Or for requirements-dev.txt:
pytest>=7.0.0
pytest-cov>=4.0.0
pytest-xdist>=3.0.0
pytest-mock>=3.10.0
pytest-benchmark>=4.0.0
pytest-slow>=0.3.0
memory-profiler>=0.60.0
psutil>=5.9.0
hypothesis>=6.0.0
```

### Test Configuration Files

#### pytest.ini
```ini
[tool:pytest]
minversion = 7.0
addopts =
    --strict-markers
    --strict-config
    --verbose
    --tb=short
    --cov=esdc
    --cov-report=html
    --cov-report=term-missing
    --cov-fail-under=85
markers =
    slow: marks tests as slow (deselect with '-m "not slow"')
    integration: marks tests as integration tests
    performance: marks tests as performance tests
    unit: marks tests as unit tests
    security: marks tests as security-related
    regression: marks tests as regression tests
testpaths = tests
python_files = test_*.py *_test.py
python_classes = Test*
python_functions = test_*
```

#### pyproject.toml (Complete)
```toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "esdc"
version = "0.1.0"
description = "ESDC (Electronic Submission of Data Center) Python module"
authors = [{name = "ESDC Team"}]
license = {text = "MIT"}
readme = "README.md"
requires-python = ">=3.8"
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
]
dependencies = [
    "typer>=0.9.0",
    "pandas>=2.0.0",
    "tabulate>=0.9.0",
    "rich>=13.0.0",
    "sqlite3",
    "platformdirs>=3.0.0",
    "openpyxl>=3.1.0",
    "requests>=2.28.0",
]

[project.optional-dependencies]
test = [
    "pytest>=7.0.0",
    "pytest-cov>=4.0.0",
    "pytest-xdist>=3.0.0",
    "pytest-mock>=3.10.0",
    "pytest-benchmark>=4.0.0",
    "pytest-slow>=0.3.0",
    "memory-profiler>=0.60.0",
    "psutil>=5.9.0",
    "hypothesis>=6.0.0",
]
dev = [
    "esdc[test]",
    "black>=23.0.0",
    "isort>=5.12.0",
    "mypy>=1.0.0",
    "pylint>=2.17.0",
    "pre-commit>=3.0.0",
]

[project.scripts]
esdc = "esdc.esdc:app"

[tool.setuptools.packages.find]
where = ["."]
include = ["esdc*"]

[tool.black]
line-length = 88
target-version = ['py38']
include = '\.pyi?$'

[tool.isort]
profile = "black"
multi_line_output = 3

[tool.mypy]
python_version = "3.8"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true

[tool.pytest.ini_options]
minversion = "7.0"
addopts = [
    "--strict-markers",
    "--strict-config",
    "--verbose",
    "--tb=short",
    "--cov=esdc",
    "--cov-report=html",
    "--cov-report=term-missing",
    "--cov-fail-under=85",
]
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    "integration: marks tests as integration tests",
    "performance: marks tests as performance tests",
    "unit: marks tests as unit tests",
    "security: marks tests as security-related",
    "regression: marks tests as regression tests",
]
testpaths = ["tests"]
```

### GitHub Actions Workflow
```yaml
# .github/workflows/test.yml
name: Tests

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]

jobs:
  test:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, windows-latest, macos-latest]
        python-version: ["3.8", "3.9", "3.10", "3.11"]

    steps:
    - uses: actions/checkout@v4

    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e .[test,dev]

    - name: Lint with pylint
      run: |
        pylint esdc --fail-under=8.0

    - name: Type check with mypy
      run: |
        mypy esdc

    - name: Run unit tests
      run: |
        pytest tests/unit -m "not slow and not performance" -v

    - name: Run integration tests
      run: |
        pytest tests/integration -m "not slow and not performance" -v

    - name: Run performance tests (Ubuntu only)
      if: matrix.os == 'ubuntu-latest' && matrix.python-version == '3.10'
      run: |
        pytest -m "performance" --benchmark-only --benchmark-json=benchmark.json

    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        flags: unittests
        name: codecov-umbrella
```

### Test Environment Setup

#### Local Development Setup Script
```bash
#!/bin/bash
# setup_test_env.sh

echo "Setting up ESDC test environment..."

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -e .[test,dev]

# Setup pre-commit hooks
pre-commit install

# Run initial test to verify setup
pytest tests/unit/test_show_command.py::TestShowCommandArguments::test_valid_table_names_lowercase -v

echo "Test environment setup complete!"
echo "Run 'pytest' to run all tests"
echo "Run 'pytest -m \"not slow\"' to run fast tests only"
echo "Run 'pytest -m \"performance\"' to run performance tests"
```

#### Multiple Python Versions Testing
```bash
# Using pyenv for multiple Python versions
pyenv install 3.8.18
pyenv install 3.9.18
pyenv install 3.10.13
pyenv install 3.11.6

# Test against all versions
pyenv local 3.8.18 3.9.18 3.10.13 3.11.6
tox
```

#### tox.ini for Cross-Version Testing
```ini
[tox]
envlist = py38,py39,py310,py311,flake8,mypy
isolated_build = true

[testenv]
deps =
    pytest>=7.0.0
    pytest-cov>=4.0.0
    pytest-mock>=3.10.0
    memory-profiler>=0.60.0
    psutil>=5.9.0
commands = pytest {posargs}

[testenv:performance]
deps =
    pytest>=7.0.0
    pytest-benchmark>=4.0.0
    psutil>=5.9.0
commands = pytest -m "performance" --benchmark-only

[testenv:flake8]
deps = flake8
commands = flake8 esdc tests

[testenv:mypy]
deps = mypy
commands = mypy esdc
```

### Database Configurations
- **SQLite**: Default local database (already implemented)
- **In-memory SQLite**: For fast unit tests
- **Network-based storage**: For integration testing (path testing)
- **Cross-platform paths**: Windows UNC paths, Unix symbolic links, macOS aliases

---

## Conclusion

This comprehensive test plan addresses all identified gaps in the current test suite and provides robust coverage for the `esdc show` command functionality. The phased implementation approach ensures systematic improvement while maintaining existing functionality.

**Key Benefits:**
- **Complete coverage** of all CLI options and edge cases
- **Performance validation** for large datasets
- **Cross-platform compatibility** assurance
- **Regression prevention** through comprehensive testing
- **Quality assurance** for production deployments

The plan establishes a foundation for maintaining high code quality and reliability as the ESDC project continues to evolve.