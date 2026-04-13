import duckdb
import pandas as pd
import pytest

from esdc.dbmanager import (
    _load_sql_script,
    invalidate_sql_cache,
    load_data_to_db,
    run_query,
)
from esdc.selection import TableName


class TestLoadSqlScript:
    """Tests for _load_sql_script()."""

    def test_load_sql_script_exists(self):
        """Test loading an existing SQL script."""
        script = _load_sql_script("create_table_project_resources.sql")
        assert script is not None
        assert "CREATE TABLE" in script.upper()

    def test_load_sql_script_view(self):
        """Test loading a view SQL script."""
        script = _load_sql_script("create_esdc_view.sql")
        assert script is not None
        assert "SELECT" in script.upper()

    def test_load_sql_script_not_found(self):
        """Test loading non-existent script raises error."""
        with pytest.raises(FileNotFoundError):
            _load_sql_script("nonexistent.sql")


class TestRunQuery:
    """Tests for run_query()."""

    def test_run_query_no_database(self, tmp_path, mocker):
        """Test run_query returns None when database doesn't exist."""
        db_file = tmp_path / "nonexistent.db"
        mocker.patch("esdc.dbmanager.Config.get_db_file", return_value=db_file)
        result = run_query(TableName.PROJECT_RESOURCES)
        assert result is None

    def test_run_query_with_filter(self, tmp_path, mocker):
        """Test run_query with where and like filters."""
        mock_df = pd.DataFrame({"id": [1], "name": ["test"]})

        db_file = tmp_path / "test.db"
        mocker.patch("esdc.dbmanager.Config.get_db_file", return_value=db_file)
        mocker.patch("pathlib.Path.exists", return_value=True)
        mock_conn = mocker.MagicMock()
        mock_conn.execute.return_value.fetchdf.return_value = mock_df
        mocker.patch("duckdb.connect", return_value=mock_conn)

        result = run_query(
            TableName.PROJECT_RESOURCES, where="project_name", like="test"
        )
        assert result is not None

    def test_run_query_with_year(self, tmp_path, mocker):
        """Test run_query with year filter."""
        mock_df = pd.DataFrame({"id": [1]})

        db_file = tmp_path / "test.db"
        mocker.patch("esdc.dbmanager.Config.get_db_file", return_value=db_file)
        mocker.patch("pathlib.Path.exists", return_value=True)
        mock_conn = mocker.MagicMock()
        mock_conn.execute.return_value.fetchdf.return_value = mock_df
        mocker.patch("duckdb.connect", return_value=mock_conn)

        result = run_query(TableName.PROJECT_RESOURCES, years=[2024])
        assert result is not None

    def test_run_query_with_columns(self, tmp_path, mocker):
        """Test run_query with specific columns."""
        mock_df = pd.DataFrame({"id": [1]})

        db_file = tmp_path / "test.db"
        db_file = tmp_path / "test.db"
        mocker.patch("esdc.dbmanager.Config.get_db_file", return_value=db_file)
        mocker.patch("pathlib.Path.exists", return_value=True)
        mock_conn = mocker.MagicMock()
        mock_conn.execute.return_value.fetchdf.return_value = mock_df
        mocker.patch("duckdb.connect", return_value=mock_conn)

        result = run_query(TableName.PROJECT_RESOURCES, columns=["id", "name"])
        assert result is not None

    def test_run_query_sql_error(self, tmp_path, mocker):
        """Test run_query handles SQL errors gracefully."""
        db_file = tmp_path / "test.db"
        mocker.patch("esdc.dbmanager.Config.get_db_file", return_value=db_file)
        mocker.patch("pathlib.Path.exists", return_value=True)
        mock_conn = mocker.MagicMock()
        mock_conn.execute.side_effect = duckdb.Error("no such table")
        mocker.patch("duckdb.connect", return_value=mock_conn)

        result = run_query(TableName.PROJECT_RESOURCES)
        assert result is None


class TestLoadDataToDb:
    """Tests for load_data_to_db()."""

    def test_load_data_to_db_handles_empty_data(self, tmp_path, mocker):
        """Test that load_data_to_db handles empty data gracefully."""
        db_file = tmp_path / "test.db"
        mocker.patch("esdc.dbmanager.Config.get_db_file", return_value=db_file)
        mocker.patch("esdc.dbmanager.Config.get_db_dir", return_value=tmp_path)
        mocker.patch(
            "esdc.dbmanager._load_sql_script",
            return_value="CREATE TABLE test (id TEXT);",
        )
        mocker.patch("pathlib.Path.mkdir", return_value=None)

        with pytest.raises(duckdb.Error):
            load_data_to_db([], [], "project_resources")


class TestInvalidateSqlCache:
    """Tests for invalidate_sql_cache()."""

    def test_invalidate_cache_removes_dir(self, tmp_path, mocker):
        """Test that invalidate_sql_cache removes the cache directory."""
        cache_dir = tmp_path / "sql_results"
        cache_dir.mkdir()
        (cache_dir / "test.cache").write_text("test")

        mocker.patch("esdc.dbmanager.Config.get_cache_dir", return_value=tmp_path)
        invalidate_sql_cache()

        assert not cache_dir.exists()

    def test_invalidate_cache_no_dir(self, tmp_path, mocker):
        """Test invalidate_sql_cache handles missing cache dir gracefully."""
        mocker.patch("esdc.dbmanager.Config.get_cache_dir", return_value=tmp_path)
        invalidate_sql_cache()
