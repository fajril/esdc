import duckdb
import pandas as pd
import pytest

from esdc.dbmanager import (
    _load_sql_script,
    check_indexes,
    check_table_stats,
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


class TestCheckIndexes:
    """Tests for check_indexes()."""

    def test_check_indexes_all_present(self, tmp_path):
        """Test check_indexes with all indexes present."""
        conn = duckdb.connect(str(tmp_path / "test.db"))
        conn.execute("INSTALL fts")
        conn.execute("LOAD fts")
        conn.execute("INSTALL vss")
        conn.execute("LOAD vss")
        conn.execute("SET hnsw_enable_experimental_persistence = true")
        conn.execute(
            "CREATE TABLE project_resources AS "
            "SELECT range AS id, 'name' || CAST(range AS VARCHAR) AS project_name "
            "FROM range(5)"
        )
        conn.execute(
            "CREATE TABLE project_timeseries AS "
            "SELECT range AS id, 'name' || CAST(range AS VARCHAR) AS project_name "
            "FROM range(5)"
        )
        conn.execute(
            "PRAGMA create_fts_index('project_resources', 'id', 'project_name')"
        )
        conn.execute(
            "PRAGMA create_fts_index('project_timeseries', 'id', 'project_name')"
        )
        conn.execute(
            "CREATE INDEX idx_project_resources_report_year ON project_resources(id)"
        )
        conn.execute(
            "CREATE TABLE project_embeddings AS "
            "SELECT CAST(range AS VARCHAR) AS uuid, "
            "[0.1, 0.2, 0.3]::FLOAT[3] AS embedding "
            "FROM range(3)"
        )
        conn.execute(
            "CREATE INDEX idx_hnsw_embeddings ON project_embeddings "
            "USING HNSW (embedding)"
        )

        result = check_indexes(conn)

        assert len(result["fts_indexes"]) == 2
        assert result["fts_indexes"][0]["table"] == "project_resources"
        assert result["fts_indexes"][0]["exists"] is True
        assert result["fts_indexes"][1]["table"] == "project_timeseries"
        assert result["fts_indexes"][1]["exists"] is True

        assert len(result["btree_indexes"]) == 4
        assert any(
            b["name"] == "idx_project_resources_report_year" and b["exists"]
            for b in result["btree_indexes"]
        )

        assert result["embeddings"]["table_exists"] is True
        assert result["embeddings"]["row_count"] == 3
        assert result["embeddings"]["hnsw_exists"] is True
        conn.close()

    def test_check_indexes_missing_indexes(self, tmp_path):
        """Test check_indexes with no indexes."""
        conn = duckdb.connect(str(tmp_path / "test.db"))
        conn.execute(
            "CREATE TABLE project_resources AS "
            "SELECT range AS id, 'name' || CAST(range AS VARCHAR) AS project_name "
            "FROM range(5)"
        )
        conn.execute(
            "CREATE TABLE project_timeseries AS "
            "SELECT range AS id, 'name' || CAST(range AS VARCHAR) AS project_name "
            "FROM range(5)"
        )

        result = check_indexes(conn)

        assert result["fts_indexes"][0]["exists"] is False
        assert result["fts_indexes"][1]["exists"] is False
        assert all(not b["exists"] for b in result["btree_indexes"])
        assert result["embeddings"]["table_exists"] is False
        assert result["embeddings"]["hnsw_exists"] is False
        conn.close()


class TestCheckTableStats:
    """Tests for check_table_stats()."""

    def test_check_table_stats_with_data(self, tmp_path):
        """Test check_table_stats with data per year."""
        conn = duckdb.connect(str(tmp_path / "test.db"))
        conn.execute(
            "CREATE TABLE project_resources AS "
            "SELECT 2023 AS report_year, 'name' || CAST(range AS VARCHAR) "
            "AS project_name FROM range(5) "
            "UNION ALL "
            "SELECT 2024 AS report_year, 'name' || CAST(range + 5 AS VARCHAR) "
            "AS project_name FROM range(3)"
        )
        conn.execute(
            "CREATE TABLE project_timeseries AS "
            "SELECT 2024 AS report_year, 'name' || CAST(range AS VARCHAR) "
            "AS project_name FROM range(7)"
        )

        result = check_table_stats(conn)

        assert len(result) == 2
        pr = result[0]
        assert pr["table"] == "project_resources"
        assert pr["total"] == 8
        assert (2023, 5) in pr["years"]
        assert (2024, 3) in pr["years"]

        pt = result[1]
        assert pt["table"] == "project_timeseries"
        assert pt["total"] == 7
        assert (2024, 7) in pt["years"]
        conn.close()

    def test_check_table_stats_empty_database(self, tmp_path):
        """Test check_table_stats with no tables."""
        conn = duckdb.connect(str(tmp_path / "test.db"))

        result = check_table_stats(conn)

        assert len(result) == 2
        assert result[0]["total"] == 0
        assert result[0]["years"] == []
        assert result[1]["total"] == 0
        assert result[1]["years"] == []
        conn.close()
