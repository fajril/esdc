"""Tests for the database manager."""

import duckdb
import pandas as pd
import pytest

from esdc.dbmanager import (
    _execute_sql_script,
    _load_sql_script,
    check_indexes,
    check_table_stats,
    get_last_updated,
    invalidate_sql_cache,
    load_data_to_db,
    run_query,
    verify_indexes,
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


class TestVerifyIndexes:
    """Tests for verify_indexes()."""

    def test_verify_indexes_with_fts(self, tmp_path):
        """Test verify_indexes detects functional FTS."""
        conn = duckdb.connect(str(tmp_path / "test.db"))
        conn.execute("INSTALL fts")
        conn.execute("LOAD fts")
        conn.execute("INSTALL vss")
        conn.execute("LOAD vss")
        conn.execute(
            "CREATE TABLE project_resources AS "
            "SELECT CAST(range AS VARCHAR) AS uuid, "
            "'duri field' AS project_name, "
            "2024 AS report_year "
            "FROM range(5)"
        )
        conn.execute(
            "CREATE TABLE project_timeseries AS "
            "SELECT CAST(range AS VARCHAR) AS uuid, "
            "'duri field' AS project_name, "
            "2024 AS report_year "
            "FROM range(5)"
        )
        conn.execute(
            "PRAGMA create_fts_index('project_resources', 'uuid', 'project_name')"
        )
        conn.execute(
            "PRAGMA create_fts_index('project_timeseries', 'uuid', 'project_name')"
        )

        result = verify_indexes(conn)

        assert len(result["fts"]) == 2
        assert result["fts"][0]["table"] == "project_resources"
        assert result["fts"][0]["functional"] is True
        assert result["fts"][0]["result_count"] > 0
        assert result["fts"][1]["table"] == "project_timeseries"
        assert result["fts"][1]["functional"] is True
        conn.close()

    def test_verify_indexes_empty_database(self, tmp_path):
        """Test verify_indexes with no indexes."""
        conn = duckdb.connect(str(tmp_path / "test.db"))

        result = verify_indexes(conn)

        assert len(result["btree"]) == 4
        assert all(not bt["functional"] for bt in result["btree"])
        conn.close()


class TestGetLastUpdated:
    """Tests for get_last_updated()."""

    def test_get_last_updated_returns_timestamp(self, tmp_path):
        """Test get_last_updated returns timestamp from _metadata table."""
        conn = duckdb.connect(str(tmp_path / "test.db"))
        conn.execute("INSTALL vss")
        conn.execute("LOAD vss")
        _execute_sql_script(conn, "create_table_metadata.sql")

        result = get_last_updated(conn)
        assert result is not None
        assert len(result) > 0
        conn.close()

    def test_get_last_updated_returns_none_when_no_table(self, tmp_path):
        """Test get_last_updated returns None when _metadata table does not exist."""
        conn = duckdb.connect(str(tmp_path / "test.db"))
        conn.execute("INSTALL vss")
        conn.execute("LOAD vss")

        result = get_last_updated(conn)
        assert result is None
        conn.close()

    def test_get_last_updated_updates_on_second_call(self, tmp_path):
        """Test that re-running metadata SQL updates the timestamp."""
        import time

        conn = duckdb.connect(str(tmp_path / "test.db"))
        conn.execute("INSTALL vss")
        conn.execute("LOAD vss")

        _execute_sql_script(conn, "create_table_metadata.sql")
        first = get_last_updated(conn)

        time.sleep(1.1)

        _execute_sql_script(conn, "create_table_metadata.sql")
        second = get_last_updated(conn)

        assert first is not None
        assert second is not None
        assert second >= first
        conn.close()


class TestGetDuckdbConnectionReadOnlyDefault:
    """get_duckdb_connection(): read-only by default, writers must opt in.

    DuckDB allows many concurrent read-only processes but a read-write
    connection takes an exclusive file lock (`esdc serve` vs `esdc status`
    conflict), so the safe default is read-only.
    """

    def test_default_connection_rejects_writes(self, tmp_path):
        from esdc.dbmanager import get_duckdb_connection

        db_file = tmp_path / "ro.db"
        seed = duckdb.connect(str(db_file))
        seed.execute("CREATE TABLE t (x INTEGER)")
        seed.close()

        conn = get_duckdb_connection(db_file)
        try:
            assert conn.execute("SELECT COUNT(*) FROM t").fetchone() == (0,)
            with pytest.raises(duckdb.Error, match="read-only"):
                conn.execute("INSERT INTO t VALUES (1)")
        finally:
            conn.close()

    def test_explicit_read_write_allows_writes(self, tmp_path):
        from esdc.dbmanager import get_duckdb_connection

        db_file = tmp_path / "rw.db"
        conn = get_duckdb_connection(db_file, read_only=False)
        try:
            conn.execute("CREATE TABLE t (x INTEGER)")
            conn.execute("INSERT INTO t VALUES (1)")
            assert conn.execute("SELECT COUNT(*) FROM t").fetchone() == (1,)
        finally:
            conn.close()

    def test_read_only_coexists_with_second_reader(self, tmp_path):
        """Two read-only connections may hold the same file simultaneously."""
        from esdc.dbmanager import get_duckdb_connection

        db_file = tmp_path / "multi.db"
        seed = duckdb.connect(str(db_file))
        seed.execute("CREATE TABLE t (x INTEGER)")
        seed.close()

        first = get_duckdb_connection(db_file)
        second = get_duckdb_connection(db_file)
        try:
            assert first.execute("SELECT COUNT(*) FROM t").fetchone() == (0,)
            assert second.execute("SELECT COUNT(*) FROM t").fetchone() == (0,)
        finally:
            first.close()
            second.close()
