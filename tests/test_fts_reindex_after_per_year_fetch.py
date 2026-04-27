"""Tests for --reindex-only flag on esdc fetch and FTS rebuild behavior.

These tests verify that:
- The fetch command accepts --reindex-only
- load_esdc_data passes the flag through correctly
- reindex_fts is called after data loading in both full-replace and per-year modes
"""

from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from esdc.configs import Config
from esdc.dbmanager import get_duckdb_connection
from esdc.esdc import app, load_esdc_data
from esdc.selection import FileType

runner = CliRunner()


class TestFetchReindexOnlyFlag:
    """Test that --reindex-only is accepted by the fetch CLI."""

    def test_fetch_help_shows_reindex_only(self):
        result = runner.invoke(app, ["fetch", "--help"])
        assert result.exit_code == 0
        assert "--reindex-only" in result.output

    @patch("esdc.configs.Config.get_credentials")
    @patch("esdc.esdc.load_esdc_data")
    def test_fetch_year_with_reindex_only(self, mock_load, mock_creds):
        mock_creds.return_value = ("user", "pass")
        result = runner.invoke(
            app,
            ["fetch", "--year", "2025", "--reindex-only"],
        )
        assert result.exit_code == 0
        mock_load.assert_called_once()
        call_kwargs = mock_load.call_args.kwargs
        assert call_kwargs.get("reindex_only") is True
        assert call_kwargs.get("years") == [2025]

    @patch("esdc.configs.Config.get_credentials")
    @patch("esdc.esdc.load_esdc_data")
    def test_fetch_without_reindex_only_defaults_false(self, mock_load, mock_creds):
        mock_creds.return_value = ("user", "pass")
        result = runner.invoke(
            app,
            ["fetch", "--year", "2025"],
        )
        assert result.exit_code == 0
        mock_load.assert_called_once()
        call_kwargs = mock_load.call_args.kwargs
        assert call_kwargs.get("reindex_only") is False


class TestLoadEsdcDataReindexOnly:
    """Test that load_esdc_data calls reindex_fts when reindex_only=True."""

    @patch("esdc.dbmanager.reindex_fts")
    @patch("esdc.esdc._detect_report_years")
    @patch("esdc.esdc.load_data_to_db")
    @patch("esdc.esdc._fetch_and_parse_table")
    def test_full_replace_calls_reindex_fts_when_true(
        self,
        mock_fetch,
        mock_load_db,
        mock_detect,
        mock_reindex,
        tmp_path,
        monkeypatch,
    ):
        """Full-replace mode with reindex_only=True triggers reindex_fts."""
        monkeypatch.setattr(Config, "get_db_file", lambda: tmp_path / "test.duckdb")
        monkeypatch.setattr(Config, "get_db_dir", lambda: tmp_path)

        mock_fetch.return_value = ([["a", "b"]], ["col1", "col2"])
        mock_detect.return_value = [2025]

        load_esdc_data(
            filetype=FileType.CSV,
            to_file=False,
            reload=True,
            username="u",
            password="p",
            years=None,
            reindex_only=True,
        )

        mock_reindex.assert_called_once()

    @patch("esdc.dbmanager.reindex_fts")
    @patch("esdc.esdc._detect_report_years")
    @patch("esdc.esdc.load_data_to_db")
    @patch("esdc.esdc._fetch_and_parse_table")
    def test_full_replace_does_not_call_reindex_when_false(
        self,
        mock_fetch,
        mock_load_db,
        mock_detect,
        mock_reindex,
        tmp_path,
        monkeypatch,
    ):
        """Full-replace mode with reindex_only=False does NOT trigger reindex_fts."""
        monkeypatch.setattr(Config, "get_db_file", lambda: tmp_path / "test.duckdb")
        monkeypatch.setattr(Config, "get_db_dir", lambda: tmp_path)

        mock_fetch.return_value = ([["a", "b"]], ["col1", "col2"])
        mock_detect.return_value = [2025]

        load_esdc_data(
            filetype=FileType.CSV,
            to_file=False,
            reload=True,
            username="u",
            password="p",
            years=None,
            reindex_only=False,
        )

        mock_reindex.assert_not_called()

    @patch("esdc.dbmanager.reindex_fts")
    @patch("esdc.esdc._append_to_table")
    @patch("esdc.esdc._fetch_and_parse_table")
    def test_per_year_calls_reindex_fts_when_true(
        self,
        mock_fetch,
        mock_append,
        mock_reindex,
        tmp_path,
        monkeypatch,
    ):
        """Per-year mode with reindex_only=True triggers reindex_fts."""
        monkeypatch.setattr(Config, "get_db_file", lambda: tmp_path / "test.duckdb")
        monkeypatch.setattr(Config, "get_db_dir", lambda: tmp_path)

        mock_fetch.return_value = ([["a", "b"]], ["col1", "col2"])

        load_esdc_data(
            filetype=FileType.CSV,
            to_file=False,
            reload=True,
            username="u",
            password="p",
            years=[2025],
            reindex_only=True,
        )

        mock_reindex.assert_called_once()

    @patch("esdc.dbmanager.reindex_fts")
    @patch("esdc.esdc._append_to_table")
    @patch("esdc.esdc._fetch_and_parse_table")
    def test_per_year_does_not_call_reindex_when_false(
        self,
        mock_fetch,
        mock_append,
        mock_reindex,
        tmp_path,
        monkeypatch,
    ):
        """Per-year mode with reindex_only=False does NOT trigger reindex_fts."""
        monkeypatch.setattr(Config, "get_db_file", lambda: tmp_path / "test.duckdb")
        monkeypatch.setattr(Config, "get_db_dir", lambda: tmp_path)

        mock_fetch.return_value = ([["a", "b"]], ["col1", "col2"])

        load_esdc_data(
            filetype=FileType.CSV,
            to_file=False,
            reload=True,
            username="u",
            password="p",
            years=[2025],
            reindex_only=False,
        )

        mock_reindex.assert_not_called()


class TestReindexFtsUnit:
    """Direct test for reindex_fts rebuilding FTS indexes."""

    @patch("esdc.dbmanager._create_fts_indexes")
    @patch("esdc.dbmanager.get_duckdb_connection")
    def test_reindex_fts_calls_create_fts_indexes(self, mock_conn, mock_fts):
        """reindex_fts must call _create_fts_indexes on valid DB."""
        from esdc.dbmanager import reindex_fts

        mock_cursor = MagicMock()
        mock_conn.return_value = mock_cursor

        db_path_mock = MagicMock()
        db_path_mock.exists.return_value = True
        db_path_mock.__str__ = lambda self: "/tmp/test.duckdb"

        with patch.object(Config, "get_db_file", return_value=db_path_mock):
            reindex_fts()

        mock_fts.assert_called_once_with(mock_cursor)
        mock_cursor.execute.assert_any_call("CHECKPOINT")
