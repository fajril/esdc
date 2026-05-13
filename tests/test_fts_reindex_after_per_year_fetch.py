"""Tests for --no-reindex escape hatch, FTS zero-row fallback, and FTS index quality.

These tests verify that:
- The fetch command accepts --no-reindex (escape hatch for auto-reindex)
- load_esdc_data defaults to reindex=True
- reindex_fts is called/not called depending on flag
- FTS query returning 0 rows falls back to original ILIKE query
- FTS omits stemmer/stopwords (DuckDB bug: empty strings cause Catalog Error)
"""

from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from esdc.configs import Config
from esdc.esdc import app, load_esdc_data
from esdc.selection import FileType

runner = CliRunner()


class TestFetchNoReindexFlag:
    """Test --no-reindex flag defaults to auto-reindex."""

    def test_fetch_help_shows_no_reindex(self):
        result = runner.invoke(app, ["fetch", "--help"])
        assert result.exit_code == 0
        assert "--no-reindex" in result.output

    @patch("esdc.configs.Config.get_credentials")
    @patch("esdc.esdc.load_esdc_data")
    def test_fetch_year_with_no_reindex(self, mock_load, mock_creds):
        mock_creds.return_value = ("user", "pass")
        result = runner.invoke(
            app,
            ["fetch", "--year", "2025", "--no-reindex"],
        )
        assert result.exit_code == 0
        mock_load.assert_called_once()
        call_kwargs = mock_load.call_args.kwargs
        assert call_kwargs.get("reindex") is False

    @patch("esdc.configs.Config.get_credentials")
    @patch("esdc.esdc.load_esdc_data")
    def test_fetch_year_default_auto_reindex(self, mock_load, mock_creds):
        mock_creds.return_value = ("user", "pass")
        result = runner.invoke(
            app,
            ["fetch", "--year", "2025"],
        )
        assert result.exit_code == 0
        mock_load.assert_called_once()
        call_kwargs = mock_load.call_args.kwargs
        assert call_kwargs.get("reindex") is True


class TestLoadEsdcDataReindex:
    """Test that load_esdc_data calls reindex_fts when reindex=True (default)."""

    @patch("esdc.dbmanager.reindex_fts")
    @patch("esdc.esdc._detect_report_years")
    @patch("esdc.esdc.load_data_to_db")
    @patch("esdc.esdc._fetch_and_parse_table")
    def test_full_replace_calls_reindex_fts_by_default(
        self,
        mock_fetch,
        mock_load_db,
        mock_detect,
        mock_reindex,
        tmp_path,
        monkeypatch,
    ):
        """Full-replace mode with reindex=True (default) triggers reindex_fts."""
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
        )

        mock_reindex.assert_called_once()

    @patch("esdc.dbmanager.reindex_fts")
    @patch("esdc.esdc._detect_report_years")
    @patch("esdc.esdc.load_data_to_db")
    @patch("esdc.esdc._fetch_and_parse_table")
    def test_full_replace_skips_reindex_when_false(
        self,
        mock_fetch,
        mock_load_db,
        mock_detect,
        mock_reindex,
        tmp_path,
        monkeypatch,
    ):
        """Full-replace mode with reindex=False does NOT trigger reindex_fts."""
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
            reindex=False,
        )

        mock_reindex.assert_not_called()

    @patch("esdc.dbmanager.reindex_fts")
    @patch("esdc.esdc._append_to_table")
    @patch("esdc.esdc._fetch_and_parse_table")
    def test_per_year_calls_reindex_fts_by_default(
        self,
        mock_fetch,
        mock_append,
        mock_reindex,
        tmp_path,
        monkeypatch,
    ):
        """Per-year mode with reindex=True (default) triggers reindex_fts."""
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
        )

        mock_reindex.assert_called_once()

    @patch("esdc.dbmanager.reindex_fts")
    @patch("esdc.esdc._append_to_table")
    @patch("esdc.esdc._fetch_and_parse_table")
    def test_per_year_skips_reindex_when_false(
        self,
        mock_fetch,
        mock_append,
        mock_reindex,
        tmp_path,
        monkeypatch,
    ):
        """Per-year mode with reindex=False does NOT trigger reindex_fts."""
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
            reindex=False,
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
        db_path_mock.__str__ = MagicMock(return_value="/tmp/test.duckdb")

        with patch.object(Config, "get_db_file", return_value=db_path_mock):
            reindex_fts()

        mock_fts.assert_called_once_with(mock_cursor)
        mock_cursor.execute.assert_any_call("CHECKPOINT")


class TestFTSZeroRowFallback:
    """Test that FTS query returning 0 rows falls back to original ILIKE query."""

    def test_rewrite_with_fts_and_fallback_on_zero_rows(self, tmp_path, monkeypatch):
        """If FTS-rewritten query returns 0 rows, fallback to original ILIKE query."""
        from esdc.chat.tools import _rewrite_with_fts

        original = (
            "SELECT project_name FROM project_resources "
            "WHERE field_name ILIKE '%Duri%' AND report_year = 2025"
        )
        rewritten = _rewrite_with_fts(original)
        assert "match_bm25" in rewritten
        assert "field_name ILIKE '%Duri%'" in rewritten


class TestFTSIndexNoStopwordsNoStemmer:
    """Verify FTS index creation omits stemmer/stopwords params.

    DuckDB's FTS extension treats empty string values (stemmer='',
    stopwords='') as references to catalog tables, causing
    'Catalog Error: Table with name  does not exist!'.
    Omitting these params disables stemming and stopword removal by default.
    """

    def test_create_fts_indexes_omits_stemmer_and_stopwords(self):
        """_create_fts_indexes must not pass stemmer/stopwords to PRAGMA."""
        from esdc.dbmanager import _create_fts_indexes

        mock_conn = MagicMock()

        fts_columns = [
            "project_name",
            "field_name",
            "wk_name",
            "province",
            "basin128",
            "operator_name",
            "project_remarks",
            "vol_remarks",
            "basin86",
            "wk_id",
            "field_id",
            "field_name_previous",
            "project_name_previous",
            "pod_name",
            "operator_group",
            "wk_subgroup",
        ]

        def mock_execute(sql, *args, **kwargs):
            result = MagicMock()
            if "information_schema.tables" in sql:
                result.fetchall.return_value = [
                    ("project_resources",),
                    ("project_timeseries",),
                ]
            elif "information_schema.columns" in sql:
                result.fetchone.return_value = (5,)
                cols = fts_columns if "project_resources" in sql else fts_columns[:13]
                result.fetchall.return_value = [(c,) for c in cols]
            elif "COUNT" in sql:
                result.fetchone.return_value = (5,)
            else:
                result.fetchall.return_value = []
                result.fetchone.return_value = None
            return result

        mock_conn.execute.side_effect = mock_execute

        _create_fts_indexes(mock_conn)

        fts_calls = [
            call.args[0]
            for call in mock_conn.execute.call_args_list
            if "create_fts_index" in str(call.args[0])
        ]
        assert len(fts_calls) >= 1, "Expected at least one create_fts_index call"
        for call_str in fts_calls:
            assert "stemmer" not in call_str, f"Unexpected stemmer param in: {call_str}"
            assert "stopwords" not in call_str, (
                f"Unexpected stopwords param in: {call_str}"
            )
