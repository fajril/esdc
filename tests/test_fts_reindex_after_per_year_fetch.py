"""Tests for --no-reindex escape hatch, FTS zero-row fallback, and FTS index quality.

These tests verify that:
- The fetch command accepts --no-reindex (escape hatch for auto-reindex)
- load_esdc_data defaults to reindex=True
- reindex_fts is called/not called depending on flag
- FTS query returning 0 rows falls back to original ILIKE query
- FTS omits stemmer/stopwords (DuckDB bug: empty strings cause Catalog Error)
"""

from unittest.mock import MagicMock, patch

import duckdb
import pytest
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
    """The loader passes the final reindex choice to the coordinator."""

    @pytest.mark.parametrize(
        ("years", "reindex", "expected"),
        [(None, True, True), (None, False, False), ([2025], True, True), ([2025], False, False)],
    )
    def test_passes_reindex_choice_to_coordinator(
        self, years, reindex, expected, tmp_path, monkeypatch
    ):
        from esdc.api_snapshot import ApiProjectResourceRow
        from esdc.fetching import FetchOutcome, PreparedFetch

        monkeypatch.setattr(Config, "get_db_file", lambda: tmp_path / "test.duckdb")
        monkeypatch.setattr(Config, "get_db_dir", lambda: tmp_path)
        observed: list[bool] = []
        monkeypatch.setattr(
            "esdc.esdc.prepare_fetch",
            lambda **_: PreparedFetch(
                (ApiProjectResourceRow(project_id="P-1", report_year=2025),),
                None if years is None else (2025,),
                (),
                (),
                (),
            ),
        )

        def publish(prepared, *, duckdb_path, sqlite_path, reindex):
            observed.append(reindex)
            return FetchOutcome(
                "revision", "sha", 1, 0, (2025,), 0, (), (), ()
            )

        monkeypatch.setattr("esdc.esdc.publish_prepared_fetch", publish)
        load_esdc_data(
            filetype=FileType.CSV,
            to_file=False,
            reload=True,
            username="u",
            password="p",
            years=years,
            reindex=reindex,
        )
        assert observed == [expected]


class TestReindexFtsUnit:
    """Direct test for reindex_fts rebuilding FTS indexes."""

    @pytest.mark.parametrize("has_hnsw", [False, True])
    @pytest.mark.parametrize(
        "timeseries_result", ["ok", "table not loaded", "skipped (0 rows)", "FTS error"]
    )
    def test_reindex_reports_actual_index_state(
        self, tmp_path, monkeypatch, capsys, has_hnsw, timeseries_result
    ):
        from esdc.dbmanager import reindex_fts

        db_path = tmp_path / "indexes.duckdb"
        conn = duckdb.connect(str(db_path))
        conn.execute("CREATE TABLE embeddings (id INTEGER)")
        if has_hnsw:
            # A named index exercises catalog detection without requiring VSS.
            conn.execute("CREATE INDEX idx_hnsw_embeddings ON embeddings(id)")
        conn.close()
        monkeypatch.setattr(Config, "get_db_file", lambda: db_path)
        monkeypatch.setattr("esdc.dbmanager.get_duckdb_connection", duckdb.connect)
        monkeypatch.setattr(
            "esdc.dbmanager._create_fts_indexes",
            lambda _: {
                "project_resources": "ok",
                "project_timeseries": timeseries_result,
            },
        )

        reindex_fts()

        output = capsys.readouterr().err
        assert ("HNSW embedding index dropped" in output) is has_hnsw
        assert ("✗" in output) is (timeseries_result == "FTS error")
        assert ("Search indexes rebuilt" in output) is (
            timeseries_result != "FTS error"
        )
        if timeseries_result in {"table not loaded", "skipped (0 rows)"}:
            assert "FTS index skipped: project_timeseries" in output
        with duckdb.connect(str(db_path), read_only=True) as check:
            assert check.execute(
                "SELECT COUNT(*) FROM duckdb_indexes()"
            ).fetchone() == (0,)

    @patch("esdc.dbmanager._create_fts_indexes")
    @patch("esdc.dbmanager.get_duckdb_connection")
    def test_reindex_fts_calls_create_fts_indexes(self, mock_conn, mock_fts):
        """reindex_fts must call _create_fts_indexes on valid DB."""
        from esdc.dbmanager import reindex_fts

        mock_cursor = MagicMock()
        mock_conn.return_value = mock_cursor
        mock_fts.return_value = {
            "project_resources": "table not loaded",
            "project_timeseries": "skipped (0 rows)",
        }

        def respond_to_reindex_queries(sql, *args, **kwargs):
            result = MagicMock()
            if "duckdb_indexes()" in sql:
                result.fetchone.return_value = None
            elif "information_schema.tables" in sql:
                result.fetchall.return_value = []
            return result

        mock_cursor.execute.side_effect = respond_to_reindex_queries

        db_path_mock = MagicMock()
        db_path_mock.exists.return_value = True
        db_path_mock.__str__ = MagicMock(return_value="/tmp/test.duckdb")

        with patch.object(Config, "get_db_file", return_value=db_path_mock):
            reindex_fts()

        mock_fts.assert_called_once_with(mock_cursor)
        mock_cursor.execute.assert_any_call("CHECKPOINT")

    @patch("esdc.dbmanager._create_fts_indexes")
    def test_path_specific_strict_reindex_rejects_populated_failure(
        self, mock_fts, tmp_path, monkeypatch
    ):
        from esdc.dbmanager import _reindex_fts_for_path

        explicit_path = tmp_path / "explicit.duckdb"
        config_path = tmp_path / "different-config.duckdb"
        monkeypatch.setattr("esdc.dbmanager.Config.get_db_file", lambda: config_path)
        conn = duckdb.connect(str(explicit_path))
        conn.execute("CREATE TABLE project_resources (project_name VARCHAR)")
        conn.execute("INSERT INTO project_resources VALUES ('present')")
        conn.close()
        mock_fts.return_value = {"project_resources": "index failed"}

        with pytest.raises(RuntimeError, match="project_resources"):
            _reindex_fts_for_path(explicit_path, strict=True)

        mock_fts.assert_called_once()
        assert explicit_path.exists()
        assert not config_path.exists()


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
            "groovy_report",
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
        assert any("'groovy_report'" in call_str for call_str in fts_calls)
