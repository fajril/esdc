import os
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from esdc.esdc import app

runner = CliRunner()


class TestCliHelp:
    """Tests for CLI help output."""

    def test_main_help(self):
        """Test main help output."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "Usage:" in result.stdout
        assert "Commands" in result.stdout
        assert "fetch" in result.stdout
        assert "show" in result.stdout

    def test_fetch_help(self):
        """Test fetch command --help."""
        result = runner.invoke(app, ["fetch", "--help"])
        assert result.exit_code == 0
        assert "filetype" in result.stdout.lower()
        assert "save" in result.stdout.lower()

    def test_show_help(self):
        """Test show command --help."""
        result = runner.invoke(app, ["show", "--help"])
        assert result.exit_code == 0
        assert "table" in result.stdout.lower()
        assert "where" in result.stdout.lower()

    def test_reload_help(self):
        """Test reload command --help."""
        result = runner.invoke(app, ["reload", "--help"])
        assert result.exit_code == 0
        assert "filetype" in result.stdout.lower()


class TestStatusCommand:
    """Tests for status command and its `fetch`/`cache` subcommands."""

    def test_status_output(self, isolated_config):
        """Test `status fetch` shows correct database info."""
        result = runner.invoke(app, ["status", "fetch"])
        assert result.exit_code == 0
        assert ".esdc" in result.stdout

    def test_status_shows_cache_section(self, isolated_config):
        """Test `status cache` shows cache diagnostics."""
        result = runner.invoke(app, ["status", "cache"])
        assert result.exit_code == 0
        assert "Cache:" in result.stdout
        assert "Directory:" in result.stdout

    def test_status_shows_hit_rate_when_db_exists(self, isolated_config):
        """Test `status cache` shows cache hit rate."""
        result = runner.invoke(app, ["status", "cache"])
        assert result.exit_code == 0
        assert "Hit rate:" in result.stdout or "N/A" in result.stdout

    def test_status_with_custom_env(self):
        """Test `status fetch` shows custom path from env var."""
        with patch.dict(os.environ, {"ESDC_DB_FILE": "/custom/path/db.db"}):
            result = runner.invoke(app, ["status", "fetch"])
            assert result.exit_code == 0
            assert "/custom/path/db.db" in result.stdout

    def test_status_no_database(self):
        """Test bare status shows no database message when DB missing."""
        with patch.dict(os.environ, {"ESDC_DB_FILE": "/nonexistent/path/db.db"}):
            result = runner.invoke(app, ["status"])
            assert result.exit_code == 0
            assert "Database exists: No" in result.stdout

    def test_status_shows_last_updated(self, isolated_config):
        """Test `status fetch` shows Last updated line."""
        import duckdb

        from esdc.configs import Config

        Config.init_config()
        db_file = Config.get_db_file()
        db_file.parent.mkdir(parents=True, exist_ok=True)
        duckdb.connect(str(db_file)).close()

        result = runner.invoke(app, ["status", "fetch"])
        assert result.exit_code == 0
        assert "Last updated" in result.stdout


class TestStatusSubcommands:
    """Tests for `esdc status` sub-app routing and the compact summary."""

    def _touch_db(self):
        import duckdb

        from esdc.configs import Config

        Config.init_config()
        db_file = Config.get_db_file()
        db_file.parent.mkdir(parents=True, exist_ok=True)
        duckdb.connect(str(db_file)).close()
        return db_file

    def test_fetch_subcommand_shows_tables_marker(self, isolated_config):
        self._touch_db()
        result = runner.invoke(app, ["status", "fetch"])
        assert result.exit_code == 0
        assert "Tables:" in result.stdout

    def test_index_subcommand_shows_fts_marker(self, isolated_config):
        self._touch_db()
        result = runner.invoke(app, ["status", "index"])
        assert result.exit_code == 0
        assert "FTS Indexes:" in result.stdout

    def test_corpus_subcommand_shows_corpus_marker(self, isolated_config):
        self._touch_db()
        result = runner.invoke(app, ["status", "corpus"])
        assert result.exit_code == 0
        assert "Corpus:" in result.stdout

    def test_cache_subcommand_shows_cache_marker(self, isolated_config):
        result = runner.invoke(app, ["status", "cache"])
        assert result.exit_code == 0
        assert "Cache:" in result.stdout

    def test_bare_status_is_compact_summary(self, seeded_database):
        """Bare `esdc status` is a one-glance summary, not the full report."""
        result = runner.invoke(app, ["status"])
        assert result.exit_code == 0
        assert "Detail: esdc status" in result.stdout
        assert "Tables:" in result.stdout
        assert "Corpus:" in result.stdout
        assert "Cache:" in result.stdout
        # Full per-year breakdown (from `status fetch`) must NOT appear here.
        assert "2024:" not in result.stdout

    def test_fetch_subcommand_shows_full_per_year_breakdown(self, seeded_database):
        result = runner.invoke(app, ["status", "fetch"])
        assert result.exit_code == 0
        assert "2024:" in result.stdout

    def test_index_verify_flag_still_works(self, isolated_config):
        self._touch_db()
        result = runner.invoke(app, ["status", "index", "--verify"])
        assert result.exit_code == 0

    def test_bare_status_verify_flag_removed(self, isolated_config):
        """--verify moved to `status index`; bare status no longer accepts it."""
        result = runner.invoke(app, ["status", "--verify"])
        assert result.exit_code == 2


class _FakeEmbedder:
    """Minimal embedder stub for corpus-store CLI tests (no Ollama needed)."""

    model = "fake-embed"

    def generate_embedding(self, text: str) -> list[float]:
        return [1.0, 0.0, 0.0]

    def generate_embeddings_batch(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0, 0.0] for _ in texts]


class TestStatusCorpus:
    """Tests for `esdc status corpus` (store-level corpus summary)."""

    def test_not_initialized_when_corpus_tables_absent(self, isolated_config):
        """Corpus section reports not-initialized when tables don't exist."""
        import duckdb

        from esdc.configs import Config

        Config.init_config()
        db_file = Config.get_db_file()
        db_file.parent.mkdir(parents=True, exist_ok=True)
        duckdb.connect(str(db_file)).close()

        result = runner.invoke(app, ["status", "corpus"])
        assert result.exit_code == 0
        assert "not initialized" in result.stdout

    def test_shows_counts_and_breakdown_for_committed_docs(self, isolated_config):
        """Corpus section shows doc/chunk counts, doc_type breakdown, model."""
        from esdc.corpus.chunker import Chunk
        from esdc.corpus.store import CorpusStore

        db_file = isolated_config / ".esdc" / "esdc.duckdb"
        store = CorpusStore(db_path=db_file, embedder=_FakeEmbedder())
        store.ensure_tables()
        store.insert_document(
            {
                "doc_id": "letter-1",
                "file_name": "letter.pdf",
                "file_path": "/x/letter.pdf",
                "file_hash": "aa" * 32,
                "doc_type": "letter",
                "doc_number": "SRT-1",
                "doc_date": "2026-01-05",
                "subject": "Persetujuan",
                "sender": "SKK",
                "recipient": "KKKS",
                "doc_level": "field",
                "wk_name": "Rokan",
                "field_name": "Duri",
                "project_name": None,
                "raw_entities": "{}",
                "metadata": "{}",
                "markdown": "# Surat\nisi",
                "extraction_method": "native",
                "page_count": 1,
            },
            [Chunk(0, "Surat", "isi surat satu"), Chunk(1, "Surat", "isi surat dua")],
        )
        store.insert_document(
            {
                "doc_id": "mom-1",
                "file_name": "mom.pdf",
                "file_path": "/x/mom.pdf",
                "file_hash": "bb" * 32,
                "doc_type": "mom",
                "doc_number": "MOM-1",
                "doc_date": "2026-01-06",
                "subject": "Rapat",
                "sender": "SKK",
                "recipient": "KKKS",
                "doc_level": "field",
                "wk_name": "Rokan",
                "field_name": "Duri",
                "project_name": None,
                "raw_entities": "{}",
                "metadata": "{}",
                "markdown": "# MoM\nisi",
                "extraction_method": "native",
                "page_count": 1,
            },
            [Chunk(0, "MoM", "isi mom satu")],
        )
        store.close()

        result = runner.invoke(app, ["status", "corpus"])
        assert result.exit_code == 0
        assert "2 documents" in result.stdout
        assert "3 chunks" in result.stdout
        assert "letter: 1" in result.stdout
        assert "mom: 1" in result.stdout
        assert "fake-embed" in result.stdout


class TestShowCommand:
    """Tests for show command."""

    def test_show_invalid_table(self):
        """Test show with invalid table name."""
        result = runner.invoke(app, ["show", "invalid_table"])
        assert result.exit_code != 0

    def test_show_with_where_filter(self):
        """Test show with where filter."""
        with patch("esdc.esdc.run_query", return_value=MagicMock()):
            result = runner.invoke(
                app, ["show", "project_resources", "--where", "id", "--search", "test"]
            )
            assert result.exit_code == 0

    def test_show_with_year_filter(self):
        """Test show with year filter."""
        with patch("esdc.esdc.run_query", return_value=MagicMock()):
            result = runner.invoke(app, ["show", "project_resources", "--year", "2024"])
            assert result.exit_code == 0

    def test_show_with_detail_level(self):
        """Test show with detail level."""
        with patch("esdc.esdc.run_query", return_value=MagicMock()):
            result = runner.invoke(
                app, ["show", "project_resources", "--detail", "reserves"]
            )
            assert result.exit_code == 0


class TestFetchCommand:
    """Tests for fetch command."""

    def test_fetch_requires_credentials(self):
        """Test fetch prompts for credentials when not in env."""
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("esdc.esdc.Config.get_credentials") as mock_creds,
            patch("esdc.esdc.load_esdc_data"),
        ):
            mock_creds.return_value = ("user", "pass")
            result = runner.invoke(app, ["fetch", "--filetype", "json"])
            assert result.exit_code == 0

    def test_fetch_with_csv(self):
        """Test fetch with csv filetype."""
        with (
            patch.dict(os.environ, {"ESDC_USER": "test", "ESDC_PASS": "test"}),
            patch("esdc.esdc.load_esdc_data"),
        ):
            result = runner.invoke(app, ["fetch", "--filetype", "csv"])
            assert result.exit_code == 0

    def test_fetch_with_invalid_filetype(self):
        """Test fetch with invalid filetype."""
        with (
            patch.dict(os.environ, {"ESDC_USER": "test", "ESDC_PASS": "test"}),
            patch("esdc.esdc.load_esdc_data"),
        ):
            result = runner.invoke(app, ["fetch", "--filetype", "invalid"])
            assert result.exit_code == 0


class TestReloadCommand:
    """Tests for reload command."""

    def test_reload_missing_file(self, tmp_path):
        """Test reload with missing file shows warning."""
        with patch("esdc.esdc.Config.get_db_dir", return_value=tmp_path):
            result = runner.invoke(
                app, ["reload", "--filetype", "csv", "--no-embeddings"]
            )
            assert result.exit_code in [0, 1]


class TestVerboseFlag:
    """Tests for --verbose flag."""

    def test_verbose_default(self):
        """Test default verbosity level."""
        with patch("esdc.esdc.Config.init_config"):
            result = runner.invoke(app, ["--help"])
            assert result.exit_code == 0

    def test_verbose_info(self):
        """Test info verbosity level."""
        with patch("esdc.esdc.Config.init_config"):
            result = runner.invoke(app, ["--verbose", "1", "status"])
            assert result.exit_code == 0

    def test_verbose_debug(self):
        """Test debug verbosity level."""
        with patch("esdc.esdc.Config.init_config"):
            result = runner.invoke(app, ["--verbose", "2", "status"])
            assert result.exit_code == 0


class TestHumanizeBytes:
    """Tests for _humanize_bytes helper."""

    def test_bytes(self):
        from esdc.esdc import _humanize_bytes

        assert _humanize_bytes(0) == "0 B"
        assert _humanize_bytes(512) == "512 B"
        assert _humanize_bytes(1023) == "1023 B"

    def test_kilobytes(self):
        from esdc.esdc import _humanize_bytes

        assert _humanize_bytes(1024) == "1.0 KB"
        assert _humanize_bytes(1536) == "1.5 KB"

    def test_megabytes(self):
        from esdc.esdc import _humanize_bytes

        assert _humanize_bytes(1024**2) == "1.0 MB"
        assert _humanize_bytes(int(1.5 * 1024**2)) == "1.5 MB"

    def test_gigabytes(self):
        from esdc.esdc import _humanize_bytes

        assert _humanize_bytes(1024**3) == "1.0 GB"
        assert _humanize_bytes(500_000_000) == "476.8 MB"

    def test_negative_bytes(self):
        from esdc.esdc import _humanize_bytes

        assert _humanize_bytes(-1) == "0 B"
        assert _humanize_bytes(-1024) == "0 B"


class TestPrintHitRate:
    """Tests for _print_hit_rate helper."""

    def test_zero_activity(self):
        from esdc.esdc import _print_hit_rate

        _print_hit_rate(0, 0)

    def test_high_hit_rate(self):
        from esdc.esdc import _print_hit_rate

        _print_hit_rate(90, 10)

    def test_low_hit_rate(self):
        from esdc.esdc import _print_hit_rate

        _print_hit_rate(10, 90)


class TestChatCommandCleanup:
    """chat command has no --setup flag and clean guard logic."""

    def test_chat_help_has_no_setup_flag(self):
        from typer.testing import CliRunner

        from esdc.esdc import app

        runner = CliRunner()
        result = runner.invoke(app, ["chat", "--help"])
        assert result.exit_code == 0
        assert "--setup" not in result.output

    def test_chat_without_config_points_to_configs(self, monkeypatch):
        from typer.testing import CliRunner

        from esdc.configs import Config
        from esdc.esdc import app

        monkeypatch.setattr(Config, "has_chat_config", staticmethod(lambda: False))
        runner = CliRunner()
        result = runner.invoke(app, ["chat"])
        assert result.exit_code == 0
        assert "esdc configs" in result.output


class TestServeCommandCleanup:
    def test_serve_help_has_no_web_flag(self):
        from typer.testing import CliRunner

        from esdc.esdc import app

        runner = CliRunner()
        result = runner.invoke(app, ["serve", "--help"])
        assert result.exit_code == 0
        assert "--web" not in result.output
        assert "--port" in result.output


def test_portal_command_invokes_run_portal(monkeypatch):
    from typer.testing import CliRunner
    from esdc.esdc import app

    called = {}

    def fake_run(host, port, log_level):
        called.update(host=host, port=port, log_level=log_level)

    monkeypatch.setattr("esdc.portal.app.run_portal", fake_run)
    result = CliRunner().invoke(app, ["portal"])
    assert result.exit_code == 0
    assert called == {"host": "127.0.0.1", "port": 13334, "log_level": "info"}
