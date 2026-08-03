import re

import pytest
from typer.testing import CliRunner

# SGR escape sequences (color/bold). On CI, Typer's Rich help renders in
# terminal mode and emits these codes even into CliRunner's non-tty buffer,
# splitting flag strings like "--from-excel" across escape codes and breaking
# plain substring assertions (they pass locally where color is off).
_ANSI_SGR_RE = re.compile(r"\x1b\[[0-9;]*m")

# Patch CliRunner.invoke on the class in place so every runner -- the `runner`
# fixture AND tests that build their own CliRunner() -- yields ANSI-stripped
# output. All test modules share this one class object via
# `from typer.testing import CliRunner`, so an in-place method patch reaches
# them regardless of import order (rebinding the symbol would not).
_orig_invoke = CliRunner.invoke


def _invoke_strip_ansi(self, *args, **kwargs):
    result = _orig_invoke(self, *args, **kwargs)
    # Result exposes several byte buffers depending on Click version:
    # stdout_bytes backs `.stdout`, output_bytes backs `.output` (stdout+stderr
    # combined), stderr_bytes backs `.stderr`. Strip SGR codes from each so
    # every accessor is color-agnostic.
    for attr in ("stdout_bytes", "output_bytes", "stderr_bytes"):
        raw = getattr(result, attr, None)
        if raw:
            stripped = _ANSI_SGR_RE.sub("", raw.decode("utf-8", "replace"))
            setattr(result, attr, stripped.encode("utf-8"))
    return result


CliRunner.invoke = _invoke_strip_ansi


@pytest.fixture
def runner():
    """Create a fresh CliRunner for each test."""
    return CliRunner()


@pytest.fixture(autouse=True)
def reset_config_cache():
    """Reset Config cache before each test."""
    from esdc.configs import Config

    Config._config_cache = None
    yield


@pytest.fixture(autouse=True)
def reset_semantic_resolver_cache():
    """Reset the cached SemanticResolver before and after each test.

    esdc.chat.tools._get_semantic_resolver() caches one instance at module
    scope so its DB-signature pin memo survives across tool calls in
    production. Tests that patch
    ``esdc.search.semantic_resolver.SemanticResolver`` (real or Mock) need a
    clean slate each time -- otherwise a resolver instance built by an
    earlier test leaks into a later one that expects its own patched class
    to be used.
    """
    import esdc.chat.tools as tools_mod

    tools_mod._semantic_resolver = None
    yield
    tools_mod._semantic_resolver = None


@pytest.fixture(autouse=True)
def _mock_provider_config(request):
    """Prevent accidental real LLM calls in tests.

    Patches get_provider_config in agent_wrapper where
    generate_response/generate_streaming_response call it. Does NOT
    patch esdc.configs globally so that config tests still work.

    Skip this fixture with @pytest.mark.allow_provider_config marker.
    """
    from unittest.mock import patch

    marker = request.node.get_closest_marker("allow_provider_config")
    if marker is not None:
        yield
        return

    with patch(
        "esdc.server.agent_wrapper.Config.get_provider_config",
        return_value=None,
    ):
        yield


@pytest.fixture
def isolated_config(tmp_path, monkeypatch):
    """Create isolated config directory for integration tests.

    Sets ESDC_CONFIG_DIR to a temporary directory and ESDC_DB_FILE
    to avoid polluting the user's actual ~/.esdc/ directory.
    """
    config_dir = tmp_path / ".esdc"
    db_file = config_dir / "esdc.duckdb"
    monkeypatch.setenv("ESDC_CONFIG_DIR", str(config_dir))
    monkeypatch.setenv("ESDC_DB_FILE", str(db_file))
    yield tmp_path


@pytest.fixture
def mock_esdc_api():
    r"""Factory to add mock ESDC API responses.

    Usage:
        mock = mock_esdc_api()
        mock.add_json("https://esdc.skkmigas.go.id/api/v2/...", {...})
        mock.add_csv("https://esdc.skkmigas.go.id/api/v2/...", "col1,col2\nval1,val2")
    """
    import re

    import responses as responses_lib

    responses_lib.start()

    def _add_response(method, pattern, **kwargs):
        url_pattern = re.compile(pattern)
        responses_lib.add(method, url_pattern, **kwargs)

    class MockAPI:
        def add_json(self, pattern, json_data, status=200):
            _add_response(responses_lib.GET, pattern, json=json_data, status=status)

        def add_csv(self, pattern, body, status=200):
            _add_response(responses_lib.GET, pattern, body=body, status=status)

        def add_status(self, pattern, status):
            _add_response(responses_lib.GET, pattern, status=status)

    yield MockAPI()

    responses_lib.stop()
    responses_lib.reset()


@pytest.fixture
def seeded_database(isolated_config):
    """Create database with sample project_resources data.

    Returns the tmp_path so tests can reference the database file.
    """
    import duckdb

    from esdc.configs import Config

    Config.init_config()

    db_file = Config.get_db_file()
    db_dir = Config.get_db_dir()
    if not db_dir.exists():
        db_dir.mkdir(parents=True, exist_ok=True)

    conn = duckdb.connect(str(db_file))
    try:
        conn.execute("""
            CREATE SEQUENCE IF NOT EXISTS project_resources_id_seq START 1;
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS project_resources (
                id INTEGER DEFAULT nextval('project_resources_id_seq'),
                report_date TEXT,
                report_year INTEGER,
                report_status TEXT,
                project_name TEXT,
                project_stage TEXT,
                project_class TEXT,
                project_level TEXT,
                uncert_level TEXT,
                wk_name TEXT,
                field_name TEXT,
                rec_oc_risked REAL,
                rec_an_risked REAL,
                res_oc REAL,
                res_an REAL,
                prj_ioip REAL,
                prj_igip REAL
            );
        """)
        conn.executemany(
            "INSERT INTO project_resources (report_date, report_year, report_status, project_name, project_stage, project_class, project_level, uncert_level, wk_name, field_name, rec_oc_risked, rec_an_risked, res_oc, res_an, prj_ioip, prj_igip) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",  # noqa: E501
            [
                (
                    "2024-01-01",
                    2024,
                    "ACTIVE",
                    "Test Project Alpha",
                    "DISCOVERED",
                    "CLASS A",
                    "LEVEL 1",
                    "UNCERT-HIGH",
                    "Work Area Alpha",
                    "Field Alpha",
                    100.0,
                    50.0,
                    1000.0,
                    500.0,
                    100.0,
                    50.0,
                ),
                (
                    "2024-01-01",
                    2024,
                    "ACTIVE",
                    "Test Project Beta",
                    "DEVELOPMENT",
                    "CLASS B",
                    "LEVEL 2",
                    "UNCERT-MED",
                    "Work Area Beta",
                    "Field Beta",
                    200.0,
                    100.0,
                    2000.0,
                    1000.0,
                    200.0,
                    100.0,
                ),
                (
                    "2023-01-01",
                    2023,
                    "ACTIVE",
                    "Test Project Gamma",
                    "PRODUCTION",
                    "CLASS C",
                    "LEVEL 3",
                    "UNCERT-LOW",
                    "Work Area Gamma",
                    "Field Gamma",
                    300.0,
                    150.0,
                    3000.0,
                    1500.0,
                    300.0,
                    150.0,
                ),
            ],
        )
    finally:
        conn.close()

    return isolated_config
