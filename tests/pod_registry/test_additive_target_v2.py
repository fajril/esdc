"""S2d1b: additive initialization on a target that carries no POD registry.

Every database here is an owned fixture under ``tmp_path``; the operational store and the
reference seed are never opened. Preparation and the public CLI dry-run stay read-only and
create nothing, an old or partial POD footprint refuses before any writer or DDL, and a
valid no-POD target initializes and applies inside the one common atomic transaction. An
injected late failure rolls initialization and the imported rows back together. Every
SQLite handle opened by a test is closed explicitly, and the default global Config is
isolated so an unguarded run cannot write outside ``tmp_path``.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from esdc.configs import Config
from esdc.esdc import app
from esdc.loaders import generate_pod_workbook_template
from esdc.pod_registry import sqlite_import
from esdc.pod_registry.contract import load_pod_contract
from esdc.pod_registry.migrations import initialize_registry_v2
from esdc.pod_registry.sqlite_import import (
    PodImportError,
    load_prepared_pod_batch,
    prepare_pod_import,
)
from esdc.pod_registry.store import (
    ensure_project_resources_table,
    ensure_snapshot_table,
    get_esdc_sqlite_path,
)
from esdc.timeseries_snapshot import ensure_timeseries_tables
from tests.pod_registry import test_importer

ACTOR = "FA:test-additive-target"
INPUT_TABLES = test_importer.INPUT_TABLES
DOCUMENTS_ROW = ("KEEP", "hash-keep", "# Preserve")
APP_ROW = ("KEEP", "unrelated app fact")


@pytest.fixture(autouse=True)
def isolated_config_for_targets(
    isolated_config: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[Path]:
    """Resolve every Config lookup inside this test's own tmp_path.

    ``isolated_config`` already points the environment at tmp_path; the classmethods are
    patched as well so an unguarded pytest run cannot reach the default global Config
    through a helper that reads Config directly instead of the environment.
    """
    db_dir = isolated_config / ".esdc"
    cache_dir = isolated_config / "cache"
    monkeypatch.setattr(Config, "get_db_dir", classmethod(lambda cls, *a, **k: db_dir))
    monkeypatch.setattr(
        Config, "get_db_file", classmethod(lambda cls, *a, **k: db_dir / "esdc.duckdb")
    )
    monkeypatch.setattr(
        Config, "get_cache_dir", classmethod(lambda cls, *a, **k: cache_dir)
    )
    yield isolated_config


def _writer(path: Path) -> sqlite3.Connection:
    """Test setup only: an owned fixture under tmp_path, never the active database."""
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _reader(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)


def _objects(conn: sqlite3.Connection) -> set[str]:
    return {
        str(row[0])
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' "
            "AND name NOT LIKE 'sqlite_%'"
        )
    }


def _application_fixture(path: Path) -> None:
    conn = _writer(path)
    try:
        conn.execute(
            "CREATE TABLE documents(doc_id TEXT PRIMARY KEY, file_hash TEXT, markdown TEXT)"
        )
        conn.execute("INSERT INTO documents VALUES (?, ?, ?)", DOCUMENTS_ROW)
        conn.execute(
            "CREATE TABLE custom_app_cache(entity_id TEXT PRIMARY KEY, note TEXT)"
        )
        conn.execute("INSERT INTO custom_app_cache VALUES (?, ?)", APP_ROW)
        conn.commit()
    finally:
        conn.close()


def _workbook(tmp_path: Path, name: str, tables: tuple[str, ...]) -> Path:
    contract = load_pod_contract()
    rows = {table: [test_importer._row(contract, table)] for table in tables}
    return test_importer._write_workbook(tmp_path / name, contract, rows)


def test_documents_only_target_loads_additively_and_preserves_rows(
    tmp_path: Path,
) -> None:
    """The actual public additive RED (gate4ytfuuzt): prepare refused a safe no-POD target.

    Prepare must not treat existing application tables as a legacy footprint, and the load
    initializes the canonical schema and applies the incoming rows in one transaction.
    """
    source = tmp_path / "input.xlsx"
    target = tmp_path / "documents.sqlite"
    cache = tmp_path / "esdc.duckdb"
    _application_fixture(target)
    generate_pod_workbook_template(output_path=source)

    batch = prepare_pod_import(
        source, "xlsx", target_path=target, stage_snapshots=False
    )
    assert batch.target_basis is not None
    assert batch.target_basis.existed is True
    assert batch.target_basis.source_generation is None
    reader = _reader(target)
    try:
        assert _objects(reader) == {"documents", "custom_app_cache"}, (
            "prepare must not create a registry"
        )
    finally:
        reader.close()

    load_prepared_pod_batch(batch, sqlite_path=target, duck_path=cache)
    reader = _reader(target)
    try:
        assert reader.execute("SELECT * FROM documents").fetchall() == [DOCUMENTS_ROW]
        assert reader.execute(
            "SELECT source_generation FROM pod_registry_state WHERE singleton_id=1"
        ).fetchone() == (1,)
        assert reader.execute("SELECT COUNT(*) FROM m_pod").fetchone()[0] == 0
    finally:
        reader.close()


def test_canonical_raw_only_target_loads_additively_and_preserves_rows(
    tmp_path: Path,
) -> None:
    """Canonical raw API source tables alone are an additive target, not a legacy shape."""
    source = _workbook(tmp_path, "pod.xlsx", INPUT_TABLES)
    target = tmp_path / "raw-only.sqlite"
    writer = _writer(target)
    try:
        ensure_project_resources_table(writer)
        ensure_snapshot_table(writer)
        ensure_timeseries_tables(writer)
        writer.execute(
            "INSERT INTO api_project_resources (project_id, report_year, wk_name) "
            "VALUES ('P-0000777-77', 2024, 'Preserved work area')"
        )
        writer.commit()
        raw_before = _objects(writer)
    finally:
        writer.close()
    assert {
        "api_project_resources",
        "api_snapshot",
        "api_project_timeseries",
    } <= raw_before

    batch = prepare_pod_import(
        source, "xlsx", target_path=target, stage_snapshots=False
    )
    load_prepared_pod_batch(
        batch,
        sqlite_path=target,
        duck_path=tmp_path / "projection.duckdb",
        publish=False,
    )

    reader = _reader(target)
    try:
        assert reader.execute(
            "SELECT project_id, report_year, wk_name FROM api_project_resources"
        ).fetchall() == [("P-0000777-77", 2024, "Preserved work area")]
        assert raw_before <= _objects(reader), "raw source tables are never replaced"
        assert reader.execute("SELECT COUNT(*) FROM m_pod").fetchone()[0] == 1
        assert reader.execute(
            "SELECT source_generation FROM pod_registry_state WHERE singleton_id=1"
        ).fetchone() == (1,)
    finally:
        reader.close()


def test_unrelated_application_table_and_imported_rows_survive_an_additive_load(
    tmp_path: Path,
) -> None:
    """An unrelated application table is preserved while the incoming batch is applied."""
    source = _workbook(tmp_path, "pod.xlsx", INPUT_TABLES)
    target = tmp_path / "application.sqlite"
    _application_fixture(target)

    batch = prepare_pod_import(
        source, "xlsx", target_path=target, stage_snapshots=False
    )
    result, publication = load_prepared_pod_batch(
        batch,
        sqlite_path=target,
        duck_path=tmp_path / "projection.duckdb",
        actor=ACTOR,
        publish=False,
    )
    assert publication is None
    assert result.source_committed
    assert result.table_counts == dict.fromkeys(INPUT_TABLES, 1)

    reader = _reader(target)
    try:
        assert reader.execute("SELECT * FROM custom_app_cache").fetchall() == [APP_ROW]
        assert reader.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 1
        assert reader.execute("SELECT COUNT(*) FROM m_pod").fetchone()[0] == 1
    finally:
        reader.close()


def test_unrelated_application_view_is_not_a_pod_shape(tmp_path: Path) -> None:
    """Only reserved registry slots are POD objects; an application view stays additive."""
    source = _workbook(tmp_path, "pod.xlsx", INPUT_TABLES)
    target = tmp_path / "app-view.sqlite"
    conn = _writer(target)
    try:
        conn.execute("CREATE TABLE documents(doc_id TEXT PRIMARY KEY)")
        conn.execute("INSERT INTO documents VALUES ('KEEP')")
        conn.execute("CREATE VIEW app_document_summary AS SELECT doc_id FROM documents")
        conn.commit()
    finally:
        conn.close()

    batch = prepare_pod_import(
        source, "xlsx", target_path=target, stage_snapshots=False
    )
    assert batch.target_basis is not None
    assert batch.target_basis.source_generation is None

    load_prepared_pod_batch(
        batch,
        sqlite_path=target,
        duck_path=tmp_path / "projection.duckdb",
        publish=False,
    )
    reader = _reader(target)
    try:
        assert reader.execute("SELECT * FROM app_document_summary").fetchall() == [
            ("KEEP",)
        ]
        assert reader.execute("SELECT COUNT(*) FROM m_pod").fetchone()[0] == 1
    finally:
        reader.close()


def test_empty_and_missing_targets_still_load_normally(tmp_path: Path) -> None:
    """The normal fresh-target path is unchanged, and prepare still creates nothing."""
    source = _workbook(tmp_path, "pod.xlsx", INPUT_TABLES)
    missing = tmp_path / "missing" / "target.sqlite"
    empty = tmp_path / "empty.sqlite"
    empty.write_bytes(b"")

    for path in (missing, empty):
        batch = prepare_pod_import(
            source, "xlsx", target_path=path, stage_snapshots=False
        )
        if path is missing:
            assert not path.exists(), "prepare must not create the target directory"
        else:
            assert path.read_bytes() == b"", "prepare stays read-only"

        result = load_prepared_pod_batch(
            batch,
            sqlite_path=path,
            duck_path=tmp_path / "projection.duckdb",
            actor=ACTOR,
            publish=False,
        )[0]
        assert result.source_generation == 1
        reader = _reader(path)
        try:
            assert reader.execute("SELECT COUNT(*) FROM m_pod").fetchone()[0] == 1
        finally:
            reader.close()


def test_master_only_profile_on_a_no_pod_target_stays_consistent(
    tmp_path: Path,
) -> None:
    """A no-POD target has no preserved input context; the master-only profile still applies."""
    source = _workbook(tmp_path, "master.xlsx", ("m_project", "m_pod", "pod_project"))
    target = tmp_path / "master-only.sqlite"
    _application_fixture(target)

    batch = prepare_pod_import(
        source,
        "xlsx",
        profile="master-only",
        target_path=target,
        stage_snapshots=False,
    )
    assert batch.profile == "master-only"
    result = load_prepared_pod_batch(
        batch,
        sqlite_path=target,
        duck_path=tmp_path / "projection.duckdb",
        actor=ACTOR,
        publish=False,
    )[0]

    assert sum(result.table_counts.values()) == 3
    reader = _reader(target)
    try:
        assert reader.execute("SELECT * FROM custom_app_cache").fetchall() == [APP_ROW]
        assert reader.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 1
        assert reader.execute("SELECT COUNT(*) FROM pod_monitoring").fetchone()[0] == 0
    finally:
        reader.close()


def test_previously_current_registry_still_imports_on_its_prior_path(
    tmp_path: Path,
) -> None:
    """Fix1 finding 1: a batch prepared against a current registry is not read as no-POD."""
    source = _workbook(tmp_path, "pod.xlsx", INPUT_TABLES)
    target = tmp_path / "current.sqlite"
    writer = _writer(target)
    try:
        initialize_registry_v2(writer)
        writer.commit()
    finally:
        writer.close()

    batch = prepare_pod_import(
        source, "xlsx", target_path=target, stage_snapshots=False
    )
    assert batch.target_basis is not None
    assert batch.target_basis.source_generation == 0

    result = load_prepared_pod_batch(
        batch,
        sqlite_path=target,
        duck_path=tmp_path / "projection.duckdb",
        actor=ACTOR,
        publish=False,
    )[0]
    assert result.source_generation == 1
    reader = _reader(target)
    try:
        assert reader.execute("SELECT COUNT(*) FROM m_pod").fetchone()[0] == 1
    finally:
        reader.close()


@pytest.mark.parametrize(
    "name", ["m_pod", "m_project", "pod_registry_state", "M_POD", "M_PROJECT"]
)
def test_reserved_pod_view_is_refused_before_any_writer(
    tmp_path: Path, name: str
) -> None:
    """Fix1 finding 2: a view in a reserved slot is a partial footprint, case-insensitive."""
    source = _workbook(tmp_path, "pod.xlsx", INPUT_TABLES)
    target = tmp_path / "view-shape.sqlite"
    conn = _writer(target)
    try:
        conn.execute("CREATE TABLE documents(doc_id TEXT PRIMARY KEY)")
        conn.execute(f'CREATE VIEW "{name}" AS SELECT 1 AS id')
        conn.commit()
    finally:
        conn.close()
    before = target.read_bytes()

    # Both accepted policy routes refuse before any writer: the shared classifier reports
    # the reserved slot, and the registry-state reader requires its canonical table.
    with pytest.raises(
        PodImportError, match="not the canonical table|must be a table"
    ) as error:
        prepare_pod_import(source, "xlsx", target_path=target, stage_snapshots=False)

    assert name in str(error.value)
    assert target.read_bytes() == before, "no writer, DDL or journal change on refusal"
    assert {p.name for p in tmp_path.iterdir()} == {"pod.xlsx", "view-shape.sqlite"}


@pytest.mark.parametrize("name", ["r_institution", "r_pod_type", "M_PROJECT"])
def test_partial_reserved_table_is_refused_before_any_writer(
    tmp_path: Path, name: str
) -> None:
    """A reserved reference table without registry metadata is partial POD, not an app table."""
    source = _workbook(tmp_path, "pod.xlsx", INPUT_TABLES)
    target = tmp_path / "partial-table.sqlite"
    conn = _writer(target)
    try:
        conn.execute("CREATE TABLE documents(doc_id TEXT PRIMARY KEY)")
        conn.execute(f'CREATE TABLE "{name}" (id INTEGER)')
        conn.commit()
    finally:
        conn.close()
    before = target.read_bytes()

    with pytest.raises(PodImportError, match="explicit validated replacement") as error:
        prepare_pod_import(source, "xlsx", target_path=target, stage_snapshots=False)

    assert name.lower() in str(error.value).casefold()
    assert target.read_bytes() == before, "no writer, DDL or journal change on refusal"
    probe = sqlite3.connect(target.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        assert probe.execute("PRAGMA journal_mode").fetchone()[0] == "delete"
        assert probe.execute("PRAGMA foreign_keys").fetchone()[0] == 0
    finally:
        probe.close()
    assert not (tmp_path / "partial-table.sqlite-wal").exists()
    assert {p.name for p in tmp_path.iterdir()} == {"pod.xlsx", "partial-table.sqlite"}


@pytest.mark.parametrize(
    ("name", "sql", "expected", "names_crosswalk"),
    [
        (
            "legacy.sqlite",
            "CREATE TABLE m_pod (id INTEGER PRIMARY KEY, pod_id TEXT);",
            "legacy POD registry",
            True,
        ),
        (
            "partial.sqlite",
            "CREATE TABLE documents(doc_id TEXT PRIMARY KEY); "
            "CREATE TABLE m_project (project_id TEXT PRIMARY KEY);",
            "v2 target table exists without registry metadata: m_project",
            False,
        ),
    ],
)
def test_old_or_partial_pod_footprint_refuses_before_any_writer(
    tmp_path: Path, name: str, sql: str, expected: str, names_crosswalk: bool
) -> None:
    """The shared policy still refuses a partial footprint, without touching the target."""
    source = _workbook(tmp_path, "pod.xlsx", INPUT_TABLES)
    target = tmp_path / name
    conn = _writer(target)
    try:
        conn.executescript(sql)
        conn.commit()
    finally:
        conn.close()
    before = target.read_bytes()

    if names_crosswalk:
        # Full replacement of a recognized old target with no protected links is
        # now permitted; its prepare phase is still read-only.
        batch = prepare_pod_import(
            source, "xlsx", target_path=target, stage_snapshots=False
        )
        assert batch.replacement_plan is not None
        assert target.read_bytes() == before
        return

    with pytest.raises(PodImportError, match=expected) as error:
        prepare_pod_import(source, "xlsx", target_path=target, stage_snapshots=False)

    # Every refusal names the explicit validated replacement; only a legacy footprint can
    # be reconciled by an approved crosswalk, and numeric POD IDs are never inferred.
    assert "explicit validated replacement" in str(error.value)
    if names_crosswalk:
        assert "approved_crosswalk" in str(error.value)
        assert "numeric POD IDs are never inferred" in str(error.value)
    assert target.read_bytes() == before
    probe = sqlite3.connect(target.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        assert probe.execute("PRAGMA journal_mode").fetchone()[0] == "delete"
        assert probe.execute("PRAGMA foreign_keys").fetchone()[0] == 0
    finally:
        probe.close()
    assert not (tmp_path / f"{name}-wal").exists()
    assert not (tmp_path / f"{name}-shm").exists()
    assert {p.name for p in tmp_path.iterdir()} == {"pod.xlsx", name}


def test_concurrent_initialization_after_prepare_is_refused(tmp_path: Path) -> None:
    """A prepared no-POD target that another writer initialized is changed, not overwriteable."""
    source = _workbook(tmp_path, "pod.xlsx", INPUT_TABLES)
    target = tmp_path / "concurrent.sqlite"
    _application_fixture(target)

    batch = prepare_pod_import(
        source, "xlsx", target_path=target, stage_snapshots=False
    )

    other = _writer(target)
    try:
        initialize_registry_v2(other)
        other.commit()
    finally:
        other.close()

    with pytest.raises(PodImportError, match="changed after import preparation"):
        load_prepared_pod_batch(
            batch,
            sqlite_path=target,
            duck_path=tmp_path / "projection.duckdb",
            actor=ACTOR,
            publish=False,
        )

    reader = _reader(target)
    try:
        assert reader.execute("SELECT * FROM custom_app_cache").fetchall() == [APP_ROW]
        assert reader.execute(
            "SELECT source_generation FROM pod_registry_state WHERE singleton_id=1"
        ).fetchone() == (0,), "the other writer's initialization survives"
        assert reader.execute("SELECT COUNT(*) FROM m_pod").fetchone()[0] == 0
    finally:
        reader.close()


def test_partial_footprint_appearing_after_prepare_refuses_before_ddl(
    tmp_path: Path,
) -> None:
    """A shape change between prepare and load is refused, not treated as additive."""
    source = _workbook(tmp_path, "pod.xlsx", INPUT_TABLES)
    target = tmp_path / "late-shape.sqlite"
    _application_fixture(target)

    batch = prepare_pod_import(
        source, "xlsx", target_path=target, stage_snapshots=False
    )

    other = sqlite3.connect(target)
    try:
        other.execute("CREATE TABLE m_project (project_id TEXT PRIMARY KEY)")
        other.commit()
    finally:
        other.close()

    with pytest.raises(PodImportError, match="explicit validated replacement"):
        load_prepared_pod_batch(
            batch,
            sqlite_path=target,
            duck_path=tmp_path / "projection.duckdb",
            actor=ACTOR,
            publish=False,
        )

    reader = _reader(target)
    try:
        assert _objects(reader) == {"documents", "custom_app_cache", "m_project"}
        assert reader.execute("SELECT * FROM custom_app_cache").fetchall() == [APP_ROW]
    finally:
        reader.close()


def test_injected_late_refresh_failure_rolls_back_initialization_and_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Initialization and the imported rows commit together, so a late failure undoes both."""
    source = _workbook(tmp_path, "pod.xlsx", INPUT_TABLES)
    target = tmp_path / "rollback.sqlite"
    _application_fixture(target)
    before = {"documents", "custom_app_cache"}

    def fail_refresh(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("injected late refresh failure")

    monkeypatch.setattr(sqlite_import, "refresh_pod_after_source_change", fail_refresh)

    batch = prepare_pod_import(
        source, "xlsx", target_path=target, stage_snapshots=False
    )
    with pytest.raises(RuntimeError, match="injected late refresh failure"):
        load_prepared_pod_batch(
            batch,
            sqlite_path=target,
            duck_path=tmp_path / "projection.duckdb",
            actor=ACTOR,
            publish=False,
        )

    reader = _reader(target)
    try:
        assert _objects(reader) == before, "no partial registry is left behind"
        assert reader.execute("SELECT * FROM custom_app_cache").fetchall() == [APP_ROW]
    finally:
        reader.close()


def test_public_cli_dry_run_on_a_no_pod_target_writes_nothing(
    runner: CliRunner, isolated_config: Path, tmp_path: Path
) -> None:
    """Dry-run prepares against the operational target without a writer or a cache file."""
    source = _workbook(tmp_path, "pod.xlsx", INPUT_TABLES)
    operational = get_esdc_sqlite_path()
    assert operational == isolated_config / ".esdc" / "esdc.sqlite"
    operational.parent.mkdir(parents=True, exist_ok=True)
    _application_fixture(operational)
    before = operational.read_bytes()

    result = runner.invoke(
        app, ["load", "--from-excel", str(source), "--schema-pod", "--dry-run"]
    )

    assert result.exit_code == 0, result.output
    assert "m_pod: 1 rows" in result.output
    assert "target was not opened for writing" in result.output
    assert operational.read_bytes() == before
    assert not Config.get_db_file().exists()
    probe = sqlite3.connect(operational.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        assert _objects(probe) == {"documents", "custom_app_cache"}
        assert probe.execute("PRAGMA journal_mode").fetchone()[0] == "delete"
    finally:
        probe.close()
