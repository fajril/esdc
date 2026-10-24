"""API-pair analytics recovery tests backed by real SQLite and DuckDB files."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Iterable
from pathlib import Path
from typing import Literal

import duckdb
import pytest

from esdc.api_snapshot import (
    ApiProjectResourceRow,
    ApiSnapshot,
    get_project_resources_snapshot,
    replace_project_resources_snapshot,
)
from esdc.configs import Config
from esdc.fetching import PreparedFetch, spool_timeseries
from esdc.pod_registry.store import get_sqlite_connection
from esdc.timeseries_snapshot import (
    normalize_timeseries_row,
    record_observed_empty_timeseries,
    record_timeseries_provenance,
    replace_timeseries_in_transaction,
)


@pytest.fixture(autouse=True)
def isolated_esdc_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ESDC_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setenv("ESDC_DB_DIR", str(tmp_path / "db"))
    monkeypatch.setenv("ESDC_DB_FILE", str(tmp_path / "db" / "active.duckdb"))
    Config._config_cache = None


def _seed_pair(
    sqlite_path: Path,
    *,
    timeseries_row: bool = True,
    project_name: str = "Canonical project",
    field_geom: str | None = None,
    project_class: str | None = None,
    project_level: str | None = None,
    wk_id: str | int | None = None,
    rec_oil: str | float | None = None,
) -> tuple[str, str]:
    source = get_sqlite_connection(sqlite_path)
    replace_project_resources_snapshot(
        source,
        [
            ApiProjectResourceRow(
                project_id="P-1",
                report_year=2025,
                project_name=project_name,
                field_name="Field one",
                field_geom=field_geom,
                project_class=project_class,
                project_level=project_level,
                wk_id=wk_id,
                rec_oil=rec_oil,
            )
        ],
        ApiSnapshot(
            "project_resources",
            "resources-r1",
            1,
            "",
            "2026-10-24T00:00:00Z",
            "committed",
        ),
    )
    resource_snapshot = get_project_resources_snapshot(source)
    assert resource_snapshot is not None
    source.execute("BEGIN")
    if timeseries_row:
        timeseries = normalize_timeseries_row(
            ("report_year", "project_id", "tpf_oc"),
            (2025, "P-1", 12.5),
            origin="api_json",
        )
        ts_snapshot = replace_timeseries_in_transaction(
            source,
            [timeseries],
            years=None,
            revision="timeseries-r1",
            fetched_at="2026-10-24T00:00:00Z",
        )
        record_timeseries_provenance(
            source,
            revision=ts_snapshot.revision,
            origin="api_json",
            details={"input_origins": ["api_json"], "observed_empty": False},
        )
    else:
        ts_snapshot = record_observed_empty_timeseries(
            source,
            revision="timeseries-empty-r1",
            fetched_at="2026-10-24T00:00:00Z",
            origin="api_json",
        )
    source.commit()
    source.close()
    return resource_snapshot.revision, ts_snapshot.revision


def _analytics_rows(path: Path, table: str) -> list[tuple[object, ...]]:
    connection = duckdb.connect(str(path), read_only=True)
    try:
        return connection.execute(f"SELECT * FROM {table}").fetchall()
    finally:
        connection.close()


def test_rebuild_pair_without_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from esdc.analytics_rebuild import rebuild_api_analytics

    sqlite_path = tmp_path / "source.sqlite"
    duckdb_path = tmp_path / "analytics.duckdb"
    resource_revision, timeseries_revision = _seed_pair(sqlite_path)
    target = duckdb.connect(str(duckdb_path))
    target.execute("CREATE TABLE user_table (value VARCHAR)")
    target.execute("INSERT INTO user_table VALUES ('preserve me')")
    target.close()

    def no_network(*_: object, **__: object) -> None:
        raise AssertionError("API recovery attempted a network request")

    monkeypatch.setattr("requests.get", no_network)
    report = rebuild_api_analytics(
        duckdb_path=duckdb_path, sqlite_path=sqlite_path, reindex=False
    )
    from esdc.fetch_publication import ensure_timeseries_baseline

    assert ensure_timeseries_baseline(duckdb_path, sqlite_path) is not None

    assert report.output_path == duckdb_path
    assert report.source_revisions == {
        "project_resources": resource_revision,
        "project_timeseries": timeseries_revision,
    }
    assert report.subsystem_status == {"api": "restored"}
    assert len(_analytics_rows(duckdb_path, "project_resources")) == 1
    assert len(_analytics_rows(duckdb_path, "project_timeseries")) == 1
    assert _analytics_rows(duckdb_path, "user_table") == [("preserve me",)]


def test_bootstrap_incremental_precision_survives_api_free_recovery(
    tmp_path: Path,
) -> None:
    from esdc.analytics_rebuild import rebuild_api_analytics
    from esdc.fetch_publication import publish_prepared_fetch
    from esdc.timeseries_bootstrap import bootstrap_timeseries_source
    from esdc.timeseries_snapshot import get_timeseries_snapshot

    sqlite_path = tmp_path / "canonical.sqlite"
    original_duckdb = tmp_path / "legacy.duckdb"
    recovered_duckdb = tmp_path / "recovered.duckdb"
    legacy = duckdb.connect(str(original_duckdb))
    from esdc.dbmanager import _execute_sql_script

    _execute_sql_script(legacy, "create_table_project_timeseries.sql")
    legacy.execute(
        "ALTER TABLE project_timeseries ALTER COLUMN report_year TYPE HUGEINT"
    )
    legacy.execute(
        "ALTER TABLE project_timeseries ALTER COLUMN year TYPE HUGEINT"
    )
    legacy.execute(
        "ALTER TABLE project_timeseries ALTER COLUMN cprd_grs_oil "
        "TYPE DECIMAL(30,4)"
    )
    legacy.executemany(
        "INSERT INTO project_timeseries "
        "(report_year, project_id, year, cprd_grs_oil) VALUES (?, ?, ?, ?)",
        [
            (2024, "legacy-2024", 9223372036854775808, "12345678901234567890.1234"),
            (2025, "legacy-2025", 9223372036854775809, "12345678901234567890.1234"),
            (None, "legacy-null-year", 9223372036854775810, "12345678901234567890.1234"),
        ],
    )
    legacy.close()

    bootstrap = bootstrap_timeseries_source(
        duckdb_path=original_duckdb, sqlite_path=sqlite_path, apply=True
    )
    assert bootstrap.applied is True
    assert bootstrap.null_year_rows == 1
    assert bootstrap.source_types["year"] == "HUGEINT"
    assert bootstrap.source_types["cprd_grs_oil"] == "DECIMAL(30,4)"

    source = get_sqlite_connection(sqlite_path)
    replace_project_resources_snapshot(
        source,
        [ApiProjectResourceRow(project_id="api-resource", report_year=2024)],
        ApiSnapshot(
            "project_resources",
            "resources-before-incremental",
            1,
            "",
            "2026-10-24T00:00:00Z",
            "committed",
        ),
    )
    source.close()

    prepared_slice = spool_timeseries(
        ("report_year", "project_id", "year", "cprd_grs_oil"),
        ((2024, "api-2024", 2024, 7),),
        report_year=2024,
        directory=tmp_path / "spool",
    )
    assert prepared_slice is not None
    outcome = publish_prepared_fetch(
        PreparedFetch(
            (ApiProjectResourceRow(project_id="api-resource", report_year=2024),),
            None,
            (prepared_slice,),
            (),
            (),
            timeseries_observed_years=(2024,),
            timeseries_complete_observation=True,
        ),
        duckdb_path=original_duckdb,
        sqlite_path=sqlite_path,
        reindex=False,
    )
    assert outcome.timeseries_revision is not None

    # The recovery target is independent of the legacy DuckDB file. Renaming
    # the source keeps the fixture recoverable while making its old path absent.
    original_duckdb.rename(tmp_path / "legacy-unavailable.duckdb")
    report = rebuild_api_analytics(
        duckdb_path=recovered_duckdb,
        sqlite_path=sqlite_path,
        reindex=False,
    )

    source = get_sqlite_connection(sqlite_path)
    snapshot = get_timeseries_snapshot(source)
    source.close()
    assert snapshot is not None
    assert report.source_revisions["project_timeseries"] == snapshot.revision
    assert report.source_hashes["project_timeseries"] == snapshot.source_sha256
    assert original_duckdb.exists() is False
    recovered = duckdb.connect(str(recovered_duckdb), read_only=True)
    try:
        types = dict(
            recovered.execute(
                "SELECT column_name, data_type FROM information_schema.columns "
                "WHERE table_name='project_timeseries'"
            ).fetchall()
        )
        assert types["year"] == "HUGEINT"
        assert types["cprd_grs_oil"] == "DECIMAL(30,4)"
        assert recovered.execute(
            "SELECT project_id, year, CAST(cprd_grs_oil AS VARCHAR) "
            "FROM project_timeseries WHERE project_id LIKE 'legacy-%' "
            "ORDER BY project_id"
        ).fetchall() == [
            ("legacy-2025", 9223372036854775809, "12345678901234567890.1234"),
            ("legacy-null-year", 9223372036854775810, "12345678901234567890.1234"),
        ]
        assert recovered.execute(
            "SELECT project_id FROM project_timeseries WHERE project_id='api-2024'"
        ).fetchall() == [("api-2024",)]
    finally:
        recovered.close()


def test_source_revision_after_c2_is_not_marked_published(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import esdc.analytics_rebuild as rebuild
    from esdc.analytics_rebuild import _ApiSources

    sqlite_path = tmp_path / "source.sqlite"
    duckdb_path = tmp_path / "analytics.duckdb"
    captured_resource_revision, captured_timeseries_revision = _seed_pair(sqlite_path)
    original_mark = rebuild._mark_sources
    advanced = False

    def advance_before_publish_mark(
        path: Path,
        sources: _ApiSources,
        state: Literal["published", "analytics_stale"],
        *,
        invalid_geometry_count: int | None = None,
    ) -> None:
        nonlocal advanced
        if state == "published" and not advanced:
            advanced = True
            source = get_sqlite_connection(path)
            replace_project_resources_snapshot(
                source,
                [
                    ApiProjectResourceRow(
                        project_id="P-1",
                        report_year=2025,
                        project_name="new revision",
                    )
                ],
                ApiSnapshot(
                    "project_resources",
                    "resources-r2",
                    1,
                    "",
                    "2026-10-24T00:01:00Z",
                    "committed",
                ),
            )
            source.close()
        original_mark(
            path,
            sources,
            state,
            invalid_geometry_count=invalid_geometry_count,
        )

    monkeypatch.setattr(rebuild, "_mark_sources", advance_before_publish_mark)
    with pytest.raises(
        RuntimeError,
        match=(
            "committed.*publication marking failed.*canonical publication remains "
            "pending or stale.*rebuild-api-analytics"
        ),
    ) as failure:
        rebuild.rebuild_api_analytics(
            duckdb_path=duckdb_path, sqlite_path=sqlite_path, reindex=False
        )
    assert failure.value.__cause__ is not None
    assert "revision changed before publication marking" in str(failure.value.__cause__)

    target = duckdb.connect(str(duckdb_path), read_only=True)
    try:
        assert target.execute(
            "SELECT source_table, revision FROM api_publication "
            "ORDER BY source_table"
        ).fetchall() == [
            ("project_resources", captured_resource_revision),
            ("project_timeseries", captured_timeseries_revision),
        ]
    finally:
        target.close()
    source = get_sqlite_connection(sqlite_path)
    try:
        resource_state = source.execute(
            "SELECT revision, publication_state FROM api_snapshot "
            "WHERE source_table='project_resources'"
        ).fetchone()
        assert resource_state is not None
        assert resource_state[0] != captured_resource_revision
        assert resource_state[1] == "committed"
        timeseries_state = source.execute(
            "SELECT publication_state FROM api_snapshot "
            "WHERE source_table='project_timeseries'"
        ).fetchone()
        assert timeseries_state is not None
        assert timeseries_state[0] == "committed"
    finally:
        source.close()


def test_post_c2_sqlite_marking_io_failure_is_reported_truthfully(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import esdc.analytics_rebuild as rebuild
    from esdc.analytics_rebuild import _ApiSources

    sqlite_path = tmp_path / "source.sqlite"
    duckdb_path = tmp_path / "analytics.duckdb"
    resource_revision, timeseries_revision = _seed_pair(sqlite_path)
    original_mark = rebuild._mark_sources

    def fail_published_mark(
        path: Path,
        sources: _ApiSources,
        state: Literal["published", "analytics_stale"],
        *,
        invalid_geometry_count: int | None = None,
    ) -> None:
        if state == "published":
            raise sqlite3.OperationalError("injected SQLite marker I/O failure")
        original_mark(
            path,
            sources,
            state,
            invalid_geometry_count=invalid_geometry_count,
        )

    monkeypatch.setattr(rebuild, "_mark_sources", fail_published_mark)
    with pytest.raises(
        RuntimeError,
        match=(
            "DuckDB API analytics committed; publication marking failed; "
            "canonical publication remains pending or stale.*injected SQLite "
            "marker I/O failure"
        ),
    ) as failure:
        rebuild.rebuild_api_analytics(
            duckdb_path=duckdb_path, sqlite_path=sqlite_path, reindex=False
        )
    assert isinstance(failure.value.__cause__, sqlite3.OperationalError)
    target = duckdb.connect(str(duckdb_path), read_only=True)
    try:
        assert target.execute(
            "SELECT source_table, revision FROM api_publication "
            "ORDER BY source_table"
        ).fetchall() == [
            ("project_resources", resource_revision),
            ("project_timeseries", timeseries_revision),
        ]
    finally:
        target.close()
    source = sqlite3.connect(sqlite_path)
    try:
        assert source.execute(
            "SELECT publication_state FROM api_snapshot ORDER BY source_table"
        ).fetchall() == [("analytics_stale",), ("analytics_stale",)]
    finally:
        source.close()


def test_rebuild_uses_same_revisions_and_preserves_other_tables(tmp_path: Path) -> None:
    from esdc.analytics_rebuild import rebuild_api_analytics
    from esdc.api_projection import get_api_publication

    sqlite_path = tmp_path / "source.sqlite"
    duckdb_path = tmp_path / "analytics.duckdb"
    resource_revision, timeseries_revision = _seed_pair(sqlite_path)
    target = duckdb.connect(str(duckdb_path))
    target.execute("CREATE TABLE private_notes (note VARCHAR)")
    target.execute("INSERT INTO private_notes VALUES ('keep')")
    target.close()

    report = rebuild_api_analytics(
        duckdb_path=duckdb_path, sqlite_path=sqlite_path, reindex=False
    )

    target = duckdb.connect(str(duckdb_path), read_only=True)
    try:
        resource_lineage = get_api_publication(target, "project_resources")
        timeseries_lineage = get_api_publication(target, "project_timeseries")
        assert resource_lineage is not None
        assert timeseries_lineage is not None
        assert resource_lineage.revision == resource_revision
        assert timeseries_lineage.revision == timeseries_revision
        assert target.execute("SELECT note FROM private_notes").fetchall() == [
            ("keep",)
        ]
    finally:
        target.close()
    source = sqlite3.connect(sqlite_path)
    try:
        assert source.execute(
            "SELECT source_table, revision, publication_state FROM api_snapshot "
            "ORDER BY source_table"
        ).fetchall() == [
            ("project_resources", resource_revision, "published"),
            ("project_timeseries", timeseries_revision, "published"),
        ]
    finally:
        source.close()
    assert report.source_hashes.keys() == {"project_resources", "project_timeseries"}


def test_missing_or_corrupt_canonical_refuses_without_target_mutation(
    tmp_path: Path,
) -> None:
    from esdc.analytics_rebuild import rebuild_api_analytics

    missing_source = tmp_path / "missing.sqlite"
    absent_target = tmp_path / "absent.duckdb"
    with pytest.raises(FileNotFoundError):
        rebuild_api_analytics(duckdb_path=absent_target, sqlite_path=missing_source)
    assert absent_target.exists() is False

    sqlite_path = tmp_path / "corrupt.sqlite"
    _seed_pair(sqlite_path)
    source = sqlite3.connect(sqlite_path)
    source.execute(
        "UPDATE api_snapshot SET source_sha256='bad' WHERE source_table='project_resources'"
    )
    source.commit()
    source.close()
    with pytest.raises(ValueError, match="count/hash"):
        rebuild_api_analytics(duckdb_path=absent_target, sqlite_path=sqlite_path)
    assert absent_target.exists() is False


@pytest.mark.parametrize(
    ("timeseries_row", "input_origins"),
    [
        (True, [{}]),
        (True, ["api_json", "untrusted"]),
        (False, [{}]),
        (False, ["api_json", "untrusted"]),
    ],
)
def test_malformed_current_timeseries_provenance_refuses_before_target_creation(
    tmp_path: Path, timeseries_row: bool, input_origins: list[object]
) -> None:
    from esdc.analytics_rebuild import rebuild_api_analytics

    sqlite_path = tmp_path / "malformed-provenance.sqlite"
    duckdb_path = tmp_path / "target.duckdb"
    _seed_pair(sqlite_path, timeseries_row=timeseries_row)
    source = sqlite3.connect(sqlite_path)
    details = {
        "input_origins": input_origins,
        "observed_empty": not timeseries_row,
    }
    source.execute(
        "UPDATE api_snapshot_provenance SET details_json=? "
        "WHERE source_table='project_timeseries'",
        (json.dumps(details),),
    )
    source.commit()
    source.close()

    with pytest.raises(ValueError, match="input_origins"):
        rebuild_api_analytics(
            duckdb_path=duckdb_path, sqlite_path=sqlite_path, reindex=False
        )
    assert duckdb_path.exists() is False


def test_rebuild_rolls_back_both_on_second_stage_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import esdc.analytics_rebuild as rebuild

    sqlite_path = tmp_path / "source.sqlite"
    duckdb_path = tmp_path / "analytics.duckdb"
    _seed_pair(sqlite_path)
    rebuild.rebuild_api_analytics(
        duckdb_path=duckdb_path, sqlite_path=sqlite_path, reindex=False
    )
    before_resources = _analytics_rows(duckdb_path, "project_resources")
    before_timeseries = _analytics_rows(duckdb_path, "project_timeseries")
    target = duckdb.connect(str(duckdb_path), read_only=True)
    try:
        before_lineage = target.execute(
            "SELECT source_table, revision, source_sha256 FROM api_publication "
            "ORDER BY source_table"
        ).fetchall()
    finally:
        target.close()
    _seed_pair(sqlite_path, project_name="Changed project")

    def fail_timeseries_stage(*_: object, **__: object) -> object:
        raise RuntimeError("injected second-stage failure")

    monkeypatch.setattr(rebuild, "stage_canonical_timeseries", fail_timeseries_stage)
    with pytest.raises(RuntimeError, match="injected second-stage failure"):
        rebuild.rebuild_api_analytics(
            duckdb_path=duckdb_path, sqlite_path=sqlite_path, reindex=False
        )

    assert _analytics_rows(duckdb_path, "project_resources") == before_resources
    assert _analytics_rows(duckdb_path, "project_timeseries") == before_timeseries
    target = duckdb.connect(str(duckdb_path), read_only=True)
    try:
        assert (
            target.execute(
                "SELECT source_table, revision, source_sha256 FROM api_publication "
                "ORDER BY source_table"
            ).fetchall()
            == before_lineage
        )
    finally:
        target.close()


def test_verified_empty_timeseries_only_rebuilds_empty_target(tmp_path: Path) -> None:
    from esdc.analytics_rebuild import rebuild_api_analytics

    sqlite_path = tmp_path / "empty.sqlite"
    duckdb_path = tmp_path / "empty.duckdb"
    _, revision = _seed_pair(sqlite_path, timeseries_row=False)
    report = rebuild_api_analytics(
        duckdb_path=duckdb_path, sqlite_path=sqlite_path, reindex=False
    )
    assert report.source_revisions["project_timeseries"] == revision
    assert report.object_counts["project_timeseries"] == 0
    assert _analytics_rows(duckdb_path, "project_timeseries") == []

    conflict_path = tmp_path / "conflict.duckdb"
    target = duckdb.connect(str(conflict_path))
    target.execute("CREATE TABLE project_timeseries (value INTEGER)")
    target.execute("INSERT INTO project_timeseries VALUES (1)")
    target.close()
    before = hashlib.sha256(conflict_path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="cannot replace a populated target"):
        rebuild_api_analytics(
            duckdb_path=conflict_path, sqlite_path=sqlite_path, reindex=False
        )
    assert hashlib.sha256(conflict_path.read_bytes()).hexdigest() == before


def test_zero_timeseries_rechecks_target_inside_pair_transaction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import esdc.analytics_rebuild as rebuild

    sqlite_path = tmp_path / "empty.sqlite"
    duckdb_path = tmp_path / "analytics.duckdb"
    _seed_pair(sqlite_path, timeseries_row=False)
    rebuild.rebuild_api_analytics(
        duckdb_path=duckdb_path, sqlite_path=sqlite_path, reindex=False
    )
    target = duckdb.connect(str(duckdb_path), read_only=True)
    try:
        lineage_before = target.execute(
            "SELECT source_table, revision, source_row_count, source_sha256 "
            "FROM api_publication ORDER BY source_table"
        ).fetchall()
    finally:
        target.close()
    original_connection = rebuild.get_duckdb_connection

    class TargetRace:
        def __init__(self, connection: duckdb.DuckDBPyConnection) -> None:
            self.connection = connection
            self.inserted = False

        def execute(self, query: str, *args: object, **kwargs: object) -> object:
            if query.strip().upper() == "BEGIN TRANSACTION" and not self.inserted:
                self.connection.execute(
                    "INSERT INTO project_timeseries (report_year, project_id) "
                    "VALUES (2099, 'concurrent row')"
                )
                self.inserted = True
            return self.connection.execute(query, *args, **kwargs)

        def __getattr__(self, name: str) -> object:
            return getattr(self.connection, name)

    monkeypatch.setattr(
        rebuild,
        "get_duckdb_connection",
        lambda *args, **kwargs: TargetRace(original_connection(*args, **kwargs)),
    )
    with pytest.raises(ValueError, match="cannot replace a populated target"):
        rebuild.rebuild_api_analytics(
            duckdb_path=duckdb_path, sqlite_path=sqlite_path, reindex=False
        )
    target = duckdb.connect(str(duckdb_path), read_only=True)
    try:
        assert target.execute(
            "SELECT report_year, project_id FROM project_timeseries"
        ).fetchall() == [(2099, "concurrent row")]
        assert (
            target.execute(
                "SELECT source_table, revision, source_row_count, source_sha256 "
                "FROM api_publication ORDER BY source_table"
            ).fetchall()
            == lineage_before
        )
    finally:
        target.close()


def test_empty_timeseries_requires_current_observed_empty_provenance(
    tmp_path: Path,
) -> None:
    from esdc.analytics_rebuild import rebuild_api_analytics

    sqlite_path = tmp_path / "missing-proof.sqlite"
    duckdb_path = tmp_path / "target.duckdb"
    _seed_pair(sqlite_path, timeseries_row=False)
    source = sqlite3.connect(sqlite_path)
    source.execute(
        "DELETE FROM api_snapshot_provenance WHERE source_table='project_timeseries'"
    )
    source.commit()
    source.close()
    with pytest.raises(ValueError, match="provenance is missing"):
        rebuild_api_analytics(duckdb_path=duckdb_path, sqlite_path=sqlite_path)
    assert duckdb_path.exists() is False


def test_missing_cached_extension_fails_before_target_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import esdc.analytics_rebuild as rebuild

    sqlite_path = tmp_path / "source.sqlite"
    absent_target = tmp_path / "absent.duckdb"
    _seed_pair(sqlite_path)

    def missing_extension() -> None:
        raise RuntimeError("required DuckDB extension 'fts' is not cached")

    monkeypatch.setattr(rebuild, "_preflight_extensions", missing_extension)
    with pytest.raises(RuntimeError, match="not cached"):
        rebuild.rebuild_api_analytics(
            duckdb_path=absent_target, sqlite_path=sqlite_path, reindex=False
        )
    assert absent_target.exists() is False

    existing_target = tmp_path / "existing.duckdb"
    target = duckdb.connect(str(existing_target))
    target.execute("CREATE TABLE protected (value VARCHAR)")
    target.execute("INSERT INTO protected VALUES ('unchanged')")
    target.close()
    before = hashlib.sha256(existing_target.read_bytes()).hexdigest()
    with pytest.raises(RuntimeError, match="not cached"):
        rebuild.rebuild_api_analytics(
            duckdb_path=existing_target, sqlite_path=sqlite_path, reindex=False
        )
    assert hashlib.sha256(existing_target.read_bytes()).hexdigest() == before


def test_extension_preflight_reports_missing_cached_extension_and_closes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from esdc.analytics_rebuild import rebuild_api_analytics

    sqlite_path = tmp_path / "source.sqlite"
    absent_target = tmp_path / "absent.duckdb"
    _seed_pair(sqlite_path)
    configs: list[object] = []
    closed: list[bool] = []

    class MissingExtensionConnection:
        def execute(self, query: str) -> object:
            if query == "LOAD spatial":
                raise duckdb.Error("spatial extension unavailable")
            return self

        def close(self) -> None:
            closed.append(True)

    def preflight_connect(
        *args: object, **kwargs: object
    ) -> MissingExtensionConnection:
        assert args == (":memory:",)
        configs.append(kwargs.get("config"))
        return MissingExtensionConnection()

    monkeypatch.setattr(duckdb, "connect", preflight_connect)
    with pytest.raises(RuntimeError, match="extension 'spatial' is not cached"):
        rebuild_api_analytics(
            duckdb_path=absent_target, sqlite_path=sqlite_path, reindex=False
        )
    assert configs == [{"autoinstall_known_extensions": "false"}]
    assert closed == [True]
    assert absent_target.exists() is False


def test_resource_only_recovery_refuses_pending_timeseries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import esdc.esdc as cli
    from esdc.analytics_rebuild import rebuild_api_analytics

    sqlite_path = tmp_path / "source.sqlite"
    duckdb_path = tmp_path / "analytics.duckdb"
    _seed_pair(sqlite_path)
    rebuild_api_analytics(
        duckdb_path=duckdb_path, sqlite_path=sqlite_path, reindex=False
    )
    before = _analytics_rows(duckdb_path, "project_resources")
    source = sqlite3.connect(sqlite_path)
    source.execute(
        "UPDATE api_snapshot SET publication_state='analytics_stale' "
        "WHERE source_table='project_timeseries'"
    )
    source.commit()
    source.close()
    monkeypatch.setattr(cli.Config, "get_db_file", lambda: duckdb_path)
    monkeypatch.setattr(cli, "get_esdc_sqlite_path", lambda: sqlite_path)

    with pytest.raises(ValueError, match="rebuild-api-analytics"):
        cli.rebuild_project_resources_projection()
    assert _analytics_rows(duckdb_path, "project_resources") == before


def test_resource_only_recovery_preflights_extensions_before_writer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import esdc.analytics_rebuild as rebuild
    import esdc.esdc as cli
    from esdc.analytics_rebuild import rebuild_api_analytics

    sqlite_path = tmp_path / "source.sqlite"
    duckdb_path = tmp_path / "analytics.duckdb"
    _seed_pair(sqlite_path)
    rebuild_api_analytics(
        duckdb_path=duckdb_path, sqlite_path=sqlite_path, reindex=False
    )
    _seed_pair(sqlite_path, project_name="New pending resource")
    source = sqlite3.connect(sqlite_path)
    source.execute(
        "UPDATE api_snapshot SET publication_state='published' "
        "WHERE source_table='project_timeseries'"
    )
    source.commit()
    source.close()
    before = hashlib.sha256(duckdb_path.read_bytes()).hexdigest()
    monkeypatch.setattr(cli.Config, "get_db_file", lambda: duckdb_path)
    monkeypatch.setattr(cli, "get_esdc_sqlite_path", lambda: sqlite_path)
    monkeypatch.setattr(
        rebuild,
        "_preflight_extensions",
        lambda: (_ for _ in ()).throw(RuntimeError("cached extension unavailable")),
    )
    with pytest.raises(RuntimeError, match="cached extension unavailable"):
        cli.rebuild_project_resources_projection()
    assert hashlib.sha256(duckdb_path.read_bytes()).hexdigest() == before


def test_status_compares_both_sources_without_creating_sqlite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import esdc.esdc as cli
    from esdc.analytics_rebuild import get_api_analytics_status, rebuild_api_analytics

    absent_sqlite = tmp_path / "absent.sqlite"
    absent_target = tmp_path / "absent.duckdb"
    status = get_api_analytics_status(
        duckdb_path=absent_target, sqlite_path=absent_sqlite
    )
    assert set(status) == {"project_resources", "project_timeseries"}
    assert all(item["status"] == "absent" for item in status.values())
    assert absent_sqlite.exists() is False
    assert absent_target.exists() is False

    sqlite_path = tmp_path / "source.sqlite"
    duckdb_path = tmp_path / "analytics.duckdb"
    _seed_pair(sqlite_path)
    rebuild_api_analytics(
        duckdb_path=duckdb_path, sqlite_path=sqlite_path, reindex=False
    )
    monkeypatch.setattr(cli.Config, "get_db_file", lambda: duckdb_path)
    monkeypatch.setattr(cli, "get_esdc_sqlite_path", lambda: sqlite_path)
    assert cli._status_fetch_report() is True


def test_status_reports_unreadable_sources_and_targets_as_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from esdc.analytics_rebuild import get_api_analytics_status

    sqlite_path = tmp_path / "missing.sqlite"
    corrupt_target = tmp_path / "corrupt.duckdb"
    corrupt_target.write_bytes(b"not a DuckDB database")
    target_hash = hashlib.sha256(corrupt_target.read_bytes()).hexdigest()
    status = get_api_analytics_status(
        duckdb_path=corrupt_target, sqlite_path=sqlite_path
    )
    assert all(item["status"] == "error" for item in status.values())
    assert sqlite_path.exists() is False
    assert hashlib.sha256(corrupt_target.read_bytes()).hexdigest() == target_hash

    unreadable_source = tmp_path / "unreadable.sqlite"
    unreadable_source.write_bytes(b"not a SQLite database")
    source_hash = hashlib.sha256(unreadable_source.read_bytes()).hexdigest()
    status = get_api_analytics_status(
        duckdb_path=tmp_path / "absent.duckdb", sqlite_path=unreadable_source
    )
    assert all(item["status"] == "error" for item in status.values())
    assert hashlib.sha256(unreadable_source.read_bytes()).hexdigest() == source_hash

    from typer.testing import CliRunner

    import esdc.esdc as cli

    monkeypatch.setattr(cli.Config, "get_db_file", lambda: corrupt_target)
    monkeypatch.setattr(cli.Config, "get_db_dir", lambda: tmp_path)
    monkeypatch.setattr(cli, "get_esdc_sqlite_path", lambda: sqlite_path)
    output = CliRunner().invoke(cli.app, ["status", "fetch"])
    assert output.exit_code == 0
    assert "project_resources: error" in output.stdout
    assert "project_timeseries: error" in output.stdout
    assert "Could not check indexes" in output.stdout


def test_status_reports_malformed_duckdb_lineage_as_error_without_mutation(
    tmp_path: Path,
) -> None:
    from esdc.analytics_rebuild import (
        get_api_analytics_status,
        rebuild_api_analytics,
    )

    sqlite_path = tmp_path / "source.sqlite"
    duckdb_path = tmp_path / "analytics.duckdb"
    _seed_pair(sqlite_path)
    rebuild_api_analytics(
        duckdb_path=duckdb_path, sqlite_path=sqlite_path, reindex=False
    )
    target = duckdb.connect(str(duckdb_path))
    target.execute("DROP TABLE api_publication")
    target.execute("CREATE TABLE api_publication (broken INTEGER)")
    target.close()
    target_hash = hashlib.sha256(duckdb_path.read_bytes()).hexdigest()
    status = get_api_analytics_status(duckdb_path=duckdb_path, sqlite_path=sqlite_path)
    assert all(item["status"] == "error" for item in status.values())
    assert hashlib.sha256(duckdb_path.read_bytes()).hexdigest() == target_hash


def test_source_revision_change_before_c2_commit_rolls_back_pair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import esdc.analytics_rebuild as rebuild
    from esdc.api_projection import ResourceStage

    sqlite_path = tmp_path / "source.sqlite"
    duckdb_path = tmp_path / "analytics.duckdb"
    original_resource_revision, original_timeseries_revision = _seed_pair(sqlite_path)
    rebuild.rebuild_api_analytics(
        duckdb_path=duckdb_path, sqlite_path=sqlite_path, reindex=False
    )
    before_resources = _analytics_rows(duckdb_path, "project_resources")
    before_timeseries = _analytics_rows(duckdb_path, "project_timeseries")
    source = sqlite3.connect(sqlite_path)
    source.execute("PRAGMA journal_mode=WAL")
    source.close()
    original_stage = rebuild._stage_project_resources
    new_timeseries = normalize_timeseries_row(
        ("report_year", "project_id", "tpf_oc"),
        (2026, "P-1", 24.5),
        origin="api_json",
    )

    def stage_then_advance(
        conn: duckdb.DuckDBPyConnection,
        rows: Iterable[ApiProjectResourceRow],
    ) -> ResourceStage:
        stage = original_stage(conn, rows)
        concurrent = sqlite3.connect(sqlite_path)
        replace_project_resources_snapshot(
            concurrent,
            [
                ApiProjectResourceRow(
                    project_id="P-1",
                    report_year=2025,
                    project_name="Concurrent canonical revision",
                )
            ],
            ApiSnapshot(
                "project_resources",
                "resources-r2",
                1,
                "",
                "2026-10-24T00:02:00Z",
                "committed",
            ),
        )
        concurrent.execute("BEGIN")
        replace_timeseries_in_transaction(
            concurrent,
            [new_timeseries],
            years=None,
            revision="timeseries-r2",
            fetched_at="2026-10-24T00:02:00Z",
        )
        concurrent.commit()
        concurrent.close()
        return stage

    monkeypatch.setattr(rebuild, "_stage_project_resources", stage_then_advance)
    with pytest.raises(ValueError, match="changed before DuckDB commit"):
        rebuild.rebuild_api_analytics(
            duckdb_path=duckdb_path, sqlite_path=sqlite_path, reindex=False
        )
    assert _analytics_rows(duckdb_path, "project_resources") == before_resources
    assert _analytics_rows(duckdb_path, "project_timeseries") == before_timeseries
    source = sqlite3.connect(sqlite_path)
    try:
        updated = source.execute(
            "SELECT revision, publication_state FROM api_snapshot "
            "WHERE source_table='project_resources'"
        ).fetchone()
        assert updated is not None
        concurrent_revision, concurrent_state = updated
        assert concurrent_revision != original_resource_revision
        assert concurrent_state == "committed"
        timeseries_state = source.execute(
            "SELECT revision, publication_state FROM api_snapshot "
            "WHERE source_table='project_timeseries'"
        ).fetchone()
        assert timeseries_state == ("timeseries-r2", "committed")
        provenance = source.execute(
            "SELECT origin, details_json FROM api_snapshot_provenance "
            "WHERE source_table='project_timeseries' AND revision=?",
            (timeseries_state[0],),
        ).fetchone()
        assert provenance is not None
        assert provenance[0] == "api_json"
        assert json.loads(provenance[1])["input_origins"] == ["api_json"]
        assert source.execute(
            "SELECT project_name FROM api_project_resources WHERE project_id='P-1'"
        ).fetchone() == ("Concurrent canonical revision",)
        assert source.execute(
            "SELECT row_json FROM api_project_timeseries"
        ).fetchone() == (new_timeseries.row_json,)
        assert timeseries_state[0] != original_timeseries_revision
    finally:
        source.close()


def test_c3_interrupt_is_preserved_and_all_maintenance_is_attempted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import esdc.analytics_rebuild as rebuild

    sqlite_path = tmp_path / "source.sqlite"
    duckdb_path = tmp_path / "analytics.duckdb"
    _seed_pair(sqlite_path)
    events: list[str] = []

    def interrupt_cache() -> None:
        events.append("cache")
        raise KeyboardInterrupt("injected C3 interrupt")

    def fail_checkpoint(_: Path) -> None:
        events.append("checkpoint")
        raise RuntimeError("injected checkpoint failure")

    def fail_reindex(*_: object, **__: object) -> None:
        events.append("reindex")
        raise RuntimeError("injected reindex failure")

    monkeypatch.setattr(rebuild, "_invalidate_analytical_caches", interrupt_cache)
    monkeypatch.setattr(rebuild, "_checkpoint_path", fail_checkpoint)
    monkeypatch.setattr(rebuild, "_reindex_fts_for_path", fail_reindex)
    with pytest.raises(KeyboardInterrupt, match="injected C3 interrupt"):
        rebuild.rebuild_api_analytics(
            duckdb_path=duckdb_path, sqlite_path=sqlite_path, reindex=True
        )
    assert events == ["cache", "checkpoint", "reindex"]
    source = sqlite3.connect(sqlite_path)
    try:
        assert source.execute(
            "SELECT publication_state FROM api_snapshot ORDER BY source_table"
        ).fetchall() == [("analytics_stale",), ("analytics_stale",)]
    finally:
        source.close()
    assert len(_analytics_rows(duckdb_path, "project_resources")) == 1
    assert len(_analytics_rows(duckdb_path, "project_timeseries")) == 1


def test_post_commit_writer_close_failure_still_attempts_c3(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import esdc.analytics_rebuild as rebuild

    sqlite_path = tmp_path / "source.sqlite"
    duckdb_path = tmp_path / "analytics.duckdb"
    _seed_pair(sqlite_path)
    original_connection = rebuild.get_duckdb_connection
    events: list[str] = []

    class CloseFailure:
        def __init__(self, connection: duckdb.DuckDBPyConnection) -> None:
            self.connection = connection

        def execute(self, query: str, *args: object, **kwargs: object) -> object:
            return self.connection.execute(query, *args, **kwargs)

        def close(self) -> None:
            self.connection.close()
            raise RuntimeError("injected writer-close failure")

        def __getattr__(self, name: str) -> object:
            return getattr(self.connection, name)

    def tracked_connection(
        db_path: str | Path, read_only: bool = True, *, allow_install: bool = True
    ) -> CloseFailure:
        connection = original_connection(
            db_path, read_only, allow_install=allow_install
        )
        return CloseFailure(connection)

    monkeypatch.setattr(rebuild, "get_duckdb_connection", tracked_connection)
    monkeypatch.setattr(
        rebuild,
        "_invalidate_analytical_caches",
        lambda: events.append("cache"),
    )
    monkeypatch.setattr(
        rebuild, "_checkpoint_path", lambda _: events.append("checkpoint")
    )
    monkeypatch.setattr(
        rebuild,
        "_reindex_fts_for_path",
        lambda *args, **kwargs: events.append("reindex"),
    )
    with pytest.raises(RuntimeError, match="post-commit maintenance failed"):
        rebuild.rebuild_api_analytics(
            duckdb_path=duckdb_path, sqlite_path=sqlite_path, reindex=True
        )
    assert events == ["cache", "checkpoint", "reindex"]
    source = sqlite3.connect(sqlite_path)
    try:
        assert source.execute(
            "SELECT publication_state FROM api_snapshot ORDER BY source_table"
        ).fetchall() == [("analytics_stale",), ("analytics_stale",)]
    finally:
        source.close()


def test_offline_recovery_never_installs_extensions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from esdc.analytics_rebuild import rebuild_api_analytics

    sqlite_path = tmp_path / "source.sqlite"
    duckdb_path = tmp_path / "analytics.duckdb"
    _seed_pair(sqlite_path)
    original_connect = duckdb.connect
    configs: list[object] = []
    statements: list[str] = []

    class ConnectionSpy:
        def __init__(self, connection: duckdb.DuckDBPyConnection) -> None:
            self.connection = connection

        def execute(self, query: str, *args: object, **kwargs: object) -> object:
            statements.append(query.strip().upper())
            return self.connection.execute(query, *args, **kwargs)

        def __getattr__(self, name: str) -> object:
            return getattr(self.connection, name)

    def tracked_connect(
        database: str | Path = ":memory:",
        read_only: bool = False,
        config: dict[str, str | bool | int | float | list[str]] | None = None,
    ) -> ConnectionSpy:
        configs.append(config)
        return ConnectionSpy(
            original_connect(database, read_only=read_only, config=config)
        )

    monkeypatch.setattr(duckdb, "connect", tracked_connect)
    rebuild_api_analytics(
        duckdb_path=duckdb_path, sqlite_path=sqlite_path, reindex=True
    )
    assert configs
    assert all(
        config == {"autoinstall_known_extensions": "false"} for config in configs
    )
    assert not any(statement.startswith("INSTALL ") for statement in statements)


def test_status_detects_copied_wrong_rows_and_validates_empty_as_published(
    tmp_path: Path,
) -> None:
    from esdc.analytics_rebuild import (
        get_api_analytics_status,
        rebuild_api_analytics,
    )

    sqlite_path = tmp_path / "source.sqlite"
    duckdb_path = tmp_path / "analytics.duckdb"
    _seed_pair(
        sqlite_path,
        timeseries_row=False,
        field_geom="POINT (1 2)",
        project_class="1 - Development",
        project_level="E",
        wk_id=123,
        rec_oil="12.5",
    )
    rebuild_api_analytics(
        duckdb_path=duckdb_path, sqlite_path=sqlite_path, reindex=False
    )
    status = get_api_analytics_status(duckdb_path=duckdb_path, sqlite_path=sqlite_path)
    assert status["project_timeseries"]["status"] == "published"
    timeseries_snapshot = status["project_timeseries"]["snapshot"]
    assert timeseries_snapshot is not None
    assert timeseries_snapshot["row_count"] == 0

    target = duckdb.connect(str(duckdb_path))
    target.execute("UPDATE project_resources SET project_name='copied wrong row'")
    target.close()
    status = get_api_analytics_status(duckdb_path=duckdb_path, sqlite_path=sqlite_path)
    assert status["project_resources"]["status"] == "stale"


def test_invalid_geometry_count_is_marked_for_exact_resource_revision(
    tmp_path: Path,
) -> None:
    from esdc.analytics_rebuild import rebuild_api_analytics

    sqlite_path = tmp_path / "source.sqlite"
    duckdb_path = tmp_path / "analytics.duckdb"
    _seed_pair(sqlite_path, field_geom="POINT (not valid)")
    report = rebuild_api_analytics(
        duckdb_path=duckdb_path, sqlite_path=sqlite_path, reindex=False
    )
    assert report.object_counts["invalid_resource_geometry"] == 1
    source = sqlite3.connect(sqlite_path)
    try:
        assert source.execute(
            "SELECT publication_state, invalid_geometry_count FROM api_snapshot "
            "WHERE source_table='project_resources'"
        ).fetchone() == ("published", 1)
    finally:
        source.close()


def test_rebuild_rejects_source_target_file_aliases(tmp_path: Path) -> None:
    from esdc.analytics_rebuild import rebuild_api_analytics

    sqlite_path = tmp_path / "source.sqlite"
    _seed_pair(sqlite_path)
    symlink_path = tmp_path / "source-link.sqlite"
    symlink_path.symlink_to(sqlite_path)
    hardlink_path = tmp_path / "source-hardlink.sqlite"
    hardlink_path.hardlink_to(sqlite_path)
    for alias in (sqlite_path, symlink_path, hardlink_path):
        with pytest.raises(ValueError, match="must be different files"):
            rebuild_api_analytics(duckdb_path=alias, sqlite_path=sqlite_path)


def test_missing_timeseries_source_refuses_without_target_mutation(
    tmp_path: Path,
) -> None:
    from esdc.analytics_rebuild import rebuild_api_analytics

    sqlite_path = tmp_path / "source.sqlite"
    duckdb_path = tmp_path / "analytics.duckdb"
    _seed_pair(sqlite_path)
    source = sqlite3.connect(sqlite_path)
    source.execute("DROP TABLE api_project_timeseries")
    source.commit()
    source.close()
    with pytest.raises(ValueError, match="missing table"):
        rebuild_api_analytics(
            duckdb_path=duckdb_path, sqlite_path=sqlite_path, reindex=False
        )
    assert duckdb_path.exists() is False
