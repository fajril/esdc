"""Read-only selective resource baseline guard regressions."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections import Counter
from collections.abc import Generator
from datetime import datetime
from pathlib import Path

import duckdb
import pytest

from esdc.api_projection import get_resource_publication
from esdc.api_snapshot import (
    ApiProjectResourceRow,
    ApiSnapshot,
    commit_project_resources_update,
    get_project_resources_snapshot,
    mark_project_resources_published,
    replace_project_resources_snapshot,
)
from esdc.fetch_publication import (
    ensure_selective_resource_baseline,
    ensure_timeseries_baseline,
    publish_prepared_fetch,
)
from esdc.fetching import (
    FetchFailure,
    PreparedFetch,
    PreparedTimeseriesSlice,
    spool_timeseries,
)
from esdc.pod_registry.store import get_sqlite_connection
from esdc.resource_audit import compare_resource_projection
from esdc.timeseries_snapshot import (
    TIMESERIES_SOURCE_COLUMNS,
    TIMESERIES_SOURCE_TYPES,
    ensure_timeseries_tables,
    get_timeseries_snapshot,
    normalize_timeseries_row,
    record_timeseries_provenance,
    replace_timeseries_in_transaction,
)

_PUBLICATION_DDL = """CREATE TABLE api_publication (
    source_table VARCHAR PRIMARY KEY,
    revision VARCHAR NOT NULL,
    source_row_count BIGINT NOT NULL,
    source_sha256 VARCHAR NOT NULL,
    invalid_geometry_count BIGINT NOT NULL,
    published_at TIMESTAMPTZ NOT NULL
)"""


@pytest.fixture(autouse=True)
def _isolated_publication_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Generator[None, None, None]:
    """Keep coordinator cache/config paths inside this test's temporary tree."""
    from esdc.configs import Config

    config_dir = tmp_path / "config"
    database_dir = tmp_path / "database"
    config_dir.mkdir()
    database_dir.mkdir()
    monkeypatch.setenv("ESDC_CONFIG_DIR", str(config_dir))
    monkeypatch.setenv("ESDC_DB_DIR", str(database_dir))
    monkeypatch.setenv("ESDC_DB_FILE", str(database_dir / "esdc.sqlite"))
    Config._config_cache = None
    yield
    Config._config_cache = None


def _write_canonical(path: Path, rows: list[ApiProjectResourceRow]) -> str:
    conn = get_sqlite_connection(path)
    replace_project_resources_snapshot(
        conn,
        rows,
        ApiSnapshot(
            "project_resources", "incoming", len(rows), "caller", "now", "committed"
        ),
    )
    snapshot = conn.execute(
        "SELECT revision FROM api_snapshot WHERE source_table='project_resources'"
    ).fetchone()[0]
    mark_project_resources_published(conn, snapshot)
    conn.close()
    return snapshot


def _write_mirror(
    path: Path,
    rows: list[tuple[str | None, int | None, str | None]],
    *,
    revision: str | None = None,
    digest: str = "",
) -> None:
    conn = duckdb.connect(str(path))
    conn.execute(
        "CREATE TABLE project_resources(project_id VARCHAR, report_year INTEGER, uncert_level VARCHAR)"
    )
    conn.executemany("INSERT INTO project_resources VALUES (?, ?, ?)", rows)
    if revision is not None:
        conn.execute(_PUBLICATION_DDL)
        conn.execute(
            "INSERT INTO api_publication VALUES ('project_resources', ?, ?, ?, 0, now())",
            [revision, len(rows), digest],
        )
    conn.close()


def _skip_population_fts(monkeypatch: pytest.MonkeyPatch) -> None:
    def create_indexes(conn, *, tables, extensions_loaded):
        assert extensions_loaded is True
        return dict.fromkeys(tables, "ok")

    monkeypatch.setattr("esdc.dbmanager._create_fts_indexes", create_indexes)


def test_populated_mirror_with_empty_canonical_refuses_selective_update(
    tmp_path: Path,
) -> None:
    sqlite_path = tmp_path / "esdc.sqlite"
    duck_path = tmp_path / "esdc.duckdb"
    _write_mirror(duck_path, [("P-1", 2025, "1P")])

    with pytest.raises(ValueError, match="Run an unfiltered 'esdc fetch'"):
        ensure_selective_resource_baseline(duck_path, sqlite_path)


@pytest.mark.parametrize("unreadable_kind", ["missing-identity-column", "invalid-file"])
def test_uninspectable_existing_mirror_refuses_without_mutation(
    tmp_path: Path, unreadable_kind: str
) -> None:
    sqlite_path = tmp_path / "missing.sqlite"
    duck_path = tmp_path / "esdc.duckdb"
    if unreadable_kind == "invalid-file":
        duck_path.write_bytes(b"not a DuckDB database")
    else:
        conn = duckdb.connect(str(duck_path))
        conn.execute("CREATE TABLE project_resources(project_id VARCHAR)")
        conn.execute("INSERT INTO project_resources VALUES ('legacy')")
        conn.close()
    before = hashlib.sha256(duck_path.read_bytes()).hexdigest()

    with pytest.raises(ValueError, match="Existing data was retained"):
        ensure_selective_resource_baseline(duck_path, sqlite_path)

    assert hashlib.sha256(duck_path.read_bytes()).hexdigest() == before


def test_equal_counts_do_not_hide_different_identity_multisets(tmp_path: Path) -> None:
    sqlite_path = tmp_path / "esdc.sqlite"
    duck_path = tmp_path / "esdc.duckdb"
    rows = [
        ApiProjectResourceRow(project_id="P-1", report_year=2025, uncert_level=None),
        ApiProjectResourceRow(project_id="P-1", report_year=2025, uncert_level=None),
        ApiProjectResourceRow(project_id="P-null", report_year=None, uncert_level=None),
    ]
    revision = _write_canonical(sqlite_path, rows)
    conn = sqlite3.connect(sqlite_path)
    digest = conn.execute(
        "SELECT source_sha256 FROM api_snapshot WHERE source_table='project_resources'"
    ).fetchone()[0]
    conn.close()
    _write_mirror(
        duck_path,
        [("P-1", 2025, "1P"), ("P-1", 2025, None), ("P-null", None, None)],
        revision=revision,
        digest=digest,
    )

    with pytest.raises(ValueError, match="Existing data was retained"):
        ensure_selective_resource_baseline(duck_path, sqlite_path)


def test_both_empty_stores_allow_selective_first_fetch(tmp_path: Path) -> None:
    assert (
        ensure_selective_resource_baseline(
            tmp_path / "missing.duckdb", tmp_path / "missing.sqlite"
        )
        is None
    )


def test_missing_lineage_refuses_populated_canonical_and_gives_rebuild_guidance(
    tmp_path: Path,
) -> None:
    sqlite_path = tmp_path / "esdc.sqlite"
    duck_path = tmp_path / "esdc.duckdb"
    _write_canonical(
        sqlite_path,
        [ApiProjectResourceRow(project_id="P-1", report_year="2025", uncert_level=1)],
    )
    _write_mirror(duck_path, [("P-1", 2025, "1")])

    with pytest.raises(ValueError, match="Run 'esdc rebuild-project-resources'"):
        ensure_selective_resource_baseline(duck_path, sqlite_path)
    conn = duckdb.connect(str(duck_path), read_only=True)
    try:
        assert conn.execute(
            "SELECT COUNT(*) FROM information_schema.tables "
            "WHERE table_name='api_publication'"
        ).fetchone() == (0,)
    finally:
        conn.close()


def test_preflight_opens_inputs_read_only_and_does_not_create_lineage_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sqlite_path = tmp_path / "esdc.sqlite"
    duck_path = tmp_path / "esdc.duckdb"
    rows = [
        ApiProjectResourceRow(project_id="P-1", report_year=2025, uncert_level="1P")
    ]
    revision = _write_canonical(sqlite_path, rows)
    conn = sqlite3.connect(sqlite_path)
    digest = conn.execute(
        "SELECT source_sha256 FROM api_snapshot WHERE source_table='project_resources'"
    ).fetchone()[0]
    conn.close()
    _write_mirror(duck_path, [("P-1", 2025, "1P")], revision=revision, digest=digest)
    before = hashlib.sha256(duck_path.read_bytes()).hexdigest()
    sqlite_calls: list[tuple[str | Path, bool]] = []
    duckdb_calls: list[tuple[str, bool]] = []
    original_sqlite_connect = sqlite3.connect
    original_duckdb_connect = duckdb.connect

    def sqlite_connect_spy(
        database: str | Path, *, uri: bool = False
    ) -> sqlite3.Connection:
        sqlite_calls.append((database, uri))
        return original_sqlite_connect(database, uri=uri)

    def duckdb_connect_spy(
        database: str, *, read_only: bool = False
    ) -> duckdb.DuckDBPyConnection:
        duckdb_calls.append((database, read_only))
        return original_duckdb_connect(database, read_only=read_only)

    monkeypatch.setattr("esdc.fetch_publication.sqlite3.connect", sqlite_connect_spy)
    monkeypatch.setattr("esdc.fetch_publication.duckdb.connect", duckdb_connect_spy)

    result = ensure_selective_resource_baseline(duck_path, sqlite_path)

    assert result is not None and result.revision == revision
    assert hashlib.sha256(duck_path.read_bytes()).hexdigest() == before
    assert len(sqlite_calls) == 1
    sqlite_uri, sqlite_mode = sqlite_calls[0]
    assert "mode=ro" in str(sqlite_uri)
    assert sqlite_mode is True
    assert duckdb_calls == [(str(duck_path), True)]
    sqlite_ro = original_sqlite_connect(f"file:{sqlite_path}?mode=ro", uri=True)
    try:
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            sqlite_ro.execute(
                "UPDATE api_snapshot SET publication_state='analytics_stale'"
            )
    finally:
        sqlite_ro.close()
    duck_ro = original_duckdb_connect(str(duck_path), read_only=True)
    try:
        publication = get_resource_publication(duck_ro)
        assert publication is not None
        assert isinstance(publication.published_at, datetime)
        assert publication.published_at.tzinfo is not None
        assert Counter(
            duck_ro.execute("SELECT project_id FROM project_resources").fetchall()
        )
    finally:
        duck_ro.close()


def test_identity_normalization_keeps_duplicate_and_null_rows(tmp_path: Path) -> None:
    # A matching baseline with numeric/string year and uncertainty values is valid.
    sqlite_path = tmp_path / "esdc.sqlite"
    duck_path = tmp_path / "esdc.duckdb"
    rows = [
        ApiProjectResourceRow(project_id="P-1", report_year="2025", uncert_level=1),
        ApiProjectResourceRow(project_id="P-null", report_year=None, uncert_level=None),
    ]
    revision = _write_canonical(sqlite_path, rows)
    conn = sqlite3.connect(sqlite_path)
    digest = conn.execute(
        "SELECT source_sha256 FROM api_snapshot WHERE source_table='project_resources'"
    ).fetchone()[0]
    conn.close()
    _write_mirror(
        duck_path,
        [("P-1", 2025, "1"), ("P-null", None, None)],
        revision=revision,
        digest=digest,
    )
    snapshot = ensure_selective_resource_baseline(duck_path, sqlite_path)
    assert snapshot is not None and snapshot.row_count == 2


def test_existing_unreadable_sqlite_refuses_without_modification(
    tmp_path: Path,
) -> None:
    sqlite_path = tmp_path / "bad.sqlite"
    sqlite_path.write_bytes(b"not a SQLite database")
    before = hashlib.sha256(sqlite_path.read_bytes()).hexdigest()

    with pytest.raises(ValueError, match="Run an unfiltered 'esdc fetch'"):
        ensure_selective_resource_baseline(tmp_path / "absent.duckdb", sqlite_path)

    assert hashlib.sha256(sqlite_path.read_bytes()).hexdigest() == before


def test_missing_snapshot_table_with_populated_canonical_rows_refuses_read_only(
    tmp_path: Path,
) -> None:
    sqlite_path = tmp_path / "orphan.sqlite"
    conn = get_sqlite_connection(sqlite_path)
    conn.execute("INSERT INTO api_project_resources (project_id) VALUES ('orphan')")
    conn.execute("DROP TABLE api_snapshot")
    conn.commit()
    conn.close()
    before = hashlib.sha256(sqlite_path.read_bytes()).hexdigest()

    with pytest.raises(ValueError, match="Run an unfiltered 'esdc fetch'"):
        ensure_selective_resource_baseline(tmp_path / "absent.duckdb", sqlite_path)

    assert hashlib.sha256(sqlite_path.read_bytes()).hexdigest() == before
    conn = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
    try:
        assert conn.execute(
            "SELECT COUNT(*) FROM api_project_resources"
        ).fetchone() == (1,)
        assert conn.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE name='api_snapshot'"
        ).fetchone() == (0,)
    finally:
        conn.close()


def test_sqlite_connect_error_for_existing_file_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sqlite_path = tmp_path / "present.sqlite"
    sqlite_path.write_bytes(b"synthetic file")
    before = hashlib.sha256(sqlite_path.read_bytes()).hexdigest()
    real_connect = sqlite3.connect

    def fail_existing(path: str, *, uri: bool = False) -> sqlite3.Connection:
        if str(path).startswith("file:") and "present.sqlite" in str(path):
            raise sqlite3.OperationalError("injected read-only open failure")
        return real_connect(path, uri=uri)

    monkeypatch.setattr("esdc.fetch_publication.sqlite3.connect", fail_existing)
    with pytest.raises(ValueError, match="Run an unfiltered 'esdc fetch'"):
        ensure_selective_resource_baseline(tmp_path / "absent.duckdb", sqlite_path)
    assert hashlib.sha256(sqlite_path.read_bytes()).hexdigest() == before


def test_empty_prepared_batch_is_noop_without_opening_databases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unexpected(*args, **kwargs):
        raise AssertionError("empty fetch must not open a database")

    monkeypatch.setattr("esdc.fetch_publication.get_duckdb_connection", unexpected)
    monkeypatch.setattr("esdc.fetch_publication.get_sqlite_connection", unexpected)
    outcome = publish_prepared_fetch(
        PreparedFetch((), None, (), (), ()),
        duckdb_path=tmp_path / "absent.duckdb",
        sqlite_path=tmp_path / "absent.sqlite",
        reindex=True,
    )
    assert outcome.resource_revision is None
    assert outcome.timeseries_rows_written == 0


def test_resource_publication_uses_committed_sqlite_rows_before_duckdb_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    row = ApiProjectResourceRow(project_id="canonical", report_year=2024)
    sqlite_path = tmp_path / "esdc.sqlite"
    duck_path = tmp_path / "esdc.duckdb"
    events: list[object] = []
    real_get_sqlite_connection = get_sqlite_connection

    def observe_c1(path):
        conn = real_get_sqlite_connection(path)

        class CommitObserver:
            def __getattr__(self, name):
                return getattr(conn, name)

            def commit(self):
                conn.commit()
                events.append("c1")

            def close(self):
                return conn.close()

        return CommitObserver()

    monkeypatch.setattr("esdc.fetch_publication.get_sqlite_connection", observe_c1)
    original_stage = __import__(
        "esdc.fetch_publication", fromlist=["x"]
    )._stage_project_resources

    def stage(conn, rows):
        rows = tuple(rows)
        events.append(("stage", rows[0].project_id))
        return original_stage(conn, rows)

    monkeypatch.setattr("esdc.fetch_publication._stage_project_resources", stage)
    monkeypatch.setattr(
        "esdc.api_projection._invalidate_analytical_caches", lambda: None
    )
    monkeypatch.setattr("esdc.api_projection._checkpoint", lambda conn: None)

    outcome = publish_prepared_fetch(
        PreparedFetch((row,), None, (), (), ()),
        duckdb_path=duck_path,
        sqlite_path=sqlite_path,
        reindex=False,
    )

    assert events.index("c1") < next(
        index
        for index, event in enumerate(events)
        if isinstance(event, tuple) and event[0] == "stage"
    )
    assert next(event for event in events if isinstance(event, tuple)) == (
        "stage",
        "canonical",
    )
    assert outcome.resource_revision is not None
    assert outcome.resource_row_count == 1


def test_duckdb_stage_failure_retains_canonical_revision_and_old_analytical_pair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    old_row = ApiProjectResourceRow(project_id="old", report_year=2024)
    sqlite_path = tmp_path / "esdc.sqlite"
    duck_path = tmp_path / "esdc.duckdb"
    old_revision = _write_canonical(sqlite_path, [old_row])
    conn = sqlite3.connect(sqlite_path)
    old_snapshot = get_project_resources_snapshot(conn)
    conn.close()
    assert old_snapshot is not None
    _write_mirror(
        duck_path,
        [("old", 2024, None)],
        revision=old_revision,
        digest=old_snapshot.source_sha256,
    )
    prepared = PreparedFetch(
        (ApiProjectResourceRow(project_id="new", report_year=2025),),
        None,
        (),
        (),
        (),
    )

    def fail_stage(*args, **kwargs):
        raise ValueError("typed stage failed")

    monkeypatch.setattr("esdc.fetch_publication._stage_project_resources", fail_stage)
    with pytest.raises(FetchFailure, match="Pair recovery is required") as error:
        publish_prepared_fetch(
            prepared,
            duckdb_path=duck_path,
            sqlite_path=sqlite_path,
            reindex=False,
        )

    assert error.value.sqlite_committed is True
    assert error.value.duckdb_committed is False
    conn = sqlite3.connect(sqlite_path)
    current = get_project_resources_snapshot(conn)
    assert current is not None and current.revision != old_revision
    assert current.publication_state == "analytics_stale"
    conn.close()
    with duckdb.connect(str(duck_path), read_only=True) as db:
        assert db.execute("SELECT project_id FROM project_resources").fetchall() == [
            ("old",)
        ]
        lineage = db.execute("SELECT revision FROM api_publication").fetchone()
        assert lineage == (old_revision,)


def test_duckdb_writer_keyboard_interrupt_preserves_interrupt_and_marks_c1_stale(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sqlite_path = tmp_path / "esdc.sqlite"
    duck_path = tmp_path / "esdc.duckdb"
    monkeypatch.setattr(
        "esdc.fetch_publication.get_duckdb_connection",
        lambda *args, **kwargs: (_ for _ in ()).throw(KeyboardInterrupt()),
    )

    with pytest.raises(KeyboardInterrupt):
        publish_prepared_fetch(
            PreparedFetch(
                (ApiProjectResourceRow(project_id="P-1", report_year=2025),),
                None,
                (),
                (),
                (),
            ),
            duckdb_path=duck_path,
            sqlite_path=sqlite_path,
            reindex=False,
        )

    conn = sqlite3.connect(sqlite_path)
    try:
        snapshot = get_project_resources_snapshot(conn)
        assert snapshot is not None
        assert snapshot.publication_state == "analytics_stale"
    finally:
        conn.close()


def test_c1_connection_close_failure_reports_durable_stale_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sqlite_path = tmp_path / "esdc.sqlite"
    duck_path = tmp_path / "esdc.duckdb"
    real_get_connection = get_sqlite_connection
    first = True

    class CloseFailureProxy:
        def __init__(self, connection: sqlite3.Connection):
            self.connection = connection

        def __getattr__(self, name):
            return getattr(self.connection, name)

        def close(self):
            self.connection.close()
            raise OSError("C1 close failed")

    def connect(path):
        nonlocal first
        connection = real_get_connection(path)
        if first:
            first = False
            return CloseFailureProxy(connection)
        return connection

    monkeypatch.setattr("esdc.fetch_publication.get_sqlite_connection", connect)

    with pytest.raises(FetchFailure, match="C1 close failed") as error:
        publish_prepared_fetch(
            PreparedFetch(
                (ApiProjectResourceRow(project_id="P-1", report_year=2025),),
                None,
                (),
                (),
                (),
            ),
            duckdb_path=duck_path,
            sqlite_path=sqlite_path,
            reindex=False,
        )

    assert error.value.sqlite_committed is True
    assert error.value.duckdb_committed is False
    conn = sqlite3.connect(sqlite_path)
    try:
        snapshot = get_project_resources_snapshot(conn)
        assert snapshot is not None
        assert snapshot.publication_state == "analytics_stale"
    finally:
        conn.close()


def test_post_c2_failure_does_not_claim_old_data_or_refetch() -> None:
    from esdc.fetch_publication import _failure

    failure = _failure(
        "reindex failed",
        phase="post_commit",
        sqlite_committed=True,
        duckdb_committed=True,
        resource_revision="revision-1",
    )
    assert "New DuckDB data is committed" in str(failure)
    assert "retains its prior" not in str(failure)
    assert "must be fetched again" not in str(failure)


def _slice(
    directory: Path, project_id: str, year: int, value: int
) -> PreparedTimeseriesSlice:
    prepared = spool_timeseries(
        ("report_year", "project_id", "year", "cprd_grs_oil"),
        ((year, project_id, year, value),),
        report_year=year,
        directory=directory,
    )
    assert prepared is not None
    return prepared


def _seed_coherent_pair(sqlite_path: Path, duck_path: Path, tmp_path: Path) -> str:
    result = publish_prepared_fetch(
        PreparedFetch(
            (ApiProjectResourceRow(project_id="old", report_year=2024),),
            None,
            (_slice(tmp_path / "old", "old", 2024, 10),),
            (),
            (),
        ),
        duckdb_path=duck_path,
        sqlite_path=sqlite_path,
        reindex=False,
    )
    assert result.resource_revision is not None
    return result.resource_revision


def _set_bootstrap_provenance(
    sqlite_path: Path,
    source_types: dict[str, str],
    *,
    publication_state: str = "committed",
) -> None:
    conn = get_sqlite_connection(sqlite_path)
    snapshot = get_timeseries_snapshot(conn)
    assert snapshot is not None
    conn.execute("UPDATE api_project_timeseries SET origin='legacy_duckdb'")
    conn.execute(
        "UPDATE api_snapshot SET publication_state=? "
        "WHERE source_table='project_timeseries'",
        (publication_state,),
    )
    conn.execute(
        "UPDATE api_snapshot_provenance SET origin='legacy_duckdb', details_json=? "
        "WHERE source_table='project_timeseries' AND revision=?",
        (json.dumps({"source_types": source_types}), snapshot.revision),
    )
    conn.commit()
    conn.close()


def test_stale_legacy_bootstrap_cannot_bypass_published_baseline_guard(
    tmp_path: Path,
) -> None:
    sqlite_path = tmp_path / "esdc.sqlite"
    duck_path = tmp_path / "esdc.duckdb"
    _seed_coherent_pair(sqlite_path, duck_path, tmp_path)
    _set_bootstrap_provenance(
        sqlite_path,
        dict(TIMESERIES_SOURCE_TYPES),
        publication_state="analytics_stale",
    )

    with pytest.raises(ValueError, match="baseline is not published"):
        ensure_timeseries_baseline(duck_path, sqlite_path, allow_bootstrap=True)


def test_published_legacy_baseline_cannot_use_bootstrap_without_lineage(
    tmp_path: Path,
) -> None:
    sqlite_path = tmp_path / "esdc.sqlite"
    duck_path = tmp_path / "esdc.duckdb"
    _seed_coherent_pair(sqlite_path, duck_path, tmp_path)
    _set_bootstrap_provenance(
        sqlite_path,
        dict(TIMESERIES_SOURCE_TYPES),
        publication_state="published",
    )
    conn = duckdb.connect(str(duck_path))
    conn.execute(
        "DELETE FROM api_publication WHERE source_table='project_timeseries'"
    )
    conn.close()

    with pytest.raises(ValueError, match="analytical project_timeseries lineage is missing"):
        ensure_timeseries_baseline(duck_path, sqlite_path, allow_bootstrap=True)


def test_resource_only_preservation_refuses_corrupt_current_bootstrap_types(
    tmp_path: Path,
) -> None:
    sqlite_path = tmp_path / "esdc.sqlite"
    duck_path = tmp_path / "esdc.duckdb"
    _seed_coherent_pair(sqlite_path, duck_path, tmp_path)
    corrupt_types = dict(TIMESERIES_SOURCE_TYPES)
    corrupt_types.pop("ghv")
    _set_bootstrap_provenance(sqlite_path, corrupt_types)
    resource_before = _canonical_state(sqlite_path)

    with pytest.raises(FetchFailure, match="type map is incomplete"):
        publish_prepared_fetch(
            PreparedFetch(
                (ApiProjectResourceRow(project_id="new-resource", report_year=2026),),
                None,
                (),
                (),
                (),
                timeseries_complete_observation=True,
            ),
            duckdb_path=duck_path,
            sqlite_path=sqlite_path,
            reindex=False,
        )

    assert _canonical_state(sqlite_path) == resource_before


def test_resource_only_preservation_refuses_missing_current_timeseries_provenance(
    tmp_path: Path,
) -> None:
    sqlite_path = tmp_path / "esdc.sqlite"
    duck_path = tmp_path / "esdc.duckdb"
    _seed_coherent_pair(sqlite_path, duck_path, tmp_path)
    resource_before = _canonical_state(sqlite_path)
    with get_sqlite_connection(sqlite_path) as conn:
        snapshot = get_timeseries_snapshot(conn)
        assert snapshot is not None
        conn.execute(
            "DELETE FROM api_snapshot_provenance "
            "WHERE source_table='project_timeseries' AND revision=?",
            (snapshot.revision,),
        )

    with pytest.raises(FetchFailure, match="provenance is missing"):
        publish_prepared_fetch(
            PreparedFetch(
                (ApiProjectResourceRow(project_id="new-resource", report_year=2026),),
                None,
                (),
                (),
                (),
                timeseries_complete_observation=True,
            ),
            duckdb_path=duck_path,
            sqlite_path=sqlite_path,
            reindex=False,
        )

    assert _canonical_state(sqlite_path) == resource_before


def _replacement(tmp_path: Path) -> PreparedFetch:
    return PreparedFetch(
        (ApiProjectResourceRow(project_id="new", report_year=2025),),
        None,
        (_slice(tmp_path / "new", "new", 2025, 20),),
        (),
        (),
    )


def _assert_resource_geometry_count(
    sqlite_path: Path, duck_path: Path, expected_count: int
) -> None:
    with sqlite3.connect(sqlite_path) as source:
        snapshot = get_project_resources_snapshot(source)
        raw_rows = source.execute(
            "SELECT project_id, field_geom FROM api_project_resources "
            "ORDER BY project_id"
        ).fetchall()
    assert snapshot is not None
    assert snapshot.invalid_geometry_count == expected_count
    with duckdb.connect(str(duck_path), read_only=True) as target:
        publication = get_resource_publication(target)
        audit = compare_resource_projection(target, sqlite_path)
        projection_columns = {
            row[0]
            for row in target.execute(
                "SELECT column_name FROM duckdb_columns() "
                "WHERE table_name='project_resources'"
            ).fetchall()
        }
    assert publication is not None
    assert publication.revision == snapshot.revision
    assert publication.invalid_geometry_count == expected_count
    assert audit.ok
    assert audit.invalid_geometry_count == expected_count
    assert audit.source_multiset_difference_count == 0
    assert not {"field_geom", "field_geometry"} & projection_columns
    if expected_count > 0:
        assert ("bad-wkt", "NOT WKT") in raw_rows


@pytest.mark.parametrize("include_timeseries", [False, True])
def test_c3_marks_invalid_geometry_count_for_resource_and_joint_publications(
    tmp_path: Path, include_timeseries: bool
) -> None:
    sqlite_path = tmp_path / "esdc.sqlite"
    duck_path = tmp_path / "esdc.duckdb"
    rows = (
        ApiProjectResourceRow(
            project_id="bad-wkt",
            report_year=2025,
            project_name="Malformed geometry",
            field_geom="NOT WKT",
        ),
    )
    timeseries = (
        (_slice(tmp_path / "timeseries", "bad-wkt", 2025, 7),)
        if include_timeseries
        else ()
    )

    outcome = publish_prepared_fetch(
        PreparedFetch(rows, None, timeseries, (), ()),
        duckdb_path=duck_path,
        sqlite_path=sqlite_path,
        reindex=False,
    )

    assert outcome.resource_revision is not None
    assert outcome.resource_invalid_geometry_count == 1
    _assert_resource_geometry_count(sqlite_path, duck_path, 1)
    if include_timeseries:
        with sqlite3.connect(sqlite_path) as source:
            timeseries_snapshot = get_timeseries_snapshot(source)
        assert timeseries_snapshot is not None
        assert timeseries_snapshot.publication_state == "published"


def test_invalid_geometry_count_survives_selected_refresh_and_full_valid_reset(
    tmp_path: Path,
) -> None:
    sqlite_path = tmp_path / "esdc.sqlite"
    duck_path = tmp_path / "esdc.duckdb"
    publish_prepared_fetch(
        PreparedFetch(
            (
                ApiProjectResourceRow(
                    project_id="bad-wkt",
                    report_year=2025,
                    field_geom="NOT WKT",
                ),
                ApiProjectResourceRow(
                    project_id="old-valid", report_year=2024, field_geom="POINT (1 2)"
                ),
            ),
            None,
            (),
            (),
            (),
        ),
        duckdb_path=duck_path,
        sqlite_path=sqlite_path,
        reindex=False,
    )
    _assert_resource_geometry_count(sqlite_path, duck_path, 1)

    publish_prepared_fetch(
        PreparedFetch(
            (ApiProjectResourceRow(project_id="new-valid", report_year=2024),),
            (2024,),
            (),
            (),
            (),
        ),
        duckdb_path=duck_path,
        sqlite_path=sqlite_path,
        reindex=False,
    )
    _assert_resource_geometry_count(sqlite_path, duck_path, 1)

    publish_prepared_fetch(
        PreparedFetch(
            (ApiProjectResourceRow(project_id="all-valid", report_year=2026),),
            None,
            (),
            (),
            (),
        ),
        duckdb_path=duck_path,
        sqlite_path=sqlite_path,
        reindex=False,
    )
    _assert_resource_geometry_count(sqlite_path, duck_path, 0)


def test_c3_old_resource_count_cannot_overwrite_newer_canonical_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from esdc import fetch_publication

    sqlite_path = tmp_path / "esdc.sqlite"
    duck_path = tmp_path / "esdc.duckdb"
    superseding: list[str] = []
    original_mark = fetch_publication._mark_source_published

    def supersede_before_c3(
        path: Path, source_table: str, snapshot: ApiSnapshot
    ) -> None:
        if source_table == "project_resources":
            with get_sqlite_connection(path) as source:
                current = get_project_resources_snapshot(source)
                assert current is not None
                newer = commit_project_resources_update(
                    source,
                    (
                        ApiProjectResourceRow(
                            project_id="newer-valid",
                            report_year=2026,
                            field_geom="POINT (1 2)",
                        ),
                    ),
                    years=None,
                    expected_snapshot=current,
                )
            superseding.append(newer.revision)
        original_mark(path, source_table, snapshot)

    monkeypatch.setattr(
        "esdc.fetch_publication._mark_source_published", supersede_before_c3
    )
    with pytest.raises(FetchFailure, match="no project_resources snapshot matches") as error:
        publish_prepared_fetch(
            PreparedFetch(
                (
                    ApiProjectResourceRow(
                        project_id="bad-wkt",
                        report_year=2025,
                        field_geom="NOT WKT",
                    ),
                ),
                None,
                (),
                (),
                (),
            ),
            duckdb_path=duck_path,
            sqlite_path=sqlite_path,
            reindex=False,
        )

    assert error.value.phase == "post_commit"
    assert error.value.sqlite_committed is True
    assert error.value.duckdb_committed is True
    assert len(superseding) == 1
    with sqlite3.connect(sqlite_path) as source:
        current = get_project_resources_snapshot(source)
    assert current is not None
    assert current.revision == superseding[0]
    assert current.invalid_geometry_count == 0
    assert current.publication_state == "committed"


def test_resource_only_fetch_refuses_unpublished_preserved_timeseries_baseline(
    tmp_path: Path,
) -> None:
    sqlite_path = tmp_path / "esdc.sqlite"
    duck_path = tmp_path / "esdc.duckdb"
    _seed_coherent_pair(sqlite_path, duck_path, tmp_path)
    conn = get_sqlite_connection(sqlite_path)
    conn.execute(
        "UPDATE api_snapshot SET publication_state='analytics_stale' "
        "WHERE source_table='project_timeseries'"
    )
    conn.commit()
    conn.close()
    before = _canonical_state(sqlite_path)

    with pytest.raises(
        FetchFailure, match="project_timeseries baseline is not published"
    ):
        publish_prepared_fetch(
            PreparedFetch(
                (ApiProjectResourceRow(project_id="new-resource", report_year=2026),),
                None,
                (),
                (),
                (),
            ),
            duckdb_path=duck_path,
            sqlite_path=sqlite_path,
            reindex=False,
        )

    assert _canonical_state(sqlite_path) == before


def test_empty_timeseries_baseline_requires_table_schema_and_lineage(
    tmp_path: Path,
) -> None:
    sqlite_path = tmp_path / "esdc.sqlite"
    duck_path = tmp_path / "esdc.duckdb"
    publish_prepared_fetch(
        PreparedFetch(
            (ApiProjectResourceRow(project_id="P-1", report_year=2025),),
            None,
            (),
            (),
            (),
            timeseries_complete_observation=True,
            timeseries_observed_years=(2025,),
        ),
        duckdb_path=duck_path,
        sqlite_path=sqlite_path,
        reindex=False,
    )
    with duckdb.connect(str(duck_path)) as conn:
        conn.execute("DROP TABLE project_timeseries")

    with pytest.raises(ValueError, match="schema is incomplete"):
        ensure_timeseries_baseline(duck_path, sqlite_path)


def _arrived_timeseries_snapshot(path: Path) -> ApiSnapshot:
    conn = get_sqlite_connection(path)
    ensure_timeseries_tables(conn)
    conn.execute("BEGIN")
    snapshot = replace_timeseries_in_transaction(
        conn,
        [
            normalize_timeseries_row(
                ("report_year", "project_id", "year", "cprd_grs_oil"),
                (2025, "arrived", 2025, 1.0),
                origin="api_json",
            )
        ],
        years=None,
        revision="arrived-r1",
        fetched_at="2026-10-24T00:00:00+00:00",
    )
    record_timeseries_provenance(
        conn,
        revision=snapshot.revision,
        origin="api_json",
        details={"observed_years": [2025]},
    )
    conn.commit()
    conn.close()
    return snapshot


def test_c1_refuses_timeseries_arrival_after_empty_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sqlite_path = tmp_path / "esdc.sqlite"
    duck_path = tmp_path / "esdc.duckdb"
    real_connect = get_sqlite_connection
    injected = False

    def connect(path: Path):
        nonlocal injected
        conn = real_connect(path)
        if not injected:
            injected = True
            ensure_timeseries_tables(conn)
            conn.execute("BEGIN")
            snapshot = replace_timeseries_in_transaction(
                conn,
                [
                    normalize_timeseries_row(
                        ("report_year", "project_id", "year", "cprd_grs_oil"),
                        (2025, "arrived", 2025, 1.0),
                        origin="api_json",
                    )
                ],
                years=None,
                revision="arrived-r1",
                fetched_at="2026-10-24T00:00:00+00:00",
            )
            record_timeseries_provenance(
                conn,
                revision=snapshot.revision,
                origin="api_json",
                details={"observed_years": [2025]},
            )
            conn.commit()
        return conn

    monkeypatch.setattr("esdc.fetch_publication.get_sqlite_connection", connect)
    with pytest.raises(
        FetchFailure, match="source appeared during preparation"
    ) as error:
        publish_prepared_fetch(
            PreparedFetch(
                (ApiProjectResourceRow(project_id="resource", report_year=2025),),
                None,
                (),
                (),
                (),
            ),
            duckdb_path=duck_path,
            sqlite_path=sqlite_path,
            reindex=False,
        )
    assert error.value.sqlite_committed is False
    assert error.value.duckdb_committed is False
    assert not duck_path.exists()
    with sqlite3.connect(sqlite_path) as conn:
        assert get_project_resources_snapshot(conn) is None
        assert get_timeseries_snapshot(conn) is not None


def test_c2_refuses_timeseries_arrival_after_empty_pinned_capture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sqlite_path = tmp_path / "esdc.sqlite"
    duck_path = tmp_path / "esdc.duckdb"
    publish_prepared_fetch(
        PreparedFetch(
            (ApiProjectResourceRow(project_id="old", report_year=2024),),
            None,
            (),
            (),
            (),
        ),
        duckdb_path=duck_path,
        sqlite_path=sqlite_path,
        reindex=False,
    )
    with sqlite3.connect(sqlite_path) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
    with duckdb.connect(str(duck_path), read_only=True) as conn:
        old_resources = conn.execute(
            "SELECT project_id, report_year FROM project_resources"
        ).fetchall()
        old_marker = conn.execute(
            "SELECT revision, source_row_count, source_sha256 FROM api_publication "
            "WHERE source_table='project_resources'"
        ).fetchone()
    real_stage = __import__(
        "esdc.fetch_publication", fromlist=["x"]
    )._stage_project_resources
    injected = False

    def stage_and_inject(conn, rows):
        nonlocal injected
        stage = real_stage(conn, rows)
        if not injected:
            injected = True
            _arrived_timeseries_snapshot(sqlite_path)
        return stage

    monkeypatch.setattr(
        "esdc.fetch_publication._stage_project_resources", stage_and_inject
    )
    with pytest.raises(FetchFailure, match="source appeared before C2 commit") as error:
        publish_prepared_fetch(
            PreparedFetch(
                (ApiProjectResourceRow(project_id="new", report_year=2025),),
                None,
                (),
                (),
                (),
            ),
            duckdb_path=duck_path,
            sqlite_path=sqlite_path,
            reindex=False,
        )
    assert error.value.sqlite_committed is True
    assert error.value.duckdb_committed is False
    with duckdb.connect(str(duck_path), read_only=True) as conn:
        assert (
            conn.execute(
                "SELECT project_id, report_year FROM project_resources"
            ).fetchall()
            == old_resources
        )
        assert (
            conn.execute(
                "SELECT revision, source_row_count, source_sha256 FROM api_publication "
                "WHERE source_table='project_resources'"
            ).fetchone()
            == old_marker
        )
    with sqlite3.connect(sqlite_path) as conn:
        resource = get_project_resources_snapshot(conn)
        timeseries = get_timeseries_snapshot(conn)
    assert resource is not None and resource.publication_state == "analytics_stale"
    assert timeseries is not None and timeseries.revision == "arrived-r1"
    assert timeseries.publication_state == "committed"


def _canonical_state(path: Path):
    with sqlite3.connect(path) as conn:
        conn.row_factory = sqlite3.Row
        return get_project_resources_snapshot(conn)


def _analytical_pair(path: Path):
    with duckdb.connect(str(path), read_only=True) as conn:
        resources = conn.execute(
            "SELECT project_id, report_year FROM project_resources ORDER BY project_id"
        ).fetchall()
        timeseries = conn.execute(
            "SELECT project_id, report_year, cprd_grs_oil "
            "FROM project_timeseries ORDER BY report_year"
        ).fetchall()
        lineage = conn.execute(
            "SELECT revision, source_row_count, source_sha256 FROM api_publication"
        ).fetchall()
    return resources, timeseries, lineage


def _publication_marker(path: Path, source_table: str) -> tuple[str, int, str] | None:
    with duckdb.connect(str(path), read_only=True) as conn:
        return conn.execute(
            "SELECT revision, source_row_count, source_sha256 FROM api_publication "
            "WHERE source_table=?",
            [source_table],
        ).fetchone()


def test_duckdb_commit_failure_rolls_back_joint_tables_and_lineage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sqlite_path = tmp_path / "esdc.sqlite"
    duck_path = tmp_path / "esdc.duckdb"
    old_revision = _seed_coherent_pair(sqlite_path, duck_path, tmp_path)
    old_pair = _analytical_pair(duck_path)
    original_connect = __import__(
        "esdc.fetch_publication", fromlist=["x"]
    ).get_duckdb_connection

    class CommitFailureProxy:
        def __init__(self, connection):
            self.connection = connection

        def execute(self, sql, *args, **kwargs):
            if sql == "COMMIT":
                raise duckdb.IOException("injected C2 commit failure")
            return self.connection.execute(sql, *args, **kwargs)

        def __getattr__(self, name):
            return getattr(self.connection, name)

    monkeypatch.setattr(
        "esdc.fetch_publication.get_duckdb_connection",
        lambda *args, **kwargs: CommitFailureProxy(original_connect(*args, **kwargs)),
    )
    with pytest.raises(FetchFailure) as error:
        publish_prepared_fetch(
            _replacement(tmp_path),
            duckdb_path=duck_path,
            sqlite_path=sqlite_path,
            reindex=False,
        )
    assert error.value.sqlite_committed is True
    assert error.value.duckdb_committed is False
    current = _canonical_state(sqlite_path)
    assert current is not None and current.revision != old_revision
    assert current.publication_state == "analytics_stale"
    assert _analytical_pair(duck_path) == old_pair


def test_timeseries_only_publication_preserves_resource_snapshot_and_lineage(
    tmp_path: Path,
) -> None:
    sqlite_path = tmp_path / "esdc.sqlite"
    duck_path = tmp_path / "esdc.duckdb"
    old_revision = _seed_coherent_pair(sqlite_path, duck_path, tmp_path)
    snapshot_before = _canonical_state(sqlite_path)
    resources_before, _, _ = _analytical_pair(duck_path)
    resource_marker_before = _publication_marker(duck_path, "project_resources")
    outcome = publish_prepared_fetch(
        PreparedFetch(
            (),
            None,
            (_slice(tmp_path / "ts-only", "new-ts", 2025, 20),),
            (),
            (),
        ),
        duckdb_path=duck_path,
        sqlite_path=sqlite_path,
        reindex=False,
    )
    snapshot_after = _canonical_state(sqlite_path)
    resources_after, timeseries_after, _ = _analytical_pair(duck_path)
    assert outcome.resource_revision is None
    assert snapshot_after == snapshot_before
    assert snapshot_after is not None and snapshot_after.revision == old_revision
    assert resources_after == resources_before
    assert _publication_marker(duck_path, "project_resources") == resource_marker_before
    assert ("new-ts", 2025, 20.0) in timeseries_after
    assert _publication_marker(duck_path, "project_timeseries") == (
        outcome.timeseries_revision,
        outcome.timeseries_row_count,
        outcome.timeseries_source_sha256,
    )


def test_post_c2_failures_attempt_cache_checkpoint_reindex_after_close_and_stale_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sqlite_path = tmp_path / "esdc.sqlite"
    duck_path = tmp_path / "esdc.duckdb"
    old_revision = _seed_coherent_pair(sqlite_path, duck_path, tmp_path)
    original_connect = __import__(
        "esdc.fetch_publication", fromlist=["x"]
    ).get_duckdb_connection
    events: list[str] = []

    class CloseObserverProxy:
        def __init__(self, connection):
            self.connection = connection

        def close(self):
            events.append("close")
            return self.connection.close()

        def __getattr__(self, name):
            return getattr(self.connection, name)

    monkeypatch.setattr(
        "esdc.fetch_publication.get_duckdb_connection",
        lambda *args, **kwargs: CloseObserverProxy(original_connect(*args, **kwargs)),
    )

    def cache_failure():
        events.append("cache")
        raise RuntimeError("injected cache failure")

    def checkpoint_failure(conn):
        events.append("checkpoint")
        raise RuntimeError("injected checkpoint failure")

    def reindex_failure(path, *, strict):
        events.append("reindex")
        raise RuntimeError("injected strict reindex failure")

    monkeypatch.setattr(
        "esdc.api_projection._invalidate_analytical_caches", cache_failure
    )
    monkeypatch.setattr("esdc.fetch_publication._checkpoint", checkpoint_failure)
    monkeypatch.setattr("esdc.fetch_publication._reindex_fts_for_path", reindex_failure)

    with pytest.raises(FetchFailure) as error:
        publish_prepared_fetch(
            _replacement(tmp_path),
            duckdb_path=duck_path,
            sqlite_path=sqlite_path,
            reindex=True,
        )

    assert error.value.sqlite_committed is True
    assert error.value.duckdb_committed is True
    assert "New DuckDB data is committed" in str(error.value)
    assert "retains its prior analytical state" not in str(error.value)
    assert events.count("cache") == 1
    assert events.count("checkpoint") == 1
    assert events.index("close") < events.index("reindex")
    assert events[-1] == "reindex"
    current = _canonical_state(sqlite_path)
    assert current is not None and current.revision != old_revision
    assert current.publication_state == "analytics_stale"
    resources, timeseries, lineage = _analytical_pair(duck_path)
    assert resources == [("new", 2025)]
    assert ("new", 2025, 20.0) in timeseries
    assert lineage[0][0] == current.revision


def test_c2_superseded_revision_rolls_back_without_marking_newer_snapshot_stale(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sqlite_path = tmp_path / "esdc.sqlite"
    duck_path = tmp_path / "esdc.duckdb"
    old_revision = _seed_coherent_pair(sqlite_path, duck_path, tmp_path)
    with sqlite3.connect(sqlite_path) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
    old_pair = _analytical_pair(duck_path)
    real_record = __import__(
        "esdc.fetch_publication", fromlist=["x"]
    )._record_api_publication
    superseding_revision: str | None = None

    def supersede(conn, snapshot, *, invalid_geometry_count=0):
        nonlocal superseding_revision
        real_record(conn, snapshot, invalid_geometry_count=invalid_geometry_count)
        if snapshot.source_table != "project_resources":
            return
        sqlite_conn = get_sqlite_connection(sqlite_path)
        try:
            newer = commit_project_resources_update(
                sqlite_conn,
                (ApiProjectResourceRow(project_id="newer", report_year=2026),),
                years=None,
                expected_snapshot=snapshot,
            )
            superseding_revision = newer.revision
        finally:
            sqlite_conn.close()

    monkeypatch.setattr("esdc.fetch_publication._record_api_publication", supersede)
    with pytest.raises(FetchFailure, match="superseded before C2 commit") as error:
        publish_prepared_fetch(
            _replacement(tmp_path),
            duckdb_path=duck_path,
            sqlite_path=sqlite_path,
            reindex=False,
        )
    assert error.value.sqlite_committed is True
    assert error.value.duckdb_committed is False
    assert _analytical_pair(duck_path) == old_pair
    current = _canonical_state(sqlite_path)
    assert current is not None
    assert current.revision == superseding_revision
    assert superseding_revision is not None
    assert current.revision != old_revision
    assert current.publication_state == "committed"


def test_post_c2_keyboard_interrupt_is_preserved_after_earlier_maintenance_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sqlite_path = tmp_path / "esdc.sqlite"
    duck_path = tmp_path / "esdc.duckdb"
    _seed_coherent_pair(sqlite_path, duck_path, tmp_path)
    monkeypatch.setattr(
        "esdc.api_projection._invalidate_analytical_caches",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("cache failed")),
    )
    monkeypatch.setattr(
        "esdc.fetch_publication._reindex_fts_for_path",
        lambda *args, **kwargs: (_ for _ in ()).throw(KeyboardInterrupt("stop")),
    )
    with pytest.raises(KeyboardInterrupt, match="stop"):
        publish_prepared_fetch(
            _replacement(tmp_path),
            duckdb_path=duck_path,
            sqlite_path=sqlite_path,
            reindex=True,
        )
    current = _canonical_state(sqlite_path)
    assert current is not None
    assert current.publication_state == "analytics_stale"
    assert _analytical_pair(duck_path)[0] == [("new", 2025)]


def test_published_mark_follows_writer_close_and_strict_reindex(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sqlite_path = tmp_path / "esdc.sqlite"
    duck_path = tmp_path / "esdc.duckdb"
    _seed_coherent_pair(sqlite_path, duck_path, tmp_path)
    original_connect = __import__(
        "esdc.fetch_publication", fromlist=["x"]
    ).get_duckdb_connection
    original_mark = __import__(
        "esdc.fetch_publication", fromlist=["x"]
    ).mark_project_resources_published
    events: list[str] = []

    class CloseObserverProxy:
        def __init__(self, connection):
            self.connection = connection

        def close(self):
            events.append("close")
            return self.connection.close()

        def __getattr__(self, name):
            return getattr(self.connection, name)

    def mark(conn, revision, *, invalid_geometry_count):
        events.append("published")
        return original_mark(
            conn, revision, invalid_geometry_count=invalid_geometry_count
        )

    def reindex(path, *, strict):
        assert strict is True
        events.append("reindex")

    monkeypatch.setattr(
        "esdc.fetch_publication.get_duckdb_connection",
        lambda *args, **kwargs: CloseObserverProxy(original_connect(*args, **kwargs)),
    )
    monkeypatch.setattr("esdc.fetch_publication.mark_project_resources_published", mark)
    monkeypatch.setattr("esdc.fetch_publication._reindex_fts_for_path", reindex)
    outcome = publish_prepared_fetch(
        _replacement(tmp_path),
        duckdb_path=duck_path,
        sqlite_path=sqlite_path,
        reindex=True,
    )
    assert outcome.resource_revision is not None
    assert events.index("close") < events.index("reindex") < events.index("published")
    snapshot = _canonical_state(sqlite_path)
    assert snapshot is not None and snapshot.publication_state == "published"


def test_bootstrap_precision_types_survive_incremental_update_and_fresh_projection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from esdc.dbmanager import _execute_sql_script
    from esdc.timeseries_projection import (
        apply_timeseries_stage,
        stage_canonical_timeseries,
    )

    monkeypatch.setattr(
        "esdc.api_projection._invalidate_analytical_caches", lambda: None
    )
    monkeypatch.setattr("esdc.api_projection._checkpoint", lambda conn: None)
    monkeypatch.setattr(
        "esdc.fetch_publication._reindex_fts_for_path", lambda *a, **k: None
    )

    sqlite_path = tmp_path / "esdc.sqlite"
    duck_path = tmp_path / "esdc.duckdb"
    source = get_sqlite_connection(sqlite_path)
    ensure_timeseries_tables(source)
    source_types = dict(TIMESERIES_SOURCE_TYPES)
    source_types["cprd_grs_oil"] = "DECIMAL(30,4)"
    source_types["ghv"] = "DOUBLE"
    legacy_rows = []
    for year, project in ((2024, "legacy-2024"), (2025, "legacy-2025")):
        values = dict.fromkeys(TIMESERIES_SOURCE_COLUMNS)
        values.update(
            report_year=year,
            project_id=project,
            year=year,
            cprd_grs_oil="12345678901234567890.1234",
            ghv=1.123456789012345,
        )
        legacy_rows.append(
            normalize_timeseries_row(
                TIMESERIES_SOURCE_COLUMNS,
                tuple(values[column] for column in TIMESERIES_SOURCE_COLUMNS),
                origin="legacy_duckdb",
            )
        )
    source.execute("BEGIN")
    snapshot = replace_timeseries_in_transaction(
        source,
        legacy_rows,
        years=None,
        revision="legacy-r1",
        fetched_at="2026-10-23T00:00:00+00:00",
    )
    record_timeseries_provenance(
        source,
        revision=snapshot.revision,
        origin="legacy_duckdb",
        details={"source_types": source_types},
    )
    source.commit()
    source.close()

    target = duckdb.connect(str(duck_path))
    _execute_sql_script(target, "create_table_project_timeseries.sql")
    target.execute("ALTER TABLE project_timeseries ADD COLUMN uuid VARCHAR")
    target.execute(
        "ALTER TABLE project_timeseries ALTER COLUMN cprd_grs_oil TYPE DECIMAL(30,4)"
    )
    target.execute("ALTER TABLE project_timeseries ALTER COLUMN ghv TYPE DOUBLE")
    target.executemany(
        "INSERT INTO project_timeseries(report_year, project_id, year, "
        "cprd_grs_oil, ghv) VALUES (?, ?, ?, ?, ?)",
        [
            (2024, "legacy-2024", 2024, "12345678901234567890.1234", 1.123456789012345),
            (2025, "legacy-2025", 2025, "12345678901234567890.1234", 1.123456789012345),
        ],
    )
    target.close()

    outcome = publish_prepared_fetch(
        PreparedFetch(
            (),
            None,
            (_slice(tmp_path / "incremental", "api-2024", 2024, 7),),
            (),
            (),
            timeseries_observed_years=(2024,),
            timeseries_complete_observation=True,
        ),
        duckdb_path=duck_path,
        sqlite_path=sqlite_path,
        reindex=False,
    )
    assert outcome.timeseries_revision is not None

    source = get_sqlite_connection(sqlite_path)
    row = source.execute(
        "SELECT details_json FROM api_snapshot_provenance "
        "WHERE source_table='project_timeseries' AND revision=?",
        (outcome.timeseries_revision,),
    ).fetchone()
    assert row is not None
    details = json.loads(row[0])
    expected_overrides = {
        column: source_types[column]
        for column, source_type in TIMESERIES_SOURCE_TYPES.items()
        if source_type != "TEXT"
    }
    expected_overrides.update(cprd_grs_oil="DECIMAL(30,4)", ghv="DOUBLE")
    assert details["legacy_type_overrides"] == expected_overrides
    source.close()

    with duckdb.connect(str(duck_path), read_only=True) as existing:
        assert existing.execute(
            "SELECT data_type FROM information_schema.columns "
            "WHERE table_name='project_timeseries' AND column_name='cprd_grs_oil'"
        ).fetchone() == ("DECIMAL(30,4)",)
        assert existing.execute(
            "SELECT data_type FROM information_schema.columns "
            "WHERE table_name='project_timeseries' AND column_name='ghv'"
        ).fetchone() == ("DOUBLE",)

    source = get_sqlite_connection(sqlite_path)
    fresh = duckdb.connect(":memory:")
    _skip_population_fts(monkeypatch)
    stage = stage_canonical_timeseries(fresh, source, years=None)
    fresh.execute("BEGIN")
    apply_timeseries_stage(fresh, stage)
    fresh.execute("COMMIT")
    assert fresh.execute(
        "SELECT data_type FROM information_schema.columns "
        "WHERE table_name='project_timeseries' AND column_name='cprd_grs_oil'"
    ).fetchone() == ("DECIMAL(30,4)",)
    assert fresh.execute(
        "SELECT data_type FROM information_schema.columns "
        "WHERE table_name='project_timeseries' AND column_name='ghv'"
    ).fetchone() == ("DOUBLE",)
    assert fresh.execute(
        "SELECT CAST(cprd_grs_oil AS VARCHAR), ghv FROM project_timeseries "
        "WHERE project_id='legacy-2025'"
    ).fetchone() == ("12345678901234567890.1234", 1.123456789012345)
    fresh.close()
    source.close()


def test_first_observed_empty_creates_snapshot_and_repeated_empty_keeps_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "esdc.api_projection._invalidate_analytical_caches", lambda: None
    )
    monkeypatch.setattr("esdc.api_projection._checkpoint", lambda conn: None)
    monkeypatch.setattr(
        "esdc.fetch_publication._reindex_fts_for_path", lambda *a, **k: None
    )
    sqlite_path = tmp_path / "esdc.sqlite"
    duck_path = tmp_path / "esdc.duckdb"

    initial = publish_prepared_fetch(
        PreparedFetch(
            (ApiProjectResourceRow(project_id="P-1", report_year=2025),),
            None,
            (),
            (),
            (),
            timeseries_complete_observation=True,
            timeseries_observed_years=(2025,),
        ),
        duckdb_path=duck_path,
        sqlite_path=sqlite_path,
        reindex=False,
    )
    assert initial.timeseries_revision is not None
    assert initial.timeseries_row_count == 0

    repeated = publish_prepared_fetch(
        PreparedFetch(
            (ApiProjectResourceRow(project_id="P-2", report_year=2026),),
            None,
            (),
            (),
            (),
            timeseries_complete_observation=True,
            timeseries_observed_years=(2026,),
        ),
        duckdb_path=duck_path,
        sqlite_path=sqlite_path,
        reindex=False,
    )
    assert repeated.timeseries_revision == initial.timeseries_revision
    assert repeated.timeseries_row_count == 0
    with sqlite3.connect(sqlite_path) as conn:
        current = conn.execute(
            "SELECT revision, row_count FROM api_snapshot "
            "WHERE source_table='project_timeseries'"
        ).fetchone()
    assert current == (initial.timeseries_revision, 0)
    with duckdb.connect(str(duck_path), read_only=True) as conn:
        assert conn.execute("SELECT COUNT(*) FROM project_timeseries").fetchone() == (
            0,
        )
        assert conn.execute(
            "SELECT revision FROM api_publication WHERE source_table='project_timeseries'"
        ).fetchone() == (initial.timeseries_revision,)


def test_c2_rollback_cleanup_error_keeps_keyboard_interrupt_and_marks_pair_stale(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sqlite_path = tmp_path / "esdc.sqlite"
    duck_path = tmp_path / "esdc.duckdb"
    _seed_coherent_pair(sqlite_path, duck_path, tmp_path)
    real_connect = __import__(
        "esdc.fetch_publication", fromlist=["x"]
    ).get_duckdb_connection

    class RollbackFailureProxy:
        def __init__(self, conn):
            self.conn = conn

        def execute(self, sql, *args, **kwargs):
            if sql == "ROLLBACK":
                raise duckdb.IOException("injected rollback cleanup failure")
            return self.conn.execute(sql, *args, **kwargs)

        def __getattr__(self, name):
            return getattr(self.conn, name)

    monkeypatch.setattr(
        "esdc.fetch_publication.get_duckdb_connection",
        lambda *args, **kwargs: RollbackFailureProxy(real_connect(*args, **kwargs)),
    )
    monkeypatch.setattr(
        "esdc.fetch_publication.stage_canonical_timeseries",
        lambda *args, **kwargs: (_ for _ in ()).throw(KeyboardInterrupt("stop")),
    )
    with pytest.raises(KeyboardInterrupt, match="stop"):
        publish_prepared_fetch(
            _replacement(tmp_path),
            duckdb_path=duck_path,
            sqlite_path=sqlite_path,
            reindex=False,
        )
    with sqlite3.connect(sqlite_path) as conn:
        states = conn.execute(
            "SELECT source_table, publication_state FROM api_snapshot "
            "WHERE source_table IN ('project_resources','project_timeseries') "
            "ORDER BY source_table"
        ).fetchall()
    assert states == [
        ("project_resources", "analytics_stale"),
        ("project_timeseries", "analytics_stale"),
    ]


def test_second_sqlite_dataset_failure_rolls_back_both(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sqlite_path = tmp_path / "esdc.sqlite"
    duck_path = tmp_path / "esdc.duckdb"
    with pytest.raises(FetchFailure, match="timeseries write failed"):
        monkeypatch.setattr(
            "esdc.fetch_publication.replace_timeseries_in_transaction",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                RuntimeError("timeseries write failed")
            ),
            raising=False,
        )
        publish_prepared_fetch(
            _replacement(tmp_path),
            duckdb_path=duck_path,
            sqlite_path=sqlite_path,
            reindex=False,
        )
    assert _canonical_state(sqlite_path) is None
    assert not duck_path.exists()


def test_c1_reads_no_duckdb_writer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []
    real_connect = __import__(
        "esdc.fetch_publication", fromlist=["x"]
    ).get_duckdb_connection

    def connect(path: Path, *, read_only: bool = False):
        events.append("duckdb-read" if read_only else "duckdb-write")
        return real_connect(path, read_only=read_only)

    real_commit = __import__(
        "esdc.fetch_publication", fromlist=["x"]
    ).get_sqlite_connection

    def sqlite_connect(path: Path):
        conn = real_commit(path)

        class CommitObserver:
            def __getattr__(self, name):
                return getattr(conn, name)

            def commit(self):
                events.append("c1-commit")
                return conn.commit()

            def close(self):
                return conn.close()

        return CommitObserver()

    monkeypatch.setattr("esdc.fetch_publication.get_duckdb_connection", connect)
    monkeypatch.setattr("esdc.fetch_publication.get_sqlite_connection", sqlite_connect)
    publish_prepared_fetch(
        _replacement(tmp_path),
        duckdb_path=tmp_path / "esdc.duckdb",
        sqlite_path=tmp_path / "esdc.sqlite",
        reindex=False,
    )
    assert events.index("c1-commit") < events.index("duckdb-write")


def test_c2_reads_timeseries_from_committed_sqlite_not_spool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepared = _replacement(tmp_path)
    rows_path = prepared.timeseries[0].rows_path
    real_connect = __import__(
        "esdc.fetch_publication", fromlist=["x"]
    ).get_sqlite_connection
    first = True

    def connect(path: Path):
        nonlocal first
        conn = real_connect(path)
        if not first:
            return conn
        first = False

        class CommitObserver:
            def __getattr__(self, name):
                return getattr(conn, name)

            def commit(self):
                conn.commit()
                rows_path.write_text("tampered after C1", encoding="utf-8")

            def close(self):
                return conn.close()

        return CommitObserver()

    monkeypatch.setattr("esdc.fetch_publication.get_sqlite_connection", connect)
    outcome = publish_prepared_fetch(
        prepared,
        duckdb_path=tmp_path / "esdc.duckdb",
        sqlite_path=tmp_path / "esdc.sqlite",
        reindex=False,
    )
    assert outcome.timeseries_row_count == 1
    with duckdb.connect(str(tmp_path / "esdc.duckdb"), read_only=True) as conn:
        assert conn.execute(
            "SELECT project_id, cprd_grs_oil FROM project_timeseries"
        ).fetchall() == [("new", 20.0)]


def test_c2_failure_preserves_old_pair_and_marks_changed_sources_stale(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sqlite_path = tmp_path / "esdc.sqlite"
    duck_path = tmp_path / "esdc.duckdb"
    _seed_coherent_pair(sqlite_path, duck_path, tmp_path)
    old_pair = _analytical_pair(duck_path)
    monkeypatch.setattr(
        "esdc.fetch_publication.stage_canonical_timeseries",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("C2 failed")),
        raising=False,
    )
    with pytest.raises(FetchFailure, match="C2 failed"):
        publish_prepared_fetch(
            _replacement(tmp_path),
            duckdb_path=duck_path,
            sqlite_path=sqlite_path,
            reindex=False,
        )
    assert _analytical_pair(duck_path) == old_pair
    with sqlite3.connect(sqlite_path) as conn:
        states = dict(
            conn.execute("SELECT source_table, publication_state FROM api_snapshot")
        )
        assert states["project_resources"] == "analytics_stale"
        assert states["project_timeseries"] == "analytics_stale"


def test_noop_dataset_revision_stays_unchanged(tmp_path: Path) -> None:
    sqlite_path = tmp_path / "esdc.sqlite"
    duck_path = tmp_path / "esdc.duckdb"
    _seed_coherent_pair(sqlite_path, duck_path, tmp_path)
    with sqlite3.connect(sqlite_path) as conn:
        before_resource = conn.execute(
            "SELECT revision FROM api_snapshot WHERE source_table='project_resources'"
        ).fetchone()[0]
    outcome = publish_prepared_fetch(
        PreparedFetch(
            (), None, (_slice(tmp_path / "only-ts", "new-ts", 2026, 30),), (), ()
        ),
        duckdb_path=duck_path,
        sqlite_path=sqlite_path,
        reindex=False,
    )
    with sqlite3.connect(sqlite_path) as conn:
        after_resource = conn.execute(
            "SELECT revision FROM api_snapshot WHERE source_table='project_resources'"
        ).fetchone()[0]
        after_timeseries_row = conn.execute(
            "SELECT revision FROM api_snapshot WHERE source_table='project_timeseries'"
        ).fetchone()
        after_timeseries = (
            None if after_timeseries_row is None else after_timeseries_row[0]
        )
    assert before_resource == after_resource
    assert outcome.resource_revision is None
    assert after_timeseries is not None
    assert outcome.timeseries_revision == after_timeseries
    assert outcome.timeseries_row_count == 2


def test_timeseries_lineage_is_committed_with_pair(tmp_path: Path) -> None:
    sqlite_path = tmp_path / "esdc.sqlite"
    duck_path = tmp_path / "esdc.duckdb"
    outcome = publish_prepared_fetch(
        _replacement(tmp_path),
        duckdb_path=duck_path,
        sqlite_path=sqlite_path,
        reindex=False,
    )
    assert outcome.resource_revision is not None
    assert outcome.timeseries_revision == outcome.resource_revision
    with duckdb.connect(str(duck_path), read_only=True) as conn:
        rows = conn.execute(
            "SELECT source_table, revision, source_row_count, source_sha256 "
            "FROM api_publication ORDER BY source_table"
        ).fetchall()
    assert {row[0] for row in rows} == {"project_resources", "project_timeseries"}
    assert {row[1] for row in rows} == {outcome.resource_revision}
