"""Task5D2b: v2 publication wiring — committed source truth, then SQLite-only publication.

The coordinator boundary is exercised for real: mocked prepare produces a prepared POD
batch, the accepted source-only apply commits it, and the shared publisher consumes the
committed SQLite truth only. The network is mocked and must stay quiet during the
publication step; nothing here re-applies the source or issues a second generation.
Fixtures live under pytest's temporary task directory and the active operational
database is never opened.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from esdc import analytics_rebuild, fetch_publication
from esdc.analytics_rebuild import AnalyticsRebuildReport, get_api_analytics_status
from esdc.configs import Config
from esdc.fetch_publication import apply_prepared_pod_fetch, publish_prepared_fetch
from esdc.fetching import FetchFailure, FetchOutcome
from esdc.pod_registry.publication import get_pod_registry_status
from esdc.source_publication import (
    committed_pod_source_generation,
    publish_committed_pod_sources,
)
from esdc.timeseries_snapshot import get_timeseries_snapshot
from tests.pod_registry.test_fetch_prepare_v2 import PROJECT_ID, _pod_transport
from tests.pod_registry.test_fetch_v2 import (
    INPUT_TABLES,
    LEGACY_EXACT_VALUE,
    LEGACY_FETCHED_AT,
    SNAPSHOT_ROWS,
    _prepare,
    _snapshot_states,
    _sqlite_counts,
)

RESOURCE_ROWS = [
    {
        "project_id": PROJECT_ID,
        "report_year": 2024,
        "project_name": "Old",
        "field_geom": "POINT (1 2)",
    },
    {
        "project_id": "P001459F01",
        "report_year": 2025,
        "project_name": "New",
        "field_geom": "POINT (3 4)",
    },
]
TIMESERIES_ROWS = {2024: [{"report_year": 2024, "project_id": PROJECT_ID}]}


def _fixture_transport(monkeypatch: pytest.MonkeyPatch):
    """The accepted v2 prepare fixture, with a complete canonical pair."""
    transport = _pod_transport(monkeypatch)
    transport.resource_rows = [dict(row) for row in RESOURCE_ROWS]
    transport.timeseries_rows = {year: list(rows) for year, rows in TIMESERIES_ROWS.items()}
    return transport


def _identity_uuids(path: Path) -> dict[str, str]:
    """The identity UUIDs committed by the accepted adapter, read from SQLite only."""
    columns = {
        "m_pod": "pod_uuid",
        "pod_project": "pod_project_uuid",
        "pod_monitoring": "monpod_uuid",
        "pod_forecast": "frcast_uuid",
    }
    connection = sqlite3.connect(path)
    try:
        return {
            table: str(connection.execute(f"SELECT {column} FROM {table} LIMIT 1").fetchone()[0])
            for table, column in columns.items()
        }
    finally:
        connection.close()


def _committed_generation(path: Path) -> int:
    connection = sqlite3.connect(path)
    try:
        return int(
            connection.execute(
                "SELECT source_generation FROM pod_registry_state WHERE singleton_id=1"
            ).fetchone()[0]
        )
    finally:
        connection.close()


def test_publish_routes_pod_batch_through_apply_then_the_shared_publisher(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import esdc.esdc as esdc

    transport = _pod_transport(monkeypatch)
    transport.timeseries_rows = {2024: [{"report_year": 2024, "project_id": PROJECT_ID}]}
    sqlite_path = tmp_path / "pod.sqlite"
    duckdb_path = tmp_path / "esdc.duckdb"

    prepared = _prepare(esdc, tmp_path / "spool", sqlite_path, None)
    assert prepared.pod_batch is not None
    prepare_calls = list(transport.calls)

    applies: list[str] = []
    real_apply = fetch_publication.apply_prepared_pod_fetch

    def counting_apply(prepared_fetch, **kwargs):
        applies.append(str(prepared_fetch.pod_batch is not None))
        return real_apply(prepared_fetch, **kwargs)

    monkeypatch.setattr(fetch_publication, "apply_prepared_pod_fetch", counting_apply)

    def fail_publisher(*args, **kwargs):
        raise RuntimeError("independent publication failure")

    monkeypatch.setattr(analytics_rebuild, "rebuild_api_analytics", fail_publisher)

    with pytest.raises(FetchFailure, match="independent publication failure") as caught:
        publish_prepared_fetch(
            prepared,
            duckdb_path=duckdb_path,
            sqlite_path=sqlite_path,
            reindex=False,
        )

    failure = caught.value
    # C1 committed source truth; the publisher failed independently, before any write.
    assert failure.sqlite_committed is True
    assert failure.duckdb_committed is False
    assert not duckdb_path.exists()

    # The prepared batch must have been committed as exactly one source generation.
    assert _sqlite_counts(sqlite_path, INPUT_TABLES) == dict.fromkeys(INPUT_TABLES, 1)
    assert _sqlite_counts(sqlite_path, SNAPSHOT_ROWS) == {
        "api_project_resources": 2,
        "api_project_timeseries": 1,
    }
    assert _snapshot_states(sqlite_path) == {
        "project_resources": "committed",
        "project_timeseries": "committed",
    }
    with sqlite3.connect(sqlite_path) as source:
        generation = int(
            source.execute(
                "SELECT source_generation FROM pod_registry_state WHERE singleton_id=1"
            ).fetchone()[0]
        )
    assert generation == 1

    # Publication reads committed SQLite only: no HTTP and no second source apply.
    assert transport.calls == prepare_calls
    assert len(applies) == 1


def test_shared_publisher_projects_pair_then_pod_and_retries_from_committed_truth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import esdc.esdc as esdc

    transport = _fixture_transport(monkeypatch)
    sqlite_path = tmp_path / "pod.sqlite"
    duckdb_path = tmp_path / "esdc.duckdb"

    prepared = _prepare(esdc, tmp_path / "spool", sqlite_path, None)
    prepare_calls = list(transport.calls)

    def no_http(table, filetype, username, password, report_year):
        raise AssertionError("publication must read committed SQLite, not the network")

    monkeypatch.setattr(esdc, "_download_table_bytes", no_http)

    outcome = publish_prepared_fetch(
        prepared,
        duckdb_path=duckdb_path,
        sqlite_path=sqlite_path,
        reindex=False,
    )

    assert outcome.pod_source_generation == 1
    assert transport.calls == prepare_calls

    status = get_api_analytics_status(duckdb_path=duckdb_path, sqlite_path=sqlite_path)
    assert status["project_resources"]["status"] == "published"
    assert status["project_timeseries"]["status"] == "published"
    registry = get_pod_registry_status(sqlite_path, duckdb_path)
    assert registry.state == "fresh"
    assert registry.source_generation == 1

    # The same committed truth republished is a SQLite-only retry: no HTTP, no new
    # generation, and no new identity UUID.
    identities = _identity_uuids(sqlite_path)
    report = publish_committed_pod_sources(
        sqlite_path,
        duckdb_path,
        expected_generation=1,
        reindex=False,
    )
    assert transport.calls == prepare_calls
    assert _committed_generation(sqlite_path) == 1
    assert _identity_uuids(sqlite_path) == identities
    assert report.subsystem_status == {"api": "restored", "pod_registry": "restored"}


def test_valid_empty_observed_snapshots_still_publish_from_committed_truth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import esdc.esdc as esdc

    transport = _fixture_transport(monkeypatch)
    transport.resource_rows = []
    transport.timeseries_rows = {2024: []}
    sqlite_path = tmp_path / "pod.sqlite"
    duckdb_path = tmp_path / "esdc.duckdb"

    prepared = _prepare(esdc, tmp_path / "spool", sqlite_path, (2024,))
    outcome = publish_prepared_fetch(
        prepared,
        duckdb_path=duckdb_path,
        sqlite_path=sqlite_path,
        reindex=False,
    )

    # A legitimate observed-empty pair is committed source truth and must publish;
    # nothing may claim rows that were never observed.
    assert outcome.resource_row_count == 0
    assert outcome.timeseries_rows_written == 0
    status = get_api_analytics_status(duckdb_path=duckdb_path, sqlite_path=sqlite_path)
    assert status["project_resources"]["status"] == "published", status
    assert status["project_timeseries"]["status"] == "published", status
    assert get_pod_registry_status(sqlite_path, duckdb_path).state == "fresh"


def test_legacy_recovery_caller_keeps_its_nonempty_guard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import esdc.esdc as esdc

    transport = _fixture_transport(monkeypatch)
    transport.resource_rows = []
    transport.timeseries_rows = {2024: []}
    sqlite_path = tmp_path / "pod.sqlite"
    duckdb_path = tmp_path / "esdc.duckdb"

    prepared = _prepare(esdc, tmp_path / "spool", sqlite_path, (2024,))
    outcome = publish_prepared_fetch(
        prepared,
        duckdb_path=duckdb_path,
        sqlite_path=sqlite_path,
        reindex=False,
    )
    assert outcome.resource_row_count == 0, outcome

    # The same committed truth publishes through the v2 route, but a legacy recovery
    # caller (no expected generation) still refuses an empty canonical resource
    # snapshot: the v2 adaptation did not weaken that guard.
    with pytest.raises(ValueError) as raised:
        analytics_rebuild.rebuild_api_analytics(
            duckdb_path=duckdb_path, sqlite_path=sqlite_path, reindex=False
        )
    assert "requires a verified nonempty source" in str(raised.value), raised.value

    report = publish_committed_pod_sources(
        sqlite_path,
        duckdb_path,
        expected_generation=outcome.pod_source_generation,
        reindex=False,
    )
    assert report.fresh, report


def test_never_observed_timeseries_partner_stays_absent_and_preserves_cached_projection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import duckdb

    import esdc.esdc as esdc

    transport = _pod_transport(monkeypatch)
    transport.resource_rows = []
    transport.timeseries_rows = {}
    sqlite_path = tmp_path / "pod.sqlite"
    duckdb_path = tmp_path / "esdc.duckdb"

    prepared = _prepare(esdc, tmp_path / "spool", sqlite_path, None)
    # Full resources answered valid-empty and no year was derived for the partner: the
    # timeseries scope was never observed, so it must not become a fabricated snapshot.
    assert prepared.timeseries_observed_years == ()
    prepare_calls = list(transport.calls)
    assert [call for call in prepare_calls if call[0] == "project_timeseries"] == []
    assert len(prepare_calls) == 6

    # Unowned analytical data that this fetch never observed must survive untouched.
    with duckdb.connect(str(duckdb_path)) as cache:
        cache.execute(
            "CREATE TABLE project_timeseries (project_id VARCHAR, report_year INTEGER)"
        )
        cache.execute("INSERT INTO project_timeseries VALUES ('P-OTHER', 2019)")

    outcome = apply_prepared_pod_fetch(
        prepared, sqlite_path=sqlite_path, duckdb_path=duckdb_path
    )
    report = publish_committed_pod_sources(
        sqlite_path,
        duckdb_path,
        expected_generation=outcome.pod_source_generation,
        reindex=False,
    )

    statuses = {component.name: component.status for component in report.components}
    assert statuses["project_resources"] == "published"
    assert statuses["project_timeseries"] == "absent"
    assert statuses["pod_projection"] == "published"
    # The pair plus the POD projection are gated together: an unobserved part cannot
    # let the whole fetch claim freshness.
    assert report.fresh is False
    assert report.object_counts["project_timeseries"] == 0
    assert transport.calls == prepare_calls
    with duckdb.connect(str(duckdb_path), read_only=True) as mirror:
        assert mirror.execute(
            "SELECT project_id, report_year FROM project_timeseries ORDER BY project_id"
        ).fetchall() == [("P-OTHER", 2019)]


def test_pod_freshness_alone_never_proves_the_whole_fetch_is_fresh(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import esdc.esdc as esdc
    from esdc.pod_registry.publication import publish_v2_registry
    from esdc.source_publication import get_fetch_publication_status

    _fixture_transport(monkeypatch)
    sqlite_path = tmp_path / "pod.sqlite"
    duckdb_path = tmp_path / "esdc.duckdb"

    prepared = _prepare(esdc, tmp_path / "spool", sqlite_path, None)
    outcome = apply_prepared_pod_fetch(
        prepared, sqlite_path=sqlite_path, duckdb_path=duckdb_path
    )

    # The POD projection is published while the canonical pair is still pending: a POD
    # fresh marker alone must not prove the whole fetch is fresh.
    publish_v2_registry(sqlite_path, duckdb_path)
    assert get_pod_registry_status(sqlite_path, duckdb_path).state == "fresh"

    gate = get_fetch_publication_status(sqlite_path=sqlite_path, duckdb_path=duckdb_path)
    statuses = {component.name: component.status for component in gate.components}
    assert statuses["project_resources"] == "pending"
    assert statuses["project_timeseries"] == "pending"
    assert statuses["pod_projection"] == "published"
    assert gate.fresh is False
    assert gate.recovery_command == "esdc rebuild-api-analytics"

    # Publishing the pair closes the gate for both parts together.
    publish_committed_pod_sources(
        sqlite_path,
        duckdb_path,
        expected_generation=outcome.pod_source_generation,
        reindex=False,
    )
    gate = get_fetch_publication_status(sqlite_path=sqlite_path, duckdb_path=duckdb_path)
    assert all(component.status == "published" for component in gate.components), gate
    assert gate.fresh is True


def test_pair_projection_failure_rolls_back_and_never_reaches_the_pod_projection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import duckdb

    import esdc.esdc as esdc
    from esdc.source_publication import get_fetch_publication_status

    transport = _fixture_transport(monkeypatch)
    sqlite_path = tmp_path / "pod.sqlite"
    duckdb_path = tmp_path / "esdc.duckdb"

    prepared = _prepare(esdc, tmp_path / "spool", sqlite_path, None)
    outcome = apply_prepared_pod_fetch(
        prepared, sqlite_path=sqlite_path, duckdb_path=duckdb_path
    )
    publish_committed_pod_sources(
        sqlite_path,
        duckdb_path,
        expected_generation=outcome.pod_source_generation,
        reindex=False,
    )

    # Protected analytical content that the shared route must never drop, plus the
    # baseline mirror produced by the first successful publication.
    with duckdb.connect(str(duckdb_path)) as mirror:
        mirror.execute("CREATE TABLE IF NOT EXISTS embeddings (id INTEGER, payload VARCHAR)")
        mirror.execute("INSERT INTO embeddings VALUES (1, 'cached')")
        resources_before = mirror.execute(
            "SELECT project_id FROM project_resources ORDER BY project_id"
        ).fetchall()
        timeseries_before = mirror.execute(
            "SELECT project_id, report_year FROM project_timeseries ORDER BY project_id"
        ).fetchall()

    # A second committed generation: the projection must not be replaced by work the
    # mirror cannot fully verify, and the POD step must never run for a failed pair.
    transport.resource_rows.append(
        {
            "project_id": "P999999",
            "report_year": 2025,
            "project_name": "Unpublished",
            "field_geom": "POINT (5 6)",
        }
    )
    second = _prepare(esdc, tmp_path / "spool-2", sqlite_path, None)
    second_outcome = apply_prepared_pod_fetch(
        second, sqlite_path=sqlite_path, duckdb_path=duckdb_path
    )
    assert second_outcome.pod_source_generation == 2

    original_swap = analytics_rebuild._swap_project_resources

    def exploding_swap(writer, stage, **kwargs):
        original_swap(writer, stage, **kwargs)
        raise RuntimeError("pair projection failed after writing")

    def pod_tripwire(*args, **kwargs):
        raise RuntimeError("the POD projection must not run when the pair fails")

    monkeypatch.setattr(analytics_rebuild, "_swap_project_resources", exploding_swap)
    monkeypatch.setattr(
        "esdc.pod_registry.publication.publish_v2_registry", pod_tripwire
    )

    prepare_calls = list(transport.calls)
    with pytest.raises(FetchFailure) as raised:
        publish_committed_pod_sources(
            sqlite_path,
            duckdb_path,
            expected_generation=second_outcome.pod_source_generation,
            reindex=False,
        )
    failure = raised.value
    assert "pair projection failed after writing" in str(failure), failure
    assert "must not run" not in str(failure)
    # The source stays committed; only the projection is refused.
    assert failure.sqlite_committed is True
    assert failure.duckdb_committed is False
    assert transport.calls == prepare_calls

    gate = get_fetch_publication_status(sqlite_path=sqlite_path, duckdb_path=duckdb_path)
    statuses = {component.name: component.status for component in gate.components}
    assert statuses["project_resources"] == "stale", gate
    assert statuses["project_timeseries"] == "stale", gate
    # The mirror marker still reports generation 1 while SQLite committed generation 2:
    # work from an older generation cannot be reported as globally fresh.
    assert statuses["pod_projection"] == "stale", gate
    assert gate.fresh is False

    with duckdb.connect(str(duckdb_path), read_only=True) as mirror:
        assert mirror.execute(
            "SELECT project_id FROM project_resources ORDER BY project_id"
        ).fetchall() == resources_before
        assert mirror.execute("SELECT id, payload FROM embeddings").fetchall() == [
            (1, "cached")
        ]
        assert mirror.execute(
            "SELECT project_id, report_year FROM project_timeseries ORDER BY project_id"
        ).fetchall() == timeseries_before


def test_generation_change_before_commit_rolls_back_and_cannot_claim_fresh(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import duckdb

    import esdc.esdc as esdc
    from esdc.source_publication import get_fetch_publication_status

    transport = _fixture_transport(monkeypatch)
    sqlite_path = tmp_path / "pod.sqlite"
    duckdb_path = tmp_path / "esdc.duckdb"

    prepared = _prepare(esdc, tmp_path / "spool", sqlite_path, None)
    outcome = apply_prepared_pod_fetch(
        prepared, sqlite_path=sqlite_path, duckdb_path=duckdb_path
    )
    publish_committed_pod_sources(
        sqlite_path,
        duckdb_path,
        expected_generation=outcome.pod_source_generation,
        reindex=False,
    )

    with duckdb.connect(str(duckdb_path)) as mirror:
        resources_before = mirror.execute(
            "SELECT project_id FROM project_resources ORDER BY project_id"
        ).fetchall()

    original_generation = analytics_rebuild._pod_generation
    observations: list[int | None] = []

    def racing_generation(connection):
        value = original_generation(connection)
        observations.append(value)
        # The captured generation is honored for the first read; the pre-commit recheck
        # sees another worker commit a newer generation, so this work is not current.
        if len(observations) == 1:
            return value
        return None if value is None else value + 1

    def pod_tripwire(*args, **kwargs):
        raise RuntimeError("the POD projection must not run for a rolled-back pair")

    monkeypatch.setattr(analytics_rebuild, "_pod_generation", racing_generation)
    monkeypatch.setattr(
        "esdc.pod_registry.publication.publish_v2_registry", pod_tripwire
    )

    prepare_calls = list(transport.calls)
    with pytest.raises(FetchFailure) as raised:
        publish_committed_pod_sources(
            sqlite_path,
            duckdb_path,
            expected_generation=outcome.pod_source_generation,
            reindex=False,
        )
    failure = raised.value
    assert "generation changed from 1 before commit" in str(failure), failure
    assert failure.sqlite_committed is True
    assert failure.duckdb_committed is False
    assert len(observations) >= 2
    assert transport.calls == prepare_calls

    # Committed SQLite truth records the failed publication as stale, and the gate
    # reports that truth instead of an old mirror marker.
    states = _snapshot_states(sqlite_path)
    assert states == {
        "project_resources": "analytics_stale",
        "project_timeseries": "analytics_stale",
    }, states
    gate = get_fetch_publication_status(sqlite_path=sqlite_path, duckdb_path=duckdb_path)
    statuses = {component.name: component.status for component in gate.components}
    assert statuses["project_resources"] == "stale", gate
    assert statuses["project_timeseries"] == "stale", gate
    # The POD marker itself is genuinely fresh for generation 1, but that alone cannot
    # prove the whole fetch: the rolled-back pair keeps the combined gate closed.
    assert statuses["pod_projection"] == "published", gate
    assert gate.fresh is False

    with duckdb.connect(str(duckdb_path), read_only=True) as mirror:
        assert mirror.execute(
            "SELECT project_id FROM project_resources ORDER BY project_id"
        ).fetchall() == resources_before
        assert mirror.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_name LIKE 'project_resources_stage%'"
        ).fetchall() == []


def test_empty_pod_projection_still_surfaces_its_projection_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import duckdb

    import esdc.esdc as esdc
    from esdc.pod_registry.publication import PodPublicationReport
    from esdc.source_publication import get_fetch_publication_status

    _fixture_transport(monkeypatch)
    sqlite_path = tmp_path / "pod.sqlite"
    duckdb_path = tmp_path / "esdc.duckdb"

    prepared = _prepare(esdc, tmp_path / "spool", sqlite_path, None)
    outcome = apply_prepared_pod_fetch(
        prepared, sqlite_path=sqlite_path, duckdb_path=duckdb_path
    )

    # An empty publication iterable must not be read as success: the report itself
    # carries the projection error and the route has to inspect it.
    monkeypatch.setattr(
        "esdc.pod_registry.publication.publish_v2_registry",
        lambda *args, **kwargs: PodPublicationReport(
            results=(),
            source_committed=True,
            projection_error="pod projection refused",
        ),
    )

    with pytest.raises(FetchFailure) as raised:
        publish_committed_pod_sources(
            sqlite_path,
            duckdb_path,
            expected_generation=outcome.pod_source_generation,
            reindex=False,
        )
    failure = raised.value
    assert "pod projection refused" in str(failure), failure
    assert "esdc rebuild-api-analytics" in str(failure), failure
    # The canonical pair committed first; only the POD projection is refused.
    assert failure.duckdb_committed is True
    assert failure.sqlite_committed is True

    gate = get_fetch_publication_status(sqlite_path=sqlite_path, duckdb_path=duckdb_path)
    statuses = {component.name: component.status for component in gate.components}
    assert statuses["project_resources"] == "published", gate
    assert statuses["project_timeseries"] == "published", gate
    assert statuses["pod_projection"] == "pending", gate
    assert gate.fresh is False

    with duckdb.connect(str(duckdb_path), read_only=True) as mirror:
        assert mirror.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema='main' AND table_name='pod_master_data'"
        ).fetchall() == []


def test_retained_legacy_exact_numeric_types_survive_publication_without_downgrade(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sqlite3

    import duckdb

    import esdc.esdc as esdc
    from esdc.pod_registry.read_models import refresh_pod_after_source_change
    from esdc.sqlite_runtime import configure_writer
    from esdc.timeseries_projection import _legacy_type_overrides
    from esdc.timeseries_snapshot import (
        TIMESERIES_SOURCE_COLUMNS,
        TIMESERIES_SOURCE_TYPES,
        normalize_timeseries_row,
        record_timeseries_provenance,
        replace_timeseries_in_transaction,
    )

    _fixture_transport(monkeypatch)
    sqlite_path = tmp_path / "pod.sqlite"
    duckdb_path = tmp_path / "esdc.duckdb"

    prepared = _prepare(esdc, tmp_path / "spool", sqlite_path, None)
    outcome = apply_prepared_pod_fetch(
        prepared, sqlite_path=sqlite_path, duckdb_path=duckdb_path
    )

    # A retained legacy canonical row whose exact numeric type is carried by the current
    # revision provenance, written through the accepted source APIs only.
    connection = sqlite3.connect(sqlite_path)
    try:
        configure_writer(connection)
        connection.row_factory = sqlite3.Row
        source_types = dict(TIMESERIES_SOURCE_TYPES)
        source_types["cprd_grs_oil"] = "DECIMAL(30,4)"
        rows = []
        for year in (2024, 2025):
            values = dict.fromkeys(TIMESERIES_SOURCE_COLUMNS)
            values.update(
                report_year=year,
                project_id=PROJECT_ID,
                year=year,
                cprd_grs_oil=LEGACY_EXACT_VALUE,
            )
            rows.append(
                normalize_timeseries_row(
                    TIMESERIES_SOURCE_COLUMNS,
                    tuple(values[column] for column in TIMESERIES_SOURCE_COLUMNS),
                    origin="legacy_duckdb",
                )
            )
        connection.execute("BEGIN IMMEDIATE")
        snapshot = replace_timeseries_in_transaction(
            connection,
            rows,
            years=None,
            revision="legacy-type-baseline",
            fetched_at=LEGACY_FETCHED_AT,
            refresh_read_models=False,
        )
        record_timeseries_provenance(
            connection,
            revision=snapshot.revision,
            origin="legacy_duckdb",
            details={"source_types": source_types},
        )
        refresh_pod_after_source_change(connection, actor="FA:parent-baseline")
        connection.commit()
        overrides = _legacy_type_overrides(connection)
    finally:
        connection.close()
    assert overrides["cprd_grs_oil"] == "DECIMAL(30,4)", overrides

    # The source APIs recorded a new generation for that write: the shared publisher
    # refuses the older expectation instead of publishing work that is not committed.
    with pytest.raises(FetchFailure) as raised:
        publish_committed_pod_sources(
            sqlite_path,
            duckdb_path,
            expected_generation=outcome.pod_source_generation,
            reindex=False,
        )
    generation = outcome.pod_source_generation
    assert generation is not None
    expected = str(raised.value)
    assert "does not match the expected committed generation" in expected, expected

    current = committed_pod_source_generation(sqlite_path)
    assert current == generation + 1, current
    report = publish_committed_pod_sources(
        sqlite_path,
        duckdb_path,
        expected_generation=current,
        reindex=False,
    )

    with duckdb.connect(str(duckdb_path), read_only=True) as mirror:
        column_type = mirror.execute(
            "SELECT data_type FROM information_schema.columns "
            "WHERE table_schema='main' AND table_name='project_timeseries' "
            "AND column_name='cprd_grs_oil'"
        ).fetchone()
        values = mirror.execute(
            "SELECT cprd_grs_oil FROM project_timeseries ORDER BY report_year"
        ).fetchall()

    assert column_type == ("DECIMAL(30,4)",), column_type
    assert [str(row[0]) for row in values] == [LEGACY_EXACT_VALUE, LEGACY_EXACT_VALUE], values
    assert report.object_counts["project_timeseries"] == 2


def test_retained_partner_stays_current_and_the_publisher_reports_its_truthful_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import esdc.esdc as esdc

    transport = _fixture_transport(monkeypatch)
    sqlite_path = tmp_path / "pod.sqlite"
    duckdb_path = tmp_path / "esdc.duckdb"

    prepared = _prepare(esdc, tmp_path / "spool", sqlite_path, None)
    first = apply_prepared_pod_fetch(
        prepared, sqlite_path=sqlite_path, duckdb_path=duckdb_path
    )
    publish_committed_pod_sources(
        sqlite_path,
        duckdb_path,
        expected_generation=first.pod_source_generation,
        reindex=False,
    )

    # The next fetch selects a year that answers empty: nothing is written, the retained
    # partner stays canonical and current, and its counts must not be reported as zero.
    transport.timeseries_rows = {}
    second = _prepare(esdc, tmp_path / "spool-2", sqlite_path, (2025,))
    outcome = apply_prepared_pod_fetch(
        second, sqlite_path=sqlite_path, duckdb_path=duckdb_path
    )
    assert outcome.timeseries_rows_written == 0
    assert outcome.pod_source_generation == 2

    connection = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
    try:
        connection.row_factory = sqlite3.Row
        snapshot = get_timeseries_snapshot(connection)
        assert snapshot is not None
        current_revision = snapshot.revision
    finally:
        connection.close()

    report = publish_committed_pod_sources(
        sqlite_path,
        duckdb_path,
        expected_generation=outcome.pod_source_generation,
        reindex=False,
    )
    components = {component.name: component for component in report.components}
    assert components["project_timeseries"].status == "published", report
    assert components["project_timeseries"].rows_written == 1, report
    # The publisher reports the current committed marker, not the outcome's zero.
    assert components["project_timeseries"].revision == current_revision, report
    assert all(component.status == "published" for component in report.components), report
    assert report.fresh is True


def test_public_route_final_outcome_merges_the_current_committed_truth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import esdc.esdc as esdc

    transport = _fixture_transport(monkeypatch)
    sqlite_path = tmp_path / "pod.sqlite"
    duckdb_path = tmp_path / "esdc.duckdb"

    prepared = _prepare(esdc, tmp_path / "spool", sqlite_path, None)
    first = publish_prepared_fetch(
        prepared, duckdb_path=duckdb_path, sqlite_path=sqlite_path, reindex=False
    )
    assert first.pod_source_generation == 1

    transport.timeseries_rows = {}
    second = _prepare(esdc, tmp_path / "spool-2", sqlite_path, (2025,))

    source_outcome: dict[str, FetchOutcome] = {}
    report_captured: dict[str, AnalyticsRebuildReport] = {}
    real_apply = fetch_publication.apply_prepared_pod_fetch
    real_publish = fetch_publication.source_publication.publish_committed_pod_sources

    def recording_apply(prepared_fetch, **kwargs):
        outcome = real_apply(prepared_fetch, **kwargs)
        source_outcome["outcome"] = outcome
        return outcome

    def recording_publish(source_path, target_path, **kwargs):
        report = real_publish(source_path, target_path, **kwargs)
        report_captured["report"] = report
        return report

    monkeypatch.setattr(fetch_publication, "apply_prepared_pod_fetch", recording_apply)
    monkeypatch.setattr(
        fetch_publication.source_publication,
        "publish_committed_pod_sources",
        recording_publish,
    )

    final = publish_prepared_fetch(
        second, duckdb_path=duckdb_path, sqlite_path=sqlite_path, reindex=False
    )
    committed = source_outcome["outcome"]
    report = report_captured["report"]

    # The outcome keeps its own observed scope, skipped years and changed-only counters.
    assert final.pod_source_generation == committed.pod_source_generation == 2
    assert final.sqlite_committed
    assert final.updated_timeseries_years == committed.updated_timeseries_years
    assert final.skipped_timeseries_years == committed.skipped_timeseries_years
    assert final.skipped_resource_years == committed.skipped_resource_years
    assert final.timeseries_rows_written == committed.timeseries_rows_written == 0

    # The published truth comes from the committed publication, not from the zero counter.
    assert final.timeseries_revision == report.source_revisions["project_timeseries"], (
        final,
        report,
    )
    assert final.timeseries_source_sha256 == report.source_hashes["project_timeseries"], (
        final,
        report,
    )
    assert final.timeseries_row_count == report.object_counts["project_timeseries"] == 1
    assert final.resource_revision == report.source_revisions["project_resources"], (
        final,
        report,
    )
    assert final.resource_source_sha256 == report.source_hashes["project_resources"], (
        final,
        report,
    )
    assert final.resource_row_count == report.object_counts["project_resources"] == 2
    assert (
        final.resource_invalid_geometry_count
        == report.object_counts["invalid_resource_geometry"]
    ), (final, report)


def test_no_reload_returns_before_writer_init_journal_or_publish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from typer.testing import CliRunner

    import esdc.esdc as esdc
    from esdc.esdc import app

    transport = _pod_transport(monkeypatch)
    sqlite_path = tmp_path / "pod.sqlite"
    missing_target = tmp_path / "mirror-missing" / "esdc.duckdb"

    touched: list[str] = []

    def tripwire(name: str):
        def fail(*args, **kwargs):
            touched.append(name)
            raise AssertionError(f"--no-reload must not run {name}")

        return fail

    monkeypatch.setattr("esdc.dbmanager.get_duckdb_connection", tripwire("writer"))
    monkeypatch.setattr("esdc.sqlite_runtime.configure_writer", tripwire("journal"))
    monkeypatch.setattr(
        "esdc.pod_registry.migrations.initialize_registry_v2", tripwire("init")
    )
    monkeypatch.setattr(esdc, "publish_prepared_fetch", tripwire("publish"))

    saved: list[str] = []
    original_save = esdc._save_download

    def record_save(table, filetype, data, report_year=None):
        saved.append(str(getattr(table, "value", table)))
        return original_save(table, filetype, data, report_year)

    monkeypatch.setattr(esdc, "_save_download", record_save)
    monkeypatch.setattr(Config, "get_credentials", lambda: ("fake-user", "fake-secret"))
    monkeypatch.setattr(esdc, "get_esdc_sqlite_path", lambda: sqlite_path)
    monkeypatch.setattr(Config, "get_db_file", lambda: missing_target)

    result = CliRunner().invoke(app, ["fetch", "--filetype", "json", "--no-reload"])

    assert result.exit_code == 0, result.output
    assert touched == []
    assert set(saved) == {"project_resources", "project_timeseries", *INPUT_TABLES}
    assert len(saved) == 8
    assert not missing_target.parent.exists()
    # Prepare validated and saved the payloads, but nothing was committed or published.
    assert "POD source generation: none created." in result.output, result.output
    assert "Nothing committed to SQLite" in result.output, result.output
    assert len(transport.calls) == 8


def test_cli_success_text_reports_generation_and_the_combined_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from typer.testing import CliRunner

    import esdc.esdc as esdc
    from esdc.esdc import app

    _fixture_transport(monkeypatch)
    sqlite_path = tmp_path / "pod.sqlite"
    duckdb_path = tmp_path / "esdc.duckdb"

    # The CLI route may publish; the legacy writer helpers must stay unused.
    monkeypatch.setattr(esdc, "publish_prepared_fetch", publish_prepared_fetch)
    monkeypatch.setattr(Config, "get_credentials", lambda: ("fake-user", "fake-secret"))
    monkeypatch.setattr(esdc, "get_esdc_sqlite_path", lambda: sqlite_path)
    monkeypatch.setattr(Config, "get_db_file", lambda: duckdb_path)

    result = CliRunner().invoke(app, ["fetch", "--filetype", "json"])

    assert result.exit_code == 0, result.output
    assert "POD source generation 1: SQLite committed." in result.output, result.output
    assert "Gate project_resources: published" in result.output, result.output
    assert "Gate project_timeseries: published" in result.output, result.output
    assert "Gate pod_projection: published" in result.output, result.output
    assert "Whole fetch is fresh for generation 1." in result.output, result.output


def test_cli_error_text_keeps_committed_source_and_pending_publication_separate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from typer.testing import CliRunner

    import esdc.esdc as esdc
    from esdc.esdc import app

    transport = _fixture_transport(monkeypatch)
    sqlite_path = tmp_path / "pod.sqlite"
    duckdb_path = tmp_path / "esdc.duckdb"

    monkeypatch.setattr(esdc, "publish_prepared_fetch", publish_prepared_fetch)
    monkeypatch.setattr(
        "esdc.analytics_rebuild.rebuild_api_analytics",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("pair publication refused")
        ),
    )
    monkeypatch.setattr(Config, "get_credentials", lambda: ("fake-user", "fake-secret"))
    monkeypatch.setattr(esdc, "get_esdc_sqlite_path", lambda: sqlite_path)
    monkeypatch.setattr(Config, "get_db_file", lambda: duckdb_path)

    result = CliRunner().invoke(app, ["fetch", "--filetype", "json"])

    assert result.exit_code == 1, result.output
    assert "pair publication refused" in result.output, result.output
    # The source is committed even though the publication is pending: the text reports
    # the two separately instead of claiming a completed fetch.
    assert "SQLite source committed at generation 1." in result.output, result.output
    assert "DuckDB publication pending." in result.output, result.output
    assert "esdc rebuild-api-analytics" in result.output, result.output
    assert len(transport.calls) == 8


def test_cli_reports_observed_empty_clears_and_an_absent_partner_truthfully(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from typer.testing import CliRunner

    import esdc.esdc as esdc
    from esdc.esdc import app

    transport = _pod_transport(monkeypatch)
    transport.resource_rows = []
    transport.timeseries_rows = {}
    sqlite_path = tmp_path / "pod.sqlite"
    duckdb_path = tmp_path / "esdc.duckdb"

    monkeypatch.setattr(esdc, "publish_prepared_fetch", publish_prepared_fetch)
    monkeypatch.setattr(Config, "get_credentials", lambda: ("fake-user", "fake-secret"))
    monkeypatch.setattr(esdc, "get_esdc_sqlite_path", lambda: sqlite_path)
    monkeypatch.setattr(Config, "get_db_file", lambda: duckdb_path)

    result = CliRunner().invoke(app, ["fetch", "--filetype", "json"])

    assert result.exit_code == 0, result.output
    assert "POD source generation 1: SQLite committed." in result.output, result.output
    assert "Gate project_resources: published" in result.output, result.output
    assert "Gate project_timeseries: absent" in result.output, result.output
    assert "Gate pod_projection: published" in result.output, result.output
    assert (
        "project_resources: observed-empty snapshot published." in result.output
    ), result.output
    assert "project_timeseries: never observed." in result.output, result.output
    assert "Whole fetch is not fresh for generation 1" in result.output, result.output


def test_verified_empty_source_clears_the_populated_mirror_for_the_v2_route(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import duckdb

    import esdc.esdc as esdc

    transport = _fixture_transport(monkeypatch)
    sqlite_path = tmp_path / "pod.sqlite"
    duckdb_path = tmp_path / "esdc.duckdb"

    def mirror_counts() -> dict[str, int]:
        with duckdb.connect(str(duckdb_path), read_only=True) as mirror:
            counts: dict[str, int] = {}
            for table in ("project_resources", "project_timeseries"):
                row = mirror.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
                assert row is not None
                counts[table] = int(row[0])
            return counts

    prepared = _prepare(esdc, tmp_path / "spool", sqlite_path, (2024,))
    first = apply_prepared_pod_fetch(
        prepared, sqlite_path=sqlite_path, duckdb_path=duckdb_path
    )
    publish_committed_pod_sources(
        sqlite_path,
        duckdb_path,
        expected_generation=first.pod_source_generation,
        reindex=False,
    )
    assert mirror_counts() == {"project_resources": 1, "project_timeseries": 1}

    # A verified empty observation for the same scope is the current canonical truth:
    # last source wins, so the populated mirror must be cleared from committed truth.
    transport.timeseries_rows = {2024: []}
    second = _prepare(esdc, tmp_path / "spool-2", sqlite_path, (2024,))
    outcome = apply_prepared_pod_fetch(second, sqlite_path=sqlite_path, duckdb_path=duckdb_path)
    assert outcome.pod_source_generation == 2
    assert _sqlite_counts(sqlite_path, SNAPSHOT_ROWS) == {
        "api_project_resources": 1,
        "api_project_timeseries": 0,
    }
    prepare_calls = list(transport.calls)
    identities_before = _identity_uuids(sqlite_path)

    # The explicit legacy route keeps its own guard for the same committed truth.
    with pytest.raises(ValueError, match="cannot replace a populated target"):
        analytics_rebuild.rebuild_api_analytics(
            duckdb_path=duckdb_path, sqlite_path=sqlite_path, reindex=False
        )
    assert mirror_counts() == {"project_resources": 1, "project_timeseries": 1}

    # One decision rule drives both sites; an optimistic preflight view must still be
    # refused inside the owned transaction when the mirror is really populated.
    assert not analytics_rebuild._empty_source_guard_decision(
        v2=False, table="project_timeseries", source_count=0, target_count=1
    )
    assert analytics_rebuild._empty_source_guard_decision(
        v2=True, table="project_timeseries", source_count=0, target_count=1
    )

    original_count = analytics_rebuild._mirror_row_count

    def optimistic_preflight(path: Path, table: str) -> int | None:
        return 0

    monkeypatch.setattr(analytics_rebuild, "_mirror_row_count", optimistic_preflight)
    with pytest.raises(ValueError, match="cannot replace a populated target"):
        analytics_rebuild.rebuild_api_analytics(
            duckdb_path=duckdb_path, sqlite_path=sqlite_path, reindex=False
        )
    assert mirror_counts() == {"project_resources": 1, "project_timeseries": 1}
    monkeypatch.setattr(analytics_rebuild, "_mirror_row_count", original_count)

    def no_http(*args, **kwargs):
        raise AssertionError("publication must read committed SQLite, not the network")

    monkeypatch.setattr(esdc, "_download_table_bytes", no_http)

    report = publish_committed_pod_sources(
        sqlite_path, duckdb_path, expected_generation=2, reindex=False
    )
    statuses = {component.name: component.status for component in report.components}
    assert statuses == {
        "project_resources": "published",
        "project_timeseries": "published",
        "pod_projection": "published",
    }
    assert report.fresh
    assert mirror_counts() == {"project_resources": 1, "project_timeseries": 0}

    # The SQLite-only retry repeats the clear with zero HTTP, no new UUID, no generation bump.
    repeat = publish_committed_pod_sources(
        sqlite_path, duckdb_path, expected_generation=2, reindex=False
    )
    assert {component.name: component.status for component in repeat.components} == statuses
    assert transport.calls == prepare_calls
    assert _identity_uuids(sqlite_path) == identities_before
    assert committed_pod_source_generation(sqlite_path) == 2
    assert duckdb_path.is_file()


def test_recovery_cli_publishes_the_v2_route_without_the_legacy_nonempty_refusal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json

    from typer.testing import CliRunner

    import esdc.esdc as esdc
    from esdc.esdc import app

    transport = _fixture_transport(monkeypatch)
    transport.resource_rows = []
    transport.timeseries_rows = {2024: []}
    sqlite_path = tmp_path / "pod.sqlite"
    duckdb_path = tmp_path / "esdc.duckdb"

    prepared = _prepare(esdc, tmp_path / "spool", sqlite_path, (2024,))
    outcome = apply_prepared_pod_fetch(
        prepared, sqlite_path=sqlite_path, duckdb_path=duckdb_path
    )
    prepare_calls = list(transport.calls)
    identities_before = _identity_uuids(sqlite_path)

    monkeypatch.setattr(Config, "get_db_file", lambda: duckdb_path)
    monkeypatch.setattr(
        "esdc.commands.analytics.get_esdc_sqlite_path", lambda: sqlite_path
    )

    result = CliRunner().invoke(app, ["rebuild-api-analytics"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["recovery_command"] == "esdc rebuild-api-analytics", payload
    assert payload["fresh"] is True, payload
    assert {component["name"]: component["status"] for component in payload["components"]} == {
        "project_resources": "published",
        "project_timeseries": "published",
        "pod_projection": "published",
    }, payload
    # The recovery route reads committed truth only: no HTTP, no source re-apply,
    # no new revision UUID and no generation bump.
    assert transport.calls == prepare_calls
    assert _identity_uuids(sqlite_path) == identities_before
    assert committed_pod_source_generation(sqlite_path) == outcome.pod_source_generation


def test_recovery_cli_keeps_a_never_observed_partner_absent_and_does_not_refuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json

    from typer.testing import CliRunner

    import esdc.esdc as esdc
    from esdc.esdc import app

    transport = _pod_transport(monkeypatch)
    transport.resource_rows = []
    transport.timeseries_rows = {}
    sqlite_path = tmp_path / "pod.sqlite"
    duckdb_path = tmp_path / "esdc.duckdb"

    prepared = _prepare(esdc, tmp_path / "spool", sqlite_path, None)
    outcome = apply_prepared_pod_fetch(
        prepared, sqlite_path=sqlite_path, duckdb_path=duckdb_path
    )
    prepare_calls = list(transport.calls)
    identities_before = _identity_uuids(sqlite_path)

    monkeypatch.setattr(Config, "get_db_file", lambda: duckdb_path)
    monkeypatch.setattr(
        "esdc.commands.analytics.get_esdc_sqlite_path", lambda: sqlite_path
    )

    result = CliRunner().invoke(app, ["rebuild-api-analytics"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert {component["name"]: component["status"] for component in payload["components"]} == {
        "project_resources": "published",
        "project_timeseries": "absent",
        "pod_projection": "published",
    }, payload
    assert payload["fresh"] is False, payload
    assert transport.calls == prepare_calls
    assert _identity_uuids(sqlite_path) == identities_before
    assert committed_pod_source_generation(sqlite_path) == outcome.pod_source_generation


def test_generation_change_after_the_pod_publish_is_not_claimed_fresh(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import duckdb

    import esdc.esdc as esdc
    from esdc.pod_registry import publication
    from esdc.source_publication import get_fetch_publication_status

    transport = _fixture_transport(monkeypatch)
    sqlite_path = tmp_path / "pod.sqlite"
    duckdb_path = tmp_path / "esdc.duckdb"

    prepared = _prepare(esdc, tmp_path / "spool", sqlite_path, (2024,))
    apply_prepared_pod_fetch(prepared, sqlite_path=sqlite_path, duckdb_path=duckdb_path)

    transport.timeseries_rows = {2024: [{"report_year": 2024, "project_id": "P-NEWER"}]}
    second = _prepare(esdc, tmp_path / "spool-2", sqlite_path, (2024,))

    real_publish = publication.publish_v2_registry

    def racing_publish(source_path: Path, target_path: Path):
        # A newer writer commits generation 2 after the pair stage has already committed.
        report = real_publish(source_path, target_path)
        apply_prepared_pod_fetch(second, sqlite_path=source_path, duckdb_path=target_path)
        return report

    monkeypatch.setattr(publication, "publish_v2_registry", racing_publish)

    with pytest.raises(FetchFailure) as raised:
        publish_committed_pod_sources(
            sqlite_path, duckdb_path, expected_generation=1, reindex=False
        )
    failure = raised.value
    assert failure.phase == "post_commit", failure
    assert failure.sqlite_committed and failure.duckdb_committed, failure
    assert "generation" in str(failure), failure

    # Nothing is rolled back: the newer committed generation and the valid cached parts
    # remain, and the gate reports the projected work as pending for generation 2.
    gate = get_fetch_publication_status(sqlite_path=sqlite_path, duckdb_path=duckdb_path)
    statuses = {component.name: component.status for component in gate.components}
    assert statuses == {
        "project_resources": "pending",
        "project_timeseries": "pending",
        "pod_projection": "stale",
    }, gate
    assert not gate.fresh, gate
    assert committed_pod_source_generation(sqlite_path) == 2

    with duckdb.connect(str(duckdb_path), read_only=True) as mirror:
        for table in ("project_resources", "project_timeseries"):
            row = mirror.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
            assert row is not None
            assert int(row[0]) == 1


def test_combined_gate_does_not_combine_reads_from_two_generations_as_fresh(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import esdc.esdc as esdc
    from esdc.pod_registry import publication
    from esdc.source_publication import get_fetch_publication_status

    transport = _fixture_transport(monkeypatch)
    sqlite_path = tmp_path / "pod.sqlite"
    duckdb_path = tmp_path / "esdc.duckdb"

    prepared = _prepare(esdc, tmp_path / "spool", sqlite_path, (2024,))
    apply_prepared_pod_fetch(prepared, sqlite_path=sqlite_path, duckdb_path=duckdb_path)
    publish_committed_pod_sources(
        sqlite_path, duckdb_path, expected_generation=1, reindex=False
    )

    transport.timeseries_rows = {2024: [{"report_year": 2024, "project_id": "P-NEWER"}]}
    second = _prepare(esdc, tmp_path / "spool-2", sqlite_path, (2024,))

    # The pair read measured generation 1; the POD read happens after generation 2 is
    # committed and published, so the gate must not combine the two bases.
    generation_one_read = analytics_rebuild.get_api_analytics_status(
        duckdb_path=duckdb_path, sqlite_path=sqlite_path
    )

    def stale_pair_read(*args, **kwargs):
        apply_prepared_pod_fetch(second, sqlite_path=sqlite_path, duckdb_path=duckdb_path)
        return generation_one_read

    original_pod_status = publication.get_pod_registry_status

    def racing_pod_status(source_path: Path, target_path: Path):
        publication.publish_v2_registry(source_path, target_path)
        return original_pod_status(source_path, target_path)

    monkeypatch.setattr(analytics_rebuild, "get_api_analytics_status", stale_pair_read)
    monkeypatch.setattr(publication, "get_pod_registry_status", racing_pod_status)

    gate = get_fetch_publication_status(sqlite_path=sqlite_path, duckdb_path=duckdb_path)

    assert not gate.fresh, gate
    assert all(component.status == "stale" for component in gate.components), gate
    assert all("generation changed" in (component.error or "") for component in gate.components), gate
    assert committed_pod_source_generation(sqlite_path) == 2
