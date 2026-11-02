"""Task5D2a: source-only atomic POD apply regressions.

Fixtures live under pytest's temporary task directory and the active operational
database is never opened. The application network is mocked; the accepted prepare
and shared apply adapters run for real. This slice must not open or publish DuckDB:
one owned SQLite writer transaction commits the validated POD batch plus the
canonical raw snapshots, then performs exactly one final read-model refresh.
"""

from __future__ import annotations

import json
import sqlite3
from decimal import Decimal
from pathlib import Path

import pytest

from esdc import fetch_publication
from esdc.api_snapshot import JSONScalar, get_project_resources_snapshot
from esdc.fetch_publication import apply_prepared_pod_fetch
from esdc.fetching import FetchFailure
from esdc.pod_registry.migrations import initialize_registry_v2
from esdc.pod_registry.read_models import refresh_pod_after_source_change
from esdc.selection import FileType, PodApiTable
from esdc.sqlite_runtime import configure_writer
from esdc.timeseries_projection import _legacy_type_overrides
from esdc.timeseries_snapshot import (
    TIMESERIES_SOURCE_COLUMNS,
    TIMESERIES_SOURCE_TYPES,
    CanonicalTimeseriesRow,
    ensure_timeseries_tables,
    get_timeseries_snapshot,
    normalize_timeseries_row,
    record_timeseries_provenance,
    replace_timeseries_in_transaction,
)
from tests.pod_registry.test_fetch_prepare_v2 import (
    POD_TABLES,
    PROJECT_ID,
    _pod_transport,
)

INPUT_TABLES = tuple(table.value for table in POD_TABLES)
SNAPSHOT_ROWS = ("api_project_resources", "api_project_timeseries")
OTHER_ACTOR = "FA:other-writer"

# Minimal old-shaped fixture SQL. This is test material only, never a production
# compatibility constant: the apply path refuses a legacy target instead of replacing
# it with the current schema.
_OLD_SHAPED_FIXTURE_SQL = """
CREATE TABLE m_pod (
    id INTEGER PRIMARY KEY,
    pod_id TEXT NOT NULL UNIQUE,
    pod_name TEXT,
    approval_date TEXT,
    approval_seq INTEGER NOT NULL UNIQUE
);
"""


def _sqlite_counts(path: Path, tables: tuple[str, ...]) -> dict[str, int]:
    connection = sqlite3.connect(path)
    try:
        return {
            table: int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
            for table in tables
        }
    finally:
        connection.close()


def _snapshot_states(path: Path) -> dict[str, str]:
    connection = sqlite3.connect(path)
    try:
        resources = get_project_resources_snapshot(connection)
        timeseries = get_timeseries_snapshot(connection)
        assert resources is not None and timeseries is not None
        return {
            "project_resources": str(resources.publication_state),
            "project_timeseries": str(timeseries.publication_state),
        }
    finally:
        connection.close()


def _prepare(esdc, spool: Path, sqlite_path: Path, years: tuple[int, ...] | None):
    return esdc.prepare_fetch(
        filetype=FileType.JSON,
        to_file=False,
        username="fake-user",
        password="fake-secret",
        years=years,
        directory=spool,
        sqlite_path=sqlite_path,
    )


def _combined_read_model_counts(path: Path) -> dict[str, int]:
    return _sqlite_counts(path, ("pod_monitoring_api", "pod_forecast_api"))


LEGACY_EXACT_VALUE = "12345678901234567890.1234"
LEGACY_BOOTSTRAP_REVISION = "legacy-bootstrap-revision"
LEGACY_FETCHED_AT = "2026-10-23T00:00:00+00:00"


def _numeric_type_map(source_types: dict[str, str]) -> dict[str, str]:
    """The projection consumer keeps only the non-TEXT columns."""
    return {
        column: source_types[column]
        for column in TIMESERIES_SOURCE_COLUMNS
        if TIMESERIES_SOURCE_TYPES[column] != "TEXT"
    }


def _legacy_row(year: int) -> CanonicalTimeseriesRow:
    """One full-width legacy row that preserves the exact numeric value as text."""
    values: list[JSONScalar] = [None] * len(TIMESERIES_SOURCE_COLUMNS)
    values[TIMESERIES_SOURCE_COLUMNS.index("report_year")] = year
    values[TIMESERIES_SOURCE_COLUMNS.index("project_id")] = PROJECT_ID
    values[TIMESERIES_SOURCE_COLUMNS.index("cprd_grs_oil")] = LEGACY_EXACT_VALUE
    return normalize_timeseries_row(
        TIMESERIES_SOURCE_COLUMNS, values, origin="legacy_duckdb"
    )


def _seed_legacy_timeseries_baseline(path: Path, years: tuple[int, ...]) -> dict[str, str]:
    """Write a legitimate legacy bootstrap baseline with the canonical writer APIs."""
    source_types = dict(TIMESERIES_SOURCE_TYPES)
    source_types["cprd_grs_oil"] = "DECIMAL(30,4)"
    connection = sqlite3.connect(path)
    try:
        configure_writer(connection)
        connection.row_factory = sqlite3.Row
        fetch_publication._ensure_canonical_snapshot_table(connection)
        ensure_timeseries_tables(connection)
        connection.execute("BEGIN IMMEDIATE")
        snapshot = replace_timeseries_in_transaction(
            connection,
            [_legacy_row(year) for year in years],
            years=None,
            revision=LEGACY_BOOTSTRAP_REVISION,
            fetched_at=LEGACY_FETCHED_AT,
            refresh_read_models=False,
        )
        record_timeseries_provenance(
            connection,
            revision=snapshot.revision,
            origin="legacy_duckdb",
            details={
                "source_types": source_types,
                "input_origins": ["legacy_duckdb"],
            },
        )
        connection.commit()
    finally:
        connection.close()
    return source_types


def _stored_timeseries_row(path: Path, year: int) -> dict[str, object]:
    connection = sqlite3.connect(path)
    try:
        row = connection.execute(
            "SELECT row_json FROM api_project_timeseries WHERE report_year = ?",
            (year,),
        ).fetchone()
    finally:
        connection.close()
    assert row is not None
    return json.loads(str(row[0]))


def _provenance_row(path: Path, revision: str | None) -> tuple[str, dict[str, object]]:
    assert revision is not None
    connection = sqlite3.connect(path)
    try:
        row = connection.execute(
            "SELECT origin, details_json FROM api_snapshot_provenance "
            "WHERE source_table = 'project_timeseries' AND revision = ?",
            (revision,),
        ).fetchone()
    finally:
        connection.close()
    assert row is not None
    return str(row[0]), json.loads(str(row[1]))


def _advance_target(path: Path) -> int:
    """Simulate another writer committing through the same shared v2 adapter."""
    connection = sqlite3.connect(path)
    try:
        configure_writer(connection)
        connection.row_factory = sqlite3.Row
        initialize_registry_v2(connection)
        connection.commit()
        connection.execute("BEGIN IMMEDIATE")
        refresh_pod_after_source_change(connection, actor=OTHER_ACTOR)
        generation = int(
            connection.execute(
                "SELECT source_generation FROM pod_registry_state WHERE singleton_id=1"
            ).fetchone()[0]
        )
        connection.commit()
        return generation
    finally:
        connection.close()


def test_source_apply_commits_the_pod_batch_and_raw_snapshots_in_one_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import esdc.esdc as esdc

    transport = _pod_transport(monkeypatch)
    transport.timeseries_rows = {2024: [{"report_year": 2024, "project_id": PROJECT_ID}]}
    sqlite_path = tmp_path / "pod.sqlite"
    duckdb_path = tmp_path / "esdc.duckdb"

    prepared = _prepare(esdc, tmp_path / "spool", sqlite_path, None)
    assert prepared.pod_batch is not None

    outcome = apply_prepared_pod_fetch(
        prepared,
        sqlite_path=sqlite_path,
        duckdb_path=duckdb_path,
    )

    assert outcome.sqlite_committed is True
    # The prepared target was fresh, so exactly one source generation is issued.
    assert outcome.pod_source_generation == 1
    assert _sqlite_counts(sqlite_path, INPUT_TABLES) == dict.fromkeys(INPUT_TABLES, 1)
    assert _sqlite_counts(sqlite_path, SNAPSHOT_ROWS) == {
        "api_project_resources": 2,
        "api_project_timeseries": 1,
    }
    assert outcome.resource_row_count == 2
    assert outcome.timeseries_row_count == 1

    # The result reports committed source truth only. Publication is a separate
    # concern and must not be claimed by the source-only helper.
    assert _snapshot_states(sqlite_path) == {
        "project_resources": "committed",
        "project_timeseries": "committed",
    }
    assert not duckdb_path.exists()


def test_one_final_refresh_records_the_combined_aggregate(tmp_path: Path, monkeypatch) -> None:
    import esdc.esdc as esdc

    transport = _pod_transport(monkeypatch)
    transport.timeseries_rows = {2024: [{"report_year": 2024, "project_id": PROJECT_ID}]}
    sqlite_path = tmp_path / "pod.sqlite"

    prepared = _prepare(esdc, tmp_path / "spool", sqlite_path, None)

    real_refresh = fetch_publication.refresh_pod_after_source_change
    reports: list = []

    def counting_refresh(conn, **kwargs):
        report = real_refresh(conn, **kwargs)
        reports.append(report)
        return report

    monkeypatch.setattr(fetch_publication, "refresh_pod_after_source_change", counting_refresh)
    outcome = apply_prepared_pod_fetch(
        prepared,
        sqlite_path=sqlite_path,
        duckdb_path=tmp_path / "esdc.duckdb",
    )

    # Three source changes (POD batch, resources, timeseries) share one final refresh.
    assert len(reports) == 1
    report = reports[0]
    assert report.source_generation == outcome.pod_source_generation == 1
    assert (report.resource_revision, report.resource_sha256) == (
        outcome.resource_revision,
        outcome.resource_source_sha256,
    )
    assert (report.timeseries_revision, report.timeseries_sha256) == (
        outcome.timeseries_revision,
        outcome.timeseries_source_sha256,
    )
    # The combined aggregate row and the physical read models agree with that report.
    connection = sqlite3.connect(sqlite_path)
    try:
        aggregate = connection.execute(
            "SELECT source_generation, resource_revision, resource_sha256, "
            "timeseries_revision, timeseries_sha256 FROM pod_read_model_snapshot "
            "WHERE singleton_id=1"
        ).fetchone()
    finally:
        connection.close()
    assert tuple(aggregate) == (
        report.source_generation,
        report.resource_revision,
        report.resource_sha256,
        report.timeseries_revision,
        report.timeseries_sha256,
    )
    assert _combined_read_model_counts(sqlite_path) == {
        "pod_monitoring_api": report.monitoring_rows,
        "pod_forecast_api": report.forecast_rows,
    }
    assert report.monitoring_rows == 1
    assert report.forecast_rows >= 1


def test_pod_facts_apply_when_raw_resources_answer_valid_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import esdc.esdc as esdc

    transport = _pod_transport(monkeypatch)
    transport.resource_rows = []
    sqlite_path = tmp_path / "pod.sqlite"

    prepared = _prepare(esdc, tmp_path / "spool", sqlite_path, None)
    outcome = apply_prepared_pod_fetch(
        prepared,
        sqlite_path=sqlite_path,
        duckdb_path=tmp_path / "esdc.duckdb",
    )

    # An empty response is still an observation; no early return is allowed.
    assert _sqlite_counts(sqlite_path, INPUT_TABLES) == dict.fromkeys(INPUT_TABLES, 1)
    assert outcome.resource_row_count == 0
    assert outcome.resource_revision is not None
    assert _sqlite_counts(sqlite_path, ("api_project_resources",)) == {
        "api_project_resources": 0
    }
    # No timeseries request was made, so that source state stays untouched.
    assert outcome.timeseries_revision is None
    connection = sqlite3.connect(sqlite_path)
    try:
        assert connection.execute(
            "SELECT COUNT(*) FROM api_snapshot WHERE source_table = 'project_timeseries'"
        ).fetchone()[0] == 0
    finally:
        connection.close()
    assert outcome.pod_source_generation == 1


def test_selected_empty_observation_clears_only_the_observed_years(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import esdc.esdc as esdc

    sqlite_path = tmp_path / "pod.sqlite"
    spool = tmp_path / "spool"
    duckdb_path = tmp_path / "esdc.duckdb"

    transport = _pod_transport(monkeypatch)
    transport.timeseries_rows = {2024: [{"report_year": 2024, "project_id": PROJECT_ID}]}
    baseline = _prepare(esdc, spool, sqlite_path, None)
    apply_prepared_pod_fetch(baseline, sqlite_path=sqlite_path, duckdb_path=duckdb_path)

    # The second fetch observes only 2025 and both endpoints answer valid-empty.
    transport.resource_rows = []
    transport.timeseries_rows = {}
    prepared = _prepare(esdc, spool, sqlite_path, (2025,))
    outcome = apply_prepared_pod_fetch(
        prepared,
        sqlite_path=sqlite_path,
        duckdb_path=duckdb_path,
    )

    # The unobserved 2024 canonical rows survive in both sources.
    connection = sqlite3.connect(sqlite_path)
    try:
        resource_years = sorted(
            str(row[0])
            for row in connection.execute("SELECT report_year FROM api_project_resources")
        )
        timeseries_years = sorted(
            str(row[0])
            for row in connection.execute("SELECT report_year FROM api_project_timeseries")
        )
        provenance = connection.execute(
            "SELECT details_json FROM api_snapshot_provenance "
            "WHERE source_table = 'project_timeseries' AND revision = ?",
            (outcome.timeseries_revision,),
        ).fetchone()
    finally:
        connection.close()
    assert resource_years == ["2024"]
    assert timeseries_years == ["2024"]
    assert outcome.updated_resource_years == (2025,)
    assert outcome.skipped_resource_years == (2025,)
    assert outcome.updated_timeseries_years == (2025,)
    assert outcome.skipped_timeseries_years == (2025,)
    assert provenance is not None
    details = json.loads(str(provenance[0]))
    assert details["observed_empty"] is True
    assert details["input_origins"] == ["api_json"]
    assert details["observed_years"] == [2025]
    assert outcome.pod_source_generation == 2


def test_selected_apply_retains_unselected_facts_and_the_global_forecast(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import esdc.esdc as esdc

    sqlite_path = tmp_path / "pod.sqlite"
    spool = tmp_path / "spool"
    duckdb_path = tmp_path / "esdc.duckdb"

    transport = _pod_transport(monkeypatch)
    baseline = _prepare(esdc, spool, sqlite_path, (2024, 2025))
    apply_prepared_pod_fetch(baseline, sqlite_path=sqlite_path, duckdb_path=duckdb_path)

    # A one-year selection carries the prior monitoring fact reported outside it.
    transport.resource_rows = []
    prepared = _prepare(esdc, spool, sqlite_path, (2024,))
    apply_prepared_pod_fetch(
        prepared,
        sqlite_path=sqlite_path,
        duckdb_path=duckdb_path,
    )

    connection = sqlite3.connect(sqlite_path)
    try:
        monitoring_years = sorted(
            str(row[0][:4]) for row in connection.execute("SELECT reported_date FROM pod_monitoring")
        )
        forecast_years = [
            int(row[0]) for row in connection.execute("SELECT year_frcast FROM pod_forecast")
        ]
    finally:
        connection.close()
    # The selected year is rewritten and the unselected fact is retained; the global
    # forecast is never filtered by the year selection.
    assert monitoring_years == ["2024", "2025"]
    assert forecast_years == [2027]


def test_stale_basis_is_rejected_after_a_concurrent_writer_advanced_the_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import esdc.esdc as esdc

    sqlite_path = tmp_path / "pod.sqlite"
    spool = tmp_path / "spool"
    duckdb_path = tmp_path / "esdc.duckdb"

    _pod_transport(monkeypatch)
    prepared = _prepare(esdc, spool, sqlite_path, None)
    assert prepared.pod_batch is not None
    # Another writer initializes and advances the target after preparation.
    assert _advance_target(sqlite_path) == 1

    with pytest.raises(FetchFailure) as stale:
        apply_prepared_pod_fetch(prepared, sqlite_path=sqlite_path, duckdb_path=duckdb_path)
    assert "prepare again" in str(stale.value)
    # Nothing was written or changed by the rejected attempt.
    connection = sqlite3.connect(sqlite_path)
    try:
        generation = int(
            connection.execute(
                "SELECT source_generation FROM pod_registry_state WHERE singleton_id=1"
            ).fetchone()[0]
        )
        assert generation == 1
        assert connection.execute("SELECT COUNT(*) FROM _pod_identity_ledger").fetchone()[0] == 0
    finally:
        connection.close()

    # The same rule protects an existing-basis preparation that another writer moved.
    refreshed = _prepare(esdc, spool, sqlite_path, None)
    assert refreshed.pod_batch is not None
    assert _advance_target(sqlite_path) == 2
    with pytest.raises(FetchFailure):
        apply_prepared_pod_fetch(
            refreshed,
            sqlite_path=sqlite_path,
            duckdb_path=duckdb_path,
        )
    connection = sqlite3.connect(sqlite_path)
    try:
        assert (
            int(
                connection.execute(
                    "SELECT source_generation FROM pod_registry_state WHERE singleton_id=1"
                ).fetchone()[0]
            )
            == 2
        )
    finally:
        connection.close()


def test_same_file_and_wrong_target_are_rejected_before_any_writer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import esdc.esdc as esdc

    sqlite_path = tmp_path / "pod.sqlite"
    prepared = _pod_transport(monkeypatch) and _prepare(
        esdc, tmp_path / "spool", sqlite_path, None
    )
    assert prepared.pod_batch is not None

    with pytest.raises(FetchFailure) as same_file:
        apply_prepared_pod_fetch(
            prepared,
            sqlite_path=sqlite_path,
            duckdb_path=sqlite_path,
        )
    assert "same file" in str(same_file.value)
    with pytest.raises(FetchFailure) as wrong_path:
        apply_prepared_pod_fetch(
            prepared,
            sqlite_path=tmp_path / "other.sqlite",
            duckdb_path=tmp_path / "esdc.duckdb",
        )
    assert "target basis" in str(wrong_path.value)
    # No writer, no mkdir: neither target exists.
    assert not sqlite_path.exists()
    assert not (tmp_path / "other.sqlite").exists()


def test_unsupported_wal_runtime_rejects_before_creating_a_fresh_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import esdc.esdc as esdc

    prepared = _pod_transport(monkeypatch) and _prepare(
        esdc, tmp_path / "spool", tmp_path / "pod.sqlite", None
    )
    assert prepared.pod_batch is not None
    monkeypatch.setattr(sqlite3, "sqlite_version_info", (3, 50, 4))
    target = tmp_path / "nested" / "pod.sqlite"

    with pytest.raises(FetchFailure) as unsupported:
        apply_prepared_pod_fetch(
            prepared,
            sqlite_path=target,
            duckdb_path=tmp_path / "esdc.duckdb",
        )
    assert "unsupported" in str(unsupported.value)
    assert not target.parent.exists()
    assert not target.exists()


def test_failure_after_the_final_refresh_rolls_back_every_source_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import esdc.esdc as esdc

    transport = _pod_transport(monkeypatch)
    transport.timeseries_rows = {2024: [{"report_year": 2024, "project_id": PROJECT_ID}]}
    sqlite_path = tmp_path / "pod.sqlite"
    duckdb_path = tmp_path / "esdc.duckdb"
    spool = tmp_path / "spool"
    checked = INPUT_TABLES + SNAPSHOT_ROWS + ("_pod_identity_ledger", "pod_read_model_snapshot")

    baseline = _prepare(esdc, spool, sqlite_path, None)
    apply_prepared_pod_fetch(baseline, sqlite_path=sqlite_path, duckdb_path=duckdb_path)
    before = _sqlite_counts(sqlite_path, checked)
    assert before["_pod_identity_ledger"] == 1

    real_refresh = fetch_publication.refresh_pod_after_source_change

    def refresh_then_fail(conn, **kwargs):
        real_refresh(conn, **kwargs)
        raise RuntimeError("injected failure after the final refresh")

    monkeypatch.setattr(fetch_publication, "refresh_pod_after_source_change", refresh_then_fail)

    with pytest.raises(FetchFailure) as failure:
        apply_prepared_pod_fetch(
            _prepare(esdc, spool, sqlite_path, None),
            sqlite_path=sqlite_path,
            duckdb_path=duckdb_path,
        )
    assert failure.value.sqlite_committed is False
    assert failure.value.recovery_command is None
    assert "Nothing was committed" in str(failure.value)
    # The failed apply changed nothing; every source write it staged is rolled back.
    assert _sqlite_counts(sqlite_path, checked) == before

    # On a fresh target the v2 schema itself is created inside the owned transaction,
    # so a rollback leaves no committed registry, inputs, snapshots, or ledger behind.
    fresh = tmp_path / "fresh.sqlite"
    with pytest.raises(FetchFailure):
        apply_prepared_pod_fetch(
            _prepare(esdc, spool, fresh, None),
            sqlite_path=fresh,
            duckdb_path=duckdb_path,
        )
    connection = sqlite3.connect(fresh)
    try:
        persisted = connection.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table'"
        ).fetchone()[0]
    finally:
        connection.close()
    assert persisted == 0


def test_exact_20_digit_numeric_from_the_prepared_batch_is_preserved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import esdc.esdc as esdc
    from tests.pod_registry.test_fetch_prepare_v2 import _payload, _record

    transport = _pod_transport(monkeypatch)
    exact = Decimal("12345678901234567.123")
    transport.payloads[(PodApiTable.POD_FORECAST.value, None)] = _payload(
        [_record("pod_forecast", usf_oil_pod=exact)], "pod_forecast"
    )
    sqlite_path = tmp_path / "pod.sqlite"

    prepared = _prepare(esdc, tmp_path / "spool", sqlite_path, None)
    outcome = apply_prepared_pod_fetch(
        prepared,
        sqlite_path=sqlite_path,
        duckdb_path=tmp_path / "esdc.duckdb",
    )

    connection = sqlite3.connect(sqlite_path)
    try:
        stored = connection.execute("SELECT usf_oil_pod FROM pod_forecast").fetchone()[0]
    finally:
        connection.close()
    assert stored == "12345678901234567.123"
    assert Decimal(str(stored)) == exact
    assert outcome.pod_source_generation == 1


def test_explicit_non_pod_argument_is_rejected_before_any_writer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dataclasses import replace

    import esdc.esdc as esdc

    prepared = _pod_transport(monkeypatch) and _prepare(
        esdc, tmp_path / "spool", tmp_path / "pod.sqlite", None
    )
    without_batch = replace(prepared, pod_batch=None)

    with pytest.raises(FetchFailure) as failure:
        apply_prepared_pod_fetch(
            without_batch,
            sqlite_path=tmp_path / "pod.sqlite",
            duckdb_path=tmp_path / "esdc.duckdb",
        )
    assert "prepared POD batch" in str(failure.value)
    assert not (tmp_path / "pod.sqlite").exists()


def test_actually_empty_sqlite_target_is_treated_as_fresh(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import esdc.esdc as esdc

    transport = _pod_transport(monkeypatch)
    transport.timeseries_rows = {2024: [{"report_year": 2024, "project_id": PROJECT_ID}]}
    sqlite_path = tmp_path / "pod.sqlite"
    # An empty regular SQLite file is a fresh initialization target, not legacy.
    sqlite_path.touch()
    prepared = _prepare(esdc, tmp_path / "spool", sqlite_path, None)
    assert prepared.pod_batch is not None
    basis = prepared.pod_batch.target_basis
    assert basis is not None and basis.existed is True
    assert basis.source_generation is None

    outcome = apply_prepared_pod_fetch(
        prepared,
        sqlite_path=sqlite_path,
        duckdb_path=tmp_path / "esdc.duckdb",
    )

    assert outcome.pod_source_generation == 1
    assert _sqlite_counts(sqlite_path, INPUT_TABLES) == dict.fromkeys(INPUT_TABLES, 1)
    assert _sqlite_counts(sqlite_path, SNAPSHOT_ROWS) == {
        "api_project_resources": 2,
        "api_project_timeseries": 1,
    }


def test_nonempty_legacy_target_is_refused_before_any_writer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import esdc.esdc as esdc

    _pod_transport(monkeypatch)
    target = tmp_path / "pod.sqlite"
    prepared = _prepare(esdc, tmp_path / "spool", target, None)
    assert prepared.pod_batch is not None
    # Another writer creates a nonempty legacy application database at the target.
    connection = sqlite3.connect(target)
    connection.row_factory = sqlite3.Row
    connection.executescript(_OLD_SHAPED_FIXTURE_SQL)
    connection.execute("INSERT INTO m_pod (id, pod_id, approval_seq) VALUES (1, 'old', 1)")
    connection.commit()
    before = {
        str(row[0]) for row in connection.execute("SELECT name FROM sqlite_master")
    }
    connection.close()

    with pytest.raises(FetchFailure) as refusal:
        apply_prepared_pod_fetch(
            prepared,
            sqlite_path=target,
            duckdb_path=tmp_path / "esdc.duckdb",
        )
    assert "inspect_migration" in str(refusal.value)
    assert "crosswalk" in str(refusal.value)

    connection = sqlite3.connect(target)
    try:
        after = {str(row[0]) for row in connection.execute("SELECT name FROM sqlite_master")}
    finally:
        connection.close()
    # The refusal is actionable only: a legacy target is never migrated or given v2 schema.
    assert "m_pod" in after
    assert "pod_registry_state" not in after
    assert after == before


def test_directory_target_is_not_a_fresh_source_writer_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import esdc.esdc as esdc

    _pod_transport(monkeypatch)
    target = tmp_path / "pod.sqlite"
    prepared = _prepare(esdc, tmp_path / "spool", target, None)
    assert prepared.pod_batch is not None
    target.mkdir()

    with pytest.raises(FetchFailure) as failure:
        apply_prepared_pod_fetch(
            prepared,
            sqlite_path=target,
            duckdb_path=tmp_path / "esdc.duckdb",
        )
    assert "regular SQLite file" in str(failure.value)
    assert not any(target.iterdir())


def test_missing_spool_slice_is_rejected_before_any_writer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import esdc.esdc as esdc

    transport = _pod_transport(monkeypatch)
    transport.timeseries_rows = {2024: [{"report_year": 2024, "project_id": PROJECT_ID}]}
    sqlite_path = tmp_path / "pod.sqlite"
    prepared = _prepare(esdc, tmp_path / "spool", sqlite_path, None)
    assert prepared.timeseries

    spool_file = prepared.timeseries[0].rows_path
    spool_file.unlink()

    with pytest.raises(FetchFailure) as failure:
        apply_prepared_pod_fetch(
            prepared,
            sqlite_path=sqlite_path,
            duckdb_path=tmp_path / "esdc.duckdb",
        )
    assert failure.value.phase == "validation"
    assert failure.value.table == "project_timeseries"
    # The spool is validated before the writer, so the target was never created.
    assert not sqlite_path.exists()


def test_source_helper_never_opens_duckdb_or_publishes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import esdc.api_snapshot as api_snapshot
    import esdc.esdc as esdc
    import esdc.pod_registry.publication as publication

    transport = _pod_transport(monkeypatch)
    transport.timeseries_rows = {2024: [{"report_year": 2024, "project_id": PROJECT_ID}]}
    sqlite_path = tmp_path / "pod.sqlite"
    duckdb_path = tmp_path / "esdc.duckdb"
    prepared = _prepare(esdc, tmp_path / "spool", sqlite_path, None)

    calls: list[str] = []

    def spy(name: str):
        def fail(*args, **kwargs):
            calls.append(name)
            raise AssertionError(f"{name} must not run during the source-only apply")

        return fail

    monkeypatch.setattr(fetch_publication, "duckdb", type("spy", (), {"connect": spy("duckdb.connect")}))
    monkeypatch.setattr(fetch_publication, "get_duckdb_connection", spy("get_duckdb_connection"))
    monkeypatch.setattr(api_snapshot, "mark_project_resources_published", spy("mark_published"))
    monkeypatch.setattr(publication, "publish_v2_registry", spy("publish_v2_registry"))

    apply_prepared_pod_fetch(
        prepared,
        sqlite_path=sqlite_path,
        duckdb_path=duckdb_path,
    )

    assert calls == []
    assert not duckdb_path.exists()
    assert _snapshot_states(sqlite_path) == {
        "project_resources": "committed",
        "project_timeseries": "committed",
    }


def test_retained_legacy_numeric_types_survive_a_selected_api_update(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import esdc.esdc as esdc

    sqlite_path = tmp_path / "pod.sqlite"
    spool = tmp_path / "spool"
    duckdb_path = tmp_path / "esdc.duckdb"

    transport = _pod_transport(monkeypatch)
    transport.timeseries_rows = {2024: [{"report_year": 2024, "project_id": PROJECT_ID}]}
    baseline = _prepare(esdc, spool, sqlite_path, None)
    apply_prepared_pod_fetch(baseline, sqlite_path=sqlite_path, duckdb_path=duckdb_path)

    # A legitimate bootstrap baseline records the exact value with its numeric type map.
    source_types = _seed_legacy_timeseries_baseline(sqlite_path, (2024,))

    transport.resource_rows = [{"project_id": PROJECT_ID, "report_year": 2025}]
    transport.timeseries_rows = {
        2025: [{"report_year": 2025, "project_id": PROJECT_ID}]
    }
    prepared = _prepare(esdc, spool, sqlite_path, (2025,))
    outcome = apply_prepared_pod_fetch(
        prepared, sqlite_path=sqlite_path, duckdb_path=duckdb_path
    )

    # The retained legacy row outside the observed scope keeps its exact precision.
    assert _stored_timeseries_row(sqlite_path, 2024)["cprd_grs_oil"] == LEGACY_EXACT_VALUE
    connection = sqlite3.connect(sqlite_path)
    try:
        assert _legacy_type_overrides(connection) == _numeric_type_map(source_types)
    finally:
        connection.close()

    origin, details = _provenance_row(sqlite_path, outcome.timeseries_revision)
    assert origin == "api_json"
    assert details["legacy_type_overrides"] == _numeric_type_map(source_types)
    assert details["observed_years"] == [2025]
    assert details["input_origins"] == ["api_json"]


def test_selected_empty_clear_carries_retained_legacy_numeric_types(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import esdc.esdc as esdc

    sqlite_path = tmp_path / "pod.sqlite"
    spool = tmp_path / "spool"
    duckdb_path = tmp_path / "esdc.duckdb"

    transport = _pod_transport(monkeypatch)
    transport.timeseries_rows = {2024: [{"report_year": 2024, "project_id": PROJECT_ID}]}
    baseline = _prepare(esdc, spool, sqlite_path, None)
    apply_prepared_pod_fetch(baseline, sqlite_path=sqlite_path, duckdb_path=duckdb_path)

    source_types = _seed_legacy_timeseries_baseline(sqlite_path, (2024,))

    # The selected year answers valid-empty; the legacy row outside it is retained.
    transport.resource_rows = [{"project_id": PROJECT_ID, "report_year": 2025}]
    transport.timeseries_rows = {}
    prepared = _prepare(esdc, spool, sqlite_path, (2025,))
    outcome = apply_prepared_pod_fetch(
        prepared, sqlite_path=sqlite_path, duckdb_path=duckdb_path
    )

    assert _stored_timeseries_row(sqlite_path, 2024)["cprd_grs_oil"] == LEGACY_EXACT_VALUE
    connection = sqlite3.connect(sqlite_path)
    try:
        assert _legacy_type_overrides(connection) == _numeric_type_map(source_types)
    finally:
        connection.close()

    origin, details = _provenance_row(sqlite_path, outcome.timeseries_revision)
    assert origin == "api_json"
    assert details["observed_empty"] is True
    assert details["observed_years"] == [2025]
    assert details["input_origins"] == ["api_json"]
    assert details["legacy_type_overrides"] == _numeric_type_map(source_types)


def test_full_replacement_without_retained_legacy_rows_records_no_overrides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import esdc.esdc as esdc

    sqlite_path = tmp_path / "pod.sqlite"
    spool = tmp_path / "spool"
    duckdb_path = tmp_path / "esdc.duckdb"

    transport = _pod_transport(monkeypatch)
    transport.timeseries_rows = {2024: [{"report_year": 2024, "project_id": PROJECT_ID}]}
    baseline = _prepare(esdc, spool, sqlite_path, None)
    apply_prepared_pod_fetch(baseline, sqlite_path=sqlite_path, duckdb_path=duckdb_path)

    _seed_legacy_timeseries_baseline(sqlite_path, (2024,))

    # A complete observation removes the last legacy row, so no map is required.
    prepared = _prepare(esdc, spool, sqlite_path, None)
    outcome = apply_prepared_pod_fetch(
        prepared, sqlite_path=sqlite_path, duckdb_path=duckdb_path
    )

    connection = sqlite3.connect(sqlite_path)
    try:
        assert _legacy_type_overrides(connection) == {}
    finally:
        connection.close()

    origin, details = _provenance_row(sqlite_path, outcome.timeseries_revision)
    assert origin == "api_json"
    assert "legacy_type_overrides" not in details
    assert details["observed_years"] == [2024, 2025]
