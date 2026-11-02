"""Regression: SQLite import prepare -> load -> DuckDB publication current state."""

import sqlite3
from contextlib import closing
from pathlib import Path

import duckdb
import pytest

from esdc.analytics_rebuild import get_api_analytics_status
from esdc.fetching import FetchFailure
from esdc.pod_registry.contract import load_pod_contract
from esdc.pod_registry.publication import get_pod_registry_status
from esdc.pod_registry.sqlite_import import load_prepared_pod_batch, prepare_pod_import
from esdc.source_publication import publish_committed_pod_sources
from tests.pod_registry.test_import_v2 import _create_sqlite_source
from tests.pod_registry.test_importer import _row, _write_workbook

INPUT_TABLES = ("m_project", "m_pod", "pod_project", "pod_monitoring", "pod_forecast")
ACTOR = "FA:test-import"


def _open_duck(path: Path) -> duckdb.DuckDBPyConnection:
    return duckdb.connect(
        str(path), read_only=True, config={"autoinstall_known_extensions": False}
    )


def _counts(path: Path) -> dict[str, int]:
    with closing(sqlite3.connect(path)) as conn:
        return {
            t: conn.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
            for t in INPUT_TABLES
        }


@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize("publish_initial", [False, True])
def test_sqlite_import_publication_current(
    tmp_path: Path, empty: bool, publish_initial: bool
) -> None:
    contract = load_pod_contract()
    source = tmp_path / "source.sqlite"
    target = tmp_path / "target.sqlite"
    duck_path = tmp_path / "projection.duckdb"

    _create_sqlite_source(source, contract)

    if not empty:
        with closing(sqlite3.connect(source)) as conn:
            conn.execute(
                "INSERT INTO project_resources (project_id, project_id_canonical, report_year, project_name) "
                "VALUES (?,?,?,?)",
                ("P001459F01", "P-001459F-01", 2024, "Imported"),
            )
            conn.execute(
                "INSERT INTO project_timeseries (project_id, project_id_canonical, report_year) VALUES (?,?,?)",
                ("P001459F01", "P-001459F-01", 2024),
            )
            conn.commit()

    batch = prepare_pod_import(source, "sqlite", contract, target_path=target)

    result, publication = load_prepared_pod_batch(
        batch,
        sqlite_path=target,
        duck_path=duck_path,
        actor=ACTOR,
        publish=publish_initial,
    )

    assert result.source_committed
    assert result.source_generation == 1
    assert getattr(publication, "projection_error", None) is None
    if not publish_initial:
        publication = publish_committed_pod_sources(
            target, duck_path, expected_generation=1, reindex=False
        )
        assert publication.fresh

    expected = 0 if empty else 1
    counts = _counts(target)
    assert counts == dict.fromkeys(INPUT_TABLES, 0)
    assert result.table_counts == dict.fromkeys(INPUT_TABLES, 0)

    with _open_duck(duck_path) as con:
        resource_count = con.execute(
            "SELECT COUNT(*) FROM project_resources"
        ).fetchone()
        timeseries_count = con.execute(
            "SELECT COUNT(*) FROM project_timeseries"
        ).fetchone()
        assert resource_count == (expected,)
        assert timeseries_count == (expected,)
        if not empty:
            name = con.execute(
                "SELECT project_name FROM project_resources",
            ).fetchone()
            assert name is not None and name[0] == "Imported"

    status = get_api_analytics_status(duckdb_path=duck_path, sqlite_path=target)
    assert status["project_resources"]["status"] == "published"
    assert status["project_timeseries"]["status"] == "published"

    registry = get_pod_registry_status(target, duck_path)
    assert registry.state == "fresh"

    generation = 1
    if not empty:
        with closing(sqlite3.connect(source)) as conn:
            conn.execute("UPDATE project_resources SET project_name='Replacement'")
            conn.commit()
        batch = prepare_pod_import(source, "sqlite", contract, target_path=target)
        replaced, _ = load_prepared_pod_batch(
            batch, sqlite_path=target, duck_path=duck_path, actor=ACTOR
        )
        generation = 2
        assert replaced.source_generation == generation
        with _open_duck(duck_path) as con:
            assert con.execute(
                "SELECT project_name FROM project_resources"
            ).fetchall() == [("Replacement",)]

    retry = publish_committed_pod_sources(
        target, duck_path, expected_generation=generation, reindex=False
    )
    assert retry.fresh
    with closing(sqlite3.connect(target)) as conn:
        assert (
            conn.execute("SELECT source_generation FROM pod_registry_state").fetchone()[
                0
            ]
            == generation
        )


def test_workbook_recovery_status_failure_keeps_committed_mirror_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = load_pod_contract()
    target = tmp_path / "target.sqlite"
    cache = tmp_path / "cache.duckdb"
    workbook = _write_workbook(
        tmp_path / "pod.xlsx",
        contract,
        {table: [_row(contract, table)] for table in INPUT_TABLES},
    )
    batch = prepare_pod_import(workbook, "xlsx", target_path=target)
    load_prepared_pod_batch(batch, sqlite_path=target, duck_path=cache, publish=False)

    def unavailable_status(*args, **kwargs):
        raise RuntimeError("status read failed after mirror commit")

    monkeypatch.setattr(
        "esdc.pod_registry.publication.get_pod_registry_status", unavailable_status
    )
    with pytest.raises(FetchFailure, match="status read failed") as caught:
        publish_committed_pod_sources(target, cache, expected_generation=1)
    assert caught.value.sqlite_committed is True
    assert caught.value.duckdb_committed is True
    assert caught.value.phase == "post_commit"
    with _open_duck(cache) as con:
        assert con.execute("SELECT COUNT(*) FROM m_pod").fetchone() == (1,)
