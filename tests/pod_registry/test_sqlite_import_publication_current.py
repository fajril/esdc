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
from esdc.pod_registry.sqlite_import import (
    PodImportError,
    load_prepared_pod_batch,
    prepare_pod_import,
)
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
                "INSERT INTO project_resources (project_id, project_id_canonical, report_year, project_name, pd) "
                "VALUES (?,?,?,?,?)",
                ("P001459F01", "P-001459F-01", 2024, "Imported", 1.0),
            )
            conn.execute(
                "INSERT INTO project_timeseries (project_id, project_id_canonical, report_year) VALUES (?,?,?)",
                ("P001459F01", "P-001459F-01", 2024),
            )
            conn.commit()
            assert conn.execute(
                "SELECT pd, typeof(pd) FROM project_resources"
            ).fetchone() == (1.0, "real")

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
            assert con.execute("SELECT pd FROM project_resources").fetchone() == (1,)

    if not empty:
        with closing(sqlite3.connect(target)) as conn:
            assert conn.execute(
                "SELECT pd, typeof(pd) FROM api_project_resources"
            ).fetchone() == (1, "integer")

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


def test_sqlite_import_rejects_changed_resource_stage_before_commit(
    tmp_path: Path,
) -> None:
    contract = load_pod_contract()
    source = tmp_path / "source.sqlite"
    target = tmp_path / "target.sqlite"
    cache = tmp_path / "cache.duckdb"
    _create_sqlite_source(source, contract)
    with closing(sqlite3.connect(source)) as conn:
        conn.execute(
            "INSERT INTO project_resources (project_id, report_year, project_name, pd) "
            "VALUES (?, ?, ?, ?)",
            ("P-001459F-01", 2024, "Imported", 1.0),
        )
        conn.commit()
    batch = prepare_pod_import(source, "sqlite", contract, target_path=target)
    stage = batch.staged_snapshots
    assert stage is not None and stage.resources_path is not None
    staged = stage.resources_path.read_text()
    assert '"Imported"' in staged
    stage.resources_path.write_text(
        staged.replace('"Imported"', '"Changed after validation"')
    )
    with pytest.raises(PodImportError, match="stage.*changed"):
        load_prepared_pod_batch(
            batch, sqlite_path=target, duck_path=cache, publish=False
        )
    with closing(sqlite3.connect(target)) as conn:
        assert conn.execute(
            "SELECT name FROM sqlite_master WHERE name IN ('m_pod', 'api_snapshot')"
        ).fetchall() == []
    assert not cache.exists()


def test_sqlite_import_rejects_stage_changed_between_hash_and_consumption(
    published_status_pair, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from esdc.pod_registry import sqlite_import

    _, target, cache = published_status_pair
    source = tmp_path / "changed-source.sqlite"
    _create_sqlite_source(source, load_pod_contract())
    with closing(sqlite3.connect(source)) as conn:
        conn.execute(
            "INSERT INTO project_resources (project_id, report_year, project_name) "
            "VALUES ('P-001459F-01', 2024, 'Validated')"
        )
        conn.commit()
    batch = prepare_pod_import(source, "sqlite", target_path=target)
    stage = batch.staged_snapshots
    assert stage is not None and stage.resources_path is not None
    original_hash = sqlite_import._sorted_resource_hash

    def mutate_after_hash(path: Path, directory: Path) -> tuple[int, str]:
        result = original_hash(path, directory)
        if path == stage.resources_path:
            path.write_text(path.read_text().replace('"Validated"', '"Unvalidated"'))
        return result

    monkeypatch.setattr(sqlite_import, "_sorted_resource_hash", mutate_after_hash)
    cache_before = cache.read_bytes()
    with closing(sqlite3.connect(target)) as conn:
        generation = conn.execute("SELECT source_generation FROM pod_registry_state").fetchone()
    with pytest.raises(PodImportError, match="stage.*changed"):
        load_prepared_pod_batch(batch, sqlite_path=target, duck_path=cache, publish=False)
    with closing(sqlite3.connect(target)) as conn:
        assert conn.execute("SELECT project_name FROM api_project_resources").fetchone() == ("Canonical",)
        assert conn.execute("SELECT source_generation FROM pod_registry_state").fetchone() == generation
    assert cache.read_bytes() == cache_before


@pytest.fixture()
def published_status_pair(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from fastapi.testclient import TestClient

    from esdc.configs import Config
    from esdc.portal.app import create_portal_app

    source, target, cache = (
        tmp_path / "source.sqlite", tmp_path / "esdc.sqlite", tmp_path / "cache.duckdb"
    )
    monkeypatch.setattr(Config, "get_db_dir", classmethod(lambda cls: tmp_path))
    monkeypatch.setattr(Config, "get_db_file", classmethod(lambda cls: cache))
    _create_sqlite_source(source, load_pod_contract())
    with closing(sqlite3.connect(source)) as conn:
        conn.execute(
            "INSERT INTO project_resources (project_id, report_year, project_name) "
            "VALUES ('P-001459F-01', 2024, 'Canonical')"
        )
        conn.execute(
            "INSERT INTO project_timeseries (project_id, report_year) "
            "VALUES ('P-001459F-01', 2024)"
        )
        conn.commit()
    batch = prepare_pod_import(source, "sqlite", target_path=target)
    load_prepared_pod_batch(batch, sqlite_path=target, duck_path=cache)
    return TestClient(create_portal_app()), target, cache


def test_status_uses_publication_markers_without_scanning_data(
    published_status_pair, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, target, _ = published_status_pair

    def forbidden(*args, **kwargs):
        raise AssertionError("portal status must not inspect business rows")

    for name in (
        "esdc.analytics_rebuild._persisted_hash",
        "esdc.analytics_rebuild.timeseries_fingerprint",
        "esdc.analytics_rebuild._resource_projection_matches",
        "esdc.fetch_publication.ensure_selective_resource_baseline",
        "esdc.fetch_publication.ensure_timeseries_baseline",
        "esdc.source_publication._projection_counts",
    ):
        monkeypatch.setattr(name, forbidden)
    before = target.read_bytes()
    response = client.get("/api/status")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["verification_level"] == "metadata"
    assert body["fetch"]["fresh"] is True
    assert body["pod"]["state"] == "fresh"
    assert {p["name"]: p["rows_written"] for p in body["fetch"]["components"]} == {
        "project_resources": 1, "project_timeseries": 1, "pod_projection": 0
    }
    assert target.read_bytes() == before


@pytest.mark.parametrize("change", ["revision", "pending", "missing_lineage", "corrupt"])
def test_fast_status_never_claims_unpublished_or_unreadable_pair_is_fresh(
    published_status_pair, change: str
) -> None:
    client, target, cache = published_status_pair
    if change in {"revision", "pending"}:
        with closing(sqlite3.connect(target)) as conn:
            if change == "revision":
                conn.execute("UPDATE api_snapshot SET revision='new-revision' WHERE source_table='project_resources'")
            else:
                conn.execute("UPDATE api_snapshot SET publication_state='committed' WHERE source_table='project_resources'")
            conn.commit()
    elif change == "missing_lineage":
        with closing(duckdb.connect(str(cache))) as conn:
            conn.execute("DELETE FROM api_publication WHERE source_table='project_resources'")
    else:
        cache.write_bytes(b"not a DuckDB database")
    reply = client.get("/api/status")
    assert reply.status_code == 200, reply.text
    assert reply.json()["verification_level"] == "metadata"
    assert reply.json()["fetch"]["fresh"] is False
    parts = {p["name"]: p for p in reply.json()["fetch"]["components"]}
    assert parts["project_resources"]["status"] != "published"
    if change == "pending":
        assert parts["project_resources"]["status"] == "pending"
    if change == "revision":
        assert parts["project_resources"]["status"] == "stale"
    if change == "corrupt":
        assert parts["project_resources"]["status"] == "error"


@pytest.mark.parametrize("change", ["revision", "generation"])
def test_fast_status_detects_source_change_during_marker_read(
    published_status_pair, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    from esdc import analytics_rebuild

    client, target, _ = published_status_pair
    real = analytics_rebuild.get_api_analytics_status

    def change_after_read(**kwargs):
        result = real(**kwargs)
        with closing(sqlite3.connect(target)) as conn:
            if change == "revision":
                conn.execute("UPDATE api_snapshot SET revision='concurrent' WHERE source_table='project_timeseries'")
            else:
                conn.execute("UPDATE pod_registry_state SET source_generation=source_generation+1")
            conn.commit()
        return result

    monkeypatch.setattr(analytics_rebuild, "get_api_analytics_status", change_after_read)
    reply = client.get("/api/status")
    assert reply.status_code == 200, reply.text
    assert reply.json()["fetch"]["fresh"] is False
    assert {p["status"] for p in reply.json()["fetch"]["components"]} == {"stale"}


def test_default_audit_still_detects_content_changes_without_new_markers(
    published_status_pair,
) -> None:
    from esdc.analytics_rebuild import get_api_analytics_status

    client, target, cache = published_status_pair
    with closing(sqlite3.connect(target)) as conn:
        conn.execute("UPDATE api_project_resources SET project_name='Manual edit'")
        conn.commit()
    reply = client.get("/api/status")
    assert reply.status_code == 200, reply.text
    assert reply.json()["verification_level"] == "metadata"
    assert reply.json()["fetch"]["fresh"] is True
    audit = get_api_analytics_status(sqlite_path=target, duckdb_path=cache)
    assert audit["project_resources"]["status"] == "error"
    assert "count/hash" in audit["project_resources"]["detail"]


def test_status_refuses_generation_change_between_combined_and_pod_reads(
    published_status_pair, monkeypatch: pytest.MonkeyPatch
) -> None:
    from esdc.portal import app as portal

    client, target, _ = published_status_pair
    real = portal.get_pod_registry_status

    def change_before_pod(*args, **kwargs):
        with closing(sqlite3.connect(target)) as conn:
            conn.execute("UPDATE pod_registry_state SET source_generation=source_generation+1")
            conn.commit()
        return real(*args, **kwargs)

    monkeypatch.setattr(portal, "get_pod_registry_status", change_before_pod)
    reply = client.get("/api/status")
    assert reply.status_code == 503
    assert "changed" in reply.json()["detail"].lower()


def test_fast_status_does_not_initialize_missing_databases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fastapi.testclient import TestClient

    from esdc.configs import Config
    from esdc.portal.app import create_portal_app

    directory = tmp_path / "missing"
    cache = tmp_path / "absent.duckdb"
    monkeypatch.setattr(Config, "get_db_dir", classmethod(lambda cls: directory))
    monkeypatch.setattr(Config, "get_db_file", classmethod(lambda cls: cache))
    reply = TestClient(create_portal_app()).get("/api/status")
    assert reply.status_code == 200, reply.text
    assert reply.json()["verification_level"] == "metadata"
    assert reply.json()["fetch"]["fresh"] is False
    assert not directory.exists() and not cache.exists()


def test_fast_status_reports_locked_duckdb_without_claiming_freshness(
    published_status_pair, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, _, cache = published_status_pair
    before = cache.read_bytes()
    real = duckdb.connect

    def locked(database, *args, **kwargs):
        if Path(database) == cache:
            raise duckdb.IOException("database file is locked")
        return real(database, *args, **kwargs)

    monkeypatch.setattr(duckdb, "connect", locked)
    reply = client.get("/api/status")
    assert reply.status_code == 200, reply.text
    assert reply.json()["fetch"]["fresh"] is False
    parts = {p["name"]: p for p in reply.json()["fetch"]["components"]}
    assert parts["project_resources"]["status"] == "error"
    assert "locked" in parts["project_resources"]["error"]
    assert cache.read_bytes() == before
