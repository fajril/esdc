"""Raw-data rehearsals retain an explicit current POD baseline snapshot."""

import json
from pathlib import Path

import pytest

from esdc import analytics_rehearsal
from esdc.configs import Config
from esdc.pod_registry.api_context import PodImportError
from esdc.pod_registry.store import get_sqlite_connection
from tests.knowledge import conftest as knowledge
from tests.pod_registry.test_migrations_v2 import _legacy_connection


@pytest.fixture(autouse=True)
def owned_config(isolated_config: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(Config, "get_db_dir", classmethod(lambda cls: isolated_config))
    monkeypatch.setattr(
        Config, "get_db_file", classmethod(lambda cls: isolated_config / "cache.duckdb")
    )
    monkeypatch.setattr(
        Config, "get_cache_dir", classmethod(lambda cls: isolated_config / "cache")
    )


def test_rehearsal_pod_payloads_preserve_current_master_uuids(tmp_path: Path):
    target = tmp_path / "source.sqlite"
    conn = get_sqlite_connection(target)
    try:
        knowledge._seed_knowledge_world(conn)
        expected = [
            tuple(row)
            for row in conn.execute(
                "SELECT pod_id_seq,pod_uuid FROM m_pod ORDER BY pod_id_seq"
            )
        ]
    finally:
        conn.close()
    before = target.read_bytes()
    assert hasattr(analytics_rehearsal, "current_pod_fixture_bodies")
    bodies = analytics_rehearsal.current_pod_fixture_bodies(target)
    records = json.loads(bodies["m_pod"])
    assert sorted((row["pod_id_seq"], row["pod_uuid"]) for row in records) == expected
    assert len(bodies) == 5
    assert target.read_bytes() == before


def test_rehearsal_without_pod_has_explicit_empty_contract_payloads(tmp_path: Path):
    target = tmp_path / "missing" / "source.sqlite"
    assert hasattr(analytics_rehearsal, "current_pod_fixture_bodies")
    bodies = analytics_rehearsal.current_pod_fixture_bodies(target)
    assert len(bodies) == 5
    assert all(json.loads(body) == [] for body in bodies.values())
    assert not target.parent.exists()


def test_rehearsal_does_not_convert_old_pod_baseline(tmp_path: Path):
    target = tmp_path / "old.sqlite"
    conn = _legacy_connection(target)
    conn.close()
    before = target.read_bytes()
    assert hasattr(analytics_rehearsal, "current_pod_fixture_bodies")
    with pytest.raises(PodImportError, match="full load|Replace it once"):
        analytics_rehearsal.current_pod_fixture_bodies(target)
    assert target.read_bytes() == before


def test_selected_rehearsal_compares_canonical_values_not_cache_identities(tmp_path):
    import duckdb

    path = tmp_path / "cache.duckdb"
    conn = duckdb.connect(str(path))
    try:
        conn.execute(
            "CREATE TABLE project_timeseries(id BIGINT,uuid VARCHAR,report_year INTEGER,project_id VARCHAR,amount DECIMAL(20,3),memo VARCHAR)"
        )
        conn.execute(
            "INSERT INTO project_timeseries VALUES (1,'old1',2023,'P-0000001-01',1.250,NULL),(2,'old2',NULL,'P-0000001-01',12345678901234567.123,''),(3,'old3',2024,'P-0000002-01',5.000,'selected')"
        )
    finally:
        conn.close()
    before = analytics_rehearsal._retained_timeseries_values(path, {2024})
    assert before["count"] == 2
    conn = duckdb.connect(str(path))
    try:
        conn.execute("UPDATE project_timeseries SET id=id+100,uuid=uuid||'-rebuilt'")
        conn.execute("UPDATE project_timeseries SET amount=99 WHERE report_year=2024")
    finally:
        conn.close()
    assert analytics_rehearsal._retained_timeseries_values(path, {2024}) == before
    conn = duckdb.connect(str(path))
    try:
        conn.execute("UPDATE project_timeseries SET memo='' WHERE report_year=2023")
    finally:
        conn.close()
    assert analytics_rehearsal._retained_timeseries_values(path, {2024}) != before
    conn = duckdb.connect(str(path))
    try:
        conn.execute("UPDATE project_timeseries SET memo=NULL WHERE report_year=2023")
        conn.execute(
            "INSERT INTO project_timeseries SELECT id+1000,uuid||'-duplicate',report_year,project_id,amount,memo FROM project_timeseries WHERE report_year=2023"
        )
    finally:
        conn.close()
    duplicate = analytics_rehearsal._retained_timeseries_values(path, {2024})
    assert duplicate["count"] == 3 and duplicate["sha256"] != before["sha256"]
