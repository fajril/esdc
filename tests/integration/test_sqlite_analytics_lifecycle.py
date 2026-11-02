"""Offline end-to-end tests for the bounded SQLite analytics rehearsal."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

import duckdb
import pytest
from typer.testing import CliRunner

from esdc.api_snapshot import (
    ApiProjectResourceRow,
    ApiSnapshot,
    replace_project_resources_snapshot,
)
from esdc.pod_registry.store import get_sqlite_connection


def _fixture_pair(root: Path) -> tuple[Path, Path]:
    from esdc.api_projection import publish_project_resources
    from esdc.api_snapshot import get_project_resources_snapshot
    from esdc.dbmanager import _execute_sql_script

    baseline = root / "baseline"
    fixtures = root / "fixtures"
    baseline.mkdir()
    fixtures.mkdir()
    sqlite_path = baseline / "esdc.sqlite"
    duckdb_path = baseline / "esdc.duckdb"
    source = get_sqlite_connection(sqlite_path)
    replace_project_resources_snapshot(
        source,
        [
            ApiProjectResourceRow(
                project_id="legacy-24", report_year=2024, project_name="Retained 24"
            ),
            ApiProjectResourceRow(
                project_id="legacy-19", report_year=2019, project_name="Retained 19"
            ),
        ],
        ApiSnapshot(
            "project_resources",
            "resources-1",
            2,
            "",
            "2026-10-24T00:00:00Z",
            "committed",
        ),
    )
    source.execute(
        "CREATE TABLE IF NOT EXISTS api_snapshot_provenance (source_table TEXT, revision TEXT, origin TEXT, details_json TEXT, PRIMARY KEY(source_table, revision))"
    )
    source.close()
    with duckdb.connect(str(duckdb_path)) as target:
        _execute_sql_script(target, "create_table_project_timeseries.sql")
        target.execute("ALTER TABLE project_timeseries ADD COLUMN uuid VARCHAR")
        target.execute(
            "ALTER TABLE project_timeseries ALTER COLUMN report_date TYPE DATE"
        )
        target.executemany(
            "INSERT INTO project_timeseries(report_date,report_year,project_id,project_name,cprd_grs_oil,uuid) VALUES (?,?,?,?,?,?)",
            [
                ("2024-01-02", 2024, "legacy-ts", "Retained 24", 1.25, "uuid-24"),
                ("2019-01-02", 2019, "legacy-old-ts", "Retained 19", 2.5, "uuid-19"),
                (
                    "2025-01-02",
                    2025,
                    "legacy-empty-ts",
                    "Retained empty slice",
                    3.5,
                    "uuid-25",
                ),
                (None, None, "legacy-null-ts", "Retained NULL", None, "uuid-null"),
            ],
        )
        target.execute("CREATE TABLE protected_sentinel(value VARCHAR)")
        target.execute("INSERT INTO protected_sentinel VALUES ('keep')")
    source = get_sqlite_connection(sqlite_path)
    snapshot = get_project_resources_snapshot(source)
    source.close()
    assert snapshot is not None
    publish_project_resources(duckdb_path, sqlite_path, snapshot.revision)
    legacy_source = sqlite3.connect(sqlite_path)
    try:
        with legacy_source:
            legacy_source.execute("DROP TABLE api_project_timeseries")
            legacy_source.execute(
                "DELETE FROM api_snapshot_provenance WHERE source_table='project_timeseries'"
            )
            legacy_source.execute(
                "DELETE FROM api_snapshot WHERE source_table='project_timeseries'"
            )
    finally:
        legacy_source.close()
    with duckdb.connect(str(duckdb_path)) as target:
        _execute_sql_script(target, "create_table_metadata.sql")
        target.execute("INSERT INTO _metadata VALUES ('operator_note', 'preserve')")

    payloads: dict[tuple[str, str, int | None], object] = {}
    for scenario in (
        "default_fetch",
        "selected_fetch",
        "before_c1_failure",
        "after_c1_failure",
    ):
        payloads[
            (
                scenario,
                "project_resources",
                None if scenario == "default_fetch" else 2024,
            )
        ] = [
            {
                "project_id": f"{scenario}-24",
                "report_year": "2024",
                "project_name": "Fixture 24",
            }
        ]
        if scenario != "default_fetch":
            payloads[(scenario, "project_resources", 2025)] = [
                {
                    "project_id": f"{scenario}-25",
                    "report_year": "2025",
                    "project_name": "Fixture 25",
                }
            ]
        payloads[(scenario, "project_timeseries", 2024)] = [
            {
                "report_year": "2024",
                "report_date": "2024-01-02",
                "project_id": f"{scenario}-ts24",
                "project_name": "Fixture TS",
                "rate_oil": 8.5,
            }
        ]
        if scenario != "default_fetch":
            payloads[(scenario, "project_timeseries", 2025)] = []
    responses = []
    for index, (key, value) in enumerate(payloads.items()):
        data = json.dumps(value, separators=(",", ":")).encode()
        filename = f"response-{index}.json"
        (fixtures / filename).write_bytes(data)
        scenario, table, year = key
        responses.append(
            {
                "scenario": scenario,
                "table": table,
                "requested_year": year,
                "payload_path": filename,
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    manifest = fixtures / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "version": 1,
                "filetype": "json",
                "selected_years": [2024, 2025],
                "responses": responses,
            }
        ),
        encoding="utf-8",
    )
    return baseline, manifest


def test_small_fixture_rehearsal_and_root_cli_failure(
    tmp_path: Path, monkeypatch
) -> None:
    from esdc.esdc import app

    baseline, manifest = _fixture_pair(tmp_path)
    output = tmp_path / "small-run"
    monkeypatch.setenv("ESDC_CONFIG_DIR", str(tmp_path / "cli-config"))
    monkeypatch.setenv("ESDC_DB_DIR", str(tmp_path / "cli-db"))
    monkeypatch.setenv("ESDC_DB_FILE", str(tmp_path / "cli-db" / "active.duckdb"))
    from esdc.configs import Config

    Config._config_cache = None
    runner = CliRunner()
    args = [
        "verify-analytics-migration",
        "--baseline-dir",
        str(baseline),
        "--fixture-manifest",
        str(manifest),
        "--output-dir",
        str(output),
    ]
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    report_path = Path(result.output.strip().splitlines()[-1])
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "PASS"
    assert {stage["name"] for stage in report["stages"]} >= {
        "bootstrap_dry_run",
        "bootstrap_apply",
        "default_fetch",
        "selected_fetch",
        "before_c1_failure",
        "after_c1_failure",
        "api_free_pair_recovery",
        "deliberate_value_corruption",
    }
    assert report["scale"]["classification"] == "small fixture lifecycle"
    assert report["telemetry"]["peak_rss_bytes"] > 0
    assert report["source_counts"] == {
        "project_resources": {"sqlite": 2, "duckdb": 2},
        "project_timeseries": {"sqlite": 0, "duckdb": 4},
    }
    stages = {stage["name"]: stage for stage in report["stages"]}
    for stage_name in ("default_fetch", "selected_fetch"):
        retained = stages[stage_name]["retained_values_after"]
        assert retained["count"] == 0
        assert retained["samples"] == []
    with duckdb.connect(
        str(Path(report["clone_paths"]["duckdb"])), read_only=True
    ) as conn:
        assert conn.execute("SELECT * FROM protected_sentinel").fetchall() == [
            ("keep",)
        ]
    assert all(
        _hash(path) == digest for path, digest in report["original_hashes"].items()
    )
    stage_rows = report["stages"]
    assert isinstance(stage_rows, list)
    recovery = next(
        stage
        for stage in stage_rows
        if isinstance(stage, dict) and stage.get("name") == "api_free_pair_recovery"
    )
    assert isinstance(recovery, dict)
    recovery_audit = recovery["timeseries_audit"]
    assert isinstance(recovery_audit, dict)
    assert recovery_audit["ok"] is True

    with duckdb.connect(
        str(Path(report["clone_paths"]["duckdb"])), read_only=True
    ) as conn:
        existing = conn.execute(
            "SELECT data_type, (SELECT report_date FROM project_timeseries "
            "WHERE project_id='after_c1_failure-ts24') "
            "FROM information_schema.columns "
            "WHERE table_name='project_timeseries' AND column_name='report_date'"
        ).fetchone()
        assert existing is not None
        existing_type, existing_date = existing
    assert existing_type == "DATE"
    assert existing_date.isoformat() == "2024-01-02"
    from esdc.analytics_rebuild import rebuild_api_analytics

    fresh_recovery = tmp_path / "date-recovery.duckdb"
    rebuild_api_analytics(
        duckdb_path=fresh_recovery,
        sqlite_path=Path(report["clone_paths"]["sqlite"]),
        reindex=False,
    )
    with duckdb.connect(str(fresh_recovery), read_only=True) as conn:
        report_date_type_row = conn.execute(
            "SELECT data_type FROM information_schema.columns "
            "WHERE table_name='project_timeseries' AND column_name='report_date'"
        ).fetchone()
        report_date_row = conn.execute(
            "SELECT report_date FROM project_timeseries WHERE project_id='after_c1_failure-ts24'"
        ).fetchone()
        assert report_date_type_row is not None and report_date_row is not None
        report_date_type = report_date_type_row[0]
        report_date = report_date_row[0]
    assert report_date_type == "VARCHAR"
    assert report_date == "2024-01-02"
    failed = runner.invoke(app, args)
    assert failed.exit_code != 0
    assert "output-dir must not already exist" in failed.output

    from esdc import analytics_rehearsal

    @contextmanager
    def fail_transport(*args, **kwargs):
        raise RuntimeError("fixture injected stage failure")
        yield []

    monkeypatch.setattr(analytics_rehearsal, "_fixture_transport", fail_transport)
    failed_output = tmp_path / "failed-run"
    failed_result = runner.invoke(
        app,
        [
            "verify-analytics-migration",
            "--baseline-dir",
            str(baseline),
            "--fixture-manifest",
            str(manifest),
            "--output-dir",
            str(failed_output),
        ],
    )
    assert failed_result.exit_code != 0
    assert str(failed_output / "report.json") in failed_result.output
    failed_report = json.loads(
        (failed_output / "report.json").read_text(encoding="utf-8")
    )
    assert failed_report["status"] == "FAIL"
    assert failed_report["failed_stage"] == "default_fetch"
    assert failed_report["failed_command"] == "load_esdc_data(filetype=json)"
    assert failed_report["failed_exit"] == 1
    assert failed_report["failed_assertion"]
    assert all(
        _hash(path) == digest
        for path, digest in failed_report["original_hashes"].items()
    )


def test_rehearsal_scale_classification_requires_initial_and_final_volume() -> None:
    from esdc.analytics_rehearsal import _scale_classification

    million = 1_000_000
    small = _scale_classification(initial_legacy_rows=4, final_canonical_rows=6)
    assert small == {
        "classification": "small fixture lifecycle",
        "initial_legacy_timeseries_rows": 4,
        "final_canonical_timeseries_rows": 6,
        "million_row_lifecycle": "UNVERIFIED",
    }

    volume = _scale_classification(
        initial_legacy_rows=million, final_canonical_rows=million + 2
    )
    assert volume == {
        "classification": "synthetic million-row lifecycle",
        "initial_legacy_timeseries_rows": million,
        "final_canonical_timeseries_rows": million + 2,
        "million_row_lifecycle": "PASS",
    }

    incomplete = _scale_classification(
        initial_legacy_rows=million, final_canonical_rows=million - 1
    )
    assert incomplete["classification"] == "incomplete large lifecycle"
    assert incomplete["million_row_lifecycle"] == "UNVERIFIED"


@pytest.mark.parametrize("corruption", ["missing", "duplicate", "hash"])
def test_fixture_manifest_coverage_and_hash_failures(
    tmp_path: Path, corruption: str
) -> None:
    from esdc.analytics_rehearsal import _read_manifest

    _, manifest_path = _fixture_pair(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if corruption == "missing":
        manifest["responses"].pop()
    elif corruption == "duplicate":
        manifest["responses"].append(dict(manifest["responses"][0]))
    else:
        manifest["responses"][0]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError):
        _read_manifest(manifest_path)


def test_fixture_transport_rejects_unspecified_request_and_restores_downloader(
    tmp_path: Path,
) -> None:
    import esdc.esdc as app
    from esdc.analytics_rehearsal import _fixture_transport, _read_manifest
    from esdc.selection import FileType, TableName

    _, manifest_path = _fixture_pair(tmp_path)
    entries, _ = _read_manifest(manifest_path)
    original = app.esdc_downloader
    with (
        pytest.raises(RuntimeError, match="unspecified API request"),
        _fixture_transport("default_fetch", entries, set()),
    ):
        app.esdc_downloader(
            app.esdc_url_builder(
                TableName.PROJECT_RESOURCES,
                file_type=FileType.JSON,
            )
        )
    assert app.esdc_downloader is original


def test_interruption_retains_report_and_restores_process_state(
    tmp_path: Path, monkeypatch
) -> None:
    import diskcache

    import esdc.chat.tools as chat_tools
    from esdc.analytics_rehearsal import run_analytics_rehearsal
    from esdc.configs import Config

    baseline, manifest = _fixture_pair(tmp_path)
    output = tmp_path / "interrupted-run"
    marker: dict[str, object] = {"t6-marker": None}
    Config._config_cache = marker
    original_sql_cache = chat_tools._sql_cache
    original_tool_cache = chat_tools._tool_cache
    caller_sql_cache = diskcache.Cache(str(tmp_path / "caller-sql-cache"))
    caller_tool_cache = diskcache.Cache(str(tmp_path / "caller-tool-cache"))
    caller_sql_cache["caller"] = "preserved"
    caller_tool_cache["caller"] = "preserved"
    chat_tools._sql_cache = caller_sql_cache
    chat_tools._tool_cache = caller_tool_cache
    monkeypatch.setenv("ESDC_CONFIG_DIR", str(tmp_path / "prior-config"))
    monkeypatch.setenv("ESDC_DB_DIR", str(tmp_path / "prior-db"))
    monkeypatch.setenv("ESDC_DB_FILE", str(tmp_path / "prior-db" / "active.duckdb"))
    prior_env = {
        key: os.environ[key]
        for key in ("ESDC_CONFIG_DIR", "ESDC_DB_DIR", "ESDC_DB_FILE")
    }

    from esdc import analytics_rehearsal

    owned_handles = []

    @contextmanager
    def interrupt_transport(*args, **kwargs):
        owned_handles.extend((chat_tools._sql_cache, chat_tools._tool_cache))
        raise KeyboardInterrupt("T6 interruption probe")
        yield []

    monkeypatch.setattr(analytics_rehearsal, "_fixture_transport", interrupt_transport)
    with pytest.raises(KeyboardInterrupt, match="T6 interruption probe"):
        run_analytics_rehearsal(
            baseline_dir=baseline, fixture_manifest=manifest, output_dir=output
        )
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert report["status"] == "FAIL"
    assert report["failed_stage"] == "default_fetch"
    assert report["failed_exit"] == 1
    assert Config._config_cache is marker
    assert chat_tools._sql_cache is caller_sql_cache
    assert chat_tools._tool_cache is caller_tool_cache
    assert caller_sql_cache["caller"] == caller_tool_cache["caller"] == "preserved"
    assert len(owned_handles) == 2
    assert all(getattr(handle._local, "con", None) is None for handle in owned_handles)
    assert {key: os.environ[key] for key in prior_env} == prior_env
    caller_sql_cache.close()
    caller_tool_cache.close()
    chat_tools._sql_cache = original_sql_cache
    chat_tools._tool_cache = original_tool_cache


def test_root_cli_imports_without_resource_module(tmp_path: Path) -> None:
    script = r"""
import builtins, json
original = builtins.__import__
def blocked(name, *args, **kwargs):
    if name == 'resource':
        raise ImportError('resource intentionally unavailable')
    return original(name, *args, **kwargs)
builtins.__import__ = blocked
from esdc.analytics_rehearsal import _rss_bytes
from esdc.esdc import app
from typer.testing import CliRunner
result = CliRunner().invoke(app, ['verify-analytics-migration', '--help'])
print(json.dumps({'exit': result.exit_code, 'telemetry': _rss_bytes()}))
raise SystemExit(result.exit_code)
"""
    env = os.environ.copy()
    env.update(
        {
            "ESDC_CONFIG_DIR": str(tmp_path / "config"),
            "ESDC_DB_DIR": str(tmp_path / "db"),
            "ESDC_DB_FILE": str(tmp_path / "db" / "active.duckdb"),
        }
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert payload["exit"] == 0
    assert payload["telemetry"]["peak_rss_bytes"] is None
    assert (
        "resource intentionally unavailable" in payload["telemetry"]["peak_rss_reason"]
    )


def _hash(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
