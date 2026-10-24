"""Offline closed-clone rehearsal tests for API resource migration."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import cast

import duckdb
import pytest
from diskcache import Cache

from scripts.verify_api_resource_migration import _logical_state, run_rehearsal
from scripts.verify_api_resource_migration import main as migration_main


def _write_inputs(root: Path) -> tuple[Path, Path]:
    baseline = root / "baseline"
    fixture_root = root / "fixtures"
    baseline.mkdir()
    fixture_root.mkdir()
    (baseline / "esdc.sqlite").touch()
    with duckdb.connect(str(baseline / "esdc.duckdb")) as conn:
        conn.execute("LOAD spatial")
        conn.execute(
            "CREATE TABLE project_resources "
            "(project_id VARCHAR, report_year INTEGER, project_name VARCHAR)"
        )
        conn.execute(
            "INSERT INTO project_resources VALUES ('legacy-keep-until-bootstrap', 2019, 'legacy row')"
        )
        conn.execute(Path("esdc/sql/create_table_project_timeseries.sql").read_text())
        for year, project_id in (
            (2019, "historical-timeseries-sentinel"),
            (2025, "empty-response-timeseries-sentinel"),
            (2023, "unrequested-timeseries-sentinel"),
            (None, "null-year-timeseries-sentinel"),
        ):
            ts_values = {"report_year": year, "project_id": project_id}
            quoted_columns = ", ".join(f'"{name}"' for name in ts_values)
            conn.execute(
                f"INSERT INTO project_timeseries ({quoted_columns}) VALUES (?, ?)",
                list(ts_values.values()),
            )
        conn.execute("CREATE TABLE protected_sentinel (value VARCHAR)")
        conn.execute("INSERT INTO protected_sentinel VALUES ('keep-me')")
        conn.execute("CREATE TABLE document_chunks (id VARCHAR, content VARCHAR)")
        conn.execute("INSERT INTO document_chunks VALUES ('doc-sentinel', 'preserve me')")
        conn.execute("CREATE TABLE project_embeddings (project_id VARCHAR, embedding FLOAT[])")
        conn.execute("INSERT INTO project_embeddings VALUES ('embedding-sentinel', [0.25, 0.5])")

    resources = [
        {
            "project_id": "001",
            "report_year": "2024",
            "project_name": "Fixture field",
            "project_class": "1P",
            "project_level": "E",
            "field_geom": "POINT (106  -6)",
            "grov_current_situation": "label one",
            "grov_executive_summary": "label two",
            "rec_oil": 0,
            "rec_con": 7.5,
        },
        {
            "project_id": "002",
            "report_year": "2025",
            "project_name": "Malformed geometry",
            "project_class": "3P",
            "project_level": "A",
            "field_geom": "NOT WKT",
            "rec_oil": None,
        },
    ]
    payloads = {
        "project_resources.json": json.dumps(resources).encode(),
        "project_timeseries_2024.json": json.dumps(
            [{"report_year": "2024", "project_id": "001", "rate_oil": 0}]
        ).encode(),
        "project_timeseries_2025.json": b"[]",
    }
    responses = []
    for filename, data in payloads.items():
        (fixture_root / filename).write_bytes(data)
        stem = filename.rsplit(".", 1)[0]
        if stem == "project_resources":
            table, year = "project_resources", None
        else:
            table, year = "project_timeseries", int(stem.rsplit("_", 1)[1])
        responses.append(
            {
                "table": table,
                "report_year": year,
                "filetype": "json",
                "path": filename,
                "sha256": hashlib.sha256(data).hexdigest(),
                "status": 200,
            }
        )
    manifest = fixture_root / "manifest.json"
    manifest.write_text(
        json.dumps(
            {"schema_version": 1, "resource_scope": "unfiltered", "responses": responses}
        ),
        encoding="utf-8",
    )
    return baseline, manifest


def _assert_pair_bootstrap_and_recovery(stages: dict[str, dict[str, object]]) -> None:
    bootstrap = stages["unfiltered_cli_bootstrap"]["observations"]
    assert isinstance(bootstrap, dict)
    proof = bootstrap["timeseries_bootstrap"]
    assert isinstance(proof, dict)
    dry_run = proof["dry_run"]
    applied = proof["apply"]
    assert isinstance(dry_run, dict) and isinstance(applied, dict)
    assert dry_run["applied"] is False
    assert dry_run["duckdb_sha256_unchanged"] is True
    assert dry_run["sqlite_sha256_unchanged"] is True
    assert dry_run["logical_state_unchanged"] is True
    assert dry_run["api_call_count_unchanged"] is True
    assert applied["applied"] is True
    assert applied["duckdb_sha256_unchanged"] is True
    assert applied["logical_state_unchanged"] is True
    assert applied["api_call_count_unchanged"] is True
    canonical_timeseries = applied["canonical_timeseries"]
    assert isinstance(canonical_timeseries, dict)
    assert canonical_timeseries["current_provenance_origin"] == "legacy_duckdb"
    assert canonical_timeseries["origins"] == ["legacy_duckdb"]

    recovery = stages["post_c1_failure_and_api_free_rebuild"]["observations"]
    assert isinstance(recovery, dict)
    before = recovery["canonical_api_pair_sources_before_rebuild"]
    after = recovery["canonical_api_pair_sources_after_rebuild"]
    rebuild = recovery["pair_rebuild_report"]
    publication = recovery["rebuild_publication"]
    fts = recovery["fts_search_after_rebuild"]
    parity = recovery["working_clone_parity_after_rebuild"]
    caches = recovery["analytical_caches_before_after_rebuild"]
    assert isinstance(before, dict) and isinstance(after, dict)
    assert isinstance(rebuild, dict) and isinstance(parity, dict)
    assert isinstance(publication, dict) and isinstance(fts, dict)
    assert isinstance(caches, dict)
    for table in ("project_resources", "project_timeseries"):
        before_source = before[table]
        after_source = after[table]
        assert isinstance(before_source, dict) and isinstance(after_source, dict)
        assert after_source["revision"] == before_source["revision"]
        assert after_source["source_sha256"] == before_source["source_sha256"]
        assert after_source["publication_state"] == "published"
        assert rebuild["source_revisions"][table] == before_source["revision"]
        assert rebuild["source_hashes"][table] == before_source["source_sha256"]
    assert publication["revision"] == before["project_resources"]["revision"]
    assert publication["row_count"] == rebuild["object_counts"]["project_resources"]
    assert publication["api_calls_during_rebuild"] == 0
    assert fts["hits"] > 0
    assert parity["resources"]["ok"] is True
    assert parity["timeseries"]["ok"] is True
    assert parity["timeseries"]["lineage_matches"] is True
    assert parity["timeseries"]["value_difference_count"] == 0
    assert caches["before"]["sql"]["entries"] > 0
    assert caches["before"]["tools"]["entries"] > 0
    assert caches["after"]["sql"]["entries"] == 0
    assert caches["after"]["tools"]["entries"] == 0


def test_small_fixture_rehearsal_writes_a_complete_report(tmp_path: Path) -> None:
    baseline, manifest = _write_inputs(tmp_path)
    resource_fixture = json.loads((manifest.parent / "project_resources.json").read_text(encoding="utf-8"))
    timeseries_fixture = json.loads((manifest.parent / "project_timeseries_2024.json").read_text(encoding="utf-8"))
    assert all(isinstance(row["report_year"], str) for row in resource_fixture)
    assert isinstance(timeseries_fixture[0]["report_year"], str)
    before = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in baseline.iterdir()
    }

    report_path = run_rehearsal(
        baseline_dir=baseline,
        fixture_manifest=manifest,
        output_dir=tmp_path / "runs",
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["schema_version"] == 1
    assert report["milestone_b"]["status"] == "PASS"
    assert report["representative_full_size"] == "UNVERIFIED"
    assert report["real_database_cutover"] == "PENDING AUTHORIZATION"
    assert all(stage["status"] == "PASS" for stage in report["stages"])
    stage_map = {stage["name"]: stage for stage in report["stages"]}
    _assert_pair_bootstrap_and_recovery(stage_map)
    target_stage = stage_map["separate_empty_target_rebuild_and_repeat"]["observations"]
    assert target_stage["unexpected_baseline_protected_tables_restored"] == []
    assert target_stage["forbidden_duckdb_owned_data_tables_present"] == []
    assert "project_timeseries" in target_stage["target_table_names"]
    assert target_stage["timeseries_parity"]["ok"] is True
    pair_sources = target_stage["canonical_pair_sources_before_after"]
    assert pair_sources["before"] == pair_sources["after"]
    assert target_stage["rebuild_source_revisions"]["project_timeseries"] == pair_sources["before"]["project_timeseries"]["revision"]
    failure_stage = stage_map["empty_and_download_failure_injections"]["observations"]["download_failure"]
    assert failure_stage["snapshot_unchanged"]
    assert failure_stage["analytical_state_unchanged"]
    assert failure_stage["cache_state_unchanged"]
    assert stage_map["typed_parity_views_spatial_search_and_protected_rows"]["observations"]["fts_search"]["hits"] > 0
    with duckdb.connect(str(Path(report["run_dir"]) / "db" / "esdc.duckdb"), read_only=True) as conn:
        conn.execute("LOAD spatial")
        derived = conn.execute(
            "SELECT project_id, is_discovered, project_stage, field_geom, "
            "typeof(rec_oil), rec_oil, ST_GeometryType(field_geometry), "
            "ST_AsText(field_geometry) FROM project_resources ORDER BY project_id"
        ).fetchall()
        assert derived == [
            ("001", 1, "1. Exploitation", "POINT (106  -6)", "FLOAT", 0.0, "POINT", "POINT (106 -6)"),
            ("002", 0, "3. Abandoned", "NOT WKT", "FLOAT", None, None, None),
        ]
        assert conn.execute(
            "SELECT report_year, project_id FROM project_timeseries WHERE project_id LIKE '%timeseries-sentinel' ORDER BY project_id"
        ).fetchall() == [
            (2025, "empty-response-timeseries-sentinel"),
            (2019, "historical-timeseries-sentinel"),
            (None, "null-year-timeseries-sentinel"),
            (2023, "unrequested-timeseries-sentinel"),
        ]
        view_counts = dict(stage_map["typed_parity_views_spatial_search_and_protected_rows"]["observations"]["view_row_counts"])
        assert {name: view_counts[name] for name in ("field_resources", "wa_resources", "nkri_resources")} == {
            "field_resources": 2,
            "wa_resources": 2,
            "nkri_resources": 2,
        }
        assert {name: view_counts[name] for name in ("field_timeseries", "wa_timeseries", "nkri_timeseries")} == {
            "field_timeseries": 5,
            "wa_timeseries": 5,
            "nkri_timeseries": 5,
        }
        for view in ("field_resources", "wa_resources", "nkri_resources"):
            assert conn.execute(f"SELECT SUM(rec_con), SUM(rec_oil) FROM {view}").fetchone() == (7.5, 0.0)
        assert conn.execute("SELECT * FROM project_embeddings").fetchall() == [
            ("embedding-sentinel", [0.25, 0.5])
        ]
    sqlite_path = Path(report["run_dir"]) / "db" / "esdc.sqlite"
    with sqlite3.connect(f"file:{sqlite_path.resolve()}?mode=ro", uri=True) as conn:
        assert conn.execute("SELECT DISTINCT typeof(report_year) FROM api_project_resources").fetchall() == [("integer",)]
        assert conn.execute("SELECT report_year FROM api_project_resources WHERE project_id='001'").fetchone() == (2024,)
        groovy = conn.execute(
            "SELECT groovy_report FROM api_project_resources WHERE project_id='001'"
        ).fetchone()
        columns = {row[1] for row in conn.execute("PRAGMA table_info(api_project_resources)")}
        assert groovy == ("Current situation:\nlabel one\n\nExecutive summary:\nlabel two",)
        assert not any(column.startswith("grov_") for column in columns)
    after = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in baseline.iterdir()
    }
    assert before == after


def test_runner_derives_scope_and_protected_data_from_changed_fixture_inputs(
    tmp_path: Path,
) -> None:
    baseline, manifest = _write_inputs(tmp_path)
    fixture_root = manifest.parent
    with duckdb.connect(str(baseline / "esdc.duckdb")) as conn:
        conn.execute('CREATE TABLE "unrelated""domain_cache" (opaque_id VARCHAR, payload VARCHAR)')
        conn.execute('INSERT INTO "unrelated""domain_cache" VALUES (\'arbitrary-key\', \'opaque payload\')')
        conn.execute("CREATE TABLE _metadata (key VARCHAR PRIMARY KEY, value VARCHAR)")
        conn.execute("INSERT INTO _metadata VALUES ('operator_note', 'preserve-this'), ('last_updated', 'old timestamp')")
        conn.execute("DELETE FROM project_timeseries")
        conn.execute(
            "INSERT INTO project_timeseries (report_year, project_id) VALUES (?, ?), (?, ?), (?, ?)",
            [2018, "unusual-old-row", None, "unusual-null-row", 2033, "unusual-empty-response-row"],
        )

    resources = [
        {
            "project_id": "north-88",
            "report_year": "2031",
            "project_name": "North Basin",
            "project_class": "1P",
            "project_level": "E",
            "field_geom": "POINT (109 -4)",
        },
        {
            "project_id": "south-14",
            "report_year": "2033",
            "project_name": "South Basin",
            "project_class": "3P",
            "project_level": "A",
        },
    ]
    fixture_payloads = {
        "project_resources.json": resources,
        "project_timeseries_2031.json": [
            {"report_year": "2031", "project_id": "north-88", "rate_oil": 7}
        ],
        "project_timeseries_2033.json": [],
    }
    responses = []
    for filename, rows in fixture_payloads.items():
        data = json.dumps(rows).encode()
        (fixture_root / filename).write_bytes(data)
        if filename == "project_resources.json":
            table, year = "project_resources", None
        else:
            table, year = "project_timeseries", int(filename.split("_")[-1].split(".")[0])
        responses.append(
            {
                "table": table,
                "report_year": year,
                "filetype": "json",
                "path": filename,
                "sha256": hashlib.sha256(data).hexdigest(),
                "status": 200,
            }
        )
    manifest.write_text(
        json.dumps({"schema_version": 1, "resource_scope": "unfiltered", "responses": responses}),
        encoding="utf-8",
    )

    report_path = run_rehearsal(
        baseline_dir=baseline,
        fixture_manifest=manifest,
        output_dir=tmp_path / "changed-runs",
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["milestone_b"]["status"] == "PASS"
    assert all(stage["status"] == "PASS" for stage in report["stages"])
    stages = {stage["name"]: stage["observations"] for stage in report["stages"]}
    _assert_pair_bootstrap_and_recovery(
        {name: {"observations": observations} for name, observations in stages.items()}
    )
    assert stages["unfiltered_cli_bootstrap"]["requested_and_scoped_years"]["timeseries_fixture_years"] == [2031, 2033]
    assert stages["typed_parity_views_spatial_search_and_protected_rows"]["preserved_non_resource_table_hashes"]['unrelated"domain_cache']["row_count"] == 1
    assert stages["typed_parity_views_spatial_search_and_protected_rows"]["unrequested_and_null_year_timeseries"]["row_count"] == 2
    assert stages["typed_parity_views_spatial_search_and_protected_rows"]["valid_empty_timeseries_preservation"]["before"] == stages["typed_parity_views_spatial_search_and_protected_rows"]["valid_empty_timeseries_preservation"]["after"]
    selective = stages["selective_year_refresh"]
    assert selective["unselected_canonical_resource_values_before_after"]["before"] == selective["unselected_canonical_resource_values_before_after"]["after"]
    assert selective["unselected_and_null_timeseries_values_before_after"]["before"] == selective["unselected_and_null_timeseries_values_before_after"]["after"]
    target_stage = stages["separate_empty_target_rebuild_and_repeat"]
    assert "project_timeseries" in target_stage["target_table_names"]
    assert target_stage["timeseries_parity"]["ok"] is True
    assert target_stage["canonical_pair_sources_before_after"]["before"] == target_stage["canonical_pair_sources_before_after"]["after"]
    with duckdb.connect(str(Path(report["run_dir"]) / "db" / "esdc.duckdb"), read_only=True) as conn:
        assert conn.execute("SELECT value FROM _metadata WHERE key='operator_note'").fetchone() == ("preserve-this",)


def test_logical_state_detects_same_count_resource_and_timeseries_value_changes(
    tmp_path: Path,
) -> None:
    path = tmp_path / "logical-state.duckdb"
    with duckdb.connect(str(path)) as conn:
        conn.execute("CREATE TABLE project_resources (project_id VARCHAR, report_year INTEGER, source_value VARCHAR)")
        conn.execute("INSERT INTO project_resources VALUES ('res-a', 2031, 'before')")
        conn.execute("CREATE TABLE project_timeseries (project_id VARCHAR, report_year INTEGER, source_value VARCHAR)")
        conn.execute("INSERT INTO project_timeseries VALUES ('res-a', 2031, 'before')")

    before = _logical_state(path)
    with duckdb.connect(str(path)) as conn:
        conn.execute("UPDATE project_resources SET source_value='after'")
    resource_changed = _logical_state(path)
    before_tables = cast(dict[str, object], before["managed_table_states"])
    resource_tables = cast(dict[str, object], resource_changed["managed_table_states"])
    assert before_tables["project_resources"] != resource_tables["project_resources"]
    with duckdb.connect(str(path)) as conn:
        conn.execute("UPDATE project_timeseries SET source_value='after'")
    timeseries_changed = _logical_state(path)
    timeseries_tables = cast(dict[str, object], timeseries_changed["managed_table_states"])
    assert resource_tables["project_timeseries"] != timeseries_tables["project_timeseries"]


def test_runner_rejects_fixture_path_escape_before_copying_inputs(
    tmp_path: Path,
) -> None:
    baseline, manifest = _write_inputs(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["responses"][0]["path"] = "../outside.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="fixture path"):
        run_rehearsal(
            baseline_dir=baseline,
            fixture_manifest=manifest,
            output_dir=tmp_path / "runs",
        )

    assert not (tmp_path / "runs").exists()


def test_runner_rejects_fixture_hash_mismatch(tmp_path: Path) -> None:
    baseline, manifest = _write_inputs(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["responses"][0]["sha256"] = "0" * 64
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="SHA-256"):
        run_rehearsal(
            baseline_dir=baseline,
            fixture_manifest=manifest,
            output_dir=tmp_path / "runs",
        )

    assert not (tmp_path / "runs").exists()


def test_runner_rejects_wal_output_overlap_and_default_live_paths(tmp_path: Path) -> None:
    baseline, manifest = _write_inputs(tmp_path)
    wal = baseline / "esdc.sqlite-wal"
    wal.write_bytes(b"incomplete clone")
    with pytest.raises(ValueError, match="WAL-bearing"):
        run_rehearsal(baseline_dir=baseline, fixture_manifest=manifest, output_dir=tmp_path / "runs")
    wal.unlink()

    with pytest.raises(ValueError, match="overlap"):
        run_rehearsal(baseline_dir=baseline, fixture_manifest=manifest, output_dir=baseline)

    with pytest.raises(ValueError, match="~/.esdc"):
        run_rehearsal(
            baseline_dir=Path.home() / ".esdc" / "closed-clone",
            fixture_manifest=manifest,
            output_dir=tmp_path / "runs",
        )


def test_runner_rejects_missing_timeseries_request_fixture(tmp_path: Path) -> None:
    baseline, manifest = _write_inputs(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["responses"] = [
        response
        for response in payload["responses"]
        if not (response["table"] == "project_timeseries" and response["report_year"] == 2024)
    ]
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="timeseries fixtures are missing"):
        run_rehearsal(baseline_dir=baseline, fixture_manifest=manifest, output_dir=tmp_path / "runs")
    assert not (tmp_path / "runs").exists()


def test_main_returns_nonzero_and_writes_report_when_rehearsal_assertion_fails(
    tmp_path: Path,
) -> None:
    baseline, manifest = _write_inputs(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["responses"][0]["status"] = 503
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    output = tmp_path / "runs"

    code = migration_main(
        [
            "--baseline-dir",
            str(baseline),
            "--fixture-manifest",
            str(manifest),
            "--output-dir",
            str(output),
        ]
    )

    assert code == 1
    reports = list(output.glob("resource-migration-*/report.json"))
    assert len(reports) == 1
    report = json.loads(reports[0].read_text(encoding="utf-8"))
    assert report["milestone_b"]["status"] == "FAIL"


def test_runner_preserves_preexisting_cache_handles_and_scopes_new_caches(
    tmp_path: Path,
) -> None:
    import esdc.chat.tools as chat_tools

    input_root = tmp_path / "inputs"
    input_root.mkdir()
    baseline, manifest = _write_inputs(input_root)
    old_sql, old_tools = chat_tools._sql_cache, chat_tools._tool_cache
    sentinel_sql = Cache(str(tmp_path / "external-sql-cache"), statistics=True)
    sentinel_tools = Cache(str(tmp_path / "external-tool-cache"), statistics=True)
    sentinel_sql.set("keep", "sql sentinel")
    sentinel_tools.set("keep", "tool sentinel")
    chat_tools._sql_cache = sentinel_sql
    chat_tools._tool_cache = sentinel_tools
    try:
        report_path = run_rehearsal(
            baseline_dir=baseline,
            fixture_manifest=manifest,
            output_dir=tmp_path / "runs",
        )
        report = json.loads(report_path.read_text(encoding="utf-8"))
        run_dir = Path(report["run_dir"]).resolve()
        assert chat_tools._sql_cache is sentinel_sql
        assert chat_tools._tool_cache is sentinel_tools
        assert sentinel_sql.get("keep") == "sql sentinel"
        assert sentinel_tools.get("keep") == "tool sentinel"
        assert sentinel_sql.set("still-usable", True)
        assert sentinel_tools.set("still-usable", True)
        assert run_dir in Path(report["cache_isolation"]["sql_cache_dir"]).resolve().parents
        assert run_dir in Path(report["cache_isolation"]["tool_cache_dir"]).resolve().parents
    finally:
        chat_tools._sql_cache = old_sql
        chat_tools._tool_cache = old_tools
        sentinel_sql.close()
        sentinel_tools.close()
