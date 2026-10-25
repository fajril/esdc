"""Run the offline, closed-clone API resource migration rehearsal."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import shutil
import sqlite3
import sys
import uuid
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import cast


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _counts_by_year(rows: Sequence[Sequence[object]]) -> dict[int | None, int]:
    counts: dict[int | None, int] = {}
    for row in rows:
        if len(row) != 2:
            raise ValueError("year-count row must contain exactly two values")
        year, count = row
        if year is not None and (not isinstance(year, int) or isinstance(year, bool)):
            raise ValueError("source report_year must be an integer or null")
        if not isinstance(count, int):
            raise ValueError("source row count must be an integer")
        counts[year] = count
    return counts


def _input_hashes(baseline: Path) -> dict[str, str]:
    return {
        path.name: _sha256(path)
        for path in (baseline / "esdc.sqlite", baseline / "esdc.duckdb")
    }


def _validate_inputs(
    baseline_dir: Path, fixture_manifest: Path, output_dir: Path
) -> tuple[Path, Path, Path, dict[tuple[str, int | None, str], dict[str, object]]]:
    baseline_candidate = baseline_dir.expanduser().resolve(strict=False)
    manifest_path = fixture_manifest.expanduser().resolve(strict=True)
    output = output_dir.expanduser().resolve(strict=False)
    home_esdc = (Path.home() / ".esdc").resolve(strict=False)
    if output == home_esdc or home_esdc in output.parents:
        raise ValueError("output under ~/.esdc is forbidden")
    if baseline_candidate == home_esdc or home_esdc in baseline_candidate.parents:
        raise ValueError("baseline under ~/.esdc is forbidden")
    baseline = baseline_candidate.resolve(strict=True)
    if not baseline.is_dir():
        raise ValueError("baseline-dir must be a directory")
    sqlite_path = baseline / "esdc.sqlite"
    duckdb_path = baseline / "esdc.duckdb"
    if not sqlite_path.is_file() or not duckdb_path.is_file():
        raise ValueError("baseline must contain esdc.sqlite and esdc.duckdb")
    for path in baseline.iterdir():
        lowered = path.name.lower()
        if path.is_symlink():
            raise ValueError(f"baseline clone files cannot be symlinks: {path.name}")
        if lowered.endswith(("-wal", "-shm", "-journal", ".wal")):
            raise ValueError(f"WAL-bearing or inconsistent baseline set: {path.name}")
    if output == baseline or output in baseline.parents or baseline in output.parents:
        raise ValueError("baseline and output directories overlap")
    fixture_root = manifest_path.parent.resolve(strict=True)
    if (
        output == fixture_root
        or output in fixture_root.parents
        or fixture_root in output.parents
    ):
        raise ValueError("fixture and output directories overlap")

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid fixture manifest: {exc}") from exc
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ValueError("fixture manifest schema_version must be 1")
    if manifest.get("resource_scope") != "unfiltered":
        raise ValueError("fixture manifest resource_scope must be 'unfiltered'")
    responses = manifest.get("responses")
    if not isinstance(responses, list):
        raise ValueError("fixture manifest responses must be a list")

    fixtures: dict[tuple[str, int | None, str], dict[str, object]] = {}
    for item in responses:
        if not isinstance(item, dict):
            raise ValueError("each fixture response must be an object")
        table, year, filetype = (
            item.get("table"),
            item.get("report_year"),
            item.get("filetype"),
        )
        if table not in {"project_resources", "project_timeseries"}:
            raise ValueError(f"unsupported fixture table: {table!r}")
        if year is not None and (not isinstance(year, int) or isinstance(year, bool)):
            raise ValueError("fixture report_year must be an integer or null")
        if filetype not in {"json", "csv"}:
            raise ValueError("fixture filetype must be json or csv")
        key = (table, year, filetype)
        if key in fixtures:
            raise ValueError(f"duplicate fixture request key: {key}")
        relative = Path(str(item.get("path", "")))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(
                f"fixture path must stay relative to the manifest: {relative}"
            )
        source = (fixture_root / relative).resolve(strict=True)
        if fixture_root not in source.parents or not source.is_file():
            raise ValueError(f"fixture path escapes fixture directory: {relative}")
        expected_hash = item.get("sha256")
        if (
            not isinstance(expected_hash, str)
            or _sha256(source).lower() != expected_hash.lower()
        ):
            raise ValueError(f"fixture SHA-256 mismatch: {relative}")
        status = item.get("status")
        if not isinstance(status, int) or isinstance(status, bool):
            raise ValueError("fixture status must be an integer")
        fixtures[key] = {
            **item,
            "resolved_path": source,
            "sha256": expected_hash.lower(),
        }
    if ("project_resources", None, "json") not in fixtures and (
        "project_resources",
        None,
        "csv",
    ) not in fixtures:
        raise ValueError("an unfiltered project_resources fixture is required")
    resource_fixture = next(
        value
        for (table, year, _), value in fixtures.items()
        if table == "project_resources" and year is None
    )
    resource_path = resource_fixture["resolved_path"]
    assert isinstance(resource_path, Path)
    resource_type = str(resource_fixture["filetype"])
    resource_years = _fixture_resource_years(resource_path, resource_type)
    if not resource_years:
        raise ValueError("unfiltered resource fixture must include report years")
    missing_years = [
        year
        for year in sorted(resource_years)
        if year >= 2020
        and not any(
            ("project_timeseries", year, kind) in fixtures for kind in ("json", "csv")
        )
    ]
    if missing_years:
        raise ValueError(
            f"timeseries fixtures are missing for resource years: {missing_years}"
        )
    if (
        output == fixture_root
        or output in fixture_root.parents
        or fixture_root in output.parents
    ):
        raise ValueError("fixture and output directories overlap")
    source = sqlite3.connect(f"file:{sqlite_path.resolve()}?mode=ro", uri=True)
    try:
        integrity = source.execute("PRAGMA integrity_check").fetchone()
        if integrity is None or integrity[0] != "ok":
            raise ValueError("SQLite baseline clone failed integrity_check")
    finally:
        source.close()
    import duckdb

    source_duckdb = duckdb.connect(str(duckdb_path), read_only=True)
    source_duckdb.close()
    return baseline, manifest_path, output, fixtures


def _fixture_resource_years(path: Path, filetype: str) -> set[int]:
    try:
        if filetype == "json":
            rows = json.loads(path.read_text(encoding="utf-8"))
        else:
            with path.open(encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.DictReader(stream, delimiter=";"))
    except (OSError, json.JSONDecodeError, csv.Error) as exc:
        raise ValueError(f"cannot inspect unfiltered resources fixture: {exc}") from exc
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError("unfiltered resources fixture must contain a list of objects")
    years: set[int] = set()
    for row in rows:
        value = row.get("report_year")
        if isinstance(value, int) and not isinstance(value, bool):
            years.add(value)
        elif isinstance(value, str) and value.strip().isdigit():
            years.add(int(value.strip()))
    return years


def _empty_timeseries_years(
    fixtures: dict[tuple[str, int | None, str], dict[str, object]],
) -> set[int]:
    empty_years: set[int] = set()
    for (table, year, filetype), fixture in fixtures.items():
        if table != "project_timeseries" or year is None:
            continue
        status = fixture["status"]
        if not isinstance(status, int) or isinstance(status, bool) or status != 200:
            continue
        path = fixture["resolved_path"]
        assert isinstance(path, Path)
        try:
            if filetype == "json":
                rows = json.loads(path.read_text(encoding="utf-8"))
            else:
                with path.open(encoding="utf-8-sig", newline="") as stream:
                    rows = list(csv.DictReader(stream, delimiter=";"))
        except (OSError, json.JSONDecodeError, csv.Error):
            continue
        if not rows:
            empty_years.add(year)
    return empty_years


@contextmanager
def _scoped_environment(values: dict[str, str]) -> Iterator[None]:
    previous = {name: os.environ.get(name) for name in values}
    try:
        os.environ.update(values)
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _assert_cached_extensions() -> None:
    import duckdb

    conn = duckdb.connect()
    try:
        conn.execute("LOAD fts")
        conn.execute("LOAD spatial")
    finally:
        conn.close()


def _record(stages: list[dict[str, object]], name: str, operation) -> object | None:
    stage: dict[str, object] = {"name": name, "status": "PASS", "observations": {}}
    stages.append(stage)
    try:
        result = operation(stage["observations"])
        return result
    except Exception as exc:  # report the first failing gate and stop dependent stages
        stage["status"] = "FAIL"
        stage["error"] = f"{type(exc).__name__}: {exc}"
        return None


def _set_stage(stage: object, key: str, value: object) -> None:
    assert isinstance(stage, dict)
    stage[key] = value


def _cli_run(app, argv: list[str]) -> tuple[int, str]:
    from typer.testing import CliRunner

    result = CliRunner().invoke(app, argv)
    return int(result.exit_code), str(result.output)


def _snapshot(path: Path) -> dict[str, object]:
    import sqlite3

    from esdc.api_snapshot import get_project_resources_snapshot

    conn = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
    try:
        snapshot = get_project_resources_snapshot(conn)
        if snapshot is None:
            raise AssertionError("canonical resource snapshot is missing")
        return {
            "revision": snapshot.revision,
            "row_count": snapshot.row_count,
            "source_sha256": snapshot.source_sha256,
            "publication_state": snapshot.publication_state,
        }
    finally:
        conn.close()


def _api_pair_source_state(path: Path) -> dict[str, object]:
    """Return and verify the current committed SQLite API source markers."""
    from esdc.api_snapshot import get_project_resources_snapshot
    from esdc.timeseries_snapshot import get_timeseries_snapshot, timeseries_fingerprint

    conn = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
    try:
        resources = get_project_resources_snapshot(conn)
        timeseries = get_timeseries_snapshot(conn)
        if timeseries is None:
            raise AssertionError("canonical timeseries snapshot is missing")
        row_count, source_hash = timeseries_fingerprint(conn)
        if row_count != timeseries.row_count or source_hash != timeseries.source_sha256:
            raise AssertionError("canonical timeseries snapshot hash/count is invalid")
        origins = [
            row[0]
            for row in conn.execute(
                "SELECT DISTINCT origin FROM api_project_timeseries ORDER BY origin"
            ).fetchall()
        ]
        provenance = conn.execute(
            "SELECT origin, details_json FROM api_snapshot_provenance "
            "WHERE source_table='project_timeseries' AND revision=?",
            (timeseries.revision,),
        ).fetchone()
        if provenance is None:
            raise AssertionError("canonical timeseries provenance is missing")
        return {
            "project_resources": (
                None
                if resources is None
                else {
                    "revision": resources.revision,
                    "row_count": resources.row_count,
                    "source_sha256": resources.source_sha256,
                    "publication_state": resources.publication_state,
                }
            ),
            "project_timeseries": {
                "revision": timeseries.revision,
                "row_count": row_count,
                "source_sha256": source_hash,
                "publication_state": timeseries.publication_state,
                "origins": origins,
                "current_provenance_origin": provenance[0],
                "current_provenance_details": json.loads(provenance[1]),
            },
        }
    finally:
        conn.close()


def _sqlite_unselected_resources_digest(
    path: Path, selected_year: int
) -> dict[str, object]:
    conn = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
    try:
        columns = [
            row[1] for row in conn.execute("PRAGMA table_info(api_project_resources)")
        ]
        projection = ", ".join(_quote_ident(name) for name in columns)
        cursor = conn.execute(
            f"SELECT {projection} FROM api_project_resources "
            "WHERE report_year IS NULL OR report_year <> ?",
            (selected_year,),
        )
        hashes: list[bytes] = []
        count = 0
        while batch := cursor.fetchmany(512):
            hashes.extend(hashlib.sha256(repr(row).encode()).digest() for row in batch)
            count += len(batch)
        hashes.sort()
        return {
            "row_count": count,
            "rows_sha256": hashlib.sha256(b"".join(hashes)).hexdigest(),
            "columns": columns,
        }
    finally:
        conn.close()


def _quote_ident(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _logical_table_digest(
    conn,
    table_name: str,
    *,
    ignore_recomputed_metadata: bool = False,
    exclude_columns: set[str] | None = None,
) -> dict[str, object] | None:
    quoted_table = _quote_ident(table_name)
    all_columns = [
        row[0] for row in conn.execute(f"DESCRIBE {quoted_table}").fetchall()
    ]
    columns = [name for name in all_columns if name not in (exclude_columns or set())]
    row_hashes: list[bytes] = []
    projection = ", ".join(_quote_ident(name) for name in columns)
    query = f"SELECT {projection} FROM {quoted_table}"
    if ignore_recomputed_metadata and table_name == "_metadata" and "key" in columns:
        query += ' WHERE "key" IS NULL OR "key" <> \'last_updated\''
    cursor = conn.execute(query)
    while batch := cursor.fetchmany(512):
        for row in batch:
            row_hashes.append(hashlib.sha256(repr(row).encode()).digest())
    row_hashes.sort()
    digest = hashlib.sha256(b"".join(row_hashes)).hexdigest()
    return {
        "row_count": len(row_hashes),
        "rows_sha256": digest,
        "column_count": len(columns),
        "columns_sha256": hashlib.sha256("\0".join(columns).encode()).hexdigest(),
    }


def _logical_state(
    path: Path, *, ignore_recomputed_metadata: bool = False
) -> dict[str, object]:
    import duckdb

    from esdc.api_projection import get_resource_publication

    conn = duckdb.connect(str(path), read_only=True)
    try:
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema='main' AND table_type='BASE TABLE'"
            ).fetchall()
        }
        state: dict[str, object] = {}
        protected_tables: dict[str, object] = {}
        for name in sorted(
            tables - {"project_resources", "api_publication", "project_timeseries"}
        ):
            if name.startswith(("fts_", "pragma_")):
                continue
            protected_tables[name] = _logical_table_digest(
                conn, name, ignore_recomputed_metadata=ignore_recomputed_metadata
            )
        state["protected_tables"] = protected_tables
        managed_table_states: dict[str, object] = {}
        for name in ("project_resources", "project_timeseries"):
            if name in tables:
                managed_table_states[name] = _logical_table_digest(conn, name)
        state["managed_table_states"] = managed_table_states
        try:
            publication = get_resource_publication(conn)
        except Exception:
            publication = None
        state["resource_publication"] = (
            None
            if publication is None
            else {
                "revision": publication.revision,
                "source_row_count": publication.source_row_count,
                "source_sha256": publication.source_sha256,
                "invalid_geometry_count": publication.invalid_geometry_count,
            }
        )
        views = conn.execute(
            "SELECT table_name FROM information_schema.views "
            "WHERE table_schema='main' ORDER BY table_name"
        ).fetchall()
        application_views = [
            row[0] for row in views if row[0].endswith(("_resources", "_timeseries"))
        ]
        view_counts: dict[str, int] = {}
        for name in application_views:
            result = conn.execute(
                f"SELECT COUNT(*) FROM {_quote_ident(name)}"
            ).fetchone()
            if result is None:
                raise AssertionError(f"view count query returned no row: {name}")
            view_counts[name] = int(result[0])
        state["view_counts"] = view_counts
        state["representative_view_states"] = {
            name: _logical_table_digest(conn, name, exclude_columns={"uuid"})
            for name in application_views
        }
        return state
    finally:
        conn.close()


def _table_columns(conn, table_name: str) -> tuple[str, ...]:
    return tuple(
        row[0]
        for row in conn.execute(f"DESCRIBE {_quote_ident(table_name)}").fetchall()
    )


def _untouched_timeseries_digest(
    path: Path, updated_years: set[int], columns: tuple[str, ...] | None = None
) -> dict[str, object] | None:
    import duckdb

    conn = duckdb.connect(str(path), read_only=True)
    try:
        exists = conn.execute(
            "SELECT 1 FROM information_schema.tables "
            "WHERE table_schema='main' AND table_name='project_timeseries'"
        ).fetchone()
        if exists is None:
            return None
        current_columns = _table_columns(conn, "project_timeseries")
        selected_columns = (
            current_columns
            if columns is None
            else tuple(name for name in columns if name in current_columns)
        )
        if not selected_columns or "report_year" not in selected_columns:
            return {
                "row_count": 0,
                "rows_sha256": hashlib.sha256().hexdigest(),
                "columns": [],
            }
        projection = ", ".join(_quote_ident(name) for name in selected_columns)
        placeholders = ", ".join("?" for _ in updated_years)
        where = (
            "report_year IS NULL"
            if not updated_years
            else f"report_year IS NULL OR report_year NOT IN ({placeholders})"
        )
        query = f"SELECT {projection} FROM project_timeseries WHERE {where}"
        row_hashes: list[bytes] = []
        cursor = conn.execute(query, sorted(updated_years))
        while batch := cursor.fetchmany(512):
            row_hashes.extend(
                hashlib.sha256(repr(row).encode()).digest() for row in batch
            )
        row_hashes.sort()
        return {
            "row_count": len(row_hashes),
            "rows_sha256": hashlib.sha256(b"".join(row_hashes)).hexdigest(),
            "columns": list(selected_columns),
        }
    finally:
        conn.close()


def _timeseries_year_digest(
    path: Path, years: set[int], columns: tuple[str, ...]
) -> dict[str, object] | None:
    import duckdb

    if not years:
        return {
            "row_count": 0,
            "rows_sha256": hashlib.sha256().hexdigest(),
            "columns": list(columns),
        }
    conn = duckdb.connect(str(path), read_only=True)
    try:
        current_columns = _table_columns(conn, "project_timeseries")
        selected_columns = tuple(name for name in columns if name in current_columns)
        projection = ", ".join(_quote_ident(name) for name in selected_columns)
        placeholders = ", ".join("?" for _ in years)
        cursor = conn.execute(
            f"SELECT {projection} FROM project_timeseries "
            f"WHERE report_year IN ({placeholders})",
            sorted(years),
        )
        hashes: list[bytes] = []
        while batch := cursor.fetchmany(512):
            hashes.extend(hashlib.sha256(repr(row).encode()).digest() for row in batch)
        hashes.sort()
        return {
            "row_count": len(hashes),
            "rows_sha256": hashlib.sha256(b"".join(hashes)).hexdigest(),
            "columns": list(selected_columns),
        }
    finally:
        conn.close()


def _baseline_counts(path: Path) -> dict[str, list[list[object]]]:
    import duckdb

    conn = duckdb.connect(str(path), read_only=True)
    try:
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema='main'"
            ).fetchall()
        }
        result: dict[str, list[list[object]]] = {}
        for table in ("project_resources", "project_timeseries"):
            if table in tables and "report_year" in {
                row[0] for row in conn.execute(f'DESCRIBE "{table}"').fetchall()
            }:
                result[table] = [
                    [year, count]
                    for year, count in conn.execute(
                        f'SELECT report_year, COUNT(*) FROM "{table}" '
                        "GROUP BY report_year ORDER BY report_year"
                    ).fetchall()
                ]
            else:
                result[table] = []
        return result
    finally:
        conn.close()


def run_rehearsal(
    *, baseline_dir: Path, fixture_manifest: Path, output_dir: Path
) -> Path:
    """Copy closed inputs, run the offline rehearsal, and return report.json."""
    baseline, manifest_path, output, fixtures = _validate_inputs(
        baseline_dir, fixture_manifest, output_dir
    )
    resource_fixture = next(
        value
        for (table, year, _), value in fixtures.items()
        if table == "project_resources" and year is None
    )
    resource_path = resource_fixture["resolved_path"]
    assert isinstance(resource_path, Path)
    selected_year = max(
        _fixture_resource_years(resource_path, str(resource_fixture["filetype"]))
    )
    fixture_years = {
        key[1]
        for key in fixtures
        if key[0] == "project_timeseries" and key[1] is not None
    }
    empty_timeseries_years = _empty_timeseries_years(fixtures)
    try:
        _assert_cached_extensions()
    except Exception as exc:
        raise RuntimeError(
            f"required cached DuckDB extensions unavailable: {exc}"
        ) from exc
    before_hashes = _input_hashes(baseline)
    baseline_counts = _baseline_counts(baseline / "esdc.duckdb")
    baseline_logical_state = _logical_state(baseline / "esdc.duckdb")
    baseline_preserved_tables = cast(
        dict[str, object],
        _logical_state(baseline / "esdc.duckdb", ignore_recomputed_metadata=True)[
            "protected_tables"
        ],
    )
    import duckdb

    with duckdb.connect(
        str(baseline / "esdc.duckdb"), read_only=True
    ) as baseline_connection:
        ts_exists = baseline_connection.execute(
            "SELECT 1 FROM information_schema.tables "
            "WHERE table_schema='main' AND table_name='project_timeseries'"
        ).fetchone()
        baseline_timeseries_columns = (
            _table_columns(baseline_connection, "project_timeseries")
            if ts_exists
            else ()
        )
    baseline_untouched_timeseries = _untouched_timeseries_digest(
        baseline / "esdc.duckdb", fixture_years, baseline_timeseries_columns
    )
    baseline_empty_timeseries = _timeseries_year_digest(
        baseline / "esdc.duckdb", empty_timeseries_years, baseline_timeseries_columns
    )
    output.mkdir(parents=True, exist_ok=True)
    run_dir = output / f"resource-migration-{uuid.uuid4().hex}"
    run_dir.mkdir()
    db_dir = run_dir / "db"
    db_dir.mkdir()
    shutil.copy2(baseline / "esdc.sqlite", db_dir / "esdc.sqlite")
    shutil.copy2(baseline / "esdc.duckdb", db_dir / "esdc.duckdb")
    config_dir = run_dir / "config"
    config_dir.mkdir()
    stages: list[dict[str, object]] = []
    report_path = run_dir / "report.json"
    report: dict[str, object] = {
        "schema_version": 1,
        "run_dir": str(run_dir),
        "inputs": {
            "baseline_dir": str(baseline),
            "fixture_manifest": str(manifest_path),
            "baseline_sha256_before": before_hashes,
            "baseline_analytical_counts_by_year": baseline_counts,
            "fixtures": [
                {
                    "table": key[0],
                    "report_year": key[1],
                    "filetype": key[2],
                    "path": str(value["path"]),
                    "sha256": value["sha256"],
                    "status": value["status"],
                }
                for key, value in sorted(
                    fixtures.items(), key=lambda item: str(item[0])
                )
            ],
        },
        "stages": stages,
        "milestone_a": "UNVERIFIED",
        "milestone_b": {"status": "RUNNING"},
        "representative_full_size": "UNVERIFIED",
        "real_database_cutover": "PENDING AUTHORIZATION",
    }

    def save_report() -> None:
        after_hashes = _input_hashes(baseline)
        report["baseline_sha256_after"] = after_hashes
        if not any(
            stage["name"] == "source_clone_hashes_unchanged" for stage in stages
        ):
            stage = {
                "name": "source_clone_hashes_unchanged",
                "status": "PASS",
                "observations": {"before": before_hashes, "after": after_hashes},
            }
            if before_hashes != after_hashes:
                stage["status"] = "FAIL"
                stage["error"] = "source clone hash changed"
            stages.append(stage)
        if not any(stage["name"] == "network_fallback_blocked" for stage in stages):
            stage = {
                "name": "network_fallback_blocked",
                "status": "PASS",
                "observations": {"network_attempt_count": len(network_attempts)},
            }
            if network_attempts:
                stage["status"] = "FAIL"
                stage["error"] = "network fallback was attempted"
            stages.append(stage)
        report["ignored_server_filter_simulations"] = list(ignored_filter_simulations)
        report["stages"] = stages
        failed = any(stage["status"] != "PASS" for stage in stages)
        report["milestone_a"] = "PASS" if not failed else "FAIL"
        report["milestone_b"] = {"status": "PASS" if not failed else "FAIL"}
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    with _scoped_environment(
        {
            "ESDC_CONFIG_DIR": str(config_dir),
            "ESDC_DB_DIR": str(db_dir),
            "ESDC_DB_FILE": str(db_dir / "esdc.duckdb"),
            "ESDC_USERNAME": "offline-fixture-user",
            "ESDC_PASSWORD": "offline-fixture-password",
        }
    ):
        from unittest.mock import patch

        import esdc.chat.tools as chat_tools
        from esdc.configs import Config

        previous_config_cache = Config._config_cache
        previous_sql_cache = chat_tools._sql_cache
        previous_tool_cache = chat_tools._tool_cache
        previous_handlers = list(logging.root.handlers)
        previous_log_level = logging.root.level
        chat_tools._sql_cache = None
        chat_tools._tool_cache = None
        logging.root.handlers = []
        Config._config_cache = None
        local_sql_cache = chat_tools._get_cache()
        local_tool_cache = chat_tools._get_tool_cache()
        report["cache_isolation"] = {
            "sql_cache_dir": str(local_sql_cache.directory),
            "tool_cache_dir": str(local_tool_cache.directory),
        }
        import esdc.esdc as cli_module
        import esdc.fetch_publication as publication_module
        from esdc.resource_audit import compare_resource_projection

        original_downloader = cli_module._download_table_bytes
        api_calls: list[dict[str, object]] = []
        ignored_filter_simulations: list[dict[str, object]] = []
        network_attempts: list[str] = []
        use_empty_resources = False
        fail_download = False

        def fixture_downloader(table, filetype, username, password, report_year):
            nonlocal use_empty_resources, fail_download
            table_name = table.value if hasattr(table, "value") else str(table)
            type_name = filetype.value if hasattr(filetype, "value") else str(filetype)
            api_calls.append(
                {"table": table_name, "report_year": report_year, "filetype": type_name}
            )
            if fail_download:
                raise RuntimeError("injected offline download failure")
            key = (table_name, report_year, type_name)
            fixture = fixtures.get(key)
            if (
                fixture is None
                and table_name == "project_resources"
                and report_year is not None
            ):
                fallback_key = (table_name, None, type_name)
                fixture = fixtures.get(fallback_key)
                if fixture is not None:
                    ignored_filter_simulations.append(
                        {
                            "requested_year": report_year,
                            "fixture_year": None,
                            "filetype": type_name,
                            "local_scoping": "required by prepare_fetch",
                        }
                    )
            if fixture is None:
                raise RuntimeError(f"missing offline fixture request: {key}")
            path = fixture["resolved_path"]
            assert isinstance(path, Path)
            if use_empty_resources and table_name == "project_resources":
                return b"[]", f"fixture://{key[0]}?year={report_year}"
            status = fixture["status"]
            if not isinstance(status, int) or isinstance(status, bool) or status != 200:
                raise RuntimeError(f"fixture returned HTTP {fixture['status']}")
            return path.read_bytes(), f"fixture://{key[0]}?year={report_year}"

        def invoke_fetch(args: list[str]) -> tuple[int, str]:
            return _cli_run(cli_module.app, args)

        def reject_network(*args, **kwargs):
            network_attempts.append("esdc_downloader invoked")
            raise AssertionError("offline migration rehearsal attempted network access")

        try:
            with (
                patch.object(
                    Config,
                    "get_credentials",
                    return_value=("offline-user", "offline-pass"),
                ),
                patch.object(
                    cli_module, "_download_table_bytes", side_effect=fixture_downloader
                ),
                patch.object(cli_module, "esdc_downloader", side_effect=reject_network),
            ):

                def selective_refusal(observations: object) -> None:
                    before_db = _sha256(db_dir / "esdc.duckdb")
                    before_sqlite = _sha256(db_dir / "esdc.sqlite")
                    code, output_text = invoke_fetch(
                        ["fetch", "--year", str(selected_year)]
                    )
                    _set_stage(observations, "cli_exit_code", code)
                    _set_stage(observations, "message", output_text.strip())
                    if (
                        code == 0
                        or "phase=preflight" not in output_text
                        or "Cannot update selected resource years" not in output_text
                        or _sha256(db_dir / "esdc.duckdb") != before_db
                        or _sha256(db_dir / "esdc.sqlite") != before_sqlite
                    ):
                        raise AssertionError(
                            "selective fetch did not refuse read-only on an "
                            "empty canonical baseline"
                        )
                    _set_stage(observations, "protected_pair_unchanged", True)

                _record(stages, "selective_fetch_refusal", selective_refusal)
                if stages[-1]["status"] != "PASS":
                    save_report()
                    return report_path

                def unfiltered_bootstrap(observations: object) -> None:
                    from esdc.timeseries_bootstrap import bootstrap_timeseries_source

                    duckdb_path = db_dir / "esdc.duckdb"
                    sqlite_path = db_dir / "esdc.sqlite"
                    duckdb_before = _sha256(duckdb_path)
                    sqlite_before = _sha256(sqlite_path)
                    logical_before = _logical_state(duckdb_path)
                    api_calls_before_bootstrap = len(api_calls)
                    legacy_rows = sum(
                        cast(int, count)
                        for _year, count in cast(
                            list[list[object]],
                            baseline_counts.get("project_timeseries", []),
                        )
                    )
                    dry_run = bootstrap_timeseries_source(
                        duckdb_path=duckdb_path, sqlite_path=sqlite_path, apply=False
                    )
                    dry_duckdb_hash = _sha256(duckdb_path)
                    dry_sqlite_hash = _sha256(sqlite_path)
                    logical_after_dry_run = _logical_state(duckdb_path)
                    if (
                        dry_run.applied
                        or dry_run.row_count != legacy_rows
                        or dry_duckdb_hash != duckdb_before
                        or dry_sqlite_hash != sqlite_before
                        or logical_after_dry_run != logical_before
                    ):
                        raise AssertionError(
                            "timeseries bootstrap dry-run changed its source or "
                            "did not inspect the complete legacy table"
                        )
                    applied = bootstrap_timeseries_source(
                        duckdb_path=duckdb_path, sqlite_path=sqlite_path, apply=True
                    )
                    apply_duckdb_hash = _sha256(duckdb_path)
                    logical_after_apply = _logical_state(duckdb_path)
                    source_state = _api_pair_source_state(sqlite_path)
                    timeseries_source = cast(
                        dict[str, object], source_state["project_timeseries"]
                    )
                    api_calls_after_bootstrap = len(api_calls)
                    provenance_details = cast(
                        dict[str, object],
                        timeseries_source["current_provenance_details"],
                    )
                    if (
                        not applied.applied
                        or applied.row_count != legacy_rows
                        or applied.source_types != dry_run.source_types
                        or apply_duckdb_hash != duckdb_before
                        or logical_after_apply != logical_before
                        or timeseries_source["row_count"] != legacy_rows
                        or timeseries_source["current_provenance_origin"]
                        != "legacy_duckdb"
                        or api_calls_before_bootstrap != api_calls_after_bootstrap
                        or provenance_details.get("source_types")
                        != applied.source_types
                        or provenance_details.get("row_count") != legacy_rows
                    ):
                        raise AssertionError(
                            "timeseries bootstrap did not preserve the DuckDB "
                            "legacy table or commit its complete canonical proof"
                        )
                    _set_stage(
                        observations,
                        "timeseries_bootstrap",
                        {
                            "dry_run": {
                                "applied": dry_run.applied,
                                "row_count": dry_run.row_count,
                                "null_year_rows": dry_run.null_year_rows,
                                "source_types": dry_run.source_types,
                                "source_sha256": dry_run.source_sha256,
                                "duckdb_sha256_unchanged": dry_duckdb_hash
                                == duckdb_before,
                                "sqlite_sha256_unchanged": dry_sqlite_hash
                                == sqlite_before,
                                "logical_state_unchanged": (
                                    logical_after_dry_run == logical_before
                                ),
                                "api_call_count_unchanged": (
                                    api_calls_before_bootstrap
                                    == api_calls_after_bootstrap
                                ),
                            },
                            "apply": {
                                "applied": applied.applied,
                                "row_count": applied.row_count,
                                "null_year_rows": applied.null_year_rows,
                                "source_types": applied.source_types,
                                "source_sha256": applied.source_sha256,
                                "duckdb_sha256_unchanged": apply_duckdb_hash
                                == duckdb_before,
                                "logical_state_unchanged": (
                                    logical_after_apply == logical_before
                                ),
                                "api_call_count_unchanged": (
                                    api_calls_before_bootstrap
                                    == api_calls_after_bootstrap
                                ),
                                "canonical_timeseries": timeseries_source,
                            },
                        },
                    )
                    code, output_text = invoke_fetch(["fetch"])
                    _set_stage(observations, "argv", ["fetch"])
                    _set_stage(observations, "cli_exit_code", code)
                    _set_stage(observations, "output", output_text.strip())
                    if code != 0:
                        raise AssertionError(
                            f"unfiltered CLI bootstrap failed: {output_text.strip()}"
                        )
                    state = _snapshot(db_dir / "esdc.sqlite")
                    _set_stage(observations, "canonical_snapshot", state)
                    sqlite_source = sqlite3.connect(
                        f"file:{(db_dir / 'esdc.sqlite').resolve()}?mode=ro", uri=True
                    )
                    try:
                        source_counts = [
                            [year, count]
                            for year, count in sqlite_source.execute(
                                "SELECT report_year, COUNT(*) FROM "
                                "api_project_resources "
                                "GROUP BY report_year ORDER BY report_year"
                            ).fetchall()
                        ]
                    finally:
                        sqlite_source.close()
                    _set_stage(
                        observations, "canonical_source_counts_by_year", source_counts
                    )
                    changes = {}
                    for table in ("project_resources",):
                        before_counts = _counts_by_year(
                            baseline_counts.get(table, [])
                        )
                        after_counts = _counts_by_year(source_counts)
                        changes[table] = {
                            str(year): {
                                "before": before_counts.get(year, 0),
                                "after": after_counts.get(year, 0),
                                "delta": after_counts.get(year, 0)
                                - before_counts.get(year, 0),
                            }
                            for year in sorted(
                                set(before_counts) | set(after_counts),
                                key=lambda value: "" if value is None else str(value),
                            )
                        }
                    _set_stage(observations, "source_change_summary_by_year", changes)
                    _set_stage(observations, "api_calls", list(api_calls))
                    timeseries_request_years = [
                        year
                        for table, year, _filetype in fixtures
                        if table == "project_timeseries"
                    ]
                    _set_stage(
                        observations,
                        "requested_and_scoped_years",
                        {
                            "requested_resource_years": [None],
                            "resource_years_returned": sorted(
                                _fixture_resource_years(
                                    resource_path, str(resource_fixture["filetype"])
                                )
                            ),
                            "timeseries_fixture_years": sorted(
                                timeseries_request_years,
                                key=lambda year: -1 if year is None else year,
                            ),
                        },
                    )
                    if len(api_calls) < 2:
                        raise AssertionError(
                            "bootstrap did not request resources and "
                            "timeseries fixtures"
                        )

                bootstrap_result = _record(
                    stages, "unfiltered_cli_bootstrap", unfiltered_bootstrap
                )
                if bootstrap_result is None and stages[-1]["status"] != "PASS":
                    save_report()
                    return report_path

                def parity_and_observations(observations: object) -> None:
                    import csv
                    import sqlite3

                    import duckdb

                    if str(resource_fixture["filetype"]) == "json":
                        fixture_rows = json.loads(
                            resource_path.read_text(encoding="utf-8")
                        )
                    else:
                        with resource_path.open(
                            encoding="utf-8-sig", newline=""
                        ) as stream:
                            fixture_rows = list(csv.DictReader(stream, delimiter=";"))
                    search_value = next(
                        (
                            str(row.get(name))
                            for row in fixture_rows
                            for name in (
                                "project_name",
                                "project_remarks",
                                "project_id",
                            )
                            if row.get(name) not in (None, "")
                        ),
                        None,
                    )
                    if search_value is None:
                        raise AssertionError("fixture has no searchable project value")
                    search_token = search_value.split()[0].replace("'", "''")

                    conn = duckdb.connect(str(db_dir / "esdc.duckdb"), read_only=True)
                    try:
                        parity = compare_resource_projection(
                            conn, db_dir / "esdc.sqlite"
                        )
                        _set_stage(
                            observations,
                            "parity",
                            {
                                "canonical_rows": parity.canonical_rows,
                                "projected_rows": parity.projected_rows,
                                "identity_difference_count": (
                                    parity.identity_difference_count
                                ),
                                "source_multiset_difference_count": (
                                    parity.source_multiset_difference_count
                                ),
                                "metadata_matches": parity.metadata_matches,
                                "invalid_geometry_count": parity.invalid_geometry_count,
                                "issues": list(parity.issues),
                            },
                        )
                        if not parity.ok:
                            raise AssertionError(
                                "resource source and lineage parity failed: "
                                + "; ".join(parity.issues)
                            )
                        state_after_bootstrap = _logical_state(db_dir / "esdc.duckdb")
                        initial_protected = baseline_preserved_tables
                        current_protected = cast(
                            dict[str, object],
                            _logical_state(
                                db_dir / "esdc.duckdb", ignore_recomputed_metadata=True
                            )["protected_tables"],
                        )
                        current_protected_full = cast(
                            dict[str, object], state_after_bootstrap["protected_tables"]
                        )
                        protected_unchanged = all(
                            current_protected.get(name) == value
                            for name, value in initial_protected.items()
                        )
                        timeseries_untouched = _untouched_timeseries_digest(
                            db_dir / "esdc.duckdb",
                            fixture_years,
                            baseline_timeseries_columns,
                        )
                        empty_timeseries_after = _timeseries_year_digest(
                            db_dir / "esdc.duckdb",
                            empty_timeseries_years,
                            baseline_timeseries_columns,
                        )
                        if (
                            not protected_unchanged
                            or timeseries_untouched != baseline_untouched_timeseries
                            or empty_timeseries_after != baseline_empty_timeseries
                        ):
                            raise AssertionError(
                                "bootstrap changed preexisting non-resource tables or "
                                "unrequested/NULL/empty-response timeseries"
                            )
                        canonical = sqlite3.connect(
                            f"file:{(db_dir / 'esdc.sqlite').resolve()}?mode=ro",
                            uri=True,
                        )
                        try:
                            source_year_counts = canonical.execute(
                                "SELECT report_year, COUNT(*) FROM "
                                "api_project_resources "
                                "GROUP BY report_year ORDER BY report_year"
                            ).fetchall()
                        finally:
                            canonical.close()
                        analytical_year_counts = conn.execute(
                            "SELECT report_year, COUNT(*) FROM project_resources "
                            "GROUP BY report_year ORDER BY report_year"
                        ).fetchall()
                        _set_stage(
                            observations,
                            "source_and_analytical_counts_by_year",
                            {
                                "canonical": [
                                    [year, count] for year, count in source_year_counts
                                ],
                                "analytical": [
                                    [year, count]
                                    for year, count in analytical_year_counts
                                ],
                            },
                        )
                        views = conn.execute(
                            "SELECT table_name FROM information_schema.views "
                            "WHERE table_schema='main' ORDER BY table_name"
                        ).fetchall()
                        resource_views = [
                            row[0]
                            for row in views
                            if row[0].endswith(("_resources", "_timeseries"))
                        ]
                        view_counts: dict[str, int] = {}
                        for name in resource_views:
                            count_row = conn.execute(
                                f"SELECT COUNT(*) FROM {_quote_ident(name)}"
                            ).fetchone()
                            if count_row is None:
                                raise AssertionError(
                                    f"view count query returned no row: {name}"
                                )
                            view_counts[name] = int(count_row[0])
                        _set_stage(observations, "view_row_counts", view_counts)
                        view_hashes = {
                            name: _logical_table_digest(
                                conn, name, exclude_columns={"uuid"}
                            )
                            for name in resource_views
                        }
                        _set_stage(
                            observations,
                            "representative_resource_view_results",
                            view_hashes,
                        )
                        resource_columns = {
                            row[0]
                            for row in conn.execute(
                                "SELECT column_name FROM duckdb_columns() "
                                "WHERE table_name='project_resources'"
                            ).fetchall()
                        }
                        _set_stage(
                            observations,
                            "spatial_query",
                            {
                                "resource_geometry_columns_present": bool(
                                    {"field_geom", "field_geometry"}
                                    & resource_columns
                                ),
                            },
                        )
                        hits_row = conn.execute(
                            "SELECT COUNT(*) FROM project_resources "
                            "WHERE fts_main_project_resources."
                            "match_bm25(uuid, ?) IS NOT NULL",
                            [search_token],
                        ).fetchone()
                        if hits_row is None:
                            raise AssertionError("FTS query returned no count")
                        search_hits = int(hits_row[0])
                        _set_stage(
                            observations,
                            "fts_search",
                            {
                                "query_sha256": hashlib.sha256(
                                    search_token.encode()
                                ).hexdigest(),
                                "hits": search_hits,
                            },
                        )
                        if search_hits < 1:
                            raise AssertionError(
                                "FTS search returned no fixture-derived match"
                            )
                        _set_stage(
                            observations,
                            "preserved_non_resource_table_hashes",
                            current_protected_full,
                        )
                        _set_stage(
                            observations,
                            "unrequested_and_null_year_timeseries",
                            timeseries_untouched,
                        )
                        _set_stage(
                            observations,
                            "valid_empty_timeseries_preservation",
                            {
                                "years": sorted(empty_timeseries_years),
                                "before": baseline_empty_timeseries,
                                "after": empty_timeseries_after,
                            },
                        )
                    finally:
                        conn.close()
                    source = sqlite3.connect(
                        f"file:{(db_dir / 'esdc.sqlite').resolve()}?mode=ro", uri=True
                    )
                    try:
                        raw = source.execute(
                            "SELECT project_id, field_geom, groovy_report, rec_oil "
                            "FROM api_project_resources ORDER BY project_id"
                        ).fetchall()
                        _set_stage(
                            observations,
                            "raw_source_value_hash",
                            hashlib.sha256(repr(raw).encode()).hexdigest(),
                        )
                        _set_stage(observations, "raw_source_row_count", len(raw))
                        _set_stage(
                            observations,
                            "grov_raw_storage_excluded",
                            not any(
                                name.startswith("grov_")
                                for name in {
                                    row[1]
                                    for row in source.execute(
                                        "PRAGMA table_info(api_project_resources)"
                                    ).fetchall()
                                }
                            ),
                        )
                    finally:
                        source.close()

                _record(
                    stages,
                    "typed_parity_views_spatial_search_and_protected_rows",
                    parity_and_observations,
                )
                if stages[-1]["status"] != "PASS":
                    save_report()
                    return report_path

                def null_year_rebuild_setup(observations: object) -> None:
                    import sqlite3

                    from esdc.api_snapshot import (
                        ApiProjectResourceRow,
                        ApiSnapshot,
                        iter_project_resources_rows,
                        replace_project_resources_snapshot,
                    )
                    from esdc.pod_registry.store import get_sqlite_connection

                    path = db_dir / "esdc.sqlite"
                    conn = get_sqlite_connection(path)
                    try:
                        rows = list(iter_project_resources_rows(conn))
                        existing_null_count = sum(
                            row.report_year is None for row in rows
                        )
                        rows.append(
                            ApiProjectResourceRow(
                                project_id=f"task6b-null-year-{uuid.uuid4().hex}",
                                report_year=None,
                                project_name="Local NULL-year preservation sentinel",
                            )
                        )
                        replace_project_resources_snapshot(
                            conn,
                            rows,
                            ApiSnapshot(
                                "project_resources",
                                "local-null-year-seed",
                                len(rows),
                                "",
                                "2026-10-23T00:00:00+00:00",
                                "committed",
                                1,
                            ),
                        )
                    finally:
                        conn.close()
                    calls_before = len(api_calls)
                    publication = cli_module.rebuild_project_resources_projection()
                    conn = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
                    try:
                        null_after = int(
                            conn.execute(
                                "SELECT COUNT(*) FROM api_project_resources "
                                "WHERE report_year IS NULL"
                            ).fetchone()[0]
                        )
                    finally:
                        conn.close()
                    if (
                        null_after != existing_null_count + 1
                        or len(api_calls) != calls_before
                    ):
                        raise AssertionError(
                            "NULL-year canonical baseline setup/rebuild did not "
                            "preserve API-free state"
                        )
                    _set_stage(
                        observations,
                        "explicit_local_setup",
                        "Added a synthetic NULL-year canonical resource using "
                        "repository APIs after fixture bootstrap, then rebuilt "
                        "resources without API calls.",
                    )
                    _set_stage(
                        observations,
                        "null_year_count_before_after",
                        [existing_null_count, null_after],
                    )
                    _set_stage(observations, "rebuild_revision", publication.revision)
                    _set_stage(
                        observations,
                        "api_calls_during_setup_rebuild",
                        len(api_calls) - calls_before,
                    )

                _record(
                    stages,
                    "null_year_canonical_preservation_setup",
                    null_year_rebuild_setup,
                )
                if stages[-1]["status"] != "PASS":
                    save_report()
                    return report_path

                def selective_refresh(observations: object) -> None:
                    before = _snapshot(db_dir / "esdc.sqlite")
                    resources_preservation_before = _sqlite_unselected_resources_digest(
                        db_dir / "esdc.sqlite", selected_year
                    )
                    timeseries_preservation_before = _untouched_timeseries_digest(
                        db_dir / "esdc.duckdb",
                        {selected_year},
                        baseline_timeseries_columns,
                    )
                    ts_before = _timeseries_year_digest(
                        db_dir / "esdc.duckdb",
                        empty_timeseries_years,
                        baseline_timeseries_columns,
                    )
                    import sqlite3

                    source = sqlite3.connect(
                        f"file:{(db_dir / 'esdc.sqlite').resolve()}?mode=ro", uri=True
                    )
                    try:
                        before_years = source.execute(
                            "SELECT report_year, COUNT(*) FROM api_project_resources "
                            "WHERE report_year IS NULL OR report_year != ? "
                            "GROUP BY report_year ORDER BY report_year",
                            (selected_year,),
                        ).fetchall()
                    finally:
                        source.close()
                    code, output_text = invoke_fetch(
                        ["fetch", "--year", str(selected_year)]
                    )
                    if code != 0:
                        raise AssertionError(
                            f"selective refresh failed: {output_text.strip()}"
                        )
                    after = _snapshot(db_dir / "esdc.sqlite")
                    resources_preservation_after = _sqlite_unselected_resources_digest(
                        db_dir / "esdc.sqlite", selected_year
                    )
                    timeseries_preservation_after = _untouched_timeseries_digest(
                        db_dir / "esdc.duckdb",
                        {selected_year},
                        baseline_timeseries_columns,
                    )
                    ts_after = _timeseries_year_digest(
                        db_dir / "esdc.duckdb",
                        empty_timeseries_years,
                        baseline_timeseries_columns,
                    )
                    source = sqlite3.connect(
                        f"file:{(db_dir / 'esdc.sqlite').resolve()}?mode=ro", uri=True
                    )
                    try:
                        years = source.execute(
                            "SELECT report_year, COUNT(*) FROM api_project_resources "
                            "GROUP BY report_year ORDER BY report_year"
                        ).fetchall()
                        selected_count = int(
                            source.execute(
                                "SELECT COUNT(*) FROM api_project_resources "
                                "WHERE report_year=?",
                                (selected_year,),
                            ).fetchone()[0]
                        )
                    finally:
                        source.close()
                    if str(resource_fixture["filetype"]) == "json":
                        selected_rows = json.loads(
                            resource_path.read_text(encoding="utf-8")
                        )
                    else:
                        with resource_path.open(
                            encoding="utf-8-sig", newline=""
                        ) as stream:
                            selected_rows = list(csv.DictReader(stream, delimiter=";"))
                    fixture_selected_count = sum(
                        str(row.get("report_year", "")).strip() == str(selected_year)
                        for row in selected_rows
                    )
                    _set_stage(observations, "before", before)
                    _set_stage(observations, "after", after)
                    _set_stage(observations, "resource_rows_by_year", years)
                    _set_stage(
                        observations, "unselected_year_counts_before", before_years
                    )
                    _set_stage(
                        observations,
                        "selected_year_expected_actual_count",
                        [fixture_selected_count, selected_count],
                    )
                    null_after = sum(count for year, count in years if year is None)
                    null_before = sum(
                        count for year, count in before_years if year is None
                    )
                    _set_stage(
                        observations,
                        "null_year_count_before_after",
                        [null_before, null_after],
                    )
                    _set_stage(
                        observations,
                        "empty_timeseries_rows_before_after",
                        {"before": ts_before, "after": ts_after},
                    )
                    _set_stage(
                        observations,
                        "unselected_canonical_resource_values_before_after",
                        {
                            "before": resources_preservation_before,
                            "after": resources_preservation_after,
                        },
                    )
                    _set_stage(
                        observations,
                        "unselected_and_null_timeseries_values_before_after",
                        {
                            "before": timeseries_preservation_before,
                            "after": timeseries_preservation_after,
                        },
                    )
                    if (
                        any((year, count) not in years for year, count in before_years)
                        or null_after != null_before
                        or selected_count != fixture_selected_count
                        or ts_before != ts_after
                        or resources_preservation_before != resources_preservation_after
                        or timeseries_preservation_before
                        != timeseries_preservation_after
                    ):
                        raise AssertionError(
                            "selective update did not preserve exact unselected/NULL "
                            "resource and timeseries values"
                        )

                _record(stages, "selective_year_refresh", selective_refresh)
                if stages[-1]["status"] != "PASS":
                    save_report()
                    return report_path

                def empty_and_download_failures(observations: object) -> None:
                    nonlocal use_empty_resources, fail_download
                    before = _snapshot(db_dir / "esdc.sqlite")
                    before_pair = _logical_state(db_dir / "esdc.duckdb")
                    from esdc.chat.tools import (
                        get_sql_cache_stats,
                        get_tool_cache_stats,
                    )

                    before_caches = {
                        "sql": get_sql_cache_stats(),
                        "tools": get_tool_cache_stats(),
                    }
                    calls_before = len(api_calls)
                    use_empty_resources = True
                    empty_code, empty_output = invoke_fetch(["fetch"])
                    use_empty_resources = False
                    empty_after = _snapshot(db_dir / "esdc.sqlite")
                    after_pair = _logical_state(db_dir / "esdc.duckdb")
                    after_caches = {
                        "sql": get_sql_cache_stats(),
                        "tools": get_tool_cache_stats(),
                    }
                    changed_sections = [
                        key
                        for key in before_pair
                        if before_pair.get(key) != after_pair.get(key)
                    ]
                    before_view_states = cast(
                        dict[str, object],
                        before_pair.get("representative_view_states", {}),
                    )
                    after_view_states = cast(
                        dict[str, object],
                        after_pair.get("representative_view_states", {}),
                    )
                    changed_views = [
                        name
                        for name, value in before_view_states.items()
                        if after_view_states.get(name) != value
                    ]
                    empty_call_delta = api_calls[calls_before:]
                    failed_calls_before = len(api_calls)
                    fail_download = True
                    try:
                        download_code, download_output = invoke_fetch(["fetch"])
                    finally:
                        fail_download = False
                    failed_snapshot = _snapshot(db_dir / "esdc.sqlite")
                    failed_analytical = _logical_state(db_dir / "esdc.duckdb")
                    failed_caches = {
                        "sql": get_sql_cache_stats(),
                        "tools": get_tool_cache_stats(),
                    }
                    failed_call_delta = api_calls[failed_calls_before:]
                    empty_ok = (
                        empty_code == 0
                        and before == empty_after
                        and before_pair == after_pair
                        and before_caches == after_caches
                        and len(empty_call_delta) == 1
                        and empty_call_delta[0]["table"] == "project_resources"
                    )
                    _set_stage(
                        observations,
                        "empty_response",
                        {
                            "exit_code": empty_code,
                            "snapshot_unchanged": before == empty_after,
                            "analytical_state_unchanged": before_pair == after_pair,
                            "analytical_state_before_sha256": hashlib.sha256(
                                json.dumps(before_pair, sort_keys=True).encode()
                            ).hexdigest(),
                            "analytical_state_after_sha256": hashlib.sha256(
                                json.dumps(after_pair, sort_keys=True).encode()
                            ).hexdigest(),
                            "changed_logical_sections": changed_sections,
                            "changed_application_views": changed_views,
                            "cache_state_unchanged": before_caches == after_caches,
                            "api_calls": empty_call_delta,
                            "output": empty_output.strip(),
                        },
                    )
                    _set_stage(
                        observations,
                        "download_failure",
                        {
                            "exit_code": download_code,
                            "output": download_output.strip(),
                            "snapshot_unchanged": before == failed_snapshot,
                            "analytical_state_unchanged": before_pair
                            == failed_analytical,
                            "analytical_state_before_sha256": hashlib.sha256(
                                json.dumps(before_pair, sort_keys=True).encode()
                            ).hexdigest(),
                            "analytical_state_after_sha256": hashlib.sha256(
                                json.dumps(failed_analytical, sort_keys=True).encode()
                            ).hexdigest(),
                            "cache_state_unchanged": before_caches == failed_caches,
                            "api_calls": failed_call_delta,
                        },
                    )
                    if (
                        empty_code != 0
                        or before != empty_after
                        or before_pair != after_pair
                        or before_caches != after_caches
                        or len(empty_call_delta) != 1
                        or empty_call_delta[0]["table"] != "project_resources"
                        or download_code == 0
                        or before != failed_snapshot
                        or before_pair != failed_analytical
                        or before_caches != failed_caches
                        or len(failed_call_delta) != 1
                        or failed_call_delta[0]["table"] != "project_resources"
                    ):
                        raise AssertionError(
                            "default empty response or download-failure contract "
                            f"failed (empty_ok={empty_ok}, "
                            f"download_failed={download_code != 0})"
                        )

                _record(
                    stages,
                    "empty_and_download_failure_injections",
                    empty_and_download_failures,
                )
                if stages[-1]["status"] != "PASS":
                    save_report()
                    return report_path

                def c1_recovery(observations: object) -> None:
                    before_analytical = _logical_state(db_dir / "esdc.duckdb")
                    before_untouched_ts = _untouched_timeseries_digest(
                        db_dir / "esdc.duckdb",
                        fixture_years,
                        baseline_timeseries_columns,
                    )
                    original_swap = publication_module._swap_project_resources

                    def fail_swap(*args, **kwargs):
                        raise RuntimeError("injected C2 publication failure after C1")

                    publication_module._swap_project_resources = fail_swap
                    try:
                        code, output_text = invoke_fetch(["fetch"])
                    finally:
                        publication_module._swap_project_resources = original_swap
                    stale = _snapshot(db_dir / "esdc.sqlite")
                    after_failed_analytical = _logical_state(db_dir / "esdc.duckdb")
                    after_failed_untouched_ts = _untouched_timeseries_digest(
                        db_dir / "esdc.duckdb",
                        fixture_years,
                        baseline_timeseries_columns,
                    )
                    _set_stage(
                        observations,
                        "failed_fetch",
                        {"exit_code": code, "output": output_text.strip()},
                    )
                    _set_stage(observations, "stale_snapshot", stale)
                    _set_stage(
                        observations,
                        "analytical_pair_after_failed_c2",
                        after_failed_analytical,
                    )
                    if (
                        code == 0
                        or "phase=publication" not in output_text
                        or stale["publication_state"] != "analytics_stale"
                    ):
                        raise AssertionError(
                            "C2 failure did not retain and mark C1 snapshot stale: "
                            f"{output_text.strip()}"
                        )
                    if (
                        after_failed_analytical != before_analytical
                        or after_failed_untouched_ts != before_untouched_ts
                    ):
                        raise AssertionError(
                            "failed C2 publication changed analytical baseline"
                        )
                    before_rebuild_calls = len(api_calls)
                    source_before_rebuild = _api_pair_source_state(
                        db_dir / "esdc.sqlite"
                    )
                    _set_stage(
                        observations,
                        "canonical_api_pair_sources_before_rebuild",
                        source_before_rebuild,
                    )
                    resource_source = cast(
                        dict[str, object],
                        source_before_rebuild["project_resources"],
                    )
                    timeseries_source = cast(
                        dict[str, object],
                        source_before_rebuild["project_timeseries"],
                    )
                    if resource_source is None:
                        raise AssertionError(
                            "C1 failure did not retain a resource source snapshot"
                        )
                    import esdc.chat.tools as chat_tools
                    from esdc.analytics_rebuild import rebuild_api_analytics
                    from esdc.chat.tools import (
                        get_sql_cache_stats,
                        get_tool_cache_stats,
                    )

                    sql_cache = chat_tools._get_cache()
                    tool_cache = chat_tools._get_tool_cache()
                    sql_cache.set("t6-c1-recovery-probe", "invalidate after rebuild")
                    tool_cache.set("t6-c1-recovery-probe", "invalidate after rebuild")
                    before_caches = {
                        "sql": get_sql_cache_stats(),
                        "tools": get_tool_cache_stats(),
                    }
                    _set_stage(
                        observations,
                        "analytical_caches_before_rebuild",
                        before_caches,
                    )
                    sql_cache_entries = cast(
                        dict[str, int], before_caches["sql"]
                    ).get("entries", 0)
                    tool_cache_entries = cast(
                        dict[str, int], before_caches["tools"]
                    ).get("entries", 0)
                    if sql_cache_entries < 1 or tool_cache_entries < 1:
                        raise AssertionError("cache recovery probes were not seeded")
                    before_protected_state = _logical_state(
                        db_dir / "esdc.duckdb", ignore_recomputed_metadata=True
                    )
                    nonlocal fail_download
                    fail_download = True
                    try:
                        rebuilt = rebuild_api_analytics(
                            duckdb_path=db_dir / "esdc.duckdb",
                            sqlite_path=db_dir / "esdc.sqlite",
                        )
                    finally:
                        fail_download = False
                    published = _snapshot(db_dir / "esdc.sqlite")
                    source_after_rebuild = _api_pair_source_state(
                        db_dir / "esdc.sqlite"
                    )
                    _set_stage(
                        observations,
                        "canonical_api_pair_sources_after_rebuild",
                        source_after_rebuild,
                    )
                    after_caches = {
                        "sql": get_sql_cache_stats(),
                        "tools": get_tool_cache_stats(),
                    }
                    _set_stage(
                        observations,
                        "analytical_caches_after_rebuild",
                        after_caches,
                    )
                    after_rebuild_analytical = _logical_state(
                        db_dir / "esdc.duckdb", ignore_recomputed_metadata=True
                    )
                    _set_stage(
                        observations,
                        "logical_state_after_rebuild",
                        after_rebuild_analytical,
                    )
                    resource_source_after = cast(
                        dict[str, object],
                        source_after_rebuild["project_resources"],
                    )
                    timeseries_source_after = cast(
                        dict[str, object],
                        source_after_rebuild["project_timeseries"],
                    )
                    expected_revisions = {
                        "project_resources": resource_source["revision"],
                        "project_timeseries": timeseries_source["revision"],
                    }
                    expected_hashes = {
                        "project_resources": resource_source["source_sha256"],
                        "project_timeseries": timeseries_source["source_sha256"],
                    }
                    if (
                        rebuilt.source_revisions != expected_revisions
                        or rebuilt.source_hashes != expected_hashes
                        or resource_source_after["revision"]
                        != resource_source["revision"]
                        or resource_source_after["source_sha256"]
                        != resource_source["source_sha256"]
                        or timeseries_source_after["revision"]
                        != timeseries_source["revision"]
                        or timeseries_source_after["source_sha256"]
                        != timeseries_source["source_sha256"]
                        or published["revision"] != resource_source["revision"]
                        or published["source_sha256"]
                        != resource_source["source_sha256"]
                        or resource_source_after["publication_state"] != "published"
                        or timeseries_source_after["publication_state"] != "published"
                    ):
                        raise AssertionError(
                            "API-free pair rebuild did not publish both retained "
                            "canonical revisions and hashes"
                        )
                    protected_before = before_protected_state["protected_tables"]
                    protected_after = after_rebuild_analytical["protected_tables"]
                    untouched_after_rebuild = _untouched_timeseries_digest(
                        db_dir / "esdc.duckdb",
                        fixture_years,
                        baseline_timeseries_columns,
                    )
                    timeseries_before_rebuild = cast(
                        dict[str, object], before_analytical["managed_table_states"]
                    ).get("project_timeseries")
                    timeseries_after_rebuild = cast(
                        dict[str, object],
                        after_rebuild_analytical["managed_table_states"],
                    ).get("project_timeseries")
                    if protected_before != protected_after:
                        raise AssertionError(
                            "API pair rebuild changed protected DuckDB data"
                        )
                    import duckdb

                    with duckdb.connect(
                        str(db_dir / "esdc.duckdb"), read_only=True
                    ) as conn:
                        resource_parity = compare_resource_projection(
                            conn, db_dir / "esdc.sqlite"
                        )
                        source_conn = sqlite3.connect(
                            f"file:{(db_dir / 'esdc.sqlite').resolve()}?mode=ro",
                            uri=True,
                        )
                        try:
                            source_search_values = source_conn.execute(
                                "SELECT project_name, project_remarks, project_id "
                                "FROM api_project_resources ORDER BY project_id"
                            ).fetchall()
                        finally:
                            source_conn.close()
                        search_value = next(
                            (
                                str(value)
                                for row in source_search_values
                                for value in row
                                if value not in (None, "")
                            ),
                            None,
                        )
                        if search_value is None:
                            raise AssertionError(
                                "resource recovery fixture has no searchable value"
                            )
                        search_tokens = search_value.split()
                        if not search_tokens:
                            raise AssertionError(
                                "resource recovery fixture search value is blank"
                            )
                        search_token = search_tokens[0]
                        fts_row = conn.execute(
                            "SELECT COUNT(*) FROM project_resources WHERE "
                            "fts_main_project_resources.match_bm25(uuid, ?) "
                            "IS NOT NULL",
                            [search_token],
                        ).fetchone()
                        if fts_row is None:
                            raise AssertionError(
                                "recovery FTS query returned no count"
                            )
                        fts_after_rebuild = {
                            "query": search_token,
                            "hits": int(fts_row[0]),
                        }
                    from esdc.analytics_audit import compare_timeseries_projection

                    timeseries_parity = compare_timeseries_projection(
                        sqlite_path=db_dir / "esdc.sqlite",
                        duckdb_path=db_dir / "esdc.duckdb",
                    )
                    if not resource_parity.ok or not timeseries_parity["ok"]:
                        raise AssertionError(
                            "working-clone typed API pair parity failed after "
                            "offline recovery rebuild"
                        )
                    if fts_after_rebuild["hits"] < 1:
                        raise AssertionError(
                            "API-free pair rebuild did not restore fixture-derived FTS"
                        )
                    if (
                        len(api_calls) != before_rebuild_calls
                        or chat_tools._sql_cache is not None
                        or chat_tools._tool_cache is not None
                        or cast(dict[str, object], after_caches["sql"]).get(
                            "entries", 0
                        )
                        != 0
                        or cast(dict[str, object], after_caches["tools"]).get(
                            "entries", 0
                        )
                        != 0
                    ):
                        raise AssertionError(
                            "API-free pair rebuild made a request or failed to "
                            "invalidate seeded analytical caches"
                        )
                    _set_stage(
                        observations,
                        "failed_fetch",
                        {"exit_code": code, "output": output_text.strip()},
                    )
                    _set_stage(observations, "stale_snapshot", stale)
                    _set_stage(
                        observations,
                        "canonical_api_pair_sources_before_rebuild",
                        source_before_rebuild,
                    )
                    _set_stage(
                        observations,
                        "canonical_api_pair_sources_after_rebuild",
                        source_after_rebuild,
                    )
                    _set_stage(
                        observations,
                        "pair_rebuild_report",
                        {
                            "source_revisions": rebuilt.source_revisions,
                            "source_hashes": rebuilt.source_hashes,
                            "object_counts": rebuilt.object_counts,
                        },
                    )
                    _set_stage(
                        observations,
                        "analytical_caches_before_after_rebuild",
                        {"before": before_caches, "after": after_caches},
                    )
                    _set_stage(
                        observations,
                        "old_analytical_pair_and_lineage_logical_state",
                        before_analytical,
                    )
                    _set_stage(
                        observations,
                        "analytical_pair_and_lineage_after_failed_c2",
                        after_failed_analytical,
                    )
                    _set_stage(
                        observations,
                        "old_analytical_state_sha256",
                        hashlib.sha256(
                            json.dumps(before_analytical, sort_keys=True).encode()
                        ).hexdigest(),
                    )
                    _set_stage(
                        observations,
                        "analytical_state_after_failed_c2_sha256",
                        hashlib.sha256(
                            json.dumps(after_failed_analytical, sort_keys=True).encode()
                        ).hexdigest(),
                    )
                    _set_stage(
                        observations,
                        "rebuild_publication",
                        {
                            "revision": rebuilt.source_revisions[
                                "project_resources"
                            ],
                            "row_count": rebuilt.object_counts["project_resources"],
                            "api_calls_during_rebuild": len(api_calls)
                            - before_rebuild_calls,
                        },
                    )
                    _set_stage(
                        observations, "fts_search_after_rebuild", fts_after_rebuild
                    )
                    _set_stage(
                        observations,
                        "protected_state_unchanged_by_rebuild",
                        {
                            "tables": protected_after,
                            "full_timeseries_table_before": timeseries_before_rebuild,
                            "full_timeseries_table_after": timeseries_after_rebuild,
                            "legacy_untouched_digest_before": before_untouched_ts,
                            "legacy_untouched_digest_after": untouched_after_rebuild,
                            "unrequested_and_null_year_timeseries": {
                                "audit_ok": timeseries_parity["ok"],
                                "canonical_rows": timeseries_parity[
                                    "canonical_rows"
                                ],
                                "projected_rows": timeseries_parity[
                                    "projected_rows"
                                ],
                                "value_difference_count": timeseries_parity[
                                    "value_difference_count"
                                ],
                            },
                        },
                    )
                    _set_stage(
                        observations,
                        "working_clone_parity_after_rebuild",
                        {
                            "resources": {
                                "ok": resource_parity.ok,
                                "issues": list(resource_parity.issues),
                            },
                            "timeseries": timeseries_parity,
                        },
                    )

                _record(stages, "post_c1_failure_and_api_free_rebuild", c1_recovery)
                if stages[-1]["status"] != "PASS":
                    save_report()
                    return report_path

                def separate_empty_target(observations: object) -> None:
                    import duckdb

                    allowed_target_bookkeeping = {
                        "_metadata",
                        "api_publication",
                        "resource_summaries",
                    }
                    forbidden_owned_data = {
                        "document_chunks",
                        "project_embeddings",
                    }
                    empty_target_dir = run_dir / "separate-target"
                    empty_target_dir.mkdir()
                    target = empty_target_dir / "esdc.duckdb"
                    os.environ["ESDC_DB_FILE"] = str(target)
                    before = _snapshot(db_dir / "esdc.sqlite")
                    pair_sources_before = _api_pair_source_state(db_dir / "esdc.sqlite")
                    calls_before = len(api_calls)
                    nonlocal fail_download
                    fail_download = True
                    try:
                        from esdc.analytics_rebuild import rebuild_api_analytics

                        first = rebuild_api_analytics(
                            duckdb_path=target,
                            sqlite_path=db_dir / "esdc.sqlite",
                        )
                        second = rebuild_api_analytics(
                            duckdb_path=target,
                            sqlite_path=db_dir / "esdc.sqlite",
                        )
                    finally:
                        fail_download = False
                    after = _snapshot(db_dir / "esdc.sqlite")
                    pair_sources_after = _api_pair_source_state(
                        db_dir / "esdc.sqlite"
                    )
                    with duckdb.connect(str(target), read_only=True) as conn:
                        parity = compare_resource_projection(
                            conn, db_dir / "esdc.sqlite"
                        )
                        target_table_names = {
                            row[0]
                            for row in conn.execute(
                                "SELECT table_name FROM information_schema.tables "
                                "WHERE table_schema='main' AND table_type='BASE TABLE'"
                            ).fetchall()
                        }
                    from esdc.analytics_audit import compare_timeseries_projection

                    timeseries_parity = compare_timeseries_projection(
                        sqlite_path=db_dir / "esdc.sqlite",
                        duckdb_path=target,
                    )
                    source_revisions = {
                        "project_resources": before["revision"],
                        "project_timeseries": cast(
                            dict[str, object],
                            pair_sources_before["project_timeseries"],
                        )["revision"],
                    }
                    source_hashes = {
                        "project_resources": before["source_sha256"],
                        "project_timeseries": cast(
                            dict[str, object],
                            pair_sources_before["project_timeseries"],
                        )["source_sha256"],
                    }
                    baseline_protected_names = set(
                        cast(
                            dict[str, object],
                            baseline_logical_state["protected_tables"],
                        )
                    )
                    unexpected_protected_tables = sorted(
                        target_table_names
                        & (baseline_protected_names - allowed_target_bookkeeping)
                    )
                    forbidden_tables_present = sorted(
                        target_table_names & forbidden_owned_data
                    )
                    _set_stage(observations, "target", str(target))
                    _set_stage(
                        observations,
                        "first_revision",
                        first.source_revisions["project_resources"],
                    )
                    _set_stage(
                        observations,
                        "second_revision",
                        second.source_revisions["project_resources"],
                    )
                    _set_stage(
                        observations,
                        "rebuild_source_revisions",
                        first.source_revisions,
                    )
                    _set_stage(
                        observations, "rebuild_source_hashes", first.source_hashes
                    )
                    _set_stage(observations, "source_before_after", [before, after])
                    _set_stage(
                        observations,
                        "canonical_pair_sources_before_after",
                        {"before": pair_sources_before, "after": pair_sources_after},
                    )
                    _set_stage(
                        observations,
                        "parity",
                        {"ok": parity.ok, "issues": list(parity.issues)},
                    )
                    _set_stage(observations, "timeseries_parity", timeseries_parity)
                    _set_stage(
                        observations,
                        "api_calls_during_rebuilds",
                        len(api_calls) - calls_before,
                    )
                    _set_stage(
                        observations, "target_table_names", sorted(target_table_names)
                    )
                    _set_stage(
                        observations,
                        "baseline_protected_table_names",
                        sorted(baseline_protected_names),
                    )
                    _set_stage(
                        observations,
                        "unexpected_baseline_protected_tables_restored",
                        unexpected_protected_tables,
                    )
                    _set_stage(
                        observations,
                        "forbidden_duckdb_owned_data_tables_present",
                        forbidden_tables_present,
                    )
                    if (
                        not parity.ok
                        or not timeseries_parity["ok"]
                        or before != after
                        or pair_sources_before != pair_sources_after
                        or first.source_revisions != source_revisions
                        or second.source_revisions != source_revisions
                        or first.source_hashes != source_hashes
                        or second.source_hashes != source_hashes
                        or first.object_counts != second.object_counts
                        or len(api_calls) != calls_before
                        or unexpected_protected_tables
                        or forbidden_tables_present
                    ):
                        raise AssertionError(
                            "separate empty-target rebuild/repetition did not "
                            "preserve canonical parity and revision"
                        )

                _record(
                    stages,
                    "separate_empty_target_rebuild_and_repeat",
                    separate_empty_target,
                )
        finally:
            cli_module._download_table_bytes = original_downloader
            runner_caches = (chat_tools._sql_cache, chat_tools._tool_cache)
            caches_to_close = (local_sql_cache, local_tool_cache, *runner_caches)
            closed_cache_ids: set[int] = set()
            for cache in caches_to_close:
                if cache is not None and id(cache) not in closed_cache_ids:
                    cache.close()
                    closed_cache_ids.add(id(cache))
            runner_handlers = [
                handler
                for handler in logging.root.handlers
                if handler not in previous_handlers
            ]
            logging.root.handlers = previous_handlers
            logging.root.setLevel(previous_log_level)
            for handler in runner_handlers:
                handler.close()
            chat_tools._sql_cache = previous_sql_cache
            chat_tools._tool_cache = previous_tool_cache
            Config._config_cache = previous_config_cache

    save_report()
    return report_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", required=True, type=Path)
    parser.add_argument("--fixture-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        report_path = run_rehearsal(
            baseline_dir=args.baseline_dir,
            fixture_manifest=args.fixture_manifest,
            output_dir=args.output_dir,
        )
    except Exception as exc:
        print(f"Migration rehearsal rejected or unavailable: {exc}", file=sys.stderr)
        return 1
    report = json.loads(report_path.read_text(encoding="utf-8"))
    passed = sum(stage["status"] == "PASS" for stage in report["stages"])
    failed = sum(stage["status"] != "PASS" for stage in report["stages"])
    print(f"{report_path} ({passed} passed, {failed} failed)")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
