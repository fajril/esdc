"""Closed-clone, offline rehearsal for SQLite-backed analytics migration."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import sqlite3
import sys
import time
from collections.abc import Iterator
from contextlib import closing, contextmanager
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import duckdb


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _read_manifest(
    path: Path,
) -> tuple[dict[tuple[str, str, int | None], Path], dict[str, object]]:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid fixture manifest: {exc}") from exc
    if not isinstance(manifest, dict) or manifest.get("version") != 1:
        raise ValueError("fixture manifest version must be 1")
    if manifest.get("filetype") != "json":
        raise ValueError("fixture manifest filetype must be json")
    years = manifest.get("selected_years")
    if years != [2024, 2025]:
        raise ValueError("fixture manifest selected_years must be [2024, 2025]")
    years = [2024, 2025]
    responses = manifest.get("responses")
    if not isinstance(responses, list):
        raise ValueError("fixture manifest responses must be a list")
    entries: dict[tuple[str, str, int | None], Path] = {}
    for item in responses:
        if not isinstance(item, dict):
            raise ValueError("fixture responses must be objects")
        scenario, table, year = (
            item.get("scenario"),
            item.get("table"),
            item.get("requested_year"),
        )
        if scenario not in {
            "default_fetch",
            "selected_fetch",
            "before_c1_failure",
            "after_c1_failure",
        }:
            raise ValueError(f"unsupported fixture scenario {scenario!r}")
        if table not in {"project_resources", "project_timeseries"}:
            raise ValueError(f"unsupported fixture table {table!r}")
        if year is not None and (not isinstance(year, int) or isinstance(year, bool)):
            raise ValueError("requested_year must be an integer or null")
        key = (scenario, table, year)
        if key in entries:
            raise ValueError(f"duplicate fixture key: {key}")
        relative = Path(str(item.get("payload_path", "")))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"fixture payload path must be relative: {relative}")
        root = path.parent.resolve(strict=True)
        payload = (root / relative).resolve(strict=True)
        if root not in payload.parents or not payload.is_file():
            raise ValueError(f"fixture payload escapes manifest directory: {relative}")
        expected = item.get("sha256")
        if (
            not isinstance(expected, str)
            or len(expected) != 64
            or any(c not in "0123456789abcdef" for c in expected)
        ):
            raise ValueError("fixture sha256 must be 64 lowercase hexadecimal digits")
        if _sha256(payload) != expected:
            raise ValueError(f"fixture SHA-256 mismatch: {relative}")
        entries[key] = payload
    resources_key = ("default_fetch", "project_resources", None)
    if resources_key not in entries:
        raise ValueError("default_fetch needs unfiltered project_resources")
    try:
        default_rows = json.loads(entries[resources_key].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"default resource payload is invalid JSON: {exc}") from exc
    if not isinstance(default_rows, list) or not all(
        isinstance(row, dict) for row in default_rows
    ):
        raise ValueError("default resource payload must be an array of objects")
    dynamic_year_set: set[int] = set()
    for row in default_rows:
        raw_year = row.get("report_year")
        if isinstance(raw_year, bool) or not isinstance(raw_year, (int, str)):
            continue
        try:
            year = int(raw_year)
        except ValueError:
            continue
        if year >= 2020:
            dynamic_year_set.add(year)
    dynamic_years = sorted(dynamic_year_set)
    required: set[tuple[str, str, int | None]] = {resources_key}
    required.update(
        ("default_fetch", "project_timeseries", year) for year in dynamic_years
    )
    for scenario in ("selected_fetch", "before_c1_failure", "after_c1_failure"):
        required.update((scenario, "project_resources", year) for year in years)
        required.update((scenario, "project_timeseries", year) for year in years)
    missing = sorted(required - entries.keys())
    extra = sorted(entries.keys() - required)
    if missing or extra:
        raise ValueError(f"fixture coverage mismatch; missing={missing}, extra={extra}")
    return entries, manifest


def _validate_baseline(
    baseline_dir: Path, fixture_manifest: Path, output_dir: Path
) -> tuple[
    Path, Path, Path, dict[tuple[str, str, int | None], Path], dict[str, object]
]:
    baseline = baseline_dir.expanduser().resolve(strict=True)
    output = output_dir.expanduser().resolve(strict=False)
    manifest_path = fixture_manifest.expanduser().resolve(strict=True)
    home_esdc = (Path.home() / ".esdc").resolve(strict=False)
    if baseline == home_esdc or home_esdc in baseline.parents:
        raise ValueError("baseline under ~/.esdc is forbidden")
    if output == home_esdc or home_esdc in output.parents:
        raise ValueError("output under ~/.esdc is forbidden")
    if output.exists():
        raise ValueError("output-dir must not already exist")
    if not baseline.is_dir():
        raise ValueError("baseline-dir must be a directory")
    source_sqlite, source_duckdb = baseline / "esdc.sqlite", baseline / "esdc.duckdb"
    for path in (source_sqlite, source_duckdb):
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"baseline requires closed regular file {path.name}")
    if any(
        path.name.lower().endswith(("-wal", "-shm", "-journal"))
        for path in baseline.iterdir()
    ):
        raise ValueError("baseline contains SQLite WAL/SHM/journal sidecars")
    if output == baseline or output in baseline.parents or baseline in output.parents:
        raise ValueError("baseline and output directories overlap")
    fixture_root = manifest_path.parent
    if (
        output == fixture_root
        or output in fixture_root.parents
        or fixture_root in output.parents
    ):
        raise ValueError("fixture and output directories overlap")
    entries, manifest = _read_manifest(manifest_path)
    return baseline, output, manifest_path, entries, manifest


def _require_legacy_bootstrap_state(sqlite_path: Path) -> None:
    with closing(
        sqlite3.connect(sqlite_path.resolve().as_uri() + "?mode=ro", uri=True)
    ) as source:
        table = source.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' "
            "AND name='api_project_timeseries'"
        ).fetchone()
        if (
            table is not None
            and source.execute(
                "SELECT 1 FROM api_project_timeseries LIMIT 1"
            ).fetchone()
        ):
            raise ValueError(
                "canonical timeseries rows exist; rehearsal will not erase "
                "canonical truth to force legacy bootstrap"
            )
        snapshot_table = source.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='api_snapshot'"
        ).fetchone()
        if (
            snapshot_table is not None
            and source.execute(
                "SELECT 1 FROM api_snapshot "
                "WHERE source_table='project_timeseries' LIMIT 1"
            ).fetchone()
        ):
            raise ValueError(
                "canonical timeseries snapshot exists; rehearsal requires "
                "an explicitly legacy baseline"
            )


def _scale_classification(
    *, initial_legacy_rows: int, final_canonical_rows: int
) -> dict[str, object]:
    """Classify completed rehearsal volume from observed endpoint counts."""
    million_rows = 1_000_000
    complete_volume = (
        initial_legacy_rows >= million_rows and final_canonical_rows >= million_rows
    )
    both_small = (
        initial_legacy_rows < million_rows and final_canonical_rows < million_rows
    )
    classification = (
        "synthetic million-row lifecycle"
        if complete_volume
        else "small fixture lifecycle"
        if both_small
        else "incomplete large lifecycle"
    )
    return {
        "classification": classification,
        "initial_legacy_timeseries_rows": initial_legacy_rows,
        "final_canonical_timeseries_rows": final_canonical_rows,
        "million_row_lifecycle": "PASS" if complete_volume else "UNVERIFIED",
    }


def current_pod_fixture_bodies(sqlite_path: Path) -> dict[str, bytes]:
    """Declare the unchanged current POD baseline for a raw-data-only rehearsal.

    These are explicit offline API fixtures, not an alternative production fetch
    path. Old POD schemas refuse; only an absent registry produces empty inputs.
    """
    from esdc.pod_registry.api_context import read_pod_api_context
    from esdc.selection import PodApiTable

    context = read_pod_api_context(sqlite_path)
    return {
        table.value: json.dumps(
            [dict(row) for row in context.existing_rows.get(table.value, ())],
            default=str,
        ).encode()
        for table in PodApiTable
    }


def pod_fixture_body(
    bodies: dict[str, bytes], table_name: str, report_year: int | None
) -> bytes:
    """Apply the monitoring endpoint's year selection to its explicit baseline."""
    body = bodies[table_name]
    if table_name == "pod_monitoring" and report_year is not None:
        rows = [
            row
            for row in json.loads(body)
            if str(row.get("reported_date")).startswith(f"{report_year:04d}-")
        ]
        return json.dumps(rows).encode()
    return body


@contextmanager
def _fixture_transport(
    scenario: str,
    entries: dict[tuple[str, str, int | None], Path],
    expected: set[tuple[str, str, int | None]],
    *,
    pod_bodies: dict[str, bytes] | None = None,
) -> Iterator[list[str]]:
    import esdc.esdc as app
    from esdc.selection import FileType, PodApiTable, TableName

    remaining = set(expected)
    urls: list[str] = []
    original = app.esdc_downloader
    pod_paths = {
        urlparse(
            app.esdc_url_builder(PodApiTable(name), file_type=FileType.JSON)
        ).path: name
        for name in (pod_bodies or {})
    }

    def download(url: str, username: str = "", password: str = "") -> bytes:
        parsed = urlparse(url)
        table_name = (
            "project_resources"
            if parsed.path.endswith("project-resources")
            else "project_timeseries"
            if parsed.path.endswith("project-timeseries")
            else "unknown"
        )
        query = parse_qs(parsed.query)
        year_values = query.get("report-year")
        year = int(year_values[0]) if year_values else None
        if parsed.path in pod_paths:
            name = pod_paths[parsed.path]
            expected_url = app.esdc_url_builder(
                PodApiTable(name), file_type=FileType.JSON, report_year=year
            )
            if url != expected_url:
                raise RuntimeError(
                    f"POD fixture URL differs from endpoint contract: {url}"
                )
            assert pod_bodies is not None
            urls.append(url)
            return pod_fixture_body(pod_bodies, name, year)
        key = (scenario, table_name, year)
        if key not in remaining:
            raise RuntimeError(f"unspecified API request: {url}")
        table = (
            TableName.PROJECT_RESOURCES
            if table_name == "project_resources"
            else TableName.PROJECT_TIMESERIES
        )
        expected_url = app.esdc_url_builder(
            table, file_type=FileType.JSON, report_year=year
        )
        if (
            url != expected_url
            or query.get("output") != ["json"]
            or query.get("verbose") != ["3"]
        ):
            raise RuntimeError(
                f"request URL/query differs from esdc_url_builder contract: {url}"
            )
        remaining.remove(key)
        urls.append(url)
        return entries[key].read_bytes()

    app.esdc_downloader = download
    try:
        yield urls
        if remaining:
            raise RuntimeError(
                f"fixture requests were not consumed: {sorted(remaining)}"
            )
    finally:
        app.esdc_downloader = original


def _file_hashes(sqlite_path: Path, duckdb_path: Path) -> dict[str, str]:
    return {str(path): _sha256(path) for path in (sqlite_path, duckdb_path)}


def _protected_digests(duckdb_path: Path) -> dict[str, str]:
    from esdc.loaders import _METADATA_TABLE
    from esdc.pod_registry.contract import load_pod_contract
    from esdc.pod_registry.ownership import pod_reserved_names

    managed = {
        "project_resources",
        "project_timeseries",
        "api_publication",
        *pod_reserved_names(load_pod_contract()),
    }
    conn = duckdb.connect(str(duckdb_path), read_only=True)
    result: dict[str, str] = {}
    try:
        names = [
            row[0]
            for row in conn.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema='main' AND table_type='BASE TABLE'"
            ).fetchall()
        ]
        for name in names:
            if name in managed:
                continue
            escaped = name.replace('"', '""')
            if name == _METADATA_TABLE:
                placeholders = ", ".join("?" for _ in managed)
                cursor = conn.execute(
                    f'SELECT * FROM "{escaped}" WHERE table_name IS NULL '
                    f"OR lower(table_name) NOT IN ({placeholders}) ORDER BY ALL",
                    sorted(managed),
                )
            elif name == "_metadata":
                cursor = conn.execute(
                    f"SELECT CAST(key AS VARCHAR), CAST(value AS VARCHAR) "
                    f"FROM \"{escaped}\" WHERE key != 'last_updated' ORDER BY key"
                )
            else:
                columns = conn.execute(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_schema='main' AND table_name=? "
                    "ORDER BY ordinal_position",
                    [name],
                ).fetchall()
                projection = ", ".join(
                    f'CAST("{row[0]}" AS VARCHAR)' for row in columns
                )
                cursor = conn.execute(
                    f'SELECT {projection} FROM "{escaped}" ORDER BY ALL'
                )
            digest = hashlib.sha256()
            count = 0
            while batch := cursor.fetchmany(1000):
                count += len(batch)
                digest.update(
                    json.dumps(batch, default=str, separators=(",", ":")).encode()
                )
            if name == _METADATA_TABLE and count == 0:
                continue
            result[name] = digest.hexdigest()
    finally:
        conn.close()
    return result


def _counts_by_year(path: Path, table: str) -> dict[str, int]:
    conn = duckdb.connect(str(path), read_only=True)
    try:
        rows = conn.execute(
            f"SELECT report_year, COUNT(*) FROM {table} "
            "GROUP BY report_year ORDER BY report_year"
        ).fetchall()
        return {
            "NULL" if year is None else str(year): int(count) for year, count in rows
        }
    finally:
        conn.close()


def _sqlite_counts_by_year(path: Path, table: str) -> dict[str, int]:
    source_table = (
        "api_project_resources"
        if table == "project_resources"
        else "api_project_timeseries"
    )
    conn = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        rows = conn.execute(
            f"SELECT report_year, COUNT(*) FROM {source_table} "
            "GROUP BY report_year ORDER BY report_year"
        ).fetchall()
        return {
            "NULL" if year is None else str(year): int(count) for year, count in rows
        }
    finally:
        conn.close()


def _sqlite_table_count(path: Path, table: str) -> int:
    conn = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
        ).fetchone()
        return (
            0
            if exists is None
            else int(conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0])
        )
    finally:
        conn.close()


def _table_digest(path: Path, table: str) -> str:
    conn = duckdb.connect(str(path), read_only=True)
    digest = hashlib.sha256()
    try:
        schema = conn.execute(
            "SELECT column_name, data_type, is_nullable, column_default "
            "FROM information_schema.columns "
            "WHERE table_schema='main' AND table_name=? "
            "ORDER BY ordinal_position",
            [table],
        ).fetchall()
        digest.update(json.dumps(schema, default=str, separators=(",", ":")).encode())
        projection = ", ".join(f'CAST("{row[0]}" AS VARCHAR)' for row in schema)
        cursor = conn.execute(f'SELECT {projection} FROM "{table}" ORDER BY ALL')
        while batch := cursor.fetchmany(1000):
            digest.update(
                json.dumps(batch, default=str, separators=(",", ":")).encode()
            )
    finally:
        conn.close()
    return digest.hexdigest()


def _identity_digest(cursor: object, *, sample_limit: int = 10) -> dict[str, object]:
    digest = hashlib.sha256()
    count = 0
    samples: list[tuple[object, ...]] = []
    while batch := cursor.fetchmany(1000):  # type: ignore[attr-defined]
        for row in batch:
            digest.update(json.dumps(row, default=str, separators=(",", ":")).encode())
            digest.update(b"\n")
            count += 1
            if len(samples) < sample_limit:
                samples.append(tuple(row))
    return {"count": count, "sha256": digest.hexdigest(), "samples": samples}


def _retained_timeseries_values(
    path: Path, replaced_years: set[int] | None = None
) -> dict[str, object]:
    """Hash retained business rows independently of regenerated cache identities."""
    conn = duckdb.connect(
        str(path), read_only=True, config={"autoinstall_known_extensions": "false"}
    )
    try:
        columns = [
            row[0]
            for row in conn.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_schema='main' AND table_name='project_timeseries' "
                "ORDER BY column_name"
            ).fetchall()
            if row[0] not in ("id", "uuid")
        ]
        projection = ", ".join('"' + name.replace('"', '""') + '"' for name in columns)
        years = sorted(replaced_years or set())
        if years:
            placeholders = ", ".join("?" for _ in years)
            cursor = conn.execute(
                f"SELECT {projection} FROM project_timeseries "
                f"WHERE report_year IS NULL OR report_year NOT IN ({placeholders}) "
                "ORDER BY ALL",
                years,
            )
        else:
            cursor = conn.execute(
                f"SELECT {projection} FROM project_timeseries ORDER BY ALL"
            )
        result = _identity_digest(cursor)
        result["columns"] = columns
        return result
    finally:
        conn.close()


def _retained_timeseries_identity(
    path: Path, replaced_years: set[int] | None = None
) -> dict[str, object]:
    conn = duckdb.connect(str(path), read_only=True)
    try:
        columns = {
            row[0]
            for row in conn.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_schema='main' AND table_name='project_timeseries'"
            ).fetchall()
        }
        uuid_projection = '"uuid"' if "uuid" in columns else "NULL AS uuid"
        id_projection = '"id"' if "id" in columns else "NULL AS id"
        years = sorted(replaced_years or set())
        if years:
            placeholders = ", ".join("?" for _ in years)
            cursor = conn.execute(
                f"SELECT {id_projection}, {uuid_projection}, report_year, project_id "
                "FROM project_timeseries "
                f"WHERE report_year IS NULL OR report_year NOT IN ({placeholders}) "
                "ORDER BY id",
                years,
            )
        else:
            cursor = conn.execute(
                f"SELECT {id_projection}, {uuid_projection}, report_year, "
                "project_id FROM project_timeseries ORDER BY id"
            )
        return _identity_digest(cursor)
    finally:
        conn.close()


def _nonempty_fixture_years(
    entries: dict[tuple[str, str, int | None], Path], scenario: str
) -> set[int]:
    years: set[int] = set()
    for (entry_scenario, table, year), path in entries.items():
        if entry_scenario != scenario or table != "project_timeseries" or year is None:
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload:
            years.add(year)
    return years


def _rss_bytes() -> dict[str, object]:
    try:
        import resource

        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if platform.system() != "Darwin":
            value *= 1024
        return {
            "peak_rss_bytes": value,
            "peak_rss_reason": (
                "resource.getrusage(RUSAGE_SELF); platform units normalized to bytes"
            ),
            "scope": "entire process cumulative high-water mark",
        }
    except (ImportError, AttributeError, OSError, ValueError) as exc:
        return {
            "peak_rss_bytes": None,
            "peak_rss_reason": (
                f"resource.getrusage unavailable: {type(exc).__name__}: {exc}"
            ),
            "scope": "entire process cumulative high-water mark",
        }


def _seed_result_caches(chat_tools: object) -> None:
    chat_tools._get_cache()["t6-rehearsal-sentinel"] = "stale"  # type: ignore[attr-defined]
    chat_tools._get_tool_cache()["t6-rehearsal-sentinel"] = "stale"  # type: ignore[attr-defined]


def _result_cache_state(chat_tools: object) -> dict[str, object]:
    sql_cache = chat_tools._get_cache()  # type: ignore[attr-defined]
    tool_cache = chat_tools._get_tool_cache()  # type: ignore[attr-defined]
    return {
        "sql_entries": len(sql_cache),
        "tool_entries": len(tool_cache),
        "sql_sentinel": sql_cache.get("t6-rehearsal-sentinel"),
        "tool_sentinel": tool_cache.get("t6-rehearsal-sentinel"),
    }


def _snapshots(sqlite_path: Path) -> list[dict[str, object]]:
    conn = sqlite3.connect(sqlite_path.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        rows = conn.execute(
            "SELECT source_table, revision, row_count, source_sha256, "
            "publication_state "
            "FROM api_snapshot ORDER BY source_table"
        ).fetchall()
        return [
            dict(
                zip(
                    (
                        "source_table",
                        "revision",
                        "row_count",
                        "source_sha256",
                        "publication_state",
                    ),
                    row,
                    strict=True,
                )
            )
            for row in rows
        ]
    finally:
        conn.close()


def _lineage(duckdb_path: Path) -> list[dict[str, object]]:
    conn = duckdb.connect(str(duckdb_path), read_only=True)
    try:
        exists = conn.execute(
            "SELECT 1 FROM information_schema.tables WHERE table_name='api_publication'"
        ).fetchone()
        if exists is None:
            return []
        rows = conn.execute(
            "SELECT source_table, revision, source_row_count, source_sha256, "
            "invalid_geometry_count "
            "FROM api_publication ORDER BY source_table"
        ).fetchall()
        return [
            dict(
                zip(
                    (
                        "source_table",
                        "revision",
                        "row_count",
                        "source_sha256",
                        "invalid_geometry_count",
                    ),
                    row,
                    strict=True,
                )
            )
            for row in rows
        ]
    finally:
        conn.close()


def _audit_resources(sqlite_path: Path, duckdb_path: Path) -> dict[str, object]:
    from esdc.resource_audit import compare_resource_projection

    conn = duckdb.connect(str(duckdb_path), read_only=True)
    try:
        report = compare_resource_projection(conn, sqlite_path)
        return {
            "ok": report.ok,
            "canonical_rows": report.canonical_rows,
            "projected_rows": report.projected_rows,
            "issues": list(report.issues),
        }
    finally:
        conn.close()


def run_analytics_rehearsal(
    *, baseline_dir: Path, fixture_manifest: Path, output_dir: Path
) -> Path:
    """Run a deterministic offline lifecycle on isolated copies of a DB pair."""
    baseline, output, manifest_path, entries, manifest = _validate_baseline(
        baseline_dir, fixture_manifest, output_dir
    )
    started = time.monotonic()
    output.mkdir(parents=True)
    clone = output / "clone"
    clone.mkdir()
    original_hashes = _file_hashes(baseline / "esdc.sqlite", baseline / "esdc.duckdb")
    shutil.copy2(baseline / "esdc.sqlite", clone / "esdc.sqlite")
    shutil.copy2(baseline / "esdc.duckdb", clone / "esdc.duckdb")
    sqlite_path, duckdb_path = clone / "esdc.sqlite", clone / "esdc.duckdb"
    clone_hashes = _file_hashes(sqlite_path, duckdb_path)
    if tuple(clone_hashes.values()) != tuple(original_hashes.values()):
        raise ValueError(
            "closed baseline clone hashes do not match source before execution"
        )
    with closing(
        sqlite3.connect(sqlite_path.resolve().as_uri() + "?mode=ro", uri=True)
    ) as count_source:
        # Inspect only the verified closed-file clone. Even a read-only SQLite
        # connection can create WAL coordination files beside its source file.
        integrity = count_source.execute("PRAGMA integrity_check").fetchone()
        if integrity is None or integrity[0] != "ok":
            raise ValueError("baseline SQLite integrity_check failed")
        resource_count_sqlite = count_source.execute(
            "SELECT COUNT(*) FROM api_project_resources"
        ).fetchone()[0]
        timeseries_table = count_source.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' "
            "AND name='api_project_timeseries'"
        ).fetchone()
        timeseries_count_sqlite = (
            0
            if timeseries_table is None
            else count_source.execute(
                "SELECT COUNT(*) FROM api_project_timeseries"
            ).fetchone()[0]
        )
    with duckdb.connect(str(duckdb_path), read_only=True) as count_target:
        resource_count_duckdb = count_target.execute(
            "SELECT COUNT(*) FROM project_resources"
        ).fetchone()
        timeseries_count_duckdb = count_target.execute(
            "SELECT COUNT(*) FROM project_timeseries"
        ).fetchone()
        assert resource_count_duckdb is not None
        assert timeseries_count_duckdb is not None
        resource_count_duckdb = resource_count_duckdb[0]
        timeseries_count_duckdb = timeseries_count_duckdb[0]
    report: dict[str, object] = {
        "version": 1,
        "status": "RUNNING",
        "original_paths": {
            "sqlite": str(baseline / "esdc.sqlite"),
            "duckdb": str(baseline / "esdc.duckdb"),
        },
        "clone_paths": {"sqlite": str(sqlite_path), "duckdb": str(duckdb_path)},
        "original_hashes": original_hashes,
        "clone_hashes_before": clone_hashes,
        "source_counts": {
            "project_resources": {
                "sqlite": resource_count_sqlite,
                "duckdb": resource_count_duckdb,
            },
            "project_timeseries": {
                "sqlite": timeseries_count_sqlite,
                "duckdb": timeseries_count_duckdb,
            },
        },
        "manifest_path": str(manifest_path),
        "manifest_response_count": len(entries),
        "stages": [],
        "sqlite_snapshots": [],
        "duckdb_lineage": [],
        "per_year_counts": {},
        "parity_differences": [],
        "protected_table_digests": {},
        "telemetry": _rss_bytes(),
        "scale": {"classification": "small fixture; million-row lifecycle pending"},
    }
    stage_list: list[dict[str, object]] = report["stages"]  # type: ignore[assignment]
    protected_before = _protected_digests(duckdb_path)
    report["protected_table_digests"] = {"before": protected_before}
    active_scenario: str | None = None
    active_command: str | None = None
    active_assertion: str | None = None
    failure_exception: BaseException | None = None
    try:
        import esdc.chat.tools as chat_tools
        import esdc.esdc as app
        from esdc import fetch_publication
        from esdc.analytics_audit import compare_timeseries_projection
        from esdc.analytics_rebuild import rebuild_api_analytics
        from esdc.configs import Config
        from esdc.selection import FileType
        from esdc.timeseries_bootstrap import bootstrap_timeseries_source

        previous_env = {
            name: os.environ.get(name)
            for name in ("ESDC_CONFIG_DIR", "ESDC_DB_DIR", "ESDC_DB_FILE")
        }
        os.environ["ESDC_CONFIG_DIR"] = str(output / "config")
        os.environ["ESDC_DB_DIR"] = str(clone)
        os.environ["ESDC_DB_FILE"] = str(duckdb_path)
        prior_config_cache = Config._config_cache
        prior_sql_cache = chat_tools._sql_cache
        prior_tool_cache = chat_tools._tool_cache
        Config._config_cache = None
        chat_tools._sql_cache = None
        chat_tools._tool_cache = None
        try:
            active_scenario = "bootstrap_dry_run"
            active_command = "bootstrap_timeseries_source(apply=False)"
            active_assertion = "dry run leaves both cloned database hashes unchanged"
            _require_legacy_bootstrap_state(sqlite_path)
            original_legacy_ids = _retained_timeseries_identity(duckdb_path)
            original_legacy_state = _table_digest(duckdb_path, "project_timeseries")
            before_dry = _file_hashes(sqlite_path, duckdb_path)
            with duckdb.connect(str(duckdb_path), read_only=True) as legacy_target:
                legacy_count = legacy_target.execute(
                    "SELECT COUNT(*) FROM project_timeseries"
                ).fetchone()
                assert legacy_count is not None
                legacy_count = legacy_count[0]
            dry = bootstrap_timeseries_source(
                duckdb_path=duckdb_path, sqlite_path=sqlite_path, apply=False
            )
            if dry.applied or _file_hashes(sqlite_path, duckdb_path) != before_dry:
                raise RuntimeError("bootstrap dry run changed source files")
            stage_list.append(
                {
                    "name": "bootstrap_dry_run",
                    "command": "bootstrap_timeseries_source(apply=False)",
                    "exit": 0,
                    "assertions": [
                        "legacy DuckDB source inspected on lifecycle clone",
                        "SQLite and DuckDB hashes unchanged by dry run",
                    ],
                    "row_count": dry.row_count,
                    "null_year_rows": dry.null_year_rows,
                    "source_types": dry.source_types,
                    "file_hashes": before_dry,
                }
            )
            active_scenario = "bootstrap_apply"
            active_command = "bootstrap_timeseries_source(apply=True)"
            active_assertion = "bootstrap preserves all legacy DuckDB row IDs and UUIDs"
            applied = bootstrap_timeseries_source(
                duckdb_path=duckdb_path, sqlite_path=sqlite_path, apply=True
            )
            if (
                not applied.applied
                or applied.row_count != dry.row_count
                or applied.source_sha256 != dry.source_sha256
            ):
                raise RuntimeError(
                    "bootstrap apply differs from the verified dry-run source"
                )
            if applied.row_count != legacy_count:
                raise RuntimeError(
                    "bootstrap did not preserve the complete legacy DuckDB row count"
                )
            if (
                _table_digest(duckdb_path, "project_timeseries")
                != original_legacy_state
                or _retained_timeseries_identity(duckdb_path) != original_legacy_ids
            ):
                raise RuntimeError(
                    "bootstrap apply changed legacy DuckDB rows, IDs, or UUIDs"
                )
            stage_list.append(
                {
                    "name": "bootstrap_apply",
                    "command": "bootstrap_timeseries_source(apply=True)",
                    "exit": 0,
                    "assertions": [
                        "legacy rows imported into SQLite with no canonical TS table",
                        "row count and source hash match dry run",
                        "DuckDB hash, typed schema/data, IDs, and UUIDs unchanged",
                        "next stage is the first complete unfiltered fetch",
                    ],
                    "row_count": applied.row_count,
                    "null_year_rows": applied.null_year_rows,
                    "legacy_identity_digest": original_legacy_ids,
                    "source_types": applied.source_types,
                    "file_hashes": _file_hashes(sqlite_path, duckdb_path),
                }
            )

            for scenario, years in (
                ("default_fetch", None),
                ("selected_fetch", [2024, 2025]),
            ):
                active_scenario = scenario
                active_command = "load_esdc_data(filetype=json)"
                active_assertion = (
                    "complete fixture fetch, typed parity, retained canonical values, "
                    "and result-cache invalidation"
                )
                changed_years = {
                    year
                    for entry_scenario, table, year in entries
                    if entry_scenario == scenario
                    and table == "project_timeseries"
                    and year is not None
                }
                _seed_result_caches(chat_tools)
                caches_before = _result_cache_state(chat_tools)
                retained_before = _retained_timeseries_values(
                    duckdb_path, changed_years
                )
                expected = {key for key in entries if key[0] == scenario}
                with _fixture_transport(
                    scenario,
                    entries,
                    expected,
                    pod_bodies=current_pod_fixture_bodies(sqlite_path),
                ) as urls:
                    outcome = app.load_esdc_data(
                        filetype=FileType.JSON,
                        to_file=False,
                        reload=True,
                        username="offline-fixture",
                        password="offline-fixture",
                        years=years,
                        reindex=False,
                    )
                audit = compare_timeseries_projection(
                    sqlite_path=sqlite_path, duckdb_path=duckdb_path
                )
                if not audit["ok"]:
                    raise RuntimeError(
                        f"{scenario} typed parity failed: {audit['issues']}"
                    )
                resource_audit = _audit_resources(sqlite_path, duckdb_path)
                if not resource_audit["ok"]:
                    raise RuntimeError(
                        f"{scenario} resource parity failed: {resource_audit['issues']}"
                    )
                retained_after = _retained_timeseries_values(duckdb_path, changed_years)
                if years is None and retained_after["count"] != 0:
                    raise RuntimeError(
                        f"{scenario} retained rows outside the complete source snapshot"
                    )
                if years is not None and retained_after != retained_before:
                    raise RuntimeError(
                        f"{scenario} changed retained older/NULL-year timeseries "
                        "canonical values"
                    )
                protected_now = _protected_digests(duckdb_path)
                if protected_now != protected_before:
                    raise RuntimeError(f"{scenario} changed protected DuckDB objects")
                caches_after = _result_cache_state(chat_tools)
                if (
                    caches_after["sql_sentinel"] is not None
                    or caches_after["tool_sentinel"] is not None
                ):
                    raise RuntimeError(
                        f"{scenario} did not invalidate both analytical result caches"
                    )
                stage_list.append(
                    {
                        "name": scenario,
                        "command": "load_esdc_data(filetype=json)",
                        "exit": 0,
                        "assertions": [
                            "all expected fixture requests consumed",
                            "all URLs matched esdc_url_builder",
                            "timeseries typed parity passed",
                            "resource parity passed",
                            "retained canonical values unchanged "
                            "outside replaced slices",
                            "both SQL and tool result caches invalidated",
                            "protected objects unchanged",
                        ],
                        "requests": urls,
                        "resource_rows": outcome.resource_row_count,
                        "timeseries_rows": outcome.timeseries_row_count,
                        "retained_values_before": retained_before,
                        "retained_values_after": retained_after,
                        "result_caches_before": caches_before,
                        "result_caches_after": caches_after,
                        "source_snapshots": _snapshots(sqlite_path),
                        "duckdb_lineage": _lineage(duckdb_path),
                        "file_hashes": _file_hashes(sqlite_path, duckdb_path),
                    }
                )

            # Inject a failure in the coordinator before its SQLite publication.
            active_scenario = "before_c1_failure"
            active_command = "load_esdc_data with injected in-transaction C1 failure"
            active_assertion = "C1 failure leaves both cloned database hashes unchanged"
            _seed_result_caches(chat_tools)
            caches_before_c1 = _result_cache_state(chat_tools)
            before_c1 = _file_hashes(sqlite_path, duckdb_path)
            original_replace = fetch_publication.replace_timeseries_in_transaction

            def fail_before_c1(*args, **kwargs):
                raise RuntimeError(
                    "fixture injected within C1 transaction before commit"
                )

            fetch_publication.replace_timeseries_in_transaction = fail_before_c1
            try:
                with _fixture_transport(
                    active_scenario,
                    entries,
                    {key for key in entries if key[0] == active_scenario},
                    pod_bodies=current_pod_fixture_bodies(sqlite_path),
                ):
                    try:
                        app.load_esdc_data(
                            FileType.JSON,
                            False,
                            True,
                            "offline-fixture",
                            "offline-fixture",
                            [2024, 2025],
                            False,
                        )
                    except RuntimeError as exc:
                        if (
                            "fixture injected within C1 transaction before commit"
                            not in str(exc)
                        ):
                            raise RuntimeError(
                                "unexpected pre-C1 failure instead of injected "
                                f"seam: {exc}"
                            ) from exc
                    else:
                        raise RuntimeError("pre-C1 injected failure was not observed")
            finally:
                fetch_publication.replace_timeseries_in_transaction = original_replace
            after_c1_failure = _file_hashes(sqlite_path, duckdb_path)
            if before_c1 != after_c1_failure:
                raise RuntimeError("pre-C1 failure changed a database file")
            caches_after_c1 = _result_cache_state(chat_tools)
            if (
                caches_after_c1["sql_sentinel"] != "stale"
                or caches_after_c1["tool_sentinel"] != "stale"
            ):
                raise RuntimeError(
                    "pre-C1 failure unexpectedly invalidated result caches"
                )
            stage_list.append(
                {
                    "name": "before_c1_failure",
                    "command": "fetch with coordinator seam failure",
                    "exit": 0,
                    "assertions": [
                        "expected injected failure",
                        "SQLite and DuckDB file hashes unchanged",
                        "both analytical result caches retained pre-commit values",
                    ],
                    "result_caches_before": caches_before_c1,
                    "result_caches_after": caches_after_c1,
                    "source_snapshots": _snapshots(sqlite_path),
                    "duckdb_lineage": _lineage(duckdb_path),
                    "file_hashes": after_c1_failure,
                }
            )

            active_scenario = "after_c1_failure"
            active_command = "load_esdc_data with injected post-C1/pre-C2 failure"
            active_assertion = (
                "C2 rollback preserves both DuckDB API tables and lineage "
                "after C1 commit"
            )
            _seed_result_caches(chat_tools)
            caches_before_c2 = _result_cache_state(chat_tools)
            before_c2 = {
                table: _table_digest(duckdb_path, table)
                for table in (
                    "project_resources",
                    "project_timeseries",
                    "api_publication",
                )
            }
            snapshots_before_c2 = _snapshots(sqlite_path)
            from esdc import analytics_rebuild

            original_stage = analytics_rebuild.apply_timeseries_stage

            def fail_after_c1(*args, **kwargs):
                raise RuntimeError(
                    "T6 injected failure after resource projection apply, "
                    "inside C2 before timeseries apply"
                )

            analytics_rebuild.apply_timeseries_stage = fail_after_c1
            try:
                with _fixture_transport(
                    active_scenario,
                    entries,
                    {key for key in entries if key[0] == active_scenario},
                    pod_bodies=current_pod_fixture_bodies(sqlite_path),
                ) as urls:
                    try:
                        app.load_esdc_data(
                            FileType.JSON,
                            False,
                            True,
                            "offline-fixture",
                            "offline-fixture",
                            [2024, 2025],
                            False,
                        )
                    except Exception as exc:
                        if (
                            not getattr(exc, "sqlite_committed", False)
                            or "T6 injected failure after resource projection apply"
                            not in str(exc)
                        ):
                            raise RuntimeError(
                                f"post-C1 failure lacks durable C1 context: {exc}"
                            ) from exc
                    else:
                        raise RuntimeError("post-C1 injected failure was not observed")
            finally:
                analytics_rebuild.apply_timeseries_stage = original_stage
            if _protected_digests(duckdb_path) != protected_before:
                raise RuntimeError("C2 failure changed a protected DuckDB object")
            after_c2 = {table: _table_digest(duckdb_path, table) for table in before_c2}
            snapshots_after_c2 = _snapshots(sqlite_path)
            prior_revisions = {
                item["source_table"]: item["revision"] for item in snapshots_before_c2
            }
            current_revisions = {
                item["source_table"]: item["revision"] for item in snapshots_after_c2
            }
            if before_c2 != after_c2 or not all(
                current_revisions.get(name) != revision
                for name, revision in prior_revisions.items()
            ):
                raise RuntimeError(
                    "C2 failure did not retain SQLite-only revisions while preserving "
                    "both DuckDB API projections and lineage"
                )
            expected_resources = {
                (row.get("project_id"), row.get("project_name"))
                for (scenario, table, _), fixture_path in entries.items()
                if scenario == "after_c1_failure" and table == "project_resources"
                for row in json.loads(fixture_path.read_text(encoding="utf-8"))
                if isinstance(row.get("project_id"), str)
            }
            expected_timeseries = {
                (row.get("project_id"), row.get("project_name"))
                for (scenario, table, _), fixture_path in entries.items()
                if scenario == "after_c1_failure" and table == "project_timeseries"
                for row in json.loads(fixture_path.read_text(encoding="utf-8"))
                if isinstance(row.get("project_id"), str)
            }
            if not expected_resources or not expected_timeseries:
                raise RuntimeError("C2 failure fixture lacks durable project IDs")
            with closing(
                sqlite3.connect(sqlite_path.resolve().as_uri() + "?mode=ro", uri=True)
            ) as source_check:
                resource_placeholders = ",".join("?" for _ in expected_resources)
                durable_resources = {
                    (row[0], row[1])
                    for row in source_check.execute(
                        "SELECT project_id, project_name FROM api_project_resources "
                        f"WHERE project_id IN ({resource_placeholders})",
                        sorted(project_id for project_id, _ in expected_resources),
                    ).fetchall()
                }
                timeseries_placeholders = ",".join("?" for _ in expected_timeseries)
                durable_timeseries = {
                    (row[0], row[1])
                    for row in source_check.execute(
                        "SELECT json_extract(row_json, '$.project_id'), "
                        "json_extract(row_json, '$.project_name') "
                        "FROM api_project_timeseries "
                        "WHERE json_extract(row_json, '$.project_id') IN "
                        f"({timeseries_placeholders})",
                        sorted(project_id for project_id, _ in expected_timeseries),
                    ).fetchall()
                }
            resources_durable = expected_resources <= durable_resources
            timeseries_durable = expected_timeseries <= durable_timeseries
            if not resources_durable or not timeseries_durable:
                raise RuntimeError(
                    "expected fixture values did not remain durable in canonical "
                    "SQLite after C2 rollback"
                )
            caches_after_c2 = _result_cache_state(chat_tools)
            if (
                caches_after_c2["sql_sentinel"] != "stale"
                or caches_after_c2["tool_sentinel"] != "stale"
            ):
                raise RuntimeError(
                    "failed C2 unexpectedly invalidated result caches before recovery"
                )
            stage_list.append(
                {
                    "name": "after_c1_failure",
                    "command": (
                        "load_esdc_data with T6 injected failure after resource "
                        "projection apply inside C2"
                    ),
                    "exit": 0,
                    "assertions": [
                        "distinctive injected C2 failure observed after C1 commit",
                        "both DuckDB API schemas, typed rows, IDs/UUIDs, geometry "
                        "values, and publication lineage unchanged",
                        "new canonical revisions and fixture values committed",
                        "both result caches retained until recovery",
                        "protected objects unchanged",
                    ],
                    "requests": urls,
                    "result_caches_before": caches_before_c2,
                    "result_caches_after": caches_after_c2,
                    "source_snapshots_before": snapshots_before_c2,
                    "source_snapshots_after": snapshots_after_c2,
                    "duckdb_lineage": _lineage(duckdb_path),
                    "file_hashes": _file_hashes(sqlite_path, duckdb_path),
                }
            )

            # Recovery runs with a transport that rejects every accidental request.
            active_scenario = "api_free_pair_recovery"
            active_command = "rebuild_api_analytics"
            active_assertion = (
                "API-free pair recovery restores resource and timeseries parity"
            )
            caches_before_recovery = _result_cache_state(chat_tools)
            original_downloader = app.esdc_downloader
            app.esdc_downloader = lambda *args, **kwargs: (_ for _ in ()).throw(
                RuntimeError("network request forbidden during recovery")
            )
            try:
                recovery = rebuild_api_analytics(
                    duckdb_path=duckdb_path, sqlite_path=sqlite_path
                )
            finally:
                app.esdc_downloader = original_downloader
            recovered = compare_timeseries_projection(
                sqlite_path=sqlite_path, duckdb_path=duckdb_path
            )
            if not recovered["ok"]:
                raise RuntimeError(
                    f"API-free pair recovery parity failed: {recovered['issues']}"
                )
            recovered_resources = _audit_resources(sqlite_path, duckdb_path)
            if not recovered_resources["ok"]:
                raise RuntimeError(
                    "API-free resource recovery parity failed: "
                    f"{recovered_resources['issues']}"
                )
            if _protected_digests(duckdb_path) != protected_before:
                raise RuntimeError(
                    "API-free recovery changed a protected DuckDB object"
                )
            caches_after_recovery = _result_cache_state(chat_tools)
            if (
                caches_after_recovery["sql_sentinel"] is not None
                or caches_after_recovery["tool_sentinel"] is not None
            ):
                raise RuntimeError(
                    "API-free recovery did not invalidate both analytical result caches"
                )
            stage_list.append(
                {
                    "name": "api_free_pair_recovery",
                    "command": "rebuild_api_analytics",
                    "exit": 0,
                    "assertions": [
                        "resource and timeseries typed parity passed",
                        "network transport rejected requests",
                        "both SQL and tool result caches invalidated",
                        "protected objects unchanged",
                    ],
                    "result_caches_before": caches_before_recovery,
                    "result_caches_after": caches_after_recovery,
                    "resource_audit": recovered_resources,
                    "timeseries_audit": recovered,
                    "source_revisions": recovery.source_revisions,
                    "source_snapshots": _snapshots(sqlite_path),
                    "duckdb_lineage": _lineage(duckdb_path),
                    "file_hashes": _file_hashes(sqlite_path, duckdb_path),
                }
            )

            # A deliberate target-only value mutation must fail independent parity.
            active_scenario = "deliberate_value_corruption"
            active_command = "target-only mutation + compare_timeseries_projection"
            active_assertion = (
                "independent audit detects equal-lineage target value corruption"
            )
            writer = duckdb.connect(str(duckdb_path))
            try:
                writer.execute(
                    "UPDATE project_timeseries "
                    "SET project_name='T6 deliberate corruption' "
                    "WHERE id=(SELECT min(id) FROM project_timeseries)"
                )
            finally:
                writer.close()
            corrupted = compare_timeseries_projection(
                sqlite_path=sqlite_path, duckdb_path=duckdb_path
            )
            if corrupted["ok"]:
                raise RuntimeError(
                    "independent auditor missed deliberate target value corruption"
                )
            report["parity_differences"] = corrupted["difference_samples"]
            stage_list.append(
                {
                    "name": "deliberate_value_corruption",
                    "command": "target-only value update + independent audit",
                    "exit": 0,
                    "assertions": ["equal-lineage value corruption detected"],
                    "difference_count": corrupted["value_difference_count"],
                }
            )
            report["per_year_counts"] = {
                table: {
                    "sqlite": _sqlite_counts_by_year(sqlite_path, table),
                    "duckdb": _counts_by_year(duckdb_path, table),
                }
                for table in ("project_resources", "project_timeseries")
            }
            for table, counts in report["per_year_counts"].items():
                if counts["sqlite"] != counts["duckdb"]:
                    raise RuntimeError(
                        f"final {table} per-year counts differ between "
                        "SQLite and DuckDB"
                    )
            report["protected_table_digests"] = {
                "before": protected_before,
                "after": _protected_digests(duckdb_path),
            }
            if (
                report["protected_table_digests"]["before"]
                != report["protected_table_digests"]["after"]
            ):
                raise RuntimeError(
                    "fetch/recovery lifecycle changed protected tables or "
                    "stable metadata"
                )
            final_canonical_rows = corrupted["canonical_rows"]
            if not isinstance(final_canonical_rows, int):
                raise RuntimeError("final canonical row count is not an integer")
            report["scale"] = _scale_classification(
                initial_legacy_rows=timeseries_count_duckdb,
                final_canonical_rows=final_canonical_rows,
            )
        finally:
            pending_exception = sys.exc_info()[1]
            cleanup_error: BaseException | None = None
            owned_handles = (
                (chat_tools._sql_cache, prior_sql_cache),
                (chat_tools._tool_cache, prior_tool_cache),
            )
            for handle, prior_handle in owned_handles:
                if handle is not None and handle is not prior_handle:
                    try:
                        handle.close()
                    except BaseException as exc:
                        cleanup_error = cleanup_error or exc
            try:
                for name, value in previous_env.items():
                    if value is None:
                        os.environ.pop(name, None)
                    else:
                        os.environ[name] = value
            except BaseException as exc:
                cleanup_error = cleanup_error or exc
            finally:
                Config._config_cache = prior_config_cache
                chat_tools._sql_cache = prior_sql_cache
                chat_tools._tool_cache = prior_tool_cache
            if cleanup_error is not None and pending_exception is None:
                raise cleanup_error
        original_hashes_after = _file_hashes(
            baseline / "esdc.sqlite", baseline / "esdc.duckdb"
        )
        report["original_hashes_after"] = original_hashes_after
        if original_hashes_after != original_hashes:
            raise RuntimeError("original baseline hashes changed during rehearsal")
        report["clone_hashes_after"] = _file_hashes(sqlite_path, duckdb_path)
        report["sqlite_snapshots"] = _snapshots(sqlite_path)
        report["duckdb_lineage"] = _lineage(duckdb_path)
        report["status"] = "PASS"
    except BaseException as exc:
        failure_exception = exc
        report["status"] = "FAIL"
        report["failed_stage"] = active_scenario or "preflight"
        report["failed_command"] = active_command
        report["failed_exit"] = 1
        report["failed_assertion"] = (
            active_assertion or "stage completed all declared assertions"
        )
        report["failure"] = f"{type(exc).__name__}: {exc}"
        report["original_hashes_after"] = _file_hashes(
            baseline / "esdc.sqlite", baseline / "esdc.duckdb"
        )
    report["telemetry"] = _rss_bytes()
    report["wall_time_seconds"] = time.monotonic() - started
    report_path = output / "report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    if failure_exception is not None:
        raise failure_exception
    return report_path
