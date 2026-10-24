"""Independent timeseries parity audit regressions."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Iterable
from pathlib import Path

import duckdb
import pytest

from esdc.api_snapshot import JSONScalar
from esdc.timeseries_projection import (
    apply_timeseries_stage,
    stage_canonical_timeseries,
)
from esdc.timeseries_snapshot import (
    TIMESERIES_SOURCE_TYPES,
    ensure_timeseries_tables,
    normalize_timeseries_row,
    replace_timeseries_in_transaction,
)


def _source(path: Path, rows: Iterable[tuple[JSONScalar, ...]]) -> tuple[str, str]:
    conn = sqlite3.connect(path)
    try:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS api_snapshot ("
            "source_table TEXT PRIMARY KEY, revision TEXT, row_count INTEGER, "
            "source_sha256 TEXT, fetched_at TEXT, publication_state TEXT, "
            "invalid_geometry_count INTEGER)"
        )
        ensure_timeseries_tables(conn)
        conn.execute("BEGIN")
        snapshot = replace_timeseries_in_transaction(
            conn,
            (
                normalize_timeseries_row(
                    ("report_year", "project_id", "year", "rate_oil", "project_name"),
                    row,
                    origin="api_json",
                )
                for row in rows
            ),
            years=None,
            revision="source-r1",
            fetched_at="2026-10-24T00:00:00+00:00",
        )
        conn.execute(
            "UPDATE api_snapshot SET revision=?, row_count=?, source_sha256=?, "
            "fetched_at=?, publication_state=?, invalid_geometry_count=? "
            "WHERE source_table=?",
            (
                snapshot.revision,
                snapshot.row_count,
                snapshot.source_sha256,
                snapshot.fetched_at,
                "published",
                0,
                "project_timeseries",
            ),
        )
        conn.commit()
        return snapshot.revision, snapshot.source_sha256
    finally:
        conn.close()


def _published(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    rows: list[tuple[JSONScalar, ...]] | None = None,
) -> tuple[Path, Path]:
    sqlite_path = tmp_path / "source.sqlite"
    duckdb_path = tmp_path / "mirror.duckdb"
    revision, source_hash = _source(
        sqlite_path,
        rows or [(2025, "P-1", 2025, 1.25, "Cendana")],
    )
    monkeypatch.setattr(
        "esdc.dbmanager._create_fts_indexes",
        lambda conn, *, tables, extensions_loaded: dict.fromkeys(tables, "ok"),
    )
    with sqlite3.connect(sqlite_path) as source:
        stage_conn = duckdb.connect(str(duckdb_path))
        stage = stage_canonical_timeseries(stage_conn, source, years=None)
        stage_conn.execute("BEGIN")
        apply_timeseries_stage(stage_conn, stage)
        stage_conn.execute(
            "CREATE TABLE api_publication ("
            "source_table VARCHAR, revision VARCHAR, "
            "source_row_count BIGINT, source_sha256 VARCHAR, "
            "invalid_geometry_count BIGINT, published_at TIMESTAMP)"
        )
        stage_conn.execute(
            "INSERT INTO api_publication VALUES ('project_timeseries', ?, ?, ?, 0, current_timestamp)",
            [
                revision,
                len(rows or [(2025, "P-1", 2025, 1.25, "Cendana")]),
                source_hash,
            ],
        )
        stage_conn.execute("COMMIT")
        stage_conn.close()
        return sqlite_path, duckdb_path


def _mutate(duckdb_path: Path, sql: str) -> None:
    conn = duckdb.connect(str(duckdb_path))
    try:
        conn.execute(sql)
    finally:
        conn.close()


def _wrong_type_copy(source_path: Path, target_path: Path) -> None:
    target = duckdb.connect(str(target_path))
    try:
        quoted = str(source_path).replace("'", "''")
        target.execute(f"ATTACH '{quoted}' AS original (READ_ONLY)")
        target.execute(
            "CREATE TABLE project_timeseries AS SELECT * REPLACE "
            "(rate_oil::DECIMAL(18, 4) AS rate_oil) "
            "FROM original.main.project_timeseries"
        )
        target.execute(
            "CREATE TABLE api_publication AS SELECT * FROM original.main.api_publication"
        )
    finally:
        target.close()


def test_equal_counts_wrong_value_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from esdc.analytics_audit import compare_timeseries_projection

    sqlite_path, duckdb_path = _published(tmp_path, monkeypatch)
    _mutate(duckdb_path, "UPDATE project_timeseries SET project_name='wrong'")
    report = compare_timeseries_projection(
        sqlite_path=sqlite_path, duckdb_path=duckdb_path
    )
    assert report["canonical_rows"] == report["projected_rows"] == 1
    assert report["lineage_matches"] is True
    difference_count = report["value_difference_count"]
    assert isinstance(difference_count, int) and difference_count > 0
    assert report["ok"] is False


def test_duplicate_loss_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from esdc.analytics_audit import compare_timeseries_projection

    sqlite_path, duckdb_path = _published(
        tmp_path,
        monkeypatch,
        [
            (2025, "P-1", 2025, 1.25, "Cendana"),
            (2025, "P-1", 2025, 1.25, "Cendana"),
        ],
    )
    _mutate(
        duckdb_path,
        "DELETE FROM project_timeseries WHERE id=(SELECT max(id) FROM project_timeseries)",
    )
    report = compare_timeseries_projection(
        sqlite_path=sqlite_path, duckdb_path=duckdb_path
    )
    assert report["value_difference_count"] == 1
    assert report["ok"] is False


def test_raw_null_empty_text_and_decimal_casts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from esdc.analytics_audit import compare_timeseries_projection

    sqlite_path, duckdb_path = _published(
        tmp_path,
        monkeypatch,
        [(2025, "", 2025, 1.25, "")],
    )
    report = compare_timeseries_projection(
        sqlite_path=sqlite_path, duckdb_path=duckdb_path
    )
    assert report["ok"] is True  # empty strings and NULLs remain distinct
    _mutate(duckdb_path, "UPDATE project_timeseries SET project_name=NULL")
    report = compare_timeseries_projection(
        sqlite_path=sqlite_path, duckdb_path=duckdb_path
    )
    difference_count = report["value_difference_count"]
    assert isinstance(difference_count, int) and difference_count > 0
    assert report["ok"] is False
    wrong_type = tmp_path / "wrong-type.duckdb"
    _wrong_type_copy(duckdb_path, wrong_type)
    report = compare_timeseries_projection(
        sqlite_path=sqlite_path, duckdb_path=wrong_type
    )
    issues = report["issues"]
    assert isinstance(issues, list)
    assert any(
        "rate_oil" in issue and "type" in issue.lower()
        for issue in issues
        if isinstance(issue, str)
    )
    assert report["ok"] is False


def test_lineage_matches_but_wrong_rows_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from esdc.analytics_audit import compare_timeseries_projection

    sqlite_path, duckdb_path = _published(tmp_path, monkeypatch)
    _mutate(duckdb_path, "UPDATE project_timeseries SET rate_oil=9.0")
    report = compare_timeseries_projection(
        sqlite_path=sqlite_path, duckdb_path=duckdb_path
    )
    assert report["lineage_matches"] is True
    assert report["ok"] is False


def test_no_source_or_target_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from esdc.analytics_audit import compare_timeseries_projection

    sqlite_path, duckdb_path = _published(tmp_path, monkeypatch)
    before = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (sqlite_path, duckdb_path)
    }
    readonly = duckdb.connect(str(duckdb_path), read_only=True)
    try:
        compare_timeseries_projection(sqlite_path=sqlite_path, duckdb_path=duckdb_path)
    finally:
        readonly.close()
    after = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (sqlite_path, duckdb_path)
    }
    assert before == after


def test_malformed_provenance_type_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from esdc.analytics_audit import compare_timeseries_projection

    sqlite_path, duckdb_path = _published(tmp_path, monkeypatch)
    source = sqlite3.connect(sqlite_path)
    try:
        source.execute("UPDATE api_project_timeseries SET origin='legacy_duckdb'")
        source_types: dict[str, object] = dict(TIMESERIES_SOURCE_TYPES)
        source_types["rate_oil"] = {"type": "DOUBLE"}
        source.execute(
            "UPDATE api_snapshot_provenance SET origin='legacy_duckdb', details_json=? "
            "WHERE source_table='project_timeseries'",
            (
                json.dumps(
                    {
                        "source_path": "fixture.duckdb",
                        "source_sha256": "0" * 64,
                        "source_size": 1,
                        "source_mtime_ns": 1,
                        "normalized_source_sha256": "0" * 64,
                        "source_types": source_types,
                        "row_count": 1,
                        "null_year_rows": 0,
                    }
                ),
            ),
        )
        source.commit()
    finally:
        source.close()
    target = duckdb.connect(str(duckdb_path), read_only=True)
    try:
        report = compare_timeseries_projection(
            sqlite_path=sqlite_path, duckdb_path=duckdb_path
        )
        assert report["ok"] is False
        issues = report["issues"]
        assert isinstance(issues, list)
        assert any(
            isinstance(issue, str) and "rate_oil" in issue and "type" in issue.lower()
            for issue in issues
        )
    finally:
        target.close()


def test_verified_legacy_decimal_and_hugeint_types_preserve_precision(
    tmp_path: Path,
) -> None:
    from esdc.analytics_audit import compare_timeseries_projection

    sqlite_path = tmp_path / "legacy.sqlite"
    duckdb_path = tmp_path / "legacy.duckdb"
    revision, source_hash = _source(
        sqlite_path,
        [(2025, "legacy", 9223372036854775809, "12345678901234567890.1234", "Legacy")],
    )
    source = sqlite3.connect(sqlite_path)
    try:
        source.execute("UPDATE api_project_timeseries SET origin='legacy_duckdb'")
        types = dict(TIMESERIES_SOURCE_TYPES)
        types.update(
            {
                "report_date": "DATE",
                "report_year": "HUGEINT",
                "year": "HUGEINT",
                "rate_oil": "DECIMAL(30,4)",
            }
        )
        details = {
            "source_path": "legacy.duckdb",
            "source_sha256": "1" * 64,
            "source_size": 1,
            "source_mtime_ns": 1,
            "normalized_source_sha256": "1" * 64,
            "source_types": types,
            "row_count": 1,
            "null_year_rows": 0,
        }
        source.execute(
            "UPDATE api_snapshot_provenance SET origin='legacy_duckdb', details_json=? "
            "WHERE source_table='project_timeseries' AND revision=?",
            (json.dumps(details), revision),
        )
        source.commit()
    finally:
        source.close()
    target = duckdb.connect(str(duckdb_path))
    try:
        target.execute(Path("esdc/sql/create_table_project_timeseries.sql").read_text())
        target.execute(
            "ALTER TABLE project_timeseries ALTER COLUMN report_year TYPE HUGEINT"
        )
        target.execute("ALTER TABLE project_timeseries ALTER COLUMN year TYPE HUGEINT")
        target.execute(
            "ALTER TABLE project_timeseries ALTER COLUMN rate_oil TYPE DECIMAL(30,4)"
        )
        target.execute(
            "INSERT INTO project_timeseries (report_year, project_id, year, rate_oil, project_name) "
            "VALUES (2025, 'legacy', 9223372036854775809, '12345678901234567890.1234', 'Legacy')"
        )
        target.execute(
            "CREATE TABLE api_publication (source_table VARCHAR, revision VARCHAR, "
            "source_row_count BIGINT, source_sha256 VARCHAR, invalid_geometry_count BIGINT, "
            "published_at TIMESTAMP)"
        )
        target.execute(
            "INSERT INTO api_publication VALUES ('project_timeseries', ?, 1, ?, 0, current_timestamp)",
            [revision, source_hash],
        )
    finally:
        target.close()
    report = compare_timeseries_projection(
        sqlite_path=sqlite_path, duckdb_path=duckdb_path
    )
    assert report["ok"] is True, report["issues"]
    _mutate(
        duckdb_path, "ALTER TABLE project_timeseries ALTER COLUMN rate_oil TYPE DOUBLE"
    )
    report = compare_timeseries_projection(
        sqlite_path=sqlite_path, duckdb_path=duckdb_path
    )
    assert report["ok"] is False
    issues = report["issues"]
    assert isinstance(issues, list)
    assert any(
        isinstance(issue, str) and "rate_oil" in issue and "type" in issue.lower()
        for issue in issues
    )


def test_bootstrap_date_audits_same_path_and_fresh_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from esdc.analytics_audit import compare_timeseries_projection
    from esdc.analytics_rebuild import rebuild_api_analytics
    from esdc.api_snapshot import (
        ApiProjectResourceRow,
        ApiSnapshot,
        replace_project_resources_snapshot,
    )
    from esdc.configs import Config
    from esdc.dbmanager import _execute_sql_script
    from esdc.pod_registry.store import get_sqlite_connection
    from esdc.timeseries_bootstrap import bootstrap_timeseries_source

    database_dir = tmp_path / "db"
    monkeypatch.setenv("ESDC_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setenv("ESDC_DB_DIR", str(database_dir))
    monkeypatch.setenv("ESDC_DB_FILE", str(database_dir / "active.duckdb"))
    monkeypatch.setattr(Config, "_config_cache", None)

    sqlite_path = tmp_path / "canonical.sqlite"
    legacy_path = tmp_path / "legacy.duckdb"
    fresh_path = tmp_path / "fresh.duckdb"
    legacy = duckdb.connect(str(legacy_path))
    _execute_sql_script(legacy, "create_table_project_timeseries.sql")
    legacy.execute(
        "ALTER TABLE project_timeseries ALTER COLUMN report_date TYPE DATE"
    )
    legacy.execute(
        "INSERT INTO project_timeseries(report_date, report_year, project_id, year) "
        "VALUES (DATE '2025-01-02', 2025, 'legacy-date', 2025)"
    )
    legacy.close()

    bootstrap = bootstrap_timeseries_source(
        duckdb_path=legacy_path, sqlite_path=sqlite_path, apply=True
    )
    assert bootstrap.applied is True
    source = get_sqlite_connection(sqlite_path)
    replace_project_resources_snapshot(
        source,
        [ApiProjectResourceRow(project_id="resource-1", report_year=2025)],
        ApiSnapshot(
            "project_resources",
            "resources-r1",
            1,
            "",
            "2026-10-24T00:00:00Z",
            "committed",
        ),
    )
    source.close()

    rebuild_api_analytics(
        duckdb_path=legacy_path, sqlite_path=sqlite_path, reindex=False
    )
    existing_audit = compare_timeseries_projection(
        sqlite_path=sqlite_path, duckdb_path=legacy_path
    )
    assert existing_audit["ok"] is True
    with duckdb.connect(str(legacy_path), read_only=True) as target:
        existing_type = target.execute(
            "SELECT data_type FROM information_schema.columns "
            "WHERE table_name='project_timeseries' AND column_name='report_date'"
        ).fetchone()
        existing_date = target.execute(
            "SELECT report_date FROM project_timeseries WHERE project_id='legacy-date'"
        ).fetchone()
    assert existing_type == ("DATE",)
    assert existing_date is not None and existing_date[0].isoformat() == "2025-01-02"

    rebuild_api_analytics(
        duckdb_path=fresh_path, sqlite_path=sqlite_path, reindex=False
    )
    fresh_audit = compare_timeseries_projection(
        sqlite_path=sqlite_path, duckdb_path=fresh_path
    )
    assert fresh_audit["ok"] is True
    with duckdb.connect(str(fresh_path), read_only=True) as target:
        fresh_type = target.execute(
            "SELECT data_type FROM information_schema.columns "
            "WHERE table_name='project_timeseries' AND column_name='report_date'"
        ).fetchone()
        fresh_date = target.execute(
            "SELECT report_date FROM project_timeseries WHERE project_id='legacy-date'"
        ).fetchone()
    assert fresh_type == ("VARCHAR",)
    assert fresh_date == ("2025-01-02",)


@pytest.mark.parametrize(
    ("statement", "expected"),
    [
        ("UPDATE api_project_timeseries SET row_sha256='0'", "hash"),
        ("UPDATE api_project_timeseries SET report_year=2024", "report_year"),
    ],
)
def test_source_storage_and_year_index_corruption_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    statement: str,
    expected: str,
) -> None:
    from esdc.analytics_audit import compare_timeseries_projection

    sqlite_path, duckdb_path = _published(tmp_path, monkeypatch)
    source = sqlite3.connect(sqlite_path)
    try:
        source.execute(statement)
        source.commit()
    finally:
        source.close()
    report = compare_timeseries_projection(
        sqlite_path=sqlite_path, duckdb_path=duckdb_path
    )
    assert report["ok"] is False
    issues = report["issues"]
    assert isinstance(issues, list)
    assert any(isinstance(issue, str) and expected in issue for issue in issues)


@pytest.mark.parametrize(
    ("observed_empty", "input_origins"),
    [
        (True, ["api_json"]),
        (False, ["api_json"]),
        (True, []),
        (True, ["unknown"]),
        (True, [["api_json"]]),
    ],
)
def test_empty_timeseries_requires_current_observed_empty_proof(
    tmp_path: Path, observed_empty: bool, input_origins: list[object]
) -> None:
    from esdc.analytics_audit import compare_timeseries_projection
    from esdc.timeseries_snapshot import record_observed_empty_timeseries

    sqlite_path = tmp_path / "empty.sqlite"
    duckdb_path = tmp_path / "empty.duckdb"
    source = sqlite3.connect(sqlite_path)
    ensure_timeseries_tables(source)
    source.execute(
        "CREATE TABLE IF NOT EXISTS api_snapshot (source_table TEXT PRIMARY KEY, revision TEXT, row_count INTEGER, source_sha256 TEXT, fetched_at TEXT, publication_state TEXT, invalid_geometry_count INTEGER)"
    )
    source.execute("BEGIN")
    snapshot = record_observed_empty_timeseries(
        source,
        revision="empty-r1",
        fetched_at="2026-10-24T00:00:00+00:00",
        origin="api_json",
    )
    source.execute(
        "UPDATE api_snapshot SET publication_state='published' WHERE source_table='project_timeseries'"
    )
    if observed_empty is not True or input_origins != ["api_json"]:
        source.execute(
            "UPDATE api_snapshot_provenance SET details_json=? "
            "WHERE source_table='project_timeseries' AND revision=?",
            (
                json.dumps(
                    {"observed_empty": observed_empty, "input_origins": input_origins}
                ),
                snapshot.revision,
            ),
        )
    source.commit()
    source.close()
    target = duckdb.connect(str(duckdb_path))
    try:
        target.execute(Path("esdc/sql/create_table_project_timeseries.sql").read_text())
        target.execute(
            "CREATE TABLE api_publication (source_table VARCHAR, revision VARCHAR, "
            "source_row_count BIGINT, source_sha256 VARCHAR, invalid_geometry_count BIGINT, "
            "published_at TIMESTAMP)"
        )
        target.execute(
            "INSERT INTO api_publication VALUES ('project_timeseries', ?, 0, ?, 0, current_timestamp)",
            [snapshot.revision, snapshot.source_sha256],
        )
    finally:
        target.close()
    report = compare_timeseries_projection(
        sqlite_path=sqlite_path, duckdb_path=duckdb_path
    )
    assert report["ok"] is (observed_empty is True and input_origins == ["api_json"])


def test_retained_csv_rows_survive_json_current_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from esdc.analytics_audit import compare_timeseries_projection
    from esdc.timeseries_snapshot import (
        TIMESERIES_SOURCE_COLUMNS,
        timeseries_fingerprint,
    )

    sqlite_path, duckdb_path = _published(tmp_path, monkeypatch)
    source = sqlite3.connect(sqlite_path)
    try:
        source.execute(
            "INSERT INTO api_project_timeseries "
            "(report_year,row_json,row_sha256,origin,schema_version) "
            "SELECT report_year,row_json,row_sha256,'api_csv',schema_version "
            "FROM api_project_timeseries"
        )
        count, source_hash = timeseries_fingerprint(source)
        source.execute(
            "UPDATE api_snapshot SET row_count=?,source_sha256=? WHERE source_table='project_timeseries'",
            (count, source_hash),
        )
        source.commit()
    finally:
        source.close()
    target = duckdb.connect(str(duckdb_path))
    try:
        columns = ", ".join(f'"{name}"' for name in TIMESERIES_SOURCE_COLUMNS)
        target.execute(
            f"INSERT INTO project_timeseries ({columns}, uuid) "
            f"SELECT {columns}, 'duplicate-uuid' FROM project_timeseries LIMIT 1"
        )
        target.execute(
            "UPDATE api_publication SET source_row_count=?,source_sha256=? "
            "WHERE source_table='project_timeseries'",
            [count, source_hash],
        )
    finally:
        target.close()
    report = compare_timeseries_projection(
        sqlite_path=sqlite_path, duckdb_path=duckdb_path
    )
    assert report["ok"] is True, report["issues"]


def test_audit_streams_more_than_ten_thousand_rows(tmp_path: Path) -> None:
    from esdc.analytics_audit import compare_timeseries_projection

    sqlite_path = tmp_path / "stream.sqlite"
    duckdb_path = tmp_path / "stream.duckdb"
    revision, source_hash = _source(
        sqlite_path,
        (
            (2024, f"P-{index}", index, index / 10, f"row-{index}")
            for index in range(10_001)
        ),
    )
    with sqlite3.connect(sqlite_path) as source:
        target = duckdb.connect(str(duckdb_path))
        try:
            stage = stage_canonical_timeseries(target, source, years=None)
            target.execute("BEGIN")
            apply_timeseries_stage(target, stage)
            target.execute(
                "CREATE TABLE api_publication (source_table VARCHAR, revision VARCHAR, "
                "source_row_count BIGINT, source_sha256 VARCHAR, "
                "invalid_geometry_count BIGINT, published_at TIMESTAMP)"
            )
            target.execute(
                "INSERT INTO api_publication VALUES ('project_timeseries', ?, 10001, ?, 0, current_timestamp)",
                [revision, source_hash],
            )
            target.execute("COMMIT")
        finally:
            target.close()
    report = compare_timeseries_projection(
        sqlite_path=sqlite_path, duckdb_path=duckdb_path
    )
    assert report["canonical_rows"] == 10_001
    assert report["projected_rows"] == 10_001
    assert report["ok"] is True, report["issues"]


def test_identity_digest_requests_bounded_batches() -> None:
    from esdc.analytics_rehearsal import _identity_digest

    class Cursor:
        def __init__(self) -> None:
            self.remaining = 2_501
            self.requests: list[int] = []

        def fetchmany(self, size: int) -> list[tuple[int]]:
            self.requests.append(size)
            count = min(size, self.remaining)
            start = 2_501 - self.remaining
            self.remaining -= count
            return [(value,) for value in range(start, start + count)]

    cursor = Cursor()
    result = _identity_digest(cursor)
    assert result["count"] == 2_501
    samples = result["samples"]
    assert isinstance(samples, list) and len(samples) == 10
    assert cursor.requests and max(cursor.requests) <= 1_000
