"""Regression tests for the read-only project-resource parity audit."""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

import duckdb

from esdc.api_projection import publish_project_resources
from esdc.api_snapshot import (
    ApiProjectResourceRow,
    ApiSnapshot,
    get_project_resources_snapshot,
    replace_project_resources_snapshot,
)
from esdc.pod_registry.store import get_sqlite_connection
from esdc.resource_audit import compare_resource_projection


def _source(path: Path, rows: list[ApiProjectResourceRow]) -> str:
    conn = get_sqlite_connection(path)
    try:
        replace_project_resources_snapshot(
            conn,
            rows,
            ApiSnapshot(
                "project_resources",
                "incoming",
                len(rows),
                "ignored",
                "now",
                "committed",
            ),
        )
        snapshot = get_project_resources_snapshot(conn)
        assert snapshot is not None
        return snapshot.revision
    finally:
        conn.close()


def _published(tmp_path: Path) -> tuple[Path, duckdb.DuckDBPyConnection]:
    sqlite_path = tmp_path / "source.sqlite"
    duckdb_path = tmp_path / "mirror.duckdb"
    revision = _source(
        sqlite_path,
        [
            ApiProjectResourceRow(
                project_id="00017",
                report_year=2025,
                project_name="Cendana",
                field_geom="POINT (106 2)",
                groovy_report="Current situation:\n  Stable",
                project_class="2P",
                project_level="X-1",
                is_discovered=0,
                project_stage="source stage",
                rec_oil=0.0,
            ),
            ApiProjectResourceRow(
                project_id="bad-wkt",
                report_year=2024,
                field_geom="NOT WKT",
                groovy_report="Future strategy:\n  Drill well",
            ),
        ],
    )
    publish_project_resources(duckdb_path, sqlite_path, revision)
    return sqlite_path, duckdb.connect(str(duckdb_path))


def test_resource_parity_compares_canonical_typed_rows_and_metadata(
    tmp_path: Path,
) -> None:
    sqlite_path, conn = _published(tmp_path)
    before_hash = hashlib.sha256(sqlite_path.read_bytes()).hexdigest()
    try:
        report = compare_resource_projection(conn, sqlite_path)
        assert report.ok
        assert report.canonical_rows == report.projected_rows == 2
        assert report.identity_difference_count == 0
        assert report.source_multiset_difference_count == 0
        assert report.metadata_matches
        assert report.invalid_geometry_count == 1
        rows = conn.execute(
            "SELECT project_id, is_discovered, project_stage, field_geom, "
            "ST_GeometryType(field_geometry), ST_AsText(field_geometry) "
            "FROM project_resources ORDER BY project_id"
        ).fetchall()
        assert rows == [
            ("00017", 1, "2. Exploration", "POINT (106 2)", "POINT", "POINT (106 2)"),
            ("bad-wkt", None, None, "NOT WKT", None, None),
        ]
        assert conn.execute(
            "SELECT typeof(rec_oil) FROM project_resources WHERE project_id='00017'"
        ).fetchone() == ("FLOAT",)
        assert conn.execute(
            "SELECT typeof(report_year) FROM project_resources WHERE project_id='00017'"
        ).fetchone() == ("INTEGER",)
        assert conn.execute(
            "SELECT typeof(project_id) FROM project_resources WHERE project_id='00017'"
        ).fetchone() == ("VARCHAR",)
        assert hashlib.sha256(sqlite_path.read_bytes()).hexdigest() == before_hash
        conn.execute(
            "UPDATE project_resources SET field_geometry=NULL WHERE project_id='00017'"
        )
        geometry_report = compare_resource_projection(conn, sqlite_path)
        assert not geometry_report.ok
        assert any("Valid raw WKT" in issue for issue in geometry_report.issues)
    finally:
        conn.close()


def test_resource_parity_detects_wrong_values_when_counts_and_lineage_match(
    tmp_path: Path,
) -> None:
    sqlite_path, conn = _published(tmp_path)
    try:
        conn.execute(
            "UPDATE project_resources SET project_name='Wrong' WHERE project_id='00017'"
        )
        report = compare_resource_projection(conn, sqlite_path)
        assert not report.ok
        assert report.projected_rows == report.canonical_rows == 2
        assert report.metadata_matches
        assert report.source_multiset_difference_count > 0
    finally:
        conn.close()


def test_resource_parity_rejects_wrong_target_types_and_missing_columns(
    tmp_path: Path,
) -> None:
    sqlite_path, conn = _published(tmp_path)
    conn.close()
    source = duckdb.connect(str(tmp_path / "mirror.duckdb"), read_only=True)
    malformed = duckdb.connect()
    try:
        malformed.execute(
            "ATTACH ? AS original (READ_ONLY)", [str(tmp_path / "mirror.duckdb")]
        )
    except duckdb.Error:
        # ATTACH parameters are unsupported by some bundled DuckDB versions.
        malformed.close()
        malformed = duckdb.connect()
        malformed.execute(
            f"ATTACH '{(tmp_path / 'mirror.duckdb').as_posix()}' AS original (READ_ONLY)"
        )
    try:
        malformed.execute(
            "CREATE TABLE project_resources AS SELECT * REPLACE "
            "(report_year::VARCHAR AS report_year, "
            "is_discovered::VARCHAR AS is_discovered) FROM original.project_resources"
        )
        report = compare_resource_projection(malformed, sqlite_path)
        assert not report.ok
        assert any(
            "type" in issue.lower() and "report_year" in issue
            for issue in report.issues
        )
        assert any(
            "type" in issue.lower() and "is_discovered" in issue
            for issue in report.issues
        )
    finally:
        source.close()
        malformed.close()

    sqlite_path, conn = _published(tmp_path)
    conn.close()
    malformed = duckdb.connect()
    try:
        malformed.execute(
            f"ATTACH '{(tmp_path / 'mirror.duckdb').as_posix()}' AS original (READ_ONLY)"
        )
        malformed.execute(
            "CREATE TABLE project_resources AS SELECT * EXCLUDE (ghv_avg) "
            "FROM original.project_resources"
        )
        report = compare_resource_projection(malformed, sqlite_path)
        assert not report.ok
        assert any("ghv_avg" in issue for issue in report.issues)
    finally:
        malformed.close()


def test_audit_is_read_only_when_sqlite_lineage_is_missing(tmp_path: Path) -> None:
    sqlite_path = tmp_path / "unpublished.sqlite"
    conn_sql = sqlite3.connect(sqlite_path)
    conn_sql.execute("CREATE TABLE another_table (item TEXT)")
    conn_sql.execute("INSERT INTO another_table VALUES ('keep')")
    conn_sql.commit()
    conn_sql.close()
    before = hashlib.sha256(sqlite_path.read_bytes()).hexdigest()
    duck = duckdb.connect()
    try:
        report = compare_resource_projection(duck, sqlite_path)
        assert not report.ok
        assert not report.metadata_matches
    finally:
        duck.close()
    after = hashlib.sha256(sqlite_path.read_bytes()).hexdigest()
    assert after == before


def test_resource_parity_preserves_duplicate_identity_multiplicity(
    tmp_path: Path,
) -> None:
    sqlite_path = tmp_path / "duplicates.sqlite"
    duckdb_path = tmp_path / "duplicates.duckdb"
    rows = [
        ApiProjectResourceRow(
            project_id="duplicate",
            report_year=2025,
            uncert_level=None,
            project_name=name,
            project_class="2p",
            project_level="e",
            is_discovered=1,
            project_stage="original",
        )
        for name in ("first", "second")
    ]
    revision = _source(sqlite_path, rows)
    publish_project_resources(duckdb_path, sqlite_path, revision)
    conn = duckdb.connect(str(duckdb_path))
    try:
        report = compare_resource_projection(conn, sqlite_path)
        assert report.ok
        assert report.canonical_rows == 2
        assert report.identity_difference_count == 0
        assert conn.execute(
            "SELECT DISTINCT is_discovered, project_stage FROM project_resources"
        ).fetchall() == [(1, None)]
        conn.execute(
            "UPDATE project_resources SET project_id='substitute' "
            "WHERE project_name='second'"
        )
        report = compare_resource_projection(conn, sqlite_path)
        assert not report.ok
        assert report.identity_difference_count > 0
        assert report.source_multiset_difference_count > 0
        assert report.projected_rows == report.canonical_rows == 2
    finally:
        conn.close()


def test_audit_rejects_derived_geometry_for_missing_or_blank_raw_wkt_read_only(
    tmp_path: Path,
) -> None:
    sqlite_path = tmp_path / "empty-wkt.sqlite"
    duckdb_path = tmp_path / "empty-wkt.duckdb"
    revision = _source(
        sqlite_path,
        [
            ApiProjectResourceRow(project_id="null-wkt", field_geom=None),
            ApiProjectResourceRow(project_id="blank-wkt", field_geom=" \t"),
        ],
    )
    publish_project_resources(duckdb_path, sqlite_path, revision)
    writable = duckdb.connect(str(duckdb_path))
    try:
        writable.execute("LOAD spatial")
        writable.execute(
            "UPDATE project_resources SET field_geometry=ST_GeomFromText('POINT (1 2)')"
        )
    finally:
        writable.close()

    source_hash_before = hashlib.sha256(sqlite_path.read_bytes()).hexdigest()
    mirror_hash_before = hashlib.sha256(duckdb_path.read_bytes()).hexdigest()
    readonly = duckdb.connect(str(duckdb_path), read_only=True)
    try:
        report = compare_resource_projection(readonly, sqlite_path)
        assert not report.ok
        assert any("empty raw wkt" in issue.lower() for issue in report.issues)
    finally:
        readonly.close()
    assert hashlib.sha256(sqlite_path.read_bytes()).hexdigest() == source_hash_before
    assert hashlib.sha256(duckdb_path.read_bytes()).hexdigest() == mirror_hash_before


def test_audit_with_missing_sqlite_lineage_is_read_only_for_both_files(
    tmp_path: Path,
) -> None:
    sqlite_path = tmp_path / "missing-lineage.sqlite"
    duckdb_path = tmp_path / "missing-lineage.duckdb"
    revision = _source(
        sqlite_path,
        [ApiProjectResourceRow(project_id="P-1", report_year=2025)],
    )
    publish_project_resources(duckdb_path, sqlite_path, revision)
    sqlite = sqlite3.connect(sqlite_path)
    sqlite.execute("DROP TABLE api_snapshot")
    sqlite.commit()
    sqlite.close()

    source_hash_before = hashlib.sha256(sqlite_path.read_bytes()).hexdigest()
    mirror_hash_before = hashlib.sha256(duckdb_path.read_bytes()).hexdigest()
    readonly = duckdb.connect(str(duckdb_path), read_only=True)
    try:
        report = compare_resource_projection(readonly, sqlite_path)
        assert not report.ok
        assert report.canonical_rows == report.projected_rows == 1
        assert any("snapshot lineage is missing" in issue for issue in report.issues)
    finally:
        readonly.close()
    assert hashlib.sha256(sqlite_path.read_bytes()).hexdigest() == source_hash_before
    assert hashlib.sha256(duckdb_path.read_bytes()).hexdigest() == mirror_hash_before
