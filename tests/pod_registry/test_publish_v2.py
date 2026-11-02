"""Tests for publishing the canonical POD v2 snapshot."""

from __future__ import annotations

import sqlite3
from datetime import date
from decimal import Decimal
from pathlib import Path

import duckdb
import pytest

from esdc.pod_registry.identity import ApprovalKey, issue_identity
from esdc.pod_registry.migrations import initialize_registry_v2
from esdc.pod_registry.publication import PodPublicationReport
from esdc.pod_registry.publish import publish_pod_registry

# Minimal old-shaped fixture SQL: test material only. The store no longer carries
# a superseded-schema constant, so refusal fixtures declare the shape they want.
_OLD_SHAPED_FIXTURE_SQL = """
CREATE TABLE m_pod (
    id INTEGER PRIMARY KEY,
    pod_id TEXT NOT NULL UNIQUE,
    pod_name TEXT,
    approval_seq INTEGER
);
CREATE TABLE project_pod (
    pod_id INTEGER NOT NULL REFERENCES m_pod(id),
    project_id TEXT NOT NULL,
    UNIQUE (pod_id, project_id)
);
CREATE TABLE pod_revision (
    successor_id TEXT NOT NULL,
    predecessor_id TEXT NOT NULL,
    UNIQUE (successor_id, predecessor_id)
);
"""


def _publish(source, target) -> PodPublicationReport:
    report = publish_pod_registry(source, target)
    assert isinstance(report, PodPublicationReport)
    return report


def _source(path):
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA foreign_keys = ON")
    initialize_registry_v2(conn)
    conn.execute(
        "INSERT INTO m_project (project_id, project_name, is_active) "
        "VALUES ('P-0000001-01', 'Project One', 1)"
    )
    conn.commit()
    conn.execute("BEGIN IMMEDIATE")
    identity = issue_identity(
        conn,
        key=ApprovalKey(date(2024, 1, 1), 1, 1, 1, 0),
        actor="FA:test-publication",
    )
    conn.execute(
        "INSERT INTO m_pod (pod_id_seq, pod_uuid, approval_date, institution_id, "
        "pod_type_id, rev_num, pod_order_no, pod_id, pod_name, pod_letter_num, "
        "is_latest, capital_drilling_pod) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            identity.pod_id_seq,
            str(identity.pod_uuid),
            "2024-01-01",
            1,
            1,
            0,
            identity.pod_order_no,
            identity.pod_id,
            "Project One POD",
            "LETTER-1",
            1,
            "123.450",
        ),
    )
    conn.execute(
        "UPDATE pod_registry_state SET source_generation = 3 WHERE singleton_id = 1"
    )
    conn.commit()
    conn.close()


def test_v2_publication_replaces_catalog_views_with_typed_snapshot(tmp_path):
    source = tmp_path / "source.sqlite"
    target = tmp_path / "analytics.duckdb"
    _source(source)
    existing = duckdb.connect(str(target))
    existing.execute("CREATE VIEW m_project AS SELECT 'stale' AS project_id")
    existing.execute("CREATE TABLE pod_registry (stale INTEGER)")
    existing.close()

    report = _publish(source, target)

    assert report.source_committed is True
    assert report.source_generation == 3
    assert report.published_generation == 3
    assert report.projection_error is None
    conn = duckdb.connect(str(target), read_only=True)
    try:
        objects = {
            row[0]: row[1]
            for row in conn.execute(
                "SELECT table_name, table_type FROM information_schema.tables "
                "WHERE table_schema = 'main'"
            ).fetchall()
        }
        assert objects["m_project"] == "BASE TABLE"
        assert objects["m_pod"] == "BASE TABLE"
        types = {
            (row[0], row[1]): row[2]
            for row in conn.execute(
                "SELECT table_name, column_name, data_type "
                "FROM information_schema.columns WHERE table_schema = 'main'"
            ).fetchall()
        }
        assert types[("m_pod", "approval_date")] == "DATE"
        assert types[("m_pod", "is_latest")] == "BOOLEAN"
        assert types[("m_pod", "capital_drilling_pod")] == "DECIMAL(20,3)"
        assert conn.execute(
            "SELECT approval_date, is_latest, capital_drilling_pod FROM m_pod"
        ).fetchone() == (date(2024, 1, 1), True, Decimal("123.450"))
        assert conn.execute(
            "SELECT pod_id_seq, doc_id FROM pod_document"
        ).fetchall() == []
        assert conn.execute(
            "SELECT source_generation FROM pod_registry_publication"
        ).fetchone() == (3,)
    finally:
        conn.close()

def test_v2_rejects_dictionary_mismatch_before_duckdb_mutation(tmp_path):
    source = tmp_path / "source.sqlite"
    target = tmp_path / "analytics.duckdb"
    _source(source)
    writer = sqlite3.connect(source)
    writer.execute(
        "UPDATE pod_registry_state SET dictionary_version='0.3.9' WHERE singleton_id=1"
    )
    writer.commit()
    writer.close()
    existing = duckdb.connect(str(target))
    existing.execute("CREATE VIEW m_project AS SELECT 'stale' AS project_id")
    existing.execute("CREATE TABLE pod_registry (stale INTEGER)")
    existing.close()

    report = _publish(source, target)

    assert report.source_committed is False
    error = report.projection_error
    assert error is not None
    assert "dictionary marker" in error
    conn = duckdb.connect(str(target), read_only=True)
    try:
        assert conn.execute("SELECT project_id FROM m_project").fetchall() == [("stale",)]
        assert conn.execute("SELECT stale FROM pod_registry").fetchall() == []
    finally:
        conn.close()


def test_v2_publication_rejects_active_borrowed_transaction_and_releases_its_own(
    tmp_path,
):
    from esdc.pod_registry.publication import publish_v2_registry

    source = tmp_path / "source.sqlite"
    target = tmp_path / "analytics.duckdb"
    _source(source)
    borrowed = sqlite3.connect(source)
    borrowed.row_factory = sqlite3.Row
    report = publish_v2_registry(source, target, sqlite_conn=borrowed)
    assert report.projection_error is None
    assert borrowed.in_transaction is False
    assert borrowed.execute(
        "SELECT source_generation FROM pod_registry_state WHERE singleton_id=1"
    ).fetchone()[0] == 3
    borrowed.close()

    active = sqlite3.connect(source)
    active.row_factory = sqlite3.Row
    active.execute("UPDATE pod_registry_state SET source_generation=99")
    report = publish_v2_registry(source, target, sqlite_conn=active)
    assert report.source_committed is False
    error = report.projection_error
    assert error is not None
    assert "active transaction" in error
    assert active.in_transaction is True
    assert active.execute(
        "SELECT source_generation FROM pod_registry_state WHERE singleton_id=1"
    ).fetchone()[0] == 99
    active.rollback()
    assert active.execute(
        "SELECT source_generation FROM pod_registry_state WHERE singleton_id=1"
    ).fetchone()[0] == 3
    active.close()


def test_v2_duckdb_setup_failure_reports_valid_captured_source(tmp_path, monkeypatch):
    from esdc.pod_registry import publication

    source = tmp_path / "source.sqlite"
    target = tmp_path / "analytics.duckdb"
    _source(source)

    def fail_connect(*args, **kwargs):
        raise duckdb.IOException("simulated writer lock")

    monkeypatch.setattr(publication.duckdb, "connect", fail_connect)
    report = publication.publish_v2_registry(source, target)

    assert report.source_committed is True
    assert report.source_generation == 3
    assert report.published_generation is None
    error = report.projection_error
    assert error is not None
    assert "writer lock" in error


def test_v2_duckdb_failure_rolls_back_replaced_catalog_objects(tmp_path, monkeypatch):
    from esdc.pod_registry import publication

    source = tmp_path / "source.sqlite"
    target = tmp_path / "analytics.duckdb"
    _source(source)
    existing = duckdb.connect(str(target))
    existing.execute("CREATE VIEW m_project AS SELECT 'stale' AS project_id")
    existing.execute("CREATE TABLE pod_registry (stale INTEGER)")
    existing.close()
    original = publication._create_table_from_sqlite

    def fail_after_project(target_conn, source_conn, table, *args, **kwargs):
        if table == "m_pod":
            raise RuntimeError("simulated table-copy failure")
        return original(target_conn, source_conn, table, *args, **kwargs)

    monkeypatch.setattr(publication, "_create_table_from_sqlite", fail_after_project)
    report = _publish(source, target)
    assert report.source_committed is True
    error = report.projection_error
    assert error is not None
    assert "simulated table-copy failure" in error

    conn = duckdb.connect(str(target), read_only=True)
    try:
        assert conn.execute("SELECT project_id FROM m_project").fetchall() == [("stale",)]
        assert conn.execute("SELECT stale FROM pod_registry").fetchall() == []
    finally:
        conn.close()


def test_v2_metadata_is_replaced_and_fast_status_reads_only_markers(
    tmp_path, monkeypatch
):
    from esdc.loaders import _METADATA_TABLE
    from esdc.pod_registry import publication

    source = tmp_path / "source.sqlite"
    target = tmp_path / "analytics.duckdb"
    _source(source)
    old = duckdb.connect(str(target))
    old.execute(
        f"CREATE TABLE {_METADATA_TABLE} "
        "(table_name VARCHAR, description VARCHAR, sheet_name VARCHAR, "
        "schema_yaml VARCHAR, loaded_at VARCHAR)"
    )
    old.execute(
        f"INSERT INTO {_METADATA_TABLE} VALUES "
        "('pod_economics','stale','x','{}','old')"
    )
    old.close()
    report = _publish(source, target)
    assert report.projection_error is None

    conn = duckdb.connect(str(target), read_only=True)
    try:
        names = {
            row[0]
            for row in conn.execute(f"SELECT table_name FROM {_METADATA_TABLE}").fetchall()
        }
        assert "m_pod" in names
        assert "pod_registry" in names
        assert "pod_economics" not in names
    finally:
        conn.close()

    def fail_full_audit(*args, **kwargs):
        raise AssertionError("fast status must not scan the full source")

    monkeypatch.setattr(publication, "_validate_source", fail_full_audit)
    status = publication.get_pod_registry_status(source, target)
    assert status.state == "fresh"


def test_v2_newer_source_during_publish_is_left_marked_stale(tmp_path, monkeypatch):
    from esdc.pod_registry import publication

    source = tmp_path / "source.sqlite"
    target = tmp_path / "analytics.duckdb"
    _source(source)
    original = publication._publish_snapshot

    def advance_source_then_publish(
        target_conn,
        source_conn,
        contract,
        generation,
        source_sha256,
        source_tables,
        source_updated_at,
        read_model_marker,
        db_path,
    ):
        writer = sqlite3.connect(source)
        writer.execute("PRAGMA journal_mode=WAL")
        writer.execute(
            "UPDATE pod_registry_state SET source_generation=4, "
            "updated_at='2026-10-29T01:00:00Z' WHERE singleton_id=1"
        )
        writer.commit()
        writer.close()
        return original(
            target_conn,
            source_conn,
            contract,
            generation,
            source_sha256,
            source_tables,
            source_updated_at,
            read_model_marker,
            db_path,
        )

    monkeypatch.setattr(publication, "_publish_snapshot", advance_source_then_publish)
    report = _publish(source, target)
    assert report.source_committed is True
    assert report.source_generation == 3
    assert report.published_generation == 3
    assert publication.get_pod_registry_status(source, target).state == "stale"

@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("capital_drilling_pod", "1.2345"),
        ("res_oil_2p_pod", "Inf"),
    ],
)
def test_v2_invalid_numeric_contract_value_preserves_good_projection(
    tmp_path, column, value
):
    source = tmp_path / "source.sqlite"
    target = tmp_path / "analytics.duckdb"
    _source(source)
    good = _publish(source, target)
    assert good.projection_error is None
    before = duckdb.connect(str(target), read_only=True)
    try:
        original_row = before.execute(
            f'SELECT "{column}" FROM m_pod'
        ).fetchone()
        assert original_row is not None
        original_value = original_row[0]
    finally:
        before.close()

    writer = sqlite3.connect(source)
    writer.execute(f'UPDATE m_pod SET "{column}"=?', (value,))
    writer.commit()
    source_row = writer.execute(
        f'SELECT "{column}" FROM m_pod'
    ).fetchone()
    assert source_row is not None
    source_value = source_row[0]
    writer.close()

    report = _publish(source, target)

    assert report.source_committed is False
    assert report.projection_error is not None
    assert column in report.projection_error
    writer = sqlite3.connect(source)
    persisted_row = writer.execute(
        f'SELECT "{column}" FROM m_pod'
    ).fetchone()
    assert persisted_row is not None
    assert persisted_row[0] == source_value
    writer.close()
    after = duckdb.connect(str(target), read_only=True)
    try:
        restored = after.execute(f'SELECT "{column}" FROM m_pod').fetchone()
        assert restored is not None
        assert restored[0] == original_value
        publication_row = after.execute(
            "SELECT source_generation FROM pod_registry_publication"
        ).fetchone()
        assert publication_row is not None
        assert publication_row == (3,)
    finally:
        after.close()


def _drift_declared_storage_type(
    path: Path, table: str, column: str, drifted_type: str
) -> None:
    """Declare a drifted physical storage type in an owned disposable fixture."""
    conn = sqlite3.connect(path)
    try:
        actual = conn.execute(f'PRAGMA table_info("{table}")').fetchall()
        current = {row[1]: row[2] for row in actual}[column]
        stored = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?",
            (table,),
        ).fetchone()[0]
        conn.execute("PRAGMA writable_schema = ON")
        conn.execute(
            "UPDATE sqlite_master SET sql = ? WHERE type = 'table' AND name = ?",
            (
                stored.replace(f"{column} {current}", f"{column} {drifted_type}", 1),
                table,
            ),
        )
        conn.execute("PRAGMA writable_schema = OFF")
        conn.commit()
    finally:
        conn.close()


def test_declared_storage_type_drift_is_refused_before_duckdb_mutation(
    tmp_path, monkeypatch
):
    from esdc.pod_registry import publication

    source = tmp_path / "drift-source.sqlite"
    target = tmp_path / "analytics.duckdb"
    exact = "12345678901234567.123"

    _source(source)
    writer = sqlite3.connect(source)
    writer.execute("UPDATE m_pod SET capital_drilling_pod = ?", (exact,))
    writer.commit()
    writer.close()

    good = _publish(source, target)
    assert good.projection_error is None
    before = duckdb.connect(str(target), read_only=True)
    try:
        prior_value = before.execute(
            "SELECT capital_drilling_pod FROM m_pod"
        ).fetchone()
        prior_marker = before.execute(
            "SELECT source_generation FROM pod_registry_publication"
        ).fetchone()
    finally:
        before.close()
    assert prior_value == (Decimal(exact),)
    assert prior_marker == (3,)

    _drift_declared_storage_type(source, "m_pod", "capital_drilling_pod", "NUMERIC")
    writer = sqlite3.connect(source)
    writer.execute("UPDATE m_pod SET capital_drilling_pod = ?", (exact,))
    writer.commit()
    # NUMERIC affinity already lost the exact contract precision in the source.
    assert writer.execute("SELECT capital_drilling_pod FROM m_pod").fetchone() == (
        12345678901234568,
    )
    writer.close()

    def reject_duckdb_open(*args, **kwargs):
        raise AssertionError("a refused source must not open the DuckDB target")

    monkeypatch.setattr(publication.duckdb, "connect", reject_duckdb_open)
    report = publish_pod_registry(source, target)
    monkeypatch.undo()

    assert report.source_committed is False
    assert report.projection_error is not None
    assert "m_pod" in report.projection_error
    assert "capital_drilling_pod" in report.projection_error
    assert "NUMERIC" in report.projection_error
    assert "TEXT" in report.projection_error

    after = duckdb.connect(str(target), read_only=True)
    try:
        assert after.execute("SELECT capital_drilling_pod FROM m_pod").fetchone() == (
            Decimal(exact),
        )
        assert after.execute(
            "SELECT source_generation FROM pod_registry_publication"
        ).fetchone() == (3,)
    finally:
        after.close()




@pytest.mark.parametrize(
    "source_kind", ["missing", "empty", "corrupt", "wrong-shape", "foreign-key"]
)
def test_invalid_source_does_not_clear_existing_projection(
    tmp_path, monkeypatch, source_kind
):
    from esdc.pod_registry import publication

    source = tmp_path / "source.sqlite"
    target = tmp_path / "analytics.duckdb"
    _source(source)
    published = _publish(source, target)
    assert published.projection_error is None

    before = duckdb.connect(str(target), read_only=True)
    try:
        expected_rows = before.execute(
            "SELECT pod_id, pod_name FROM pod_registry ORDER BY pod_id"
        ).fetchall()
    finally:
        before.close()

    invalid = tmp_path / f"{source_kind}.sqlite"
    if source_kind == "empty":
        sqlite3.connect(invalid).close()
    elif source_kind == "corrupt":
        invalid.write_bytes(b"not a SQLite database")
    elif source_kind == "wrong-shape":
        wrong_shape = sqlite3.connect(invalid)
        wrong_shape.execute("CREATE TABLE m_pod (id INTEGER)")
        wrong_shape.commit()
        wrong_shape.close()
    elif source_kind == "foreign-key":
        invalid_source = sqlite3.connect(invalid)
        invalid_source.executescript(_OLD_SHAPED_FIXTURE_SQL)
        invalid_source.execute("PRAGMA foreign_keys=OFF")
        invalid_source.execute(
            "INSERT INTO project_pod (pod_id, project_id) VALUES (999, 'P-9999999-99')"
        )
        invalid_source.commit()
        invalid_source.close()

    def reject_duckdb_open(*args, **kwargs):
        raise AssertionError("invalid source must be rejected before DuckDB open")

    monkeypatch.setattr(publication.duckdb, "connect", reject_duckdb_open)
    report = _publish(invalid, target)

    assert report.projection_error is not None
    monkeypatch.undo()

    after = duckdb.connect(str(target), read_only=True)
    try:
        assert after.execute(
            "SELECT pod_id, pod_name FROM pod_registry ORDER BY pod_id"
        ).fetchall() == expected_rows
    finally:
        after.close()


@pytest.mark.parametrize("source_kind", ["missing", "empty", "corrupt"])
def test_publication_rejects_invalid_source_before_replacing_target_objects(
    tmp_path, source_kind
):
    """The single path refuses a bad source without touching the target."""
    from esdc.pod_registry.publication import publish_v2_registry

    target = tmp_path / "analytics.duckdb"
    conn = duckdb.connect(str(target))
    conn.execute("CREATE TABLE pod_registry (pod_id VARCHAR, pod_name VARCHAR)")
    conn.execute("INSERT INTO pod_registry VALUES ('POD-1', 'Keep me')")

    invalid = tmp_path / f"direct-{source_kind}.sqlite"
    if source_kind == "empty":
        sqlite3.connect(invalid).close()
    elif source_kind == "corrupt":
        invalid.write_bytes(b"not a SQLite database")

    report = publish_v2_registry(invalid, target, duck_conn=conn)

    assert report.projection_error is not None
    assert conn.execute("SELECT pod_id, pod_name FROM pod_registry").fetchall() == [
        ("POD-1", "Keep me")
    ]
    conn.close()


def test_metadata_view_is_rejected_without_clearing_projection(tmp_path):
    source = tmp_path / "source.sqlite"
    target = tmp_path / "analytics.duckdb"
    _source(source)
    published = _publish(source, target)
    assert published.projection_error is None

    writer = sqlite3.connect(source)
    writer.execute("DROP TABLE pod_registry_state")
    writer.execute(
        "CREATE VIEW pod_registry_state AS SELECT "
        "1 AS singleton_id, 2 AS schema_version, 'test' AS contract_id, "
        "'v2' AS contract_version, '0.4.0' AS dictionary_version, "
        "3 AS source_generation, 'now' AS updated_at"
    )
    writer.commit()
    writer.close()

    report = _publish(source, target)

    assert report.projection_error is not None
    after = duckdb.connect(str(target), read_only=True)
    try:
        assert after.execute("SELECT pod_id FROM pod_registry").fetchall()
        publication_row = after.execute(
            "SELECT source_generation FROM pod_registry_publication"
        ).fetchone()
        assert publication_row is not None
        assert publication_row == (3,)
    finally:
        after.close()



def test_fully_initialized_empty_v2_source_is_publishable(tmp_path):
    source = tmp_path / "empty-v2.sqlite"
    target = tmp_path / "analytics.duckdb"
    conn = sqlite3.connect(source)
    initialize_registry_v2(conn)
    conn.close()

    report = _publish(source, target)

    assert report.projection_error is None
    mirror = duckdb.connect(str(target), read_only=True)
    try:
        assert mirror.execute("SELECT COUNT(*) FROM m_pod").fetchone() == (0,)
        assert mirror.execute(
            "SELECT source_generation FROM pod_registry_publication"
        ).fetchone() == (0,)
    finally:
        mirror.close()
