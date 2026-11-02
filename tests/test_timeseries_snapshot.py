"""Behavioral tests for canonical SQLite timeseries snapshots."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from typing import cast

import pytest

from esdc.fetching import (
    TIMESERIES_COLUMNS,
    TIMESERIES_SOURCE_COLUMNS,
    TIMESERIES_SOURCE_TYPES,
)
from esdc.timeseries_snapshot import (
    CanonicalTimeseriesRow,
    get_timeseries_snapshot,
    iter_timeseries_rows,
    normalize_timeseries_row,
    replace_timeseries_in_transaction,
    timeseries_fingerprint,
)


def _row(**values):
    return normalize_timeseries_row(
        tuple(values), tuple(values.values()), origin="api_json"
    )


def _new_connection() -> sqlite3.Connection:
    from esdc.pod_registry.store import ensure_tables

    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA foreign_keys = ON")
    ensure_tables(conn)
    return conn


def test_round_trip_preserves_scalars_and_duplicate_rows() -> None:
    conn = _new_connection()
    row = _row(
        report_year="2025",
        project_id="001",
        cprd_grs_oil=0,
        project_name="",
        field_name=None,
        project_remarks="baris satu\nbaris dua — minyak",
        is_offshore=True,
        rate_oil="0.125",
    )
    conn.execute("BEGIN")
    replace_timeseries_in_transaction(
        conn, iter((row, row)), years=None, revision="r1", fetched_at="now"
    )
    conn.commit()

    persisted = list(iter_timeseries_rows(conn))
    assert len(persisted) == 2
    assert json.loads(persisted[0].row_json) == json.loads(row.row_json)
    assert json.loads(persisted[1].row_json) == json.loads(row.row_json)
    actual = json.loads(persisted[0].row_json)
    assert actual["report_year"] == 2025
    assert actual["project_id"] == "001"
    assert actual["cprd_grs_oil"] == 0
    assert actual["project_name"] == ""
    assert actual["field_name"] is None
    assert actual["project_remarks"] == "baris satu\nbaris dua — minyak"
    assert actual["is_offshore"] is True
    assert actual["rate_oil"] == "0.125"
    records = sorted(item.row_json.encode("utf-8") for item in persisted)
    expected_hash = hashlib.sha256(b"[" + b",".join(records) + b"]").hexdigest()
    assert timeseries_fingerprint(conn) == (2, expected_hash)
    snapshot = get_timeseries_snapshot(conn)
    assert snapshot is not None
    assert snapshot.row_count == 2


def test_api_requires_valid_year_but_legacy_allows_null() -> None:
    with pytest.raises(ValueError, match="report_year"):
        normalize_timeseries_row(("report_year",), (None,), origin="api_json")
    legacy = normalize_timeseries_row(
        ("project_id",), ("legacy",), origin="legacy_duckdb"
    )
    assert legacy.report_year is None


def test_nonfinite_and_unknown_fields_rejected() -> None:
    for value in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError, match="column cprd_grs_oil"):
            normalize_timeseries_row(
                ("report_year", "cprd_grs_oil"), (2025, value), origin="api_json"
            )
    with pytest.raises(ValueError, match="unknown.*surprise"):
        normalize_timeseries_row(("surprise",), ("x",), origin="api_json")


def test_source_column_set_matches_ddl() -> None:
    import re
    from pathlib import Path

    ddl = Path("esdc/sql/create_table_project_timeseries.sql").read_text()
    source_columns = tuple(
        (match.group(1), match.group(2))
        for match in re.finditer(
            r"^\s{4}([a-z][a-z0-9_]*)\s+(TEXT|INTEGER|REAL)", ddl, re.M
        )
        if match.group(1) not in {"id", "uuid"}
    )
    assert set(TIMESERIES_SOURCE_COLUMNS) == TIMESERIES_COLUMNS
    assert tuple(name for name, _ in source_columns) == TIMESERIES_SOURCE_COLUMNS
    assert tuple(TIMESERIES_SOURCE_TYPES.items()) == source_columns


def test_selected_replace_preserves_other_and_null_years() -> None:
    conn = _new_connection()
    rows = (
        normalize_timeseries_row(
            ("report_year", "project_id"), (2018, "keep"), origin="legacy_duckdb"
        ),
        normalize_timeseries_row(
            ("project_id",), ("null-year",), origin="legacy_duckdb"
        ),
        normalize_timeseries_row(
            ("report_year", "project_id"), (2025, "replace"), origin="api_json"
        ),
    )
    conn.execute("BEGIN")
    replace_timeseries_in_transaction(
        conn, rows, years=None, revision="r1", fetched_at="now"
    )
    conn.commit()
    replacement = _row(report_year=2025, project_id="new")
    conn.execute("BEGIN")
    replace_timeseries_in_transaction(
        conn, (replacement,), years=(2025,), revision="r2", fetched_at="now"
    )
    conn.commit()
    stored = list(iter_timeseries_rows(conn))
    assert {item.report_year for item in stored} == {2018, 2025, None}
    assert "keep" in [json.loads(item.row_json).get("project_id") for item in stored]
    assert "null-year" in [
        json.loads(item.row_json).get("project_id") for item in stored
    ]
    assert "new" in [json.loads(item.row_json).get("project_id") for item in stored]


def test_hash_is_order_independent_and_duplicate_sensitive() -> None:
    conn = _new_connection()
    first = _row(report_year=2025, project_id="first")
    second = _row(report_year=2025, project_id="second")
    conn.execute("BEGIN")
    replace_timeseries_in_transaction(
        conn, (first, second), years=None, revision="r1", fetched_at="now"
    )
    conn.commit()
    original = timeseries_fingerprint(conn)
    conn.execute("BEGIN")
    replace_timeseries_in_transaction(
        conn, (second, first), years=None, revision="r2", fetched_at="now"
    )
    conn.commit()
    assert timeseries_fingerprint(conn) == original
    conn.execute("BEGIN")
    replace_timeseries_in_transaction(
        conn, (first, second, second), years=None, revision="r3", fetched_at="now"
    )
    conn.commit()
    assert timeseries_fingerprint(conn)[0] == 3
    assert timeseries_fingerprint(conn)[1] != original[1]


def test_transaction_free_writer_rolls_back_with_caller() -> None:
    conn = _new_connection()
    with pytest.raises(RuntimeError, match="active caller transaction"):
        replace_timeseries_in_transaction(
            conn, (_row(report_year=2025),), years=None, revision="r0", fetched_at="now"
        )
    with pytest.raises(RuntimeError, match="rollback"):
        conn.execute("BEGIN")
        replace_timeseries_in_transaction(
            conn, (_row(report_year=2025),), years=None, revision="r1", fetched_at="now"
        )
        raise RuntimeError("rollback")
    conn.rollback()
    assert (
        conn.execute("SELECT COUNT(*) FROM api_project_timeseries").fetchone()[0] == 0
    )


def test_schema_hook_does_not_commit_callers_transaction() -> None:
    conn = sqlite3.connect(":memory:")
    conn.execute("BEGIN")
    from esdc.timeseries_snapshot import ensure_timeseries_tables

    ensure_timeseries_tables(conn)
    assert conn.in_transaction
    conn.rollback()
    assert (
        conn.execute(
            "SELECT 1 FROM sqlite_master WHERE name='api_project_timeseries'"
        ).fetchone()
        is None
    )


@pytest.mark.parametrize("tamper", ["row_hash", "indexed_year"])
def test_fingerprint_rejects_persisted_metadata_tampering(tamper: str) -> None:
    conn = _new_connection()
    row = _row(report_year=2025, project_id="integrity")
    conn.execute("BEGIN")
    replace_timeseries_in_transaction(
        conn, (row,), years=None, revision="r1", fetched_at="now"
    )
    conn.commit()
    if tamper == "row_hash":
        conn.execute("UPDATE api_project_timeseries SET row_sha256=?", ("0" * 64,))
        message = "hash does not match"
    else:
        conn.execute("UPDATE api_project_timeseries SET report_year=2024")
        message = "indexed and JSON report_year"
    with pytest.raises(ValueError, match=message):
        timeseries_fingerprint(conn)


def test_public_canonical_rows_cannot_bypass_scalar_or_year_validation() -> None:
    conn = _new_connection()
    valid = _row(report_year=2025)
    nested_json = valid.row_json.replace(
        '"project_name":null', '"project_name":{"x":1}'
    )
    nested = CanonicalTimeseriesRow(
        2025,
        nested_json,
        hashlib.sha256(nested_json.encode("utf-8")).hexdigest(),
        "api_json",
    )
    parsed = json.loads(valid.row_json)
    parsed["report_year"] = None
    null_json = json.dumps(
        parsed, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    invalid_year = CanonicalTimeseriesRow(
        None,
        null_json,
        hashlib.sha256(null_json.encode("utf-8")).hexdigest(),
        "api_json",
    )
    for invalid, message in (
        (nested, "not a JSON scalar"),
        (invalid_year, "report_year"),
    ):
        conn.execute("BEGIN")
        with pytest.raises(ValueError, match=message):
            replace_timeseries_in_transaction(
                conn, (invalid,), years=None, revision="bad", fetched_at="now"
            )
        conn.rollback()


def test_writer_and_reads_use_one_shot_batches_without_fetchall() -> None:
    conn = _new_connection()
    batch_sizes: list[int] = []

    class CursorProxy:
        def __init__(self, cursor):
            self._cursor = cursor

        def fetchmany(self, size=1):
            result = self._cursor.fetchmany(size)
            batch_sizes.append(len(result))
            assert len(result) <= 1000
            return result

        def fetchall(self):
            raise AssertionError("timeseries path must stream with fetchmany")

        def __getattr__(self, name):
            return getattr(self._cursor, name)

    class ConnectionProxy:
        def __init__(self, connection):
            self._connection = connection

        @property
        def in_transaction(self):
            return self._connection.in_transaction

        def execute(self, *args, **kwargs):
            cursor = self._connection.execute(*args, **kwargs)
            normalized = " ".join(str(args[0]).lower().replace(chr(34), "").split())
            if "from api_project_timeseries" in normalized:
                return CursorProxy(cursor)
            return cursor

        def executemany(self, *args, **kwargs):
            return CursorProxy(self._connection.executemany(*args, **kwargs))

        def __getattr__(self, name):
            return getattr(self._connection, name)

    class OneShotRows:
        consumed = False

        def __iter__(self):
            assert not self.consumed
            self.consumed = True
            return (_row(report_year=2025, project_id=str(i)) for i in range(1001))

    conn.execute("BEGIN")
    instrumented = cast(sqlite3.Connection, ConnectionProxy(conn))
    snapshot = replace_timeseries_in_transaction(
        instrumented,
        OneShotRows(),
        years=None,
        revision="r1",
        fetched_at="now",
    )
    assert snapshot.row_count == 1001
    conn.commit()
    assert timeseries_fingerprint(instrumented)[0] == 1001
    assert len(list(iter_timeseries_rows(instrumented))) == 1001
    assert batch_sizes and max(batch_sizes) <= 1000
