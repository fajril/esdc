"""Task5C1: read-only POD reconciliation context (one snapshot) regressions.

Every fixture here is created in a throwaway directory under the OS temporary
directory, so the test is portable across platforms. The active operational
database is never opened. The existing v2 initializer is used only as test
setup, and the production reader must never create, initialize, migrate, or
write anything.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import sqlite3
import tempfile
from collections.abc import Iterator
from dataclasses import replace
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest

from esdc.pod_registry import api_context
from esdc.pod_registry.api_context import (
    PodApiContext,
    PodImportError,
    read_identity_ledger_rows,
    read_pod_api_context,
)
from esdc.pod_registry.api_import import INPUT_TABLES, prepare_pod_api_batch
from esdc.pod_registry.contract import load_pod_contract
from esdc.pod_registry.migrations import initialize_registry_v2
from esdc.pod_registry.models import ImportTargetBasis

CONTRACT = load_pod_contract()
PROJECT_ID = "p001459f01"
POD_ID = "PL-2024-0021-1-1-1"
POD_UUID = "4aac3a88-10f4-4c74-87d5-12246ce43ca2"
MON_UUID = "00000001-1111-4111-8111-111111111111"
PROJ_UUID = "22222222-5222-4222-8222-222222222222"
FRC_UUID = "33333333-5333-4333-8333-333333333333"
STAGE_ID = 3
MARKER = "UNCHECKPOINTED-WAL-MARKER"
FRESH_SHA = hashlib.sha256(b"fresh").hexdigest()
NON_POD_TARGET_SHA = hashlib.sha256(b"non-pod-target").hexdigest()
WRITE_VERBS = ("INSERT", "UPDATE", "DELETE", "CREATE", "DROP", "ALTER")
OUTPUT_TABLES = ("pod_monitoring_api", "pod_forecast_api")


@pytest.fixture
def fixture_root() -> Iterator[Path]:
    with tempfile.TemporaryDirectory(prefix="esdc-pod-5c1-") as name:
        yield Path(name)


class _RecordingConnection(sqlite3.Connection):
    """Records the statements and the closure of one read-only connection."""

    # When set, a matching setup statement fails, which is how the reader's
    # connection-setup error path is exercised.
    fail_on: str | None = None
    # A matching setup statement is cancelled instead, which is how the reader's
    # connection-setup cancellation path is exercised.
    cancel_on: str | None = None

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.events: list[tuple[str, str]] = []
        self.closed = False

    def execute(self, sql: str, values: Any = ()) -> sqlite3.Cursor:
        self.events.append(("execute", str(sql)))
        if self.cancel_on and str(sql).strip().upper() == self.cancel_on.upper():
            raise KeyboardInterrupt
        if self.fail_on and str(sql).strip().upper() == self.fail_on.upper():
            raise sqlite3.OperationalError(f"setup step failed: {sql}")
        return super().execute(sql, values)

    def close(self) -> None:
        self.closed = True
        self.events.append(("close", ""))
        super().close()


class Recorder:
    """Routes the reader's connections through a recording factory."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.connections: list[_RecordingConnection] = []
        self.uris: list[str] = []
        self.real_connect = sqlite3.connect
        monkeypatch.setattr(api_context.sqlite3, "connect", self._connect)

    def _connect(self, *args: Any, **kwargs: Any) -> sqlite3.Connection:
        connection = self.real_connect(
            *args, factory=_RecordingConnection, **kwargs
        )
        self.connections.append(connection)
        self.uris.append(str(args[0]) if args else "")
        return connection


@pytest.fixture
def recorder(monkeypatch: pytest.MonkeyPatch) -> Recorder:
    return Recorder(monkeypatch)


def _statements(recorder: Recorder) -> list[str]:
    assert len(recorder.connections) == 1, (
        "basis, rows, ledger, and links must be read in one snapshot connection"
    )
    return [event[1] for event in recorder.connections[0].events if event[0] == "execute"]


def _columns(table: str) -> dict[str, Any]:
    return {column.name: column for column in CONTRACT.tables[table].columns}


def _raw_value(column: Any) -> object:
    logical = column.logical_type
    if logical == "text":
        return "Raw text"
    if logical == "integer":
        return 1
    if logical == "decimal":
        return "20.3"
    if logical == "boolean":
        return 1
    if logical == "date":
        return "2024-01-15"
    if logical == "datetime":
        return "2024-01-15T08:00:00+00:00"
    if logical == "uuid":
        return str(UUID("55555555-5555-4555-8555-555555555555"))
    raise AssertionError(f"unhandled logical type {logical!r}")


# The dependent tables copy the issued POD identity, so the fixture rows must agree
# with the master and with the ledger reservation.
_IDENTITY_COPIES = {
    "pod_id_seq": 1,
    "pod_id": POD_ID,
    "pod_uuid": POD_UUID,
    "approval_date": "2024-01-02",
    "pod_order_no": 21,
    "institution_id": 1,
    "pod_type_id": 1,
    "rev_num": 1,
}


# Raw fixture values must still satisfy the contract's value guards.
_PLAUSIBLE_VALUES = {"year_frcast": 2024}


def _raw_row(table: str, **overrides: object) -> dict[str, object]:
    row = {column.name: _raw_value(column) for column in CONTRACT.tables[table].columns}
    # A raw fixture row is not a supersession, so the self-reference stays null.
    if "superseded_by" in row:
        row["superseded_by"] = None
    for name, value in _IDENTITY_COPIES.items():
        if name in row:
            row[name] = value
    for name, value in _PLAUSIBLE_VALUES.items():
        if name in row:
            row[name] = value
    row.update(overrides)
    return row


def _insert(conn: sqlite3.Connection, table: str, row: dict[str, object]) -> None:
    names = [column.name for column in CONTRACT.tables[table].columns]
    conn.execute(
        f"INSERT INTO {table} ({', '.join(names)}) VALUES ("
        + ", ".join("?" for _ in names)
        + ")",
        [row[name] for name in names],
    )


def _v2_target(recorder: Recorder, path: Path) -> sqlite3.Connection:
    """Test-fixture-only v2 initialization; the production reader never initializes."""
    conn = recorder.real_connect(path)
    conn.execute("PRAGMA foreign_keys = ON")
    initialize_registry_v2(conn)
    conn.commit()
    return conn


def _issue_ledger(conn: sqlite3.Connection) -> None:
    conn.execute(
        "INSERT INTO _pod_identity_ledger (approval_year, approval_date, daily_slot, "
        "pod_order_no, pod_id_seq, pod_id, pod_uuid, institution_id, pod_type_id, "
        "rev_num, actor, issued_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            2024,
            "2024-01-02",
            1,
            21,
            1,
            POD_ID,
            POD_UUID,
            1,
            1,
            1,
            "FA:initial-pod-identity-allocation",
            "2024-01-02T08:10:26.428Z",
        ),
    )


def _pod_master_row(**overrides: object) -> dict[str, object]:
    return _raw_row("m_pod", **overrides)


def _monitoring_row(monpod_id: int, remarks: str, reported: str) -> dict[str, object]:
    return _raw_row(
        "pod_monitoring",
        monpod_id=monpod_id,
        monpod_uuid=str(UUID(f"{monpod_id:08x}-1111-4111-8111-111111111111")),
        pod_stage_id=STAGE_ID,
        remarks_monitoring=remarks,
        reported_date=reported,
    )


def _task4_source(recorder: Recorder, path: Path) -> sqlite3.Connection:
    """A Task4-shaped source fixture: contract columns plus its audit ledger fields."""
    conn = recorder.real_connect(path)
    for table in INPUT_TABLES:
        ddl = ", ".join(
            f'"{column.name}" {column.data_type}'
            for column in CONTRACT.tables[table].columns
        )
        conn.execute(f"CREATE TABLE {table} ({ddl})")
    conn.execute(
        "CREATE TABLE _pod_identity_ledger (approval_year, approval_date, daily_slot, "
        "pod_order_no, pod_id_seq, pod_id, issued_at, issued_by)"
    )
    conn.execute("PRAGMA user_version=1")
    return conn


def _pod_project_row() -> dict[str, object]:
    # The reference guard compares the raw stored project id with the m_project row,
    # so both tables must store the same compact form.
    return _raw_row(
        "pod_project",
        pod_project_uuid=PROJ_UUID,
        project_id=PROJECT_ID,
    )


def _seed_target(recorder: Recorder, path: Path) -> sqlite3.Connection:
    conn = _v2_target(recorder, path)
    conn.execute("PRAGMA foreign_keys = OFF")
    _issue_ledger(conn)
    _insert(conn, "m_project", _raw_row("m_project", project_id=PROJECT_ID))
    _insert(conn, "m_pod", _pod_master_row())
    _insert(conn, "pod_project", _pod_project_row())
    _insert(conn, "pod_monitoring", _monitoring_row(1, "BASELINE", "2024-01-15"))
    _insert(conn, "pod_forecast", _raw_row("pod_forecast", forecast_uuid=FRC_UUID))
    conn.executemany(
        "INSERT INTO pod_document (pod_id_seq, doc_id) VALUES (?, ?)",
        [(1, "DOC-B"), (1, "DOC-A"), (2, "DOC-C")],
    )
    conn.commit()
    return conn


def test_public_context_api_is_a_frozen_snapshot_record() -> None:
    fields = PodApiContext.__dataclass_fields__
    assert set(fields) == {
        "target_basis",
        "existing_rows",
        "reserved_pod_identities",
        "linked_pod_ids",
    }
    assert PodApiContext.__annotations__ == {
        "target_basis": "ImportTargetBasis",
        "existing_rows": "Mapping[str, Sequence[PreparedRow]]",
        "reserved_pod_identities": "Sequence[PreparedRow]",
        "linked_pod_ids": "tuple[int, ...]",
    }
    context = PodApiContext(
        ImportTargetBasis(False, None, FRESH_SHA, Path("/private/tmp/unchanged")),
        {},
        (),
        (),
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        context.__setattr__("linked_pod_ids", (1,))


def test_absent_target_with_absent_parent_returns_fresh_basis_without_io(
    fixture_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbid_connect(*args: Any, **kwargs: Any) -> sqlite3.Connection:
        raise AssertionError("an absent target must not open a connection")

    def forbid_mkdir(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("an absent target must not create a path")

    monkeypatch.setattr(api_context.sqlite3, "connect", forbid_connect)
    monkeypatch.setattr(Path, "mkdir", forbid_mkdir)
    monkeypatch.setattr(
        "esdc.pod_registry.migrations.initialize_registry_v2",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("an absent target must not be initialized")
        ),
    )

    target = fixture_root / "absent-parent" / "target.sqlite"
    context = read_pod_api_context(target)

    assert context.target_basis == ImportTargetBasis(
        False, None, FRESH_SHA, target.resolve(strict=False)
    )
    assert context.existing_rows == {}
    assert context.reserved_pod_identities == ()
    assert context.linked_pod_ids == ()
    assert not target.parent.exists()
    assert not target.exists()


def test_empty_closed_sqlite_file_is_an_acceptable_fresh_basis(
    fixture_root: Path, recorder: Recorder
) -> None:
    target = fixture_root / "empty.sqlite"
    target.write_bytes(b"")

    context = read_pod_api_context(target)

    assert context.target_basis == ImportTargetBasis(
        True, None, NON_POD_TARGET_SHA, target.resolve()
    )
    assert context.existing_rows == {}
    assert context.reserved_pod_identities == ()
    assert context.linked_pod_ids == ()

    verifier = recorder.real_connect(target)
    try:
        assert verifier.execute("SELECT count(*) FROM sqlite_master").fetchone()[0] == 0
    finally:
        verifier.close()


def test_legacy_pod_registry_refuses_explicit_crosswalk(
    fixture_root: Path, recorder: Recorder
) -> None:
    target = fixture_root / "legacy.sqlite"
    conn = recorder.real_connect(target)
    conn.executescript(
        "CREATE TABLE m_pod (id INTEGER PRIMARY KEY, pod_id TEXT);"
        "CREATE TABLE pod_document (pod_id INTEGER, doc_id TEXT);"
    )
    conn.commit()
    before = target.read_bytes()
    conn.close()

    with pytest.raises(PodImportError, match="--pod-crosswalk") as error:
        read_pod_api_context(target)

    # The useful guidance is the explicit replacement route; the wording is matched
    # case-insensitively because it is sentence-cased, and only the mapping name and the
    # public CLI route are exact.
    message = str(error.value)
    assert "legacy pod registry" in message.casefold()
    assert "explicit validated replacement" in message.casefold()
    assert "numeric pod ids are never inferred" in message.casefold()
    assert "esdc load --from-sqlite" in message and "--pod-crosswalk" in message
    assert target.read_bytes() == before


def test_application_tables_alone_are_an_additive_no_pod_target(
    fixture_root: Path, recorder: Recorder
) -> None:
    target = fixture_root / "application.sqlite"
    conn = recorder.real_connect(target)
    conn.execute("CREATE TABLE api_project_resources (project_id TEXT)")
    conn.execute("INSERT INTO api_project_resources VALUES ('P-0000777-77')")
    conn.commit()
    conn.close()

    context = read_pod_api_context(target)

    assert context.target_basis == ImportTargetBasis(
        True, None, NON_POD_TARGET_SHA, target.resolve()
    )
    assert context.target_basis.source_generation is None
    assert context.existing_rows == {}
    assert context.reserved_pod_identities == ()
    assert context.linked_pod_ids == ()

    statements = _statements(recorder)
    assert not any(statement.split()[0].upper() in WRITE_VERBS for statement in statements)
    verifier = recorder.real_connect(target)
    try:
        assert verifier.execute(
            "SELECT project_id FROM api_project_resources"
        ).fetchall() == [("P-0000777-77",)]
        assert verifier.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall() == [("api_project_resources",)], "no registry is created"
    finally:
        verifier.close()


def test_corrupt_target_refuses_contextually_without_repair(
    fixture_root: Path, recorder: Recorder
) -> None:
    target = fixture_root / "corrupt.sqlite"
    target.write_bytes(b"not a database at all")

    with pytest.raises(PodImportError, match="Could not read POD target") as error:
        read_pod_api_context(target)

    assert "not a database" in str(error.value)
    assert target.read_bytes() == b"not a database at all"


def test_incompatible_v2_metadata_refuses(fixture_root: Path, recorder: Recorder) -> None:
    target = fixture_root / "incompatible.sqlite"
    conn = _seed_target(recorder, target)
    conn.execute("UPDATE pod_registry_state SET dictionary_version='0.3.9'")
    conn.commit()
    conn.close()

    with pytest.raises(PodImportError, match="incompatible"):
        read_pod_api_context(target)


def test_existing_v2_context_normalizes_the_five_inputs_in_one_snapshot(
    fixture_root: Path, recorder: Recorder
) -> None:
    target = fixture_root / "v2.sqlite"
    conn = _seed_target(recorder, target)
    expected_digest = api_context.target_protected_state_digest(conn)
    conn.close()

    context = read_pod_api_context(target)

    assert context.target_basis == ImportTargetBasis(
        True, 0, expected_digest, target.resolve()
    )
    assert set(context.existing_rows) == set(INPUT_TABLES)

    project = context.existing_rows["m_project"][0]
    assert project["project_id"] == "P-001459F-01"
    master = context.existing_rows["m_pod"][0]
    assert master["pod_uuid"] == UUID(POD_UUID)
    assert master["approval_date"] == date(2024, 1, 2)
    monitoring = context.existing_rows["pod_monitoring"][0]
    assert monitoring["reported_date"] == date(2024, 1, 15)
    assert monitoring["monpod_uuid"] == UUID(MON_UUID)
    assert monitoring["is_completed"] == 1
    assert not isinstance(monitoring["is_completed"], bool)
    assert isinstance(monitoring["is_completed"], int)
    decimal_column = _columns("pod_monitoring")["wap_oil_real"]
    assert decimal_column.data_type == "NUMERIC(20,6)"
    assert monitoring["wap_oil_real"] == Decimal("20.3")
    assert isinstance(monitoring["wap_oil_real"], Decimal)

    for table in INPUT_TABLES:
        for row in context.existing_rows[table]:
            for name, value in row.items():
                if value is None:
                    continue
                logical = _columns(table)[name].logical_type
                if logical == "uuid":
                    assert isinstance(value, UUID), f"{table}.{name} stayed raw text"
                elif logical == "date":
                    assert isinstance(value, date), f"{table}.{name} stayed raw text"
                elif logical == "datetime":
                    assert isinstance(value, datetime), f"{table}.{name} stayed raw text"
                elif logical == "decimal":
                    assert isinstance(value, (Decimal, float)), (
                        f"{table}.{name} stayed raw text"
                    )
                elif logical == "boolean":
                    assert isinstance(value, int) and value in (0, 1)

    ledger = context.reserved_pod_identities
    assert len(ledger) == 1
    # The operational ledger is read against the canonical v2 DDL columns.
    assert set(ledger[0]) == {
        "approval_year",
        "approval_date",
        "daily_slot",
        "pod_order_no",
        "pod_id_seq",
        "pod_id",
        "pod_uuid",
        "institution_id",
        "pod_type_id",
        "rev_num",
        "actor",
        "issued_at",
    }
    assert ledger[0]["actor"] == "FA:initial-pod-identity-allocation"
    assert ledger[0]["pod_uuid"] == POD_UUID
    assert isinstance(ledger[0]["pod_uuid"], str)
    assert context.linked_pod_ids == (1, 2)

    statements = _statements(recorder)
    assert any("query_only" in statement for statement in statements)
    assert any(statement.strip().upper() == "BEGIN" for statement in statements)
    assert not any(statement.split()[0].upper() in WRITE_VERBS for statement in statements)
    assert not any("journal_mode" in statement for statement in statements)
    # The output tables may be verified as metadata, but their rows must never be
    # loaded by this reader.
    assert not any(
        f'FROM "{table}"' in statement or f"FROM {table}" in statement
        for statement in statements
        for table in OUTPUT_TABLES
    )
    for table in INPUT_TABLES:
        assert any(f'FROM "{table}"' in statement for statement in statements)
    assert recorder.connections[0].closed is True
    assert recorder.connections[0].events[-1][0] == "close"
    assert "mode=ro" in recorder.uris[0]
    assert "immutable" not in recorder.uris[0]


def test_concurrent_commit_during_read_does_not_split_the_snapshot(
    fixture_root: Path, recorder: Recorder, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = fixture_root / "concurrent.sqlite"
    conn = _seed_target(recorder, target)
    original_digest = api_context.target_protected_state_digest
    expected_basis = ImportTargetBasis(
        True, 0, original_digest(conn), target.resolve()
    )

    def digest_with_concurrent_commit(
        reader: sqlite3.Connection,
        *,
        need_preserved_inputs: bool = False,
        contract: Any = None,
    ) -> str:
        # Another project commits while the reader is inside its transaction.
        conn.execute("INSERT INTO pod_document (pod_id_seq, doc_id) VALUES (3, 'DOC-X')")
        conn.commit()
        return original_digest(
            reader, need_preserved_inputs=need_preserved_inputs, contract=contract
        )

    monkeypatch.setattr(api_context, "target_protected_state_digest", digest_with_concurrent_commit)

    context = read_pod_api_context(target)

    # The basis and the rows must describe the same snapshot, so the concurrent
    # document link appears in neither.
    assert context.linked_pod_ids == (1, 2)
    assert context.target_basis == expected_basis
    assert recorder.connections[0].closed is True
    conn.close()


def test_committed_uncheckpointed_wal_row_is_visible_and_the_main_file_is_untouched(
    fixture_root: Path, recorder: Recorder
) -> None:
    target = fixture_root / "wal.sqlite"
    conn = _seed_target(recorder, target)
    assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
    _insert(conn, "pod_monitoring", _monitoring_row(2, MARKER, "2024-02-01"))
    conn.commit()

    wal = Path(f"{target}-wal")
    assert wal.exists() and wal.stat().st_size > 0
    assert MARKER.encode() not in target.read_bytes()

    context = read_pod_api_context(target)

    rows = [dict(row) for row in context.existing_rows["pod_monitoring"]]
    assert [row["remarks_monitoring"] for row in rows] == ["BASELINE", MARKER]
    assert MARKER.encode() not in target.read_bytes()
    assert "immutable" not in recorder.uris[0]
    statements = _statements(recorder)
    assert not any(statement.split()[0].upper() in WRITE_VERBS for statement in statements)
    assert not any("journal_mode" in statement for statement in statements)
    conn.close()


def _json_scalar(value: object) -> object:
    if isinstance(value, (Decimal, UUID, date, datetime)):
        return str(value)
    return value


def _payload_json(rows: object) -> bytes:
    return json.dumps(
        [
            {name: _json_scalar(value) for name, value in dict(row).items()}
            for row in rows  # type: ignore[union-attr]
        ]
    ).encode()


def test_context_rows_feed_the_accepted_api_batch_for_selected_year_preservation(
    fixture_root: Path, recorder: Recorder
) -> None:
    target = fixture_root / "feed.sqlite"
    conn = _seed_target(recorder, target)
    _insert(conn, "pod_monitoring", _monitoring_row(3, "OUTSIDE_YEAR", "2023-12-31"))
    conn.commit()
    conn.close()

    context = read_pod_api_context(target)

    # The fetch carries the masters; only the selected year's monitoring payload is
    # empty, so the fact reported outside that year must survive on the context rows.
    payloads = {table: _payload_json(context.existing_rows[table]) for table in INPUT_TABLES}
    payloads["pod_monitoring"] = b"[]"
    batch = prepare_pod_api_batch(
        payloads,
        existing_rows=context.existing_rows,
        reserved_pod_identities=context.reserved_pod_identities,
        monitoring_years=[2024],
    )

    retained = [dict(row) for row in batch.rows["pod_monitoring"]]
    assert [row["remarks_monitoring"] for row in retained] == ["OUTSIDE_YEAR"]
    # The reader's typed values, not raw SQLite text, are what the batch receives.
    assert retained[0]["reported_date"] == date(2023, 12, 31)
    assert retained[0]["monpod_uuid"] == UUID("00000003-1111-4111-8111-111111111111")
    assert batch.target_basis is None
    # The basis the reader captured is the basis a later apply must still use.
    assert replace(batch, target_basis=context.target_basis).target_basis == (
        context.target_basis
    )
    assert context.linked_pod_ids == (1, 2)


def test_adapter_target_state_still_uses_placeholder_preserved_rows(
    fixture_root: Path, recorder: Recorder
) -> None:
    import importlib

    adapter = importlib.import_module("esdc.pod_registry.sqlite_import")
    target = fixture_root / "adapter.sqlite"
    conn = _seed_target(recorder, target)
    conn.close()

    basis, headers, rows, linked = adapter._read_target_state(
        target, CONTRACT, need_preserved_inputs=True
    )

    assert basis.existed is True
    assert basis.source_generation == 0
    assert basis.target_path == target.resolve()
    assert set(headers) == {"pod_monitoring", "pod_forecast"}
    assert set(rows) == {"pod_monitoring", "pod_forecast"}
    assert all(values == () for values in rows.values())
    assert headers["pod_monitoring"] == tuple(
        column.name for column in CONTRACT.tables["pod_monitoring"].columns
    )
    assert linked == (1, 2)
    assert adapter.PodImportError is PodImportError


def test_document_link_orphan_is_refused_before_any_writer(
    fixture_root: Path, recorder: Recorder
) -> None:
    import importlib

    adapter = importlib.import_module("esdc.pod_registry.sqlite_import")
    target = fixture_root / "orphan-target.sqlite"
    conn = _seed_target(recorder, target)
    conn.close()

    source = fixture_root / "orphan-source.sqlite"
    # The source is read with Task4's source semantics, which still require its own
    # audit ledger fields rather than the operational v2 ledger shape.
    source_conn = _task4_source(recorder, source)
    source_conn.execute("PRAGMA foreign_keys = OFF")
    _insert(source_conn, "m_project", _raw_row("m_project", project_id=PROJECT_ID))
    _insert(source_conn, "m_pod", _pod_master_row())
    _insert(source_conn, "pod_project", _pod_project_row())
    _insert(source_conn, "pod_monitoring", _monitoring_row(1, "BASELINE", "2024-01-15"))
    _insert(source_conn, "pod_forecast", _raw_row("pod_forecast", forecast_uuid=FRC_UUID))
    source_conn.commit()
    before = target.read_bytes()
    source_conn.close()

    with pytest.raises(adapter.PodImportError, match="omits POD identities") as error:
        adapter.prepare_pod_import(
            source, "sqlite", CONTRACT, target_path=target, stage_snapshots=False
        )

    assert "2" in str(error.value)
    assert target.read_bytes() == before


def test_unnormalizable_target_value_reports_row_context_and_closes(
    fixture_root: Path, recorder: Recorder
) -> None:
    target = fixture_root / "bad-value.sqlite"
    conn = _seed_target(recorder, target)
    bad = _raw_row(
        "pod_monitoring",
        monpod_id=2,
        monpod_uuid=str(UUID("00000002-1111-4111-8111-111111111111")),
        pod_stage_id=STAGE_ID,
        reported_date="2024-01-16",
        wap_oil_real="not a number",
    )
    _insert(conn, "pod_monitoring", bad)
    conn.commit()
    conn.close()

    with pytest.raises(PodImportError, match="pod_monitoring") as error:
        read_pod_api_context(target)

    assert "wap_oil_real" in str(error.value)
    assert recorder.connections[0].closed is True


def test_reader_closes_the_connection_when_the_read_fails(
    fixture_root: Path, recorder: Recorder, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = fixture_root / "failing.sqlite"
    conn = _seed_target(recorder, target)
    conn.close()

    def failing_digest(
        reader: sqlite3.Connection,
        *,
        need_preserved_inputs: bool = False,
        contract: Any = None,
    ) -> str:
        raise ValueError("reader interrupted")

    monkeypatch.setattr(api_context, "target_protected_state_digest", failing_digest)

    with pytest.raises(ValueError, match="reader interrupted"):
        read_pod_api_context(target)

    assert recorder.connections[0].closed is True


@pytest.mark.parametrize("setup_step", ["PRAGMA query_only = ON", "BEGIN"])
def test_read_setup_failure_closes_the_connection(
    fixture_root: Path, recorder: Recorder, monkeypatch: pytest.MonkeyPatch, setup_step: str
) -> None:
    target = fixture_root / "setup-failure.sqlite"
    target.write_bytes(b"")

    monkeypatch.setattr(_RecordingConnection, "fail_on", setup_step)

    with pytest.raises(PodImportError, match="Could not read POD target") as error:
        read_pod_api_context(target)

    # The helper owns a connection it created, even when its setup step fails.
    assert str(target) in str(error.value)
    assert len(recorder.connections) == 1
    assert recorder.connections[0].closed is True


@pytest.mark.parametrize("setup_step", ["PRAGMA query_only = ON", "BEGIN"])
def test_read_setup_cancellation_closes_the_connection(
    fixture_root: Path,
    recorder: Recorder,
    monkeypatch: pytest.MonkeyPatch,
    setup_step: str,
) -> None:
    target = fixture_root / "setup-cancellation.sqlite"
    target.write_bytes(b"")

    monkeypatch.setattr(_RecordingConnection, "cancel_on", setup_step)

    # A cancellation must propagate unchanged, and the helper must still close the
    # connection it created before the error leaves.
    with pytest.raises(KeyboardInterrupt):
        read_pod_api_context(target)

    assert len(recorder.connections) == 1
    assert recorder.connections[0].closed is True


def test_source_ledger_audit_semantics_are_not_weakened(
    fixture_root: Path, recorder: Recorder
) -> None:
    target = fixture_root / "ledger-shape.sqlite"
    conn = _seed_target(recorder, target)

    # Task4's source parser still requires its audit field, so the operational reader
    # reads the canonical v2 ledger against the v2 DDL instead of remapping the source.
    with pytest.raises(PodImportError, match="issued_by"):
        read_identity_ledger_rows(conn)

    context = read_pod_api_context(target)
    assert context.reserved_pod_identities[0]["actor"] == (
        "FA:initial-pod-identity-allocation"
    )
