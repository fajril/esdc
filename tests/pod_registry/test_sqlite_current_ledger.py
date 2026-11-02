"""S2d1a: the canonical current SQLite source-ledger adapter.

Every database here is an owned fixture under ``tmp_path``; the operational store and the
reference seed are never opened, and the Config db/config/cache paths are patched to owned
tmp paths before seeding. Assertions run through the public ``prepare_pod_import`` path, so
the profile decision, the metadata validation and the ledger read all happen inside one
read-only transaction that opens no writer and creates no target. The replacement case loads
a prepared canonical source into an owned current target through the public path, and every
SQLite, DuckDB and native-graph handle opened by the test is closed explicitly.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path
from typing import cast

import duckdb
import pytest

from esdc.chat.domain_knowledge.instance_graph import InstanceGraphManager
from esdc.configs import Config
from esdc.knowledge.freshness import compute_knowledge_basis, mark_knowledge_current
from esdc.knowledge.linker import run_deterministic_linking
from esdc.knowledge.store import KnowledgeStore
from esdc.pod_registry import api_context
from esdc.pod_registry.api_context import PodImportError
from esdc.pod_registry.contract import load_pod_contract
from esdc.pod_registry.migrations import SCHEMA_VERSION
from esdc.pod_registry.publication import publish_v2_registry
from esdc.pod_registry.sqlite_import import (
    load_prepared_pod_batch,
    prepare_pod_import,
)
from esdc.pod_registry.store import get_sqlite_connection
from tests.knowledge import conftest as knowledge
from tests.pod_registry import test_import_v2

LEDGER = "_pod_identity_ledger"
POD_DURI_REV1 = knowledge.POD_DURI_REV1


@pytest.fixture()
def owned_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[Path]:
    """Point the Config db/config/cache paths at owned tmp paths before seeding."""
    monkeypatch.setattr(Config, "get_db_dir", classmethod(lambda cls: tmp_path))
    monkeypatch.setattr(Config, "get_cache_dir", classmethod(lambda cls: tmp_path / "cache"))
    monkeypatch.setattr(
        Config, "get_db_file", classmethod(lambda cls: tmp_path / "esdc.duckdb")
    )
    yield tmp_path


def _ledger_rows(conn: sqlite3.Connection) -> list[dict[str, object]]:
    """Full source ledger rows as they are stored, in stable key order."""
    cursor = conn.execute(f"SELECT * FROM {LEDGER} ORDER BY pod_id_seq")
    return [dict(row) for row in cursor.fetchall()]


def _renamed(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """The same rows with only the issuer representation renamed."""
    return [
        {name: value for name, value in row.items() if name != "actor"}
        | {"issued_by": row["actor"]}
        for row in rows
    ]


def _canonical_source(owned_config: Path, name: str) -> sqlite3.Connection:
    """A canonical current registry on the shared single schema (issuer is ``actor``)."""
    conn = get_sqlite_connection(owned_config / name)
    knowledge._seed_knowledge_world(conn)
    return conn


def _rebuild(conn: sqlite3.Connection, memory: duckdb.DuckDBPyConnection) -> None:
    """Rebuild the derived graph through the shared knowledge helpers."""
    store = KnowledgeStore(conn, allow_stale=True)
    store.ensure_tables()
    run_deterministic_linking(conn, memory, store)
    conn.commit()
    conn.execute("BEGIN IMMEDIATE")
    assert mark_knowledge_current(conn, compute_knowledge_basis(conn))
    conn.commit()


def _raw_documents(conn: sqlite3.Connection) -> list[tuple[object, ...]]:
    """Raw document facts, excluding the intentionally derived suggestion column."""
    columns = [str(row[1]) for row in conn.execute("PRAGMA table_info(documents)")]
    columns.remove("suggested_pod_ids")
    selected = ", ".join(api_context.quote_identifier(name) for name in columns)
    return [
        tuple(row)
        for row in conn.execute(f"SELECT {selected} FROM documents ORDER BY doc_id")
    ]


def test_canonical_source_prepare_maps_actor_to_issued_by(
    owned_config: Path,
) -> None:
    """The independently proven RED (gate79e_qjd9): a valid canonical source is refused.

    The canonical current ledger records the issuer as ``actor``; the source reader still
    applied the supplied ePOD profile, which requires ``issued_by``. The adapter resolves
    the profile from the source's own declared shape and renames only that technical
    representation, so every identity, reservation and audit value stays as stored.
    """
    source = owned_config / "canonical-source.sqlite"
    conn = _canonical_source(owned_config, "canonical-source.sqlite")
    try:
        original = _ledger_rows(conn)
    finally:
        conn.close()

    assert original, "the accepted seed must issue canonical identities"
    assert all("actor" in row for row in original)
    before = source.read_bytes()

    target = owned_config / "missing-target.sqlite"
    batch = prepare_pod_import(source, "sqlite", target_path=target, stage_snapshots=False)

    prepared = [dict(row) for row in batch.identity_ledger or ()]
    assert prepared == _renamed(original)

    # Prepare is read-only: the source is unchanged and no target or cache is created.
    reader = sqlite3.connect(source)
    reader.row_factory = sqlite3.Row
    try:
        assert _ledger_rows(reader) == original
    finally:
        reader.close()
    assert source.read_bytes() == before
    assert not target.exists()
    assert not Config.get_db_file().exists()


def test_target_reader_semantics_are_not_changed_by_the_source_adapter(
    owned_config: Path,
) -> None:
    """The shared canonical reader keeps its explicit fields; only the source path adapts.

    Task4's source audit profile still requires ``issued_by`` on the shared reader, and the
    target snapshot read stays on the canonical fields with ``actor`` intact.
    """
    conn = _canonical_source(owned_config, "target-reader.sqlite")
    try:
        original = _ledger_rows(conn)
        with pytest.raises(PodImportError, match="identity ledger is missing fields"):
            api_context.read_identity_ledger_rows(conn)

        rows = api_context.read_source_identity_ledger_rows(conn)
        assert [dict(row) for row in rows or ()] == _renamed(original)
    finally:
        conn.close()


def test_canonical_source_refuses_a_deviating_ledger_shape_before_writes(
    owned_config: Path,
) -> None:
    """The strict canonical profile requires exactly the canonical fields, in prepare."""
    conn = _canonical_source(owned_config, "malformed-shape.sqlite")
    try:
        conn.execute(f"ALTER TABLE {LEDGER} DROP COLUMN actor")
        conn.commit()
    finally:
        conn.close()

    source = owned_config / "malformed-shape.sqlite"
    before = source.read_bytes()
    target = owned_config / "malformed-target.sqlite"
    with pytest.raises(
        PodImportError, match="Canonical SQLite source ledger fields mismatch"
    ) as error:
        prepare_pod_import(source, "sqlite", target_path=target, stage_snapshots=False)
    assert "missing=['actor']" in str(error.value)
    assert source.read_bytes() == before
    assert not target.exists()

    # An unexpected column on the canonical ledger is refused the same way.
    conn = _canonical_source(owned_config, "extra-column.sqlite")
    try:
        conn.execute(f"ALTER TABLE {LEDGER} ADD COLUMN source_type TEXT")
        conn.commit()
    finally:
        conn.close()

    second = owned_config / "extra-column.sqlite"
    before = second.read_bytes()
    with pytest.raises(
        PodImportError, match="Canonical SQLite source ledger fields mismatch"
    ) as error:
        prepare_pod_import(
            second,
            "sqlite",
            target_path=owned_config / "extra-target.sqlite",
            stage_snapshots=False,
        )
    assert "unexpected=['source_type']" in str(error.value)
    assert second.read_bytes() == before
    assert not (owned_config / "extra-target.sqlite").exists()


def test_canonical_profile_refuses_unvalidated_metadata_before_writes(
    owned_config: Path,
) -> None:
    """Canonical metadata is validated before the canonical profile is taken."""
    contract = load_pod_contract()
    source = owned_config / "metadata-source.sqlite"
    conn = sqlite3.connect(source)
    try:
        conn.execute(
            "CREATE TABLE pod_registry_state (singleton_id, contract_id, contract_version, "
            "dictionary_version, schema_version, source_generation, knowledge_generation)"
        )
        conn.execute(
            "INSERT INTO pod_registry_state VALUES (1, ?, ?, ?, ?, 0, NULL)",
            (
                contract.contract_id,
                contract.contract_version,
                "0.0.0",
                SCHEMA_VERSION,
            ),
        )
        fields = ", ".join(sorted(api_context._CANONICAL_V2_LEDGER_FIELDS))
        conn.execute(f"CREATE TABLE {LEDGER} ({fields})")
        conn.commit()
    finally:
        conn.close()

    target = owned_config / "metadata-target.sqlite"
    with pytest.raises(
        PodImportError, match="SQLite source POD contract metadata is unsupported"
    ):
        prepare_pod_import(source, "sqlite", target_path=target, stage_snapshots=False)
    assert not target.exists()
    assert not Config.get_db_file().exists()


def test_arbitrary_actor_column_is_never_guessed_as_the_canonical_profile(
    owned_config: Path,
) -> None:
    """A source is read under the profile its own declared shape selects, never guessed.

    A ledger that merely happens to have an ``actor`` column is not the canonical current
    registry: without contract metadata the supplied profile still applies, so the arbitrary
    column is not reinterpreted as the issuer and the read refuses before any writer opens.
    """
    conn = sqlite3.connect(owned_config / "guessed-actor.sqlite")
    conn.row_factory = sqlite3.Row
    try:
        conn.execute(f"CREATE TABLE {LEDGER} (pod_id TEXT, actor TEXT, source_type TEXT)")
        with pytest.raises(
            PodImportError, match="identity ledger is missing fields"
        ) as error:
            api_context.read_source_identity_ledger_rows(conn)
        message = str(error.value)
        assert "issued_by" in message
        assert "actor" not in message
    finally:
        conn.close()


def test_present_empty_source_ledger_stays_empty_and_absent_stays_none(
    owned_config: Path,
) -> None:
    """An empty ledger table is ``()``; only a missing ledger object is ``None``."""
    conn = get_sqlite_connection(owned_config / "present-empty-ledger.sqlite")
    try:
        assert api_context.read_source_identity_ledger_rows(conn) == ()
    finally:
        conn.close()

    absent = sqlite3.connect(owned_config / "absent-ledger.sqlite")
    try:
        assert api_context.read_source_identity_ledger_rows(absent) is None
    finally:
        absent.close()


def test_canonical_source_replacement_preserves_protected_document_uuid_and_graph(
    owned_config: Path,
) -> None:
    """The actual current replacement through the public path, protected document and graph.

    The prepared source ledger is the stored canonical ledger with only the issuer renamed,
    the protected document row and its POD link survive the replacement, the identity UUID
    and audit values are unchanged, and the derived graph is rebuilt from SQLite truth.
    """
    target_path = owned_config / "replacement-target.sqlite"
    cache = owned_config / "projection.duckdb"
    target = _canonical_source(owned_config, "replacement-target.sqlite")
    memory = duckdb.connect(":memory:")
    manager: InstanceGraphManager | None = None
    try:
        sequence = target.execute(
            "SELECT pod_id_seq FROM m_pod WHERE pod_id = ?", (POD_DURI_REV1,)
        ).fetchone()[0]
        target.execute(
            "INSERT INTO documents (doc_id, file_name, file_hash, doc_type, doc_topic, "
            "markdown) VALUES ('DOC-PROTECTED', 'protected.pdf', 'hash-protected', "
            "'letter', '[\"pod\"]', '# Keep')"
        )
        target.execute(
            "INSERT INTO pod_document (pod_id_seq, doc_id) VALUES (?, 'DOC-PROTECTED')",
            (sequence,),
        )
        target.commit()
        _rebuild(target, memory)
        documents_before = _raw_documents(target)
        links_before = [
            tuple(row) for row in target.execute("SELECT * FROM pod_document")
        ]
        ledger_before = _ledger_rows(target)
        uuid_before = target.execute(
            "SELECT pod_uuid FROM m_pod WHERE pod_id_seq = ?", (sequence,)
        ).fetchone()[0]
        generation_before = target.execute(
            "SELECT source_generation FROM pod_registry_state WHERE singleton_id = 1"
        ).fetchone()[0]

        incoming = owned_config / "incoming.sqlite"
        clone = sqlite3.connect(incoming)
        try:
            target.backup(clone)
            clone.execute(
                "UPDATE m_pod SET pod_name = 'Duri Current Import' WHERE pod_id_seq = ?",
                (sequence,),
            )
            clone.commit()
        finally:
            clone.close()

        publish_v2_registry(target_path, cache)
        manager = InstanceGraphManager(sqlite_path=target_path)
        assert manager.is_available()
        assert manager.query(
            "MATCH (d:Document {doc_id: 'DOC-PROTECTED'})-[:ABOUT_POD]->(p:POD) "
            "RETURN p.pod_id"
        ) == [{"p.pod_id": POD_DURI_REV1}]

        batch = prepare_pod_import(
            incoming, "sqlite", target_path=target_path, stage_snapshots=False
        )
        prepared = sorted(
            (dict(row) for row in batch.identity_ledger or ()),
            key=lambda row: cast(int, row["pod_id_seq"]),
        )
        assert prepared == _renamed(ledger_before)

        load_prepared_pod_batch(batch, sqlite_path=target_path, duck_path=cache)

        generation_after = target.execute(
            "SELECT source_generation FROM pod_registry_state WHERE singleton_id = 1"
        ).fetchone()[0]
        assert generation_after == generation_before + 1
        assert _raw_documents(target) == documents_before
        links_after = [tuple(row) for row in target.execute("SELECT * FROM pod_document")]
        assert links_after == links_before
        assert target.execute(
            "SELECT pod_uuid FROM m_pod WHERE pod_id_seq = ?", (sequence,)
        ).fetchone()[0] == uuid_before
        assert _ledger_rows(target) == ledger_before
        assert target.execute(
            "SELECT pod_name FROM m_pod WHERE pod_id_seq = ?", (sequence,)
        ).fetchone()[0] == "Duri Current Import"
        assert not manager.is_available(), "the stale graph must be unavailable after load"

        _rebuild(target, memory)
        assert manager.is_available()
        assert manager.query(
            f"MATCH (p:POD) WHERE p.pod_id = '{POD_DURI_REV1}' RETURN p.pod_name"
        ) == [{"p.pod_name": "Duri Current Import"}]
        assert manager.query(
            "MATCH (d:Document {doc_id: 'DOC-PROTECTED'})-[:ABOUT_POD]->(p:POD) "
            "RETURN p.pod_id"
        ) == [{"p.pod_id": POD_DURI_REV1}]
        projection = duckdb.connect(str(cache), read_only=True)
        try:
            projection_row = projection.execute(
                "SELECT pod_name FROM m_pod WHERE pod_id_seq = ?", (sequence,)
            ).fetchone()
            assert projection_row is not None
            assert projection_row[0] == "Duri Current Import"
        finally:
            projection.close()
    finally:
        if manager is not None:
            manager.close()
        memory.close()
        target.close()


def test_supplied_epod_source_profile_still_prepares_unchanged(
    owned_config: Path,
) -> None:
    """The supplied ePOD source keeps its own profile and its own row representation."""
    from esdc.pod_registry.migrations import _sqlite_value
    from tests.pod_registry.test_importer import _row

    contract = load_pod_contract()
    source = owned_config / "epod-source.sqlite"
    test_import_v2._create_sqlite_source(source, contract)
    conn = sqlite3.connect(source)
    conn.row_factory = sqlite3.Row
    try:
        master = _row(contract, "m_pod")
        master.update(
            pod_id_seq=7,
            pod_id="PL-2024-0001-4-1-0",
            approval_date="2024-01-01",
            pod_order_no=1,
            institution_id=4,
            pod_type_id=1,
        )
        names = ", ".join(api_context.quote_identifier(name) for name in master)
        placeholders = ", ".join("?" for _ in master)
        conn.execute(
            f"INSERT INTO m_pod ({names}) VALUES ({placeholders})",
            tuple(_sqlite_value(value) for value in master.values()),
        )
        conn.execute(
            f"INSERT INTO {LEDGER} VALUES (2024, '2024-01-01', 1, 1, 7, "
            "'PL-2024-0001-4-1-0', '2024-01-01T00:00:00.000Z', 'FA:test-epod-issuer')"
        )
        conn.commit()
        original = _ledger_rows(conn)
    finally:
        conn.close()

    target = owned_config / "epod-target.sqlite"
    batch = prepare_pod_import(source, "sqlite", target_path=target, stage_snapshots=False)
    assert [dict(row) for row in batch.identity_ledger or ()] == original
    assert not target.exists()
