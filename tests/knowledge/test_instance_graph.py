"""Tests for the instance graph."""

from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
import threading
from pathlib import Path
from typing import Any

import pytest
import yaml

import esdc.chat.domain_knowledge.instance_graph as graph_module
from esdc.chat.domain_knowledge.instance_graph import (
    InstanceGraphManager,
    _load_instance_graph_schema,
    _node_ddl,
    _rel_ddl,
)
from esdc.configs import Config
from esdc.knowledge.freshness import (
    compute_knowledge_basis,
    invalidate_knowledge_after_dependency_change,
    mark_knowledge_current,
)
from esdc.knowledge.linker import run_deterministic_linking
from esdc.knowledge.store import Edge, KnowledgeStore
from esdc.pod_registry.store import (
    get_sqlite_connection,
    get_sqlite_readonly_connection,
    get_sqlite_writer_connection,
)
from tests.knowledge.conftest import (
    ACTOR,
    POD_DURI,
    POD_DURI_REV1,
    POD_KAMPUNG,
    PROJECT_DURI,
    PROJECT_KAMPUNG,
    _seed_knowledge_world,
)

# Captured at import time, before the autouse isolation fixtures replace this
# getter. The constructor regression restores only Config.get_db_file from this
# capture, so the guideline, cache and other autouse isolation stays in place.
_REAL_GET_DB_FILE = Config.get_db_file


@pytest.fixture
def learned_sqlite(sqlite_conn, duck_conn, tmp_path: Path) -> Path:
    store = KnowledgeStore(sqlite_conn)
    store.ensure_tables()
    run_deterministic_linking(sqlite_conn, duck_conn, store)
    sqlite_conn.commit()
    return tmp_path / "esdc.sqlite"


@pytest.fixture(autouse=True)
def _reset_singleton():
    InstanceGraphManager.reset_for_tests()
    yield
    InstanceGraphManager.reset_for_tests()


def test_unavailable_before_learn(tmp_path: Path):
    """A current registry whose derived knowledge is not marked current is not loaded."""
    conn = get_sqlite_connection(tmp_path / "fresh.sqlite")
    conn.close()
    mgr = InstanceGraphManager(sqlite_path=tmp_path / "fresh.sqlite")
    assert mgr.is_available() is False


def _finish_source_write(conn: sqlite3.Connection) -> None:
    """Close a source write the way reconciliation does.

    A source-row insert invalidates the knowledge basis, so the write boundary
    re-marks it for the same snapshot the readers measure. Without that the test
    would be reading a correctly-refused stale knowledge store.
    """
    if not conn.in_transaction:
        conn.execute("BEGIN IMMEDIATE")
    assert mark_knowledge_current(conn, compute_knowledge_basis(conn))
    conn.commit()


PROJECT_DURI_NAME = "Duri Steamflood"
PROJECT_KAMPUNG_NAME = "Kampung Baru Dev"
PROJECT_DURI_RENAMED = "Duri Steamflood Phase 2"


def _project_names(mgr: InstanceGraphManager) -> dict[str, str]:
    """Read the projected project names through the public API only."""
    return {
        str(row["pr.project_id"]): str(row["pr.project_name"])
        for row in mgr.query(
            "MATCH (pr:Project) RETURN pr.project_id, pr.project_name "
            "ORDER BY pr.project_id"
        )
    }


# -- cache lifecycle follows the committed markers ---------------------


def test_same_manager_follows_committed_marker_changes(
    learned_sqlite: Path,
    sqlite_conn: sqlite3.Connection,
):
    """First RED: one manager must stop serving a graph its markers replaced.

    The same manager builds the canonical world, then a source write moves
    ``source_generation`` forward with knowledge marked stale at the write
    boundary, so the old in-memory projection must no longer be served. When the
    producers complete mark-current for the new source state, that same manager
    rebuilds and exposes the changed master name. `_ensure_built()` returns
    forever while `_built` is true, so this fails before the lifecycle work.
    """
    mgr = InstanceGraphManager(sqlite_path=learned_sqlite)
    assert mgr.is_available() is True
    assert _project_names(mgr) == {
        PROJECT_DURI: PROJECT_DURI_NAME,
        PROJECT_KAMPUNG: PROJECT_KAMPUNG_NAME,
    }

    source_generation = int(
        sqlite_conn.execute(
            "SELECT source_generation FROM pod_registry_state WHERE singleton_id=1"
        ).fetchone()[0]
    )

    sqlite_conn.execute("BEGIN IMMEDIATE")
    sqlite_conn.execute(
        "UPDATE m_project SET project_name = ? WHERE project_id = ?",
        (PROJECT_DURI_RENAMED, PROJECT_DURI),
    )
    sqlite_conn.execute(
        "UPDATE pod_registry_state SET source_generation = ?, "
        "knowledge_generation = NULL WHERE singleton_id = 1",
        [source_generation + 1],
    )
    sqlite_conn.commit()

    assert mgr.is_available() is False, (
        "a moved source generation with stale knowledge must not serve the "
        "old graph from the same manager"
    )

    sqlite_conn.execute("BEGIN IMMEDIATE")
    basis = compute_knowledge_basis(sqlite_conn)
    assert mark_knowledge_current(sqlite_conn, basis)
    sqlite_conn.commit()

    assert mgr.is_available() is True
    assert _project_names(mgr) == {
        PROJECT_DURI: PROJECT_DURI_RENAMED,
        PROJECT_KAMPUNG: PROJECT_KAMPUNG_NAME,
    }
    mgr.close()


def test_same_generation_marker_replacement_rebuilds_links(
    learned_sqlite: Path,
    sqlite_conn: sqlite3.Connection,
):
    """A marker replaced at the same generation must still rebuild the projection.

    A corpus-only dependency change does not advance `source_generation`, so a
    cache that compared generations alone would keep serving the old graph. The
    producers invalidate the stored basis at the write boundary and then complete
    mark-current for the new one; the same manager must refuse the replaced
    marker and rebuild from a new snapshot.
    """
    mgr = InstanceGraphManager(sqlite_path=learned_sqlite)
    assert mgr.is_available() is True

    def links() -> set[tuple[str, str]]:
        return {
            (str(row["d.doc_id"]), str(row["f.name"]))
            for row in mgr.query(
                "MATCH (d:Document)-[:ABOUT_FIELD]->(f:Field) RETURN d.doc_id, f.name"
            )
        }

    markers_before = mgr._build_markers
    assert links() == {("DOC-A", "Duri"), ("DOC-B", "Duri")}
    source_generation = int(
        sqlite_conn.execute(
            "SELECT source_generation FROM pod_registry_state WHERE singleton_id=1"
        ).fetchone()[0]
    )

    sqlite_conn.execute("BEGIN IMMEDIATE")
    sqlite_conn.execute(
        "INSERT INTO documents (doc_id, file_hash, doc_type, doc_topic, doc_number, "
        "doc_date, subject, wk_name, field_name, project_name, pod_name, "
        "suggested_pod_ids, markdown) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            "DOC-D",
            "hash-d",
            "letter",
            '["pod"]',
            "SRT-0300/SKKMA0000/2024/S1",
            "2024-05-01",
            "Kampung Baru review",
            "Pertamina EP",
            "Kampung Baru",
            None,
            "POD Kampung Baru",
            json.dumps([POD_KAMPUNG]),
            "# Review\nKampung Baru scope.",
        ),
    )
    assert invalidate_knowledge_after_dependency_change(sqlite_conn, actor=ACTOR)
    sqlite_conn.commit()

    # The stale boundary: the same manager must not serve the captured projection.
    assert mgr.is_available() is False

    sqlite_conn.execute("BEGIN IMMEDIATE")
    assert mark_knowledge_current(sqlite_conn, compute_knowledge_basis(sqlite_conn))
    # The producers' pass for the new basis writes this link.
    store = KnowledgeStore(sqlite_conn, allow_stale=True)
    store.upsert_edges(
        [
            Edge(
                "document",
                "DOC-D",
                "ABOUT_FIELD",
                "field",
                "Kampung Baru",
                confidence=1.0,
                method="test",
            )
        ],
        commit=False,
    )
    sqlite_conn.commit()

    assert (
        int(
            sqlite_conn.execute(
                "SELECT source_generation FROM pod_registry_state WHERE singleton_id=1"
            ).fetchone()[0]
        )
        == source_generation
    ), "a corpus change must not advance the registry generation"

    assert mgr.is_available() is True
    assert links() == {("DOC-D", "Kampung Baru")}, (
        "the rebuilt projection must reflect the current committed source"
    )
    assert mgr._build_markers != markers_before, "the marker replacement invalidated it"
    mgr.close()


def test_corrupted_marker_never_serves_the_cached_graph_or_writes_the_source(
    learned_sqlite: Path,
    sqlite_conn: sqlite3.Connection,
    tmp_path: Path,
):
    """A missing marker row refuses the cache; the graph cannot repair the source."""
    mgr = InstanceGraphManager(sqlite_path=learned_sqlite)
    assert mgr.is_available() is True

    sqlite_conn.execute("BEGIN IMMEDIATE")
    sqlite_conn.execute("DELETE FROM pod_knowledge_state WHERE singleton_id=1")
    sqlite_conn.commit()

    # Captured after the test's own write, so anything the graph adds is its own.
    before_files = set(tmp_path.iterdir())
    assert mgr.is_available() is False
    assert mgr._conn is None and mgr._db is None, "the old graph must be discarded"
    assert set(tmp_path.iterdir()) == before_files, (
        "a refused marker read must not create side files or initialize the source"
    )
    assert (
        sqlite_conn.execute(
            "SELECT COUNT(*) FROM pod_knowledge_state WHERE singleton_id=1"
        ).fetchone()[0]
        == 0
    ), "a refused reader must not write the marker back"

    # The same manager can rebuild once the producers complete the marker.
    sqlite_conn.execute("BEGIN IMMEDIATE")
    assert mark_knowledge_current(sqlite_conn, compute_knowledge_basis(sqlite_conn))
    sqlite_conn.commit()
    assert mgr.is_available() is True
    assert _project_names(mgr) == {
        PROJECT_DURI: PROJECT_DURI_NAME,
        PROJECT_KAMPUNG: PROJECT_KAMPUNG_NAME,
    }
    mgr.close()


def test_source_created_after_a_refusal_becomes_available_on_the_same_manager(
    tmp_path: Path,
):
    """The first refusal creates nothing; a later current source serves the same manager."""
    source = tmp_path / "esdc.sqlite"
    mgr = InstanceGraphManager(sqlite_path=source)
    assert mgr.is_available() is False
    assert not source.exists(), "a missing source is refused, never created"

    # The test owns the write boundary: canonical schema, seeded world, and the
    # knowledge marker measured at that boundary.
    conn = get_sqlite_connection(source)
    try:
        _seed_knowledge_world(conn)
        store = KnowledgeStore(conn)
        store.ensure_tables()
        store.upsert_edges(
            [
                Edge(
                    "document",
                    "DOC-A",
                    "ABOUT_PROJECT",
                    "project",
                    PROJECT_DURI,
                    confidence=1.0,
                    method="test",
                )
            ]
        )
    finally:
        conn.close()

    assert mgr.is_available() is True
    # Generic entity nodes are projected from the rows an edge references, so the
    # names visible here are the ones this test linked to the master.
    assert _project_names(mgr) == {PROJECT_DURI: PROJECT_DURI_NAME}
    pods = {row["p.pod_id"] for row in mgr.query("MATCH (p:POD) RETURN p.pod_id")}
    assert pods == {POD_DURI, POD_DURI_REV1, POD_KAMPUNG}
    mgr.close()


def test_marker_read_is_its_own_short_read_only_snapshot(
    learned_sqlite: Path, monkeypatch
):
    """Every public access reads the markers on its own connection, per call.

    The marker read is a short read transaction through the shared read-only
    policy, separate from the build's snapshot connection, and is released before
    the call returns. It can neither write nor upgrade the source.
    """
    mgr = InstanceGraphManager(sqlite_path=learned_sqlite)
    original_opener = graph_module.get_sqlite_readonly_connection
    original_read = mgr._read_source_snapshot
    opened: list[dict[str, Any]] = []
    captured: dict[str, Any] = {}

    def open_read(path: Path) -> sqlite3.Connection | None:
        conn = original_opener(path)
        if conn is None:
            return None
        record: dict[str, Any] = {"conn": conn}
        try:
            conn.execute(
                "INSERT INTO kg_edge (src_type, src_id, rel, dst_type, dst_id) "
                "VALUES ('pod', 'x', 'ABOUT_POD', 'pod', 'y')"
            )
        except sqlite3.OperationalError as e:
            record["write_error"] = str(e)
        # A failed write can still leave an implicit transaction open; clear it so
        # the caller's own read transaction is the only one on this connection.
        conn.rollback()
        opened.append(record)
        return conn

    def read(conn: sqlite3.Connection) -> Any:
        captured["build_conn"] = conn
        return original_read(conn)

    monkeypatch.setattr(graph_module, "get_sqlite_readonly_connection", open_read)
    monkeypatch.setattr(mgr, "_read_source_snapshot", read)

    assert mgr.is_available() is True
    first_call = list(opened)
    # Pre-build marker read, the build's own snapshot, then the post-build re-read
    # that decides whether the projection may be published.
    assert len(first_call) == 3, f"expected three short reads, got {len(first_call)}"
    assert first_call[0]["conn"] is not captured["build_conn"], (
        "the marker read is not the build snapshot"
    )
    assert first_call[2]["conn"] is not first_call[0]["conn"]
    for record in first_call:
        assert "readonly" in record["write_error"], (
            "the marker read must stay on the read-only policy"
        )
        with pytest.raises(sqlite3.ProgrammingError):
            record["conn"].execute("SELECT 1")

    opened.clear()
    assert mgr.is_available() is True
    assert len(opened) == 1, "a cache hit is one marker read on one short connection"
    assert opened[0]["conn"] is not first_call[0]["conn"], (
        "connections are per call, not kept across accesses"
    )
    with pytest.raises(sqlite3.ProgrammingError):
        opened[0]["conn"].execute("SELECT 1")
    mgr.close()


def test_mtime_and_data_version_are_not_the_cache_signal(
    learned_sqlite: Path,
    sqlite_conn: sqlite3.Connection,
):
    """Reuse is decided by the committed marker, not by filesystem or per-connection state.

    Every WAL write moves mtime and every new reader sees a different
    `PRAGMA data_version`, so neither can be a cross-call fingerprint. A marker
    that still matches the captured build basis reuses the cache; producers always
    rewrite the committed marker inside their write transaction, so a corpus write
    that does not move it is not a committed knowledge transition.
    """
    mgr = InstanceGraphManager(sqlite_path=learned_sqlite)
    assert mgr.is_available() is True
    connection = mgr._conn
    markers = mgr._build_markers

    stat = learned_sqlite.stat()
    os.utime(learned_sqlite, (stat.st_atime, stat.st_mtime + 10_000))
    assert learned_sqlite.stat().st_mtime_ns != stat.st_mtime_ns

    writer = get_sqlite_writer_connection(learned_sqlite)
    try:
        writer.execute("BEGIN IMMEDIATE")
        writer.execute(
            "INSERT INTO kg_edge (src_type, src_id, rel, dst_type, dst_id, "
            "confidence, method) VALUES "
            "('document','DOC-F','ABOUT_FIELD','field','Duri',0.9,'no_marker_write')"
        )
        writer.commit()
    finally:
        writer.close()

    assert mgr.is_available() is True
    assert mgr._conn is connection, "an unchanged committed marker reuses the cache"
    assert mgr._build_markers == markers

    # A real dependency change at its write boundary moves the committed marker, so
    # the same manager drops the projection captured from the previous one.
    sqlite_conn.execute("BEGIN IMMEDIATE")
    sqlite_conn.execute(
        "INSERT INTO documents (doc_id, file_hash, doc_type, doc_topic, doc_number, "
        "doc_date, subject, markdown) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            "DOC-G",
            "hash-g",
            "letter",
            '["pod"]',
            "SRT-0500/SKKMA0000/2025/S1",
            "2025-03-01",
            "Marker moved by a real dependency change",
            "# Change",
        ),
    )
    assert invalidate_knowledge_after_dependency_change(sqlite_conn, actor=ACTOR)
    sqlite_conn.commit()

    # The captured basis is the previous marker; the committed marker now describes
    # a different state, so the cache and its basis are gone.
    assert mgr._read_committed_markers() is None, (
        "the replaced marker is not the basis the projection was captured from"
    )
    assert mgr.is_available() is False, "the stale boundary serves nothing"
    assert mgr._conn is None, "the projection captured from the old marker is discarded"
    assert mgr._build_markers is None, "a discarded cache no longer has a basis"

    # The same manager rebuilds once the producers complete the marker.
    sqlite_conn.execute("BEGIN IMMEDIATE")
    assert mark_knowledge_current(sqlite_conn, compute_knowledge_basis(sqlite_conn))
    KnowledgeStore(sqlite_conn, allow_stale=True).upsert_edges(
        [
            Edge(
                "document",
                "DOC-G",
                "ABOUT_FIELD",
                "field",
                "Duri",
                confidence=1.0,
                method="test",
            )
        ],
        commit=False,
    )
    sqlite_conn.commit()

    assert mgr.is_available() is True
    assert mgr._conn is not connection, "the same manager rebuilt on the new marker"
    assert {
        row["d.doc_id"]
        for row in mgr.query(
            "MATCH (d:Document)-[:ABOUT_FIELD]->(f:Field) RETURN d.doc_id"
        )
    } == {"DOC-G"}
    mgr.close()


def test_second_writer_commit_during_node_loading_discards_the_projection(
    learned_sqlite: Path,
    monkeypatch,
):
    """Node loading reads one snapshot, and a commit during it cannot publish a mix.

    A second WAL writer commits inside the projection. The open reader keeps the
    captured snapshot, so the first projection cannot contain its row; the
    committed marker moved, so that projection is discarded and the same manager
    rebuilds from the current source. The published graph must be the second
    attempt, never the first projection with a new status stamped on it.
    """
    mgr = InstanceGraphManager(sqlite_path=learned_sqlite)
    original_read = mgr._read_source_snapshot
    original_project = mgr._project_graph
    captured: dict[str, Any] = {}
    attempts: list[list[tuple[Any, ...]]] = []
    observations: dict[str, Any] = {}

    def edge_rows(conn: sqlite3.Connection) -> list[tuple[Any, ...]]:
        return [
            tuple(row)
            for row in conn.execute(
                "SELECT src_type, src_id, rel, dst_type, dst_id FROM kg_edge "
                "ORDER BY 1, 2, 3, 4, 5"
            ).fetchall()
        ]

    def read(conn: sqlite3.Connection) -> Any:
        captured["conn"] = conn
        return original_read(conn)

    def project(snapshot: Any) -> None:
        attempts.append([tuple(row[:5]) for row in snapshot.edges])
        observations[f"graph_attempt_{len(attempts)}"] = mgr._db
        if len(attempts) == 1:
            reader = captured["conn"]
            observations["before"] = edge_rows(reader)
            writer = get_sqlite_writer_connection(learned_sqlite)
            try:
                writer.execute("BEGIN IMMEDIATE")
                writer.execute(
                    "INSERT INTO documents (doc_id, file_hash, doc_type, doc_topic, "
                    "doc_number, doc_date, subject, markdown) VALUES "
                    "('DOC-E', 'hash-e', 'letter', '[\"pod\"]', 'SRT-0400/2025', "
                    "'2025-02-01', 'Committed during load', '# Injected')"
                )
                writer.execute(
                    "INSERT INTO kg_edge (src_type, src_id, rel, dst_type, dst_id, "
                    "confidence, method) VALUES "
                    "('document','DOC-E','ABOUT_PROJECT','project','P-TEST002-01',"
                    "0.9,'mid_load_writer')"
                )
                assert mark_knowledge_current(writer, compute_knowledge_basis(writer))
                writer.commit()
            finally:
                writer.close()
            # The open reader is still on the captured snapshot.
            observations["during"] = edge_rows(reader)
        original_project(snapshot)

    monkeypatch.setattr(mgr, "_read_source_snapshot", read)
    monkeypatch.setattr(mgr, "_project_graph", project)

    assert mgr.is_available() is True
    injected = ("document", "DOC-E", "ABOUT_PROJECT", "project", PROJECT_KAMPUNG)
    assert len(attempts) == 2, "the moved marker is retried once, then bounded"
    assert attempts[0] == observations["before"] == observations["during"]
    assert injected not in attempts[0], "the first projection is the captured snapshot"
    assert injected in attempts[1], "the published graph is the current committed state"
    assert observations["graph_attempt_1"] is not mgr._db, (
        "the first projection was discarded, not published"
    )

    def graph_links() -> set[tuple[str, str]]:
        return {
            (str(row["d.doc_id"]), str(row["pr.project_id"]))
            for row in mgr.query(
                "MATCH (d:Document)-[:ABOUT_PROJECT]->(pr:Project) "
                "RETURN d.doc_id, pr.project_id"
            )
        }

    first_projection_links = {
        (row[1], row[4])
        for row in attempts[0]
        if row[2] == "ABOUT_PROJECT" and row[3] == "project"
    }
    published = graph_links()
    assert published == {
        (row[1], row[4]) for row in attempts[1] if row[2] == "ABOUT_PROJECT"
    }
    assert published != first_projection_links, "no mixed new-status/old-graph result"

    # The published basis is the marker re-read after the snapshot closed.
    current = graph_module.get_sqlite_readonly_connection(learned_sqlite)
    assert current is not None
    try:
        current.execute("BEGIN")
        markers = graph_module.knowledge_freshness_markers(current)
        assert markers.fresh
    finally:
        current.close()
    assert mgr._build_markers is not None
    assert mgr._build_markers[0] == markers.source_generation
    assert mgr._build_markers[2] == markers.stored_basis_sha256

    # A later current marker lets the same manager rebuild again.
    later_writer = get_sqlite_writer_connection(learned_sqlite)
    try:
        later_writer.execute("BEGIN IMMEDIATE")
        later_writer.execute(
            "UPDATE m_project SET project_name = ? WHERE project_id = ?",
            (PROJECT_DURI_RENAMED, PROJECT_DURI),
        )
        generation = int(
            later_writer.execute(
                "SELECT source_generation FROM pod_registry_state WHERE singleton_id=1"
            ).fetchone()[0]
        )
        later_writer.execute(
            "UPDATE pod_registry_state SET source_generation = ?, "
            "knowledge_generation = NULL WHERE singleton_id = 1",
            [generation + 1],
        )
        later_writer.execute(
            "INSERT INTO kg_edge (src_type, src_id, rel, dst_type, dst_id, "
            "confidence, method) VALUES "
            "('document','DOC-E','ABOUT_PROJECT','project','P-TEST001-01',0.9,'later')"
        )
        assert mark_knowledge_current(
            later_writer, compute_knowledge_basis(later_writer)
        )
        later_writer.commit()
    finally:
        later_writer.close()

    assert mgr.is_available() is True
    assert graph_links() == {
        ("DOC-E", PROJECT_KAMPUNG),
        ("DOC-E", PROJECT_DURI),
    }
    assert _project_names(mgr)[PROJECT_DURI] == PROJECT_DURI_RENAMED
    mgr.close()


def test_retry_is_bounded_and_never_loops(learned_sqlite: Path, monkeypatch):
    """A marker that keeps moving stops at the attempt bound instead of rebuilding forever."""
    mgr = InstanceGraphManager(sqlite_path=learned_sqlite)
    original_markers = mgr._read_committed_markers
    calls: dict[str, int] = {"n": 0}

    def moving_markers() -> tuple[Any, ...] | None:
        calls["n"] += 1
        markers = original_markers()
        if markers is None:
            return None
        # Each read reports a different committed state, so no projection built
        # against it can be published.
        return markers + (calls["n"],)

    monkeypatch.setattr(mgr, "_read_committed_markers", moving_markers)

    assert mgr.is_available() is False
    # Two attempts, each with the pre-build read and the post-build re-read.
    assert calls["n"] == graph_module._BUILD_ATTEMPTS * 2, (
        f"the retry must stop at its bound, got {calls['n']} marker reads"
    )
    assert mgr._conn is None and mgr._db is None, (
        "a projection that no longer matches its markers is not served"
    )

    # The same manager rebuilds for a current marker; nothing is permanently broken.
    mgr._read_committed_markers = original_markers
    assert mgr.is_available() is True
    assert _project_names(mgr) == {
        PROJECT_DURI: PROJECT_DURI_NAME,
        PROJECT_KAMPUNG: PROJECT_KAMPUNG_NAME,
    }
    mgr.close()


def test_invalidation_waits_for_a_query_in_progress(
    learned_sqlite: Path,
    monkeypatch,
):
    """An automatic invalidation cannot close a Ladybug connection another thread is using.

    One thread executes on the graph while holding the lifecycle section; the
    invalidating call must block until that execution ends, then rebuild on the
    moved marker. The query in flight completes on the same connection, so nothing
    is torn down underneath it.
    """
    mgr = InstanceGraphManager(sqlite_path=learned_sqlite)
    assert mgr.is_available() is True
    connection = mgr._conn
    original_execute = mgr._execute_cypher

    executing = threading.Event()
    releaser = threading.Event()
    attempted = threading.Event()
    finished = threading.Event()
    observations: dict[str, Any] = {}
    invalidation_result: list[bool] = []
    query_result: list[list[dict[str, Any]]] = []

    def execute(cypher: str) -> list[list[Any]]:
        if not executing.is_set():
            observations["connection_while_executing"] = mgr._conn
            executing.set()
            # The invalidator has reached the lifecycle lock and is waiting here.
            if not releaser.wait(10.0):
                raise AssertionError(
                    "the invalidation never reached the lifecycle lock"
                )
            observations["connection_after_invalidation_attempt"] = mgr._conn
        return original_execute(cypher)

    def invalidator() -> None:
        if not executing.wait(10.0):
            observations["invalidator_saw_executing"] = False
            return
        observations["invalidator_saw_executing"] = True
        writer = get_sqlite_writer_connection(learned_sqlite)
        try:
            writer.execute("BEGIN IMMEDIATE")
            writer.execute(
                "UPDATE m_project SET project_name = ? WHERE project_id = ?",
                (PROJECT_DURI_RENAMED, PROJECT_DURI),
            )
            generation = int(
                writer.execute(
                    "SELECT source_generation FROM pod_registry_state "
                    "WHERE singleton_id=1"
                ).fetchone()[0]
            )
            writer.execute(
                "UPDATE pod_registry_state SET source_generation = ?, "
                "knowledge_generation = NULL WHERE singleton_id = 1",
                [generation + 1],
            )
            writer.commit()
        finally:
            writer.close()

        attempted.set()
        invalidation_result.append(mgr.is_available())
        finished.set()

    def searcher() -> None:
        query_result.append(mgr.find(PROJECT_DURI_NAME, entity_type="project"))

    monkeypatch.setattr(mgr, "_execute_cypher", execute)

    query_thread = threading.Thread(target=searcher, name="query-in-progress")
    invalidator_thread = threading.Thread(target=invalidator, name="invalidator")
    query_thread.start()
    try:
        assert executing.wait(10.0), "the query never executed"
        invalidator_thread.start()
        assert attempted.wait(10.0), "the invalidation never started"
        assert not finished.wait(0.5), (
            "invalidation must wait for the query holding the connection"
        )
        assert observations["connection_while_executing"] is connection
        assert observations["invalidator_saw_executing"] is True
        releaser.set()
        invalidator_thread.join(10.0)
        query_thread.join(10.0)

        assert finished.is_set() and query_result
        assert query_result[0], "the query in flight completed on its own connection"
        assert query_result[0][0]["entity_id"] == PROJECT_DURI, (
            "the query in flight answered from its own connection"
        )
        # Recorded before the query released the section, so the invalidator had
        # not been able to close anything.
        assert observations["connection_after_invalidation_attempt"] is connection
        assert invalidation_result == [False], (
            "the moved marker with stale knowledge is refused, not answered by "
            "the old graph"
        )
        assert mgr._conn is None, (
            "the connection was released only after the query ended"
        )

        # Completing mark-current lets the same manager rebuild; nothing stayed
        # permanently broken.
        writer = get_sqlite_writer_connection(learned_sqlite)
        try:
            writer.execute("BEGIN IMMEDIATE")
            assert mark_knowledge_current(writer, compute_knowledge_basis(writer))
            writer.commit()
        finally:
            writer.close()

        assert mgr.is_available() is True
        assert mgr._conn is not connection, (
            "the same manager rebuilt on the moved marker"
        )
        assert _project_names(mgr)[PROJECT_DURI] == PROJECT_DURI_RENAMED
    finally:
        releaser.set()
        invalidator_thread.join(10.0)
        query_thread.join(10.0)
        mgr.close()


def test_concurrent_construction_cannot_reinitialize_a_live_cache(
    learned_sqlite: Path,
    monkeypatch,
):
    """Two no-arg constructions initialize the singleton exactly once.

    The first constructor is held inside its source-path resolution while it owns
    the initialization section. The second cannot install a second lifecycle lock or
    reset the state behind it; it waits for that section and then returns the
    instance that was already initialized, so a native graph published on the shared
    singleton is never discarded by a later construction.
    """
    installs: list[dict[str, Any]] = []
    original_install = InstanceGraphManager._install_cache_state

    def install(self: InstanceGraphManager, sqlite_path: Path) -> None:
        original_install(self, sqlite_path)
        installs.append(
            {"instance": self, "path": sqlite_path, "lock": self._lifecycle_lock}
        )

    monkeypatch.setattr(InstanceGraphManager, "_install_cache_state", install)

    resolving = threading.Event()
    release = threading.Event()

    def resolve_path() -> Path:
        if not resolving.is_set():
            resolving.set()
            if not release.wait(10.0):
                raise AssertionError("the constructor gate was never released")
        return learned_sqlite

    monkeypatch.setattr(graph_module, "get_esdc_sqlite_path", resolve_path)

    first: list[InstanceGraphManager] = []
    second: list[InstanceGraphManager] = []
    second_done = threading.Event()
    observations: dict[str, Any] = {}

    def construct_first() -> None:
        first.append(InstanceGraphManager())

    def construct_second() -> None:
        observations["installs_while_first_was_open"] = len(installs)
        second.append(InstanceGraphManager())
        observations["installs_when_second_returned"] = len(installs)
        second_done.set()

    first_thread = threading.Thread(target=construct_first, name="first-constructor")
    second_thread = threading.Thread(target=construct_second, name="second-constructor")
    first_thread.start()
    try:
        assert resolving.wait(10.0), "the first constructor never resolved the path"
        second_thread.start()
        assert not second_done.wait(1.0), (
            "the second construction must not install state behind an open init"
        )
        assert observations["installs_while_first_was_open"] == 0, (
            "the open initialization section owns the only state install"
        )

        release.set()
        first_thread.join(10.0)
        second_thread.join(10.0)

        assert observations["installs_when_second_returned"] == 1, (
            "the second construction returned without re-initializing"
        )

        assert first and second
        assert first[0] is second[0], "both constructions got the same singleton"
        assert len(installs) == 1, "initialization happened exactly once"
        lock = installs[0]["lock"]
        mgr = first[0]
        assert mgr._lifecycle_lock is lock
        assert mgr._sqlite_path == learned_sqlite

        # A native graph published on the shared instance survives later constructions.
        assert mgr.is_available() is True
        graph_connection = mgr._conn
        later = InstanceGraphManager()
        assert later is mgr
        assert later._lifecycle_lock is lock, (
            "a completed initialization cannot replace a live manager's lock"
        )
        assert later._conn is graph_connection, (
            "a completed initialization cannot discard a live native graph"
        )
        assert len(installs) == 1, "later constructions do not re-initialize"
        assert _project_names(mgr) == {
            PROJECT_DURI: PROJECT_DURI_NAME,
            PROJECT_KAMPUNG: PROJECT_KAMPUNG_NAME,
        }
    finally:
        release.set()
        first_thread.join(10.0)
        second_thread.join(10.0)
        # Closes the singleton's native handle outside the class lock.
        InstanceGraphManager.reset_for_tests()


def test_current_projection_builds_from_sqlite_only(learned_sqlite: Path):
    """First RED: the graph is a projection of the current SQLite source.

    Built from the canonical seeded-and-linked fixture with no Duck resource
    mirror available at all (the shared conftest points Config.get_db_file at a
    path that does not exist), the graph must still be available and expose the
    three canonical issued POD IDs and the linked projects named from current
    `m_project`. The old source joins `r_pod_type.code = m.pod_type_code`, which
    the single current schema does not have.
    """
    mgr = InstanceGraphManager(sqlite_path=learned_sqlite)
    assert mgr.is_available() is True

    pods = {row["p.pod_id"] for row in mgr.query("MATCH (p:POD) RETURN p.pod_id")}
    assert pods == {POD_DURI, POD_DURI_REV1, POD_KAMPUNG}

    projects = {
        row["pr.project_id"]: row["pr.project_name"]
        for row in mgr.query(
            "MATCH (pr:Project) RETURN pr.project_id, pr.project_name "
            "ORDER BY pr.project_id"
        )
    }
    assert projects == {PROJECT_DURI: PROJECT_DURI_NAME, PROJECT_KAMPUNG: PROJECT_KAMPUNG_NAME}

    hits = mgr.find(PROJECT_DURI_NAME, entity_type="project")
    assert hits and hits[0]["entity_id"] == PROJECT_DURI


# -- read-only source and single captured snapshot --------------------


def test_read_snapshot_cannot_write_and_ends_at_build_close(
    learned_sqlite: Path, monkeypatch
):
    """The build uses the shared read-only policy and ends its read transaction.

    The live source is WAL, so neither `immutable=1` nor a writer upgrade is
    permitted: the same connection must report the source's journal mode, refuse
    a write, stay open through the projection, and be closed once the build ends.
    """
    mgr = InstanceGraphManager(sqlite_path=learned_sqlite)
    original_read = mgr._read_source_snapshot
    original_project = mgr._project_graph
    captured: dict[str, Any] = {}

    def read(conn: sqlite3.Connection) -> Any:
        captured["conn"] = conn
        return original_read(conn)

    def project(snapshot: Any) -> None:
        conn = captured["conn"]
        captured["journal_mode"] = conn.execute("PRAGMA journal_mode").fetchone()[0]
        try:
            conn.execute(
                "INSERT INTO kg_edge (src_type, src_id, rel, dst_type, dst_id, method) "
                "VALUES ('pod', 'x', 'ABOUT_POD', 'pod', 'y', 'test')"
            )
        except sqlite3.OperationalError as e:
            captured["write_error"] = str(e)
        captured["open_during_projection"] = conn.in_transaction
        original_project(snapshot)
        # The read transaction is only ended when the build itself ends.
        captured["open_at_build_end"] = conn.in_transaction

    monkeypatch.setattr(mgr, "_read_source_snapshot", read)
    monkeypatch.setattr(mgr, "_project_graph", project)

    assert mgr.is_available() is True
    assert captured["journal_mode"] == "wal"
    assert "readonly" in captured["write_error"]
    assert captured["open_during_projection"] is True
    assert captured["open_at_build_end"] is True
    with pytest.raises(sqlite3.ProgrammingError):
        captured["conn"].execute("SELECT 1")


def test_all_loaded_fields_come_from_one_captured_snapshot(
    learned_sqlite: Path, monkeypatch
):
    """A second connection cannot change what the first snapshot loaded.

    The write commits while the build's read transaction is still open, so the
    open reader must keep seeing the pre-change rows and the projected graph must
    contain only the captured snapshot.
    """
    mgr = InstanceGraphManager(sqlite_path=learned_sqlite)
    original_read = mgr._read_source_snapshot
    original_project = mgr._project_graph
    captured: dict[str, sqlite3.Connection] = {}
    observations: dict[str, list[tuple[Any, ...]]] = {}

    def edge_rows(conn: sqlite3.Connection) -> list[tuple[Any, ...]]:
        return [
            tuple(row)
            for row in conn.execute(
                "SELECT src_type, src_id, rel, dst_type, dst_id FROM kg_edge "
                "ORDER BY 1, 2, 3, 4, 5"
            ).fetchall()
        ]

    def read(conn: sqlite3.Connection) -> Any:
        captured["conn"] = conn
        return original_read(conn)

    def project(snapshot: Any) -> None:
        reader = captured["conn"]
        observations["captured"] = [tuple(row[:5]) for row in snapshot.edges]
        observations["before"] = edge_rows(reader)
        writer = get_sqlite_writer_connection(learned_sqlite)
        try:
            writer.execute(
                "INSERT INTO kg_edge (src_type, src_id, rel, dst_type, dst_id, "
                "confidence, method) VALUES "
                "('document','DOC-C','ABOUT_FIELD','field','Duri',0.9,'mid_build_writer')"
            )
            writer.commit()
        finally:
            writer.close()
        observations["during"] = edge_rows(reader)
        original_project(snapshot)

    monkeypatch.setattr(mgr, "_read_source_snapshot", read)
    monkeypatch.setattr(mgr, "_project_graph", project)

    assert mgr.is_available() is True
    injected = ("document", "DOC-C", "ABOUT_FIELD", "field", "Duri")
    assert observations["captured"] == observations["before"]
    assert observations["during"] == observations["before"]
    assert injected not in observations["captured"]

    graph_edges = sorted(
        (row["d.doc_id"], row["f.name"])
        for row in mgr.query(
            "MATCH (d:Document)-[:ABOUT_FIELD]->(f:Field) RETURN d.doc_id, f.name"
        )
    )
    expected = sorted(
        (row[1], row[4])
        for row in observations["before"]
        if row[2] == "ABOUT_FIELD" and row[3] == "field"
    )
    assert graph_edges == expected
    assert len(graph_edges) == 2

    # A fresh reader after the commit does see the row: the difference here is
    # snapshot isolation, not a lost write.
    after = get_sqlite_readonly_connection(learned_sqlite)
    assert after is not None
    try:
        assert injected in edge_rows(after)
    finally:
        after.close()


def test_cancellation_ends_the_read_snapshot(learned_sqlite: Path, monkeypatch):
    """Cancellation releases the source instead of leaving a reader holding it."""
    mgr = InstanceGraphManager(sqlite_path=learned_sqlite)
    original_read = mgr._read_source_snapshot
    captured: dict[str, sqlite3.Connection] = {}

    def read(conn: sqlite3.Connection) -> Any:
        captured["conn"] = conn
        return original_read(conn)

    def project(snapshot: Any) -> None:
        raise KeyboardInterrupt("cancelled mid-build")

    monkeypatch.setattr(mgr, "_read_source_snapshot", read)
    monkeypatch.setattr(mgr, "_project_graph", project)

    with pytest.raises(KeyboardInterrupt):
        mgr.is_available()

    assert mgr._conn is None
    assert mgr._db is None
    assert mgr._available is False
    with pytest.raises(sqlite3.ProgrammingError):
        captured["conn"].execute("SELECT 1")


def test_missing_source_is_refused_without_creating_anything(tmp_path: Path):
    source = tmp_path / "missing" / "esdc.sqlite"
    mgr = InstanceGraphManager(sqlite_path=source)
    assert mgr.is_available() is False
    assert not source.parent.exists()


def test_nonregular_source_is_refused_without_touching_it(tmp_path: Path):
    directory = tmp_path / "esdc.sqlite"
    directory.mkdir()
    mgr = InstanceGraphManager(sqlite_path=directory)
    assert mgr.is_available() is False
    assert list(directory.iterdir()) == []


def test_wrong_shape_source_is_refused_without_changing_it(tmp_path: Path):
    """A source that is not the current registry is refused, never repaired."""
    source = tmp_path / "esdc.sqlite"
    conn = sqlite3.connect(source)
    conn.execute(
        "CREATE TABLE kg_edge (src_type TEXT, src_id TEXT, rel TEXT, "
        "dst_type TEXT, dst_id TEXT)"
    )
    conn.execute("INSERT INTO kg_edge VALUES ('pod','PL-1','ABOUT_POD','pod','PL-2')")
    conn.commit()
    conn.close()

    before = source.read_bytes()
    before_mtime = source.stat().st_mtime_ns
    mgr = InstanceGraphManager(sqlite_path=source)
    assert mgr.is_available() is False
    assert source.read_bytes() == before
    assert source.stat().st_mtime_ns == before_mtime
    assert [entry.name for entry in tmp_path.iterdir()] == ["esdc.sqlite"]


def test_constructor_never_resolves_a_default_duck_path(tmp_path: Path, monkeypatch):
    """Construction must not run the legacy DuckDB default migration.

    `Config.get_db_file()` is not a pure getter: with no ESDC_DB_FILE it can rename
    a legacy default database and rewrite config.yaml. This SQLite-only reader has
    no Duck input, so a construction that refuses a missing source must leave the
    unrelated default and config paths untouched. Every path here is an owned tmp
    file; nothing under the developer's real ~/.esdc is read or written, and the
    other autouse isolation fixtures are left in place.
    """
    monkeypatch.setattr(Config, "get_config_dir", classmethod(lambda cls: tmp_path))
    monkeypatch.setattr(Config, "get_db_file", _REAL_GET_DB_FILE)
    monkeypatch.setattr(Config, "_config_cache", None)
    monkeypatch.delenv("ESDC_DB_FILE", raising=False)
    monkeypatch.delenv("ESDC_CONFIG_DIR", raising=False)

    legacy = tmp_path / Config.LEGACY_DB_FILENAME
    legacy.write_text("sentinel database, must not be renamed", encoding="utf-8")
    config_file = tmp_path / "config.yaml"
    config_file.write_text(f"database_path: {legacy}\n", encoding="utf-8")
    before_config = config_file.read_text(encoding="utf-8")

    mgr = InstanceGraphManager(sqlite_path=tmp_path / "missing" / "esdc.sqlite")
    assert mgr.is_available() is False

    assert legacy.read_text(encoding="utf-8") == "sentinel database, must not be renamed"
    assert not (tmp_path / Config.DB_FILENAME).exists()
    assert config_file.read_text(encoding="utf-8") == before_config
    # Iteration order is not filesystem-guaranteed, so compare names as a set.
    assert {entry.name for entry in tmp_path.iterdir()} == {"config.yaml", legacy.name}


def test_valid_graph_never_calls_config_get_db_file(learned_sqlite: Path, monkeypatch):
    """A valid SQLite source is read without resolving any Duck default path."""

    def forbidden(cls) -> Path:
        raise AssertionError("the graph must not resolve a default DuckDB path")

    monkeypatch.setattr(Config, "get_db_file", classmethod(forbidden))

    mgr = InstanceGraphManager(sqlite_path=learned_sqlite)
    assert mgr.is_available() is True
    pods = {row["p.pod_id"] for row in mgr.query("MATCH (p:POD) RETURN p.pod_id")}
    assert pods == {POD_DURI, POD_DURI_REV1, POD_KAMPUNG}


def test_generic_field_and_working_area_traversal(learned_sqlite: Path):
    """Generic linked entities traverse without any Duck resource mirror."""
    mgr = InstanceGraphManager(sqlite_path=learned_sqlite)
    assert mgr.is_available() is True

    in_field = {
        (row["pr.project_id"], row["f.name"])
        for row in mgr.query(
            "MATCH (pr:Project)-[:IN_FIELD]->(f:Field) "
            "RETURN pr.project_id, f.name"
        )
    }
    assert in_field == {(PROJECT_DURI, "Duri"), (PROJECT_KAMPUNG, "Kampung Baru")}

    in_wk = {
        (row["f.name"], row["w.name"])
        for row in mgr.query("MATCH (f:Field)-[:IN_WK]->(w:WorkingArea) RETURN f.name, w.name")
    }
    assert in_wk == {("Duri", "Rokan"), ("Kampung Baru", "Pertamina EP")}

    # Claim subjects resolve to the same generic entity nodes.
    field_hits = mgr.find("Duri", entity_type="field")
    assert field_hits and field_hits[0]["entity_id"] == "Duri"
    wk_hits = mgr.find("Rokan", entity_type="working_area")
    assert wk_hits and wk_hits[0]["entity_id"] == "Rokan"
    documents_by_field = {
        row["d.doc_id"]
        for row in mgr.query(
            "MATCH (d:Document)-[:ABOUT_FIELD]->(f:Field) "
            "WHERE f.name = 'Duri' RETURN d.doc_id ORDER BY d.doc_id"
        )
    }
    assert documents_by_field == {"DOC-A", "DOC-B"}


def test_master_owns_names_including_inactive_and_master_only_projects(
    learned_sqlite: Path, sqlite_conn
):
    """`m_project` names every project it owns; raw rows only fill projects outside it."""
    sqlite_conn.execute("BEGIN IMMEDIATE")
    sqlite_conn.execute(
        "INSERT INTO m_project (project_id, project_name, is_active, created_by, updated_by) "
        "VALUES (?, ?, 0, 'FA:test-knowledge', 'FA:test-knowledge')",
        ("P-INACTIVE-01", "Sengkang Inactive"),
    )
    sqlite_conn.execute(
        "INSERT INTO api_project_resources (project_id, report_year, project_name, "
        "field_name, wk_name, uncert_level, project_remarks) "
        "VALUES (?, ?, ?, ?, ?, '1P', ?)",
        (
            "P-OUTSIDE-MASTER-01",
            2024,
            "Generic Outside Master",
            "Barru",
            "Pertamina EP",
            "No master row",
        ),
    )
    _finish_source_write(sqlite_conn)

    KnowledgeStore(sqlite_conn).upsert_edges(
        [
            Edge(
                "document",
                "DOC-C",
                "ABOUT_PROJECT",
                "project",
                project_id,
                confidence=1.0,
                method="test",
            )
            for project_id in ("P-INACTIVE-01", "P-OUTSIDE-MASTER-01")
        ]
    )
    sqlite_conn.commit()

    mgr = InstanceGraphManager(sqlite_path=learned_sqlite)
    assert mgr.is_available() is True
    names = {
        row["pr.project_id"]: row["pr.project_name"]
        for row in mgr.query(
            "MATCH (pr:Project) RETURN pr.project_id, pr.project_name "
            "ORDER BY pr.project_id"
        )
    }
    assert names[PROJECT_DURI] == PROJECT_DURI_NAME
    assert names["P-INACTIVE-01"] == "Sengkang Inactive"
    assert names["P-OUTSIDE-MASTER-01"] == "Generic Outside Master"


def test_stale_duck_mirror_name_is_ignored(
    learned_sqlite: Path, duck_conn, tmp_path: Path
):
    """A stale Duck mirror can never override the SQLite master name."""
    duck_conn.execute(
        "UPDATE project_resources SET project_name = 'Stale Mirror Name' "
        "WHERE project_id = ?",
        [PROJECT_DURI],
    )
    duck_conn.close()

    mgr = InstanceGraphManager(
        sqlite_path=learned_sqlite, duckdb_path=tmp_path / "esdc.duckdb"
    )
    assert mgr.is_available() is True
    names = {
        row["pr.project_id"]: row["pr.project_name"]
        for row in mgr.query(
            "MATCH (pr:Project) RETURN pr.project_id, pr.project_name"
        )
    }
    assert names[PROJECT_DURI] == PROJECT_DURI_NAME
    assert not mgr.find("Stale Mirror Name", entity_type="project"), (
        "the Duck mirror name must not be indexed"
    )


def test_graph_builds_with_no_duckdb_input_at_all(learned_sqlite: Path):
    """Duck is no longer a name source, so a missing mirror cannot degrade names."""
    mgr = InstanceGraphManager(
        sqlite_path=learned_sqlite,
        duckdb_path=Path("/nonexistent/path/esdc.duckdb"),
    )
    assert mgr.is_available() is True
    names = {
        row["pr.project_id"]: row["pr.project_name"]
        for row in mgr.query(
            "MATCH (pr:Project) RETURN pr.project_id, pr.project_name "
            "ORDER BY pr.project_id"
        )
    }
    assert names == {PROJECT_DURI: PROJECT_DURI_NAME, PROJECT_KAMPUNG: PROJECT_KAMPUNG_NAME}


def test_unreadable_duck_path_is_never_opened(learned_sqlite: Path, tmp_path: Path):
    """An existing-but-unreadable Duck path stays an unused caller argument."""
    unreadable = tmp_path / "esdc.duckdb"
    unreadable.write_bytes(b"not a real duckdb file")
    mgr = InstanceGraphManager(
        sqlite_path=learned_sqlite,
        duckdb_path=unreadable,
    )
    assert mgr.is_available() is True
    hits = mgr.find(PROJECT_KAMPUNG_NAME, entity_type="project")
    assert hits and hits[0]["entity_id"] == PROJECT_KAMPUNG
    assert hits[0]["name"] == PROJECT_KAMPUNG_NAME


def test_unissued_null_pod_id_is_not_invented_as_a_node(
    learned_sqlite: Path, sqlite_conn
):
    """The canonical unissued carve-out master has no graph node.

    Row 856 is the schema's only permitted NULL pod_id master. Building the graph
    must skip it rather than materialise a node named 'None'.
    """
    sqlite_conn.execute("BEGIN IMMEDIATE")
    sqlite_conn.execute(
        "INSERT INTO m_pod (pod_id_seq, pod_uuid, approval_date, institution_id, "
        "pod_type_id, rev_num, pod_order_no, pod_id, pod_name, pod_letter_num, "
        "is_latest, created_by, updated_by) VALUES (?, ?, ?, ?, ?, ?, NULL, NULL, ?, ?, 1, ?, ?)",
        (
            856,
            "00000000-0000-0000-0000-000000000856",
            "1997-06-01",
            4,
            1,
            0,
            "Pengembangan Lapangan Semoga",
            "SRT-0001/SKKMA0000/1997/S1",
            "FA:test-knowledge",
            "FA:test-knowledge",
        ),
    )
    _finish_source_write(sqlite_conn)

    mgr = InstanceGraphManager(sqlite_path=learned_sqlite)
    assert mgr.is_available() is True
    pods = {row["p.pod_id"] for row in mgr.query("MATCH (p:POD) RETURN p.pod_id")}
    assert pods == {POD_DURI, POD_DURI_REV1, POD_KAMPUNG}
    assert "None" not in pods


def test_schema_contract_declares_current_topology():
    """The executable YAML contract must match the current LadybugDB graph."""
    schema = _load_instance_graph_schema()

    assert set(schema.nodes) == {
        "Document",
        "POD",
        "Project",
        "Field",
        "WorkingArea",
    }
    expected_nodes = {
        "Document": ("document", "doc_id", "subject"),
        "POD": ("pod", "pod_id", "pod_name"),
        "Project": ("project", "project_id", "project_name"),
        "Field": ("field", "name", "name"),
        "WorkingArea": ("working_area", "name", "name"),
    }
    for label, (corpus_type, key, name) in expected_nodes.items():
        node = schema.nodes[label]
        assert node.corpus_type == corpus_type
        assert node.key == key
        assert node.name == name

    assert set(schema.relationships) == {
        "ABOUT_POD",
        "ABOUT_PROJECT",
        "ABOUT_FIELD",
        "ABOUT_WK",
        "HAS_PROJECT",
        "REVISES",
        "SUPERSEDES",
        "IN_FIELD",
        "IN_WK",
    }
    endpoints = {
        "ABOUT_POD": ("Document", "POD"),
        "ABOUT_PROJECT": ("Document", "Project"),
        "ABOUT_FIELD": ("Document", "Field"),
        "ABOUT_WK": ("Document", "WorkingArea"),
        "HAS_PROJECT": ("POD", "Project"),
        "REVISES": ("POD", "POD"),
        "SUPERSEDES": ("POD", "POD"),
        "IN_FIELD": ("Project", "Field"),
        "IN_WK": ("Field", "WorkingArea"),
    }
    temporal = {"valid_from", "valid_to", "properties_json"}
    evidence = {"confidence", "method"}
    for rel, (from_label, to_label) in endpoints.items():
        definition = schema.relationships[rel]
        assert definition.from_label == from_label
        assert definition.to_label == to_label
        assert temporal <= set(definition.properties)
        if rel.startswith("ABOUT_"):
            assert evidence <= set(definition.properties)


def _base_schema_dict() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "nodes": {
            "Document": {
                "corpus_type": "document",
                "key": "doc_id",
                "name": "subject",
                "properties": {
                    "doc_id": "STRING",
                    "file_name": "STRING",
                    "doc_type": "STRING",
                    "doc_date": "STRING",
                    "subject": "STRING",
                },
                "fts": {"index": "document_fts", "properties": ["subject"]},
            },
            "POD": {
                "corpus_type": "pod",
                "key": "pod_id",
                "name": "pod_name",
                "properties": {
                    "pod_id": "STRING",
                    "pod_name": "STRING",
                    "rev_num": "INT64",
                },
            },
        },
        "relationships": {
            "ABOUT_POD": {
                "from": "Document",
                "to": "POD",
                "properties": {
                    "confidence": "DOUBLE",
                    "method": "STRING",
                    "valid_from": "STRING",
                    "valid_to": "STRING",
                    "properties_json": "STRING",
                },
            },
        },
    }


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda d: d.pop("schema_version"), "schema_version"),
        (lambda d: d.update(schema_version=2), "schema_version"),
        (lambda d: d.update(nodes="nodes"), "nodes"),
        (lambda d: d.update(relationships=[]), "relationships"),
        (
            lambda d: d["nodes"].update(
                {"Bad-Label": _base_schema_dict()["nodes"]["POD"]}
            ),
            "nodes.Bad-Label",
        ),
        (
            lambda d: d["relationships"].update(
                {"ABOUT POD": _base_schema_dict()["relationships"]["ABOUT_POD"]}
            ),
            "relationships.ABOUT POD",
        ),
        (lambda d: d["nodes"]["Document"].update(key="doc id"), "key"),
        (
            lambda d: d["nodes"]["Document"]["properties"].update(
                {"bad prop": "STRING"}
            ),
            "properties.bad prop",
        ),
        (
            lambda d: d["nodes"]["POD"].update(corpus_type="document"),
            "corpus_type",
        ),
        (
            lambda d: d["nodes"]["POD"].update(key="missing_key"),
            "nodes.POD.key",
        ),
        (
            lambda d: d["nodes"]["POD"].update(name="missing_name"),
            "nodes.POD.name",
        ),
        (
            lambda d: d["nodes"]["Document"]["properties"].update(subject="FLOAT"),
            "properties.subject",
        ),
        (
            lambda d: d["relationships"]["ABOUT_POD"].pop("from"),
            "relationships.ABOUT_POD.from",
        ),
        (
            lambda d: d["relationships"]["ABOUT_POD"].update(to="Ghost"),
            "ABOUT_POD.to",
        ),
        (
            lambda d: d["nodes"]["Document"]["fts"]["properties"].append("ghost"),
            "fts.properties.ghost",
        ),
        (
            lambda d: d["nodes"]["Document"].update(fts={"properties": ["subject"]}),
            "fts.index",
        ),
        (
            lambda d: d["nodes"]["Document"]["fts"].update(index="bad-index"),
            "fts.index",
        ),
        (
            lambda d: d["relationships"]["ABOUT_POD"]["properties"].pop("valid_from"),
            "properties.valid_from",
        ),
        (
            lambda d: d["relationships"]["ABOUT_POD"]["properties"].pop("confidence"),
            "properties.confidence",
        ),
        (
            lambda d: d["relationships"]["ABOUT_POD"]["properties"].pop("method"),
            "properties.method",
        ),
    ],
)
def test_schema_validation_rejects_invalid_yaml(tmp_path: Path, mutate, message: str):
    """Invalid contract files fail with path-aware ValueError messages."""
    data = _base_schema_dict()
    mutate(data)
    path = tmp_path / "schema.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")

    with pytest.raises(ValueError) as exc:
        _load_instance_graph_schema(path)

    assert message in str(exc.value)
    assert "schema.yaml" in str(exc.value)


def test_node_ddl_from_schema_pins_primary_key_and_types(tmp_path: Path):
    """Schema node definitions generate exact node tables with key as PK."""
    path = tmp_path / "schema.yaml"
    path.write_text(
        yaml.safe_dump(_base_schema_dict(), sort_keys=False), encoding="utf-8"
    )
    schema = _load_instance_graph_schema(path)

    doc_ddl = _node_ddl(schema.nodes["Document"])
    assert doc_ddl == (
        "CREATE NODE TABLE Document (doc_id STRING PRIMARY KEY, "
        "file_name STRING, doc_type STRING, doc_date STRING, subject STRING)"
    )
    pod_ddl = _node_ddl(schema.nodes["POD"])
    assert "pod_id STRING PRIMARY KEY" in pod_ddl
    assert "rev_num INT64" in pod_ddl
    for label, node in schema.nodes.items():
        stmt = _node_ddl(node)
        assert stmt.startswith(f"CREATE NODE TABLE {label} ")
        assert re.fullmatch(r"CREATE NODE TABLE [A-Za-z_][A-Za-z0-9_]* \([^)]+\)", stmt)


def test_rel_ddl_from_schema_pins_temporal_and_evidence_props(tmp_path: Path):
    """Schema relationship definitions generate exact rel tables."""
    path = tmp_path / "schema.yaml"
    path.write_text(
        yaml.safe_dump(_base_schema_dict(), sort_keys=False), encoding="utf-8"
    )
    schema = _load_instance_graph_schema(path)

    stmt = _rel_ddl(schema.relationships["ABOUT_POD"])
    assert stmt == (
        "CREATE REL TABLE ABOUT_POD (FROM Document TO POD, "
        "confidence DOUBLE, method STRING, valid_from STRING, "
        "valid_to STRING, properties_json STRING)"
    )
    assert re.fullmatch(
        r"CREATE REL TABLE [A-Za-z_][A-Za-z0-9_]* "
        r"\(FROM [A-Za-z_][A-Za-z0-9_]* TO [A-Za-z_][A-Za-z0-9_]*"
        r", [^)]+\)",
        stmt,
    )


def test_runtime_topology_constants_derive_from_default_schema():
    """Every hardcoded topology constant is a projection of the YAML schema."""
    from esdc.chat.domain_knowledge.instance_graph import (
        _FTS_INDEXES,
        _LABEL_TO_TYPE,
        _NODE_DDL,
        _NODE_KEY_COL,
        _NODE_NAME_COL,
        _REL_DDL,
        _REL_MAP,
        _TYPE_TO_LABEL,
    )

    schema = _load_instance_graph_schema()
    assert set(_NODE_DDL) == {_node_ddl(node) for node in schema.nodes.values()}
    assert set(_REL_DDL) == {_rel_ddl(rel) for rel in schema.relationships.values()}
    assert {
        label: node.corpus_type for label, node in schema.nodes.items()
    } == _LABEL_TO_TYPE
    assert {
        node.corpus_type: label for label, node in schema.nodes.items()
    } == _TYPE_TO_LABEL
    assert {label: node.key for label, node in schema.nodes.items()} == _NODE_KEY_COL
    assert {label: node.name for label, node in schema.nodes.items()} == _NODE_NAME_COL
    assert {(idx[0], idx[1], tuple(idx[2])) for idx in _FTS_INDEXES} == {
        (label, node.fts_index, tuple(node.fts_properties))
        for label, node in schema.nodes.items()
        if node.fts_index is not None
    }
    expected_rel_map = {
        rel_name: (
            rel_def.from_label,
            schema.nodes[rel_def.from_label].key,
            rel_def.to_label,
            schema.nodes[rel_def.to_label].key,
            {"confidence", "method"} <= set(rel_def.properties),
        )
        for rel_name, rel_def in schema.relationships.items()
    }
    assert expected_rel_map == _REL_MAP


def test_find_resolves_pod_by_name(learned_sqlite: Path):
    mgr = InstanceGraphManager(sqlite_path=learned_sqlite)
    hits = mgr.find("POD Duri")
    assert hits
    top = hits[0]
    assert top["entity_type"] == "pod"
    assert top["entity_id"] in (POD_DURI, POD_DURI_REV1)


def test_find_trailing_backslash_does_not_raise(learned_sqlite: Path):
    r"""A trailing backslash in the query text must not swallow the closing.

    quote of the QUERY_FTS_INDEX string literal. The hand-rolled escaping
    previously used in find() (only `'` -> `\'`) left a lone backslash at
    the end of the Cypher string literal, which escapes the closing quote
    instead of being escaped itself -- every per-table query then raised
    and find() silently degraded to an empty result (surfacing as a
    misleading not_found in explore_entity). _cypher_escape escapes
    backslashes first, so this must return normally instead of raising.
    """
    mgr = InstanceGraphManager(sqlite_path=learned_sqlite)
    hits = mgr.find("Duri\\")
    assert isinstance(hits, list)


def test_find_ranks_entities_over_documents(learned_sqlite: Path):
    """Type-priority ranking must hold even for a non-top same-type hit.

    Query choice ("Duri") is deliberate, not arbitrary: the fixture seeds
    TWO PODs matching it ("POD I Duri", "POD I Duri Revisi 1") plus one
    Field "Duri" (empirically confirmed via mgr.find("Duri") -- 2 pod hits,
    1 field hit, 2 document hits). That gives a *non-top* entity hit
    (the weaker POD) to check against the lower-priority Field hit.

    Under the OLD per-table max-normalized scoring, every table's top hit
    collapsed to exactly 1.0: POD's top ("POD I Duri") -> 1.0, Field's only
    hit ("Duri") -> 1.0, but the second/weaker POD hit ("POD I Duri Revisi
    1") normalized to < 1.0. Sorting by normalized score then put the
    Field hit (1.0) BETWEEN the two POD hits (1.0, then <1.0) -- i.e. NOT
    both PODs before Field. The fixed type-priority-then-raw-score ranking
    keeps all pod-priority (0) hits ahead of field-priority (2) hits
    regardless of BM25 magnitude, so both PODs must precede Field.
    """
    mgr = InstanceGraphManager(sqlite_path=learned_sqlite)
    hits = mgr.find("Duri", top_k=10)
    types = [h["entity_type"] for h in hits]
    assert types.count("pod") >= 2
    assert "field" in types
    assert "document" in types
    pod_positions = [i for i, t in enumerate(types) if t == "pod"]
    field_position = types.index("field")
    assert max(pod_positions) < field_position, (
        f"expected both POD hits before the Field hit, got order={types}"
    )
    assert field_position < types.index("document")


def test_find_orders_same_type_by_raw_score(learned_sqlite: Path):
    """Within one entity type, ties are broken by RAW (uncollapsed) score.

    Raw BM25 is comparable within the same FTS index, so this should not
    fall back to insertion order -- and it should not be per-table
    max-normalized either. The OLD implementation forced every table's top
    hit to exactly 1.0 before sorting; asserting the top score here is NOT
    1.0 (plus that the two scores are genuinely distinct raw magnitudes,
    not just a normalized order) fails under that old behavior and passes
    under the fixed raw-score implementation.
    """
    mgr = InstanceGraphManager(sqlite_path=learned_sqlite)
    hits = mgr.find("POD Duri", top_k=10)
    pod_hits = [h for h in hits if h["entity_type"] == "pod"]
    assert len(pod_hits) >= 2
    scores = [h["score"] for h in pod_hits]
    assert scores == sorted(scores, reverse=True)
    assert scores[0] != 1.0
    assert scores[0] > scores[1]


def test_neighbors_groups_by_relation(learned_sqlite: Path, sqlite_conn):
    """Neighbours are grouped by relationship label, in both directions.

    Retired assertion: the fixture used to carry a document->POD link on the base
    POD. Canonical deterministic matching resolves DOC-A's exact letter number to
    the current identity (POD_DURI_REV1), so POD_DURI has no document link. The
    project group needs one explicit edge here because the fixture's documents name
    no project.
    """
    store = KnowledgeStore(sqlite_conn)
    store.upsert_edges(
        [
            Edge(
                "document",
                "DOC-B",
                "ABOUT_PROJECT",
                "project",
                PROJECT_DURI,
                confidence=1.0,
                method="test",
            ),
        ]
    )

    mgr = InstanceGraphManager(sqlite_path=learned_sqlite)
    n = mgr.neighbors("pod", POD_DURI)
    assert any(item["entity_id"] == PROJECT_DURI for item in n.get("HAS_PROJECT", []))
    assert any(
        item["entity_id"] == POD_DURI_REV1 and item["direction"] == "inbound"
        for item in n.get("REVISES", [])
    )
    assert n.get("ABOUT_POD", []) == []

    # The current identity owns the canonical exact-letter document link.
    current = mgr.neighbors("pod", POD_DURI_REV1)
    assert any(
        item["entity_type"] == "document" and item["entity_id"] == "DOC-A"
        for item in current.get("ABOUT_POD", [])
    )

    # Grouping is per relationship label: the project side keeps its own group and
    # does not mix the document link into the POD links.
    project_neighbours = mgr.neighbors("project", PROJECT_DURI)
    assert any(
        item["entity_type"] == "pod" and item["entity_id"] == POD_DURI
        for item in project_neighbours.get("HAS_PROJECT", [])
    )
    assert any(
        item["entity_type"] == "document" and item["entity_id"] == "DOC-B"
        for item in project_neighbours.get("ABOUT_PROJECT", [])
    )
    assert all(
        item["entity_type"] == "document"
        for item in project_neighbours.get("ABOUT_PROJECT", [])
    )


def test_all_declared_relationships_traversable(learned_sqlite: Path, sqlite_conn):
    """Every schema relationship must be reachable after building.

    Supersession is pointer-derived from `m_pod.superseded_by`, so REVISES and
    SUPERSEDES already exist for the canonical fixture; only ABOUT_PROJECT has to
    be added here because the fixture's documents carry no project name.
    """
    store = KnowledgeStore(sqlite_conn)
    store.upsert_edges(
        [
            Edge(
                "document",
                "DOC-A",
                "ABOUT_PROJECT",
                "project",
                PROJECT_DURI,
                confidence=1.0,
                method="test",
            )
        ]
    )
    mgr = InstanceGraphManager(sqlite_path=learned_sqlite)
    for rel in _load_instance_graph_schema().relationships:
        rows = mgr.query(f"MATCH ()-[r:{rel}]->() RETURN r LIMIT 1")
        assert rows, f"relationship {rel} missing from traversable graph"

    n = mgr.neighbors("pod", POD_DURI)
    revises = n.get("REVISES", [])
    supersedes = n.get("SUPERSEDES", [])
    assert any(
        item["entity_id"] == POD_DURI_REV1 and item["direction"] == "inbound"
        for item in revises
    )
    assert any(
        item["entity_id"] == POD_DURI_REV1 and item["direction"] == "inbound"
        for item in supersedes
    )


def test_cypher_query_passthrough(learned_sqlite: Path):
    """Public Cypher traversal resolves the canonical document link.

    Retired assertion: this previously required `len(rows) >= 2`. Under the
    accepted model only the letter-number match links DOC-A to the current
    identity; DOC-B's `suggested_pod_ids` is review-only and never produces a
    graph edge.
    """
    mgr = InstanceGraphManager(sqlite_path=learned_sqlite)
    rows = mgr.query(
        "MATCH (d:Document)-[:ABOUT_POD]->(p:POD) "
        "RETURN d.doc_id, p.pod_id ORDER BY d.doc_id"
    )
    assert rows
    assert all(row["p.pod_id"] == POD_DURI_REV1 for row in rows)
    assert "DOC-A" in [row["d.doc_id"] for row in rows]


def test_find_resolves_project_with_master_name(learned_sqlite: Path):
    """Project FTS lookup must exist and surface the master-owned name.

    Without a Project FTS index, find() never returns project hits at all
    -- this is the Issue #1 bug (entity_type='project' always not_found).

    Retired assertion: this previously required a DuckDB mirror to name the
    project. `m_project` owns project names, so the graph resolves entirely from
    its own SQLite snapshot and no Duck connection is opened.
    """
    mgr = InstanceGraphManager(sqlite_path=learned_sqlite)
    hits = mgr.find(PROJECT_DURI_NAME)
    assert hits
    projects = [h for h in hits if h["entity_type"] == "project"]
    assert projects, (
        f"expected a project hit, got types={[h['entity_type'] for h in hits]}"
    )
    top = projects[0]
    assert top["entity_id"] == PROJECT_DURI
    assert top["name"] == PROJECT_DURI_NAME


def test_project_neighbor_reports_master_name(learned_sqlite: Path):
    """A Project reached as a POD neighbor must show its master-owned name.

    Project nodes used to be created with project_name = project_id because the
    graph was built only from kg_edge, which has no names. Retired assertion: the
    name previously came from DuckDB; it now comes from `m_project` in the same
    snapshot that produced the nodes.
    """
    mgr = InstanceGraphManager(sqlite_path=learned_sqlite)
    n = mgr.neighbors("pod", POD_DURI)
    has_project = n.get("HAS_PROJECT", [])
    assert any(
        item["entity_id"] == PROJECT_DURI and item["name"] == PROJECT_DURI_NAME
        for item in has_project
    ), f"expected master-owned project name among neighbors, got {has_project}"


def test_find_entity_type_filter_avoids_cross_type_truncation(learned_sqlite: Path):
    """entity_type filter must apply BEFORE top_k truncation, not after.

    "Duri Steamflood" matches both POD entities ("POD I Duri", "POD I Duri
    Revisi 1" via the "Duri" token) and the Project "Duri Steamflood". POD
    outranks Project under _TYPE_PRIORITY, so with top_k=1 a plain find()
    surfaces a POD, not the project -- demonstrating that a caller asking
    specifically for entity_type="project" would otherwise lose the match
    to higher-priority cross-type hits before ever getting to filter.
    """
    mgr = InstanceGraphManager(sqlite_path=learned_sqlite)
    plain = mgr.find(PROJECT_DURI_NAME, top_k=1)
    assert plain
    assert plain[0]["entity_type"] == "pod", (
        "test setup assumption broken: expected POD to outrank Project in "
        f"the unfiltered top_k=1 result, got {plain}"
    )

    filtered = mgr.find(PROJECT_DURI_NAME, top_k=1, entity_type="project")
    assert filtered
    assert all(h["entity_type"] == "project" for h in filtered)
    assert filtered[0]["entity_id"] == PROJECT_DURI
    assert filtered[0]["name"] == PROJECT_DURI_NAME


def test_query_rejects_mutation(learned_sqlite: Path):
    """Public Ladybug queries must reject mutation.

    The instance graph is a disposable read-only index over the relational
    source of truth. query() is exposed to chat tools, so a write Cypher
    statement must raise instead of mutating the graph.
    """
    mgr = InstanceGraphManager(sqlite_path=learned_sqlite)
    with pytest.raises(ValueError, match="read-only"):
        mgr.query("CREATE (:POD {pod_id: 'forbidden'})")


@pytest.mark.parametrize(
    "bad",
    [
        "CREATE (:POD {pod_id: 'x'})",
        "MERGE (a:POD {pod_id: 'x'})",
        "MATCH (a:POD) SET a.rev_num = 5",
        "MATCH (a:POD) DELETE a",
        "CALL some_proc() RETURN 1",
        "LOAD CSV FROM 'x' AS row RETURN row",
        "DROP TABLE POD",
        "MATCH (a:POD) RETURN a.pod_id; MATCH (b:POD) RETURN b.pod_id",
        "MATCH (a:POD) RETURN a.pod_id UNION MATCH (b:POD) RETURN b.pod_id",
        "MATCH (a:POD) RETURN a.pod_id YIELD something",
    ],
)
def test_query_rejects_non_read(learned_sqlite: Path, bad: str):
    """Write/DDL/unsupported/multi-statement queries raise before executing."""
    mgr = InstanceGraphManager(sqlite_path=learned_sqlite)
    with pytest.raises(ValueError, match="read-only"):
        mgr.query(bad)


def test_query_read_only_validation_masks_literals(learned_sqlite: Path):
    """Keywords inside string literals/comments must not trip the guard."""
    mgr = InstanceGraphManager(sqlite_path=learned_sqlite)
    rows = mgr.query(
        "MATCH (d:Document)-[:ABOUT_POD]->(p:POD) "
        "WHERE d.subject = 'CREATE UNION SET' "
        "RETURN d.doc_id, p.pod_id ORDER BY d.doc_id LIMIT 5"
    )
    assert isinstance(rows, list)
    assert not rows
    rows = mgr.query(
        "MATCH (d:Document)-[:ABOUT_POD]->(p:POD) "
        "// CREATE MERGE SET\n"
        "RETURN d.doc_id, p.pod_id ORDER BY d.doc_id LIMIT 5"
    )
    assert rows


def test_query_allows_semicolon_in_literal(learned_sqlite: Path):
    """A semicolon inside a quoted literal is not a statement boundary.

    The semicolon guard must run on the masked query, so ';' inside a
    string literal is ignored; a single real trailing semicolon is allowed.
    The literal matches nothing in the corpus, so the query may return empty
    -- only the list shape is asserted.
    """
    mgr = InstanceGraphManager(sqlite_path=learned_sqlite)
    rows = mgr.query(
        "MATCH (d:Document) WHERE d.subject = 'a;b' "
        "RETURN d.doc_id ORDER BY d.doc_id LIMIT 5;"
    )
    assert isinstance(rows, list)


def test_query_validates_before_build(learned_sqlite: Path, monkeypatch):
    """A write query must raise before the graph is built or checked."""
    mgr = InstanceGraphManager(sqlite_path=learned_sqlite)

    def _should_not_build():
        raise AssertionError("_ensure_built must not run for an invalid query")

    monkeypatch.setattr(mgr, "_ensure_built", _should_not_build)
    with pytest.raises(ValueError, match="read-only"):
        mgr.query("CREATE (:POD {pod_id: 'forbidden'})")


@pytest.mark.parametrize(
    ("edge", "fragment"),
    [
        (
            Edge(
                "document",
                "DOC-A",
                "MYSTERY_REL",
                "pod",
                POD_DURI,
                method="test",
            ),
            "MYSTERY_REL",
        ),
        (
            Edge(
                "field",
                "Duri",
                "ABOUT_POD",
                "pod",
                POD_DURI,
                method="test",
            ),
            "ABOUT_POD",
        ),
        (
            Edge(
                "document",
                "DOC-A",
                "ABOUT_POD",
                "field",
                "Duri",
                method="test",
            ),
            "ABOUT_POD",
        ),
        (
            Edge(
                "document",
                "DOC-MISSING",
                "ABOUT_POD",
                "pod",
                POD_DURI,
                method="test",
            ),
            "DOC-MISSING",
        ),
        (
            Edge(
                "document",
                "DOC-A",
                "ABOUT_POD",
                "pod",
                "PL-MISSING",
                method="test",
            ),
            "PL-MISSING",
        ),
    ],
)
def test_schema_drift_fails_build_without_touching_sqlite(
    learned_sqlite: Path,
    sqlite_conn,
    caplog,
    tmp_path: Path,
    edge: Edge,
    fragment: str,
):
    """Drifted kg_edge rows make the projection unavailable, never mutating."""
    store = KnowledgeStore(sqlite_conn)
    store.upsert_edges([edge])
    sqlite_conn.commit()
    before = sqlite_conn.execute(
        "SELECT src_type, src_id, rel, dst_type, dst_id, confidence, method, "
        "valid_from, valid_to, properties_json FROM kg_edge ORDER BY 1,2,3,4,5"
    ).fetchall()
    before_mtime = learned_sqlite.stat().st_mtime_ns
    before_files = set(tmp_path.iterdir())

    mgr = InstanceGraphManager(sqlite_path=learned_sqlite)
    # esdc.chat.app may have globally disabled logging on import (chat level
    # "0") and stopped esdc.chat from propagating to the root, where pytest's
    # caplog handler lives. Attach caplog directly to the target logger for
    # this assertion, and always restore global disable plus logger state.
    target_logger = logging.getLogger("esdc.chat.domain_knowledge.instance_graph")
    prior_disable = logging.root.manager.disable
    prior_logger_level = target_logger.level
    logging.disable(logging.NOTSET)
    target_logger.setLevel(logging.ERROR)
    target_logger.addHandler(caplog.handler)
    try:
        assert mgr.is_available() is False
    finally:
        target_logger.removeHandler(caplog.handler)
        target_logger.setLevel(prior_logger_level)
        logging.disable(prior_disable)

    assert mgr._conn is None
    assert mgr._db is None
    assert any(
        "build_failed" in record.message and fragment in record.message
        for record in caplog.records
    ), f"expected actionable drift log for {fragment}, got {caplog.records}"
    after = sqlite_conn.execute(
        "SELECT src_type, src_id, rel, dst_type, dst_id, confidence, method, "
        "valid_from, valid_to, properties_json FROM kg_edge ORDER BY 1,2,3,4,5"
    ).fetchall()
    assert after == before, "a refused build must not mutate the source rows"
    # Nothing is written and nothing is created: the source file is only opened.
    assert learned_sqlite.stat().st_mtime_ns == before_mtime, (
        "a refused build must not write to the source file"
    )
    assert set(tmp_path.iterdir()) == before_files, (
        "a refused build must not create side files"
    )


def test_query_read_only_allowlist_succeeds(learned_sqlite: Path):
    """Valid MATCH with WHERE/WITH/RETURN/ORDER BY/SKIP/LIMIT passes."""
    mgr = InstanceGraphManager(sqlite_path=learned_sqlite)
    rows = mgr.query(
        "MATCH (d:Document)-[:ABOUT_POD]->(p:POD) "
        "WHERE d.doc_type = 'letter' "
        "WITH d, p "
        "RETURN d.doc_id, p.pod_id ORDER BY d.doc_id SKIP 0 LIMIT 10"
    )
    assert rows
    optional = mgr.query(
        "OPTIONAL MATCH (d:Document) WHERE d.doc_type = 'letter' "
        "RETURN d.doc_id ORDER BY d.doc_id LIMIT 5"
    )
    assert len(optional) >= 1


def test_temporal_relation_projection(learned_sqlite: Path, sqlite_conn):
    """REVISES/SUPERSEDES must expose valid_from/valid_to/properties_json."""
    store = KnowledgeStore(sqlite_conn)
    store.upsert_edges(
        [
            Edge(
                "pod",
                POD_KAMPUNG,
                "SUPERSEDES",
                "pod",
                POD_DURI,
                confidence=1.0,
                method="registry",
                valid_from="2024-06-01",
                valid_to="2025-01-01",
                properties_json='{"revision_effect": "full_replacement"}',
            )
        ]
    )
    mgr = InstanceGraphManager(sqlite_path=learned_sqlite)
    rows = mgr.query(
        "MATCH (a:POD)-[r:SUPERSEDES]->(b:POD) "
        "RETURN a.pod_id, b.pod_id, r.valid_from, r.valid_to, r.properties_json"
    )
    assert rows
    hit = next(r for r in rows if r["a.pod_id"] == POD_KAMPUNG)
    assert hit["b.pod_id"] == POD_DURI
    assert hit["r.valid_from"] == "2024-06-01"
    assert hit["r.valid_to"] == "2025-01-01"
    assert '"revision_effect": "full_replacement"' in hit["r.properties_json"]

    revises = mgr.query(
        "MATCH (a:POD)-[r:REVISES]->(b:POD) "
        "RETURN a.pod_id, b.pod_id, r.valid_from, r.properties_json"
    )
    # Canonical supersession is pointer-derived from m_pod.superseded_by, so the
    # fixture already carries this edge and valid_from is the successor's approval
    # date. Retired assertion: the old build emitted NULL valid_from here.
    assert any(
        r["a.pod_id"] == POD_DURI_REV1
        and r["b.pod_id"] == POD_DURI
        and r["r.valid_from"] == "2022-01-01"
        and '"revision_effect": "full_replacement"' in r["r.properties_json"]
        and '"source": "m_pod.superseded_by"' in r["r.properties_json"]
        for r in revises
    )


def test_rebuild_reflects_sqlite_changes(learned_sqlite: Path, sqlite_conn):
    """Rebuilding from the same sqlite is identical; edits become visible.

    The canonical fixture already contributes one pointer-derived SUPERSEDES edge
    (POD_DURI_REV1 -> POD_DURI from m_pod.superseded_by), so the counted set is
    the two edges written here plus that canonical one. Retired assertion: the old
    test assumed two hand-made edges only, before supersession was pointer-derived.
    """
    store = KnowledgeStore(sqlite_conn)
    store.upsert_edges(
        [
            Edge(
                "pod",
                POD_KAMPUNG,
                "SUPERSEDES",
                "pod",
                POD_DURI,
                confidence=1.0,
                method="test",
                valid_from="2024-06-01",
                properties_json='{"revision_effect": "full_replacement", "written_by": "test"}',
            ),
            Edge(
                "pod",
                POD_KAMPUNG,
                "SUPERSEDES",
                "pod",
                POD_DURI_REV1,
                confidence=1.0,
                method="test",
                valid_from="2024-07-01",
                properties_json='{"revision_effect": "full_replacement", "written_by": "test"}',
            ),
        ]
    )

    def supersedes_edges(mgr) -> list[dict[str, Any]]:
        return mgr.query(
            "MATCH (a:POD)-[r:SUPERSEDES]->(b:POD) "
            "RETURN a.pod_id, b.pod_id, r.properties_json "
            "ORDER BY a.pod_id, b.pod_id"
        )

    def written(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Only the edges this test wrote; the canonical pointer edge is separate.

        SUPERSEDES carries no method column in the public schema contract, so the
        marker travels in properties_json.
        """
        return [r for r in rows if '"written_by": "test"' in r["r.properties_json"]]

    mgr = InstanceGraphManager(sqlite_path=learned_sqlite)
    state_a = supersedes_edges(mgr)
    assert len(written(state_a)) == 2
    assert any(
        '"source": "m_pod.superseded_by"' in r["r.properties_json"]
        and r["a.pod_id"] == POD_DURI_REV1
        and r["b.pod_id"] == POD_DURI
        for r in state_a
    ), "pointer-derived supersession must survive the projection"
    mgr.close()
    InstanceGraphManager.reset_for_tests()

    rebuilt = InstanceGraphManager(sqlite_path=learned_sqlite)
    assert supersedes_edges(rebuilt) == state_a
    rebuilt.close()

    sqlite_conn.execute(
        f"DELETE FROM kg_edge WHERE rel = 'SUPERSEDES' AND dst_id = '{POD_DURI_REV1}'"
    )
    # The projection reads through its own read-only connection, so the write
    # boundary has to be closed before the rebuild can observe it.
    _finish_source_write(sqlite_conn)
    store.upsert_edges(
        [
            Edge(
                "pod",
                POD_DURI,
                "SUPERSEDES",
                "pod",
                POD_KAMPUNG,
                confidence=1.0,
                method="test",
                valid_from="2025-01-01",
                properties_json='{"revision_effect": "full_replacement", "written_by": "test"}',
            )
        ]
    )

    mgr3 = InstanceGraphManager(sqlite_path=learned_sqlite)
    state_b = supersedes_edges(mgr3)
    assert len(written(state_b)) == 2
    assert not any(r["b.pod_id"] == POD_DURI_REV1 for r in written(state_b)), (
        "old in-memory edge must be absent after rebuild"
    )
    assert any(
        r["a.pod_id"] == POD_DURI and r["b.pod_id"] == POD_KAMPUNG
        for r in written(state_b)
    ), "new sqlite edge must be visible after rebuild"
    assert any(
        '"source": "m_pod.superseded_by"' in r["r.properties_json"]
        and r["a.pod_id"] == POD_DURI_REV1
        and r["b.pod_id"] == POD_DURI
        for r in state_b
    ), "the canonical pointer-derived edge is regenerated on rebuild"
