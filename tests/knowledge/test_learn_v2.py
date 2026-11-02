"""Generation guards for the v2 corpus learning pipeline."""

from __future__ import annotations

import json
import sqlite3
from datetime import date
from typing import cast

import duckdb

from esdc.knowledge.dossier import ensure_dossier_table, gather_pod_context, get_dossier
from esdc.knowledge.freshness import (
    compute_knowledge_basis,
    ensure_knowledge_state_table,
    knowledge_freshness,
    knowledge_freshness_markers,
)
from esdc.knowledge.learn import run_learn
from esdc.knowledge.store import Edge, KnowledgeStore
from esdc.pod_registry.identity import ApprovalKey, issue_identity
from esdc.pod_registry.migrations import initialize_registry_v2
from esdc.pod_registry.store import ensure_project_resources_table


def _database(path, *, docs: int = 1, current: bool = True):
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    initialize_registry_v2(conn)
    # The canonical raw resource shape is the only shape the readers accept.
    ensure_project_resources_table(conn)
    conn.execute(
        "INSERT INTO m_project (project_id, project_name, is_active) "
        "VALUES ('P-1', 'Initial project', 1)"
    )
    conn.execute(
        "INSERT INTO api_project_resources (project_id, report_year, project_name, "
        "field_name, wk_name, uncert_level, project_remarks) "
        "VALUES ('P-1', 2024, 'Initial project', 'Field one', 'WK one', '1P', 'Remarks')"
    )
    conn.execute(
        "CREATE TABLE documents ("
        "doc_id TEXT PRIMARY KEY, file_hash TEXT NOT NULL, doc_type TEXT, "
        "doc_date TEXT, subject TEXT, markdown TEXT, suggested_pod_ids TEXT)"
    )
    for index in range(docs):
        conn.execute(
            "INSERT INTO documents VALUES (?, ?, 'letter', '2024-01-01', "
            "'Subject', 'Source document text', NULL)",
            (f"DOC-{index + 1}", f"hash-{index + 1}"),
        )
    store = KnowledgeStore(conn, allow_stale=True)
    store.ensure_tables()
    ensure_knowledge_state_table(conn)
    basis = compute_knowledge_basis(conn)
    conn.execute(
        "INSERT INTO pod_knowledge_state "
        "(singleton_id, basis_sha256, source_generation) VALUES (1, ?, 0)",
        (basis,),
    )
    conn.execute(
        "UPDATE pod_registry_state SET knowledge_generation=? WHERE singleton_id=1",
        (0 if current else None,),
    )
    conn.commit()
    return conn


def _llm(prompt: str) -> str:
    return json.dumps(
        {
            "entities": [],
            "claims": [],
            "unknown_types": [{"kind": "claim_type", "name": "new", "why": "x"}],
        }
    )


def test_v2_learning_discards_result_when_basis_changes_during_llm(
    tmp_path, monkeypatch
):
    from esdc.corpus import mirror

    conn = _database(tmp_path / "source.sqlite")
    duck_conn = duckdb.connect(":memory:")
    monkeypatch.setattr(mirror, "refresh_all", lambda *_args: None)
    calls = 0

    def changing_llm(prompt: str) -> str:
        nonlocal calls
        calls += 1
        assert not conn.in_transaction
        conn.execute(
            "UPDATE api_project_resources SET project_name='Changed during LLM' "
            "WHERE project_id='P-1'"
        )
        conn.commit()
        return _llm(prompt)

    try:
        before_basis = compute_knowledge_basis(conn)
        report = run_learn(
            sqlite_conn=conn,
            duck_conn=duck_conn,
            llm_caller=changing_llm,
            force=True,
            progress=False,
        )
        assert calls == 1
        assert report.docs_processed == 0
        assert report.docs_failed == 1
        assert conn.execute("SELECT COUNT(*) FROM kg_proposal").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM learn_state").fetchone()[0] == 0
        assert compute_knowledge_basis(conn) != before_basis
        assert not knowledge_freshness(conn).fresh
    finally:
        conn.close()
        duck_conn.close()


def test_v2_limited_learning_preserves_progress_until_full_completion(
    tmp_path, monkeypatch
):
    from esdc.corpus import mirror

    conn = _database(tmp_path / "source.sqlite", docs=2, current=False)
    duck_conn = duckdb.connect(":memory:")
    monkeypatch.setattr(mirror, "refresh_all", lambda *_args: None)
    calls: list[str] = []

    def counting_llm(prompt: str) -> str:
        calls.append(prompt)
        return _llm(prompt)

    try:
        first = run_learn(
            sqlite_conn=conn,
            duck_conn=duck_conn,
            llm_caller=counting_llm,
            limit=1,
            progress=False,
        )
        assert first.docs_processed == 1
        assert first.docs_failed == 0
        assert len(calls) == 1
        assert conn.execute("SELECT COUNT(*) FROM learn_state").fetchone()[0] == 1
        assert not knowledge_freshness(conn).fresh

        second = run_learn(
            sqlite_conn=conn,
            duck_conn=duck_conn,
            llm_caller=counting_llm,
            limit=1,
            progress=False,
        )
        assert second.docs_processed == 1
        assert second.docs_skipped == 1
        assert len(calls) == 2
        assert conn.execute("SELECT COUNT(*) FROM learn_state").fetchone()[0] == 2
        assert not knowledge_freshness(conn).fresh

        complete = run_learn(
            sqlite_conn=conn,
            duck_conn=duck_conn,
            llm_caller=counting_llm,
            progress=False,
        )
        assert complete.docs_processed == 0
        assert complete.docs_skipped == 2
        assert len(calls) == 2
        assert knowledge_freshness(conn).fresh
        assert knowledge_freshness_markers(conn).fresh
    finally:
        conn.close()
        duck_conn.close()


def test_knowledge_basis_ignores_numeric_edits_and_resource_row_order(tmp_path):
    conn = _database(tmp_path / "source.sqlite", docs=0)
    columns = (
        "project_id, report_year, project_name, project_remarks, "
        "field_name, wk_name, uncert_level, rec_oil"
    )
    try:
        conn.execute("DELETE FROM api_project_resources")
        conn.executemany(
            f"INSERT INTO api_project_resources ({columns}) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [
                ("P-2", 2024, "Second", "R2", "Field B", "WK B", "1P", 2.0),
                ("P-1", 2024, "First", "R1", "Field A", "WK A", "1P", 1.0),
            ],
        )
        conn.commit()
        first = compute_knowledge_basis(conn)
        conn.execute("UPDATE api_project_resources SET rec_oil=999 WHERE project_id='P-1'")
        conn.commit()
        assert compute_knowledge_basis(conn) == first
        rows = conn.execute(
            f"SELECT {columns} FROM api_project_resources ORDER BY project_id"
        ).fetchall()
        conn.execute("DELETE FROM api_project_resources")
        conn.executemany(
            f"INSERT INTO api_project_resources ({columns}) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [tuple(row) for row in reversed(rows)],
        )
        conn.commit()
        assert compute_knowledge_basis(conn) == first
    finally:
        conn.close()


def test_v2_dossier_reads_current_project_descriptors_from_sqlite(tmp_path):
    conn = _database(tmp_path / "source.sqlite")
    duck_conn = duckdb.connect(":memory:")
    try:
        conn.execute("BEGIN IMMEDIATE")
        identity = issue_identity(
            conn,
            key=ApprovalKey(date(2024, 1, 1), 1, 1, 1, 0),
            actor="FA:test-v2-dossier",
        )
        conn.execute(
            "INSERT INTO m_pod (pod_id_seq, pod_uuid, approval_date, institution_id, "
            "pod_type_id, rev_num, pod_order_no, pod_id, pod_name, pod_letter_num, "
            "is_latest) VALUES (?, ?, '2024-01-01', 1, 1, 0, ?, ?, 'POD One', "
            "'LETTER-1', 1)",
            (
                identity.pod_id_seq,
                str(identity.pod_uuid),
                identity.pod_order_no,
                identity.pod_id,
            ),
        )
        conn.execute(
            "INSERT INTO pod_project (pod_project_uuid, pod_id_seq, project_id, "
            "is_main_project, is_auto_inferred) VALUES ('map-1', ?, 'P-1', 1, 0)",
            (identity.pod_id_seq,),
        )
        conn.commit()
        store = KnowledgeStore(conn, allow_stale=True)
        store.ensure_tables()
        store.upsert_edges(
            [
                Edge(
                    "pod",
                    identity.pod_id,
                    "HAS_PROJECT",
                    "project",
                    "P-1",
                    method="registry_current",
                )
            ]
        )

        context = gather_pod_context(identity.pod_id, conn, duck_conn, store)

        assert context["pod"]["pod_type"] == "POD I"
        assert context["projects"] == [
            {
                "project_id": "P-1",
                "project_name": "Initial project",
                "report_year": 2024,
                "project_remarks": "Remarks",
            }
        ]
    finally:
        conn.close()
        duck_conn.close()


def test_v2_dossier_read_hides_stale_cached_content(tmp_path):
    from esdc.knowledge.dossier import generate_pod_dossier
    from esdc.knowledge.guideline import load_guideline

    conn = _database(tmp_path / "source.sqlite")
    duck_conn = duckdb.connect(":memory:")
    try:
        store = KnowledgeStore(conn, allow_stale=True)
        store.ensure_tables()
        ensure_dossier_table(duck_conn)
        assert (
            generate_pod_dossier(
                "POD-1",
                conn,
                duck_conn,
                store,
                load_guideline().content_hash,
                lambda _prompt: "stale dossier",
            )
            == "built"
        )
        dossier = get_dossier(duck_conn, "pod", "POD-1", sqlite_conn=conn)
        assert dossier is not None
        assert dossier["dossier_text"] == "stale dossier"
        conn.execute(
            "UPDATE api_project_resources SET project_name='New project name' "
            "WHERE project_id='P-1'"
        )
        conn.commit()
        assert get_dossier(duck_conn, "pod", "POD-1", sqlite_conn=conn) is None
    finally:
        conn.close()
        duck_conn.close()



def test_v2_learning_captures_basis_and_document_from_same_snapshot(
    tmp_path, monkeypatch
):
    from esdc.corpus import mirror

    source_path = tmp_path / "source.sqlite"
    conn = _database(source_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.commit()
    writer = sqlite3.connect(source_path)
    duck_conn = duckdb.connect(":memory:")
    monkeypatch.setattr(mirror, "refresh_all", lambda *_args: None)
    prompts: list[str] = []

    class CursorAfterFetch:
        def __init__(self, cursor):
            self.cursor = cursor

        def fetchall(self):
            rows = self.cursor.fetchall()
            writer.execute(
                "UPDATE documents SET markdown='Source document text' "
                "WHERE doc_id='DOC-1'"
            )
            writer.commit()
            return rows

        def __getattr__(self, name):
            return getattr(self.cursor, name)

    class AbaConnection:
        def __init__(self, connection):
            self.connection = connection

        def execute(self, sql, parameters=()):
            if sql.startswith(
                "SELECT doc_id, file_hash, doc_type, doc_date, subject, markdown "
                "FROM documents ORDER BY doc_date, doc_id"
            ):
                writer.execute(
                    "UPDATE documents SET markdown='TRANSIENT DOCUMENT B' "
                    "WHERE doc_id='DOC-1'"
                )
                writer.commit()
                return CursorAfterFetch(self.connection.execute(sql, parameters))
            return self.connection.execute(sql, parameters)

        def __getattr__(self, name):
            return getattr(self.connection, name)

    def recording_llm(prompt: str) -> str:
        prompts.append(prompt)
        return _llm(prompt)

    try:
        report = run_learn(
            sqlite_conn=cast(sqlite3.Connection, AbaConnection(conn)),
            duck_conn=duck_conn,
            llm_caller=recording_llm,
            force=True,
            progress=False,
        )
        assert report.docs_processed == 1
        assert report.docs_failed == 0
        assert len(prompts) == 1
        assert "TRANSIENT DOCUMENT B" not in prompts[0]
        assert "Source document text" in prompts[0]
        assert knowledge_freshness(conn).fresh
    finally:
        conn.close()
        writer.close()
        duck_conn.close()



def test_v2_dossier_captures_context_and_basis_from_same_snapshot(tmp_path):
    from esdc.knowledge.dossier import generate_pod_dossier

    source_path = tmp_path / "source.sqlite"
    conn = _database(source_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.commit()
    identity = issue_identity(
        conn,
        key=ApprovalKey(date(2024, 1, 1), 1, 1, 1, 0),
        actor="FA:test-v2-dossier-snapshot",
    )
    conn.execute(
        "INSERT INTO m_pod (pod_id_seq, pod_uuid, approval_date, institution_id, "
        "pod_type_id, rev_num, pod_order_no, pod_id, pod_name, pod_letter_num, "
        "is_latest) VALUES (?, ?, '2024-01-01', 1, 1, 0, ?, ?, 'POD One', "
        "'LETTER-1', 1)",
        (
            identity.pod_id_seq,
            str(identity.pod_uuid),
            identity.pod_order_no,
            identity.pod_id,
        ),
    )
    conn.execute(
        "INSERT INTO pod_project (pod_project_uuid, pod_id_seq, project_id, "
        "is_main_project, is_auto_inferred) VALUES ('map-1', ?, 'P-1', 1, 0)",
        (identity.pod_id_seq,),
    )
    conn.commit()
    store = KnowledgeStore(conn, allow_stale=True)
    store.ensure_tables()
    store.upsert_edges(
        [
            Edge(
                "pod",
                identity.pod_id,
                "HAS_PROJECT",
                "project",
                "P-1",
                method="registry_current",
            )
        ]
    )

    writer = sqlite3.connect(source_path)
    duck_conn = duckdb.connect(":memory:")
    ensure_dossier_table(duck_conn)
    prompts: list[str] = []
    initial_context_read = True

    class CursorAfterFetch:
        def __init__(self, cursor):
            self.cursor = cursor

        def fetchone(self):
            row = self.cursor.fetchone()
            writer.execute(
                "UPDATE api_project_resources SET project_name='Initial project' "
                "WHERE project_id='P-1'"
            )
            writer.commit()
            return row

        def __getattr__(self, name):
            return getattr(self.cursor, name)

    class AbaConnection:
        def __init__(self, connection):
            self.connection = connection

        def execute(self, sql, parameters=()):
            nonlocal initial_context_read
            if (
                initial_context_read
                and "FROM api_project_resources WHERE project_id = ?" in sql
            ):
                initial_context_read = False
                writer.execute(
                    "UPDATE api_project_resources "
                    "SET project_name='TRANSIENT DESCRIPTOR B' "
                    "WHERE project_id='P-1'"
                )
                writer.commit()
                return CursorAfterFetch(self.connection.execute(sql, parameters))
            return self.connection.execute(sql, parameters)

        def __getattr__(self, name):
            return getattr(self.connection, name)

    def recording_llm(prompt: str) -> str:
        prompts.append(prompt)
        return "Current dossier"

    try:
        status = generate_pod_dossier(
            identity.pod_id,
            cast(sqlite3.Connection, AbaConnection(conn)),
            duck_conn,
            store,
            "guideline-hash",
            recording_llm,
            force=True,
        )
        assert status == "built"
        assert len(prompts) == 1
        assert "TRANSIENT DESCRIPTOR B" not in prompts[0]
        assert "Initial project" in prompts[0]
    finally:
        conn.close()
        writer.close()
        duck_conn.close()



def test_default_learn_writer_opens_existing_v2_source(tmp_path, monkeypatch):
    from esdc.corpus import mirror
    from esdc.pod_registry import store as pod_store

    source_path = tmp_path / "source.sqlite"
    source = _database(source_path, current=False)
    source.execute("PRAGMA journal_mode=WAL")
    source.commit()
    source.close()
    monkeypatch.setattr(pod_store, "get_esdc_sqlite_path", lambda: source_path)
    monkeypatch.setattr(mirror, "refresh_all", lambda *_args: None)
    duck_conn = duckdb.connect(":memory:")

    try:
        report = run_learn(
            sqlite_conn=None,
            duck_conn=duck_conn,
            llm_caller=_llm,
            progress=False,
        )
        assert report.docs_processed == 1
        assert report.docs_failed == 0
        check = sqlite3.connect(source_path)
        try:
            assert check.execute("SELECT COUNT(*) FROM learn_state").fetchone()[0] == 1
        finally:
            check.close()
    finally:
        duck_conn.close()


def test_default_learn_dry_run_uses_read_only_source_without_schema_writes(
    tmp_path, monkeypatch
):
    from esdc.pod_registry import store as pod_store

    source_path = tmp_path / "source.sqlite"
    source = _database(source_path, current=False)
    source.execute("DROP TABLE learn_state")
    source.commit()
    source.close()
    monkeypatch.setattr(pod_store, "get_esdc_sqlite_path", lambda: source_path)
    duck_conn = duckdb.connect(":memory:")

    try:
        report = run_learn(
            sqlite_conn=None,
            duck_conn=duck_conn,
            dry_run=True,
            progress=False,
        )
        assert report.docs_total == 1
        assert report.docs_processed == 1
        check = sqlite3.connect(f"{source_path.as_uri()}?mode=ro", uri=True)
        try:
            assert check.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='learn_state'"
            ).fetchone() is None
        finally:
            check.close()
    finally:
        duck_conn.close()


def test_default_learn_with_empty_v2_corpus_does_not_open_provider(
    tmp_path, monkeypatch
):
    from esdc.corpus import mirror
    from esdc.knowledge import learn as learn_module
    from esdc.pod_registry import store as pod_store

    source_path = tmp_path / "source.sqlite"
    source = _database(source_path, docs=0, current=True)
    source.execute("PRAGMA journal_mode=WAL")
    source.commit()
    source.close()
    monkeypatch.setattr(pod_store, "get_esdc_sqlite_path", lambda: source_path)
    monkeypatch.setattr(mirror, "refresh_all", lambda *_args: None)
    monkeypatch.setattr(
        learn_module,
        "_open_default_llm",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("empty corpus should not open a provider")
        ),
    )
    duck_conn = duckdb.connect(":memory:")

    try:
        report = run_learn(
            sqlite_conn=None,
            duck_conn=duck_conn,
            progress=False,
        )
        assert report.docs_total == 0
        assert report.docs_processed == 0
        assert report.docs_failed == 0
    finally:
        duck_conn.close()



def test_corpus_learn_cli_uses_default_v2_source_writer(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from esdc.configs import Config
    from esdc.corpus import mirror
    from esdc.esdc import app
    from esdc.knowledge import learn as learn_module
    from esdc.pod_registry import store as pod_store

    source_path = tmp_path / "source.sqlite"
    source = _database(source_path, current=False)
    source.execute("PRAGMA journal_mode=WAL")
    source.commit()
    source.close()
    duckdb_path = tmp_path / "analytics.duckdb"
    monkeypatch.setattr(pod_store, "get_esdc_sqlite_path", lambda: source_path)
    monkeypatch.setattr(Config, "get_db_file", staticmethod(lambda: duckdb_path))
    monkeypatch.setattr(mirror, "refresh_all", lambda *_args: None)
    actual_run_learn = learn_module.run_learn

    def injected_learn(**kwargs):
        return actual_run_learn(**kwargs, llm_caller=_llm)

    monkeypatch.setattr(learn_module, "run_learn", injected_learn)
    result = CliRunner().invoke(app, ["corpus", "learn"])

    assert result.exit_code == 0, result.output
    assert "Documents: 1 processed" in result.output



def test_corpus_store_writer_opens_v2_without_legacy_schema(tmp_path):
    from esdc.corpus.store import CorpusStore

    source_path = tmp_path / "source.sqlite"
    source = _database(source_path, docs=0, current=True)
    source.execute("PRAGMA journal_mode=WAL")
    source.commit()
    source.close()

    store = CorpusStore(
        db_path=tmp_path / "analytics.duckdb",
        sqlite_path=source_path,
        read_only=False,
    )
    try:
        conn = store._get_sqlite()
        assert conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='pod_registry_state'"
        ).fetchone() is not None
        assert conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='project_pod'"
        ).fetchone() is None
    finally:
        store.close()


def test_v2_dossier_rebuilds_after_pod_metadata_changes_and_refreshes(tmp_path):
    from esdc.knowledge.dossier import generate_pod_dossier
    from esdc.knowledge.freshness import mark_knowledge_current
    from esdc.knowledge.guideline import load_guideline
    from esdc.pod_registry.read_models import refresh_pod_after_source_change

    conn = _database(tmp_path / "source.sqlite")
    from esdc.api_snapshot import ApiProjectResourceRow
    from esdc.timeseries_snapshot import ensure_timeseries_tables

    resource_columns = tuple(ApiProjectResourceRow.__dataclass_fields__)
    existing_columns = {
        str(row[1])
        for row in conn.execute("PRAGMA table_info(api_project_resources)").fetchall()
    }
    for column in resource_columns:
        if column not in existing_columns:
            conn.execute(
                f'ALTER TABLE api_project_resources ADD COLUMN "{column}" TEXT'
            )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS api_snapshot ("
        "source_table TEXT PRIMARY KEY, revision TEXT NOT NULL, row_count INTEGER NOT NULL, "
        "source_sha256 TEXT NOT NULL, fetched_at TEXT NOT NULL, publication_state TEXT NOT NULL, "
        "invalid_geometry_count INTEGER NOT NULL DEFAULT 0)"
    )
    ensure_timeseries_tables(conn)
    duck_conn = duckdb.connect(":memory:")
    identity = issue_identity(
        conn,
        key=ApprovalKey(date(2024, 1, 1), 1, 1, 1, 0),
        actor="FA:test-v2-dossier-cache-key",
    )
    conn.execute(
        "INSERT INTO m_pod (pod_id_seq, pod_uuid, approval_date, institution_id, "
        "pod_type_id, rev_num, pod_order_no, pod_id, pod_name, pod_letter_num, "
        "is_latest) VALUES (?, ?, '2024-01-01', 1, 1, 0, ?, ?, 'POD Name A', "
        "'LETTER-1', 1)",
        (
            identity.pod_id_seq,
            str(identity.pod_uuid),
            identity.pod_order_no,
            identity.pod_id,
        ),
    )
    conn.commit()
    store = KnowledgeStore(conn, allow_stale=True)
    store.ensure_tables()
    prompts: list[str] = []
    guideline_hash = load_guideline().content_hash

    def dossier_llm(prompt: str) -> str:
        prompts.append(prompt)
        return "Name B dossier" if "name: POD Name B" in prompt else "Name A dossier"

    try:
        conn.execute("BEGIN IMMEDIATE")
        assert mark_knowledge_current(conn, compute_knowledge_basis(conn))
        conn.commit()
        ensure_dossier_table(duck_conn)

        first = generate_pod_dossier(
            identity.pod_id, conn, duck_conn, store, guideline_hash, dossier_llm
        )
        assert first == "built"
        assert len(prompts) == 1
        assert "name: POD Name A" in prompts[0]

        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            "UPDATE m_pod SET pod_name='POD Name B' WHERE pod_id=?",
            (identity.pod_id,),
        )
        refresh_pod_after_source_change(
            conn, actor="FA:test-v2-dossier-cache-key"
        )
        conn.commit()
        assert get_dossier(
            duck_conn, "pod", identity.pod_id, sqlite_conn=conn
        ) is None

        second = generate_pod_dossier(
            identity.pod_id, conn, duck_conn, store, guideline_hash, dossier_llm
        )
        assert second == "built"
        assert len(prompts) == 2
        assert "name: POD Name B" in prompts[1]

        conn.execute("BEGIN IMMEDIATE")
        assert mark_knowledge_current(conn, compute_knowledge_basis(conn))
        conn.commit()
        dossier = get_dossier(
            duck_conn, "pod", identity.pod_id, sqlite_conn=conn
        )
        assert dossier is not None
        assert dossier["dossier_text"] == "Name B dossier"
    finally:
        conn.close()
        duck_conn.close()


def test_v2_dossier_discards_older_same_basis_claim_context(tmp_path):
    from esdc.knowledge.dossier import generate_pod_dossier
    from esdc.knowledge.freshness import mark_knowledge_current
    from esdc.knowledge.guideline import load_guideline
    from esdc.knowledge.store import Claim

    conn = _database(tmp_path / "source.sqlite")
    duck_conn = duckdb.connect(":memory:")
    identity = issue_identity(
        conn,
        key=ApprovalKey(date(2024, 1, 1), 1, 1, 1, 0),
        actor="FA:test-v2-dossier-context-race",
    )
    conn.execute(
        "INSERT INTO m_pod (pod_id_seq, pod_uuid, approval_date, institution_id, "
        "pod_type_id, rev_num, pod_order_no, pod_id, pod_name, pod_letter_num, "
        "is_latest) VALUES (?, ?, '2024-01-01', 1, 1, 0, ?, ?, 'POD One', "
        "'LETTER-1', 1)",
        (
            identity.pod_id_seq,
            str(identity.pod_uuid),
            identity.pod_order_no,
            identity.pod_id,
        ),
    )
    conn.commit()
    store = KnowledgeStore(conn, allow_stale=True)
    store.ensure_tables()
    store.upsert_edges(
        [
            Edge(
                "document",
                "DOC-1",
                "ABOUT_POD",
                "pod",
                identity.pod_id,
                method="registry_current",
            )
        ]
    )

    def claim(value: str) -> Claim:
        return Claim(
            "DOC-1",
            "issue",
            "delay_cause",
            value,
            subject_type="pod",
            subject_id=identity.pod_id,
            evidence=value,
        )

    store.replace_claims("DOC-1", [claim("old claim context")])
    conn.execute("BEGIN IMMEDIATE")
    assert mark_knowledge_current(conn, compute_knowledge_basis(conn))
    conn.commit()
    ensure_dossier_table(duck_conn)
    guideline_hash = load_guideline().content_hash
    newer_prompts: list[str] = []

    def old_dossier_llm(prompt: str) -> str:
        assert "old claim context" in prompt
        store.replace_claims("DOC-1", [claim("new claim context")])
        status = generate_pod_dossier(
            identity.pod_id,
            conn,
            duck_conn,
            store,
            guideline_hash,
            lambda new_prompt: newer_prompts.append(new_prompt)
            or "newer context dossier",
            force=True,
        )
        assert status == "built"
        return "older context dossier"

    try:
        status = generate_pod_dossier(
            identity.pod_id,
            conn,
            duck_conn,
            store,
            guideline_hash,
            old_dossier_llm,
            force=True,
        )
        assert status == "stale"
        assert len(newer_prompts) == 1
        assert "new claim context" in newer_prompts[0]
        dossier = get_dossier(
            duck_conn, "pod", identity.pod_id, sqlite_conn=conn
        )
        assert dossier is not None
        assert dossier["dossier_text"] == "newer context dossier"
    finally:
        conn.close()
        duck_conn.close()


def test_v2_dossier_read_hides_cached_context_after_link_removal(tmp_path):
    from esdc.knowledge.dossier import generate_pod_dossier
    from esdc.knowledge.freshness import mark_knowledge_current
    from esdc.knowledge.guideline import load_guideline

    conn = _database(tmp_path / "source.sqlite")
    duck_conn = duckdb.connect(":memory:")
    identity = issue_identity(
        conn,
        key=ApprovalKey(date(2024, 1, 1), 1, 1, 1, 0),
        actor="FA:test-v2-dossier-read-context",
    )
    conn.execute(
        "INSERT INTO m_pod (pod_id_seq, pod_uuid, approval_date, institution_id, "
        "pod_type_id, rev_num, pod_order_no, pod_id, pod_name, pod_letter_num, "
        "is_latest) VALUES (?, ?, '2024-01-01', 1, 1, 0, ?, ?, 'POD One', "
        "'LETTER-1', 1)",
        (
            identity.pod_id_seq,
            str(identity.pod_uuid),
            identity.pod_order_no,
            identity.pod_id,
        ),
    )
    conn.commit()
    store = KnowledgeStore(conn, allow_stale=True)
    store.ensure_tables()
    store.upsert_edges(
        [
            Edge(
                "document",
                "DOC-1",
                "ABOUT_POD",
                "pod",
                identity.pod_id,
                method="registry_current",
            )
        ]
    )
    conn.execute("BEGIN IMMEDIATE")
    assert mark_knowledge_current(conn, compute_knowledge_basis(conn))
    conn.commit()
    ensure_dossier_table(duck_conn)
    guideline_hash = load_guideline().content_hash

    try:
        assert (
            generate_pod_dossier(
                identity.pod_id,
                conn,
                duck_conn,
                store,
                guideline_hash,
                lambda _prompt: "dossier with linked source",
            )
            == "built"
        )
        store.delete_doc_edges("DOC-1")
        assert knowledge_freshness(conn).fresh

        assert (
            get_dossier(
                duck_conn, "pod", identity.pod_id, sqlite_conn=conn
            )
            is None
        )
    finally:
        conn.close()
        duck_conn.close()
