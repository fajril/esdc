"""Focused regression tests for read-only DuckDB serving.

Covers the guarantees from docs/plans/2026-09-10-duckdb-read-only-serving.md:

* A — every chat serving tool opens read-only connections and closes them.
* B — read-only mutators reject before any side effect.
* C — a read-only resolver/store never creates files or directories.
* D — retrieval succeeds while another process holds a read-only connection.
* E — AST guard: no executable raw ``duckdb.connect`` in ``esdc/chat``.

The runtime spy test (A) is the actual connection guarantee; the AST guard
(E) is a cheap tripwire that only sees executable code in ``esdc/chat``.
"""

from __future__ import annotations

import ast
import importlib
import json
import multiprocessing
import sys
from pathlib import Path

import diskcache
import duckdb
import pytest

import esdc
from esdc.corpus.chunker import Chunk
from esdc.corpus.store import CorpusNotReadyError, CorpusStore
from esdc.search.semantic_resolver import SemanticResolver

_DOC = {
    "doc_id": "abc123",
    "file_name": "s.pdf",
    "file_path": "/x/s.pdf",
    "file_hash": "ab" * 32,
    "doc_type": "surat",
    "doc_number": "SRT-1",
    "doc_date": "2026-01-05",
    "subject": "Persetujuan",
    "sender": "SKK",
    "recipient": "KKKS",
    "doc_level": "field",
    "wk_name": "Rokan",
    "field_name": "Duri",
    "project_name": None,
    "raw_entities": "{}",
    "metadata": "{}",
    "markdown": "# Surat\nisi",
    "extraction_method": "native",
    "page_count": 1,
}


class FixedEmbedder:
    """Deterministic embedder with a stable probe vector and call counter."""

    model = "qwen3-embedding-0.6b-q8_0"

    def __init__(self, vec: tuple[float, ...] = (1.0, 0.0, 0.0)):
        self._vec = list(vec)
        self.calls = 0

    def generate_embedding(self, text: str) -> list[float]:
        self.calls += 1
        return list(self._vec)

    def generate_embeddings_batch(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        return [list(self._vec) for _ in texts]


@pytest.fixture
def ro_env(tmp_path: Path, monkeypatch):
    """Tool environment pinned to a tmp DB with fake embedders + tmp cache."""
    tools_mod = importlib.import_module("esdc.chat.tools")
    embedder_mod = importlib.import_module("esdc.corpus.embedder")
    sr_mod = importlib.import_module("esdc.search.semantic_resolver")
    configs = importlib.import_module("esdc.configs")

    db_path = tmp_path / "corpus.duckdb"
    sqlite_path = tmp_path / "esdc.sqlite"

    monkeypatch.setattr(configs.Config, "get_db_file", classmethod(lambda cls: db_path))
    monkeypatch.setattr(configs.Config, "get_db_dir", classmethod(lambda cls: tmp_path))
    monkeypatch.setattr(
        configs.Config, "get_cache_dir", classmethod(lambda cls: tmp_path / "cache")
    )
    monkeypatch.setattr(configs.Config, "_load_config", classmethod(lambda cls: None))
    monkeypatch.setattr(
        embedder_mod, "InternalEmbedder", lambda: FixedEmbedder((1.0, 0.0, 0.0))
    )
    monkeypatch.setattr(
        sr_mod, "InternalEmbedder", lambda: FixedEmbedder((1.0, 0.0))
    )
    monkeypatch.setattr(tools_mod, "_corpus_embedder", None)

    cache = diskcache.Cache(str(tmp_path / "tool_cache"))
    monkeypatch.setattr(tools_mod, "_get_tool_cache", lambda: cache)

    if hasattr(tools_mod._semantic_resolver_tls, "resolver"):
        del tools_mod._semantic_resolver_tls.resolver

    yield db_path, sqlite_path

    cache.close()
    if hasattr(tools_mod._semantic_resolver_tls, "resolver"):
        del tools_mod._semantic_resolver_tls.resolver


def _seed_everything(db_path: Path, sqlite_path: Path) -> None:
    """Seed corpus tables/indexes and a project_embeddings space (writable)."""
    store = CorpusStore(
        db_path=db_path,
        sqlite_path=sqlite_path,
        embedder=FixedEmbedder((1.0, 0.0, 0.0)),
        read_only=False,
    )
    store.ensure_tables()
    store.insert_document(_DOC, [Chunk(0, None, "persetujuan POD lapangan Duri")])
    store.refresh_mirror()
    store.rebuild_indexes()
    store.close()

    resolver = SemanticResolver(
        db_path=db_path, embedder=FixedEmbedder((1.0, 0.0)), read_only=False
    )
    assert resolver.build_embeddings_table() is True
    resolver._get_connection().execute(
        "INSERT INTO project_embeddings "
        "(project_id, report_year, table_name, project_remarks, embedding) "
        "VALUES (?, ?, ?, ?, ?)",
        ["P1", 2024, "project_resources", "kendala teknis", [1.0, 0.0]],
    )
    resolver.close()


class _ConnectSpy:
    """Wrap the module-level duckdb.connect callable and record every call."""

    def __init__(self, monkeypatch):
        self.calls: list[dict] = []
        self.created: list[duckdb.DuckDBPyConnection] = []
        self._real = duckdb.connect
        monkeypatch.setattr(duckdb, "connect", self)

    def __call__(self, *args, **kwargs):
        self.calls.append(dict(kwargs))
        conn = self._real(*args, **kwargs)
        self.created.append(conn)
        return conn

    def reset(self) -> None:
        self.calls.clear()
        self.created.clear()

    def assert_read_only_and_closed(self, tool: str) -> None:
        assert self.calls, f"{tool} opened no DuckDB connection"
        assert all(c.get("read_only") is True for c in self.calls), self.calls
        for conn in self.created:
            with pytest.raises(duckdb.ConnectionException):
                conn.execute("SELECT 1")


# ---------------------------------------------------------------------------
# A — read-only serving guarantee for all four chat tools
# ---------------------------------------------------------------------------


def test_search_documents_serving_connections_are_read_only(ro_env, monkeypatch):
    db_path, sqlite_path = ro_env
    _seed_everything(db_path, sqlite_path)
    spy = _ConnectSpy(monkeypatch)

    from esdc.chat.tools import search_documents

    result = json.loads(search_documents.invoke({"query": "persetujuan"}))
    assert result["status"] in ("success", "no_results")
    spy.assert_read_only_and_closed("search_documents")


def test_aggregate_documents_serving_connections_are_read_only(ro_env, monkeypatch):
    db_path, sqlite_path = ro_env
    _seed_everything(db_path, sqlite_path)
    spy = _ConnectSpy(monkeypatch)

    from esdc.chat.tools import aggregate_documents

    result = json.loads(aggregate_documents.invoke({"query": "persetujuan"}))
    assert result["status"] in ("success", "no_results")
    spy.assert_read_only_and_closed("aggregate_documents")


def test_read_document_serving_connections_are_read_only(ro_env, monkeypatch):
    db_path, sqlite_path = ro_env
    _seed_everything(db_path, sqlite_path)
    spy = _ConnectSpy(monkeypatch)

    from esdc.chat.tools import read_document

    result = json.loads(read_document.invoke({"doc_id": "abc123"}))
    assert result["status"] == "success"
    spy.assert_read_only_and_closed("read_document")


def test_semantic_search_both_branches_open_read_only_connections(ro_env, monkeypatch):
    db_path, sqlite_path = ro_env
    _seed_everything(db_path, sqlite_path)
    spy = _ConnectSpy(monkeypatch)

    import esdc.chat.tools as tools_mod
    from esdc.chat.tools import semantic_search

    tools_mod.invalidate_tool_cache()
    result = json.loads(semantic_search.invoke({"query": "kendala teknis"}))

    # Both branches produced real results: remarks via the resolver, documents
    # via the corpus fan-out.
    assert result["remarks"]["status"] == "success"
    assert result["documents"]["status"] == "success"
    spy.assert_read_only_and_closed("semantic_search")

    # The memoized thread-local resolver must not retain a live connection.
    resolver = tools_mod._semantic_resolver_tls.resolver
    assert resolver._conn is None


def test_semantic_search_missing_corpus_still_read_only(ro_env, monkeypatch):
    db_path, _ = ro_env
    spy = _ConnectSpy(monkeypatch)

    import esdc.chat.tools as tools_mod
    from esdc.chat.tools import semantic_search

    tools_mod.invalidate_tool_cache()
    result = json.loads(semantic_search.invoke({"query": "kendala unik"}))

    assert result["documents"]["status"] == "not_available"
    # The resolver still opened a read-only connection even though the file
    # was missing; the fan-out degrades before connecting.
    assert spy.calls
    assert all(c.get("read_only") is True for c in spy.calls)
    assert not db_path.exists()


# ---------------------------------------------------------------------------
# B — mutator ordering / no side effects
# ---------------------------------------------------------------------------

_SR_MUTATORS = [
    ("build_embeddings_table", lambda r: r.build_embeddings_table()),
    ("generate_and_store_embeddings", lambda r: r.generate_and_store_embeddings()),
    ("_create_hnsw_index", lambda r: r._create_hnsw_index()),
    ("_create_embedding_indexes", lambda r: r._create_embedding_indexes()),
]

_CS_MUTATORS = [
    ("ensure_tables", lambda s: s.ensure_tables()),
    ("_migrate_legacy_entity_columns", lambda s: s._migrate_legacy_entity_columns()),
    ("_create_document_indexes", lambda s: s._create_document_indexes()),
    (
        "fill_blank_entities",
        lambda s: s.fill_blank_entities("abc123", {"wk_name": ["X"]}),
    ),
    ("insert_document", lambda s: s.insert_document(_DOC, [Chunk(0, None, "x")])),
    ("delete_document", lambda s: s.delete_document("abc123")),
    ("clear", lambda s: s.clear()),
    ("replace_chunks", lambda s: s.replace_chunks(_DOC, [Chunk(0, None, "x")])),
    (
        "_repin_legacy_model",
        lambda s: s._repin_legacy_model(
            None, "qwen3-embedding:0.6b", "qwen3-embedding-0.6b-q8_0"
        ),
    ),
    ("set_meta", lambda s: s.set_meta("m", 3)),
    ("rebuild_indexes", lambda s: s.rebuild_indexes()),
    ("refresh_mirror", lambda s: s.refresh_mirror()),
]


@pytest.mark.parametrize("name,call", _SR_MUTATORS)
def test_resolver_mutators_reject_before_side_effects(tmp_path, name, call):
    db = tmp_path / "sub" / "missing.duckdb"
    sqlite = tmp_path / "sub" / "missing.sqlite"
    emb = FixedEmbedder((1.0, 0.0))
    resolver = SemanticResolver(db_path=db, embedder=emb, read_only=True)

    with pytest.raises(PermissionError, match="read_only=False"):
        call(resolver)

    assert emb.calls == 0
    assert not db.exists()
    assert not sqlite.exists()
    assert resolver._conn is None
    resolver.close()


@pytest.mark.parametrize("name,call", _CS_MUTATORS)
def test_corpus_mutators_reject_before_side_effects(tmp_path, name, call):
    db = tmp_path / "missing.duckdb"
    sqlite = tmp_path / "missing.sqlite"
    emb = FixedEmbedder((1.0, 0.0, 0.0))
    store = CorpusStore(
        db_path=db, sqlite_path=sqlite, embedder=emb, read_only=True
    )

    with pytest.raises(PermissionError, match="read_only=False"):
        call(store)

    assert emb.calls == 0
    assert not db.exists()
    assert not sqlite.exists()
    assert store._conn is None
    assert store._sconn is None
    store.close()


def _seed_corpus(db_path: Path, sqlite_path: Path) -> None:
    store = CorpusStore(
        db_path=db_path,
        sqlite_path=sqlite_path,
        embedder=FixedEmbedder((1.0, 0.0, 0.0)),
        read_only=False,
    )
    store.ensure_tables()
    store.insert_document(_DOC, [Chunk(0, None, "isi surat")])
    store.refresh_mirror()
    store.close()


def _snapshot(db_path: Path) -> dict[str, int]:
    conn = duckdb.connect(str(db_path), read_only=True)
    try:
        counts: dict[str, int] = {}
        for table in ("documents", "document_chunks", "corpus_meta"):
            row = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
            assert row is not None
            counts[table] = int(row[0])
        return counts
    finally:
        conn.close()


def test_corpus_mutators_leave_content_unchanged(tmp_path):
    db = tmp_path / "ro_content.duckdb"
    sqlite = tmp_path / "ro_content.sqlite"
    _seed_corpus(db, sqlite)
    before = _snapshot(db)

    emb = FixedEmbedder((1.0, 0.0, 0.0))
    store = CorpusStore(db_path=db, sqlite_path=sqlite, embedder=emb, read_only=True)
    for _name, call in _CS_MUTATORS:
        with pytest.raises(PermissionError, match="read_only=False"):
            call(store)
    store.close()

    assert emb.calls == 0
    assert _snapshot(db) == before


def test_resolver_mutators_leave_content_unchanged(tmp_path):
    db = tmp_path / "ro_semantic.duckdb"
    writer = SemanticResolver(
        db_path=db, embedder=FixedEmbedder((1.0, 0.0)), read_only=False
    )
    assert writer.build_embeddings_table() is True
    writer.close()

    before = _semantic_snapshot(db)
    resolver = SemanticResolver(
        db_path=db, embedder=FixedEmbedder((1.0, 0.0)), read_only=True
    )
    for _name, call in _SR_MUTATORS:
        with pytest.raises(PermissionError, match="read_only=False"):
            call(resolver)
    resolver.close()
    assert _semantic_snapshot(db) == before


def _semantic_snapshot(db_path: Path) -> dict[str, int]:
    conn = duckdb.connect(str(db_path), read_only=True)
    try:
        counts: dict[str, int] = {}
        for table in ("project_embeddings", "semantic_meta"):
            row = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
            assert row is not None
            counts[table] = int(row[0])
        return counts
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# C — non-creation
# ---------------------------------------------------------------------------


def test_readonly_resolver_missing_db_creates_nothing(tmp_path):
    db = tmp_path / "sub" / "missing.duckdb"
    resolver = SemanticResolver(
        db_path=db, embedder=FixedEmbedder((1.0, 0.0)), read_only=True
    )
    # A failed connect degrades to an error envelope instead of raising, so
    # query paths cannot propagate an IOException mid-serve.
    out = resolver._ensure_semantic_meta()
    assert out is not None
    assert out["status"] == "error"
    assert not db.exists()
    assert not db.parent.exists()
    resolver.close()


def test_readonly_corpus_missing_db_creates_nothing(tmp_path):
    db = tmp_path / "sub" / "missing.duckdb"
    sqlite = tmp_path / "sub" / "missing.sqlite"
    store = CorpusStore(
        db_path=db, sqlite_path=sqlite, embedder=FixedEmbedder(), read_only=True
    )
    with pytest.raises(CorpusNotReadyError):
        store.validate_readiness("search")
    assert not db.exists()
    assert not sqlite.exists()
    assert not db.parent.exists()
    store.close()


def test_missing_corpus_tool_reports_not_available_without_creating_db(ro_env):
    db_path, sqlite_path = ro_env
    from esdc.chat.tools import search_documents

    result = json.loads(search_documents.invoke({"query": "apapun"}))
    assert result["status"] == "not_available"
    assert "esdc corpus commit" in result["message"]
    assert not db_path.exists()
    assert not sqlite_path.exists()


# ---------------------------------------------------------------------------
# D — separate-process reader coexists with serving
# ---------------------------------------------------------------------------


def _hold_readonly_connection(db_path_str: str, ready, release) -> None:
    root = str(Path(__file__).resolve().parent.parent)
    if root not in sys.path:
        sys.path.insert(0, root)
    from esdc.dbmanager import get_duckdb_connection

    conn = get_duckdb_connection(db_path_str, read_only=True)
    try:
        conn.execute("SELECT 1")
        ready.set()
        release.wait(30)
    finally:
        conn.close()


def test_separate_process_reader_holds_connection_during_retrieval(ro_env):
    db_path, sqlite_path = ro_env
    _seed_everything(db_path, sqlite_path)

    ctx = multiprocessing.get_context("spawn")
    ready = ctx.Event()
    release = ctx.Event()
    proc = ctx.Process(
        target=_hold_readonly_connection, args=(str(db_path), ready, release)
    )
    proc.start()
    try:
        assert ready.wait(30), "child process never held a read-only connection"
        from esdc.chat.tools import read_document, search_documents

        result = json.loads(search_documents.invoke({"query": "persetujuan"}))
        assert result["status"] in ("success", "no_results")
        doc = json.loads(read_document.invoke({"doc_id": "abc123"}))
        assert doc["status"] == "success"
    finally:
        release.set()
        proc.join(30)
    assert proc.exitcode == 0


# ---------------------------------------------------------------------------
# E — AST guard against executable raw duckdb.connect in esdc/chat
# ---------------------------------------------------------------------------


def _raw_connect_violations(path: Path) -> list[str]:
    """Executable duckdb.connect call sites; strings/docstrings are ignored.

    Covers ``import duckdb``, ``import duckdb as X`` and
    ``from duckdb import connect [as Y]``. Only executable AST nodes are
    inspected, so example snippets inside prompt/docstring strings never
    register.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    duckdb_names = {"duckdb"}
    direct_connects: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "duckdb":
                    duckdb_names.add(alias.asname or "duckdb")
        elif isinstance(node, ast.ImportFrom) and node.module == "duckdb":
            for alias in node.names:
                if alias.name == "connect":
                    direct_connects.add(alias.asname or "connect")

    violations: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if (
            isinstance(func, ast.Attribute)
            and func.attr == "connect"
            and isinstance(func.value, ast.Name)
            and func.value.id in duckdb_names
        ) or isinstance(func, ast.Name) and func.id in direct_connects:
            violations.append(f"{path}:{node.lineno}")
    return violations


def test_no_executable_raw_duckdb_connect_in_chat():
    chat_dir = Path(esdc.__file__).resolve().parent / "chat"
    files = sorted(chat_dir.rglob("*.py"))
    assert files, "esdc/chat has no Python files; guard scanned nothing"

    violations: list[str] = []
    for source_file in files:
        violations.extend(_raw_connect_violations(source_file))

    assert violations == [], (
        "Executable raw duckdb.connect in esdc/chat bypasses the read-only "
        f"serving path: {violations}. The runtime spy test "
        "(test_*_serving_connections_are_read_only) is the real guarantee."
    )


# ---------------------------------------------------------------------------
# F — durable safe defaults, factories and explicit writers (Phase 2)
# ---------------------------------------------------------------------------


class _NoopEmbedder:
    """Deterministic 3-dim embedder for default-constructor tests."""

    model = "fake-embed"

    def generate_embedding(self, text: str) -> list[float]:
        return [1.0, 0.0, 0.0]

    def generate_embeddings_batch(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0, 0.0] for _ in texts]


def test_default_constructor_instances_reject_writes(isolated_config, monkeypatch):
    """Both classes default to read-only and refuse writes outside chat."""
    import esdc.corpus.embedder as corpus_embedder_mod
    import esdc.search.semantic_resolver as sr_mod

    monkeypatch.setattr(corpus_embedder_mod, "InternalEmbedder", _NoopEmbedder)
    monkeypatch.setattr(sr_mod, "InternalEmbedder", _NoopEmbedder)

    resolver = SemanticResolver()
    assert resolver._read_only is True
    with pytest.raises(PermissionError, match="read_only=False"):
        resolver.build_embeddings_table()
    resolver.close()

    store = CorpusStore()
    assert store._read_only is True
    with pytest.raises(PermissionError, match="read_only=False"):
        store.ensure_tables()
    store.close()

    assert not (isolated_config / ".esdc" / "esdc.duckdb").exists()


def test_open_corpus_store_reader_validates_without_creating(isolated_config):
    """The default factory is a reader: it validates and never creates files."""
    import typer

    import esdc.esdc as cli

    with pytest.raises(typer.Exit):
        cli._open_corpus_store()
    assert not (isolated_config / ".esdc" / "esdc.duckdb").exists()


def test_open_corpus_store_writer_ensures_schema(isolated_config, monkeypatch):
    """An explicit writer factory ensures schema; the default validates it."""
    import esdc.corpus.embedder as corpus_embedder_mod
    import esdc.esdc as cli

    monkeypatch.setattr(corpus_embedder_mod, "InternalEmbedder", _NoopEmbedder)

    writer = cli._open_corpus_store(read_only=False)
    try:
        assert writer._read_only is False
        assert writer._db_path.exists()
    finally:
        writer.close()

    reader = cli._open_corpus_store()
    try:
        assert reader._read_only is True
    finally:
        reader.close()


def test_run_reembed_documents_rejects_injected_read_only_store(tmp_path):
    """An injected read-only store fails before any mutation side effect."""
    from esdc.corpus.pipeline import run_reembed_documents

    db = tmp_path / "inj.duckdb"
    sqlite = tmp_path / "inj.sqlite"
    writer = CorpusStore(
        db_path=db, sqlite_path=sqlite, embedder=_NoopEmbedder(), read_only=False
    )
    writer.ensure_tables()
    writer.insert_document(_DOC, [Chunk(0, None, "isi surat")])
    writer.refresh_mirror()
    writer.close()

    before = _snapshot(db)
    reader = CorpusStore(
        db_path=db, sqlite_path=sqlite, embedder=_NoopEmbedder(), read_only=True
    )
    try:
        with pytest.raises(PermissionError, match="read_only=False"):
            run_reembed_documents([_DOC["doc_id"]], store=reader)
    finally:
        reader.close()
    assert _snapshot(db) == before


def test_explicit_writer_workflow_round_trips(tmp_path):
    """An explicit writer creates, inserts, re-embeds, refreshes and deletes."""
    from esdc.corpus.pipeline import run_reembed_documents

    db = tmp_path / "writer.duckdb"
    sqlite = tmp_path / "writer.sqlite"
    store = CorpusStore(
        db_path=db, sqlite_path=sqlite, embedder=_NoopEmbedder(), read_only=False
    )
    try:
        store.ensure_tables()
        store.insert_document(_DOC, [Chunk(0, None, "isi surat")])
        store.refresh_mirror()
        assert store.get_document(_DOC["doc_id"]) is not None
        store.rebuild_indexes()

        report = run_reembed_documents([_DOC["doc_id"]], store=store)
        assert report.failed == {}

        store.delete_document(_DOC["doc_id"])
        assert store.get_document(_DOC["doc_id"]) is None
    finally:
        store.close()


def test_reader_lookup_list_evaluate_export_paths(tmp_path, monkeypatch):
    """Reader lookup/list/eval/export paths run against a committed corpus."""
    import esdc.configs as configs
    import esdc.corpus.embedder as corpus_embedder_mod
    from esdc.corpus.evaluate import run_eval
    from esdc.corpus.pipeline import run_export

    db = tmp_path / "esdc.duckdb"
    sqlite = tmp_path / "esdc.sqlite"
    monkeypatch.setattr(configs.Config, "get_db_file", classmethod(lambda cls: db))
    monkeypatch.setattr(configs.Config, "get_db_dir", classmethod(lambda cls: tmp_path))
    monkeypatch.setattr(corpus_embedder_mod, "InternalEmbedder", _NoopEmbedder)

    src = tmp_path / "doc.corpus.md"
    doc = dict(_DOC)
    doc["file_path"] = str(src)
    writer = CorpusStore(
        db_path=db, sqlite_path=sqlite, embedder=_NoopEmbedder(), read_only=False
    )
    writer.ensure_tables()
    writer.insert_document(doc, [Chunk(0, None, "isi surat")])
    writer.refresh_mirror()
    writer.rebuild_indexes()
    writer.close()

    before = _snapshot(db)

    reader = CorpusStore(
        db_path=db, sqlite_path=sqlite, embedder=_NoopEmbedder(), read_only=True
    )
    try:
        assert reader.get_document(_DOC["doc_id"]) is not None
        assert [d["doc_id"] for d in reader.list_documents()] == [_DOC["doc_id"]]
        assert reader.fingerprint_rows() == [(_DOC["doc_id"], _DOC["file_hash"])]
        assert reader.document_exists(_DOC["file_hash"]) is True
    finally:
        reader.close()

    queries = tmp_path / "queries.jsonl"
    queries.write_text(
        json.dumps({"query": "isi surat", "expected": [_DOC["doc_id"]]}) + "\n",
        encoding="utf-8",
    )
    eval_report = run_eval(queries, ks=(1,))
    assert eval_report.n_queries == 1

    export_report = run_export([src])
    assert export_report.processed == [src.name]
    assert src.exists()

    assert _snapshot(db) == before
