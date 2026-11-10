# tests/search/test_semantic_resolver_embedder.py
"""SemanticResolver embeds queries locally and accepts an injected embedder."""

from __future__ import annotations

import hashlib
import inspect
import json
import os
from typing import cast

import duckdb
import pytest

from esdc import embedders
from esdc.embedders import EmbeddingError
from esdc.search.semantic_resolver import SemanticResolver


@pytest.fixture(autouse=True)
def _default_embedding_signature(monkeypatch):
    """Probe tests run against one explicit effective embedding config."""
    monkeypatch.setattr(
        embedders,
        "config_signature",
        lambda backend=None: (
            "ollama",
            "http://localhost:11434",
            "qwen3-embedding-0.6b-q8_0",
            "",
        ),
    )


class _FixedEmbedder:
    model = "qwen3-embedding-0.6b-q8_0"

    def __init__(self, vec):
        self._vec = list(vec)
        self.calls = 0

    def generate_embedding(self, text):
        self.calls += 1
        return list(self._vec)

    def generate_embeddings_batch(self, texts):
        return [list(self._vec) for _ in texts]


def test_accepts_injected_embedder(tmp_path):
    e = _FixedEmbedder([1.0])
    r = SemanticResolver(db_path=tmp_path / "x.duckdb", embedder=e)
    assert r._embedder is e


def test_model_param_is_gone():
    assert "model" not in inspect.signature(SemanticResolver.__init__).parameters


def test_query_path_uses_the_injected_embedder(tmp_path):
    e = _FixedEmbedder([1.0])
    db_path = tmp_path / "x.duckdb"
    duckdb.connect(str(db_path)).close()  # empty DB: reader opens, table absent
    r = SemanticResolver(db_path=db_path, embedder=e)
    # No embeddings table exists, so search short-circuits before embedding.
    out = r.search_by_text("anything")
    assert out["status"] == "not_available"
    assert e.calls == 0


def _seeded(tmp_path, vec, dim=None):
    """A resolver whose semantic_meta is pinned by an embedder producing vec.

    ``dim`` defaults to ``len(vec)`` so the stored probe length always
    matches the recorded dimension; pass it explicitly only to build a
    deliberately inconsistent pin.
    """
    if dim is None:
        dim = len(vec)
    r = SemanticResolver(
        db_path=tmp_path / "s.duckdb", embedder=_FixedEmbedder(vec), read_only=False
    )
    conn = r._get_connection()
    conn.execute(
        "CREATE TABLE IF NOT EXISTS semantic_meta "
        "(embedding_model VARCHAR, dim INTEGER, probe_vec JSON, "
        "content_version INTEGER)"
    )
    conn.execute(
        "INSERT INTO semantic_meta VALUES (?, ?, ?, ?)",
        [
            "qwen3-embedding-0.6b-q8_0",
            dim,
            json.dumps(list(vec)),
            SemanticResolver.NARRATIVE_CONTENT_VERSION,
        ],
    )
    return r


def test_meta_table_name():
    assert SemanticResolver.SEMANTIC_META == "semantic_meta"


def test_drifted_embedder_returns_not_available_not_exception(tmp_path):
    """Same-dimension cosine drift must degrade, not raise.

    Both probes have length 2, so the only thing that can reject them is
    the cosine comparison against PROBE_TOLERANCE — an orthogonal vector
    scores 0.0, well below it. The message must name the cosine, proving
    the dimension guard did not fire instead.
    """
    r = _seeded(tmp_path, [1.0, 0.0])
    r._embedder = _FixedEmbedder([0.0, 1.0])
    out = r._ensure_semantic_meta()
    r.close()
    assert out is not None
    assert out["status"] == "not_available"
    assert out["results"] == []
    assert "cosine" in out["message"]
    assert "dimension" not in out["message"]
    assert "esdc reload" in out["message"]


def test_legacy_space_without_meta_degrades_then_writable_repair(tmp_path):
    """A missing semantic_meta is read-only degradation, not query seeding.

    The reader must NOT create or seed the table. Repair is the advertised
    maintenance writer path (what ``esdc reload --embeddings-only`` runs:
    ``build_embeddings_table``); after it, the reader validates.
    """
    db_path = tmp_path / "s.duckdb"
    duckdb.connect(str(db_path)).close()  # reader opens an empty DB; no table
    r = SemanticResolver(db_path=db_path, embedder=_FixedEmbedder([1.0, 0.0]))
    out = r._ensure_semantic_meta()
    assert out is not None
    assert out["status"] == "not_available"
    assert "esdc reload --embeddings-only" in out["message"]
    # Reader never seeded the pin: the table was not created.
    present = (
        r._get_connection()
        .execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = 'semantic_meta'"
        )
        .fetchall()
    )
    assert present == []
    r.close()

    # Writable maintenance path repairs the pin.
    writer = SemanticResolver(
        db_path=db_path, embedder=_FixedEmbedder([1.0, 0.0]), read_only=False
    )
    assert writer.build_embeddings_table() is True
    writer.close()

    # Reader now validates against the repaired pin.
    reader = SemanticResolver(db_path=db_path, embedder=_FixedEmbedder([1.0, 0.0]))
    assert reader._ensure_semantic_meta() is None
    reader.close()


def test_legacy_meta_missing_probe_column_repairable_by_reload(tmp_path):
    """Gap 6: a legacy semantic_meta without probe_vec is repairable.

    The reader advertises ``esdc reload --embeddings-only``; the writer it
    runs (``build_embeddings_table``) must actually add the probe column and
    (re)seed the probe, after which the reader validates.
    """
    db_path = tmp_path / "s.duckdb"
    legacy = SemanticResolver(
        db_path=db_path, embedder=_FixedEmbedder([1.0, 0.0]), read_only=False
    )
    conn = legacy._get_connection()
    conn.execute("CREATE TABLE semantic_meta (embedding_model VARCHAR, dim INTEGER)")
    conn.execute(
        "INSERT INTO semantic_meta (embedding_model, dim) VALUES (?, ?)",
        ["qwen3-embedding-0.6b-q8_0", 2],
    )
    out = legacy._ensure_semantic_meta()
    assert out is not None
    assert out["status"] == "not_available"
    assert "esdc reload --embeddings-only" in out["message"]
    legacy.close()

    writer = SemanticResolver(
        db_path=db_path, embedder=_FixedEmbedder([1.0, 0.0]), read_only=False
    )
    assert writer.build_embeddings_table() is True
    writer.close()

    reader = SemanticResolver(db_path=db_path, embedder=_FixedEmbedder([1.0, 0.0]))
    assert reader._ensure_semantic_meta() is None
    reader.close()


@pytest.mark.parametrize("version_state", ["missing", "stale"])
def test_unavailable_content_version_preserves_read_only_database(
    tmp_path, version_state
):
    """Legacy or stale narrative metadata degrades without reader writes."""
    db_path = tmp_path / "s.duckdb"
    writer = SemanticResolver(
        db_path=db_path, embedder=_FixedEmbedder([1.0, 0.0]), read_only=False
    )
    conn = writer._get_connection()
    if version_state == "missing":
        conn.execute(
            "CREATE TABLE semantic_meta "
            "(embedding_model VARCHAR, dim INTEGER, probe_vec JSON)"
        )
        conn.execute(
            "INSERT INTO semantic_meta VALUES (?, ?, ?)",
            ["qwen3-embedding-0.6b-q8_0", 2, json.dumps([1.0, 0.0])],
        )
    else:
        conn.execute(
            "CREATE TABLE semantic_meta "
            "(embedding_model VARCHAR, dim INTEGER, probe_vec JSON, "
            "content_version INTEGER)"
        )
        conn.execute(
            "INSERT INTO semantic_meta VALUES (?, ?, ?, ?)",
            [
                "qwen3-embedding-0.6b-q8_0",
                2,
                json.dumps([1.0, 0.0]),
                SemanticResolver.NARRATIVE_CONTENT_VERSION - 1,
            ],
        )
    writer.close()

    before = hashlib.sha256(db_path.read_bytes()).hexdigest()
    embedder = _FixedEmbedder([1.0, 0.0])
    reader = SemanticResolver(db_path=db_path, embedder=embedder, read_only=True)
    out = reader._ensure_semantic_meta()
    reader.close()
    after = hashlib.sha256(db_path.read_bytes()).hexdigest()

    assert out is not None
    assert out["status"] == "not_available"
    assert "esdc reload --embeddings-only" in out["message"]
    assert embedder.calls == 0
    assert after == before


def test_pin_check_embeds_probe_only_once(tmp_path):
    """A stable (config signature, db signature) pair means the memo hits again.

    Repointed for Task 2 (D2): the memo is keyed by the effective config and the facts
    the check was computed from — the DB file plus the stored pin, dimension and probe —
    and the proof is the request count, not an inspection of the cache.
    """
    r = _seeded(tmp_path, [1.0, 0.0])
    assert r._ensure_semantic_meta() is None
    embedder = cast(_FixedEmbedder, r._embedder)
    embedder.calls = 0
    assert r._ensure_semantic_meta() is None
    r.close()
    # The second check hit the memo: no probe request was made for the same state.
    assert embedder.calls == 0


def test_modern_database_returns_results_read_only(tmp_path):
    """A properly pinned database still serves results on a read-only reader.

    Seeded by a writable resolver, then served by a separate read-only
    resolver: the pin validates (memo held) and the vector search returns
    the stored row. This is the "preserve modern-database result behavior"
    guard for the new read-only validation path.
    """
    db_path = tmp_path / "s.duckdb"
    writer = SemanticResolver(
        db_path=db_path, embedder=_FixedEmbedder([1.0, 0.0]), read_only=False
    )
    assert writer.build_embeddings_table() is True
    conn = writer._get_connection()
    conn.execute(
        "INSERT INTO project_embeddings "
        "(project_id, report_year, table_name, project_remarks, embedding) "
        "VALUES (?, ?, ?, ?, ?)",
        ["P1", 2024, "project_resources", "masalah teknis sumur", [1.0, 0.0]],
    )
    writer.close()

    reader = SemanticResolver(
        db_path=db_path, embedder=_FixedEmbedder([1.0, 0.0]), read_only=True
    )
    out = reader.search_by_text("masalah teknis", limit=5)
    reader.close()
    assert out["status"] == "success"
    assert out["results"][0]["project_id"] == "P1"


def test_pin_recheck_on_db_signature_change(tmp_path):
    """A changed DB signature forces the pin to be re-verified.

    Covers e.g. esdc fetch/reload replacing the DB file underneath a
    cached resolver, instead of trusting a stale memo.
    """
    r = _seeded(tmp_path, [1.0, 0.0])
    assert r._ensure_semantic_meta() is None
    embedder = cast(_FixedEmbedder, r._embedder)
    embedder.calls = 0

    # Simulate esdc fetch/reload replacing the file underneath the cached
    # resolver: same path, a different file. The recorded signature no longer
    # matches, so the pin is read again rather than trusted from the memo.
    replacement = r._db_path.parent / "replaced.duckdb"
    duckdb.connect(str(replacement)).close()
    os.replace(replacement, r._db_path)

    assert r._ensure_semantic_meta() is None
    r.close()
    assert embedder.calls == 1


def test_build_writes_the_pin(tmp_path):
    r = SemanticResolver(
        db_path=tmp_path / "s.duckdb",
        embedder=_FixedEmbedder([1.0, 0.0]),
        read_only=False,
    )
    r.build_embeddings_table()
    row = (
        r._get_connection()
        .execute("SELECT embedding_model, dim FROM semantic_meta")
        .fetchone()
    )
    assert row is not None
    r.close()
    assert row[0] == "qwen3-embedding-0.6b-q8_0"
    assert row[1] == 2


# --- Task 2: lazy resolution, cached probe validation, transactional generation ---


def test_defaults_to_internal_embedder(tmp_path, monkeypatch):
    """The default is whatever the factory resolves, not a class named in the resolver.

    Repointed for Task 2: the resolver no longer carries an in-process default of its
    own — it asks the factory, and the factory is what the config decides.
    """
    _patch_config(monkeypatch, "service-a")
    client = _CountingEmbedder([1.0, 0.0], "service-a")
    monkeypatch.setattr(embedders, "get_embedder", lambda backend=None: client)

    r = SemanticResolver(db_path=tmp_path / "x.duckdb")
    assert r.embedder is client
    assert client.calls == []
    r.close()


class _CountingEmbedder:
    """Fixed-vector client that records what it was asked to embed."""

    def __init__(self, vec, name="qwen3-embedding-0.6b-q8_0"):
        self._vec = list(vec)
        self.model = name
        self.calls: list[tuple[str, str]] = []

    def generate_embedding(self, text):
        self.calls.append(
            ("probe", text) if text == embedders.PROBE_TEXT else ("query", text)
        )
        return list(self._vec)

    def generate_embeddings_batch(self, texts):
        if not texts:
            return []
        self.calls.append(("batch", texts[0]))
        return [list(self._vec) for _ in texts]

    def check_available(self) -> str | None:
        return None


class _FailingBatchClient(_CountingEmbedder):
    """Answers for the first batch, then fails for the next one."""

    def __init__(self, vec, name="qwen3-embedding-0.6b-q8_0", fail_after=1):
        super().__init__(vec, name)
        self.fail_after = fail_after

    def generate_embeddings_batch(self, texts):
        if not texts:
            return []
        self.calls.append(("batch", texts[0]))
        if len([kind for kind, _ in self.calls if kind == "batch"]) > self.fail_after:
            raise EmbeddingError("[Embedding] service at fake failed")
        return [list(self._vec) for _ in texts]


def _patch_config(monkeypatch, model, api_key=""):
    state = {"model": model, "api_key": api_key}
    monkeypatch.setattr(
        embedders,
        "config_signature",
        lambda backend=None: (
            "ollama",
            "http://localhost:11434",
            state["model"],
            state["api_key"],
        ),
    )
    return state


def _pinned_space(tmp_path, model, db_name="pinned.duckdb"):
    """A resolver space pinned to `model` with one stored vector and its probe.

    Built through a writer so the schema is the one a reader expects, then the pin is
    renamed: the pin, the probe and the vector are simply what a reader finds.
    """
    db_path = tmp_path / db_name
    writer = SemanticResolver(
        db_path=db_path, embedder=_FixedEmbedder([1.0, 0.0]), read_only=False
    )
    assert writer.build_embeddings_table() is True
    conn = writer._get_connection()
    conn.execute(
        "INSERT INTO project_embeddings "
        "(project_id, report_year, table_name, project_remarks, embedding) "
        "VALUES (?, ?, ?, ?, ?)",
        ["P1", 2024, "project_resources", "masalah teknis sumur", [1.0, 0.0]],
    )
    conn.execute("UPDATE semantic_meta SET embedding_model = ?", [model])
    writer.close()
    return db_path


def test_constructor_does_not_call_the_service(tmp_path, monkeypatch):
    """Constructing the resolver resolves nothing; the first vector op does."""
    _patch_config(monkeypatch, "service-a")
    resolved: list = []
    client = _CountingEmbedder([1.0, 0.0])

    def factory(backend=None):
        resolved.append(client)
        return client

    monkeypatch.setattr(embedders, "get_embedder", factory)

    db_path = tmp_path / "x.duckdb"
    space = _pinned_space(tmp_path, "service-a", db_name="x.duckdb")
    assert space == db_path

    r = SemanticResolver(db_path=db_path)
    assert resolved == []
    assert client.calls == []

    out = r.search_by_text("anything")
    assert out["status"] == "success", out
    assert resolved == [client]  # resolved on the operation that needed a vector
    assert client.calls == [
        ("probe", embedders.PROBE_TEXT),
        ("query", "anything"),
    ]
    r.close()


def test_client_replaced_when_effective_config_changes(tmp_path, monkeypatch):
    """A superseded config (including its credential) must not keep its client."""
    state = _patch_config(monkeypatch, "service-a")
    clients = {"service-a": _CountingEmbedder([1.0, 0.0], "service-a")}
    resolved: list[tuple] = []

    def factory(backend=None):
        sig = embedders.config_signature()
        resolved.append(sig)
        return clients[sig[2]]

    monkeypatch.setattr(embedders, "get_embedder", factory)

    r = SemanticResolver(db_path=tmp_path / "x.duckdb")
    assert r.embedder is clients["service-a"]
    assert r.embedder is clients["service-a"]
    assert len(resolved) == 1

    state["model"] = "service-b"
    clients["service-b"] = _CountingEmbedder([1.0, 0.0], "service-b")
    assert r.embedder is clients["service-b"]
    assert len(resolved) == 2

    state["api_key"] = "sk-test"
    assert r.embedder is clients["service-b"]
    assert len(resolved) == 3
    r.close()


def test_query_does_not_write_probe_metadata(tmp_path, monkeypatch):
    """A read validates the stored probe without writing anything to the DB."""
    _patch_config(monkeypatch, "service-a")
    db_path = tmp_path / "s.duckdb"
    writer = SemanticResolver(
        db_path=db_path,
        embedder=_CountingEmbedder([1.0, 0.0], "space-a"),
        read_only=False,
    )
    assert writer.build_embeddings_table() is True
    writer._get_connection().execute(
        "INSERT INTO project_embeddings (project_id, report_year, table_name, "
        "project_remarks, embedding) VALUES (?, ?, ?, ?, ?)",
        ["P1", 2024, "project_resources", "masalah teknis sumur", [1.0, 0.0]],
    )
    writer.close()

    before = hashlib.sha256(db_path.read_bytes()).hexdigest()
    client = _CountingEmbedder([1.0, 0.0], "space-b")
    reader = SemanticResolver(db_path=db_path, embedder=client, read_only=True)
    out = reader.search_by_text("masalah teknis", limit=5)
    reader.close()

    assert out["status"] == "success", out
    assert client.calls == [
        ("probe", embedders.PROBE_TEXT),
        ("query", "masalah teknis"),
    ]
    assert hashlib.sha256(db_path.read_bytes()).hexdigest() == before


def test_validation_runs_once_per_config_signature_and_db_signature(
    tmp_path, monkeypatch
):
    """The pin is validated once per (config signature, db signature), not per query."""
    state = _patch_config(monkeypatch, "service-a")
    db_path = tmp_path / "s.duckdb"
    writer = SemanticResolver(
        db_path=db_path,
        embedder=_CountingEmbedder([1.0, 0.0], "service-a"),
        read_only=False,
    )
    assert writer.build_embeddings_table() is True
    writer._get_connection().execute(
        "INSERT INTO project_embeddings (project_id, report_year, table_name, "
        "project_remarks, embedding) VALUES (?, ?, ?, ?, ?)",
        ["P1", 2024, "project_resources", "masalah teknis sumur", [1.0, 0.0]],
    )
    writer.close()

    client = _CountingEmbedder([1.0, 0.0], "service-a")
    reader = SemanticResolver(db_path=db_path, embedder=client, read_only=True)
    reader.search_by_text("first query", limit=5)
    reader.search_by_text("second query", limit=5)
    assert client.calls == [
        ("probe", embedders.PROBE_TEXT),
        ("query", "first query"),
        ("query", "second query"),
    ]
    # The memo hit is proved by the count above: the second query embedded only what it
    # was asked about, so the state it validated is the state it recorded.

    # A different effective config is a different space: validate again, once.
    state["model"] = "service-b"
    reader.search_by_text("third query", limit=5)
    assert [text for kind, text in client.calls if kind == "probe"] == [
        embedders.PROBE_TEXT,
        embedders.PROBE_TEXT,
    ]
    reader.search_by_text("fourth query", limit=5)
    assert [text for kind, text in client.calls if kind == "probe"] == [
        embedders.PROBE_TEXT,
        embedders.PROBE_TEXT,
    ]

    # A validation that never succeeded is not memoized as verified.
    reader.close()
    drifted = _CountingEmbedder([0.0, 1.0], "service-b")
    second = SemanticResolver(db_path=db_path, embedder=drifted, read_only=True)
    out = second.search_by_text("flare", limit=5)
    assert out is not None and out["status"] == "not_available", out
    second.search_by_text("flare", limit=5)
    assert [text for kind, text in drifted.calls if kind == "probe"] == [
        embedders.PROBE_TEXT,
        embedders.PROBE_TEXT,
    ]
    second.close()


def test_index_without_probe_is_not_solemnized_by_seeding_from_new_service(tmp_path):
    """A probe from a new service cannot certify vectors produced elsewhere.

    The reader must not seed the pin as a side effect of a query: seeding is only the
    migration path for a space that is still empty.
    """
    new_service = _CountingEmbedder([1.0, 0.0], "new-service")
    db_path = _pinned_space(tmp_path, "stored-service", db_name="s.duckdb")
    # The space holds vectors from another service and no probe to compare against.
    conn = duckdb.connect(str(db_path))
    conn.execute("UPDATE semantic_meta SET probe_vec = NULL")
    conn.close()

    resolver = SemanticResolver(db_path=db_path, embedder=new_service)
    result = resolver._ensure_semantic_meta()
    assert result is not None
    assert result["status"] == "not_available", result
    assert "stored-service" in result["message"]
    reader = resolver._get_connection()
    probe_row = reader.execute("SELECT probe_vec FROM semantic_meta").fetchone()
    count_row = reader.execute("SELECT COUNT(*) FROM project_embeddings").fetchone()
    assert probe_row is not None and count_row is not None
    assert probe_row[0] is None
    assert count_row[0] == 1
    resolver.close()


def test_mid_batch_failure_leaves_index_and_source_intact(tmp_path, monkeypatch):
    """A generation run that fails partway writes nothing: the old index survives."""
    _patch_config(monkeypatch, "service-a")
    client = _FailingBatchClient([1.0, 0.0], "service-a", fail_after=1)
    db_path = tmp_path / "s.duckdb"
    resolver = SemanticResolver(db_path=db_path, embedder=client, read_only=False)
    conn = resolver._get_connection()
    conn.execute(
        "CREATE TABLE project_resources (project_id VARCHAR, report_year INTEGER, "
        "field_name VARCHAR, project_name VARCHAR, pod_name VARCHAR, wk_name VARCHAR, "
        "province VARCHAR, basin128 VARCHAR, project_class VARCHAR, project_stage VARCHAR, "
        "project_level VARCHAR, operator_name VARCHAR, operator_group VARCHAR, "
        "wk_subgroup VARCHAR, wk_regionisasi_ngi VARCHAR, wk_area_perwakilan_skkmigas VARCHAR, "
        "project_remarks TEXT, groovy_report TEXT)"
    )
    assert resolver.build_embeddings_table() is True
    conn.execute(
        "INSERT INTO project_embeddings (project_id, report_year, table_name, project_remarks, "
        "embedding) VALUES (?, ?, ?, ?, ?)",
        ["P0", 2024, "project_resources", "row already indexed", [1.0, 0.0]],
    )
    for i in range(4):
        conn.execute(
            "INSERT INTO project_resources (project_id, report_year, project_remarks) "
            "VALUES (?, ?, ?)",
            [f"P{i + 1}", 2024, f"remarks {i + 1}"],
        )

    out = resolver.generate_and_store_embeddings(batch_size=2)
    resolver.close()

    assert out["status"] == "error", out
    conn = duckdb.connect(str(db_path), read_only=True)
    rows = conn.execute(
        "SELECT project_id, embedding FROM project_embeddings ORDER BY project_id"
    ).fetchall()
    conn.close()
    # Only the row that existed before the run is still in the index.
    assert [row[0] for row in rows] == ["P0"], rows


def test_memo_invalidated_when_the_stored_pin_or_probe_changes_without_replacing_the_file(
    tmp_path,
):
    """A metadata row rewritten underneath a cached client re-validates the space.

    The DB half of the memo key is the file's inode, and `esdc fetch`/`esdc reload`
    publish in place: the same inode can hold a different pin, dimension or probe. So
    the key carries the facts the validation was computed from, read from the row, and
    the entry is recorded as the state stands after a successful check. Request counts
    are the proof; nothing here looks inside the cache.
    """
    db_path = _pinned_space(tmp_path, "space-a", db_name="in-place.duckdb")
    client = _CountingEmbedder([1.0, 0.0], "space-a")
    # Writable so the rewrite underneath comes from a second connection with the same
    # configuration this process already has open; the memo behaviour is the same in
    # either mode, and the resolver still never writes the pin itself.
    reader = SemanticResolver(db_path=db_path, embedder=client, read_only=False)

    out = reader.search_by_text("first query", limit=5)
    assert out["status"] == "success", out
    assert [text for kind, text in client.calls if kind == "probe"] == [
        embedders.PROBE_TEXT
    ]
    inode = os.stat(db_path).st_ino

    # Same file, same inode: the pin and the stored probe were rewritten underneath the
    # cached client, and the new probe still agrees, so the next query checks again.
    conn = duckdb.connect(str(db_path))
    conn.execute(
        "UPDATE semantic_meta SET embedding_model = ?, probe_vec = ?",
        ["space-b", json.dumps([0.999, 0.001])],
    )
    conn.close()

    out = reader.search_by_text("second query", limit=5)
    assert out["status"] == "success", out
    assert os.stat(db_path).st_ino == inode, "the row was changed by replacing the file"
    assert [text for kind, text in client.calls if kind == "probe"] == [
        embedders.PROBE_TEXT,
        embedders.PROBE_TEXT,
    ]

    # The recorded key is the state after the successful check, so operations on that
    # state do not pay for the probe again.
    reader.search_by_text("third query", limit=5)
    assert [text for kind, text in client.calls if kind == "probe"] == [
        embedders.PROBE_TEXT,
        embedders.PROBE_TEXT,
    ]

    # A probe that no longer agrees is caught the same way, and a check that did not
    # succeed is not remembered as verified: each refused read checks again.
    conn = duckdb.connect(str(db_path))
    conn.execute("UPDATE semantic_meta SET probe_vec = ?", [json.dumps([0.0, 1.0])])
    conn.close()

    out = reader.search_by_text("fourth query", limit=5)
    assert out["status"] == "not_available", out
    assert "space-b" in out["message"]
    assert [text for kind, text in client.calls if kind == "probe"] == [
        embedders.PROBE_TEXT,
        embedders.PROBE_TEXT,
        embedders.PROBE_TEXT,
    ]

    reader.search_by_text("fifth query", limit=5)
    assert [text for kind, text in client.calls if kind == "probe"] == [
        embedders.PROBE_TEXT,
        embedders.PROBE_TEXT,
        embedders.PROBE_TEXT,
        embedders.PROBE_TEXT,
    ]
    reader.close()


class _MemoEmbedder:
    """Stable client identity, like an HTTP client built from the same config."""

    model = "qwen3-embedding-0.6b-q8_0"

    def __init__(self, vec, cache_key=("semantic-test",)):
        self._vec = list(vec)
        self.cache_key = cache_key
        self.calls = 0

    def generate_embedding(self, text):
        self.calls += 1
        return list(self._vec)

    def generate_embeddings_batch(self, texts):
        return [list(self._vec) for _ in texts]


def _meta_db(tmp_path, embedder, *, content_version=None, include_version=True):
    db_path = tmp_path / "memo.duckdb"
    columns = "embedding_model VARCHAR, dim INTEGER, probe_vec JSON"
    if include_version:
        columns += ", content_version INTEGER"
    conn = duckdb.connect(str(db_path))
    conn.execute(f"CREATE TABLE semantic_meta ({columns})")
    values = [
        embedder.model,
        len(embedder._vec),
        json.dumps(list(embedder._vec)),
    ]
    if include_version:
        values.append(
            SemanticResolver.NARRATIVE_CONTENT_VERSION
            if content_version is None
            else content_version
        )
    placeholders = ", ".join("?" for _ in values)
    conn.execute(
        f"INSERT INTO semantic_meta VALUES ({placeholders})",
        values,
    )
    conn.close()
    return db_path


def test_semantic_content_version_checked_after_memo_hit(tmp_path):
    embedders.clear_probe_memo()
    try:
        client = _MemoEmbedder([1.0, 0.0])
        db_path = _meta_db(tmp_path, client)
        resolver = SemanticResolver(db_path=db_path, embedder=client)
        assert resolver._ensure_semantic_meta() is None
        assert client.calls == 1
        resolver.close()

        conn = duckdb.connect(str(db_path))
        conn.execute(
            "UPDATE semantic_meta SET content_version = ?",
            [SemanticResolver.NARRATIVE_CONTENT_VERSION - 1],
        )
        conn.close()

        resolver = SemanticResolver(db_path=db_path, embedder=client)
        out = resolver._ensure_semantic_meta()
        resolver.close()

        assert out is not None
        assert out["status"] == "not_available"
        assert "current project narrative" in out["message"]
        # The version guard fires before any remote parity probe.
        assert client.calls == 1
    finally:
        embedders.clear_probe_memo()


def test_semantic_missing_version_column_rejected_after_warm_memo(tmp_path):
    embedders.clear_probe_memo()
    try:
        client = _MemoEmbedder([1.0, 0.0])
        db_path = _meta_db(tmp_path, client)
        resolver = SemanticResolver(db_path=db_path, embedder=client)
        assert resolver._ensure_semantic_meta() is None
        resolver.close()

        conn = duckdb.connect(str(db_path))
        conn.execute("DROP TABLE semantic_meta")
        conn.execute(
            "CREATE TABLE semantic_meta "
            "(embedding_model VARCHAR, dim INTEGER, probe_vec JSON)"
        )
        conn.execute(
            "INSERT INTO semantic_meta VALUES (?, ?, ?)",
            [client.model, 2, json.dumps([1.0, 0.0])],
        )
        conn.close()

        resolver = SemanticResolver(db_path=db_path, embedder=client)
        out = resolver._ensure_semantic_meta()
        present = resolver._get_connection().execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = 'semantic_meta'"
        ).fetchall()
        resolver.close()

        assert out is not None
        assert out["status"] == "not_available"
        assert "esdc reload --embeddings-only" in out["message"]
        assert "content_version" not in {row[0] for row in present}
    finally:
        embedders.clear_probe_memo()


def test_unchanged_semantic_state_reuses_one_probe(tmp_path):
    embedders.clear_probe_memo()
    try:
        first = _MemoEmbedder([1.0, 0.0])
        db_path = _meta_db(tmp_path, first)
        resolver = SemanticResolver(db_path=db_path, embedder=first)
        assert resolver._ensure_semantic_meta() is None
        assert resolver._ensure_semantic_meta() is None
        resolver.close()

        second = _MemoEmbedder([1.0, 0.0])
        fresh = SemanticResolver(db_path=db_path, embedder=second)
        assert fresh._ensure_semantic_meta() is None
        fresh.close()

        assert first.calls == 1
        assert second.calls == 0
    finally:
        embedders.clear_probe_memo()


def test_failed_semantic_validation_never_populates_success_memo(tmp_path):
    embedders.clear_probe_memo()
    try:
        client = _MemoEmbedder([1.0, 0.0])
        db_path = _meta_db(tmp_path, client)

        conn = duckdb.connect(str(db_path))
        conn.execute(
            "UPDATE semantic_meta SET probe_vec = ?",
            [json.dumps([0.0, 1.0])],
        )
        conn.close()

        resolver = SemanticResolver(db_path=db_path, embedder=client)
        out = resolver._ensure_semantic_meta()
        resolver.close()
        assert out is not None
        assert out["status"] == "not_available"
        assert client.calls == 1

        conn = duckdb.connect(str(db_path))
        conn.execute(
            "UPDATE semantic_meta SET probe_vec = ?",
            [json.dumps([1.0, 0.0])],
        )
        conn.close()

        resolver = SemanticResolver(db_path=db_path, embedder=client)
        assert resolver._ensure_semantic_meta() is None
        resolver.close()
        assert client.calls == 2
    finally:
        embedders.clear_probe_memo()
