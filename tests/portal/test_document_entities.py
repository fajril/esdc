"""Tests for portal document entities."""

import json
from pathlib import Path

import duckdb
import pytest

from esdc.corpus.store import _SQLITE_DOC_DDL
from esdc.knowledge.freshness import invalidate_knowledge_after_dependency_change
from esdc.pod_registry.store import get_sqlite_connection
from esdc.portal.document_entities import ENTITY_FIELDS, apply_document_entity_changeset

pytestmark = pytest.mark.usefixtures("offline_corpus_embedder")


class FakeResolver:
    """Stand-in for EntityResolver — pattern from tests/corpus/test_pipeline.py.

    `matches` keys on (entity_type, raw_name.lower()) -> list of match dicts
    (each needs at least a "name" key); an empty/missing key means 0 matches.
    `suggestions` keys on (entity_type, raw_name) -> list of candidate names.
    """

    def __init__(self, matches=None, suggestions=None):
        self._matches = matches or {}
        self._suggestions = suggestions or {}

    def resolve_name(self, name, entity_type, parent_filter=None):
        return list(self._matches.get((entity_type, name.strip().lower()), []))

    def suggest_names(self, name, entity_type, limit=5):
        return list(self._suggestions.get((entity_type, name), []))[:limit]


def _seed_sqlite(tmp_path: Path, doc_ids=("D1",), extra: dict | None = None) -> Path:
    path = tmp_path / "esdc.sqlite"
    conn = get_sqlite_connection(path)
    conn.execute(_SQLITE_DOC_DDL)
    for doc_id in doc_ids:
        row = {
            "doc_id": doc_id,
            "file_name": f"{doc_id}.pdf",
            "file_path": f"/x/{doc_id}.pdf",
            "file_hash": doc_id * 4,
            "markdown": "# isi",
            "extraction_method": "native",
            "embedding_model": "fake-embed",
        }
        if extra:
            row.update(extra)
        cols = ", ".join(row.keys())
        placeholders = ", ".join("?" for _ in row)
        conn.execute(
            f"INSERT INTO documents ({cols}) VALUES ({placeholders})",
            tuple(row.values()),
        )
    conn.commit()
    conn.close()
    return path


def _make_duckdb_mirror(tmp_path: Path, doc_ids=("D1",), name="mirror.duckdb") -> Path:
    path = tmp_path / name
    conn = duckdb.connect(str(path))
    conn.execute(
        "CREATE TABLE documents (doc_id VARCHAR PRIMARY KEY, wk_name JSON,"
        " field_name JSON, project_name JSON)"
    )
    for doc_id in doc_ids:
        conn.execute("INSERT INTO documents VALUES (?, NULL, NULL, NULL)", [doc_id])
    conn.close()
    return path


def _read_doc(sqlite_path: Path, doc_id: str) -> dict:
    conn = get_sqlite_connection(sqlite_path)
    row = conn.execute(
        "SELECT wk_name, field_name, project_name FROM documents WHERE doc_id = ?",
        (doc_id,),
    ).fetchone()
    conn.close()
    return dict(row)


def test_entity_fields_constant():
    assert ENTITY_FIELDS == ("wk_name", "field_name", "project_name")


def test_update_entities_canonicalizes_and_stores_json_array(tmp_path):
    sqlite_path = _seed_sqlite(tmp_path)
    duckdb_path = _make_duckdb_mirror(tmp_path)
    resolver = FakeResolver(
        matches={
            ("wk_name", "rokan"): [{"name": "Rokan"}],
            ("wk_name", "kampar"): [{"name": "Kampar"}],
        }
    )

    result = apply_document_entity_changeset(
        {"updates": [{"doc_id": "D1", "wk_name": "rokan; kampar"}]},
        sqlite_path=sqlite_path,
        db_path=duckdb_path,
        resolver=resolver,
    )

    assert result.ok is True
    assert result.warnings == []
    assert _read_doc(sqlite_path, "D1")["wk_name"] == json.dumps(["Rokan", "Kampar"])

    mirror = duckdb.connect(str(duckdb_path))
    mirrored_row = mirror.execute(
        "SELECT wk_name FROM documents WHERE doc_id = 'D1'"
    ).fetchone()
    assert mirrored_row is not None
    mirrored = mirrored_row[0]
    mirror.close()
    assert mirrored == json.dumps(["Rokan", "Kampar"])


def test_empty_cell_clears_to_null(tmp_path):
    sqlite_path = _seed_sqlite(tmp_path, extra={"wk_name": '["Rokan"]'})
    resolver = FakeResolver()

    result = apply_document_entity_changeset(
        {"updates": [{"doc_id": "D1", "wk_name": ""}]},
        sqlite_path=sqlite_path,
        db_path=tmp_path / "no-mirror.duckdb",
        resolver=resolver,
    )

    assert result.ok is True
    assert _read_doc(sqlite_path, "D1")["wk_name"] is None


def test_unknown_entity_name_rejected_with_suggestions(tmp_path):
    sqlite_path = _seed_sqlite(tmp_path)
    resolver = FakeResolver(
        suggestions={("wk_name", "Rokann"): ["Rokan", "Rokan Hilir"]}
    )

    result = apply_document_entity_changeset(
        {"updates": [{"doc_id": "D1", "wk_name": "Rokann"}]},
        sqlite_path=sqlite_path,
        db_path=tmp_path / "no-mirror.duckdb",
        resolver=resolver,
    )

    assert result.ok is False
    assert len(result.errors) == 1
    message = result.errors[0].message
    assert "Rokann" in message
    assert "Rokan" in message
    # no unvalidated write happened
    assert _read_doc(sqlite_path, "D1")["wk_name"] is None


def test_inserts_and_deletes_rejected(tmp_path):
    sqlite_path = _seed_sqlite(tmp_path)
    resolver = FakeResolver()

    result = apply_document_entity_changeset(
        {
            "inserts": [{"doc_id": "D2"}],
            "deletes": [{"doc_id": "D1"}],
        },
        sqlite_path=sqlite_path,
        db_path=tmp_path / "no-mirror.duckdb",
        resolver=resolver,
    )

    assert result.ok is False
    kinds = {e.kind for e in result.errors}
    assert kinds == {"insert", "delete"}


def test_unknown_doc_id_rejected(tmp_path):
    sqlite_path = _seed_sqlite(tmp_path)
    resolver = FakeResolver(matches={("wk_name", "rokan"): [{"name": "Rokan"}]})

    result = apply_document_entity_changeset(
        {"updates": [{"doc_id": "nope", "wk_name": "rokan"}]},
        sqlite_path=sqlite_path,
        db_path=tmp_path / "no-mirror.duckdb",
        resolver=resolver,
    )

    assert result.ok is False
    assert "nope" in result.errors[0].message


def test_non_entity_field_rejected(tmp_path):
    sqlite_path = _seed_sqlite(tmp_path)
    resolver = FakeResolver()

    result = apply_document_entity_changeset(
        {"updates": [{"doc_id": "D1", "subject": "new subject"}]},
        sqlite_path=sqlite_path,
        db_path=tmp_path / "no-mirror.duckdb",
        resolver=resolver,
    )

    assert result.ok is False
    assert "subject" in result.errors[0].message


def test_resolver_unavailable_rejects_save(tmp_path):
    sqlite_path = _seed_sqlite(tmp_path)
    # No resolver injected, and db_path points at a DuckDB file that does
    # not exist -- read_only=True cannot create it, simulating "locked or
    # missing" without needing to fake a connection object.
    missing_db_path = tmp_path / "does-not-exist.duckdb"

    result = apply_document_entity_changeset(
        {"updates": [{"doc_id": "D1", "wk_name": "rokan"}]},
        sqlite_path=sqlite_path,
        db_path=missing_db_path,
        resolver=None,
    )

    assert result.ok is False
    assert result.errors
    # no unvalidated write happened
    assert _read_doc(sqlite_path, "D1")["wk_name"] is None


def test_default_resolver_and_mirror_share_db_path(tmp_path):
    # Production lifecycle: resolver=None so the module opens its own
    # read-only DuckDB connection for EntityResolver, and the mirror then
    # needs a read-write connection to the SAME file. If the resolver
    # connection is still open, DuckDB rejects the second open ("different
    # configuration") and the mirror silently degrades to a warning — so
    # this asserts the full sequence works with real connections.
    sqlite_path = _seed_sqlite(tmp_path)
    duckdb_path = tmp_path / "esdc.duckdb"
    conn = duckdb.connect(str(duckdb_path))
    # canonical lookup table queried by EntityResolver (all entity specs
    # use project_resources as lookup_table)
    conn.execute("CREATE TABLE project_resources (wk_name VARCHAR, wk_id VARCHAR)")
    conn.execute("INSERT INTO project_resources VALUES ('Rokan', 'WK1')")
    conn.execute(
        "CREATE TABLE documents (doc_id VARCHAR PRIMARY KEY, wk_name JSON,"
        " field_name JSON, project_name JSON)"
    )
    conn.execute("INSERT INTO documents VALUES ('D1', NULL, NULL, NULL)")
    conn.close()

    result = apply_document_entity_changeset(
        {"updates": [{"doc_id": "D1", "wk_name": "rokan"}]},
        sqlite_path=sqlite_path,
        db_path=duckdb_path,
        resolver=None,
    )

    assert result.ok is True
    assert result.warnings == []
    assert _read_doc(sqlite_path, "D1")["wk_name"] == json.dumps(["Rokan"])

    mirror = duckdb.connect(str(duckdb_path))
    mirrored_row = mirror.execute(
        "SELECT wk_name FROM documents WHERE doc_id = 'D1'"
    ).fetchone()
    assert mirrored_row is not None
    mirrored = mirrored_row[0]
    mirror.close()
    assert mirrored == json.dumps(["Rokan"])


def test_update_with_no_entity_fields_is_a_noop(tmp_path):
    # A row carrying only doc_id changes nothing: it must not count in
    # applied["updates"] nor trigger a mirror attempt.
    sqlite_path = _seed_sqlite(tmp_path)
    resolver = FakeResolver()
    # nonexistent mirror file: a mirror attempt would produce a warning
    missing_db_path = tmp_path / "does-not-exist.duckdb"

    result = apply_document_entity_changeset(
        {"updates": [{"doc_id": "D1"}]},
        sqlite_path=sqlite_path,
        db_path=missing_db_path,
        resolver=resolver,
    )

    assert result.ok is True
    assert result.applied["updates"] == 0
    assert result.warnings == []


def test_mirror_failure_returns_warning_but_saves(tmp_path):
    """A refresh that loses the DuckDB lock degrades to a warning, not a failed save.

    `refresh_mirror()` rebuilds `documents` wholesale (`CREATE OR REPLACE
    TABLE`), so a missing table no longer reproduces a mirror failure —
    unlike the old row-by-row UPDATE mirror, it just creates the table.
    Instead, hold a read-only DuckDB connection open on the same file:
    DuckDB refuses a second connection under a different configuration,
    which is exactly what happens if a `esdc corpus commit` (or another
    portal save) still holds the write lock.
    """
    sqlite_path = _seed_sqlite(tmp_path)
    duckdb_path = tmp_path / "locked.duckdb"
    duckdb.connect(str(duckdb_path)).close()
    lock_conn = duckdb.connect(str(duckdb_path), read_only=True)
    resolver = FakeResolver(matches={("wk_name", "rokan"): [{"name": "Rokan"}]})

    try:
        result = apply_document_entity_changeset(
            {"updates": [{"doc_id": "D1", "wk_name": "rokan"}]},
            sqlite_path=sqlite_path,
            db_path=duckdb_path,
            resolver=resolver,
        )
    finally:
        lock_conn.close()

    assert result.ok is True
    assert result.warnings
    assert "esdc corpus sync" in result.warnings[0]
    assert _read_doc(sqlite_path, "D1")["wk_name"] == json.dumps(["Rokan"])


@pytest.mark.parametrize(
    "payload",
    [
        {"updates": [{"doc_id": "D1", "wk_name": "rokan"}, "not-a-row"]},
        {"updates": "not-a-list"},
        {"inserts": "not-a-list"},
        {"deletes": None},
        {"updates": [{"doc_id": "D1", "wk_name": "rokan", "source_generation": 7}]},
        {"updates": [{f"wk_name{i}": "x"} for i in range(501)]},
    ],
)
def test_malformed_payload_rejected_before_db(tmp_path, payload):
    """Structural validation runs before any resolver or DB access."""
    sqlite_path = _seed_sqlite(tmp_path)
    resolver = FakeResolver(matches={("wk_name", "rokan"): [{"name": "Rokan"}]})

    result = apply_document_entity_changeset(
        payload,
        sqlite_path=sqlite_path,
        db_path=tmp_path / "no-mirror.duckdb",
        resolver=resolver,
    )

    assert result.ok is False
    assert result.errors
    assert result.source_committed is False
    # no unvalidated write happened
    assert _read_doc(sqlite_path, "D1")["wk_name"] is None


def test_successful_save_never_loads_embedding_models(monkeypatch, tmp_path):
    """Ordinary saves do no automatic LLM/embedding work."""
    from esdc.corpus.pipeline import run_reembed_documents

    monkeypatch.setattr(
        "esdc.corpus.pipeline.run_reembed_documents",
        lambda *a, **k: pytest.fail("reembed"),
    )
    assert run_reembed_documents is not None

    sqlite_path = _seed_sqlite(tmp_path)
    duckdb_path = _make_duckdb_mirror(tmp_path)
    resolver = FakeResolver(matches={("wk_name", "rokan"): [{"name": "Rokan"}]})

    result = apply_document_entity_changeset(
        {"updates": [{"doc_id": "D1", "wk_name": "rokan"}]},
        sqlite_path=sqlite_path,
        db_path=duckdb_path,
        resolver=resolver,
    )

    assert result.ok is True
    # successful mirror publication -> no reminders at all
    assert result.warnings == []
    assert result.source_committed is True
    assert result.source_generation == 0
    assert _read_doc(sqlite_path, "D1")["wk_name"] == json.dumps(["Rokan"])

    mirror = duckdb.connect(str(duckdb_path))
    mirrored_row = mirror.execute(
        "SELECT wk_name FROM documents WHERE doc_id = 'D1'"
    ).fetchone()
    mirror.close()
    assert mirrored_row is not None
    assert mirrored_row[0] == json.dumps(["Rokan"])


def test_mirror_failure_warns_and_reminds_manual_reembed(monkeypatch, tmp_path):
    """A lost DuckDB lock degrades to a warning plus a manual reembed reminder."""
    from esdc.corpus.pipeline import run_reembed_documents

    monkeypatch.setattr(
        "esdc.corpus.pipeline.run_reembed_documents",
        lambda *a, **k: pytest.fail("reembed"),
    )
    assert run_reembed_documents is not None

    sqlite_path = _seed_sqlite(tmp_path)
    duckdb_path = tmp_path / "locked.duckdb"
    duckdb.connect(str(duckdb_path)).close()
    lock_conn = duckdb.connect(str(duckdb_path), read_only=True)
    resolver = FakeResolver(matches={("wk_name", "rokan"): [{"name": "Rokan"}]})

    try:
        result = apply_document_entity_changeset(
            {"updates": [{"doc_id": "D1", "wk_name": "rokan"}]},
            sqlite_path=sqlite_path,
            db_path=duckdb_path,
            resolver=resolver,
        )
    finally:
        lock_conn.close()

    assert result.ok is True
    assert result.source_committed is True
    assert "esdc corpus sync" in result.warnings[0]
    assert any("reembed --stale" in w for w in result.warnings)
    assert _read_doc(sqlite_path, "D1")["wk_name"] == json.dumps(["Rokan"])


def test_knowledge_invalidation_runs_inside_the_source_transaction(
    monkeypatch, tmp_path
):
    """KG invalidation happens in the same transaction as the updates, before commit."""
    from esdc.portal import document_entities

    calls: list[dict] = []
    real_helper = invalidate_knowledge_after_dependency_change

    def _spy(conn, *, actor, force=False):
        # The helper itself requires an open transaction; asserting here proves
        # the call is inside it rather than after the commit.
        assert conn.in_transaction
        calls.append({"actor": actor, "force": force})
        return real_helper(conn, actor=actor, force=force)

    monkeypatch.setattr(
        document_entities,
        "invalidate_knowledge_after_dependency_change",
        _spy,
    )

    sqlite_path = _seed_sqlite(tmp_path)
    duckdb_path = _make_duckdb_mirror(tmp_path)
    resolver = FakeResolver(matches={("wk_name", "rokan"): [{"name": "Rokan"}]})

    result = apply_document_entity_changeset(
        {"updates": [{"doc_id": "D1", "wk_name": "rokan"}]},
        sqlite_path=sqlite_path,
        db_path=duckdb_path,
        resolver=resolver,
    )

    assert result.ok is True
    assert calls == [
        {"actor": "FA:portal-document", "force": True},
    ]
    # Metadata edits leave the current POD source generation unchanged.
    assert result.source_generation == 0
    assert _read_doc(sqlite_path, "D1")["wk_name"] == json.dumps(["Rokan"])
