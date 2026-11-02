"""Tests for entity dossiers."""

from __future__ import annotations

import json
import sqlite3

import duckdb
import pytest

from esdc.knowledge.dossier import (
    ensure_dossier_table,
    gather_pod_context,
    generate_pod_dossier,
    get_dossier,
    pod_source_hash,
)
from esdc.knowledge.freshness import (
    compute_knowledge_basis,
    knowledge_freshness,
    mark_knowledge_current,
)
from esdc.knowledge.guideline import load_guideline
from esdc.knowledge.linker import run_deterministic_linking
from esdc.knowledge.store import Claim, KnowledgeStore
from tests.knowledge.conftest import POD_DURI, POD_DURI_REV1, PROJECT_DURI

POD = POD_DURI

# The reader re-derives the hash from the packaged guideline (the guideline
# isolation fixture guarantees it), so generation must use the same hash for the
# cache to be served.
GUIDELINE_HASH = load_guideline().content_hash

_DOSSIER_CACHE_KEYS = (
    "entity_type",
    "entity_id",
    "entity_name",
    "dossier_text",
    "source_hash",
    "provider",
    "model",
    "generated_at",
)


def _prepared(sqlite_conn, duck_conn):
    store = KnowledgeStore(sqlite_conn)
    store.ensure_tables()
    run_deterministic_linking(sqlite_conn, duck_conn, store)
    store.replace_claims(
        "DOC-B",
        [
            Claim(
                "DOC-B",
                "issue",
                "delay_cause",
                "rig availability",
                subject_type="pod",
                subject_id=POD,
                evidence="q",
            )
        ],
    )
    ensure_dossier_table(duck_conn)
    return store


def _cached_row(duck_conn: duckdb.DuckDBPyConnection) -> tuple[object, ...] | None:
    row = duck_conn.execute(
        "SELECT "
        + ", ".join(_DOSSIER_CACHE_KEYS)
        + " FROM knowledge_dossiers WHERE entity_type = 'pod' AND entity_id = ?",
        [POD],
    ).fetchone()
    return None if row is None else tuple(row)


def _pod_seq(sqlite_conn: sqlite3.Connection, pod_id: str) -> int:
    row = sqlite_conn.execute(
        "SELECT pod_id_seq FROM m_pod WHERE pod_id = ?", (pod_id,)
    ).fetchone()
    assert row is not None
    return int(row["pod_id_seq"])


def _link_document(sqlite_conn: sqlite3.Connection, pod_id: str, doc_id: str) -> None:
    """Record a verified document link, then re-mark knowledge current for it.

    A source-row insert invalidates the knowledge basis, so readers refuse to
    serve until the write boundary re-marked it; the tests must model that same
    boundary instead of bypassing the guard.
    """
    sqlite_conn.execute("BEGIN IMMEDIATE")
    sqlite_conn.execute(
        "INSERT INTO pod_document (pod_id_seq, doc_id) VALUES (?, ?)",
        (_pod_seq(sqlite_conn, pod_id), doc_id),
    )
    assert mark_knowledge_current(sqlite_conn, compute_knowledge_basis(sqlite_conn))
    sqlite_conn.commit()


@pytest.mark.parametrize(
    "source_context", ["no_source", "source_without_current_registry"]
)
def test_pod_dossier_cache_refuses_to_serve_without_current_source(
    source_context: str,
) -> None:
    """A cached POD dossier is derived data; without its current source it is not served.

    The cache row itself and the supplied source connection must both stay
    untouched: the boundary is a read gate, not a write or a fallback.
    """
    duck_conn = duckdb.connect(":memory:")
    ensure_dossier_table(duck_conn)
    duck_conn.execute(
        "INSERT INTO knowledge_dossiers VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        [
            "pod",
            POD,
            "POD I Duri",
            "cached dossier text",
            "fixed-source-hash",
            "fake-provider",
            "fake-model",
            "2024-01-01T00:00:00+00:00",
        ],
    )
    cached_before = _cached_row(duck_conn)
    assert cached_before is not None

    sqlite_context: sqlite3.Connection | None = None
    if source_context == "source_without_current_registry":
        sqlite_context = sqlite3.connect(":memory:")
        sqlite_context.row_factory = sqlite3.Row
        assert sqlite_context.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='pod_registry_state'"
        ).fetchone() is None

    try:
        assert get_dossier(duck_conn, "pod", POD, sqlite_conn=sqlite_context) is None
        # Source untouched: nothing created or written there, and no transaction
        # left open on a connection the caller owns.
        if sqlite_context is not None:
            assert sqlite_context.execute(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table'"
            ).fetchone()[0] == 0
            assert not sqlite_context.in_transaction
        # Cache untouched: the row is still readable through the raw cache.
        assert _cached_row(duck_conn) == cached_before
    finally:
        if sqlite_context is not None:
            sqlite_context.close()
        duck_conn.close()


def test_pod_dossier_serves_with_current_source_and_generic_cache_without_one(
    sqlite_conn, duck_conn
) -> None:
    """A current source context serves the POD cache; generic rows are unchanged."""
    store = _prepared(sqlite_conn, duck_conn)
    generate_pod_dossier(POD, sqlite_conn, duck_conn, store, GUIDELINE_HASH, lambda _p: "dossier")

    pod_dossier = get_dossier(duck_conn, "pod", POD, sqlite_conn=sqlite_conn)
    assert pod_dossier is not None
    assert pod_dossier["entity_type"] == "pod"

    # Generic knowledge dossiers keep their existing behaviour: they are not POD
    # derived data and are served without a POD source context.
    ensure_dossier_table(duck_conn)
    duck_conn.execute(
        "INSERT INTO knowledge_dossiers VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        [
            "document",
            "DOC-B",
            "MoM Monitoring POD I Duri",
            "generic dossier",
            "generic-hash",
            "fake-provider",
            "fake-model",
            "2024-01-01T00:00:00+00:00",
        ],
    )
    generic = get_dossier(duck_conn, "document", "DOC-B")
    assert generic is not None
    assert generic["dossier_text"] == "generic dossier"


def test_current_only_pod_gate_leaves_non_pod_entities_on_the_generic_path(
    sqlite_conn, duck_conn
) -> None:
    """A stale POD source refuses the POD dossier but not a generic cache row.

    The gate is source validation for derived POD data only; a document dossier is
    not POD-derived and keeps its existing cache behaviour. Refusing must not
    re-mark knowledge current as a side effect, and must leave the connection the
    caller owns idle.
    """
    store = _prepared(sqlite_conn, duck_conn)
    assert (
        generate_pod_dossier(
            POD, sqlite_conn, duck_conn, store, GUIDELINE_HASH, lambda _p: "pod dossier"
        )
        == "built"
    )
    ensure_dossier_table(duck_conn)
    duck_conn.execute(
        "INSERT INTO knowledge_dossiers VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        [
            "document",
            "DOC-C",
            "Pedoman Insentif Hulu Migas",
            "generic dossier",
            "generic-hash",
            "fake-provider",
            "fake-model",
            "2024-01-01T00:00:00+00:00",
        ],
    )

    # A committed source change invalidates the basis without re-marking it.
    sqlite_conn.execute(
        "UPDATE documents SET subject = 'Changed subject' WHERE doc_id = 'DOC-C'"
    )
    sqlite_conn.commit()
    before = knowledge_freshness(sqlite_conn)
    assert not before.fresh

    assert get_dossier(duck_conn, "pod", POD, sqlite_conn=sqlite_conn) is None
    generic = get_dossier(duck_conn, "document", "DOC-C", sqlite_conn=sqlite_conn)
    assert generic is not None
    assert generic["dossier_text"] == "generic dossier"

    # Nothing was written on the caller's behalf: the stored marker still disagrees
    # with the live basis, so the refusal is a read gate and not a re-mark.
    after = knowledge_freshness(sqlite_conn)
    assert after.stored_basis_sha256 == before.stored_basis_sha256
    assert after.basis_sha256 != after.stored_basis_sha256
    assert not sqlite_conn.in_transaction


def test_malformed_current_metadata_is_a_cache_miss_not_an_error(
    sqlite_conn, duck_conn
) -> None:
    """Current metadata that cannot be read is an incompatible source: no serve.

    The read boundary must treat a source it cannot validate as unavailable -- a
    cache miss -- rather than raising or serving last-good cache. The prior cache
    row and the source itself stay untouched, and the caller's connection is left
    idle.
    """
    store = _prepared(sqlite_conn, duck_conn)
    assert (
        generate_pod_dossier(
            POD, sqlite_conn, duck_conn, store, GUIDELINE_HASH, lambda _p: "dossier"
        )
        == "built"
    )
    cached_before = _cached_row(duck_conn)
    assert cached_before is not None
    assert get_dossier(duck_conn, "pod", POD, sqlite_conn=sqlite_conn) is not None

    # Malformed current metadata on an otherwise canonical current source.
    sqlite_conn.execute(
        "ALTER TABLE pod_registry_state RENAME COLUMN schema_version TO broken_version"
    )
    sqlite_conn.commit()
    # The rename does not change the knowledge basis, so the refusal below can only
    # come from the source-validation boundary, not from staleness.
    assert knowledge_freshness(sqlite_conn).fresh

    assert get_dossier(duck_conn, "pod", POD, sqlite_conn=sqlite_conn) is None

    # Prior cache untouched, source not repaired or written, marker not re-marked,
    # and no transaction left open on a connection the caller owns.
    assert _cached_row(duck_conn) == cached_before
    state = sqlite_conn.execute(
        "SELECT broken_version, contract_id, contract_version, dictionary_version "
        "FROM pod_registry_state WHERE singleton_id = 1"
    ).fetchone()
    assert state is not None
    assert state[1] == "esdc.pod"
    assert knowledge_freshness(sqlite_conn).fresh
    assert not sqlite_conn.in_transaction


def test_gather_pod_context_joins_all_sources(sqlite_conn, duck_conn):
    store = _prepared(sqlite_conn, duck_conn)
    ctx = gather_pod_context(POD, sqlite_conn, duck_conn, store)
    assert ctx["pod"]["pod_name"] == "POD I Duri"
    assert ctx["revisions"] == [POD_DURI_REV1]
    # m_project is the project authority; the latest-year raw row supplies remarks.
    assert ctx["projects"][0]["project_id"] == PROJECT_DURI
    assert ctx["projects"][0]["project_name"] == "Duri Steamflood"
    assert ctx["projects"][0]["report_year"] == 2024
    assert ctx["projects"][0]["project_remarks"] == (
        "Water handling constraint ongoing"
    )
    assert any(c.predicate == "delay_cause" for c in ctx["claims"])


def test_suggested_pod_ids_are_review_only_and_create_no_document_edge(
    sqlite_conn, duck_conn
):
    """DOC-B carries a suggested id, but suggestions are never accepted links."""
    store = _prepared(sqlite_conn, duck_conn)
    ctx = gather_pod_context(POD, sqlite_conn, duck_conn, store)
    assert ctx["documents"] == []
    assert not [
        e
        for e in store.edges_for("document", "DOC-B")
        if e.rel == "ABOUT_POD"
    ]


def test_verified_document_link_surfaces_document_as_explicit_edge(
    sqlite_conn, duck_conn
):
    """A protected pod_document row is the only source of a document edge."""
    store = _prepared(sqlite_conn, duck_conn)
    _link_document(sqlite_conn, POD, "DOC-B")
    run_deterministic_linking(sqlite_conn, duck_conn, store)
    ctx = gather_pod_context(POD, sqlite_conn, duck_conn, store)
    assert [d["doc_id"] for d in ctx["documents"]] == ["DOC-B"]
    assert ctx["documents"][0]["link_method"] == "pod_document_explicit"
    assert ctx["documents"][0]["excerpt"]


def test_revision_edge_is_derived_from_the_current_pointer(sqlite_conn, duck_conn):
    """The superseded_by pointer is the revision contract; there is no revision table."""
    store = _prepared(sqlite_conn, duck_conn)
    edges = [e for e in store.edges_for("pod", POD) if e.rel in ("REVISES", "SUPERSEDES")]
    assert edges
    for edge in edges:
        assert edge.dst_id == POD
        assert edge.src_id == POD_DURI_REV1
        assert edge.valid_from == "2022-01-01"
        properties = json.loads(edge.properties_json or "{}")
        assert properties["source"] == "m_pod.superseded_by"
        assert properties["revision_effect"] == "full_replacement"
    assert not sqlite_conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='pod_revision'"
    ).fetchone()


def test_project_master_without_resource_row_is_reported_without_fabricated_values(
    sqlite_conn, duck_conn
):
    """An inactive or resource-less master still appears, with NULL year and remarks."""
    store = _prepared(sqlite_conn, duck_conn)
    sqlite_conn.execute("BEGIN IMMEDIATE")
    sqlite_conn.execute(
        "INSERT INTO m_project (project_id, project_name, is_active, created_by, updated_by) "
        "VALUES (?, ?, 0, ?, ?)",
        ("P-TEST003-01", "Idle Project", "FA:test-knowledge", "FA:test-knowledge"),
    )
    sqlite_conn.execute(
        "INSERT INTO pod_project (pod_project_uuid, pod_id_seq, project_id, "
        "is_main_project, is_auto_inferred, created_by, updated_by) "
        "VALUES (?, ?, ?, 1, 0, ?, ?)",
        (
            "uuid-test003",
            _pod_seq(sqlite_conn, POD),
            "P-TEST003-01",
            "FA:test-knowledge",
            "FA:test-knowledge",
        ),
    )
    assert mark_knowledge_current(sqlite_conn, compute_knowledge_basis(sqlite_conn))
    sqlite_conn.commit()
    run_deterministic_linking(sqlite_conn, duck_conn, store)

    ctx = gather_pod_context(POD, sqlite_conn, duck_conn, store)
    idle = [p for p in ctx["projects"] if p["project_id"] == "P-TEST003-01"]
    assert len(idle) == 1
    assert idle[0]["project_name"] == "Idle Project"
    assert idle[0]["report_year"] is None
    assert idle[0]["project_remarks"] is None


def test_source_hash_stable_and_sensitive(sqlite_conn, duck_conn):
    store = _prepared(sqlite_conn, duck_conn)
    ctx = gather_pod_context(POD, sqlite_conn, duck_conn, store)
    h1 = pod_source_hash(ctx, GUIDELINE_HASH)
    h2 = pod_source_hash(
        gather_pod_context(POD, sqlite_conn, duck_conn, store), GUIDELINE_HASH
    )
    assert h1 == h2
    assert pod_source_hash(ctx, "other-guideline") != h1


def test_generate_skips_when_hash_matches_and_force_rebuilds(sqlite_conn, duck_conn):
    store = _prepared(sqlite_conn, duck_conn)
    calls = []

    def llm(prompt):
        calls.append(prompt)
        return "## Approval & Revisions\nApproved 2019 [DOC-B]"

    status1 = generate_pod_dossier(
        POD,
        sqlite_conn,
        duck_conn,
        store,
        GUIDELINE_HASH,
        llm,
        provider="ollama",
        model="m",
    )
    status2 = generate_pod_dossier(POD, sqlite_conn, duck_conn, store, GUIDELINE_HASH, llm)
    status3 = generate_pod_dossier(
        POD, sqlite_conn, duck_conn, store, GUIDELINE_HASH, llm, force=True
    )
    assert (status1, status2, status3) == ("built", "skipped", "built")
    assert len(calls) == 2
    dossier = get_dossier(duck_conn, "pod", POD, sqlite_conn=sqlite_conn)
    assert dossier is not None
    assert "Approved 2019" in dossier["dossier_text"]
    assert dossier["model"] == "m"


def test_prompt_contains_context(sqlite_conn, duck_conn):
    store = _prepared(sqlite_conn, duck_conn)
    _link_document(sqlite_conn, POD, "DOC-B")
    run_deterministic_linking(sqlite_conn, duck_conn, store)
    seen = {}

    def llm(prompt):
        seen["prompt"] = prompt
        return "dossier"

    generate_pod_dossier(POD, sqlite_conn, duck_conn, store, GUIDELINE_HASH, llm)
    p = seen["prompt"]
    assert "POD I Duri" in p
    assert "Water handling constraint ongoing" in p
    assert "delay_cause" in p
    assert "MoM Monitoring POD I Duri" in p
    assert "[doc_id]" in p or "cite" in p.lower()


def test_generate_pod_dossier_strips_reasoning_prose(sqlite_conn, duck_conn):
    """A reasoning-model response must not be persisted verbatim.

    Pre-fix, `text = llm_caller(...)` was stored as-is: `dossier_text` would
    still contain the `<think>...</think>` block and its prose. Because
    `source_hash` gating skips regeneration on an unchanged doc set, that
    pollution would survive indefinitely without `--force`. This asserts
    the stored text is exactly the post-tag content, matching what
    `strip_thinking_tags` + `.strip()` would produce.
    """
    store = _prepared(sqlite_conn, duck_conn)

    def llm(prompt):
        return "<think>Let me plan the dossier sections first.</think>\n## Approval"

    generate_pod_dossier(POD, sqlite_conn, duck_conn, store, GUIDELINE_HASH, llm)
    dossier = get_dossier(duck_conn, "pod", POD, sqlite_conn=sqlite_conn)
    assert dossier is not None
    assert dossier["dossier_text"] == "## Approval"
    assert "<think>" not in dossier["dossier_text"]
    assert "plan the dossier" not in dossier["dossier_text"]


def test_failed_dossier_insert_preserves_previous_cache(sqlite_conn, duck_conn):
    from typing import cast

    import duckdb
    import pytest

    store = _prepared(sqlite_conn, duck_conn)
    assert (
        generate_pod_dossier(
            POD, sqlite_conn, duck_conn, store, GUIDELINE_HASH, lambda _prompt: "last good dossier"
        )
        == "built"
    )

    class FailingInsertDuckConnection:
        def __init__(self, connection):
            self.connection = connection

        def execute(self, query, parameters=None):
            if query.startswith("INSERT INTO knowledge_dossiers VALUES"):
                raise RuntimeError("injected dossier insert failure")
            if parameters is None:
                return self.connection.execute(query)
            return self.connection.execute(query, parameters)

        def __getattr__(self, name):
            return getattr(self.connection, name)

    with pytest.raises(RuntimeError, match="injected dossier insert failure"):
        generate_pod_dossier(
            POD,
            sqlite_conn,
            cast(duckdb.DuckDBPyConnection, FailingInsertDuckConnection(duck_conn)),
            store,
            GUIDELINE_HASH,
            lambda _prompt: "replacement dossier",
            force=True,
        )

    dossier = get_dossier(duck_conn, "pod", POD, sqlite_conn=sqlite_conn)
    assert dossier is not None
    assert dossier["dossier_text"] == "last good dossier"


def test_source_hash_covers_prompt_metadata_and_typed_claim_fields(
    sqlite_conn, duck_conn
):
    from dataclasses import replace

    store = _prepared(sqlite_conn, duck_conn)
    _link_document(sqlite_conn, POD, "DOC-B")
    run_deterministic_linking(sqlite_conn, duck_conn, store)
    context = gather_pod_context(POD, sqlite_conn, duck_conn, store)
    assert context["documents"]
    assert context["projects"]
    assert context["claims"]
    original_hash = pod_source_hash(context, GUIDELINE_HASH)

    changed_pod = {
        **context,
        "pod": {**context["pod"], "pod_letter_num": "LETTER-CHANGED"},
    }
    changed_document = {
        **context,
        "documents": [{**context["documents"][0], "subject": "Changed subject"}],
    }
    changed_project = {
        **context,
        "projects": [{**context["projects"][0], "project_name": "Changed project"}],
    }
    changed_claim = {
        **context,
        "claims": [
            replace(context["claims"][0], evidence="Changed source quote"),
            *context["claims"][1:],
        ],
    }

    assert pod_source_hash(changed_pod, GUIDELINE_HASH) != original_hash
    assert pod_source_hash(changed_document, GUIDELINE_HASH) != original_hash
    assert pod_source_hash(changed_project, GUIDELINE_HASH) != original_hash
    assert pod_source_hash(changed_claim, GUIDELINE_HASH) != original_hash
