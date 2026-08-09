"""Tests for entity dossiers."""

from __future__ import annotations

from esdc.knowledge.dossier import (
    ensure_dossier_table,
    gather_pod_context,
    generate_pod_dossier,
    get_dossier,
    pod_source_hash,
)
from esdc.knowledge.linker import run_deterministic_linking
from esdc.knowledge.store import Claim, KnowledgeStore

POD = "PL-2019-0001-2-2-0"


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


def test_gather_pod_context_joins_all_sources(sqlite_conn, duck_conn):
    store = _prepared(sqlite_conn, duck_conn)
    ctx = gather_pod_context(POD, sqlite_conn, duck_conn, store)
    assert ctx["pod"]["pod_name"] == "POD I Duri"
    assert "PL-2022-0002-2-2-1" in ctx["revisions"]
    doc_ids = [d["doc_id"] for d in ctx["documents"]]
    assert "DOC-B" in doc_ids
    assert ctx["documents"][0]["excerpt"]
    assert any(c.predicate == "delay_cause" for c in ctx["claims"])
    assert ctx["projects"][0]["project_id"] == "PRJ-001"
    assert ctx["projects"][0]["project_remarks"] == (
        "Water handling constraint ongoing"
    )


def test_source_hash_stable_and_sensitive(sqlite_conn, duck_conn):
    store = _prepared(sqlite_conn, duck_conn)
    ctx = gather_pod_context(POD, sqlite_conn, duck_conn, store)
    h1 = pod_source_hash(ctx, "guideline-hash")
    h2 = pod_source_hash(
        gather_pod_context(POD, sqlite_conn, duck_conn, store), "guideline-hash"
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
        "gh",
        llm,
        provider="ollama",
        model="m",
    )
    status2 = generate_pod_dossier(POD, sqlite_conn, duck_conn, store, "gh", llm)
    status3 = generate_pod_dossier(
        POD, sqlite_conn, duck_conn, store, "gh", llm, force=True
    )
    assert (status1, status2, status3) == ("built", "skipped", "built")
    assert len(calls) == 2
    dossier = get_dossier(duck_conn, "pod", POD)
    assert dossier is not None
    assert "Approved 2019" in dossier["dossier_text"]
    assert dossier["model"] == "m"


def test_prompt_contains_context(sqlite_conn, duck_conn):
    store = _prepared(sqlite_conn, duck_conn)
    seen = {}

    def llm(prompt):
        seen["prompt"] = prompt
        return "dossier"

    generate_pod_dossier(POD, sqlite_conn, duck_conn, store, "gh", llm)
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

    generate_pod_dossier(POD, sqlite_conn, duck_conn, store, "gh", llm)
    dossier = get_dossier(duck_conn, "pod", POD)
    assert dossier is not None
    assert dossier["dossier_text"] == "## Approval"
    assert "<think>" not in dossier["dossier_text"]
    assert "plan the dossier" not in dossier["dossier_text"]
