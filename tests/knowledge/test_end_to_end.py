"""End-to-end smoke test tying the whole `esdc corpus learn` pipeline together.

Deterministic linking -> guideline-driven extraction -> registry resolution
-> dossier synthesis -> LadybugDB instance graph -> the `explore_entity`
chat tool.

Uses injected FAKE llm_caller(s) -- never a real LLM/embedder -- and
tmp_path-backed sqlite/duckdb files (via the shared autouse fixtures in
tests/knowledge/conftest.py, which already isolate Config.get_db_dir and
Config.get_cache_dir). Nothing here touches a real ~/.esdc.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from langchain_core.tools import StructuredTool

from esdc.chat.domain_knowledge.instance_graph import InstanceGraphManager
from esdc.knowledge.freshness import (
    committed_marker_metadata,
    knowledge_freshness,
    knowledge_freshness_markers,
)
from esdc.knowledge.learn import run_learn
from esdc.pod_registry.store import get_sqlite_readonly_connection
from tests.knowledge.conftest import POD_DURI_REV1

EXTRACTION = {
    "entities": [{"type": "pod", "name": "POD I Duri"}],
    "claims": [
        {
            "type": "issue",
            "subject": "POD I Duri",
            "subject_type": "pod",
            "predicate": "delay_cause",
            "value": "rig availability",
            "evidence": "q",
        }
    ],
    "unknown_types": [],
}


def _fake_llm(prompt: str) -> str:
    """Deterministic stand-in for a real model -- no network, no Ollama."""
    if "case file" in prompt:
        return "## Approval & Revisions\nStub dossier for smoke test."
    return json.dumps(EXTRACTION)


def _refusal_context(graph: InstanceGraphManager, expected: Path) -> str:
    """Describe why the graph refused to build, evaluated only on assertion failure.

    Reports the path the graph itself resolved against the path expected from
    the test's temporary database, plus the facts the reader checks: marker-only freshness,
    recomputed basis freshness, marker metadata presence, and the projected edge
    count. Fingerprints appear only as match/no-match; no rows, config, credentials
    or environment are read.
    """
    resolved = getattr(graph, "_sqlite_path", None)
    probe_path: Path = resolved if isinstance(resolved, Path) else expected
    parts = [
        f"resolved_path={resolved}",
        f"expected_path={expected}",
        f"file={probe_path.is_file()}",
    ]
    if probe_path != expected:
        return " | ".join([*parts, "unexpected_source_not_opened"])
    try:
        conn = get_sqlite_readonly_connection(probe_path)
        if conn is None:
            return " | ".join([*parts, "readonly_connection=None"])
        try:
            markers = knowledge_freshness_markers(conn)
            parts.append(
                "marker_fresh="
                f"{markers.fresh} source_generation={markers.source_generation} "
                f"knowledge_generation={markers.knowledge_generation}"
            )
            full = knowledge_freshness(conn)
            parts.append(f"full_fresh={full.fresh}")
            parts.append(
                f"marker_metadata_present={committed_marker_metadata(conn) is not None}"
            )
            edge_table = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='kg_edge'"
            ).fetchone()
            if edge_table is None:
                parts.append("kg_edge_table=absent")
            else:
                edge_rows = conn.execute("SELECT COUNT(*) FROM kg_edge").fetchone()[0]
                parts.append(f"kg_edge_rows={edge_rows}")
        finally:
            conn.close()
    except Exception as exc:
        parts.append(f"context_error={type(exc).__name__}")
    return " | ".join(parts)


@pytest.fixture(autouse=True)
def _reset_instance_graph_singleton():
    InstanceGraphManager.reset_for_tests()
    yield
    InstanceGraphManager.reset_for_tests()


def test_learn_to_explore_entity_pipeline(
    sqlite_conn, duck_conn, tmp_path, monkeypatch
):
    """Real pipeline modules, fake LLM, tmp_path dbs -- no mocked seams.

    Runs the actual orchestrator (phases 1-4), then points the chat-tool
    layer's Config-resolved default paths at the same tmp_path files (it
    resolves its own connections rather than accepting injected ones), and
    exercises the real `InstanceGraphManager` singleton and `explore_entity`
    tool with no mocking of pipeline internals.
    """
    report = run_learn(
        sqlite_conn=sqlite_conn,
        duck_conn=duck_conn,
        llm_caller=_fake_llm,
        progress=False,
    )
    assert report.docs_total == 3
    assert report.docs_processed == 3
    assert report.docs_failed == 0
    assert report.edges_written > 0
    assert report.claims_written > 0
    # Dossier scope follows the canonical document link. Only DOC-A matches the
    # current identity exactly, so exactly one POD has a document and gets a
    # dossier. Retired assertion: `>= 2` assumed DOC-B's suggested_pod_ids became a
    # link; under the current model that column is review-only and never an edge,
    # and no pod_document bridge exists in the fixture (pod_document_added == 0).
    assert report.dossiers_built == 1
    assert report.dossiers_skipped == 0
    assert report.pod_document_added == 0

    # run_learn's store/linker calls already commit internally and it
    # CHECKPOINTs duck_conn, but close both explicitly before the chat-tool
    # layer opens its own connections to the same files below -- DuckDB
    # does not allow a second read-write handle while one is still open.
    sqlite_conn.commit()
    sqlite_conn.close()
    duck_conn.close()

    from esdc.configs import Config

    monkeypatch.setattr(Config, "get_db_dir", classmethod(lambda cls: tmp_path))
    monkeypatch.setattr(
        Config, "get_chat_db_path", classmethod(lambda cls: tmp_path / "esdc.duckdb")
    )

    # Instance graph: built exactly as production does it, via the no-arg
    # singleton resolving its sqlite path through Config.get_db_dir().
    graph = InstanceGraphManager()
    assert graph.is_available() is True, _refusal_context(graph, tmp_path / "esdc.sqlite")
    hits = graph.find("Duri", top_k=10)
    assert any(h["entity_type"] == "pod" for h in hits)

    from esdc.chat.tools import explore_entity, invalidate_tool_cache

    invalidate_tool_cache()

    assert isinstance(explore_entity, StructuredTool)
    assert explore_entity.func is not None
    # The canonical identity name resolves through the real FTS index to the POD
    # that owns the document link, which is the only POD with a dossier. Retired
    # assertion: the old fixture had dossiers for both PODs, so the ambiguous short
    # name "POD Duri" was enough; here it is checked only as a type resolution.
    out = json.loads(explore_entity.func(entity="POD I Duri Revisi 1", entity_type=None))
    assert out["status"] == "success"
    assert out["entity"]["entity_type"] == "pod"
    assert out["entity"]["entity_id"] == POD_DURI_REV1
    assert out["dossier"]
    assert "Stub dossier for smoke test" in out["dossier"]
    assert isinstance(out["related"], dict)
    assert isinstance(out["claims"], list)

    ambiguous = json.loads(explore_entity.func(entity="POD Duri", entity_type=None))
    assert ambiguous["status"] == "success"
    assert ambiguous["entity"]["entity_type"] == "pod"

    # A name that resolves to nothing must be a clean not_found, never an
    # exception -- the brief's "bad name" expectation.
    assert explore_entity.func is not None
    out_missing = json.loads(
        explore_entity.func(
            entity="Zzyzx Nonexistent Entity 999888777", entity_type=None
        )
    )
    assert out_missing["status"] == "not_found"
