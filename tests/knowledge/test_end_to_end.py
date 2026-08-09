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

import pytest
from langchain_core.tools import StructuredTool

from esdc.chat.domain_knowledge.instance_graph import InstanceGraphManager
from esdc.knowledge.learn import run_learn

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
    # both linked pods get dossiers (DOC-A -> rev1 pod, DOC-B -> base pod)
    assert report.dossiers_built >= 2

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
    assert graph.is_available() is True
    hits = graph.find("Duri", top_k=10)
    assert any(h["entity_type"] == "pod" for h in hits)

    from esdc.chat.tools import explore_entity, invalidate_tool_cache

    invalidate_tool_cache()

    assert isinstance(explore_entity, StructuredTool)
    assert explore_entity.func is not None
    out = json.loads(explore_entity.func(entity="POD Duri", entity_type=None))
    assert out["status"] == "success"
    assert out["entity"]["entity_type"] == "pod"
    assert out["dossier"]
    assert "Stub dossier for smoke test" in out["dossier"]
    assert isinstance(out["related"], dict)
    assert isinstance(out["claims"], list)

    # A name that resolves to nothing must be a clean not_found, never an
    # exception -- the brief's "bad name" expectation.
    assert explore_entity.func is not None
    out_missing = json.loads(
        explore_entity.func(
            entity="Zzyzx Nonexistent Entity 999888777", entity_type=None
        )
    )
    assert out_missing["status"] == "not_found"
