"""Tests for entity exploration."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest
from langchain_core.tools import StructuredTool


@pytest.fixture(autouse=True)
def _clear_tool_cache(monkeypatch):
    """Reset the module-level tool results cache handle between tests.

    Config.get_cache_dir() isolation from the real ~/.esdc/cache is handled
    globally by the `_isolate_tool_cache_dir` autouse fixture in
    tests/knowledge/conftest.py. This fixture only drops any
    already-initialized module-level cache handle (which may point at a
    directory from a prior test) so it gets recreated against the
    currently-patched tmp dir, then clears it before/after each test.
    """
    import esdc.chat.tools as tools_mod

    monkeypatch.setattr(tools_mod, "_tool_cache", None)

    from esdc.chat.tools import invalidate_tool_cache

    invalidate_tool_cache()
    yield
    invalidate_tool_cache()


def _invoke(entity, entity_type=None):
    from esdc.chat.tools import explore_entity

    assert isinstance(explore_entity, StructuredTool)
    assert explore_entity.func is not None
    return json.loads(explore_entity.func(entity=entity, entity_type=entity_type))


@patch("esdc.chat.tools._get_instance_graph")
def test_not_available_before_learn(mock_graph):
    mock_graph.return_value.is_available.return_value = False
    out = _invoke("POD Duri")
    assert out["status"] == "not_available"
    assert "esdc corpus learn" in out["message"]


@patch("esdc.chat.tools._get_knowledge_context")
@patch("esdc.chat.tools._get_instance_graph")
def test_success_bundles_dossier_neighbors_claims(mock_graph, mock_ctx):
    mgr = mock_graph.return_value
    mgr.is_available.return_value = True
    mgr.find.return_value = [
        {"entity_type": "pod", "entity_id": "PL-1", "name": "POD I Duri", "score": 3.2}
    ]
    mgr.neighbors.return_value = {
        "HAS_PROJECT": [
            {
                "entity_type": "project",
                "entity_id": "PRJ-001",
                "name": "Duri Steamflood",
                "direction": "outbound",
            }
        ]
    }
    mock_ctx.return_value = (
        {"dossier_text": "## Approval\nstuff"},
        [
            {
                "doc_id": "DOC-B",
                "type": "issue",
                "predicate": "delay_cause",
                "value": "rig",
                "evidence": "q",
            }
        ],
    )
    out = _invoke("POD I Duri", entity_type="pod")
    assert out["status"] == "success"
    assert out["entity"]["entity_id"] == "PL-1"
    assert "## Approval" in out["dossier"]
    assert out["related"]["HAS_PROJECT"][0]["entity_id"] == "PRJ-001"
    assert out["claims"][0]["predicate"] == "delay_cause"


@patch("esdc.chat.tools._get_instance_graph")
def test_not_found(mock_graph):
    mgr = mock_graph.return_value
    mgr.is_available.return_value = True
    mgr.find.return_value = []
    out = _invoke("Atlantis Phase 9")
    assert out["status"] == "not_found"


@patch("esdc.chat.tools._get_knowledge_context")
@patch("esdc.chat.tools._get_instance_graph")
def test_entity_type_uppercase_normalizes_like_lowercase(mock_graph, mock_ctx):
    """entity_type='POD' must resolve exactly like 'pod', not not_found.

    Regression coverage: explore_entity previously passed entity_type
    through to graph.find() unnormalized, so an LLM-provided variant like
    'POD' never matched find()'s (and the safety-net filter's) lowercase
    entity_type values.
    """
    mgr = mock_graph.return_value
    mgr.is_available.return_value = True
    mgr.find.return_value = [
        {"entity_type": "pod", "entity_id": "PL-1", "name": "POD I Duri", "score": 3.2}
    ]
    mgr.neighbors.return_value = {}
    mock_ctx.return_value = (None, [])

    out = _invoke("POD I Duri", entity_type="POD")

    assert out["status"] == "success"
    assert out["entity"]["entity_id"] == "PL-1"
    mgr.find.assert_called_once_with("POD I Duri", top_k=5, entity_type="pod")


@patch("esdc.chat.tools._get_instance_graph")
def test_unknown_entity_type_returns_error(mock_graph):
    out = _invoke("POD I Duri", entity_type="banana")
    assert out["status"] == "error"
    assert "banana" in out["message"]
    for valid in ("pod", "project", "field", "working_area", "document"):
        assert valid in out["message"]
    # Unknown entity_type must fail before ever touching the graph.
    mock_graph.assert_not_called()


def test_project_entity_type_resolves_via_real_graph(sqlite_conn, duck_conn, tmp_path):
    """entity_type='project' must resolve through the real find() filter.

    Regression coverage for Issue #1: explore_entity previously always
    returned not_found for entity_type='project' because the instance
    graph had no Project FTS index. The other explore_entity tests in this
    file mock the graph entirely, so they cannot catch that -- this test
    exercises a real InstanceGraphManager (built from the same
    sqlite/duckdb fixtures as tests/knowledge/test_instance_graph.py) and
    only mocks the dossier lookup (`_get_knowledge_context`), which is
    unrelated to the bug being fixed here.
    """
    from esdc.chat.domain_knowledge.instance_graph import InstanceGraphManager
    from esdc.knowledge.linker import run_deterministic_linking
    from esdc.knowledge.store import KnowledgeStore

    store = KnowledgeStore(sqlite_conn)
    store.ensure_tables()
    run_deterministic_linking(sqlite_conn, duck_conn, store)
    sqlite_conn.commit()
    # duck_conn holds the only read-write connection to the tmp duckdb
    # file; InstanceGraphManager opens its own read-only connection to
    # the same file, which duckdb refuses while a different-config
    # connection is still open (see test_instance_graph.py for the same
    # pattern).
    duck_conn.close()

    real_graph = InstanceGraphManager(
        sqlite_path=tmp_path / "esdc.sqlite",
        duckdb_path=tmp_path / "esdc.duckdb",
    )

    with (
        patch("esdc.chat.tools._get_instance_graph", return_value=real_graph),
        patch("esdc.chat.tools._get_knowledge_context", return_value=(None, [])),
    ):
        out = _invoke("Duri Steamflood", entity_type="project")

    assert out["status"] == "success"
    assert out["entity"]["entity_type"] == "project"
    assert out["entity"]["entity_id"] == "PRJ-001"
    assert out["entity"]["name"] == "Duri Steamflood"


def test_tool_registered_in_agent_defaults():
    import esdc.chat.agent as agent_mod

    with open(agent_mod.__file__) as f:
        src = f.read()
    assert "explore_entity" in src
