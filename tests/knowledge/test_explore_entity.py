from __future__ import annotations

import json
from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def _clear_tool_cache():
    from esdc.chat.tools import invalidate_tool_cache

    invalidate_tool_cache()
    yield
    invalidate_tool_cache()


def _invoke(entity, entity_type=None):
    from esdc.chat.tools import explore_entity

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


def test_tool_registered_in_agent_defaults():
    import esdc.chat.agent as agent_mod

    with open(agent_mod.__file__) as f:
        src = f.read()
    assert "explore_entity" in src
