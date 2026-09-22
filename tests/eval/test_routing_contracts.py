"""Red contracts for route-aware capability and skill selection."""

import json

import pytest

from esdc.chat.query_classifier import route_query
from esdc.chat.smart_query import simple_data_query
from esdc.chat.tools import (
    aggregate_documents,
    entity_resolver,
    execute_sql,
    knowledge_traversal,
    resolve_spatial,
    search_documents,
    semantic_search,
)


@pytest.mark.parametrize(
    ("query", "required", "excluded", "skills", "knowledge"),
    [
        (
            "Bolehkah E3 ke E4?",
            {"Knowledge Traversal"},
            {"Semantic Search", "Document Search"},
            {"ksmi_reasoning"},
            {("transition", "E3")},
        ),
        (
            "Bagaimana transisi E3 ke E4 pada proyek WK Rokan tahun 2025?",
            {"Knowledge Traversal", "SQL Executor"},
            {"Semantic Search", "Document Search"},
            {"ksmi_reasoning"},
            {("transition", "E3")},
        ),
        (
            "Apa isi GROOVY dan kendala proyek di WK Rokan?",
            {"Semantic Search", "SQL Executor"},
            {"Knowledge Traversal", "Document Search"},
            {"project_narrative_analysis"},
            set(),
        ),
        (
            "Apa formula GROOVY untuk level E3?",
            {"Knowledge Traversal"},
            {"Semantic Search", "Document Search", "SQL Executor"},
            {"ksmi_reasoning"},
            {("document", "GROOVY")},
        ),
        (
            "Berapa cadangan minyak Lapangan Duri tahun 2025?",
            {"Simple Data Query"},
            {"Knowledge Traversal", "Document Search", "Spatial Resolver"},
            {"resource_sql"},
            set(),
        ),
    ],
)
def test_route_query_selects_only_required_capabilities(
    query: str,
    required: set[str],
    excluded: set[str],
    skills: set[str],
    knowledge: set[tuple[str, str]],
) -> None:
    decision = route_query(query)
    assert required <= set(decision.allowed_tools)
    assert not excluded & set(decision.allowed_tools)
    assert set(decision.skill_names) == skills
    assert {(r.topic, r.entity) for r in decision.required_knowledge} == knowledge


def _serialize_tool_schema(tool: object) -> str:
    """Return the measured LLM-facing schema payload for one selected tool."""
    args_schema = getattr(tool, "args_schema", None)
    schema = args_schema.model_json_schema() if args_schema is not None else {}
    return json.dumps(
        {
            "name": getattr(tool, "name", ""),
            "description": getattr(tool, "description", ""),
            "args_schema": schema,
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def test_simple_factual_route_schema_baseline(capsys: pytest.CaptureFixture[str]) -> None:
    """Measure serialized tool schemas before per-route limits are introduced."""
    decision = route_query("Berapa cadangan minyak Lapangan Duri tahun 2025?")
    assert len(decision.allowed_tools) <= 2

    tools = {
        tool.name: tool
        for tool in (
            aggregate_documents,
            entity_resolver,
            execute_sql,
            knowledge_traversal,
            resolve_spatial,
            search_documents,
            semantic_search,
            simple_data_query,
        )
    }
    serialized = [
        _serialize_tool_schema(tools[name])
        for name in decision.allowed_tools
        if name in tools
    ]
    total_chars = sum(map(len, serialized))
    print(
        "simple factual route schema chars:",
        [len(item) for item in serialized],
        "total:",
        total_chars,
    )
    captured = capsys.readouterr()
    assert "simple factual route schema chars:" in captured.out


def test_visualization_is_additive_to_the_spatial_route() -> None:
    ordinary = route_query("Which fields are within a 10 km radius of Duri?")
    visualization = route_query(
        "Show a map overlay of fields within a 10 km radius of Duri."
    )

    assert ordinary.visualization is False
    assert visualization.visualization is True
    # The visualization flag is additive: the route keeps its spatial/data
    # capability set; configured rendering tools are added at binding time
    # (covered by tests/test_agent_single_system_message.py).
    assert set(visualization.allowed_tools) == set(ordinary.allowed_tools)
    assert {"Spatial Resolver", "SQL Executor"} <= set(visualization.allowed_tools)
    assert visualization.skill_names == ordinary.skill_names == ("spatial_analysis",)
