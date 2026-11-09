"""Red contracts for route-aware capability and skill selection."""

import pytest

from esdc.chat.query_classifier import route_query


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


def test_simple_factual_route_selects_at_most_two_tools() -> None:
    """The simple factual route exposes at most two tools to the model."""
    decision = route_query("Berapa cadangan minyak Lapangan Duri tahun 2025?")
    assert len(decision.allowed_tools) <= 2


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
