"""Unit tests for deterministic chat-skill selection."""

from esdc.chat.query_classifier import route_query
from esdc.chat.skills import (
    Skill,
    discover_skills,
    select_skills,
    select_skills_for_route,
    skill_matches_query,
)


def _strategic_skill(triggers: list[str] | None = None) -> Skill:
    return Skill(
        name="strategic_analysis",
        description="test strategic skill",
        triggers=tuple(
            triggers
            if triggers is not None
            else ["analisis strategis", "exploration highlight"]
        ),
        instructions="# Strategic Analysis Skill\n",
    )


def test_select_skills_returns_no_skill_for_greeting() -> None:
    assert select_skills("halo", [_strategic_skill()]) == []


def test_select_skills_returns_no_skill_for_ordinary_resource_lookup() -> None:
    query = "berapa potensi lapangan dan cadangan minyak tahun 2025?"
    assert select_skills(query, [_strategic_skill()]) == []


def test_select_skills_returns_strategic_skill_for_explicit_request() -> None:
    selected = select_skills("Buat ANALISIS STRATEGIS nasional", [_strategic_skill()])
    assert [skill.name for skill in selected] == ["strategic_analysis"]


def test_select_skills_normalizes_whitespace() -> None:
    selected = select_skills("buat  exploration\n highlight", [_strategic_skill()])
    assert [skill.name for skill in selected] == ["strategic_analysis"]


def test_skill_matches_query_does_not_match_inside_word() -> None:
    skill = _strategic_skill(["analisis"])
    assert skill_matches_query(skill, "praanalisis data") is False


def test_skill_matches_query_ignores_empty_trigger() -> None:
    assert skill_matches_query(_strategic_skill([]), "analisis strategis") is False
    assert skill_matches_query(_strategic_skill([""]), "analisis strategis") is False
    assert (
        skill_matches_query(
            _strategic_skill(["", "analisis strategis"]), "analisis strategis"
        )
        is True
    )


def test_repository_strategic_skill_requires_explicit_analysis_request() -> None:
    skills = discover_skills()
    assert select_skills("berapa potensi lapangan Duri?", skills) == []
    selected = select_skills("buat analisis strategis nasional", skills)
    assert [skill.name for skill in selected] == ["strategic_analysis"]


def test_ksmi_skill_does_not_need_literal_trigger() -> None:
    decision = route_query("Bolehkah proyek berpindah dari E3 ke E4?")
    selected = select_skills_for_route(decision, discover_skills())
    assert [skill.name for skill in selected] == ["ksmi_reasoning"]


def test_groovy_content_route_selects_narrative_playbook() -> None:
    decision = route_query("Apa isi GROOVY dan kendala proyek di WK Rokan?")
    selected = select_skills_for_route(decision, discover_skills())
    assert [skill.name for skill in selected] == ["project_narrative_analysis"]


def test_route_selection_rejects_unbound_tool_contract() -> None:
    decision = route_query("berapa cadangan minyak Lapangan Duri tahun 2025?")
    skill = Skill(
        name="resource_sql",
        description="resource workflow",
        query_types=("SIMPLE_FACTUAL",),
        tools=("SQL Executor",),
    )
    assert select_skills_for_route(decision, [skill]) == []


def test_repository_playbooks_have_expected_route_contracts() -> None:
    by_name = {skill.name: skill for skill in discover_skills()}
    assert {"ksmi_reasoning", "resource_sql", "project_narrative_analysis",
            "spatial_analysis", "document_evidence"} <= set(by_name)
    assert by_name["ksmi_reasoning"].tools == ("Knowledge Traversal",)
    assert by_name["document_evidence"].query_types == ("DOCUMENT",)


def test_missing_yaml_lists_are_empty_tuples() -> None:
    skill = Skill(name="minimal", description="minimal")
    assert skill.query_types == ()
    assert skill.tools == ()
    assert skill.triggers == ()
