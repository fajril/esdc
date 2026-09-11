"""Unit tests for deterministic chat-skill selection."""

from esdc.chat.skills import (
    Skill,
    discover_skills,
    select_skills,
    skill_matches_query,
)


def _strategic_skill(triggers: list[str] | None = None) -> Skill:
    return Skill(
        name="strategic_analysis",
        description="test strategic skill",
        triggers=(
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
