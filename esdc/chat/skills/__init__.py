"""Declarative, route-aware procedural skills for the chat agent."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from esdc.chat.query_classifier import RoutingDecision


@dataclass(frozen=True)
class Skill:
    """A small procedural playbook loaded from a skill directory."""

    name: str
    description: str
    query_types: tuple[str, ...] = ()
    tools: tuple[str, ...] = ()
    triggers: tuple[str, ...] = ()
    instructions: str | None = None


def _normalize_phrase(value: str) -> str:
    return " ".join(value.casefold().split())


def skill_matches_query(skill: Skill, query: str) -> bool:
    """Match legacy literal triggers using word boundaries."""
    normalized_query = _normalize_phrase(query)
    if not normalized_query:
        return False
    for trigger in skill.triggers:
        normalized_trigger = _normalize_phrase(trigger)
        if not normalized_trigger:
            continue
        if re.search(
            rf"(?<!\w){re.escape(normalized_trigger)}(?!\w)",
            normalized_query,
        ):
            return True
    return False


def select_skills(query: str, skills: list[Skill]) -> list[Skill]:
    """Compatibility selector for callers that still have raw query text."""
    return [skill for skill in skills if skill_matches_query(skill, query)]


def select_skills_for_route(
    decision: RoutingDecision, skills: list[Skill]
) -> list[Skill]:
    """Select only route-named skills whose type and tool contracts match."""
    query_type = decision.classification.query_type.name
    requested = set(decision.skill_names)
    allowed_tools = set(decision.allowed_tools)
    selected: list[Skill] = []
    for skill in skills:
        if skill.name not in requested:
            continue
        if skill.query_types and query_type not in skill.query_types:
            continue
        if not set(skill.tools).issubset(allowed_tools):
            continue
        selected.append(skill)
    return selected


_SKILLS_DIR = Path(__file__).parent


def _as_tuple(value: Any) -> tuple[str, ...]:
    """Normalize optional YAML lists; malformed values become empty tuples."""
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(item for item in value if isinstance(item, str) and item)


def _load_skill(skill_dir: Path) -> Skill | None:
    yaml_path = skill_dir / "skill.yaml"
    if not yaml_path.exists():
        return None

    with yaml_path.open(encoding="utf-8") as file:
        loaded = yaml.safe_load(file) or {}
    if not isinstance(loaded, dict):
        return None
    meta: dict[str, Any] = loaded

    instructions: str | None = None
    instructions_rel = meta.get("instructions")
    if isinstance(instructions_rel, str) and instructions_rel:
        instructions_path = skill_dir / instructions_rel
        if instructions_path.exists():
            instructions = instructions_path.read_text(encoding="utf-8")

    name = meta.get("name", skill_dir.name)
    description = meta.get("description", "")
    return Skill(
        name=name if isinstance(name, str) else skill_dir.name,
        description=description if isinstance(description, str) else "",
        query_types=_as_tuple(meta.get("query_types")),
        tools=_as_tuple(meta.get("tools")),
        triggers=_as_tuple(meta.get("triggers")),
        instructions=instructions,
    )


def discover_skills() -> list[Skill]:
    """Scan skills/ subdirectories for skill.yaml files and load each skill."""
    skills: list[Skill] = []
    for entry in sorted(_SKILLS_DIR.iterdir()):
        if not entry.is_dir() or entry.name.startswith("_"):
            continue
        skill = _load_skill(entry)
        if skill is not None:
            skills.append(skill)
    return skills


def inject_skills_into_prompt(base_prompt: str, skills: list[Skill]) -> str:
    """Append selected skill instructions to a system prompt."""
    parts = [base_prompt]
    for skill in skills:
        if skill.instructions:
            parts.extend(("", skill.instructions))
    return "\n".join(parts)
