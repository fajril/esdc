from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class Skill:
    name: str
    description: str
    triggers: list[str]
    instructions: str | None = None


_SKILLS_DIR = Path(__file__).parent


def _load_skill(skill_dir: Path) -> Skill | None:
    yaml_path = skill_dir / "skill.yaml"
    if not yaml_path.exists():
        return None

    with open(yaml_path) as f:
        meta: dict = yaml.safe_load(f) or {}

    instructions: str | None = None
    instructions_rel = meta.get("instructions")
    if instructions_rel:
        instructions_path = skill_dir / instructions_rel
        if instructions_path.exists():
            instructions = instructions_path.read_text(encoding="utf-8")

    return Skill(
        name=meta.get("name", skill_dir.name),
        description=meta.get("description", ""),
        triggers=meta.get("triggers", []),
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
    """Append skill instructions to the base system prompt."""
    parts = [base_prompt]
    for skill in skills:
        if skill.instructions:
            parts.append("")
            parts.append(skill.instructions)
    return "\n".join(parts)
