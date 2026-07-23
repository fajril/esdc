"""Guideline-driven LLM extraction for a single document.

Output is raw (entity names as written in the doc). Resolution to registry
ids happens in esdc.knowledge.resolver.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from esdc.knowledge.guideline import Guideline, build_extraction_prompt


@dataclass
class ExtractionResult:
    entities: list[dict[str, str]] = field(default_factory=list)
    claims: list[dict[str, Any]] = field(default_factory=list)
    unknown_types: list[dict[str, str]] = field(default_factory=list)


def _clean_entities(raw: Any, valid_types: set[str]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    if not isinstance(raw, list):
        return out
    for item in raw:
        if not isinstance(item, dict):
            continue
        etype = item.get("type")
        name = item.get("name")
        if etype in valid_types and isinstance(name, str) and name.strip():
            out.append({"type": etype, "name": name.strip()})
    return out


def _clean_claims(raw: Any, valid_types: set[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not isinstance(raw, list):
        return out
    for item in raw:
        if not isinstance(item, dict):
            continue
        ctype = item.get("type")
        predicate = item.get("predicate")
        if ctype not in valid_types:
            continue
        if not isinstance(predicate, str) or not predicate.strip():
            continue
        if "value" not in item:
            continue
        out.append(
            {
                "type": ctype,
                "subject": item.get("subject"),
                "subject_type": item.get("subject_type"),
                "predicate": predicate.strip(),
                "value": item.get("value"),
                "evidence": item.get("evidence"),
            }
        )
    return out


def _clean_unknowns(raw: Any) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    if not isinstance(raw, list):
        return out
    for item in raw:
        if not isinstance(item, dict):
            continue
        kind = item.get("kind")
        name = item.get("name")
        if kind in ("claim_type", "entity_type") and isinstance(name, str):
            out.append(
                {"kind": kind, "name": name.strip(), "why": str(item.get("why") or "")}
            )
    return out


def extract_knowledge(
    markdown: str,
    doc_meta: dict[str, Any],
    guideline: Guideline,
    llm_caller: Callable[[str], str],
) -> ExtractionResult:
    prompt = build_extraction_prompt(guideline, doc_meta, markdown)
    raw = llm_caller(prompt)

    # Validate that the response contains valid JSON before parsing.
    # This ensures we raise ValueError for both:
    # 1. No {..} object found at all
    # 2. {..} found but fails to parse (unquoted keys, trailing commas, etc.)
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        raise ValueError(f"LLM response contained no valid JSON: {raw[:100]}")

    try:
        parsed_dict = json.loads(match.group(0))
    except json.JSONDecodeError as e:
        raise ValueError(f"LLM response contained invalid JSON: {raw[:100]}") from e

    if not isinstance(parsed_dict, dict):
        raise ValueError(f"LLM response JSON is not an object: {raw[:100]}")

    return ExtractionResult(
        entities=_clean_entities(
            parsed_dict.get("entities"), set(guideline.entity_types)
        ),
        claims=_clean_claims(parsed_dict.get("claims"), set(guideline.claim_types)),
        unknown_types=_clean_unknowns(parsed_dict.get("unknown_types")),
    )
