"""Guideline-driven LLM extraction for a single document.

Output is raw (entity names as written in the doc). Resolution to registry
ids happens in esdc.knowledge.resolver.
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from esdc.knowledge.guideline import Guideline, build_extraction_prompt
from esdc.llm_text import strip_thinking_tags

logger = logging.getLogger(__name__)


def _dump_raw_response(raw: str, doc_meta: dict[str, Any], reason: str) -> str | None:
    """Write an unparseable LLM response to disk; return the path, or None.

    The exception message truncates to 100 chars, which is not enough to tell a
    reasoning-prefixed response (JSON present, wrong slice taken) from one where
    the model never emitted JSON at all. Diagnostics must never break the run,
    so every failure here is swallowed.
    """
    try:
        from esdc.configs import Config

        dump_dir = Config.get_cache_dir() / "extract_failures"
        dump_dir.mkdir(parents=True, exist_ok=True)
        doc_id = str(doc_meta.get("doc_id") or "unknown")
        path = dump_dir / f"{time.strftime('%Y%m%d-%H%M%S')}-{doc_id}.txt"
        path.write_text(
            f"# reason: {reason}\n"
            f"# doc_id: {doc_id}\n"
            f"# response_len: {len(raw)}\n"
            f"# has_think_open: {'<think' in raw.lower()}\n"
            f"# has_think_close: {'</think' in raw.lower()}\n"
            f"# brace_open: {raw.count('{')} brace_close: {raw.count('}')}\n"
            f"{'-' * 70}\n{raw}",
            encoding="utf-8",
        )
        return str(path)
    except Exception:
        logger.debug("[Extract] raw_dump_failed", exc_info=True)
        return None


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

    # Strip reasoning blocks before matching, to prevent greedy {.*} from
    # matching from a brace inside <think>…</think> to the closing brace
    # of the real JSON. Keep original raw for all diagnostics.
    candidate = strip_thinking_tags(raw)

    # Validate that the response contains valid JSON before parsing.
    # This ensures we raise ValueError for both:
    # 1. No {..} object found at all
    # 2. {..} found but fails to parse (unquoted keys, trailing commas, etc.)
    match = re.search(r"\{.*\}", candidate, re.DOTALL)
    if not match:
        dump = _dump_raw_response(raw, doc_meta, "no_json_object")
        raise ValueError(
            f"LLM response contained no valid JSON: {raw[:100]}"
            + (f" | raw dumped to {dump}" if dump else "")
        )

    try:
        parsed_dict = json.loads(match.group(0))
    except json.JSONDecodeError as e:
        dump = _dump_raw_response(raw, doc_meta, "json_decode_error")
        raise ValueError(
            f"LLM response contained invalid JSON: {raw[:100]}"
            + (f" | raw dumped to {dump}" if dump else "")
        ) from e

    return ExtractionResult(
        entities=_clean_entities(
            parsed_dict.get("entities"), set(guideline.entity_types)
        ),
        claims=_clean_claims(parsed_dict.get("claims"), set(guideline.claim_types)),
        unknown_types=_clean_unknowns(parsed_dict.get("unknown_types")),
    )
