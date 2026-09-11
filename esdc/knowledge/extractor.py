"""Guideline-driven LLM extraction for a single document.

Output is raw (entity names as written in the doc). Resolution to registry
ids happens in esdc.knowledge.resolver.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from esdc.knowledge.guideline import Guideline, build_extraction_prompt
from esdc.llm_text import has_degenerate_repetition, strip_thinking_tags

logger = logging.getLogger(__name__)

_RETRY_SUFFIX = """

Your previous response was rejected. Return a single valid JSON object only.
Start with { and end with }. No prose, markdown fences, or repeated items.
Escape quote characters inside JSON strings.
"""


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
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        path = dump_dir / f"{stamp}-{doc_id}-{reason}.txt"
        if len(raw) <= 64_000:
            body = raw
            truncated = False
        else:
            body = raw[:48_000] + "\n\n... RESPONSE ELIDED ...\n\n" + raw[-16_000:]
            truncated = True
        path.write_text(
            f"# reason: {reason}\n"
            f"# doc_id: {doc_id}\n"
            f"# response_len: {len(raw)}\n"
            f"# response_truncated: {str(truncated).lower()}\n"
            f"# has_think_open: {'<think' in raw.lower()}\n"
            f"# has_think_close: {'</think' in raw.lower()}\n"
            f"# brace_open: {raw.count('{')} brace_close: {raw.count('}')}\n"
            f"{'-' * 70}\n{body}",
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
    *,
    max_attempts: int = 2,
) -> ExtractionResult:
    prompt = build_extraction_prompt(guideline, doc_meta, markdown)
    reason = "no_json_object"
    raw = ""
    for attempt in range(max_attempts):
        # Provider-level failure is handled by fallback in the caller; do not
        # catch or retry llm_caller exceptions (one timeout window per call).
        raw = llm_caller(prompt)

        if not has_degenerate_repetition(raw):
            # Strip reasoning blocks before matching, to prevent greedy {.*}
            # from matching from a brace inside <think>…</think> to the
            # closing brace of the real JSON. Keep original raw for all
            # diagnostics.
            candidate = strip_thinking_tags(raw)
            match = re.search(r"\{.*\}", candidate, re.DOTALL)
            if match:
                try:
                    parsed_dict = json.loads(match.group(0))
                except json.JSONDecodeError:
                    reason = "json_decode_error"
                else:
                    return ExtractionResult(
                        entities=_clean_entities(
                            parsed_dict.get("entities"),
                            set(guideline.entity_types),
                        ),
                        claims=_clean_claims(
                            parsed_dict.get("claims"), set(guideline.claim_types)
                        ),
                        unknown_types=_clean_unknowns(parsed_dict.get("unknown_types")),
                    )
            else:
                reason = "no_json_object"
        else:
            reason = "repetition_loop"

        if attempt < max_attempts - 1:
            prompt = prompt + _RETRY_SUFFIX
            logger.warning("[Extract] %s, retrying %s", reason, doc_meta.get("doc_id"))

    dump = _dump_raw_response(raw, doc_meta, reason)
    raise ValueError(
        f"LLM response contained no valid JSON: {raw[:100]}"
        + (f" | raw dumped to {dump}" if dump else "")
    )
