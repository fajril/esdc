"""Pre-fill document metadata via local LLM."""

import json
import re
from collections.abc import Callable
from typing import Any

from esdc.chat.domain_knowledge.doc_schema import (
    doc_level_rules,
    enum_values,
    legacy_doc_type_map,
    legacy_topic_seed,
    render_prompt_definitions,
)

DOC_TYPES = enum_values("doc_type")
DOC_LEVELS = enum_values("doc_level")
DOC_TOPICS = enum_values("doc_topic")
MAX_PROMPT_CHARS = 8000

ENTITY_FIELDS = ("wk_name", "field_name", "project_name")

_PROMPT_SKELETON = """You extract metadata from Indonesian oil & gas official documents.
Given the markdown of a document, return ONLY a JSON object with these keys
(use null when unknown, never guess):
{definitions}

Document markdown:
---
{{markdown}}
---
JSON:"""

METADATA_PROMPT = _PROMPT_SKELETON.format(definitions=render_prompt_definitions())

# Same schema, phrased for the image path (GLM-OCR on the first page).
# Build it from the shared field list so the two prompts can't drift.
METADATA_PROMPT_IMAGE = METADATA_PROMPT.split("Document markdown:")[0] + (
    "Extract from this first page of the document. JSON:"
)


def parse_llm_json(raw: str) -> dict[str, Any]:
    """Parse LLM output into a dict; tolerate code fences and chatter."""
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return {}
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def normalize_metadata(parsed: dict[str, Any]) -> dict[str, Any]:
    """Clamp LLM output to the allowed vocabulary and apply deterministic rules.

    Order (each stage sees the previous stage's output):
    1. Legacy doc_type remap (e.g. "surat" -> "letter", "psc" -> "contract")
       + doc_topic seed for legacy form/topic splits (e.g. "psc" -> topic
       "psc") — only seeds doc_topic when it isn't already set.
    2. doc_type clamp to the allowed vocabulary (else "others").
    3. doc_topic normalize to a deduped list clamped to the allowed
       vocabulary (else "others" per entry), or None.
    4. doc_level clamp to the allowed vocabulary (else "unknown").
    5. Deterministic level rules: a doc_type rule (e.g. "uu" -> regulation)
       takes precedence over a doc_topic rule; a doc_type rule that implies
       "regulation" also nulls wk_name/field_name/project_name. A doc_topic
       rule only applies when every topic present implies the same level.
    """
    out = dict(parsed)

    # --- doc_type: legacy remap (case-insensitive) + topic seed ---
    raw_type = out.get("doc_type")
    # Vocab codes are acronyms (UU, PSC, ...) the LLM often capitalizes —
    # compare lowercase so casing never demotes a valid type to "others".
    if isinstance(raw_type, str):
        raw_type = raw_type.lower()
    canon_type = legacy_doc_type_map().get(raw_type, raw_type)
    seeded_topic = legacy_topic_seed().get(raw_type)
    out["doc_type"] = canon_type if canon_type in DOC_TYPES else "others"

    # --- doc_topic: normalize to a deduped, clamped list (or None) ---
    raw_topic = out.get("doc_topic")
    if raw_topic is None and seeded_topic is not None:
        raw_topic = [seeded_topic]
    if raw_topic is None:
        out["doc_topic"] = None
    else:
        topic_list = raw_topic if isinstance(raw_topic, list) else [raw_topic]
        clamped_topics: list[str] = []
        for t in topic_list:
            t = t.lower() if isinstance(t, str) else t
            t = t if t in DOC_TOPICS else "others"
            if t not in clamped_topics:
                clamped_topics.append(t)
        out["doc_topic"] = clamped_topics

    # --- doc_level: clamp ---
    raw_level = out.get("doc_level")
    if isinstance(raw_level, str):
        raw_level = raw_level.lower()
    out["doc_level"] = raw_level if raw_level in DOC_LEVELS else "unknown"

    # --- deterministic level rules: doc_type rule beats doc_topic rule ---
    rules = doc_level_rules()
    type_rule = rules.get("doc_type", {}).get(out["doc_type"])
    if type_rule is not None:
        out["doc_level"] = type_rule
        if type_rule == "regulation":
            for key in ENTITY_FIELDS:
                out[key] = None
    else:
        topic_rule_map = rules.get("doc_topic", {})
        topics = out.get("doc_topic") or []
        implied = {topic_rule_map[t] for t in topics if t in topic_rule_map}
        if len(implied) == 1:
            out["doc_level"] = implied.pop()

    return out


def normalize_entity_fields(meta: dict[str, Any]) -> dict[str, Any]:
    """Convert scalar entity fields to lists for backward compatibility.

    Handles:
    - None → None
    - "Rokan" → ["Rokan"]
    - ["Rokan", "Mahakam"] → ["Rokan", "Mahakam"] (already a list)
    """
    out = dict(meta)
    for key in ENTITY_FIELDS:
        val = out.get(key)
        if val is None:
            continue
        if isinstance(val, str):
            out[key] = [val]
        elif isinstance(val, list):
            # Already a list — keep as-is
            pass
        else:
            # Unexpected type — wrap in list
            out[key] = [str(val)]
    return out


def llm_extract(markdown: str, llm_caller: Callable[[str], str]) -> dict[str, Any]:
    """Metadata candidates for the sidecar. llm_caller: prompt -> raw response."""
    prompt = METADATA_PROMPT.format(markdown=markdown[:MAX_PROMPT_CHARS])
    return normalize_metadata(parse_llm_json(llm_caller(prompt)))

