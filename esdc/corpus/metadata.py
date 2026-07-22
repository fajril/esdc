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


def metadata_image_prompt(filename: str | None = None) -> str:
    """First-page-image metadata prompt, optionally prefixed with a filename hint."""
    if filename:
        return (
            f"Filename (may hint doc_type/date/subject): {filename}\n\n"
            f"{METADATA_PROMPT_IMAGE}"
        )
    return METADATA_PROMPT_IMAGE


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


def remap_legacy_doc_type(value: Any) -> Any:
    """Remap a legacy doc_type alias (case-insensitive) to its replacement.

    Values that aren't a recognized legacy alias pass through completely
    unchanged (including non-strings, already-canonical values, unrecognized
    garbage, and None) — unlike ``normalize_doc_type``, this never invents
    or clamps a value. Used by callers (e.g. ``esdc corpus meta``) that must
    not silently mutate a field the caller didn't touch.
    """
    if not isinstance(value, str):
        return value
    return legacy_doc_type_map().get(value.lower(), value)


def normalize_doc_type(value: Any) -> str | None:
    """Canonicalize a doc_type value: legacy remap + case-fold + vocab clamp.

    ``None`` passes through unchanged — callers that want a real doc_type
    invented for a field the sidecar simply hasn't been touched on yet
    (e.g. ``esdc corpus meta``) should not call this on a ``None`` value.
    """
    if value is None:
        return None
    # Vocab codes are acronyms (UU, PSC, ...) the LLM often capitalizes —
    # compare lowercase so casing never demotes a valid type to "others".
    v = value.lower() if isinstance(value, str) else value
    v = legacy_doc_type_map().get(v, v)
    return v if v in DOC_TYPES else "others"


def seed_topic_from_legacy(raw_doc_type: Any) -> str | None:
    """The doc_topic a legacy doc_type value seeds (e.g. "psc" -> "psc"), if any."""
    v = raw_doc_type.lower() if isinstance(raw_doc_type, str) else raw_doc_type
    return legacy_topic_seed().get(v)


def normalize_topic(value: Any) -> list[str] | None:
    """Normalize a doc_topic value to a deduped, vocab-clamped list, or None.

    Handles a scalar string, an existing list, or None (passthrough).
    """
    if value is None:
        return None
    topic_list = value if isinstance(value, list) else [value]
    clamped: list[str] = []
    for t in topic_list:
        t = t.lower() if isinstance(t, str) else t
        t = t if t in DOC_TOPICS else "others"
        if t not in clamped:
            clamped.append(t)
    return clamped


def doc_level_rule(doc_type: Any, doc_topic: Any) -> tuple[str, str, str] | None:
    """The doc_level rule that applies to this doc_type/doc_topic, if any.

    Returns ``(kind, triggering_value, implied_level)`` where ``kind`` is
    ``"doc_type"`` or ``"doc_topic"``. The doc_type rule takes precedence;
    the doc_topic rule only applies when every topic present implies the
    same level (picks one triggering topic for the message).
    """
    rules = doc_level_rules()
    type_rule = rules.get("doc_type", {}).get(doc_type)
    if type_rule is not None:
        return ("doc_type", doc_type, type_rule)

    topic_rule_map = rules.get("doc_topic", {})
    topics = doc_topic or []
    implied = {topic_rule_map[t] for t in topics if t in topic_rule_map}
    if len(implied) == 1:
        level = next(iter(implied))
        trigger = next(t for t in topics if topic_rule_map.get(t) == level)
        return ("doc_topic", trigger, level)
    return None


def apply_doc_level_rules(meta: dict[str, Any]) -> dict[str, Any]:
    """Apply the deterministic doc_level rule implied by doc_type/doc_topic.

    Does not clamp doc_type/doc_topic/doc_level to the allowed vocab first
    (see ``normalize_metadata`` for the full clamp-then-rule pipeline) — this
    only overwrites doc_level when a rule actually fires, and nulls
    wk_name/field_name/project_name when the fired rule implies "regulation".
    """
    out = dict(meta)
    rule = doc_level_rule(out.get("doc_type"), out.get("doc_topic"))
    if rule is not None:
        _, _, level = rule
        out["doc_level"] = level
        if level == "regulation":
            for key in ENTITY_FIELDS:
                out[key] = None
    return out


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

    raw_type = out.get("doc_type")
    seeded_topic = seed_topic_from_legacy(raw_type)
    out["doc_type"] = normalize_doc_type(raw_type)

    raw_topic = out.get("doc_topic")
    if raw_topic is None and seeded_topic is not None:
        raw_topic = [seeded_topic]
    out["doc_topic"] = normalize_topic(raw_topic)

    raw_level = out.get("doc_level")
    if isinstance(raw_level, str):
        raw_level = raw_level.lower()
    out["doc_level"] = raw_level if raw_level in DOC_LEVELS else "unknown"

    return apply_doc_level_rules(out)


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


def llm_extract(
    markdown: str,
    llm_caller: Callable[[str], str],
    filename: str | None = None,
) -> dict[str, Any]:
    """Metadata candidates for the sidecar. llm_caller: prompt -> raw response.

    ``filename`` is an optional hint prepended to the prompt (used by
    ``corpus rename``); the model still returns the full metadata dict.
    """
    prompt = METADATA_PROMPT.format(markdown=markdown[:MAX_PROMPT_CHARS])
    if filename:
        prompt = f"Filename (may hint doc_type/date/subject): {filename}\n\n{prompt}"
    return normalize_metadata(parse_llm_json(llm_caller(prompt)))

