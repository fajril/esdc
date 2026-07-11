"""Pre-fill document metadata via local LLM."""

import json
import re
from collections.abc import Callable
from typing import Any

DOC_TYPES = ("surat", "mom", "ba", "other")
DOC_LEVELS = ("wk", "field", "project", "unknown")
MAX_PROMPT_CHARS = 8000

METADATA_PROMPT = """You extract metadata from Indonesian oil & gas official documents.
Given the markdown of a document, return ONLY a JSON object with these keys
(use null when unknown, never guess):
- doc_type: "surat" (official letter) | "mom" (minutes of meeting)
  | "ba" (berita acara) | "other"
- doc_number: the document/letter number exactly as written
- doc_date: ISO date YYYY-MM-DD
- subject: perihal or meeting title
- sender: issuing organization or signatory org
- recipient: addressed organization (letters only)
- doc_level: "wk" | "field" | "project" | "unknown" — the scope this document is about
- wk_name: list of working area (wilayah kerja) names mentioned
  (e.g. ["Rokan", "Mahakam"])
- field_name: list of field (lapangan) names mentioned (e.g. ["Duri", "Minas"])
- project_name: list of project or POD names mentioned (e.g. ["POD Duri", "POD Minas"])
- extras: object with doc_type-specific fields, e.g. for mom:
  {{"peserta": [...], "keputusan": [...]}}

Document markdown:
---
{markdown}
---
JSON:"""

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
    """Clamp LLM output to the allowed vocabulary."""
    out = dict(parsed)
    if out.get("doc_type") not in DOC_TYPES:
        out["doc_type"] = "other"
    if out.get("doc_level") not in DOC_LEVELS:
        out["doc_level"] = "unknown"
    return out


ENTITY_FIELDS = ("wk_name", "field_name", "project_name")


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

