"""Pre-fill document metadata via local LLM; resolve entities to canonical names."""

import difflib
import json
import logging
import re
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

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
- wk_name: working area (wilayah kerja) name mentioned, raw text
- field_name: field (lapangan) name mentioned, raw text
- project_name: project or POD name mentioned, raw text
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


def llm_extract(markdown: str, llm_caller: Callable[[str], str]) -> dict[str, Any]:
    """Metadata candidates for the sidecar. llm_caller: prompt -> raw response."""
    prompt = METADATA_PROMPT.format(markdown=markdown[:MAX_PROMPT_CHARS])
    return normalize_metadata(parse_llm_json(llm_caller(prompt)))


def resolve_entity(
    raw: str | None, canonical: list[str]
) -> tuple[str | None, float]:
    """Map a raw name to a canonical one.

    exact (case-insensitive) -> 1.0
    substring either direction -> 0.85
    difflib fuzzy (cutoff 0.75) -> ratio
    otherwise (None, 0.0) — caller stores NULL and warns, never guesses.
    """
    if not raw or not canonical:
        return None, 0.0
    folded = raw.casefold().strip()
    by_fold = {c.casefold(): c for c in canonical}
    if folded in by_fold:
        return by_fold[folded], 1.0
    for cand_fold, cand in by_fold.items():
        if cand_fold in folded or folded in cand_fold:
            return cand, 0.85
    close = difflib.get_close_matches(folded, list(by_fold), n=1, cutoff=0.75)
    if close:
        ratio = difflib.SequenceMatcher(None, folded, close[0]).ratio()
        return by_fold[close[0]], round(ratio, 2)
    return None, 0.0


def load_canonical_names(conn) -> dict[str, list[str]]:
    """Distinct canonical names from project_resources for entity resolution."""
    names: dict[str, list[str]] = {}
    for key in ("wk_name", "field_name", "project_name"):
        try:
            rows = conn.execute(
                f"SELECT DISTINCT {key} FROM project_resources WHERE {key} IS NOT NULL"
            ).fetchall()
            names[key] = [r[0] for r in rows]
        except Exception as e:  # table may not exist on fresh installs
            logger.warning("[Corpus] cannot load canonical %s: %s", key, e)
            names[key] = []
    return names
