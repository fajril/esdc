"""General domain glossary loader and lookup module.

Loads glossary.yaml and provides exact-match term lookup for the
knowledge_traversal tool. glossary.yaml holds general oil & gas /
commercial / PSC terms that are NOT part of the KSMI framework schema
(those live in ksmi_schema.yaml) — for example TBS (Trustee Borrowing
Scheme), which is a financing term, not a KSMI classification concept.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

_SCHEMA_PATH = Path(__file__).parent / "glossary.yaml"


@lru_cache(maxsize=1)
def _load_glossary() -> dict[str, Any]:
    """Load and cache the glossary YAML.

    Never raises: returns an empty dict on any load failure so callers
    can degrade gracefully instead of breaking the knowledge_traversal tool.
    """
    if not _SCHEMA_PATH.exists():
        logger.warning("Glossary file not found: %s", _SCHEMA_PATH)
        return {}
    try:
        with open(_SCHEMA_PATH) as f:
            data = yaml.safe_load(f)
    except Exception:
        logger.warning("Failed to load glossary: %s", _SCHEMA_PATH, exc_info=True)
        return {}
    if not isinstance(data, dict):
        logger.warning("Glossary is empty or malformed: %s", _SCHEMA_PATH)
        return {}
    return data


def _ci_match(haystack: str, needle: str) -> bool:
    """Case-insensitive exact string match."""
    return haystack.lower() == needle.lower()


def _ci_in(needle: str, haystack: list[str]) -> bool:
    """Case-insensitive membership check: needle in haystack."""
    return needle.lower() in [h.lower() for h in haystack]


def _format_entity(key: str, data: dict[str, Any]) -> str:
    """Format a single glossary entity as readable markdown."""
    name = data.get("name", key)
    lines = [f"## {name}"]

    code = data.get("code")
    if code:
        lines.append(f"**Code:** {code}")

    aliases = data.get("aliases")
    if aliases:
        lines.append(f"**Aliases:** {', '.join(str(a) for a in aliases)}")

    definition = data.get("definition")
    if definition:
        lines.append(f"**Definition:** {definition.strip()}")

    key_concepts = data.get("key_concepts")
    if key_concepts:
        lines.append("**Key Concepts:**")
        for kc in key_concepts:
            if isinstance(kc, dict):
                for k, v in kc.items():
                    lines.append(f"  - {k}: {v}")
            else:
                lines.append(f"  - {kc}")

    return "\n".join(lines)


def glossary_lookup(term: str) -> str | None:
    """Look up a term in the general domain glossary via exact match.

    Matches (case-insensitive, after stripping) against each entry's
    top-level key, code, name, or any of its aliases. Matching is exact
    only — no fuzzy/substring matching — so the glossary can never shadow
    KSMI schema content.

    Args:
        term: The term to look up (e.g. 'TBS', 'Trustee Borrowing Scheme').

    Returns:
        Formatted markdown text on an exact hit, or None on a miss or if
        the glossary could not be loaded.
    """
    try:
        glossary = _load_glossary()
        if not glossary:
            return None

        term_norm = term.strip()
        if not term_norm:
            return None

        for key, data in glossary.items():
            if not isinstance(data, dict):
                continue
            if _ci_match(key, term_norm):
                return _format_entity(key, data)

        for key, data in glossary.items():
            if not isinstance(data, dict):
                continue
            code = data.get("code", "")
            name = data.get("name", "")
            aliases = data.get("aliases", [])
            if (
                _ci_match(code, term_norm)
                or _ci_match(name, term_norm)
                or _ci_in(term_norm, aliases)
            ):
                return _format_entity(key, data)

        return None
    except Exception:
        logger.warning("Glossary lookup failed for term=%r", term, exc_info=True)
        return None


def glossary_terms() -> list[str]:
    """Return a sorted list of glossary term codes for docstring/debug use.

    Falls back to the entry's top-level key when it has no code.
    """
    try:
        glossary = _load_glossary()
    except Exception:
        logger.warning("Failed to enumerate glossary terms", exc_info=True)
        return []

    terms = []
    for key, data in glossary.items():
        if isinstance(data, dict) and data.get("code"):
            terms.append(str(data["code"]))
        else:
            terms.append(key)
    return sorted(terms)
