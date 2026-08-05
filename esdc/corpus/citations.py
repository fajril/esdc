"""Letter-number citation extraction for Indonesian correspondence.

Documents in this corpus cite each other by nomor surat in body text
("menindaklanjuti surat No. SRT-0184/SKKO0000/2016/S1"). Extracting
those deterministically gives a citation graph with no LLM and no
hallucination surface.

Normalization strips ALL whitespace rather than collapsing runs: real
doc_number values carry erratic internal spacing ('SRT- 0204
/SKKIE1000/2025/S1'), so pod_matcher._norm — which only collapses runs —
compares two spellings of one letter as different strings.
"""

from __future__ import annotations

import re

# A nomor surat is: an optional dash-terminated series prefix (SRT-, T-),
# a 2-5 digit sequence, then two or three slash-separated segments.
# The (?<![A-Z0-9]) guard stops an Indonesian word preceding a bare number
# ("surat 0433 /...") from being absorbed as the series prefix. Segments
# allow dots because ministry numbers use them (T-37/MG.04/MEM.M/2025), but
# must END on an alphanumeric — otherwise a citation closing a sentence
# ("... T-37/MG.04/MEM.M/2025.") swallows the full stop.
LETTER_NUM_RE = re.compile(
    r"(?<![A-Z0-9])(?:[A-Z]{1,6}-\s*)?\d{2,5}\s*/\s*[A-Z0-9.]{1,19}[A-Z0-9]"
    r"\s*/\s*[A-Z0-9.-]{0,19}[A-Z0-9](?:\s*/\s*[A-Z0-9.-]{0,9}[A-Z0-9])?",
    re.IGNORECASE,
)


def normalize_letter_number(value: str | None) -> str:
    """Canonical form of a letter number: no whitespace, uppercased."""
    if not value:
        return ""
    return re.sub(r"\s+", "", str(value)).upper()


def extract_letter_numbers(markdown: str) -> list[str]:
    """Normalized letter numbers cited in body text, deduped, in order."""
    if not markdown:
        return []
    seen: set[str] = set()
    out: list[str] = []
    for raw in LETTER_NUM_RE.findall(markdown):
        norm = normalize_letter_number(raw)
        if norm and norm not in seen:
            seen.add(norm)
            out.append(norm)
    return out
