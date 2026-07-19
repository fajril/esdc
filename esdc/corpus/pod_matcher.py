"""Match extracted document metadata against the POD registry.

Produces SUGGESTED pod_ids only — pod_document links are always
human-confirmed in the portal. Two signals, strongest first:
letter number exact (normalized) and pod_name text similarity.
"""

from __future__ import annotations

import difflib
import re
import sqlite3
from typing import Any

_MAX_SUGGESTIONS = 5
_MIN_SUBSTRING_LEN = 6
_FUZZY_CUTOFF = 0.85


def _norm(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value)).strip().casefold() if value else ""


class PodMatcher:
    """In-memory matcher over m_pod; safe to build when the table is absent."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        try:
            rows = conn.execute(
                "SELECT pod_id, pod_name, pod_letter_num FROM m_pod"
            ).fetchall()
        except sqlite3.OperationalError:
            rows = []
        self._by_letter_num: dict[str, str] = {}
        self._names: list[tuple[str, str, str]] = []  # (norm_name, name, pod_id)
        for r in rows:
            if r["pod_letter_num"]:
                self._by_letter_num[_norm(r["pod_letter_num"])] = r["pod_id"]
            if r["pod_name"]:
                self._names.append((_norm(r["pod_name"]), r["pod_name"], r["pod_id"]))

    def suggest(
        self, doc_number: Any, pod_names: list[str] | str | None
    ) -> tuple[list[str], list[str]]:
        """(suggested pod_ids, human-readable reasons) for one document."""
        suggestions: list[str] = []
        reasons: list[str] = []

        letter = self._by_letter_num.get(_norm(doc_number))
        if letter:
            suggestions.append(letter)
            reasons.append(f"letter number matches {letter}")

        raw_names = pod_names if isinstance(pod_names, list) else [pod_names]
        for raw in raw_names:
            norm = _norm(raw)
            if not norm:
                continue
            matched = self._match_name(norm)
            for pod_id, why in matched:
                if pod_id not in suggestions:
                    suggestions.append(pod_id)
                    reasons.append(f"pod_name '{raw}' {why} {pod_id}")

        return suggestions[:_MAX_SUGGESTIONS], reasons[:_MAX_SUGGESTIONS]

    def _match_name(self, norm: str) -> list[tuple[str, str]]:
        exact = [(pid, "matches") for n, _, pid in self._names if n == norm]
        if exact:
            return exact
        if len(norm) >= _MIN_SUBSTRING_LEN:
            sub = [
                (pid, "contains/contained-in")
                for n, _, pid in self._names
                if (len(n) >= _MIN_SUBSTRING_LEN and (n in norm or norm in n))
            ]
            if sub:
                return sub
        close = difflib.get_close_matches(
            norm, [n for n, _, _ in self._names], n=3, cutoff=_FUZZY_CUTOFF
        )
        by_norm = {n: pid for n, _, pid in self._names}
        return [(by_norm[c], "fuzzy-matches") for c in close]
