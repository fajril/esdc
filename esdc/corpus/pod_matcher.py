"""Suggest current POD matches without assigning uncertain business identity."""

from __future__ import annotations

import difflib
import re
import sqlite3
from collections import defaultdict
from typing import Any

from esdc.pod_registry.store import is_missing_table_error

_MAX_SUGGESTIONS = 5
_MIN_SUBSTRING_LEN = 6
_FUZZY_CUTOFF = 0.85


def _norm(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value)).strip().casefold() if value else ""


class PodMatcher:
    """In-memory index of current, issued POD identities."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._by_letter: dict[str, list[str]] = defaultdict(list)
        self._by_name: dict[str, list[str]] = defaultdict(list)
        self._names: list[tuple[str, str]] = []
        try:
            available = {
                str(row[1]) for row in conn.execute("PRAGMA table_info(m_pod)")
            }
            if not available:
                return
            columns = ["pod_id", "pod_name", "pod_letter_num"]
            if "is_latest" in available:
                columns.append("is_latest")
            projection = ", ".join(f'"{column}"' for column in columns)
            rows = conn.execute(
                f"SELECT {projection} FROM m_pod WHERE pod_id IS NOT NULL "
                "AND trim(pod_id) <> ''"
            ).fetchall()
        except sqlite3.OperationalError as exc:
            if is_missing_table_error(exc):
                return
            raise

        for raw in rows:
            record = dict(zip(columns, raw, strict=True))
            if "is_latest" in record and not bool(record["is_latest"]):
                continue
            pod_id = str(record["pod_id"]).strip()
            if not pod_id:
                continue
            letter = _norm(record.get("pod_letter_num"))
            name = _norm(record.get("pod_name"))
            if letter and pod_id not in self._by_letter[letter]:
                self._by_letter[letter].append(pod_id)
            if name:
                if pod_id not in self._by_name[name]:
                    self._by_name[name].append(pod_id)
                self._names.append((name, pod_id))

    def resolve_exact(
        self, doc_number: Any = None, pod_names: list[str] | str | None = None
    ) -> tuple[str | None, str | None]:
        """Resolve only a unique exact letter or exact current POD name."""
        letter = _norm(doc_number)
        if letter:
            matches = self._by_letter.get(letter, [])
            if len(matches) == 1:
                return matches[0], "letter_num_exact"
            if matches:
                return None, "letter_num_ambiguous"

        raw_names = pod_names if isinstance(pod_names, list) else [pod_names]
        exact_matches: set[str] = set()
        for raw in raw_names:
            exact_matches.update(self._by_name.get(_norm(raw), ()))
        if len(exact_matches) == 1:
            return next(iter(exact_matches)), "pod_name_exact_unique"
        if exact_matches:
            return None, "pod_name_ambiguous"
        return None, None

    def suggest(
        self, doc_number: Any, pod_names: list[str] | str | None
    ) -> tuple[list[str], list[str]]:
        """Return ranked candidates and reasons; loose name matches are review-only."""
        suggestions: list[str] = []
        reasons: list[str] = []

        def add(pod_id: str, reason: str) -> None:
            if pod_id not in suggestions and len(suggestions) < _MAX_SUGGESTIONS:
                suggestions.append(pod_id)
                reasons.append(reason)

        letter = _norm(doc_number)
        letter_matches = self._by_letter.get(letter, []) if letter else []
        for pod_id in letter_matches[:_MAX_SUGGESTIONS]:
            reason = (
                f"exact letter number matches {pod_id}"
                if len(letter_matches) == 1
                else f"ambiguous exact letter number; review candidate {pod_id}"
            )
            add(pod_id, reason)

        raw_names = pod_names if isinstance(pod_names, list) else [pod_names]
        for raw in raw_names:
            norm = _norm(raw)
            if not norm or len(suggestions) >= _MAX_SUGGESTIONS:
                continue
            exact = self._by_name.get(norm, [])
            for pod_id in exact:
                reason = (
                    f"exact unique POD name matches {pod_id}"
                    if len(exact) == 1
                    else f"ambiguous exact POD name; review candidate {pod_id}"
                )
                add(pod_id, reason)

            if len(suggestions) >= _MAX_SUGGESTIONS:
                continue
            substring_matches = [
                (name, pod_id)
                for name, pod_id in self._names
                if len(norm) >= _MIN_SUBSTRING_LEN
                and len(name) >= _MIN_SUBSTRING_LEN
                and (name in norm or norm in name)
            ]
            for _, pod_id in substring_matches:
                add(pod_id, f"review-only substring candidate {pod_id}")

            if len(suggestions) >= _MAX_SUGGESTIONS:
                continue
            names = list(dict.fromkeys(name for name, _ in self._names))
            close = difflib.get_close_matches(
                norm, names, n=3, cutoff=_FUZZY_CUTOFF
            )
            for candidate in close:
                for _, pod_id in self._names:
                    if candidate == _ and len(suggestions) < _MAX_SUGGESTIONS:
                        add(pod_id, f"review-only fuzzy candidate {pod_id}")

        return suggestions, reasons
