"""Resolve extracted entity mentions to registry-backed canonical ids.

This step is what keeps the auto-built graph clean: LLM mentions never
become edges unless they land on a registry entity (m_pod, project_resources).
"""

from __future__ import annotations

import difflib
import logging
import sqlite3
from dataclasses import dataclass, field
from typing import Any

import duckdb

from esdc.corpus.pod_matcher import PodMatcher
from esdc.knowledge.extractor import ExtractionResult
from esdc.knowledge.store import Claim, Edge

logger = logging.getLogger(__name__)

_FUZZY_CUTOFF = 0.85


@dataclass
class ResolutionResult:
    edges: list[Edge] = field(default_factory=list)
    claims: list[Claim] = field(default_factory=list)
    unresolved: list[dict[str, str]] = field(default_factory=list)


def _name_lookup(duck_conn: duckdb.DuckDBPyConnection, sql: str) -> dict[str, str]:
    """Lowercase name -> canonical id from a 2-column (name, id) query."""
    rows = duck_conn.execute(sql).fetchall()
    return {
        str(name).strip().lower(): str(ident)
        for name, ident in rows
        if name is not None and ident is not None
    }


def _resolve_name(name: str, lookup: dict[str, str]) -> tuple[str | None, float]:
    key = name.strip().lower()
    if key in lookup:
        return lookup[key], 0.9
    close = difflib.get_close_matches(key, list(lookup), n=1, cutoff=_FUZZY_CUTOFF)
    if close:
        return lookup[close[0]], 0.75
    return None, 0.0


class _Resolvers:
    def __init__(
        self,
        sqlite_conn: sqlite3.Connection,
        duck_conn: duckdb.DuckDBPyConnection,
    ) -> None:
        self.pod_matcher = PodMatcher(sqlite_conn)
        self.fields = _name_lookup(
            duck_conn,
            "SELECT DISTINCT field_name, field_name FROM project_resources "
            "WHERE field_name IS NOT NULL",
        )
        self.wks = _name_lookup(
            duck_conn,
            "SELECT DISTINCT wk_name, wk_name FROM project_resources "
            "WHERE wk_name IS NOT NULL",
        )
        self.projects = _name_lookup(
            duck_conn,
            "SELECT DISTINCT project_name, project_id FROM project_resources "
            "WHERE project_name IS NOT NULL",
        )

    def resolve(self, etype: str, name: str) -> tuple[str | None, float]:
        if etype == "pod":
            suggestions, _ = self.pod_matcher.suggest(None, [name])
            return (suggestions[0], 0.7) if suggestions else (None, 0.0)
        if etype == "field":
            return _resolve_name(name, self.fields)
        if etype == "working_area":
            return _resolve_name(name, self.wks)
        if etype == "project":
            return _resolve_name(name, self.projects)
        return None, 0.0


_REL_BY_TYPE = {
    "pod": ("ABOUT_POD", "pod"),
    "field": ("ABOUT_FIELD", "field"),
    "working_area": ("ABOUT_WK", "working_area"),
    "project": ("ABOUT_PROJECT", "project"),
}


def resolve_extraction(
    doc_id: str,
    extraction: ExtractionResult,
    sqlite_conn: sqlite3.Connection,
    duck_conn: duckdb.DuckDBPyConnection,
) -> ResolutionResult:
    resolvers = _Resolvers(sqlite_conn, duck_conn)
    result = ResolutionResult()
    resolved_mentions: dict[tuple[str, str], str] = {}

    for mention in extraction.entities:
        etype, name = mention["type"], mention["name"]
        ident, confidence = resolvers.resolve(etype, name)
        if ident is None:
            result.unresolved.append({"type": etype, "name": name})
            continue
        resolved_mentions[(etype, name.strip().lower())] = ident
        rel, dst_type = _REL_BY_TYPE[etype]
        # pod matcher confidence is fixed at 0.7 by contract
        conf = 0.7 if etype == "pod" else confidence
        result.edges.append(
            Edge(
                "document",
                doc_id,
                rel,
                dst_type,
                ident,
                confidence=conf,
                method="llm_extracted",
                evidence=f"mention: {name}",
            )
        )

    for raw in extraction.claims:
        subject_type: str | None = raw.get("subject_type")
        subject_id: str | None = None
        subject: Any = raw.get("subject")
        if subject_type and isinstance(subject, str):
            subject_id = resolved_mentions.get((subject_type, subject.strip().lower()))
            if subject_id is None:
                subject_id, _ = resolvers.resolve(subject_type, subject)
        if subject_id is None:
            subject_type = None
        result.claims.append(
            Claim(
                doc_id=doc_id,
                claim_type=raw["type"],
                predicate=raw["predicate"],
                value=raw["value"],
                subject_type=subject_type,
                subject_id=subject_id,
                evidence=raw.get("evidence"),
                confidence=0.7,
            )
        )
    return result
