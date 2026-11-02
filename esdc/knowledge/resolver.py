"""Resolve extracted entity mentions to registry-backed canonical ids.

This step is what keeps the auto-built graph clean: LLM mentions never
become edges unless they land on a registry entity (m_pod, m_project, the
canonical raw API resource rows).
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
from esdc.pod_registry.migrations import _validate_existing_v2_registry

logger = logging.getLogger(__name__)

_FUZZY_CUTOFF = 0.85


@dataclass
class ResolutionResult:
    edges: list[Edge] = field(default_factory=list)
    claims: list[Claim] = field(default_factory=list)
    unresolved: list[dict[str, str]] = field(default_factory=list)


def _unique_name_lookup(rows: list[tuple[Any, Any]]) -> dict[str, str]:
    """Exact lowercase name to id, excluding ambiguous display names."""
    candidates: dict[str, set[str]] = {}
    for name, ident in rows:
        if name is None or ident is None:
            continue
        key = str(name).strip().casefold()
        if key:
            candidates.setdefault(key, set()).add(str(ident))
    return {
        name: next(iter(ids)) for name, ids in candidates.items() if len(ids) == 1
    }


def _resolve_name(name: str, lookup: dict[str, str]) -> tuple[str | None, float]:
    """Resolve field/project names; POD identity uses stricter matcher rules."""
    key = name.strip().casefold()
    if key in lookup:
        return lookup[key], 0.9
    close = difflib.get_close_matches(
        key, list(lookup), n=1, cutoff=_FUZZY_CUTOFF
    )
    if close:
        return lookup[close[0]], 0.75
    return None, 0.0


def _sqlite_name_lookup(
    sqlite_conn: sqlite3.Connection,
    table: str,
    name_column: str,
    id_column: str,
) -> dict[str, str]:
    exists = sqlite_conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type IN ('table','view') AND name=?",
        [table],
    ).fetchone()
    if exists is None:
        return {}
    columns = {
        str(row[1]) for row in sqlite_conn.execute(f'PRAGMA table_info("{table}")')
    }
    if name_column not in columns or id_column not in columns:
        return {}
    rows = sqlite_conn.execute(
        f'SELECT DISTINCT "{name_column}", "{id_column}" FROM "{table}" '
        f'WHERE "{name_column}" IS NOT NULL AND "{id_column}" IS NOT NULL'
    ).fetchall()
    return _unique_name_lookup([tuple(row) for row in rows])


class _Resolvers:
    def __init__(self, sqlite_conn: sqlite3.Connection) -> None:
        self.pod_matcher = PodMatcher(sqlite_conn)
        # Canonical sources only: m_project owns project identity, and the raw API
        # resource rows are the only source of field and working-area names.
        self.fields = _sqlite_name_lookup(
            sqlite_conn, "api_project_resources", "field_name", "field_name"
        )
        self.wks = _sqlite_name_lookup(
            sqlite_conn, "api_project_resources", "wk_name", "wk_name"
        )
        self.projects = _sqlite_name_lookup(
            sqlite_conn, "m_project", "project_name", "project_id"
        )

    def resolve(self, etype: str, name: str) -> tuple[str | None, float]:
        if etype == "pod":
            ident, _method = self.pod_matcher.resolve_exact(pod_names=[name])
            return (ident, 0.9) if ident is not None else (None, 0.0)
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
    """Resolve mentions against the single current source.

    ``duck_conn`` is accepted for caller compatibility while the instance graph is
    still on its pending conversion; the disposable dossier cache is never a
    source of canonical names. A source that is not the current schema is refused
    before anything is resolved.
    """
    _validate_existing_v2_registry(sqlite_conn)
    resolvers = _Resolvers(sqlite_conn)
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
