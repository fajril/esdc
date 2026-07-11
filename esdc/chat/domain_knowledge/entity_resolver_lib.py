"""Entity resolver for ESDC.

Resolves entities from the database using pattern matching and FTS.
Returns structured context for single-shot SQL generation.
"""

from __future__ import annotations

import logging
import re
from typing import Any

import duckdb

from .entity_patterns import QueryPatternMatcher
from .entity_registry import ENTITY_LABEL_TO_KEY, ENTITY_REGISTRY, EntitySpec
from .entity_schema import KGSchema

logger = logging.getLogger(__name__)

_CONFIDENCE_THRESHOLD = 0.7

_YEAR_PATTERN = re.compile(r"\b(20\d{2})\b")
_PHRASE_WORD_PATTERN = re.compile(r"[a-zA-Z0-9\u00C0-\u024F][\w&./'-]*")

# resolve_name() looks up bare entity strings (not NL queries), so it needs
# its own fixed acceptance bar independent of _CONFIDENCE_THRESHOLD (which
# tunes the chat NL path and is currently 0.7, not 0.8).
_NAME_CONFIDENCE_THRESHOLD = 0.8

# Indonesian/English domain-boilerplate words stripped only inside
# resolve_name()'s normalization fallback. Deliberately NOT merged into the
# global _STOP_WORDS/_ENTITY_HINTS below -- those drive entity-type hint
# detection on the chat NL path and widening them would change that
# behavior too (e.g. "field"/"project" are entity-type hints there).
_NAME_NOISE_WORDS: frozenset[str] = frozenset(
    {
        "proyek",
        "project",
        "pengembangan",
        "development",
        "lapangan",
        "field",
        "wilayah",
        "kerja",
        "pod",
    }
)

# Single-token segments resolve_name()'s greedy token segmentation never
# tries to match on their own.
_NAME_SEGMENT_SKIP_TOKENS: frozenset[str] = frozenset(
    {"dan", "and", "&", "the", "of", "di"}
)

_UNCERTAINTY_MAP: dict[str, str] = {
    "1p": "1. Low Value",
    "proven": "1. Low Value",
    "terbukti": "1. Low Value",
    "low": "1. Low Value",
    "2p": "2. Middle Value",
    "probable": "2. Middle Value",
    "mungkin": "2. Middle Value",
    "middle": "2. Middle Value",
    "mid": "2. Middle Value",
    "3p": "3. High Value",
    "possible": "3. High Value",
    "harapan": "3. High Value",
    "high": "3. High Value",
    "1c": "1. Low Value",
    "2c": "2. Middle Value",
    "3c": "3. High Value",
}

_CLASS_MAP: dict[str, str] = {
    "reserves": "1. Reserves & GRR",
    "cadangan": "1. Reserves & GRR",
    "grr": "1. Reserves & GRR",
    "contingent": "2. Contingent Resources",
    "kontingen": "2. Contingent Resources",
    "prospective": "3. Prospective Resources",
    "prospektif": "3. Prospective Resources",
    "abandoned": "4. Abandoned",
}

_ENTITY_HINTS: dict[str, str] = {
    "lapangan": "Field",
    "field": "Field",
    "wk": "WorkingArea",
    "wilayah": "WorkingArea",
    "wilayah kerja": "WorkingArea",
    "working area": "WorkingArea",
    "proyek": "Project",
    "project": "Project",
    "operator": "Operator",
    "perusahaan": "Operator",
    "oleh": "Operator",
}

_STOP_WORDS: set[str] = {
    "di",
    "yang",
    "dan",
    "atau",
    "dengan",
    "untuk",
    "dari",
    "ke",
    "pada",
    "ini",
    "itu",
    "adalah",
    "tidak",
    "sudah",
    "akan",
    "apa",
    "saja",
    "berapa",
    "siapa",
    "dimana",
    "kapan",
    "data",
    "info",
    "informasi",
    "lihat",
    "tampilkan",
    "show",
    "the",
    "of",
    "in",
    "for",
    "at",
    "to",
    "from",
    "with",
    "by",
    "about",
    "how",
    "what",
    "when",
    "where",
    "which",
}


class EntityResolver:
    """Resolve entities from database for Knowledge Traversal."""

    def __init__(self, db: duckdb.DuckDBPyConnection) -> None:
        """Initialize the knowledge graph resolver."""
        self.schema = KGSchema()
        self.pattern_matcher = QueryPatternMatcher(self.schema)
        self.db = db
        self._columns_cache: dict[str, set[str]] = {}

    def resolve(self, query: str, return_multiple: bool = True) -> dict[str, Any]:
        """Resolve entities and patterns from a natural language query.

        Returns structured context for single-shot SQL generation.
        """
        entities = self._resolve_entities(query, return_multiple)
        pattern_result = self.pattern_matcher.match(query)

        if not entities and not pattern_result:
            return {
                "status": "failed",
                "fallback": "multi_round",
                "message": "Could not resolve any entities or patterns from query",
                "entities": [],
                "pattern": None,
                "suggested_table": None,
                "where_conditions": [],
                "required_columns": [],
                "confidence": 0.0,
            }

        best_confidence = max((e.get("confidence", 0.0) for e in entities), default=0.0)

        if (
            not return_multiple
            and best_confidence < _CONFIDENCE_THRESHOLD
            and len(entities) > 1
        ):
            return {
                "status": "ambiguous",
                "message": (
                    f"Found {len(entities)} possible matches. Please specify which one."
                ),
                "candidates": entities,
                "pattern": pattern_result,
                "suggested_table": pattern_result.get("suggested_table")
                if pattern_result
                else None,
                "confidence": best_confidence,
            }

        where_conditions = self._build_where_conditions(entities)
        suggested_table = self._determine_table(query, entities, pattern_result)

        return {
            "status": "success",
            "entities": entities,
            "pattern": pattern_result,
            "suggested_table": suggested_table,
            "where_conditions": where_conditions,
            "required_columns": pattern_result.get("suggested_columns", [])
            if pattern_result
            else [],
            "confidence": best_confidence,
        }

    def resolve_name(self, name: str, entity_type: str) -> list[dict[str, Any]]:
        """Resolve a bare entity name against one registry spec.

        Unlike `resolve()`, there is no NL parsing. Returns the FINAL
        confident picks -- callers should not re-filter by confidence.
        A raw name legitimately naming several DB rows (e.g. "Arung
        Nowera" naming two separate fields) can come back with more than
        one match. Fallback chain, each step short-circuiting on a hit:

        1. Whole string as the search term -> single best match.
        2. Domain-word-stripped phrase (drops hint/noise words like
           "proyek"/"pengembangan"/"lapangan") -> single best match.
        3. Greedy longest-first token segmentation of that phrase, so a
           multi-entity raw name resolves to multiple canonical rows
           without also emitting spurious sub-token matches for names
           that already matched in full (no fan-out).

        Every candidate match is word-boundary guarded (see
        `_best_confident_match`) so a bare substring hit like "arung"
        matching "GARUNG - BASE" is rejected.
        """
        if entity_type not in ENTITY_REGISTRY:
            raise ValueError(f"unknown entity_type: {entity_type}")
        spec = ENTITY_REGISTRY[entity_type]

        name = (name or "").strip()
        if not name:
            return []

        # 1. Whole string.
        best = self._best_confident_match(spec, name)
        if best is not None:
            return [best]

        # 2. Domain-word-stripped phrase.
        phrase = self._normalize_candidate_phrase(name, remove_domain_keywords=True)
        phrase = self._strip_name_noise_words(phrase)
        if phrase and phrase.lower() != name.lower():
            best = self._best_confident_match(spec, phrase)
            if best is not None:
                return [best]

        # 3. Greedy longest-first token segmentation over the normalized
        # phrase's tokens (so noise/stop words never become search terms).
        tokens = phrase.split()
        results: list[dict[str, Any]] = []
        seen: set[str] = set()
        pos = 0
        while pos < len(tokens):
            matched = False
            for size in range(min(5, len(tokens) - pos), 0, -1):
                term_tokens = tokens[pos : pos + size]
                if size == 1 and term_tokens[0].lower() in _NAME_SEGMENT_SKIP_TOKENS:
                    continue
                term = " ".join(term_tokens)
                best = self._best_confident_match(spec, term)
                if best is not None:
                    canonical = best["name"].strip().lower()
                    if canonical not in seen:
                        seen.add(canonical)
                        results.append(best)
                    pos += size
                    matched = True
                    break
            if not matched:
                pos += 1

        return results

    def _best_confident_match(
        self, spec: EntitySpec, term: str
    ) -> dict[str, Any] | None:
        """Highest-confidence word-boundary-guarded match for `term`, or None."""
        term = term.strip()
        if not term:
            return None
        matches = self._query_entity_spec(spec, term, return_multiple=True)
        confident = [
            m
            for m in matches
            if m["confidence"] >= _NAME_CONFIDENCE_THRESHOLD
            and self._name_boundary_match(term, m["name"])
        ]
        if not confident:
            return None
        return max(confident, key=lambda m: m["confidence"])

    @staticmethod
    def _name_boundary_match(term: str, entity_name: str) -> bool:
        """True if `term` sits at word boundaries in `entity_name` (or is equal).

        Kills bare-substring false positives ("arung" -> "GARUNG - BASE")
        while keeping boundary-aligned prefix/substring hits ("arung" ->
        "ARUNG - BASE").
        """
        if term.strip().lower() == entity_name.strip().lower():
            return True
        return bool(re.search(rf"\b{re.escape(term)}\b", entity_name, re.IGNORECASE))

    @staticmethod
    def _strip_name_noise_words(phrase: str) -> str:
        if not phrase:
            return phrase
        words = [w for w in phrase.split() if w.lower() not in _NAME_NOISE_WORDS]
        return " ".join(words)

    def _resolve_entities(
        self, query: str, return_multiple: bool
    ) -> list[dict[str, Any]]:
        entities: list[dict[str, Any]] = []

        year_match = _YEAR_PATTERN.search(query)
        if year_match:
            entities.append(
                {
                    "type": "Year",
                    "value": int(year_match.group(1)),
                    "name": year_match.group(1),
                    "confidence": 1.0,
                }
            )

        uncertainty = self._resolve_uncertainty(query)
        if uncertainty:
            entities.append(uncertainty)

        project_class = self._resolve_class(query)
        if project_class:
            entities.append(project_class)

        entity_type_hint = self._detect_entity_type(query)
        named_entities = self._resolve_named_entities(
            query, entity_type_hint, return_multiple
        )
        entities.extend(named_entities)

        return entities

    def _resolve_uncertainty(self, query: str) -> dict[str, Any] | None:
        query_lower = query.lower()
        for keyword, db_value in _UNCERTAINTY_MAP.items():
            if keyword in query_lower:
                return {
                    "type": "UncertaintyLevel",
                    "name": keyword.upper(),
                    "db_value": db_value,
                    "confidence": 1.0,
                }
        return None

    def _resolve_class(self, query: str) -> dict[str, Any] | None:
        query_lower = query.lower()
        for keyword, db_value in _CLASS_MAP.items():
            if keyword in query_lower:
                return {
                    "type": "ProjectClass",
                    "name": keyword,
                    "db_value": db_value,
                    "confidence": 0.9,
                }
        return None

    def _detect_entity_type(self, query: str) -> str | None:
        query_lower = query.lower()
        for keyword, entity_type in _ENTITY_HINTS.items():
            if keyword in query_lower:
                return entity_type
        return None

    def _resolve_named_entities(
        self, query: str, entity_type_hint: str | None, return_multiple: bool
    ) -> list[dict[str, Any]]:
        candidates = self._extract_entity_candidates(query, entity_type_hint)
        all_results: list[dict[str, Any]] = []
        seen: set[tuple[str, str, str]] = set()

        for entity_key, search_term, hinted in candidates:
            specs = (
                [ENTITY_REGISTRY[entity_key]]
                if entity_key
                else list(ENTITY_REGISTRY.values())
            )
            for spec in specs:
                results = self._query_entity_spec(spec, search_term, return_multiple)
                for result in results:
                    dedupe_key = (
                        result["entity_type"],
                        str(result.get("id")),
                        result["name"].lower(),
                    )
                    if dedupe_key in seen:
                        continue
                    seen.add(dedupe_key)
                    if hinted:
                        result["confidence"] = min(result["confidence"] + 0.08, 1.0)
                    result["matched_text"] = search_term
                    result["hinted"] = hinted
                    all_results.append(result)

        if return_multiple:
            return sorted(
                all_results,
                key=lambda x: (
                    x["confidence"],
                    -ENTITY_REGISTRY[x["entity_type"]].priority,
                    len(x["name"]),
                ),
                reverse=True,
            )

        return self._select_named_entities(all_results)

    def _extract_entity_name(self, query: str) -> str:
        words = _PHRASE_WORD_PATTERN.findall(query)
        skip_words = (
            set(_ENTITY_HINTS.keys())
            | set(_CLASS_MAP.keys())
            | set(_UNCERTAINTY_MAP.keys())
            | _STOP_WORDS
        )
        candidates = [w for w in words if len(w) > 2 and w.lower() not in skip_words]
        pattern_keywords = set()
        for pattern in self.schema.query_patterns.values():
            pattern_keywords.update(kw.lower() for kw in pattern.get("keywords", []))
        candidates = [w for w in candidates if w.lower() not in pattern_keywords]

        return " ".join(candidates) if candidates else ""

    def _extract_entity_candidates(
        self, query: str, entity_type_hint: str | None
    ) -> list[tuple[str | None, str, bool]]:
        candidates: list[tuple[str | None, str, bool]] = []
        seen: set[tuple[str | None, str]] = set()

        def add(entity_key: str | None, phrase: str, hinted: bool) -> None:
            normalized = self._normalize_candidate_phrase(
                phrase, remove_domain_keywords=not hinted
            )
            if not normalized:
                return
            key = (entity_key, normalized.lower())
            if key in seen:
                return
            seen.add(key)
            candidates.append((entity_key, normalized, hinted))

        for quoted in re.findall(r'"([^"]+)"|' r"'([^']+)'", query):
            phrase = quoted[0] or quoted[1]
            add(None, phrase, True)

        query_lower = query.lower()
        for entity_key, spec in ENTITY_REGISTRY.items():
            for hint in sorted(spec.hints, key=len, reverse=True):
                pattern = (
                    rf"\b{re.escape(hint)}\s+(.+?)"
                    rf"(?=\s+(?:tahun|year|oleh|operator|di|pada|untuk|dengan|dan|yang)\b|$)"
                )
                for match in re.finditer(pattern, query_lower, re.IGNORECASE):
                    add(entity_key, match.group(1), True)

        if entity_type_hint:
            entity_key = ENTITY_LABEL_TO_KEY.get(entity_type_hint)
            add(entity_key, self._extract_entity_name(query), True)
        else:
            fallback = self._extract_entity_name(query)
            words = fallback.split()
            for size in range(min(5, len(words)), 0, -1):
                for start in range(0, len(words) - size + 1):
                    add(None, " ".join(words[start : start + size]), False)

        return candidates

    def _normalize_candidate_phrase(
        self, phrase: str, *, remove_domain_keywords: bool
    ) -> str:
        words = _PHRASE_WORD_PATTERN.findall(phrase)
        if not words:
            return ""
        skip_words = (
            set(_ENTITY_HINTS.keys())
            | set(_CLASS_MAP.keys())
            | set(_UNCERTAINTY_MAP.keys())
            | _STOP_WORDS
        )
        pattern_keywords = set()
        if remove_domain_keywords:
            for pattern in self.schema.query_patterns.values():
                pattern_keywords.update(
                    kw.lower() for kw in pattern.get("keywords", [])
                )
        filtered = [
            w
            for w in words
            if not _YEAR_PATTERN.fullmatch(w)
            and w.lower() not in skip_words
            and w.lower() not in pattern_keywords
        ]
        return " ".join(filtered).strip()

    def _select_named_entities(
        self, results: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        if not results:
            return []

        best_by_type: dict[str, dict[str, Any]] = {}
        for result in sorted(
            results,
            key=lambda x: (
                x["confidence"],
                -ENTITY_REGISTRY[x["entity_type"]].priority,
                len(x["name"]),
            ),
            reverse=True,
        ):
            best_by_type.setdefault(result["entity_type"], result)

        selected = list(best_by_type.values())
        selected.sort(
            key=lambda x: (
                x["confidence"],
                -ENTITY_REGISTRY[x["entity_type"]].priority,
                len(x["name"]),
            ),
            reverse=True,
        )

        hinted = [e for e in selected if e.get("hinted")]
        if len(hinted) > 1:
            return [e for e in selected if e["confidence"] >= _CONFIDENCE_THRESHOLD]

        if selected[0]["confidence"] >= _CONFIDENCE_THRESHOLD:
            return [selected[0]]
        return []

    def _query_entity_spec(
        self, spec: EntitySpec, search_term: str, return_multiple: bool
    ) -> list[dict[str, Any]]:
        columns = self._table_columns(spec.lookup_table)
        if spec.name_column not in columns:
            return []

        id_expr = spec.id_column if spec.id_column in columns else "NULL"
        id_condition = (
            f" OR {spec.id_column} ILIKE '%' || ? || '%'"
            if spec.id_column in columns
            else ""
        )
        limit = 10 if return_multiple else 3
        sql = f"""
            SELECT entity_id, entity_name
            FROM (
                SELECT DISTINCT
                    {id_expr} AS entity_id,
                    {spec.name_column} AS entity_name
                FROM {spec.lookup_table}
                WHERE {spec.name_column} ILIKE '%' || ? || '%'{id_condition}
            ) matches
            ORDER BY
                CASE
                    WHEN lower(trim(entity_name)) = lower(trim(?)) THEN 0
                    WHEN lower(trim(entity_name)) LIKE lower(trim(?)) || '%' THEN 1
                    WHEN lower(trim(entity_name)) LIKE '%' || lower(trim(?)) || '%'
                        THEN 2
                    ELSE 3
                END,
                length(trim(entity_name)),
                lower(trim(entity_name))
            LIMIT {limit}
        """

        try:
            params = [search_term, search_term] if id_condition else [search_term]
            params.extend([search_term, search_term, search_term])
            result = self.db.execute(sql, params).fetchall()
        except Exception:
            logger.debug(
                "[KG] entity_query_failed | type=%s term=%s", spec.label, search_term
            )
            return []

        if not result:
            return []

        entities: list[dict[str, Any]] = []
        for row in result:
            entity_id, entity_name = row[0], row[1]
            if not entity_name:
                continue

            search_lower = search_term.lower()
            name_lower = entity_name.lower() if entity_name else ""
            if name_lower == search_lower:
                confidence = 1.0
            elif name_lower.startswith(search_lower):
                confidence = 0.9
            elif search_lower in name_lower:
                confidence = 0.8
            else:
                confidence = 0.6

            entity: dict[str, Any] = {
                "type": spec.label,
                "entity_type": spec.key,
                "entity_level": spec.level,
                "id": entity_id,
                "name": entity_name,
                "confidence": confidence,
                "match_type": self._match_type(search_lower, name_lower),
                "filter_column": spec.name_column,
                "recommended_table": spec.default_table,
            }

            entities.append(entity)

        return entities

    def _table_columns(self, table: str) -> set[str]:
        if table not in self._columns_cache:
            rows = self.db.execute(f"PRAGMA table_info('{table}')").fetchall()
            self._columns_cache[table] = {str(row[1]) for row in rows}
        return self._columns_cache[table]

    def _match_type(self, search_lower: str, name_lower: str) -> str:
        if name_lower == search_lower:
            return "exact"
        if name_lower.startswith(search_lower):
            return "prefix"
        if search_lower in name_lower:
            return "substring"
        return "weak"

    def _build_where_conditions(self, entities: list[dict[str, Any]]) -> list[str]:
        conditions: list[str] = []
        for entity in entities:
            etype = entity.get("type")
            if etype == "Year":
                conditions.append(f"report_year = {entity['value']}")
            elif etype == "UncertaintyLevel":
                conditions.append(f"uncert_level = '{entity['db_value']}'")
            elif etype == "ProjectClass":
                conditions.append(f"project_class = '{entity['db_value']}'")
            elif entity.get("entity_type") in ENTITY_REGISTRY and entity.get("name"):
                col = entity["filter_column"]
                name = entity["name"].replace("'", "''")
                conditions.append(f"{col} = '{name}'")
        return conditions

    def _determine_table(
        self, query: str, entities: list[dict[str, Any]], pattern: dict[str, Any] | None
    ) -> str:
        named_entities = [
            e for e in entities if e.get("entity_type") in ENTITY_REGISTRY
        ]
        if named_entities:
            if self._needs_project_detail(query) or any(
                e["entity_type"] in ("project_name", "operator_name")
                and e.get("hinted")
                for e in named_entities
            ):
                if self._is_production_query(query):
                    return "project_timeseries"
                return "project_resources"
            if pattern and pattern.get("primary_entity"):
                primary_key = ENTITY_LABEL_TO_KEY.get(pattern["primary_entity"])
                if primary_key:
                    for entity in named_entities:
                        if entity["entity_type"] == primary_key:
                            spec = ENTITY_REGISTRY[primary_key]
                            if (
                                self._is_production_query(query)
                                and spec.timeseries_table
                            ):
                                return spec.timeseries_table
                            return spec.default_table
            primary = min(
                named_entities,
                key=lambda e: ENTITY_REGISTRY[e["entity_type"]].priority,
            )
            spec = ENTITY_REGISTRY[primary["entity_type"]]
            if self._is_production_query(query) and spec.timeseries_table:
                return spec.timeseries_table
            return spec.default_table

        for entity in entities:
            etype = entity.get("type")
            if etype == "Field":
                if self._is_production_query(query):
                    return "field_timeseries"
                return "field_resources"
            elif etype == "WorkingArea":
                if self._is_production_query(query):
                    return "wa_timeseries"
                return "wa_resources"
            elif etype == "Operator":
                return "project_resources"

        if pattern and pattern.get("suggested_table"):
            return pattern["suggested_table"]

        return "project_resources"

    def _is_production_query(self, query: str) -> bool:
        return any(
            kw in query.lower()
            for kw in ("produksi", "forecast", "profil", "rate", "production")
        )

    def _needs_project_detail(self, query: str) -> bool:
        return any(
            kw in query.lower()
            for kw in ("proyek", "project", "operator", "perusahaan", "oleh")
        )
