"""KSMI domain knowledge graph manager using LadybugDB.

Provides three query modes:
1. find(query) — BM25 FTS search for entity resolution (replaces buggy string matching)
2. traverse(entity_code, rel_type) — Cypher relationship traversal
3. query(cypher) — Raw Cypher for complex queries

Wraps an in-memory LadybugDB instance loaded from ksmi_schema.yaml.
Lazy-initializes on first access. Thread-safe via module-level singleton.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from esdc.chat.domain_knowledge.ksmi_graph_builder import KSMIGraphBuilder

logger = logging.getLogger(__name__)


class KSMIGraphManager:
    """Manage KSMI knowledge graph queries via LadybugDB.

    Wraps an in-memory LadybugDB instance loaded from ksmi_schema.yaml.
    Lazy-initializes on first access. Uses module-level singleton pattern.
    """

    _instance: KSMIGraphManager | None = None
    _builder: KSMIGraphBuilder
    _initialized: bool
    _lock: threading.Lock = threading.Lock()

    def __new__(cls) -> KSMIGraphManager:
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._builder = KSMIGraphBuilder()
                cls._instance._initialized = False
            return cls._instance

    def _ensure_initialized(self) -> None:
        with KSMIGraphManager._lock:
            if not self._initialized:
                try:
                    self._builder.build_from_yaml()
                    self._initialized = True
                    logger.info("[KSMI-KG] graph_initialized_successfully")
                except Exception as e:
                    logger.error("[KSMI-KG] initialization_failed | error=%s", e)
                    raise

    def _execute_cypher(self, cypher: str) -> list[list[Any]]:
        """Execute a Cypher query and return result rows.

        Wraps LadybugDB's result iteration API which differs from
        the type stubs (uses has_next/get_next at runtime).
        """
        result = self._builder.conn.execute(cypher)
        rows: list[list[Any]] = []
        while result.has_next():  # type: ignore[union-attr]
            rows.append(result.get_next())  # type: ignore[union-attr]
        return rows

    @property
    def conn(self) -> Any:
        self._ensure_initialized()
        return self._builder.conn

    @property
    def is_available(self) -> bool:
        return self._initialized and self._builder.is_built

    def find(
        self, query: str, top_k: int = 5, table: str = "KSMIEntity"
    ) -> list[dict[str, Any]]:
        """Find entities using BM25 full-text search.

        Uses Indonesian stemmer for Indonesian/English queries.
        Searches across code, name, aliases, definition, and entity_type.

        Args:
            query: Search string (e.g., "reserves", "cadangan", "E0")
            top_k: Maximum number of results to return
            table: Table to search (KSMIEntity, ProjectLevel, VolumeType,
                    VolumeSubstance, KSMIFrameworkNode)

        Returns:
            List of dicts with node properties and relevance scores.
        """
        self._ensure_initialized()
        index_map = {
            "KSMIEntity": "entity_fts",
            "ProjectLevel": "level_fts",
            "VolumeType": "volume_fts",
            "VolumeSubstance": "substance_fts",
            "KSMIFrameworkNode": "framework_fts",
        }
        idx = index_map.get(table, "entity_fts")
        escaped = query.replace("'", "\\'")
        return_cols_map = {
            "KSMIEntity": (
                "node.code, node.name, node.aliases, "
                "node.definition, node.entity_type, score"
            ),
            "ProjectLevel": (
                "node.code, node.name, "
                "node.project_classification, node.definition, score"
            ),
            "VolumeType": "node.code, node.name, node.description, score",
            "VolumeSubstance": (
                "node.code, node.name, node.substance, node.volume_category, score"
            ),
            "KSMIFrameworkNode": (
                "node.code, node.name, node.aliases, node.definition, score"
            ),
        }
        return_cols = return_cols_map.get(table, return_cols_map["KSMIEntity"])
        cypher = (
            f"CALL QUERY_FTS_INDEX('{table}', '{idx}', '{escaped}', "
            f"top := {top_k}) RETURN {return_cols}"
        )
        try:
            raw_rows = self._execute_cypher(cypher)
            rows: list[dict[str, Any]] = []
            for row in raw_rows:
                row_dict: dict[str, Any] = {"code": row[0], "name": row[1]}
                if table == "KSMIEntity":
                    row_dict["aliases"] = row[2]
                    row_dict["definition"] = row[3]
                    row_dict["entity_type"] = row[4]
                    row_dict["score"] = row[5]
                elif table == "ProjectLevel":
                    row_dict["project_classification"] = row[2]
                    row_dict["definition"] = row[3]
                    row_dict["score"] = row[4]
                elif table == "VolumeType":
                    row_dict["description"] = row[2]
                    row_dict["score"] = row[3]
                elif table == "VolumeSubstance":
                    row_dict["substance"] = row[2]
                    row_dict["volume_category"] = row[3]
                    row_dict["score"] = row[4]
                elif table == "KSMIFrameworkNode":
                    row_dict["aliases"] = row[2]
                    row_dict["definition"] = row[3]
                    row_dict["score"] = row[4]
                else:
                    msg = f"Unknown table for FTS: {table}"
                    raise ValueError(msg)
                rows.append(row_dict)
            return rows
        except Exception as e:
            logger.warning("[KSMI-KG] find_error | query=%s error=%s", query, e)
            return []

    def find_all(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        """Search across all tables and merge results by score.

        Args:
            query: Search string
            top_k: Max results per table

        Returns:
            Merged and deduplicated results sorted by score descending.
        """
        all_results: list[dict[str, Any]] = []
        seen_codes: set[str] = set()
        for table in [
            "KSMIEntity",
            "ProjectLevel",
            "VolumeType",
            "VolumeSubstance",
            "KSMIFrameworkNode",
        ]:
            results = self.find(query, top_k=top_k, table=table)
            for r in results:
                code = r.get("code", "")
                if code not in seen_codes:
                    seen_codes.add(code)
                    r["source_table"] = table
                    all_results.append(r)
        all_results.sort(key=lambda x: x.get("score", 0), reverse=True)
        return all_results

    def traverse(self, entity_code: str, rel_type: str) -> list[dict[str, Any]]:
        """Traverse relationships from an entity.

        Args:
            entity_code: Code of the source entity (e.g., "E0", "Reserves")
            rel_type: Relationship type (e.g., "CLASSIFIED_AS",
                      "REPORTED_AS", "CAN_TRANSITION_TO")

        Returns:
            List of target node properties.
        """
        self._ensure_initialized()
        escaped = entity_code.replace("'", "\\'")
        cypher = (
            f"MATCH (e {{code: '{escaped}'}})-[:{rel_type}]->(target) "
            f"RETURN target.code, target.name, "
            f"labels(target)[0] AS target_type"
        )
        try:
            raw_rows = self._execute_cypher(cypher)
            return [
                {"code": row[0], "name": row[1], "type": row[2]} for row in raw_rows
            ]
        except Exception as e:
            logger.warning(
                "[KSMI-KG] traverse_error | entity=%s rel=%s error=%s",
                entity_code,
                rel_type,
                e,
            )
            return []

    def traverse_reverse(self, entity_code: str, rel_type: str) -> list[dict[str, Any]]:
        """Traverse reverse relationships (incoming edges).

        Args:
            entity_code: Code of the target entity
            rel_type: Relationship type (traversed backwards)

        Returns:
            List of source node properties.
        """
        self._ensure_initialized()
        escaped = entity_code.replace("'", "\\'")
        cypher = (
            f"MATCH (source)-[:{rel_type}]->(target {{code: '{escaped}'}}) "
            f"RETURN source.code, source.name, "
            f"labels(source)[0] AS source_type"
        )
        try:
            raw_rows = self._execute_cypher(cypher)
            return [
                {"code": row[0], "name": row[1], "type": row[2]} for row in raw_rows
            ]
        except Exception as e:
            logger.warning(
                "[KSMI-KG] traverse_reverse_error | entity=%s rel=%s error=%s",
                entity_code,
                rel_type,
                e,
            )
            return []

    def query(self, cypher: str) -> list[dict[str, Any]]:
        """Execute arbitrary Cypher query on KSMI graph.

        Args:
            cypher: Cypher query string (must start with MATCH)

        Returns:
            List of result rows as dicts with string keys.
        """
        self._ensure_initialized()
        if not cypher.strip().upper().startswith("MATCH"):
            logger.warning("[KSMI-KG] query_rejected | non-MATCH cypher")
            return []
        try:
            raw_rows = self._execute_cypher(cypher)
            rows: list[dict[str, Any]] = []
            for row in raw_rows:
                row_dict = {str(i): v for i, v in enumerate(row)}
                rows.append(row_dict)
            return rows
        except Exception as e:
            logger.warning(
                "[KSMI-KG] query_error | cypher=%s error=%s",
                cypher[:80],
                e,
            )
            return []

    def find_levels(self, project_class: str | None = None) -> list[dict[str, Any]]:
        """Find project levels, optionally filtered by classification.

        Args:
            project_class: Filter by classification code
                          (e.g., "Reserves & GRR").
                          If None, returns all levels.

        Returns:
            List of project level properties.
        """
        self._ensure_initialized()
        if project_class:
            escaped = project_class.replace("'", "\\'")
            cypher = (
                f"MATCH (l:ProjectLevel)"
                f" WHERE l.project_classification CONTAINS '{escaped}'"
                f" RETURN l.code, l.name, l.is_pod_approved, "
                f"l.is_pse_approved, l.project_classification"
            )
        else:
            cypher = (
                "MATCH (l:ProjectLevel)"
                " RETURN l.code, l.name, l.is_pod_approved,"
                " l.is_pse_approved, l.project_classification"
            )
        try:
            raw_rows = self._execute_cypher(cypher)
            return [
                {
                    "code": row[0],
                    "name": row[1],
                    "is_pod_approved": row[2],
                    "is_pse_approved": row[3],
                    "project_classification": row[4],
                }
                for row in raw_rows
            ]
        except Exception as e:
            logger.warning("[KSMI-KG] find_levels_error | error=%s", e)
            return []

    def get_transitions(self, from_level: str | None = None) -> list[dict[str, Any]]:
        """Get level transition rules.

        Args:
            from_level: Source level code (e.g., "E0"). If None, returns all.

        Returns:
            List of transition dicts with from, to, condition, rule_note.
        """
        self._ensure_initialized()
        if from_level:
            escaped = from_level.replace("'", "\\'")
            cypher = (
                f"MATCH (from:ProjectLevel {{code: '{escaped}'}})"
                f"-[r:CAN_TRANSITION_TO]->(to:ProjectLevel)"
                f" RETURN from.code, to.code, r.condition, r.rule_note"
            )
        else:
            cypher = (
                "MATCH (from:ProjectLevel)"
                "-[r:CAN_TRANSITION_TO]->(to:ProjectLevel)"
                " RETURN from.code, to.code, r.condition, r.rule_note"
            )
        try:
            raw_rows = self._execute_cypher(cypher)
            return [
                {
                    "from": row[0],
                    "to": row[1],
                    "condition": row[2],
                    "rule_note": row[3],
                }
                for row in raw_rows
            ]
        except Exception as e:
            logger.warning("[KSMI-KG] get_transitions_error | error=%s", e)
            return []

    def format_reachability(
        self, highlight: str | None = None
    ) -> str:
        """Format reachability matrix as compact markdown.

        Reads CAN_TRANSITION_TO edges from the graph and renders a
        level-by-level table of allowed targets. Optionally highlights
        one source level with a marker for visual focus.

        A1 and A2 (Abandoned absorbing states) are always included even
        if they are not stored as ProjectLevel nodes in the graph.

        Args:
            highlight: Optional level code to mark (e.g., "E3").

        Returns:
            Formatted string covering all 18 levels (E0-E8, X0-X6, A1, A2).
        """
        self._ensure_initialized()
        transitions = self.get_transitions()

        from collections import defaultdict

        matrix: dict[str, set[str]] = defaultdict(set)
        for t in transitions:
            matrix[t["from"]].add(t["to"])

        for absorbing in ("A1", "A2"):
            if absorbing not in matrix:
                matrix[absorbing].add(absorbing)

        def sort_key(code: str) -> tuple[int, int]:
            if code.startswith("E"):
                return (0, int(code[1:]))
            if code.startswith("X"):
                return (1, int(code[1:]))
            return (2, int(code[1:]))

        lines = ["## Reachability Matrix (Level → Allowed Targets)", ""]
        for code in sorted(matrix.keys(), key=sort_key):
            targets = sorted(matrix[code], key=sort_key)
            prefix = ">>>" if code == highlight else "   "
            lines.append(f"{prefix} {code} → {', '.join(targets)}")
        return "\n".join(lines)

    def close(self) -> None:
        with KSMIGraphManager._lock:
            if self._initialized:
                self._builder.close()
                self._initialized = False
