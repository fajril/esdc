"""Knowledge Graph schema loader and validator.

Loads entity types, relationships, query patterns, and aliases from graph_schema.yaml.
Compiled once at startup for fast access.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

_SCHEMA_PATH = (
    Path(__file__).parent.parent / "chat" / "domain_knowledge" / "graph_schema.yaml"
)


class KGSchema:
    """Knowledge Graph schema loaded from YAML.

    Provides fast access to entity types, relationships, query patterns,
    and aliases compiled at startup.
    """

    def __init__(self, schema_path: Path | str | None = None) -> None:
        """Initialize the schema loader."""
        path = Path(schema_path) if schema_path else _SCHEMA_PATH
        if not path.exists():
            raise FileNotFoundError(f"KG schema file not found: {path}")
        self._data = self._load_yaml(path)
        self.entity_types: dict[str, dict[str, Any]] = self._data.get(
            "entity_types", {}
        )
        self.relationships: dict[str, dict[str, Any]] = self._data.get(
            "relationships", {}
        )
        self.query_patterns: dict[str, dict[str, Any]] = self._data.get(
            "query_patterns", {}
        )
        self.aliases: dict[str, dict[str, Any]] = self._data.get("aliases", {})
        self.enums: dict[str, dict[str, Any]] = self._data.get("enums", {}) or {}
        if not self.enums:
            et = self._data.get("entity_types", {})
            if isinstance(et, dict):
                self.enums = et.get("enums", {}) or {}
        self._compile_lookups()

    @staticmethod
    def _load_yaml(path: Path) -> dict[str, Any]:
        with open(path) as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            raise ValueError(f"Expected dict from YAML, got {type(data)}")
        logger.info("[KG] schema_loaded | path=%s", path)
        return data

    def _compile_lookups(self) -> None:
        self._entity_name_index: dict[str, str] = {}
        for entity_name, entity_def in self.entity_types.items():
            self._entity_name_index[entity_name.lower()] = entity_name
            if "description" in entity_def:
                for word in entity_def["description"].lower().split():
                    if len(word) > 3:
                        self._entity_name_index.setdefault(word, entity_name)

        self._pattern_keyword_index: dict[str, list[str]] = {}
        for pattern_name, pattern in self.query_patterns.items():
            for keyword in pattern.get("keywords", []):
                key = keyword.lower()
                if key not in self._pattern_keyword_index:
                    self._pattern_keyword_index[key] = []
                self._pattern_keyword_index[key].append(pattern_name)

    def get_entity_type(self, name: str) -> str | None:
        """Look up entity type by name using compiled index."""
        return self._entity_name_index.get(name.lower())

    def get_pattern_for_keywords(self, keywords: list[str]) -> list[dict[str, Any]]:
        """Find query patterns matching any of the given keywords."""
        matched_patterns: dict[str, dict[str, Any]] = {}
        for kw in keywords:
            key = kw.lower()
            for pattern_name in self._pattern_keyword_index.get(key, []):
                if pattern_name not in matched_patterns:
                    matched_patterns[pattern_name] = self.query_patterns[
                        pattern_name
                    ] | {"name": pattern_name}
        return list(matched_patterns.values())

    def get_primary_entity_type(self, pattern_name: str) -> str | None:
        """Get the primary entity type for a query pattern."""
        pattern = self.query_patterns.get(pattern_name)
        if pattern:
            return pattern.get("primary_entity")
        return None

    def get_suggested_table(self, pattern_name: str) -> str | None:
        """Get the suggested default table for a query pattern."""
        pattern = self.query_patterns.get(pattern_name)
        if pattern:
            return pattern.get("default_table")
        return None

    def get_suggested_columns(self, pattern_name: str) -> list[str]:
        """Get suggested columns for a query pattern."""
        pattern = self.query_patterns.get(pattern_name)
        if pattern:
            return pattern.get("suggested_columns", [])
        return []

    def get_enum_values(self, enum_name: str) -> list[str]:
        """Get enum values by name."""
        enum_def = self.enums.get(enum_name, {})
        return enum_def.get("values", [])
