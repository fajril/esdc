"""EntityResolver query-pattern schema loader and validator."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

_SCHEMA_PATH = Path(__file__).parent / "entity_query_schema.yaml"


class KGSchema:
    """EntityResolver query guidance loaded from YAML."""

    def __init__(self, schema_path: Path | str | None = None) -> None:
        """Initialize the schema loader."""
        path = Path(schema_path) if schema_path else _SCHEMA_PATH
        if not path.exists():
            raise FileNotFoundError(f"KG schema file not found: {path}")
        self._data = self._load_yaml(path)
        self.schema_version = self._data.get("schema_version")
        if self.schema_version != 1:
            raise ValueError(
                f"{path}: schema_version must be 1, got {self.schema_version!r}"
            )
        query_patterns = self._data.get("query_patterns")
        if not isinstance(query_patterns, dict):
            raise ValueError(f"{path}: query_patterns must be a mapping")
        if not query_patterns:
            raise ValueError(f"{path}: query_patterns must not be empty")
        for name, pattern in query_patterns.items():
            if not isinstance(name, str) or not isinstance(pattern, dict):
                raise ValueError(f"{path}: query pattern {name!r} must be a mapping")
            keywords = pattern.get("keywords")
            if not isinstance(keywords, list) or not all(
                isinstance(keyword, str) and keyword for keyword in keywords
            ):
                raise ValueError(f"{path}: query pattern {name!r} keywords invalid")
        self.query_patterns: dict[str, dict[str, Any]] = query_patterns

    @staticmethod
    def _load_yaml(path: Path) -> dict[str, Any]:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            raise ValueError(f"{path}: expected a YAML mapping, got {type(data)}")
        logger.info("[KG] schema_loaded | path=%s", path)
        return data

    def get_pattern_for_keywords(self, keywords: list[str]) -> list[dict[str, Any]]:
        """Find query patterns matching any of the given keywords."""
        keys = {keyword.lower() for keyword in keywords}
        return [
            pattern | {"name": name}
            for name, pattern in self.query_patterns.items()
            if any(keyword.lower() in keys for keyword in pattern["keywords"])
        ]

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
