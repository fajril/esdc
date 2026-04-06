"""Entity resolution from ESDC database."""

import sqlite3
from typing import Any

from esdc.configs import Config


class EntityResolver:
    """Resolve entity names to types by querying ESDC database."""

    def __init__(self, db_connection: sqlite3.Connection | None = None):
        """Initialize resolver with optional database connection.

        Args:
            db_connection: Optional database connection. If None, creates own connection.
        """
        self._owns_connection = db_connection is None
        if db_connection is not None:
            self._db = db_connection
        else:
            db_path = Config.get_chat_db_path()
            self._db = sqlite3.connect(db_path)

    def resolve(self, entity_name: str) -> dict[str, Any]:
        """Resolve entity name to type by querying ESDC DB.

        Args:
            entity_name: Entity name to resolve.

        Returns:
            Dict with keys:
                - type: Entity type (field, project, wk, operator, pod, custom)
                - name: Original entity name
                - source: "esdc_db" or "custom"
                - esdc_id: ESDC identifier if found (optional)
        """
        entity_type_mapping = [
            ("field_name", "field"),
            ("project_name", "project"),
            ("wk_name", "wk"),
            ("operator_name", "operator"),
            ("pod_name", "pod"),
        ]

        cursor = self._db.cursor()

        for column_name, entity_type in entity_type_mapping:
            query = f"""
                SELECT {column_name}, {column_name}
                FROM project_resources
                WHERE LOWER({column_name}) = LOWER(?)
                LIMIT 1
            """
            cursor.execute(query, (entity_name,))
            result = cursor.fetchone()

            if result is not None:
                esdc_id = result[0]
                return {
                    "type": entity_type,
                    "name": entity_name,
                    "source": "esdc_db",
                    "esdc_id": esdc_id,
                }

        cursor.close()

        return {
            "type": "custom",
            "name": entity_name,
            "source": "custom",
        }

    def close(self) -> None:
        """Close database connection if owned by this instance."""
        if self._owns_connection:
            self._db.close()
