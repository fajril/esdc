"""Load master entities from ESDC database."""

import logging
import sqlite3

from esdc.configs import Config
from esdc.knowledge_graph.embeddings import EmbeddingGenerator
from esdc.knowledge_graph.entity_embeddings import EntityEmbeddingStore

logger = logging.getLogger(__name__)


class MasterEntityLoader:
    """Load master entities from ESDC database with embeddings."""

    EXTERNAL_ENTITIES: dict[str, list[dict]] = {
        "ministry": [
            {"name": "Kementerian ESDM", "esdc_id": None},
            {"name": "Kementerian Keuangan", "esdc_id": None},
            {"name": "Kementerian ESDM", "esdc_id": None},
            {"name": "ESDM", "esdc_id": None},
            {"name": "Kementerian Energi dan Sumber Daya Mineral", "esdc_id": None},
        ],
        "directorate": [
            {"name": "Ditjen Migas", "esdc_id": None},
            {"name": "Direktorat Jenderal Minyak dan Gas Bumi", "esdc_id": None},
        ],
    }

    ENTITY_TYPE_MAPPING: list[tuple[str, str]] = [
        ("field_name", "field"),
        ("project_name", "project"),
        ("wk_name", "wk"),
        ("pod_name", "pod"),
        ("operator_name", "operator"),
    ]

    def __init__(
        self,
        db_connection: sqlite3.Connection | None = None,
        embedding_generator: EmbeddingGenerator | None = None,
    ):
        """Initialize master entity loader.

        Args:
            db_connection: Optional database connection
            embedding_generator: Optional embedding generator
        """
        self._owns_connection = db_connection is None
        self._db = db_connection or self._get_default_connection()
        self.embedding_gen = embedding_generator or EmbeddingGenerator()
        self.store = EntityEmbeddingStore(self.embedding_gen)

    def _get_default_connection(self) -> sqlite3.Connection:
        """Get default database connection."""
        db_path = Config.get_chat_db_path()
        return sqlite3.connect(db_path)

    def load_all(self) -> dict[str, list[dict]]:
        """Load all entities from project_resources table.

        Returns:
            Dict of {entity_type: [{"name": ..., "esdc_id": ...}, ...]}
        """
        entities: dict[str, list[dict]] = {}

        cursor = self._db.cursor()

        for column_name, entity_type in self.ENTITY_TYPE_MAPPING:
            query = f"""
                SELECT DISTINCT {column_name}
                FROM project_resources
                WHERE {column_name} IS NOT NULL
                  AND {column_name} != ''
                ORDER BY {column_name}
            """

            cursor.execute(query)
            results = cursor.fetchall()

            entities[entity_type] = [
                {"name": row[0], "esdc_id": row[0]} for row in results if row[0]
            ]

            logger.info(f"Loaded {len(entities[entity_type])} {entity_type} entities")

        cursor.close()

        # Add external entities
        for entity_type, external_list in self.EXTERNAL_ENTITIES.items():
            if entity_type not in entities:
                entities[entity_type] = []

            entities[entity_type].extend(external_list)
            entities[entity_type] = self._deduplicate_entities(entities[entity_type])

        return entities

    def load_all_with_embeddings(
        self, force_recreate: bool = False
    ) -> tuple[dict[str, list[dict]], dict[str, list[list[float]]]]:
        """Load entities and generate or load embeddings.

        Args:
            force_recreate: Force embedding regeneration

        Returns:
            Tuple of:
                - Dict of {entity_type: [{"name": ..., "esdc_id": ...}]}
                - Dict of {entity_type: [embeddings...]}
        """
        entities = self.load_all()
        embeddings = self.store.load_or_create(entities, force_recreate)

        return entities, embeddings

    def count_entities(self) -> dict[str, int]:
        """Count total entities per type.

        Returns:
            Dict of {entity_type: count}
        """
        entities = self.load_all()
        return {
            entity_type: len(entity_list)
            for entity_type, entity_list in entities.items()
        }

    def close(self) -> None:
        """Close database connection if owned."""
        if self._owns_connection:
            self._db.close()

    def _deduplicate_entities(self, entities: list[dict]) -> list[dict]:
        """Remove duplicate entities by name.

        Args:
            entities: List of entity dicts

        Returns:
            Deduplicated list
        """
        seen = set()
        result = []

        for entity in entities:
            name = entity.get("name")
            if name and name not in seen:
                seen.add(name)
                result.append(entity)

        return result
