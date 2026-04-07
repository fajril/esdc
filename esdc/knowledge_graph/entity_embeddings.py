"""Entity embedding store with persistence."""

import hashlib
import json
import logging
import pickle
from pathlib import Path
from typing import Any

from esdc.knowledge_graph.embeddings import EmbeddingGenerator

logger = logging.getLogger(__name__)


class EntityEmbeddingStore:
    """Store and manage embeddings for master entities with caching."""

    CACHE_DIR = Path.home() / ".esdc" / "cache"
    CACHE_FILE = "entity_embeddings.pkl"

    def __init__(self, embedding_generator: EmbeddingGenerator | None = None):
        """Initialize embedding store.

        Args:
            embedding_generator: Embedding generator instance. Creates one if None.
        """
        self.generator = embedding_generator or EmbeddingGenerator()
        self.embeddings: dict[str, list[float]] = {}
        self._metadata: dict[str, Any] = {}

    def load_or_create(
        self,
        entities: dict[str, list[dict]],
        force_recreate: bool = False,
    ) -> dict[str, list[list[float]]]:
        """Load embeddings from cache if exists and not stale, else generate.

        Args:
            entities: Dict of {entity_type: [{"name": ..., "esdc_id": ...}, ...]}
            force_recreate: Force regeneration even if cache exists

        Returns:
            Dict of {entity_type: [embedding1, embedding2, ...]}
            Order matches input entities order.
        """
        if force_recreate or self._is_cache_stale(entities):
            logger.info("Generating new entity embeddings...")
            embeddings = self._generate_embeddings(entities)
            self._save_cache()
            return embeddings

        logger.info("Loading cached entity embeddings...")
        if not self._load_cache():
            logger.warning("Cache load failed, regenerating...")
            embeddings = self._generate_embeddings(entities)
            self._save_cache()

        return self._get_embeddings_by_type(entities)

    def _generate_embeddings(
        self, entities: dict[str, list[dict]]
    ) -> dict[str, list[list[float]]]:
        """Generate embeddings for all entities.

        Args:
            entities: Dict of {entity_type: [{"name": ...}, ...]}

        Returns:
            Dict of {entity_type: [embeddings...]}
        """
        result: dict[str, list[list[float]]] = {}

        for entity_type, entity_list in entities.items():
            if not entity_list:
                result[entity_type] = []
                continue

            names = [e["name"] for e in entity_list]

            try:
                embeddings_list = self.generator.generate(names)
                result[entity_type] = embeddings_list

                # Store in flat dict for quick lookup
                for entity, embedding in zip(
                    entity_list, embeddings_list, strict=False
                ):
                    key = f"{entity_type}:{entity['name']}"
                    self.embeddings[key] = embedding

                logger.info(
                    f"Generated {len(embeddings_list)} embeddings for {entity_type}"
                )

            except Exception as e:
                logger.error(f"Failed to generate embeddings for {entity_type}: {e}")
                # Fallback: zero vectors
                result[entity_type] = [
                    [0.0] * self.generator.EMBEDDING_DIM for _ in entity_list
                ]

        # Store metadata
        self._metadata["entity_hash"] = self._compute_entity_hash(entities)
        self._metadata["embedding_dim"] = self.generator.EMBEDDING_DIM

        return result

    def get_embedding(self, entity_type: str, entity_name: str) -> list[float] | None:
        """Get embedding for a specific entity.

        Args:
            entity_type: Entity type (project, field, etc.)
            entity_name: Entity name

        Returns:
            Embedding vector or None if not found
        """
        key = f"{entity_type}:{entity_name}"
        return self.embeddings.get(key)

    def _is_cache_stale(self, entities: dict[str, list[dict]]) -> bool:
        """Check if cached embeddings are outdated.

        Args:
            entities: Current entity dict

        Returns:
            True if cache is stale or missing
        """
        cache_path = self.CACHE_DIR / self.CACHE_FILE

        if not cache_path.exists():
            return True

        try:
            with open(cache_path, "rb") as f:
                data = pickle.load(f)
                cached_hash = data.get("metadata", {}).get("entity_hash")

                current_hash = self._compute_entity_hash(entities)

                return cached_hash != current_hash

        except Exception as e:
            logger.warning(f"Failed to check cache staleness: {e}")
            return True

    def _compute_entity_hash(self, entities: dict[str, list[dict]]) -> str:
        """Compute hash of entity list for cache invalidation."""
        content = json.dumps(entities, sort_keys=True)
        return hashlib.md5(content.encode()).hexdigest()

    def _save_cache(self) -> None:
        """Persist embeddings to cache file."""
        try:
            self.CACHE_DIR.mkdir(parents=True, exist_ok=True)

            cache_path = self.CACHE_DIR / self.CACHE_FILE

            with open(cache_path, "wb") as f:
                pickle.dump(
                    {
                        "embeddings": self.embeddings,
                        "metadata": self._metadata,
                    },
                    f,
                )

            logger.info(f"Saved entity embeddings to {cache_path}")

        except Exception as e:
            logger.error(f"Failed to save cache: {e}")

    def _load_cache(self) -> bool:
        """Load embeddings from cache file.

        Returns:
            True if successful, False otherwise
        """
        try:
            cache_path = self.CACHE_DIR / self.CACHE_FILE

            with open(cache_path, "rb") as f:
                data = pickle.load(f)

            self.embeddings = data.get("embeddings", {})
            self._metadata = data.get("metadata", {})

            logger.info(f"Loaded {len(self.embeddings)} embeddings from cache")

            return bool(self.embeddings)

        except Exception as e:
            logger.error(f"Failed to load cache: {e}")
            return False

    def _get_embeddings_by_type(
        self, entities: dict[str, list[dict]]
    ) -> dict[str, list[list[float]]]:
        """Get embeddings grouped by type from cache.

        Args:
            entities: Entity dict for reference

        Returns:
            Dict of {entity_type: [embeddings...]}
        """
        result: dict[str, list[list[float]]] = {}

        for entity_type, entity_list in entities.items():
            embeddings_list = []

            for entity in entity_list:
                key = f"{entity_type}:{entity['name']}"
                embedding = self.embeddings.get(key)

                if embedding is None:
                    # Generate on-the-fly if missing
                    embedding = self.generator.generate_single(entity["name"])
                    self.embeddings[key] = embedding

                embeddings_list.append(embedding)

            result[entity_type] = embeddings_list

        return result

    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        cache_path = self.CACHE_DIR / self.CACHE_FILE

        if cache_path.exists():
            cache_path.unlink()
            logger.info(f"Cleared cache: {cache_path}")

        self.embeddings.clear()
        self._metadata.clear()
