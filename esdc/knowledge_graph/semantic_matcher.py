"""Semantic entity matching using embeddings."""

import logging
import math

from esdc.knowledge_graph.embeddings import EmbeddingGenerator

logger = logging.getLogger(__name__)


class SemanticEntityMatcher:
    """Match entities using semantic similarity."""

    EXTERNAL_ENTITIES: dict[str, list[dict]] = {
        "ministry": [
            {"name": "Kementerian ESDM", "esdc_id": None},
            {"name": "Kementerian Keuangan", "esdc_id": None},
        ],
        "directorate": [
            {"name": "Ditjen Migas", "esdc_id": None},
        ],
    }

    def __init__(
        self,
        embedding_generator: EmbeddingGenerator | None = None,
        threshold: float = 0.7,
    ):
        """Initialize matcher.

        Args:
            embedding_generator: Embedding generator instance
            threshold: Similarity threshold for matching (default: 0.7)
        """
        self.generator = embedding_generator or EmbeddingGenerator()
        self.threshold = threshold

    def find_matches(
        self,
        document_text: str,
        master_entities: dict[str, list[dict]],
        master_embeddings: dict[str, list[list[float]]],
    ) -> list[dict]:
        """Find all entities mentioned in document text.

        Args:
            document_text: Full document text
            master_entities: Dict of {type: [{"name": ..., "esdc_id": ...}]}
            master_embeddings: Dict of {type: [embeddings...]}

        Returns:
            List of matched entities:
            [
                {
                    "name": "Arung Nowera",
                    "type": "project",
                    "esdc_id": "...",
                    "similarity": 0.92,
                    "source": "esdc_db"
                },
                ...
            ]
        """
        doc_embedding = self.generator.generate_single(document_text)

        matches = []

        # Match against ESDC database entities
        for entity_type, entities_list in master_entities.items():
            embeddings_list = master_embeddings.get(entity_type, [])

            for i, entity in enumerate(entities_list):
                if i >= len(embeddings_list):
                    continue

                embedding = embeddings_list[i]
                similarity = self.cosine_similarity(doc_embedding, embedding)

                if similarity >= self.threshold:
                    matches.append(
                        {
                            "name": entity["name"],
                            "type": entity_type,
                            "esdc_id": entity.get("esdc_id"),
                            "similarity": round(similarity, 3),
                            "source": "esdc_db",
                        }
                    )

        # Match against external entities
        for entity_type, entities_list in self.EXTERNAL_ENTITIES.items():
            for entity in entities_list:
                entity_embedding = self.generator.generate_single(entity["name"])
                similarity = self.cosine_similarity(doc_embedding, entity_embedding)

                if similarity >= self.threshold:
                    matches.append(
                        {
                            "name": entity["name"],
                            "type": entity_type,
                            "esdc_id": entity.get("esdc_id"),
                            "similarity": round(similarity, 3),
                            "source": "external",
                        }
                    )

        # Sort by similarity descending
        matches.sort(key=lambda x: x["similarity"], reverse=True)

        return matches

    def find_top_k(
        self,
        document_text: str,
        master_entities: dict[str, list[dict]],
        master_embeddings: dict[str, list[list[float]]],
        k: int = 20,
    ) -> list[dict]:
        """Find top-k candidate entities for LLM verification.

        Args:
            document_text: Full document text
            master_entities: Dict of {type: [{"name": ..., "esdc_id": ...}]}
            master_embeddings: Dict of {type: [embeddings...]}
            k: Number of top candidates to return

        Returns:
            Top-k candidates regardless of threshold
        """
        doc_embedding = self.generator.generate_single(document_text)

        candidates = []

        # Get all entities with similarities
        for entity_type, entities_list in master_entities.items():
            embeddings_list = master_embeddings.get(entity_type, [])

            for i, entity in enumerate(entities_list):
                if i >= len(embeddings_list):
                    continue

                embedding = embeddings_list[i]
                similarity = self.cosine_similarity(doc_embedding, embedding)

                candidates.append(
                    {
                        "name": entity["name"],
                        "type": entity_type,
                        "esdc_id": entity.get("esdc_id"),
                        "similarity": round(similarity, 3),
                        "source": "esdc_db",
                    }
                )

        # Add external entities
        for entity_type, entities_list in self.EXTERNAL_ENTITIES.items():
            for entity in entities_list:
                entity_embedding = self.generator.generate_single(entity["name"])
                similarity = self.cosine_similarity(doc_embedding, entity_embedding)

                candidates.append(
                    {
                        "name": entity["name"],
                        "type": entity_type,
                        "esdc_id": entity.get("esdc_id"),
                        "similarity": round(similarity, 3),
                        "source": "external",
                    }
                )

        # Sort and take top-k
        candidates.sort(key=lambda x: x["similarity"], reverse=True)

        return candidates[:k]

    @staticmethod
    def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
        """Calculate cosine similarity between two vectors.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Cosine similarity score between 0 and 1
        """
        if not vec1 or not vec2:
            return 0.0

        if len(vec1) != len(vec2):
            logger.warning(f"Vector dimension mismatch: {len(vec1)} vs {len(vec2)}")
            return 0.0

        dot_product = sum(a * b for a, b in zip(vec1, vec2, strict=False))

        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(b * b for b in vec2))

        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0

        return dot_product / (magnitude1 * magnitude2)
