"""Hybrid entity extraction combining semantic search and LLM verification."""

import json
import logging
import re
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

from esdc.knowledge_graph.embeddings import EmbeddingGenerator
from esdc.knowledge_graph.semantic_matcher import SemanticEntityMatcher

logger = logging.getLogger(__name__)

VERIFICATION_PROMPT = """You are analyzing a document from the Indonesian oil & gas sector.

Candidate entities from database (matched by semantic similarity):
{candidates}

Document text (first 3000 characters):
{document_text}

Task:
1. Review each candidate entity carefully
2. Determine which entities are ACTUALLY mentioned in the document
3. Consider:
   - Exact matches ("Arung Nowera", "Pertamina")
   - Abbreviations ("ESDM" = "Kementerian ESDM")
   - Partial mentions ("Arung" might refer to "Arung Nowera" or "Arung Field")
   - Context cues (is this entity relevant to this document?)

Return JSON array of matched entities:
[
    {{
        "name": "<exact entity name from candidates>",
        "type": "<entity type>",
        "confidence": <float 0-1>,
        "evidence": "<brief quote from document>"
    }}
]

IMPORTANT:
- Only include entities with confidence >= 0.5
- Use exact names from candidates list
- Provide evidence from document text
- Return ONLY the JSON array, no other text"""


class HybridEntityExtractor:
    """Two-stage entity extraction: semantic candidates + LLM verification."""

    def __init__(
        self,
        semantic_matcher: SemanticEntityMatcher,
        llm: BaseChatModel | None = None,
        top_k: int = 20,
    ):
        """Initialize hybrid extractor.

        Args:
            semantic_matcher: Semantic matcher for candidate generation
            llm: Optional LLM for verification (falls back to semantic-only if None)
            top_k: Number of top candidates to consider
        """
        self.matcher = semantic_matcher
        self.llm = llm
        self.top_k = top_k

    def extract(
        self,
        document_text: str,
        master_entities: dict[str, list[dict]],
        master_embeddings: dict[str, list[list[float]]],
    ) -> list[dict]:
        """Extract entities via two-stage process.

        Args:
            document_text: Full document text
            master_entities: Dict of {type: [{"name": ..., "esdc_id": ...}]}
            master_embeddings: Dict of {type: [embeddings...]}

        Returns:
            List of matched entities:
            [
                {
                    "name": "...",
                    "type": "...",
                    "esdc_id": "...",
                    "similarity": 0.92,
                    "confidence": 0.95,
                    "source": "esdc_db" | "external",
                    "evidence": "..."
                }
            ]
        """
        # Stage 1: Semantic candidate generation
        candidates = self.matcher.find_top_k(
            document_text=document_text,
            master_entities=master_entities,
            master_embeddings=master_embeddings,
            k=self.top_k,
        )

        if not candidates:
            logger.info("No semantic candidates found")
            return []

        # Stage 2: LLM verification (or fall back to semantic-only)
        if self.llm is None:
            logger.info("No LLM provided, using semantic-only matching")
            return self._filter_by_threshold(candidates)

        try:
            verified = self._verify_with_llm(document_text, candidates)
            return verified
        except Exception as e:
            logger.error(f"LLM verification failed: {e}, falling back to semantic-only")
            return self._filter_by_threshold(candidates)

    def _verify_with_llm(
        self, document_text: str, candidates: list[dict]
    ) -> list[dict]:
        """Use LLM to verify candidate entities.

        Args:
            document_text: Full document text
            candidates: Top-k candidates from semantic search

        Returns:
            Verified entities with confidence scores
        """
        # Format candidates for LLM
        candidates_str = self._format_candidates(candidates)

        # Truncate document text
        doc_text = document_text[:3000]

        prompt = VERIFICATION_PROMPT.format(
            candidates=candidates_str, document_text=doc_text
        )

        # Call LLM
        response = self.llm.invoke(prompt)
        response_content = response.content

        if isinstance(response_content, list):
            response_content = " ".join(str(item) for item in response_content)

        # Parse JSON
        json_str = self._extract_json(str(response_content))

        if json_str is None:
            logger.warning("Failed to parse LLM response as JSON")
            return self._filter_by_threshold(candidates)

        try:
            verified_entities = json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error: {e}")
            return self._filter_by_threshold(candidates)

        # Match verified entities back to candidates
        result = []
        for verified in verified_entities:
            # Find matching candidate
            matching_candidate = self._find_matching_candidate(verified, candidates)

            if matching_candidate:
                result.append(
                    {
                        "name": verified.get("name", matching_candidate["name"]),
                        "type": verified.get("type", matching_candidate["type"]),
                        "esdc_id": matching_candidate.get("esdc_id"),
                        "similarity": matching_candidate.get("similarity"),
                        "confidence": verified.get("confidence", 0.8),
                        "source": matching_candidate.get("source"),
                        "evidence": verified.get("evidence", ""),
                    }
                )

        return result

    def _format_candidates(self, candidates: list[dict]) -> str:
        """Format candidates for LLM prompt.

        Args:
            candidates: List of candidate entities

        Returns:
            Formatted string
        """
        lines = []
        for i, candidate in enumerate(candidates, 1):
            source = candidate.get("source", "unknown")
            similarity = candidate.get("similarity", 0.0)
            entity_type = candidate.get("type", "unknown")
            name = candidate.get("name", "unknown")
            lines.append(
                f"{i}. {name} ({entity_type}) - [{source}] - similarity: {similarity:.2f}"
            )

        return "\n".join(lines)

    def _filter_by_threshold(self, candidates: list[dict]) -> list[dict]:
        """Filter candidates by similarity threshold.

        Args:
            candidates: List of candidate entities

        Returns:
            Filtered entities above threshold
        """
        return [
            {
                "name": c["name"],
                "type": c["type"],
                "esdc_id": c.get("esdc_id"),
                "similarity": c["similarity"],
                "confidence": c["similarity"],  # Use similarity as confidence
                "source": c.get("source"),
                "evidence": "",
            }
            for c in candidates
            if c.get("similarity", 0) >= self.matcher.threshold
        ]

    def _extract_json(self, content: str) -> str | None:
        """Extract JSON array from LLM response.

        Args:
            content: LLM response content

        Returns:
            JSON string if found, None otherwise
        """
        content = content.strip()

        # Try to find JSON array
        json_match = re.search(r"\[[\s\S]*\]", content)
        if json_match:
            return json_match.group(0)

        return None

    def _find_matching_candidate(
        self, verified: dict, candidates: list[dict]
    ) -> dict | None:
        """Find matching candidate for verified entity.

        Args:
            verified: Verified entity from LLM
            candidates: List of candidates

        Returns:
            Matching candidate or None
        """
        name = verified.get("name", "").strip().lower()
        entity_type = verified.get("type", "").strip().lower()

        for candidate in candidates:
            cand_name = candidate.get("name", "").strip().lower()
            cand_type = candidate.get("type", "").strip().lower()

            # Exact match
            if name == cand_name and entity_type == cand_type:
                return candidate

            # Partial match (verified name is substring of candidate)
            if name in cand_name and entity_type == cand_type:
                return candidate

        return None
