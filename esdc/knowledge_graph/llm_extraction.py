"""LLM-based document structure extraction."""

import json
import re
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

EXTRACTION_PROMPT = """Analyze the following document and extract its structure.

Document filename: {filename}

Document content:
{content}

Extract and return a JSON object with the following structure:
{{
    "doc_type": "One of: DEFINITION, MoM, SK, POD, PERATURAN, TECHNICAL_REPORT",
    "confidence": <float between 0 and 1>,
    "date": "<ISO date if found, null if timeless or not found>",
    "date_source": "<where date was found: 'content', 'filename', or null>",
    "is_timeless": <true if document is a timeless reference, false otherwise>,
    "sections": {{
        "<section_name>": {{
            "text": "<extracted section content>",
            "confidence": <float between 0 and 1>
        }}
    }},
    "title": "<document title>"
}}

Important:
- Identify the document type based on content structure and keywords
- Extract sections with their content and confidence scores
- For timeless documents (definitions, glossaries), set is_timeless to true
- Extract date only if clearly stated, otherwise set to null
- Return ONLY the JSON object, no other text"""


class LLMExtractor:
    """Extracts document structure using LLM."""

    def __init__(self, llm: BaseChatModel | None = None):
        """Initialize the extractor.

        Args:
            llm: Optional LangChain chat model. If not provided,
                 will need to be set before extraction.
        """
        self.llm = llm

    def extract_structure(self, content: str, filename: str) -> dict[str, Any]:
        """Extract document structure using LLM.

        Args:
            content: Document content
            filename: Document filename

        Returns:
            Dictionary with extracted structure containing:
            - doc_type: Document type
            - confidence: Confidence score
            - date: ISO date if found
            - date_source: Where date was found
            - is_timeless: Whether document is timeless
            - sections: Extracted sections with text and confidence
            - title: Document title

        Raises:
            ValueError: If LLM not initialized or JSON parsing fails
        """
        if self.llm is None:
            raise ValueError("LLM not initialized. Provide llm parameter.")

        prompt = EXTRACTION_PROMPT.format(filename=filename, content=content)
        response = self.llm.invoke(prompt)

        response_content = response.content
        if isinstance(response_content, list):
            response_content = " ".join(
                str(item) if isinstance(item, str) else str(item)
                for item in response_content
            )

        json_str = self._extract_json(str(response_content))
        if json_str is None:
            raise ValueError(
                f"Failed to parse LLM response as JSON. Response: {response_content}"
            )

        try:
            result = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON from LLM response: {e}") from e

        return self._normalize_result(result)

    def _extract_json(self, content: str) -> str | None:
        """Extract JSON from LLM response.

        Args:
            content: LLM response content

        Returns:
            JSON string if found, None otherwise
        """
        content = content.strip()

        if content.startswith("{"):
            return content

        json_match = re.search(r"\{[\s\S]*\}", content)
        if json_match:
            return json_match.group(0)

        return None

    def _normalize_result(self, result: dict[str, Any]) -> dict[str, Any]:
        """Normalize extracted result with defaults.

        Args:
            result: Raw extraction result

        Returns:
            Normalized result with all required fields
        """
        return {
            "doc_type": result.get("doc_type", "TECHNICAL_REPORT"),
            "confidence": result.get("confidence", 0.5),
            "date": result.get("date"),
            "date_source": result.get("date_source"),
            "is_timeless": result.get("is_timeless", False),
            "sections": result.get("sections", {}),
            "title": result.get("title", ""),
        }
