"""Document metadata schema loader and renderers.

Loads the corpus document metadata schema from doc_schema.yaml — the single
source for the LLM extraction prompt (``METADATA_PROMPT``), the
``DOC_TYPES``/``DOC_LEVELS`` vocab, and the iris ``search_documents`` tool
context. Compiled once at import time for fast access.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

import yaml

_SCHEMA_PATH = Path(__file__).parent / "doc_schema.yaml"


@functools.lru_cache(maxsize=1)
def load_doc_schema() -> dict[str, Any]:
    """Load and cache the document metadata schema."""
    with open(_SCHEMA_PATH) as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict from YAML, got {type(data)}")
    return data


def llm_field_names() -> tuple[str, ...]:
    """Names of LLM-extracted fields, in prompt definition order."""
    return tuple(f["name"] for f in load_doc_schema().get("llm_fields", []))


def housekeeping_field_names() -> tuple[str, ...]:
    """Names of pipeline-set (non-LLM) fields."""
    return tuple(f["name"] for f in load_doc_schema().get("housekeeping_fields", []))


def enum_values(name: str) -> tuple[str, ...]:
    """Allowed values for a named enum (e.g. ``enum_values("doc_type")``)."""
    return tuple(load_doc_schema().get("enums", {}).get(name, []))


def render_prompt_definitions() -> str:
    """Render the ``- field: definition`` lines used in METADATA_PROMPT.

    Each line is transcribed verbatim from the schema's ``prompt`` value so
    the regenerated prompt stays byte-identical to the legacy literal.
    """
    lines = [f"- {f['name']}: {f['prompt']}" for f in load_doc_schema()["llm_fields"]]
    return "\n".join(lines)


def render_tool_context() -> str:
    """Render a compact, human/iris-readable field guide.

    One ``- field: description`` line per field (LLM + housekeeping), plus
    enum values. Used verbatim in the search_documents tool description.
    """
    schema = load_doc_schema()
    lines = ["Fields:"]
    for f in schema.get("llm_fields", []):
        lines.append(f"- {f['name']}: {f['description']}")
    for f in schema.get("housekeeping_fields", []):
        lines.append(f"- {f['name']}: {f['description']}")
    enums = schema.get("enums", {})
    if enums:
        lines.append("Enums:")
        for enum_name, values in enums.items():
            lines.append(f"- {enum_name}: {', '.join(values)}")
    return "\n".join(lines)
