"""Load the extraction guideline and build the extraction prompt.

The YAML file is the user's contract with the extractor: it defines what
entity types, claim types, and per-doc-type focus the LLM should apply.
Its content hash is part of every document's learn-state hash, so editing
the guideline re-triggers extraction corpus-wide.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

_DEFAULT_PATH = Path(__file__).parent / "guideline.yaml"


@dataclass(frozen=True)
class Guideline:
    version: int
    entity_types: dict[str, str]
    claim_types: dict[str, str]
    doc_type_hints: dict[str, str]
    content_hash: str


def default_guideline_path() -> Path:
    """User-level guideline (~/.esdc/guideline.yaml) wins over the packaged one.

    The user file is created by hand or by `esdc corpus learn --init-guideline`.
    Keeping it in the db dir means it migrates to the server together with
    esdc.sqlite / esdc.duckdb.
    """
    from esdc.configs import Config

    user_path = Config.get_db_dir() / "guideline.yaml"
    return user_path if user_path.exists() else _DEFAULT_PATH


def load_guideline(path: Path | None = None) -> Guideline:
    p = path or default_guideline_path()
    raw = p.read_bytes()
    data: dict[str, Any] = yaml.safe_load(raw) or {}
    return Guideline(
        version=int(data.get("version", 1)),
        entity_types=dict(data.get("entity_types") or {}),
        claim_types=dict(data.get("claim_types") or {}),
        doc_type_hints=dict(data.get("doc_type_hints") or {}),
        content_hash=hashlib.sha256(raw).hexdigest(),
    )


def build_extraction_prompt(
    guideline: Guideline,
    doc_meta: dict[str, Any],
    markdown: str,
    max_chars: int = 24000,
) -> str:
    entity_lines = "\n".join(
        f"- {name}: {desc}" for name, desc in guideline.entity_types.items()
    )
    claim_lines = "\n".join(
        f"- {name}: {desc}" for name, desc in guideline.claim_types.items()
    )
    doc_type = str(doc_meta.get("doc_type") or "")
    hint = guideline.doc_type_hints.get(doc_type, "")
    body = markdown[:max_chars]
    return f"""You extract structured knowledge from Indonesian upstream \
oil & gas documents (letters, MoM, regulations). Documents mix Indonesian \
and English.

Document metadata:
- type: {doc_type}
- date: {doc_meta.get("doc_date") or "unknown"}
- subject: {doc_meta.get("subject") or "unknown"}
{f"Focus for this document type: {hint}" if hint else ""}

Entity types you may reference:
{entity_lines}

Claim types you may emit:
{claim_lines}

Return ONLY a JSON object, no prose, with exactly these keys:
{{
  "entities": [
    {{"type": "<one of the entity types>", "name": "<name as written>"}}
  ],
  "claims": [
    {{"type": "<one of the claim types>",
      "subject": "<entity name the claim is about, or null>",
      "subject_type": "<entity type of subject, or null>",
      "predicate": "<snake_case predicate>",
      "value": <number | string | object>,
      "evidence": "<short verbatim quote, max 25 words>"}}
  ],
  "unknown_types": [
    {{"kind": "claim_type" | "entity_type", "name": "<snake_case>",
      "why": "<one sentence>"}}
  ]
}}

Rules:
- Only emit claims stated in the document. Never infer or compute.
- Keep numbers with their units in the predicate (e.g. npv_musd).
- If a recurring fact fits no claim type, put it in unknown_types instead
  of forcing a wrong type.
- Empty lists are fine.

Document markdown:
---
{body}
---"""
