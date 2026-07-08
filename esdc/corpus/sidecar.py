"""Sidecar .corpus.md files: the human-review handoff between extract and commit."""

import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

DELIMITER = "---"


def sidecar_path(pdf_path: Path) -> Path:
    return pdf_path.with_suffix(".corpus.md")


def write_sidecar(pdf_path: Path, meta: dict[str, Any], body: str) -> Path:
    path = sidecar_path(pdf_path)
    front = yaml.safe_dump(meta, allow_unicode=True, sort_keys=False).strip()
    path.write_text(f"{DELIMITER}\n{front}\n{DELIMITER}\n{body}\n", encoding="utf-8")
    return path


def read_sidecar(path: Path) -> tuple[dict[str, Any], str]:
    text = path.read_text(encoding="utf-8")
    if not text.startswith(DELIMITER):
        raise ValueError(f"{path.name}: missing YAML frontmatter")
    try:
        _, front, body = text.split(DELIMITER, 2)
    except ValueError as e:
        raise ValueError(f"{path.name}: malformed frontmatter") from e
    meta = yaml.safe_load(front)
    if not isinstance(meta, dict) or "file_hash" not in meta:
        raise ValueError(f"{path.name}: frontmatter must be a mapping with file_hash")
    return meta, body
