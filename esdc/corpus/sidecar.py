"""Sidecar .corpus.md files: the human-review handoff between extract and commit."""

from pathlib import Path
from typing import Any

import yaml

DELIMITER = "---"


def sidecar_path(pdf_path: Path) -> Path:
    return pdf_path.with_suffix(".corpus.md")


def write_sidecar_file(path: Path, meta: dict[str, Any], body: str) -> Path:
    front = yaml.safe_dump(meta, allow_unicode=True, sort_keys=False).strip()
    # read_sidecar returns the body with a leading newline (from splitting on
    # the frontmatter delimiter); strip outer newlines so read->write cycles
    # are idempotent instead of accumulating blank lines around the body.
    body = body.strip("\n")
    path.write_text(f"{DELIMITER}\n{front}\n{DELIMITER}\n{body}\n", encoding="utf-8")
    return path


def write_sidecar(pdf_path: Path, meta: dict[str, Any], body: str) -> Path:
    return write_sidecar_file(sidecar_path(pdf_path), meta, body)


def read_sidecar(path: Path) -> tuple[dict[str, Any], str]:
    text = path.read_text(encoding="utf-8").lstrip()
    if not text.startswith(DELIMITER):
        raise ValueError(f"{path.name}: missing YAML frontmatter")
    try:
        _, front, body = text.split(DELIMITER, 2)
    except ValueError as e:
        raise ValueError(f"{path.name}: malformed frontmatter") from e
    try:
        meta = yaml.safe_load(front)
    except yaml.YAMLError as e:
        raise ValueError(f"{path.name}: invalid YAML in frontmatter: {e}") from e
    if not isinstance(meta, dict) or "file_hash" not in meta:
        raise ValueError(f"{path.name}: frontmatter must be a mapping with file_hash")
    return meta, body
