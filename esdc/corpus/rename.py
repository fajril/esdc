"""Rename corpus source files (and their sidecars) to a canonical structure.

Target name: ``DOC_TYPE - YYYY.MM.DD - title.<ext>`` where DOC_TYPE is
``doc_type``, the date is the ISO ``doc_date`` (issue date) rendered with
dots, and title is ``subject``. The three fields are resolved per file
through a ladder: existing ``.corpus.md`` sidecar frontmatter -> committed
corpus DB (by file_hash) -> LLM/OCR inference (with the filename passed as
a hint). Dry-run is the default; on-disk renames happen only when the CLI
passes ``apply=True``.
"""

import datetime
import hashlib
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_ILLEGAL_TITLE_CHARS = re.compile(r'[\\/:*?"<>|\x00-\x1f]')
_MAX_TITLE_CHARS = 150


def format_doc_date(value: Any) -> str | None:
    """Render an ISO date (str or date/datetime) as ``YYYY.MM.DD``; None if invalid."""
    if value is None or value == "":
        return None
    if isinstance(value, (datetime.date, datetime.datetime)):
        d: datetime.date = value
    else:
        try:
            d = datetime.date.fromisoformat(str(value).strip())
        except ValueError:
            return None
    return f"{d.year:04d}.{d.month:02d}.{d.day:02d}"


def sanitize_title(title: Any) -> str | None:
    """Make a filesystem-safe title fragment; None when empty after cleaning."""
    if not title:
        return None
    cleaned = _ILLEGAL_TITLE_CHARS.sub(" ", str(title))
    cleaned = re.sub(r"\s+", " ", cleaned).strip().strip(".").strip()
    if not cleaned:
        return None
    return cleaned[:_MAX_TITLE_CHARS].strip()
