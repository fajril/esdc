"""Data models for knowledge graph."""

from dataclasses import dataclass
from datetime import date


@dataclass
class Document:
    """Document node in the knowledge graph."""

    id: str
    title: str
    doc_type: str
    date: date | None
    file_path: str
    file_name: str
    is_timeless: bool
    extracted_at: str
    extraction_confidence: float


@dataclass
class Chunk:
    """Text chunk from a document."""

    id: str
    document_id: str
    chunk_index: int
    text: str
    section_type: str
    is_summary: bool


@dataclass
class Entity:
    """Entity extracted from document text."""

    id: str
    name: str
    normalized_name: str
    type: str
    source: str
    esdc_id: str | None = None
