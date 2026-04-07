# Standard library
import json
from typing import Annotated

# Third-party
from langchain.tools import tool

# Local
from esdc.knowledge_graph.graph_db import LadybugDB


def _get_ladybug_db() -> LadybugDB:
    """Get LadybugDB instance."""
    return LadybugDB()


@tool("Document Search")
def search_documents(
    query: Annotated[str, "Natural language query"],
    entity_filter: Annotated[str | None, "Optional entity name"] = None,
    doc_type_filter: Annotated[str | None, "Optional doc type"] = None,
    temporal: Annotated[str, "'latest', 'all', or date range"] = "latest",
) -> str:
    """Search knowledge graph for documents.

    Use when user asks about document content, updates, history.

    Args:
        query: Natural language query about documents
        entity_filter: Optional entity name to filter by
        doc_type_filter: Optional document type (e.g., "MoM", "Report")
        temporal: Temporal filter ('latest', 'all', or date range)

    Returns:
        JSON string with query, filters, and matching documents
    """
    db = _get_ladybug_db()

    if not db.is_available():
        result = {
            "query": query,
            "entity_filter": entity_filter,
            "doc_type_filter": doc_type_filter,
            "temporal": temporal,
            "documents": [],
            "available": False,
            "note": (
                "Document knowledge graph not available. "
                "Install real-ladybug to enable."
            ),
        }
        return json.dumps(result, indent=2)

    if entity_filter:
        entity_type = _guess_entity_type(entity_filter)
        documents = db.get_entity_documents(
            entity_name=entity_filter,
            entity_type=entity_type,
            temporal=temporal,
        )

        if doc_type_filter:
            documents = [d for d in documents if d.get("doc_type") == doc_type_filter]
    else:
        documents = []

    result = {
        "query": query,
        "entity_filter": entity_filter,
        "doc_type_filter": doc_type_filter,
        "temporal": temporal,
        "documents": documents,
        "available": True,
    }
    return json.dumps(result, indent=2)


@tool("Entity Document Context")
def get_entity_document_context(
    entity_name: Annotated[str, "Entity name"],
    entity_type: Annotated[str, "Entity type: project, field, wk, operator, pod"],
) -> str:
    """Get document context for entity.

    Use after SQL query to enrich with document context.

    Args:
        entity_name: Name of the entity (e.g., "Abadi", "Rokan")
        entity_type: Type of entity (project, field, wk, operator, pod)

    Returns:
        JSON string with entity info and related documents
    """
    db = _get_ladybug_db()

    if not db.is_available():
        result = {
            "entity": entity_name,
            "type": entity_type,
            "documents": [],
            "available": False,
            "note": (
                "Document knowledge graph not available. "
                "Install real-ladybug to enable."
            ),
        }
        return json.dumps(result, indent=2)

    documents = db.get_entity_documents(
        entity_name=entity_name,
        entity_type=entity_type,
        temporal="latest",
    )

    title_sections = []
    other_sections = []

    for doc in documents:
        if doc.get("doc_type") in ["DEFINITION", "MoM"]:
            title_sections.append(doc)
        else:
            other_sections.append(doc)

    prioritized_docs = title_sections[:3] + other_sections[:7]

    result = {
        "entity": entity_name,
        "type": entity_type,
        "documents": prioritized_docs,
        "total_found": len(documents),
        "available": True,
    }
    return json.dumps(result, indent=2)


def _guess_entity_type(entity_name: str) -> str:
    """Guess entity type from name based on patterns.

    This is a simple heuristic. For better accuracy, use the EntityResolver.

    Args:
        entity_name: Entity name to analyze

    Returns:
        Guessed entity type (project, field, wk, operator, pod)
    """
    name_lower = entity_name.lower()

    wk_patterns = ["wk", "working area", "wilayah kerja"]
    if any(p in name_lower for p in wk_patterns):
        return "wk"

    operator_patterns = ["pt ", "cv ", "persero", "ltd"]
    if any(p in name_lower for p in operator_patterns):
        return "operator"

    pod_patterns = ["pod", "plan of development"]
    if any(p in name_lower for p in pod_patterns):
        return "pod"

    field_patterns = ["field", "lapangan"]
    if any(p in name_lower for p in field_patterns):
        return "field"

    return "project"
