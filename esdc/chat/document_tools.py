# Standard library
import json
from typing import Annotated

# Third-party
from langchain.tools import tool


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
    # TODO: Query LadybugDB for documents
    # Placeholder implementation - will be wired to LadybugDB later
    result = {
        "query": query,
        "entity_filter": entity_filter,
        "doc_type_filter": doc_type_filter,
        "temporal": temporal,
        "documents": [],
        "note": "Document search not yet implemented - will query LadybugDB",
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
    # TODO: Query LadybugDB for entity documents
    # Prioritize TITLE sections and latest documents
    # Placeholder implementation - will be wired to LadybugDB later
    result = {
        "entity": entity_name,
        "type": entity_type,
        "documents": [],
        "note": "Entity document context not yet implemented - will query LadybugDB",
    }
    return json.dumps(result, indent=2)
