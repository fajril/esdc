"""External tool categorization for hybrid tool execution.

Separates tools into internal (ESDC) and external (OpenTerminal etc.)
categories for the Responses API tool passthrough feature.

When OpenWebUI sends a /responses request with `tools` parameter containing
OpenTerminal tool definitions (run_command, write_file, etc.), ESDC categorizes
them as external tools. Internal tools (execute_sql, get_schema, etc.) are
executed server-side as before. External tools are not executed by ESDC;
instead, when the LLM calls an external tool, a marker is returned which
is converted to a function_call output item for OpenWebUI to handle.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("esdc.chat.external_tools")

INTERNAL_TOOL_NAMES: set[str] = {
    "knowledge_traversal",
    "resolve_spatial",
    "semantic_search",
    "execute_cypher",
    "execute_sql",
    "get_schema",
    "list_tables",
    "get_recommended_table",
    "resolve_uncertainty_level",
    "search_problem_cluster",
    "get_timeseries_columns",
    "get_resources_columns",
    "list_available_models",
    "Compute Engine",
    "File Processing",
    "View File",
}

EXTERNAL_TOOL_MARKER_PREFIX = "[EXTERNAL_TOOL_CALL:"


def categorize_tools(
    tools: list[dict[str, Any]] | None = None,
) -> tuple[set[str], set[str], list[dict[str, Any]]]:
    """Categorize tools into internal (ESDC) and external (OpenTerminal etc.).

    Args:
        tools: List of tool definitions from the Responses API request.
            Each tool is a dict with format:
            {"type": "function", "function": {"name": "...", ...}}
            If None, returns all internal tools and no external tools.

    Returns:
        Tuple of:
        - internal_names: Set of internal tool names (always includes all ESDC tools)
        - external_names: Set of external tool names from the request
        - external_specs: List of external tool specs (for binding to LLM)
    """
    internal_names = INTERNAL_TOOL_NAMES.copy()
    external_names: set[str] = set()
    external_specs: list[dict[str, Any]] = []

    if tools is None:
        return internal_names, external_names, external_specs

    for tool in tools:
        func_def = tool.get("function", tool)
        name = func_def.get("name", "")

        if name and name not in INTERNAL_TOOL_NAMES:
            external_names.add(name)
            external_specs.append(tool)

    logger.debug(
        "[EXTERNAL_TOOLS] Categorized: internal=%d, external=%d, external_names=%s",
        len(internal_names),
        len(external_names),
        sorted(external_names),
    )

    return internal_names, external_names, external_specs


def convert_external_specs_to_langchain(
    external_specs: list[dict[str, Any]],
) -> list:
    """Convert OpenAI-style function specs to LangChain tool format.

    Creates StructuredTool objects that can be bound to LLMs but whose
    execution is handled by the external_tool marker in tool_node.
    The actual execution never happens — when the LLM calls an external
    tool, tool_node returns a marker string which event_streamer detects
    and converts to a function_call output item.
    """
    from langchain_core.tools import StructuredTool
    from pydantic import create_model

    langchain_tools: list = []

    for spec in external_specs:
        func = spec.get("function", spec)
        name = func.get("name", "")
        description = func.get("description", "")
        parameters = func.get("parameters", {"type": "object", "properties": {}})

        properties = parameters.get("properties", {})
        required_fields = parameters.get("required", [])

        field_definitions: dict[str, Any] = {}
        for param_name, param_schema in properties.items():
            param_type: type = str
            if param_schema.get("type") == "integer":
                param_type = int
            elif param_schema.get("type") == "number":
                param_type = float
            elif param_schema.get("type") == "boolean":
                param_type = bool

            is_required = param_name in required_fields
            if is_required:
                field_definitions[param_name] = (
                    param_type,
                    param_schema.get("description", ""),
                )
            else:
                field_definitions[param_name] = (
                    param_type | None,
                    param_schema.get("description", ""),
                )

        args_model = create_model(f"{name}Args", **field_definitions)

        def _make_external_fn(tool_name: str):
            async def _external_tool_callback(**kwargs):
                return f"{EXTERNAL_TOOL_MARKER_PREFIX}{tool_name}]"

            return _external_tool_callback

        tool = StructuredTool.from_function(
            coroutine=_make_external_fn(name),
            name=name,
            description=description,
            args_schema=args_model,
        )
        langchain_tools.append(tool)

    logger.debug(
        "[EXTERNAL_TOOLS] Converted %d external specs to LangChain tools: %s",
        len(langchain_tools),
        [t.name for t in langchain_tools],
    )

    return langchain_tools


def is_external_tool_marker(result: str) -> bool:
    """Check if a tool result is an external tool call marker."""
    return result.strip().startswith(EXTERNAL_TOOL_MARKER_PREFIX)


def parse_external_tool_name(result: str) -> str | None:
    """Extract the external tool name from a marker result string.

    Args:
        result: Tool result string, possibly containing an external tool marker.

    Returns:
        Tool name if the result is an external tool marker, None otherwise.
    """
    result = result.strip()
    if not result.startswith(EXTERNAL_TOOL_MARKER_PREFIX):
        return None
    inner = result[len(EXTERNAL_TOOL_MARKER_PREFIX) : -1]
    return inner if inner else None
