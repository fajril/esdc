"""SSE event payload builders for streaming responses."""

# Standard library
import copy
import json
import time
from typing import Any

# Local

# aiohttp-based SSE clients (e.g. Open WebUI) reject any single stream line
# longer than 131072 bytes (2x the 64 KiB StreamReader read buffer).
SSE_LINE_BYTE_LIMIT = 131_072
# Budget for the serialized output list in terminal events, leaving headroom
# for the event envelope, the "data: " prefix, and JSON string escaping.
MAX_SSE_EVENT_BYTES = 100_000
# Cap for tool-result text embedded in function_call_output items. The LLM
# consumes the full result internally; the SSE copy is display/citation only.
MAX_TOOL_RESULT_TEXT_CHARS = 40_000
TRUNCATION_SUFFIX = "... [truncated]"
# Stub length used when slimming oversized terminal events.
_SLIM_TEXT_CHARS = 500


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + TRUNCATION_SUFFIX


def create_function_call_output_item(
    item_id: str,
    call_id: str,
    text: str,
    max_text_chars: int = MAX_TOOL_RESULT_TEXT_CHARS,
) -> dict[str, Any]:
    """Build a function_call_output item with size-capped result text.

    Uncapped tool results (document search dumps, query output) can push a
    single SSE event past SSE_LINE_BYTE_LIMIT and break streaming clients.
    """
    return {
        "id": item_id,
        "type": "function_call_output",
        "status": "completed",
        "call_id": call_id,
        "output": [{"type": "input_text", "text": _truncate(text, max_text_chars)}],
    }


def _output_byte_size(output: list[dict[str, Any]]) -> int:
    return len(json.dumps(output).encode("utf-8"))


def slim_output_items(
    output: list[dict[str, Any]],
    max_bytes: int = MAX_SSE_EVENT_BYTES,
) -> list[dict[str, Any]]:
    """Shrink bulky non-message fields until output fits within max_bytes.

    Terminal events (response.completed / response.incomplete) re-send the
    full accumulated output as one SSE line; over long agent runs this can
    exceed what streaming clients will read. Truncates, in order:
    function_call_output texts, reasoning summaries, function_call arguments.
    Message text is never touched -- it is what the user sees. Returns the
    input unchanged when it already fits; never mutates the input.
    """
    if _output_byte_size(output) <= max_bytes:
        return output

    slimmed = copy.deepcopy(output)

    fco_fields: list[tuple[dict[str, Any], str]] = []
    reasoning_fields: list[tuple[dict[str, Any], str]] = []
    args_fields: list[tuple[dict[str, Any], str]] = []
    for item in slimmed:
        item_type = item.get("type")
        if item_type == "function_call_output":
            for part in item.get("output", []):
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    fco_fields.append((part, "text"))
        elif item_type == "reasoning":
            for part in item.get("summary", []):
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    reasoning_fields.append((part, "text"))
        elif item_type == "function_call" and isinstance(item.get("arguments"), str):
            args_fields.append((item, "arguments"))

    for fields in (fco_fields, reasoning_fields, args_fields):
        for container, key in fields:
            container[key] = _truncate(container[key], _SLIM_TEXT_CHARS)
        if _output_byte_size(slimmed) <= max_bytes:
            break
    return slimmed


def format_sse_event(event: dict[str, Any]) -> str:
    r"""Format an event dict as an SSE string.

    Per Open Responses spec:
    - event field MUST match the type in the event body
    - data is JSON-encoded
    - Terminal event is literal [DONE]

    Args:
        event: Event dictionary with 'type' field

    Returns:
        SSE-formatted string: "event: <type>\ndata: <json>\n\n"
    """
    event_type = event.get("type", "unknown")
    return f"event: {event_type}\ndata: {json.dumps(event)}\n\n"


# =============================================================================
# Response Lifecycle Events
# =============================================================================


def create_response_created_event(
    response_id: str,
    model: str,
    sequence_number: int = 1,
) -> dict[str, Any]:
    """Event: response.created.

    Emitted when a response starts.
    Per spec: The first event in a response stream.
    """
    return {
        "type": "response.created",
        "sequence_number": sequence_number,
        "response": {
            "id": response_id,
            "object": "response",
            "created_at": time.time(),
            "model": model,
            "status": "in_progress",
            "output": [],
        },
    }


def create_response_completed_event(
    sequence_number: int,
    response_id: str,
    model: str,
    output: list[dict[str, Any]],
    usage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Event: response.completed.

    Emitted when a response finishes successfully.
    """
    return {
        "type": "response.completed",
        "sequence_number": sequence_number,
        "response": {
            "id": response_id,
            "object": "response",
            "created_at": time.time(),
            "model": model,
            "status": "completed",
            "output": slim_output_items(output),
            "usage": usage,
        },
    }


def create_response_failed_event(
    sequence_number: int,
    response_id: str,
    model: str,
    error: dict[str, Any],
) -> dict[str, Any]:
    """Event: response.failed.

    Emitted when a response encounters an error.
    """
    return {
        "type": "response.failed",
        "sequence_number": sequence_number,
        "response": {
            "id": response_id,
            "object": "response",
            "created_at": time.time(),
            "model": model,
            "status": "failed",
            "output": [],
            "error": error,
        },
    }


def create_response_incomplete_event(
    sequence_number: int,
    response_id: str,
    model: str,
    output: list[dict[str, Any]],
    error: dict[str, Any],
) -> dict[str, Any]:
    """Event: response.incomplete.

    Emitted when a response is interrupted but has partial output.
    Per Open Responses spec and OpenWebUI v0.9.0: clients should
    display partial results and offer regeneration.
    """
    return {
        "type": "response.incomplete",
        "sequence_number": sequence_number,
        "response": {
            "id": response_id,
            "object": "response",
            "created_at": time.time(),
            "model": model,
            "status": "incomplete",
            "output": slim_output_items(output),
            "error": error,
        },
    }


# =============================================================================
# Output Item Events
# =============================================================================


def create_output_item_added_event(
    sequence_number: int,
    output_index: int,
    item: dict[str, Any],
) -> dict[str, Any]:
    """Event: response.output_item.added.

    Emitted when a new output item starts.
    Per spec: First event for each item, contains item with minimal fields.
    """
    return {
        "type": "response.output_item.added",
        "sequence_number": sequence_number,
        "output_index": output_index,
        "item": item,
    }


def create_output_item_done_event(
    sequence_number: int,
    output_index: int,
    item: dict[str, Any],
) -> dict[str, Any]:
    """Event: response.output_item.done.

    Emitted when an output item is complete.
    Per spec: Final event for each item, contains complete item.
    """
    return {
        "type": "response.output_item.done",
        "sequence_number": sequence_number,
        "output_index": output_index,
        "item": item,
    }


# =============================================================================
# Content Part Events (for message items)
# =============================================================================


def create_content_part_added_event(
    sequence_number: int,
    output_index: int,
    content_index: int,
    item_id: str,
    part: dict[str, Any],
) -> dict[str, Any]:
    """Event: response.content_part.added.

    Emitted when a content part starts within a message item.
    Per spec: MUST be emitted before any delta events for that content.
    """
    return {
        "type": "response.content_part.added",
        "sequence_number": sequence_number,
        "item_id": item_id,
        "output_index": output_index,
        "content_index": content_index,
        "part": part,
    }


def create_output_text_delta_event(
    sequence_number: int,
    output_index: int,
    content_index: int,
    item_id: str,
    delta: str,
) -> dict[str, Any]:
    """Event: response.output_text.delta.

    Emitted for each chunk of text output.
    Per spec: Delta event representing incremental text change.
    """
    return {
        "type": "response.output_text.delta",
        "sequence_number": sequence_number,
        "output_index": output_index,
        "content_index": content_index,
        "item_id": item_id,
        "delta": delta,
    }


def create_output_text_done_event(
    sequence_number: int,
    output_index: int,
    content_index: int,
    item_id: str,
    text: str,
) -> dict[str, Any]:
    """Event: response.output_text.done.

    Emitted when a text content part is complete.
    Per spec: Contains the full accumulated text.
    """
    return {
        "type": "response.output_text.done",
        "sequence_number": sequence_number,
        "output_index": output_index,
        "content_index": content_index,
        "item_id": item_id,
        "text": text,
    }


def create_content_part_done_event(
    sequence_number: int,
    output_index: int,
    content_index: int,
    item_id: str,
    part: dict[str, Any],
) -> dict[str, Any]:
    """Event: response.content_part.done.

    Emitted when a content part is complete.
    Per spec: Final event for a content part, contains complete part.
    """
    return {
        "type": "response.content_part.done",
        "sequence_number": sequence_number,
        "item_id": item_id,
        "output_index": output_index,
        "content_index": content_index,
        "part": part,
    }


# =============================================================================
# Function Call Events
# =============================================================================


def create_function_call_arguments_delta_event(
    sequence_number: int,
    output_index: int,
    item_id: str,
    delta: str,
) -> dict[str, Any]:
    """Event: response.function_call_arguments.delta.

    Emitted for each chunk of function call arguments (JSON string).
    """
    return {
        "type": "response.function_call_arguments.delta",
        "sequence_number": sequence_number,
        "output_index": output_index,
        "item_id": item_id,
        "delta": delta,
    }


def create_function_call_arguments_done_event(
    sequence_number: int,
    output_index: int,
    item_id: str,
    arguments: str,
) -> dict[str, Any]:
    """Event: response.function_call_arguments.done.

    Emitted when function call arguments are complete.
    Per spec: Contains the full JSON arguments string.
    """
    return {
        "type": "response.function_call_arguments.done",
        "sequence_number": sequence_number,
        "output_index": output_index,
        "item_id": item_id,
        "arguments": arguments,
    }


# =============================================================================
# Reasoning Summary Events (Open Responses API spec)
# =============================================================================


def create_reasoning_summary_text_delta_event(
    sequence_number: int,
    output_index: int,
    content_index: int,
    item_id: str,
    delta: str,
) -> dict[str, Any]:
    """Event: response.reasoning_summary_text.delta.

    Emitted for each chunk of reasoning summary text.
    Per Open Responses API spec, reasoning items contain summary
    text parts that explain the model's thinking process.
    """
    return {
        "type": "response.reasoning_summary_text.delta",
        "sequence_number": sequence_number,
        "output_index": output_index,
        "content_index": content_index,
        "item_id": item_id,
        "delta": delta,
    }


def create_reasoning_summary_text_done_event(
    sequence_number: int,
    output_index: int,
    content_index: int,
    item_id: str,
    text: str,
) -> dict[str, Any]:
    """Event: response.reasoning_summary_text.done.

    Emitted when a reasoning summary text part is complete.
    Contains the full accumulated reasoning text.
    """
    return {
        "type": "response.reasoning_summary_text.done",
        "sequence_number": sequence_number,
        "output_index": output_index,
        "content_index": content_index,
        "item_id": item_id,
        "text": text,
    }


# =============================================================================
# Non-Streaming Response Helper
# =============================================================================


def create_non_streaming_response(
    response_id: str,
    model: str,
    output: list[dict[str, Any]],
    usage: dict[str, Any] | None = None,
    error: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a complete non-streaming response object.

    This is used for non-streaming requests where the entire response
    is returned at once.
    """
    result = {
        "id": response_id,
        "object": "response",
        "created_at": time.time(),
        "model": model,
        "status": "failed" if error else "completed",
        "output": output,
        "usage": usage,
    }
    if error:
        result["error"] = error
    return result
