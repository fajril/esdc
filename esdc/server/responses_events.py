# Standard library
import json
import time
from typing import Any

# Local


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
            "output": output,
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
