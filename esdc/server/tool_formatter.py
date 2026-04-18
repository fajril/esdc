"""Tool call formatting for OpenAI-compatible native tool calling."""

# Standard library
import json
import os
import time
import uuid
from typing import Any


def should_use_native_format(headers: dict[str, str], stream: bool) -> bool:
    """Determine whether to use native tool calling format.

    Priority:
    1. Environment variable ESDC_TOOL_FORMAT (highest priority)
    2. Config file tool_format setting
    3. Auto-detect based on headers (lowest priority)

    Valid values for both env var and config:
    - "native": Force native format
    - "markdown": Force markdown format
    - "auto": Auto-detect based on headers
    """
    # Priority 1: Environment variable
    env_format = os.getenv("ESDC_TOOL_FORMAT", "").lower()
    if env_format == "native":
        return True
    elif env_format == "markdown":
        return False

    # Priority 2: Config file (import here to avoid circular import)
    from esdc.configs import Config

    config_format = Config.get_tool_format()
    if config_format == "native":
        return True
    elif config_format == "markdown":
        return False

    # Priority 3: Auto-detect
    return detect_native_format(headers, stream)


def create_tool_call_chunk(
    tool_calls: list[dict[str, Any]], model: str = "iris"
) -> dict[str, Any]:
    """Create OpenAI-compatible streaming chunk with tool calls.

    Args:
        tool_calls: List of tool call dicts with 'name', 'args', 'id' keys
        model: Model identifier

    Returns:
        OpenAI-compatible chat.completion.chunk dict
    """
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {
                    "tool_calls": [
                        {
                            "index": i,
                            "id": tc.get("id") or f"call_{uuid.uuid4().hex[:8]}",
                            "type": "function",
                            "function": {
                                "name": tc.get("name", "unknown"),
                                "arguments": json.dumps(tc.get("args", {})),
                            },
                        }
                        for i, tc in enumerate(tool_calls)
                    ]
                },
                "finish_reason": None,
            }
        ],
    }


def format_tool_calls_for_response(
    tool_calls: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Format tool calls for non-streaming response.

    Args:
        tool_calls: List of tool call dicts

    Returns:
        List of OpenAI-compatible tool call dicts
    """
    return [
        {
            "id": tc.get("id") or f"call_{uuid.uuid4().hex[:8]}",
            "type": "function",
            "function": {
                "name": tc.get("name", "unknown"),
                "arguments": json.dumps(tc.get("args", {})),
            },
        }
        for tc in tool_calls
    ]


def detect_native_format(headers: dict[str, str], stream: bool) -> bool:
    """Detect if client supports native tool calling format.

    Detection strategy:
    - SSE streaming clients (text/event-stream) support native
    - JSON clients on non-streaming support native
    - Plain text or unknown clients get markdown fallback

    Args:
        headers: Request headers dict
        stream: Whether this is a streaming request

    Returns:
        True if native format should be used, False for markdown
    """
    accept = headers.get("accept", "").lower()

    return ("text/event-stream" in accept) or (
        "application/json" in accept and not stream
    )


def create_tool_role_chunk(
    tool_call_id: str,
    content: str,
    model: str = "iris",
    max_content_length: int = 1000,
) -> dict[str, Any]:
    """Create OpenAI-compatible streaming chunk for tool role message.

    This is sent after tool execution to satisfy OpenWebUI's expectation
    that tool_calls should be followed by tool role messages.

    Args:
        tool_call_id: ID of the tool call this result corresponds to
        content: Tool execution result (will be truncated if too long)
        model: Model identifier
        max_content_length: Maximum content length before truncation

    Returns:
        OpenAI-compatible chat.completion.chunk dict with tool role
    """
    # Truncate content if too long to avoid huge payloads
    if len(content) > max_content_length:
        content = content[:max_content_length] + "... [truncated]"

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": content,
                },
                "finish_reason": None,
            }
        ],
    }


def create_final_chunk(model: str = "iris") -> dict[str, Any]:
    """Create final SSE chunk indicating completion.

    Args:
        model: Model identifier

    Returns:
        Final completion chunk
    """
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
