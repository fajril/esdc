"""Detect OpenWebUI ancillary requests (title/tag generation).

OpenWebUI sends these as stream=False requests with a single
user message starting with '### Task:'. They should bypass the
full agent pipeline to avoid loading the 18K system prompt,
running query classification, and potentially executing tools.

Two types of ancillary requests:
- Title: "Generate a concise, 3-5 word title..."
- Tags: "Generate 1-3 broad tags categorizing..."
"""

import json
import time
import uuid


def is_ancillary_request(input_messages: str | list) -> bool:
    """Check if a request is an OpenWebUI ancillary request.

    Matches any single-message input starting with '### Task:'.
    This covers title generation, tag generation, and any future
    OpenWebUI ancillary request types.

    Returns True only for ancillary requests that should bypass
    the agent pipeline.
    """
    text = _extract_text_from_input(input_messages)
    if not text:
        return False
    return text.strip().startswith("### Task:")


def is_title_generation_request(input_messages: str | list) -> bool:
    """Backward-compatible alias for is_ancillary_request.

    Previously this only matched title requests. Now it matches
    all ### Task: requests (title, tags, and future types).
    """
    return is_ancillary_request(input_messages)


def get_ancillary_type(input_messages: str | list) -> str | None:
    """Determine the type of ancillary request.

    Returns:
        "title" if this is a title generation request
        "tags" if this is a tag generation request
        "other" if it's an unknown ### Task: type
        None if not an ancillary request
    """
    text = _extract_text_from_input(input_messages)
    if not text or not text.strip().startswith("### Task:"):
        return None

    lower = text.lower()
    if "title" in lower:
        return "title"
    if "tag" in lower:
        return "tags"
    return "other"


def extract_user_query(input_messages: str | list) -> str:
    r"""Extract the original user query from an ancillary request.

    The request format is:
    '### Task:\n<instructions>\n\nUser: <actual query>\n\nTitle:'

    Works for both title and tag generation requests.

    Returns the user query portion, or the full text as fallback.
    """
    text = _extract_text_from_input(input_messages)
    if not text:
        return ""

    if "User:" in text:
        user_start = text.find("User:")
        query_part = text[user_start + 5 :].strip()
        for ending in ("\n\nTitle:", "\n\nTags:"):
            if ending in query_part:
                query_part = query_part[: query_part.find(ending)].strip()
        return query_part

    return text


def extract_user_query_from_title_request(input_messages: str | list) -> str:
    """Backward-compatible alias for extract_user_query."""
    return extract_user_query(input_messages)


def create_title_sync_response(title: str, response_id: str) -> dict:
    """Create a non-streaming Responses API response for title generation."""
    msg_id = f"msg_{uuid.uuid4().hex[:24]}"
    return {
        "id": response_id,
        "object": "response",
        "created_at": time.time(),
        "model": "iris",
        "status": "completed",
        "output": [
            {
                "type": "message",
                "id": msg_id,
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": title,
                    }
                ],
            }
        ],
        "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
    }


def create_tags_sync_response(tags: str, response_id: str) -> dict:
    """Create a non-streaming Responses API response for tag generation."""
    msg_id = f"msg_{uuid.uuid4().hex[:24]}"
    return {
        "id": response_id,
        "object": "response",
        "created_at": time.time(),
        "model": "iris",
        "status": "completed",
        "output": [
            {
                "type": "message",
                "id": msg_id,
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": tags,
                    }
                ],
            }
        ],
        "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
    }


def create_title_stream_events(
    title: str, response_id: str, model: str = "iris"
) -> list[str]:
    """Create SSE events for a title generation streaming response.

    Returns a list of SSE-formatted event strings that can be yielded
    by the streaming endpoint.
    """
    return _create_ancillary_stream_events(title, response_id, model)


def create_tags_stream_events(
    tags: str, response_id: str, model: str = "iris"
) -> list[str]:
    """Create SSE events for a tag generation streaming response."""
    return _create_ancillary_stream_events(tags, response_id, model)


def create_ancillary_chat_response(content: str) -> dict:
    """Create a Chat Completions response for ancillary requests.

    Returns an OpenAI-compatible non-streaming response dict.
    """
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "iris",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                },
                "finish_reason": "stop",
            }
        ],
    }


def create_ancillary_chat_stream_chunks(content: str) -> list[str]:
    """Create Chat Completions streaming chunks for ancillary requests.

    Returns a list of JSON-encoded chunk strings.
    """
    chunk_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    chunks = []

    content_chunk = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": "iris",
        "choices": [
            {
                "index": 0,
                "delta": {"content": content},
                "finish_reason": None,
            }
        ],
    }
    chunks.append(json.dumps(content_chunk))

    done_chunk = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": "iris",
        "choices": [
            {
                "index": 0,
                "delta": {},
                "finish_reason": "stop",
            }
        ],
    }
    chunks.append(json.dumps(done_chunk))

    return chunks


def _create_ancillary_stream_events(
    text: str, response_id: str, model: str = "iris"
) -> list[str]:
    """Create SSE events for any ancillary streaming response."""
    events = []

    events.append(f"event: response.created\ndata: {uuid.uuid4().hex}\n\n")
    events.append(f"event: response.output_item.added\ndata: {uuid.uuid4().hex}\n\n")
    events.append(f"event: response.content_part.added\ndata: {uuid.uuid4().hex}\n\n")

    delta_event = {
        "type": "response.output_text.delta",
        "output_index": 0,
        "content_index": 0,
        "delta": text,
    }
    events.append(
        f"event: response.output_text.delta\ndata: {json.dumps(delta_event)}\n\n"
    )

    done_event = {
        "type": "response.output_text.done",
        "output_index": 0,
        "content_index": 0,
        "text": text,
    }
    events.append(
        f"event: response.output_text.done\ndata: {json.dumps(done_event)}\n\n"
    )

    events.append(f"event: response.content_part.done\ndata: {uuid.uuid4().hex}\n\n")
    events.append(f"event: response.output_item.done\ndata: {uuid.uuid4().hex}\n\n")

    return events


def _extract_text_from_input(input_messages: str | list) -> str:
    """Extract text content from various input formats.

    Handles:
    - Plain strings
    - Responses API format: list of dicts with 'content' field
    - Chat Completions format: list of Message-like objects with 'content' field
    """
    if isinstance(input_messages, str):
        return input_messages

    if isinstance(input_messages, list):
        if len(input_messages) != 1:
            return ""

        item = input_messages[0]
        content = (
            item.get("content", "")
            if isinstance(item, dict)
            else getattr(item, "content", "")
        )

        if isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, dict):
                    parts.append(part.get("text", ""))
                elif isinstance(part, str):
                    parts.append(part)
                elif hasattr(part, "text"):
                    parts.append(str(part.text))
            return " ".join(parts)

        return str(content) if content else ""

    return ""
