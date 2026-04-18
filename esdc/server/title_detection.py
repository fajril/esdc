"""Detect OpenWebUI title-generation requests.

OpenWebUI sends these as stream=False requests with a single
user message starting with '### Task:' containing title generation
instructions. They should bypass the full agent pipeline.
"""

import time
import uuid


def is_title_generation_request(input_messages: str | list) -> bool:
    """Check if a request is an OpenWebUI title-generation request.

    Title requests have the pattern:
    - Single user message (input=list(1) or string)
    - Content starts with '### Task:'
    - Contains 'title' keyword

    Returns True only for title-generation requests.
    """
    text = _extract_text_from_input(input_messages)
    if not text:
        return False
    text = text.strip()
    return text.startswith("### Task:") and "title" in text.lower()


def extract_user_query_from_title_request(input_messages: str | list) -> str:
    r"""Extract the original user query from a title-generation request.

    The title request format is:
    '### Task:\nGenerate a ...title...\n\nUser: <actual query>\n\nTitle:'

    Returns the user query portion, or the full text as fallback.
    """
    text = _extract_text_from_input(input_messages)
    if not text:
        return ""

    if "User:" in text:
        user_start = text.find("User:")
        query_part = text[user_start + 5 :].strip()
        if "\n\nTitle:" in query_part:
            query_part = query_part[: query_part.find("\n\nTitle:")].strip()
        return query_part

    return text


def create_title_sync_response(title: str, response_id: str) -> dict:
    """Create a non-streaming response object for title generation."""
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


def create_title_stream_events(
    title: str, response_id: str, model: str = "iris"
) -> list[str]:
    """Create SSE events for a title generation streaming response.

    Returns a list of SSE-formatted event strings that can be yielded
    by the streaming endpoint.
    """
    events = []
    seq = 1

    # response.created
    events.append(f"event: response.created\ndata: {uuid.uuid4().hex}\n\n")
    seq += 1

    # output_item.added (message)
    events.append(f"event: response.output_item.added\ndata: {uuid.uuid4().hex}\n\n")
    seq += 1

    # content_part.added
    events.append(f"event: response.content_part.added\ndata: {uuid.uuid4().hex}\n\n")
    seq += 1

    # output_text.delta (full title as single chunk)
    import json

    delta_event = {
        "type": "response.output_text.delta",
        "output_index": 0,
        "content_index": 0,
        "delta": title,
    }
    events.append(
        f"event: response.output_text.delta\ndata: {json.dumps(delta_event)}\n\n"
    )
    seq += 1

    # output_text.done
    done_event = {
        "type": "response.output_text.done",
        "output_index": 0,
        "content_index": 0,
        "text": title,
    }
    events.append(
        f"event: response.output_text.done\ndata: {json.dumps(done_event)}\n\n"
    )
    seq += 1

    # content_part.done
    events.append(f"event: response.content_part.done\ndata: {uuid.uuid4().hex}\n\n")
    seq += 1

    # output_item.done (message)
    events.append(f"event: response.output_item.done\ndata: {uuid.uuid4().hex}\n\n")
    seq += 1

    return events


def _extract_text_from_input(input_messages: str | list) -> str:
    """Extract text content from various input formats."""
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
