"""Shared message utilities for LangGraph event handling.

This module contains common functions used by both Chat Completions API
(agent_wrapper.py) and Responses API (responses_wrapper.py).
"""

from typing import Any

from langchain_core.messages import AIMessage


def extract_ai_message_from_event(event: dict[str, Any]) -> AIMessage | None:
    """Extract AIMessage from LangGraph event.

    LangGraph events have structure: {node_name: {messages: [...]}}.
    We look for the 'agent' node which contains AI responses.

    Args:
        event: LangGraph event dictionary with node data

    Returns:
        AIMessage if found in event, None otherwise

    Example:
        >>> event = {"agent": {"messages": [AIMessage(content="Hello")]}}
        >>> msg = extract_ai_message_from_event(event)
        >>> msg.content
        'Hello'
    """
    if "agent" in event:
        agent_data = event["agent"]
        if isinstance(agent_data, dict) and "messages" in agent_data:
            messages = agent_data["messages"]
            if messages:
                last_msg = messages[-1]
                if isinstance(last_msg, AIMessage):
                    return last_msg

    for _key, value in event.items():
        if isinstance(value, dict) and "messages" in value:
            messages = value["messages"]
            if messages:
                last_msg = messages[-1]
                if isinstance(last_msg, AIMessage):
                    return last_msg

    return None


def extract_content_str(content: Any) -> str:
    """Extract string content from AIMessage content.

    Handles various content formats used by different LLM providers:
    - str: Direct string content
    - list: List of content parts (strings, dicts, or objects)
    - dict: Dictionary content (formatted as JSON)

    Args:
        content: Content from AIMessage (can be str, list, dict, or None)

    Returns:
        Extracted string representation of content

    Example:
        >>> extract_content_str("Hello")
        'Hello'
        >>> extract_content_str([{"type": "text", "text": "Hello"}])
        'Hello'
    """
    if content is None:
        return ""

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                text = part.get("text") or part.get("content") or ""
                parts.append(str(text))
            elif hasattr(part, "text"):
                parts.append(str(part.text))
            else:
                parts.append(str(part) if part else "")
        return " ".join(parts) if parts else ""

    if isinstance(content, dict):
        import json

        return json.dumps(content, indent=2)

    return str(content)
