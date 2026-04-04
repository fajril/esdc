# Standard library
import json
import logging
import uuid
from collections.abc import AsyncGenerator, Generator
from typing import Any

# Third-party
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

# Local
from esdc.chat.agent import create_agent
from esdc.configs import Config
from esdc.providers import create_llm_from_config
from esdc.server.responses_events import (
    create_content_part_added_event,
    create_content_part_done_event,
    create_function_call_arguments_delta_event,
    create_function_call_arguments_done_event,
    create_output_item_added_event,
    create_output_item_done_event,
    create_output_text_delta_event,
    create_output_text_done_event,
    create_response_completed_event,
    create_response_created_event,
    format_sse_event,
)
from esdc.server.responses_models import ResponseInputItem

logger = logging.getLogger("esdc.server.responses")


class SequenceCounter:
    """Manages monotonically increasing sequence numbers for SSE events."""

    def __init__(self, start: int = 1):
        self._value = start

    def next(self) -> int:
        """Get next sequence number."""
        current = self._value
        self._value += 1
        return current


def convert_responses_input_to_langchain(
    input_data: str | list[ResponseInputItem],
    instructions: str | None = None,
) -> list[Any]:
    """Convert Responses API input to LangChain messages.

    Args:
        input_data: Either a string (single user message) or list of input items
        instructions: Optional system message / instructions

    Returns:
        List of LangChain messages
    """
    messages = []

    # Add system message if instructions provided
    if instructions:
        messages.append(SystemMessage(content=instructions))

    # Handle string input (simple case)
    if isinstance(input_data, str):
        messages.append(HumanMessage(content=input_data))
        return messages

    # Handle list of input items
    for item in input_data:
        if item.type == "message":
            # Handle message item
            role = item.role
            content = item.content

            # Extract text content
            if isinstance(content, str):
                text = content
            elif isinstance(content, list):
                # List of content parts
                texts = []
                for part in content:
                    if isinstance(part, dict):
                        if part.get("type") == "input_text":
                            texts.append(part.get("text", ""))
                    elif hasattr(part, "text"):
                        texts.append(part.text)
                text = "\n".join(texts)
            else:
                text = str(content) if content else ""

            if role == "user":
                messages.append(HumanMessage(content=text))
            elif role == "assistant":
                messages.append(AIMessage(content=text))
            elif role == "system":
                messages.append(SystemMessage(content=text))

        elif item.type == "function_call_output":
            # Tool result from client (for multi-turn conversations)
            # This would be used in stateful mode, but we include it for completeness
            # LangGraph expects this as a ToolMessage
            from langchain_core.messages import ToolMessage

            messages.append(
                ToolMessage(
                    content=item.output or "",
                    tool_call_id=item.call_id or "",
                )
            )

    return messages


def extract_ai_message_from_event(event: dict) -> AIMessage | None:
    """Extract AIMessage from LangGraph event.

    LangGraph events have structure: {node_name: {messages: [...]}}
    We need to look for the 'agent' node which contains AI responses.
    """
    # Check for 'agent' node first (contains AI responses)
    if "agent" in event:
        agent_data = event["agent"]
        if isinstance(agent_data, dict) and "messages" in agent_data:
            messages = agent_data["messages"]
            if messages:
                last_msg = messages[-1]
                if isinstance(last_msg, AIMessage):
                    return last_msg

    # Fallback: check all keys for messages
    for _key, value in event.items():
        if isinstance(value, dict) and "messages" in value:
            messages = value["messages"]
            if messages:
                last_msg = messages[-1]
                if isinstance(last_msg, AIMessage):
                    return last_msg

    return None


def extract_tool_messages_from_event(event: dict) -> list[dict[str, Any]]:
    """Extract tool results from LangGraph event.

    LangGraph's tool_node returns dicts with 'role': 'tool',
    NOT ToolMessage objects.

    Args:
        event: LangGraph event dictionary

    Returns:
        List of tool result dicts from the event
    """
    tool_messages = []

    # Check for 'tools' node (contains tool results)
    if "tools" in event:
        tools_data = event["tools"]
        if isinstance(tools_data, dict) and "messages" in tools_data:
            messages = tools_data["messages"]
            for msg in messages:
                # LangGraph returns dicts with role="tool"
                if isinstance(msg, dict) and msg.get("role") == "tool":
                    tool_messages.append(msg)

    return tool_messages


def extract_content_str(content) -> str:
    """Extract string content from AIMessage content (handles str, list, dict)."""
    if content is None:
        return ""
    if isinstance(content, list):
        return str(content[0]) if content else ""
    elif isinstance(content, dict):
        return json.dumps(content, indent=2)
    else:
        return str(content)


def generate_item_id(prefix: str = "msg") -> str:
    """Generate a unique item ID with proper prefix.

    Prefixes:
    - msg: message items
    - fc: function_call items
    - fco: function_call_output items
    """
    return f"{prefix}_{uuid.uuid4().hex[:24]}"


def chunk_text(text: str, chunk_size: int = 5) -> Generator[str, None, None]:
    """Split text into chunks for streaming.

    Args:
        text: Text to chunk
        chunk_size: Approximate characters per chunk (default 5)

    Yields:
        Text chunks
    """
    # For Responses API, we chunk at word boundaries for better UX
    words = text.split()
    chunks: list[str] = []

    for i, word in enumerate(words):
        if i == 0:
            chunks.append(word)
        else:
            # Add space before word
            chunks.append(" " + word)

        # Yield every few words
        if len(chunks) >= 3:
            yield "".join(chunks)
            chunks = []

    # Yield remaining
    if chunks:
        yield "".join(chunks)


def chunk_json(json_str: str, chunk_size: int = 10) -> Generator[str, None, None]:
    """Split JSON into chunks for streaming function call arguments.

    Args:
        json_str: JSON string to chunk
        chunk_size: Characters per chunk

    Yields:
        JSON string chunks
    """
    for i in range(0, len(json_str), chunk_size):
        yield json_str[i : i + chunk_size]


async def generate_responses_stream(
    input_messages: str | list[ResponseInputItem],
    model: str = "esdc-agent",
    instructions: str | None = None,
    tools: list[dict[str, Any]] | None = None,
    temperature: float = 0.7,
) -> AsyncGenerator[str, None]:
    """Generate Responses API streaming events from LangGraph agent.

    Yields SSE-formatted events following Open Responses specification.

    Args:
        input_messages: Input (string or list of items)
        model: Model ID
        instructions: System message / instructions
        tools: Tools available to the model (not used by ESDC, tools are internal)
        temperature: Sampling temperature

    Yields:
        SSE-formatted event strings
    """
    response_id = f"resp_{uuid.uuid4().hex[:24]}"
    seq = SequenceCounter()

    # Emit response.created
    yield format_sse_event(
        create_response_created_event(response_id, model, seq.next())
    )

    # Create LLM and agent
    provider_config = Config.get_provider_config()
    if not provider_config:
        # Emit error event
        yield format_sse_event(
            {
                "type": "error",
                "sequence_number": seq.next(),
                "error": {"message": "No provider configured", "type": "server_error"},
            }
        )
        return

    provider_name = provider_config.get("provider_type", "ollama")
    provider_model = provider_config.get("model")
    base_url = provider_config.get("base_url")
    api_key = provider_config.get("api_key")

    provider_config_obj = {
        "provider_type": provider_name,
        "model": provider_model,
        "base_url": base_url,
        "api_key": api_key,
    }

    llm = create_llm_from_config(provider_config_obj)
    agent = create_agent(llm, checkpointer=None)

    # Convert input to LangChain messages
    lc_messages = convert_responses_input_to_langchain(input_messages, instructions)

    # Track output items and indices
    output_items: list[dict[str, Any]] = []
    output_index = 0

    logger.debug(f"[RESPONSES] Starting response stream, id={response_id}")

    try:
        # Stream agent events
        async for event in agent.astream({"messages": lc_messages}):
            # Check for tool results FIRST (before AI messages)
            tool_messages = extract_tool_messages_from_event(event)

            if tool_messages:
                logger.debug(f"[RESPONSES] Found {len(tool_messages)} tool result(s)")
                # Emit function_call_output items for each tool result
                for tool_msg in tool_messages:
                    item_id = generate_item_id("fco")
                    tool_call_id = tool_msg.get("tool_call_id", "")
                    tool_content = str(tool_msg.get("content", ""))

                    function_call_output = {
                        "id": item_id,
                        "type": "function_call_output",
                        "status": "completed",
                        "call_id": tool_call_id,
                        "output": tool_content,
                    }

                    # Emit item added
                    yield format_sse_event(
                        create_output_item_added_event(
                            seq.next(), output_index, function_call_output
                        )
                    )

                    # Emit item done (output is included, no deltas needed)
                    function_call_output["status"] = "completed"
                    yield format_sse_event(
                        create_output_item_done_event(
                            seq.next(), output_index, function_call_output
                        )
                    )

                    output_items.append(function_call_output)
                    output_index += 1

                # Tool results processed, continue to next event
                continue

            # Extract AI message
            ai_msg = extract_ai_message_from_event(event)
            if not ai_msg:
                continue

            # Handle message with content
            if ai_msg.content:
                item_id = generate_item_id("msg")
                content_str = extract_content_str(ai_msg.content)

                if content_str.strip():
                    # Emit message item added
                    message_item = {
                        "id": item_id,
                        "type": "message",
                        "status": "in_progress",
                        "role": "assistant",
                        "content": [],
                    }
                    yield format_sse_event(
                        create_output_item_added_event(
                            seq.next(), output_index, message_item
                        )
                    )

                    # Emit content part added
                    content_part = {"type": "output_text", "text": ""}
                    yield format_sse_event(
                        create_content_part_added_event(
                            seq.next(), output_index, 0, item_id, content_part
                        )
                    )

                    # Stream text deltas
                    accumulated_text = ""
                    for chunk in chunk_text(content_str):
                        accumulated_text += chunk
                        yield format_sse_event(
                            create_output_text_delta_event(
                                seq.next(), output_index, 0, item_id, chunk
                            )
                        )

                    # Emit text done
                    yield format_sse_event(
                        create_output_text_done_event(
                            seq.next(), output_index, 0, item_id, accumulated_text
                        )
                    )

                    # Emit content part done
                    content_part_done = {
                        "type": "output_text",
                        "text": accumulated_text,
                    }
                    yield format_sse_event(
                        create_content_part_done_event(
                            seq.next(), output_index, 0, item_id, content_part_done
                        )
                    )

                    # Complete message item
                    message_item["status"] = "completed"
                    message_item["content"] = [content_part_done]
                    yield format_sse_event(
                        create_output_item_done_event(
                            seq.next(), output_index, message_item
                        )
                    )

                    output_items.append(message_item)
                    output_index += 1

            # Handle function calls
            if hasattr(ai_msg, "tool_calls") and ai_msg.tool_calls:
                for tc in ai_msg.tool_calls:
                    item_id = generate_item_id("fc")
                    tc_name = tc.get("name", "")
                    tc_id = tc.get("id", "")
                    tc_args = json.dumps(tc.get("args", {}))

                    function_call_item = {
                        "id": item_id,
                        "type": "function_call",
                        "status": "in_progress",
                        "name": tc_name,
                        "call_id": tc_id,
                        "arguments": "",
                    }

                    # Emit item added
                    yield format_sse_event(
                        create_output_item_added_event(
                            seq.next(), output_index, function_call_item
                        )
                    )

                    # Stream argument deltas
                    for arg_chunk in chunk_json(tc_args):
                        function_call_item["arguments"] += arg_chunk
                        yield format_sse_event(
                            create_function_call_arguments_delta_event(
                                seq.next(), output_index, item_id, arg_chunk
                            )
                        )

                    # Emit args done
                    yield format_sse_event(
                        create_function_call_arguments_done_event(
                            seq.next(), output_index, item_id, tc_args
                        )
                    )

                    # Complete item
                    function_call_item["status"] = "completed"
                    yield format_sse_event(
                        create_output_item_done_event(
                            seq.next(), output_index, function_call_item
                        )
                    )

                    output_items.append(function_call_item)
                    output_index += 1

        # Emit response.completed
        yield format_sse_event(
            create_response_completed_event(
                seq.next(), response_id, model, output_items
            )
        )

    except Exception as e:
        logger.exception(f"[RESPONSES] Error in stream: {e}")
        yield format_sse_event(
            {
                "type": "error",
                "sequence_number": seq.next(),
                "error": {"message": str(e), "type": "server_error"},
            }
        )


async def generate_responses_sync(
    input_messages: str | list[ResponseInputItem],
    model: str = "esdc-agent",
    instructions: str | None = None,
    tools: list[dict[str, Any]] | None = None,
    temperature: float = 0.7,
) -> dict[str, Any]:
    """Generate non-streaming Responses API response.

    Collects all output from agent and returns a complete response object.

    Args:
        input_messages: Input (string or list of items)
        model: Model ID
        instructions: System message / instructions
        tools: Tools available to the model (not used by ESDC)
        temperature: Sampling temperature

    Returns:
        Complete response object
    """
    from esdc.server.responses_events import create_non_streaming_response

    response_id = f"resp_{uuid.uuid4().hex[:24]}"
    output_items: list[dict[str, Any]] = []

    # Create LLM and agent
    provider_config = Config.get_provider_config()
    if not provider_config:
        return create_non_streaming_response(
            response_id,
            model,
            [],
            error={"message": "No provider configured", "type": "server_error"},
        )

    provider_name = provider_config.get("provider_type", "ollama")
    provider_model = provider_config.get("model")
    base_url = provider_config.get("base_url")
    api_key = provider_config.get("api_key")

    provider_config_obj = {
        "provider_type": provider_name,
        "model": provider_model,
        "base_url": base_url,
        "api_key": api_key,
    }

    llm = create_llm_from_config(provider_config_obj)
    agent = create_agent(llm, checkpointer=None)

    # Convert input to LangChain messages
    lc_messages = convert_responses_input_to_langchain(input_messages, instructions)

    try:
        # Collect all messages from agent
        all_tool_results: list[dict[str, Any]] = []
        all_ai_messages: list[AIMessage] = []

        async for event in agent.astream({"messages": lc_messages}):
            # Collect tool results
            tool_messages = extract_tool_messages_from_event(event)
            if tool_messages:
                for tool_msg in tool_messages:
                    all_tool_results.append(
                        {
                            "id": generate_item_id("fco"),
                            "type": "function_call_output",
                            "status": "completed",
                            "call_id": tool_msg.get("tool_call_id", ""),
                            "output": str(tool_msg.get("content", "")),
                        }
                    )
                continue

            # Collect AI messages
            ai_msg = extract_ai_message_from_event(event)
            if ai_msg:
                all_ai_messages.append(ai_msg)

        # Build output items
        # First: all function_calls
        for ai_msg in all_ai_messages:
            if hasattr(ai_msg, "tool_calls") and ai_msg.tool_calls:
                for tc in ai_msg.tool_calls:
                    output_items.append(
                        {
                            "id": generate_item_id("fc"),
                            "type": "function_call",
                            "status": "completed",
                            "name": tc.get("name", ""),
                            "call_id": tc.get("id", ""),
                            "arguments": json.dumps(tc.get("args", {})),
                        }
                    )

        # Second: all function_call_outputs
        output_items.extend(all_tool_results)

        # Last: final message (if any AI message had content)
        for ai_msg in reversed(all_ai_messages):
            if ai_msg.content:
                content_str = extract_content_str(ai_msg.content)
                if content_str.strip():
                    output_items.append(
                        {
                            "id": generate_item_id("msg"),
                            "type": "message",
                            "status": "completed",
                            "role": "assistant",
                            "content": [
                                {
                                    "type": "output_text",
                                    "text": content_str,
                                    "annotations": [],
                                }
                            ],
                        }
                    )
                break

        return create_non_streaming_response(response_id, model, output_items)

    except Exception as e:
        logger.exception(f"[RESPONSES] Error in sync generation: {e}")
        return create_non_streaming_response(
            response_id,
            model,
            [],
            error={"message": str(e), "type": "server_error"},
        )
