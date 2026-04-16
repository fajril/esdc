"""Responses API wrapper for LangGraph agent.

This module provides Open Responses API compatible endpoints that:
- Handle Responses API specific format with input items
- Support string or list input formats
- Stream responses with SSE events
- Support discriminated union input types

Key Functions:
    - convert_responses_input_to_langchain: Convert Responses API input
    - generate_responses_stream: SSE streaming for Responses API
    - generate_responses_sync: Non-streaming response generation
"""

# Standard library
import json
import logging
import time
import uuid
from collections.abc import AsyncGenerator
from typing import Any

# Third-party
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

# Local
from esdc.chat.agent import create_agent, generate_conversation_title
from esdc.configs import Config
from esdc.providers import create_llm_from_config
from esdc.server.cache import get_parsed_json
from esdc.server.message_utils import extract_ai_message_from_event, extract_content_str
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
from esdc.server.stream_utils import chunk_json, chunk_text
from esdc.server.title_detection import (
    create_title_stream_events,
    create_title_sync_response,
    extract_user_query_from_title_request,
    is_title_generation_request,
)

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
    input_data: str | list[Any],
    instructions: str | None = None,
) -> list[Any]:
    """Convert Responses API input to LangChain messages.

    Args:
        input_data: Either a string (single user message) or list of input items.
            Items can be ResponseInputItem Pydantic models or plain dicts.
        instructions: Optional system message / instructions

    Returns:
        List of LangChain messages

    Example:
        >>> messages = convert_responses_input_to_langchain("Hello")
        >>> len(messages)
        1
        >>> messages[0].content
        'Hello'

        >>> messages = convert_responses_input_to_langchain(
        ...     [{"type": "message", "role": "user", "content": "Hi"}],
        ...     instructions="Be helpful"
        ... )
        >>> len(messages)
        2
        >>> messages[0].content
        'Be helpful'
    """
    messages: list[Any] = []

    # Add system message if instructions provided
    if instructions:
        messages.append(SystemMessage(content=instructions))

    # Handle string input (simple case)
    if isinstance(input_data, str):
        messages.append(HumanMessage(content=input_data))
        return messages

    # Handle list of input items
    for idx, item in enumerate(input_data):
        # Handle dict items (from JSON/API)
        if isinstance(item, dict):
            item_type = item.get("type")
            role = item.get("role")
            content = item.get("content")
            call_id = item.get("call_id")
            output = item.get("output")
        else:
            # Handle Pydantic model items
            item_type = getattr(item, "type", None)
            role = getattr(item, "role", None)
            content = getattr(item, "content", None)
            call_id = getattr(item, "call_id", None)
            output = getattr(item, "output", None)

        if item_type == "message":
            # Handle message item
            # Extract text content
            text = ""
            if isinstance(content, str):
                text = content
                logger.debug(
                    f"[convert_responses_input] Item {idx}: str, len={len(text)}"
                )
            elif isinstance(content, list):
                # List of content parts - can include strings, dicts, or Pydantic
                logger.debug(
                    f"[convert_responses_input] Item {idx}: list, len={len(content)}"
                )
                texts = []

                for part in content:
                    # Handle string parts
                    if isinstance(part, str):
                        texts.append(part)
                    # Handle dict parts
                    elif isinstance(part, dict):
                        ptype = part.get("type", "")
                        if ptype in ("input_text", "text", "output_text"):
                            texts.append(part.get("text", ""))
                        else:
                            texts.append(part.get("text", ""))
                    # Handle Pydantic model objects or objects with .text
                    elif hasattr(part, "text"):
                        texts.append(str(part.text))
                    # Unknown type - convert to string as fallback
                    else:
                        texts.append(str(part) if part else "")

                text = "\n".join(texts)
                logger.debug(
                    f"[convert_responses_input] Item {idx}: combined len={len(text)}"
                )
            else:
                # Unknown content type - convert to string
                text = str(content) if content else ""
                logger.warning(
                    f"[convert_responses_input] Item {idx}: unknown content type "
                    f"{type(content).__name__}"
                )

            if role == "user":
                messages.append(HumanMessage(content=text))
            elif role == "assistant":
                messages.append(AIMessage(content=text))
            elif role == "system":
                messages.append(SystemMessage(content=text))

        elif item_type == "function_call":
            # Function call from previous assistant turn
            # Create AIMessage with tool_calls for LangGraph
            # Note: OpenWebUI may use 'id' or 'call_id' interchangeably
            call_id = ""
            if isinstance(item, dict):
                call_id = item.get("call_id") or item.get("id", "")
            else:
                call_id = getattr(item, "call_id", None) or getattr(item, "id", "")

            name = (
                item.get("name", "")
                if isinstance(item, dict)
                else getattr(item, "name", "")
            )
            args_str = (
                item.get("arguments", "{}")
                if isinstance(item, dict)
                else getattr(item, "arguments", "{}")
            )

            args = get_parsed_json(args_str)

            logger.debug(
                f"[convert_responses_input] Item {idx}: function_call, "
                f"call_id={call_id}, name={name}"
            )

            messages.append(
                AIMessage(
                    content="",
                    tool_calls=[{"name": name, "args": args, "id": call_id}],
                )
            )

        elif item_type == "function_call_output":
            # Tool result from client (for multi-turn conversations)
            # LangGraph expects this as a ToolMessage
            # Output can be either a string or array of content parts (OpenWebUI format)
            from langchain_core.messages import ToolMessage

            output_content = ""
            if isinstance(output, str):
                output_content = output
            elif isinstance(output, list):
                # Extract text from content parts
                text_parts = []
                output_list: list[Any] = output
                for part in output_list:
                    if isinstance(part, dict):
                        ptype = part.get("type", "")
                        if ptype in ("input_text", "output_text", "text"):
                            text_parts.append(part.get("text", ""))
                    elif isinstance(part, str):
                        text_parts.append(part)
                output_content = "\n".join(text_parts) if text_parts else str(output)

            output_len = len(output_content) if output_content else 0
            logger.debug(
                f"[convert_responses_input] Item {idx}: function_call_output, "
                f"call_id={call_id}, output_len={output_len}"
            )

            messages.append(
                ToolMessage(
                    content=output_content,
                    tool_call_id=call_id or "",
                )
            )

        else:
            logger.warning(
                f"[convert_responses_input] Item {idx}: unknown type={item_type}"
            )

    logger.debug(f"[convert_responses_input] Created {len(messages)} messages")
    return messages


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


def generate_item_id(prefix: str = "msg") -> str:
    """Generate a unique item ID with proper prefix.

    Prefixes:
    - msg: message items
    - fc: function_call items
    - fco: function_call_output items
    """
    return f"{prefix}_{uuid.uuid4().hex[:24]}"


async def generate_responses_stream(
    input_messages: str | list[ResponseInputItem],
    model: str = "iris",
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

    # Bypass: Detect OpenWebUI title-generation requests
    if is_title_generation_request(input_messages):
        logger.info(
            "[RESPONSES %s] TITLE_GEN_STREAM: bypassing agent pipeline",
            response_id,
        )
        user_query = extract_user_query_from_title_request(input_messages)

        provider_config = Config.get_provider_config()
        if not provider_config:
            created_seq = seq.next()
            yield format_sse_event(
                create_response_created_event(response_id, model, created_seq)
            )
            yield "data: [DONE]\n\n"
            return

        provider_config_obj = {
            "provider_type": provider_config.get("provider_type", "ollama"),
            "model": provider_config.get("model"),
            "base_url": provider_config.get("base_url"),
            "api_key": provider_config.get("api_key"),
        }
        llm = create_llm_from_config(provider_config_obj)

        title = await generate_conversation_title(llm, user_query)
        logger.info(
            "[RESPONSES %s] TITLE_GEN_STREAM: completed, title=%r",
            response_id,
            title,
        )

        for event in create_title_stream_events(title, response_id, model):
            yield event
        return

    # DEBUG: Track execution
    event_counter = 0

    # Emit response.created
    created_seq = seq.next()
    logger.debug(
        f"[RESPONSES {response_id}] Emitting response.created seq={created_seq}"
    )
    yield format_sse_event(
        create_response_created_event(response_id, model, created_seq)
    )

    # Create LLM and agent
    provider_config = Config.get_provider_config()
    if not provider_config:
        logger.error(f"[RESPONSES {response_id}] No provider configured")
        # Emit error event
        error_seq = seq.next()
        yield format_sse_event(
            {
                "type": "error",
                "sequence_number": error_seq,
                "error": {"message": "No provider configured", "type": "server_error"},
            }
        )
        return

    provider_name = provider_config.get("provider_type", "ollama")
    provider_model = provider_config.get("model")
    base_url = provider_config.get("base_url")
    api_key = provider_config.get("api_key")

    logger.debug(
        f"[RESPONSES {response_id}] Provider: {provider_name}, "
        f"model: {provider_model}, base_url: {base_url}"
    )

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

    # DEBUG: Log input
    input_type = (
        "string" if isinstance(input_messages, str) else f"list({len(input_messages)})"
    )
    instructions_preview = instructions[:100] if instructions else "None"
    logger.debug(
        f"[RESPONSES {response_id}] Input type: {input_type}, "
        f"messages: {len(lc_messages)}, instructions: {instructions_preview}"
    )
    for i, msg in enumerate(lc_messages):
        content_preview = str(msg.content)[:50] if msg.content else "None"
        logger.debug(
            f"[RESPONSES {response_id}] Message {i}: type={type(msg).__name__}, "
            f"content_preview={content_preview}..."
        )

    # Track output items and indices
    output_items: list[dict[str, Any]] = []
    output_index = 0

    logger.info(
        f"[RESPONSES {response_id}] START - "
        f"input_type={input_type}, messages={len(lc_messages)}"
    )
    logger.debug(f"[TIMING] {response_id} stream_start")

    # Log inference input details
    total_chars = sum(len(str(m.content)) for m in lc_messages)
    system_msgs = len([m for m in lc_messages if isinstance(m, SystemMessage)])
    user_msgs = len([m for m in lc_messages if isinstance(m, HumanMessage)])
    ai_msgs = len([m for m in lc_messages if isinstance(m, AIMessage)])
    logger.debug(
        "[INFERENCE] responses_stream_input | response_id=%s | messages=%d | system=%d | user=%d | ai=%d | total_chars=%d",
        response_id,
        len(lc_messages),
        system_msgs,
        user_msgs,
        ai_msgs,
        total_chars,
    )

    first_ai_response = True
    stream_start = time.perf_counter()
    last_event_time = stream_start

    try:
        # Stream agent events
        async for event in agent.astream({"messages": lc_messages}):
            event_counter += 1
            current_time = time.perf_counter()
            event_keys = list(event.keys())

            # Calculate gap from previous event
            gap_ms = (current_time - last_event_time) * 1000
            if gap_ms > 2000:
                logger.debug(
                    f"[GAP_DETECTED] {response_id} | gap={gap_ms:.2f}ms | "
                    f"since_event=#{event_counter - 1} | to_event=#{event_counter}"
                )
            last_event_time = current_time

            logger.debug(
                f"[RESPONSES {response_id}] EVENT #{event_counter} keys: {event_keys}"
            )

            # Check for tool results FIRST (before AI messages)
            tool_messages = extract_tool_messages_from_event(event)

            if tool_messages:
                logger.debug(
                    f"[RESPONSES {response_id}] EVENT #{event_counter} "
                    f"Found {len(tool_messages)} tool result(s)"
                )
                logger.debug(
                    f"[TIMING] {response_id} tool_result | event=#{event_counter} count={len(tool_messages)}"
                )
                # Emit function_call_output items for each tool result
                for i, tool_msg in enumerate(tool_messages):
                    item_id = generate_item_id("fco")
                    tool_call_id = tool_msg.get("tool_call_id", "")
                    tool_content = str(tool_msg.get("content", ""))
                    content_len = len(tool_content)
                    content_preview = (
                        tool_content[:100] if content_len > 100 else tool_content
                    )

                    logger.debug(
                        f"[RESPONSES {response_id}] EVENT #{event_counter} "
                        f"Tool result #{i}: call_id={tool_call_id}, "
                        f"content_len={content_len}, preview={content_preview}..."
                    )

                    function_call_output = {
                        "id": item_id,
                        "type": "function_call_output",
                        "status": "completed",
                        "call_id": tool_call_id,
                        "output": [{"type": "input_text", "text": tool_content}],
                    }

                    # Emit item added
                    added_seq = seq.next()
                    logger.debug(
                        f"[RESPONSES {response_id}] EMIT response.output_item.added "
                        f"seq={added_seq}, output_index={output_index}, "
                        f"type=function_call_output, id={item_id}, "
                        f"call_id={tool_call_id}"
                    )
                    yield format_sse_event(
                        create_output_item_added_event(
                            added_seq, output_index, function_call_output
                        )
                    )

                    # Emit item done (output is included, no deltas needed)
                    function_call_output["status"] = "completed"
                    done_seq = seq.next()
                    logger.debug(
                        f"[RESPONSES {response_id}] EMIT response.output_item.done "
                        f"seq={done_seq}, output_index={output_index}, "
                        f"type=function_call_output, id={item_id}"
                    )
                    yield format_sse_event(
                        create_output_item_done_event(
                            done_seq, output_index, function_call_output
                        )
                    )

                    output_items.append(function_call_output)
                    output_index += 1

                # Tool results processed, continue to next event
                continue

            # Extract AI message
            ai_msg = extract_ai_message_from_event(event)
            if not ai_msg:
                logger.debug(
                    f"[RESPONSES {response_id}] EVENT #{event_counter} "
                    f"No AIMessage extracted, skipping"
                )
                continue

            if first_ai_response:
                first_response_ms = (time.perf_counter() - stream_start) * 1000
                logger.debug(
                    f"[TIMING] {response_id} first_ai_response | event=#{event_counter}"
                )
                first_ai_response = False

            # DEBUG: Log AIMessage details
            msg_id = getattr(ai_msg, "id", "no-id")
            msg_content_preview = (
                str(ai_msg.content)[:100] if ai_msg.content else "None"
            )
            has_tool_calls = hasattr(ai_msg, "tool_calls") and ai_msg.tool_calls
            tool_call_count = len(ai_msg.tool_calls) if has_tool_calls else 0
            logger.debug(
                f"[RESPONSES {response_id}] EVENT #{event_counter} "
                f"AIMessage: id={msg_id}, content_preview={msg_content_preview}..., "
                f"tool_calls={tool_call_count}"
            )

            # Log inference response details for first AI message
            if first_ai_response is False and tool_call_count > 0:
                logger.debug(
                    "[INFERENCE] responses_first_response | response_id=%s | elapsed_ms=%.2f | content_preview=%s | tool_calls=%d",
                    response_id,
                    first_response_ms,
                    msg_content_preview[:50],
                    tool_call_count,
                )

            # Handle message with content
            if ai_msg.content:
                item_id = generate_item_id("msg")
                content_str = extract_content_str(ai_msg.content)

                if content_str.strip():
                    logger.debug(
                        f"[RESPONSES {response_id}] Processing message content: "
                        f"len={len(content_str)}, preview={content_str[:50]}..."
                    )

                    # Emit message item added
                    message_item = {
                        "id": item_id,
                        "type": "message",
                        "status": "in_progress",
                        "role": "assistant",
                        "content": [],
                    }
                    added_seq = seq.next()
                    logger.debug(
                        f"[RESPONSES {response_id}] EMIT response.output_item.added "
                        f"seq={added_seq}, output_index={output_index}, "
                        f"type=message, id={item_id}"
                    )
                    yield format_sse_event(
                        create_output_item_added_event(
                            added_seq, output_index, message_item
                        )
                    )

                    # Emit content part added
                    content_part = {"type": "output_text", "text": ""}
                    part_added_seq = seq.next()
                    logger.debug(
                        f"[RESPONSES {response_id}] EMIT response.content_part.added "
                        f"seq={part_added_seq}, output_index={output_index}, "
                        f"content_index=0, item_id={item_id}"
                    )
                    yield format_sse_event(
                        create_content_part_added_event(
                            part_added_seq, output_index, 0, item_id, content_part
                        )
                    )

                    # Stream text deltas
                    accumulated_text = ""
                    delta_count = 0
                    for chunk in chunk_text(content_str):
                        accumulated_text += chunk
                        delta_count += 1
                        delta_seq = seq.next()
                        logger.debug(
                            f"[RESPONSES {response_id}] "
                            f"EMIT response.output_text.delta "
                            f"seq={delta_seq}, delta_len={len(chunk)}, "
                            f"accumulated_len={len(accumulated_text)}"
                        )
                        yield format_sse_event(
                            create_output_text_delta_event(
                                delta_seq, output_index, 0, item_id, chunk
                            )
                        )

                    # Emit text done
                    text_done_seq = seq.next()
                    logger.debug(
                        f"[RESPONSES {response_id}] EMIT response.output_text.done "
                        f"seq={text_done_seq}, text_len={len(accumulated_text)}, "
                        f"deltas_sent={delta_count}"
                    )
                    yield format_sse_event(
                        create_output_text_done_event(
                            text_done_seq, output_index, 0, item_id, accumulated_text
                        )
                    )

                    # Emit content part done
                    content_part_done = {
                        "type": "output_text",
                        "text": accumulated_text,
                    }
                    part_done_seq = seq.next()
                    logger.debug(
                        f"[RESPONSES {response_id}] EMIT response.content_part.done "
                        f"seq={part_done_seq}, output_index={output_index}"
                    )
                    yield format_sse_event(
                        create_content_part_done_event(
                            part_done_seq, output_index, 0, item_id, content_part_done
                        )
                    )

                    # Complete message item
                    message_item["status"] = "completed"
                    message_item["content"] = [content_part_done]
                    item_done_seq = seq.next()
                    logger.debug(
                        f"[RESPONSES {response_id}] EMIT response.output_item.done "
                        f"seq={item_done_seq}, output_index={output_index}, "
                        f"type=message, id={item_id}"
                    )
                    yield format_sse_event(
                        create_output_item_done_event(
                            item_done_seq, output_index, message_item
                        )
                    )

                    output_items.append(message_item)
                    output_index += 1

            # Handle function calls
            if hasattr(ai_msg, "tool_calls") and ai_msg.tool_calls:
                logger.debug(
                    f"[RESPONSES {response_id}] EVENT #{event_counter} "
                    f"Processing {len(ai_msg.tool_calls)} tool call(s)"
                )
                for tc_idx, tc in enumerate(ai_msg.tool_calls):
                    # DIAGNOSTIC: Log type and value of each tool_call
                    logger.debug(
                        f"[RESPONSES {response_id}] EVENT #{event_counter} "
                        f"tool_call[{tc_idx}] type: {type(tc).__name__}, "
                        f"value: {repr(tc)[:200]}"
                    )
                    # If it's not a dict, skip and log warning
                    if not isinstance(tc, dict):
                        logger.warning(
                            f"[RESPONSES {response_id}] EVENT #{event_counter} "
                            f"SKIPPING malformed tool_call[{tc_idx}] "
                            f"- expected dict, got {type(tc).__name__}: "
                            f"{repr(tc)[:100]}"
                        )
                        continue
                    item_id = generate_item_id("fc")
                    tc_name = tc.get("name", "")
                    tc_id = tc.get("id", "")
                    tc_args = json.dumps(tc.get("args", {}))

                    logger.debug(
                        f"[RESPONSES {response_id}] EVENT #{event_counter} "
                        f"Tool call #{tc_idx}: name={tc_name}, call_id={tc_id}, "
                        f"args_len={len(tc_args)}"
                    )

                    function_call_item = {
                        "id": item_id,
                        "type": "function_call",
                        "status": "in_progress",
                        "name": tc_name,
                        "call_id": tc_id,
                        "arguments": "",
                    }

                    # Emit item added
                    added_seq = seq.next()
                    logger.debug(
                        f"[RESPONSES {response_id}] EMIT "
                        f"response.output_item.added "
                        f"seq={added_seq}, output_index={output_index}, "
                        f"type=function_call, id={item_id}, "
                        f"name={tc_name}, call_id={tc_id}"
                    )
                    yield format_sse_event(
                        create_output_item_added_event(
                            added_seq, output_index, function_call_item
                        )
                    )

                    # Stream argument deltas
                    delta_count = 0
                    for arg_chunk in chunk_json(tc_args):
                        function_call_item["arguments"] += arg_chunk
                        delta_count += 1
                        delta_seq = seq.next()
                        logger.debug(
                            f"[RESPONSES {response_id}] "
                            f"EMIT response.function_call_arguments."
                            f"delta seq={delta_seq}, chunk_len={len(arg_chunk)}"
                        )
                        yield format_sse_event(
                            create_function_call_arguments_delta_event(
                                delta_seq, output_index, item_id, arg_chunk
                            )
                        )

                    # Emit args done
                    args_done_seq = seq.next()
                    logger.debug(
                        f"[RESPONSES {response_id}] "
                        f"EMIT response.function_call_arguments.done "
                        f"seq={args_done_seq}, args_len={len(tc_args)}, "
                        f"deltas_sent={delta_count}"
                    )
                    yield format_sse_event(
                        create_function_call_arguments_done_event(
                            args_done_seq, output_index, item_id, tc_args
                        )
                    )

                    # Complete item
                    function_call_item["status"] = "completed"
                    item_done_seq = seq.next()
                    logger.debug(
                        f"[RESPONSES {response_id}] EMIT response.output_item.done "
                        f"seq={item_done_seq}, output_index={output_index}, "
                        f"type=function_call, id={item_id}, name={tc_name}"
                    )
                    yield format_sse_event(
                        create_output_item_done_event(
                            item_done_seq, output_index, function_call_item
                        )
                    )

                    output_items.append(function_call_item)
                    output_index += 1

        # Emit response.completed
        completed_seq = seq.next()
        output_summary = [
            f"{item.get('type')}:{item.get('name', item.get('call_id', 'msg'))}"
            for item in output_items
        ]
        logger.info(
            f"[RESPONSES {response_id}] COMPLETED - "
            f"events={event_counter}, output_items={len(output_items)}, "
            f"items=[{', '.join(output_summary)}], final_seq={completed_seq}"
        )

        # Log inference stream completion
        total_elapsed_ms = (time.perf_counter() - stream_start) * 1000
        logger.debug(
            f"[TIMING] {response_id} stream_complete | events={event_counter} items={len(output_items)}"
        )
        logger.debug(
            "[INFERENCE] responses_stream_complete | response_id=%s | total_ms=%.2f | events=%d | output_items=%d",
            response_id,
            total_elapsed_ms,
            event_counter,
            len(output_items),
        )
        yield format_sse_event(
            create_response_completed_event(
                completed_seq, response_id, model, output_items
            )
        )

    except Exception as e:
        logger.exception(f"[RESPONSES {response_id}] ERROR: {e}")
        yield format_sse_event(
            {
                "type": "error",
                "sequence_number": seq.next(),
                "error": {"message": str(e), "type": "server_error"},
            }
        )


async def generate_responses_sync(
    input_messages: str | list[ResponseInputItem],
    model: str = "iris",
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

    # Bypass: Detect OpenWebUI title-generation requests
    if is_title_generation_request(input_messages):
        logger.info("[RESPONSES %s] TITLE_GEN: bypassing agent pipeline", response_id)
        user_query = extract_user_query_from_title_request(input_messages)

        provider_config = Config.get_provider_config()
        if not provider_config:
            return create_non_streaming_response(
                response_id,
                model,
                [],
                error={"message": "No provider configured", "type": "server_error"},
            )

        provider_config_obj = {
            "provider_type": provider_config.get("provider_type", "ollama"),
            "model": provider_config.get("model"),
            "base_url": provider_config.get("base_url"),
            "api_key": provider_config.get("api_key"),
        }
        llm = create_llm_from_config(provider_config_obj)

        title_start = time.perf_counter()
        title = await generate_conversation_title(llm, user_query)
        title_elapsed = (time.perf_counter() - title_start) * 1000
        logger.info(
            "[RESPONSES %s] TITLE_GEN: completed in %.0fms, title=%r",
            response_id,
            title_elapsed,
            title,
        )

        return create_title_sync_response(title, response_id)

    # Create LLM and agent
    provider_config = Config.get_provider_config()
    if not provider_config:
        logger.error(f"[RESPONSES {response_id}] SYNC: No provider configured")
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

    logger.debug(
        f"[RESPONSES {response_id}] SYNC: Provider: {provider_name}, "
        f"model: {provider_model}, base_url: {base_url}"
    )

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

    # DEBUG: Log input
    input_type = (
        "string" if isinstance(input_messages, str) else f"list({len(input_messages)})"
    )
    instructions_preview = instructions[:100] if instructions else "None"
    logger.debug(
        f"[RESPONSES {response_id}] SYNC: Input type: {input_type}, "
        f"messages: {len(lc_messages)}, instructions: {instructions_preview}"
    )

    # DEBUG: Track execution
    event_counter = 0

    logger.info(
        f"[RESPONSES {response_id}] SYNC START - "
        f"input_type={input_type}, messages={len(lc_messages)}"
    )
    logger.debug(f"[TIMING] {response_id} sync_start")

    # Log inference input details
    total_chars = sum(len(str(m.content)) for m in lc_messages)
    system_msgs = len([m for m in lc_messages if isinstance(m, SystemMessage)])
    user_msgs = len([m for m in lc_messages if isinstance(m, HumanMessage)])
    ai_msgs = len([m for m in lc_messages if isinstance(m, AIMessage)])
    logger.debug(
        "[INFERENCE] responses_sync_input | response_id=%s | messages=%d | system=%d | user=%d | ai=%d | total_chars=%d",
        response_id,
        len(lc_messages),
        system_msgs,
        user_msgs,
        ai_msgs,
        total_chars,
    )

    sync_start = time.perf_counter()

    try:
        # Collect all messages from agent
        all_tool_results: list[dict[str, Any]] = []
        all_ai_messages: list[AIMessage] = []

        async for event in agent.astream({"messages": lc_messages}):
            event_counter += 1
            event_keys = list(event.keys())
            logger.debug(
                f"[RESPONSES {response_id}] SYNC EVENT #{event_counter} "
                f"keys: {event_keys}"
            )

            # Collect tool results
            tool_messages = extract_tool_messages_from_event(event)
            if tool_messages:
                logger.debug(
                    f"[RESPONSES {response_id}] SYNC EVENT #{event_counter} "
                    f"Found {len(tool_messages)} tool result(s)"
                )
                logger.debug(
                    f"[TIMING] {response_id} sync_tool_result | event=#{event_counter} count={len(tool_messages)}"
                )
                for i, tool_msg in enumerate(tool_messages):
                    call_id = tool_msg.get("tool_call_id", "")
                    content_len = len(str(tool_msg.get("content", "")))
                    logger.debug(
                        f"[RESPONSES {response_id}] SYNC Tool result #{i}: "
                        f"call_id={call_id}, content_len={content_len}"
                    )
                    all_tool_results.append(
                        {
                            "id": generate_item_id("fco"),
                            "type": "function_call_output",
                            "status": "completed",
                            "call_id": call_id,
                            "output": [
                                {
                                    "type": "input_text",
                                    "text": str(tool_msg.get("content", "")),
                                }
                            ],
                        }
                    )
                continue

            # Collect AI messages
            ai_msg = extract_ai_message_from_event(event)
            if ai_msg:
                msg_id = getattr(ai_msg, "id", "no-id")
                has_tool_calls = hasattr(ai_msg, "tool_calls") and ai_msg.tool_calls
                tool_call_count = len(ai_msg.tool_calls) if has_tool_calls else 0
                logger.debug(
                    f"[RESPONSES {response_id}] SYNC EVENT #{event_counter} "
                    f"AIMessage: id={msg_id}, tool_calls={tool_call_count}"
                )
                all_ai_messages.append(ai_msg)

        # Build output items
        # First: all function_calls
        for ai_msg in all_ai_messages:
            if hasattr(ai_msg, "tool_calls") and ai_msg.tool_calls:
                logger.debug(
                    f"[RESPONSES {response_id}] SYNC tool_calls type: "
                    f"{type(ai_msg.tool_calls)}, count: {len(ai_msg.tool_calls)}"
                )
                for tc_idx, tc in enumerate(ai_msg.tool_calls):
                    # DIAGNOSTIC: Log type and value of each tool_call
                    logger.debug(
                        f"[RESPONSES {response_id}] SYNC tool_call[{tc_idx}] "
                        f"type: {type(tc).__name__}, "
                        f"value: {repr(tc)[:200]}"
                    )
                    # If it's not a dict, skip and log warning
                    if not isinstance(tc, dict):
                        logger.warning(
                            f"[RESPONSES {response_id}] SYNC SKIPPING malformed "
                            f"tool_call[{tc_idx}] - expected dict, got "
                            f"{type(tc).__name__}: {repr(tc)[:100]}"
                        )
                        continue
                    tc_name = tc.get("name", "")
                    tc_id = tc.get("id", "")
                    logger.debug(
                        f"[RESPONSES {response_id}] SYNC Building function_call: "
                        f"name={tc_name}, call_id={tc_id}"
                    )
                    output_items.append(
                        {
                            "id": generate_item_id("fc"),
                            "type": "function_call",
                            "status": "completed",
                            "name": tc_name,
                            "call_id": tc_id,
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
                    logger.debug(
                        f"[RESPONSES {response_id}] SYNC Building message item: "
                        f"content_len={len(content_str)}"
                    )
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

        output_summary = [
            f"{item.get('type')}:{item.get('name', item.get('call_id', 'msg'))}"
            for item in output_items
        ]
        logger.info(
            f"[RESPONSES {response_id}] SYNC COMPLETED - "
            f"events={event_counter}, output_items={len(output_items)}, "
            f"items=[{', '.join(output_summary)}]"
        )

        # Log inference sync completion
        sync_elapsed_ms = (time.perf_counter() - sync_start) * 1000
        logger.debug(
            f"[TIMING] {response_id} sync_complete | events={event_counter} items={len(output_items)}"
        )
        logger.debug(
            "[INFERENCE] responses_sync_complete | response_id=%s | elapsed_ms=%.2f | events=%d | output_items=%d",
            response_id,
            sync_elapsed_ms,
            event_counter,
            len(output_items),
        )
        return create_non_streaming_response(response_id, model, output_items)

    except Exception as e:
        # Log inference error
        error_elapsed_ms = (time.perf_counter() - sync_start) * 1000
        logger.debug(
            "[INFERENCE] responses_sync_error | response_id=%s | elapsed_ms=%.2f | error=%s",
            response_id,
            error_elapsed_ms,
            str(e)[:100],
        )
        logger.exception(f"[RESPONSES {response_id}] SYNC ERROR: {e}")
        return create_non_streaming_response(
            response_id,
            model,
            [],
            error={"message": str(e), "type": "server_error"},
        )
