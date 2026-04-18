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
import asyncio
import json
import logging
import time
import uuid
from collections.abc import AsyncGenerator
from typing import Any

# Third-party
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

# Local
from esdc.chat.agent import (
    create_agent,
    generate_conversation_tags,
    generate_conversation_title,
)
from esdc.chat.external_tools import (
    categorize_tools,
    convert_external_specs_to_langchain,
)
from esdc.configs import Config
from esdc.providers import create_llm_from_config
from esdc.server.cache import get_parsed_json
from esdc.server.constants import SSE_STREAM_TIMEOUT
from esdc.server.event_streamer import astream_agent_events
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
    create_response_failed_event,
    format_sse_event,
)
from esdc.server.responses_models import ResponseInputItem
from esdc.server.stream_utils import chunk_json
from esdc.server.title_detection import (
    create_tags_stream_events,
    create_tags_sync_response,
    create_title_stream_events,
    create_title_sync_response,
    extract_user_query,
    get_ancillary_type,
    is_ancillary_request,
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


class _RecursionLimitExceededError(Exception):
    """Raised when LangGraph recursion limit is exceeded during streaming."""

    pass


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
    reasoning_effort: str | None = None,
) -> AsyncGenerator[str, None]:
    """Generate Responses API streaming events from LangGraph agent.

    Yields SSE-formatted events following Open Responses specification.

    Args:
        input_messages: Input (string or list of items)
        model: Model ID
        instructions: System message / instructions
        tools: Tools available to the model. If provided, tools are categorized
            into internal (ESDC) and external (e.g. OpenTerminal). Internal tools
            are executed server-side. External tools are not executed; when the
            LLM calls an external tool, a function_call item is emitted for
            OpenWebUI to handle.
        temperature: Sampling temperature
        reasoning_effort: Reasoning effort level (none/minimal/low/medium/high/xhigh)

    Yields:
        SSE-formatted event strings
    """
    response_id = f"resp_{uuid.uuid4().hex[:24]}"
    seq = SequenceCounter()

    # Bypass: Detect OpenWebUI ancillary requests (title/tag generation)
    if is_ancillary_request(input_messages):
        anc_type = get_ancillary_type(input_messages) or "other"
        user_query = extract_user_query(input_messages)

        logger.info(
            "[RESPONSES %s] ANCILLARY_STREAM: bypassing agent pipeline, type=%s",
            response_id,
            anc_type,
        )

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
            "reasoning_effort": reasoning_effort,
        }
        llm = create_llm_from_config(provider_config_obj)

        if anc_type == "tags":
            result = await generate_conversation_tags(llm, user_query)
            logger.info(
                "[RESPONSES %s] ANCILLARY_STREAM: completed, type=tags, result=%r",
                response_id,
                result,
            )
            for event in create_tags_stream_events(result, response_id, model):
                yield event
        else:
            result = await generate_conversation_title(llm, user_query)
            logger.info(
                "[RESPONSES %s] ANCILLARY_STREAM: completed, type=title, result=%r",
                response_id,
                result,
            )
            for event in create_title_stream_events(result, response_id, model):
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
        "reasoning_effort": reasoning_effort,
    }

    # Categorize tools into internal (ESDC) and external (OpenTerminal etc.)
    (
        internal_tool_names,
        external_tool_names,
        external_tool_specs,
    ) = categorize_tools(tools)
    external_langchain_tools = convert_external_specs_to_langchain(external_tool_specs)

    # DEBUG: Log tool categorization for OpenTerminal passthrough investigation
    raw_tool_names = []
    if tools:
        raw_tool_names = [
            t.get("function", {}).get("name", t.get("name", "?"))
            if isinstance(t, dict)
            else str(t)
            for t in tools
        ]
    logger.debug(
        "[RESPONSES %s] TOOL_CATEGORIZATION: "
        "tools_received=%d, raw_names=%s, "
        "internal_count=%d, internal_names=%s, "
        "external_count=%d, external_names=%s",
        response_id,
        len(tools) if tools else 0,
        raw_tool_names[:20],
        len(internal_tool_names),
        sorted(internal_tool_names)[:5],
        len(external_tool_names),
        sorted(external_tool_names),
    )

    if external_tool_names:
        logger.info(
            "[RESPONSES %s] External tools detected: %s",
            response_id,
            sorted(external_tool_names),
        )

    llm = create_llm_from_config(provider_config_obj)
    agent = create_agent(
        llm,
        checkpointer=None,
        external_tool_names=external_tool_names,
        external_tools=external_langchain_tools if external_langchain_tools else None,
    )

    # DEBUG: Log agent creation params
    logger.debug(
        "[RESPONSES %s] AGENT_CREATED: external_tool_names=%s, "
        "external_langchain_count=%d",
        response_id,
        sorted(external_tool_names),
        len(external_langchain_tools) if external_langchain_tools else 0,
    )

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
        "[INFERENCE] responses_stream_input | response_id=%s | messages=%d | system=%d | user=%d | ai=%d | total_chars=%d",  # noqa: E501
        response_id,
        len(lc_messages),
        system_msgs,
        user_msgs,
        ai_msgs,
        total_chars,
    )

    stream_start = time.perf_counter()

    # State tracking for real token-level streaming
    current_item_id = ""
    content_started = False
    accumulated_text = ""
    delta_count = 0
    content_index = 0
    last_usage: dict[str, Any] | None = None

    # Track external tool call arguments for streaming in function_call items
    # Maps tool_call_id -> JSON args string
    pending_external_tool_args: dict[str, str] = {}

    try:
        stream_deadline = time.perf_counter() + SSE_STREAM_TIMEOUT

        # Stream agent events using astream_events for real token-level streaming
        async for event in astream_agent_events(agent, lc_messages):
            if time.perf_counter() > stream_deadline:
                raise asyncio.TimeoutError()

            event_type = event["type"]
            event_counter += 1

            # ---- Token events: emit deltas immediately ----
            if event_type == "token":
                content = event["content"]
                if not content:
                    continue

                # First token for this message → emit item.added + content_part.added
                if not content_started:
                    current_item_id = generate_item_id("msg")
                    content_started = True
                    accumulated_text = ""
                    delta_count = 0

                    message_item = {
                        "id": current_item_id,
                        "type": "message",
                        "status": "in_progress",
                        "role": "assistant",
                        "content": [],
                    }
                    added_seq = seq.next()
                    logger.debug(
                        "[RESPONSES %s] EMIT response.output_item.added "
                        "seq=%d, output_index=%d, type=message, id=%s",
                        response_id,
                        added_seq,
                        output_index,
                        current_item_id,
                    )
                    yield format_sse_event(
                        create_output_item_added_event(
                            added_seq, output_index, message_item
                        )
                    )

                    content_part = {"type": "output_text", "text": ""}
                    part_added_seq = seq.next()
                    logger.debug(
                        "[RESPONSES %s] EMIT response.content_part.added "
                        "seq=%d, output_index=%d, content_index=%d, item_id=%s",
                        response_id,
                        part_added_seq,
                        output_index,
                        content_index,
                        current_item_id,
                    )
                    yield format_sse_event(
                        create_content_part_added_event(
                            part_added_seq,
                            output_index,
                            content_index,
                            current_item_id,
                            content_part,
                        )
                    )

                # Emit delta immediately — real streaming!
                if not current_item_id:
                    continue
                accumulated_text += content
                delta_count += 1
                delta_seq = seq.next()
                yield format_sse_event(
                    create_output_text_delta_event(
                        delta_seq, output_index, content_index, current_item_id, content
                    )
                )

            # ---- Reasoning token events ----
            elif event_type == "reasoning_token":
                # Models that emit reasoning_content field get a reasoning output item
                reasoning_content = event["content"]
                if not reasoning_content:
                    continue

                # For now, emit reasoning tokens as text deltas in the same
                # content stream. OpenWebUI detects thinking tags natively.
                # TODO: Emit as proper reasoning output item per Open Responses spec.
                if not content_started:
                    current_item_id = generate_item_id("msg")
                    content_started = True
                    accumulated_text = ""
                    delta_count = 0

                    message_item = {
                        "id": current_item_id,
                        "type": "message",
                        "status": "in_progress",
                        "role": "assistant",
                        "content": [],
                    }
                    added_seq = seq.next()
                    yield format_sse_event(
                        create_output_item_added_event(
                            added_seq, output_index, message_item
                        )
                    )

                    content_part = {"type": "output_text", "text": ""}
                    part_added_seq = seq.next()
                    yield format_sse_event(
                        create_content_part_added_event(
                            part_added_seq,
                            output_index,
                            content_index,
                            current_item_id,
                            content_part,
                        )
                    )

                if not current_item_id:
                    continue
                accumulated_text += reasoning_content
                delta_count += 1
                delta_seq = seq.next()
                yield format_sse_event(
                    create_output_text_delta_event(
                        delta_seq,
                        output_index,
                        content_index,
                        current_item_id,
                        reasoning_content,
                    )
                )

            # ---- Message complete: close content, emit tool calls ----
            elif event_type == "message_complete":
                ai_message = event["ai_message"]

                # Track usage from last LLM response
                if hasattr(ai_message, "usage_metadata") and ai_message.usage_metadata:
                    last_usage = ai_message.usage_metadata

                # Close any open text content
                if content_started and accumulated_text:
                    # Emit text done using accumulated_text (exactly what was streamed)
                    text_done_seq = seq.next()
                    logger.debug(
                        "[RESPONSES %s] EMIT response.output_text.done "
                        "seq=%d, text_len=%d, deltas_sent=%d",
                        response_id,
                        text_done_seq,
                        len(accumulated_text),
                        delta_count,
                    )
                    yield format_sse_event(
                        create_output_text_done_event(
                            text_done_seq,
                            output_index,
                            content_index,
                            current_item_id,
                            accumulated_text,
                        )
                    )

                    # Emit content part done
                    content_part_done = {
                        "type": "output_text",
                        "text": accumulated_text,
                    }
                    part_done_seq = seq.next()
                    yield format_sse_event(
                        create_content_part_done_event(
                            part_done_seq,
                            output_index,
                            content_index,
                            current_item_id,
                            content_part_done,
                        )
                    )

                    # Complete message item
                    message_item = {
                        "id": current_item_id,
                        "type": "message",
                        "status": "completed",
                        "role": "assistant",
                        "content": [content_part_done],
                    }
                    item_done_seq = seq.next()
                    logger.debug(
                        "[RESPONSES %s] EMIT response.output_item.done "
                        "seq=%d, output_index=%d, type=message, id=%s",
                        response_id,
                        item_done_seq,
                        output_index,
                        current_item_id,
                    )
                    yield format_sse_event(
                        create_output_item_done_event(
                            item_done_seq, output_index, message_item
                        )
                    )

                    output_items.append(message_item)
                    output_index += 1

                    # Reset state for next LLM turn (after tool calls)
                    content_started = False
                    accumulated_text = ""
                    delta_count = 0
                    content_index = 0
                    current_item_id = ""

                # Handle tool calls
                if hasattr(ai_message, "tool_calls") and ai_message.tool_calls:
                    for tc_idx, tc in enumerate(ai_message.tool_calls):
                        if not isinstance(tc, dict):
                            logger.warning(
                                "[RESPONSES %s] SKIPPING malformed tool_call[%d]"
                                " - expected dict, got %s",
                                response_id,
                                tc_idx,
                                type(tc).__name__,
                            )
                            continue

                        item_id = generate_item_id("fc")
                        tc_name = tc.get("name", "")
                        tc_id = tc.get("id", "")
                        tc_args = json.dumps(tc.get("args", {}))

                        # Track args for external tool calls
                        if tc_name in external_tool_names:
                            pending_external_tool_args[tc_id] = tc_args

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
                            "[RESPONSES %s] EMIT response.output_item.added "
                            "seq=%d, output_index=%d, "
                            "type=function_call, id=%s, name=%s",
                            response_id,
                            added_seq,
                            output_index,
                            item_id,
                            tc_name,
                        )
                        yield format_sse_event(
                            create_output_item_added_event(
                                added_seq, output_index, function_call_item
                            )
                        )

                        # Stream argument deltas
                        arg_delta_count = 0
                        for arg_chunk in chunk_json(tc_args):
                            function_call_item["arguments"] += arg_chunk
                            arg_delta_count += 1
                            delta_seq = seq.next()
                            yield format_sse_event(
                                create_function_call_arguments_delta_event(
                                    delta_seq, output_index, item_id, arg_chunk
                                )
                            )

                        # Emit args done
                        args_done_seq = seq.next()
                        logger.debug(
                            "[RESPONSES %s] EMIT response.function_call_arguments.done "
                            "seq=%d, args_len=%d, deltas_sent=%d",
                            response_id,
                            args_done_seq,
                            len(tc_args),
                            arg_delta_count,
                        )
                        yield format_sse_event(
                            create_function_call_arguments_done_event(
                                args_done_seq, output_index, item_id, tc_args
                            )
                        )

                        # Complete item
                        function_call_item["status"] = "completed"
                        item_done_seq = seq.next()
                        yield format_sse_event(
                            create_output_item_done_event(
                                item_done_seq, output_index, function_call_item
                            )
                        )

                        output_items.append(function_call_item)
                        output_index += 1

            # ---- Tool result events ----
            elif event_type == "tool_result":
                tool_name = event["tool_name"]
                tool_result_content = event["result"]
                tool_call_id = event.get("tool_call_id", "")

                item_id = generate_item_id("fco")
                content_preview = (
                    tool_result_content[:100]
                    if len(tool_result_content) > 100
                    else tool_result_content
                )

                logger.debug(
                    "[RESPONSES %s] Tool result: tool=%s, call_id=%s, content_len=%d",
                    response_id,
                    tool_name,
                    tool_call_id,
                    len(tool_result_content),
                )

                function_call_output = {
                    "id": item_id,
                    "type": "function_call_output",
                    "status": "completed",
                    "call_id": tool_call_id,
                    "output": [{"type": "input_text", "text": tool_result_content}],
                }

                # Emit item added
                added_seq = seq.next()
                yield format_sse_event(
                    create_output_item_added_event(
                        added_seq, output_index, function_call_output
                    )
                )

                # Emit item done
                function_call_output["status"] = "completed"
                done_seq = seq.next()
                yield format_sse_event(
                    create_output_item_done_event(
                        done_seq, output_index, function_call_output
                    )
                )

                output_items.append(function_call_output)
                output_index += 1

            # ---- External tool call events (passthrough to OpenWebUI) ----
            elif event_type == "external_tool_call":
                tool_name = event.get("tool_name", "unknown")
                tool_call_id = event.get("tool_call_id", "")
                marker_tool_name = event.get("marker_tool_name", tool_name)

                # Get stored args for this external tool call
                args_str = pending_external_tool_args.pop(tool_call_id, "{}")

                logger.info(
                    "[RESPONSES %s] EXTERNAL_TOOL_CALL: name=%s, "
                    "call_id=%s, args_len=%d",
                    response_id,
                    marker_tool_name,
                    tool_call_id,
                    len(args_str),
                )

                # Emit function_call output item for OpenWebUI to execute
                item_id = generate_item_id("fc")
                function_call_item = {
                    "id": item_id,
                    "type": "function_call",
                    "status": "in_progress",
                    "name": marker_tool_name,
                    "call_id": tool_call_id,
                    "arguments": "",
                }

                added_seq = seq.next()
                yield format_sse_event(
                    create_output_item_added_event(
                        added_seq, output_index, function_call_item
                    )
                )

                # Stream argument deltas
                for arg_chunk in chunk_json(args_str):
                    function_call_item["arguments"] += arg_chunk
                    delta_seq = seq.next()
                    yield format_sse_event(
                        create_function_call_arguments_delta_event(
                            delta_seq, output_index, item_id, arg_chunk
                        )
                    )

                # Emit args done
                args_done_seq = seq.next()
                yield format_sse_event(
                    create_function_call_arguments_done_event(
                        args_done_seq, output_index, item_id, args_str
                    )
                )

                # Complete item — external tools are completed immediately
                function_call_item["status"] = "completed"
                item_done_seq = seq.next()
                yield format_sse_event(
                    create_output_item_done_event(
                        item_done_seq, output_index, function_call_item
                    )
                )

                output_items.append(function_call_item)
                output_index += 1

            # ---- Context metadata (informational, not emitted as SSE) ----
            elif event_type == "context_metadata":
                logger.debug(
                    "[RESPONSES %s] Context metadata: %s",
                    response_id,
                    event.get("metadata"),
                )

            # ---- Recursion error: emit response.failed ----
            elif event_type == "recursion_error":
                logger.error(
                    "[RESPONSES %s] RECURSION_ERROR: %s",
                    response_id,
                    event["message"],
                )
                raise _RecursionLimitExceededError(event["message"])

        # Emit response.completed
        completed_seq = seq.next()
        output_summary = [
            f"{item.get('type')}:{item.get('name', item.get('call_id', 'msg'))}"
            for item in output_items
        ]
        logger.info(
            "[RESPONSES %s] COMPLETED - events=%d, "
            "output_items=%d, items=[%s], final_seq=%d",
            response_id,
            event_counter,
            len(output_items),
            ", ".join(output_summary),
            completed_seq,
        )

        total_elapsed_ms = (time.perf_counter() - stream_start) * 1000
        logger.debug(
            "[TIMING] %s stream_complete | events=%d items=%d total_ms=%.2f",
            response_id,
            event_counter,
            len(output_items),
            total_elapsed_ms,
        )
        yield format_sse_event(
            create_response_completed_event(
                completed_seq, response_id, model, output_items, usage=last_usage
            )
        )

    except asyncio.TimeoutError:
        logger.error(
            "[RESPONSES %s] TIMEOUT after %ds", response_id, SSE_STREAM_TIMEOUT
        )
        yield format_sse_event(
            create_response_failed_event(
                seq.next(),
                response_id,
                model,
                {
                    "message": f"Response timed out after {SSE_STREAM_TIMEOUT} seconds",
                    "type": "timeout",
                },
            )
        )
    except _RecursionLimitExceededError as e:
        logger.error("[RESPONSES %s] RECURSION_LIMIT: %s", response_id, str(e))
        yield format_sse_event(
            create_response_failed_event(
                seq.next(),
                response_id,
                model,
                {"message": str(e), "type": "recursion_limit"},
            )
        )
    except Exception as e:
        logger.exception("[RESPONSES %s] ERROR: %s", response_id, e)
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
    reasoning_effort: str | None = None,
) -> dict[str, Any]:
    """Generate non-streaming Responses API response.

    Collects all output from agent and returns a complete response object.

    Args:
        input_messages: Input (string or list of items)
        model: Model ID
        instructions: System message / instructions
        tools: Tools available to the model. If provided, tools are categorized
            into internal (ESDC) and external (e.g. OpenTerminal). Internal tools
            are executed server-side. External tools are returned as function_call
            items for OpenWebUI to handle.
        temperature: Sampling temperature
        reasoning_effort: Reasoning effort level (none/minimal/low/medium/high/xhigh)

    Returns:
        Complete response object
    """
    from esdc.server.responses_events import create_non_streaming_response

    response_id = f"resp_{uuid.uuid4().hex[:24]}"
    output_items: list[dict[str, Any]] = []

    # Bypass: Detect OpenWebUI ancillary requests (title/tag generation)
    if is_ancillary_request(input_messages):
        anc_type = get_ancillary_type(input_messages) or "other"
        user_query = extract_user_query(input_messages)

        logger.info(
            "[RESPONSES %s] ANCILLARY: bypassing agent pipeline, type=%s",
            response_id,
            anc_type,
        )

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
            "reasoning_effort": reasoning_effort,
        }
        llm = create_llm_from_config(provider_config_obj)

        anc_start = time.perf_counter()
        if anc_type == "tags":
            result = await generate_conversation_tags(llm, user_query)
            anc_elapsed = (time.perf_counter() - anc_start) * 1000
            logger.info(
                "[RESPONSES %s] ANCILLARY: completed in %.0fms, type=tags, result=%r",
                response_id,
                anc_elapsed,
                result,
            )
            return create_tags_sync_response(result, response_id)
        else:
            result = await generate_conversation_title(llm, user_query)
            anc_elapsed = (time.perf_counter() - anc_start) * 1000
            logger.info(
                "[RESPONSES %s] ANCILLARY: completed in %.0fms, type=title, result=%r",
                response_id,
                anc_elapsed,
                result,
            )
            return create_title_sync_response(result, response_id)

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
        "reasoning_effort": reasoning_effort,
    }

    # Categorize tools into internal and external
    (
        sync_internal_tool_names,
        sync_external_tool_names,
        sync_external_tool_specs,
    ) = categorize_tools(tools)
    sync_external_langchain_tools = convert_external_specs_to_langchain(
        sync_external_tool_specs
    )

    # DEBUG: Log tool categorization for OpenTerminal passthrough investigation
    sync_raw_tool_names = []
    if tools:
        sync_raw_tool_names = [
            t.get("function", {}).get("name", t.get("name", "?"))
            if isinstance(t, dict)
            else str(t)
            for t in tools
        ]
    logger.debug(
        "[RESPONSES %s] SYNC TOOL_CATEGORIZATION: "
        "tools_received=%d, raw_names=%s, "
        "internal_count=%d, internal_names=%s, "
        "external_count=%d, external_names=%s",
        response_id,
        len(tools) if tools else 0,
        sync_raw_tool_names[:20],
        len(sync_internal_tool_names),
        sorted(sync_internal_tool_names)[:5],
        len(sync_external_tool_names),
        sorted(sync_external_tool_names),
    )

    if sync_external_tool_names:
        logger.info(
            "[RESPONSES %s] SYNC External tools detected: %s",
            response_id,
            sorted(sync_external_tool_names),
        )

    llm = create_llm_from_config(provider_config_obj)
    agent = create_agent(
        llm,
        checkpointer=None,
        external_tool_names=sync_external_tool_names,
        external_tools=sync_external_langchain_tools
        if sync_external_langchain_tools
        else None,
    )

    # DEBUG: Log sync agent creation params
    logger.debug(
        "[RESPONSES %s] SYNC AGENT_CREATED: external_tool_names=%s, "
        "external_langchain_count=%d",
        response_id,
        sorted(sync_external_tool_names),
        len(sync_external_langchain_tools) if sync_external_langchain_tools else 0,
    )

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
        "[INFERENCE] responses_sync_input | response_id=%s | messages=%d | system=%d | user=%d | ai=%d | total_chars=%d",  # noqa: E501
        response_id,
        len(lc_messages),
        system_msgs,
        user_msgs,
        ai_msgs,
        total_chars,
    )

    sync_start = time.perf_counter()

    try:
        all_function_calls: list[dict[str, Any]] = []
        all_tool_results: list[dict[str, Any]] = []
        accumulated_text = ""
        last_message_content = ""
        had_recursion_error = False

        async for event in astream_agent_events(agent, lc_messages):
            event_counter += 1
            event_type = event["type"]

            if event_type == "token" or event_type == "reasoning_token":
                content = event["content"]
                if content:
                    accumulated_text += content

            elif event_type == "message_complete":
                ai_message = event["ai_message"]
                has_tool_calls = bool(ai_message.tool_calls)

                if has_tool_calls:
                    for tc in ai_message.tool_calls:
                        if isinstance(tc, dict):
                            all_function_calls.append(
                                {
                                    "id": generate_item_id("fc"),
                                    "type": "function_call",
                                    "status": "completed",
                                    "name": tc.get("name", ""),
                                    "call_id": tc.get("id", ""),
                                    "arguments": json.dumps(tc.get("args", {})),
                                }
                            )

                if accumulated_text.strip():
                    last_message_content = accumulated_text
                    accumulated_text = ""

            elif event_type == "tool_result":
                tool_name = event.get("tool_name", "unknown")
                tool_call_id = event.get("tool_call_id", "")
                result_text = event.get("result", "")

                all_tool_results.append(
                    {
                        "id": generate_item_id("fco"),
                        "type": "function_call_output",
                        "status": "completed",
                        "call_id": tool_call_id,
                        "output": [
                            {
                                "type": "input_text",
                                "text": result_text,
                            }
                        ],
                    }
                )
                logger.debug(
                    "[RESPONSES %s] SYNC tool_result:"
                    " tool=%s, call_id=%s, result_len=%d",
                    response_id,
                    tool_name,
                    tool_call_id,
                    len(result_text),
                )

            elif event_type == "external_tool_call":
                tool_name = event.get("tool_name", "unknown")
                tool_call_id = event.get("tool_call_id", "")
                marker_tool_name = event.get("marker_tool_name", tool_name)

                logger.info(
                    "[RESPONSES %s] SYNC EXTERNAL_TOOL_CALL: name=%s, call_id=%s",
                    response_id,
                    marker_tool_name,
                    tool_call_id,
                )

                all_function_calls.append(
                    {
                        "id": generate_item_id("fc"),
                        "type": "function_call",
                        "status": "completed",
                        "name": marker_tool_name,
                        "call_id": tool_call_id,
                        "arguments": "{}",
                    }
                )

            elif event_type == "recursion_error":
                logger.error(
                    "[RESPONSES %s] SYNC RECURSION_ERROR: %s",
                    response_id,
                    event["message"],
                )
                return create_non_streaming_response(
                    response_id,
                    model,
                    [],
                    error={"message": event["message"], "type": "recursion_limit"},
                )

            elif event_type == "context_metadata":
                pass

        output_items.extend(all_function_calls)
        output_items.extend(all_tool_results)

        if last_message_content.strip():
            logger.debug(
                "[RESPONSES %s] SYNC Building message item: content_len=%d",
                response_id,
                len(last_message_content),
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
                            "text": last_message_content,
                            "annotations": [],
                        }
                    ],
                }
            )

        output_summary = [
            f"{item.get('type')}:{item.get('name', item.get('call_id', 'msg'))}"
            for item in output_items
        ]
        logger.info(
            f"[RESPONSES {response_id}] SYNC COMPLETED - "
            f"events={event_counter}, output_items={len(output_items)}, "
            f"items=[{', '.join(output_summary)}]"
        )

        sync_elapsed_ms = (time.perf_counter() - sync_start) * 1000
        logger.debug(
            "[INFERENCE] responses_sync_complete"
            " | response_id=%s | elapsed_ms=%.2f"
            " | events=%d | output_items=%d"
            " | recursion_error=%s",
            response_id,
            sync_elapsed_ms,
            event_counter,
            len(output_items),
            had_recursion_error,
        )
        return create_non_streaming_response(response_id, model, output_items)

    except Exception as e:
        error_elapsed_ms = (time.perf_counter() - sync_start) * 1000
        logger.debug(
            "[INFERENCE] responses_sync_error"
            " | response_id=%s | elapsed_ms=%.2f | error=%s",
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
