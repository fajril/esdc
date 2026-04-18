"""Chat Completions API wrapper for LangGraph agent.

This module provides OpenAI-compatible chat completion endpoints that:
- Convert messages between OpenAI and LangChain formats
- Handle OpenWebUI's extended format with output arrays
- Stream responses with real token-level streaming via astream_events
- Support native tool calling and markdown formats

Key Functions:
    - convert_messages_to_langchain: Convert OpenAI messages to LangChain
    - generate_streaming_response: SSE streaming for chat completions
    - generate_response: Non-streaming response generation
"""

# Standard library
import json
import logging
import time
import uuid
from collections.abc import AsyncGenerator
from typing import Any

# Third-party
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

# Local
from esdc.chat.agent import create_agent
from esdc.configs import Config
from esdc.providers import create_llm_from_config
from esdc.server.cache import get_parsed_json
from esdc.server.constants import SSE_STREAM_TIMEOUT
from esdc.server.content_accumulator import ContentAccumulator
from esdc.server.event_streamer import astream_agent_events
from esdc.server.tool_formatter import (
    create_final_chunk,
    create_tool_call_chunk,
    create_tool_role_chunk,
)

logger = logging.getLogger("esdc.server.agent")


def convert_messages_to_langchain(messages: list[Any]) -> list[Any]:
    """Convert OpenAI-compatible messages to LangChain messages.

    Handles both standard Chat Completions format and OpenWebUI format
    with 'output' field for assistant messages containing function_call items.

    OpenWebUI sends conversation history with output arrays like:
    [
      {"type": "message", "content": [{"type": "output_text", "text": "..."}]},
      {"type": "function_call", "call_id": "...", "name": "..."},
      {"type": "function_call_output", "call_id": "...", "output": [...]}
    ]

    This needs to be converted to LangChain's format of AIMessage (with tool_calls)
    followed by ToolMessage objects.

    Example:
        >>> messages = [{"role": "user", "content": "Hello"}]
        >>> lc_messages = convert_messages_to_langchain(messages)
        >>> isinstance(lc_messages[0], HumanMessage)
        True
    """
    lc_messages: list[Any] = []

    for msg in messages:
        if isinstance(msg, dict):
            role = msg.get("role", "")
            content = msg.get("content", "") or ""
            output = msg.get("output")

            if role == "assistant" and output and isinstance(output, list):
                lc_messages.extend(_convert_output_to_langchain_messages(output))
            elif role == "system":
                lc_messages.append(SystemMessage(content=content))
            elif role == "user":
                lc_messages.append(HumanMessage(content=content))
            elif role == "assistant":
                lc_messages.append(AIMessage(content=content))
            elif role == "tool":
                tool_call_id = msg.get("tool_call_id", "")
                lc_messages.append(
                    ToolMessage(content=content, tool_call_id=tool_call_id)
                )

        else:
            role = getattr(msg, "role", "")
            content = getattr(msg, "content", "") or ""
            output = getattr(msg, "output", None)

            if role == "assistant" and output and isinstance(output, list):
                lc_messages.extend(_convert_output_to_langchain_messages(output))
            elif role == "system":
                lc_messages.append(SystemMessage(content=content))
            elif role == "user":
                lc_messages.append(HumanMessage(content=content))
            elif role == "assistant":
                lc_messages.append(AIMessage(content=content))
            elif role == "tool":
                tool_call_id = getattr(msg, "tool_call_id", "")
                lc_messages.append(
                    ToolMessage(content=content, tool_call_id=tool_call_id)
                )

    return lc_messages


def _convert_output_to_langchain_messages(output: list[Any]) -> list[Any]:
    """Convert OpenWebUI output array to LangChain messages.

    The output array can contain:
    - message items (assistant responses)
    - function_call items (tool invocations)
    - function_call_output items (tool results)

    Returns a list of LangChain messages preserving order and context.
    """
    lc_messages: list[Any] = []

    for item in output:
        if not isinstance(item, dict):
            logger.debug(
                f"[convert_output] Skipping non-dict item: type={type(item).__name__}"
            )
            continue

        item_type = item.get("type")

        if item_type == "message":
            role = item.get("role", "assistant")
            content_parts = item.get("content", [])

            text_parts_str = []
            for part in content_parts:
                if isinstance(part, dict):
                    ptype = part.get("type", "")
                    if ptype in ("output_text", "input_text", "text"):
                        text_parts_str.append(part.get("text", ""))
                elif isinstance(part, str):
                    text_parts_str.append(part)

            text = "\n".join(text_parts_str) if text_parts_str else ""

            if role == "assistant":
                lc_messages.append(AIMessage(content=text))
            elif role == "user":
                lc_messages.append(HumanMessage(content=text))
            elif role == "system":
                lc_messages.append(SystemMessage(content=text))

        elif item_type == "function_call":
            call_id = item.get("call_id", "")
            name = item.get("name", "")
            args_str = item.get("arguments", "{}")

            args = get_parsed_json(args_str)

            lc_messages.append(
                AIMessage(
                    content="",
                    tool_calls=[{"name": name, "args": args, "id": call_id}],
                )
            )

        elif item_type == "function_call_output":
            call_id = item.get("call_id", "")
            output_content = item.get("output", "")

            if isinstance(output_content, list):
                text_parts = []
                for part in output_content:
                    if isinstance(part, dict):
                        ptype = part.get("type", "")
                        if ptype in ("input_text", "output_text", "text"):
                            text_parts.append(part.get("text", ""))
                    elif isinstance(part, str):
                        text_parts.append(part)
                output_text = (
                    "\n".join(text_parts) if text_parts else str(output_content)
                )
            else:
                output_text = str(output_content) if output_content else ""

            lc_messages.append(ToolMessage(content=output_text, tool_call_id=call_id))

        else:
            logger.warning(
                f"[convert_output] Unknown item type '{item_type}' in output array"
            )

    return lc_messages


def create_openai_chunk(
    content: str = "",
    model: str = "iris",
    finish_reason: str | None = None,
    chunk_id: str | None = None,
) -> dict:
    """Create OpenAI-compatible streaming chunk.

    Args:
        content: Content for this chunk
        model: Model ID
        finish_reason: Finish reason (stop, tool_calls, etc.)
        chunk_id: Optional chunk ID (generated if not provided)

    Returns:
        OpenAI-compatible chunk dict
    """
    return {
        "id": chunk_id or f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {"content": content},
                "finish_reason": finish_reason,
            }
        ],
    }


async def generate_streaming_response(
    messages: list,
    model: str = "iris",
    temperature: float = 0.7,
    use_native_format: bool = True,
    request_id: str | None = None,
    reasoning_effort: str | None = None,
) -> AsyncGenerator[str, None]:
    """Generate streaming chat completion response with real token-level streaming.

    Uses astream_agent_events() for token-by-token streaming instead of
    agent.astream() which only yields complete AIMessages per node.

    Args:
        messages: List of conversation messages
        model: Model ID
        temperature: Sampling temperature
        use_native_format: Whether to use native tool_calls format or markdown
        request_id: Optional request ID for tracking (auto-generated if not provided)
        reasoning_effort: Reasoning effort level (none/minimal/low/medium/high/xhigh)

    Yields:
        OpenAI-compatible streaming chunks as SSE formatted strings
    """
    stream_uuid = uuid.uuid4().hex[:12]
    if not request_id:
        request_id = f"chatcmpl-{stream_uuid}"

    chunk_counter = 0

    def next_chunk_id() -> str:
        nonlocal chunk_counter
        chunk_counter += 1
        return f"chatcmpl-{stream_uuid}-{chunk_counter:04d}"

    yield_counter = 0
    stream_start = time.perf_counter()

    logger.debug("=" * 80)
    logger.debug(
        f"[{request_id}] STREAM_START - use_native_format={use_native_format}, "
        f"num_messages={len(messages)}"
    )

    try:
        provider_config = Config.get_provider_config()
        if not provider_config:
            yield_counter += 1
            yield json.dumps(
                create_openai_chunk(
                    content="Error: No provider configured.",
                    model=model,
                    finish_reason="stop",
                    chunk_id=next_chunk_id(),
                )
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
            "reasoning_effort": reasoning_effort,
        }

        llm = create_llm_from_config(provider_config_obj)
        agent = create_agent(llm, checkpointer=None)
        logger.debug(
            f"[{request_id}] [TIMING] llm_and_agent_created | "
            f"elapsed={((time.perf_counter() - stream_start) * 1000):.2f}ms"
        )

        lc_messages = convert_messages_to_langchain(messages)

        total_chars = sum(len(str(m.content)) for m in lc_messages)
        system_msgs = len([m for m in lc_messages if isinstance(m, SystemMessage)])
        user_msgs = len([m for m in lc_messages if isinstance(m, HumanMessage)])
        logger.debug(
            "[INFERENCE] stream_input_prepared"
            " | messages=%d | system=%d | user=%d | total_chars=%d",
            len(lc_messages),
            system_msgs,
            user_msgs,
            total_chars,
        )

        inference_start = time.perf_counter()
        stream_deadline = time.perf_counter() + SSE_STREAM_TIMEOUT

        async for event in astream_agent_events(agent, lc_messages):
            if time.perf_counter() > stream_deadline:
                logger.error(
                    "[%s] STREAM_TIMEOUT after %ds, yielding error and stopping",
                    request_id,
                    SSE_STREAM_TIMEOUT,
                )
                yield_counter += 1
                yield json.dumps(
                    create_openai_chunk(
                        content=(
                            "Maaf, permintaan Anda memerlukan waktu"
                            " terlalu lama untuk diproses."
                            " Silakan sederhanakan pertanyaan atau"
                            " coba lagi."
                        ),
                        model=model,
                        finish_reason="stop",
                        chunk_id=next_chunk_id(),
                    )
                )
                return

            event_type = event["type"]

            if event_type == "token" or event_type == "reasoning_token":
                content = event["content"]
                if content:
                    yield_counter += 1
                    yield json.dumps(
                        create_openai_chunk(
                            content=content, model=model, chunk_id=next_chunk_id()
                        )
                    )

            elif event_type == "message_complete":
                ai_message = event["ai_message"]
                has_tool_calls = bool(ai_message.tool_calls)

                if use_native_format and has_tool_calls:
                    tool_calls_dict = [
                        {
                            "name": tc.get("name", ""),
                            "args": tc.get("args", {}),
                            "id": tc.get("id", ""),
                        }
                        for tc in ai_message.tool_calls
                    ]
                    chunk = create_tool_call_chunk(tool_calls_dict, model)
                    yield_counter += 1
                    logger.debug(
                        f"[{request_id}] YIELD #{yield_counter} - "
                        f"tool_calls_chunk, tool_count={len(tool_calls_dict)}"
                    )
                    yield json.dumps(chunk)

                if not use_native_format and has_tool_calls:
                    from esdc.server.content_accumulator import format_tool_section

                    for tool_call in ai_message.tool_calls:
                        tool_name = tool_call.get("name", "unknown")
                        tool_args = tool_call.get("args", {})
                        tool_section = format_tool_section(tool_name, tool_args)
                        yield_counter += 1
                        yield json.dumps(
                            create_openai_chunk(
                                content=tool_section,
                                model=model,
                                chunk_id=next_chunk_id(),
                            )
                        )

            elif event_type == "tool_result":
                if use_native_format:
                    tool_call_id = event.get("tool_call_id", "")
                    result = event.get("result", "")
                    chunk = create_tool_role_chunk(
                        tool_call_id=tool_call_id,
                        content=result,
                        model=model,
                    )
                    yield_counter += 1
                    logger.debug(
                        f"[{request_id}] YIELD #{yield_counter} - "
                        f"tool_role_chunk, tool_call_id={tool_call_id}"
                    )
                    yield json.dumps(chunk)

            elif event_type == "recursion_error":
                error_message = event["message"]
                yield_counter += 1
                yield json.dumps(
                    create_openai_chunk(
                        content=error_message,
                        model=model,
                        finish_reason="stop",
                        chunk_id=next_chunk_id(),
                    )
                )
                return

            elif event_type == "context_metadata":
                pass

        total_ms = (time.perf_counter() - stream_start) * 1000
        inference_elapsed_ms = (time.perf_counter() - inference_start) * 1000
        logger.debug(f"[{request_id}] STREAM_END - yields={yield_counter}")
        logger.debug(
            "[INFERENCE] stream_complete"
            " | total_ms=%.2f | inference_ms=%.2f | yields=%d",
            total_ms,
            inference_elapsed_ms,
            yield_counter,
        )

        if use_native_format:
            final_chunk = create_final_chunk(model)
            yield_counter += 1
            yield json.dumps(final_chunk)
        else:
            final_chunk = create_openai_chunk(
                content="",
                model=model,
                finish_reason="stop",
                chunk_id=next_chunk_id(),
            )
            yield_counter += 1
            yield json.dumps(final_chunk)

    except Exception as e:
        logger.error(
            f"[{request_id}] FATAL ERROR in streaming: {type(e).__name__}: {str(e)}",
            exc_info=True,
        )
        error_chunk = create_openai_chunk(
            content=f"Error: {str(e)}",
            model=model,
            finish_reason="stop",
            chunk_id=next_chunk_id(),
        )
        yield_counter += 1
        yield json.dumps(error_chunk)


async def generate_response(
    messages: list,
    model: str = "iris",
    temperature: float = 0.7,
    use_native_format: bool = True,
    reasoning_effort: str | None = None,
) -> dict[str, Any]:
    """Generate non-streaming chat completion response.

    Uses astream_agent_events() for consistency with streaming path.

    Args:
        messages: List of conversation messages
        model: Model ID
        temperature: Sampling temperature
        use_native_format: Whether to use native tool_calls format or markdown
        reasoning_effort: Reasoning effort level (none/minimal/low/medium/high/xhigh)

    Returns:
        OpenAI-compatible response dictionary
    """
    response_uuid = uuid.uuid4().hex[:12]
    inference_start = time.perf_counter()

    try:
        provider_config = Config.get_provider_config()
        if not provider_config:
            return {
                "content": "Error: No provider configured.",
                "role": "assistant",
                "finish_reason": "stop",
            }

        provider_name = provider_config.get("provider_type", "ollama")
        provider_model = provider_config.get("model")
        base_url = provider_config.get("base_url")
        api_key = provider_config.get("api_key")

        provider_config_obj = {
            "provider_type": provider_name,
            "model": provider_model,
            "base_url": base_url,
            "api_key": api_key,
            "reasoning_effort": reasoning_effort,
        }

        llm = create_llm_from_config(provider_config_obj)
        agent = create_agent(llm, checkpointer=None)

        lc_messages = convert_messages_to_langchain(messages)

        total_chars = sum(len(str(m.content)) for m in lc_messages)
        system_msgs = len([m for m in lc_messages if isinstance(m, SystemMessage)])
        user_msgs = len([m for m in lc_messages if isinstance(m, HumanMessage)])
        logger.debug(
            "[INFERENCE] sync_input_prepared"
            " | messages=%d | system=%d | user=%d | total_chars=%d",
            len(lc_messages),
            system_msgs,
            user_msgs,
            total_chars,
        )

        buffer = ContentAccumulator()
        stored_tool_calls: list[dict[str, Any]] = []
        accumulated_text = ""
        had_recursion_error = False

        async for event in astream_agent_events(agent, lc_messages):
            event_type = event["type"]

            if event_type == "token" or event_type == "reasoning_token":
                content = event["content"]
                if content:
                    accumulated_text += content

            elif event_type == "message_complete":
                ai_message = event["ai_message"]
                has_tool_calls = bool(ai_message.tool_calls)

                if has_tool_calls:
                    if not use_native_format and buffer.has_content():
                        buffer.flush()

                    stored_tool_calls = [
                        {
                            "name": tc.get("name", ""),
                            "args": tc.get("args", {}),
                            "id": tc.get("id", ""),
                        }
                        for tc in ai_message.tool_calls
                    ]

                    if not use_native_format:
                        for tool_call in ai_message.tool_calls:
                            tool_name = tool_call.get("name", "unknown")
                            tool_args = tool_call.get("args", {})
                            buffer.add_tool_call(tool_name, tool_args)
                            buffer.flush()

                if accumulated_text.strip():
                    buffer.add_content(accumulated_text)
                    accumulated_text = ""

            elif event_type == "recursion_error":
                buffer.add_content(event["message"])
                had_recursion_error = True
                break

        if use_native_format:
            from esdc.server.tool_formatter import format_tool_calls_for_response

            final_content = buffer.flush_final()
            tool_calls_formatted = (
                format_tool_calls_for_response(stored_tool_calls)
                if stored_tool_calls
                else None
            )

            inference_elapsed_ms = (time.perf_counter() - inference_start) * 1000
            logger.debug(
                "[INFERENCE] sync_complete"
                " | elapsed_ms=%.2f | content_len=%d"
                " | tool_calls=%d | recursion_error=%s",
                inference_elapsed_ms,
                len(final_content) if final_content else 0,
                len(stored_tool_calls),
                had_recursion_error,
            )

            return {
                "id": f"chatcmpl-{response_uuid}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": final_content if final_content else None,
                            "tool_calls": tool_calls_formatted,
                        },
                        "finish_reason": "stop",
                    }
                ],
            }
        else:
            final_content = buffer.flush_final()

            inference_elapsed_ms = (time.perf_counter() - inference_start) * 1000
            logger.debug(
                "[INFERENCE] sync_complete (legacy) | elapsed_ms=%.2f | content_len=%d",
                inference_elapsed_ms,
                len(final_content) if final_content else 0,
            )

            return {
                "content": final_content if final_content else "No response generated",
                "role": "assistant",
                "finish_reason": "stop",
            }

    except Exception as e:
        error_elapsed_ms = (time.perf_counter() - inference_start) * 1000
        logger.debug(
            "[INFERENCE] sync_error | elapsed_ms=%.2f | error=%s",
            error_elapsed_ms,
            str(e)[:100],
        )
        logger.error(f"Error in response generation: {e}", exc_info=True)
        return {
            "content": f"Error: {str(e)}",
            "role": "assistant",
            "finish_reason": "stop",
        }
