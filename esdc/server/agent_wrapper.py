"""Chat Completions API wrapper for LangGraph agent.

This module provides OpenAI-compatible chat completion endpoints that:
- Convert messages between OpenAI and LangChain formats
- Handle OpenWebUI's extended format with output arrays
- Stream responses with character-level chunking
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
from langchain_core.runnables import RunnableConfig

# Local
from esdc.chat.agent import create_agent
from esdc.configs import Config
from esdc.providers import create_llm_from_config
from esdc.server.content_accumulator import ContentAccumulator, format_tool_section
from esdc.server.message_utils import extract_ai_message_from_event, extract_content_str
from esdc.server.responses_wrapper import extract_tool_messages_from_event
from esdc.server.stream_utils import chunk_text
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
            # Pydantic model
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
            # Default to assistant role if not specified
            role = item.get("role", "assistant")
            content_parts = item.get("content", [])

            # Extract text from content parts
            text_parts = []
            for part in content_parts:
                if isinstance(part, dict):
                    ptype = part.get("type", "")
                    if ptype in ("output_text", "input_text", "text"):
                        text_parts.append(part.get("text", ""))
                elif isinstance(part, str):
                    text_parts.append(part)

            text = "\n".join(text_parts) if text_parts else ""

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

            try:
                args = json.loads(args_str) if args_str else {}
            except json.JSONDecodeError:
                args = {}

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
    model: str = "esdc-agent",
    finish_reason: str | None = None,
) -> dict:
    """Create OpenAI-compatible streaming chunk."""
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
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
    model: str = "esdc-agent",
    temperature: float = 0.7,
    use_native_format: bool = True,
    request_id: str | None = None,
) -> AsyncGenerator[str, None]:
    """Generate streaming chat completion response with character-level streaming.

    Streams content 3 characters at a time using chunk_text(), matching
    the Responses API implementation for consistent streaming behavior.

    Args:
        messages: List of conversation messages
        model: Model ID
        temperature: Sampling temperature
        use_native_format: Whether to use native tool_calls format or markdown
        request_id: Optional request ID for tracking (auto-generated if not provided)

    Yields:
        OpenAI-compatible streaming chunks as SSE formatted strings
    """
    if not request_id:
        request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

    # DEBUG: Counters for tracking execution paths
    event_counter = 0
    yield_counter = 0
    seen_msg_ids: set[str] = set()

    logger.debug("=" * 80)
    logger.debug(
        f"[{request_id}] STREAM_START - use_native_format={use_native_format}, "
        f"num_messages={len(messages)}"
    )

    try:
        # Create LLM and agent
        provider_config = Config.get_provider_config()
        if not provider_config:
            yield_counter += 1
            logger.warning(
                f"[{request_id}] ERROR - No provider configured, yield #{yield_counter}"
            )
            yield json.dumps(
                create_openai_chunk(
                    content="Error: No provider configured.",
                    model=model,
                    finish_reason="stop",
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
        }

        llm = create_llm_from_config(provider_config_obj)
        agent = create_agent(llm, checkpointer=None)

        # Convert messages
        lc_messages = convert_messages_to_langchain(messages)

        # Track first message
        is_first_ai_message = True

        # Stream the response
        async for event in agent.astream(
            {"messages": lc_messages},
            config=RunnableConfig(configurable={"thread_id": request_id}),
        ):
            event_counter += 1
            logger.debug(
                f"[{request_id}] EVENT #{event_counter} - keys={list(event.keys())}"
            )

            # Check for tool messages FIRST (before AI messages)
            if use_native_format:
                tool_messages = extract_tool_messages_from_event(event)
                logger.debug(
                    f"[{request_id}] EVENT #{event_counter} - "
                    f"extracted {len(tool_messages)} tool_messages from event"
                )
                if tool_messages:
                    for tool_msg in tool_messages:
                        tool_call_id = tool_msg.get("tool_call_id", "")
                        content = tool_msg.get("content", "")
                        chunk = create_tool_role_chunk(
                            tool_call_id=tool_call_id,
                            content=content,
                            model=model,
                        )
                        yield_counter += 1
                        logger.debug(
                            f"[{request_id}] YIELD #{yield_counter} - tool_role_chunk, "
                            f"tool_call_id={tool_call_id}, content_len={len(content)}"
                        )
                        yield json.dumps(chunk)
                    continue  # Skip AI message handling for tool events

            ai_msg = extract_ai_message_from_event(event)
            if not ai_msg:
                logger.debug(
                    f"[{request_id}] EVENT #{event_counter} - "
                    f"No AIMessage extracted, skipping"
                )
                continue

            # DEBUG: Log AIMessage details with duplicate detection
            msg_id = getattr(ai_msg, "id", "no-id")
            msg_content_preview = (
                str(ai_msg.content)[:100] if ai_msg.content else "None"
            )
            has_tool_calls = hasattr(ai_msg, "tool_calls") and ai_msg.tool_calls
            tool_call_count = len(ai_msg.tool_calls) if has_tool_calls else 0

            if msg_id in seen_msg_ids:
                logger.warning(
                    f"[{request_id}] EVENT #{event_counter} - "
                    f"DUPLICATE AI MESSAGE! msg_id={msg_id} already seen, "
                    f"content_preview='{msg_content_preview}...'"
                )
            else:
                seen_msg_ids.add(msg_id)
                logger.debug(
                    f"[{request_id}] EVENT #{event_counter} - "
                    f"AIMessage id={msg_id}, "
                    f"content_preview='{msg_content_preview}...', "
                    f"tool_calls={tool_call_count}"
                )

            # Handle different message types
            if is_first_ai_message:
                content_str = extract_content_str(ai_msg.content)

                # Stream content character-by-character (3 chars at a time)
                if content_str and content_str.strip():
                    for chunk in chunk_text(content_str):
                        yield_counter += 1
                        yield json.dumps(
                            create_openai_chunk(content=chunk, model=model)
                        )

                # Emit tool calls after content
                if (
                    hasattr(ai_msg, "tool_calls")
                    and ai_msg.tool_calls
                    and use_native_format
                ):
                    tool_calls_dict = [
                        {
                            "name": tc.get("name", ""),
                            "args": tc.get("args", {}),
                            "id": tc.get("id", ""),
                        }
                        for tc in ai_msg.tool_calls
                    ]
                    chunk = create_tool_call_chunk(tool_calls_dict, model)
                    yield_counter += 1
                    logger.debug(
                        f"[{request_id}] YIELD #{yield_counter} - tool_calls_chunk, "
                        f"tool_count={len(tool_calls_dict)}"
                    )
                    yield json.dumps(chunk)

                is_first_ai_message = False

            elif hasattr(ai_msg, "tool_calls") and ai_msg.tool_calls:
                if use_native_format:
                    # Emit tool calls chunk (no buffering)
                    tool_calls_dict = [
                        {
                            "name": tc.get("name", ""),
                            "args": tc.get("args", {}),
                            "id": tc.get("id", ""),
                        }
                        for tc in ai_msg.tool_calls
                    ]
                    chunk = create_tool_call_chunk(tool_calls_dict, model)
                    yield_counter += 1
                    logger.debug(
                        f"[{request_id}] YIELD #{yield_counter} - tool_calls_chunk, "
                        f"tool_count={len(tool_calls_dict)}"
                    )
                    yield json.dumps(chunk)
                else:
                    # Legacy markdown mode: stream tool info
                    for tool_call in ai_msg.tool_calls:
                        tool_name = tool_call.get("name", "unknown")
                        tool_args = tool_call.get("args", {})
                        tool_section = format_tool_section(tool_name, tool_args)
                        for chunk in chunk_text(tool_section):
                            yield_counter += 1
                            yield json.dumps(
                                create_openai_chunk(content=chunk, model=model)
                            )

            else:
                # Stream content character-by-character
                content_str = extract_content_str(ai_msg.content)
                if content_str and content_str.strip():
                    for chunk in chunk_text(content_str):
                        yield_counter += 1
                        yield json.dumps(
                            create_openai_chunk(content=chunk, model=model)
                        )

        # Send final chunk
        logger.debug(
            f"[{request_id}] STREAM_END - events={event_counter}, "
            f"yields={yield_counter}, unique_msg_ids={len(seen_msg_ids)}"
        )

        if use_native_format:
            final_chunk = create_final_chunk(model)
            yield_counter += 1
            logger.debug(
                f"[{request_id}] YIELD #{yield_counter} - "
                f"final_chunk, finish_reason=stop"
            )
            yield json.dumps(final_chunk)
        else:
            final_chunk = create_openai_chunk(
                content="",
                model=model,
                finish_reason="stop",
            )
            yield_counter += 1
            logger.debug(
                f"[{request_id}] YIELD #{yield_counter} - "
                f"final_chunk (legacy), finish_reason=stop"
            )
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
        )
        yield_counter += 1
        yield json.dumps(error_chunk)


async def generate_response(
    messages: list,
    model: str = "esdc-agent",
    temperature: float = 0.7,
    use_native_format: bool = True,
) -> dict[str, Any]:
    """Generate non-streaming chat completion response with markdown formatting.

    Args:
        messages: List of conversation messages
        model: Model ID
        temperature: Sampling temperature
        use_native_format: Whether to use native tool_calls format or markdown

    Returns:
        OpenAI-compatible response dictionary
    """
    try:
        # Create LLM and agent
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
        }

        llm = create_llm_from_config(provider_config_obj)
        agent = create_agent(llm, checkpointer=None)

        # Convert messages
        lc_messages = convert_messages_to_langchain(messages)

        # Accumulate content menggunakan buffer
        buffer = ContentAccumulator()
        stored_tool_calls = []  # Store tool calls for native format
        is_first_ai_message = True

        # Run the agent
        async for event in agent.astream(
            {"messages": lc_messages},
            config=RunnableConfig(
                configurable={"request_id": f"esdc-{int(time.time())}"}
            ),
        ):
            ai_msg = extract_ai_message_from_event(event)
            if not ai_msg:
                continue

            # Handle different message types
            if is_first_ai_message:
                content_str = extract_content_str(ai_msg.content)
                if content_str and content_str.strip():
                    buffer.add_content(content_str)
                is_first_ai_message = False

            elif hasattr(ai_msg, "tool_calls") and ai_msg.tool_calls:
                # Only flush buffer for markdown mode
                if not use_native_format and buffer.has_content():
                    buffer.flush()

                # Store tool calls for final response (convert to dict format)
                stored_tool_calls = [
                    {
                        "name": tc.get("name", ""),
                        "args": tc.get("args", {}),
                        "id": tc.get("id", ""),
                    }
                    for tc in ai_msg.tool_calls
                ]

                if not use_native_format:
                    # Add to buffer for markdown mode
                    for tool_call in ai_msg.tool_calls:
                        tool_name = tool_call.get("name", "unknown")
                        tool_args = tool_call.get("args", {})
                        buffer.add_tool_call(tool_name, tool_args)
                        buffer.flush()  # Flush each tool immediately

            else:
                # Final content
                content_str = extract_content_str(ai_msg.content)
                if content_str:
                    buffer.add_content(content_str)

        # Build final response
        if use_native_format:
            # Import formatter
            from esdc.server.tool_formatter import format_tool_calls_for_response

            final_content = buffer.flush_final()
            tool_calls_formatted = (
                format_tool_calls_for_response(stored_tool_calls)
                if stored_tool_calls
                else None
            )

            return {
                "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
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
            # Legacy markdown response
            final_content = buffer.flush_final()
            return {
                "content": final_content if final_content else "No response generated",
                "role": "assistant",
                "finish_reason": "stop",
            }

    except Exception as e:
        logger.error(f"Error in response generation: {e}", exc_info=True)
        return {
            "content": f"Error: {str(e)}",
            "role": "assistant",
            "finish_reason": "stop",
        }
