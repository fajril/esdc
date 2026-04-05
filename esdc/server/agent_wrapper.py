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
from esdc.server.stream_utils import chunk_text
from esdc.server.tool_formatter import (
    create_final_chunk,
    create_tool_call_chunk,
)

logger = logging.getLogger("esdc.server.agent")


def convert_messages_to_langchain(messages: list) -> list:
    """Convert OpenAI-compatible messages to LangChain messages.

    Handles both standard Chat Completions format and Responses API format
    with 'output' field for assistant messages.
    """
    lc_messages = []

    for msg in messages:
        if isinstance(msg, dict):
            role = msg.get("role", "")
            content = msg.get("content", "") or ""

            if role == "assistant" and "output" in msg:
                output = msg["output"]
                if isinstance(output, list):
                    texts = []
                    for item in output:
                        if isinstance(item, dict) and item.get("type") == "message":
                            content_parts = item.get("content", [])
                            for part in content_parts:
                                if (
                                    isinstance(part, dict)
                                    and part.get("type") == "output_text"
                                ):
                                    texts.append(part.get("text", ""))
                    if texts:
                        content = "\n".join(texts)
        else:
            role = getattr(msg, "role", "")
            content = getattr(msg, "content", "") or ""

        if role == "system":
            lc_messages.append(SystemMessage(content=content))
        elif role == "user":
            lc_messages.append(HumanMessage(content=content))
        elif role == "assistant":
            lc_messages.append(AIMessage(content=content))
        elif role == "tool":
            tool_call_id = (
                msg.get("tool_call_id", "")
                if isinstance(msg, dict)
                else getattr(msg, "tool_call_id", "")
            )
            lc_messages.append(ToolMessage(content=content, tool_call_id=tool_call_id))

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


async def generate_streaming_response(
    messages: list,
    model: str = "esdc-agent",
    temperature: float = 0.7,
    use_native_format: bool = True,
) -> AsyncGenerator[str, None]:
    """Generate streaming chat completion response with character-level streaming.

    Streams content 3 characters at a time using chunk_text(), matching
    the Responses API implementation for consistent streaming behavior.

    Args:
        messages: List of conversation messages
        model: Model ID
        temperature: Sampling temperature
        use_native_format: Whether to use native tool_calls format or markdown

    Yields:
        OpenAI-compatible streaming chunks as SSE formatted strings
    """
    request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

    # DEBUG: Counters for tracking execution paths
    event_counter = 0
    path_first_msg_count = 0
    path_tool_calls_count = 0
    path_final_content_count = 0

    logger.debug("=" * 80)
    logger.debug(
        f"[STREAM_START] request_id={request_id}, use_native_format={use_native_format}"
    )
    logger.debug(f"[STREAM_START] num_messages={len(messages)}")

    try:
        # Create LLM and agent
        provider_config = Config.get_provider_config()
        if not provider_config:
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
            logger.debug(f"[EVENT #{event_counter}] Event keys: {list(event.keys())}")

            ai_msg = extract_ai_message_from_event(event)
            if not ai_msg:
                logger.debug(
                    f"[EVENT #{event_counter}] No AIMessage extracted, skipping"
                )
                continue

            # DEBUG: Log AIMessage details
            msg_id = getattr(ai_msg, "id", "no-id")
            msg_content_preview = (
                str(ai_msg.content)[:100] if ai_msg.content else "None"
            )
            has_tool_calls = hasattr(ai_msg, "tool_calls") and ai_msg.tool_calls
            tool_call_count = len(ai_msg.tool_calls) if has_tool_calls else 0
            logger.debug(
                f"[EVENT #{event_counter}] AIMessage: id={msg_id}, "
                f"content_preview='{msg_content_preview}...', "
                f"tool_calls={tool_call_count}"
            )

            # Handle different message types
            if is_first_ai_message:
                path_first_msg_count += 1
                logger.debug(
                    f"[PATH] First AI message block executed "
                    f"(count={path_first_msg_count}, is_first={is_first_ai_message})"
                )

                content_str = extract_content_str(ai_msg.content)

                # Stream content character-by-character (3 chars at a time)
                if content_str and content_str.strip():
                    for chunk in chunk_text(content_str):
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
                    logger.debug(
                        f"[YIELD] First msg - tool_calls chunk: "
                        f"{len(tool_calls_dict)} tool(s)"
                    )
                    yield json.dumps(chunk)

                is_first_ai_message = False
                logger.debug("[STATE] is_first_ai_message set to False")

            elif hasattr(ai_msg, "tool_calls") and ai_msg.tool_calls:
                path_tool_calls_count += 1
                logger.debug(
                    f"[PATH] Elif tool_calls block executed "
                    f"(count={path_tool_calls_count}, "
                    f"tool_ids={[tc.get('id', 'no-id') for tc in ai_msg.tool_calls]})"
                )

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
                    logger.debug(
                        f"[YIELD] Elif block - tool_calls chunk: "
                        f"{len(tool_calls_dict)} tool(s)"
                    )
                    yield json.dumps(chunk)
                else:
                    # Legacy markdown mode: stream tool info
                    for tool_call in ai_msg.tool_calls:
                        tool_name = tool_call.get("name", "unknown")
                        tool_args = tool_call.get("args", {})
                        tool_section = format_tool_section(tool_name, tool_args)
                        for chunk in chunk_text(tool_section):
                            yield json.dumps(
                                create_openai_chunk(content=chunk, model=model)
                            )

            else:
                path_final_content_count += 1
                logger.debug(
                    f"[PATH] Final content block executed "
                    f"(count={path_final_content_count})"
                )

                # Stream content character-by-character
                content_str = extract_content_str(ai_msg.content)
                if content_str and content_str.strip():
                    for chunk in chunk_text(content_str):
                        yield json.dumps(
                            create_openai_chunk(content=chunk, model=model)
                        )

        # Flush any remaining content
        logger.debug(
            f"[SUMMARY] Events processed: {event_counter}, "
            f"First msg path: {path_first_msg_count}, "
            f"Tool calls path: {path_tool_calls_count}, "
            f"Final content path: {path_final_content_count}"
        )

        if use_native_format:
            # Send final chunk
            final_chunk = create_final_chunk(model)
            logger.debug("[YIELD] Final [DONE] chunk")
            yield json.dumps(final_chunk)
        else:
            # Legacy markdown final chunk
            final_chunk = create_openai_chunk(
                content="",
                model=model,
                finish_reason="stop",
            )
            logger.debug("[YIELD] Final [DONE] chunk (legacy)")
            yield json.dumps(final_chunk)

        logger.debug(f"[STREAM_END] request_id={request_id}")

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
