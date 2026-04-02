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
from esdc.server.stream_buffer import StreamingBuffer
from esdc.server.thinking_parser import (
    extract_thinking_content,
    has_thinking_tags,
)
from esdc.server.thinking_state import ThinkingState
from esdc.server.tool_formatter import (
    create_tool_call_chunk,
    create_final_chunk,
)

logger = logging.getLogger("esdc.server.agent")


def convert_messages_to_langchain(messages: list) -> list:
    """Convert OpenAI-compatible messages to LangChain messages."""
    lc_messages = []

    for msg in messages:
        role = msg.role
        content = msg.content or ""

        if role == "system":
            lc_messages.append(SystemMessage(content=content))
        elif role == "user":
            lc_messages.append(HumanMessage(content=content))
        elif role == "assistant":
            lc_messages.append(AIMessage(content=content))
        elif role == "tool":
            lc_messages.append(
                ToolMessage(content=content, tool_call_id=msg.tool_call_id or "")
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
    for key, value in event.items():
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


def extract_thinking_for_interleaved(message: AIMessage) -> str | None:
    """Extract thinking content from AIMessage when interleaved with tool calls.

    Only extracts thinking when message has tool_calls (interleaved scenario).

    Args:
        message: AIMessage that may contain reasoning_content

    Returns:
        Thinking content if present and message has tool calls, None otherwise
    """
    # Only extract for interleaved scenario (tool calls present)
    if not hasattr(message, "tool_calls") or not message.tool_calls:
        return None

    # Try reasoning_content from additional_kwargs
    if hasattr(message, "additional_kwargs") and message.additional_kwargs:
        reasoning = message.additional_kwargs.get("reasoning_content")
        if reasoning:
            return str(reasoning).strip()

    # Try extracting from content with thinking tags
    content_str = extract_content_str(message.content)
    if has_thinking_tags(content_str):
        thinking, _ = extract_thinking_content(content_str)
        if thinking:
            return thinking.strip()

    return None


async def generate_streaming_response(
    messages: list,
    model: str = "esdc-agent",
    temperature: float = 0.7,
    use_native_format: bool = True,
) -> AsyncGenerator[str, None]:
    """Generate streaming chat completion response with markdown formatting.

    Uses StreamingBuffer for accumulating and flushing content at checkpoints:
    - Tool calls are flushed immediately with previous content
    - Content is flushed when buffer exceeds 500 chars or 500ms timeout
    - Final flush at end of stream

    Args:
        messages: List of conversation messages
        model: Model ID
        temperature: Sampling temperature
        use_native_format: Whether to use native tool_calls format or markdown

    Yields:
        OpenAI-compatible streaming chunks as SSE formatted strings
    """
    request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    buffer = StreamingBuffer()
    thinking_state = ThinkingState()

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

        # Track first message for thinking
        is_first_ai_message = True

        # Stream the response
        async for event in agent.astream(
            {"messages": lc_messages},
            config=RunnableConfig(configurable={"thread_id": request_id}),
        ):
            ai_msg = extract_ai_message_from_event(event)
            if not ai_msg:
                continue

            # Handle different message types
            if is_first_ai_message:
                # First message might contain thinking/reasoning
                content_str = extract_content_str(ai_msg.content)
                if content_str and content_str.strip():
                    # Check if content has thinking tags
                    if has_thinking_tags(content_str):
                        thinking, final = extract_thinking_content(content_str)
                        if thinking:
                            # Add thinking to buffer (will be formatted with <think> tags)
                            should_flush = buffer.add_thinking(thinking)
                            if should_flush:
                                content = buffer.flush()
                                if content:
                                    chunk = create_openai_chunk(
                                        content=content, model=model
                                    )
                                    yield json.dumps(chunk)
                        if final:
                            # Add final response
                            should_flush = buffer.add_content(final)
                            if should_flush:
                                content = buffer.flush()
                                if content:
                                    chunk = create_openai_chunk(
                                        content=content, model=model
                                    )
                                    yield json.dumps(chunk)
                    else:
                        # No thinking tags, treat as regular content
                        should_flush = buffer.add_thinking(content_str)
                        if should_flush:
                            content = buffer.flush()
                            if content:
                                chunk = create_openai_chunk(
                                    content=content, model=model
                                )
                                yield json.dumps(chunk)
                is_first_ai_message = False

            elif hasattr(ai_msg, "tool_calls") and ai_msg.tool_calls:
                # Extract and preserve thinking before tool execution
                thinking = extract_thinking_for_interleaved(ai_msg)
                if thinking:
                    thinking_state.preserve_thinking(thinking)

                if use_native_format:
                    # Emit native tool_calls chunk
                    chunk = create_tool_call_chunk(ai_msg.tool_calls, model)
                    yield json.dumps(chunk)
                else:
                    # Fallback to markdown format
                    if buffer.has_content():
                        content = buffer.flush()
                        if content:
                            chunk = create_openai_chunk(content=content, model=model)
                            yield json.dumps(chunk)

                    for tool_call in ai_msg.tool_calls:
                        tool_name = tool_call.get("name", "unknown")
                        tool_args = tool_call.get("args", {})
                        buffer.add_tool_call(tool_name, tool_args)
                        content = buffer.flush()
                        if content:
                            chunk = create_openai_chunk(content=content, model=model)
                            yield json.dumps(chunk)

            else:
                # Check if we have preserved thinking to inject before final content
                if thinking_state.has_thinking():
                    preserved = thinking_state.get_thinking()
                    if preserved:
                        buffer.add_preserved_thinking(preserved)

                # Final content - accumulate in buffer
                content_str = extract_content_str(ai_msg.content)
                if content_str:
                    should_flush = buffer.add_content(content_str)
                    if should_flush:
                        content = buffer.flush()
                        if content:
                            chunk = create_openai_chunk(content=content, model=model)
                            yield json.dumps(chunk)

        # Flush any remaining content
        if buffer.has_content():
            final_content = buffer.flush_final()
            if final_content:
                chunk = create_openai_chunk(content=final_content, model=model)
                yield json.dumps(chunk)

        if use_native_format:
            # Send final chunk
            final_chunk = create_final_chunk(model)
            yield json.dumps(final_chunk)
        else:
            # Legacy markdown final chunk
            final_chunk = create_openai_chunk(
                content="",
                model=model,
                finish_reason="stop",
            )
            yield json.dumps(final_chunk)

    except Exception as e:
        logger.error(f"Error in streaming response: {e}", exc_info=True)
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
        buffer = StreamingBuffer()
        thinking_state = ThinkingState()
        stored_tool_calls = []  # Store tool calls for native format
        is_first_ai_message = True

        # Run the agent
        async for event in agent.astream(
            {"messages": lc_messages},
            config=RunnableConfig(
                configurable={"thread_id": f"esdc-{int(time.time())}"}
            ),
        ):
            ai_msg = extract_ai_message_from_event(event)
            if not ai_msg:
                continue

            # Handle different message types
            if is_first_ai_message:
                content_str = extract_content_str(ai_msg.content)
                if content_str and content_str.strip():
                    # Check if content has thinking tags
                    if has_thinking_tags(content_str):
                        thinking, final = extract_thinking_content(content_str)
                        if thinking:
                            buffer.add_thinking(thinking)
                        if final:
                            buffer.add_content(final)
                    else:
                        # No thinking tags, treat as regular content
                        buffer.add_thinking(content_str)
                is_first_ai_message = False

            elif hasattr(ai_msg, "tool_calls") and ai_msg.tool_calls:
                # Extract and preserve thinking
                thinking = extract_thinking_for_interleaved(ai_msg)
                if thinking:
                    thinking_state.preserve_thinking(thinking)

                if not use_native_format:
                    # Only flush buffer for markdown mode
                    if buffer.has_content():
                        buffer.flush()

                # Store tool calls for final response
                stored_tool_calls = ai_msg.tool_calls

                if not use_native_format:
                    # Add to buffer for markdown mode
                    for tool_call in ai_msg.tool_calls:
                        tool_name = tool_call.get("name", "unknown")
                        tool_args = tool_call.get("args", {})
                        buffer.add_tool_call(tool_name, tool_args)
                        buffer.flush()  # Flush each tool immediately

            else:
                # Check if we have preserved thinking to inject before final content
                if thinking_state.has_thinking():
                    preserved = thinking_state.get_thinking()
                    if preserved:
                        buffer.add_preserved_thinking(preserved)

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
