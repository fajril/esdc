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


async def generate_streaming_response(
    messages: list,
    model: str = "esdc-agent",
    temperature: float = 0.7,
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

    Yields:
        OpenAI-compatible streaming chunks as SSE formatted strings
    """
    request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    buffer = StreamingBuffer()

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
                    # Check if this is thinking content (not just empty)
                    should_flush = buffer.add_thinking(content_str)
                    if should_flush:
                        content = buffer.flush()
                        if content:
                            chunk = create_openai_chunk(content=content, model=model)
                            yield json.dumps(chunk)
                is_first_ai_message = False

            elif hasattr(ai_msg, "tool_calls") and ai_msg.tool_calls:
                # Tool call detected - flush current buffer first
                if buffer.has_content():
                    content = buffer.flush()
                    if content:
                        chunk = create_openai_chunk(content=content, model=model)
                        yield json.dumps(chunk)

                # Stream each tool call immediately
                for tool_call in ai_msg.tool_calls:
                    tool_name = tool_call.get("name", "unknown")
                    tool_args = tool_call.get("args", {})
                    # Add to buffer and flush immediately
                    buffer.add_tool_call(tool_name, tool_args)
                    content = buffer.flush()
                    if content:
                        chunk = create_openai_chunk(content=content, model=model)
                        yield json.dumps(chunk)

            else:
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

        # Send final chunk
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
                    buffer.add_thinking(content_str)
                is_first_ai_message = False

            elif hasattr(ai_msg, "tool_calls") and ai_msg.tool_calls:
                # Flush current content first
                if buffer.has_content():
                    buffer.flush()

                # Add tool calls
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

        # Build final markdown response
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
