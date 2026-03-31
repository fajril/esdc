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


async def generate_streaming_response(
    messages: list,
    model: str = "esdc-agent",
    temperature: float = 0.7,
) -> AsyncGenerator[str, None]:
    """Generate streaming chat completion response.

    Args:
        messages: List of conversation messages
        model: Model ID
        temperature: Sampling temperature

    Yields:
        OpenAI-compatible streaming chunks as SSE formatted strings
    """
    request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())
    full_content = ""

    try:
        # Create LLM and agent
        provider_config = Config.get_provider_config()
        if not provider_config:
            yield json.dumps(
                create_openai_chunk(
                    content="Error: No provider configured. Please run 'esdc provider add' first.",
                    model=model,
                    finish_reason="stop",
                )
            )
            return

        provider_name = provider_config.get("provider", "ollama")
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

        # Stream the response
        last_content = ""
        async for event in agent.astream(
            {"messages": lc_messages},
            config=RunnableConfig(configurable={"thread_id": request_id}),
        ):
            # Handle different event types from LangGraph
            if "messages" in event:
                messages_list = event["messages"]
                if messages_list:
                    last_message = messages_list[-1]
                    if isinstance(last_message, AIMessage):
                        content = last_message.content
                        if isinstance(content, list):
                            content = str(content[0]) if content else ""

                        # Only send if content changed
                        if content and content != last_content:
                            last_content = content
                            full_content = content

                            chunk = {
                                "id": request_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": model,
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {"content": content},
                                        "finish_reason": None,
                                    }
                                ],
                            }
                            yield json.dumps(chunk)

        # Send final chunk with stop reason
        final_chunk = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": ""},
                    "finish_reason": "stop",
                }
            ],
        }
        yield json.dumps(final_chunk)

    except Exception as e:
        logger.error(f"Error in streaming response: {e}")
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
    """Generate non-streaming chat completion response.

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
                "content": "Error: No provider configured. Please run 'esdc provider add' first.",
                "role": "assistant",
                "finish_reason": "stop",
            }

        provider_name = provider_config.get("provider", "ollama")
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

        # Run the agent
        final_content = ""
        async for event in agent.astream(
            {"messages": lc_messages},
            config=RunnableConfig(
                configurable={"thread_id": f"esdc-{int(time.time())}"}
            ),
        ):
            if "messages" in event:
                messages_list = event["messages"]
                if messages_list:
                    last_message = messages_list[-1]
                    if isinstance(last_message, AIMessage):
                        content = last_message.content
                        if isinstance(content, list):
                            content = str(content[0]) if content else ""
                        if content:
                            final_content = content

        if final_content:
            return {
                "content": final_content,
                "role": "assistant",
                "finish_reason": "stop",
            }

        return {
            "content": "No response generated",
            "role": "assistant",
            "finish_reason": "stop",
        }

    except Exception as e:
        logger.error(f"Error in response generation: {e}")
        return {
            "content": f"Error: {str(e)}",
            "role": "assistant",
            "finish_reason": "stop",
        }
