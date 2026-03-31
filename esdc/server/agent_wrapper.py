# Standard library
import logging
import time
from collections.abc import AsyncGenerator
from typing import Any

# Third-party
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig

# Local
from esdc.chat.agent import create_agent
from esdc.chat.memory import get_thread_config as create_thread_config
from esdc.configs import Config
from esdc.providers import get_provider

logger = logging.getLogger("esdc.server.agent")


async def create_llm_from_config():
    """Create LLM instance from configuration."""
    provider_config = Config.get_provider_config()

    if not provider_config:
        raise ValueError(
            "No provider configured. Please run 'esdc provider add' first."
        )

    provider_name = provider_config.get("provider", "ollama")
    model = provider_config.get("model")
    base_url = provider_config.get("base_url")
    api_key = provider_config.get("api_key")

    provider_class = get_provider(provider_name)
    if not provider_class:
        raise ValueError(f"Unknown provider: {provider_name}")

    return provider_class.create_llm(
        model=model,
        base_url=base_url,
        api_key=api_key,
        temperature=0.7,
    )


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


async def generate_streaming_response(
    messages: list,
    model: str = "esdc-agent",
    temperature: float = 0.7,
) -> AsyncGenerator[dict, None]:
    """Generate streaming chat completion response.

    Args:
        messages: List of conversation messages
        model: Model ID
        temperature: Sampling temperature

    Yields:
        OpenAI-compatible streaming chunks
    """
    try:
        # Create LLM and agent
        llm = await create_llm_from_config()
        agent = create_agent(llm, checkpointer=None)

        # Convert messages
        lc_messages = convert_messages_to_langchain(messages)

        # Create thread config for this request
        thread_config = create_thread_config(f"esdc-{int(time.time())}")

        # Stream the response
        async for event in agent.astream(
            {"messages": lc_messages},
            config=thread_config,  # type: ignore[arg-type]
        ):
            # Handle different event types
            if "messages" in event:
                messages = event["messages"]
                if messages:
                    last_message = messages[-1]
                    if isinstance(last_message, AIMessage):
                        content = last_message.content
                        if isinstance(content, list):
                            content = str(content[0]) if content else ""

                        if content:
                            yield {
                                "content": content,
                                "role": "assistant",
                                "finish_reason": None,
                            }

            # Handle tool calls
            if "tool_calls" in event:
                tool_calls = event["tool_calls"]
                if tool_calls:
                    # Send tool calls as part of response
                    for tool_call in tool_calls:
                        yield {
                            "tool_calls": [tool_call],
                            "role": "assistant",
                            "finish_reason": None,
                        }

        # Send final message
        yield {
            "content": "",
            "role": "assistant",
            "finish_reason": "stop",
        }

    except Exception as e:
        logger.error(f"Error in streaming response: {e}")
        yield {
            "content": f"Error: {str(e)}",
            "role": "assistant",
            "finish_reason": "stop",
        }


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
        llm = await create_llm_from_config()
        agent = create_agent(llm, checkpointer=None)

        # Convert messages
        lc_messages = convert_messages_to_langchain(messages)

        # Create thread config for this request
        thread_config = create_thread_config(f"esdc-{int(time.time())}")

        # Run the agent
        result = await agent.ainvoke(
            {"messages": lc_messages},
            config=thread_config,  # type: ignore[arg-type]
        )

        # Extract the final response
        final_messages = result.get("messages", [])
        if final_messages:
            last_message = final_messages[-1]
            if isinstance(last_message, AIMessage):
                content = last_message.content
                if isinstance(content, list):
                    content = str(content[0]) if content else ""

                return {
                    "content": content,
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
