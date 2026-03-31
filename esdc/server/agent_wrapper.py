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


def format_thinking_section(content: str | list | dict | None) -> str:
    """Format thinking/reasoning section in markdown.

    Args:
        content: The thinking/reasoning text (can be str, list, dict, or None)

    Returns:
        Markdown formatted thinking section
    """
    # Convert content to string
    if content is None:
        return ""
    if isinstance(content, list):
        content = str(content[0]) if content else ""
    elif isinstance(content, dict):
        content = json.dumps(content, indent=2)
    else:
        content = str(content)

    if not content.strip():
        return ""

    return f"""### 🧠 Thinking Process

{content}

"""


def format_tool_section(tool_name: str, tool_args: dict) -> str:
    """Format tool call section in markdown with syntax highlighting.

    Args:
        tool_name: Name of the tool
        tool_args: Tool arguments dictionary

    Returns:
        Markdown formatted tool section
    """
    if tool_name == "execute_sql":
        sql = tool_args.get("query", "")
        return f"""### 🛠️ Tool: {tool_name}

```sql
{sql}
```

"""
    else:
        # For other tools, show as JSON
        args_str = json.dumps(tool_args, indent=2, ensure_ascii=False)
        return f"""### 🛠️ Tool: {tool_name}

```json
{args_str}
```

"""


def build_markdown_response(
    thinking: str, tool_calls: list[dict], final_response: str
) -> str:
    """Build complete markdown response with all sections.

    Args:
        thinking: The thinking/reasoning text
        tool_calls: List of tool call dictionaries
        final_response: The final response content

    Returns:
        Complete markdown formatted response
    """
    sections = []

    # Add thinking section
    if thinking:
        sections.append(format_thinking_section(thinking))

    # Add tool sections
    for tool_call in tool_calls:
        tool_name = tool_call.get("name", "unknown")
        tool_args = tool_call.get("args", {})
        sections.append(format_tool_section(tool_name, tool_args))

    # Add final response
    if final_response:
        sections.append(final_response)

    return "\n".join(sections)


async def generate_streaming_response(
    messages: list,
    model: str = "esdc-agent",
    temperature: float = 0.7,
) -> AsyncGenerator[str, None]:
    """Generate streaming chat completion response with markdown formatting."""
    request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

    # Accumulate content for markdown
    thinking_content = ""
    tool_calls = []
    final_content = ""
    is_first_ai_message = True

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

        # Stream the response
        async for event in agent.astream(
            {"messages": lc_messages},
            config=RunnableConfig(configurable={"thread_id": request_id}),
        ):
            ai_msg = extract_ai_message_from_event(event)
            if not ai_msg:
                continue

            # Capture thinking from first message
            if is_first_ai_message:
                content_str = extract_content_str(ai_msg.content)
                if content_str and content_str.strip():
                    thinking_content = content_str
                is_first_ai_message = False

            # Capture tool calls
            if hasattr(ai_msg, "tool_calls") and ai_msg.tool_calls:
                tool_calls.extend(ai_msg.tool_calls)

            # Capture final response (last message with content and no tool calls)
            content_str = extract_content_str(ai_msg.content)
            if content_str and not (
                hasattr(ai_msg, "tool_calls") and ai_msg.tool_calls
            ):
                final_content = content_str

        # Stream the markdown response in chunks
        # Send thinking section
        if thinking_content:
            chunk = create_openai_chunk(
                content=format_thinking_section(thinking_content),
                model=model,
            )
            yield json.dumps(chunk)

        # Send tool sections
        for tool_call in tool_calls:
            tool_name = tool_call.get("name", "unknown")
            tool_args = tool_call.get("args", {})
            chunk = create_openai_chunk(
                content=format_tool_section(tool_name, tool_args),
                model=model,
            )
            yield json.dumps(chunk)

        # Send final response
        if final_content:
            chunk = create_openai_chunk(
                content=final_content,
                model=model,
            )
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
    """Generate non-streaming chat completion response with markdown formatting."""
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

        # Accumulate content
        thinking_content = ""
        tool_calls = []
        final_content = ""
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

            # Capture thinking from first message
            if is_first_ai_message:
                content_str = extract_content_str(ai_msg.content)
                if content_str and content_str.strip():
                    thinking_content = content_str
                is_first_ai_message = False

            # Capture tool calls
            if hasattr(ai_msg, "tool_calls") and ai_msg.tool_calls:
                tool_calls.extend(ai_msg.tool_calls)

            # Capture final response
            content_str = extract_content_str(ai_msg.content)
            if content_str and not (
                hasattr(ai_msg, "tool_calls") and ai_msg.tool_calls
            ):
                final_content = content_str

        # Build markdown response
        markdown_response = build_markdown_response(
            thinking=thinking_content,
            tool_calls=tool_calls,
            final_response=final_content,
        )

        return {
            "content": markdown_response
            if markdown_response
            else "No response generated",
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
