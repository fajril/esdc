import json
import logging
from typing import Any, AsyncGenerator, cast

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
)
from langchain_core.runnables import Runnable, RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import StateGraph, MessagesState, START, END

from esdc.chat.prompts import get_system_prompt
from esdc.chat.tools import (
    execute_sql,
    get_schema,
    get_recommended_table,
    list_tables,
    resolve_uncertainty_level,
)

# Logger is configured by app.py (runs first)
logger = logging.getLogger("esdc.chat.agent")

TOKEN_CHARS_PER_TOKEN = 4


async def generate_conversation_title(
    llm: BaseChatModel,
    user_query: str,
) -> str:
    """Generate a short title/summary for the conversation based on first query.

    Args:
        llm: Language model to use for generation
        user_query: First user query

    Returns:
        Short title (max 50 chars) summarizing the conversation
    """
    prompt = """Generate a very short title (max 50 characters) summarizing this user query.
The title should be concise and descriptive. Respond with ONLY the title, no quotes or explanation.

Examples:
- "how much oil reserves in Rokan field" -> "Rokan Field Oil Reserves"
- "list all working areas with gas production" -> "Working Areas Gas Production"
- "compare reserves between 2020 and 2023" -> "Reserve Comparison 2020-2023"

User query: {query}

Title:"""

    try:
        messages = [
            SystemMessage(
                content="You are a helpful assistant that generates concise conversation titles."
            ),
            HumanMessage(content=prompt.format(query=user_query)),
        ]

        response = await llm.ainvoke(messages)
        content = response.content
        if isinstance(content, list):
            title = str(content[0]) if content else ""
        else:
            title = str(content)
        title = title.strip().strip("\"'")

        if len(title) > 50:
            title = title[:47] + "..."

        return title
    except Exception:
        query_clean = user_query.strip()
        if len(query_clean) > 50:
            return query_clean[:47] + "..."
        return query_clean


def create_agent(
    llm: BaseChatModel,
    tools: list | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
) -> Runnable:
    """Create a LangGraph agent with tools.

    Args:
        llm: A LangChain chat model
        tools: Optional list of tools. Defaults to [execute_sql, get_schema, list_tables]
        checkpointer: Optional checkpointer for memory persistence

    Returns:
        A compiled StateGraph agent
    """
    if tools is None:
        tools = [
            execute_sql,
            get_schema,
            list_tables,
            get_recommended_table,
            resolve_uncertainty_level,
        ]

    tools_by_name = {tool.name: tool for tool in tools}
    llm_with_tools = llm.bind_tools(tools)

    def agent_node(state: MessagesState) -> dict[str, list[AnyMessage]]:
        """Agent node that calls the LLM with tools."""
        system_prompt = get_system_prompt()
        messages_with_system = [SystemMessage(content=system_prompt)] + state[
            "messages"
        ]

        response = llm_with_tools.invoke(messages_with_system)
        return {"messages": [cast(AnyMessage, response)]}

    def should_continue(state: MessagesState) -> str:
        """Determine if we should continue to tools or end."""
        messages = state["messages"]
        last_message = messages[-1]

        ai_message = cast(AIMessage, last_message)
        if hasattr(ai_message, "tool_calls") and ai_message.tool_calls:
            return "tools"

        return END

    async def tool_node(state: MessagesState) -> dict[str, list[AnyMessage]]:
        """Tool execution node."""
        result = []
        last_message = state["messages"][-1]

        ai_message = cast(AIMessage, last_message)
        if not hasattr(ai_message, "tool_calls") or not ai_message.tool_calls:
            return {"messages": []}

        logger.info(f"🔧 TOOL_NODE: Processing {len(ai_message.tool_calls)} tool calls")

        for tool_call in ai_message.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call.get("args", {})
            tool_id = tool_call.get("id", "unknown")

            if tool_name in tools_by_name:
                tool = tools_by_name[tool_name]
                try:
                    if isinstance(tool_args, str):
                        tool_args = json.loads(tool_args)

                    logger.info(f"🔧 TOOL_NODE: Invoking {tool_name} (id={tool_id})")
                    observation = await tool.ainvoke(tool_args)
                    logger.info(
                        f"🔧 TOOL_NODE: {tool_name} returned {len(str(observation))} chars"
                    )
                except Exception as e:
                    logger.error(f"🔧 TOOL_NODE: {tool_name} failed: {e}")
                    observation = f"Error: {str(e)}"

                result.append(
                    {
                        "tool_call_id": tool_id,
                        "role": "tool",
                        "name": tool_name,
                        "content": observation,
                    }
                )

        logger.info(f"🔧 TOOL_NODE: Returning {len(result)} tool results")
        return {"messages": result}

    graph = (
        StateGraph(MessagesState)
        .add_node("agent", agent_node)
        .add_node("tools", tool_node)
        .add_edge(START, "agent")
        .add_conditional_edges(
            "agent",
            should_continue,
            {
                "tools": "tools",
                END: END,
            },
        )
        .add_edge("tools", "agent")
    )

    return graph.compile(checkpointer=checkpointer)


async def run_agent_stream(
    agent: Runnable,
    user_input: str,
    thread_id: str,
    checkpointer: BaseCheckpointSaver | None = None,
) -> AsyncGenerator[dict[str, Any], None]:
    """Run the agent with streaming output.

    Args:
        agent: Compiled LangGraph agent
        user_input: User message
        thread_id: Conversation thread ID
        checkpointer: Optional checkpointer for memory (agent should already be compiled)

    Yields:
        Dict with 'type' (message/tool/error) and 'content' or 'token_usage'
    """
    config: RunnableConfig = {  # type: ignore[assignment]
        "recursion_limit": 50,
        "configurable": {
            "thread_id": thread_id,
            "checkpoint_ns": "esdc_chat",
        },
    }

    messages = [HumanMessage(content=user_input)]

    if checkpointer:
        agent = agent.compile(checkpointer=checkpointer)  # type: ignore[attr-defined]

    stored_tool_calls: list[dict[str, Any]] = []

    logger.info("=" * 60)
    logger.info("🔔 AGENT_STREAM_STARTED: thread_id=%s", thread_id)
    logger.info("=" * 60)

    event_count = 0
    token_event_count = 0

    async for event in agent.astream_events(
        {"messages": messages},
        config=config,
        version="v2",
    ):
        event_count += 1
        event_type = event.get("event")
        data = event.get("data", {})

        # Log every event (but not too spammy)
        if event_count <= 5 or event_count % 20 == 0:
            logger.info("🔔 AGENT_EVENT #%d: type=%s", event_count, event_type)

        # Handle token streaming (character-by-character)
        # ChatOllama may emit either on_chat_model_stream or on_llm_stream
        if event_type in ("on_chat_model_stream", "on_llm_stream"):
            token_event_count += 1
            chunk = data.get("chunk")
            if chunk:
                # Handle different chunk formats
                if hasattr(chunk, "content"):
                    content = chunk.content
                elif isinstance(chunk, str):
                    content = chunk
                else:
                    content = str(chunk) if chunk else ""

                if content:
                    # Log token details (every 50 tokens)
                    if token_event_count % 50 == 0:
                        logger.info(
                            "✅ TOKEN_EVENT #%d: len=%d, preview='%s'",
                            token_event_count,
                            len(content),
                            content[:40],
                        )

                    yield {
                        "type": "token",
                        "content": content,
                    }

        # Handle completion (tool calls, final message)
        elif event_type == "on_chat_model_end":
            logger.info("🔚 CHAT_MODEL_END: completing LLM call")
            output = data.get("output")
            if output:
                # Token usage
                tokens_used = _extract_token_usage(output, user_input)
                if tokens_used > 0:
                    logger.info("📊 TOKEN_USAGE: %d tokens", tokens_used)
                    yield {"type": "token_usage", "tokens": tokens_used}

                # Tool calls
                if hasattr(output, "tool_calls") and output.tool_calls:
                    logger.info(
                        "🛠️ TOOL_CALLS_DETECTED: count=%d", len(output.tool_calls)
                    )
                    for tc in output.tool_calls:
                        # Store for later SQL extraction
                        args = tc.get("args", {})
                        if isinstance(args, str):
                            try:
                                args = json.loads(args)
                            except Exception:
                                args = {}
                        query = args.get("query", "")

                        stored_tool_calls.append(
                            {
                                "name": tc["name"],
                                "args": args,
                                "query": query,
                                "id": tc.get("id", ""),
                            }
                        )

                        logger.info(
                            f"AGENT_STORING: tool_call={tc['name']}, query={query[:50] if query else 'N/A'}..."
                        )

                        yield {
                            "type": "tool_call",
                            "tool": tc["name"],
                            "args": args,
                        }

                # Content (apply existing SQL filtering)
                if hasattr(output, "content") and output.content:
                    content = output.content
                    # Apply SQL filtering (lines 249-267)
                    if "```sql" in content.lower():
                        import re

                        sql_pattern = r"```sql\s*?\n?(.*?)\n?```"
                        content = re.sub(
                            sql_pattern, "", content, flags=re.DOTALL | re.IGNORECASE
                        )
                        table_pattern = r"\|.*\|.*\n\|[-:| ]+\|"
                        content = re.sub(table_pattern, "", content)
                        content = re.sub(r"\n{3,}", "\n\n", content)
                        content = content.strip()
                        logger.info(
                            "AGENT_FILTERED: Removed SQL code block from message"
                        )

                    if content:
                        yield {
                            "type": "message",
                            "content": content,
                            "additional_kwargs": output.additional_kwargs
                            if hasattr(output, "additional_kwargs")
                            else {},
                        }

        # Handle tool results
        elif event_type == "on_tool_end":
            tool_result = data.get("output")
            tool_name = event.get("name", "unknown")

            logger.info(
                "🔧 AGENT_TOOL_END: tool=%s, result_len=%d",
                tool_name,
                len(str(tool_result)),
            )

            # Extract SQL from stored_tool_calls (pop from front - FIFO order)
            sql = ""
            if stored_tool_calls:
                stored_tc = stored_tool_calls.pop(0)
                if stored_tc.get("name") == "execute_sql":
                    args = stored_tc.get("args", {})
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except Exception:
                            args = {}
                    sql = args.get("query", "")
                    logger.info(
                        "📤 YIELDING tool_result: tool=%s, sql_len=%d, result_len=%d",
                        tool_name,
                        len(sql),
                        len(str(tool_result)),
                    )
            else:
                logger.info(
                    "YIELDING tool_result NO SQL - tool=%s, result_length=%d (no stored calls)",
                    tool_name,
                    len(str(tool_result)),
                )

            yield {
                "type": "tool_result",
                "tool": tool_name,
                "result": str(tool_result),
                "sql": sql,
            }


def _extract_token_usage(message: AIMessage, user_input: str) -> int:
    """Extract token usage from an AIMessage response.

    Tries multiple sources:
    1. message.usage_metadata (LangChain format)
    2. message.response_metadata with usage (OpenAI format)
    3. Estimate from text length

    Args:
        message: AIMessage from LLM
        user_input: Original user input for estimation fallback

    Returns:
        Estimated or actual token count
    """
    # Try LangChain usage_metadata format
    if hasattr(message, "usage_metadata") and message.usage_metadata:
        usage = message.usage_metadata
        if isinstance(usage, dict):
            # LangChain format: {'input_tokens': X, 'output_tokens': Y, 'total_tokens': Z}
            if "total_tokens" in usage:
                return int(usage["total_tokens"])
            elif "output_tokens" in usage and "input_tokens" in usage:
                return int(usage.get("input_tokens", 0)) + int(
                    usage.get("output_tokens", 0)
                )

    # Try OpenAI response_metadata format
    if hasattr(message, "response_metadata") and message.response_metadata:
        metadata = message.response_metadata
        if isinstance(metadata, dict):
            usage = metadata.get("usage") or metadata.get("Usage")
            if usage:
                if hasattr(usage, "total_tokens"):
                    return int(usage.total_tokens)
                if isinstance(usage, dict):
                    if "total_tokens" in usage:
                        return int(usage["total_tokens"])
                    elif "output_tokens" in usage and "prompt_tokens" in usage:
                        return int(usage.get("prompt_tokens", 0)) + int(
                            usage.get("output_tokens", 0)
                        )

    # Fallback: estimate from text content
    if hasattr(message, "content") and message.content:
        content = message.content
        if isinstance(content, list):
            # Handle list content (e.g., [{"type": "text", "text": "..."}])
            text = " ".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in content
            )
        else:
            text = str(content)
        return _estimate_tokens(text)

    return 0


def _estimate_tokens(text: str) -> int:
    """Estimate token count from text.

    Uses rough approximation: ~4 characters per token.

    Args:
        text: Text to estimate tokens for

    Returns:
        Estimated token count
    """
    if not text:
        return 0
    return len(text) // TOKEN_CHARS_PER_TOKEN
