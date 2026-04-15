# Standard library
import json
import logging
import time
from collections.abc import AsyncGenerator
from typing import Any, cast

# Third-party
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langchain_core.runnables import Runnable, RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, MessagesState, StateGraph

# Local
from esdc.chat.context_manager import manage_context_node
from esdc.chat.prompts import get_system_prompt
from esdc.chat.query_classifier import (
    QueryClassifier,
    format_classification_for_prompt,
)
from esdc.chat.tools import (
    execute_cypher,
    execute_sql,
    get_recommended_table,
    get_resources_columns,
    get_schema,
    get_timeseries_columns,
    knowledge_traversal,
    list_tables,
    resolve_spatial,
    resolve_uncertainty_level,
    search_problem_cluster,
    semantic_search,
)

# Logger is configured by app.py (runs first)
logger = logging.getLogger("esdc.chat.agent")

TOKEN_CHARS_PER_TOKEN = 4
MAX_TOOL_RESULT_CHARS = 10000


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

        # Log inference start with prompt details
        total_chars = sum(len(str(m.content)) for m in messages)
        logger.debug(
            "[INFERENCE] title_generation_start | messages=%d | total_chars=%d",
            len(messages),
            total_chars,
        )
        inference_start = time.perf_counter()

        response = await llm.ainvoke(messages)

        # Log inference completion
        inference_elapsed_ms = (time.perf_counter() - inference_start) * 1000
        response_content = response.content if hasattr(response, "content") else str(response)
        content_len = len(response_content) if response_content else 0
        logger.debug(
            "[INFERENCE] title_generation_complete | elapsed=%.2fms | response_len=%d",
            inference_elapsed_ms,
            content_len,
        )
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
    context_length: int = 6000,
) -> Runnable:
    """Create a LangGraph agent with tools.

    Args:
        llm: A LangChain chat model
        tools: Optional list of tools. Defaults to all registered tools
        checkpointer: Optional checkpointer for memory persistence
        context_length: Maximum context length in tokens for context management. Default: 6000

    Returns:
        A compiled StateGraph agent
    """
    if tools is None:
        tools = [
            knowledge_traversal,
            resolve_spatial,
            semantic_search,
            execute_cypher,
            execute_sql,
            get_schema,
            list_tables,
            get_recommended_table,
            resolve_uncertainty_level,
            search_problem_cluster,
            get_timeseries_columns,
            get_resources_columns,
        ]

    tools_by_name = {tool.name: tool for tool in tools}
    llm_with_tools = llm.bind_tools(tools)

    def agent_node(state: MessagesState) -> dict[str, list[AnyMessage]]:
        """Agent node that calls the LLM with tools."""
        system_prompt = get_system_prompt()
        messages_with_system = [SystemMessage(content=system_prompt)] + state[
            "messages"
        ]

        # Log inference start with detailed prompt metrics
        total_chars = sum(len(str(m.content)) for m in messages_with_system)
        system_prompt_len = len(system_prompt)
        user_messages = len([m for m in messages_with_system if isinstance(m, HumanMessage)])
        logger.debug(
            "[INFERENCE] llm_invoke_start | messages=%d | user_messages=%d | "
            "system_prompt_len=%d | total_chars=%d",
            len(messages_with_system),
            user_messages,
            system_prompt_len,
            total_chars,
        )
        inference_start = time.perf_counter()

        response = llm_with_tools.invoke(messages_with_system)

        # Log inference completion with response metrics
        inference_elapsed_ms = (time.perf_counter() - inference_start) * 1000
        response_len = len(str(response.content)) if hasattr(response, "content") else 0
        has_tool_calls = hasattr(response, "tool_calls") and bool(response.tool_calls)
        tool_call_count = len(response.tool_calls) if has_tool_calls else 0
        logger.debug(
            "[INFERENCE] llm_invoke_complete | elapsed=%.2fms | response_len=%d | "
            "has_tool_calls=%s | tool_calls=%d",
            inference_elapsed_ms,
            response_len,
            has_tool_calls,
            tool_call_count,
        )

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
                    observation_str = str(observation)
                    logger.info(
                        f"🔧 TOOL_NODE: {tool_name} returned {len(observation_str)} chars"
                    )

                    if len(observation_str) > MAX_TOOL_RESULT_CHARS:
                        observation = (
                            observation_str[:MAX_TOOL_RESULT_CHARS]
                            + "\n\n[Result truncated to first 10000 characters for context efficiency]"
                        )
                        logger.info(
                            f"🔧 TOOL_NODE: {tool_name} result truncated from {len(observation_str)} to {MAX_TOOL_RESULT_CHARS} chars"
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

    def manage_context_with_length(state: MessagesState) -> dict[str, list[AnyMessage]]:
        """Wrapper for manage_context_node with bound context_length."""
        return manage_context_node(state, context_length=context_length)

    def entity_resolution_node(state: MessagesState) -> dict[str, list[AnyMessage]]:
        """Pre-resolve entities from the last user message via knowledge graph.

        Injects resolved entities as a system message so the LLM can use
        them directly without needing to call knowledge_traversal as a tool.
        """
        messages = state["messages"]
        if not messages:
            return {"messages": []}

        last_human = None
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage) and msg.content:
                last_human = msg.content
                break

        if not last_human:
            return {"messages": []}

        try:
            import json as _json

            from esdc.chat.tools import knowledge_traversal

            result_str = knowledge_traversal.invoke(
                {"query": last_human, "return_multiple": False}
            )
            result = _json.loads(result_str)
            logger.info(
                "entity_resolution: status=%s entities=%d",
                result.get("status"),
                len(result.get("entities", [])),
            )

            if result.get("status") == "failed":
                return {"messages": []}

            entity_names = []
            for e in result.get("entities", []):
                if e.get("confidence", 0) >= 0.7:
                    entity_names.append(
                        f"- {e['type']}: {e['name']} (id={e.get('id', '?')})"
                    )

            if not entity_names:
                return {"messages": []}

            context_parts = [
                "[Knowledge Graph - Auto-resolved entities]",
                "The following entities were automatically resolved from your query.",
                "Use these to write precise SQL WHERE clauses "
                "instead of ILIKE patterns.",
            ]
            context_parts.extend(entity_names)

            if result.get("where_conditions"):
                context_parts.append("Suggested WHERE conditions:")
                for wc in result["where_conditions"]:
                    context_parts.append(f"  {wc}")

            if result.get("suggested_table"):
                context_parts.append(f"Suggested table: {result['suggested_table']}")

            if result.get("required_columns"):
                context_parts.append(
                    f"Required columns: {', '.join(result['required_columns'])}"
                )

            context_msg = SystemMessage(content="\n".join(context_parts))
            logger.info(
                "entity_resolution: injecting %d entities for query='%s'",
                len(entity_names),
                last_human[:50],
            )
            return {"messages": [cast(AnyMessage, context_msg)]}

        except Exception as e:
            logger.warning("entity_resolution: failed - %s", e)
            return {"messages": []}

    def query_classification_node(state: MessagesState) -> dict[str, list[AnyMessage]]:
        """Classify query and inject strategy into system prompt."""
        messages = state["messages"]
        if not messages:
            return {"messages": []}

        # Find last human message
        last_human = None
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage) and msg.content:
                last_human = msg.content
                break

        if not last_human:
            return {"messages": []}

        try:
            classifier = QueryClassifier()
            classification = classifier.classify(last_human)

            strategy_text = format_classification_for_prompt(classification)

            strategy_msg = SystemMessage(content=strategy_text)
            logger.info(
                "query_classification: type=%s confidence=%.2f query='%s...'",
                classification.query_type.name,
                classification.confidence,
                last_human[:50],
            )
            return {"messages": [cast(AnyMessage, strategy_msg)]}

        except Exception as e:
            logger.warning("query_classification: failed - %s", e)
            return {"messages": []}

    # Build graph with query classification
    graph = (
        StateGraph(MessagesState)
        .add_node("manage_context", manage_context_with_length)
        .add_node("query_classification", query_classification_node)
        .add_node("entity_resolution", entity_resolution_node)
        .add_node("agent", agent_node)
        .add_node("tools", tool_node)
        .add_edge(START, "manage_context")
        .add_edge("manage_context", "query_classification")
        .add_edge("query_classification", "entity_resolution")
        .add_edge("entity_resolution", "agent")
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
    conversation_messages: list[
        AnyMessage
    ] = []  # Track messages for real-time token count

    stream_start = time.perf_counter()
    first_token_time: float | None = None
    first_llm_time: float | None = None
    tool_call_time: float | None = None

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
            elapsed_ms = (time.perf_counter() - stream_start) * 1000
            logger.debug(
                "🔔 AGENT_EVENT #%d: type=%s | elapsed=%.0fms",
                event_count,
                event_type,
                elapsed_ms,
            )

        # Handle token streaming (character-by-character)
        # ChatOllama may emit either on_chat_model_stream or on_llm_stream
        if event_type in ("on_chat_model_stream", "on_llm_stream"):
            if first_token_time is None:
                first_token_time = time.perf_counter()
                ttft_ms = (first_token_time - stream_start) * 1000
                logger.debug("[TIMING] first_token | ttft=%.2fms", ttft_ms)
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

        # Handle context management node completion
        elif event_type == "on_chain_end":
            node_name = event.get("name", "")
            if node_name == "manage_context":
                output_data = data.get("output", {})
                messages = output_data.get("messages", [])
                context_metadata = output_data.get("context_metadata")
                if context_metadata:
                    logger.info("📦 CONTEXT_METADATA: %s", context_metadata)
                    yield {"type": "context_metadata", "metadata": context_metadata}
                # Initialize conversation_messages with managed messages
                if messages:
                    conversation_messages = messages.copy()
                    yield {
                        "type": "messages_state",
                        "messages": conversation_messages.copy(),
                        "message_count": len(conversation_messages),
                    }

        # Handle completion (tool calls, final message)
        elif event_type == "on_chat_model_end":
            if first_llm_time is None:
                first_llm_time = time.perf_counter()
                elapsed_ms = (first_llm_time - stream_start) * 1000
                logger.debug("[TIMING] first_llm_response | elapsed=%.2fms", elapsed_ms)
            logger.info("🔚 CHAT_MODEL_END: completing LLM call")
            output = data.get("output")
            if output:
                # Add AI message to conversation for token tracking
                conversation_messages.append(output)
                yield {
                    "type": "messages_state",
                    "messages": conversation_messages.copy(),
                    "message_count": len(conversation_messages),
                }

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

            if tool_call_time is None:
                tool_call_time = time.perf_counter()
                elapsed_ms = (tool_call_time - stream_start) * 1000
                logger.debug(
                    "[TIMING] first_tool_result | elapsed=%.2fms | tool=%s",
                    elapsed_ms,
                    tool_name,
                )

            logger.info(
                "🔧 AGENT_TOOL_END: tool=%s, result_len=%d",
                tool_name,
                len(str(tool_result)),
            )

            # Extract SQL from stored_tool_calls (pop from front - FIFO order)
            sql = ""
            tool_call_id = "unknown"
            if stored_tool_calls:
                stored_tc = stored_tool_calls.pop(0)
                tool_call_id = stored_tc.get("id", "unknown")
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

            # Add ToolMessage to conversation for token tracking
            from langchain_core.messages import ToolMessage

            tool_msg = ToolMessage(content=str(tool_result), tool_call_id=tool_call_id)
            conversation_messages.append(tool_msg)
            yield {
                "type": "messages_state",
                "messages": conversation_messages.copy(),
                "message_count": len(conversation_messages),
            }

            yield {
                "type": "tool_result",
                "tool": tool_name,
                "result": str(tool_result),
                "sql": sql,
            }

    # Stream complete - log final timing summary
    total_ms = (time.perf_counter() - stream_start) * 1000
    logger.debug("=" * 60)
    logger.debug(
        "[TIMING] STREAM_COMPLETE | total=%.2fms | events=%d", total_ms, event_count
    )
    if first_token_time:
        ttft_ms = (first_token_time - stream_start) * 1000
        logger.debug("[TIMING] time_to_first_token=%.2fms", ttft_ms)
    if first_llm_time:
        llm_ms = (first_llm_time - stream_start) * 1000
        logger.debug("[TIMING] time_to_first_llm_response=%.2fms", llm_ms)
    if tool_call_time:
        tool_ms = (tool_call_time - stream_start) * 1000
        logger.debug("[TIMING] time_to_first_tool_result=%.2fms", tool_ms)
    logger.debug("=" * 60)
    logger.info(
        "[TIMING] STREAM_COMPLETE | total=%.2fms | events=%d", total_ms, event_count
    )
    if first_token_time:
        ttft_ms = (first_token_time - stream_start) * 1000
        logger.info("[TIMING] time_to_first_token=%.2fms", ttft_ms)
    if first_llm_time:
        llm_ms = (first_llm_time - stream_start) * 1000
        logger.info("[TIMING] time_to_first_llm_response=%.2fms", llm_ms)
    if tool_call_time:
        tool_ms = (tool_call_time - stream_start) * 1000
        logger.info("[TIMING] time_to_first_tool_result=%.2fms", tool_ms)
    logger.info("=" * 60)


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
