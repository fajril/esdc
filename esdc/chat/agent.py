# Standard library
import asyncio
import contextlib
import json
import logging
import time
from typing import Any, cast

# Third-party
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.runnables import Runnable
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

# Local
from esdc.chat.context_manager import AgentState, manage_context_node
from esdc.chat.openterminal import get_openterminal_tools
from esdc.chat.prompts import get_system_prompt
from esdc.chat.query_classifier import (
    QueryClassifier,
    format_classification_for_prompt,
    get_tools_for_classification,
)
from esdc.chat.skills import discover_skills, inject_skills_into_prompt
from esdc.chat.smart_query import simple_data_query
from esdc.chat.tools import (
    entity_resolver,
    execute_sql,
    get_recommended_table,
    get_resources_columns,
    get_schema,
    get_timeseries_columns,
    knowledge_traversal,
    list_tables,
    read_document,
    resolve_spatial,
    resolve_uncertainty_level,
    search_documents,
    search_problem_cluster,
    semantic_search,
)

# Logger is configured by app.py (runs first)
logger = logging.getLogger("esdc.chat.agent")

MAX_TOOL_CALLS = 50


_context_length_cache: dict[str, int] = {}


def _detect_context_length(llm: BaseChatModel) -> int:
    """Auto-detect model context length from LLM instance.

    Priority:
    1. Check ``_esdc_context_length`` metadata set by the provider's
       ``create_llm()``. This is dynamically fetched from provider APIs
       (Ollama API, Anthropic API, etc.) and is the source of truth.
    2. Fallback to ``isinstance()`` detection + static dicts (legacy).
    3. Fallback to ``DEFAULT_CONTEXT_LENGTH``.

    Results are cached by a deterministic model key.
    """
    # ── Priority 1: provider metadata ──────────────────────────
    val: int = 0
    try:
        raw = getattr(llm, "_esdc_context_length", 0)
        if isinstance(raw, int) and raw > 0:
            val = raw
    except AttributeError:
        pass

    if val > 0:
        logger.info(
            "[CONTEXT_LENGTH] from_provider_metadata | context_length=%d",
            val,
        )
        return val

    # Build a cache key from the LLM instance's identity.
    model_key = ""
    try:
        from langchain_ollama import ChatOllama

        if isinstance(llm, ChatOllama):
            model = getattr(llm, "model", "")
            model_key = f"ollama:{model}"
    except ImportError:
        pass

    if not model_key:
        try:
            from langchain_openai import ChatOpenAI

            if isinstance(llm, ChatOpenAI):
                model_key = f"openai:{llm.model_name}"
        except ImportError:
            pass

    if not model_key:
        try:
            from langchain_anthropic import ChatAnthropic

            if isinstance(llm, ChatAnthropic):
                model_key = f"anthropic:{llm.model}"
        except ImportError:
            pass

    if not model_key:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI

            if isinstance(llm, ChatGoogleGenerativeAI):
                model_key = f"google:{llm.model}"
        except ImportError:
            pass

    if not model_key:
        try:
            from langchain_openai import AzureChatOpenAI

            if isinstance(llm, AzureChatOpenAI):
                model_key = f"azure_openai:{llm.azure_deployment}"  # type: ignore[attr-defined]
        except ImportError:
            pass

    if not model_key:
        try:
            from langchain_groq import ChatGroq

            if isinstance(llm, ChatGroq):
                model_key = f"groq:{llm.model}"  # type: ignore[attr-defined]
        except ImportError:
            pass

    if model_key:
        if model_key in _context_length_cache:
            return _context_length_cache[model_key]

        # ── Priority 2: static provider dicts ────────────────────
        from esdc.providers import get_provider

        provider_type = model_key.split(":")[0]
        provider_cls = get_provider(provider_type)
        if provider_cls is not None:
            model_name = model_key.split(":", 1)[1]
            model_context_length = provider_cls.get_actual_context_length(model_name)
            if model_context_length > 0:
                _context_length_cache[model_key] = model_context_length
                logger.info(
                    "[CONTEXT_LENGTH] from_provider_static | "
                    "model_key=%s | context_length=%d",
                    model_key,
                    model_context_length,
                )
                return model_context_length

    # ── Priority 3: default fallback ─────────────────────────
    from esdc.providers.base import DEFAULT_CONTEXT_LENGTH

    model_context_length = DEFAULT_CONTEXT_LENGTH
    logger.debug(
        "[CONTEXT_LENGTH] using_default | context_length=%d",
        model_context_length,
    )
    return model_context_length


def _merge_allowed_tools(
    classifier_tools: list[str],
    conditional_tool_names: set[str],
) -> list[str]:
    """Classifier-selected tools plus conditionally-registered ones.

    Conditionally-registered tools (OpenTerminal sandbox tools, external
    passthrough tools) are not known to the classifier, so they must always
    stay allowed. Everything else follows the classifier's restriction.
    """
    return sorted(set(classifier_tools) | conditional_tool_names)


MAX_TOOL_RESULT_CHARS = 10000

# A signature is blocked once it would be the 3rd+ execution of an
# identical tool+args call (i.e. 2 prior executions already happened):
# repeating past this point cannot change the output and only burns the
# MAX_TOOL_CALLS budget (see the Entity Resolver death-spiral in
# docs/plans/2026-07-13-improve-document-search-usage.md).
_LOOP_DETECTION_MAX_PRIOR_EXECUTIONS = 2

_REPEATED_CALL_BLOCKED_PREFIX = "REPEATED CALL BLOCKED"


def _tool_call_signature(tool_name: str, tool_args: Any) -> str:
    """Build a stable signature identifying a tool call by name + args.

    Used by the tool_node loop guard to recognize when the LLM repeats an
    identical call instead of trying something different.
    """
    if isinstance(tool_args, str):
        with contextlib.suppress(json.JSONDecodeError, TypeError):
            tool_args = json.loads(tool_args)
    return f"{tool_name}:{json.dumps(tool_args, sort_keys=True, default=str)}"


def _count_executed_tool_signatures(messages: list[AnyMessage]) -> dict[str, int]:
    """Count prior tool executions per call signature in the current turn.

    Only messages after the last HumanMessage are counted: a user asking
    the same question in a later turn legitimately re-runs the same tool
    calls and must not inherit counts from earlier turns.

    Pairs each ToolMessage with the tool_call it answers (matched by
    tool_call_id against preceding AIMessage.tool_calls) to recover the
    tool+args combination it represents. Calls the loop guard itself
    already blocked (content prefixed with "REPEATED CALL BLOCKED") never
    actually executed, so they are not counted.
    """
    turn_start = 0
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], HumanMessage):
            turn_start = i + 1
            break
    turn_messages = messages[turn_start:]

    call_index: dict[str, tuple[str, Any]] = {}
    for msg in turn_messages:
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            for tc in msg.tool_calls:
                tc_id = tc.get("id")
                if tc_id:
                    call_index[tc_id] = (tc.get("name", "unknown"), tc.get("args", {}))

    counts: dict[str, int] = {}
    for msg in turn_messages:
        if isinstance(msg, ToolMessage) and not str(msg.content).startswith(
            _REPEATED_CALL_BLOCKED_PREFIX
        ):
            entry = call_index.get(msg.tool_call_id)
            if entry is None:
                continue
            signature = _tool_call_signature(*entry)
            counts[signature] = counts.get(signature, 0) + 1
    return counts


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
    prompt = (
        "Buatkan judul singkat (maks. 50 karakter) yang merangkum "
        "pertanyaan user berikut dalam BAHASA yang SAMA dengan "
        "pertanyaan user.\n"
        "Judul harus ringkas dan padat.\n\n"
        "INSTRUCTION: Selalu respons dalam format JSON berikut (tanpa markdown):\n"
        '{{"title": "judul ringkas di sini"}}\n\n'
        "Contoh:\n"
        '- "berapa cadangan nasional" -> '
        '{{"title": "Cadangan nasional"}}\n'
        '- "buatkan profil produksi EOR" -> '
        '{{"title": "Profil produksi EOR"}}\n'
        '- "berapa cadangan lapangan Duri" -> '
        '{{"title": "Cadangan lapangan Duri"}}\n'
        '- "how much oil reserves in Rokan field" -> '
        '{{"title": "Oil Reserves Rokan Field"}}\n'
        '- "list all working areas with gas production" -> '
        '{{"title": "Working Areas Gas Production"}}\n'
        '- "compare reserves between 2020 and 2023" -> '
        '{{"title": "Reserve Comparison 2020-2023"}}\n\n'
        "Pertanyaan user: {query}\n\n"
        "Judul:"
    )

    try:
        messages = [
            SystemMessage(
                content=(
                    "Kamu adalah asisten yang membuat judul percakapan "
                    "singkat dan ringkas. Gunakan bahasa yang sama dengan "
                    "pertanyaan user. Hanya respons dalam format JSON "
                    'tanpa markdown: {{"title": "..."}}.'
                )
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
        response_content = (
            response.content if hasattr(response, "content") else str(response)
        )
        content_len = len(response_content) if response_content else 0
        logger.debug(
            "[INFERENCE] title_generation_complete | elapsed=%.2fms | response_len=%d",
            inference_elapsed_ms,
            content_len,
        )

        raw_text = str(response.content).strip() if response.content else ""

        # Parse JSON title
        title = ""
        if raw_text:
            try:
                json_start = raw_text.find("{")
                json_end = raw_text.rfind("}") + 1
                if json_start >= 0 and json_end > json_start:
                    parsed = json.loads(raw_text[json_start:json_end])
                    title = parsed.get("title", "")
            except (json.JSONDecodeError, ValueError):
                pass

        # Fallback: plain text extraction if JSON parsing fails
        if not title:
            title = raw_text.strip().strip("\"'")

        if len(title) > 50:
            title = title[:47] + "..."

        return title
    except Exception:
        logger.exception(
            "[INFERENCE] title_generation_error | query=%r",
            user_query[:80],
        )
        query_clean = user_query.strip()
        if len(query_clean) > 50:
            return query_clean[:47] + "..."
        return query_clean


async def generate_conversation_tags(
    llm: BaseChatModel,
    user_query: str,
) -> str:
    """Generate broad tags categorizing the conversation.

    Args:
        llm: Language model to use for generation
        user_query: First user query

    Returns:
        Comma-separated tags (max 100 chars)
    """
    prompt = (
        "Generate 1-3 broad tags categorizing this user query.\n"
        "The tags should be general categories.\n\n"
        "INSTRUCTION: Selalu respons dalam format JSON berikut (tanpa markdown):\n"
        '{{"tags": "tag1, tag2, tag3"}}\n\n'
        "Examples:\n"
        '- "how much oil reserves in Rokan field" -> '
        '{{"tags": "Reserves, Oil, Rokan"}}\n'
        '- "list all working areas with gas production" -> '
        '{{"tags": "Working Areas, Gas Production"}}\n'
        '- "compare reserves between 2020 and 2023" -> '
        '{{"tags": "Reserves, Comparison"}}\n\n'
        "User query: {query}\n\n"
        "Tags:"
    )

    try:
        messages = [
            SystemMessage(
                content=(
                    "You are a helpful assistant that "
                    "generates broad categorization tags. "
                    "Respond ONLY in JSON format: "
                    '{{"tags": "tag1, tag2, ..."}}. '
                    "No markdown, no explanation."
                )
            ),
            HumanMessage(content=prompt.format(query=user_query)),
        ]

        total_chars = sum(len(str(m.content)) for m in messages)
        logger.debug(
            "[INFERENCE] tag_generation_start | messages=%d | total_chars=%d",
            len(messages),
            total_chars,
        )
        inference_start = time.perf_counter()

        response = await llm.ainvoke(messages)

        inference_elapsed_ms = (time.perf_counter() - inference_start) * 1000
        response_content = (
            response.content if hasattr(response, "content") else str(response)
        )
        content_len = len(response_content) if response_content else 0
        logger.debug(
            "[INFERENCE] tag_generation_complete | elapsed=%.2fms | response_len=%d",
            inference_elapsed_ms,
            content_len,
        )

        raw_text = str(response.content).strip() if response.content else ""

        # Parse JSON tags
        tags = ""
        if raw_text:
            try:
                json_start = raw_text.find("{")
                json_end = raw_text.rfind("}") + 1
                if json_start >= 0 and json_end > json_start:
                    parsed = json.loads(raw_text[json_start:json_end])
                    tags = parsed.get("tags", "")
            except (json.JSONDecodeError, ValueError):
                pass

        # Fallback: plain text extraction if JSON parsing fails
        if not tags:
            tags = raw_text.strip().strip("\"'")

        if len(tags) > 100:
            tags = tags[:97] + "..."

        return tags
    except Exception:
        logger.exception(
            "[INFERENCE] tag_generation_error | query=%r",
            user_query[:80],
        )
        query_clean = user_query.strip()
        if len(query_clean) > 100:
            return query_clean[:97] + "..."
        return query_clean


def create_agent(
    llm: BaseChatModel,
    tools: list | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
    context_length: int | None = None,
    external_tool_names: set[str] | None = None,
    external_tools: list | None = None,
) -> Runnable:
    """Create a LangGraph agent with tools.

    Args:
        llm: A LangChain chat model
        tools: Optional list of tools. Defaults to all registered tools
        checkpointer: Optional checkpointer for memory persistence
        context_length: Maximum context length in tokens for messages.
            If None, auto-detected from the LLM model's context window.
            Explicit values (e.g. from TUI) take precedence.
        external_tool_names: Set of tool names that are external (e.g. OpenTerminal).
            External tool calls are not executed server-side; instead, a marker
            is returned for the Responses API to pass through to the client.
        external_tools: Optional list of LangChain tool objects for external tools.
            These are bound to the LLM so it can call them, but their execution
            is intercepted by tool_node and returned as markers.

    Returns:
        A compiled StateGraph agent
    """
    if context_length is None:
        context_length = _detect_context_length(llm)
    provider_type = getattr(llm, "_esdc_provider_type", None)
    model_name = getattr(llm, "_esdc_model_name", None)
    base_url = getattr(llm, "_esdc_base_url", None)
    if tools is None:
        tools = [
            simple_data_query,
            entity_resolver,
            knowledge_traversal,
            resolve_spatial,
            semantic_search,
            search_documents,
            read_document,
            execute_sql,
            get_schema,
            list_tables,
            get_recommended_table,
            resolve_uncertainty_level,
            search_problem_cluster,
            get_timeseries_columns,
            get_resources_columns,
        ]

    # Conditionally add OpenTerminal tools when configured
    openterminal_tools = get_openterminal_tools()
    if openterminal_tools:
        tools = tools + openterminal_tools

    if external_tools:
        tools = tools + external_tools

    _external_tool_names = external_tool_names or set()

    conditional_tool_names: set[str] = set(_external_tool_names)
    if openterminal_tools:
        conditional_tool_names |= {t.name for t in openterminal_tools}

    all_tools: dict[str, Any] = {tool.name: tool for tool in tools}

    # Discover skills and inject their instructions into the system prompt
    skills = discover_skills()

    tools_by_name = dict(all_tools)

    def init_node(state: AgentState) -> dict[str, Any]:
        """Initialize system prompt and defaults in state (runs once)."""
        system_prompt = get_system_prompt()
        if skills:
            system_prompt = inject_skills_into_prompt(system_prompt, skills)
        logger.debug("[INIT] system_prompt_set | len=%d", len(system_prompt))
        return {
            "system_prompt": system_prompt,
            "allowed_tools": list(all_tools.keys()),
            "tool_call_count": 0,
        }

    _max_tool_calls_reached_msg = (
        "Tool call limit reached ({limit}). "
        "You must now provide your final answer using the data you already have. "
        "Do NOT make any more tool calls."
    )

    async def agent_node(state: AgentState) -> dict[str, list[AnyMessage]]:
        """Agent node that calls the LLM with tools."""
        system_prompt = state.get("system_prompt", "")
        if not system_prompt:
            system_prompt = get_system_prompt()
            logger.warning(
                "[AGENT] system_prompt missing from state, regenerating | len=%d",
                len(system_prompt),
            )

        big_sys_in_msgs = [
            m
            for m in state["messages"]
            if isinstance(m, SystemMessage) and len(str(m.content)) > 10000
        ]
        if big_sys_in_msgs:
            logger.warning(
                "[AGENT] DUPLICATE_SYSTEM_PROMPT_DETECTED | big_sys_count=%d | "
                "total_msgs=%d | prompt_len=%d",
                len(big_sys_in_msgs),
                len(state["messages"]),
                len(system_prompt),
            )

        # Filter out empty assistant messages that can cause death spiral
        # when accumulated in conversation history
        filtered_messages = []
        empty_count = 0
        for m in state["messages"]:
            if isinstance(m, AIMessage) and not m.content and not m.tool_calls:
                empty_count += 1
                continue
            filtered_messages.append(m)
        if empty_count:
            logger.warning(
                "[AGENT] Filtered %d empty AIMessage from history",
                empty_count,
            )

        messages_with_system = [
            SystemMessage(content=system_prompt)
        ] + filtered_messages

        tool_call_count = state.get("tool_call_count", 0)
        if tool_call_count >= 35:
            nudge = (
                f"CRITICAL: You have already made {tool_call_count} tool calls. "
                "You are approaching the limit of 50. "
                "You MUST synthesize the data you already have and respond directly. "
                "Do NOT make more tool calls unless absolutely essential."
            )
            messages_with_system.append(SystemMessage(content=nudge))
            logger.info(
                "[AGENT] strong_nudge_injected | tool_call_count=%d",
                tool_call_count,
            )
        elif tool_call_count >= 20:
            nudge = (
                f"NOTE: You have made {tool_call_count} tool calls. "
                "If you have enough data to answer the user's question, "
                "consider synthesizing your findings rather than making more calls."
            )
            messages_with_system.append(SystemMessage(content=nudge))
            logger.info(
                "[AGENT] gentle_nudge_injected | tool_call_count=%d",
                tool_call_count,
            )

        allowed_tools = state.get("allowed_tools", list(all_tools.keys()))
        selected_tools = [
            all_tools[name] for name in allowed_tools if name in all_tools
        ]
        if not selected_tools:
            selected_tools = list(all_tools.values())

        llm_with_selected_tools = llm.bind_tools(selected_tools)

        total_chars = sum(len(str(m.content)) for m in messages_with_system)
        system_prompt_len = len(system_prompt)
        user_messages = len(
            [m for m in messages_with_system if isinstance(m, HumanMessage)]
        )
        logger.debug(
            "[INFERENCE] llm_invoke_start | messages=%d | user_messages=%d | "
            "system_prompt_len=%d | total_chars=%d | allowed_tools=%s",
            len(messages_with_system),
            user_messages,
            system_prompt_len,
            total_chars,
            allowed_tools,
        )
        inference_start = time.perf_counter()

        try:
            response = await asyncio.wait_for(
                llm_with_selected_tools.ainvoke(messages_with_system),
                timeout=120,
            )
        except asyncio.TimeoutError:
            inference_elapsed_ms = (time.perf_counter() - inference_start) * 1000
            logger.error(
                "[INFERENCE] llm_invoke_timeout | elapsed=%.2fms | timeout=120s",
                inference_elapsed_ms,
            )
            return {
                "messages": [
                    cast(
                        AnyMessage,
                        AIMessage(
                            content="Maaf, permintaan Anda memakan waktu terlalu lama untuk diproses. Silakan coba lagi atau sederhanakan pertanyaan Anda."  # noqa: E501
                        ),
                    )
                ]
            }

        inference_elapsed_ms = (time.perf_counter() - inference_start) * 1000
        response_len = len(str(response.content)) if hasattr(response, "content") else 0
        ai_response = cast(AIMessage, response)
        has_tool_calls = bool(ai_response.tool_calls)
        tool_call_count = len(ai_response.tool_calls) if has_tool_calls else 0
        logger.debug(
            "[INFERENCE] llm_invoke_complete | elapsed=%.2fms | response_len=%d | "
            "has_tool_calls=%s | tool_calls=%d",
            inference_elapsed_ms,
            response_len,
            has_tool_calls,
            tool_call_count,
        )

        if not response.content and not has_tool_calls:
            logger.warning("[AGENT] Empty LLM response, injecting fallback")
            return {
                "messages": [
                    cast(
                        AnyMessage,
                        AIMessage(
                            content="Maaf, saya tidak dapat memproses permintaan Anda. Silakan coba lagi."  # noqa: E501
                        ),
                    )
                ]
            }

        return {"messages": [cast(AnyMessage, response)]}

    def should_continue(state: AgentState) -> str:
        """Determine if we should continue to tools or end."""
        tool_call_count = state.get("tool_call_count", 0)
        if tool_call_count >= MAX_TOOL_CALLS:
            logger.warning(
                "[TOOL] TOOL_LIMIT: Reached %d tool calls, forcing end",
                tool_call_count,
            )
            return END
        messages = state["messages"]
        if not messages:
            logger.warning("[AGENT] should_continue: empty messages, ending")
            return END

        last_message = messages[-1]

        ai_message = cast(AIMessage, last_message)
        if hasattr(ai_message, "tool_calls") and ai_message.tool_calls:
            tool_call_count = state.get("tool_call_count", 0)
            if tool_call_count >= MAX_TOOL_CALLS:
                # Check if we already sent a "limit reached" response
                # (ToolMessage with the limit-reached message). If so, the
                # LLM ignored it and tried to call tools again — force END.
                for msg in reversed(messages):
                    if isinstance(
                        msg, ToolMessage
                    ) and _max_tool_calls_reached_msg.format(
                        limit=MAX_TOOL_CALLS
                    ) in str(msg.content):
                        logger.warning(
                            "🔧 TOOL_LIMIT: LLM still calling tools after "
                            "limit-reached message, forcing END"
                        )
                        return END

                logger.warning(
                    "🔧 TOOL_LIMIT: Reached %d tool calls, routing to tool_node "
                    "for limit-reached response before final synthesis",
                    tool_call_count,
                )
            return "tools"

        return END

    async def tool_node(state: AgentState) -> dict[str, Any]:
        """Tool execution node."""
        result = []
        last_message = state["messages"][-1]
        allowed_tools = set(state.get("allowed_tools", list(all_tools.keys())))
        current_tool_count = state.get("tool_call_count", 0)

        ai_message = cast(AIMessage, last_message)
        if not hasattr(ai_message, "tool_calls") or not ai_message.tool_calls:
            return {"messages": [], "tool_call_count": current_tool_count}

        if current_tool_count >= MAX_TOOL_CALLS:
            logger.warning(
                "[TOOL_NODE] Forced stop: tool_call_count=%d >= MAX_TOOL_CALLS=%d | "
                "returning limit-reached messages for %d pending calls",
                current_tool_count,
                MAX_TOOL_CALLS,
                len(ai_message.tool_calls),
            )
            for tool_call in ai_message.tool_calls:
                tool_id = tool_call.get("id", "unknown")
                tool_name = tool_call.get("name", "unknown")
                result.append(
                    {
                        "tool_call_id": tool_id,
                        "role": "tool",
                        "name": tool_name,
                        "content": _max_tool_calls_reached_msg.format(
                            limit=MAX_TOOL_CALLS
                        ),
                    }
                )
            return {
                "messages": result,
                "tool_call_count": current_tool_count,
            }

        logger.info(
            "[TOOL] TOOL_NODE: Processing %d tool calls", len(ai_message.tool_calls)
        )

        signature_counts = _count_executed_tool_signatures(state["messages"])

        for tool_call in ai_message.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call.get("args", {})
            tool_id = tool_call.get("id", "unknown")

            signature = _tool_call_signature(tool_name, tool_args)
            prior_executions = signature_counts.get(signature, 0)
            if prior_executions >= _LOOP_DETECTION_MAX_PRIOR_EXECUTIONS:
                logger.warning(
                    "[TOOL_NODE] Loop detected: %s already executed %d times with "
                    "identical args, blocking | signature=%s",
                    tool_name,
                    prior_executions,
                    signature,
                )
                result.append(
                    {
                        "tool_call_id": tool_id,
                        "role": "tool",
                        "name": tool_name,
                        "content": (
                            f"{_REPEATED_CALL_BLOCKED_PREFIX}: '{tool_name}' already "
                            f"returned identical results for these exact arguments "
                            f"{prior_executions} times. Calling again will not change "
                            "the output. Use a DIFFERENT tool (e.g. search_documents "
                            "for document queries) or give your final answer with "
                            "what you have."
                        ),
                    }
                )
                continue

            if tool_name in _external_tool_names:
                logger.info(
                    "[TOOL] TOOL_NODE: External tool call detected: %s (id=%s). "
                    "Returning marker for passthrough.",
                    tool_name,
                    tool_id,
                )
                result.append(
                    {
                        "tool_call_id": tool_id,
                        "role": "tool",
                        "name": tool_name,
                        "content": f"[EXTERNAL_TOOL_CALL:{tool_name}]",
                    }
                )
                continue

            if tool_name not in allowed_tools:
                logger.warning(
                    "[TOOL] TOOL_NODE: Blocked disallowed tool %s (allowed: %s)",
                    tool_name,
                    sorted(allowed_tools),
                )
                result.append(
                    {
                        "tool_call_id": tool_id,
                        "role": "tool",
                        "name": tool_name,
                        "content": (
                            f"Error: Tool '{tool_name}' is not available "
                            f"for this query type. Use only: "
                            f"{', '.join(sorted(allowed_tools))}"
                        ),
                    }
                )
                continue

            if tool_name in tools_by_name:
                tool = tools_by_name[tool_name]
                try:
                    if isinstance(tool_args, str):
                        tool_args = json.loads(tool_args)

                    logger.info(
                        "[TOOL] TOOL_NODE: Invoking %s (id=%s)", tool_name, tool_id
                    )
                    observation = await tool.ainvoke(tool_args)

                    # JSON-serialize dict/list observations for richer content
                    if isinstance(observation, (dict, list)):
                        observation = json.dumps(observation, ensure_ascii=False)

                    observation_str = str(observation)
                    logger.info(
                        "[TOOL] TOOL_NODE: %s returned %d chars",
                        tool_name,
                        len(observation_str),
                    )

                    if len(observation_str) > MAX_TOOL_RESULT_CHARS:
                        observation = (
                            observation_str[:MAX_TOOL_RESULT_CHARS]
                            + f"\n\n[Result truncated to first {MAX_TOOL_RESULT_CHARS} characters for context efficiency]"  # noqa: E501
                        )
                        logger.info(
                            "[TOOL] TOOL_NODE: %s result truncated from %d to %d chars",
                            tool_name,
                            len(observation_str),
                            MAX_TOOL_RESULT_CHARS,
                        )
                except Exception as e:
                    logger.error("[TOOL] TOOL_NODE: %s failed: %s", tool_name, e)
                    observation = f"Error: {str(e)}"

                signature_counts[signature] = prior_executions + 1
                result.append(
                    {
                        "tool_call_id": tool_id,
                        "role": "tool",
                        "name": tool_name,
                        "content": observation,
                    }
                )
            else:
                logger.warning(
                    "[TOOL] TOOL_NODE: Unknown tool %s (not in tools_by_name: %s)",
                    tool_name,
                    sorted(tools_by_name.keys()),
                )
                result.append(
                    {
                        "tool_call_id": tool_id,
                        "role": "tool",
                        "name": tool_name,
                        "content": f"Error: Tool '{tool_name}' is not available.",
                    }
                )

        logger.info("[TOOL] TOOL_NODE: Returning %d tool results", len(result))
        return {
            "messages": result,
            "tool_call_count": current_tool_count + len(ai_message.tool_calls),
        }

    def manage_context_with_length(state: AgentState) -> dict[str, Any]:
        """Wrapper for manage_context_node with bound context_length."""
        return manage_context_node(
            state,
            context_length=context_length,
            provider_type=provider_type,
            model=model_name,
            base_url=base_url,
        )

    def query_classification_node(state: AgentState) -> dict[str, Any]:
        """Classify query and inject strategy into system prompt."""
        messages = state["messages"]
        if not messages:
            return {"messages": [], "allowed_tools": list(all_tools.keys())}

        # Find last human message
        last_human: str | list[str | dict[str, Any]] | None = None
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage) and msg.content:
                last_human = msg.content
                break

        if not last_human:
            return {"messages": [], "allowed_tools": list(all_tools.keys())}

        query_text = str(last_human) if not isinstance(last_human, str) else last_human

        try:
            classifier = QueryClassifier()
            classification = classifier.classify(query_text)

            strategy_text = format_classification_for_prompt(classification)
            allowed_tools = get_tools_for_classification(classification)

            # Preserve conditionally-registered tools (e.g. OpenTerminal
            # Shell Executor) that exist in
            # all_tools but are not returned by the classifier. Without
            # this, query_classification_node would override allowed_tools
            # with only classifier-selected tools, dropping any tools that
            # were added by create_agent conditionally (like sandbox tools),
            # making the LLM unable to call them.
            before = set(allowed_tools)
            allowed_tools = _merge_allowed_tools(allowed_tools, conditional_tool_names)
            added = set(allowed_tools) - before
            if added:
                logger.debug(
                    "[CLASSIFICATION] Preserving conditionally-registered "
                    "tools not in classifier output: %s",
                    sorted(added),
                )

            strategy_msg = SystemMessage(content=strategy_text)
            logger.info(
                "query_classification: type=%s confidence=%.2f tools=%s query='%s...'",
                classification.query_type.name,
                classification.confidence,
                allowed_tools,
                last_human[:50],
            )
            return {
                "messages": [cast(AnyMessage, strategy_msg)],
                "allowed_tools": allowed_tools,
            }

        except Exception as e:
            logger.warning("query_classification: failed - %s", e)
            return {"messages": [], "allowed_tools": list(all_tools.keys())}

    def _validate_tool_name_mapping() -> None:
        """Warn if any tool name from classifier doesn't exist in all_tools."""
        from esdc.chat.query_classifier import (
            QueryClassification,
            QueryType,
            get_tools_for_classification,
        )

        classifier_tools: set[str] = set()
        for qtype in QueryType:
            classification = QueryClassification(
                query_type=qtype,
                confidence=0.9,
                detected_entities={},
                suggested_table=None,
                suggested_columns=[],
                reason="Validation",
            )
            for name in get_tools_for_classification(classification):
                classifier_tools.add(name)

        missing = classifier_tools - set(all_tools.keys())
        if missing:
            logger.warning(
                "[AGENT] Tool name mismatch: classifier references %s "
                "but not found in @tool() decorators: %s",
                missing,
                sorted(all_tools.keys()),
            )

    _validate_tool_name_mapping()

    # Build graph with query classification
    graph = (
        StateGraph(AgentState)
        .add_node("init", init_node)
        .add_node("manage_context", manage_context_with_length)
        .add_node("query_classification", query_classification_node)
        .add_node("agent", agent_node)
        .add_node("tools", tool_node)
        .add_edge(START, "init")
        .add_edge("init", "manage_context")
        .add_edge("manage_context", "query_classification")
        .add_edge("query_classification", "agent")
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
