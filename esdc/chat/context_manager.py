# esdc/chat/context_manager.py

# Standard library
from collections.abc import Sequence
from typing import Annotated, Any, TypedDict

# Third-party
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """LangGraph agent state with separate system_prompt field.

    Stores the system prompt separately from messages to prevent
    duplication across agent iterations via the add_messages reducer.
    """

    messages: Annotated[list[AnyMessage], add_messages]
    system_prompt: str
    context_metadata: dict
    allowed_tools: list[str]
    tool_call_count: int


def _get_content_str(content: str | list[str | dict[str, Any]] | None) -> str:
    """Extract string from content that can be str or list."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and "text" in item:
                parts.append(item["text"])
        return " ".join(parts)
    return str(content)


class ContextManager:
    """Manages conversation context using hybrid strategy with proactive compaction.

    Strategy:
    - Monitor token usage proactively (trigger at 75% threshold)
    - Keep last N messages verbatim (exact context)
    - Summarize older messages into condensed form
    - Preserve key facts, queries, and user intent
    """

    def __init__(
        self,
        max_tokens: int = 6000,
        compaction_threshold: float = 0.75,
        recent_messages: int = 6,
    ):
        """Initialize context manager with token budget and compaction settings."""
        self.max_tokens = max_tokens
        self.compaction_threshold = int(max_tokens * compaction_threshold)
        self.recent_messages = recent_messages
        self.compaction_count = 0

    def should_compact(self, messages: Sequence[AnyMessage]) -> bool:
        """Check if compaction is needed (proactive at 75%)."""
        if not messages:
            return False
        token_count = self._estimate_tokens(messages)
        return token_count >= self.compaction_threshold

    def manage_context(
        self, messages: Sequence[AnyMessage], force: bool = False
    ) -> tuple[list[AnyMessage], dict]:
        """Apply hybrid compaction if needed.

        Returns:
            Tuple of (managed_messages, metadata)
            metadata includes: was_compacted, original_count, summary_info
        """
        if not messages:
            return list(messages), {"was_compacted": False}

        # Proactive check: only compact if over threshold
        if not force and not self.should_compact(messages):
            return list(messages), {"was_compacted": False}

        self.compaction_count += 1
        return self._hybrid(messages)

    def _hybrid(self, messages: Sequence[AnyMessage]) -> tuple[list[AnyMessage], dict]:
        """Hybrid: Keep recent exact + summarized older."""
        # Separate system messages (always keep)
        system_messages = [m for m in messages if isinstance(m, SystemMessage)]
        other_messages = [m for m in messages if not isinstance(m, SystemMessage)]

        # Keep last N messages exact
        recent = other_messages[-self.recent_messages :]
        older = other_messages[: -self.recent_messages]

        if not older:
            return list(messages), {"was_compacted": False}

        # Create summary of older messages
        summary = self._create_summary(older)

        managed = [
            *system_messages,
            SystemMessage(
                content=f"[Context automatically compacted to manage token usage. Previous conversation summary below:]\n\n{summary}\n\n[End of summary. Recent {len(recent)} messages follow verbatim:]"  # noqa: E501
            ),
            *recent,
        ]

        metadata = {
            "was_compacted": True,
            "original_count": len(messages),
            "new_count": len(managed),
            "summarized_count": len(older),
            "recent_kept": len(recent),
            "summary_preview": summary[:200] + "..." if len(summary) > 200 else summary,
        }

        return managed, metadata

    def _create_summary(self, messages: Sequence[AnyMessage]) -> str:
        """Create intelligent summary of older messages."""
        summary_parts = []

        for msg in messages:
            if isinstance(msg, HumanMessage):
                content_str = _get_content_str(msg.content)
                content = (
                    content_str[:80] + "..." if len(content_str) > 80 else content_str
                )
                summary_parts.append(f"User asked: {content}")

            elif isinstance(msg, ToolMessage):
                content_preview = (
                    str(msg.content)[:50] + "..."
                    if len(str(msg.content)) > 50
                    else str(msg.content)
                )
                summary_parts.append(f"Tool returned: {content_preview}")

            elif isinstance(msg, AIMessage):
                if msg.tool_calls:
                    tool_names = [tc.get("name", "unknown") for tc in msg.tool_calls]
                    summary_parts.append(f"AI used tools: {', '.join(tool_names)}")
                elif msg.content:
                    content_str = _get_content_str(msg.content)
                    content = (
                        content_str[:60] + "..."
                        if len(content_str) > 60
                        else content_str
                    )
                    summary_parts.append(f"AI responded: {content}")

        return "\n".join(summary_parts[-10:])  # Keep last 10 significant actions

    def _estimate_tokens(self, messages: Sequence[AnyMessage]) -> int:
        """Estimate token count using ~4 chars per token.

        Counts tokens from all message types including:
        - HumanMessage/AIMessage/SystemMessage content
        - ToolMessage content (can be very large, 100K+ chars)
        - AIMessage tool_calls arguments
        """
        total_chars = 0

        for m in messages:
            # Count message content (applies to HumanMessage, AIMessage, ToolMessage, SystemMessage)  # noqa: E501
            if m.content:
                total_chars += len(str(m.content))

            # Count AIMessage tool_calls arguments (these contribute to token usage)
            if isinstance(m, AIMessage) and m.tool_calls:
                for tc in m.tool_calls:
                    total_chars += len(str(tc.get("name", "")))
                    args = tc.get("args", {})
                    total_chars += len(str(args))

        return total_chars // 4


def estimate_tokens(messages: Sequence[AnyMessage]) -> int:
    """Estimate token count from messages list.

    Public API for calculating tokens from messages in state.
    Uses the same algorithm as ContextManager._estimate_tokens.

    Args:
        messages: List of messages to estimate tokens for

    Returns:
        Estimated token count
    """
    total_chars = 0

    for m in messages:
        if m.content:
            total_chars += len(str(m.content))

        if isinstance(m, AIMessage) and m.tool_calls:
            for tc in m.tool_calls:
                total_chars += len(str(tc.get("name", "")))
                args = tc.get("args", {})
                total_chars += len(str(args))

    return total_chars // 4


def manage_context_node(
    state: AgentState, context_length: int = 6000
) -> dict[str, Any]:
    """LangGraph node wrapper for context management.

    This node is called before the agent to proactively manage context.
    The context_length parameter applies to the messages field only.
    The system prompt is stored separately in AgentState and does not
    compete with messages for the context budget.

    Args:
        state: LangGraph state with 'messages' key (and 'system_prompt')
        context_length: Maximum context length in tokens for messages (default: 6000)

    Returns:
        State dict with managed messages and context metadata
    """
    messages = state.get("messages", [])
    if not messages:
        return {"messages": [], "context_metadata": {"was_compacted": False}}

    manager = ContextManager(
        max_tokens=context_length,
        compaction_threshold=0.75,
        recent_messages=6,
    )

    managed_messages, metadata = manager.manage_context(messages)

    return {
        "messages": managed_messages,
        "context_metadata": metadata,
    }
