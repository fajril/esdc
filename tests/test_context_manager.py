# tests/test_context_manager.py

import pytest
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from esdc.chat.context_manager import ContextManager, manage_context_node


class TestContextManager:
    """Tests for context management with proactive compaction."""

    def test_no_action_when_under_threshold(self):
        """Test no compaction when under 75% threshold."""
        manager = ContextManager(max_tokens=10000)
        messages = [HumanMessage(content="Hello")]

        assert not manager.should_compact(messages)

        result, metadata = manager.manage_context(messages)
        assert not metadata["was_compacted"]
        assert len(result) == 1

    def test_should_compact_at_75_percent(self):
        """Test compaction triggers at 75% threshold."""
        manager = ContextManager(max_tokens=100, compaction_threshold=0.75)
        # Create messages that exceed 75 tokens (75 chars // 4 = ~18 tokens)
        messages = [HumanMessage(content="x" * 80) for _ in range(4)]

        # Should trigger at 75% (75 tokens threshold)
        assert manager.should_compact(messages)

    def test_hybrid_compaction(self):
        """Test hybrid strategy creates summary and keeps recent."""
        manager = ContextManager(
            max_tokens=100, compaction_threshold=0.75, recent_messages=2
        )
        messages = [
            HumanMessage(content="Query 1"),
            AIMessage(content="Response 1"),
            ToolMessage(content="Result 1", tool_call_id="1"),
            HumanMessage(content="Query 2"),
            AIMessage(content="Response 2"),
            ToolMessage(content="Result 2", tool_call_id="2"),
            HumanMessage(content="Query 3"),
            AIMessage(content="Response 3"),
        ]

        result, metadata = manager.manage_context(messages, force=True)

        assert metadata["was_compacted"]
        assert metadata["summarized_count"] == 6  # 8 - 2
        assert metadata["recent_kept"] == 2

        # Check summary system message was added
        assert any("Context automatically compacted" in str(m.content) for m in result)

    def test_preserves_system_messages(self):
        """Test system messages are always preserved."""
        manager = ContextManager(
            max_tokens=100, compaction_threshold=0.75, recent_messages=2
        )
        messages = [
            SystemMessage(content="System prompt"),
            HumanMessage(content="Query 1"),
            HumanMessage(content="Query 2"),
            HumanMessage(content="Query 3"),
        ]

        result, _ = manager.manage_context(messages, force=True)

        # System message should be present
        assert any(isinstance(m, SystemMessage) for m in result)

    def test_summary_content_extraction(self):
        """Test summary correctly extracts key information."""
        manager = ContextManager()
        messages = [
            HumanMessage(content="What is the oil reserves?"),
            AIMessage(
                content="I will check",
                tool_calls=[
                    {"name": "get_recommended_table", "args": {}, "id": "call1"}
                ],
            ),
            ToolMessage(content="Table: field_resources", tool_call_id="call1"),
        ]

        summary = manager._create_summary(messages)

        assert "User asked" in summary
        assert "What is the oil reserves" in summary


class TestManageContextNode:
    """Tests for LangGraph node wrapper."""

    def test_empty_messages(self):
        """Test node handles empty messages."""
        state = {"messages": []}
        result = manage_context_node(state)

        assert result["messages"] == []
        assert result["context_metadata"]["was_compacted"] == False

    def test_returns_context_metadata(self):
        """Test node returns context metadata."""
        state = {"messages": [HumanMessage(content=f"Query {i}") for i in range(20)]}
        result = manage_context_node(state)

        assert "context_metadata" in result
        assert "was_compacted" in result["context_metadata"]
