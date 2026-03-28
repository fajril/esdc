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


class TestEstimateTokens:
    """Tests for token estimation across message types."""

    def test_human_message_tokens(self):
        """Test token estimation for HumanMessage content."""
        manager = ContextManager()
        messages = [HumanMessage(content="Hello world")]  # 11 chars
        tokens = manager._estimate_tokens(messages)
        assert tokens == 2  # 11 // 4 = 2

    def test_ai_message_content_tokens(self):
        """Test token estimation for AIMessage content."""
        manager = ContextManager()
        messages = [AIMessage(content="Response text here")]  # 18 chars
        tokens = manager._estimate_tokens(messages)
        assert tokens == 4  # 18 // 4 = 4

    def test_system_message_tokens(self):
        """Test token estimation for SystemMessage content."""
        manager = ContextManager()
        messages = [SystemMessage(content="System prompt")]  # 13 chars
        tokens = manager._estimate_tokens(messages)
        assert tokens == 3  # 13 // 4 = 3

    def test_tool_message_large_content(self):
        """Test token estimation for large ToolMessage content."""
        manager = ContextManager()
        large_content = "x" * 100000  # 100K chars (simulating SQL query result)
        messages = [ToolMessage(content=large_content, tool_call_id="1")]
        tokens = manager._estimate_tokens(messages)
        assert tokens == 25000  # 100000 // 4 = 25000

    def test_ai_message_tool_calls(self):
        """Test token estimation includes tool_calls arguments."""
        manager = ContextManager()
        messages = [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "execute_sql",
                        "args": {"query": "SELECT * FROM reserves"},
                        "id": "1",
                    }
                ],
            )
        ]
        # name: 11 chars, args: {"query": "SELECT * FROM reserves"} = ~30 chars
        # total ~41 chars, 41 // 4 = 10
        tokens = manager._estimate_tokens(messages)
        assert tokens >= 10

    def test_multiple_message_types(self):
        """Test token estimation for mixed message types."""
        manager = ContextManager()
        messages = [
            SystemMessage(content="System prompt"),  # 13 chars
            HumanMessage(content="Query"),  # 5 chars
            AIMessage(
                content="Response",  # 8 chars
                tool_calls=[
                    {"name": "exec", "args": {"q": "test"}, "id": "1"}
                ],  # ~15 chars
            ),
            ToolMessage(content="Result data", tool_call_id="1"),  # 11 chars
        ]
        # Total: 13 + 5 + 8 + 15 + 11 = 52 chars, 52 // 4 = 13
        tokens = manager._estimate_tokens(messages)
        assert tokens == 13

    def test_empty_content(self):
        """Test token estimation handles empty content."""
        manager = ContextManager()
        messages = [HumanMessage(content="")]
        tokens = manager._estimate_tokens(messages)
        assert tokens == 0

    def test_ai_message_empty_tool_calls(self):
        """Test AIMessage with empty tool_calls list."""
        manager = ContextManager()
        messages = [AIMessage(content="Hello", tool_calls=[])]
        tokens = manager._estimate_tokens(messages)
        assert tokens == 1  # 5 chars // 4 = 1


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

    def test_context_length_parameter_default(self):
        """Test node uses default context_length (6000)."""
        state = {"messages": [HumanMessage(content="Test")]}
        result = manage_context_node(state)

        assert "context_metadata" in result
        assert result["context_metadata"]["was_compacted"] == False

    def test_context_length_parameter_custom(self):
        """Test node uses custom context_length."""
        # Create enough messages to trigger compaction (need more than recent_messages=6)
        # 10 messages * 80 chars = 800 chars / 4 = 200 tokens
        state = {"messages": [HumanMessage(content="x" * 80) for _ in range(10)]}

        # With default context_length=6000, should NOT compact (under threshold: 200 < 4500)
        result_default = manage_context_node(state, context_length=6000)
        assert result_default["context_metadata"]["was_compacted"] == False

        # With context_length=100, should compact (over 75% threshold: 200 > 75)
        # 10 messages, recent_messages=6 means 4 older messages to summarize
        result_custom = manage_context_node(state, context_length=100)
        assert result_custom["context_metadata"]["was_compacted"] == True
        assert result_custom["context_metadata"]["summarized_count"] == 4
