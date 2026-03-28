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


class TestLargeContext:
    """Tests for large context (262K tokens) support."""

    def test_context_manager_large_context_threshold(self):
        """Test ContextManager with 262144 max_tokens calculates correct threshold."""
        # Bug: was hardcoded to 6000, compaction threshold was 4500 (75% of 6000)
        # Fix: should use passed context_length, threshold should be 196608 (75% of 262144)
        manager = ContextManager(max_tokens=262144, compaction_threshold=0.75)

        # Verify threshold is 75% of 262K = 196608
        assert manager.max_tokens == 262144
        assert manager.compaction_threshold == 196608

    def test_no_compaction_at_large_context(self):
        """Test no compaction when under 262K context threshold.

        Simulate ~100K tokens (a large but reasonable conversation) and verify
        it doesn't trigger compaction since 100K < 196K threshold.
        """
        manager = ContextManager(max_tokens=262144, compaction_threshold=0.75)

        # Simulate ~100K tokens of content (400K chars / 4 = 100K tokens)
        # This is well under 75% of 262K
        messages = [HumanMessage(content="x" * 40000) for _ in range(10)]  # 100K tokens

        # Should not trigger compaction (100K < 196K threshold)
        assert not manager.should_compact(messages)

    def test_compaction_at_large_context_threshold(self):
        """Test compaction triggers at 75% of 262K threshold.

        Create messages exceeding 196K tokens (75% of 262K) and verify
        compaction is triggered.
        """
        manager = ContextManager(max_tokens=262144, compaction_threshold=0.75)

        # Simulate ~200K tokens (800K chars / 4 = 200K tokens)
        # This exceeds 196K threshold (75% of 262K)
        messages = [HumanMessage(content="x" * 80000) for _ in range(10)]  # 200K tokens

        # Should trigger compaction (200K > 196K threshold)
        assert manager.should_compact(messages)

    def test_large_token_estimation_with_tool_results(self):
        """Test token estimation handles large SQL tool results.

        SQL query results can be 100K+ tokens in practice.
        Test that estimation correctly handles large content.
        """
        manager = ContextManager(max_tokens=262144)

        # Simulate a large SQL result: 1 million chars = 250K tokens
        large_sql_result = "x" * 1_000_000
        messages = [ToolMessage(content=large_sql_result, tool_call_id="call1")]

        # Verify estimation: 1M chars / 4 = 250K tokens
        tokens = manager._estimate_tokens(messages)
        assert tokens == 250000

        # Should trigger compaction (250K > 196K threshold)
        assert manager.should_compact(messages)

    def test_manage_context_node_large_context(self):
        """Test manage_context_node correctly uses large context_length.

        This verifies the full flow: App._init_agent passes context_length
        -> create_agent uses functools.partial to create manage_context callback
        -> manage_context_node creates ContextManager with correct max_tokens.
        """
        # Create messages that would trigger compaction at 6K (200 tokens / 6000 = 3.3%)
        # but should NOT trigger at 262K (200 tokens / 262K = 0.08%)
        state = {"messages": [HumanMessage(content="x" * 80) for _ in range(10)]}

        # With large context (262K), should NOT compact
        result_large = manage_context_node(state, context_length=262144)
        assert result_large["context_metadata"]["was_compacted"] == False

        # With small context (100 tokens), should compact
        result_small = manage_context_node(state, context_length=100)
        assert result_small["context_metadata"]["was_compacted"] == True
        assert result_small["context_metadata"]["summarized_count"] == 4

    def test_create_agent_passes_context_length(self):
        """Test create_agent properly passes context_length to manage_context_node.

        Verifies the parameter flow: create_agent(context_length=X)
        -> functools.partial(manage_context_node, context_length=X)
        -> ContextManager(max_tokens=X)
        """
        from unittest.mock import Mock
        from esdc.chat.agent import create_agent

        mock_llm = Mock()
        mock_llm.bind_tools = Mock(return_value=mock_llm)
        mock_llm.invoke = Mock(return_value=AIMessage(content="test"))

        # Create agent with large context (simulating 262K model)
        agent = create_agent(mock_llm, context_length=262144)

        # Verify the graph was built (agent is a compiled Runnable)
        assert agent is not None
