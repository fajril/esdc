"""Tests for tool call limit enforcement."""

from esdc.chat.agent import MAX_TOOL_CALLS


class TestToolCallLimit:
    """Test that MAX_TOOL_CALLS is set correctly."""

    def test_max_tool_calls_is_6(self):
        """MAX_TOOL_CALLS should be 6 as defined in agent.py."""
        assert MAX_TOOL_CALLS == 6

    def test_agent_state_has_tool_call_count(self):
        """AgentState should include tool_call_count field."""
        from esdc.chat.context_manager import AgentState

        annotations = AgentState.__annotations__
        assert "tool_call_count" in annotations, (
            f"tool_call_count not in AgentState: {list(annotations.keys())}"
        )

    def test_agent_state_has_allowed_tools(self):
        """AgentState should include allowed_tools field."""
        from esdc.chat.context_manager import AgentState

        annotations = AgentState.__annotations__
        assert "allowed_tools" in annotations, (
            f"allowed_tools not in AgentState: {list(annotations.keys())}"
        )
