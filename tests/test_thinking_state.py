"""Tests for ThinkingState class."""

# Third-party
import pytest
from langchain_core.messages import AIMessage

# Local
from esdc.server.thinking_state import ThinkingState


class TestThinkingState:
    """Test suite for ThinkingState class."""

    def test_empty_state_on_init(self):
        """Test that ThinkingState initializes empty."""
        state = ThinkingState()

        assert state.has_thinking() is False
        assert state.peek_thinking() is None
        assert state.get_thinking() is None

    def test_preserve_and_retrieve_thinking(self):
        """Test preserving and retrieving thinking content."""
        state = ThinkingState()
        thinking_content = "This is some thinking content"

        state.preserve_thinking(thinking_content)

        assert state.has_thinking() is True
        assert state.peek_thinking() == thinking_content
        assert state.get_thinking() == thinking_content
        assert state.has_thinking() is False  # Cleared after get

    def test_multiple_thinking_parts_concatenation(self):
        """Test that multiple thinking parts are concatenated with '\n\n'."""
        state = ThinkingState()
        part1 = "First thinking part"
        part2 = "Second thinking part"
        part3 = "Third thinking part"

        state.preserve_thinking(part1)
        state.preserve_thinking(part2)
        state.preserve_thinking(part3)

        expected = "First thinking part\n\nSecond thinking part\n\nThird thinking part"
        assert state.get_thinking() == expected

    def test_clear_behavior(self):
        """Test clearing all thinking content."""
        state = ThinkingState()
        state.preserve_thinking("Some thinking")

        assert state.has_thinking() is True

        state.clear()

        assert state.has_thinking() is False
        assert state.peek_thinking() is None
        assert state.get_thinking() is None

    def test_preserve_empty_string(self):
        """Test that empty strings are not preserved."""
        state = ThinkingState()

        state.preserve_thinking("")
        assert state.has_thinking() is False

        state.preserve_thinking("   ")
        assert state.has_thinking() is False

        state.preserve_thinking(None)  # type: ignore
        assert state.has_thinking() is False

    def test_extract_from_aimessage_with_tool_calls_and_reasoning_content(self):
        """Test extraction from AIMessage with tool_calls and reasoning_content."""
        state = ThinkingState()
        message = AIMessage(
            content="Final response",
            tool_calls=[{"name": "test_tool", "args": {}, "id": "call_1"}],
            additional_kwargs={"reasoning_content": "This is reasoning content"},
        )

        result = state.extract_and_preserve(message)

        assert result is True
        assert state.get_thinking() == "This is reasoning content"

    def test_extract_from_aimessage_with_tool_calls_and_thinking_tags(self):
        """Test extraction from AIMessage with tool_calls and thinking tags."""
        state = ThinkingState()
        message = AIMessage(
            content="<thinking>This is thinking content</thinking>Final response",
            tool_calls=[{"name": "test_tool", "args": {}, "id": "call_1"}],
        )

        result = state.extract_and_preserve(message)

        assert result is True
        assert state.get_thinking() == "This is thinking content"

    def test_no_extraction_without_tool_calls(self):
        """Test that no extraction happens without tool_calls."""
        state = ThinkingState()
        message = AIMessage(
            content="<thinking>This is thinking content</thinking>Final response",
            # No tool_calls
        )

        result = state.extract_and_preserve(message)

        assert result is False
        assert state.has_thinking() is False

    def test_no_extraction_with_empty_tool_calls(self):
        """Test that no extraction happens with empty tool_calls."""
        state = ThinkingState()
        message = AIMessage(
            content="<thinking>This is thinking content</thinking>Final response",
            tool_calls=[],  # Empty list
        )

        result = state.extract_and_preserve(message)

        assert result is False
        assert state.has_thinking() is False

    def test_reasoning_content_takes_precedence_over_thinking_tags(self):
        """Test that reasoning_content in additional_kwargs is preferred over tags."""
        state = ThinkingState()
        message = AIMessage(
            content="<thinking>Tagged thinking</thinking>Final response",
            tool_calls=[{"name": "test_tool", "args": {}, "id": "call_1"}],
            additional_kwargs={"reasoning_content": "Explicit reasoning content"},
        )

        result = state.extract_and_preserve(message)

        assert result is True
        assert state.get_thinking() == "Explicit reasoning content"

    def test_peek_does_not_clear(self):
        """Test that peek_thinking doesn't clear the thinking."""
        state = ThinkingState()
        state.preserve_thinking("Some thinking")

        # Peek multiple times
        assert state.peek_thinking() == "Some thinking"
        assert state.peek_thinking() == "Some thinking"
        assert state.peek_thinking() == "Some thinking"

        # Still there after peeks
        assert state.has_thinking() is True
        assert state.get_thinking() == "Some thinking"
        assert state.has_thinking() is False

    def test_multiple_extractions_concatenate(self):
        """Test that multiple extractions are concatenated."""
        state = ThinkingState()
        message1 = AIMessage(
            content="Final response 1",
            tool_calls=[{"name": "test_tool", "args": {}, "id": "call_1"}],
            additional_kwargs={"reasoning_content": "First reasoning"},
        )
        message2 = AIMessage(
            content="Final response 2",
            tool_calls=[{"name": "test_tool", "args": {}, "id": "call_2"}],
            additional_kwargs={"reasoning_content": "Second reasoning"},
        )

        state.extract_and_preserve(message1)
        state.extract_and_preserve(message2)

        expected = "First reasoning\n\nSecond reasoning"
        assert state.get_thinking() == expected

    def test_strip_whitespace_on_preserve(self):
        """Test that whitespace is stripped when preserving thinking."""
        state = ThinkingState()

        state.preserve_thinking("  Thinking with whitespace  ")
        assert state.peek_thinking() == "Thinking with whitespace"

    def test_extract_from_aimessage_with_multiple_thinking_tags(self):
        """Test extraction when multiple thinking tags are present."""
        state = ThinkingState()
        message = AIMessage(
            content="<thinking>First part</thinking>middle<thinking>Second part</thinking>final",
            tool_calls=[{"name": "test_tool", "args": {}, "id": "call_1"}],
        )

        result = state.extract_and_preserve(message)

        assert result is True
        expected = "First part\n\nSecond part"
        assert state.get_thinking() == expected

    def test_extract_returns_false_for_no_thinking_found(self):
        """Test that extract_and_preserve returns False when no thinking found."""
        state = ThinkingState()
        message = AIMessage(
            content="Just a regular response with no thinking",
            tool_calls=[{"name": "test_tool", "args": {}, "id": "call_1"}],
        )

        result = state.extract_and_preserve(message)

        assert result is False
        assert state.has_thinking() is False
