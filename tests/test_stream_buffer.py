"""Tests for StreamingBuffer class."""

import pytest

from esdc.server.stream_buffer import StreamingBuffer, format_thinking_section


class TestStreamingBuffer:
    """Test cases for StreamingBuffer."""

    def test_add_preserved_thinking_stores_content(self):
        """Test that add_preserved_thinking stores content correctly."""
        buffer = StreamingBuffer()
        buffer.add_preserved_thinking("Previous thinking content")
        assert buffer.preserved_thinking == "Previous thinking content"

    def test_add_preserved_thinking_overwrites_existing(self):
        """Test that add_preserved_thinking overwrites existing preserved thinking."""
        buffer = StreamingBuffer()
        buffer.add_preserved_thinking("Old content")
        buffer.add_preserved_thinking("New content")
        assert buffer.preserved_thinking == "New content"

    def test_flush_includes_preserved_thinking(self):
        """Test that flush includes preserved thinking with proper tags."""
        buffer = StreamingBuffer()
        buffer.add_preserved_thinking("Previous reasoning")
        result = buffer.flush()
        expected = format_thinking_section("Previous reasoning")
        assert result == expected

    def test_flush_clears_preserved_thinking(self):
        """Test that preserved thinking is cleared after flush."""
        buffer = StreamingBuffer()
        buffer.add_preserved_thinking("Previous reasoning")
        buffer.flush()
        assert buffer.preserved_thinking is None

    def test_combination_preserved_and_current_thinking(self):
        """Test that preserved thinking comes before current thinking."""
        buffer = StreamingBuffer()
        buffer.add_preserved_thinking("Previous reasoning")
        buffer.add_thinking("Current reasoning")
        result = buffer.flush()
        # Preserved thinking should come first, then current thinking
        expected_parts = [
            format_thinking_section("Previous reasoning"),
            format_thinking_section("Current reasoning"),
        ]
        expected = "\n".join(expected_parts)
        assert result == expected

    def test_only_preserved_thinking_no_other_content(self):
        """Test buffer with only preserved thinking flushes correctly."""
        buffer = StreamingBuffer()
        buffer.add_preserved_thinking("Only preserved thinking")
        result = buffer.flush()
        expected = format_thinking_section("Only preserved thinking")
        assert result == expected
        assert buffer.preserved_thinking is None

    def test_full_flush_order(self):
        """Test the full order: preserved thinking, current thinking, tool calls, content."""
        buffer = StreamingBuffer()
        buffer.add_preserved_thinking("Preserved from previous")
        buffer.add_thinking("Current thinking")
        buffer.add_tool_call("execute_sql", {"query": "SELECT * FROM users"})
        buffer.add_content("Final answer")
        result = buffer.flush()

        # Check order in result
        lines = result.split("\n")
        # First should be preserved thinking
        assert "Preserved from previous" in result
        assert "Current thinking" in result
        assert "execute_sql" in result
        assert "Final answer" in result

        # Verify preserved thinking comes before current thinking
        preserved_idx = result.find("Preserved from previous")
        current_idx = result.find("Current thinking")
        assert preserved_idx < current_idx

    def test_flush_empty_buffer_with_preserved_thinking(self):
        """Test flush with only preserved thinking clears it properly."""
        buffer = StreamingBuffer()
        buffer.add_preserved_thinking("Only preserved")
        result = buffer.flush()

        assert result == format_thinking_section("Only preserved")
        assert buffer.preserved_thinking is None
        assert buffer.thinking_buffer == []
        assert buffer.tool_calls_buffer == []
        assert buffer.content_buffer == []

    def test_multiple_flushes_with_preserved_thinking(self):
        """Test that preserved thinking only appears in first flush."""
        buffer = StreamingBuffer()
        buffer.add_preserved_thinking("Preserved")
        buffer.add_thinking("Current")

        # First flush
        result1 = buffer.flush()
        assert "Preserved" in result1
        assert "Current" in result1

        # Second flush - no preserved thinking
        buffer.add_thinking("More thinking")
        result2 = buffer.flush()
        assert "Preserved" not in result2
        assert "More thinking" in result2

    def test_has_content_includes_preserved_thinking(self):
        """Test that has_content returns True when only preserved thinking exists."""
        buffer = StreamingBuffer()
        assert not buffer.has_content()
        buffer.add_preserved_thinking("Preserved")
        assert buffer.has_content()

    def test_preserved_thinking_with_tool_calls(self):
        """Test preserved thinking followed by tool calls."""
        buffer = StreamingBuffer()
        buffer.add_preserved_thinking("Previous reasoning")
        buffer.add_tool_call("execute_sql", {"query": "SELECT 1"})
        result = buffer.flush()

        # Preserved thinking should come before tool call
        preserved_idx = result.find("Previous reasoning")
        tool_idx = result.find("execute_sql")
        assert preserved_idx < tool_idx
        assert buffer.preserved_thinking is None

    def test_preserved_thinking_with_content_only(self):
        """Test preserved thinking followed by content only."""
        buffer = StreamingBuffer()
        buffer.add_preserved_thinking("Previous reasoning")
        buffer.add_content("Final response")
        result = buffer.flush()

        # Preserved thinking should come before content
        preserved_idx = result.find("Previous reasoning")
        content_idx = result.find("Final response")
        assert preserved_idx < content_idx
        assert buffer.preserved_thinking is None
