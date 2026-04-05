"""Tests for ContentAccumulator class."""

from esdc.server.content_accumulator import ContentAccumulator


class TestContentAccumulator:
    """Test cases for ContentAccumulator."""

    def test_add_tool_call_returns_true(self):
        """Test that add_tool_call returns True (always flush)."""
        buffer = ContentAccumulator()
        result = buffer.add_tool_call("execute_sql", {"query": "SELECT 1"})
        assert result is True

    def test_add_tool_call_stores_in_buffer(self):
        """Test that add_tool_call stores tool call in buffer."""
        buffer = ContentAccumulator()
        buffer.add_tool_call("execute_sql", {"query": "SELECT 1"})
        assert len(buffer.tool_calls_buffer) == 1
        assert buffer.tool_calls_buffer[0]["name"] == "execute_sql"
        assert buffer.tool_calls_buffer[0]["args"] == {"query": "SELECT 1"}

    def test_add_content_returns_bool(self):
        """Test that add_content returns boolean."""
        buffer = ContentAccumulator()
        result = buffer.add_content("Some text")
        assert isinstance(result, bool)

    def test_add_content_stores_in_buffer(self):
        """Test that add_content stores content in buffer."""
        buffer = ContentAccumulator()
        buffer.add_content("Hello")
        assert len(buffer.content_buffer) == 1
        assert buffer.content_buffer[0] == "Hello"

    def test_add_content_triggers_flush_when_exceeds_limit(self):
        """Test that add_content triggers flush when exceeds size limit."""
        buffer = ContentAccumulator(buffer_size_limit=10)
        result = buffer.add_content("This is more than 10 characters")
        assert result is True

    def test_flush_returns_tool_sections(self):
        """Test that flush returns formatted tool sections."""
        buffer = ContentAccumulator()
        buffer.add_tool_call("execute_sql", {"query": "SELECT 1"})
        result = buffer.flush()
        assert "execute_sql" in result
        assert "SELECT 1" in result

    def test_flush_returns_content(self):
        """Test that flush returns content."""
        buffer = ContentAccumulator()
        buffer.add_content("Hello world")
        result = buffer.flush()
        assert result == "Hello world"

    def test_flush_returns_tools_then_content(self):
        """Test that flush returns tools then content in order."""
        buffer = ContentAccumulator()
        buffer.add_tool_call("execute_sql", {"query": "SELECT 1"})
        buffer.add_content("Final answer")
        result = buffer.flush()
        # Tool section should appear before content
        tool_idx = result.find("execute_sql")
        content_idx = result.find("Final answer")
        assert tool_idx < content_idx

    def test_flush_clears_buffers(self):
        """Test that flush clears buffers."""
        buffer = ContentAccumulator()
        buffer.add_tool_call("test", {})
        buffer.add_content("text")
        buffer.flush()
        assert len(buffer.tool_calls_buffer) == 0
        assert len(buffer.content_buffer) == 0

    def test_flush_empty_buffer_returns_empty_string(self):
        """Test that flush on empty buffer returns empty string."""
        buffer = ContentAccumulator()
        result = buffer.flush()
        assert result == ""

    def test_flush_final_calls_flush(self):
        """Test that flush_final calls flush."""
        buffer = ContentAccumulator()
        buffer.add_content("test")
        result = buffer.flush_final()
        assert result == "test"

    def test_has_content_returns_false_when_empty(self):
        """Test that has_content returns False when empty."""
        buffer = ContentAccumulator()
        assert buffer.has_content() is False

    def test_has_content_returns_true_with_tool_calls(self):
        """Test that has_content returns True with tool calls."""
        buffer = ContentAccumulator()
        buffer.add_tool_call("test", {})
        assert buffer.has_content() is True

    def test_has_content_returns_true_with_content(self):
        """Test that has_content returns True with content."""
        buffer = ContentAccumulator()
        buffer.add_content("test")
        assert buffer.has_content() is True
