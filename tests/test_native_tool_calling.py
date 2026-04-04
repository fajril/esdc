"""Tests for native tool calling functionality."""

from esdc.server.tool_formatter import (
    create_tool_call_chunk,
    create_tool_result_chunk,
    detect_native_format,
    format_tool_calls_for_response,
)


class TestToolFormatter:
    def test_create_tool_call_chunk_single(self):
        tool_calls = [
            {"name": "execute_sql", "args": {"query": "SELECT 1"}, "id": "call_123"}
        ]
        chunk = create_tool_call_chunk(tool_calls)

        assert chunk["object"] == "chat.completion.chunk"
        assert len(chunk["choices"]) == 1
        assert "tool_calls" in chunk["choices"][0]["delta"]

    def test_format_tool_calls_for_response(self):
        tool_calls = [
            {"name": "get_schema", "args": {"table": "users"}, "id": "call_1"}
        ]
        formatted = format_tool_calls_for_response(tool_calls)

        assert len(formatted) == 1
        assert formatted[0]["type"] == "function"
        assert formatted[0]["function"]["name"] == "get_schema"

    def test_detect_native_format_sse(self):
        headers = {"accept": "text/event-stream"}
        assert detect_native_format(headers, stream=True) is True

    def test_detect_native_format_json(self):
        headers = {"accept": "application/json"}
        assert detect_native_format(headers, stream=False) is True

    def test_detect_native_format_fallback(self):
        headers = {"accept": "text/plain"}
        assert detect_native_format(headers, stream=False) is False


class TestToolResultChunk:
    """Test suite for tool result chunk creation."""

    def test_create_tool_result_chunk_basic(self):
        """Test creating basic tool result chunk."""
        chunk = create_tool_result_chunk(
            tool_call_id="call_123",
            content="Query returned 5 rows",
            model="esdc-agent",
        )

        assert chunk["object"] == "chat.completion.chunk"
        assert chunk["model"] == "esdc-agent"
        assert len(chunk["choices"]) == 1

        delta = chunk["choices"][0]["delta"]
        assert delta["role"] == "tool"
        assert delta["tool_call_id"] == "call_123"
        assert delta["content"] == "Query returned 5 rows"
        assert chunk["choices"][0]["finish_reason"] is None

    def test_create_tool_result_chunk_with_model_default(self):
        """Test tool result chunk with default model."""
        chunk = create_tool_result_chunk(
            tool_call_id="call_abc",
            content="Success",
        )

        assert chunk["model"] == "esdc-agent"
        assert chunk["choices"][0]["delta"]["tool_call_id"] == "call_abc"

    def test_create_tool_result_chunk_has_required_fields(self):
        """Test chunk has all required OpenAI fields."""
        chunk = create_tool_result_chunk(
            tool_call_id="call_xyz",
            content="Result",
        )

        assert "id" in chunk
        assert "object" in chunk
        assert "created" in chunk
        assert "model" in chunk
        assert "choices" in chunk
        assert isinstance(chunk["created"], int)
        assert chunk["object"] == "chat.completion.chunk"


class TestRoutes:
    def test_chat_completions_endpoint_accepts_request_obj(self):
        # Just verify the endpoint signature works
        import inspect

        from esdc.server.routes import chat_completions

        sig = inspect.signature(chat_completions)
        params = list(sig.parameters.keys())
        assert "request_obj" in params


# Run with: python -m pytest tests/test_native_tool_calling.py -v
