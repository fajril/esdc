"""Tests for native tool calling functionality."""

from esdc.server.tool_formatter import (
    create_tool_call_chunk,
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


class TestRoutes:
    def test_chat_completions_endpoint_accepts_request_obj(self):
        # Just verify the endpoint signature works
        import inspect

        from esdc.server.routes import chat_completions

        sig = inspect.signature(chat_completions)
        params = list(sig.parameters.keys())
        assert "request_obj" in params


# Run with: python -m pytest tests/test_native_tool_calling.py -v
