"""Tests for the shared event streamer module."""

from unittest.mock import MagicMock

import pytest

from esdc.server.event_streamer import astream_agent_events


class AsyncEventIterator:
    """Helper class to make a list of events iterable as an async iterator."""

    def __init__(self, events):
        self._events = iter(events)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._events)
        except StopIteration:
            raise StopAsyncIteration


def make_chunk(content, reasoning_content=None):
    """Create a mock AIMessageChunk with content and optional reasoning_content."""
    chunk = MagicMock()
    chunk.content = content
    if reasoning_content is not None:
        chunk.reasoning_content = reasoning_content
    else:
        del chunk.reasoning_content
    return chunk


class TestAstreamAgentEvents:
    """Tests for astream_agent_events async generator."""

    @pytest.mark.asyncio
    async def test_token_events(self):
        """Test that token events are yielded correctly."""
        events = [
            {
                "event": "on_chat_model_stream",
                "data": {"chunk": make_chunk("Hello")},
                "name": "chat_model",
            },
        ]

        mock_agent = MagicMock()
        mock_agent.astream_events = MagicMock(return_value=AsyncEventIterator(events))

        result = []
        async for event in astream_agent_events(mock_agent, []):
            result.append(event)

        token_events = [e for e in result if e["type"] == "token"]
        assert len(token_events) == 1
        assert token_events[0]["content"] == "Hello"

    @pytest.mark.asyncio
    async def test_message_complete_event(self):
        """Test that message_complete events are yielded correctly."""
        from langchain_core.messages import AIMessage

        ai_message = AIMessage(content="Test response")

        events = [
            {
                "event": "on_chat_model_end",
                "data": {"output": ai_message},
                "name": "chat_model",
            },
        ]

        mock_agent = MagicMock()
        mock_agent.astream_events = MagicMock(return_value=AsyncEventIterator(events))

        result = []
        async for event in astream_agent_events(mock_agent, []):
            result.append(event)

        msg_events = [e for e in result if e["type"] == "message_complete"]
        assert len(msg_events) == 1
        assert msg_events[0]["ai_message"].content == "Test response"

    @pytest.mark.asyncio
    async def test_message_complete_content_is_unfiltered(self):
        """Test that message_complete content is NOT SQL-filtered (streamer is passthrough)."""
        from langchain_core.messages import AIMessage

        content = "Here:\n```sql\nSELECT 1;\n```\nDone."
        ai_message = AIMessage(content=content)

        events = [
            {
                "event": "on_chat_model_end",
                "data": {"output": ai_message},
                "name": "chat_model",
            },
        ]

        mock_agent = MagicMock()
        mock_agent.astream_events = MagicMock(return_value=AsyncEventIterator(events))

        result = []
        async for event in astream_agent_events(mock_agent, []):
            result.append(event)

        msg_events = [e for e in result if e["type"] == "message_complete"]
        assert len(msg_events) == 1
        assert "SELECT" in msg_events[0]["ai_message"].content

    @pytest.mark.asyncio
    async def test_tool_call_id_correlation(self):
        """Test that tool_call_id is correlated from message_complete to tool_result."""
        from langchain_core.messages import AIMessage

        ai_message = AIMessage(content="")
        ai_message.tool_calls = [
            {"name": "execute_sql", "args": {"query": "SELECT 1"}, "id": "call_abc"},
        ]

        events = [
            {
                "event": "on_chat_model_end",
                "data": {"output": ai_message},
                "name": "chat_model",
            },
            {
                "event": "on_tool_end",
                "data": {"output": "result data"},
                "name": "execute_sql",
            },
        ]

        mock_agent = MagicMock()
        mock_agent.astream_events = MagicMock(return_value=AsyncEventIterator(events))

        result = []
        async for event in astream_agent_events(mock_agent, []):
            result.append(event)

        tool_result_events = [e for e in result if e["type"] == "tool_result"]
        assert len(tool_result_events) == 1
        assert tool_result_events[0]["tool_call_id"] == "call_abc"
        assert tool_result_events[0]["tool_name"] == "execute_sql"

    @pytest.mark.asyncio
    async def test_tool_call_id_multiple_tools(self):
        """Test tool_call_id correlation with multiple tool calls in sequence."""
        from langchain_core.messages import AIMessage

        ai_message = AIMessage(content="")
        ai_message.tool_calls = [
            {"name": "list_tables", "args": {}, "id": "call_first"},
            {"name": "execute_sql", "args": {"query": "SELECT 1"}, "id": "call_second"},
        ]

        events = [
            {
                "event": "on_chat_model_end",
                "data": {"output": ai_message},
                "name": "chat_model",
            },
            {
                "event": "on_tool_end",
                "data": {"output": "table list"},
                "name": "list_tables",
            },
            {
                "event": "on_tool_end",
                "data": {"output": "query result"},
                "name": "execute_sql",
            },
        ]

        mock_agent = MagicMock()
        mock_agent.astream_events = MagicMock(return_value=AsyncEventIterator(events))

        result = []
        async for event in astream_agent_events(mock_agent, []):
            result.append(event)

        tool_result_events = [e for e in result if e["type"] == "tool_result"]
        assert len(tool_result_events) == 2
        assert tool_result_events[0]["tool_call_id"] == "call_first"
        assert tool_result_events[1]["tool_call_id"] == "call_second"

    @pytest.mark.asyncio
    async def test_tool_result_without_pending_call(self):
        """Test that tool_result with no pending tool_call gets empty call_id + warning."""
        events = [
            {
                "event": "on_tool_end",
                "data": {"output": "orphan result"},
                "name": "execute_sql",
            },
        ]

        mock_agent = MagicMock()
        mock_agent.astream_events = MagicMock(return_value=AsyncEventIterator(events))

        result = []
        async for event in astream_agent_events(mock_agent, []):
            result.append(event)

        tool_result_events = [e for e in result if e["type"] == "tool_result"]
        assert len(tool_result_events) == 1
        assert tool_result_events[0]["tool_call_id"] == ""

    @pytest.mark.asyncio
    async def test_reasoning_content_token(self):
        """Test that reasoning_content field yields reasoning_token events."""
        events = [
            {
                "event": "on_chat_model_stream",
                "data": {"chunk": make_chunk("", reasoning_content="Let me think...")},
                "name": "chat_model",
            },
        ]

        mock_agent = MagicMock()
        mock_agent.astream_events = MagicMock(return_value=AsyncEventIterator(events))

        result = []
        async for event in astream_agent_events(mock_agent, []):
            result.append(event)

        reasoning_events = [e for e in result if e["type"] == "reasoning_token"]
        assert len(reasoning_events) == 1
        assert reasoning_events[0]["content"] == "Let me think..."

    @pytest.mark.asyncio
    async def test_empty_chunk_skipped(self):
        """Test that empty content chunks produce no token events."""
        events = [
            {
                "event": "on_chat_model_stream",
                "data": {"chunk": make_chunk("")},
                "name": "chat_model",
            },
        ]

        mock_agent = MagicMock()
        mock_agent.astream_events = MagicMock(return_value=AsyncEventIterator(events))

        result = []
        async for event in astream_agent_events(mock_agent, []):
            result.append(event)

        token_events = [e for e in result if e["type"] == "token"]
        assert len(token_events) == 0

    @pytest.mark.asyncio
    async def test_on_llm_stream_event(self):
        """Test that on_llm_stream events are also handled."""
        events = [
            {
                "event": "on_llm_stream",
                "data": {"chunk": make_chunk("World")},
                "name": "ChatOllama",
            },
        ]

        mock_agent = MagicMock()
        mock_agent.astream_events = MagicMock(return_value=AsyncEventIterator(events))

        result = []
        async for event in astream_agent_events(mock_agent, []):
            result.append(event)

        token_events = [e for e in result if e["type"] == "token"]
        assert len(token_events) == 1
        assert token_events[0]["content"] == "World"

    @pytest.mark.asyncio
    async def test_multi_token_streaming(self):
        """Test streaming multiple tokens in sequence."""
        events = [
            {
                "event": "on_chat_model_stream",
                "data": {"chunk": make_chunk("Hello")},
                "name": "chat_model",
            },
            {
                "event": "on_chat_model_stream",
                "data": {"chunk": make_chunk(" world")},
                "name": "chat_model",
            },
            {
                "event": "on_chat_model_stream",
                "data": {"chunk": make_chunk("!")},
                "name": "chat_model",
            },
        ]

        mock_agent = MagicMock()
        mock_agent.astream_events = MagicMock(return_value=AsyncEventIterator(events))

        result = []
        async for event in astream_agent_events(mock_agent, []):
            result.append(event)

        token_events = [e for e in result if e["type"] == "token"]
        assert len(token_events) == 3
        assert token_events[0]["content"] == "Hello"
        assert token_events[1]["content"] == " world"
        assert token_events[2]["content"] == "!"

    @pytest.mark.asyncio
    async def test_tool_calls_on_message_complete(self):
        """Test that tool_calls are preserved on message_complete ai_message."""
        from langchain_core.messages import AIMessage

        ai_message = AIMessage(content="")
        ai_message.tool_calls = [
            {"name": "execute_sql", "args": {"query": "SELECT 1"}, "id": "call_1"},
        ]

        events = [
            {
                "event": "on_chat_model_end",
                "data": {"output": ai_message},
                "name": "chat_model",
            },
        ]

        mock_agent = MagicMock()
        mock_agent.astream_events = MagicMock(return_value=AsyncEventIterator(events))

        result = []
        async for event in astream_agent_events(mock_agent, []):
            result.append(event)

        msg_events = [e for e in result if e["type"] == "message_complete"]
        assert len(msg_events) == 1
        assert len(msg_events[0]["ai_message"].tool_calls) == 1
        assert msg_events[0]["ai_message"].tool_calls[0]["name"] == "execute_sql"

    @pytest.mark.asyncio
    async def test_usage_metadata_preserved(self):
        """Test that usage_metadata is preserved on message_complete."""
        from langchain_core.messages import AIMessage

        ai_message = AIMessage(content="test")
        ai_message.usage_metadata = {
            "input_tokens": 100,
            "output_tokens": 50,
            "total_tokens": 150,
        }

        events = [
            {
                "event": "on_chat_model_end",
                "data": {"output": ai_message},
                "name": "chat_model",
            },
        ]

        mock_agent = MagicMock()
        mock_agent.astream_events = MagicMock(return_value=AsyncEventIterator(events))

        result = []
        async for event in astream_agent_events(mock_agent, []):
            result.append(event)

        msg_events = [e for e in result if e["type"] == "message_complete"]
        assert len(msg_events) == 1
        assert msg_events[0]["ai_message"].usage_metadata["input_tokens"] == 100
