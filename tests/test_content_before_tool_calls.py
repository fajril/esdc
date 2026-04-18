"""Test that content is emitted before tool_calls in streaming response.

This test verifies the fix for the ordering issue where tool_calls were
being emitted before content in the web API, causing OpenWebUI to display
tools before the thinking/reasoning text.
"""

import json
from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage

from esdc.server.agent_wrapper import (
    generate_streaming_response,
)
from esdc.server.models import Message


class TestContentBeforeToolCalls:
    """Test that content always comes before tool_calls in streaming."""

    @pytest.mark.asyncio
    async def test_content_emitted_before_tool_calls(self):
        """Verify that when message has both content and tool_calls,
        content is emitted first, then tool_calls.
        """
        # Track emitted chunks
        emitted_chunks = []

        messages = [Message(role="user", content="test query")]

        # Mock the normalized event stream from astream_agent_events
        async def mock_events(*args, **kwargs):
            yield {"type": "token", "content": "I need to analyze the database."}
            yield {"type": "token", "content": " Let me query the data."}
            yield {
                "type": "message_complete",
                "ai_message": AIMessage(
                    content="I need to analyze the database. Let me query the data.",
                    tool_calls=[
                        {
                            "name": "execute_sql",
                            "args": {"query": "SELECT * FROM table"},
                            "id": "test-123",
                        }
                    ],
                ),
            }
            yield {
                "type": "tool_result",
                "tool_name": "execute_sql",
                "result": "query result",
                "tool_call_id": "test-123",
            }
            yield {"type": "token", "content": "Here is the answer."}

        # Mock Config
        with patch("esdc.server.agent_wrapper.Config") as mock_config:
            mock_config.get_provider_config.return_value = {
                "provider_type": "ollama",
                "model": "test-model",
                "base_url": "http://localhost:11434",
            }

            with patch("esdc.server.agent_wrapper.create_llm_from_config"):
                with patch("esdc.server.agent_wrapper.create_agent"):
                    with patch(
                        "esdc.server.agent_wrapper.astream_agent_events",
                        mock_events,
                    ):
                        # Collect chunks
                        async for chunk in generate_streaming_response(
                            messages=messages, use_native_format=True
                        ):
                            emitted_chunks.append(json.loads(chunk))

        # Verify order: content should come BEFORE tool_calls
        content_found = False
        tool_calls_found = False
        first_content_index = -1
        first_tool_calls_index = -1

        for i, chunk in enumerate(emitted_chunks):
            delta = chunk.get("choices", [{}])[0].get("delta", {})

            if "content" in delta and delta["content"]:
                if not content_found:
                    content_found = True
                    first_content_index = i

            if "tool_calls" in delta:
                if not tool_calls_found:
                    tool_calls_found = True
                    first_tool_calls_index = i

        # Assertions
        assert content_found, "Content should be emitted"
        assert tool_calls_found, "Tool calls should be emitted"
        assert first_content_index < first_tool_calls_index, (
            f"Content (index {first_content_index}) should come before"
            f" tool_calls (index {first_tool_calls_index})"
        )

    @pytest.mark.asyncio
    async def test_content_is_streamed_as_plain_text(self):
        """Content is streamed as-is without think tags (thinking logic removed)."""
        emitted_chunks = []
        messages = [Message(role="user", content="test")]

        async def mock_events(*args, **kwargs):
            yield {"type": "token", "content": "Let me think about this query..."}
            yield {
                "type": "message_complete",
                "ai_message": AIMessage(
                    content="Let me think about this query...",
                    tool_calls=[{"name": "list_tables", "args": {}, "id": "test-456"}],
                ),
            }

        with patch("esdc.server.agent_wrapper.Config") as mock_config:
            mock_config.get_provider_config.return_value = {
                "provider_type": "ollama",
                "model": "test-model",
                "base_url": "http://localhost:11434",
            }

            with patch("esdc.server.agent_wrapper.create_llm_from_config"):
                with patch("esdc.server.agent_wrapper.create_agent"):
                    with patch(
                        "esdc.server.agent_wrapper.astream_agent_events",
                        mock_events,
                    ):
                        async for chunk in generate_streaming_response(
                            messages=messages, use_native_format=True
                        ):
                            emitted_chunks.append(json.loads(chunk))

        content_chunks = [
            chunk
            for chunk in emitted_chunks
            if "content" in chunk.get("choices", [{}])[0].get("delta", {})
        ]

        assert len(content_chunks) > 0, "Should have content chunks"

        content = "".join(c["choices"][0]["delta"]["content"] for c in content_chunks)
        assert "Let me think" in content, (
            f"Content should contain original text. Got: {content[:100]}"
        )

    @pytest.mark.asyncio
    async def test_no_duplicate_thinking(self):
        """Verify that thinking content is not duplicated."""
        emitted_chunks = []

        messages = [Message(role="user", content="test")]

        async def mock_events(*args, **kwargs):
            yield {"type": "token", "content": "I will help you with that."}
            yield {
                "type": "message_complete",
                "ai_message": AIMessage(
                    content="I will help you with that.",
                    tool_calls=[
                        {
                            "name": "execute_sql",
                            "args": {"query": "SELECT 1"},
                            "id": "test-789",
                        }
                    ],
                ),
            }

        with patch("esdc.server.agent_wrapper.Config") as mock_config:
            mock_config.get_provider_config.return_value = {
                "provider_type": "ollama",
                "model": "test-model",
                "base_url": "http://localhost:11434",
            }

            with patch("esdc.server.agent_wrapper.create_llm_from_config"):
                with patch("esdc.server.agent_wrapper.create_agent"):
                    with patch(
                        "esdc.server.agent_wrapper.astream_agent_events",
                        mock_events,
                    ):
                        async for chunk in generate_streaming_response(
                            messages=messages, use_native_format=True
                        ):
                            emitted_chunks.append(json.loads(chunk))

        # Count how many times the thinking content appears
        thinking_count = 0
        for chunk in emitted_chunks:
            delta = chunk.get("choices", [{}])[0].get("delta", {})
            if "content" in delta and delta["content"]:
                if "I will help you with that" in delta["content"]:
                    thinking_count += 1

        # Thinking should appear only ONCE (not duplicated)
        assert thinking_count <= 1, (
            f"Thinking content should appear at most once,"
            f" but found {thinking_count} times"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
