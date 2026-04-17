"""Test that content is emitted before tool_calls in streaming response.

This test verifies the fix for the ordering issue where tool_calls were
being emitted before content in the web API, causing OpenWebUI to display
tools before the thinking/reasoning text.
"""

import json
from unittest.mock import MagicMock, patch

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
        # Create a message with both content (thinking) and tool_calls
        ai_msg = AIMessage(
            content="I need to analyze the database. Let me query the data.",
            tool_calls=[
                {
                    "name": "execute_sql",
                    "args": {"query": "SELECT * FROM table"},
                    "id": "test-123",
                }
            ],
        )

        # Track emitted chunks
        emitted_chunks = []

        messages = [Message(role="user", content="test query")]

        # Mock Config
        with patch("esdc.server.agent_wrapper.Config") as mock_config:
            mock_config.get_provider_config.return_value = {
                "provider_type": "ollama",
                "model": "test-model",
                "base_url": "http://localhost:11434",
            }

            # Mock agent that yields the event
            async def mock_astream(*args, **kwargs):
                event = {"agent": {"messages": [ai_msg]}}
                yield event

            mock_agent = MagicMock()
            mock_agent.astream = mock_astream

            with patch("esdc.server.agent_wrapper.create_llm_from_config"):
                with patch(
                    "esdc.server.agent_wrapper.create_agent", return_value=mock_agent
                ):
                    # Collect chunks
                    async for chunk in generate_streaming_response(
                        messages=messages, use_native_format=True
                    ):
                        emitted_chunks.append(json.loads(chunk))

        # Verify order: content should come BEFORE tool_calls
        content_found = False
        tool_calls_found = False
        content_index = -1
        tool_calls_index = -1

        for i, chunk in enumerate(emitted_chunks):
            delta = chunk.get("choices", [{}])[0].get("delta", {})

            if "content" in delta and delta["content"]:
                content_found = True
                content_index = i

            if "tool_calls" in delta:
                tool_calls_found = True
                tool_calls_index = i

        # Debug output
        print(f"Total chunks: {len(emitted_chunks)}")
        for i, chunk in enumerate(emitted_chunks):
            print(
                f"Chunk {i}: {list(chunk.get('choices', [{}])[0].get('delta', {}).keys())}"  # noqa: E501
            )

        # Assertions
        assert content_found, "Content should be emitted"
        assert tool_calls_found, "Tool calls should be emitted"
        assert content_index < tool_calls_index, (
            f"Content (index {content_index}) should come before tool_calls (index {tool_calls_index})"  # noqa: E501
        )

    @pytest.mark.asyncio
    async def test_content_is_streamed_as_plain_text(self):
        """Content is streamed as-is without think tags (thinking logic removed)."""
        ai_msg = AIMessage(
            content="Let me think about this query...",
            tool_calls=[{"name": "list_tables", "args": {}, "id": "test-456"}],
        )
        emitted_chunks = []
        messages = [Message(role="user", content="test")]

        with patch("esdc.server.agent_wrapper.Config") as mock_config:
            mock_config.get_provider_config.return_value = {
                "provider_type": "ollama",
                "model": "test-model",
                "base_url": "http://localhost:11434",
            }

            async def mock_astream(*args, **kwargs):
                event = {"agent": {"messages": [ai_msg]}}
                yield event

            mock_agent = MagicMock()
            mock_agent.astream = mock_astream

            with patch("esdc.server.agent_wrapper.create_llm_from_config"):
                with patch(
                    "esdc.server.agent_wrapper.create_agent", return_value=mock_agent
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
        assert "laissez" not in content, "Content should not have think tags (removed)"
        assert "liziez" not in content, "Content should not have think tags (removed)"

    @pytest.mark.asyncio
    async def test_no_duplicate_thinking(self):
        """Verify that thinking content is not duplicated."""
        ai_msg = AIMessage(
            content="I will help you with that.",
            tool_calls=[
                {"name": "execute_sql", "args": {"query": "SELECT 1"}, "id": "test-789"}
            ],
        )

        emitted_chunks = []

        messages = [Message(role="user", content="test")]

        with patch("esdc.server.agent_wrapper.Config") as mock_config:
            mock_config.get_provider_config.return_value = {
                "provider_type": "ollama",
                "model": "test-model",
                "base_url": "http://localhost:11434",
            }

            async def mock_astream(*args, **kwargs):
                event = {"agent": {"messages": [ai_msg]}}
                yield event

            mock_agent = MagicMock()
            mock_agent.astream = mock_astream

            with patch("esdc.server.agent_wrapper.create_llm_from_config"):
                with patch(
                    "esdc.server.agent_wrapper.create_agent", return_value=mock_agent
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
            f"Thinking content should appear at most once, but found {thinking_count} times"  # noqa: E501
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
