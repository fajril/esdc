"""Tests for tool role message handling in Chat Completions streaming."""

# Standard library
import json
from unittest.mock import MagicMock, patch

# Third-party
import pytest
from langchain_core.messages import AIMessage

# Local
from esdc.server.tool_formatter import create_tool_role_chunk


class TestCreateToolRoleChunk:
    """Test suite for create_tool_role_chunk function."""

    def test_create_tool_role_chunk_basic(self):
        """Test creating basic tool role chunk."""
        chunk = create_tool_role_chunk(
            tool_call_id="call_123",
            content="Tool executed successfully",
            model="test-model",
        )

        assert chunk["object"] == "chat.completion.chunk"
        assert chunk["model"] == "test-model"
        assert len(chunk["choices"]) == 1

        delta = chunk["choices"][0]["delta"]
        assert delta["role"] == "tool"
        assert delta["tool_call_id"] == "call_123"
        assert delta["content"] == "Tool executed successfully"
        assert chunk["choices"][0]["finish_reason"] is None

    def test_create_tool_role_chunk_with_metadata(self):
        """Test creating tool role chunk with execution metadata."""
        chunk = create_tool_role_chunk(
            tool_call_id="call_456",
            content="Query returned 42 rows",
            model="iris",
        )

        delta = chunk["choices"][0]["delta"]
        assert delta["role"] == "tool"
        assert delta["tool_call_id"] == "call_456"
        assert "42 rows" in delta["content"]

    def test_create_tool_role_chunk_content_truncation(self):
        """Test that large content is truncated."""
        large_content = "x" * 2000  # 2000 chars
        chunk = create_tool_role_chunk(
            tool_call_id="call_789", content=large_content, model="test-model"
        )

        delta = chunk["choices"][0]["delta"]
        # Should be truncated to 1000 chars + "... [truncated]"
        assert len(delta["content"]) <= 1020  # 1000 + truncation message

    def test_create_tool_role_chunk_short_content(self):
        """Test that short content is not truncated."""
        short_content = "Result: OK"
        chunk = create_tool_role_chunk(
            tool_call_id="call_abc", content=short_content, model="test-model"
        )

        delta = chunk["choices"][0]["delta"]
        assert delta["content"] == "Result: OK"
        assert "[truncated]" not in delta["content"]

    def test_create_tool_role_chunk_has_required_fields(self):
        """Test that chunk has all required OpenAI fields."""
        chunk = create_tool_role_chunk(tool_call_id="call_def", content="Test content")

        assert "id" in chunk
        assert "object" in chunk
        assert "created" in chunk
        assert "model" in chunk
        assert "choices" in chunk
        assert isinstance(chunk["created"], int)
        assert chunk["id"].startswith("chatcmpl-")

    def test_create_tool_role_chunk_empty_content(self):
        """Test handling empty tool result."""
        chunk = create_tool_role_chunk(
            tool_call_id="call_empty", content="", model="test-model"
        )

        delta = chunk["choices"][0]["delta"]
        assert delta["role"] == "tool"
        assert delta["content"] == ""

    def test_create_tool_role_chunk_special_chars(self):
        """Test handling special characters in content."""
        chunk = create_tool_role_chunk(
            tool_call_id="call_special",
            content='Result with "quotes" and \\backs\\ash',
            model="test-model",
        )

        # Should not raise JSON encoding errors
        json_str = json.dumps(chunk)
        parsed = json.loads(json_str)
        assert (
            parsed["choices"][0]["delta"]["content"]
            == 'Result with "quotes" and \\backs\\ash'
        )


class TestToolRoleMessagesInStreaming:
    """Test suite for tool role messages in streaming response."""

    @pytest.mark.asyncio
    async def test_streaming_includes_tool_role_after_tool_calls(self):
        """Test that streaming response includes tool role messages after tool calls."""
        from esdc.server.agent_wrapper import generate_streaming_response

        # Mock event sequence: AI message with tool_calls -> tool result -> AI message with content  # noqa: E501
        mock_events = [
            # Event 1: AI message with tool_calls
            {
                "agent": {
                    "messages": [
                        AIMessage(
                            content="I'll help you",
                            tool_calls=[
                                {"name": "get_table", "args": {}, "id": "call_1"}
                            ],
                        )
                    ]
                }
            },
            # Event 2: Tool result
            {
                "tools": {
                    "messages": [
                        {
                            "role": "tool",
                            "tool_call_id": "call_1",
                            "content": "Table: reserves",
                        }
                    ]
                }
            },
            # Event 3: AI message with final content
            {"agent": {"messages": [AIMessage(content="Here's the answer")]}},
        ]

        # Mock agent astrem
        async def mock_astream(*args, **kwargs):
            for event in mock_events:
                yield event

        mock_agent = MagicMock()
        mock_agent.astream = mock_astream

        # Mock dependencies
        with (
            patch(
                "esdc.server.agent_wrapper.Config.get_provider_config"
            ) as mock_config,
            patch("esdc.server.agent_wrapper.create_llm_from_config"),
            patch("esdc.server.agent_wrapper.create_agent", return_value=mock_agent),
        ):
            mock_config.return_value = {"provider_type": "test", "model": "test-model"}

            chunks = []
            async for chunk in generate_streaming_response(
                messages=[{"role": "user", "content": "test"}], model="test-model"
            ):
                chunks.append(json.loads(chunk))

        # Verify sequence
        # Should have: content chunks -> tool_call chunk -> tool_role chunk -> content chunks -> final chunk  # noqa: E501
        assert len(chunks) > 0

        # Find tool_call chunk
        tool_call_chunks = [
            c for c in chunks if "tool_calls" in c["choices"][0]["delta"]
        ]
        assert len(tool_call_chunks) == 1, "Expected one tool_call chunk"

        # Find tool role chunks (role='tool')
        tool_role_chunks = [
            c for c in chunks if c["choices"][0]["delta"].get("role") == "tool"
        ]
        assert len(tool_role_chunks) >= 1, "Expected at least one tool role chunk"

        # Verify tool role chunk structure
        tool_role = tool_role_chunks[0]
        assert tool_role["choices"][0]["delta"]["role"] == "tool"
        assert "tool_call_id" in tool_role["choices"][0]["delta"]

    @pytest.mark.asyncio
    async def test_streaming_multiple_tool_calls_multiple_results(self):
        """Test streaming with multiple tool calls and results."""
        from esdc.server.agent_wrapper import generate_streaming_response

        mock_events = [
            # Event 1: AI message with 2 tool_calls
            {
                "agent": {
                    "messages": [
                        AIMessage(
                            content="",
                            tool_calls=[
                                {"name": "get_table", "args": {}, "id": "call_1"},
                                {"name": "execute_sql", "args": {}, "id": "call_2"},
                            ],
                        )
                    ]
                }
            },
            # Event 2: Tool result 1
            {
                "tools": {
                    "messages": [
                        {
                            "role": "tool",
                            "tool_call_id": "call_1",
                            "content": "Result 1",
                        }
                    ]
                }
            },
            # Event 3: Tool result 2
            {
                "tools": {
                    "messages": [
                        {
                            "role": "tool",
                            "tool_call_id": "call_2",
                            "content": "Result 2",
                        }
                    ]
                }
            },
            # Event 4: AI message with content
            {"agent": {"messages": [AIMessage(content="Final answer")]}},
        ]

        async def mock_astream(*args, **kwargs):
            for event in mock_events:
                yield event

        mock_agent = MagicMock()
        mock_agent.astream = mock_astream

        with (
            patch(
                "esdc.server.agent_wrapper.Config.get_provider_config"
            ) as mock_config,
            patch("esdc.server.agent_wrapper.create_llm_from_config"),
            patch("esdc.server.agent_wrapper.create_agent", return_value=mock_agent),
        ):
            mock_config.return_value = {"provider_type": "test", "model": "test-model"}

            chunks = []
            async for chunk in generate_streaming_response(
                messages=[{"role": "user", "content": "test"}], model="test-model"
            ):
                chunks.append(json.loads(chunk))

        # Should have 2 tool role chunks (one per tool result)
        tool_role_chunks = [
            c for c in chunks if c["choices"][0]["delta"].get("role") == "tool"
        ]
        assert len(tool_role_chunks) == 2, (
            f"Expected 2 tool role chunks, got {len(tool_role_chunks)}"
        )

        # Verify each has correct tool_call_id
        tool_call_ids = [
            c["choices"][0]["delta"]["tool_call_id"] for c in tool_role_chunks
        ]
        assert "call_1" in tool_call_ids
        assert "call_2" in tool_call_ids


class TestToolRoleMessagesInNonStreaming:
    """Test suite for tool role messages in non-streaming response."""

    @pytest.mark.asyncio
    async def test_non_streaming_includes_tool_messages_in_history(self):
        """Test that non-streaming response includes tool messages in conversation."""
        # This test verifies the behavior, actual implementation may differ
        # The key point is that OpenAI API expects:
        # messages = [
        #   {role: "user", content: "question"},
        #   {role: "assistant", content: "...", tool_calls: [...]},
        #   {role: "tool", tool_call_id: "...", content: "result"},
        #   {role: "assistant", content: "answer"}
        # ]
        pass  # Implementation needed
