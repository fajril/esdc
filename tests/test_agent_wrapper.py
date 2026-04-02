"""Tests for agent wrapper module."""

# Standard library
import json
from unittest.mock import AsyncMock, MagicMock, patch

# Third-party
import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

# Local
from esdc.server.agent_wrapper import (
    convert_messages_to_langchain,
    create_openai_chunk,
    extract_thinking_for_interleaved,
)


class TestConvertMessages:
    """Test suite for message conversion."""

    def test_convert_system_message(self):
        """Test converting system message."""
        from esdc.server.models import Message

        messages = [Message(role="system", content="You are a helpful assistant")]
        result = convert_messages_to_langchain(messages)

        assert len(result) == 1
        assert isinstance(result[0], SystemMessage)
        assert result[0].content == "You are a helpful assistant"

    def test_convert_user_message(self):
        """Test converting user message."""
        from esdc.server.models import Message

        messages = [Message(role="user", content="Hello")]
        result = convert_messages_to_langchain(messages)

        assert len(result) == 1
        assert isinstance(result[0], HumanMessage)
        assert result[0].content == "Hello"

    def test_convert_assistant_message(self):
        """Test converting assistant message."""
        from esdc.server.models import Message

        messages = [Message(role="assistant", content="Hi there")]
        result = convert_messages_to_langchain(messages)

        assert len(result) == 1
        assert isinstance(result[0], AIMessage)
        assert result[0].content == "Hi there"

    def test_convert_tool_message(self):
        """Test converting tool message."""
        from esdc.server.models import Message

        messages = [Message(role="tool", content="Tool result", tool_call_id="123")]
        result = convert_messages_to_langchain(messages)

        assert len(result) == 1
        assert isinstance(result[0], ToolMessage)
        assert result[0].content == "Tool result"

    def test_convert_empty_content(self):
        """Test converting message with empty content."""
        from esdc.server.models import Message

        messages = [Message(role="user", content=None)]
        result = convert_messages_to_langchain(messages)

        assert len(result) == 1
        assert result[0].content == ""

    def test_convert_multiple_messages(self):
        """Test converting multiple messages."""
        from esdc.server.models import Message

        messages = [
            Message(role="system", content="System prompt"),
            Message(role="user", content="User message"),
            Message(role="assistant", content="Assistant response"),
        ]
        result = convert_messages_to_langchain(messages)

        assert len(result) == 3
        assert isinstance(result[0], SystemMessage)
        assert isinstance(result[1], HumanMessage)
        assert isinstance(result[2], AIMessage)


class TestCreateOpenAIChunk:
    """Test suite for creating OpenAI chunks."""

    def test_create_chunk_with_content(self):
        """Test creating chunk with content."""
        chunk = create_openai_chunk(content="Hello", model="test-model")

        assert chunk["object"] == "chat.completion.chunk"
        assert chunk["model"] == "test-model"
        assert len(chunk["choices"]) == 1
        assert chunk["choices"][0]["delta"]["content"] == "Hello"
        assert chunk["choices"][0]["finish_reason"] is None

    def test_create_chunk_with_finish_reason(self):
        """Test creating chunk with finish reason."""
        chunk = create_openai_chunk(
            content="", model="test-model", finish_reason="stop"
        )

        assert chunk["choices"][0]["delta"]["content"] == ""
        assert chunk["choices"][0]["finish_reason"] == "stop"

    def test_create_chunk_defaults(self):
        """Test creating chunk with default values."""
        chunk = create_openai_chunk()

        assert chunk["object"] == "chat.completion.chunk"
        assert chunk["model"] == "esdc-agent"
        assert chunk["choices"][0]["delta"]["content"] == ""
        assert chunk["choices"][0]["finish_reason"] is None

    def test_chunk_has_required_fields(self):
        """Test chunk has all required OpenAI fields."""
        chunk = create_openai_chunk(content="Test")

        assert "id" in chunk
        assert "object" in chunk
        assert "created" in chunk
        assert "model" in chunk
        assert "choices" in chunk
        assert isinstance(chunk["created"], int)


class TestAgentWrapperIntegration:
    """Integration tests for agent wrapper."""

    @pytest.mark.asyncio
    async def test_generate_response_mock(self):
        """Test generate_response with mocked agent."""
        from esdc.server.agent_wrapper import generate_response

        mock_event = {"messages": [MagicMock(spec=AIMessage, content="Test response")]}

        with patch(
            "esdc.server.agent_wrapper.Config.get_provider_config"
        ) as mock_config:
            with patch("esdc.server.agent_wrapper.create_llm_from_config") as mock_llm:
                with patch("esdc.server.agent_wrapper.create_agent") as mock_agent:
                    mock_config.return_value = {
                        "provider": "ollama",
                        "model": "test-model",
                    }
                    mock_llm.return_value = MagicMock()
                    mock_agent.return_value = MagicMock()
                    mock_agent.return_value.astream = AsyncMock(
                        return_value=[mock_event]
                    )

                    result = await generate_response(
                        messages=[MagicMock(role="user", content="Hello")],
                        model="esdc-agent",
                    )

                    assert result["role"] == "assistant"
                    assert result["finish_reason"] == "stop"

    @pytest.mark.asyncio
    async def test_generate_streaming_response_mock(self):
        """Test generate_streaming_response with mocked agent."""
        from esdc.server.agent_wrapper import generate_streaming_response

        mock_event = {"messages": [MagicMock(spec=AIMessage, content="Test")]}

        with patch(
            "esdc.server.agent_wrapper.Config.get_provider_config"
        ) as mock_config:
            with patch("esdc.server.agent_wrapper.create_llm_from_config") as mock_llm:
                with patch("esdc.server.agent_wrapper.create_agent") as mock_agent:
                    mock_config.return_value = {
                        "provider": "ollama",
                        "model": "test-model",
                    }
                    mock_llm.return_value = MagicMock()
                    mock_agent.return_value = MagicMock()
                    mock_agent.return_value.astream = AsyncMock(
                        return_value=[mock_event]
                    )

                    chunks = []
                    async for chunk in generate_streaming_response(
                        messages=[MagicMock(role="user", content="Hello")],
                        model="esdc-agent",
                    ):
                        chunks.append(json.loads(chunk))

                    assert len(chunks) > 0
                    assert all("choices" in c for c in chunks)


class TestExtractThinkingForInterleaved:
    """Test suite for extract_thinking_for_interleaved function."""

    def test_extract_thinking_from_reasoning_content(self):
        """Test extracting thinking from reasoning_content in additional_kwargs."""
        ai_msg = AIMessage(
            content="Final response",
            additional_kwargs={"reasoning_content": "This is my thinking process"},
            tool_calls=[
                {"id": "1", "name": "test_tool", "args": {}, "type": "tool_call"}
            ],
        )

        result = extract_thinking_for_interleaved(ai_msg)

        assert result == "This is my thinking process"

    def test_extract_thinking_from_thinking_tags(self):
        """Test extracting thinking from thinking tags in content."""
        ai_msg = AIMessage(
            content="<thinking>Let me analyze this</thinking>Final answer",
            tool_calls=[
                {"id": "1", "name": "test_tool", "args": {}, "type": "tool_call"}
            ],
        )

        result = extract_thinking_for_interleaved(ai_msg)

        assert result == "Let me analyze this"

    def test_no_extraction_without_tool_calls(self):
        """Test that no extraction happens without tool_calls."""
        ai_msg = AIMessage(
            content="<thinking>Some thinking</thinking>Final answer",
            additional_kwargs={"reasoning_content": "Reasoning content"},
            tool_calls=[],
        )

        result = extract_thinking_for_interleaved(ai_msg)

        assert result is None

    def test_no_extraction_without_tool_calls_attribute(self):
        """Test that no extraction happens without tool_calls attribute."""
        ai_msg = AIMessage(
            content="<thinking>Some thinking</thinking>Final answer",
            additional_kwargs={"reasoning_content": "Reasoning content"},
        )
        delattr(ai_msg, "tool_calls")

        result = extract_thinking_for_interleaved(ai_msg)

        assert result is None

    def test_no_thinking_content_returns_none(self):
        """Test that None is returned when no thinking content exists."""
        ai_msg = AIMessage(
            content="Just a regular response",
            tool_calls=[
                {"id": "1", "name": "test_tool", "args": {}, "type": "tool_call"}
            ],
        )

        result = extract_thinking_for_interleaved(ai_msg)

        assert result is None

    def test_reasoning_content_priority_over_thinking_tags(self):
        """Test that reasoning_content takes priority over thinking tags."""
        ai_msg = AIMessage(
            content="<thinking>Tagged thinking</thinking>Final",
            additional_kwargs={"reasoning_content": "Reasoning from kwargs"},
            tool_calls=[
                {"id": "1", "name": "test_tool", "args": {}, "type": "tool_call"}
            ],
        )

        result = extract_thinking_for_interleaved(ai_msg)

        assert result == "Reasoning from kwargs"

    def test_extract_thinking_strips_whitespace(self):
        """Test that extracted thinking is stripped of whitespace."""
        ai_msg = AIMessage(
            content="",
            additional_kwargs={"reasoning_content": "  Thinking with spaces  "},
            tool_calls=[
                {"id": "1", "name": "test_tool", "args": {}, "type": "tool_call"}
            ],
        )

        result = extract_thinking_for_interleaved(ai_msg)

        assert result == "Thinking with spaces"

    @pytest.mark.asyncio
    async def test_generate_response_no_provider(self):
        """Test generate_response when no provider configured."""
        from esdc.server.agent_wrapper import generate_response

        with patch(
            "esdc.server.agent_wrapper.Config.get_provider_config"
        ) as mock_config:
            mock_config.return_value = None

            result = await generate_response(
                messages=[MagicMock(role="user", content="Hello")],
            )

            assert "Error" in result["content"]
            assert result["finish_reason"] == "stop"

    @pytest.mark.asyncio
    async def test_generate_streaming_response_no_provider(self):
        """Test generate_streaming_response when no provider configured."""
        from esdc.server.agent_wrapper import generate_streaming_response

        with patch(
            "esdc.server.agent_wrapper.Config.get_provider_config"
        ) as mock_config:
            mock_config.return_value = None

            chunks = []
            async for chunk in generate_streaming_response(
                messages=[MagicMock(role="user", content="Hello")],
            ):
                chunks.append(json.loads(chunk))

            assert len(chunks) == 1
            assert "Error" in chunks[0]["choices"][0]["delta"]["content"]
