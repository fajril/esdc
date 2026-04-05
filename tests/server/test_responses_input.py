"""Tests for Responses API input handling with discriminated unions."""

from esdc.server.responses_models import (
    ResponseInputFunctionCallItem,
    ResponseInputFunctionCallOutputItem,
    ResponseInputItem,
    ResponseInputMessageItem,
)
from esdc.server.responses_wrapper import convert_responses_input_to_langchain


class TestSimpleInput:
    """Tests for simple string input."""

    def test_string_input(self) -> None:
        """Test simple string input conversion."""
        messages = convert_responses_input_to_langchain("Hello")
        assert len(messages) == 1
        assert messages[0].content == "Hello"


class TestMessageItems:
    """Tests for message input items."""

    def test_user_message_with_string_content(self) -> None:
        """Test user message with string content."""
        item = ResponseInputItem(type="message", role="user", content="Hello world")
        messages = convert_responses_input_to_langchain([item])
        assert len(messages) == 1
        assert messages[0].content == "Hello world"

    def test_user_message_with_list_content(self) -> None:
        """Test user message with list of content parts."""
        item = ResponseInputItem(
            type="message",
            role="user",
            content=[{"type": "input_text", "text": "Hello"}],
        )
        messages = convert_responses_input_to_langchain([item])
        assert len(messages) == 1
        assert "Hello" in messages[0].content

    def test_user_message_with_mixed_content(self) -> None:
        """Test user message with mixed string and dict content parts."""
        item = ResponseInputItem(
            type="message",
            role="user",
            content=["Hello", {"type": "text", "text": "World"}],
        )
        messages = convert_responses_input_to_langchain([item])
        assert len(messages) == 1
        assert "Hello" in messages[0].content
        assert "World" in messages[0].content

    def test_assistant_message(self) -> None:
        """Test assistant message."""
        item = ResponseInputItem(
            type="message",
            role="assistant",
            content="I can help with that.",
        )
        messages = convert_responses_input_to_langchain([item])
        assert len(messages) == 1
        assert messages[0].content == "I can help with that."

    def test_system_message(self) -> None:
        """Test system message with instructions."""
        item = ResponseInputItem(
            type="message",
            role="system",
            content="You are a helpful assistant.",
        )
        messages = convert_responses_input_to_langchain([item])
        assert len(messages) == 1


class TestFunctionCallItems:
    """Tests for function_call input items."""

    def test_function_call_item_skipped(self) -> None:
        """Test that function_call items are skipped gracefully."""
        item = ResponseInputItem(
            type="function_call",
            id="fc_123",
            name="get_data",
            arguments='{"param": "value"}',
        )
        messages = convert_responses_input_to_langchain([item])
        assert len(messages) == 0

    def test_new_function_call_item_model(self) -> None:
        """Test new ResponseInputFunctionCallItem model."""
        item = ResponseInputFunctionCallItem(
            type="function_call",
            id="fc_123",
            name="query_reserves",
            arguments='{"entity_type": "national"}',
        )
        messages = convert_responses_input_to_langchain([item])
        assert len(messages) == 0


class TestFunctionCallOutputItems:
    """Tests for function_call_output input items."""

    def test_function_call_output_with_string(self) -> None:
        """Test function_call_output with string output."""
        item = ResponseInputItem(
            type="function_call_output",
            call_id="call_123",
            output="Result data here",
        )
        messages = convert_responses_input_to_langchain([item])
        assert len(messages) == 1
        assert messages[0].content == "Result data here"

    def test_function_call_output_with_array(self) -> None:
        """Test function_call_output with array format."""
        item = ResponseInputItem(
            type="function_call_output",
            call_id="call_123",
            output=[{"type": "input_text", "text": "Result"}],
        )
        messages = convert_responses_input_to_langchain([item])
        assert len(messages) == 1
        assert messages[0].content == "Result"

    def test_new_function_call_output_item_model(self) -> None:
        """Test new ResponseInputFunctionCallOutputItem model."""
        item = ResponseInputFunctionCallOutputItem(
            type="function_call_output",
            call_id="call_123",
            output=[{"type": "input_text", "text": "Result"}],
        )
        messages = convert_responses_input_to_langchain([item])
        assert len(messages) == 1
        # Output may be a FlexibleContentPart object or dict
        assert "Result" in str(messages[0].content)


class TestRealOpenWebUIConversation:
    """Tests with actual OpenWebUI conversation format."""

    def test_full_conversation_with_tool_calls(self) -> None:
        """Test full conversation from OpenWebUI with tool calls."""
        conversation = [
            {
                "type": "message",
                "role": "user",
                "content": "What are reserves?",
            },
            {
                "type": "function_call",
                "id": "fc_1",
                "name": "query_reserves",
                "arguments": '{"entity_type": "national"}',
            },
            {
                "type": "function_call_output",
                "call_id": "fc_1",
                "output": [{"type": "input_text", "text": "Data here"}],
            },
            {
                "type": "message",
                "role": "assistant",
                "content": [
                    {"type": "output_text", "text": "Here are the reserves..."}
                ],
            },
            {
                "type": "message",
                "role": "user",
                "content": "Can you tell me more?",
            },
        ]

        messages = convert_responses_input_to_langchain(conversation)
        assert len(messages) == 4

    def test_conversation_with_system_instructions(self) -> None:
        """Test conversation with system instructions."""
        messages = convert_responses_input_to_langchain(
            [{"type": "message", "role": "user", "content": "Hello"}],
            instructions="You are a helpful assistant.",
        )
        assert len(messages) == 2
        assert messages[0].content == "You are a helpful assistant."
        assert messages[1].content == "Hello"


class TestBackwardCompatibility:
    """Tests for backward compatibility."""

    def test_old_response_input_item_message(self) -> None:
        """Test that old ResponseInputItem syntax still works for messages."""
        item = ResponseInputItem(type="message", role="user", content="Test")
        assert item.type == "message"
        assert item.role == "user"
        assert item.content == "Test"

    def test_old_response_input_item_function_call_output(self) -> None:
        """Test old ResponseInputItem syntax for function_call_output."""
        item = ResponseInputItem(
            type="function_call_output",
            call_id="call_123",
            output="result",
        )
        assert item.type == "function_call_output"
        assert item.call_id == "call_123"
        assert item.output == "result"

    def test_old_response_input_item_with_function_call(self) -> None:
        """Test that function_call type is now accepted."""
        item = ResponseInputItem(
            type="function_call",
            id="fc_123",
            name="get_data",
            arguments="{}",
        )
        assert item.type == "function_call"
        assert item.name == "get_data"
