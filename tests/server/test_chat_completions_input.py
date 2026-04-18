"""Tests for Chat Completions API input handling with output arrays."""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from esdc.server.agent_wrapper import convert_messages_to_langchain


class TestSimpleMessages:
    """Tests for simple message formats."""

    def test_user_message(self) -> None:
        """Test simple user message."""
        messages = [{"role": "user", "content": "Hello"}]
        lc_messages = convert_messages_to_langchain(messages)
        assert len(lc_messages) == 1
        assert isinstance(lc_messages[0], HumanMessage)
        assert lc_messages[0].content == "Hello"

    def test_assistant_message(self) -> None:
        """Test simple assistant message."""
        messages = [{"role": "assistant", "content": "Hi there!"}]
        lc_messages = convert_messages_to_langchain(messages)
        assert len(lc_messages) == 1
        assert isinstance(lc_messages[0], AIMessage)
        assert lc_messages[0].content == "Hi there!"

    def test_system_message(self) -> None:
        """Test system message."""
        messages = [{"role": "system", "content": "You are helpful."}]
        lc_messages = convert_messages_to_langchain(messages)
        assert len(lc_messages) == 1
        assert isinstance(lc_messages[0], SystemMessage)
        assert lc_messages[0].content == "You are helpful."


class TestToolMessages:
    """Tests for tool messages."""

    def test_tool_message(self) -> None:
        """Test tool message."""
        messages = [{"role": "tool", "content": "Result", "tool_call_id": "call_123"}]
        lc_messages = convert_messages_to_langchain(messages)
        assert len(lc_messages) == 1
        assert isinstance(lc_messages[0], ToolMessage)
        assert lc_messages[0].content == "Result"
        assert lc_messages[0].tool_call_id == "call_123"


class TestOutputArrayWithoutTools:
    """Tests for output arrays without tool calls."""

    def test_assistant_with_output_message_only(self) -> None:
        """Test assistant message with output array containing only message."""
        messages = [
            {
                "role": "assistant",
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": "Hello there!"}],
                    }
                ],
            }
        ]
        lc_messages = convert_messages_to_langchain(messages)
        assert len(lc_messages) == 1
        assert isinstance(lc_messages[0], AIMessage)
        assert "Hello there!" in lc_messages[0].content


class TestOutputArrayWithToolCalls:
    """Tests for output arrays containing tool call history."""

    def test_assistant_with_function_call(self) -> None:
        """Test assistant output containing function_call."""
        messages = [
            {
                "role": "assistant",
                "output": [
                    {
                        "type": "function_call",
                        "call_id": "call_1",
                        "name": "get_data",
                        "arguments": '{"param": "value"}',
                    }
                ],
            }
        ]
        lc_messages = convert_messages_to_langchain(messages)
        assert len(lc_messages) == 1
        assert isinstance(lc_messages[0], AIMessage)
        assert len(lc_messages[0].tool_calls) == 1
        assert lc_messages[0].tool_calls[0]["name"] == "get_data"
        assert lc_messages[0].tool_calls[0]["id"] == "call_1"
        assert lc_messages[0].tool_calls[0]["args"] == {"param": "value"}

    def test_function_call_output(self) -> None:
        """Test function_call_output in output array."""
        messages = [
            {
                "role": "assistant",
                "output": [
                    {
                        "type": "function_call_output",
                        "call_id": "call_1",
                        "output": "Result data here",
                    }
                ],
            }
        ]
        lc_messages = convert_messages_to_langchain(messages)
        assert len(lc_messages) == 1
        assert isinstance(lc_messages[0], ToolMessage)
        assert lc_messages[0].content == "Result data here"
        assert lc_messages[0].tool_call_id == "call_1"

    def test_function_call_output_with_array(self) -> None:
        """Test function_call_output with array format."""
        messages = [
            {
                "role": "assistant",
                "output": [
                    {
                        "type": "function_call_output",
                        "call_id": "call_1",
                        "output": [{"type": "input_text", "text": "Result"}],
                    }
                ],
            }
        ]
        lc_messages = convert_messages_to_langchain(messages)
        assert len(lc_messages) == 1
        assert isinstance(lc_messages[0], ToolMessage)
        assert "Result" in lc_messages[0].content
        assert lc_messages[0].tool_call_id == "call_1"


class TestFullConversation:
    """Tests for full conversations with tool calls."""

    def test_full_conversation_with_tools(self) -> None:
        """Test complete conversation with tool calls in output arrays."""
        messages = [
            {"role": "user", "content": "What are reserves?"},
            {
                "role": "assistant",
                "output": [
                    {
                        "type": "function_call",
                        "call_id": "call_1",
                        "name": "query_reserves",
                        "arguments": '{"entity_type": "national"}',
                    }
                ],
            },
            {
                "role": "assistant",
                "output": [
                    {
                        "type": "function_call_output",
                        "call_id": "call_1",
                        "output": [{"type": "input_text", "text": "Reserves data..."}],
                    }
                ],
            },
            {
                "role": "assistant",
                "output": [
                    {
                        "type": "message",
                        "content": [
                            {"type": "output_text", "text": "Here are the reserves..."}
                        ],
                    }
                ],
            },
            {"role": "user", "content": "Tell me more"},
        ]

        lc_messages = convert_messages_to_langchain(messages)
        # User message + AIMessage(tool_call) + ToolMessage + AIMessage(response) + User message  # noqa: E501
        assert len(lc_messages) == 5
        assert isinstance(lc_messages[0], HumanMessage)
        assert isinstance(lc_messages[1], AIMessage)
        assert len(lc_messages[1].tool_calls) == 1
        assert isinstance(lc_messages[2], ToolMessage)
        assert isinstance(lc_messages[3], AIMessage)
        assert isinstance(lc_messages[4], HumanMessage)

    def test_multiple_tool_calls_in_sequence(self) -> None:
        """Test multiple tool calls in sequence."""
        messages = [
            {"role": "user", "content": "Compare reserves"},
            {
                "role": "assistant",
                "output": [
                    {
                        "type": "function_call",
                        "call_id": "call_1",
                        "name": "query_reserves",
                        "arguments": '{"filter": "national"}',
                    }
                ],
            },
            {
                "role": "assistant",
                "output": [
                    {
                        "type": "function_call_output",
                        "call_id": "call_1",
                        "output": "National data...",
                    }
                ],
            },
            {
                "role": "assistant",
                "output": [
                    {
                        "type": "function_call",
                        "call_id": "call_2",
                        "name": "query_reserves",
                        "arguments": '{"filter": "state"}',
                    }
                ],
            },
            {
                "role": "assistant",
                "output": [
                    {
                        "type": "function_call_output",
                        "call_id": "call_2",
                        "output": "State data...",
                    }
                ],
            },
        ]

        lc_messages = convert_messages_to_langchain(messages)
        assert len(lc_messages) == 5

        assert isinstance(lc_messages[0], HumanMessage)
        assert isinstance(lc_messages[1], AIMessage)
        assert lc_messages[1].tool_calls[0]["id"] == "call_1"
        assert isinstance(lc_messages[2], ToolMessage)
        assert lc_messages[2].tool_call_id == "call_1"
        assert isinstance(lc_messages[3], AIMessage)
        assert lc_messages[3].tool_calls[0]["id"] == "call_2"
        assert isinstance(lc_messages[4], ToolMessage)
        assert lc_messages[4].tool_call_id == "call_2"


class TestMessageRoleInOutput:
    """Tests for message type with different roles in output array."""

    def test_user_message_in_output(self) -> None:
        """Test user message inside output array."""
        messages = [
            {
                "role": "assistant",
                "output": [{"type": "message", "role": "user", "content": "Override"}],
            }
        ]
        lc_messages = convert_messages_to_langchain(messages)
        assert len(lc_messages) == 1
        assert isinstance(lc_messages[0], HumanMessage)

    def test_system_message_in_output(self) -> None:
        """Test system message inside output array."""
        messages = [
            {
                "role": "assistant",
                "output": [
                    {"type": "message", "role": "system", "content": "System msg"}
                ],
            }
        ]
        lc_messages = convert_messages_to_langchain(messages)
        assert len(lc_messages) == 1
        assert isinstance(lc_messages[0], SystemMessage)


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_output(self) -> None:
        """Test empty output array returns empty AIMessage."""
        # When output is empty list, falls through to standard assistant handling
        messages = [{"role": "assistant", "output": []}]
        lc_messages = convert_messages_to_langchain(messages)
        # Empty output is falsy, so falls through to standard assistant message
        assert len(lc_messages) == 1
        assert isinstance(lc_messages[0], AIMessage)
        assert lc_messages[0].content == ""

    def test_mixed_content_parts(self) -> None:
        """Test message with mixed content parts."""
        messages = [
            {
                "role": "assistant",
                "output": [
                    {
                        "type": "message",
                        "content": [
                            "Plain string",
                            {"type": "output_text", "text": "Dict text"},
                        ],
                    }
                ],
            }
        ]
        lc_messages = convert_messages_to_langchain(messages)
        assert len(lc_messages) == 1
        assert "Plain string" in lc_messages[0].content
        assert "Dict text" in lc_messages[0].content

    def test_invalid_json_arguments(self) -> None:
        """Test function_call with invalid JSON arguments."""
        messages = [
            {
                "role": "assistant",
                "output": [
                    {
                        "type": "function_call",
                        "call_id": "call_1",
                        "name": "test",
                        "arguments": "not valid json",
                    }
                ],
            }
        ]
        lc_messages = convert_messages_to_langchain(messages)
        assert len(lc_messages) == 1
        assert lc_messages[0].tool_calls[0]["args"] == {}

    def test_missing_call_id_in_function_call(self) -> None:
        """Test function_call with missing call_id defaults to empty."""
        messages = [
            {
                "role": "assistant",
                "output": [
                    {
                        "type": "function_call",
                        "name": "get_data",
                        "arguments": "{}",
                    }
                ],
            }
        ]
        lc_messages = convert_messages_to_langchain(messages)
        assert len(lc_messages) == 1
        assert lc_messages[0].tool_calls[0]["id"] == ""

    def test_non_dict_item_in_output(self) -> None:
        """Test output array with non-dict item is skipped."""
        messages = [
            {
                "role": "assistant",
                "output": [
                    "string_item",
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": "Valid"}],
                    },
                ],
            }
        ]
        lc_messages = convert_messages_to_langchain(messages)
        assert len(lc_messages) == 1
        assert "Valid" in lc_messages[0].content


class TestPydanticModelInput:
    """Tests for Pydantic model inputs."""

    def test_pydantic_user_message(self) -> None:
        """Test user message as Pydantic model."""
        from esdc.server.models import Message

        messages = [Message(role="user", content="Hello from Pydantic")]
        lc_messages = convert_messages_to_langchain(messages)
        assert len(lc_messages) == 1
        assert isinstance(lc_messages[0], HumanMessage)
        assert lc_messages[0].content == "Hello from Pydantic"

    def test_pydantic_assistant_message(self) -> None:
        """Test assistant message as Pydantic model."""
        from esdc.server.models import Message

        messages = [Message(role="assistant", content="Response")]
        lc_messages = convert_messages_to_langchain(messages)
        assert len(lc_messages) == 1
        assert isinstance(lc_messages[0], AIMessage)
        assert lc_messages[0].content == "Response"
