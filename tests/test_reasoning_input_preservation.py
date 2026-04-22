"""Tests for reasoning content preservation in multi-turn input."""


class TestReasoningContentPreservation:
    """Test reasoning_content preservation in convert_responses_input_to_langchain."""

    def test_function_call_item_with_reasoning_preserves_reasoning_content(self):
        """When a function_call input item has reasoning_content, it should be.

        Preserved in the reconstructed AIMessage.
        """
        from langchain_core.messages import AIMessage

        from esdc.server.responses_wrapper import convert_responses_input_to_langchain

        input_items = [
            {
                "type": "message",
                "role": "user",
                "content": "Berapa cadangan Duri?",
            },
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "execute_sql",
                "arguments": '{"query": "SELECT * FROM project_resources"}',
                "reasoning_content": (
                    "I need to query the project_resources table for Duri's reserves data."
                ),
            },
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": "Cadangan minyak Duri: 123 MMSTB",
            },
        ]

        messages = convert_responses_input_to_langchain(input_items)
        ai_messages = [m for m in messages if isinstance(m, AIMessage)]
        assert len(ai_messages) == 1
        assert ai_messages[0].additional_kwargs.get("reasoning_content") == (
            "I need to query the project_resources table for Duri's reserves data."
        )

    def test_function_call_item_without_reasoning_works_as_before(self):
        """Existing function_call items without reasoning_content should still work."""
        from langchain_core.messages import AIMessage

        from esdc.server.responses_wrapper import convert_responses_input_to_langchain

        input_items = [
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "execute_sql",
                "arguments": '{"query": "SELECT 1"}',
            },
        ]

        messages = convert_responses_input_to_langchain(input_items)
        ai_messages = [m for m in messages if isinstance(m, AIMessage)]
        assert len(ai_messages) == 1
        # reasoning_content should not be set (or None)
        assert not ai_messages[0].additional_kwargs.get("reasoning_content")

    def test_message_item_with_reasoning_in_content_parts(self):
        """When an assistant message input item has reasoning in content parts,

        the reasoning should be extracted and preserved.
        """
        from langchain_core.messages import AIMessage

        from esdc.server.responses_wrapper import convert_responses_input_to_langchain

        input_items = [
            {
                "type": "message",
                "role": "assistant",
                "content": "The reserves are 123 MMSTB.",
                "reasoning_content": "I analyzed the SQL results to compute the total.",
            },
        ]

        messages = convert_responses_input_to_langchain(input_items)
        ai_messages = [m for m in messages if isinstance(m, AIMessage)]
        assert len(ai_messages) == 1
        assert ai_messages[0].additional_kwargs.get("reasoning_content") == (
            "I analyzed the SQL results to compute the total."
        )

    def test_message_item_without_reasoning_works_as_before(self):
        """Assistant messages without reasoning_content should work normally."""
        from langchain_core.messages import AIMessage

        from esdc.server.responses_wrapper import convert_responses_input_to_langchain

        input_items = [
            {
                "type": "message",
                "role": "assistant",
                "content": "The reserves are 123 MMSTB.",
            },
        ]

        messages = convert_responses_input_to_langchain(input_items)
        ai_messages = [m for m in messages if isinstance(m, AIMessage)]
        assert len(ai_messages) == 1
        assert ai_messages[0].content == "The reserves are 123 MMSTB."
        # reasoning_content should not be set
        assert not ai_messages[0].additional_kwargs.get("reasoning_content")


class TestReasoningContentInComplexConversation:
    """Test reasoning preservation in complex multi-turn conversations."""

    def test_multiple_turns_with_mixed_reasoning(self):
        """Multi-turn conversation with some turns having reasoning."""
        from langchain_core.messages import AIMessage

        from esdc.server.responses_wrapper import convert_responses_input_to_langchain

        input_items = [
            {"type": "message", "role": "user", "content": "Berapa cadangan Duri?"},
            {
                "type": "function_call",
                "call_id": "fc1",
                "name": "execute_sql",
                "arguments": '{"query": "SELECT res_oc FROM t"}',
                "reasoning_content": "Looking for oil reserves specifically.",
            },
            {"type": "function_call_output", "call_id": "fc1", "output": "123 MMSTB"},
            {
                "type": "message",
                "role": "assistant",
                "content": "Cadangan minyak Duri: 123 MMSTB",
            },
            {"type": "message", "role": "user", "content": "Dan gas?"},
            {
                "type": "function_call",
                "call_id": "fc2",
                "name": "execute_sql",
                "arguments": '{"query": "SELECT res_an FROM t"}',
                # No reasoning_content here
            },
            {"type": "function_call_output", "call_id": "fc2", "output": "45 BSCF"},
            {
                "type": "message",
                "role": "assistant",
                "content": "Cadangan gas: 45 BSCF",
            },
        ]

        messages = convert_responses_input_to_langchain(input_items)
        ai_messages = [m for m in messages if isinstance(m, AIMessage)]

        # First AI message should have reasoning
        assert (
            ai_messages[0].additional_kwargs.get("reasoning_content")
            == "Looking for oil reserves specifically."
        )
        # Second AI message should not have reasoning
        assert not ai_messages[1].additional_kwargs.get("reasoning_content")

    def test_empty_reasoning_content_not_added(self):
        """Empty or None reasoning_content should not be added to AIMessage."""
        from langchain_core.messages import AIMessage

        from esdc.server.responses_wrapper import convert_responses_input_to_langchain

        input_items = [
            {
                "type": "message",
                "role": "assistant",
                "content": "Hello",
                "reasoning_content": "",  # Empty string
            },
        ]

        messages = convert_responses_input_to_langchain(input_items)
        ai_messages = [m for m in messages if isinstance(m, AIMessage)]
        assert len(ai_messages) == 1
        # Empty reasoning should not be added
        assert not ai_messages[0].additional_kwargs.get("reasoning_content")
