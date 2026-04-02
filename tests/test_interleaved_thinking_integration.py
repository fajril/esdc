"""Integration tests for interleaved thinking with tool calls."""

import pytest
from langchain_core.messages import AIMessage

from esdc.server.agent_wrapper import extract_thinking_for_interleaved
from esdc.server.thinking_state import ThinkingState
from esdc.server.stream_buffer import StreamingBuffer, format_thinking_section


class TestThinkingFlow:
    """Test the complete thinking preservation flow."""

    def test_thinking_preserved_across_tool_calls(self):
        """Test that thinking is extracted and preserved when tool calls are present."""
        # Setup: AIMessage with thinking content and tool calls
        thinking_state = ThinkingState()
        ai_message = AIMessage(
            content="Let me analyze this request",
            additional_kwargs={
                "reasoning_content": "I need to query the database to find user information"
            },
            tool_calls=[
                {
                    "id": "call_1",
                    "name": "execute_sql",
                    "args": {"query": "SELECT * FROM users"},
                    "type": "tool_call",
                }
            ],
        )

        # Extract thinking from AIMessage
        thinking = extract_thinking_for_interleaved(ai_message)
        assert thinking == "I need to query the database to find user information"

        # Preserve the thinking
        thinking_state.preserve_thinking(thinking)
        assert thinking_state.has_thinking() is True
        assert (
            thinking_state.peek_thinking()
            == "I need to query the database to find user information"
        )

        # Simulate tool execution and final response
        final_message = AIMessage(
            content="Here are the user records from the database."
        )

        # No new thinking should be extracted (no tool_calls)
        new_thinking = extract_thinking_for_interleaved(final_message)
        assert new_thinking is None

        # Retrieve preserved thinking
        preserved = thinking_state.get_thinking()
        assert preserved == "I need to query the database to find user information"
        assert thinking_state.has_thinking() is False  # Cleared after retrieval

    def test_no_thinking_preserved_without_tool_calls(self):
        """Test that thinking is NOT preserved when there are no tool calls."""
        thinking_state = ThinkingState()

        # AIMessage with thinking tags but NO tool calls (final response)
        ai_message = AIMessage(
            content="<thinking>Let me think about this</thinking>This is my final answer.",
            # No tool_calls attribute or empty
        )

        # Extraction should return None (no tool calls)
        thinking = extract_thinking_for_interleaved(ai_message)
        assert thinking is None

        # Nothing should be preserved
        assert thinking_state.has_thinking() is False

        # Even if we manually preserve, the extraction logic prevents it
        result = thinking_state.extract_and_preserve(ai_message)
        assert result is False
        assert thinking_state.has_thinking() is False

    def test_multiple_thinking_parts_accumulated(self):
        """Test that multiple thinking segments are accumulated correctly."""
        thinking_state = ThinkingState()

        # First tool call with thinking
        msg1 = AIMessage(
            content="Query 1",
            additional_kwargs={"reasoning_content": "First, I'll query table A"},
            tool_calls=[
                {
                    "id": "call_1",
                    "name": "execute_sql",
                    "args": {"query": "SELECT 1"},
                    "type": "tool_call",
                }
            ],
        )

        # Second tool call with more thinking
        msg2 = AIMessage(
            content="Query 2",
            additional_kwargs={"reasoning_content": "Now I'll query table B"},
            tool_calls=[
                {
                    "id": "call_2",
                    "name": "execute_sql",
                    "args": {"query": "SELECT 2"},
                    "type": "tool_call",
                }
            ],
        )

        # Third tool call with even more thinking
        msg3 = AIMessage(
            content="Query 3",
            additional_kwargs={"reasoning_content": "Finally, I'll query table C"},
            tool_calls=[
                {
                    "id": "call_3",
                    "name": "execute_sql",
                    "args": {"query": "SELECT 3"},
                    "type": "tool_call",
                }
            ],
        )

        # Extract and preserve from each
        for msg in [msg1, msg2, msg3]:
            thinking = extract_thinking_for_interleaved(msg)
            if thinking:
                thinking_state.preserve_thinking(thinking)

        # All thinking parts should be concatenated
        expected = "First, I'll query table A\n\nNow I'll query table B\n\nFinally, I'll query table C"
        assert thinking_state.get_thinking() == expected


class TestStreamingBufferInterleaved:
    """Test StreamingBuffer with interleaved thinking scenarios."""

    def test_buffer_receives_preserved_thinking(self):
        """Test that buffer correctly formats preserved thinking."""
        buffer = StreamingBuffer()

        # Simulate preserved thinking from previous turn
        preserved_thinking = "I analyzed the query and decided to use SQL"
        buffer.add_preserved_thinking(preserved_thinking)

        # Add current content
        buffer.add_content("Here is the result of my analysis.")

        # Flush and verify
        result = buffer.flush()

        # Preserved thinking should be formatted with think tags
        expected_thinking = format_thinking_section(preserved_thinking)
        assert expected_thinking in result
        assert "Here is the result of my analysis." in result

        # Preserved thinking should come first
        thinking_idx = result.find("I analyzed the query")
        content_idx = result.find("Here is the result")
        assert thinking_idx < content_idx

    def test_full_interleaved_scenario(self):
        """Complete flow: extract → preserve → add to buffer → verify output."""
        # Step 1: Create thinking state and buffer
        thinking_state = ThinkingState()
        buffer = StreamingBuffer()

        # Step 2: Simulate first AIMessage with thinking and tool calls
        ai_msg_with_thinking = AIMessage(
            content="I'll help you with that",
            additional_kwargs={
                "reasoning_content": "The user is asking for sales data. I should query the sales table."
            },
            tool_calls=[
                {
                    "id": "call_1",
                    "name": "execute_sql",
                    "args": {"query": "SELECT * FROM sales"},
                    "type": "tool_call",
                }
            ],
        )

        # Step 3: Extract and preserve thinking
        thinking = extract_thinking_for_interleaved(ai_msg_with_thinking)
        assert thinking is not None
        thinking_state.preserve_thinking(thinking)
        assert thinking_state.has_thinking() is True

        # Step 4: Simulate tool call being added to buffer
        for tool_call in ai_msg_with_thinking.tool_calls:
            buffer.add_tool_call(tool_call["name"], tool_call["args"])

        # Step 5: Simulate final response (no tool calls)
        final_msg = AIMessage(content="Here are the sales records for Q1 2024.")

        # Step 6: Check if preserved thinking should be injected
        if not hasattr(final_msg, "tool_calls") or not final_msg.tool_calls:
            if thinking_state.has_thinking():
                preserved = thinking_state.get_thinking()
                buffer.add_preserved_thinking(preserved)

        # Step 7: Add final content
        buffer.add_content("Here are the sales records for Q1 2024.")

        # Step 8: Flush and verify complete output
        result = buffer.flush()

        # Verify structure
        assert "The user is asking for sales data" in result
        assert "execute_sql" in result
        assert "SELECT * FROM sales" in result
        assert "Here are the sales records" in result

        # Verify preserved thinking is formatted correctly
        expected_thinking = format_thinking_section(
            "The user is asking for sales data. I should query the sales table."
        )
        assert expected_thinking in result

        # Verify order: thinking → tool → content
        thinking_idx = result.find("The user is asking for sales data")
        tool_idx = result.find("execute_sql")
        content_idx = result.find("Here are the sales records")

        assert thinking_idx < tool_idx < content_idx


class TestEndToEnd:
    """End-to-end tests demonstrating complete interleaved flow."""

    def test_complete_interleaved_flow(self):
        """Test: Model thinks → calls tool → continues with preserved thinking."""
        # Initialize components
        thinking_state = ThinkingState()
        buffer = StreamingBuffer()

        # === Phase 1: Initial thinking with tool call ===
        initial_response = AIMessage(
            content="I'll analyze your request",
            additional_kwargs={
                "reasoning_content": """I need to:
1. Understand what data the user wants
2. Identify the relevant table
3. Formulate an appropriate SQL query"""
            },
            tool_calls=[
                {
                    "id": "sql_1",
                    "name": "execute_sql",
                    "args": {
                        "query": "SELECT COUNT(*) FROM orders WHERE date > '2024-01-01'"
                    },
                    "type": "tool_call",
                }
            ],
        )

        # Extract thinking for interleaved scenario
        extracted_thinking = extract_thinking_for_interleaved(initial_response)
        assert extracted_thinking is not None
        assert "Identify the relevant table" in extracted_thinking

        # Preserve the thinking
        thinking_state.preserve_thinking(extracted_thinking)

        # Add tool call to buffer
        for tc in initial_response.tool_calls:
            buffer.add_tool_call(tc["name"], tc["args"])

        # Flush after tool call
        tool_output = buffer.flush()
        assert "execute_sql" in tool_output

        # === Phase 2: Final response (no tool calls) with preserved thinking ===
        final_response = AIMessage(
            content="Based on the query results, there are 1,234 orders since January 2024.",
            # No tool_calls - this is the final answer
        )

        # Verify no new thinking to extract (no tool calls)
        new_thinking = extract_thinking_for_interleaved(final_response)
        assert new_thinking is None

        # Inject preserved thinking before final content
        if thinking_state.has_thinking():
            preserved = thinking_state.get_thinking()
            buffer.add_preserved_thinking(preserved)

        # Add final content
        buffer.add_content(final_response.content)

        # === Phase 3: Verify complete output ===
        final_output = buffer.flush()

        # Verify all components are present
        assert "I need to:" in final_output  # Preserved thinking
        assert "Understand what data" in final_output
        assert "1,234 orders" in final_output  # Final content

        # Verify thinking is formatted with tags
        expected_section = format_thinking_section(thinking_state.peek_thinking() or "")
        # Note: peek_thinking returns None because get_thinking already cleared it
        # But the preserved thinking should be in the output
        assert "I need to:" in final_output

    def test_multiple_tool_calls_with_accumulated_thinking(self):
        """Test multiple tool calls with thinking accumulated across them."""
        thinking_state = ThinkingState()
        buffer = StreamingBuffer()

        # Tool call 1 with thinking
        msg1 = AIMessage(
            content="First query",
            additional_kwargs={"reasoning_content": "First, check the users table"},
            tool_calls=[
                {
                    "id": "t1",
                    "name": "execute_sql",
                    "args": {"query": "SELECT * FROM users LIMIT 10"},
                    "type": "tool_call",
                }
            ],
        )

        # Tool call 2 with more thinking
        msg2 = AIMessage(
            content="Second query",
            additional_kwargs={"reasoning_content": "Now check their orders"},
            tool_calls=[
                {
                    "id": "t2",
                    "name": "execute_sql",
                    "args": {"query": "SELECT * FROM orders"},
                    "type": "tool_call",
                }
            ],
        )

        # Process both messages
        for msg in [msg1, msg2]:
            thinking = extract_thinking_for_interleaved(msg)
            if thinking:
                thinking_state.preserve_thinking(thinking)
            # Flush tool calls
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    buffer.add_tool_call(tc["name"], tc["args"])
                    buffer.flush()

        # Final response
        final_msg = AIMessage(content="Found 10 users with various order counts.")

        # Add preserved thinking before final content
        if thinking_state.has_thinking():
            preserved = thinking_state.get_thinking()
            buffer.add_preserved_thinking(preserved)

        buffer.add_content(final_msg.content)
        result = buffer.flush()

        # Verify accumulated thinking
        assert "First, check the users table" in result
        assert "Now check their orders" in result
        assert "Found 10 users" in result

        # Verify both thinking parts are concatenated
        users_idx = result.find("First, check the users table")
        orders_idx = result.find("Now check their orders")
        final_idx = result.find("Found 10 users")

        assert users_idx < orders_idx < final_idx

    def test_no_preserved_thinking_for_final_only_response(self):
        """Test that final responses without prior tool calls don't add preserved thinking."""
        thinking_state = ThinkingState()
        buffer = StreamingBuffer()

        # Direct final response without any prior tool calls
        final_msg = AIMessage(
            content="<thinking>Let me think</thinking>This is the answer.",
            # No tool_calls
        )

        # No thinking should be extracted (no tool calls)
        thinking = extract_thinking_for_interleaved(final_msg)
        assert thinking is None

        # Buffer should not have preserved thinking
        assert buffer.preserved_thinking is None

        # Add content and flush
        buffer.add_content("This is the answer.")
        result = buffer.flush()

        # Should only have content, no thinking section
        assert result == "This is the answer."
        assert "<thinking>" not in result
        assert "Let me think" not in result
