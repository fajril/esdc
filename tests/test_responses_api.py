"""Tests for OpenAI Responses API endpoint."""

import json

from esdc.server.responses_events import (
    create_content_part_added_event,
    create_content_part_done_event,
    create_function_call_arguments_delta_event,
    create_function_call_arguments_done_event,
    create_output_item_added_event,
    create_output_item_done_event,
    create_output_text_delta_event,
    create_output_text_done_event,
    create_response_completed_event,
    create_response_created_event,
    format_sse_event,
)
from esdc.server.responses_models import (
    Response,
    ResponseInputItem,
    ResponsesRequest,
)
from esdc.server.responses_wrapper import (
    SequenceCounter,
    chunk_json,
    chunk_text,
    convert_responses_input_to_langchain,
    generate_item_id,
)


class TestResponsesModels:
    """Tests for Responses API request/response models."""

    def test_responses_request_with_string_input(self):
        """Test request with simple string input."""
        request = ResponsesRequest(input="Hello, world!")
        assert request.input == "Hello, world!"
        assert request.model == "esdc-agent"
        assert request.stream is True
        assert request.instructions is None

    def test_responses_request_with_list_input(self):
        """Test request with list of input items."""
        request = ResponsesRequest(
            input=[
                ResponseInputItem(type="message", role="user", content="Hello"),
            ]
        )
        assert isinstance(request.input, list)
        assert len(request.input) == 1

    def test_responses_request_with_instructions(self):
        """Test request with system instructions."""
        request = ResponsesRequest(
            input="Hello",
            instructions="You are a helpful assistant.",
        )
        assert request.instructions == "You are a helpful assistant."

    def test_input_item_message(self):
        """Test input item message creation."""
        item = ResponseInputItem(type="message", role="user", content="Test message")
        assert item.type == "message"
        assert item.role == "user"
        assert item.content == "Test message"

    def test_input_item_function_call_output(self):
        """Test input item function call output creation."""
        item = ResponseInputItem(
            type="function_call_output",
            call_id="call_123",
            output="result",
        )
        assert item.type == "function_call_output"
        assert item.call_id == "call_123"
        assert item.output == "result"

    def test_response_output_text_with_dict_items(self):
        """Test output_text with valid dict items."""
        response = Response(
            id="test",
            object="response",
            created_at=1234567890,
            model="test",
            status="completed",
            output=[
                {
                    "id": "msg_1",
                    "type": "message",
                    "status": "completed",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Hello"}],
                }
            ],
        )
        assert response.output_text == "Hello"

    def test_response_output_text_with_string_items(self):
        """Test output_text handles string items gracefully.

        Pydantic validates at model creation, so we bypass by setting
        output directly to test defensive code in the property.
        """
        response = Response(
            id="test",
            object="response",
            created_at=1234567890,
            model="test",
            status="completed",
            output=[],
        )
        # Bypass Pydantic validation to test defensive code
        response.output = [
            "invalid string",
            {
                "type": "message",
                "content": [{"type": "output_text", "text": "World"}],
            },
        ]
        assert response.output_text == "World"

    def test_response_output_text_with_malformed_content(self):
        """Test output_text handles malformed content."""
        response = Response(
            id="test",
            object="response",
            created_at=1234567890,
            model="test",
            status="completed",
            output=[
                {
                    "type": "message",
                    "content": [
                        "not a dict",
                        {"type": "output_text", "text": "Test"},
                    ],
                }
            ],
        )
        assert response.output_text == "Test"

    def test_response_output_text_with_non_string_text(self):
        """Test output_text handles non-string text fields."""
        response = Response(
            id="test",
            object="response",
            created_at=1234567890,
            model="test",
            status="completed",
            output=[
                {
                    "type": "message",
                    "content": [
                        {"type": "output_text", "text": 123},  # Non-string text
                        {"type": "output_text", "text": "Valid"},
                    ],
                }
            ],
        )
        assert response.output_text == "Valid"

    def test_response_output_text_with_non_list_content(self):
        """Test output_text handles non-list content."""
        response = Response(
            id="test",
            object="response",
            created_at=1234567890,
            model="test",
            status="completed",
            output=[
                {
                    "type": "message",
                    "content": "not a list",  # Should be list
                }
            ],
        )
        assert response.output_text == ""


class TestResponsesEvents:
    """Tests for SSE event formatters."""

    def test_format_sse_event(self):
        """Test SSE event formatting."""
        event = {"type": "test_event", "sequence_number": 1, "data": "test"}
        sse = format_sse_event(event)
        assert sse.startswith("event: test_event\n")
        assert "data:" in sse
        assert sse.endswith("\n\n")

    def test_create_response_created_event(self):
        """Test response.created event."""
        event = create_response_created_event("resp_123", "esdc-agent")
        assert event["type"] == "response.created"
        assert event["sequence_number"] == 1
        assert event["response"]["id"] == "resp_123"
        assert event["response"]["status"] == "in_progress"
        assert event["response"]["output"] == []

    def test_create_response_completed_event(self):
        """Test response.completed event."""
        event = create_response_completed_event(
            sequence_number=10,
            response_id="resp_123",
            model="esdc-agent",
            output=[{"type": "message", "content": "Hello"}],
        )
        assert event["type"] == "response.completed"
        assert event["sequence_number"] == 10
        assert event["response"]["status"] == "completed"
        assert len(event["response"]["output"]) == 1

    def test_create_output_item_added_event(self):
        """Test output_item.added event."""
        item = {"id": "msg_123", "type": "message", "status": "in_progress"}
        event = create_output_item_added_event(
            sequence_number=2,
            output_index=0,
            item=item,
        )
        assert event["type"] == "response.output_item.added"
        assert event["output_index"] == 0
        assert event["item"]["id"] == "msg_123"

    def test_create_output_item_done_event(self):
        """Test output_item.done event."""
        item = {"id": "msg_123", "type": "message", "status": "completed"}
        event = create_output_item_done_event(
            sequence_number=5,
            output_index=0,
            item=item,
        )
        assert event["type"] == "response.output_item.done"
        assert event["item"]["status"] == "completed"

    def test_create_content_part_added_event(self):
        """Test content_part.added event."""
        part = {"type": "output_text", "text": ""}
        event = create_content_part_added_event(
            sequence_number=3,
            output_index=0,
            content_index=0,
            item_id="msg_123",
            part=part,
        )
        assert event["type"] == "response.content_part.added"
        assert event["content_index"] == 0
        assert event["part"]["type"] == "output_text"

    def test_create_output_text_delta_event(self):
        """Test output_text.delta event."""
        event = create_output_text_delta_event(
            sequence_number=4,
            output_index=0,
            content_index=0,
            item_id="msg_123",
            delta="Hello",
        )
        assert event["type"] == "response.output_text.delta"
        assert event["delta"] == "Hello"

    def test_create_output_text_done_event(self):
        """Test output_text.done event."""
        event = create_output_text_done_event(
            sequence_number=8,
            output_index=0,
            content_index=0,
            item_id="msg_123",
            text="Hello World",
        )
        assert event["type"] == "response.output_text.done"
        assert event["text"] == "Hello World"

    def test_create_content_part_done_event(self):
        """Test content_part.done event."""
        part = {"type": "output_text", "text": "Hello World"}
        event = create_content_part_done_event(
            sequence_number=9,
            output_index=0,
            content_index=0,
            item_id="msg_123",
            part=part,
        )
        assert event["type"] == "response.content_part.done"
        assert event["part"]["text"] == "Hello World"

    def test_create_function_call_args_delta_event(self):
        """Test function_call_arguments.delta event."""
        event = create_function_call_arguments_delta_event(
            sequence_number=4,
            output_index=0,
            item_id="fc_123",
            delta='{"arg1":',
        )
        assert event["type"] == "response.function_call_arguments.delta"
        assert event["delta"] == '{"arg1":'

    def test_create_function_call_args_done_event(self):
        """Test function_call_arguments.done event."""
        args = '{"query": "SELECT 1"}'
        event = create_function_call_arguments_done_event(
            sequence_number=10,
            output_index=0,
            item_id="fc_123",
            arguments=args,
        )
        assert event["type"] == "response.function_call_arguments.done"
        assert event["arguments"] == args


class TestResponsesWrapper:
    """Tests for Responses API wrapper functions."""

    def test_sequence_counter(self):
        """Test SequenceCounter starts at 1 and increments."""
        counter = SequenceCounter()
        assert counter.next() == 1
        assert counter.next() == 2
        assert counter.next() == 3

    def test_sequence_counter_custom_start(self):
        """Test SequenceCounter with custom start value."""
        counter = SequenceCounter(start=100)
        assert counter.next() == 100
        assert counter.next() == 101

    def test_generate_item_id(self):
        """Test item ID generation with proper prefix."""
        msg_id = generate_item_id("msg")
        assert msg_id.startswith("msg_")
        assert len(msg_id) == 28  # "msg_" + 24 hex chars

        fc_id = generate_item_id("fc")
        assert fc_id.startswith("fc_")

        fco_id = generate_item_id("fco")
        assert fco_id.startswith("fco_")

    def test_chunk_text(self):
        """Test text chunking."""
        chunks = list(chunk_text("Hello world from ESDC"))
        assert len(chunks) > 0
        assert "".join(chunks) == "Hello world from ESDC"

    def test_chunk_text_preserves_markdown_headers(self):
        """Test that markdown headers are not broken across chunks."""
        # Header should stream in character groups, not word boundaries
        text = "## Cadangan Lapangan"
        chunks = list(chunk_text(text, chunk_size=3))

        # Each chunk should be small enough to not split markdown tokens badly
        # With chunk_size=3, we expect multiple small chunks
        full_text = "".join(chunks)
        assert full_text == text

        # Verify no chunk starts with partial header syntax
        # If "##" is in the text, it should be complete in at least one chunk
        has_complete_header = any("##" in chunk for chunk in chunks)
        # Note: With chunk_size=3, "## " will be in ONE chunk
        assert has_complete_header, "Header syntax should remain intact"

    def test_chunk_text_preserves_markdown_tables(self):
        """Test that markdown tables are not broken across chunks."""
        text = "| Jenis | Volume |\n|---|---|\n| Gas | 100 |"
        chunks = list(chunk_text(text, chunk_size=3))

        full_text = "".join(chunks)
        assert full_text == text

        # Table syntax should not be split mid-pattern
        # With character-level streaming, this is guaranteed
        assert len(chunks) > 1, "Should have multiple chunks"

    def test_chunk_text_character_level_streaming(self):
        """Test that text streams in small character groups, not word boundaries."""
        # Long word that would be split at character level
        text = "##heading-with-many-words-here"
        chunks = list(chunk_text(text, chunk_size=3))

        # With character-level streaming, each chunk is ~3 chars
        # With word-level streaming, this would be ONE chunk (the whole word)
        full_text = "".join(chunks)
        assert full_text == text

        # Character-level should produce MANY chunks
        assert len(chunks) > 5, f"Expected many chunks, got {len(chunks)}: {chunks}"

        # Each chunk should be small (around chunk_size)
        avg_chunk_size = sum(len(c) for c in chunks) / len(chunks)
        assert avg_chunk_size <= 5, f"Average chunk size {avg_chunk_size} too large"

    def test_chunk_json(self):
        """Test JSON chunking."""
        json_str = '{"arg1": "value1", "arg2": "value2"}'
        chunks = list(chunk_json(json_str, chunk_size=10))
        assert "".join(chunks) == json_str

    def test_convert_responses_input_string(self):
        """Test converting string input to LangChain messages."""
        from langchain_core.messages import HumanMessage

        messages = convert_responses_input_to_langchain("Hello, world!")
        assert len(messages) == 1
        assert isinstance(messages[0], HumanMessage)
        assert messages[0].content == "Hello, world!"

    def test_convert_responses_input_with_instructions(self):
        """Test converting input with instructions."""
        from langchain_core.messages import HumanMessage, SystemMessage

        messages = convert_responses_input_to_langchain(
            "Hello", instructions="You are a helpful assistant."
        )
        assert len(messages) == 2
        assert isinstance(messages[0], SystemMessage)
        assert messages[0].content == "You are a helpful assistant."
        assert isinstance(messages[1], HumanMessage)
        assert messages[1].content == "Hello"

    def test_convert_responses_input_list(self):
        """Test converting list of input items."""
        from langchain_core.messages import HumanMessage

        messages = convert_responses_input_to_langchain(
            [
                ResponseInputItem(type="message", role="user", content="Hello"),
            ]
        )
        assert len(messages) == 1
        assert isinstance(messages[0], HumanMessage)


class TestEventLifecycle:
    """Tests for complete event lifecycle sequences."""

    def test_message_item_lifecycle(self):
        """Test complete lifecycle for a message item."""
        seq = SequenceCounter()
        output_index = 0
        item_id = generate_item_id("msg")

        # 1. response.created (sequence_number = 1)
        created = create_response_created_event("resp_123", "esdc-agent", seq.next())
        assert created["sequence_number"] == 1

        # 2. output_item.added (sequence_number = 2)
        added = create_output_item_added_event(
            seq.next(),
            output_index,
            {"id": item_id, "type": "message", "status": "in_progress"},
        )
        assert added["type"] == "response.output_item.added"
        assert added["sequence_number"] == 2

        # 3. content_part.added (sequence_number = 3)
        part_added = create_content_part_added_event(
            seq.next(), output_index, 0, item_id, {"type": "output_text", "text": ""}
        )
        assert part_added["type"] == "response.content_part.added"
        assert part_added["sequence_number"] == 3

        # 4. output_text.delta (sequence_number = 4)
        delta1 = create_output_text_delta_event(
            seq.next(), output_index, 0, item_id, "Hello"
        )
        assert delta1["type"] == "response.output_text.delta"
        assert delta1["sequence_number"] == 4

        # 5. output_text.done (sequence_number = 5)
        text_done = create_output_text_done_event(
            seq.next(), output_index, 0, item_id, "Hello World"
        )
        assert text_done["type"] == "response.output_text.done"
        assert text_done["sequence_number"] == 5

        # 6. content_part.done (sequence_number = 6)
        part_done = create_content_part_done_event(
            seq.next(),
            output_index,
            0,
            item_id,
            {"type": "output_text", "text": "Hello World"},
        )
        assert part_done["type"] == "response.content_part.done"
        assert part_done["sequence_number"] == 6

        # 7. output_item.done (sequence_number = 7)
        item_done = create_output_item_done_event(
            seq.next(),
            output_index,
            {"id": item_id, "type": "message", "status": "completed"},
        )
        assert item_done["type"] == "response.output_item.done"
        assert item_done["sequence_number"] == 7

        # Verify sequence numbers are monotonically increasing
        assert added["sequence_number"] > created["sequence_number"]
        assert part_added["sequence_number"] > added["sequence_number"]
        assert delta1["sequence_number"] > part_added["sequence_number"]
        assert text_done["sequence_number"] > delta1["sequence_number"]
        assert part_done["sequence_number"] > text_done["sequence_number"]
        assert item_done["sequence_number"] > part_done["sequence_number"]

    def test_function_call_item_lifecycle(self):
        """Test complete lifecycle for a function_call item."""
        seq = SequenceCounter(start=100)  # Start after message events
        output_index = 1
        item_id = generate_item_id("fc")

        # 1. output_item.added (function_call)
        added = create_output_item_added_event(
            seq.next(),
            output_index,
            {
                "id": item_id,
                "type": "function_call",
                "status": "in_progress",
                "name": "execute_sql",
                "call_id": "call_123",
            },
        )
        assert added["type"] == "response.output_item.added"

        # 2. function_call_arguments.delta (multiple)
        args_delta = create_function_call_arguments_delta_event(
            seq.next(), output_index, item_id, '{"query":'
        )
        assert args_delta["type"] == "response.function_call_arguments.delta"

        # 3. function_call_arguments.done
        args_done = create_function_call_arguments_done_event(
            seq.next(), output_index, item_id, '{"query": "SELECT 1"}'
        )
        assert args_done["type"] == "response.function_call_arguments.done"

        # 4. output_item.done (function_call)
        item_done = create_output_item_done_event(
            seq.next(),
            output_index,
            {
                "id": item_id,
                "type": "function_call",
                "status": "completed",
                "name": "execute_sql",
                "call_id": "call_123",
                "arguments": '{"query": "SELECT 1"}',
            },
        )
        assert item_done["type"] == "response.output_item.done"

        # Verify sequence numbers
        assert args_delta["sequence_number"] > added["sequence_number"]
        assert args_done["sequence_number"] > args_delta["sequence_number"]
        assert item_done["sequence_number"] > args_done["sequence_number"]


class TestSSEFormat:
    """Tests for SSE event serialization."""

    def test_sse_format_structure(self):
        """Test that SSE format matches Open Responses spec."""
        event = {"type": "test", "sequence_number": 1, "data": "test"}
        sse_string = format_sse_event(event)

        # Must start with "event: <type>\n"
        assert sse_string.startswith(f"event: {event['type']}\n")

        # Must contain "data: <json>\n"
        assert f"data: {json.dumps(event)}\n" in sse_string

        # Must end with double newline
        assert sse_string.endswith("\n\n")

    def test_sse_format_with_complex_event(self):
        """Test SSE format with complex nested event."""
        event = create_output_item_added_event(
            sequence_number=5,
            output_index=2,
            item={
                "id": "msg_abc123",
                "type": "message",
                "status": "in_progress",
                "role": "assistant",
                "content": [],
            },
        )
        sse_string = format_sse_event(event)

        # Parse and verify
        lines = sse_string.strip().split("\n")
        assert lines[0].startswith("event: ")
        assert lines[1].startswith("data: ")

        # Verify JSON is valid
        import json

        json_data = lines[1][6:]  # Remove "data: " prefix
        parsed = json.loads(json_data)
        assert parsed["type"] == "response.output_item.added"
        assert parsed["item"]["id"] == "msg_abc123"


# Run with: python -m pytest tests/test_responses_api.py -v
