"""Tests for incomplete status handling on interrupted responses."""

import time


class TestCreateResponseIncompleteEvent:
    """Test create_response_incomplete_event helper function."""

    def test_create_response_incomplete_event_structure(self):
        """Incomplete event should have status='incomplete' and partial output."""
        from esdc.server.responses_events import create_response_incomplete_event

        event = create_response_incomplete_event(
            sequence_number=42,
            response_id="resp_test123",
            model="iris",
            output=[{"type": "message", "content": "partial text..."}],
            error={"message": "Connection timed out", "type": "timeout"},
        )
        assert event["type"] == "response.incomplete"
        assert event["response"]["status"] == "incomplete"
        assert len(event["response"]["output"]) == 1
        assert event["response"]["error"]["type"] == "timeout"

    def test_incomplete_event_preserves_partial_output(self):
        """Partial output should be preserved in the event."""
        from esdc.server.responses_events import create_response_incomplete_event

        partial = [
            {
                "type": "message",
                "id": "msg_1",
                "content": [{"type": "output_text", "text": "Cadangan minyak"}],
            },
            {"type": "function_call_output", "call_id": "fc_1", "output": "SQL result"},
        ]
        event = create_response_incomplete_event(
            sequence_number=10,
            response_id="resp_abc",
            model="iris",
            output=partial,
            error={"message": "Timeout", "type": "timeout"},
        )
        assert event["response"]["output"] == partial
        assert event["response"]["error"]["message"] == "Timeout"

    def test_incomplete_event_with_function_calls(self):
        """Incomplete event can include function call items in partial output."""
        from esdc.server.responses_events import create_response_incomplete_event

        partial = [
            {
                "type": "function_call",
                "id": "fc_1",
                "name": "execute_sql",
                "arguments": '{"query": "SELECT * FROM t"}',
            },
            {
                "type": "function_call_output",
                "call_id": "fc_1",
                "output": [{"type": "input_text", "text": "Result"}],
            },
        ]
        event = create_response_incomplete_event(
            sequence_number=5,
            response_id="resp_test",
            model="iris",
            output=partial,
            error={"message": "Interrupted", "type": "interrupted"},
        )
        assert event["response"]["status"] == "incomplete"
        assert len(event["response"]["output"]) == 2


class TestIncompleteVsFailedStatus:
    """Test that incomplete status is used when partial output exists."""

    def test_incomplete_when_partial_output_exists(self):
        """Should emit incomplete when output_items is non-empty."""
        from esdc.server.responses_events import (
            create_response_failed_event,
            create_response_incomplete_event,
        )

        output_items = [{"type": "message", "content": "partial"}]
        error = {"message": "Timeout", "type": "timeout"}
        # When partial output exists, use incomplete
        incomplete = create_response_incomplete_event(
            sequence_number=1,
            response_id="r",
            model="m",
            output=output_items,
            error=error,
        )
        assert incomplete["response"]["status"] == "incomplete"
        assert incomplete["response"]["output"] == output_items
        # Failed should have empty output
        failed = create_response_failed_event(
            sequence_number=2, response_id="r", model="m", error=error
        )
        assert failed["response"]["status"] == "failed"
        assert failed["response"]["output"] == []

    def test_failed_when_no_partial_output(self):
        """Should emit failed when output_items is empty."""
        from esdc.server.responses_events import create_response_failed_event

        error = {"message": "Error", "type": "server_error"}
        failed = create_response_failed_event(
            sequence_number=1, response_id="r", model="m", error=error
        )
        assert failed["response"]["status"] == "failed"


class TestResponseModelStatus:
    """Test that Response model accepts incomplete status."""

    def test_response_model_accepts_incomplete_status(self):
        """Response Pydantic model should accept 'incomplete' status."""
        from esdc.server.responses_models import Response

        response = Response(
            id="resp_test",
            model="iris",
            status="incomplete",
            output=[{"type": "message", "content": "partial"}],
            error={"message": "Timeout", "type": "timeout"},
            created_at=time.time(),
        )
        assert response.status == "incomplete"

    def test_response_model_output_text_with_incomplete(self):
        """output_text property should work with incomplete responses."""
        from esdc.server.responses_models import Response

        response = Response(
            id="resp_test",
            model="iris",
            status="incomplete",
            output=[
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "Cadangan: 123"}],
                }
            ],
            created_at=time.time(),
        )
        assert response.output_text == "Cadangan: 123"
