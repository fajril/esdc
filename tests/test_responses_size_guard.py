"""Tests for the response size guard."""

# Standard library
import json

# Local
from esdc.server.responses_events import (
    MAX_SSE_EVENT_BYTES,
    MAX_TOOL_RESULT_TEXT_CHARS,
    SSE_LINE_BYTE_LIMIT,
    TRUNCATION_SUFFIX,
    create_function_call_output_item,
    create_response_completed_event,
    create_response_incomplete_event,
    format_sse_event,
    slim_output_items,
)


def _fco_text(item: dict) -> str:
    return item["output"][0]["text"]


def _message_item(text: str) -> dict:
    return {
        "id": "msg_1",
        "type": "message",
        "status": "completed",
        "role": "assistant",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }


def _reasoning_item(text: str) -> dict:
    return {
        "id": "rs_1",
        "type": "reasoning",
        "status": "completed",
        "summary": [{"type": "reasoning_summary_text", "text": text}],
    }


def _max_sse_line_bytes(sse: str) -> int:
    return max(len(line.encode("utf-8")) for line in sse.split("\n"))


class TestCreateFunctionCallOutputItem:
    def test_structure(self):
        item = create_function_call_output_item("fco_1", "call_1", "result text")
        assert item["id"] == "fco_1"
        assert item["type"] == "function_call_output"
        assert item["status"] == "completed"
        assert item["call_id"] == "call_1"
        assert item["output"] == [{"type": "input_text", "text": "result text"}]

    def test_short_text_unchanged(self):
        item = create_function_call_output_item("fco_1", "call_1", "short")
        assert _fco_text(item) == "short"

    def test_long_text_truncated(self):
        long_text = "x" * (MAX_TOOL_RESULT_TEXT_CHARS + 5000)
        item = create_function_call_output_item("fco_1", "call_1", long_text)
        text = _fco_text(item)
        assert text.endswith(TRUNCATION_SUFFIX)
        assert len(text) == MAX_TOOL_RESULT_TEXT_CHARS + len(TRUNCATION_SUFFIX)

    def test_text_at_limit_unchanged(self):
        text = "x" * MAX_TOOL_RESULT_TEXT_CHARS
        item = create_function_call_output_item("fco_1", "call_1", text)
        assert _fco_text(item) == text


class TestSlimOutputItems:
    def test_small_output_returned_unchanged(self):
        items = [
            _message_item("hello"),
            create_function_call_output_item("fco_1", "call_1", "result"),
        ]
        assert slim_output_items(items, MAX_SSE_EVENT_BYTES) == items

    def test_oversized_tool_outputs_slimmed(self):
        items = [
            create_function_call_output_item(f"fco_{i}", f"call_{i}", "x" * 30_000)
            for i in range(5)
        ]
        items.append(_message_item("final answer"))
        slimmed = slim_output_items(items, MAX_SSE_EVENT_BYTES)
        assert len(json.dumps(slimmed).encode("utf-8")) <= MAX_SSE_EVENT_BYTES
        # message text untouched
        assert slimmed[-1]["content"][0]["text"] == "final answer"
        # tool output texts truncated
        for item in slimmed[:-1]:
            assert _fco_text(item).endswith(TRUNCATION_SUFFIX)

    def test_original_items_not_mutated(self):
        items = [
            create_function_call_output_item("fco_1", "call_1", "x" * 200_000),
        ]
        original_text = _fco_text(items[0])
        slim_output_items(items, MAX_SSE_EVENT_BYTES)
        assert _fco_text(items[0]) == original_text

    def test_oversized_reasoning_slimmed(self):
        items = [_reasoning_item("r" * 200_000), _message_item("answer")]
        slimmed = slim_output_items(items, MAX_SSE_EVENT_BYTES)
        assert len(json.dumps(slimmed).encode("utf-8")) <= MAX_SSE_EVENT_BYTES
        assert slimmed[0]["summary"][0]["text"].endswith(TRUNCATION_SUFFIX)
        assert slimmed[1]["content"][0]["text"] == "answer"

    def test_oversized_function_call_arguments_slimmed(self):
        items = [
            {
                "id": "fc_1",
                "type": "function_call",
                "status": "completed",
                "name": "search",
                "call_id": "call_1",
                "arguments": json.dumps({"q": "y" * 200_000}),
            },
            _message_item("answer"),
        ]
        slimmed = slim_output_items(items, MAX_SSE_EVENT_BYTES)
        assert len(json.dumps(slimmed).encode("utf-8")) <= MAX_SSE_EVENT_BYTES
        assert slimmed[0]["arguments"].endswith(TRUNCATION_SUFFIX)


class TestTerminalEventSizeGuard:
    def test_completed_event_fits_sse_line_limit(self):
        # Regression: aiohttp clients (Open WebUI) reject SSE lines > 131072 bytes.
        items = [
            create_function_call_output_item(
                f"fco_{i}", f"call_{i}", "x" * MAX_TOOL_RESULT_TEXT_CHARS
            )
            for i in range(10)
        ]
        items.append(_message_item("final answer"))
        event = create_response_completed_event(918, "resp_x", "iris", items)
        sse = format_sse_event(event)
        assert _max_sse_line_bytes(sse) < SSE_LINE_BYTE_LIMIT
        # message survives intact
        output = event["response"]["output"]
        assert output[-1]["content"][0]["text"] == "final answer"

    def test_completed_event_small_output_unchanged(self):
        items = [_message_item("hi")]
        event = create_response_completed_event(2, "resp_x", "iris", items)
        assert event["response"]["output"] == items

    def test_incomplete_event_fits_sse_line_limit(self):
        items = [
            create_function_call_output_item(
                f"fco_{i}", f"call_{i}", "x" * MAX_TOOL_RESULT_TEXT_CHARS
            )
            for i in range(10)
        ]
        event = create_response_incomplete_event(
            10, "resp_x", "iris", items, {"message": "timeout", "type": "timeout"}
        )
        sse = format_sse_event(event)
        assert _max_sse_line_bytes(sse) < SSE_LINE_BYTE_LIMIT
        assert event["response"]["error"] == {"message": "timeout", "type": "timeout"}
