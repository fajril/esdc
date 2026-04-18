"""Tests for title generation request detection."""

from esdc.server.title_detection import (
    create_title_sync_response,
    extract_user_query_from_title_request,
    is_title_generation_request,
)


def test_detect_task_with_title():
    assert is_title_generation_request(
        [
            {
                "type": "message",
                "role": "user",
                "content": "### Task:\nGenerate a concise, 3-5 word title with emojis. Summarize the conversation.\n\nUser: di mana lokasi lapangan kinanti?",  # noqa: E501
            }
        ]
    )


def test_detect_task_with_title_string_input():
    assert is_title_generation_request(
        "### Task:\nGenerate a concise, 3-5 word title with emojis."
    )


def test_detect_task_with_content_list():
    assert is_title_generation_request(
        [
            {
                "type": "message",
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": "### Task:\nGenerate a concise, 3-5 word title",
                    }
                ],
            }
        ]
    )


def test_reject_normal_query():
    assert not is_title_generation_request(
        [
            {
                "type": "message",
                "role": "user",
                "content": "berapa cadangan lapangan handil?",
            }
        ]
    )


def test_reject_string_query():
    assert not is_title_generation_request("berapa cadangan lapangan handil?")


def test_reject_multi_item_input():
    assert not is_title_generation_request(
        [
            {"type": "message", "role": "user", "content": "hello"},
            {"type": "message", "role": "assistant", "content": "hi"},
            {
                "type": "message",
                "role": "user",
                "content": "### Task:\nGenerate title",
            },
        ]
    )


def test_reject_empty_input():
    assert not is_title_generation_request("")
    assert not is_title_generation_request([])


def test_extract_user_query():
    query = extract_user_query_from_title_request(
        [
            {
                "type": "message",
                "role": "user",
                "content": "### Task:\nGenerate a concise title.\n\nUser: di mana lokasi lapangan kinanti?\n\nTitle:",  # noqa: E501
            }
        ]
    )
    assert query == "di mana lokasi lapangan kinanti?"


def test_extract_user_query_fallback():
    query = extract_user_query_from_title_request("### Task:\nGenerate a title")
    assert query == "### Task:\nGenerate a title"


def test_create_title_sync_response():
    result = create_title_sync_response("Test Title", "resp_123")
    assert result["id"] == "resp_123"
    assert result["status"] == "completed"
    assert result["model"] == "iris"
    assert result["output"][0]["content"][0]["text"] == "Test Title"
    assert result["output"][0]["role"] == "assistant"


def test_reject_task_without_title_keyword():
    assert not is_title_generation_request(
        [
            {
                "type": "message",
                "role": "user",
                "content": "### Task:\nSummarize the document.",
            }
        ]
    )


def test_detect_task_with_pydantic_model():
    from esdc.server.responses_models import ResponseInputItem

    item = ResponseInputItem(
        type="message",
        role="user",
        content=[
            {
                "type": "input_text",
                "text": "### Task:\nGenerate a concise, 3-5 word title with emojis. Summarize the conversation.\n\nUser: di mana lokasi lapangan kinanti?\n\nTitle:",  # noqa: E501
            }
        ],
    )
    assert is_title_generation_request([item])


def test_extract_user_query_with_pydantic_model():
    from esdc.server.responses_models import ResponseInputItem

    item = ResponseInputItem(
        type="message",
        role="user",
        content=[
            {
                "type": "input_text",
                "text": "### Task:\nGenerate a concise title.\n\nUser: di mana lokasi lapangan kinanti?\n\nTitle:",  # noqa: E501
            }
        ],
    )
    query = extract_user_query_from_title_request([item])
    assert query == "di mana lokasi lapangan kinanti?"


def test_reject_normal_query_pydantic_model():
    from esdc.server.responses_models import ResponseInputItem

    item = ResponseInputItem(
        type="message",
        role="user",
        content=[{"type": "input_text", "text": "berapa cadangan lapangan handil?"}],
    )
    assert not is_title_generation_request([item])
