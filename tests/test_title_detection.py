"""Tests for ancillary request detection (title/tag generation)."""

# ruff: noqa: D103

from esdc.server.title_detection import (
    create_ancillary_chat_response,
    create_tags_sync_response,
    create_title_sync_response,
    extract_user_query,
    get_ancillary_type,
    is_ancillary_request,
    is_title_generation_request,
)


def test_detect_title_task():
    assert is_ancillary_request(
        [
            {
                "type": "message",
                "role": "user",
                "content": "### Task:\nGenerate a concise, 3-5 word title with emojis. Summarize the conversation.\n\nUser: di mana lokasi lapangan kinanti?",  # noqa: E501
            }
        ]
    )


def test_detect_tag_task():
    assert is_ancillary_request(
        [
            {
                "type": "message",
                "role": "user",
                "content": "### Task:\nGenerate 1-3 broad tags categorizing the conversation.\n\nUser: di mana lokasi lapangan kinanti?",  # noqa: E501
            }
        ]
    )


def test_detect_task_with_string_input():
    assert is_ancillary_request(
        "### Task:\nGenerate a concise, 3-5 word title with emojis."
    )


def test_detect_tag_with_string_input():
    assert is_ancillary_request(
        "### Task:\nGenerate 1-3 broad tags categorizing the conversation."
    )


def test_detect_task_with_content_list():
    assert is_ancillary_request(
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
    assert not is_ancillary_request(
        [
            {
                "type": "message",
                "role": "user",
                "content": "berapa cadangan lapangan handil?",
            }
        ]
    )


def test_reject_string_query():
    assert not is_ancillary_request("berapa cadangan lapangan handil?")


def test_reject_multi_item_input():
    assert not is_ancillary_request(
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
    assert not is_ancillary_request("")
    assert not is_ancillary_request([])


def test_get_ancillary_type_title():
    assert (
        get_ancillary_type(
            [
                {
                    "type": "message",
                    "role": "user",
                    "content": "### Task:\nGenerate a concise title.",
                }
            ]
        )
        == "title"
    )


def test_get_ancillary_type_tags():
    assert (
        get_ancillary_type(
            [
                {
                    "type": "message",
                    "role": "user",
                    "content": "### Task:\nGenerate 1-3 broad tags categorizing.",
                }
            ]
        )
        == "tags"
    )


def test_get_ancillary_type_other():
    assert (
        get_ancillary_type(
            [
                {
                    "type": "message",
                    "role": "user",
                    "content": "### Task:\nSummarize the document.",
                }
            ]
        )
        == "other"
    )


def test_get_ancillary_type_non_ancillary():
    assert get_ancillary_type("normal query") is None


def test_extract_user_query():
    query = extract_user_query(
        [
            {
                "type": "message",
                "role": "user",
                "content": "### Task:\nGenerate a concise title.\n\nUser: di mana lokasi lapangan kinanti?\n\nTitle:",  # noqa: E501
            }
        ]
    )
    assert query == "di mana lokasi lapangan kinanti?"


def test_extract_user_query_from_tag_request():
    query = extract_user_query(
        [
            {
                "type": "message",
                "role": "user",
                "content": "### Task:\nGenerate tags.\n\nUser: oil reserves in Rokan\n\nTags:",  # noqa: E501
            }
        ]
    )
    assert query == "oil reserves in Rokan"


def test_extract_user_query_fallback():
    query = extract_user_query("### Task:\nGenerate a title")
    assert query == "### Task:\nGenerate a title"


def test_create_title_sync_response():
    result = create_title_sync_response("Test Title", "resp_123")
    assert result["id"] == "resp_123"
    assert result["status"] == "completed"
    assert result["model"] == "iris"
    assert result["output"][0]["content"][0]["text"] == "Test Title"
    assert result["output"][0]["role"] == "assistant"


def test_create_tags_sync_response():
    result = create_tags_sync_response("Reserves, Oil, Rokan", "resp_456")
    assert result["id"] == "resp_456"
    assert result["status"] == "completed"
    assert result["output"][0]["content"][0]["text"] == "Reserves, Oil, Rokan"


def test_create_ancillary_chat_response():
    result = create_ancillary_chat_response("Rokan Field Oil Reserves")
    assert result["object"] == "chat.completion"
    assert result["choices"][0]["message"]["content"] == "Rokan Field Oil Reserves"
    assert result["choices"][0]["finish_reason"] == "stop"


def test_backward_compat_is_title_generation_request():
    """is_title_generation_request now matches all ### Task: requests."""
    assert is_title_generation_request(
        [
            {
                "type": "message",
                "role": "user",
                "content": "### Task:\nGenerate 1-3 broad tags.",
            }
        ]
    )
    assert not is_title_generation_request("normal query")


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
    assert is_ancillary_request([item])
    assert get_ancillary_type([item]) == "title"


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
    query = extract_user_query([item])
    assert query == "di mana lokasi lapangan kinanti?"


def test_reject_normal_query_pydantic_model():
    from esdc.server.responses_models import ResponseInputItem

    item = ResponseInputItem(
        type="message",
        role="user",
        content=[{"type": "input_text", "text": "berapa cadangan lapangan handil?"}],
    )
    assert not is_ancillary_request([item])
    assert get_ancillary_type([item]) is None
