from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from esdc.chat.token_counter import (
    estimate_messages_tokens,
    estimate_text_tokens,
    extract_usage_from_message,
)


def test_openai_text_uses_tiktoken_when_available():
    text = "2 + 2 = 4. " * 50

    count = estimate_text_tokens(text, provider_type="openai", model="gpt-4o")

    assert count > len(text) // 4


def test_openai_unknown_model_falls_back_to_base_encoding():
    text = "hello world"

    count = estimate_text_tokens(
        text, provider_type="openai", model="future-gpt-model"
    )

    assert count > 0


def test_non_openai_provider_uses_heuristic():
    text = "x" * 80

    count = estimate_text_tokens(text, provider_type="anthropic", model="claude")

    assert count == 20


def test_openai_compatible_non_gpt_uses_heuristic():
    text = "x" * 80

    count = estimate_text_tokens(
        text, provider_type="openai_compatible", model="qwen3:latest"
    )

    assert count == 20


def test_message_counting_includes_system_tool_and_tool_call_args():
    messages = [
        HumanMessage(content="Query"),
        AIMessage(
            content="",
            tool_calls=[
                {"name": "execute_sql", "args": {"query": "SELECT 1"}, "id": "1"}
            ],
        ),
        ToolMessage(content="Result data", tool_call_id="1"),
    ]

    count = estimate_messages_tokens(messages, system_prompt="System prompt")

    assert count >= 10


def test_extract_usage_from_langchain_usage_metadata():
    message = AIMessage(content="hello")
    message.usage_metadata = {  # type: ignore[attr-defined]
        "input_tokens": 100,
        "output_tokens": 50,
        "total_tokens": 150,
    }

    usage = extract_usage_from_message(message)

    assert usage is not None
    assert usage.input_tokens == 100
    assert usage.output_tokens == 50
    assert usage.total_tokens == 150
    assert usage.source == "provider_usage"
    assert usage.confidence == "exact"


def test_extract_usage_from_openai_response_metadata():
    message = AIMessage(
        content="hello",
        response_metadata={
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            }
        },
    )

    usage = extract_usage_from_message(message)

    assert usage is not None
    assert usage.input_tokens == 10
    assert usage.output_tokens == 5
    assert usage.total_tokens == 15


def test_extract_usage_from_ollama_style_response_metadata():
    message = AIMessage(
        content="hello",
        response_metadata={
            "usage": {
                "prompt_eval_count": 12,
                "eval_count": 8,
            }
        },
    )

    usage = extract_usage_from_message(message)

    assert usage is not None
    assert usage.input_tokens == 12
    assert usage.output_tokens == 8
    assert usage.total_tokens == 20


def test_extract_usage_from_object_response_metadata():
    message = AIMessage(
        content="hello",
        response_metadata={
            "usage": SimpleNamespace(prompt_tokens=7, completion_tokens=3)
        },
    )

    usage = extract_usage_from_message(message)

    assert usage is not None
    assert usage.input_tokens == 7
    assert usage.output_tokens == 3
    assert usage.total_tokens == 10


def test_extract_usage_returns_none_when_missing():
    message = AIMessage(content="hello")

    usage = extract_usage_from_message(message)

    assert usage is None
