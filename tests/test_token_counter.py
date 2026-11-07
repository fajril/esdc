"""Tests for the token counter."""

from types import SimpleNamespace
from typing import get_args

import pytest
import tiktoken
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from esdc.chat.token_counter import (
    TokenCountSource,
    TokenUsage,
    estimate_messages_tokens,
    estimate_text_tokens,
    extract_usage_from_message,
)

pytestmark = pytest.mark.usefixtures("offline_tiktoken")


def test_openai_text_uses_tiktoken_when_available(offline_tiktoken):
    encoding, model_lookups, base_lookups = offline_tiktoken
    text = "2 + 2 = 4. " * 50

    count = estimate_text_tokens(text, provider_type="openai", model="gpt-4o")

    assert count > len(text) // 4
    assert count == len(encoding.encode(text))
    assert model_lookups == ["gpt-4o"]
    assert base_lookups == []


def test_openai_unknown_model_falls_back_to_base_encoding(offline_tiktoken):
    _, model_lookups, base_lookups = offline_tiktoken
    text = "hello world"

    count = estimate_text_tokens(text, provider_type="openai", model="future-gpt-model")

    assert count > 0
    assert model_lookups == ["future-gpt-model"]
    assert base_lookups[0] == "o200k_base"


def test_openai_unknown_model_uses_cl100k_after_o200k_failure(
    offline_tiktoken, monkeypatch
):
    encoding, _, base_lookups = offline_tiktoken
    text = "fallback remains an actual encoded count"

    def get_encoding(name):
        base_lookups.append(name)
        if name == "o200k_base":
            raise LookupError(name)
        if name == "cl100k_base":
            return encoding
        raise KeyError(name)

    monkeypatch.setattr(tiktoken, "get_encoding", get_encoding)

    count = estimate_text_tokens(
        text, provider_type="openai", model="future-gpt-model"
    )

    assert count == len(encoding.encode(text))
    assert base_lookups == ["o200k_base", "cl100k_base"]


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


def test_extract_usage_from_zero_tokens_via_usage_metadata():
    """Zero is a valid token count; should be preserved, not dropped by or-chain."""
    message = AIMessage(content="hello")
    message.usage_metadata = {  # type: ignore[attr-defined]
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
    }

    usage = extract_usage_from_message(message)

    assert usage is not None
    assert usage.input_tokens == 0
    assert usage.output_tokens == 0
    assert usage.total_tokens == 0
    assert usage.source == "provider_usage"
    assert usage.confidence == "exact"


def test_extract_usage_from_zero_tokens_via_response_metadata():
    """vLLM-style: response_metadata with all-zero usage."""
    message = AIMessage(
        content="hello",
        response_metadata={
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            }
        },
    )

    usage = extract_usage_from_message(message)

    assert usage is not None
    assert usage.input_tokens == 0
    assert usage.output_tokens == 0
    assert usage.total_tokens == 0


def test_extract_usage_mixed_zero_and_nonzero():
    """input_tokens=0 but output_tokens=50 — zero should not block output."""
    message = AIMessage(content="hello")
    message.usage_metadata = {  # type: ignore[attr-defined]
        "input_tokens": 0,
        "output_tokens": 50,
        "total_tokens": 50,
    }

    usage = extract_usage_from_message(message)

    assert usage is not None
    assert usage.input_tokens == 0
    assert usage.output_tokens == 50
    assert usage.total_tokens == 50


def test_token_source_literal_has_no_removed_label():
    """D9: nothing produces llama_cpp_tokenize once the native library is gone."""
    sources = set(get_args(TokenCountSource))
    assert "llama_cpp_tokenize" not in sources
    assert sources == {"provider_usage", "tiktoken", "heuristic"}


def test_every_remaining_source_still_produces_usage():
    for source in sorted(get_args(TokenCountSource)):
        usage = TokenUsage(
            input_tokens=10,
            output_tokens=5,
            total_tokens=15,
            source=source,
            confidence="exact" if source == "provider_usage" else "estimated",
        )
        assert usage.to_dict()["source"] == source
