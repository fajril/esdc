"""Regression: reasoning-model thinking tags must not leak into titles/tags."""

import pytest
from langchain_core.messages import AIMessage

from esdc.chat.agent import (
    _strip_thinking_tags,
    generate_conversation_tags,
    generate_conversation_title,
)


class StubLLM:
    def __init__(self, content: str):
        self._content = content

    async def ainvoke(self, messages, **kwargs):
        return AIMessage(content=self._content)


def test_strip_thinking_tags_removes_qwen3_blocks():
    text = (
        "<thinking>User asks about reserves. Output JSON like "
        '{"title": "..."}</thinking>\n\n{"title": "Cadangan nasional"}'
    )
    assert _strip_thinking_tags(text) == '\n\n{"title": "Cadangan nasional"}'


def test_strip_thinking_tags_handles_think_variant_and_unbalanced():
    assert _strip_thinking_tags("before <think>a</think> after") == "before  after"
    # Response cut off mid-reasoning: drop from the opening tag to the end, so
    # the unterminated reasoning prose cannot become the title.
    assert _strip_thinking_tags("a <thinking>unclosed") == "a "
    assert _strip_thinking_tags("<think>reasoning only, truncated") == ""
    # A closing tag with no opener is a stray token, not a block: keep content.
    assert _strip_thinking_tags("kept</think> also kept") == "kept also kept"
    assert _strip_thinking_tags("plain text") == "plain text"


QWEN_TITLE_WITH_THINKING = (
    "<thinking>User asks about national reserves. Output JSON like "
    '{"title": "..."}</thinking>\n\n'
    '{"title": "Cadangan nasional"}'
)


@pytest.mark.asyncio
async def test_title_ignores_thinking_block():
    llm = StubLLM(QWEN_TITLE_WITH_THINKING)
    assert await generate_conversation_title(llm, "berapa cadangan nasional") == (
        "Cadangan nasional"
    )


@pytest.mark.asyncio
async def test_thinking_only_response_yields_empty():
    # Old behavior returned the raw thinking text as the title; new behavior
    # strips it, leaving the same empty result a contentless response gives.
    llm = StubLLM("<thinking>only reasoning, no final answer</thinking>")
    assert await generate_conversation_title(llm, "halo") == ""


@pytest.mark.asyncio
async def test_truncated_thinking_response_yields_empty_title():
    # Token limit hit mid-reasoning: no closing tag, no JSON. Without the
    # unterminated-block guard the reasoning prose itself became the title.
    llm = StubLLM("<thinking>The user asks about reserves, so the title should")
    assert await generate_conversation_title(llm, "berapa cadangan") == ""


QWEN_TAGS_WITH_THINKING = (
    "<thinking>The query mentions gas production, so tags should include "
    'working areas. Format: {"tags": "..."}</thinking>\n\n'
    '{"tags": "Working Areas, Gas Production"}'
)


@pytest.mark.asyncio
async def test_tags_ignore_thinking_block():
    llm = StubLLM(QWEN_TAGS_WITH_THINKING)
    assert await generate_conversation_tags(llm, "list all working areas") == (
        "Working Areas, Gas Production"
    )
