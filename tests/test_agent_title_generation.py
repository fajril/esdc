"""Regression: reasoning-model thinking tags must not leak into titles/tags."""

import pytest
from langchain_core.messages import AIMessage

from esdc.chat.agent import generate_conversation_tags, generate_conversation_title


class StubLLM:
    def __init__(self, content: str):
        self._content = content

    async def ainvoke(self, messages, **kwargs):
        return AIMessage(content=self._content)


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
