"""Tests that agent LLM requests carry exactly one leading system message.

Regression for: Qwen-family chat templates (Qwen3.x, e.g. Qwen3.6-27B)
reject ANY system message that is not the very first message. The template
raises a Jinja exception ("System message must be at the beginning") which
llama.cpp surfaces as HTTP 400 "Unable to generate parser for this template.
Automatic parser generation failed". The query-classifier strategy, the
tool-limit nudges, and compaction summaries are all SystemMessages that used
to land after the user message, producing [system, user, system] requests.
agent_node now merges all trailing system content into the leading prompt.
"""

from typing import cast

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage

from esdc.chat.agent import create_agent


class RecordingLLM:
    """Minimal LLM stub that records every message list it receives."""

    def __init__(self, response: AIMessage):
        self._response = response
        self.seen: list[list] = []

    def bind_tools(self, tools):
        return self

    async def ainvoke(self, messages, **kwargs):
        self.seen.append(list(messages))
        return self._response


@pytest.mark.asyncio
async def test_outgoing_request_has_single_leading_system_message():
    """Classifier strategy SystemMessage must merge into the leading prompt."""
    llm = RecordingLLM(AIMessage(content="jawaban final"))
    agent = create_agent(
        cast(BaseChatModel, llm), tools=[], checkpointer=None, context_length=8000
    )

    await agent.ainvoke(
        {"messages": [HumanMessage(content="halo")]},
        config={"recursion_limit": 10},
    )

    assert llm.seen, "agent must invoke the LLM"
    request = llm.seen[-1]

    roles = [type(m).__name__ for m in request]
    assert roles[0] == "SystemMessage"
    # Qwen3.x templates 400 on any system message after the first position.
    assert "SystemMessage" not in roles[1:], f"trailing system messages: {roles}"

    # The classifier's strategy guidance must survive inside the merged prompt.
    merged = str(request[0].content)
    assert "## Query Analysis (Pre-computed)" in merged


@pytest.mark.asyncio
async def test_non_qwen_model_keeps_trailing_system_messages():
    """Providers that support mid-conversation system messages are untouched.

    The Qwen merge is gated on the model name so OpenAI/Anthropic/Gemini
    requests keep the classifier strategy exactly where it was before the
    merge existed.
    """
    llm = RecordingLLM(AIMessage(content="jawaban final"))
    object.__setattr__(llm, "_esdc_model_name", "gpt-4o")
    agent = create_agent(
        cast(BaseChatModel, llm), tools=[], checkpointer=None, context_length=8000
    )

    await agent.ainvoke(
        {"messages": [HumanMessage(content="halo")]},
        config={"recursion_limit": 10},
    )

    assert llm.seen, "agent must invoke the LLM"
    request = llm.seen[-1]

    roles = [type(m).__name__ for m in request]
    assert roles[0] == "SystemMessage"
    # Not a Qwen template: the classifier's strategy guidance keeps its
    # trailing placement, as it did before the merge was introduced.
    assert "SystemMessage" in roles[1:], f"expected trailing system message: {roles}"


@pytest.mark.asyncio
async def test_manual_trailing_system_messages_are_merged():
    """Any system message in history (compaction summaries, nudges) merges."""
    llm = RecordingLLM(AIMessage(content="jawaban final"))
    agent = create_agent(
        cast(BaseChatModel, llm), tools=[], checkpointer=None, context_length=8000
    )

    from langchain_core.messages import SystemMessage

    await agent.ainvoke(
        {
            "messages": [
                HumanMessage(content="halo"),
                SystemMessage(content="strategy-1"),
                SystemMessage(content="strategy-2"),
            ]
        },
        config={"recursion_limit": 10},
    )

    assert llm.seen, "agent must invoke the LLM"
    request = llm.seen[-1]
    roles = [type(m).__name__ for m in request]
    assert roles[0] == "SystemMessage"
    assert "SystemMessage" not in roles[1:]

    merged = str(request[0].content)
    assert "strategy-1" in merged
    assert "strategy-2" in merged


@pytest.mark.asyncio
async def test_ordinary_request_excludes_strategic_skill_in_system_prompt():
    llm = RecordingLLM(AIMessage(content="jawaban"))
    agent = create_agent(
        cast(BaseChatModel, llm), tools=[], checkpointer=None, context_length=8000
    )

    await agent.ainvoke(
        {"messages": [HumanMessage(content="halo")]},
        config={"recursion_limit": 10},
    )

    system_text = str(llm.seen[-1][0].content)
    assert "# Strategic Analysis Skill" not in system_text

    await agent.ainvoke(
        {"messages": [HumanMessage(content="berapa potensi lapangan Duri?")]},
        config={"recursion_limit": 10},
    )

    resource_system_text = str(llm.seen[-1][0].content)
    assert "# Strategic Analysis Skill" not in resource_system_text


@pytest.mark.asyncio
async def test_explicit_strategic_request_includes_strategic_skill_in_system_prompt():
    llm = RecordingLLM(AIMessage(content="jawaban"))
    agent = create_agent(
        cast(BaseChatModel, llm), tools=[], checkpointer=None, context_length=8000
    )

    await agent.ainvoke(
        {"messages": [HumanMessage(content="buat analisis strategis nasional")]},
        config={"recursion_limit": 10},
    )

    system_text = str(llm.seen[-1][0].content)
    assert "# Strategic Analysis Skill" in system_text


@pytest.mark.asyncio
async def test_strategic_prompt_is_much_larger_than_ordinary_prompt():
    ordinary_llm = RecordingLLM(AIMessage(content="jawaban"))
    ordinary_agent = create_agent(
        cast(BaseChatModel, ordinary_llm),
        tools=[],
        checkpointer=None,
        context_length=8000,
    )
    await ordinary_agent.ainvoke(
        {"messages": [HumanMessage(content="halo")]},
        config={"recursion_limit": 10},
    )

    strategic_llm = RecordingLLM(AIMessage(content="jawaban"))
    strategic_agent = create_agent(
        cast(BaseChatModel, strategic_llm),
        tools=[],
        checkpointer=None,
        context_length=8000,
    )
    await strategic_agent.ainvoke(
        {"messages": [HumanMessage(content="buat analisis strategis nasional")]},
        config={"recursion_limit": 10},
    )

    ordinary_len = len(str(ordinary_llm.seen[-1][0].content))
    strategic_len = len(str(strategic_llm.seen[-1][0].content))

    assert strategic_len > ordinary_len
    assert strategic_len - ordinary_len > 10_000
