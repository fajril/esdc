"""Tests for tool_node loop detection in agent.py.

See docs/plans/2026-07-13-improve-document-search-usage.md #5: the
2026-07-13 death-spiral called Entity Resolver 20+ times with identical
arguments before ever calling search_documents. tool_node now blocks the
3rd+ identical (tool_name, args) call instead of re-executing it, and
returns a redirect ToolMessage so the LLM tries something else.
"""

import json
from typing import cast

import pytest
from langchain.tools import tool
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from esdc.chat.agent import create_agent

_TOOL_CALL_COUNTER: dict[str, int] = {}


@tool("Entity Resolver")
def _fake_entity_resolver(name: str) -> str:
    """Fake stand-in for the real Entity Resolver tool."""
    _TOOL_CALL_COUNTER["n"] = _TOOL_CALL_COUNTER.get("n", 0) + 1
    return f"resolved-{_TOOL_CALL_COUNTER['n']}"


class FakeLLM:
    """Minimal LLM stub: bind_tools returns self, ainvoke pops queued responses."""

    def __init__(self, responses: list[AIMessage]):
        self._responses = list(responses)

    def bind_tools(self, tools):
        return self

    async def ainvoke(self, messages, **kwargs):
        if not self._responses:
            return AIMessage(content="No more responses queued.")
        return self._responses.pop(0)


def _identical_tool_call(call_id: str) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": "Entity Resolver",
                "args": {"name": "POD I Abadi Revisi 1"},
                "id": call_id,
            }
        ],
    )


@pytest.mark.asyncio
async def test_third_identical_tool_call_is_blocked():
    """3 identical Entity Resolver calls -> the 3rd is blocked, not executed."""
    _TOOL_CALL_COUNTER.clear()

    responses = [
        _identical_tool_call("call_1"),
        _identical_tool_call("call_2"),
        _identical_tool_call("call_3"),
        AIMessage(content="Final answer based on what I have."),
    ]
    llm = FakeLLM(responses)

    agent = create_agent(
        cast(BaseChatModel, llm),
        tools=[_fake_entity_resolver],
        checkpointer=None,
        context_length=8000,
    )

    final_state = await agent.ainvoke(
        {"messages": [HumanMessage(content="test query about POD revisions")]},
        config={"recursion_limit": 50},
    )

    # Only 2 identical calls actually reached the tool.
    assert _TOOL_CALL_COUNTER.get("n", 0) == 2

    tool_messages = [m for m in final_state["messages"] if isinstance(m, ToolMessage)]
    blocked = [m for m in tool_messages if m.tool_call_id == "call_3"]
    assert len(blocked) == 1
    assert "REPEATED CALL BLOCKED" in str(blocked[0].content)
    assert "Entity Resolver" in str(blocked[0].content)

    # The two executed calls returned normal (non-blocked) results.
    executed = [m for m in tool_messages if m.tool_call_id in ("call_1", "call_2")]
    assert len(executed) == 2
    for m in executed:
        assert "REPEATED CALL BLOCKED" not in str(m.content)


@pytest.mark.asyncio
async def test_different_args_are_not_blocked():
    """Identical tool name but different args must not trip the loop guard."""
    _TOOL_CALL_COUNTER.clear()

    responses = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "Entity Resolver",
                    "args": {"name": f"Entity {i}"},
                    "id": f"call_{i}",
                }
            ],
        )
        for i in range(4)
    ] + [AIMessage(content="Done.")]
    llm = FakeLLM(responses)

    agent = create_agent(
        cast(BaseChatModel, llm),
        tools=[_fake_entity_resolver],
        checkpointer=None,
        context_length=8000,
    )

    await agent.ainvoke(
        {"messages": [HumanMessage(content="test query")]},
        config={"recursion_limit": 50},
    )

    # All 4 distinct-args calls should have executed.
    assert _TOOL_CALL_COUNTER.get("n", 0) == 4


@pytest.mark.asyncio
async def test_identical_calls_from_previous_turns_do_not_block():
    """2 identical executions in a previous turn + 1 now -> NOT blocked.

    The loop guard counts only within the current turn (messages after the
    last HumanMessage). A user re-asking the same question in a later turn
    legitimately re-runs the same tool call.
    """
    _TOOL_CALL_COUNTER.clear()

    # Previous turn: two identical Entity Resolver executions, already
    # answered, in the conversation history.
    previous_turn = [
        HumanMessage(content="first ask"),
        _identical_tool_call("old_1"),
        ToolMessage(content="resolved-old", tool_call_id="old_1"),
        _identical_tool_call("old_2"),
        ToolMessage(content="resolved-old", tool_call_id="old_2"),
        AIMessage(content="Answer for the first ask."),
        HumanMessage(content="same question again"),
    ]

    responses = [
        _identical_tool_call("new_1"),
        AIMessage(content="Answer for the second ask."),
    ]
    llm = FakeLLM(responses)

    agent = create_agent(
        cast(BaseChatModel, llm),
        tools=[_fake_entity_resolver],
        checkpointer=None,
        context_length=8000,
    )

    final_state = await agent.ainvoke(
        {"messages": previous_turn},
        config={"recursion_limit": 50},
    )

    # The current-turn call executed for real.
    assert _TOOL_CALL_COUNTER.get("n", 0) == 1
    new_msgs = [
        m
        for m in final_state["messages"]
        if isinstance(m, ToolMessage) and m.tool_call_id == "new_1"
    ]
    assert len(new_msgs) == 1
    assert "REPEATED CALL BLOCKED" not in str(new_msgs[0].content)


def test_count_scoped_to_current_turn():
    """_count_executed_tool_signatures ignores executions before the last HumanMessage."""
    from esdc.chat.agent import (
        _count_executed_tool_signatures,
        _tool_call_signature,
    )

    signature = _tool_call_signature(
        "Entity Resolver", {"name": "POD I Abadi Revisi 1"}
    )

    previous_turn = [
        HumanMessage(content="first ask"),
        _identical_tool_call("old_1"),
        ToolMessage(content="resolved", tool_call_id="old_1"),
        _identical_tool_call("old_2"),
        ToolMessage(content="resolved", tool_call_id="old_2"),
        AIMessage(content="answer"),
    ]
    current_turn = [
        HumanMessage(content="same question again"),
        _identical_tool_call("new_1"),
        ToolMessage(content="resolved", tool_call_id="new_1"),
    ]

    counts = _count_executed_tool_signatures(previous_turn + current_turn)
    assert counts.get(signature, 0) == 1

    # Without a newer HumanMessage, all three count (same turn).
    same_turn = previous_turn + [
        _identical_tool_call("new_1"),
        ToolMessage(content="resolved", tool_call_id="new_1"),
    ]
    counts = _count_executed_tool_signatures(same_turn)
    assert counts.get(signature, 0) == 3


@pytest.mark.asyncio
async def test_tool_call_signature_ignores_arg_key_order():
    """Signature must be stable regardless of dict key ordering / JSON-str args."""
    from esdc.chat.agent import _tool_call_signature

    sig_a = _tool_call_signature("Entity Resolver", {"a": 1, "b": 2})
    sig_b = _tool_call_signature("Entity Resolver", {"b": 2, "a": 1})
    sig_c = _tool_call_signature("Entity Resolver", json.dumps({"a": 1, "b": 2}))
    assert sig_a == sig_b == sig_c
