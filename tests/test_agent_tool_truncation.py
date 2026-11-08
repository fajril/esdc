"""Tests for tool result truncation in agent.py's tool node.

Truncation happens inside `tool_node`, so these drive the real graph with a stub
tool whose result length the test controls. The previous version of this file
reimplemented the `> MAX_TOOL_RESULT_CHARS` branch inside the test body and
asserted its own arithmetic, so it could not fail on any change to `agent.py`.
"""

from typing import cast

import pytest
from langchain.tools import tool
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from esdc.chat.agent import MAX_TOOL_RESULT_CHARS, create_agent

TRUNCATION_NOTICE = (
    f"\n\n[Result truncated to first {MAX_TOOL_RESULT_CHARS} characters "
    "for context efficiency]"
)


@tool("Entity Resolver")
def _echo_tool(count: int) -> str:
    """Stub tool whose observation length is set by the caller."""
    return "x" * count


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


async def _tool_message(count: int) -> ToolMessage:
    """Run one tool call through the real tool node and return its result."""
    llm = FakeLLM(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "Entity Resolver",
                        "args": {"count": count},
                        "id": "call_1",
                    }
                ],
            ),
            AIMessage(content="Done."),
        ]
    )
    agent = create_agent(
        cast(BaseChatModel, llm),
        tools=[_echo_tool],
        checkpointer=None,
        context_length=8000,
    )

    state = await agent.ainvoke(
        {"messages": [HumanMessage(content="test query about POD revisions")]},
        config={"recursion_limit": 50},
    )

    results = [
        m
        for m in state["messages"]
        if isinstance(m, ToolMessage) and m.tool_call_id == "call_1"
    ]
    assert len(results) == 1, (
        f"tool node did not execute the stub call: {state['messages']}"
    )
    return results[0]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "count, truncated",
    [
        (15000, True),  # well over the limit
        (5000, False),  # under the limit
        (MAX_TOOL_RESULT_CHARS, False),  # exactly at the limit is not truncated
        (MAX_TOOL_RESULT_CHARS + 1, True),  # one char over
    ],
)
async def test_tool_node_truncates_only_above_the_limit(count: int, truncated: bool):
    """`tool_node` truncates the observation it hands back to the LLM."""
    content = str((await _tool_message(count)).content)

    if truncated:
        # The tool returned more than the limit, so tool_node must have cut it.
        assert content == "x" * MAX_TOOL_RESULT_CHARS + TRUNCATION_NOTICE
    else:
        # The tool returned exactly what the stub produced: nothing was cut.
        assert content == "x" * count
        assert "[Result truncated" not in content
