"""Tests for provider-safe tool-name translation."""

import re
from typing import cast

import pytest
from langchain.tools import tool
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from esdc.chat.agent import create_agent
from esdc.chat.tool_names import ToolNameMap, sanitize_tool_name

_VALID_TOOL_NAME = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class TestSanitizeToolName:
    def test_spaces_become_snake_case(self):
        assert sanitize_tool_name("SQL Executor") == "sql_executor"

    def test_invalid_characters_are_collapsed(self):
        assert sanitize_tool_name("Entity  Resolver!") == "entity_resolver"

    def test_camel_case_boundaries_become_underscores(self):
        assert sanitize_tool_name("myTool") == "my_tool"
        assert sanitize_tool_name("SQLExecutor") == "sql_executor"

    def test_already_valid_name_is_unchanged(self):
        assert sanitize_tool_name("execute_sql") == "execute_sql"

    def test_empty_name_falls_back(self):
        assert sanitize_tool_name("   ") == "tool"

    def test_result_is_capped_at_openai_limit(self):
        assert len(sanitize_tool_name("a" * 100)) == 64


class TestToolNameMap:
    def test_round_trip(self):
        names = ToolNameMap(["SQL Executor"])
        assert names.to_wire("SQL Executor") == "sql_executor"
        assert names.to_display("sql_executor") == "SQL Executor"

    def test_unknown_names_pass_through(self):
        names = ToolNameMap(["SQL Executor"])
        assert names.to_wire("unknown_tool") == "unknown_tool"
        assert names.to_display("unknown_tool") == "unknown_tool"

    def test_case_and_separator_variants_resolve(self):
        names = ToolNameMap(["SQL Executor", "Entity Resolver"])
        assert names.to_display("sql_executor") == "SQL Executor"
        assert names.to_display("entity_resolver") == "Entity Resolver"
        assert names.to_wire("entity_resolver") == "entity_resolver"

    def test_collisions_get_distinct_wire_names(self):
        names = ToolNameMap(["SQL Executor", "SQL_Executor"])
        first = names.to_wire("SQL Executor")
        second = names.to_wire("SQL_Executor")
        assert first != second
        assert names.to_display(first) == "SQL Executor"
        assert names.to_display(second) == "SQL_Executor"

    def test_wire_tools_clone_names_without_mutating_originals(self):
        @tool("SQL Executor")
        def execute_sql(query: str) -> str:
            """Run SQL."""
            return query

        names = ToolNameMap([execute_sql.name])
        wire_tools = names.wire_tools([execute_sql])

        assert wire_tools[0].name == "sql_executor"
        assert execute_sql.name == "SQL Executor"

    def test_messages_translate_tool_calls_and_results(self):
        names = ToolNameMap(["SQL Executor"])
        ai = AIMessage(
            content="",
            tool_calls=[
                {"name": "SQL Executor", "args": {"query": "SELECT 1"}, "id": "c1"}
            ],
        )
        tool_message = ToolMessage(
            content="1", tool_call_id="c1", name="SQL Executor"
        )

        wire_ai, wire_tool = names.messages_to_wire([ai, tool_message])

        assert cast(AIMessage, wire_ai).tool_calls[0]["name"] == "sql_executor"
        assert cast(ToolMessage, wire_tool).name == "sql_executor"

        restored = names.message_from_wire(wire_ai)
        assert cast(AIMessage, restored).tool_calls[0]["name"] == "SQL Executor"


class RecordingLLM:
    """LLM double that records bound tools and the messages it receives."""

    def __init__(self) -> None:
        self.bound_tools: list = []
        self.seen_messages: list[list] = []
        self._calls = 0

    def bind_tools(self, tools):
        self.bound_tools = list(tools)
        return self

    async def ainvoke(self, messages, **kwargs):
        self.seen_messages.append(list(messages))
        self._calls += 1
        if self._calls == 1:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "sql_executor",
                        "args": {"query": "SELECT 1"},
                        "id": "call_1",
                    }
                ],
            )
        return AIMessage(content="Done.")


@pytest.mark.asyncio
async def test_agent_binds_wire_names_and_restores_display_names():
    @tool("SQL Executor")
    def execute_sql(query: str) -> str:
        """Run SQL."""
        return f"ran:{query}"

    llm = RecordingLLM()
    agent = create_agent(
        cast(BaseChatModel, llm),
        tools=[execute_sql],
        checkpointer=None,
        context_length=8000,
    )

    final_state = await agent.ainvoke(
        {"messages": [HumanMessage(content="explain economic issues")]},
        config={"recursion_limit": 20},
    )

    # The provider only sees valid OpenAI-style names.
    assert llm.bound_tools
    for bound_tool in llm.bound_tools:
        assert _VALID_TOOL_NAME.match(bound_tool.name), bound_tool.name

    # The second LLM call receives wire names in history.
    second_call = llm.seen_messages[1]
    history_ai = [m for m in second_call if isinstance(m, AIMessage)]
    assert history_ai[-1].tool_calls[0]["name"] == "sql_executor"
    history_tool = [m for m in second_call if isinstance(m, ToolMessage)]
    assert history_tool[-1].name == "sql_executor"

    # Internal state and client-facing output use the proper display name.
    state_ai = [
        m
        for m in final_state["messages"]
        if isinstance(m, AIMessage) and m.tool_calls
    ]
    assert state_ai[0].tool_calls[0]["name"] == "SQL Executor"
    state_tool = [m for m in final_state["messages"] if isinstance(m, ToolMessage)]
    assert state_tool[0].name == "SQL Executor"
    assert str(state_tool[0].content) == "ran:SELECT 1"
