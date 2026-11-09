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
from esdc.chat.tool_names import sanitize_tool_name


def _wire_names(*names: str) -> list[str]:
    """Return the provider-facing names bound for the given display names."""
    return [sanitize_tool_name(name) for name in names]


class RecordingLLM:
    """Minimal LLM stub that records every message list it receives."""

    def __init__(self, response: AIMessage):
        self._response = response
        self.seen: list[list] = []
        self.bound_tool_names: list[list[str]] = []

    def bind_tools(self, tools):
        self.bound_tool_names.append([tool.name for tool in tools])
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
    assert "# Strategic analysis playbook" not in system_text

    await agent.ainvoke(
        {"messages": [HumanMessage(content="berapa potensi lapangan Duri?")]},
        config={"recursion_limit": 10},
    )

    resource_system_text = str(llm.seen[-1][0].content)
    assert "# Strategic analysis playbook" not in resource_system_text


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
    assert "# Strategic analysis playbook" in system_text


@pytest.mark.asyncio
async def test_strategic_skill_is_replaced_by_ksmi_skill_on_follow_up():
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

    strategic_text = str(strategic_llm.seen[-1][0].content)
    assert "# Strategic analysis playbook" in strategic_text

    await strategic_agent.ainvoke(
        {"messages": [HumanMessage(content="Bolehkah E3 ke E4?")]},
        config={"recursion_limit": 10},
    )
    follow_up_text = str(strategic_llm.seen[-1][0].content)
    assert "# Strategic analysis playbook" not in follow_up_text
    assert "# KSMI reasoning playbook" in follow_up_text


class NamedTool:
    def __init__(self, name: str):
        self.name = name


def _route_tools() -> list[NamedTool]:
    return [
        NamedTool(name)
        for name in (
            "Simple Data Query",
            "Knowledge Traversal",
            "Semantic Search",
            "SQL Executor",
            "Document Search",
            "Document Reader",
            "Document Aggregator",
            "Spatial Resolver",
        )
    ]


@pytest.mark.asyncio
async def test_follow_up_route_binds_ksmi_only_and_narrative_route_binds_semantic_sql():
    llm = RecordingLLM(AIMessage(content="jawaban"))
    agent = create_agent(
        cast(BaseChatModel, llm),
        tools=_route_tools(),
        checkpointer=None,
        context_length=8000,
    )

    await agent.ainvoke(
        {"messages": [HumanMessage(content="buat analisis strategis nasional")]},
        config={"recursion_limit": 10},
    )
    await agent.ainvoke(
        {"messages": [HumanMessage(content="Bolehkah E3 ke E4?")]},
        config={"recursion_limit": 10},
    )
    assert llm.bound_tool_names[-1] == _wire_names("Knowledge Traversal")

    await agent.ainvoke(
        {
            "messages": [
                HumanMessage(content="Apa isi GROOVY dan kendala proyek di WK Rokan?")
            ]
        },
        config={"recursion_limit": 10},
    )
    assert llm.bound_tool_names[-1] == _wire_names("Semantic Search", "SQL Executor")


@pytest.mark.asyncio
async def test_simple_reserves_route_excludes_unrelated_tools():
    llm = RecordingLLM(AIMessage(content="jawaban"))
    agent = create_agent(
        cast(BaseChatModel, llm),
        tools=_route_tools(),
        checkpointer=None,
        context_length=8000,
    )
    await agent.ainvoke(
        {"messages": [HumanMessage(content="Berapa cadangan minyak Lapangan Duri tahun 2025?")]},
        config={"recursion_limit": 10},
    )
    assert llm.bound_tool_names[-1] == _wire_names("Simple Data Query")


@pytest.mark.asyncio
async def test_multi_turn_routes_replace_stale_strategy_and_bind_current_tools():
    """A persisted conversation must not let an earlier route direct a new turn."""
    from langchain_core.runnables import RunnableConfig
    from langgraph.checkpoint.memory import InMemorySaver

    llm = RecordingLLM(AIMessage(content="jawaban"))
    agent = create_agent(
        cast(BaseChatModel, llm),
        tools=_route_tools(),
        checkpointer=InMemorySaver(),
        context_length=8000,
    )
    config: RunnableConfig = {
        "configurable": {"thread_id": "route-thread"},
        "recursion_limit": 10,
    }

    await agent.ainvoke(
        {"messages": [HumanMessage(content="Bolehkah E3 ke E4?")]}, config=config
    )
    assert llm.bound_tool_names[-1] == _wire_names("Knowledge Traversal")
    assert "# KSMI reasoning playbook" in str(llm.seen[-1][0].content)

    await agent.ainvoke(
        {
            "messages": [
                HumanMessage(content="Berapa cadangan minyak Lapangan Duri tahun 2025?")
            ]
        },
        config=config,
    )
    assert llm.bound_tool_names[-1] == _wire_names("Simple Data Query")
    current_system = str(llm.seen[-1][0].content)
    assert "# KSMI reasoning playbook" not in current_system
    assert "Mandatory first action" not in current_system
    humans = [m.content for m in llm.seen[-1] if isinstance(m, HumanMessage)]
    assert "Bolehkah E3 ke E4?" in humans
    assert "Berapa cadangan minyak Lapangan Duri tahun 2025?" in humans

    await agent.ainvoke(
        {
            "messages": [
                HumanMessage(content="dokumen apa saja yang membahas separator?")
            ]
        },
        config=config,
    )
    assert llm.bound_tool_names[-1] == _wire_names(
        "Document Search",
        "Document Reader",
        "Document Aggregator",
    )

    await agent.ainvoke(
        {
            "messages": [
                HumanMessage(content="berapa cadangan minyak Lapangan Duri tahun 2023?")
            ]
        },
        config=config,
    )
    await agent.ainvoke(
        {"messages": [HumanMessage(content="bagaimana tahun 2024?")]}, config=config
    )
    assert llm.bound_tool_names[-1] == _wire_names("Simple Data Query")
    anaphoric_system = str(llm.seen[-1][0].content)
    assert "field_name: 'duri'" in anaphoric_system
    assert "report_year: '2024'" in anaphoric_system


@pytest.mark.asyncio
async def test_production_question_after_ksmi_turn_has_no_mandatory_ksmi():
    """A complete factual turn must not inherit the prior KSMI mandate."""
    from langchain_core.runnables import RunnableConfig
    from langgraph.checkpoint.memory import InMemorySaver

    llm = RecordingLLM(AIMessage(content="jawaban"))
    agent = create_agent(
        cast(BaseChatModel, llm),
        tools=_route_tools(),
        checkpointer=InMemorySaver(),
        context_length=8000,
    )
    config: RunnableConfig = {
        "configurable": {"thread_id": "production-thread"},
        "recursion_limit": 10,
    }

    await agent.ainvoke(
        {"messages": [HumanMessage(content="Bolehkah E3 ke E4?")]}, config=config
    )
    assert llm.bound_tool_names[-1] == _wire_names("Knowledge Traversal")

    await agent.ainvoke(
        {"messages": [HumanMessage(content="Berapa produksi sumur tahun 2025?")]},
        config=config,
    )
    bound = llm.bound_tool_names[-1]
    assert sanitize_tool_name("Knowledge Traversal") not in bound
    assert sanitize_tool_name("SQL Executor") in bound
    current_system = str(llm.seen[-1][0].content)
    assert "# KSMI reasoning playbook" not in current_system
    assert "Mandatory first action" not in current_system
    humans = [m.content for m in llm.seen[-1] if isinstance(m, HumanMessage)]
    assert "Bolehkah E3 ke E4?" in humans


def _configured_openterminal(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "esdc.chat.agent.get_openterminal_tools",
        lambda: [
            NamedTool("Shell Executor"),
            NamedTool("Code Interpreter"),
        ],
    )


@pytest.mark.asyncio
async def test_visualization_binds_configured_openterminal_tools_additively(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configured_openterminal(monkeypatch)
    llm = RecordingLLM(AIMessage(content="jawaban"))
    agent = create_agent(
        cast(BaseChatModel, llm),
        tools=_route_tools(),
        checkpointer=None,
        context_length=8000,
    )

    await agent.ainvoke(
        {"messages": [HumanMessage(content="Buat peta sebaran lapangan Duri")]},
        config={"recursion_limit": 10},
    )
    bound = llm.bound_tool_names[-1]
    assert sanitize_tool_name("Shell Executor") in bound
    assert sanitize_tool_name("Code Interpreter") in bound
    assert sanitize_tool_name("SQL Executor") in bound

    await agent.ainvoke(
        {
            "messages": [
                HumanMessage(content="Berapa cadangan minyak Lapangan Duri tahun 2025?")
            ]
        },
        config={"recursion_limit": 10},
    )
    assert llm.bound_tool_names[-1] == _wire_names("Simple Data Query")


@pytest.mark.asyncio
async def test_mixed_negated_map_positive_chart_binds_and_allows_rendering_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A negated map must not suppress an explicitly requested chart."""
    _configured_openterminal(monkeypatch)
    llm = RecordingLLM(AIMessage(content="jawaban"))
    agent = create_agent(
        cast(BaseChatModel, llm),
        tools=_route_tools(),
        checkpointer=None,
        context_length=8000,
    )

    result = await agent.ainvoke(
        {
            "messages": [
                HumanMessage(
                    content="Jangan buat peta, tapi tampilkan grafik lapangan Duri"
                )
            ]
        },
        config={"recursion_limit": 10},
    )
    bound = llm.bound_tool_names[-1]
    assert sanitize_tool_name("Shell Executor") in bound
    assert sanitize_tool_name("Code Interpreter") in bound
    assert "Shell Executor" in result["allowed_tools"]

    ordinary = await agent.ainvoke(
        {
            "messages": [
                HumanMessage(content="Berapa cadangan minyak Lapangan Duri tahun 2025?")
            ]
        },
        config={"recursion_limit": 10},
    )
    assert sanitize_tool_name("Shell Executor") not in llm.bound_tool_names[-1]
    assert "Shell Executor" not in ordinary["allowed_tools"]

    negated = await agent.ainvoke(
        {"messages": [HumanMessage(content="Jangan buat peta lapangan Duri")]},
        config={"recursion_limit": 10},
    )
    assert sanitize_tool_name("Shell Executor") not in llm.bound_tool_names[-1]
    assert "Shell Executor" not in negated["allowed_tools"]


@pytest.mark.asyncio
async def test_fully_negated_visualization_does_not_bind_rendering_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configured_openterminal(monkeypatch)
    llm = RecordingLLM(AIMessage(content="jawaban"))
    agent = create_agent(
        cast(BaseChatModel, llm),
        tools=_route_tools(),
        checkpointer=None,
        context_length=8000,
    )

    result = await agent.ainvoke(
        {
            "messages": [
                HumanMessage(
                    content="Jangan buat peta, grafik, atau gambar Rokan"
                )
            ]
        },
        config={"recursion_limit": 10},
    )

    assert sanitize_tool_name("Shell Executor") not in llm.bound_tool_names[-1]
    assert sanitize_tool_name("Code Interpreter") not in llm.bound_tool_names[-1]
    assert "Shell Executor" not in result["allowed_tools"]


@pytest.mark.asyncio
async def test_visualization_without_openterminal_binds_no_phantom_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("esdc.chat.agent.get_openterminal_tools", lambda: None)
    llm = RecordingLLM(AIMessage(content="jawaban"))
    agent = create_agent(
        cast(BaseChatModel, llm),
        tools=_route_tools(),
        checkpointer=None,
        context_length=8000,
    )

    await agent.ainvoke(
        {"messages": [HumanMessage(content="Buat peta sebaran lapangan Duri")]},
        config={"recursion_limit": 10},
    )
    bound = llm.bound_tool_names[-1]
    assert sanitize_tool_name("Shell Executor") not in bound
    assert sanitize_tool_name("Code Interpreter") not in bound


class _RecordingTool:
    def __init__(self, name: str) -> None:
        self.name = name
        self.calls: list[object] = []

    async def ainvoke(self, args):
        self.calls.append(args)
        return f"{self.name} output"


class _ScriptedLLM:
    def __init__(self, responses) -> None:
        self._responses = list(responses)
        self.bound_tool_names: list[list[str]] = []
        self.seen: list[list] = []

    def bind_tools(self, tools):
        self.bound_tool_names.append([tool.name for tool in tools])
        return self

    async def ainvoke(self, messages, **kwargs):
        self.seen.append(list(messages))
        return self._responses.pop(0)


@pytest.mark.asyncio
async def test_visualization_tool_call_is_executed_not_blocked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A configured rendering tool must be executable, not only bound."""
    shell = _RecordingTool("Shell Executor")
    monkeypatch.setattr("esdc.chat.agent.get_openterminal_tools", lambda: [shell])
    first = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "Shell Executor",
                "args": {"command": "echo hi"},
                "id": "call-1",
                "type": "tool_call",
            }
        ],
    )
    llm = _ScriptedLLM([first, AIMessage(content="selesai")])
    agent = create_agent(
        cast(BaseChatModel, llm),
        tools=_route_tools(),
        checkpointer=None,
        context_length=8000,
    )

    result = await agent.ainvoke(
        {"messages": [HumanMessage(content="Buat peta sebaran lapangan Duri")]},
        config={"recursion_limit": 10},
    )

    assert shell.calls == [{"command": "echo hi"}]
    tool_messages = [
        m for m in result["messages"] if type(m).__name__ == "ToolMessage"
    ]
    assert any(
        getattr(message, "name", None) == "Shell Executor"
        and "output" in str(message.content)
        for message in tool_messages
    )


@pytest.mark.asyncio
async def test_arbitrary_external_tools_are_not_default_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("esdc.chat.agent.get_openterminal_tools", lambda: None)
    llm = RecordingLLM(AIMessage(content="jawaban"))
    agent = create_agent(
        cast(BaseChatModel, llm),
        tools=_route_tools(),
        checkpointer=None,
        context_length=8000,
        external_tool_names={"Some External"},
        external_tools=[NamedTool("Some External")],
    )

    await agent.ainvoke(
        {"messages": [HumanMessage(content="Buat peta sebaran lapangan Duri")]},
        config={"recursion_limit": 10},
    )
    assert sanitize_tool_name("Some External") not in llm.bound_tool_names[-1]

    await agent.ainvoke(
        {
            "messages": [
                HumanMessage(content="Berapa cadangan minyak Lapangan Duri tahun 2025?")
            ]
        },
        config={"recursion_limit": 10},
    )
    assert sanitize_tool_name("Some External") not in llm.bound_tool_names[-1]
