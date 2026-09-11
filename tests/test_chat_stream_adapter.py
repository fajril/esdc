"""Tests for ESDCChatApp._stream_response adapter over astream_agent_events."""

from typing import cast

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable


def _make_app():
    from esdc.chat.app import ESDCChatApp

    app = ESDCChatApp()
    app._agent = cast(Runnable, object())  # sentinel; adapter passes it through
    app._thread_id = "esdc-test1234"
    return app


async def _collect(app, monkeypatch, events):
    async def fake_stream(agent, messages, config=None):
        assert agent is app._agent
        assert config is not None
        assert config["configurable"]["thread_id"] == "esdc-test1234"
        for e in events:
            yield e

    import esdc.chat.event_streamer as es

    monkeypatch.setattr(es, "astream_agent_events", fake_stream)
    return [chunk async for chunk in app._stream_response("hello")]


@pytest.mark.asyncio
async def test_tokens_and_reasoning_pass_through(monkeypatch):
    app = _make_app()
    chunks = await _collect(
        app,
        monkeypatch,
        [
            {"type": "reasoning_token", "content": "thinking..."},
            {"type": "token", "content": "Hel"},
            {"type": "token", "content": "lo"},
        ],
    )
    assert {"type": "reasoning_token", "content": "thinking..."} in chunks
    assert {"type": "token", "content": "Hel"} in chunks


@pytest.mark.asyncio
async def test_message_complete_with_tool_calls_emits_tool_call(monkeypatch):
    app = _make_app()
    ai = AIMessage(
        content="",
        tool_calls=[
            {"name": "execute_sql", "args": {"query": "SELECT 1"}, "id": "tc1"}
        ],
    )
    chunks = await _collect(
        app,
        monkeypatch,
        [
            {"type": "message_complete", "ai_message": ai},
            {
                "type": "tool_result",
                "tool_name": "execute_sql",
                "result": "1 row",
                "tool_call_id": "tc1",
            },
        ],
    )
    tool_calls = [c for c in chunks if c["type"] == "tool_call"]
    assert tool_calls == [
        {"type": "tool_call", "tool": "execute_sql", "args": {"query": "SELECT 1"}}
    ]
    tool_results = [c for c in chunks if c["type"] == "tool_result"]
    assert tool_results == [
        {
            "type": "tool_result",
            "tool": "execute_sql",
            "result": "1 row",
            "sql": "SELECT 1",
        }
    ]


@pytest.mark.asyncio
async def test_final_message_and_messages_state(monkeypatch):
    app = _make_app()
    ai = AIMessage(content="The answer is 42.")
    chunks = await _collect(
        app, monkeypatch, [{"type": "message_complete", "ai_message": ai}]
    )
    assert {"type": "message", "content": "The answer is 42."} in chunks
    states = [c for c in chunks if c["type"] == "messages_state"]
    assert states and states[-1]["message_count"] == 2  # human + ai


@pytest.mark.asyncio
async def test_recursion_error_becomes_friendly_message(monkeypatch):
    app = _make_app()
    chunks = await _collect(
        app, monkeypatch, [{"type": "recursion_error", "message": "too deep"}]
    )
    messages = [c for c in chunks if c["type"] == "message"]
    assert len(messages) == 1
    assert "too deep" in messages[0]["content"]


def test_run_agent_stream_is_deleted():
    import esdc.chat.agent as agent_mod

    assert not hasattr(agent_mod, "run_agent_stream")


@pytest.mark.asyncio
async def test_message_complete_with_display_name_tool_call_emits_sql(monkeypatch):
    """Tools register with display names, not python names.

    e.g. 'SQL Executor', not 'execute_sql'. The adapter must match on the
    real runtime name.
    """
    app = _make_app()
    ai = AIMessage(
        content="",
        tool_calls=[
            {"name": "SQL Executor", "args": {"query": "SELECT 2"}, "id": "tc9"}
        ],
    )
    chunks = await _collect(
        app,
        monkeypatch,
        [
            {"type": "message_complete", "ai_message": ai},
            {
                "type": "tool_result",
                "tool_name": "SQL Executor",
                "result": "1 row",
                "tool_call_id": "tc9",
            },
        ],
    )
    tool_results = [c for c in chunks if c["type"] == "tool_result"]
    assert tool_results == [
        {
            "type": "tool_result",
            "tool": "SQL Executor",
            "result": "1 row",
            "sql": "SELECT 2",
        }
    ]
