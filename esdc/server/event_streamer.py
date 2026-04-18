"""Shared event streamer for LangGraph agent token-level streaming.

Wraps agent.astream_events(version="v2") and yields normalized event dicts
that both Responses API and Chat Completions API can consume.

Model-agnostic: does not filter or transform content. Thinking tags
pass through as-is in content tokens — the consuming frontend
(e.g. OpenWebUI) handles detection and rendering.
If a model emits a `reasoning_content` field on AIMessageChunk,
it is yielded as a separate `reasoning_token` event.

Handles GraphRecursionError by yielding a `recursion_error` event
so consumers can emit a user-friendly response instead of a raw error.
"""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncGenerator
from typing import Any

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from esdc.chat.external_tools import is_external_tool_marker, parse_external_tool_name

logger = logging.getLogger("esdc.server.event_streamer")

DEFAULT_RECURSION_LIMIT = 100


async def astream_agent_events(
    agent,
    messages: list,
    config: RunnableConfig | None = None,
) -> AsyncGenerator[dict[str, Any], None]:
    """Stream agent events with real token-level streaming.

    Yields normalized event dicts:
      {"type": "token", "content": "..."}
          Individual LLM token from on_chat_model_stream.
          Content is passthrough — no filtering applied.

      {"type": "reasoning_token", "content": "..."}
          Token from AIMessageChunk.reasoning_content, if present.
          Models that don't emit this field simply won't yield these events.

      {"type": "message_complete", "ai_message": AIMessage}
          Full AIMessage after on_chat_model_end, with content and tool_calls.
          Content is unfiltered — same as what was streamed via tokens.

      {"type": "tool_result", "tool_name": "...",
       "result": "...", "tool_call_id": "..."}
          Tool execution result from on_tool_end.
          tool_call_id is correlated from the preceding message_complete
          event's tool_calls via a FIFO queue.

      {"type": "external_tool_call", "tool_name": "...",
       "tool_call_id": "...", "marker_tool_name": "..."}
          External tool call detected (e.g. OpenTerminal run_command).
          The tool was not executed server-side; instead, a marker was
          returned. The Responses API wrapper converts this to a
          function_call output item for OpenWebUI to handle.

      {"type": "context_metadata", "metadata": {...}}
          Context management metadata from on_chain_end.

      {"type": "recursion_error", "message": "..."}
          Emitted when LangGraph's recursion limit is exceeded.
          Consumers should stop and return a user-friendly error.
    """
    if config is None:
        config = RunnableConfig(
            configurable={"thread_id": f"stream_{time.time_ns()}"},
            recursion_limit=DEFAULT_RECURSION_LIMIT,
        )

    stream_start = time.perf_counter()
    first_token_time: float | None = None
    event_count = 0
    token_count = 0
    pending_tool_calls: list[dict[str, Any]] = []

    try:
        async for event in agent.astream_events(
            {"messages": messages},
            config=config,
            version="v2",
        ):
            event_count += 1
            event_type = event.get("event")
            data = event.get("data", {})

            if event_type in ("on_chat_model_stream", "on_llm_stream"):
                chunk = data.get("chunk")
                if not chunk:
                    continue

                content = ""
                if hasattr(chunk, "content"):
                    content = chunk.content
                elif isinstance(chunk, str):
                    content = chunk
                elif chunk:
                    content = str(chunk)

                if content:
                    if first_token_time is None:
                        first_token_time = time.perf_counter()
                        ttft_ms = (first_token_time - stream_start) * 1000
                        logger.debug("[STREAM_START] ttft=%.2fms", ttft_ms)

                    token_count += 1
                    yield {"type": "token", "content": content}

                reasoning_content = chunk.additional_kwargs.get(
                    "reasoning_content"
                ) or getattr(chunk, "reasoning_content", None)
                if reasoning_content:
                    token_count += 1
                    yield {"type": "reasoning_token", "content": reasoning_content}

            elif event_type == "on_chat_model_end":
                output = data.get("output")
                if output and isinstance(output, AIMessage):
                    cleaned = AIMessage(
                        content=output.content or "",
                        additional_kwargs=output.additional_kwargs
                        if hasattr(output, "additional_kwargs")
                        else {},
                    )
                    if hasattr(output, "tool_calls") and output.tool_calls:
                        cleaned.tool_calls = output.tool_calls  # type: ignore[attr-defined]
                        for tc in output.tool_calls:
                            pending_tool_calls.append(
                                {"name": tc.get("name", ""), "id": tc.get("id", "")}
                            )
                    if hasattr(output, "usage_metadata") and output.usage_metadata:
                        cleaned.usage_metadata = output.usage_metadata  # type: ignore[attr-defined]
                    if hasattr(output, "id") and output.id:
                        cleaned.id = output.id  # type: ignore[attr-defined]

                    yield {"type": "message_complete", "ai_message": cleaned}

            elif event_type == "on_tool_end":
                tool_result = data.get("output")
                tool_name = event.get("name", "unknown")
                tool_call_id = ""

                if pending_tool_calls:
                    stored_tc = pending_tool_calls.pop(0)
                    tool_call_id = stored_tc.get("id", "")
                else:
                    logger.warning(
                        "[STREAM] on_tool_end for %s but no pending tool_call_id",
                        tool_name,
                    )

                result_str = str(tool_result) if tool_result else ""

                # Check if this is an external tool call marker
                if is_external_tool_marker(result_str):
                    external_tool_name = parse_external_tool_name(result_str)
                    logger.info(
                        "[STREAM] External tool call detected: %s (call_id=%s)",
                        external_tool_name,
                        tool_call_id,
                    )
                    yield {
                        "type": "external_tool_call",
                        "tool_name": tool_name,
                        "tool_call_id": tool_call_id,
                        "marker_tool_name": external_tool_name or tool_name,
                    }
                    continue

                yield {
                    "type": "tool_result",
                    "tool_name": tool_name,
                    "result": result_str,
                    "tool_call_id": tool_call_id,
                }

            elif event_type == "on_chain_end":
                node_name = event.get("name", "")
                if node_name == "manage_context":
                    output_data = data.get("output", {})
                    context_metadata = output_data.get("context_metadata")
                    if context_metadata:
                        yield {
                            "type": "context_metadata",
                            "metadata": context_metadata,
                        }

    except Exception as e:
        from langgraph.errors import GraphRecursionError

        if isinstance(e, GraphRecursionError):
            recursion_limit = config.get("recursion_limit", "unknown")  # type: ignore[arg-type]
            logger.error(
                "[STREAM] GraphRecursionError | limit=%s | events=%d | tokens=%d",
                recursion_limit,
                event_count,
                token_count,
            )
            yield {
                "type": "recursion_error",
                "message": (
                    "Maaf, permintaan Anda memerlukan terlalu banyak"
                    " langkah pemrosesan. Silakan sederhanakan"
                    " pertanyaan atau coba lagi."
                ),
            }
            return

        raise

    total_ms = (time.perf_counter() - stream_start) * 1000
    logger.debug(
        "[STREAM_END] total=%.2fms events=%d tokens=%d",
        total_ms,
        event_count,
        token_count,
    )
