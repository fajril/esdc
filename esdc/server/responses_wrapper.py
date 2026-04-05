# Standard library
import json
import logging
import uuid
from collections.abc import AsyncGenerator
from typing import Any

# Third-party
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

# Local
from esdc.chat.agent import create_agent
from esdc.configs import Config
from esdc.providers import create_llm_from_config
from esdc.server.responses_events import (
    create_content_part_added_event,
    create_content_part_done_event,
    create_function_call_arguments_delta_event,
    create_function_call_arguments_done_event,
    create_output_item_added_event,
    create_output_item_done_event,
    create_output_text_delta_event,
    create_output_text_done_event,
    create_response_completed_event,
    create_response_created_event,
    format_sse_event,
)
from esdc.server.responses_models import ResponseInputItem
from esdc.server.stream_utils import chunk_json, chunk_text

logger = logging.getLogger("esdc.server.responses")


class SequenceCounter:
    """Manages monotonically increasing sequence numbers for SSE events."""

    def __init__(self, start: int = 1):
        self._value = start

    def next(self) -> int:
        """Get next sequence number."""
        current = self._value
        self._value += 1
        return current


def convert_responses_input_to_langchain(
    input_data: str | list[ResponseInputItem],
    instructions: str | None = None,
) -> list[Any]:
    """Convert Responses API input to LangChain messages.

    Args:
        input_data: Either a string (single user message) or list of input items
        instructions: Optional system message / instructions

    Returns:
        List of LangChain messages
    """
    messages = []

    # Add system message if instructions provided
    if instructions:
        messages.append(SystemMessage(content=instructions))

    # Handle string input (simple case)
    if isinstance(input_data, str):
        messages.append(HumanMessage(content=input_data))
        return messages

    # Handle list of input items
    for item in input_data:
        if item.type == "message":
            # Handle message item
            role = item.role
            content = item.content

            # Extract text content
            if isinstance(content, str):
                text = content
            elif isinstance(content, list):
                # List of content parts
                texts = []
                for part in content:
                    if isinstance(part, dict):
                        if part.get("type") == "input_text":
                            texts.append(part.get("text", ""))
                    elif hasattr(part, "text"):
                        texts.append(part.text)
                text = "\n".join(texts)
            else:
                text = str(content) if content else ""

            if role == "user":
                messages.append(HumanMessage(content=text))
            elif role == "assistant":
                messages.append(AIMessage(content=text))
            elif role == "system":
                messages.append(SystemMessage(content=text))

        elif item.type == "function_call_output":
            # Tool result from client (for multi-turn conversations)
            # This would be used in stateful mode, but we include it for completeness
            # LangGraph expects this as a ToolMessage
            from langchain_core.messages import ToolMessage

            messages.append(
                ToolMessage(
                    content=item.output or "",
                    tool_call_id=item.call_id or "",
                )
            )

    return messages


def extract_ai_message_from_event(event: dict) -> AIMessage | None:
    """Extract AIMessage from LangGraph event.

    LangGraph events have structure: {node_name: {messages: [...]}}
    We need to look for the 'agent' node which contains AI responses.
    """
    # Check for 'agent' node first (contains AI responses)
    if "agent" in event:
        agent_data = event["agent"]
        if isinstance(agent_data, dict) and "messages" in agent_data:
            messages = agent_data["messages"]
            if messages:
                last_msg = messages[-1]
                if isinstance(last_msg, AIMessage):
                    return last_msg

    # Fallback: check all keys for messages
    for _key, value in event.items():
        if isinstance(value, dict) and "messages" in value:
            messages = value["messages"]
            if messages:
                last_msg = messages[-1]
                if isinstance(last_msg, AIMessage):
                    return last_msg

    return None


def extract_tool_messages_from_event(event: dict) -> list[dict[str, Any]]:
    """Extract tool results from LangGraph event.

    LangGraph's tool_node returns dicts with 'role': 'tool',
    NOT ToolMessage objects.

    Args:
        event: LangGraph event dictionary

    Returns:
        List of tool result dicts from the event
    """
    tool_messages = []

    # Check for 'tools' node (contains tool results)
    if "tools" in event:
        tools_data = event["tools"]
        if isinstance(tools_data, dict) and "messages" in tools_data:
            messages = tools_data["messages"]
            for msg in messages:
                # LangGraph returns dicts with role="tool"
                if isinstance(msg, dict) and msg.get("role") == "tool":
                    tool_messages.append(msg)

    return tool_messages


def extract_content_str(content) -> str:
    """Extract string content from AIMessage content (handles str, list, dict)."""
    if content is None:
        return ""
    if isinstance(content, list):
        return str(content[0]) if content else ""
    elif isinstance(content, dict):
        return json.dumps(content, indent=2)
    else:
        return str(content)


def generate_item_id(prefix: str = "msg") -> str:
    """Generate a unique item ID with proper prefix.

    Prefixes:
    - msg: message items
    - fc: function_call items
    - fco: function_call_output items
    """
    return f"{prefix}_{uuid.uuid4().hex[:24]}"


async def generate_responses_stream(
    input_messages: str | list[ResponseInputItem],
    model: str = "esdc-agent",
    instructions: str | None = None,
    tools: list[dict[str, Any]] | None = None,
    temperature: float = 0.7,
) -> AsyncGenerator[str, None]:
    """Generate Responses API streaming events from LangGraph agent.

    Yields SSE-formatted events following Open Responses specification.

    Args:
        input_messages: Input (string or list of items)
        model: Model ID
        instructions: System message / instructions
        tools: Tools available to the model (not used by ESDC, tools are internal)
        temperature: Sampling temperature

    Yields:
        SSE-formatted event strings
    """
    response_id = f"resp_{uuid.uuid4().hex[:24]}"
    seq = SequenceCounter()

    # DEBUG: Track execution
    event_counter = 0

    # Emit response.created
    created_seq = seq.next()
    logger.debug(
        f"[RESPONSES {response_id}] Emitting response.created seq={created_seq}"
    )
    yield format_sse_event(
        create_response_created_event(response_id, model, created_seq)
    )

    # Create LLM and agent
    provider_config = Config.get_provider_config()
    if not provider_config:
        logger.error(f"[RESPONSES {response_id}] No provider configured")
        # Emit error event
        error_seq = seq.next()
        yield format_sse_event(
            {
                "type": "error",
                "sequence_number": error_seq,
                "error": {"message": "No provider configured", "type": "server_error"},
            }
        )
        return

    provider_name = provider_config.get("provider_type", "ollama")
    provider_model = provider_config.get("model")
    base_url = provider_config.get("base_url")
    api_key = provider_config.get("api_key")

    logger.debug(
        f"[RESPONSES {response_id}] Provider: {provider_name}, "
        f"model: {provider_model}, base_url: {base_url}"
    )

    provider_config_obj = {
        "provider_type": provider_name,
        "model": provider_model,
        "base_url": base_url,
        "api_key": api_key,
    }

    llm = create_llm_from_config(provider_config_obj)
    agent = create_agent(llm, checkpointer=None)

    # Convert input to LangChain messages
    lc_messages = convert_responses_input_to_langchain(input_messages, instructions)

    # DEBUG: Log input
    input_type = (
        "string" if isinstance(input_messages, str) else f"list({len(input_messages)})"
    )
    instructions_preview = instructions[:100] if instructions else "None"
    logger.debug(
        f"[RESPONSES {response_id}] Input type: {input_type}, "
        f"messages: {len(lc_messages)}, instructions: {instructions_preview}"
    )
    for i, msg in enumerate(lc_messages):
        content_preview = str(msg.content)[:50] if msg.content else "None"
        logger.debug(
            f"[RESPONSES {response_id}] Message {i}: type={type(msg).__name__}, "
            f"content_preview={content_preview}..."
        )

    # Track output items and indices
    output_items: list[dict[str, Any]] = []
    output_index = 0

    logger.info(
        f"[RESPONSES {response_id}] START - "
        f"input_type={input_type}, messages={len(lc_messages)}"
    )

    try:
        # Stream agent events
        async for event in agent.astream({"messages": lc_messages}):
            event_counter += 1
            event_keys = list(event.keys())
            logger.debug(
                f"[RESPONSES {response_id}] EVENT #{event_counter} keys: {event_keys}"
            )

            # Check for tool results FIRST (before AI messages)
            tool_messages = extract_tool_messages_from_event(event)

            if tool_messages:
                logger.debug(
                    f"[RESPONSES {response_id}] EVENT #{event_counter} "
                    f"Found {len(tool_messages)} tool result(s)"
                )
                # Emit function_call_output items for each tool result
                for i, tool_msg in enumerate(tool_messages):
                    item_id = generate_item_id("fco")
                    tool_call_id = tool_msg.get("tool_call_id", "")
                    tool_content = str(tool_msg.get("content", ""))
                    content_len = len(tool_content)
                    content_preview = (
                        tool_content[:100] if content_len > 100 else tool_content
                    )

                    logger.debug(
                        f"[RESPONSES {response_id}] EVENT #{event_counter} "
                        f"Tool result #{i}: call_id={tool_call_id}, "
                        f"content_len={content_len}, preview={content_preview}..."
                    )

                    function_call_output = {
                        "id": item_id,
                        "type": "function_call_output",
                        "status": "completed",
                        "call_id": tool_call_id,
                        "output": tool_content,
                    }

                    # Emit item added
                    added_seq = seq.next()
                    logger.debug(
                        f"[RESPONSES {response_id}] EMIT response.output_item.added "
                        f"seq={added_seq}, output_index={output_index}, "
                        f"type=function_call_output, id={item_id}, "
                        f"call_id={tool_call_id}"
                    )
                    yield format_sse_event(
                        create_output_item_added_event(
                            added_seq, output_index, function_call_output
                        )
                    )

                    # Emit item done (output is included, no deltas needed)
                    function_call_output["status"] = "completed"
                    done_seq = seq.next()
                    logger.debug(
                        f"[RESPONSES {response_id}] EMIT response.output_item.done "
                        f"seq={done_seq}, output_index={output_index}, "
                        f"type=function_call_output, id={item_id}"
                    )
                    yield format_sse_event(
                        create_output_item_done_event(
                            done_seq, output_index, function_call_output
                        )
                    )

                    output_items.append(function_call_output)
                    output_index += 1

                # Tool results processed, continue to next event
                continue

            # Extract AI message
            ai_msg = extract_ai_message_from_event(event)
            if not ai_msg:
                logger.debug(
                    f"[RESPONSES {response_id}] EVENT #{event_counter} "
                    f"No AIMessage extracted, skipping"
                )
                continue

            # DEBUG: Log AIMessage details
            msg_id = getattr(ai_msg, "id", "no-id")
            msg_content_preview = (
                str(ai_msg.content)[:100] if ai_msg.content else "None"
            )
            has_tool_calls = hasattr(ai_msg, "tool_calls") and ai_msg.tool_calls
            tool_call_count = len(ai_msg.tool_calls) if has_tool_calls else 0
            logger.debug(
                f"[RESPONSES {response_id}] EVENT #{event_counter} "
                f"AIMessage: id={msg_id}, content_preview={msg_content_preview}..., "
                f"tool_calls={tool_call_count}"
            )

            # Handle message with content
            if ai_msg.content:
                item_id = generate_item_id("msg")
                content_str = extract_content_str(ai_msg.content)

                if content_str.strip():
                    logger.debug(
                        f"[RESPONSES {response_id}] Processing message content: "
                        f"len={len(content_str)}, preview={content_str[:50]}..."
                    )

                    # Emit message item added
                    message_item = {
                        "id": item_id,
                        "type": "message",
                        "status": "in_progress",
                        "role": "assistant",
                        "content": [],
                    }
                    added_seq = seq.next()
                    logger.debug(
                        f"[RESPONSES {response_id}] EMIT response.output_item.added "
                        f"seq={added_seq}, output_index={output_index}, "
                        f"type=message, id={item_id}"
                    )
                    yield format_sse_event(
                        create_output_item_added_event(
                            added_seq, output_index, message_item
                        )
                    )

                    # Emit content part added
                    content_part = {"type": "output_text", "text": ""}
                    part_added_seq = seq.next()
                    logger.debug(
                        f"[RESPONSES {response_id}] EMIT response.content_part.added "
                        f"seq={part_added_seq}, output_index={output_index}, "
                        f"content_index=0, item_id={item_id}"
                    )
                    yield format_sse_event(
                        create_content_part_added_event(
                            part_added_seq, output_index, 0, item_id, content_part
                        )
                    )

                    # Stream text deltas
                    accumulated_text = ""
                    delta_count = 0
                    for chunk in chunk_text(content_str):
                        accumulated_text += chunk
                        delta_count += 1
                        delta_seq = seq.next()
                        logger.debug(
                            f"[RESPONSES {response_id}] "
                            f"EMIT response.output_text.delta "
                            f"seq={delta_seq}, delta_len={len(chunk)}, "
                            f"accumulated_len={len(accumulated_text)}"
                        )
                        yield format_sse_event(
                            create_output_text_delta_event(
                                delta_seq, output_index, 0, item_id, chunk
                            )
                        )

                    # Emit text done
                    text_done_seq = seq.next()
                    logger.debug(
                        f"[RESPONSES {response_id}] EMIT response.output_text.done "
                        f"seq={text_done_seq}, text_len={len(accumulated_text)}, "
                        f"deltas_sent={delta_count}"
                    )
                    yield format_sse_event(
                        create_output_text_done_event(
                            text_done_seq, output_index, 0, item_id, accumulated_text
                        )
                    )

                    # Emit content part done
                    content_part_done = {
                        "type": "output_text",
                        "text": accumulated_text,
                    }
                    part_done_seq = seq.next()
                    logger.debug(
                        f"[RESPONSES {response_id}] EMIT response.content_part.done "
                        f"seq={part_done_seq}, output_index={output_index}"
                    )
                    yield format_sse_event(
                        create_content_part_done_event(
                            part_done_seq, output_index, 0, item_id, content_part_done
                        )
                    )

                    # Complete message item
                    message_item["status"] = "completed"
                    message_item["content"] = [content_part_done]
                    item_done_seq = seq.next()
                    logger.debug(
                        f"[RESPONSES {response_id}] EMIT response.output_item.done "
                        f"seq={item_done_seq}, output_index={output_index}, "
                        f"type=message, id={item_id}"
                    )
                    yield format_sse_event(
                        create_output_item_done_event(
                            item_done_seq, output_index, message_item
                        )
                    )

                    output_items.append(message_item)
                    output_index += 1

            # Handle function calls
            if hasattr(ai_msg, "tool_calls") and ai_msg.tool_calls:
                logger.debug(
                    f"[RESPONSES {response_id}] EVENT #{event_counter} "
                    f"Processing {len(ai_msg.tool_calls)} tool call(s)"
                )
                for tc_idx, tc in enumerate(ai_msg.tool_calls):
                    # DIAGNOSTIC: Log type and value of each tool_call
                    logger.debug(
                        f"[RESPONSES {response_id}] EVENT #{event_counter} "
                        f"tool_call[{tc_idx}] type: {type(tc).__name__}, "
                        f"value: {repr(tc)[:200]}"
                    )
                    # If it's not a dict, skip and log warning
                    if not isinstance(tc, dict):
                        logger.warning(
                            f"[RESPONSES {response_id}] EVENT #{event_counter} "
                            f"SKIPPING malformed tool_call[{tc_idx}] "
                            f"- expected dict, got {type(tc).__name__}: "
                            f"{repr(tc)[:100]}"
                        )
                        continue
                    item_id = generate_item_id("fc")
                    tc_name = tc.get("name", "")
                    tc_id = tc.get("id", "")
                    tc_args = json.dumps(tc.get("args", {}))

                    logger.debug(
                        f"[RESPONSES {response_id}] EVENT #{event_counter} "
                        f"Tool call #{tc_idx}: name={tc_name}, call_id={tc_id}, "
                        f"args_len={len(tc_args)}"
                    )

                    function_call_item = {
                        "id": item_id,
                        "type": "function_call",
                        "status": "in_progress",
                        "name": tc_name,
                        "call_id": tc_id,
                        "arguments": "",
                    }

                    # Emit item added
                    added_seq = seq.next()
                    logger.debug(
                        f"[RESPONSES {response_id}] EMIT "
                        f"response.output_item.added "
                        f"seq={added_seq}, output_index={output_index}, "
                        f"type=function_call, id={item_id}, "
                        f"name={tc_name}, call_id={tc_id}"
                    )
                    yield format_sse_event(
                        create_output_item_added_event(
                            added_seq, output_index, function_call_item
                        )
                    )

                    # Stream argument deltas
                    delta_count = 0
                    for arg_chunk in chunk_json(tc_args):
                        function_call_item["arguments"] += arg_chunk
                        delta_count += 1
                        delta_seq = seq.next()
                        logger.debug(
                            f"[RESPONSES {response_id}] "
                            f"EMIT response.function_call_arguments."
                            f"delta seq={delta_seq}, chunk_len={len(arg_chunk)}"
                        )
                        yield format_sse_event(
                            create_function_call_arguments_delta_event(
                                delta_seq, output_index, item_id, arg_chunk
                            )
                        )

                    # Emit args done
                    args_done_seq = seq.next()
                    logger.debug(
                        f"[RESPONSES {response_id}] "
                        f"EMIT response.function_call_arguments.done "
                        f"seq={args_done_seq}, args_len={len(tc_args)}, "
                        f"deltas_sent={delta_count}"
                    )
                    yield format_sse_event(
                        create_function_call_arguments_done_event(
                            args_done_seq, output_index, item_id, tc_args
                        )
                    )

                    # Complete item
                    function_call_item["status"] = "completed"
                    item_done_seq = seq.next()
                    logger.debug(
                        f"[RESPONSES {response_id}] EMIT response.output_item.done "
                        f"seq={item_done_seq}, output_index={output_index}, "
                        f"type=function_call, id={item_id}, name={tc_name}"
                    )
                    yield format_sse_event(
                        create_output_item_done_event(
                            item_done_seq, output_index, function_call_item
                        )
                    )

                    output_items.append(function_call_item)
                    output_index += 1

        # Emit response.completed
        completed_seq = seq.next()
        output_summary = [
            f"{item.get('type')}:{item.get('name', item.get('call_id', 'msg'))}"
            for item in output_items
        ]
        logger.info(
            f"[RESPONSES {response_id}] COMPLETED - "
            f"events={event_counter}, output_items={len(output_items)}, "
            f"items=[{', '.join(output_summary)}], final_seq={completed_seq}"
        )
        yield format_sse_event(
            create_response_completed_event(
                completed_seq, response_id, model, output_items
            )
        )

    except Exception as e:
        logger.exception(f"[RESPONSES {response_id}] ERROR: {e}")
        yield format_sse_event(
            {
                "type": "error",
                "sequence_number": seq.next(),
                "error": {"message": str(e), "type": "server_error"},
            }
        )


async def generate_responses_sync(
    input_messages: str | list[ResponseInputItem],
    model: str = "esdc-agent",
    instructions: str | None = None,
    tools: list[dict[str, Any]] | None = None,
    temperature: float = 0.7,
) -> dict[str, Any]:
    """Generate non-streaming Responses API response.

    Collects all output from agent and returns a complete response object.

    Args:
        input_messages: Input (string or list of items)
        model: Model ID
        instructions: System message / instructions
        tools: Tools available to the model (not used by ESDC)
        temperature: Sampling temperature

    Returns:
        Complete response object
    """
    from esdc.server.responses_events import create_non_streaming_response

    response_id = f"resp_{uuid.uuid4().hex[:24]}"
    output_items: list[dict[str, Any]] = []

    # Create LLM and agent
    provider_config = Config.get_provider_config()
    if not provider_config:
        logger.error(f"[RESPONSES {response_id}] SYNC: No provider configured")
        return create_non_streaming_response(
            response_id,
            model,
            [],
            error={"message": "No provider configured", "type": "server_error"},
        )

    provider_name = provider_config.get("provider_type", "ollama")
    provider_model = provider_config.get("model")
    base_url = provider_config.get("base_url")
    api_key = provider_config.get("api_key")

    logger.debug(
        f"[RESPONSES {response_id}] SYNC: Provider: {provider_name}, "
        f"model: {provider_model}, base_url: {base_url}"
    )

    provider_config_obj = {
        "provider_type": provider_name,
        "model": provider_model,
        "base_url": base_url,
        "api_key": api_key,
    }

    llm = create_llm_from_config(provider_config_obj)
    agent = create_agent(llm, checkpointer=None)

    # Convert input to LangChain messages
    lc_messages = convert_responses_input_to_langchain(input_messages, instructions)

    # DEBUG: Log input
    input_type = (
        "string" if isinstance(input_messages, str) else f"list({len(input_messages)})"
    )
    instructions_preview = instructions[:100] if instructions else "None"
    logger.debug(
        f"[RESPONSES {response_id}] SYNC: Input type: {input_type}, "
        f"messages: {len(lc_messages)}, instructions: {instructions_preview}"
    )

    # DEBUG: Track execution
    event_counter = 0

    logger.info(
        f"[RESPONSES {response_id}] SYNC START - "
        f"input_type={input_type}, messages={len(lc_messages)}"
    )

    try:
        # Collect all messages from agent
        all_tool_results: list[dict[str, Any]] = []
        all_ai_messages: list[AIMessage] = []

        async for event in agent.astream({"messages": lc_messages}):
            event_counter += 1
            event_keys = list(event.keys())
            logger.debug(
                f"[RESPONSES {response_id}] SYNC EVENT #{event_counter} "
                f"keys: {event_keys}"
            )

            # Collect tool results
            tool_messages = extract_tool_messages_from_event(event)
            if tool_messages:
                logger.debug(
                    f"[RESPONSES {response_id}] SYNC EVENT #{event_counter} "
                    f"Found {len(tool_messages)} tool result(s)"
                )
                for i, tool_msg in enumerate(tool_messages):
                    call_id = tool_msg.get("tool_call_id", "")
                    content_len = len(str(tool_msg.get("content", "")))
                    logger.debug(
                        f"[RESPONSES {response_id}] SYNC Tool result #{i}: "
                        f"call_id={call_id}, content_len={content_len}"
                    )
                    all_tool_results.append(
                        {
                            "id": generate_item_id("fco"),
                            "type": "function_call_output",
                            "status": "completed",
                            "call_id": call_id,
                            "output": str(tool_msg.get("content", "")),
                        }
                    )
                continue

            # Collect AI messages
            ai_msg = extract_ai_message_from_event(event)
            if ai_msg:
                msg_id = getattr(ai_msg, "id", "no-id")
                has_tool_calls = hasattr(ai_msg, "tool_calls") and ai_msg.tool_calls
                tool_call_count = len(ai_msg.tool_calls) if has_tool_calls else 0
                logger.debug(
                    f"[RESPONSES {response_id}] SYNC EVENT #{event_counter} "
                    f"AIMessage: id={msg_id}, tool_calls={tool_call_count}"
                )
                all_ai_messages.append(ai_msg)

        # Build output items
        # First: all function_calls
        for ai_msg in all_ai_messages:
            if hasattr(ai_msg, "tool_calls") and ai_msg.tool_calls:
                logger.debug(
                    f"[RESPONSES {response_id}] SYNC tool_calls type: "
                    f"{type(ai_msg.tool_calls)}, count: {len(ai_msg.tool_calls)}"
                )
                for tc_idx, tc in enumerate(ai_msg.tool_calls):
                    # DIAGNOSTIC: Log type and value of each tool_call
                    logger.debug(
                        f"[RESPONSES {response_id}] SYNC tool_call[{tc_idx}] "
                        f"type: {type(tc).__name__}, "
                        f"value: {repr(tc)[:200]}"
                    )
                    # If it's not a dict, skip and log warning
                    if not isinstance(tc, dict):
                        logger.warning(
                            f"[RESPONSES {response_id}] SYNC SKIPPING malformed "
                            f"tool_call[{tc_idx}] - expected dict, got "
                            f"{type(tc).__name__}: {repr(tc)[:100]}"
                        )
                        continue
                    tc_name = tc.get("name", "")
                    tc_id = tc.get("id", "")
                    logger.debug(
                        f"[RESPONSES {response_id}] SYNC Building function_call: "
                        f"name={tc_name}, call_id={tc_id}"
                    )
                    output_items.append(
                        {
                            "id": generate_item_id("fc"),
                            "type": "function_call",
                            "status": "completed",
                            "name": tc_name,
                            "call_id": tc_id,
                            "arguments": json.dumps(tc.get("args", {})),
                        }
                    )

        # Second: all function_call_outputs
        output_items.extend(all_tool_results)

        # Last: final message (if any AI message had content)
        for ai_msg in reversed(all_ai_messages):
            if ai_msg.content:
                content_str = extract_content_str(ai_msg.content)
                if content_str.strip():
                    logger.debug(
                        f"[RESPONSES {response_id}] SYNC Building message item: "
                        f"content_len={len(content_str)}"
                    )
                    output_items.append(
                        {
                            "id": generate_item_id("msg"),
                            "type": "message",
                            "status": "completed",
                            "role": "assistant",
                            "content": [
                                {
                                    "type": "output_text",
                                    "text": content_str,
                                    "annotations": [],
                                }
                            ],
                        }
                    )
                break

        output_summary = [
            f"{item.get('type')}:{item.get('name', item.get('call_id', 'msg'))}"
            for item in output_items
        ]
        logger.info(
            f"[RESPONSES {response_id}] SYNC COMPLETED - "
            f"events={event_counter}, output_items={len(output_items)}, "
            f"items=[{', '.join(output_summary)}]"
        )
        return create_non_streaming_response(response_id, model, output_items)

    except Exception as e:
        logger.exception(f"[RESPONSES {response_id}] SYNC ERROR: {e}")
        return create_non_streaming_response(
            response_id,
            model,
            [],
            error={"message": str(e), "type": "server_error"},
        )
