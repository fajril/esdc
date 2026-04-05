# Fix: OpenWebUI Follow-Up Questions Not Working

**Date:** 2026-04-05
**Status:** Resolved
**Commits:** `f3e0683`, `c5ec3ca`

## Problem

Follow-up questions in OpenWebUI chats received empty or very brief responses because the conversation history with tool execution context wasn't being passed to the LLM agent.

### Symptoms
- First question in a chat worked fine with detailed tool calls
- Follow-up questions got extremely brief or empty responses
- Server logs showed `Created 1 messages` instead of multiple messages with tool history

## Root Cause

ESDC has two separate endpoints for handling chat requests:

1. **`/v1/responses`** (Responses API) - uses `responses_wrapper.py:convert_responses_input_to_langchain`
2. **`/v1/chat/completions`** (Chat Completions API) - uses `agent_wrapper.py:convert_messages_to_langchain`

OpenWebUI uses the `/v1/responses` endpoint. Both conversion functions had the same bug: they were **skipping** `function_call` items instead of converting them to `AIMessage` objects with `tool_calls`.

### Why This Matters

When OpenWebUI sends conversation history, it includes tool execution context:

```json
{
  "type": "function_call",
  "id": "fc_123",
  "name": "query_reserves",
  "arguments": "{\"entity_type\": \"national\"}"
}
```

The LLM agent needs this context to:
1. Know which tools were called previously
2. Understand the tool results that followed
3. Continue the conversation with full context

Without these messages, the LLM has no history of what tools were called, making it impossible to provide contextually relevant follow-up responses.

## The Fix

### 1. Chat Completions API (`agent_wrapper.py`)

Added `_convert_output_to_langchain_messages()` helper function to handle OpenWebUI's `output` array format:

```python
def _convert_output_to_langchain_messages(output: list) -> list:
    """Convert OpenWebUI output array to LangChain messages."""
    lc_messages = []

    for item in output:
        item_type = item.get("type")

        if item_type == "function_call":
            # Convert to AIMessage with tool_calls
            call_id = item.get("call_id") or item.get("id", "")
            name = item.get("name", "")
            args = json.loads(item.get("arguments", "{}"))

            lc_messages.append(
                AIMessage(
                    content="",
                    tool_calls=[{"name": name, "args": args, "id": call_id}],
                )
            )

        elif item_type == "function_call_output":
            # Convert to ToolMessage
            lc_messages.append(
                ToolMessage(content=output_text, tool_call_id=call_id)
            )

        elif item_type == "message":
            # Convert to HumanMessage/AIMessage based on role
            ...

    return lc_messages
```

### 2. Responses API (`responses_wrapper.py`)

Changed `function_call` handling from **skipping** to **creating AIMessage**:

**Before (BUG):**
```python
elif item_type == "function_call":
    # Just log and skip
    logger.debug(f"[convert_responses_input] Item {idx}: function_call, skipping")
```

**After (FIX):**
```python
elif item_type == "function_call":
    # Create AIMessage with tool_calls for LangGraph
    call_id = item.get("call_id") or item.get("id", "")
    name = item.get("name", "")
    args = json.loads(item.get("arguments", "{}"))

    messages.append(
        AIMessage(
            content="",
            tool_calls=[{"name": name, "args": args, "id": call_id}],
        )
    )
```

### Key Details

1. **Field name handling**: OpenWebUI may use `id` OR `call_id` for tool call IDs. Both must be checked.

2. **JSON parsing**: Arguments come as JSON strings and need parsing.

3. **Message order**: Proper order is:
   - User message (question)
   - AIMessage with tool_calls (tool invocation)
   - ToolMessage (tool result)
   - AIMessage (final response)
   - User message (follow-up question)

4. **Bug found during fix**: There was duplicate message creation code that created TWO AIMessages for each function_call. This was removed.

## Lessons Learned

### 1. Test Both Endpoints

ESDC has two endpoints that serve different clients:
- `/v1/responses` - OpenWebUI uses this
- `/v1/chat/completions` - Other clients might use this

**Always test fixes against both endpoints.** The fix needed to be applied to both `responses_wrapper.py` AND `agent_wrapper.py`.

### 2. Debug Logging is Critical

The key insight came from server logs:
```
[convert_responses_input] Created 1 messages
```

This showed that only 1 message was created from a conversation history that should have created 5+ messages (user, tool calls, tool results, assistant response).

### 3. Chrome DevTools for End-to-End Testing

Using Chrome DevTools MCP allowed:
- Capturing the actual request body OpenWebUI sends
- Seeing the `output` array format with `function_call` items
- Verifying that responses now contain tool-driven data

### 4. Test Conversation Formats

The test suite needed tests for:
```python
# OpenWebUI conversation format
conversation = [
    {"type": "message", "role": "user", "content": "Question"},
    {"type": "function_call", "id": "...", "name": "...", "arguments": "..."},
    {"type": "function_call_output", "call_id": "...", "output": [...]},
    {"type": "message", "role": "assistant", "content": [...]},
    {"type": "message", "role": "user", "content": "Follow-up"},
]
```

### 5. Pydantic Models Can Have Multiple Fields for Same Concept

`ResponseInputFunctionCallItem` has BOTH `id` and `call_id` fields:
```python
class ResponseInputFunctionCallItem(BaseModel):
    id: str = Field(default="", description="Function call ID")
    call_id: str = Field(default="", description="Call ID for tracking")
```

The fix needed to handle both:
```python
call_id = item.get("call_id") or item.get("id", "")
```

## Files Changed

1. `esdc/server/agent_wrapper.py`
   - Added `_convert_output_to_langchain_messages()` helper
   - Updated `convert_messages_to_langchain()` to use the helper

2. `esdc/server/responses_wrapper.py`
   - Changed `function_call` handling from skip to create AIMessage

3. `tests/server/test_chat_completions_input.py` (new)
   - Tests for Chat Completions message conversion

4. `tests/server/test_responses_input.py` (updated)
   - Tests now expect AIMessage for `function_call` items

## Verification

1. All 31 unit tests pass
2. Chrome DevTools test shows follow-up question gets detailed tool-driven response
3. Server logs now show correct message count

```
[convert_responses_input] Item 0: message (user)
[convert_responses_input] Item 1: function_call -> AIMessage with tool_calls
[convert_responses_input] Item 2: function_call_output -> ToolMessage
[convert_responses_input] Item 3: message (assistant)
[convert_responses_input] Item 4: message (user)
[convert_responses_input] Created 5 messages
```