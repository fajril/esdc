# Fix Tool Call Bugs Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix two bugs causing infinite loops and SQL loss in tool execution:
1. Multiple tool calls in one LLM response lose SQL query info
2. Tool execution may not complete properly

**Architecture:** 
- Track tool calls as a list instead of single dict to preserve order
- Match tool results to tool calls by order (on_tool_end events should come in same order as tool calls)
- Keep tool_node as async (LangGraph supports async nodes)

**Tech Stack:** Python, LangGraph, asyncio

---

## Root Cause Analysis

### Bug 1: SQL Loss for Multiple Tool Calls

**Problem:** `stored_tool_call` is a single dict, overwritten when multiple tool calls occur.

**Evidence from logs:**
```
AGENT_STORING: tool_call=execute_sql, query=SELECT report_year...    # Stored
AGENT_STORING: tool_call=execute_sql, query=SELECT project_class...  # Overwrites!
AGENT_TOOL_END: tool=execute_sql, result_len=67
YIELDING tool_result: tool=execute_sql, sql_len=61, result_len=67  # Uses wrong SQL
AGENT_TOOL_END: tool=execute_sql, result_len=125
YIELDING tool_result NO SQL - tool=execute_sql, result_length=125  # SQL cleared
```

**Fix:** Use `stored_tool_calls` list to track all tool calls in order, pop from front for each on_tool_end.

### Bug 2: Tool Execution Loop

**Observation:** Tools ARE executing (on_tool_end events appear), but LLM keeps calling tools repeatedly until timeout.

**Possible causes:**
1. Tool results not properly integrated into conversation
2. LLM doesn't see tool results and keeps retrying
3. Checkpointer/memory issue - LLM doesn't remember previous responses

**Fix steps:**
1. First fix Bug 1 (SQL tracking)
2. Add more logging to see if tool results are being added to messages
3. Check if ToolMessage is properly formed with tool_call_id

---

## Files to Modify

- `esdc/chat/agent.py` - Fix tool call tracking and add logging

---

## Task 1: Fix Multiple Tool Call Tracking

**Files:**
- Modify: `esdc/chat/agent.py:145-150` (stored_tool_call initialization)
- Modify: `esdc/chat/agent.py:215-239` (store tool calls in list)
- Modify: `esdc/chat/agent.py:270-307` (pop from list in on_tool_end)

**Step 1: Change stored_tool_call from dict to list**

In `run_agent_stream` around line 148, change:
```python
stored_tool_call: dict[str, Any] = {}
```

To:
```python
stored_tool_calls: list[dict[str, Any]] = []
```

**Step 2: Append to list instead of overwriting dict**

Around line 215-239, in `on_chat_model_end` tool calls loop, change:
```python
stored_tool_call = {
    "name": tc["name"],
    "args": args,
    "query": query,
}
```

To:
```python
stored_tool_calls.append({
    "name": tc["name"],
    "args": args,
    "query": query,
    "id": tc.get("id", ""),
})
```

And remove the line that stores the tool call dict (it's now append to list).

**Step 3: Pop from list in on_tool_end**

Around line 270-307, in `on_tool_end` handler, change:
```python
# Extract SQL from stored_tool_call
sql = ""
if stored_tool_call.get("name") == "execute_sql":
    args = stored_tool_call.get("args", {})
    # ... rest of code
```

To:
```python
# Extract SQL from stored_tool_calls (pop from front - FIFO order)
sql = ""
if stored_tool_calls:
    stored_tc = stored_tool_calls.pop(0)
    if stored_tc.get("name") == "execute_sql":
        args = stored_tc.get("args", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                args = {}
        sql = args.get("query", "")
        logger.info(
            "📤 YIELDING tool_result: tool=%s, sql_len=%d, result_len=%d",
            tool_name,
            len(sql),
            len(str(tool_result)),
        )
else:
    logger.info(
        "YIELDING tool_result NO SQL - tool=%s, result_length=%d (no stored calls)",
        tool_name,
        len(str(tool_result)),
    )
```

And remove the line `stored_tool_call = {}`.

**Step 4: Run type check and lint**

Run: `uv run basedpyright esdc/chat/agent.py`
Expected: No errors

Run: `uv run ruff check esdc/chat/agent.py`
Expected: All checks pass

**Step 5: Test with esdc chat**

Run: `esdc chat`
Then ask: "berapa cadangan lapangan karamba di wilayah kerja wain?"

Expected:
1. Tool calls should appear with SQL
2. Multiple tool calls should preserve SQL for each
3. Tool results should be yielded
4. LLM should respond with answer, not infinite loop

**Step 6: Check logs**

Run: `tail -100 esdc_chat.log | grep -E "(TOOL_CALL|TOOL_END|YIELDING)"`

Expected:
- Each `AGENT_STORING` followed by `AGENT_TOOL_END`
- Each `YIELDING tool_result` should have SQL info
- No "YIELDING tool_result NO SQL" when execute_sql was called

**Step 7: Commit if tests pass**

```bash
git add esdc/chat/agent.py
git commit -m "fix: track multiple tool calls in list to prevent SQL loss

Changed stored_tool_call from single dict to stored_tool_calls list.
When multiple tool calls occur in one LLM response, each is appended
to the list. When on_tool_end events arrive, pop from front (FIFO)
to match tool results with their corresponding tool calls.

This fixes the issue where:
- First tool call SQL was lost when second tool call overwrote it
- Second tool call had no SQL because stored_tool_call was cleared"
```

---

## Task 2: Add Logging to Debug Tool Result Flow

If Task 1 doesn't fix the loop, add more logging to understand the flow.

**Files:**
- Modify: `esdc/chat/agent.py:67-98` (tool_node function)

**Step 1: Add logging in tool_node**

Add at start of tool_node function (around line 69):
```python
async def tool_node(state: MessagesState) -> dict[str, list[AnyMessage]]:
    """Tool execution node."""
    result = []
    last_message = state["messages"][-1]

    ai_message = cast(AIMessage, last_message)
    if not hasattr(ai_message, "tool_calls") or not ai_message.tool_calls:
        return {"messages": []}

    logger.info(f"🔧 TOOL_NODE: Processing {len(ai_message.tool_calls)} tool calls")

    for tool_call in ai_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call.get("args", {})
        tool_id = tool_call.get("id", "unknown")

        if tool_name in tools_by_name:
            tool = tools_by_name[tool_name]
            try:
                if isinstance(tool_args, str):
                    tool_args = json.loads(tool_args)
                
                logger.info(f"🔧 TOOL_NODE: Invoking {tool_name} (id={tool_id})")
                observation = await tool.ainvoke(tool_args)
                logger.info(f"🔧 TOOL_NODE: {tool_name} returned {len(str(observation))} chars")
            except Exception as e:
                logger.error(f"🔧 TOOL_NODE: {tool_name} failed: {e}")
                observation = f"Error: {str(e)}"

            result.append(
                {
                    "tool_call_id": tool_id,
                    "role": "tool",
                    "name": tool_name,
                    "content": observation,
                }
            )

    logger.info(f"🔧 TOOL_NODE: Returning {len(result)} tool results")
    return {"messages": result}
```

**Step 2: Add json import if not present**

At top of file, ensure:
```python
import json
```

**Step 3: Run type check and lint**

Run: `uv run basedpyright esdc/chat/agent.py`
Run: `uv run ruff check esdc/chat/agent.py`
Expected: All checks pass

**Step 4: Test with esdc chat**

Run: `esdc chat`
Ask a question that triggers tool use.

Check logs for:
- How many tool calls per invocation
- Whether all tool calls get results
- Whether results are returned to LLM

**Step 5: Commit if helpful**

```bash
git add esdc/chat/agent.py
git commit -m "debug: add detailed logging in tool_node for flow analysis"
```

---

## Task 3: Investigate if Tool Results Reach LLM

If tools execute but LLM keeps calling same queries, tool results may not be reaching the LLM.

**Potential issues:**
1. ToolMessage format incorrect
2. Messages not being added to state properly
3. LLM not receiving tool results

**Investigation steps:**

**Step 1: Check message count in tool_node**

Add logging:
```python
logger.info(f"🔧 TOOL_NODE: State has {len(state['messages'])} messages before")
# ... execute tools ...
logger.info(f"🔧 TOOL_NODE: State has {len(state['messages']) + len(result)} messages after")
```

**Step 2: Check on_tool_end payload**

In `run_agent_stream`, add logging for the event structure:
```python
elif event_type == "on_tool_end":
    logger.info(f"🔧 ON_TOOL_END: event keys = {event.keys()}, data keys = {data.keys()}")
    # existing code...
```

**Step 3: Look at LangGraph docs**

Check if there's a specific way to handle async tools in LangGraph StateGraph.

**Step 4: If found issue, implement fix**

Document findings and implement appropriate fix.

---

## Task 4: Check Checkpointer/Thread State

If tool results are properly formed but LLM doesn't remember them, checkpointer may not be working.

**Step 1: Add thread_id logging**

In `run_agent_stream`, log the thread_id:
```python
logger.info(f"🔔 STREAM: thread_id={thread_id}, checkpointer={checkpointer is not None}")
```

**Step 2: Check if messages accumulate**

Add logging in tool_node to see message history:
```python
for i, msg in enumerate(state["messages"][-5:]):
    logger.info(f"  Message {i}: {type(msg).__name__}")
```

**Step 3: Test and analyze**

Run `esdc chat` and check if messages grow with each call.

---

## Verification

After all fixes:

1. Run `esdc chat`
2. Ask a database question
3. Verify:
   - ✓ Tool calls execute (on_tool_end appears in logs)
   - ✓ SQL preserved for all tool calls (YIELDING with sql_len)
   - ✓ Tool results appear (tool_result chunks)
   - ✓ LLM responds with answer (not infinite loop)
   - ✓ Follow-up questions work (memory persists)

---

## Expected Outcome

- Multiple tool calls preserve SQL for each
- Tool results reach LLM
- LLM responds appropriately
- No infinite loops or recursion errors