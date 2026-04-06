# Error Logging Enhancement Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add comprehensive error logging throughout the ESDC server to identify the root cause of 500 Internal Server Errors.

**Architecture:** Enhance error handling at three critical points: route entry/exit, streaming response generator, and LangGraph agent execution. Each layer will catch, log, and re-raise exceptions with full context.

**Tech Stack:** Python logging, FastAPI exception handlers, AsyncGenerator error handling

---

## Task 1: Enhance Routes Error Logging

**Files:**
- Modify: `esdc/server/routes.py:1-200` (chat_completions endpoint)

**Step 1: Add detailed request logging at entry**

Add at the beginning of `chat_completions` function:
```python
import uuid

@router.post("/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    request_id = str(uuid.uuid4())[:12]
    logger.info(f"[REQUEST {request_id}] START - conversation_id={request.conversation_id}, stream={request.stream}, message_count={len(request.messages)}")
    
    try:
        # existing code...
    except Exception as e:
        logger.exception(f"[REQUEST {request_id}] ERROR: {type(e).__name__}: {str(e)}")
        raise
    finally:
        logger.info(f"[REQUEST {request_id}] END")
```

**Step 2: Run server and test**

Run: `esdc serve --web`
Make a request and check logs show request_id
Expected: See "[REQUEST xxx] START" and "[REQUEST xxx] END"

**Step 3: Commit**

```bash
git add esdc/server/routes.py
git commit -m "feat: add request tracing logs to chat_completions endpoint"
```

---

## Task 2: Add Streaming Response Error Context

**Files:**
- Modify: `esdc/server/agent_wrapper.py:148-180` (generate_streaming_response)
- Modify: `esdc/server/agent_wrapper.py:417-424` (exception handler)

**Step 1: Wrap streaming generator with error context**

In `generate_streaming_response`, add try-except around the main loop:
```python
async def generate_streaming_response(...):
    request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    
    try:
        # existing setup code...
        
        async for event in agent.astream(...):
            try:
                # existing event processing...
            except Exception as e:
                logger.exception(f"[{request_id}] ERROR processing event: {type(e).__name__}")
                raise
                
    except Exception as e:
        logger.exception(f"[{request_id}] FATAL ERROR in streaming: {type(e).__name__}: {str(e)}")
        raise
```

**Step 2: Run tests**

Run: `pytest tests/test_agent_wrapper.py -v`
Expected: Tests pass

**Step 3: Commit**

```bash
git add esdc/server/agent_wrapper.py
git commit -m "feat: add detailed error context to streaming response"
```

---

## Task 3: Add Tool Execution Logging

**Files:**
- Modify: `esdc/server/agent_wrapper.py:334-380` (elif tool_calls block)

**Step 1: Log tool execution details**

In the `elif hasattr(ai_msg, "tool_calls")` block:
```python
elif hasattr(ai_msg, "tool_calls") and ai_msg.tool_calls:
    for i, tool_call in enumerate(ai_msg.tool_calls):
        tool_name = tool_call.get("name", "unknown")
        tool_id = tool_call.get("id", "no-id")
        logger.debug(f"[TOOL {request_id}] #{i}: name={tool_name}, id={tool_id}")
        
        try:
            # existing tool handling...
        except Exception as e:
            logger.exception(f"[TOOL {request_id}] ERROR in {tool_name}: {str(e)}")
            raise
```

**Step 2: Run tests**

Run: `pytest tests/test_agent_wrapper.py::test_streaming_response -v`
Expected: Tests pass

**Step 3: Commit**

```bash
git add esdc/server/agent_wrapper.py
git commit -m "feat: add tool execution logging"
```

---

## Task 4: Enhance Exception Handler with Context

**Files:**
- Modify: `esdc/server/app.py:46-53` (exception_handler)

**Step 1: Add detailed exception logging**

Update exception handler:
```python
@app.exception_handler(Exception)
async def generic_exception_handler(request, exc):
    """Handle generic exceptions with full context."""
    import traceback
    
    error_msg = f"Unhandled exception: {type(exc).__name__}: {str(exc)}"
    stack_trace = traceback.format_exc()
    
    logger.error(f"[EXCEPTION] {error_msg}")
    logger.error(f"[STACK TRACE]\n{stack_trace}")
    
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "message": str(exc),
                "type": type(exc).__name__,
                "detail": "Check server logs for full traceback"
            }
        },
    )
```

**Step 2: Run server and test**

Run: `esdc serve --web`
Trigger an error and check logs show stack trace
Expected: See full stack trace in logs

**Step 3: Commit**

```bash
git add esdc/server/app.py
git commit -m "feat: enhance exception handler with stack traces"
```

---

## Task 5: Add Response Summary Logging

**Files:**
- Modify: `esdc/server/agent_wrapper.py:405-430` (end of streaming)

**Step 1: Log response summary**

At the end of `generate_streaming_response`:
```python
logger.info(f"[{request_id}] SUMMARY: "
            f"events={event_counter}, "
            f"first_msg={path_first_msg_count}, "
            f"tool_calls={path_tool_calls_count}, "
            f"final_content={path_final_content_count}, "
            f"yielded_chunks={len(yielded_chunks)}")
```

**Step 2: Run tests**

Run: `pytest tests/test_agent_wrapper.py -v`
Expected: Tests pass

**Step 3: Commit**

```bash
git add esdc/server/agent_wrapper.py
git commit -m "feat: add response summary logging"
```

---

## Task 6: Verify Error Logging Works

**Step 1: Restart server**

```bash
pkill -f "esdc serve"
esdc serve --web
```

**Step 2: Make a test request**

Send a query that previously caused 500 error

**Step 3: Check logs**

```bash
tail -f ~/.esdc/logs/esdc_server.log | grep -E "ERROR|EXCEPTION|REQUEST"
```

Expected: See detailed error messages with request IDs and stack traces

**Step 4: Final commit**

```bash
git add docs/plans/
git commit -m "docs: add error logging enhancement plan"
```

---

## Success Criteria

- [ ] Every request has START/END log entry with request_id
- [ ] All exceptions include full stack traces
- [ ] Tool calls are logged with their IDs
- [ ] Streaming errors include the event that caused them
- [ ] Response summary shows event counts

## Files Changed

1. `esdc/server/routes.py` - Request entry/exit logging
2. `esdc/server/agent_wrapper.py` - Streaming and tool logging
3. `esdc/server/app.py` - Exception handler enhancement
4. `docs/plans/2026-04-04-error-logging-enhancement.md` - This plan
