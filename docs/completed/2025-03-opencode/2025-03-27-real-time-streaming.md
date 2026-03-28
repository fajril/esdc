# Implement Real-time Streaming Chat with Inline Indicator

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement true token-by-token streaming with inline indicator (like ChatGPT/GPT-4), removing bursty updates and intermediate widgets.

**Architecture:** Switch from `astream(stream_mode="values")` to `astream_events(version="v2")` for token-level streaming. Update UI to handle `token` events for real-time display.

**Tech Stack:** Python, Textual TUI, LangGraph with streaming events

---

## Task 1: Update Agent to Use Token Streaming

**Problem:** Current `stream_mode="values"` emits complete messages, not tokens

**Files:**
- Modify: `esdc/chat/agent.py` (lines ~121-276)

**Step 1: Change streaming method**

Replace (around line 153-157):
```python
async for chunk in agent.astream(
    {"messages": messages},
    config=config,
    stream_mode="values",
):
```

With:
```python
async for event in agent.astream_events(
    {"messages": messages},
    config=config,
    version="v2",
):
```

**Step 2: Add token event handler**

Add after the loop starts (new code):
```python
    event_type = event.get("event")
    
    # Stream individual tokens
    if event_type == "on_chat_model_stream":
        chunk = event["data"]["chunk"]
        if hasattr(chunk, "content") and chunk.content:
            yield {
                "type": "token",
                "content": chunk.content,
            }
        continue
```

**Step 3: Update completion handling**

Change message handling (around line 170-200) from:
```python
if "messages" in chunk:
    messages = chunk["messages"]
    if messages:
        last_msg = messages[-1]
        ...
```

To handle `on_chat_model_end` event:
```python
    # Handle completion
    elif event_type == "on_chat_model_end":
        output = event["data"].get("output")
        if output:
            # Token usage
            tokens_used = _extract_token_usage(output, user_input)
            if tokens_used > 0:
                yield {"type": "token_usage", "tokens": tokens_used}
            
            # Tool calls
            if hasattr(output, "tool_calls") and output.tool_calls:
                for tc in output.tool_calls:
                    yield {
                        "type": "tool_call",
                        "tool": tc["name"],
                        "args": tc.get("args", {}),
                    }
        continue
```

**Step 4: Remove old chunk processing**

Remove the entire block that processes `chunk["messages"]` (lines ~170-230) since we're now handling via events.

**Step 5: Run tests**

```bash
cd /Users/fajril/Documents/GitHub/esdc && python -m pytest tests/test_chat_dom.py tests/test_chat_mounting.py -v
```

**Expected:** Tests may fail if they expect old message format

**Step 6: Commit**

```bash
git add esdc/chat/app.py esdc/chat/agent.py
git commit -m "feat: implement token-level streaming with astream_events"
```

---

## Task 2: Update App to Handle Token Events

**Problem:** App currently expects `message` type chunks, needs to handle `token` events

**Files:**
- Modify: `esdc/chat/app.py` (lines ~1080-1096, ~1140-1163)

**Step 1: Update _stream_response to pass tokens**

Around line 1140-1163, ensure tokens are passed through:
```python
async for chunk in run_agent_stream(...):
    if chunk["type"] == "token":
        yield chunk  # Pass through directly
    elif chunk["type"] == "message":
        yield chunk  # Legacy fallback
    elif chunk["type"] == "tool_call":
        yield chunk
    elif chunk["type"] == "tool_result":
        yield chunk
    elif chunk["type"] == "token_usage":
        yield chunk
```

**Step 2: Update on_input_submitted to handle token streaming**

Around line 1080-1096, add token handling:

```python
async def run_query():
    nonlocal accumulated_content
    async for chunk in self._stream_response(user_input):
        if self._cancelled:
            self.display_message("system", "Query cancelled.")
            return
        
        if chunk["type"] == "token":
            # Real-time token streaming
            token = chunk.get("content", "")
            if token and streaming_message:
                accumulated_content += token
                streaming_message.update(accumulated_content)
                
        elif chunk["type"] == "message":
            # Legacy: complete message (fallback)
            content = chunk.get("content", "")
            if content and streaming_message:
                accumulated_content += content
                streaming_message.update(accumulated_content)
                
        elif chunk["type"] == "tool_call":
            tool_name = chunk.get("tool", "")
            logger.debug(f"Tool called: {tool_name}")
            # Optionally show inline indicator here
            if streaming_message:
                current = streaming_message.content
                if "🔍" not in current:
                    streaming_message.update(current + "\n\n🔍 Querying database...")
                
        elif chunk["type"] == "tool_result":
            # Result received, continue streaming
            pass
```

**Step 3: Add inline indicator support**

When tool is called, add visual indicator inline:
```python
elif chunk["type"] == "tool_call":
    tool_name = chunk.get("tool", "")
    if streaming_message:
        current = streaming_message.content
        # Add indicator if not already present
        if not current.endswith("..."):
            streaming_message.update(current + "\n\n🔍 Querying data...")
```

**Step 4: Run tests**

```bash
cd /Users/fajril/Documents/GitHub/esdc && python -m pytest tests/test_chat_dom.py tests/test_chat_mounting.py -v
```

**Expected:** Tests pass with token streaming

**Step 5: Commit**

```bash
git add esdc/chat/app.py
git commit -m "feat: handle token events for real-time streaming display"
```

---

## Task 3: Update Tests for Token Streaming

**Problem:** Tests may expect old `message` type chunks

**Files:**
- Modify: `tests/test_chat_dom.py` (if needed)
- Modify: `tests/test_chat_mounting.py` (if needed)

**Step 1: Run tests to identify failures**

```bash
cd /Users/fajril/Documents/GitHub/esdc && python -m pytest tests/test_chat_dom.py tests/test_chat_mounting.py -v 2&1 | grep -E "(FAILED|PASSED)"
```

**Step 2: Update failing tests**

If tests check for chunk types, update to support `token`:
```python
# Old test
def test_chunk_type(self):
    assert chunk["type"] in ["message", "tool_call", "tool_result"]

# New test
def test_chunk_type(self):
    assert chunk["type"] in ["token", "message", "tool_call", "tool_result", "token_usage"]
```

**Step 3: Add token streaming test**

New test in `tests/test_chat_dom.py`:
```python
def test_token_streaming(self):
    """Verify token-level streaming produces incremental updates."""
    from esdc.chat.app import ChatMessage
    
    msg = ChatMessage("ai", "")
    # Simulate token streaming
    msg.update("Hello")
    msg.update("Hello world")
    msg.update("Hello world!")
    
    assert "Hello world!" in str(msg.render())
```

**Step 4: Run final tests**

```bash
cd /Users/fajril/Documents/GitHub/esdc && python -m pytest tests/test_chat_dom.py tests/test_chat_mounting.py -v
```

**Expected:** All 43+ tests pass

**Step 5: Commit**

```bash
git add tests/
git commit -m "test: update tests for token streaming support"
```

---

## Task 4: Final Verification

**Step 1: Code review**

Check the changes:
```bash
git diff HEAD~3
```

**Step 2: Manual test**

```bash
cd /Users/fajril/Documents/GitHub/esdc && esdc chat
```

**Verify:**
1. Type a query
2. Watch AI response appear character-by-character (not bursty)
3. When tool is called, see inline "🔍 Querying data..." indicator
4. Response continues after tool completes
5. No black space between user message and AI response
6. Smooth streaming like ChatGPT/GPT-4

**Step 3: Check log output**

Verify no errors:
```bash
tail -f esdc_chat.log
```

**Step 4: Final commit (if all good)**

```bash
git log --oneline -5
```

---

## Expected Behavior

**Before:**
- Bursty updates (full messages appear at once)
- Black space from intermediate widgets
- Jumpy UI

**After:**
- Token-by-token streaming (smooth character-by-character)
- Inline indicator (🔍) when tool is called
- Clean layout (just user + AI messages)
- Responsive, ChatGPT-like experience

---

## Execution Options

**Plan saved to `.opencode/plans/2025-03-27-real-time-streaming.md`**

**Options:**
1. **Subagent-Driven (this session)** - Dispatch agent per task
2. **Parallel Session (separate)** - New session with executing-plans
3. **Direct execution** - Implement all tasks now

**Recommendation:** Option 1 for step-by-step verification

Which approach do you prefer?
