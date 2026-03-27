# Simplify Chat UI - Remove Complex Widgets

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Remove complex intermediate widgets (ThinkingIndicator, SQLPanel, ResultsPanel) and implement simple real-time streaming chat like standard LLM interfaces.

**Architecture:** Simplify to only show user message and AI response. Remove all Collapsible widgets that cause layout issues. Stream AI response token-by-token in real-time.

**Tech Stack:** Python, Textual TUI

---

## Task 1: Remove ThinkingIndicator from on_input_submitted

**Files:**
- Modify: `esdc/chat/app.py:1071-1176`

**Problem:** ThinkingIndicator creates black space when collapsed

**Changes needed:**

1. Remove ThinkingIndicator creation (line 1072-1076):
```python
# REMOVE this block:
# Create thinking indicator for this query and mount to chat flow
thinking = ThinkingIndicator()
if self.chat_panel:
    self.chat_panel.mount(thinking)
    thinking.scroll_visible()
    logger.debug("Mounted ThinkingIndicator as Collapsible widget")
```

2. Remove seen_tools tracking (line 1078-1079):
```python
# REMOVE:
# Track tools already added to prevent duplicates
seen_tools = set()
```

3. Update tool_call handler (lines 1097-1104) - just log, no UI:
```python
elif chunk["type"] == "tool_call":
    tool_name = chunk.get("tool", "")
    logger.debug(f"Tool called: {tool_name}")
```

4. Remove SQLPanel and ResultsPanel mounting (lines 1120-1140):
```python
# REMOVE entire block that mounts SQLPanel and ResultsPanel
```

5. Remove thinking cleanup from success/error handlers (lines 1161-1162, 1173-1174, 1180-1181):
```python
# REMOVE all thinking.collapsed and thinking.title assignments
```

**Step-by-step:**

**Step 1:** Read lines 1071-1182 to understand full context

**Step 2:** Comment out or remove ThinkingIndicator-related code

**Step 3:** Simplify tool_call handler to just logging

**Step 4:** Remove SQLPanel/ResultsPanel mounting

**Step 5:** Run tests
```bash
cd /Users/fajril/Documents/GitHub/esdc && python -m pytest tests/test_chat_dom.py tests/test_chat_mounting.py -v
```
Expected: Tests may fail initially if they test removed widgets

**Step 6:** Commit
```bash
git add esdc/chat/app.py
git commit -m "refactor: remove ThinkingIndicator, SQLPanel, ResultsPanel mounting"
```

---

## Task 2: Implement Real-time Streaming AI Response

**Files:**
- Modify: `esdc/chat/app.py:1053-1182`

**Problem:** Currently accumulating content and mounting at end. Should stream token-by-token.

**Changes needed:**

1. Create streaming message at start (after user message), update it in real-time:

Replace lines 1081-1096:
```python
# Accumulate AI response content for mounting after streaming
accumulated_content = ""
```

With:
```python
# Create streaming AI message
streaming_message = ChatMessage("ai", "")
if self.chat_panel:
    self.chat_panel.mount(streaming_message)
    streaming_message.scroll_visible()
```

2. Update streaming logic to append content in real-time:

Replace lines 1092-1096:
```python
if chunk["type"] == "message":
    content = chunk.get("content", "")
    if content:
        # Accumulate content without mounting yet
        accumulated_content += content
```

With:
```python
if chunk["type"] == "message":
    content = chunk.get("content", "")
    if content and streaming_message:
        # Append content in real-time
        current = streaming_message.content
        streaming_message.update(current + content)
```

3. Remove final mounting logic since message already created:

Replace lines 1158-1170:
```python
try:
    await asyncio.wait_for(run_query(), timeout=120.0)
    # After streaming complete, collapse thinking and mount AI message
    thinking.collapsed = True
    thinking.title = "✓ Thinking complete"
    if accumulated_content and self.chat_panel:
        # Strip excess whitespace from filtered content
        cleaned_content = accumulated_content.strip()
        if cleaned_content:  # Only mount if there's actual content
            streaming_message = ChatMessage("ai", cleaned_content)
            self.chat_panel.mount(streaming_message)
            streaming_message.scroll_visible()
            logger.info(f"Mounted AI message with {len(cleaned_content)} chars")
```

With:
```python
try:
    await asyncio.wait_for(run_query(), timeout=120.0)
    logger.info("Query completed successfully")
except asyncio.TimeoutError:
    logger.warning("Query timed out after 120 seconds")
    if streaming_message:
        streaming_message.update("Request timed out after 2 minutes. Please try again.")
except Exception as e:
    logger.exception(f"Query failed with error: {e}")
    if streaming_message:
        streaming_message.update(f"Error: {str(e)}")
```

**Step-by-step:**

**Step 1:** Create streaming message at start of query

**Step 2:** Update message content in real-time as tokens arrive

**Step 3:** Handle timeout and errors by updating the existing message

**Step 4:** Run tests
```bash
cd /Users/fajril/Documents/GitHub/esdc && python -m pytest tests/test_chat_dom.py tests/test_chat_mounting.py -v
```

**Step 5:** Commit
```bash
git add esdc/chat/app.py
git commit -m "feat: implement real-time streaming AI responses"
```

---

## Task 3: Remove Unused Widget Classes and CSS

**Files:**
- Modify: `esdc/chat/app.py`

**Problem:** ThinkingIndicator, SQLPanel, ResultsPanel classes no longer used but still in code

**Note:** Keep classes for now (they might be used in tests), just remove CSS from App class

**Changes needed:**

The App class DEFAULT_CSS still references these widgets. Keep the classes but they're no longer mounted.

**Alternative:** If you want to delete classes completely:

1. Remove ThinkingIndicator class (lines 517-586)
2. Remove SQLPanel class (lines 593-636)
3. Remove ResultsPanel class (lines 649-722)
4. Remove mount_collapsible and mount_collapsible_async methods from ChatPanel

**Recommendation:** Keep classes, just don't mount them. This allows tests to still pass.

**Step 1:** Verify classes are not imported elsewhere
```bash
grep -r "ThinkingIndicator\|SQLPanel\|ResultsPanel" esdc/ --include="*.py" | grep -v "app.py"
```

**Step 2:** If only used in app.py and tests, skip deletion for now

**Step 3:** Commit (if deletions made)
```bash
git add esdc/chat/app.py
git commit -m "refactor: remove unused Collapsible widget classes"
```

---

## Task 4: Update Tests for Simplified UI

**Files:**
- Modify: `tests/test_chat_dom.py`
- Modify: `tests/test_chat_mounting.py`

**Problem:** Tests may fail if they expect removed widgets

**Changes needed:**

1. Remove or update tests for ThinkingIndicator
2. Remove or update tests for SQLPanel
3. Remove or update tests for ResultsPanel
4. Add test for real-time streaming

**Step-by-step:**

**Step 1:** Run tests to see which fail
```bash
cd /Users/fajril/Documents/GitHub/esdc && python -m pytest tests/test_chat_dom.py tests/test_chat_mounting.py -v
```

**Step 2:** Update failing tests

For tests checking removed widgets, either:
- Remove the test
- Update to test streaming behavior instead

**Step 3:** Add new test for streaming
```python
def test_ai_message_streams_realtime(self):
    """AI messages should update content in real-time."""
    msg = ChatMessage("ai", "")
    # Simulate streaming
    msg.update("Hello")
    msg.update("Hello world")
```

**Step 4:** Run tests
```bash
cd /Users/fajril/Documents/GitHub/esdc && python -m pytest tests/test_chat_dom.py tests/test_chat_mounting.py -v
```
Expected: All tests pass

**Step 5:** Commit
```bash
git add tests/
git commit -m "test: update tests for simplified chat UI"
```

---

## Task 5: Final Verification

**Manual test:**
```bash
cd /Users/fajril/Documents/GitHub/esdc && esdc chat
```

**Verify:**
1. User message appears (with left border style)
2. AI response streams in real-time (token by token)
3. No black space between messages
4. No intermediate widgets (thinking, sql, results)
5. Query completes successfully
6. Final message is complete and readable

**Visual check:**
- Layout is clean with just user and AI messages
- No borders or boxes except user message left border
- Spacing is minimal between messages
- No collapsed widgets taking space

**Final commit:**
```bash
git log --oneline -10
```

---

## Summary

**Before:** Complex UI with 3 Collapsible widgets causing black space

**After:** Simple chat UI like standard LLM interfaces
- User message (with left border)
- AI response (streaming real-time)
- Clean layout, no intermediate widgets

**Benefits:**
1. No black space issues
2. Simpler code (less than 100 lines removed)
3. Real-time streaming feels more responsive
4. Easier to maintain
5. Consistent with standard chat interfaces (ChatGPT, Claude, etc.)

---

## Execution Options

**Plan saved to `.opencode/plans/2025-03-27-simplify-chat-ui.md`**

**Options:**
1. **Subagent-Driven (this session)** - I dispatch per task
2. **Parallel Session (separate)** - Use executing-plans in new session
3. **Direct execution** - I implement all tasks now

**Recommendation:** Option 1 (Subagent-Driven) so we can review each task before proceeding.

Which approach do you prefer?
