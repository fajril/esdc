# Fix UI Issues Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix 3 UI issues: (1) ThinkingIndicator removal causing black space, (2) excessive spacing between collapsible widgets, (3) user message blue highlight should be minimal block quote style.

**Architecture:** Keep ThinkingIndicator widget but collapse it instead of removing to prevent layout collapse. Reduce CSS margins for tighter spacing. Convert user message from blue background highlight to subtle left border block quote style.

**Tech Stack:** Python, Textual TUI framework

---

## Task 1: Fix ThinkingIndicator Black Space

**Problem:** When `thinking.remove()` is called after query completion, it leaves empty black space where the widget was.

**Files:**
- Modify: `esdc/chat/app.py:1140,1148,1154` (3 locations)

**Step 1: Identify current code**

Read lines around 1137-1156 in `esdc/chat/app.py`:
```python
# Line 1137
try:
    await asyncio.wait_for(run_query(), timeout=120.0)
    # After streaming complete, remove thinking and mount AI message
    thinking.remove()  # <-- Line 1140
    if accumulated_content and self.chat_panel:
        streaming_message = ChatMessage("ai", accumulated_content)
        self.chat_panel.mount(streaming_message)
        streaming_message.scroll_visible()
        logger.info(f"Mounted AI message with {len(accumulated_content)} chars")
except asyncio.TimeoutError:
    logger.warning("Query timed out after 120 seconds")
    thinking.remove()  # <-- Line 1148
    self.display_message(
        "ai", "Request timed out after 2 minutes. Please try again."
    )
except Exception as e:
    logger.exception(f"Query failed with error: {e}")
    thinking.remove()  # <-- Line 1154
    self.display_message("ai", f"Error: {str(e)}")
```

**Step 2: Replace remove() with collapsed state**

Change all three occurrences from:
```python
thinking.remove()
```

To:
```python
thinking.collapsed = True
thinking.title = "✓ Thinking complete"
```

**Step 3: Run tests**

Command:
```bash
cd /Users/fajril/Documents/GitHub/esdc && python -m pytest tests/test_chat_dom.py tests/test_chat_mounting.py -v
```

Expected: All 43 tests pass

**Step 4: Manual test**

Command:
```bash
cd /Users/fajril/Documents/GitHub/esdc && esdc chat
```

Verify:
- Query a database
- ThinkingIndicator shows "✓ Thinking complete" in collapsed state after completion
- No black space where widget was removed
- Widget is expandable to see thinking steps

**Step 5: Commit**

```bash
git add esdc/chat/app.py
git commit -m "fix: collapse thinking indicator instead of remove to prevent black space"
```

---

## Task 2: Reduce Spacing Between Collapsible Widgets

**Problem:** Current CSS `margin: 1 0` creates too much vertical space between widgets.

**Files:**
- Modify: `esdc/chat/app.py:522,592,643`

**Step 1: Locate CSS sections**

Read around lines 519-527 (ThinkingIndicator CSS):
```
ThinkingIndicator {
    padding: 1 2;
    margin: 1 0;  # <-- Line 522
    ...
}
```

Read around lines 590-597 (SQLPanel CSS):
```
SQLPanel {
    margin: 1 0;  # <-- Line 592
    ...
}
```

Read around lines 641-648 (ResultsPanel CSS):
```
ResultsPanel {
    margin: 1 0;  # <-- Line 643
    ...
}
```

**Step 2: Update margins**

Change from `margin: 1 0` to `margin: 0 0 1 0` (top: 0, right: 0, bottom: 1, left: 0) on all three:

Line 522: `margin: 1 0;` → `margin: 0 0 1 0;`
Line 592: `margin: 1 0;` → `margin: 0 0 1 0;`
Line 643: `margin: 1 0;` → `margin: 0 0 1 0;`

**Step 3: Run tests**

Command:
```bash
cd /Users/fajril/Documents/GitHub/esdc && python -m pytest tests/test_chat_dom.py tests/test_chat_mounting.py -v
```

Expected: All 43 tests pass

**Step 4: Manual test**

Command:
```bash
cd /Users/fajril/Documents/GitHub/esdc && esdc chat
```

Verify:
- Spacing between widgets is tighter
- Widgets still have 1 line of space below them
- No overlap between widgets

**Step 5: Commit**

```bash
git add esdc/chat/app.py
git commit -m "fix: reduce spacing between collapsible widgets"
```

---

## Task 3: Convert User Message to Minimal Block Quote Style

**Problem:** User messages have prominent blue background (`$primary-darken-2`) which is too distracting. Should be minimal like block quotes in terminal.

**Files:**
- Modify: `esdc/chat/app.py:370-375` (CSS)
- Keep: `esdc/chat/app.py:392-393` (content formatting with `>`)

**Step 1: Locate user message CSS**

Read around lines 370-381 in `esdc/chat/app.py`:
```
ChatMessage.user {
    background: $primary-darken-2;
    color: $text;
    align-horizontal: right;
    border: none;
}
```

**Step 2: Update CSS to minimal block quote style**

Change from:
```css
ChatMessage.user {
    background: $primary-darken-2;
    color: $text;
    align-horizontal: right;
    border: none;
}
```

To:
```css
ChatMessage.user {
    background: transparent;
    color: $text;
    align-horizontal: right;
    border-left: solid $primary-darken-2;
    padding-left: 1;
}
```

**Step 3: Keep block quote content format**

Verify that content formatting at line 393 is preserved:
```python
if role == "user":
    formatted = f"> {content}"  # <-- This stays as is
```

**Step 4: Run tests**

Command:
```bash
cd /Users/fajril/Documents/GitHub/esdc && python -m pytest tests/test_chat_dom.py tests/test_chat_mounting.py -v
```

Expected: All 43 tests pass

**Step 5: Manual test**

Command:
```bash
cd /Users/fajril/Documents/GitHub/esdc && esdc chat
```

Verify:
- User messages no longer have blue background
- User messages have subtle left border in primary color
- Content still prefixed with `>` (block quote style)
- Alignment stays right-aligned
- AI messages unchanged (left-aligned, transparent background)

**Step 6: Commit**

```bash
git add esdc/chat/app.py
git commit -m "fix: convert user message to minimal block quote style"
```

---

## Final Verification

**Run all tests:**
```bash
cd /Users/fajril/Documents/GitHub/esdc && python -m pytest tests/test_chat_dom.py tests/test_chat_mounting.py -v
```

**Expected result:** All 43 tests pass

**Manual integration test:**
```bash
cd /Users/fajril/Documents/GitHub/esdc && esdc chat
```

**Verify all three fixes:**
1. Query database → ThinkingIndicator collapses (not removed) showing "✓ Thinking complete"
2. Widgets have tighter spacing between them
3. User messages show with left border block quote style, no blue background
4. No black space after query completion
5. All widgets remain visible and functional

---

## Execution Options

**Plan complete and saved to `.opencode/plans/2025-03-27-fix-ui-issues.md`. Two execution options:**

**1. Subagent-Driven (this session)** - I dispatch fresh subagent per task, review between tasks, fast iteration

**2. Parallel Session (separate)** - Open new session with executing-plans, batch execution with checkpoints

Which approach do you prefer?
