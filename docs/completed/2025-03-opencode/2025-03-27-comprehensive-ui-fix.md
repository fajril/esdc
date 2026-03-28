# Comprehensive UI Fix Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix remaining visual issues: excessive black space, duplicate tool call logs, and content whitespace

**Architecture:** Optimize CSS margins, deduplicate tool call tracking, strip content whitespace

**Tech Stack:** Python, Textual TUI

---

## Task 1: Fix ChatMessage Margin Causing Black Space

**Problem:** `ChatMessage` has `margin: 1 0` creating gaps between messages

**Files:**
- Modify: `esdc/chat/app.py:366`

**Step 1: Locate current CSS**

Line 363-369:
```css
ChatMessage {
    padding: 1 2;
    margin: 1 0;  /* <-- Too much space */
    border: none;
    max-width: 85%;
}
```

**Step 2: Update margin**

Change from:
```css
margin: 1 0;
```

To:
```css
margin: 0 0 1 0;  /* Only bottom margin */
```

**Step 3: Verify**

- User messages still have spacing
- AI messages have less gap from previous content

**Step 4: Commit**

```bash
git add esdc/chat/app.py
git commit -m "fix: reduce ChatMessage margin to eliminate black space"
```

---

## Task 2: Deduplicate Tool Call Steps

**Problem:** Same tool called multiple times creating duplicate "Running: execute_sql" entries

**Files:**
- Modify: `esdc/chat/app.py:1079-1084`

**Step 1: Add tracking for seen tools**

Add `seen_tools` set before the async loop (around line 1063):

```python
# Track tools already added to prevent duplicates
seen_tools = set()
```

**Step 2: Update tool_call handler**

Change from (lines 1079-1084):
```python
elif chunk["type"] == "tool_call":
    tool_name = chunk.get("tool", "")
    thinking.add_step(f"Running: {tool_name}")
    logger.debug(
        f"Added thinking step: {tool_name}, total steps: {len(thinking.steps)}"
    )
```

To:
```python
elif chunk["type"] == "tool_call":
    tool_name = chunk.get("tool", "")
    if tool_name not in seen_tools:
        seen_tools.add(tool_name)
        thinking.add_step(f"Running: {tool_name}")
        logger.debug(
            f"Added thinking step: {tool_name}, total steps: {len(thinking.steps)}"
        )
```

**Step 3: Test**

- Query should only show "Running: execute_sql" once per unique tool
- Multiple SQL executions in one query still show once

**Step 4: Commit**

```bash
git add esdc/chat/app.py
git commit -m "fix: deduplicate tool call steps in thinking indicator"
```

---

## Task 3: Strip Excess Whitespace from AI Content

**Problem:** AI messages may have leading/trailing newlines from SQL filtering

**Files:**
- Modify: `esdc/chat/app.py:1143-1147`

**Step 1: Locate AI message mounting**

Lines 1143-1147:
```python
if accumulated_content and self.chat_panel:
    streaming_message = ChatMessage("ai", accumulated_content)
    self.chat_panel.mount(streaming_message)
```

**Step 2: Strip whitespace before creating message**

Change to:
```python
if accumulated_content and self.chat_panel:
    # Strip excess whitespace from filtered content
    cleaned_content = accumulated_content.strip()
    if cleaned_content:  # Only mount if there's actual content
        streaming_message = ChatMessage("ai", cleaned_content)
        self.chat_panel.mount(streaming_message)
        streaming_message.scroll_visible()
```

**Step 3: Test**

- Messages don't start with blank lines
- No excessive spacing at end of messages

**Step 4: Commit**

```bash
git add esdc/chat/app.py
git commit -m "fix: strip excess whitespace from AI message content"
```

---

## Task 4: Optimize Collapsible Widget Height When Collapsed

**Problem:** Collapsed widgets still take significant vertical space

**Files:**
- Modify: `esdc/chat/app.py` (CSS sections for all 3 widgets)

**Step 1: Add collapsed height optimization**

For ThinkingIndicator (around line 519), add:
```css
ThinkingIndicator.collapsed {
    height: auto;
    min-height: 1;
}
```

For SQLPanel (around line 590), add:
```css
SQLPanel.collapsed {
    height: auto;
    min-height: 1;
}
```

For ResultsPanel (around line 641), add:
```css
ResultsPanel.collapsed {
    height: auto;
    min-height: 1;
}
```

**Step 2: Test**

- Collapsed widgets take minimal vertical space
- Still clickable to expand

**Step 3: Commit**

```bash
git add esdc/chat/app.py
git commit -m "fix: minimize collapsed widget height"
```

---

## Task 5: Final Integration Test

**Run all tests:**
```bash
cd /Users/fajril/Documents/GitHub/esdc && python -m pytest tests/test_chat_dom.py tests/test_chat_mounting.py -v
```

**Manual verification:**
```bash
cd /Users/fajril/Documents/GitHub/esdc && esdc chat
```

**Verify:**
1. Black space between messages is minimal
2. Each tool appears only once in thinking indicator
3. AI messages don't have leading/trailing blank lines
4. Collapsed widgets are compact
5. All tests pass

**Final commit:**
```bash
git log --oneline -5  # Review commits
git push  # If ready
```

---

## Execution Options

**Plan saved to `.opencode/plans/2025-03-27-comprehensive-ui-fix.md`**

**Options:**
1. **Subagent-Driven (this session)** - I dispatch per task
2. **Parallel Session (separate)** - Use executing-plans in new session
3. **Direct execution** - I implement all tasks now

Which approach do you prefer?
