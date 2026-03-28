# Chat Layout Collapsible Widget Fix Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix ESDC TUI chat layout to display Thinking, SQL Query, and Results as separate Collapsible widgets in correct chronological order (User → Thinking → SQL → Results → AI Answer), with right panel showing only Session Info.

**Architecture:** Refactor mounting logic to create Collapsible containers as standalone widgets in the chat flow, not as content within AI messages. Clean up CSS to avoid widget conflicts. Right panel simplified to display static Session Info only.

**Tech Stack:** Textual TUI, Python 3.11+

---

## Current State Analysis

### Problems Identified from Screenshot:

1. **SQL Appears as Raw Code Block, Not Collapsible Widget**
   - Shows: ` ```sql \nSELECT...``` ` as raw text
   - Should be: SQLPanel Collapsible with "📝 SQL Query" header
   
2. **Results Appears as Raw Markdown Table, Not Collapsible Widget**
   - Shows: Raw markdown table in AI message
   - Should be: ResultsPanel Collapsible with "📊 Query Results" header
   
3. **Thinking Shows Plain Text, Not Collapsible**
   - Shows: "Running: execute_sql" as plain text
   - Should be: ThinkingIndicator Collapsible with "▶ Thinking..." header
   
4. **Widget Ordering Wrong**
   - Current: User → Thinking (plain) → AI Message (with SQL inside) → Results (inside AI)
   - Target: User → Thinking (collapsible) → SQL (collapsible) → Results (collapsible) → AI Message

5. **Right Panel Shows Document Metadata**
   - Shows: "alaman 31 dari 65", "Revisi ke: 03" (retrieved document info)
   - Should be: Only Session Info (Provider, Model, Thread)

### Root Causes:

1. **Agent yields message with embedded SQL** - SQL filtering logic exists but may not be working correctly
2. **SQLPanel and ResultsPanel mount correctly but content displays wrong** - Widgets mount but show raw content instead of formatted display
3. **ThinkingIndicator uses wrong mounting method** - Should mount as Collapsible widget, not plain text
4. **CSS conflicts between multiple definitions** - ChatMessage CSS defined 3 times (class, panel, app levels)
5. **Document metadata in LLM context** - Retrieved documents included in LLM context but should not be displayed

---

## Implementation Plan

### Task 1: Fix CSS Conflicts for ChatMessage

**Files:**
- Modify: `esdc/chat/app.py` - Remove duplicate ChatMessage CSS

**Analysis:**
Current code has ChatMessage CSS defined in 3 places:
1. Lines 354-380: ChatMessage.DEFAULT_CSS
2. Lines 482-508: Inside ChatPanel.DEFAULT_CSS (duplicate)
3. Lines 809-838: Inside ESDCChatApp.CSS (triple duplicate)

This causes specificity conflicts where styles override each other unpredictably.

**Step 1: Read current CSS definitions**

```bash
grep -n "ChatMessage {" esdc/chat/app.py | head -10
```

**Step 2: Keep only ChatMessage.DEFAULT_CSS, remove duplicates**

Remove ChatMessage CSS from:
- ChatPanel.DEFAULT_CSS (lines ~482-508)
- ESDCChatApp.CSS (lines ~809-838)

Keep only the class-level DEFAULT_CSS on lines 354-380.

**Step 3: Verify CSS compiles**

```python
python -c "from esdc.chat.app import ChatMessage, ChatPanel, ESDCChatApp; print('CSS OK')"
```

**Step 4: Commit**

```bash
git add esdc/chat/app.py
git commit -m "fix: remove duplicate ChatMessage CSS definitions"
```

---

### Task 2: Fix ThinkingIndicator to Display as Collapsible

**Files:**
- Modify: `esdc/chat/app.py:1085-1089`

**Analysis:**
Current code:
```python
thinking = ThinkingIndicator()
if self.chat_panel:
    self.chat_panel.mount_collapsible(thinking)
```

ThinkingIndicator IS a Collapsible (extends Collapsible), so it should mount directly to chat panel, not use mount_collapsible.

**Step 1: Change mounting method**

Replace:
```python
thinking = ThinkingIndicator()
if self.chat_panel:
    self.chat_panel.mount_collapsible(thinking)
```

With:
```python
thinking = ThinkingIndicator()
if self.chat_panel:
    self.chat_panel.mount(thinking)
    logger.debug("Mounted ThinkingIndicator as Collapsible widget")
```

**Step 2: Test ThinkingIndicator display**

Run query and verify:
- Shows collapsible header "▶ Thinking..."
- Shows step count in title
- Expands to show "Running: execute_sql" inside

**Step 3: Commit**

```bash
git add esdc/chat/app.py
git commit -m "fix: mount ThinkingIndicator as Collapsible widget"
```

---

### Task 3: Fix SQLPanel Content Display

**Files:**
- Modify: `esdc/chat/app.py:632-638` (SQLPanel.compose)
- Modify: `esdc/chat/app.py:601-625` (SQLPanel.DEFAULT_CSS)

**Analysis:**
Current SQLPanel content shows as raw markdown. The compose method uses Static with markdown code block:
```python
def compose(self) -> ComposeResult:
    content = f"```sql\n{self.sql_content}\n```" if self.sql_content else "Executing query..."
    yield Static(content, classes="sql-content")
```

This renders as text, not formatted SQL. Need to use Markdown widget or Static with proper styling.

**Step 1: Fix SQLPanel.compose to use proper formatting**

Current:
```python
def compose(self) -> ComposeResult:
    content = (
        f"```sql\n{self.sql_content}\n```"
        if self.sql_content
        else "Executing query..."
    )
    yield Static(content, classes="sql-content")
```

Better:
```python
def compose(self) -> ComposeResult:
    if self.sql_content:
        # Use Markdown for syntax highlighting
        yield Markdown(f"```sql\n{self.sql_content}\n```", classes="sql-content")
    else:
        yield Static("Executing query...", classes="sql-content")
```

**Step 2: Update CSS for SQLPanel content**

Add to SQLPanel.DEFAULT_CSS:
```css
SQLPanel Markdown.sql-content {
    background: $surface;
    color: $text;
}
```

**Step 3: Test SQLPanel display**

Run query and verify:
- SQLPanel shows as Collapsible with "📝 SQL Query" header
- SQL displays with syntax highlighting
- Can expand/collapse

**Step 4: Commit**

```bash
git add esdc/chat/app.py
git commit -m "fix: format SQLPanel content with proper markdown rendering"
```

---

### Task 4: Fix ResultsPanel Content Display

**Files:**
- Modify: `esdc/chat/app.py:678-684` (ResultsPanel.compose)

**Analysis:**
Current ResultsPanel uses Markdown for results:
```python
def compose(self) -> ComposeResult:
    content = self._format_results_as_markdown(self.results_content)
    yield Markdown(content, classes="results-content")
```

This should work for markdown tables. But verify the _format_results_as_markdown is working.

**Step 1: Read current implementation**

```bash
grep -A 30 "_format_results_as_markdown" esdc/chat/app.py
```

**Step 2: Fix formatting if needed**

Ensure results are properly formatted as markdown tables. The current implementation looks correct.

**Step 3: Test ResultsPanel display**

Run query and verify:
- ResultsPanel shows as Collapsible with "📊 Query Results" header
- Results display as formatted markdown table
- Can expand/collapse

**Step 4: Commit**

```bash
git add esdc/chat/app.py
git commit -m "fix: ResultsPanel content display"
```

---

### Task 5: Fix Widget Ordering (Chronological Flow)

**Files:**
- Modify: `esdc/chat/app.py:1091-1115`

**Analysis:**
Current order of operations:
1. Mount ThinkingIndicator (line 1088)
2. Start streaming (line 1098)
3. Mount ChatMessage during streaming (line 1105-1108)
4. Mount SQLPanel on tool_result (line 1142)
5. Mount ResultsPanel on tool_result (line 1150)

Problem: AI message mounts BEFORE SQL/Results, so they appear after AI message.

**Step 1: Defer AI message mounting**

Change logic to:
1. Mount ThinkingIndicator
2. Accumulate AI content (don't mount yet)
3. On tool_result: mount SQLPanel, ResultsPanel
4. After all tool_results: remove Thinking, then mount AI message

**Step 2: Implement deferred mounting**

Replace streaming logic in run_query():

```python
async def run_query():
    nonlocal streaming_message, accumulated_content
    logger.debug("Starting query execution")
    
    # Accumulate content without mounting
    async for chunk in self._stream_response(user_input):
        if self._cancelled:
            self.display_message("system", "Query cancelled.")
            return
        if chunk["type"] == "message":
            content = chunk.get("content", "")
            if content:
                accumulated_content += content
        elif chunk["type"] == "tool_call":
            tool_name = chunk.get("tool", "")
            thinking.add_step(f"Running: {tool_name}")
        elif chunk["type"] == "tool_result":
            result = chunk.get("result", "")
            sql = chunk.get("sql", "")
            
            if result:
                if sql:
                    sql_panel = SQLPanel(sql)
                    self.chat_panel.mount(sql_panel)
                results_panel = ResultsPanel(result)
                self.chat_panel.mount(results_panel)
    
    # After streaming complete, mount AI message
    thinking.remove()
    if accumulated_content:
        self.chat_panel.add_message("ai", accumulated_content)
```

**Step 3: Test ordering**

Run query and verify order:
1. User message
2. ThinkingIndicator (collapsible)
3. SQLPanel (collapsible)
4. ResultsPanel (collapsible)
5. AI message

**Step 4: Commit**

```bash
git add esdc/chat/app.py
git commit -m "fix: ensure correct widget ordering in chat flow"
```

---

### Task 6: Fix Right Panel to Show Only Session Info

**Files:**
- Modify: `esdc/chat/app.py:282-348` (ContextPanel)
- Modify: `esdc/chat/app.py:779-784` (#context-panel CSS)

**Analysis:**
ContextPanel currently only shows Session Info (which is correct). The "alaman 31 dari 65" text is from LLM response content (retrieved documents), not from ContextPanel.

However, we should ensure ContextPanel is truly static and not scrollable.

**Step 1: Fix CSS for right panel**

Add to #context-panel CSS:
```css
#context-panel {
    width: 1fr;
    height: 100%;
    border-left: solid $surface;
    background: $surface;
    padding: 1;
    overflow: hidden;  /* Not scrollable */
}
```

**Step 2: Test right panel**

Verify:
- Right panel shows only "Session Info" section
- Displays Provider, Model, Thread
- Panel is static (not scrollable)

**Step 3: Commit**

```bash
git add esdc/chat/app.py
git commit -m "fix: ensure right panel is static and shows only Session Info"
```

---

### Task 7: Fix Document Metadata in LLM Response

**Files:**
- Modify: `esdc/chat/prompts.py` (if exists and has RAG prompts)
- Modify: `esdc/chat/agent.py` (if RAG context is added to messages)

**Analysis:**
The "alaman 31 dari 65" text is document metadata from retrieved documents. This should not be displayed to user. Need to investigate where RAG context is added and ensure it's not included in visible output.

**Step 1: Search for RAG/document retrieval**

```bash
grep -r "retriev\|RAG\|context\|document" esdc/chat/ --include="*.py" | head -20
```

**Step 2: If found, fix to exclude from output**

Ensure retrieved documents are only in LLM context, not in message content.

**Step 3: Commit**

```bash
git add esdc/chat/prompts.py esdc/chat/agent.py
git commit -m "fix: exclude document metadata from user-visible output"
```

---

### Task 8: Comprehensive Testing

**Files:**
- Test: Manual via UI

**Step 1: Run full test scenario**

```bash
# Start app
python -m esdc chat

# Test query: "berapa cadangan lapangan karamba di wilayah kerja wain?"
```

**Step 2: Verify all requirements**

- [ ] User message appears correctly
- [ ] ThinkingIndicator appears as Collapsible with "▶ Thinking..." header
- [ ] SQLPanel appears as Collapsible with "📝 SQL Query" header
- [ ] ResultsPanel appears as Collapsible with "📊 Query Results" header
- [ ] AI message appears after all collapsibles
- [ ] Right panel shows only Session Info (Provider, Model, Thread)
- [ ] No document metadata visible in chat
- [ ] All collapsibles can expand/collapse
- [ ] SQL displays with syntax highlighting
- [ ] Results display as formatted table

**Step 3: Test edge cases**

- Query without SQL execution (simple chat)
- Query with empty results
- Multiple queries in sequence

**Step 4: Commit final changes**

```bash
git add -A
git commit -m "feat: implement clean collapsible chat layout with proper ordering"
```

---

## Verification Commands

**Check implementation:**
```bash
# Test imports
python -c "from esdc.chat.app import ESDCChatApp; print('Imports OK')"

# Check CSS syntax
python -c "from esdc.chat.app import ESDCChatApp; print('CSS OK')"

# Run app
python -m esdc chat
```

**Check logs:**
```bash
# Monitor debug logs
tail -f esdc_chat.log | grep -E "MOUNT|DEBUG_TOOL|AGENT_"
```

---

## Files Modified Summary

1. `esdc/chat/app.py` - CSS fixes, mounting logic, widget ordering, right panel
2. `esdc/chat/agent.py` - Document metadata filtering (if needed)
3. `esdc/chat/prompts.py` - RAG prompt fixes (if needed)

---

**Estimated Time:** 3-4 hours
**Priority:** High
**Breaking Changes:** None (UI only)
