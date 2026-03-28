# Chat Flow Collapsibles Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Move collapsible blocks (Thinking, SQL, Results) from fixed widgets to dynamic messages that appear in the chat flow as AI processes.

**Architecture:** 
- Remove fixed `sql_panel`, `results_panel`, `thinking` from ChatPanel
- Create collapsible widgets dynamically during query processing
- Mount them to the message container in the order events occur
- When AI finishes, remove thinking collapsible, mount answer message

**Tech Stack:** Textual TUI, Collapsible widget, existing ThinkingIndicator/SQLPanel/ResultsPanel classes

---

### Task 1: Remove Fixed Panels from ChatPanel

**Files:**
- Modify: `esdc/chat/app.py:456-498` (ChatPanel class)

**Step 1: Update ChatPanel.__init__ to remove fixed panels**

Remove from `__init__`:
```python
self.sql_panel = SQLPanel()
self.results_panel = ResultsPanel()
self.thinking = ThinkingIndicator()
```

Keep only:
```python
def __init__(self):
    super().__init__()
    self.messages: list[tuple[str, str]] = []
    self._message_container: ScrollableContainer | None = None
```

**Step 2: Update ChatPanel.compose to only yield message container**

Change from:
```python
def compose(self) -> ComposeResult:
    yield ScrollableContainer(id="message-container")
    yield self.thinking
    yield self.sql_panel
    yield self.results_panel
```

To:
```python
def compose(self) -> ComposeResult:
    yield ScrollableContainer(id="message-container")
```

**Step 3: Remove set_sql and set_results methods from ChatPanel**

These methods were for fixed panels, no longer needed.

**Step 4: Add mount_collapsible method to ChatPanel**

```python
def mount_collapsible(self, collapsible: "Collapsible") -> None:
    """Mount a collapsible widget to the message container."""
    if self._message_container:
        self._message_container.mount(collapsible)
```

**Step 5: Update tests**

In `tests/test_chat_app.py`, remove tests that reference:
- `chat_panel.sql_panel`
- `chat_panel.results_panel`
- `chat_panel.thinking`
- `chat_panel.set_sql`
- `chat_panel.set_results`

**Step 6: Run tests**

```bash
uv run pytest tests/test_chat_app.py -v
```

Expected: Tests pass or need minimal updates

**Step 7: Commit**

```bash
git add esdc/chat/app.py tests/test_chat_app.py
git commit -m "refactor: ChatPanel only contains message container"
```

---

### Task 2: Create Factory Functions for Collapsibles

**Files:**
- Modify: `esdc/chat/app.py` (add helper functions)

**Step 1: Add create_thinking_collapsible function**

```python
def create_thinking_collapsible() -> ThinkingIndicator:
    """Create a thinking indicator collapsible for the chat flow."""
    return ThinkingIndicator()
```

**Step 2: Add create_sql_collapsible function**

```python
def create_sql_collapsible(sql: str) -> "Collapsible":
    """Create a SQL query collapsible for the chat flow."""
    from textual.widgets import Collapsible, Static
    
    collapsible = Collapsible(title="📝 SQL Query", collapsed=False)
    content = Static(f"```sql\n{sql}\n```", classes="sql-content")
    # Mount content inside collapsible
    collapsible._content_widgets = [content]
    return collapsible
```

**Step 3: Add create_results_collapsible function**

```python
def create_results_collapsible(results: str) -> "Collapsible":
    """Create a query results collapsible for the chat flow."""
    from textual.widgets import Collapsible, Static
    
    collapsible = Collapsible(title="📊 Query Results", collapsed=False)
    content = Static(results, classes="results-content")
    collapsible._content_widgets = [content]
    return collapsible
```

**Step 4: Run tests**

```bash
uv run pytest tests/test_chat_app.py -v
```

**Step 5: Commit**

```bash
git add esdc/chat/app.py
git commit -m "feat: add factory functions for collapsibles"
```

---

### Task 3: Update Query Flow to Mount Collapsibles Dynamically

**Files:**
- Modify: `esdc/chat/app.py:992-1060` (on_input_submitted method)

**Step 1: Change thinking handling to create and mount dynamically**

Replace:
```python
if self.chat_panel and self.chat_panel.thinking:
    self.chat_panel.thinking.remove()

if self.chat_panel:
    self.chat_panel.thinking = ThinkingIndicator()
    self.chat_panel.mount(self.chat_panel.thinking)
```

With:
```python
# Create thinking indicator for this query
thinking = create_thinking_collapsible()
if self.chat_panel:
    self.chat_panel.mount_collapsible(thinking)
```

**Step 2: Update tool_call handler to add steps to thinking**

Keep existing:
```python
elif chunk["type"] == "tool_call":
    tool_name = chunk.get("tool", "")
    if self.chat_panel and self.chat_panel.thinking:
        self.chat_panel.thinking.add_step(f"Running: {tool_name}")
```

Change to:
```python
elif chunk["type"] == "tool_call":
    tool_name = chunk.get("tool", "")
    thinking.add_step(f"Running: {tool_name}")
```

**Step 3: Update tool_result handler to mount SQL and Results**

Replace:
```python
elif chunk["type"] == "tool_result":
    result = chunk.get("result", "")
    if result and self.chat_panel:
        self.chat_panel.set_sql(chunk.get("sql", ""))
        self.chat_panel.set_results(result)
```

With:
```python
elif chunk["type"] == "tool_result":
    result = chunk.get("result", "")
    if result and self.chat_panel:
        # Mount SQL collapsible in chat flow
        sql = chunk.get("sql", "")
        if sql:
            sql_collapsible = SQLPanel()
            sql_collapsible.set_sql(sql)
            self.chat_panel.mount_collapsible(sql_collapsible)
        
        # Mount Results collapsible in chat flow
        results_collapsible = ResultsPanel()
        results_collapsible.set_results(result)
        self.chat_panel.mount_collapsible(results_collapsible)
```

**Step 4: Update message handler to remove thinking and mount answer**

Replace:
```python
if chunk["type"] == "message":
    content = chunk.get("content", "")
    if content:
        if self.chat_panel and self.chat_panel.thinking:
            self.chat_panel.thinking.remove()
        self.display_message("ai", content)
```

With:
```python
if chunk["type"] == "message":
    content = chunk.get("content", "")
    if content:
        # Remove thinking indicator when answer arrives
        thinking.remove()
        # Mount AI answer in chat flow
        self.display_message("ai", content)
```

**Step 5: Update error handlers**

Replace:
```python
except asyncio.TimeoutError:
    if self.chat_panel and self.chat_panel.thinking:
        self.chat_panel.thinking.remove()
    self.display_message("ai", "Request timed out...")
except Exception as e:
    if self.chat_panel and self.chat_panel.thinking:
        self.chat_panel.thinking.remove()
    self.display_message("ai", f"Error: {str(e)}")
```

With:
```python
except asyncio.TimeoutError:
    thinking.remove()
    self.display_message("ai", "Request timed out after 2 minutes. Please try again.")
except Exception as e:
    thinking.remove()
    self.display_message("ai", f"Error: {str(e)}")
```

**Step 6: Remove cancel_query action's reference to thinking**

In `action_cancel_query()`, remove:
```python
if self.chat_panel and self.chat_panel.thinking:
    self.chat_panel.thinking.remove()
```

(The thinking will be tracked locally in the async function, not as instance variable)

**Step 7: Test manually**

```bash
uv run python esdc/chat/app.py
```

Ask a question and verify:
- Thinking appears in chat flow
- SQL appears after thinking (if SQL was run)
- Results appear after SQL
- Thinking disappears when answer arrives
- Answer appears at the end

**Step 8: Commit**

```bash
git add esdc/chat/app.py
git commit -m "feat: mount collapsibles dynamically during query processing"
```

---

### Task 4: Clean Up and Final Verification

**Files:**
- Modify: `esdc/chat/app.py`
- Run: Full test suite

**Step 1: Check for unused code**

- Verify `create_thinking_collapsible` etc. are used
- Remove any dead code

**Step 2: Run full test suite**

```bash
uv run pytest tests/test_chat_app.py -v
```

Expected: All tests pass

**Step 3: Manual verification**

```bash
uv run python esdc/chat/app.py
```

Test multiple scenarios:
1. Simple question (no tools) → thinking appears, then answer
2. SQL query → thinking appears, SQL appears, results appear, answer appears
3. Cancel query → thinking removed, error message shown

**Step 4: Update documentation**

Update `docs/ui-layout.md` to reflect dynamic collapsibles.

**Step 5: Final commit**

```bash
git add esdc/chat/app.py tests/test_chat_app.py docs/ui-layout.md
git commit -m "chore: clean up and finalize dynamic collapsibles"
```

---

**Plan complete.**

**Two execution options:**

**1. Subagent-Driven (this session)** - I dispatch fresh subagent per task, review between tasks, fast iteration

**2. Parallel Session (separate)** - Open new session with executing-plans, batch execution with checkpoints

**Which approach?**