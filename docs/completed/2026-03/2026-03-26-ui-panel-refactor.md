# UI Panel Refactor Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Refactor UI layout so right panel is static/minimal and all dynamic content (thinking, SQL, results) appears in collapsible blocks on the left panel.

**Architecture:** 
- Right panel becomes a narrow static sidebar showing only session info (Provider, Model, Thread)
- Left panel contains all chat content including collapsible blocks for thinking, SQL query, and results table
- Remove scrollable container from right panel, make it fixed width

**Tech Stack:** Textual TUI framework, Python, existing Collapsible widget

---

### Task 1: Make Right Panel Static (Not Scrollable)

**Files:**
- Modify: `esdc/chat/app.py:275-290` (ContextPanel class)

**Step 1: Change ContextPanel base class from ScrollableContainer to Vertical**

```python
class ContextPanel(Vertical):  # Was: ScrollableContainer
    """Context panel with minimal session info only."""

    DEFAULT_CSS = """
    ContextPanel {
        width: 25%;
        padding: 1;
        background: $surface;
    }
    """
```

**Step 2: Remove all sections except Session Info**

Remove from compose():
- Token Usage section
- Tools Available section  
- Last SQL Query section
- Query Results section
- Schema Browser section
- Query History section

Keep only:
```python
def compose(self) -> ComposeResult:
    with ContextSection(
        "Session Info",
        expanded=True,
        id="session-section",
    ):
        yield Static(
            f"Provider: {self._provider_name}\nModel: {self._model_name}\nThread: {self._session_thread_id[:8] if self._session_thread_id else 'N/A'}...",
            classes="session-content",
            id="session-content",
        )
```

**Step 3: Remove unused widget initializations from __init__**

Remove:
```python
self.token_widget = TokenUsageWidget()
self.tool_list = ToolStatusList()
self.query_history = QueryHistory()
self._sql_section = None
self._results_section = None
self._schema_section = None
```

Keep:
```python
def __init__(self, id: str | None = None):
    super().__init__(id=id)
    self._provider_name: str = ""
    self._model_name: str = ""
    self._session_thread_id: str = ""
```

**Step 4: Remove unused methods**

Remove:
- `update_tokens()`
- `set_tables()`
- `add_query()`
- `set_sql()`
- `set_results()`

Keep:
- `update_session_info()`

**Step 5: Test**

```bash
uv run pytest tests/test_chat_app.py::TestContextPanel -v
```

Expected: Tests may fail (need updating)

**Step 6: Update tests**

Modify `tests/test_chat_app.py`:
- Remove tests for removed widgets (token_widget, tool_list, etc.)
- Keep tests for session info

**Step 7: Commit**

```bash
git add esdc/chat/app.py tests/test_chat_app.py
git commit -m "refactor: right panel is static, minimal session info only"
```

---

### Task 2: Add Collapsible SQL Block to Left Panel

**Files:**
- Modify: `esdc/chat/app.py:727-770` (SQLPanel class)
- Modify: `esdc/chat/app.py:ChatPanel.compose()` 

**Step 1: Convert SQLPanel to use Collapsible**

```python
class SQLPanel(Collapsible):
    """Collapsible SQL query display in chat panel."""

    DEFAULT_CSS = """
    SQLPanel {
        margin: 1 0;
        background: transparent;
        border: none;
    }

    SQLPanel .title {
        color: $accent;
        text-style: bold;
    }

    .sql-content {
        color: $text;
        padding: 1 2;
        background: $surface;
    }
    """

    def __init__(self):
        super().__init__(title="📝 SQL Query", collapsed=True)
        self.sql_content = ""

    def compose(self) -> ComposeResult:
        yield Static("", classes="sql-content")

    def on_mount(self) -> None:
        self._content_widget = self.query_one(".sql-content", Static)

    def set_sql(self, sql: str) -> None:
        self.sql_content = sql
        if self._content_widget:
            self._content_widget.update(f"```sql\n{sql}\n```")
        # Auto-expand when new SQL is shown
        self.collapsed = False
```

**Step 2: Mount SQLPanel in ChatPanel**

```python
class ChatPanel(ScrollableContainer):
    def compose(self) -> ComposeResult:
        yield self.messages_container
        yield self.thinking  # Already exists
        yield SQLPanel(id="sql-panel")  # New
        yield ResultsPanel(id="results-panel")  # Will create in Task 3
```

**Step 3: Update ChatPanel to store SQL panel reference**

```python
def __init__(self):
    super().__init__()
    self.messages_container = MessagesContainer()
    self.sql_panel = SQLPanel()
    self.results_panel = ResultsPanel()

def compose(self) -> ComposeResult:
    yield self.messages_container
    yield self.thinking
    yield self.sql_panel
    yield self.results_panel
```

**Step 4: Test**

```bash
uv run pytest tests/test_chat_app.py::TestSQLPanel -v
```

**Step 5: Commit**

```bash
git add esdc/chat/app.py
git commit -m "feat: SQL query shown in collapsible block on left panel"
```

---

### Task 3: Add Collapsible Results Block to Left Panel

**Files:**
- Modify: `esdc/chat/app.py` (create ResultsPanel class)
- Modify: `esdc/chat/app.py:ChatPanel` 

**Step 1: Create ResultsPanel as Collapsible**

```python
class ResultsPanel(Collapsible):
    """Collapsible query results display in chat panel."""

    DEFAULT_CSS = """
    ResultsPanel {
        margin: 1 0;
        background: transparent;
        border: none;
    }

    ResultsPanel .title {
        color: $accent;
        text-style: bold;
    }

    .results-content {
        color: $text;
        padding: 1 2;
        background: $surface;
    }
    """

    def __init__(self):
        super().__init__(title="📊 Query Results", collapsed=True)
        self.results_content = ""

    def compose(self) -> ComposeResult:
        yield Static("", classes="results-content")

    def on_mount(self) -> None:
        self._content_widget = self.query_one(".results-content", Static)

    def set_results(self, results: str) -> None:
        self.results_content = results
        if self._content_widget:
            self._content_widget.update(results)
        self.collapsed = False
```

**Step 2: Add to ChatPanel.compose()**

Already done in Task 2.

**Step 3: Test**

```bash
uv run pytest tests/test_chat_app.py -v -k results
```

**Step 4: Commit**

```bash
git add esdc/chat/app.py
git commit -m "feat: query results shown in collapsible block on left panel"
```

---

### Task 4: Wire Up SQL and Results to Chat Flow

**Files:**
- Modify: `esdc/chat/app.py` (ESDCChatApp._handle_query)

**Step 1: Update _handle_query to use new panels**

Find where `set_sql` and `set_results` are called (currently on context panel).

Change from:
```python
self._context_panel.set_sql(sql, schema_tips)
self._context_panel.set_results(sql, results[:500])
```

To:
```python
if self.chat_panel and self.chat_panel.sql_panel:
    self.chat_panel.sql_panel.set_sql(sql)

if self.chat_panel and self.chat_panel.results_panel:
    self.chat_panel.results_panel.set_results(results)
```

**Step 2: Remove old set_sql/set_results calls**

Search for all calls to removed methods and update.

**Step 3: Test full query flow**

```bash
uv run python esdc/chat/app.py
# Ask: "Show me all projects"
# Verify: SQL block appears and expands
# Verify: Results block appears and expands
# Verify: Right panel shows only Session Info
```

**Step 4: Commit**

```bash
git add esdc/chat/app.py
git commit -m "feat: wire up SQL and results panels to chat flow"
```

---

### Task 5: Clean Up and Final Verification

**Files:**
- Modify: `esdc/chat/app.py`
- Modify: `tests/test_chat_app.py`

**Step 1: Remove unused CSS**

Remove from ESDCChatApp.CSS:
- TokenUsageWidget styles
- ToolStatusList styles  
- Old SQLPanel styles (if any remain)
- Old ResultsPanel styles

**Step 2: Run full test suite**

```bash
uv run pytest tests/test_chat_app.py -v
```

Expected: All tests pass (may need to update/remove tests for removed features)

**Step 3: Run type check**

```bash
uv run mypy esdc/chat/app.py
```

Expected: 0 errors

**Step 4: Manual verification**

```bash
uv run python esdc/chat/app.py
```

Verify:
- [ ] Right panel is narrow, static, shows only Session Info
- [ ] Session Info shows Provider, Model, Thread (not frozen)
- [ ] Thinking indicator appears in chat during processing
- [ ] SQL query appears in collapsible block after query
- [ ] Results appear in collapsible block after query
- [ ] All collapsible blocks can be toggled

**Step 5: Commit**

```bash
git add esdc/chat/app.py tests/test_chat_app.py
git commit -m "chore: clean up unused CSS and update tests"
```

---

### Task 6: Update Documentation

**Files:**
- Create: `docs/ui-layout.md`

**Step 1: Document new layout**

```markdown
# ESDC UI Layout

## Left Panel (75% width)
- Chat messages (scrollable)
- Thinking indicator (collapsible, auto-expands during processing)
- SQL Query (collapsible, auto-expands when SQL generated)
- Query Results (collapsible, auto-expands when results ready)

## Right Panel (25% width)
- Session Info only (static, not scrollable)
  - Provider name
  - Model name
  - Thread ID
```

**Step 2: Commit**

```bash
git add docs/ui-layout.md
git commit -m "docs: document new UI layout"
```

---

**Plan complete.**

Two execution options:

**1. Subagent-Driven (this session)** - I dispatch fresh subagent per task, review between tasks, fast iteration

**2. Parallel Session (separate)** - Open new session with executing-plans, batch execution with checkpoints

**Which approach?**
