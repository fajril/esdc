# ESDC Chat TUI Enhancement - Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Transform ESDC Chat TUI with Markdown support, improved layout (adjustable 60/40 split), colored message blocks, and split right panel (SQL+schema top, results bottom).

**Architecture:** Refactor app.py to use Textual's Markdown widget, create new panel components, add CSS styling for colored blocks and resizable layout.

**Tech Stack:** Textual (Markdown, ScrollView, Dock), Rich (tables), Python 3.10+ type hints

---

## Task 1: Setup and Basic Markdown Integration

**Files:**
- Modify: `esdc/chat/app.py:1-10` (imports)

**Step 1: Update imports in app.py**

Add Markdown widget import:
```python
from textual.widgets import Static, Input, Markdown
```

Run: `uv run basedpyright esdc/chat/app.py`
Expected: PASS (0 errors)

---

## Task 2: Refactor ChatMessage to Markdown

**Files:**
- Modify: `esdc/chat/app.py:13-16` (ChatMessage class)

**Step 1: Write failing test**

Create `tests/test_chat_app.py`:
```python
def test_chat_message_markdown():
    msg = ChatMessage("user", "Hello **world**")
    assert msg.role == "user"

def test_chat_message_role_colors():
    msg_ai = ChatMessage("ai", "Response")
    msg_user = ChatMessage("user", "Query")
    msg_system = ChatMessage("system", "Tool call")
    assert msg_ai.role == "ai"
    assert msg_user.role == "user"
    assert msg_system.role == "system"
```

Run: `pytest tests/test_chat_app.py -v`
Expected: FAIL (ChatMessage not defined in test file)

**Step 2: Refactor ChatMessage class**

Replace lines 13-16:
```python
class ChatMessage(Markdown):
    """A Markdown-formatted chat message with role-based styling."""

    DEFAULT_CSS = """
    ChatMessage {
        padding: 1;
        margin: 1 0;
    }
    ChatMessage.user {
        background: $primary 20%;
        border: solid $primary;
    }
    ChatMessage.ai {
        background: $success 20%;
        border: solid $success;
    }
    ChatMessage.system {
        background: $warning 20%;
        border: solid $warning;
    }
    """

    def __init__(self, role: str, content: str):
        formatted = f"**{role.upper()}:**\n\n{content}"
        super().__init__(formatted)
        self.role = role
        self.add_class(role)
```

Run: `pytest tests/test_chat_app.py -v`
Expected: PASS

**Step 3: Commit**
```bash
git add esdc/chat/app.py tests/test_chat_app.py
git commit -m "feat: refactor ChatMessage to use Markdown with role-based styling"
```

---

## Task 3: Create SQLPanel Component

**Files:**
- Modify: `esdc/chat/app.py:30-50` (add SQLPanel class)

**Step 1: Add SQLPanel class after ChatPanel**

```python
class SQLPanel(Vertical):
    """Panel showing generated SQL and schema context."""

    DEFAULT_CSS = """
    SQLPanel {
        height: 50%;
        border: solid $accent;
    }
    """

    def __init__(self):
        super().__init__()
        self.sql_content = ""
        self.schema_tips = ""

    def set_sql(self, sql: str, schema_tips: str = ""):
        self.sql_content = sql
        self.schema_tips = schema_tips
        self.update_display()

    def update_display(self):
        content = f"```sql\n{self.sql_content}\n```"
        if self.schema_tips:
            content += f"\n\n**Schema:**\n{self.schema_tips}"
        self.remove_children()
        self.mount(Markdown(content))
```

Run: `pytest tests/test_chat_app.py -v`
Expected: PASS

**Step 4: Commit**
```bash
git add esdc/chat/app.py
git commit -m "feat: add SQLPanel component for generated SQL display"
```

---

## Task 4: Create ResultsPanel with Markdown

**Files:**
- Modify: `esdc/chat/app.py:50-70` (ResultsPanel)

**Step 1: Refactor ResultsPanel to use Markdown and ScrollView**

Replace ResultsPanel class:
```python
class ResultsPanel(Vertical):
    """Panel showing query results in formatted table."""

    DEFAULT_CSS = """
    ResultsPanel {
        height: 50%;
        border: solid $primary;
    }
    """

    def __init__(self):
        super().__init__()
        self.results_content = ""

    def set_results(self, results: str):
        self.results_content = results
        self.update_display()

    def update_display(self):
        content = f"**Query Results:**\n\n{self.results_content}"
        self.remove_children()
        self.mount(Markdown(content))
```

Run: `pytest tests/test_chat_app.py -v`
Expected: PASS

**Step 2: Commit**
```bash
git add esdc/chat/app.py
git commit -m "feat: refactor ResultsPanel to use Markdown"
```

---

## Task 5: Create Right Panel Container (Split Layout)

**Files:**
- Modify: `esdc/chat/app.py:70-100` (RightPanel container)

**Step 1: Add RightPanel container class**

```python
class RightPanel(Vertical):
    """Container for SQL and Results panels."""

    DEFAULT_CSS = """
    RightPanel {
        width: 40%;
    }
    SQLPanel {
        height: 50%;
    }
    ResultsPanel {
        height: 50%;
    }
    """

    def __init__(self):
        super().__init__()
        self.sql_panel = SQLPanel()
        self.results_panel = ResultsPanel()

    def compose(self):
        yield self.sql_panel
        yield self.results_panel

    def set_sql(self, sql: str, schema_tips: str = ""):
        self.sql_panel.set_sql(sql, schema_tips)

    def set_results(self, results: str):
        self.results_panel.set_results(results)
```

Run: `pytest tests/test_chat_app.py -v`
Expected: PASS

**Step 2: Commit**
```bash
git add esdc/chat/app.py
git commit -m "feat: add RightPanel container with split SQL/Results layout"
```

---

## Task 6: Update Main App Layout (60/40 Split)

**Files:**
- Modify: `esdc/chat/app.py:100-160` (ESDCChatApp)

**Step 1: Update CSS for 60/40 split with resizable panels**

Replace CSS:
```python
CSS = """
Screen {
    layout: horizontal;
}

#main {
    width: 100%;
    height: 100%;
}

#left-panel {
    width: 60%;
    border: solid $success;
}

#right-panel {
    width: 40%;
    border: solid $accent;
}

ChatPanel {
    height: 100%;
    overflow-y: auto;
}

Input {
    dock: bottom;
    height: 3;
}
"""
```

**Step 2: Update compose method**

Replace compose to use RightPanel:
```python
def compose(self):
    yield Horizontal(
        Vertical(id="left-panel"),
        Vertical(id="right-panel"),
    )
```

**Step 3: Update on_mount to setup panels**

```python
def on_mount(self) -> None:
    left_container = self.query_one("#left-panel", Vertical)
    right_container = self.query_one("#right-panel", Vertical)

    self.chat_panel = ChatPanel()
    right_panel = RightPanel()

    left_container.mount(self.chat_panel)
    right_container.mount(right_panel)

    self.user_input = Input(placeholder="Ask about your data...", id="user_input")
    self.chat_panel.mount(self.user_input)

    self._right_panel = right_panel
    self._init_agent()
```

Run: `pytest tests/test_chat_app.py -v`
Expected: PASS

**Step 4: Commit**
```bash
git add esdc/chat/app.py
git commit -m "feat: update app layout to 60/40 split with RightPanel"
```

---

## Task 7: Update Streaming Logic for New Panels

**Files:**
- Modify: `esdc/chat/app.py:160-200` (streaming and display methods)

**Step 1: Update set_results method**

```python
def set_results(self, sql: str, results: str):
    if hasattr(self, '_right_panel'):
        self._right_panel.set_sql(sql, "")
        self._right_panel.set_results(results[:500])
```

**Step 2: Update on_input_submitted to clear previous results**

```python
async def on_input_submitted(self, event: Input.Submitted) -> None:
    # Clear right panel on new query
    if hasattr(self, '_right_panel'):
        self._right_panel.set_sql("", "")
        self._right_panel.set_results("Waiting for results...")
    # ... rest of method
```

Run: `pytest tests/test_chat_app.py -v`
Expected: PASS

**Step 3: Commit**
```bash
git add esdc/chat/app.py
git commit -m "feat: integrate new panels with streaming logic"
```

---

## Task 8: Add Keyboard Navigation (Optional Enhancement)

**Files:**
- Modify: `esdc/chat/app.py` (add key bindings)

**Step 1: Add BINDINGS to ESDCChatApp**

```python
BINDINGS = [
    Binding("ctrl+b", "toggle_sidebar", "Toggle Sidebar"),
    Binding("ctrl+c", "copy_results", "Copy Results"),
    Binding("ctrl+r", "focus_results", "Focus Results"),
]

def action_toggle_sidebar(self) -> None:
    """Toggle between 60/40 and 50/50 split."""
    # Toggle logic

def action_copy_results(self) -> None:
    """Copy current results to clipboard."""
    if hasattr(self, '_right_panel'):
        # Copy logic

def action_focus_results(self) -> None:
    """Focus the results panel."""
    self.query_one("#right-panel", Vertical).focus()
```

Run: `pytest tests/test_chat_app.py -v`
Expected: PASS

**Step 2: Commit**
```bash
git add esdc/chat/app.py
git commit -m "feat: add keyboard navigation for panel control"
```

---

## Task 9: Integration Testing

**Files:**
- Test: `tests/test_chat_app.py` (comprehensive tests)

**Step 1: Run all tests**

```bash
uv run pytest tests/ -v
```

Expected: All tests pass (113+ tests)

**Step 2: Run type checking**

```bash
uv run basedpyright esdc/chat/app.py
```

Expected: 0 errors

---

## Summary

| Task | Description | Files Modified |
|------|-------------|----------------|
| 1 | Import Markdown | app.py |
| 2 | ChatMessage with Markdown + styling | app.py, test_chat_app.py |
| 3 | SQLPanel component | app.py |
| 4 | ResultsPanel with Markdown | app.py |
| 5 | RightPanel container | app.py |
| 6 | 60/40 split layout | app.py |
| 7 | Streaming integration | app.py |
| 8 | Keyboard navigation | app.py |
| 9 | Integration testing | tests/ |

---

## Plan complete and saved to `docs/plans/2026-03-17-chat-tui-enhancement.md`. 

Two execution options:

**1. Subagent-Driven (this session)** - I dispatch fresh subagent per task, review between tasks, fast iteration

**2. Parallel Session (separate)** - Open new session with executing-plans, batch execution with checkpoints

Which approach?
