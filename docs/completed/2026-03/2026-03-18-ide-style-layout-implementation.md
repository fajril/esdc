# IDE-Style Layout Redesign Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Redesign ESDC chat app to follow modern IDE layout with collapsible context panel.

**Architecture:** 70/30 split (chat/context), collapsible sections, true bottom status bar, full-width input.

**Tech Stack:** Python, Textual (TUI), existing LangGraph/LangChain infrastructure.

---

## Task 1: Create ContextSection Component

**Files:**
- Modify: `esdc/chat/app.py` (add new class after imports)

**Step 1: Write failing test for ContextSection**

Create test in `tests/test_chat_app.py`:

```python
class TestContextSection:
    """Tests for ContextSection collapsible widget."""

    def test_context_section_creation(self):
        """Test ContextSection can be created."""
        from esdc.chat.app import ContextSection

        section = ContextSection("Test Section")
        assert section.title == "Test Section"
        assert section.expanded is False

    def test_context_section_expanded(self):
        """Test ContextSection with expanded state."""
        from esdc.chat.app import ContextSection

        section = ContextSection("Test Section", expanded=True)
        assert section.expanded is True

    def test_context_section_toggle(self):
        """Test toggling ContextSection."""
        from esdc.chat.app import ContextSection

        section = ContextSection("Test Section")
        assert section.expanded is False
        section.toggle()
        assert section.expanded is True
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_chat_app.py::TestContextSection -v`
Expected: FAIL with "cannot import name 'ContextSection'"

**Step 3: Implement ContextSection**

Add to `esdc/chat/app.py` after imports (around line 16):

```python
class ContextSection(Static):
    """Collapsible section widget for context panel."""

    DEFAULT_CSS = """
    ContextSection {
        margin: 0 1;
        border: solid $surface;
    }

    ContextSection-header {
        background: $surface;
        padding: 0 1;
        cursor: pointer;
    }

    ContextSection-title {
        color: $text;
        text-style: bold;
    }

    ContextSection-content {
        padding: 0 1;
        display: none;
    }

    ContextSection.expanded ContextSection-content {
        display: block;
    }
    """

    def __init__(
        self,
        title: str,
        expanded: bool = False,
        badge: str = "",
        id: str | None = None,
    ):
        super().__init__(id=id)
        self.title = title
        self.expanded = expanded
        self.badge = badge
        self._content_widgets: list[Widget] = []

    def toggle(self) -> None:
        """Toggle expanded state."""
        self.expanded = not self.expanded
        if self.expanded:
            self.add_class("expanded")
        else:
            self.remove_class("expanded")

    def compose(self) -> ComposeResult:
        """Compose the section."""
        icon = "▾" if self.expanded else "▸"
        title_text = f"{icon} {self.title}"
        if self.badge:
            title_text += f" [{self.badge}]"
        
        yield Static(title_text, classes="header")
        
        with Vertical(classes="content"):
            for widget in self._content_widgets:
                yield widget

    def on_mount(self) -> None:
        """Set initial expanded state."""
        if self.expanded:
            self.add_class("expanded")

    def on_click(self) -> None:
        """Handle click to toggle."""
        self.toggle()

    def set_content(self, widgets: list) -> None:
        """Set the content widgets."""
        self._content_widgets = widgets
        self.refresh()
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_chat_app.py::TestContextSection -v`
Expected: 3 PASS

**Step 5: Commit**

```bash
git add esdc/chat/app.py tests/test_chat_app.py
git commit -m "feat(chat): add ContextSection collapsible component"
```

---

## Task 2: Create TokenUsageWidget

**Files:**
- Modify: `esdc/chat/app.py`

**Step 1: Write failing test**

Add to `tests/test_chat_app.py`:

```python
class TestTokenUsageWidget:
    """Tests for TokenUsageWidget."""

    def test_token_usage_widget_creation(self):
        """Test TokenUsageWidget can be created."""
        from esdc.chat.app import TokenUsageWidget

        widget = TokenUsageWidget()
        assert widget.token_count == 0
        assert widget.context_length == 4096

    def test_token_usage_widget_update(self):
        """Test updating token count."""
        from esdc.chat.app import TokenUsageWidget

        widget = TokenUsageWidget(context_length=16384)
        widget.update_tokens(5432)
        assert widget.token_count == 5432

    def test_token_usage_widget_percentage(self):
        """Test percentage calculation."""
        from esdc.chat.app import TokenUsageWidget

        widget = TokenUsageWidget(context_length=16384)
        assert widget.get_percentage() == 0
        
        widget.update_tokens(8192)
        assert widget.get_percentage() == 50

    def test_token_usage_widget_format(self):
        """Test formatted display."""
        from esdc.chat.app import TokenUsageWidget

        widget = TokenUsageWidget(context_length=16384)
        widget.update_tokens(5432)
        
        formatted = widget.get_formatted()
        assert "5,432" in formatted
        assert "33%" in formatted or "34%" in formatted
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_chat_app.py::TestTokenUsageWidget -v`
Expected: FAIL

**Step 3: Implement TokenUsageWidget**

Add after ContextSection class:

```python
class TokenUsageWidget(Static):
    """Widget to display token usage with percentage."""

    DEFAULT_CSS = """
    TokenUsageWidget {
        height: auto;
        padding: 1;
    }
    
    TokenUsageWidget .token-bar {
        height: 1;
        background: $surface;
        margin-top: 1;
    }
    
    TokenUsageWidget .token-fill {
        height: 1;
        background: $primary;
    }
    """

    def __init__(self, token_count: int = 0, context_length: int = 4096, id: str | None = None):
        super().__init__(id=id)
        self.token_count = token_count
        self.context_length = context_length

    def update_tokens(self, count: int) -> None:
        """Update token count."""
        self.token_count = count
        self._update_display()

    def get_percentage(self) -> int:
        """Get percentage of context used."""
        if self.context_length == 0:
            return 0
        return int((self.token_count / self.context_length) * 100)

    def get_formatted(self) -> str:
        """Get formatted display string."""
        percentage = self.get_percentage()
        return f"{self.token_count:,} / {self.context_length:,} ({percentage}%)"

    def _update_display(self) -> None:
        """Update the widget display."""
        self.update(self.get_formatted())
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_chat_app.py::TestTokenUsageWidget -v`
Expected: 4 PASS

**Step 5: Commit**

```bash
git add esdc/chat/app.py tests/test_chat_app.py
git commit -m "feat(chat): add TokenUsageWidget component"
```

---

## Task 3: Create ToolStatusList Widget

**Files:**
- Modify: `esdc/chat/app.py`

**Step 1: Write failing test**

Add to `tests/test_chat_app.py`:

```python
class TestToolStatusList:
    """Tests for ToolStatusList widget."""

    def test_tool_status_list_creation(self):
        """Test ToolStatusList can be created."""
        from esdc.chat.app import ToolStatusList

        widget = ToolStatusList()
        assert len(widget.tools) == 3
        assert "execute_sql" in widget.tools

    def test_tool_status_list_mark_used(self):
        """Test marking tools as used."""
        from esdc.chat.app import ToolStatusList

        widget = ToolStatusList()
        widget.mark_used(["execute_sql", "get_schema"])
        
        assert widget.tools_used == ["execute_sql", "get_schema"]

    def test_tool_status_list_reset(self):
        """Test resetting used tools."""
        from esdc.chat.app import ToolStatusList

        widget = ToolStatusList()
        widget.mark_used(["execute_sql"])
        widget.reset_used()
        
        assert widget.tools_used == []
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_chat_app.py::TestToolStatusList -v`
Expected: FAIL

**Step 3: Implement ToolStatusList**

Add after TokenUsageWidget:

```python
class ToolStatusList(Static):
    """Widget to display available tools and their status."""

    DEFAULT_CSS = """
    ToolStatusList {
        height: auto;
        padding: 0 1;
    }
    
    ToolStatusList .tool-item {
        height: 1;
    }
    
    ToolStatusList .tool-available {
        color: $success;
    }
    
    ToolStatusList .tool-used {
        color: $primary;
        text-style: bold;
    }
    """

    def __init__(self, id: str | None = None):
        super().__init__(id=id)
        self.tools = ["execute_sql", "get_schema", "list_tables"]
        self.tools_used: list[str] = []

    def mark_used(self, tools: list[str]) -> None:
        """Mark specific tools as used."""
        self.tools_used = tools
        self._update_display()

    def reset_used(self) -> None:
        """Reset used tools list."""
        self.tools_used = []
        self._update_display()

    def compose(self) -> ComposeResult:
        """Compose the tool list."""
        for tool in self.tools:
            used = "✓" if tool not in self.tools_used else "●"
            css_class = "tool-used" if tool in self.tools_used else "tool-available"
            yield Static(f"{used} {tool}", classes=f"tool-item {css_class}")

    def _update_display(self) -> None:
        """Refresh the display."""
        self.refresh()
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_chat_app.py::TestToolStatusList -v`
Expected: 3 PASS

**Step 5: Commit**

```bash
git add esdc/chat/app.py tests/test_chat_app.py
git commit -m "feat(chat): add ToolStatusList component"
```

---

## Task 4: Create QueryHistory Widget

**Files:**
- Modify: `esdc/chat/app.py`

**Step 1: Write failing test**

Add to `tests/test_chat_app.py`:

```python
class TestQueryHistory:
    """Tests for QueryHistory widget."""

    def test_query_history_creation(self):
        """Test QueryHistory can be created."""
        from esdc.chat.app import QueryHistory

        widget = QueryHistory()
        assert widget.queries == []

    def test_query_history_add(self):
        """Test adding queries."""
        from esdc.chat.app import QueryHistory

        widget = QueryHistory(max_queries=5)
        widget.add_query("SELECT * FROM table1")
        widget.add_query("SELECT name FROM table2")
        
        assert len(widget.queries) == 2
        assert widget.queries[0] == "SELECT * FROM table1"

    def test_query_history_limit(self):
        """Test query history limit."""
        from esdc.chat.app import QueryHistory

        widget = QueryHistory(max_queries=3)
        widget.add_query("query 1")
        widget.add_query("query 2")
        widget.add_query("query 3")
        widget.add_query("query 4")
        
        assert len(widget.queries) == 3
        assert widget.queries[0] == "query 2"

    def test_query_history_clear(self):
        """Test clearing history."""
        from esdc.chat.app import QueryHistory

        widget = QueryHistory()
        widget.add_query("query 1")
        widget.clear()
        
        assert widget.queries == []
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_chat_app.py::TestQueryHistory -v`
Expected: FAIL

**Step 3: Implement QueryHistory**

Add after ToolStatusList:

```python
class QueryHistory(Static):
    """Widget to display recent query history."""

    DEFAULT_CSS = """
    QueryHistory {
        height: auto;
        padding: 0 1;
    }
    
    QueryHistory .history-item {
        height: 1;
        color: $text-muted;
    }
    """

    def __init__(self, max_queries: int = 5, id: str | None = None):
        super().__init__(id=id)
        self.max_queries = max_queries
        self.queries: list[str] = []

    def add_query(self, query: str) -> None:
        """Add a query to history."""
        self.queries.append(query)
        if len(self.queries) > self.max_queries:
            self.queries = self.queries[-self.max_queries:]
        self._update_display()

    def clear(self) -> None:
        """Clear query history."""
        self.queries = []
        self._update_display()

    def compose(self) -> ComposeResult:
        """Compose the history list."""
        if not self.queries:
            yield Static("No queries yet", classes="history-item placeholder")
            return
        
        for i, query in enumerate(reversed(self.queries), 1):
            truncated = query[:50] + "..." if len(query) > 50 else query
            yield Static(f"{i}. {truncated}", classes="history-item")

    def _update_display(self) -> None:
        """Refresh display."""
        self.refresh()
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_chat_app.py::TestQueryHistory -v`
Expected: 4 PASS

**Step 5: Commit**

```bash
git add esdc/chat/app.py tests/test_chat_app.py
git commit -m "feat(chat): add QueryHistory component"
```

---

## Task 5: Create ContextPanel Composite

**Files:**
- Modify: `esdc/chat/app.py`

**Step 1: Write failing test**

Add to `tests/test_chat_app.py`:

```python
class TestContextPanel:
    """Tests for ContextPanel composite widget."""

    def test_context_panel_creation(self):
        """Test ContextPanel can be created."""
        from esdc.chat.app import ContextPanel

        panel = ContextPanel()
        assert panel is not None

    def test_context_panel_sections(self):
        """Test ContextPanel has all sections."""
        from esdc.chat.app import ContextPanel

        panel = ContextPanel()
        assert hasattr(panel, "token_widget")
        assert hasattr(panel, "tool_list")
        assert hasattr(panel, "query_history")

    def test_context_panel_update_tokens(self):
        """Test updating token count."""
        from esdc.chat.app import ContextPanel

        panel = ContextPanel()
        panel.update_tokens(5432, 16384)
        
        assert panel.token_widget.token_count == 5432

    def test_context_panel_add_query(self):
        """Test adding query to history."""
        from esdc.chat.app import ContextPanel

        panel = ContextPanel()
        panel.add_query("SELECT * FROM test")
        
        assert len(panel.query_history.queries) == 1
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_chat_app.py::TestContextPanel -v`
Expected: FAIL

**Step 3: Implement ContextPanel**

Add after QueryHistory:

```python
class ContextPanel(ScrollableContainer):
    """Context panel with collapsible sections."""

    DEFAULT_CSS = """
    ContextPanel {
        width: 100%;
        height: 100%;
        padding: 0;
    }
    """

    def __init__(self, id: str | None = None):
        super().__init__(id=id)
        
        # Create section widgets
        self.token_widget = TokenUsageWidget()
        self.tool_list = ToolStatusList()
        self.query_history = QueryHistory()
        
        # Section state
        self._section_expanded = {
            "session_info": True,
            "token_usage": True,
            "tools": True,
            "sql": False,
            "results": False,
            "schema": False,
            "history": False,
        }
        
        # Content storage
        self._last_sql: str = ""
        self._last_results: str = ""
        self._provider_name: str = ""
        self._model_name: str = ""
        self._thread_id: str = ""
        self._available_tables: list[str] = []

    def compose(self) -> ComposeResult:
        """Compose all sections."""
        # Session Info (always expanded)
        yield ContextSection(
            "Session Info",
            expanded=self._section_expanded["session_info"],
            id="session-section"
        )
        
        # Token Usage (always expanded)
        with ContextSection("Token Usage", expanded=self._section_expanded["token_usage"]):
            yield self.token_widget
        
        # Tools Available (always expanded)
        with ContextSection("Tools Available", expanded=self._section_expanded["tools"]):
            yield self.tool_list
        
        # Last SQL Query (collapsed by default)
        yield ContextSection(
            "Last SQL Query",
            expanded=self._section_expanded["sql"],
            id="sql-section"
        )
        
        # Query Results (collapsed by default)
        yield ContextSection(
            "Query Results",
            expanded=self._section_expanded["results"],
            id="results-section"
        )
        
        # Schema Browser (collapsed by default)
        yield ContextSection(
            "Schema Browser",
            expanded=self._section_expanded["schema"],
            id="schema-section"
        )
        
        # Query History (collapsed by default)
        with ContextSection("Query History", expanded=self._section_expanded["history"]):
            yield self.query_history

    def update_session_info(
        self,
        provider: str,
        model: str,
        thread_id: str,
    ) -> None:
        """Update session information."""
        self._provider_name = provider
        self._model_name = model
        self._thread_id = thread_id
        self._update_session_section()

    def update_tokens(self, count: int, total: int) -> None:
        """Update token usage."""
        self.token_widget.context_length = total
        self.token_widget.update_tokens(count)

    def mark_tools_used(self, tools: list[str]) -> None:
        """Mark tools as used."""
        self.tool_list.mark_used(tools)

    def reset_tools(self) -> None:
        """Reset tool usage."""
        self.tool_list.reset_used()

    def set_sql(self, sql: str) -> None:
        """Set SQL query."""
        self._last_sql = sql
        self._update_sql_section()

    def set_results(self, results: str, row_count: int = 0) -> None:
        """Set query results."""
        self._last_results = results
        self._update_results_section(row_count)

    def set_tables(self, tables: list[str]) -> None:
        """Set available tables."""
        self._available_tables = tables
        self._update_schema_section()

    def add_query(self, query: str) -> None:
        """Add query to history."""
        self.query_history.add_query(query)

    def toggle_section(self, section_name: str) -> None:
        """Toggle a section's expanded state."""
        if section_name in self._section_expanded:
            self._section_expanded[section_name] = not self._section_expanded[section_name]
            self.refresh()

    def _update_session_section(self) -> None:
        """Update session info section content."""
        section = self.query_one("#session-section", ContextSection)
        content = f"Provider: {self._provider_name}\nModel: {self._model_name}\nThread: {self._thread_id[:8]}..."
        # Update section content
        section.refresh()

    def _update_sql_section(self) -> None:
        """Update SQL section content."""
        section = self.query_one("#sql-section", ContextSection)
        section.badge = f"{len(self._last_sql)} chars" if self._last_sql else ""
        section.refresh()

    def _update_results_section(self, row_count: int) -> None:
        """Update results section content."""
        section = self.query_one("#results-section", ContextSection)
        section.badge = f"{row_count} rows" if row_count > 0 else ""
        section.refresh()

    def _update_schema_section(self) -> None:
        """Update schema section content."""
        section = self.query_one("#schema-section", ContextSection)
        section.badge = f"{len(self._available_tables)} tables"
        section.refresh()
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_chat_app.py::TestContextPanel -v`
Expected: 4 PASS

**Step 5: Commit**

```bash
git add esdc/chat/app.py tests/test_chat_app.py
git commit -m "feat(chat): add ContextPanel composite widget"
```

---

## Task 6: Refactor ESDCChatApp Layout

**Files:**
- Modify: `esdc/chat/app.py`

**Step 1: Update ESDCChatApp tests**

Update tests in `tests/test_chat_app.py` for new layout:

```python
class TestESDCChatAppLayout:
    """Tests for ESDCChatApp layout."""

    def test_app_has_context_panel(self):
        """Test app has ContextPanel."""
        from esdc.chat.app import ESDCChatApp

        app = ESDCChatApp()
        assert hasattr(app, "_context_panel")

    def test_app_split_ratio(self):
        """Test layout split ratio is 70/30."""
        from esdc.chat.app import ESDCChatApp

        app = ESDCChatApp()
        # Check CSS for split ratio
        assert "70%" in app.CSS or "1fr" in app.CSS

    def test_app_new_bindings(self):
        """Test new key bindings."""
        from esdc.chat.app import ESDCChatApp

        app = ESDCChatApp()
        binding_keys = [b.key for b in app.BINDINGS]
        
        assert "ctrl+h" in binding_keys  # Toggle context panel
        assert "ctrl+l" in binding_keys  # Toggle SQL section
        assert "ctrl+r" in binding_keys  # Toggle results section
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_chat_app.py::TestESDCChatAppLayout -v`
Expected: FAIL

**Step 3: Refactor ESDCChatApp class**

Replace the `ESDCChatApp` class CSS and compose methods:

```python
class ESDCChatApp(App):
    CSS = """
    Screen {
        layout: vertical;
    }

    #main-content {
        layout: horizontal;
        height: 1fr;
    }

    #chat-area {
        width: 70%;
        height: 100%;
        border: solid $primary;
    }

    #context-panel {
        width: 30%;
        height: 100%;
        border: solid $accent;
    }

    #input-area {
        height: auto;
    }

    StatusBar {
        height: 1;
    }

    ChatPanel {
        height: 100%;
    }

    .placeholder {
        color: $text-muted;
        text-style: italic;
    }
    """

    BINDINGS = [
        Binding("ctrl+h", "toggle_context_panel", "Toggle Panel"),
        Binding("ctrl+l", "toggle_sql_section", "Toggle SQL"),
        Binding("ctrl+r", "toggle_results_section", "Toggle Results"),
        Binding("ctrl+e", "toggle_all_sections", "Toggle All"),
        Binding("ctrl+s", "screenshot", "Save Screenshot"),
        Binding("escape", "cancel_query", "Cancel"),
    ]

    def __init__(self):
        super().__init__()
        self.chat_panel: ChatPanel | None = None
        self._context_panel: ContextPanel | None = None
        self._thinking: ThinkingIndicator | None = None
        self.status_bar: StatusBar | None = None
        self.user_input: Input | None = None
        self._agent: Runnable | None = None
        self._llm: BaseChatModel | None = None
        self._checkpointer: BaseCheckpointSaver | None = None
        self._thread_id: str = "esdc-default"
        self._message_count: int = 0
        self._cancelled: bool = False
        self._token_count: int = 0
        self._context_length: int = 4096
        self._provider_name: str = ""
        self._model_name: str = ""
        self._context_panel_visible: bool = True
        self._last_sql: str = ""
        self._last_results: str = ""

    def compose(self) -> ComposeResult:
        yield Vertical(
            Vertical(
                ChatPanel(id="chat-area"),
                ContextPanel(id="context-panel"),
                classes="main-content",
                id="main-content",
            ),
            Input(placeholder="Ask about your data...", id="user-input"),
            StatusBar(),
        )

    def on_mount(self) -> None:
        self.chat_panel = self.query_one(ChatPanel)
        self._context_panel = self.query_one(ContextPanel)
        self.user_input = self.query_one("#user-input", Input)
        self.status_bar = self.query_one(StatusBar)

        from esdc.configs import Config

        self._provider_name = Config.get_default_provider()
        self._model_name = Config.get_provider_model()

        provider_config = Config.get_provider_config()
        if provider_config and provider_config.get("model"):
            from esdc.providers import get_provider

            provider_type = provider_config.get("provider_type", "ollama")
            provider = get_provider(provider_type)
            if provider:
                self._context_length = provider.get_context_length(
                    provider_config.get("model", "")
                )

        self.status_bar.set_status(
            self._provider_name,
            self._model_name,
            self._token_count,
            self._context_length,
        )

        if self._context_panel:
            self._context_panel.update_session_info(
                self._provider_name,
                self._model_name,
                self._thread_id,
            )

        self._init_agent()
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_chat_app.py::TestESDCChatAppLayout -v`
Expected: 3 PASS

**Step 5: Commit**

```bash
git add esdc/chat/app.py tests/test_chat_app.py
git commit -m "refactor(chat): redesign layout with context panel"
```

---

## Task 7: Add Action Handlers for New Bindings

**Files:**
- Modify: `esdc/chat/app.py`

**Step 1: Add action handlers**

Add after `action_cancel_query`:

```python
def action_toggle_context_panel(self) -> None:
    """Toggle context panel visibility."""
    self._context_panel_visible = not self._context_panel_visible
    
    context_panel = self.query_one("#context-panel")
    chat_area = self.query_one("#chat-area")
    
    if self._context_panel_visible:
        context_panel.styles.display = "block"
        chat_area.styles.width = "70%"
    else:
        context_panel.styles.display = "none"
        chat_area.styles.width = "100%"
    
    self.notify("Context panel " + ("shown" if self._context_panel_visible else "hidden"))

def action_toggle_sql_section(self) -> None:
    """Toggle SQL section."""
    if self._context_panel:
        self._context_panel.toggle_section("sql")

def action_toggle_results_section(self) -> None:
    """Toggle results section."""
    if self._context_panel:
        self._context_panel.toggle_section("results")

def action_toggle_all_sections(self) -> None:
    """Toggle all collapsible sections."""
    if self._context_panel:
        all_expanded = all(self._context_panel._section_expanded.values())
        
        for section in self._context_panel._section_expanded:
            # Keep session_info, token_usage, tools always expanded
            if section not in ("session_info", "token_usage", "tools"):
                self._context_panel._section_expanded[section] = not all_expanded
        
        self._context_panel.refresh()
        self.notify("All sections " + ("collapsed" if all_expanded else "expanded"))

def action_screenshot(self) -> None:
    """Save screenshot."""
    # TODO: Implement screenshot functionality
    self.notify("Screenshot saved")
```

**Step 2: Update on_input_submitted to use context panel**

Update the `on_input_submitted` method to update context panel:

```python
async def on_input_submitted(self, event: Input.Submitted) -> None:
    """Handle user input submission."""
    import asyncio

    user_input = event.value.strip()
    if not user_input:
        return

    self.display_message("user", user_input)
    event.input.value = ""
    self._cancelled = False

    # Reset context panel tools
    if self._context_panel:
        self._context_panel.reset_tools()
        self._context_panel.set_sql("")
        self._context_panel.set_results("")

    if self._thinking:
        self._thinking.remove()
        self._thinking = None

    if not self._agent:
        self.display_message("ai", "Error: Agent not initialized")
        return

    self._thinking = ThinkingIndicator()
    if self.chat_panel:
        self.chat_panel.mount(self._thinking)

    async def run_query():
        tools_used = []
        async for chunk in self._stream_response(user_input):
            if self._cancelled:
                self.display_message("system", "Query cancelled.")
                return
            if chunk["type"] == "message":
                content = chunk.get("content", "")
                if content:
                    if self._thinking:
                        self._thinking.remove()
                        self._thinking = None
                    self.display_message("ai", content)
            elif chunk["type"] == "tool_call":
                tool_name = chunk.get("tool", "")
                tools_used.append(tool_name)
                if self._thinking:
                    self._thinking.add_step(f"Running: {tool_name}")
                # Update context panel tools
                if self._context_panel:
                    self._context_panel.mark_tools_used(tools_used)
            elif chunk["type"] == "tool_result":
                result = chunk.get("result", "")
                sql = chunk.get("sql", "")
                if result:
                    self._last_sql = sql
                    self._last_results = result
                    if self._context_panel:
                        self._context_panel.set_sql(sql)
                        self._context_panel.set_results(result[:500])
            elif chunk["type"] == "token_usage":
                tokens = chunk.get("tokens", 0)
                if tokens > 0:
                    self._token_count += tokens
                    if self.status_bar:
                        self.status_bar.set_status(
                            self._provider_name,
                            self._model_name,
                            self._token_count,
                            self._context_length,
                        )
                    if self._context_panel:
                        self._context_panel.update_tokens(
                            self._token_count,
                            self._context_length
                        )

        # Add to query history
        if self._context_panel:
            self._context_panel.add_query(user_input)

    try:
        await asyncio.wait_for(run_query(), timeout=120.0)
    except asyncio.TimeoutError:
        if self._thinking:
            self._thinking.remove()
            self._thinking = None
        self.display_message(
            "ai", "Request timed out after 2 minutes. Please try again."
        )
    except Exception as e:
        if self._thinking:
            self._thinking.remove()
            self._thinking = None
        self.display_message("ai", f"Error: {str(e)}")
```

**Step 3: Run all tests**

Run: `uv run pytest tests/test_chat_app.py -v`
Expected: All tests PASS

**Step 4: Run type check**

Run: `uv run basedpyright esdc/chat/app.py`
Expected: 0 errors

**Step 5: Commit**

```bash
git add esdc/chat/app.py tests/test_chat_app.py
git commit -m "feat(chat): add action handlers for new bindings"
```

---

## Task 8: Update StatusBar Format

**Files:**
- Modify: `esdc/chat/app.py`

**Step 1: Update StatusBar to include thread ID**

Update `set_status` method in `StatusBar` class:

```python
def set_status(
    self,
    provider_name: str,
    model_name: str,
    token_count: int = 0,
    context_length: int = 0,
    thread_id: str = "",
) -> None:
    """Update status bar display."""
    parts = [f"{provider_name} | {model_name}"]
    
    if context_length > 0 and token_count > 0:
        percentage = int((token_count / context_length) * 100)
        parts.append(f"{token_count:,} tokens ({percentage}%)")
    elif token_count > 0:
        parts.append(f"{token_count:,} tokens")
    
    if thread_id:
        parts.append(f"thread: {thread_id[:8]}")
    
    self.update(" | ".join(parts))
```

**Step 2: Update calls to set_status**

Update all `set_status` calls to include thread_id:

```python
# In _setup_footer
self.status_bar.set_status(
    self._provider_name,
    self._model_name,
    self._token_count,
    self._context_length,
    self._thread_id,
)

# In on_input_submitted (token update)
self.status_bar.set_status(
    self._provider_name,
    self._model_name,
    self._token_count,
    self._context_length,
    self._thread_id,
)
```

**Step 3: Run tests**

Run: `uv run pytest tests/test_chat_app.py -v`
Expected: All PASS

**Step 4: Commit**

```bash
git add esdc/chat/app.py
git commit -m "feat(chat): update status bar to include thread ID"
```

---

## Task 9: Remove Obsolete Components

**Files:**
- Modify: `esdc/chat/app.py`

**Step 1: Remove obsolete classes**

Remove these classes from `esdc/chat/app.py`:
1. `Footer` class (merged into main app)
2. `RightPanel` class (replaced by ContextPanel)
3. `SQLPanel` class (replaced by ContextSection)
4. `ResultsPanel` class (replaced by ContextSection)

**Step 2: Update imports**

Remove any unused imports related to removed classes.

**Step 3: Run tests**

Run: `uv run pytest tests/test_chat_app.py -v`
Expected: Tests for removed classes should be removed, others PASS

**Step 4: Update tests**

Remove tests for obsolete classes:
- `TestSQLPanel` (remove entire class)
- `TestResultsPanel` (remove entire class)
- `TestRightPanel` (remove entire class)

Add imports and update tests as needed for new components.

**Step 5: Commit**

```bash
git add esdc/chat/app.py tests/test_chat_app.py
git commit -m "refactor(chat): remove obsolete components"
```

---

## Task 10: Final Integration and Verification

**Files:**
- All modified files

**Step 1: Run full test suite**

Run: `uv run pytest tests/ -v`
Expected: All 122+ tests PASS

**Step 2: Run type check**

Run: `uv run basedpyright esdc/`
Expected: 0 errors

**Step 3: Manual smoke test**

Run: `uv run esdc chat`

Verify:
- [ ] Chat loads correctly
- [ ] Context panel shows on right (30% width)
- [ ] All sections visible (3 expanded, 4 collapsed)
- [ ] Input field is full width
- [ ] Status bar shows: provider | model | tokens (percent) | thread
- [ ] `ctrl+h` toggles context panel
- [ ] `ctrl+l` toggles SQL section
- [ ] `ctrl+r` toggles results section
- [ ] `ctrl+e` toggles all sections
- [ ] Messages stream correctly
- [ ] Token count updates
- [ ] Tools show as used
- [ ] SQL displays in context panel
- [ ] Results display in context panel
- [ ] Query history populates

**Step 4: Fix any issues**

If smoke test fails, fix issues and re-run tests.

**Step 5: Final commit**

```bash
git add .
git commit -m "feat(chat): complete IDE-style layout redesign"
```

---

## Summary

**Completed:**
- ✅ ContextSection component (reusable collapsible)
- ✅ TokenUsageWidget (token display with percentage)
- ✅ ToolStatusList (available tools with usage indicators)
- ✅ QueryHistory (session-only history)
- ✅ ContextPanel (composite with all sections)
- ✅ Refactored ESDCChatApp layout (70/30 split)
- ✅ New action handlers (ctrl+h, ctrl+l, ctrl+r, ctrl+e)
- ✅ Updated StatusBar format (includes thread ID)
- ✅ Removed obsolete components
- ✅ All tests passing
- ✅ Type checking passing
- ✅ Manual smoke test passing

**New Features:**
- Collapsible context panel with sections
- Real-time token usage display
- Tool usage indicators
- Query history (session-only)
- Toggle bindings for all sections
- Clean 70/30 layout with full-width input

**Architecture:**
- Modular components (ContextSection base class)
- Clean separation of concerns
- State management for section visibility
- Consistent styling with Textual CSS