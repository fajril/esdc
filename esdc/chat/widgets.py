"""Textual widgets for the ESDC chat TUI."""

# Standard library
import logging
import os

from textual import events
from textual.app import ComposeResult
from textual.containers import Container, ScrollableContainer, Vertical
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Collapsible, Markdown, Static, TextArea

logger = logging.getLogger("esdc.chat.widgets")

MAX_MESSAGE_HISTORY = 100
DEFAULT_CONTEXT_LENGTH = 4096
IRIS_VERSION = "0.6.0"


class ContextSection(Container):
    """Collapsible section widget for context panel."""

    DEFAULT_CSS = """
    ContextSection {
        margin: 0 0 1 0;
        border: none;
    }

    ContextSection .header {
        background: transparent;
        padding: 1 1;
        text-style: bold;
        color: $text;
    }

    ContextSection .header:hover {
        color: $accent;
    }

    ContextSection .content {
        padding: 0 1 1 1;
        background: transparent;
        border: none;
    }
    """

    def __init__(
        self,
        title: str,
        expanded: bool = True,
        badge: str = "",
        id: str | None = None,
    ):
        """Initialize a collapsible context section.

        Args:
            title: Display title for the section.
            expanded: Whether section starts expanded (default True).
            badge: Optional badge text displayed next to title.
            id: Widget ID for CSS targeting.
        """
        super().__init__(id=id)
        self.section_title = title
        self.expanded = expanded
        self.badge = badge
        self._header: Static | None = None
        self._content_children: list[Widget] = []

    def compose_add_child(self, widget: "Widget") -> None:
        """Capture children from 'with' block to render after header."""
        self._content_children.append(widget)

    def compose(self) -> ComposeResult:
        """Header first, then content container with children."""
        icon = "▼" if self.expanded else "▶"
        title_text = f"{icon} {self.section_title}"
        if self.badge:
            title_text += f" [{self.badge}]"

        self._header = Static(title_text, classes="header")
        yield self._header

        with Vertical(classes="content") as content:
            if not self.expanded:
                content.display = False
            yield from self._content_children

        # Clear after yielding to prevent accumulation on recompose
        self._content_children = []

    def on_click(self) -> None:
        """Handle click to toggle."""
        self.toggle()

    def toggle(self) -> None:
        """Toggle expanded state and update header."""
        self.expanded = not self.expanded

        # Find content container and toggle display
        for child in self.children:
            if "content" in child.classes:
                child.display = self.expanded
                break

        if self._header:
            icon = "▼" if self.expanded else "▶"
            title_text = f"{icon} {self.section_title}"
            if self.badge:
                title_text += f" [{self.badge}]"
            self._header.update(title_text)


class ContextUsageWidget(Static):
    """Display context usage with compaction status."""

    DEFAULT_CSS = """
    ContextUsageWidget {
        height: auto;
        padding: 1;
        background: transparent;
        border: none;
        margin: 1 0;
    }

    .context-display {
        color: $text;
        text-style: bold;
    }

    .context-muted {
        color: $text-muted;
    }

    .context-warning {
        color: $warning;
    }

    .context-danger {
        color: $error;
    }

    .context-compacted {
        color: $primary;
        text-style: italic;
    }
    """

    def __init__(
        self,
        token_count: int = 0,
        context_length: int = DEFAULT_CONTEXT_LENGTH,
        id: str | None = None,
    ):
        """Initialize context usage display widget.

        Args:
            token_count: Current token count (default 0).
            context_length: Maximum context length in tokens.
            id: Widget ID for CSS targeting.
        """
        super().__init__(id=id)
        self.token_count = token_count
        self.context_length = context_length
        self.message_count: int = 0
        self.compaction_info: dict | None = None

    def update_usage(
        self,
        token_count: int,
        message_count: int = 0,
        compaction_info: dict | None = None,
    ) -> None:
        """Update token count, message count, and compaction info."""
        self.token_count = token_count
        self.message_count = message_count
        self.compaction_info = compaction_info
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
        """Update the widget display with color coding and compaction status."""
        percentage = self.get_percentage()
        text = self.get_formatted()

        lines = []

        if percentage >= 90:
            lines.append(f"[context-danger]{text}[/]")
        elif percentage >= 75:
            lines.append(f"[context-warning]{text}[/]")
        else:
            lines.append(text)

        if self.message_count > 0:
            lines.append(
                f"[context-muted]{self.message_count} messages in conversation[/]"
            )

        if self.compaction_info and self.compaction_info.get("was_compacted"):
            original_count = self.compaction_info.get("original_count", 0)
            new_count = self.compaction_info.get("new_count", 0)
            summarized_count = self.compaction_info.get("summarized_count", 0)
            lines.append(
                f"[context-compacted]📦 Compacted: {original_count} → {new_count} messages ({summarized_count} summarized)[/]"  # noqa: E501
            )

        self.update("\n".join(lines))


class QueryHistory(Static):
    """Widget to display recent query history."""

    DEFAULT_CSS = """
    QueryHistory {
        height: auto;
        padding: 1;
        background: transparent;
        border: none;
    }

    .history-item {
        height: auto;
        padding: 0 1;
        margin: 1 0;
    }

    .placeholder {
        color: $text-muted;
    }

    .history-number {
        color: $text-muted;
        text-style: bold;
    }
    """

    def __init__(self, max_queries: int = 5, id: str | None = None):
        """Initialize the query history widget."""
        super().__init__(id=id)
        self.max_queries = max_queries
        self.queries: list[str] = []

    def add_query(self, query: str) -> None:
        """Add a query to history."""
        self.queries.append(query)
        if len(self.queries) > self.max_queries:
            self.queries = self.queries[-self.max_queries :]
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


class ConversationTitle(Static):
    """Static conversation title displayed at top of context panel."""

    DEFAULT_CSS = """
    ConversationTitle {
        height: auto;
        padding: 1;
        background: transparent;
        border: none;
        text-style: bold;
        color: $text;
        content-align: center middle;
    }
    """

    def __init__(self, title: str = "", id: str | None = None):
        """Initialize conversation title widget.

        Args:
            title: Initial conversation title (empty shows as "New Conversation").
            id: Widget ID for CSS targeting.
        """
        super().__init__(title if title else "New Conversation", id=id)
        self._title = title

    def set_title(self, title: str) -> None:
        """Update the conversation title."""
        self._title = title
        self.update(title)


class ToolTimeline(Static):
    """Live timeline of tool calls for the current conversation turn."""

    DEFAULT_CSS = """
    ToolTimeline {
        padding: 0 1;
        color: $text-muted;
        background: transparent;
    }
    """

    MAX_ENTRIES = 20

    def __init__(self, id: str | None = None):
        """Initialize the tool timeline widget."""
        super().__init__("", id=id)
        # each entry: [name, state ("running"|"done"), elapsed_or_start]
        self._entries: list[list] = []

    @property
    def entries(self) -> list[tuple[str, str, float]]:
        """Return the current timeline entries as (name, state, elapsed)."""
        import time as _time

        out = []
        for name, state, t in self._entries:
            elapsed = (_time.monotonic() - t) if state == "running" else t
            out.append((name, state, elapsed))
        return out

    def start_tool(self, name: str) -> None:
        """Record a tool starting execution."""
        import time as _time

        self._entries.append([name, "running", _time.monotonic()])
        if len(self._entries) > self.MAX_ENTRIES:
            self._entries = self._entries[-self.MAX_ENTRIES :]
        self._render_entries()

    def finish_tool(self, name: str) -> None:
        """Record a tool finishing execution."""
        import time as _time

        for entry in reversed(self._entries):
            if entry[0] == name and entry[1] == "running":
                entry[1] = "done"
                entry[2] = _time.monotonic() - entry[2]
                break
        self._render_entries()

    def reset(self) -> None:
        """Clear all timeline entries."""
        self._entries = []
        self._render_entries()

    def _render_entries(self) -> None:
        """Refresh the rendered timeline text."""
        lines = []
        for name, state, elapsed in self.entries:
            if state == "running":
                lines.append(f"⏳ {name} …")
            else:
                lines.append(f"[green]✓[/green] {name} {elapsed:.1f}s")
        self.update("\n".join(lines))


class ContextPanel(Vertical):
    """Static context panel showing session info and tool status."""

    DEFAULT_CSS = """
    ContextPanel {
        width: 25%;
        padding: 1;
        background: $surface;
        border: none;
    }

    ContextPanel > * {
        border: none;
    }

    .tool-status {
        margin-top: 1;
        padding: 0;
        color: $text-muted;
        background: transparent;
    }

    .tool-status.querying {
        color: $warning;
    }

    .tool-status.completed {
        color: $success;
    }

    .tool-status.idle {
        color: $text-muted;
    }
    """

    def __init__(self, id: str | None = None):
        """Initialize the context panel widget."""
        super().__init__(id=id)
        self._provider_name: str = ""
        self._model_name: str = ""
        self._session_thread_id: str = ""
        self._current_directory: str = ""
        self._tool_status: str = "🔍 Idle"
        self._conversation_title: str = ""
        self._token_count: int = 0
        self._context_length: int = 4096

    def compose(self) -> ComposeResult:
        """Compose the working-state panel."""
        yield ConversationTitle(self._conversation_title, id="conversation-title")
        yield ToolTimeline(id="tool-timeline")

    @property
    def timeline(self) -> "ToolTimeline":
        """Return the mounted ToolTimeline widget."""
        return self.query_one("#tool-timeline", ToolTimeline)

    def on_mount(self) -> None:
        """Called when panel is mounted."""
        self._current_directory = os.getcwd()

        logger.debug(
            f"ContextPanel mounted, provider={self._provider_name!r}, model={self._model_name!r}"  # noqa: E501
        )
        self.refresh()

    def update_conversation_title(self, title: str) -> None:
        """Update the conversation title."""
        self._conversation_title = title
        try:
            title_widget = self.query_one("#conversation-title", ConversationTitle)
            title_widget.set_title(title)
        except Exception as e:
            logger.debug(f"Failed to update conversation title: {e}")

    def update_context_usage(self, token_count: int, context_length: int) -> None:
        """Store context usage values.

        The context/session sections were removed from the panel in favor of
        the consolidated StatusBar (see app.py). This method is kept
        no-op-compatible for existing callers and tests.
        """
        self._token_count = token_count
        self._context_length = context_length

    def update_session_info(
        self,
        provider: str,
        model: str,
        thread_id: str,
    ) -> None:
        """Store session info values.

        The Session Info section was removed from the panel in favor of the
        consolidated StatusBar (see app.py). This method is kept
        no-op-compatible for existing callers and tests.
        """
        self._provider_name = provider
        self._model_name = model
        self._session_thread_id = thread_id
        self._current_directory = os.getcwd()

    def update_tool_status(self, status: str) -> None:
        """Store tool status value.

        The #tool-status static was removed; tool status now lives in the
        StatusBar (Task 9 adds the timeline). Kept as a no-op-compatible
        method for existing callers and tests.
        """
        self._tool_status = status

    def reset_tool_status(self) -> None:
        """Reset tool status to idle state."""
        self.update_tool_status("🔍 Idle")


class ChatMessage(Markdown):
    """A Markdown-formatted chat message with role-based styling."""

    DEFAULT_CSS = """
    ChatMessage {
        padding: 1 2;
        margin: 0 0 1 0;
        border: none;
    }
    ChatMessage.user {
        background: transparent;
        color: $text;
        align-horizontal: right;
        border-left: solid #F97316;
        padding: 1 2 1 1;
    }
    ChatMessage.ai {
        background: transparent;
        color: $text;
        align-horizontal: left;
        border: none;
    }
    ChatMessage.system {
        background: transparent;
        color: $text-muted;
        text-style: italic;
        border: none;
        text-align: center;
    }
    """

    def __init__(self, role: str, content: str):
        """Initialize a chat message widget."""
        if role == "user" or role == "ai":
            formatted = content
        else:
            formatted = f"**[{role.upper()}]** {content}"
        super().__init__(formatted)
        self.role = role
        self.add_class(role)


class StatusBar(Static):
    """One-line status bar: version, model, thread, context usage, tool state."""

    DEFAULT_CSS = """
    StatusBar {
        height: 1;
        padding: 0 2;
        color: $text-muted;
        background: $background;
        border-top: solid $surface;
    }
    """

    def __init__(self):
        """Initialize the status bar widget."""
        super().__init__("Loading...")
        self._status_text: str = "Loading..."

    def set_status(
        self,
        model_name: str,
        thread_id: str,
        token_count: int = 0,
        context_length: int = 0,
        tool_status: str = "",
    ) -> None:
        """Update status bar display."""
        parts = [f"IRIS v{IRIS_VERSION}"]
        if model_name:
            parts.append(model_name)
        if thread_id:
            parts.append(f"thread {str(thread_id)[:8]}")
        if context_length > 0:
            pct = int((token_count / context_length) * 100)
            usage = f"{token_count:,}/{context_length:,} ({pct}%)"
            if pct >= 75:
                usage = f"[red]{usage}[/red]"
            parts.append(usage)
        if tool_status:
            parts.append(tool_status)
        self._status_text = " │ ".join(parts)
        self.update(self._status_text)


class ChatInput(TextArea):
    """TextArea that supports Shift+Enter for newlines and Enter for submission."""

    class Submitted(Message):
        """Message posted when user presses Enter."""

        def __init__(self, text_area: "ChatInput") -> None:
            """Initialize the submitted message."""
            self.text_area = text_area
            super().__init__()

        @property
        def control(self) -> "ChatInput":
            """Return the text area control."""
            return self.text_area

    async def _on_key(self, event: events.Key) -> None:
        """Handle Enter vs Shift+Enter at the TextArea level."""
        if event.key == "shift+enter":
            # Shift+Enter: insert newline
            event.stop()
            event.prevent_default()
            self.insert("\n")
        elif event.key == "enter":
            # Enter: submit
            event.stop()
            event.prevent_default()
            self.post_message(self.Submitted(self))
        else:
            await super()._on_key(event)


class Footer(Vertical):
    """Footer container for status bar and input field."""

    DEFAULT_CSS = """
    Footer {
        height: auto;
        padding: 1 2;
        background: $background;
        border-top: solid $surface;
    }
    StatusBar {
        height: 1;
    }
    Footer ChatInput {
        height: 3;
        min-height: 3;
        max-height: 10;
        width: 100%;
        border: none;
        border-left: solid #F97316;
        background: $surface;
        padding: 1 1 0 1;
    }
    Footer ChatInput:focus {
        border: none;
        border-left: solid #F97316;
    }
    """

    def __init__(self):
        """Initialize the footer widget."""
        super().__init__()
        self.status_bar = StatusBar()
        self.user_input = ChatInput(
            placeholder="Ask about your data... (Enter to send, Shift+Enter for new line)",  # noqa: E501
            id="user_input",
            show_line_numbers=False,
            soft_wrap=True,
        )
        self.user_input.styles.height = 3  # Set initial height via styles

    def compose(self) -> ComposeResult:
        """Compose the footer layout."""
        yield self.user_input
        yield self.status_bar


class ChatPanel(ScrollableContainer):
    """Scrollable chat panel for displaying messages and collapsible widgets."""

    DEFAULT_CSS = """
    ChatPanel {
        height: 100%;
        width: 100%;
        padding: 1;
    }
    """

    def __init__(self):
        """Initialize the chat panel widget."""
        super().__init__()
        self.messages: list[tuple[str, str]] = []

    def add_message(self, role: str, content: str):
        """Add a message to the chat panel."""
        self.messages.append((role, content))
        self.mount(ChatMessage(role, content))

        if len(self.messages) > MAX_MESSAGE_HISTORY:
            self.messages = self.messages[-MAX_MESSAGE_HISTORY:]

        # Always scroll to bottom when adding new message (after DOM update + small delay)  # noqa: E501
        self.call_after_refresh(
            lambda: self.set_timer(
                0.1, lambda: self.scroll_end(animate=False, immediate=True)
            )
        )

    def mount_collapsible(self, collapsible: "Collapsible") -> None:
        """Mount a collapsible widget to the chat panel."""
        logger.debug(f"Mounting collapsible {type(collapsible).__name__} to ChatPanel")
        self.mount(collapsible)

    async def mount_collapsible_async(self, collapsible: "Collapsible") -> None:
        """Mount a collapsible widget and scroll to make it visible."""
        self.mount(collapsible)
        collapsible.scroll_visible()
        self.refresh()


class ThinkingIndicator(Collapsible):
    """Collapsible thinking indicator that shows AI reasoning progress."""

    DEFAULT_CSS = """
    ThinkingIndicator {
        padding: 1 2;
        margin: 0 0 1 0;
        background: $surface;
        border: solid $primary;
        min-height: 2;
    }

    ThinkingIndicator .title {
        color: $accent;
        text-style: bold;
    }

    .thinking-steps {
        color: $text;
        padding: 0 1;
        margin: 1 0;
    }

    .thinking-steps .bullet {
        color: $accent;
    }

    ThinkingIndicator.collapsed {
        height: auto;
        min-height: 1;
    }
    """

    def __init__(self):
        """Initialize the thinking indicator widget."""
        super().__init__(title="▶ Thinking...", collapsed=False)
        self.steps: list[str] = []
        self._content_widget: Static | None = None

    def compose(self) -> ComposeResult:
        """Compose the thinking indicator layout."""
        yield Static("", classes="thinking-steps")

    def on_mount(self) -> None:
        """Handle widget mount event."""
        self._content_widget = self.query_one(".thinking-steps", Static)
        # Update display if steps were added before mount
        if self.steps:
            self._update_display()

    def add_step(self, step: str):
        """Add a thinking step to display."""
        self.steps.append(step)
        self._update_display()

    def _update_display(self):
        """Update the display with current steps."""
        if not self.steps:
            self.title = "▶ Thinking..."
            return

        step_count = len(self.steps)
        self.title = f"▶ Thinking... ({step_count} steps)"

        if self._content_widget:
            content = "\n".join(f"  • {s}" for s in self.steps)
            self._content_widget.update(content)

    def on_collapsible_expand(self) -> None:
        """Handle expand - show all steps."""
        if self._content_widget:
            content = "\n".join(f"  • {s}" for s in self.steps)
            self._content_widget.update(content)

    def on_collapsible_collapse(self) -> None:
        """Handle collapse - show summary."""
        self.title = f"▶ Thinking... ({len(self.steps)} steps)"


class SQLPanel(Collapsible):
    """Collapsible SQL query display in chat panel."""

    DEFAULT_CSS = """
    SQLPanel {
        margin: 0 0 1 0;
        background: $surface;
        border: solid $primary;
        min-height: 3;
    }

    SQLPanel .title {
        color: $accent;
        text-style: bold;
    }

    SQLPanel Markdown.sql-content {
        max-height: 20;
    }

    .sql-content {
        color: $text;
        padding: 1 2;
    }

    SQLPanel.collapsed {
        height: auto;
        min-height: 1;
    }
    """

    def __init__(self, sql: str = ""):
        """Initialize the SQL panel widget."""
        super().__init__(title="📝 SQL Query", collapsed=not sql)
        self.sql_content = sql
        self._content_widget: Markdown | None = None

    def compose(self) -> ComposeResult:
        """Compose the SQL panel layout."""
        content = (
            f"```sql\n{self.sql_content}\n```"
            if self.sql_content
            else "Executing query..."
        )
        yield Markdown(content, classes="sql-content")

    def on_mount(self) -> None:
        """Handle widget mount event."""
        self._content_widget = self.query_one(".sql-content", Markdown)

    def set_sql(self, sql: str) -> None:
        """Update the SQL content after mount."""
        self.sql_content = sql
        if self._content_widget:
            content = f"```sql\n{sql}\n```" if sql else "Executing query..."
            self._content_widget.update(content)
        self.collapsed = not sql


class ResultsPanel(Collapsible):
    """Collapsible query results display in chat panel."""

    DEFAULT_CSS = """
    ResultsPanel {
        margin: 0 0 1 0;
        background: $surface;
        border: solid $primary;
        min-height: 3;
    }

    ResultsPanel .title {
        color: $accent;
        text-style: bold;
    }

    ResultsPanel Markdown.results-content {
        max-height: 30;
    }

    .results-content {
        color: $text;
        padding: 1 2;
    }

    ResultsPanel.collapsed {
        height: auto;
        min-height: 1;
    }
    """

    def __init__(self, results: str = ""):
        """Initialize the results panel widget."""
        super().__init__(title="📊 Query Results", collapsed=not results)
        self.results_content = results
        self._content_widget: Markdown | None = None

    def compose(self) -> ComposeResult:
        """Compose the results panel layout."""
        content = (
            self._format_results_as_markdown(self.results_content)
            if self.results_content
            else "Waiting for results..."
        )
        yield Markdown(content, classes="results-content")

    def _format_results_as_markdown(self, results: str) -> str:
        """Convert raw query results to markdown table format."""
        if not results or results.strip() == "":
            return "No results returned."

        # Check if already formatted as markdown table
        if results.startswith("|") and "|---" in results:
            return results

        # Convert pipe-separated text to markdown table
        lines = results.strip().split("\n")
        if len(lines) < 2:
            return results

        # Assume first line is headers, second line is separator if present
        md_lines = []

        # Header row
        headers = lines[0].split("|")
        headers = [h.strip() for h in headers if h.strip()]
        md_lines.append("| " + " | ".join(headers) + " |")

        # Separator
        md_lines.append("|" + "---|" * len(headers))

        # Data rows
        for line in lines[1:]:
            if line.strip():
                cells = line.split("|")
                cells = [c.strip() for c in cells if c.strip()]
                md_lines.append("| " + " | ".join(cells) + " |")

        return "\n".join(md_lines)

    def on_mount(self) -> None:
        """Handle widget mount event."""
        self._content_widget = self.query_one(".results-content", Markdown)

    def set_results(self, results: str) -> None:
        """Update the results content after mount."""
        self.results_content = results
        if self._content_widget:
            formatted = self._format_results_as_markdown(results)
            self._content_widget.update(formatted)
        # Update collapsed state
        self.collapsed = not results
