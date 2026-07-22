"""Textual widgets for the ESDC chat TUI."""

# Standard library
import logging
import os

from textual import events
from textual.app import ComposeResult
from textual.containers import ScrollableContainer, Vertical
from textual.message import Message
from textual.widgets import Collapsible, Markdown, Static, TextArea

logger = logging.getLogger("esdc.chat.widgets")

MAX_MESSAGE_HISTORY = 100
IRIS_VERSION = "0.6.0"


class ContextHealth(Static):
    """Context pressure indicator — signals hallucination risk.

    States:
      <50%              green  "Full memory"
      50-75%            yellow "Getting long"
      >=75% or compacted red   "Compacted — older detail is summarized"
    Compaction latches: once compacted, stays red until reset().
    """

    DEFAULT_CSS = """
    ContextHealth {
        padding: 0 1;
        color: $text-muted;
        background: transparent;
    }
    """

    _BAR_CELLS = 10

    def __init__(self, id: str | None = None):
        """Initialize the context health widget."""
        super().__init__("Context ░░░░░░░░░░ 0%", id=id)
        self._compacted = False

    def update_health(
        self,
        token_count: int,
        context_length: int,
        message_count: int,
        compacted: bool = False,
        exact: bool = True,
    ) -> None:
        """Update the context pressure display."""
        if compacted:
            self._compacted = True
        pct = (
            int((token_count / context_length) * 100) if context_length > 0 else 0
        )
        filled = min(self._BAR_CELLS, round(pct / self._BAR_CELLS))
        bar = "▓" * filled + "░" * (self._BAR_CELLS - filled)
        if self._compacted:
            state = "[red]● Compacted — older detail is summarized[/red]"
        elif pct >= 50:
            state = "[yellow]● Getting long[/yellow]"
        else:
            state = "[green]● Full memory[/green]"
        approx = "" if exact else "≈"
        self.update(
            f"Context {bar} {approx}{pct}%\n"
            f"{approx}{token_count:,} / {context_length:,} · {message_count} messages\n"
            f"{state}"
        )

    def reset(self) -> None:
        """Reset the context health display and clear the compaction latch."""
        self._compacted = False
        self.update("Context ░░░░░░░░░░ 0%")


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

    def on_mount(self) -> None:
        """Refresh running entries every second so elapsed time ticks."""
        self.set_interval(1.0, self._tick)

    def _tick(self) -> None:
        """Re-render if any entry is still running (elapsed time ticks)."""
        if any(state == "running" for _, state, _ in self._entries):
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
        yield SQLPanel(id="sql-panel")
        yield ResultsPanel(id="results-panel")
        yield ContextHealth(id="context-health")

    @property
    def timeline(self) -> "ToolTimeline":
        """Return the mounted ToolTimeline widget."""
        return self.query_one("#tool-timeline", ToolTimeline)

    @property
    def sql_panel(self) -> "SQLPanel":
        """Return the mounted SQLPanel widget."""
        return self.query_one("#sql-panel", SQLPanel)

    @property
    def results_panel(self) -> "ResultsPanel":
        """Return the mounted ResultsPanel widget."""
        return self.query_one("#results-panel", ResultsPanel)

    @property
    def context_health(self) -> "ContextHealth":
        """Return the mounted ContextHealth widget."""
        return self.query_one("#context-health", ContextHealth)

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
        exact: bool = True,
    ) -> None:
        """Update status bar display."""
        parts = [f"IRIS v{IRIS_VERSION}"]
        if model_name:
            parts.append(model_name)
        if thread_id:
            parts.append(f"thread {str(thread_id)[:8]}")
        if context_length > 0:
            pct = int((token_count / context_length) * 100)
            approx = "" if exact else "≈"
            usage = f"{approx}{token_count:,}/{context_length:,} ({pct}%)"
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

    ThinkingIndicator.done {
        color: $text-muted;
        height: 1;
        overflow: hidden;
    }
    """

    _MAX_VISIBLE_LINES = 12

    def __init__(self):
        """Initialize the thinking indicator widget."""
        super().__init__(title="▶ Thinking...", collapsed=False)
        self.steps: list[str] = []
        self._content_widget: Static | None = None
        self._reasoning_text: str = ""
        self._done: bool = False

    def compose(self) -> ComposeResult:
        """Compose the thinking indicator layout."""
        self._content_widget = Static("", classes="thinking-steps")
        yield self._content_widget

    def on_mount(self) -> None:
        """Handle widget mount event."""
        # Update display if steps were added before mount
        if self._content_widget is not None and self.steps:
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

    def append_reasoning(self, text: str) -> None:
        """Accumulate reasoning text and show the tail."""
        if not hasattr(self, "_reasoning_text"):
            self._reasoning_text = ""
            self._done = False
        self._reasoning_text += text
        lines = self._reasoning_text.splitlines() or [self._reasoning_text]
        tail = "\n".join(lines[-self._MAX_VISIBLE_LINES :])
        if self._content_widget:
            self._content_widget.update(tail)

    def mark_done(self) -> None:
        """Dim/collapse the indicator once real answer tokens start."""
        self._done = True
        self.add_class("done")


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

    def __init__(self, sql: str = "", id: str | None = None):
        """Initialize the SQL panel widget."""
        super().__init__(title="📝 SQL Query", collapsed=not sql, id=id)
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

    def __init__(self, results: str = "", id: str | None = None):
        """Initialize the results panel widget."""
        super().__init__(title="📊 Query Results", collapsed=not results, id=id)
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
