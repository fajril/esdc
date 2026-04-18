# esdc/chat/app.py

# Standard library
import asyncio
import json
import logging
import os
from collections.abc import AsyncGenerator
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

# Configure logging FIRST - before any other imports that might use logging
# Log location: ~/.esdc/logs/
log_dir = Path.home() / ".esdc" / "logs"
log_dir.mkdir(parents=True, exist_ok=True)
log_file = log_dir / "esdc_chat.log"

# Import Config for logging configuration
# Note: This is safe because configs.py doesn't import from chat/
from esdc.configs import Config  # noqa: E402

# Get logging configuration
log_config = Config.get_logging_config()
chat_level_str = log_config.get("chat", {}).get("level", "WARNING").upper()
file_config = log_config.get("file", {})

# Configure esdc.chat logger
logger = logging.getLogger("esdc.chat")

if chat_level_str == "0":
    # Disable logging completely
    logging.disable(logging.CRITICAL)
else:
    # Parse log level
    chat_level = getattr(logging, chat_level_str, logging.WARNING)

    # Rotating file handler: from config or defaults
    max_size = 10 * 1024 * 1024  # Default 10MB
    backup_count = 5  # Default 5 backups
    if file_config.get("max_size"):
        # Parse size string like '10MB'
        size_str = file_config["max_size"].upper().strip()
        for suffix, multiplier in [("KB", 1024), ("MB", 1024**2), ("GB", 1024**3)]:
            if size_str.endswith(suffix):
                max_size = int(size_str[: -len(suffix)]) * multiplier
                break

    if file_config.get("backup_count"):
        backup_count = file_config["backup_count"]

    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=max_size,
        backupCount=backup_count,
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
    )

    logger.setLevel(chat_level)
    logger.addHandler(file_handler)
    logger.propagate = False  # Critical: prevent logs from propagating to root logger

    logger.info(f"ESDC Chat starting, log file: {log_file}")

# Suppress verbose httpcore and markdown_it logs (only show WARNING+)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("markdown_it").setLevel(logging.WARNING)

# Third-party
from langchain_core.language_models import BaseChatModel  # noqa: E402
from langchain_core.runnables import Runnable  # noqa: E402
from langgraph.checkpoint.base import BaseCheckpointSaver  # noqa: E402
from textual import events  # noqa: E402
from textual.app import App, ComposeResult  # noqa: E402
from textual.binding import Binding  # noqa: E402
from textual.containers import (  # noqa: E402
    Container,
    Horizontal,
    ScrollableContainer,
    Vertical,
)
from textual.message import Message  # noqa: E402
from textual.widget import Widget  # noqa: E402
from textual.widgets import Collapsible, Markdown, Static, TextArea  # noqa: E402

MAX_MESSAGE_HISTORY = 100
MAX_QUERY_HISTORY = 5
DEFAULT_CONTEXT_LENGTH = 4096
TOOLS_LIST = ["execute_sql", "get_schema", "list_tables"]


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


class ToolStatusList(Static):
    """Widget to display available tools and their status."""

    DEFAULT_CSS = """
    ToolStatusList {
        height: auto;
        padding: 1;
        background: transparent;
        border: none;
    }

    .tool-item {
        height: auto;
        padding: 0 1;
        margin: 1 0;
    }

    .tool-available {
        color: $text-muted;
    }

    .tool-available .icon {
        color: $text-disabled;
    }

    .tool-used {
        color: $primary;
        text-style: bold;
    }

    .tool-used .icon {
        color: $primary;
    }
    """

    def __init__(self, id: str | None = None):
        super().__init__(id=id)
        self.tools = TOOLS_LIST
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
        super().__init__(title if title else "New Conversation", id=id)
        self._title = title

    def set_title(self, title: str) -> None:
        """Update the conversation title."""
        self._title = title
        self.update(title)


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
        """Compose all sections of context panel."""
        from textual.widgets import Static

        # 1. Conversation Title (static top)
        yield ConversationTitle(
            self._conversation_title,
            id="conversation-title",
        )

        # 2. Session Info (collapsible, expanded by default)

        self._current_directory = os.getcwd()
        thread_display = (
            str(self._session_thread_id)[:8] if self._session_thread_id else "N/A"
        )
        session_content = (
            f"IRIS v0.5.0\nThread: {thread_display}...\nDir: {self._current_directory}"
        )

        with ContextSection(
            "Session Info",
            expanded=True,
            id="session-section",
        ):
            yield Static(
                session_content,
                classes="session-content",
                id="session-content",
            )

        # 3. Context (collapsible, expanded by default)
        with ContextSection(
            "Context",
            expanded=True,
            id="context-section",
        ):
            yield ContextUsageWidget(
                token_count=self._token_count,
                context_length=self._context_length,
                id="context-usage",
            )

        # 4. Tool status indicator (static)
        yield Static(self._tool_status, classes="tool-status idle", id="tool-status")

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
        """Update context usage display."""
        self._token_count = token_count
        self._context_length = context_length
        try:
            context_widget = self.query_one("#context-usage", ContextUsageWidget)
            context_widget.token_count = token_count
            context_widget.context_length = context_length
            context_widget._update_display()
            logger.debug(
                f"🔍 Updated context usage: {token_count:,} / {context_length:,}"
            )
        except Exception as e:
            logger.warning(f"❌ Failed to update context usage: {e}")
        self.refresh()

    def update_session_info(
        self,
        provider: str,
        model: str,
        thread_id: str,
    ) -> None:
        """Update session information displayed in the context panel."""
        self._provider_name = provider
        self._model_name = model
        self._session_thread_id = thread_id

        # Get current directory

        self._current_directory = os.getcwd()

        # Update the static content
        try:
            session_content = self.query_one("#session-content", Static)
            thread_display = str(thread_id)[:8] if thread_id else "N/A"
            session_content.update(
                f"IRIS v0.5.0\nThread: {thread_display}...\nDir: {self._current_directory}"  # noqa: E501
            )
        except Exception:
            pass
        self.refresh()

    def update_tool_status(self, status: str) -> None:
        """Update tool execution status with emoji+text."""
        self._tool_status = status
        try:
            status_widget = self.query_one("#tool-status", Static)
            status_widget.update(status)
            # Set appropriate class based on status
            if "⏳" in status:
                status_widget.set_class(True, "querying")
                status_widget.set_class(False, "completed")
                status_widget.set_class(False, "idle")
            elif "✅" in status:
                status_widget.set_class(False, "querying")
                status_widget.set_class(True, "completed")
                status_widget.set_class(False, "idle")
            else:
                status_widget.set_class(False, "querying")
                status_widget.set_class(False, "completed")
                status_widget.set_class(True, "idle")
        except Exception:
            pass

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
        if role == "user" or role == "ai":
            formatted = content
        else:
            formatted = f"**[{role.upper()}]** {content}"
        super().__init__(formatted)
        self.role = role
        self.add_class(role)


class StatusBar(Static):
    """Status line showing IRIS and token count."""

    DEFAULT_CSS = """
    StatusBar {
        height: 1;
        padding: 0 2;
        color: $text-muted;
        background: $background;
        border-top: solid $surface;
    }

    .status-provider {
        color: $text;
        text-style: bold;
    }

    .status-model {
        color: $text;
    }

    .status-tokens {
        color: $text-muted;
    }
    """

    def __init__(self):
        super().__init__("Loading...")

    def set_status(
        self,
        provider_name: str,
        model_name: str,
        token_count: int = 0,
        context_length: int = 0,
        thread_id: str = "",
    ) -> None:
        """Update status bar display."""
        parts = ["IRIS"]

        if context_length > 0 and token_count > 0:
            percentage = int((token_count / context_length) * 100)
            parts.append(f"{token_count:,} tokens ({percentage}%)")
        elif token_count > 0:
            parts.append(f"{token_count:,} tokens")

        if thread_id:
            thread_id_str = str(thread_id)
            parts.append(f"thread: {thread_id_str[:8]}")

        self.update(" | ".join(parts))


class ChatInput(TextArea):
    """TextArea that supports Shift+Enter for newlines and Enter for submission."""

    class Submitted(Message):
        """Message posted when user presses Enter."""

        def __init__(self, text_area: "ChatInput") -> None:
            self.text_area = text_area
            super().__init__()

        @property
        def control(self) -> "ChatInput":
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
        super().__init__(title="▶ Thinking...", collapsed=False)
        self.steps: list[str] = []
        self._content_widget: Static | None = None

    def compose(self) -> ComposeResult:
        yield Static("", classes="thinking-steps")

    def on_mount(self) -> None:
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
        super().__init__(title="📝 SQL Query", collapsed=not sql)
        self.sql_content = sql
        self._content_widget: Markdown | None = None

    def compose(self) -> ComposeResult:
        content = (
            f"```sql\n{self.sql_content}\n```"
            if self.sql_content
            else "Executing query..."
        )
        yield Markdown(content, classes="sql-content")

    def on_mount(self) -> None:
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
        super().__init__(title="📊 Query Results", collapsed=not results)
        self.results_content = results
        self._content_widget: Markdown | None = None

    def compose(self) -> ComposeResult:
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
        self._content_widget = self.query_one(".results-content", Markdown)

    def set_results(self, results: str) -> None:
        """Update the results content after mount."""
        self.results_content = results
        if self._content_widget:
            formatted = self._format_results_as_markdown(results)
            self._content_widget.update(formatted)
        # Update collapsed state
        self.collapsed = not results


class RightPanel(Vertical):
    """Container for SQL and Results panels."""

    DEFAULT_CSS = """
    RightPanel {
        width: 100%;
        height: 100%;
    }
    """

    def __init__(self):
        super().__init__()
        self.sql_panel = SQLPanel()
        self.results_panel = ResultsPanel()

    def compose(self) -> ComposeResult:
        yield self.sql_panel
        yield self.results_panel

    def set_sql(self, sql: str):
        self.sql_panel.set_sql(sql)

    def set_results(self, results: str):
        self.results_panel.set_results(results)


class ESDCChatApp(App):
    CSS = """
    /* ===== Minimalist Clean UI - Applied UX Laws =====
       Occam's Razor: Simplest effective design
       Hick's Law: Minimize cognitive load
       Law of Prägnanz: Simple, clear visual hierarchy
       Aesthetic-Usability: Clean = usable
    ================================================================ */

    Screen {
        layout: vertical;
        background: $background;
    }

    #main-content {
        layout: horizontal;
        height: 1fr;
        width: 100%;
        padding: 0;
    }

    #chat-area {
        width: 3fr;
        height: 100%;
        border: none;
        background: $background;
        padding: 0;
    }

    #context-panel {
        width: 1fr;
        height: 100%;
        border-left: solid $surface;
        background: $surface;
        padding: 0;
        overflow: hidden;
    }

    StatusBar {
        height: 1;
        padding: 0 2;
        color: $text-muted;
        background: $background;
        border-top: solid $surface;
    }

    ChatPanel {
        height: 100%;
        width: 100%;
        padding: 1;
        scrollbar-gutter: stable;
    }

    .placeholder {
        color: $text-muted;
        text-style: italic;
        text-align: center;
        padding: 2;
    }

    /* Content widgets for sections */
    .sql-content, .results-content, .schema-content, .session-content {
        padding: 1 1;
        color: #ffffff;
        background: transparent;
        text-style: bold;
    }

    .sql-content {
        color: #5dade2;
    }

    .results-content {
        color: #ffffff;
    }

    .schema-content {
        color: #a0a0a0;
    }

    .session-content {
        color: #a0a0a0;
    }

    /* ===== Widget - Clean Design ===== */
    ContextUsageWidget {
        height: auto;
        padding: 0;
        background: transparent;
        border: none;
    }

    ToolStatusList {
        height: auto;
        padding: 1;
        background: transparent;
        border: none;
    }

    .tool-item {
        height: auto;
        padding: 0 1;
        margin: 1 0;
    }

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

    /* ===== Thinking Indicator - Subtle ===== */
    ThinkingIndicator {
        padding: 1 2;
        margin: 1 0;
        background: transparent;
        border: none;
    }

    /* ===== SQL & Results Panels - Clean ===== */
    SQLPanel {
        width: 100%;
        height: 1fr;
        border: none;
        padding: 1;
        background: transparent;
    }

    ResultsPanel {
        width: 100%;
        height: 1fr;
        border: none;
        padding: 1;
        background: transparent;
    }

    /* ===== Footer & Input - Clear Chat Bar ===== */
    Footer {
        height: auto;
        padding: 1 2;
        background: $background;
        border-top: solid $surface;
    }

    /* TextArea Chat Input - Plain styling, no color box */
    .chat-input {
        height: 3;
        min-height: 3;
        max-height: 10;
        width: 100%;
        /* No border, no background - completely plain */
    }

    .chat-input TextArea {
        height: 100%;
        width: 100%;
        /* No special styling */
    }
    """

    BINDINGS = [
        Binding("ctrl+h", "toggle_context_panel", "Toggle Panel"),
        Binding("ctrl+l", "toggle_sql_section", "Toggle SQL"),
        Binding("ctrl+r", "toggle_results_section", "Toggle Results"),
        Binding("ctrl+e", "toggle_all_sections", "Toggle All"),
        Binding("ctrl+shift+s", "save_screenshot", "Save Screenshot"),
        Binding("escape", "cancel_query", "Cancel"),
    ]

    def __init__(self):
        super().__init__()
        self.chat_panel: ChatPanel | None = None
        self._context_panel: ContextPanel | None = None
        self.status_bar: StatusBar | None = None
        self.user_input: ChatInput | None = None
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
        self._context_metadata: dict | None = None

        # Queue-based streaming infrastructure
        self._event_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._streaming_message: ChatMessage | None = None
        self._accumulated_content: str = ""
        self._conversation_title: str = ""
        self._title_generated: bool = False

    def compose(self) -> ComposeResult:
        # Main horizontal split container
        chat = ChatPanel()
        chat.id = "chat-area"

        ctx = ContextPanel()
        ctx.id = "context-panel"

        main = Horizontal(chat, ctx)
        main.id = "main-content"

        yield main
        yield Footer()

    def on_mount(self) -> None:
        # Get the chat panel and context panel from compose
        self.chat_panel = self.query_one("#chat-area", ChatPanel)
        self._context_panel = self.query_one("#context-panel", ContextPanel)

        # Get input and status bar from Footer
        footer = self.query_one(Footer)
        self.user_input = footer.user_input
        self.status_bar = footer.status_bar
        self.user_input.focus()

        from esdc.configs import Config

        # Initialize config if not exists (creates directory and default config)
        Config.init_config()

        self._provider_name = Config.get_default_provider()
        self._model_name = Config.get_provider_model()

        # Get context length from provider
        provider_config = Config.get_provider_config()
        if provider_config and provider_config.get("model"):
            from esdc.providers import get_provider

            provider_type = provider_config.get("provider_type", "ollama")
            provider = get_provider(provider_type)
            if provider:
                # Use dynamic API fetching for Ollama
                if provider_type == "ollama":
                    from esdc.providers.ollama import OllamaProvider

                    self._context_length = OllamaProvider.get_context_length_from_api(
                        provider_config.get("model", ""),
                        provider_config.get("base_url"),
                    )
                else:
                    self._context_length = provider.get_context_length(
                        provider_config.get("model", "")
                    )

        # Log context length
        logger.info(
            f"📊 Context length: {self._model_name} = {self._context_length:,} tokens"
        )

        # Update context panel with session info
        if self._context_panel:
            self._context_panel.update_session_info(
                self._provider_name,
                self._model_name,
                self._thread_id,
            )
            # Initialize context usage display
            self._context_panel.update_context_usage(
                self._token_count,
                self._context_length,
            )

        # Set up timer to consume events from queue (runs every 50ms)
        self.set_interval(0.05, self._consume_events)

        self.status_bar.set_status(
            self._provider_name,
            self._model_name,
            self._token_count,
            self._context_length,
            self._thread_id,
        )

        self._init_agent()

    def _init_agent(self) -> None:
        """Initialize the LLM and agent."""
        from esdc.chat.agent import create_agent
        from esdc.chat.memory import create_checkpointer, create_thread_id
        from esdc.configs import Config
        from esdc.providers import create_llm_from_config

        provider_config = Config.get_provider_config()
        if not provider_config:
            self.display_message(
                "system",
                "Error: No provider configured. Run 'esdc chat --setup' first.",
            )
            return

        self._llm = create_llm_from_config(provider_config)
        self._checkpointer = create_checkpointer()
        self._agent = create_agent(
            self._llm,
            checkpointer=self._checkpointer,
            context_length=self._context_length,
        )
        self._thread_id = create_thread_id()

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        """Auto-expand TextArea based on content lines."""
        if not self.user_input:
            return

        # Calculate required height based on number of lines
        line_count = self.user_input.text.count("\n") + 1
        new_height = min(max(line_count + 1, 3), 10)  # Min 3, Max 10

        # Update height if changed
        if self.user_input.styles.height != new_height:
            self.user_input.styles.height = new_height

    def on_chat_input_submitted(self, event: ChatInput.Submitted) -> None:
        """Handle message submission from ChatInput."""
        if not self.user_input:
            return

        user_input = self.user_input.text.strip()
        if not user_input:
            return

        # Generate conversation title on first query (in background)
        if not self._title_generated and self._llm:
            self._title_generated = True
            asyncio.create_task(self._generate_and_set_title(user_input, self._llm))

        logger.info(f"User input submitted: {user_input[:50]}...")
        self.display_message("user", user_input)

        # Clear the input and reset height
        self.user_input.text = ""
        self.user_input.styles.height = 3
        self._cancelled = False

        if not self._agent:
            logger.error("Agent not initialized")
            self.display_message("ai", "Error: Agent not initialized")
            return

        # Create streaming AI message
        self._streaming_message = ChatMessage("ai", "")
        self._accumulated_content = ""

        if self.chat_panel:
            self.chat_panel.mount(self._streaming_message)
            self._streaming_message.scroll_visible()
            # Scroll to bottom when mounting new AI message (after DOM update)
            chat_panel = self.chat_panel
            self.call_after_refresh(
                lambda: chat_panel.scroll_end(animate=False, immediate=True)
            )

        # Start background streaming task (NON-blocking)
        asyncio.create_task(self._stream_in_background(user_input))
        logger.info("🚀 Started background streaming task")

    async def _stream_in_background(self, user_input: str) -> None:
        """Run agent streaming in background, post events to queue."""
        logger.info("=" * 60)
        logger.info("🚀 QUERY START: user_input='%s'", user_input[:80])
        logger.info("=" * 60)

        try:
            async for chunk in self._stream_response(user_input):
                if self._cancelled:
                    await self._event_queue.put(
                        {"type": "complete", "success": False, "error": "Cancelled"}
                    )
                    return
                await self._event_queue.put(chunk)

            # Signal completion
            await self._event_queue.put({"type": "complete", "success": True})
            logger.info("Query completed successfully")

        except asyncio.TimeoutError:
            logger.warning("Query timed out after 120 seconds")
            await self._event_queue.put(
                {
                    "type": "complete",
                    "success": False,
                    "error": "Request timed out after 2 minutes. Please try again.",
                }
            )
        except Exception as e:
            logger.exception(f"Query failed with error: {e}")
            await self._event_queue.put(
                {"type": "complete", "success": False, "error": str(e)}
            )
        finally:
            # Reset tool status after streaming completes
            if self._context_panel:
                self._context_panel.reset_tool_status()

    async def _generate_and_set_title(self, user_input: str, llm: Any) -> None:
        """Generate conversation title in background and update UI."""
        try:
            from esdc.chat.agent import generate_conversation_title

            title = await generate_conversation_title(llm, user_input)
            self._conversation_title = title

            # Update context panel from main thread
            if self._context_panel:
                self._context_panel.update_conversation_title(title)
        except Exception as e:
            logger.debug(f"Failed to generate conversation title: {e}")
            # Fallback: use first 50 chars of query
            if len(user_input) > 50:
                self._conversation_title = user_input[:47] + "..."
            else:
                self._conversation_title = user_input
            if self._context_panel:
                self._context_panel.update_conversation_title(self._conversation_title)

    async def _consume_events(self) -> None:
        """Consume events from queue in main thread (called by timer every 50ms)."""
        try:
            # Process up to 10 events per tick to avoid blocking
            for _ in range(10):
                try:
                    chunk = self._event_queue.get_nowait()
                    await self._process_chunk(chunk)
                except asyncio.QueueEmpty:
                    break
        except Exception as e:
            logger.error(f"Error consuming events: {e}")

    async def _process_chunk(self, chunk: dict[str, Any]) -> None:
        """Process a single chunk and update UI."""
        chunk_type = chunk.get("type", "unknown")

        if chunk_type == "token":
            token = chunk.get("content", "")
            if token and self._streaming_message:
                self._accumulated_content += token
                self._streaming_message.update(self._accumulated_content)

                # Always scroll to bottom on new content (after DOM update + delay for Markdown)  # noqa: E501
                if self.chat_panel:
                    chat_panel = self.chat_panel
                    self.call_after_refresh(
                        lambda: self.set_timer(
                            0.1,
                            lambda: chat_panel.scroll_end(
                                animate=False, immediate=True
                            ),
                        )
                    )

        elif chunk_type == "message":
            content = chunk.get("content", "")
            if content and not self._accumulated_content:
                self._accumulated_content = content
                if self._streaming_message:
                    self._streaming_message.update(self._accumulated_content)

                # Always scroll to bottom on new content (after DOM update + delay for Markdown)  # noqa: E501
                if self.chat_panel:
                    chat_panel = self.chat_panel
                    self.call_after_refresh(
                        lambda: self.set_timer(
                            0.1,
                            lambda: chat_panel.scroll_end(
                                animate=False, immediate=True
                            ),
                        )
                    )

        elif chunk_type == "tool_call":
            tool_name = chunk.get("tool", "")
            tool_args = chunk.get("args", {})

            # Tool-specific status messages
            TOOL_STATUS_MAP = {  # noqa: N806
                "execute_sql": "⏳ Executing SQL query...",
                "SQL Executor": "🛠️ Using SQL Executor...",
                "get_schema": "⏳ Getting table schema...",
                "Schema Inspector": "🛠️ Using Schema Inspector...",
                "list_tables": "⏳ Listing available tables...",
                "Table Lister": "🛠️ Using Table Lister...",
                "get_recommended_table": "⏳ Finding recommended table...",
                "Table Selector": "🛠️ Using Table Selector...",
                "resolve_uncertainty_level": "⏳ Resolving uncertainty level...",
                "Uncertainty Resolver": "🛠️ Using Uncertainty Resolver...",
                "search_problem_cluster": "⏳ Searching problem cluster definitions...",
                "Problem Cluster Search": "🛠️ Using Problem Cluster Search...",
            }

            sql_query = ""
            if isinstance(tool_args, dict):
                sql_query = tool_args.get("query", "")
            elif isinstance(tool_args, str):
                try:
                    parsed = json.loads(tool_args)
                    sql_query = parsed.get("query", "")
                except (json.JSONDecodeError, TypeError):
                    pass

            logger.info(
                "🛠️ TOOL_CALL: name=%s, sql_len=%d",
                tool_name,
                len(sql_query) if sql_query else 0,
            )

            # Get appropriate status message
            status_msg = TOOL_STATUS_MAP.get(tool_name, f"⏳ Using {tool_name}...")

            # Update tool status
            if self._context_panel:
                self._context_panel.update_tool_status(status_msg)

            # Add indicator to message
            if self._streaming_message:
                indicator_text = f"\n\n{status_msg}"
                if sql_query and tool_name in ("execute_sql", "SQL Executor"):
                    indicator_text += f"\n\n```sql\n{sql_query}\n```\n"

                self._accumulated_content += indicator_text
                self._streaming_message.update(self._accumulated_content)

                # Always scroll to bottom when adding tool indicator (after DOM update + delay for Markdown)  # noqa: E501
                if self.chat_panel:
                    chat_panel = self.chat_panel
                    self.call_after_refresh(
                        lambda: self.set_timer(
                            0.1,
                            lambda: chat_panel.scroll_end(
                                animate=False, immediate=True
                            ),
                        )
                    )

        elif chunk_type == "tool_result":
            result = chunk.get("result", "")
            tool_name = chunk.get("tool", "")

            logger.info(
                "[TOOL] TOOL_RESULT: tool=%s, result_len=%d",
                tool_name,
                len(result),
            )

            # Tool-specific completion messages
            TOOL_COMPLETED_MAP = {  # noqa: N806
                "execute_sql": "✅ SQL query completed",
                "SQL Executor": "✅ SQL Executor completed",
                "get_schema": "✅ Schema retrieved",
                "Schema Inspector": "✅ Schema Inspector completed",
                "list_tables": "✅ Tables listed",
                "Table Lister": "✅ Table Lister completed",
                "get_recommended_table": "✅ Recommended table found",
                "Table Selector": "✅ Table Selector completed",
                "resolve_uncertainty_level": "✅ Uncertainty level resolved",
                "Uncertainty Resolver": "✅ Uncertainty Resolver completed",
                "search_problem_cluster": "✅ Problem cluster definition found",
                "Problem Cluster Search": "✅ Problem Cluster Search completed",
            }

            # Update tool status
            if self._context_panel:
                completed_msg = TOOL_COMPLETED_MAP.get(tool_name, "✅ Tool completed")
                self._context_panel.update_tool_status(completed_msg)

        elif chunk_type == "context_metadata":
            metadata = chunk.get("metadata")
            # Only update if compaction occurred (don't overwrite with non-compaction)
            if metadata and metadata.get("was_compacted"):
                self._context_metadata = metadata

        elif chunk_type == "messages_state":
            messages = chunk.get("messages", [])
            message_count = chunk.get("message_count", len(messages))
            if messages:
                from esdc.chat.context_manager import estimate_tokens

                self._token_count = estimate_tokens(messages)
                if self.status_bar:
                    self.status_bar.set_status(
                        self._provider_name,
                        self._model_name,
                        self._token_count,
                        self._context_length,
                        self._thread_id,
                    )
                if self._context_panel:
                    self._context_panel.update_context_usage(
                        self._token_count,
                        self._context_length,
                    )
                    try:
                        context_widget = self._context_panel.query_one(
                            "#context-usage", ContextUsageWidget
                        )
                        context_widget.update_usage(
                            self._token_count, message_count, self._context_metadata
                        )
                    except Exception:
                        pass

        elif chunk_type == "token_usage":
            # DEPRECATED: messages_state provides more accurate token count
            # Keep for backward compatibility with older agent versions
            pass

        elif chunk_type == "complete":
            success = chunk.get("success", True)
            error = chunk.get("error")

            if not success and error and self._streaming_message:
                self._streaming_message.update(f"Error: {error}")

            # Final scroll to bottom after streaming completes (critical fix)
            if self.chat_panel:
                chat_panel = self.chat_panel
                self.call_after_refresh(
                    lambda: self.set_timer(
                        0.1,
                        lambda: chat_panel.scroll_end(animate=False, immediate=True),
                    )
                )

            # Reset state
            self._streaming_message = None
            self._accumulated_content = ""

    async def _stream_response(
        self, user_input: str
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Stream response from the agent."""
        from esdc.chat.agent import run_agent_stream

        if not self._agent:
            return

        async for chunk in run_agent_stream(
            self._agent,
            user_input,
            self._thread_id,
        ):
            # CRITICAL: Forward token events for real-time streaming
            if chunk["type"] == "token":
                yield chunk
            elif chunk["type"] == "message":
                content = chunk.get("content", "")
                if content:
                    yield {"type": "message", "content": content}
            elif chunk["type"] == "tool_call":
                yield {
                    "type": "tool_call",
                    "tool": chunk.get("tool", ""),
                    "args": chunk.get("args", {}),
                }
            elif chunk["type"] == "tool_result":
                result = chunk.get("result", "")
                sql = chunk.get("sql", "")
                yield {"type": "tool_result", "result": result, "sql": sql}
            elif (
                chunk["type"] == "token_usage"
                or chunk["type"] == "messages_state"
                or chunk["type"] == "context_metadata"
            ):
                yield chunk

    def display_message(self, role: str, content: str) -> None:
        if self.chat_panel:
            self.chat_panel.add_message(role, content)
            self._message_count += 1

    def action_cancel_query(self) -> None:
        """Cancel the current query."""
        self._cancelled = True
        self.notify("Query cancelled")

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

        self.notify(
            f"Context panel {'shown' if self._context_panel_visible else 'hidden'}"
        )

    def action_save_screenshot(self, filename: str | None = None) -> None:
        """Save screenshot of the current screen.

        Args:
            filename: Optional filename for the screenshot.
            Defaults to timestamped filename.
        """
        from pathlib import Path

        if filename is None:
            timestamp = datetime.now().strftime("%Y-%m-%d at %H.%M.%S")
            filename = f"esdc-chat-{timestamp}.png"

        # Save to screenshot directory
        screenshot_dir = Path.cwd() / "screenshot"
        screenshot_dir.mkdir(exist_ok=True)

        filepath = screenshot_dir / filename

        # Note: Full terminal screenshot requires external tools like screenshot CLI
        # For now, notify user and suggest using system screenshot
        self.notify(f"Use terminal screenshot tool. Suggested path: {filepath}")
