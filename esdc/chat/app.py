# esdc/chat/app.py
import asyncio
from typing import Any, AsyncGenerator

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
from textual.widgets import Static, Input, Markdown, Button, Collapsible
from textual.widget import Widget
from textual.events import Key

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable
from langgraph.checkpoint.base import BaseCheckpointSaver

MAX_MESSAGE_HISTORY = 100
MAX_QUERY_HISTORY = 5
DEFAULT_CONTEXT_LENGTH = 4096
TOOLS_LIST = ["execute_sql", "get_schema", "list_tables"]


class ContextSection(Static):
    """Collapsible section widget for context panel."""

    DEFAULT_CSS = """
    ContextSection {
        margin: 0;
        border: none;
    }

    .header {
        background: transparent;
        padding: 1 1;
        border-bottom: solid $primary-background;
    }

    .title {
        color: $text;
        text-style: bold;
    }

    .content {
        padding: 1 1;
        background: transparent;
    }

    ContextSection.expanded .content {
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


class TokenUsageWidget(Static):
    """Widget to display token usage with percentage."""

    DEFAULT_CSS = """
    TokenUsageWidget {
        height: auto;
        padding: 1;
        background: transparent;
        border: none;
        margin: 1 0;
    }

    .token-display {
        color: $text;
        text-style: bold;
    }

    .token-muted {
        color: $text-muted;
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


class ContextPanel(Vertical):
    """Static context panel showing only session info."""

    DEFAULT_CSS = """
    ContextPanel {
        width: 25%;
        padding: 1;
        background: $surface;
    }
    """

    def __init__(self, id: str | None = None):
        super().__init__(id=id)
        self._provider_name: str = ""
        self._model_name: str = ""
        self._session_thread_id: str = ""

    def compose(self) -> ComposeResult:
        """Compose session info section."""
        from textual.widgets import Static

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

    def update_session_info(
        self,
        provider: str,
        model: str,
        thread_id: str,
    ) -> None:
        """Update session information displayed in the context panel.

        Args:
            provider: The LLM provider name (e.g., 'ollama', 'openai').
            model: The model name (e.g., 'llama3.2', 'gpt-4o').
            thread_id: The conversation thread ID for memory persistence.
        """
        self._provider_name = provider
        self._model_name = model
        self._session_thread_id = thread_id

        try:
            session_content = self.query_one("#session-content", Static)
            thread_display = thread_id[:8] if thread_id else "N/A"
            session_content.update(
                f"Provider: {provider}\nModel: {model}\nThread: {thread_display}..."
            )
        except Exception:
            pass
        self.refresh()


class ChatMessage(Markdown):
    """A Markdown-formatted chat message with role-based styling."""

    DEFAULT_CSS = """
    ChatMessage {
        padding: 1 2;
        margin: 1 0;
        border: none;
        max-width: 85%;
    }
    ChatMessage.user {
        background: $primary;
        color: $text;
        align-horizontal: right;
        border: none;
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
        if role == "user":
            formatted = f"> {content}"
        elif role == "ai":
            formatted = content
        else:
            formatted = f"**[{role.upper()}]** {content}"
        super().__init__(formatted)
        self.role = role
        self.add_class(role)


class StatusBar(Static):
    """Status line showing current provider, model, and token count."""

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
        parts = [f"{provider_name} | {model_name}"]

        if context_length > 0 and token_count > 0:
            percentage = int((token_count / context_length) * 100)
            parts.append(f"{token_count:,} tokens ({percentage}%)")
        elif token_count > 0:
            parts.append(f"{token_count:,} tokens")

        if thread_id:
            thread_id_str = str(thread_id)
            parts.append(f"thread: {thread_id_str[:8]}")

        self.update(" | ".join(parts))


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
    """

    def __init__(self):
        super().__init__()
        self.status_bar = StatusBar()
        self.user_input = Input(placeholder="Ask about your data...", id="user_input")

    def compose(self) -> ComposeResult:
        yield self.user_input
        yield self.status_bar


class ChatPanel(ScrollableContainer):
    DEFAULT_CSS = """
    ChatPanel {
        height: 100%;
    }
    #message-container {
        height: 1fr;
    }
    """

    def __init__(self):
        super().__init__()
        self.messages: list[tuple[str, str]] = []
        self._message_container: ScrollableContainer | None = None
        self.sql_panel = SQLPanel()
        self.results_panel = ResultsPanel()
        self.thinking = ThinkingIndicator()

    def compose(self) -> ComposeResult:
        yield ScrollableContainer(id="message-container")
        yield self.thinking
        yield self.sql_panel
        yield self.results_panel

    def on_mount(self) -> None:
        self._message_container = self.query_one(
            "#message-container", ScrollableContainer
        )

    def add_message(self, role: str, content: str):
        self.messages.append((role, content))
        if self._message_container:
            self._message_container.mount(ChatMessage(role, content))

        if len(self.messages) > MAX_MESSAGE_HISTORY:
            self.messages = self.messages[-MAX_MESSAGE_HISTORY:]

    def set_sql(self, sql: str) -> None:
        self.sql_panel.set_sql(sql)

    def set_results(self, results: str) -> None:
        self.results_panel.set_results(results)


class ThinkingIndicator(Collapsible):
    """Collapsible thinking indicator that shows AI reasoning progress."""

    DEFAULT_CSS = """
    ThinkingIndicator {
        padding: 1 2;
        margin: 1 0;
        background: transparent;
        border: none;
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
    """

    def __init__(self):
        super().__init__(title="▶ Thinking...", collapsed=False)
        self.steps: list[str] = []
        self._content_widget: Static | None = None

    def compose(self) -> ComposeResult:
        yield Static("", classes="thinking-steps")

    def on_mount(self) -> None:
        self._content_widget = self.query_one(".thinking-steps", Static)

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


class SQLPanel(Vertical):
    """Panel showing generated SQL and schema context."""

    DEFAULT_CSS = """
    SQLPanel {
        width: 100%;
        height: 1fr;
        border: none;
        padding: 1;
        background: transparent;
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
        if not self.sql_content and not self.schema_tips:
            self.remove_children()
            self.mount(
                Static("No SQL generated yet. Ask a question!", classes="placeholder")
            )
            return

        content_parts = []
        if self.sql_content:
            content_parts.append(f"```sql\n{self.sql_content}\n```")
        if self.schema_tips:
            content_parts.append(f"**Schema:**\n{self.schema_tips}")

        content = "\n\n".join(content_parts)
        self.remove_children()
        self.mount(Markdown(content))


class ResultsPanel(Vertical):
    """Panel showing query results in formatted table."""

    DEFAULT_CSS = """
    ResultsPanel {
        width: 100%;
        height: 1fr;
        border: none;
        padding: 1;
        background: transparent;
    }
    """

    def __init__(self):
        super().__init__()
        self.results_content = ""

    def set_results(self, results: str):
        self.results_content = results
        self.update_display()

    def update_display(self):
        if not self.results_content:
            self.remove_children()
            self.mount(Static("Waiting for results...", classes="placeholder"))
            return

        content = f"**Query Results:**\n\n{self.results_content}"
        self.remove_children()
        self.mount(Markdown(content))


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

    def set_sql(self, sql: str, schema_tips: str = ""):
        self.sql_panel.set_sql(sql, schema_tips)

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
        padding: 0;
    }

    #chat-area {
        width: 75%;
        height: 100%;
        border: none;
        background: $background;
        padding: 0;
    }

    #context-panel {
        width: 25%;
        height: 100%;
        border-left: solid $surface;
        background: $surface;
        padding: 0;
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
        padding: 0;
    }

    #message-container {
        height: 1fr;
        padding: 1;
        scrollbar-gutter: stable;
    }

    .placeholder {
        color: $text-muted;
        text-style: italic;
        text-align: center;
        padding: 2;
    }

    /* ===== Message Bubbles - Clean & Clear ===== */
    ChatMessage {
        padding: 1 2;
        margin: 1 0;
        border: none;
        max-width: 85%;
        background: transparent;
    }

    ChatMessage.user {
        background: $primary-darken-2;
        color: $text;
        align-horizontal: right;
        border: none;
    }

    ChatMessage.ai {
        background: $surface;
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

    /* ===== Context Section - Minimal Borders ===== */
    ContextSection {
        margin: 0;
        border: none;
    }

    ContextSection .header {
        background: transparent;
        padding: 1 1;
        border-bottom: solid $primary-background;
    }

    ContextSection .title {
        color: $text;
        text-style: bold;
    }

    ContextSection .content {
        padding: 1 1;
        background: transparent;
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
    TokenUsageWidget {
        height: auto;
        padding: 1;
        background: transparent;
        border: none;
        margin: 1 0;
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
    
    /* Global Input styling - minimal defaults */
    Footer Input {
        height: 3;
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

        from esdc.configs import Config

        self._provider_name = Config.get_default_provider()
        self._model_name = Config.get_provider_model()

        # Get context length from provider
        provider_config = Config.get_provider_config()
        if provider_config and provider_config.get("model"):
            from esdc.providers import get_provider

            provider_type = provider_config.get("provider_type", "ollama")
            provider = get_provider(provider_type)
            if provider:
                self._context_length = provider.get_context_length(
                    provider_config.get("model", "")
                )

        # Update context panel with session info
        if self._context_panel:
            self._context_panel.update_session_info(
                self._provider_name,
                self._model_name,
                self._thread_id,
            )

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
        from esdc.configs import Config
        from esdc.providers import create_llm_from_config
        from esdc.chat.agent import create_agent
        from esdc.chat.memory import create_checkpointer, create_thread_id

        provider_config = Config.get_provider_config()
        if not provider_config:
            self.display_message(
                "system",
                "Error: No provider configured. Run 'esdc chat --setup' first.",
            )
            return

        self._llm = create_llm_from_config(provider_config)
        self._checkpointer = create_checkpointer()
        self._agent = create_agent(self._llm, checkpointer=self._checkpointer)
        self._thread_id = create_thread_id()

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle user input submission."""
        import asyncio

        user_input = event.value.strip()
        if not user_input:
            return

        self.display_message("user", user_input)
        event.input.value = ""
        self._cancelled = False

        if self.chat_panel and self.chat_panel.thinking:
            self.chat_panel.thinking.remove()

        if not self._agent:
            self.display_message("ai", "Error: Agent not initialized")
            return

        if self.chat_panel:
            self.chat_panel.thinking = ThinkingIndicator()
            self.chat_panel.mount(self.chat_panel.thinking)

        async def run_query():
            async for chunk in self._stream_response(user_input):
                if self._cancelled:
                    self.display_message("system", "Query cancelled.")
                    return
                if chunk["type"] == "message":
                    content = chunk.get("content", "")
                    if content:
                        if self.chat_panel and self.chat_panel.thinking:
                            self.chat_panel.thinking.remove()
                        self.display_message("ai", content)
                elif chunk["type"] == "tool_call":
                    tool_name = chunk.get("tool", "")
                    if self.chat_panel and self.chat_panel.thinking:
                        self.chat_panel.thinking.add_step(f"Running: {tool_name}")
                elif chunk["type"] == "tool_result":
                    result = chunk.get("result", "")
                    if result and self.chat_panel:
                        self.chat_panel.set_sql(chunk.get("sql", ""))
                        self.chat_panel.set_results(result)
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
                                self._thread_id,
                            )

        try:
            await asyncio.wait_for(run_query(), timeout=120.0)
        except asyncio.TimeoutError:
            if self.chat_panel and self.chat_panel.thinking:
                self.chat_panel.thinking.remove()
            self.display_message(
                "ai", "Request timed out after 2 minutes. Please try again."
            )
        except Exception as e:
            if self.chat_panel and self.chat_panel.thinking:
                self.chat_panel.thinking.remove()
            self.display_message("ai", f"Error: {str(e)}")

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
            if chunk["type"] == "message":
                content = chunk.get("content", "")
                if content:
                    yield {"type": "message", "content": content}
            elif chunk["type"] == "tool_call":
                yield {"type": "tool_call", "tool": chunk.get("tool", "")}
            elif chunk["type"] == "tool_result":
                result = chunk.get("result", "")
                sql = chunk.get("sql", "")
                yield {"type": "tool_result", "result": result, "sql": sql}

    def display_message(self, role: str, content: str) -> None:
        if self.chat_panel:
            self.chat_panel.add_message(role, content)
            self._message_count += 1

    def action_cancel_query(self) -> None:
        """Cancel the current query."""
        self._cancelled = True
        if self.chat_panel and self.chat_panel.thinking:
            self.chat_panel.thinking.remove()
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
            all_expanded = all(
                v
                for k, v in self._context_panel._section_expanded.items()
                if k not in ("session_info", "token_usage", "tools")
            )

            for section in self._context_panel._section_expanded:
                if section not in ("session_info", "token_usage", "tools"):
                    self._context_panel._section_expanded[section] = not all_expanded

            self._context_panel.refresh()
            self.notify(f"All sections {'collapsed' if all_expanded else 'expanded'}")

    def action_save_screenshot(self, filename: str | None = None) -> None:
        """Save screenshot of the current screen.

        Args:
            filename: Optional filename for the screenshot. Defaults to timestamped filename.
        """
        from datetime import datetime
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
