# esdc/chat/app.py
import asyncio
from typing import Any, AsyncGenerator

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
from textual.widgets import Static, Input, Markdown, Button, Collapsible
from textual.events import Key

MAX_MESSAGE_HISTORY = 100


class ChatMessage(Markdown):
    """A Markdown-formatted chat message with role-based styling."""

    DEFAULT_CSS = """
    ChatMessage {
        padding: 1;
        margin: 1 0;
    }
    ChatMessage.user {
        color: $text;
    }
    ChatMessage.ai {
        color: $text;
    }
    ChatMessage.system {
        color: $warning;
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
    """Status line showing current provider and model."""

    DEFAULT_CSS = """
    StatusBar {
        height: 1;
        padding: 0 1;
        color: $text-muted;
        background: $surface;
    }
    """

    def __init__(self):
        super().__init__("Loading...")

    def set_status(self, provider_name: str, model_name: str) -> None:
        self.update(f"{provider_name} | {model_name}")


class Footer(Vertical):
    """Footer container for status bar and input field."""

    DEFAULT_CSS = """
    Footer {
        height: auto;
    }
    StatusBar {
        height: 1;
    }
    Input {
        height: 3;
    }
    """

    def __init__(self):
        super().__init__()
        self.status_bar = StatusBar()
        self.user_input = Input(placeholder="Ask about your data...", id="user_input")

    def compose(self) -> ComposeResult:
        yield self.user_input
        yield self.status_bar


class ChatPanel(Vertical):
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
        self._footer: Footer | None = None

    def compose(self) -> ComposeResult:
        yield ScrollableContainer(id="message-container")
        yield Footer()

    def on_mount(self) -> None:
        self._message_container = self.query_one(
            "#message-container", ScrollableContainer
        )
        self._footer = self.query_one(Footer)

    def add_message(self, role: str, content: str):
        self.messages.append((role, content))
        if self._message_container:
            self._message_container.mount(ChatMessage(role, content))

        if len(self.messages) > MAX_MESSAGE_HISTORY:
            self.messages = self.messages[-MAX_MESSAGE_HISTORY:]


class ThinkingIndicator(Collapsible):
    """Collapsible thinking indicator that shows AI reasoning progress."""

    def __init__(self):
        super().__init__(title="▶ Thinking...", collapsed=True)
        self.steps: list[str] = []
        self._content_widget: Static | None = None

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
        border: solid $accent;
        padding: 1;
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
        border: solid $primary;
        padding: 1;
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
    Screen {
        layout: horizontal;
    }

    #left-panel {
        width: 60%;
        border: solid $success;
    }

    #right-panel {
        width: 40%;
        height: 100%;
        border: solid $accent;
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
        Binding("ctrl+b", "toggle_split", "Toggle Split"),
        Binding("ctrl+shift+c", "copy_results", "Copy Results"),
        Binding("ctrl+s", "screenshot", "Save Screenshot"),
    ]

    def __init__(self):
        super().__init__()
        self.chat_panel: ChatPanel | None = None
        self._right_panel: RightPanel | None = None
        self._thinking: ThinkingIndicator | None = None
        self.status_bar: StatusBar | None = None
        self.user_input: Input | None = None
        self._agent: Any = None
        self._llm: Any = None
        self._thread_id: str = "esdc-default"
        self._message_count: int = 0

    def compose(self) -> ComposeResult:
        yield Horizontal(
            Vertical(id="left-panel"),
            Vertical(id="right-panel"),
        )

    def on_mount(self) -> None:
        left_container = self.query_one("#left-panel", Vertical)
        right_container = self.query_one("#right-panel", Vertical)

        self.chat_panel = ChatPanel()
        right_panel = RightPanel()

        left_container.mount(self.chat_panel)
        right_container.mount(right_panel)

        self._right_panel = right_panel

        self.call_later(self._setup_footer)

    def _setup_footer(self) -> None:
        footer = self.chat_panel.query_one(Footer)
        self.user_input = footer.user_input
        self.status_bar = footer.status_bar

        from esdc.configs import Config

        provider_name = Config.get_default_provider()
        model_name = Config.get_provider_model()
        self.status_bar.set_status(provider_name, model_name)

        self._init_agent()

    def _init_agent(self) -> None:
        """Initialize the LLM and agent."""
        from esdc.configs import Config
        from esdc.providers import create_llm_from_config
        from esdc.chat.agent import create_agent
        from esdc.chat.memory import create_thread_id

        provider_config = Config.get_provider_config()
        if not provider_config:
            self.display_message(
                "system",
                "Error: No provider configured. Run 'esdc chat --setup' first.",
            )
            return

        self._llm = create_llm_from_config(provider_config)
        self._agent = create_agent(self._llm)
        self._thread_id = create_thread_id()

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle user input submission."""
        user_input = event.value.strip()
        if not user_input:
            return

        self.display_message("user", user_input)
        event.input.value = ""

        if self._right_panel:
            self._right_panel.set_sql("", "")
            self._right_panel.set_results("")

        if self._thinking:
            self._thinking.remove()
            self._thinking = None

        if not self._agent:
            self.display_message("ai", "Error: Agent not initialized")
            return

        self._thinking = ThinkingIndicator()
        if self.chat_panel:
            self.chat_panel.mount(self._thinking)

        try:
            async for chunk in self._stream_response(user_input):
                if chunk["type"] == "message":
                    content = chunk.get("content", "")
                    if content:
                        if self._thinking:
                            self._thinking.remove()
                            self._thinking = None
                        self.display_message("ai", content)
                elif chunk["type"] == "tool_call":
                    tool_name = chunk.get("tool", "")
                    if self._thinking:
                        self._thinking.add_step(f"Running: {tool_name}")
                elif chunk["type"] == "tool_result":
                    result = chunk.get("result", "")
                    if result:
                        self.set_results(chunk.get("sql", ""), result[:500])
        except Exception as e:
            if self._thinking:
                self._thinking.remove()
                self._thinking = None
            self.display_message("ai", f"Error: {str(e)}")

    async def _stream_response(
        self, user_input: str
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Stream response from the agent."""
        from esdc.chat.agent import run_agent_stream

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

    def set_results(self, sql: str, results: str) -> None:
        if self._right_panel:
            self._right_panel.set_sql(sql, "")
            self._right_panel.set_results(results[:500])

    def action_toggle_split(self) -> None:
        """Toggle between 60/40 and 50/50 split."""
        left = self.query_one("#left-panel", Vertical)
        right = self.query_one("#right-panel", Vertical)

        if left.styles.width == "60%":
            left.styles.width = "50%"
            right.styles.width = "50%"
        else:
            left.styles.width = "60%"
            right.styles.width = "40%"

    def action_copy_results(self) -> None:
        """Copy current results to clipboard."""
        if self._right_panel and self._right_panel.results_panel:
            results = self._right_panel.results_panel.results_content
            if results:
                self.copy_to_clipboard(results)
                self.notify("Results copied to clipboard")
