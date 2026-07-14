# esdc/chat/app.py

# Standard library
import asyncio
import json
import logging
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
from textual.app import App, ComposeResult  # noqa: E402
from textual.binding import Binding  # noqa: E402
from textual.containers import Horizontal  # noqa: E402
from textual.widgets import Static, TextArea  # noqa: E402

from esdc.chat.widgets import (  # noqa: F401,E402  (re-exported for tests/back-compat)
    ChatInput,
    ChatMessage,
    ChatPanel,
    ContextPanel,
    ContextSection,
    ContextUsageWidget,
    ConversationTitle,
    Footer,
    QueryHistory,
    ResultsPanel,
    SQLPanel,
    StatusBar,
    ThinkingIndicator,
)

MAX_QUERY_HISTORY = 5
TOOLS_LIST = ["execute_sql", "get_schema", "list_tables"]


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
        """Initialize the tool status list widget."""
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


class ESDCChatApp(App):
    """Main ESDC chat application."""

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
        """Initialize the ESDC chat application."""
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
        self._provider_type: str = ""
        self._model_name: str = ""
        self._base_url: str = ""
        self._system_prompt: str = ""
        self._context_panel_visible: bool = True
        self._context_metadata: dict | None = None

        # Queue-based streaming infrastructure
        self._event_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._streaming_message: ChatMessage | None = None
        self._accumulated_content: str = ""
        self._render_dirty: bool = False
        self._conversation_title: str = ""
        self._title_generated: bool = False

    def compose(self) -> ComposeResult:
        """Compose the main application layout."""
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
        """Handle application mount event."""
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
        self._system_prompt = ""

        # Get context length from provider
        provider_config = Config.get_provider_config()
        if provider_config and provider_config.get("model"):
            from esdc.providers import get_provider

            self._provider_type = provider_config.get("provider_type", "ollama")
            self._base_url = provider_config.get("base_url", "")
            provider = get_provider(self._provider_type)
            if provider:
                # Use static fallback; dynamic resolution from the provider API
                # happens later inside the agent pipeline when the LLM instance
                # is already available (see _detect_context_length / agent_node).
                self._context_length = provider.get_context_length(
                    provider_config.get("model", "")
                )
        try:
            from esdc.chat.prompts import get_system_prompt

            self._system_prompt = get_system_prompt()
        except Exception as exc:
            logger.debug("Failed to load system prompt for token panel: %s", exc)

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
        self.set_interval(0.1, self._flush_stream_render)

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

        try:
            from esdc.phoenix import setup_phoenix_tracing

            setup_phoenix_tracing()
        except ImportError:
            logger.debug("Phoenix dependencies not available, skipping tracing")

        provider_config = Config.get_provider_config()
        if not provider_config:
            self.display_message(
                "system",
                "Error: No provider configured. Run 'esdc configs' first.",
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

    def _consume_events(self) -> None:
        """Consume events from queue in main thread (called by timer every 50ms)."""
        try:
            # Process up to 10 events per tick to avoid blocking
            for _ in range(10):
                try:
                    chunk = self._event_queue.get_nowait()
                    self._handle_stream_chunk(chunk)
                except asyncio.QueueEmpty:
                    break
        except Exception as e:
            logger.error(f"Error consuming events: {e}")

    def _flush_stream_render(self) -> None:
        """Render accumulated stream content at most 10x/sec."""
        if not self._render_dirty or not self._streaming_message:
            return
        self._render_dirty = False
        self._streaming_message.update(self._accumulated_content)
        if self.chat_panel:
            self.chat_panel.scroll_end(animate=False)

    def _handle_stream_chunk(self, chunk: dict[str, Any]) -> None:
        """Process a single chunk and update UI."""
        chunk_type = chunk.get("type", "unknown")

        if chunk_type == "token":
            token = chunk.get("content", "")
            if token and self._streaming_message:
                self._accumulated_content += token
                self._render_dirty = True

        elif chunk_type == "message":
            content = chunk.get("content", "")
            if content and not self._accumulated_content:
                self._accumulated_content = content
                if self._streaming_message:
                    self._render_dirty = True

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
                self._render_dirty = True

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

                self._token_count = estimate_tokens(
                    messages,
                    system_prompt=self._system_prompt,
                    provider_type=self._provider_type,
                    model=self._model_name,
                    base_url=self._base_url,
                )
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

            # Force a final render of any pending accumulated content before
            # resetting streaming state.
            self._flush_stream_render()

            if not success and error and self._streaming_message:
                self._streaming_message.update(f"Error: {error}")

            # Final scroll to bottom after streaming completes (critical fix)
            if self.chat_panel:
                self.chat_panel.scroll_end(animate=False)

            # Reset state
            self._streaming_message = None
            self._accumulated_content = ""

    async def _stream_response(
        self, user_input: str
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Adapt shared astream_agent_events into UI chunks."""
        from langchain_core.messages import HumanMessage, ToolMessage
        from langchain_core.runnables import RunnableConfig

        import esdc.chat.event_streamer as event_streamer

        if not self._agent:
            return

        config = RunnableConfig(
            configurable={
                "thread_id": self._thread_id,
                "checkpoint_ns": "esdc_chat",
            },
            recursion_limit=event_streamer.DEFAULT_RECURSION_LIMIT,
        )

        tracked_messages: list[Any] = [HumanMessage(content=user_input)]
        pending_sql: dict[str, str] = {}

        async for event in event_streamer.astream_agent_events(
            self._agent, [HumanMessage(content=user_input)], config=config
        ):
            etype = event.get("type")

            if etype == "token":
                yield {"type": "token", "content": event.get("content", "")}

            elif etype == "reasoning_token":
                yield {
                    "type": "reasoning_token",
                    "content": event.get("content", ""),
                }

            elif etype == "message_complete":
                ai_message = event.get("ai_message")
                if ai_message is None:
                    continue
                tracked_messages.append(ai_message)
                tool_calls = getattr(ai_message, "tool_calls", None) or []
                for tc in tool_calls:
                    args = tc.get("args", {}) or {}
                    if tc.get("name") == "execute_sql" and isinstance(args, dict):
                        pending_sql[tc.get("id") or ""] = args.get("query", "")
                    yield {
                        "type": "tool_call",
                        "tool": tc.get("name", ""),
                        "args": args,
                    }
                if not tool_calls:
                    content = str(ai_message.content or "")
                    if content:
                        yield {"type": "message", "content": content}
                yield {
                    "type": "messages_state",
                    "messages": list(tracked_messages),
                    "message_count": len(tracked_messages),
                }

            elif etype == "tool_result":
                tool_call_id = event.get("tool_call_id") or ""
                result = str(event.get("result", ""))
                tracked_messages.append(
                    ToolMessage(content=result, tool_call_id=tool_call_id)
                )
                yield {
                    "type": "tool_result",
                    "tool": event.get("tool_name", ""),
                    "result": result,
                    "sql": pending_sql.pop(tool_call_id, ""),
                }

            elif etype == "context_metadata":
                yield {
                    "type": "context_metadata",
                    "metadata": event.get("metadata"),
                }

            elif etype == "recursion_error":
                yield {
                    "type": "message",
                    "content": (
                        "The agent hit its step limit for this question: "
                        f"{event.get('message', 'recursion limit exceeded')}. "
                        "Try a more specific question."
                    ),
                }

    def display_message(self, role: str, content: str) -> None:
        """Display a message in the chat panel."""
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
