"""Streaming buffer for accumulating and formatting agent events."""

# Standard library
import json
import time


def format_thinking_section(content: str) -> str:
    """Format thinking/reasoning section in markdown."""
    if not content or not content.strip():
        return ""
    return f"""### 🧠 Thinking Process

{content}

"""


def format_tool_section(tool_name: str, tool_args: dict) -> str:
    """Format tool call section in markdown with syntax highlighting."""
    if tool_name == "execute_sql":
        sql = tool_args.get("query", "")
        return f"""### 🛠️ Tool: {tool_name}

```sql
{sql}
```

"""
    else:
        args_str = json.dumps(tool_args, indent=2, ensure_ascii=False)
        return f"""### 🛠️ Tool: {tool_name}

```json
{args_str}
```

"""


class StreamingBuffer:
    """Buffer for accumulating events and flushing at checkpoints.

    This class accumulates agent events (thinking, tool calls, content)
    and flushes them as formatted markdown sections. Flushing occurs on:
    - Tool call detected (immediate flush)
    - Buffer exceeds 500 characters
    - Timeout of 500ms exceeded
    - Final flush at end of stream

    Attributes:
        thinking_buffer: List of thinking content strings
        tool_calls_buffer: List of tool call dictionaries
        content_buffer: List of content strings
        last_flush_time: Timestamp of last flush
        buffer_size_limit: Maximum characters before auto-flush
        timeout_ms: Maximum milliseconds between flushes
    """

    def __init__(self, buffer_size_limit: int = 500, timeout_ms: int = 500):
        """Initialize streaming buffer.

        Args:
            buffer_size_limit: Maximum characters before auto-flush
            timeout_ms: Maximum milliseconds between flushes
        """
        self.thinking_buffer: list[str] = []
        self.tool_calls_buffer: list[dict] = []
        self.content_buffer: list[str] = []
        self.last_flush_time: float = time.time()
        self.buffer_size_limit: int = buffer_size_limit
        self.timeout_ms: int = timeout_ms

    def add_thinking(self, content: str) -> bool:
        """Add thinking content to buffer.

        Args:
            content: Thinking/reasoning text

        Returns:
            True if flush is recommended (buffer full or timeout)
        """
        self.thinking_buffer.append(content)
        return self._should_flush()

    def add_tool_call(self, tool_name: str, args: dict) -> bool:
        """Add tool call to buffer.

        Args:
            tool_name: Name of the tool
            args: Tool arguments

        Returns:
            True (always flush on tool call)
        """
        self.tool_calls_buffer.append({"name": tool_name, "args": args})
        return True  # Always flush on tool call

    def add_content(self, content: str) -> bool:
        """Add final content to buffer.

        Args:
            content: Response content text

        Returns:
            True if flush is recommended (buffer full or timeout)
        """
        self.content_buffer.append(content)
        return self._should_flush()

    def _should_flush(self) -> bool:
        """Check if buffer should be flushed.

        Returns:
            True if buffer exceeds size limit or timeout
        """
        total_chars = sum(len(s) for s in self.thinking_buffer + self.content_buffer)

        if total_chars > self.buffer_size_limit:
            return True

        elapsed_ms = (time.time() - self.last_flush_time) * 1000
        if elapsed_ms > self.timeout_ms:
            return True

        return False

    def flush(self) -> str:
        """Build markdown from buffers and return.

        Returns:
            Formatted markdown string from accumulated buffers
        """
        sections: list[str] = []

        # Build thinking section
        if self.thinking_buffer:
            thinking = "".join(self.thinking_buffer)
            sections.append(format_thinking_section(thinking))
            self.thinking_buffer = []

        # Build tool sections
        for tool in self.tool_calls_buffer:
            sections.append(format_tool_section(tool["name"], tool["args"]))
        self.tool_calls_buffer = []

        # Build content
        if self.content_buffer:
            content = "".join(self.content_buffer)
            sections.append(content)
            self.content_buffer = []

        self.last_flush_time = time.time()
        return "\n".join(sections) if sections else ""

    def flush_final(self) -> str:
        """Flush remaining buffers at end of stream.

        Returns:
            Formatted markdown string from any remaining content
        """
        return self.flush()

    def has_content(self) -> bool:
        """Check if buffer has any content.

        Returns:
            True if any buffer is not empty
        """
        return bool(
            self.thinking_buffer or self.tool_calls_buffer or self.content_buffer
        )
