"""State management for preserving thinking content across tool calls."""

# Third-party
from langchain_core.messages import AIMessage

# Local
from esdc.server.thinking_parser import extract_thinking_content


class ThinkingState:
    """Manages preservation and retrieval of thinking content in LLM conversations.

    This class handles the interleaved thinking scenario where an AIMessage
    contains both thinking content and tool calls. The thinking content is
    extracted and preserved to be displayed with the final response.

    Attributes:
        _thinking_parts: List of thinking content strings accumulated
    """

    def __init__(self) -> None:
        """Initialize empty thinking state."""
        self._thinking_parts: list[str] = []

    def preserve_thinking(self, thinking: str) -> None:
        """Store thinking content for later retrieval.

        Args:
            thinking: The thinking content to preserve
        """
        if thinking and thinking.strip():
            self._thinking_parts.append(thinking.strip())

    def has_thinking(self) -> bool:
        """Check if thinking content exists.

        Returns:
            True if any thinking content has been preserved
        """
        return len(self._thinking_parts) > 0

    def get_thinking(self) -> str | None:
        """Retrieve and clear preserved thinking.

        Returns:
            Concatenated thinking content joined by "\n\n", or None if empty
        """
        if not self._thinking_parts:
            return None

        thinking = "\n\n".join(self._thinking_parts)
        self._thinking_parts = []
        return thinking if thinking else None

    def peek_thinking(self) -> str | None:
        """View thinking content without clearing.

        Returns:
            Concatenated thinking content joined by "\n\n", or None if empty
        """
        if not self._thinking_parts:
            return None

        thinking = "\n\n".join(self._thinking_parts)
        return thinking if thinking else None

    def clear(self) -> None:
        """Clear all preserved thinking content."""
        self._thinking_parts = []

    def extract_and_preserve(self, message: AIMessage) -> bool:
        """Extract thinking from AIMessage when tool calls are present.

        This method only extracts thinking when the message has tool_calls
        (interleaved scenario). It first checks for reasoning_content in
        additional_kwargs, then falls back to parsing thinking tags from content.

        Args:
            message: The AIMessage to extract thinking from

        Returns:
            True if thinking was extracted and preserved, False otherwise
        """
        # Only extract thinking when tool_calls are present (interleaved scenario)
        if not hasattr(message, "tool_calls") or not message.tool_calls:
            return False

        thinking_content = None

        # Check for reasoning_content in additional_kwargs first
        if hasattr(message, "additional_kwargs") and message.additional_kwargs:
            thinking_content = message.additional_kwargs.get("reasoning_content")

        # Fall back to parsing thinking tags from content
        if not thinking_content and message.content:
            content_str = str(message.content)
            thinking, _ = extract_thinking_content(content_str)
            if thinking:
                thinking_content = thinking

        # Preserve the thinking content if found
        if thinking_content and thinking_content.strip():
            self.preserve_thinking(thinking_content)
            return True

        return False
