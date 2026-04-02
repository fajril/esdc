"""Parser for extracting thinking content from model responses."""

# Standard library
import re


# Thinking tag patterns supported by OpenWebUI
# Reference: https://docs.openwebui.com/features/chat-conversations/chat-features/reasoning-models
THINKING_PATTERNS = [
    # Standard think tags
    (r"<think>", r"</think>"),
    (r"<thinking>", r"</thinking>"),
    # Reason tags
    (r"<reason>", r"</reason>"),
    (r"<reasoning>", r"</reasoning>"),
    # Thought tags
    (r"<thought>", r"</thought>"),
    # Special tokens (e.g., Qwen, DeepSeek variants)
    (r"<\|begin_of_thought\|>", r"<\|end_of_thought\|>"),
]


def has_thinking_tags(text: str) -> bool:
    """Check if text contains any thinking tags.

    Args:
        text: The text to check

    Returns:
        True if text contains thinking tags, False otherwise
    """
    if not text:
        return False

    for open_tag, close_tag in THINKING_PATTERNS:
        if re.search(open_tag, text, re.DOTALL | re.IGNORECASE):
            return True

    return False


def extract_thinking_content(text: str) -> tuple[str | None, str]:
    """Extract thinking content from text with thinking tags.

    Args:
        text: The text containing thinking tags

    Returns:
        Tuple of (thinking_content, final_response)
        thinking_content is None if no thinking tags found
    """
    if not text:
        return None, ""

    thinking_contents = []
    final_text = text

    # Try each pattern
    for open_tag, close_tag in THINKING_PATTERNS:
        # Find all occurrences
        pattern = f"({open_tag})(.*?)({close_tag})"
        matches = list(re.finditer(pattern, final_text, re.DOTALL | re.IGNORECASE))

        for match in matches:
            # Extract thinking content (group 2 is the content between tags)
            thinking = match.group(2).strip()
            if thinking:
                thinking_contents.append(thinking)

            # Remove the thinking block from final text
            final_text = final_text.replace(match.group(0), "")

    # Clean up final text
    final_text = final_text.strip()

    # Remove extra newlines
    final_text = re.sub(r"\n{3,}", "\n\n", final_text)

    if thinking_contents:
        combined_thinking = "\n\n".join(thinking_contents)
        return combined_thinking, final_text

    return None, final_text


def format_thinking_section(content: str) -> str:
    """Format thinking content with OpenWebUI compatible tags.

    Args:
        content: The thinking content to format

    Returns:
        Formatted string with think tags
    """
    if not content:
        return ""

    return f"""<think>
{content}
</think>"""


def process_response_with_thinking(text: str) -> dict[str, str | None]:
    """Process a response and extract thinking if present.

    Args:
        text: The response text

    Returns:
        Dict with 'thinking' and 'response' keys
        'thinking' is None if no thinking found
    """
    if not text:
        return {"thinking": None, "response": ""}

    thinking, final = extract_thinking_content(text)

    return {
        "thinking": thinking,
        "response": final if final else text,
    }
