"""Shared streaming utilities for SSE responses."""

from collections.abc import Generator


def chunk_text(text: str, chunk_size: int = 3) -> Generator[str, None, None]:
    """Split text into small character-level chunks for streaming.

    This ensures markdown tokens (##, **, |, etc.) are never split
    across SSE chunks, preventing rendering issues in clients.

    Args:
        text: Text to chunk
        chunk_size: Characters per chunk (default 3 for markdown safety)

    Yields:
        Text chunks of approximately chunk_size characters

    Example:
        >>> list(chunk_text("## Header", chunk_size=3))
        ['## ', 'Hea', 'der']
    """
    for i in range(0, len(text), chunk_size):
        yield text[i : i + chunk_size]


def chunk_json(json_str: str, chunk_size: int = 10) -> Generator[str, None, None]:
    """Split JSON into chunks for streaming function call arguments.

    Args:
        json_str: JSON string to chunk
        chunk_size: Characters per chunk

    Yields:
        JSON string chunks
    """
    for i in range(0, len(json_str), chunk_size):
        yield json_str[i : i + chunk_size]
