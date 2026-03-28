# Standard library
import uuid

# Third-party
from langgraph.checkpoint.memory import MemorySaver


def create_checkpointer() -> MemorySaver:
    """Create an in-memory checkpointer for conversation persistence.

    This creates a new MemorySaver instance. For persistent conversations
    across sessions, you'd want to use a persistent checkpointer like
    SQLiteSaver or PostgreSQLSaver.
    """
    return MemorySaver()


def create_thread_id() -> str:
    """Create a new unique thread ID for a conversation session."""
    return f"esdc-{uuid.uuid4().hex[:8]}"


def get_thread_config(thread_id: str) -> dict:
    """Get the config dict for a thread ID.

    Args:
        thread_id: The thread/conversation ID

    Returns:
        Config dict for LangGraph
    """
    return {
        "configurable": {
            "thread_id": thread_id,
            "checkpoint_ns": "esdc_chat",
        }
    }
