"""Shared role-based text transport for corpus metadata and cleanup tasks.

Runtime never parses ``main``, ``provider:<name>`` or a bare Ollama name here.
Those legacy meanings belong only to the one-time converter; this module reads
the resolved canonical assignment for the role.
"""

import logging
from collections.abc import Callable
from typing import Literal

from esdc.configs import Config
from esdc.model_runtime import ResolvedTask
from esdc.model_settings import ModelConfigurationRequired
from esdc.providers import create_llm_from_resolved_task

logger = logging.getLogger(__name__)

CorpusTextRole = Literal["metadata", "cleanup"]


def resolve_text_task(role: CorpusTextRole) -> ResolvedTask | None:
    """Resolve one corpus text role, returning None on an old/invalid config."""
    try:
        return Config.resolve_model_task(role)
    except ModelConfigurationRequired as exc:
        logger.warning("[Corpus] %s", exc)
        return None


def task_target_label(task: ResolvedTask) -> str:
    """Short, credential-free label for warnings and errors."""
    if task.primary is not None:
        return (
            f"connection {task.primary.connection!r} model "
            f"{task.primary.model!r}"
        )
    return f"mode {task.mode!r}"


def text_caller_for_resolved_task(
    task: ResolvedTask | None,
) -> Callable[[str], str] | None:
    """Build a prompt->text callable for a resolved model/chat task, or None."""
    if task is None or task.mode not in {"model", "chat"} or task.primary is None:
        return None

    try:
        llm = create_llm_from_resolved_task(task)
    except Exception as exc:  # best effort: a failed model never aborts a batch
        logger.warning(
            "[Corpus] %s task caller unavailable (%s): %s",
            task.role,
            task_target_label(task),
            exc,
        )
        return None

    def call(prompt: str) -> str:
        return str(llm.invoke(prompt).content)

    return call


def text_caller_for_task(role: CorpusTextRole) -> Callable[[str], str] | None:
    """Resolve a corpus text role and build its shared transport caller."""
    return text_caller_for_resolved_task(resolve_text_task(role))
