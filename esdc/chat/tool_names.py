"""Translate human-readable tool names to provider-safe wire names.

ESDC registers tools with display names such as ``"SQL Executor"``. Those
names are convenient in the UI, but OpenAI-style function names must match
``^[a-zA-Z0-9_-]{1,64}$``. Some OpenAI-compatible backends enforce that rule
strictly: vLLM's Hermes tool-call parser, for example, silently leaves a
``<tool_call><function=SQL Executor>`` block in the message content instead of
returning a parsed tool call, so the model never actually calls the tool.

The agent therefore binds sanitized names to the model and translates the
model's calls back to display names before executing a tool or returning a
tool call to a client such as OpenWebUI. This keeps the client-visible name
correct while every provider receives a wire-safe name.
"""

from __future__ import annotations

import copy
import re
from collections.abc import Iterable
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.tools import BaseTool

MAX_TOOL_NAME_LENGTH = 64
_INVALID_NAME_CHARS = re.compile(r"[^A-Za-z0-9_-]+")
_NON_ALNUM_CHARS = re.compile(r"[^a-z0-9]+")
_LOWER_OR_DIGIT_BEFORE_UPPER = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_ACRONYM_BEFORE_WORD = re.compile(r"(?<=[A-Z])(?=[A-Z][a-z])")


def _normalize_tool_name(name: str) -> str:
    """Fold case and separators so prompt variants still resolve."""
    return _NON_ALNUM_CHARS.sub("", name.lower())


def sanitize_tool_name(name: str) -> str:
    """Return an OpenAI-compatible snake_case function name for ``name``.

    CamelCase boundaries become underscores, invalid characters are collapsed
    into underscores, and the result is lowercased and capped at the OpenAI
    limit of 64 characters. An empty name becomes ``"tool"``.
    """
    name = _LOWER_OR_DIGIT_BEFORE_UPPER.sub("_", name.strip())
    name = _ACRONYM_BEFORE_WORD.sub("_", name)
    sanitized = _INVALID_NAME_CHARS.sub("_", name).strip("_").lower()
    return (sanitized or "tool")[:MAX_TOOL_NAME_LENGTH]


class ToolNameMap:
    """Bidirectional display-name/wire-name map with collision handling.

    A tool's display name is what ESDC and its clients use. Its wire name is
    what the model provider sees. Unknown names pass through unchanged, so
    providers or test doubles that already return display names keep working.
    """

    def __init__(self, display_names: Iterable[str] = ()) -> None:
        self._display_to_wire: dict[str, str] = {}
        self._wire_to_display: dict[str, str] = {}
        self._normalized_to_display: dict[str, str] = {}
        for display_name in display_names:
            self.add(display_name)

    def add(self, display_name: str) -> str:
        """Register ``display_name`` and return its wire name."""
        if display_name in self._display_to_wire:
            return self._display_to_wire[display_name]

        wire_name = sanitize_tool_name(display_name)
        if wire_name in self._wire_to_display:
            base = wire_name
            suffix_index = 2
            while True:
                suffix = f"_{suffix_index}"
                candidate = f"{base[: MAX_TOOL_NAME_LENGTH - len(suffix)]}{suffix}"
                if candidate not in self._wire_to_display:
                    wire_name = candidate
                    break
                suffix_index += 1

        self._display_to_wire[display_name] = wire_name
        self._wire_to_display[wire_name] = display_name
        self._normalized_to_display.setdefault(
            _normalize_tool_name(display_name), display_name
        )
        self._normalized_to_display.setdefault(
            _normalize_tool_name(wire_name), display_name
        )
        return wire_name

    def to_wire(self, name: str) -> str:
        """Return the provider-facing name for a display name.

        Exact display names win. Case/separator variants (for example a
        snake_case name quoted by the prompt) resolve to the same tool.
        """
        if name in self._display_to_wire:
            return self._display_to_wire[name]
        display_name = self._normalized_to_display.get(_normalize_tool_name(name))
        if display_name is not None:
            return self._display_to_wire[display_name]
        return name

    def to_display(self, name: str) -> str:
        """Return the display name for a provider-facing name.

        Exact wire names win, then case/separator variants. Unknown names pass
        through unchanged so test doubles and lenient providers keep working.
        """
        if name in self._wire_to_display:
            return self._wire_to_display[name]
        if name in self._display_to_wire:
            return name
        display_name = self._normalized_to_display.get(_normalize_tool_name(name))
        return display_name if display_name is not None else name

    def wire_tools(self, tools: Iterable[BaseTool]) -> list[BaseTool]:
        """Return copies of ``tools`` whose names are wire-safe."""
        wire_tools: list[BaseTool] = []
        for tool in tools:
            wire_name = self.to_wire(tool.name)
            if wire_name == tool.name:
                wire_tools.append(tool)
            elif hasattr(tool, "model_copy"):
                wire_tools.append(tool.model_copy(update={"name": wire_name}))
            else:
                # Test doubles and other non-pydantic tool-like objects.
                clone = copy.copy(tool)
                clone.name = wire_name
                wire_tools.append(clone)
        return wire_tools

    def messages_to_wire(self, messages: Iterable[BaseMessage]) -> list[BaseMessage]:
        """Return message copies with tool-call names translated for the model."""
        return [self._message_to_wire(message) for message in messages]

    def message_from_wire(self, message: BaseMessage) -> BaseMessage:
        """Return a message copy with provider tool-call names translated back."""
        if not isinstance(message, AIMessage) or not message.tool_calls:
            return message

        tool_calls: list[Any] = [
            {**tool_call, "name": self.to_display(tool_call.get("name", ""))}
            for tool_call in message.tool_calls
        ]
        return message.model_copy(update={"tool_calls": tool_calls})

    def _message_to_wire(self, message: BaseMessage) -> BaseMessage:
        if isinstance(message, AIMessage) and message.tool_calls:
            tool_calls: list[Any] = [
                {**tool_call, "name": self.to_wire(tool_call.get("name", ""))}
                for tool_call in message.tool_calls
            ]
            message = message.model_copy(update={"tool_calls": tool_calls})

        if isinstance(message, ToolMessage) and message.name:
            wire_name = self.to_wire(message.name)
            if wire_name != message.name:
                message = message.model_copy(update={"name": wire_name})

        return message
