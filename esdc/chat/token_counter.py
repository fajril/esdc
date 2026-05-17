# esdc/chat/token_counter.py

# Standard library
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

# Third-party
from langchain_core.messages import AIMessage, AnyMessage

TOKEN_CHARS_PER_TOKEN = 4

TokenCountSource = Literal[
    "provider_usage",
    "tiktoken",
    "llama_cpp_tokenize",
    "heuristic",
]
TokenCountConfidence = Literal["exact", "estimated"]

_OPENAI_FAMILY_PREFIXES = ("gpt-", "o1", "o3", "o4")
_OPENAI_FAMILY_MARKERS = ("openai/", "chatgpt")


@dataclass(frozen=True)
class TokenUsage:
    """Normalized token usage across providers."""

    input_tokens: int
    output_tokens: int
    total_tokens: int
    source: TokenCountSource
    confidence: TokenCountConfidence

    def to_dict(self) -> dict[str, int | str]:
        """Return a serializable normalized usage payload."""
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "source": self.source,
            "confidence": self.confidence,
        }


def _as_int(value: Any) -> int | None:
    """Best-effort integer conversion for provider usage fields."""
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _content_to_text(content: Any) -> str:
    """Extract countable text from LangChain/OpenAI-style content."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if "text" in item:
                    parts.append(str(item["text"]))
                elif item.get("type") in {"image_url", "image", "input_image"}:
                    # Image token costs are provider-specific; v1 only counts text.
                    continue
                else:
                    parts.append(str(item))
            else:
                parts.append(str(item))
        return " ".join(parts)
    return str(content)


def _is_openai_tokenizer_candidate(
    provider_type: str | None,
    model: str | None,
) -> bool:
    """Return true when tiktoken is appropriate for preflight counting."""
    provider = (provider_type or "").lower()
    model_lower = (model or "").lower()

    if provider in {"openai", "azure_openai"}:
        return True

    if provider == "openai_compatible":
        return model_lower.startswith(_OPENAI_FAMILY_PREFIXES) or any(
            marker in model_lower for marker in _OPENAI_FAMILY_MARKERS
        )

    return False


def _tiktoken_count(text: str, model: str | None) -> int | None:
    """Count text with tiktoken, returning None if it cannot be used."""
    try:
        import tiktoken
    except ImportError:
        return None

    try:
        encoding = tiktoken.encoding_for_model(model or "")
    except Exception:
        try:
            encoding = tiktoken.get_encoding("o200k_base")
        except Exception:
            encoding = tiktoken.get_encoding("cl100k_base")

    return len(encoding.encode(text))


def _heuristic_count(text: str) -> int:
    """Cheap fallback: roughly four characters per token."""
    if not text:
        return 0
    return len(text) // TOKEN_CHARS_PER_TOKEN


def estimate_text_tokens(
    text: str,
    provider_type: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
) -> int:
    """Estimate tokens for a text string using the best safe local strategy.

    ``base_url`` is accepted as an extension point for future llama.cpp/Ollama
    ``/tokenize`` support. v1 avoids network calls during preflight counting.
    """
    del base_url

    if _is_openai_tokenizer_candidate(provider_type, model):
        count = _tiktoken_count(text, model)
        if count is not None:
            return count

    return _heuristic_count(text)


def estimate_messages_tokens(
    messages: Sequence[AnyMessage],
    system_prompt: str = "",
    provider_type: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
) -> int:
    """Estimate token count for a LangChain message sequence."""
    text_parts: list[str] = []
    if system_prompt:
        text_parts.append(system_prompt)

    for message in messages:
        content = getattr(message, "content", None)
        if content:
            text_parts.append(_content_to_text(content))

        if isinstance(message, AIMessage) and message.tool_calls:
            for tool_call in message.tool_calls:
                text_parts.append(str(tool_call.get("name", "")))
                text_parts.append(str(tool_call.get("args", {})))

    return estimate_text_tokens(
        "".join(text_parts),
        provider_type=provider_type,
        model=model,
        base_url=base_url,
    )


def _usage_from_mapping(usage: dict[str, Any]) -> TokenUsage | None:
    """Normalize common provider usage mapping shapes."""
    input_tokens = (
        _as_int(usage.get("input_tokens"))
        or _as_int(usage.get("prompt_tokens"))
        or _as_int(usage.get("prompt_eval_count"))
    )
    output_tokens = (
        _as_int(usage.get("output_tokens"))
        or _as_int(usage.get("completion_tokens"))
        or _as_int(usage.get("eval_count"))
    )
    total_tokens = _as_int(usage.get("total_tokens"))

    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens

    if total_tokens is None:
        return None

    return TokenUsage(
        input_tokens=input_tokens or 0,
        output_tokens=output_tokens or 0,
        total_tokens=total_tokens,
        source="provider_usage",
        confidence="exact",
    )


def _usage_from_object(usage: Any) -> TokenUsage | None:
    """Normalize provider usage objects with attributes."""
    if isinstance(usage, dict):
        return _usage_from_mapping(usage)

    input_tokens = (
        _as_int(getattr(usage, "input_tokens", None))
        or _as_int(getattr(usage, "prompt_tokens", None))
        or _as_int(getattr(usage, "prompt_eval_count", None))
    )
    output_tokens = (
        _as_int(getattr(usage, "output_tokens", None))
        or _as_int(getattr(usage, "completion_tokens", None))
        or _as_int(getattr(usage, "eval_count", None))
    )
    total_tokens = _as_int(getattr(usage, "total_tokens", None))

    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens

    if total_tokens is None:
        return None

    return TokenUsage(
        input_tokens=input_tokens or 0,
        output_tokens=output_tokens or 0,
        total_tokens=total_tokens,
        source="provider_usage",
        confidence="exact",
    )


def extract_usage_from_message(message: Any) -> TokenUsage | None:
    """Extract normalized usage from a LangChain AI message if available."""
    usage_metadata = getattr(message, "usage_metadata", None)
    if usage_metadata:
        normalized = _usage_from_object(usage_metadata)
        if normalized:
            return normalized

    response_metadata = getattr(message, "response_metadata", None)
    if isinstance(response_metadata, dict):
        usage = (
            response_metadata.get("usage")
            or response_metadata.get("Usage")
            or response_metadata
        )
        normalized = _usage_from_object(usage)
        if normalized:
            return normalized

    return None


def estimate_message_output_tokens(
    message: Any,
    provider_type: str | None = None,
    model: str | None = None,
) -> int:
    """Estimate generated output tokens from message content only."""
    text = _content_to_text(getattr(message, "content", None))
    return estimate_text_tokens(text, provider_type=provider_type, model=model)
