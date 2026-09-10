"""OpenCode Go provider implementation.

OpenCode Go (https://opencode.ai/docs/go) requires every request to carry a
stable ``x-opencode-session`` header and a self-identifying ``User-Agent``;
without them the gateway answers ``400 MissingSessionID``. This provider is an
OpenAI-compatible client pointed at the Go endpoint with those headers
injected on every request it makes.
"""

import os
import uuid
from typing import Any

from openai import OpenAI

from esdc.providers.base import ProviderConfig
from esdc.providers.openai_compatible import OpenAICompatibleProvider

_SESSION_ID = f"esdc-{uuid.uuid4().hex}"


def _package_version() -> str:
    """Return the installed ESDC version, or '0' when unavailable."""
    try:
        from importlib.metadata import version

        return version("esdc")
    except Exception:
        return "0"


def _session_id() -> str:
    """Return a process-stable session ID, overridable by env var."""
    return os.environ.get("ESDC_OPENCODE_SESSION") or _SESSION_ID


class OpencodeProvider(OpenAICompatibleProvider):
    """Provider for OpenCode Go / Zen (OpenAI-compatible chat completions)."""

    NAME = "OpenCode Go"
    BASE_URL = "https://opencode.ai/zen/go/v1"
    DEFAULT_MODEL = "deepseek-v4-flash"

    CONTEXT_LENGTHS = {
        "glm-5": 200_000,
        "glm-4": 200_000,
        "kimi": 256_000,
        "deepseek": 128_000,
        "qwen3": 262_144,
        "minimax": 204_800,
        "longcat": 128_000,
        "mimo": 128_000,
        "grok": 256_000,
        "gpt-5": 272_000,
        "muse": 131_072,
    }

    @classmethod
    def _default_headers(cls) -> dict[str, str]:
        """Headers the OpenCode gateway requires on every request."""
        return {
            "x-opencode-session": _session_id(),
            "User-Agent": f"esdc/{_package_version()}",
        }

    @classmethod
    def get_default_model(cls, **kwargs: Any) -> str:
        return cls.DEFAULT_MODEL

    @classmethod
    def is_configured(cls, config: ProviderConfig) -> bool:
        return bool(config.api_key)

    @classmethod
    def test_connection(cls, config: ProviderConfig) -> tuple[bool, str]:
        """Test OpenCode connection, defaulting the base URL when omitted."""
        if not config.api_key:
            return False, "Not configured. Please set your OpenCode API key."

        effective_base_url = config.base_url or cls.BASE_URL
        try:
            models = cls.list_models(
                base_url=effective_base_url,
                api_key=config.api_key,
            )
            if not models:
                return False, "Connected but no models available. Check your API key."

            llm = cls.create_llm(
                model=config.model or cls.DEFAULT_MODEL,
                base_url=effective_base_url,
                api_key=config.api_key,
            )
            llm.invoke("Hello")
            return True, f"Connected. Available models: {len(models)} models"
        except Exception as e:
            return False, str(e)

    @classmethod
    def create_llm(
        cls,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.0,
        reasoning_effort: str | None = None,
        config: ProviderConfig | None = None,
        **kwargs: Any,
    ):
        """Create a ChatOpenAI client wired for OpenCode Go."""
        model = model or cls.DEFAULT_MODEL
        base_url = base_url or cls.BASE_URL
        merged_headers = {
            **cls._default_headers(),
            **(kwargs.pop("default_headers", None) or {}),
        }
        kwargs["default_headers"] = merged_headers

        return super().create_llm(
            model=model,
            base_url=base_url,
            api_key=api_key,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            config=config,
            **kwargs,
        )

    @classmethod
    def list_models(
        cls,
        base_url: str | None = None,
        api_key: str | None = None,
        **kwargs: Any,
    ) -> list[str]:
        """List models from the OpenCode /models endpoint."""
        try:
            client = OpenAI(
                base_url=base_url or cls.BASE_URL,
                api_key=api_key or "none",
                default_headers=cls._default_headers(),
            )
            return [m.id for m in client.models.list()]  # type: ignore[union-attr]
        except Exception:
            return []

    @classmethod
    def get_context_length_from_api(
        cls,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> int:
        """Fetch context length from the OpenCode /models endpoint."""
        try:
            client = OpenAI(
                base_url=base_url or cls.BASE_URL,
                api_key=api_key or "none",
                default_headers=cls._default_headers(),
            )
            for m in client.models.list():
                if m.id == model:
                    ctx = cls._extract_context_length(m)
                    if ctx > 0:
                        return ctx
        except Exception:
            pass
        return 0
