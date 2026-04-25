# Standard library
import logging
import time
from typing import Any

# Third-party
from langchain_anthropic import ChatAnthropic

# Local
from esdc.providers.base import Provider, ProviderConfig

logger = logging.getLogger(__name__)


class AnthropicProvider(Provider):
    """Provider implementation for Anthropic Claude API."""

    NAME = "Anthropic (Claude)"
    DEFAULT_MODEL = "claude-sonnet-4-6"

    CONTEXT_LENGTHS = {
        "claude-opus-4-7": 200000,
        "claude-opus-4-6": 200000,
        "claude-sonnet-4-6": 1_000_000,
        "claude-sonnet-4-5": 1_000_000,
        "claude-3-opus": 200000,
        "claude-3-sonnet": 200000,
        "claude-3-haiku": 200000,
        "claude-3-5-sonnet": 200000,
        "claude-3-5-haiku": 200000,
        "claude-3-7-sonnet": 200000,
    }

    @classmethod
    def list_models(cls, **kwargs: Any) -> list[str]:
        """Return a static list of known Anthropic models."""
        return [
            "claude-sonnet-4-6",
            "claude-sonnet-4-5-20250929",
            "claude-opus-4-7",
            "claude-opus-4-6",
            "claude-3-7-sonnet-20250219",
            "claude-3-5-sonnet-20241022",
            "claude-3-5-sonnet-20240620",
            "claude-3-5-haiku-20241022",
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
        ]

    @classmethod
    def get_default_model(cls, **kwargs: Any) -> str:
        """Get default model."""
        return cls.DEFAULT_MODEL

    @classmethod
    def is_configured(cls, config: ProviderConfig) -> bool:
        """Check if Anthropic API key is configured."""
        return bool(config.api_key)

    @classmethod
    def create_llm(
        cls,
        model: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.0,
        reasoning_effort: str | None = None,
        **kwargs,
    ) -> ChatAnthropic:
        """Create a ChatAnthropic instance.

        Args:
            model: Model name (e.g. "claude-3-5-sonnet-20241022")
            api_key: Anthropic API key
            temperature: Sampling temperature
            reasoning_effort: Reasoning effort level.
                Passed via ``extra_body`` when set.
            **kwargs: Additional keyword arguments passed to ChatAnthropic.
        """
        if not model:
            model = cls.get_default_model()

        if reasoning_effort is not None:
            kwargs["extra_body"] = {
                **kwargs.get("extra_body", {}),
                "reasoning_effort": reasoning_effort,
            }

        return ChatAnthropic(
            model_name=model,
            api_key=api_key or "",  # type: ignore[arg-type]
            temperature=temperature,
            **kwargs,
        )

    @classmethod
    def test_connection(cls, config: ProviderConfig) -> tuple[bool, str]:
        """Test Anthropic connection."""
        try:
            if not config.api_key:
                return (
                    False,
                    "Not configured. Set your Anthropic API key to connect.",
                )

            llm = cls.create_llm(
                model=config.model or cls.DEFAULT_MODEL,
                api_key=config.api_key,
            )

            logger.debug(
                "[INFERENCE] anthropic_test_connection_invoke | model=%s",
                config.model or cls.DEFAULT_MODEL,
            )
            invoke_start = time.perf_counter()

            llm.invoke("Hello")

            invoke_elapsed_ms = (time.perf_counter() - invoke_start) * 1000
            logger.debug(
                "[INFERENCE] anthropic_test_connection_complete | model=%s | elapsed=%.2fms",  # noqa: E501
                config.model or cls.DEFAULT_MODEL,
                invoke_elapsed_ms,
            )
            return True, "Connected. API key is valid."
        except Exception as e:
            return False, str(e)
