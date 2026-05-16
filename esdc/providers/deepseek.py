"""DeepSeek provider implementation."""

import logging
import time
from typing import Any

from langchain_openai import ChatOpenAI
from openai import OpenAI

from esdc.providers.base import Provider, ProviderConfig

logger = logging.getLogger(__name__)


class DeepSeekProvider(Provider):
    """Provider implementation for the DeepSeek OpenAI-compatible API."""

    NAME = "DeepSeek"
    DEFAULT_MODEL = "deepseek-v4-flash"
    BASE_URL = "https://api.deepseek.com"
    BETA_BASE_URL = "https://api.deepseek.com/beta"

    CONTEXT_LENGTHS = {
        "deepseek-v4-flash": 1_000_000,
        "deepseek-v4-pro": 1_000_000,
        "deepseek-chat": 1_000_000,
        "deepseek-reasoner": 1_000_000,
        "deepseek-v4": 1_000_000,
    }

    @classmethod
    def list_models(cls, api_key: str | None = None, **kwargs: Any) -> list[str]:
        """List available DeepSeek models."""
        try:
            client = OpenAI(api_key=api_key or "", base_url=cls.BASE_URL)

            logger.debug("[INFERENCE] deepseek_list_models | base_url=%s", cls.BASE_URL)
            list_start = time.perf_counter()
            models = client.models.list()

            elapsed_ms = (time.perf_counter() - list_start) * 1000
            model_list = [m.id for m in models]  # type: ignore[union-attr]
            logger.debug(
                "[INFERENCE] deepseek_list_models_complete | "
                "elapsed=%.2fms | models=%d",
                elapsed_ms,
                len(model_list),
            )
            return model_list
        except Exception as e:
            logger.debug(
                "[INFERENCE] deepseek_list_models_error | error=%s",
                str(e)[:100],
            )
            return []

    @classmethod
    def get_default_model(cls, **kwargs: Any) -> str:
        """Get default model."""
        return cls.DEFAULT_MODEL

    @classmethod
    def is_configured(cls, config: ProviderConfig) -> bool:
        """Check if DeepSeek is configured."""
        return bool(config.api_key)

    @classmethod
    def _normalize_reasoning_effort(
        cls,
        reasoning_effort: str | None,
    ) -> tuple[str | None, dict[str, Any] | None]:
        """Map ESDC reasoning values to DeepSeek thinking parameters."""
        if reasoning_effort is None:
            return None, None

        effort = reasoning_effort.lower()
        if effort == "none":
            return None, {"thinking": {"type": "disabled"}}
        if effort in {"low", "medium", "high"}:
            return "high", {"thinking": {"type": "enabled"}}
        if effort in {"xhigh", "max"}:
            return "max", {"thinking": {"type": "enabled"}}

        logger.debug(
            "[INFERENCE] deepseek_reasoning_effort_unknown | "
            "reasoning_effort=%s | using_high",
            reasoning_effort,
        )
        return "high", {"thinking": {"type": "enabled"}}

    @classmethod
    def create_llm(
        cls,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.0,
        reasoning_effort: str | None = None,
        config: ProviderConfig | None = None,
        **kwargs,
    ) -> ChatOpenAI:
        """Create a ChatOpenAI instance configured for DeepSeek.

        DeepSeek uses OpenAI-compatible chat completions but has its own
        thinking toggle in ``extra_body`` and only supports ``high`` or ``max``
        for reasoning effort.
        """
        _ = config  # consumed to prevent leak into ChatOpenAI kwargs
        if not model:
            model = cls.get_default_model()

        normalized_effort, thinking_body = cls._normalize_reasoning_effort(
            reasoning_effort
        )
        if normalized_effort is not None:
            kwargs["reasoning_effort"] = normalized_effort
        if thinking_body is not None:
            kwargs["extra_body"] = {
                **kwargs.get("extra_body", {}),
                **thinking_body,
            }

        llm = ChatOpenAI(
            model=model,
            api_key=api_key or "",  # type: ignore[arg-type]
            base_url=base_url or cls.BASE_URL,
            temperature=temperature,
            **kwargs,
        )
        val = cls.get_actual_context_length(model, api_key=api_key)
        llm._esdc_context_length = val  # type: ignore[attr-defined]
        return llm

    @classmethod
    def test_connection(cls, config: ProviderConfig) -> tuple[bool, str]:
        """Test DeepSeek connection."""
        try:
            if not config.api_key:
                return False, "Not configured. Please set your DeepSeek API key."

            models = cls.list_models(api_key=config.api_key)
            if not models:
                return (
                    False,
                    "Connected but no models available. Check your API key.",
                )

            llm = cls.create_llm(
                model=config.model or cls.DEFAULT_MODEL,
                api_key=config.api_key,
                base_url=config.base_url or None,
                reasoning_effort=config.reasoning_effort,
            )

            logger.debug(
                "[INFERENCE] deepseek_test_connection_invoke | model=%s",
                config.model or cls.DEFAULT_MODEL,
            )
            invoke_start = time.perf_counter()
            llm.invoke("Hello")

            invoke_elapsed_ms = (time.perf_counter() - invoke_start) * 1000
            logger.debug(
                "[INFERENCE] deepseek_test_connection_complete | "
                "model=%s | elapsed=%.2fms",
                config.model or cls.DEFAULT_MODEL,
                invoke_elapsed_ms,
            )
            return True, f"Connected. Available models: {len(models)} models"
        except Exception as e:
            return False, str(e)
