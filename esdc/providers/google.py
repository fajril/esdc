# Standard library
import logging
import time
from typing import Any

# Third-party
from langchain_google_genai import ChatGoogleGenerativeAI

# Local
from esdc.providers.base import Provider, ProviderConfig

logger = logging.getLogger(__name__)


class GoogleProvider(Provider):
    """Provider implementation for Google Gemini API."""

    NAME = "Google (Gemini)"
    DEFAULT_MODEL = "gemini-2.5-flash"

    CONTEXT_LENGTHS = {
        "gemini-2.5-pro": 1_048_576,
        "gemini-2.5-flash": 1_048_576,
        "gemini-2.0-pro": 1_048_576,
        "gemini-2.0-flash": 1_048_576,
        "gemini-1.5-pro": 2_097_152,
        "gemini-1.5-flash": 1_048_576,
        "gemini-1.0-pro": 32_768,
    }

    @classmethod
    def list_models(cls, **kwargs: Any) -> list[str]:
        """List available Gemini models."""
        logger.debug("[INFERENCE] google_list_models")
        list_start = time.perf_counter()

        models = list(cls.CONTEXT_LENGTHS.keys())

        elapsed_ms = (time.perf_counter() - list_start) * 1000
        logger.debug(
            "[INFERENCE] google_list_models_complete | elapsed=%.2fms | models=%d",
            elapsed_ms,
            len(models),
        )
        return models

    @classmethod
    def get_default_model(cls, **kwargs: Any) -> str:
        """Get default model."""
        return cls.DEFAULT_MODEL

    @classmethod
    def is_configured(cls, config: ProviderConfig) -> bool:
        """Check if Google provider is configured (API key present)."""
        return bool(config.api_key)

    @classmethod
    def create_llm(
        cls,
        model: str | None = None,
        api_key: str | None = None,
        config: ProviderConfig | None = None,
        temperature: float = 0.0,
        reasoning_effort: str | None = None,
        **kwargs,
    ) -> ChatGoogleGenerativeAI:
        """Create a ChatGoogleGenerativeAI instance."""
        if not model:
            model = cls.get_default_model()

        effective_api_key = api_key or (config.api_key if config else None) or ""

        if reasoning_effort is not None:
            logger.debug(
                "[INFERENCE] google_create_llm | "
                "reasoning_effort is not supported by Google, ignoring"
            )

        return ChatGoogleGenerativeAI(
            model=model,
            google_api_key=effective_api_key,
            temperature=temperature,
            **kwargs,
        )

    @classmethod
    def test_connection(cls, config: ProviderConfig) -> tuple[bool, str]:
        """Test Gemini API connection."""
        try:
            if not config.api_key:
                return False, "Not configured. Please provide a Google API key."

            llm = cls.create_llm(
                model=config.model or cls.DEFAULT_MODEL,
                api_key=config.api_key,
            )

            logger.debug(
                "[INFERENCE] google_test_connection_invoke | model=%s",
                config.model or cls.DEFAULT_MODEL,
            )
            invoke_start = time.perf_counter()

            llm.invoke("Hello")

            invoke_elapsed_ms = (time.perf_counter() - invoke_start) * 1000
            logger.debug(
                "[INFERENCE] google_test_connection_complete | "
                "model=%s | elapsed=%.2fms",
                config.model or cls.DEFAULT_MODEL,
                invoke_elapsed_ms,
            )
            return True, "Connected. Gemini is ready."
        except Exception as e:
            return False, str(e)
