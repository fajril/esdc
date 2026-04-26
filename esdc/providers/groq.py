# Standard library
import logging
import time
from typing import Any

# Third-party
from langchain_groq import ChatGroq

# Local
from esdc.providers.base import Provider, ProviderConfig

logger = logging.getLogger(__name__)


class GroqProvider(Provider):
    """Provider implementation for Groq API."""

    NAME = "Groq"
    DEFAULT_MODEL = "llama-3.3-70b-versatile"
    BASE_URL = "https://api.groq.com/openai/v1"

    CONTEXT_LENGTHS = {
        "llama-4-scout": 128000,
        "llama-3.3-70b": 128000,
        "llama-3.1-8b": 128000,
        "llama-3.1-70b": 128000,
        "llama-3.1-405b": 128000,
        "llama-3.2": 128000,
        "mixtral-8x7b": 32768,
        "gemma2-9b": 8192,
        "qwen-2.5-32b": 128000,
        "qwen-qwq-32b": 128000,
        "deepseek-r1-distill-llama-70b": 128000,
        "mistral-saba-24b": 32768,
    }

    @classmethod
    def list_models(cls, api_key: str | None = None, **kwargs: Any) -> list[str]:
        """List available Groq models."""
        try:
            import groq

            client = groq.Groq(
                api_key=api_key or "",
            )

            logger.debug("[INFERENCE] groq_list_models | base_url=%s", cls.BASE_URL)
            list_start = time.perf_counter()

            models = client.models.list()

            elapsed_ms = (time.perf_counter() - list_start) * 1000
            model_list = [m.id for m in models]  # type: ignore[union-attr]
            logger.debug(
                "[INFERENCE] groq_list_models_complete | elapsed=%.2fms | models=%d",
                elapsed_ms,
                len(model_list),
            )
            return model_list
        except Exception as e:
            logger.debug("[INFERENCE] groq_list_models_error | error=%s", str(e)[:100])
            return []

    @classmethod
    def get_default_model(cls, **kwargs: Any) -> str:
        """Get default model."""
        return cls.DEFAULT_MODEL

    @classmethod
    def get_context_length(cls, model: str) -> int:
        model_clean = model.split(":")[0].split("/")[-1].lower()
        for key, value in cls.CONTEXT_LENGTHS.items():
            if key in model_clean:
                return value
        logger.debug(
            "[INFERENCE] groq_get_context_length | model=%s | fallback=4096",
            model_clean,
        )
        return 4096

    @classmethod
    def is_configured(cls, config: ProviderConfig) -> bool:
        """Check if Groq is configured (API key)."""
        return bool(config.api_key)

    @classmethod
    def create_llm(
        cls,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.0,
        reasoning_effort: str | None = None,
        **kwargs,
    ) -> ChatGroq:
        """Create a ChatGroq instance.

        Args:
            model: Model name (e.g. "llama-3.3-70b-versatile")
            api_key: API key
            base_url: Base URL for Groq API
            temperature: Sampling temperature
            reasoning_effort: Reasoning effort level. Ignored for Groq.
            **kwargs: Additional keyword arguments passed to ChatGroq.
        """
        if not model:
            model = cls.get_default_model()

        if reasoning_effort is not None:
            logger.debug(
                "[INFERENCE] groq_create_llm | reasoning_effort=%s ignored",
                reasoning_effort,
            )

        llm = ChatGroq(
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
        """Test Groq connection."""
        try:
            if not config.api_key:
                return (
                    False,
                    "Not configured. Please set your Groq API key.",
                )

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
            )

            logger.debug(
                "[INFERENCE] groq_test_connection_invoke | model=%s",
                config.model or cls.DEFAULT_MODEL,
            )
            invoke_start = time.perf_counter()

            llm.invoke("Hello")

            invoke_elapsed_ms = (time.perf_counter() - invoke_start) * 1000
            logger.debug(
                "[INFERENCE] groq_test_connection_complete | model=%s | elapsed=%.2fms",  # noqa: E501
                config.model or cls.DEFAULT_MODEL,
                invoke_elapsed_ms,
            )
            return True, f"Connected. Available models: {len(models)} models"
        except Exception as e:
            return False, str(e)
