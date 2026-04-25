"""Ollama Cloud provider implementation."""

from typing import Any

from langchain_openai import ChatOpenAI
from openai import OpenAI

from esdc.providers.base import Provider, ProviderConfig


class OllamaCloudProvider(Provider):
    """Provider for Ollama Cloud (https://ollama.com/v1).

    Ollama Cloud provides OpenAI-compatible API access to cloud-hosted
    models without requiring local Ollama daemon.
    """

    NAME = "Ollama Cloud"
    BASE_URL = "https://ollama.com/v1"
    DEFAULT_MODEL = "llama3.2"

    CONTEXT_LENGTHS: dict[str, int] = {
        "llama3.2": 128000,
        "llama3.1": 128000,
        "deepseek-v3.1": 128000,
        "qwen3-coder": 128000,
        "gpt-oss:120b": 128000,
        "gpt-oss:20b": 128000,
    }

    @classmethod
    def list_models(cls, api_key: str | None = None, **kwargs: Any) -> list[str]:
        """List available models from Ollama Cloud API.

        Returns model names without :cloud suffix (e.g. "llama3.2").
        """
        try:
            client = OpenAI(
                base_url=cls.BASE_URL,
                api_key=api_key or "none",
            )
            models = client.models.list()
            return [m.id for m in models]  # type: ignore[union-attr]
        except Exception:
            return []

    @classmethod
    def get_default_model(cls, **kwargs: Any) -> str:
        """Return default model for Ollama Cloud."""
        return cls.DEFAULT_MODEL

    @classmethod
    def is_configured(cls, config: ProviderConfig) -> bool:
        """Check if api_key is configured."""
        return bool(config.api_key)

    @classmethod
    def create_llm(
        cls,
        model: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.0,
        reasoning_effort: str | None = None,
        **kwargs,
    ) -> ChatOpenAI:
        """Create a ChatOpenAI instance for Ollama Cloud.

        Args:
            model: Model name (e.g. "llama3.2"). Ollama Cloud auto-appends
                :cloud suffix if needed.
            api_key: Ollama Cloud API key.
            temperature: Sampling temperature.
            reasoning_effort: Not supported by Ollama Cloud, ignored.
            **kwargs: Additional keyword arguments.

        Returns:
            ChatOpenAI instance configured for Ollama Cloud.
        """
        if not model:
            raise ValueError("model is required for Ollama Cloud provider")

        if reasoning_effort is not None:
            pass  # Ollama Cloud does not support reasoning_effort

        return ChatOpenAI(
            model=model,
            base_url=cls.BASE_URL,
            api_key=api_key or "none",  # type: ignore[arg-type]
            temperature=temperature,
            **kwargs,
        )

    @classmethod
    def test_connection(cls, config: ProviderConfig) -> tuple[bool, str]:
        """Test Ollama Cloud connection."""
        if not config.api_key:
            return False, "API key not configured"

        try:
            models = cls.list_models(api_key=config.api_key)
            if models:
                return True, f"Connected. {len(models)} models available."
            return False, "No models found"
        except Exception as exc:
            return False, str(exc)
