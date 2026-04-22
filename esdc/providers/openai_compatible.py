from typing import Any

from langchain_openai import ChatOpenAI
from openai import OpenAI

from esdc.providers.base import Provider, ProviderConfig


class OpenAICompatibleProvider(Provider):
    """Provider for OpenAI-compatible API servers."""

    NAME = "OpenAI-Compatible"

    @classmethod
    def list_models(
        cls, base_url: str | None = None, api_key: str | None = None, **kwargs: Any
    ) -> list[str]:
        """List available models from OpenAI-compatible server."""
        if not base_url:
            return []

        try:
            client = OpenAI(base_url=base_url, api_key=api_key or "none")
            models = client.models.list()
            return [m.id for m in models]  # type: ignore[union-attr]
        except Exception:
            return []

    @classmethod
    def get_default_model(cls, **kwargs: Any) -> str:
        """No default model - user must specify."""
        return ""

    @classmethod
    def is_configured(cls, config: ProviderConfig) -> bool:
        """Check if base_url is configured."""
        return bool(config.base_url)

    @classmethod
    def create_llm(
        cls,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.0,
        reasoning_effort: str | None = None,
        **kwargs,
    ) -> ChatOpenAI:
        """Create a ChatOpenAI instance with custom base URL.

        Args:
            model: Model name (required)
            base_url: OpenAI-compatible API base URL (required)
            api_key: API key for authentication
            temperature: Sampling temperature
            reasoning_effort: Reasoning effort level passed as extra_body
                parameter. Supported by some OpenAI-compatible backends.
            **kwargs: Additional keyword arguments passed to ChatOpenAI.
        """
        if not model:
            raise ValueError("model is required for OpenAI-Compatible provider")
        if not base_url:
            raise ValueError("base_url is required for OpenAI-Compatible provider")

        if reasoning_effort is not None:
            kwargs["extra_body"] = {
                **kwargs.get("extra_body", {}),
                "reasoning_effort": reasoning_effort,
            }

        return ChatOpenAI(
            model=model,
            base_url=base_url,
            api_key=api_key or "none",  # type: ignore[arg-type]
            temperature=temperature,
            **kwargs,
        )

    @classmethod
    def test_connection(cls, config: ProviderConfig) -> tuple[bool, str]:
        """Test OpenAI-compatible server connection."""
        if not config.base_url:
            return False, "base_url not configured"

        try:
            models = cls.list_models(config.base_url, config.api_key or None)

            if not models:
                return (
                    False,
                    "Connected but couldn't list models. Try specifying a model name.",
                )

            if config.model:
                llm = cls.create_llm(
                    model=config.model,
                    base_url=config.base_url,
                    api_key=config.api_key,
                )
                llm.invoke("Hello")

            return (
                True,
                f"Connected. Available models: {', '.join(models[:5])}{'...' if len(models) > 5 else ''}",  # noqa: E501
            )
        except Exception as e:
            return False, str(e)
