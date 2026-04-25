# Standard library
import logging
import time
from typing import Any

# Third-party
from langchain_openai import AzureChatOpenAI

# Local
from esdc.providers.base import Provider, ProviderConfig

logger = logging.getLogger(__name__)


class AzureOpenAIProvider(Provider):
    """Provider implementation for Azure OpenAI API."""

    NAME = "Azure OpenAI"
    DEFAULT_MODEL = "gpt-4o"

    CONTEXT_LENGTHS = {
        "gpt-4o": 128000,
        "gpt-4o-mini": 128000,
        "gpt-4-turbo": 128000,
        "gpt-4": 8192,
        "gpt-3.5-turbo": 16385,
        "o1": 128000,
        "o1-mini": 128000,
        "o3": 200000,
        "o3-mini": 128000,
    }

    @classmethod
    def list_models(cls, **kwargs: Any) -> list[str]:
        """List available Azure OpenAI models."""
        return list(cls.CONTEXT_LENGTHS.keys())

    @classmethod
    def get_default_model(cls, **kwargs: Any) -> str:
        """Get default model."""
        return cls.DEFAULT_MODEL

    @classmethod
    def is_configured(cls, config: ProviderConfig) -> bool:
        """Check if Azure OpenAI is configured."""
        return bool(config.api_key and config.base_url)

    @classmethod
    def create_llm(
        cls,
        model: str | None = None,
        api_key: str | None = None,
        config: ProviderConfig | None = None,
        temperature: float = 0.0,
        reasoning_effort: str | None = None,
        **kwargs,
    ) -> AzureChatOpenAI:
        """Create an AzureChatOpenAI instance.

        Args:
            model: Azure deployment name
            api_key: API key
            config: Provider configuration
            temperature: Sampling temperature
            reasoning_effort: Reasoning effort level passed as extra_body
            **kwargs: Additional keyword arguments passed to AzureChatOpenAI.
        """
        if not model:
            model = cls.get_default_model()

        effective_api_key = api_key or ""
        azure_endpoint = ""
        api_version = "2024-02-01"

        if config:
            effective_api_key = effective_api_key or config.api_key
            azure_endpoint = config.base_url or ""
            if config.oauth:
                api_version = config.oauth.get("api_version", api_version)

        api_version = kwargs.pop("api_version", api_version)

        if reasoning_effort is not None:
            kwargs["extra_body"] = {
                **kwargs.get("extra_body", {}),
                "reasoning_effort": reasoning_effort,
            }

        return AzureChatOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=effective_api_key,  # type: ignore[arg-type]
            azure_deployment=model,
            api_version=api_version,
            temperature=temperature,
            **kwargs,
        )

    @classmethod
    def test_connection(cls, config: ProviderConfig) -> tuple[bool, str]:
        """Test Azure OpenAI connection."""
        try:
            if not cls.is_configured(config):
                return (
                    False,
                    "Not configured. Set API key and Azure endpoint (base URL).",
                )

            llm = cls.create_llm(
                model=config.model or cls.DEFAULT_MODEL,
                config=config,
            )

            logger.debug(
                "[INFERENCE] azure_openai_test_connection_invoke | model=%s",
                config.model or cls.DEFAULT_MODEL,
            )
            invoke_start = time.perf_counter()

            llm.invoke("Hello")

            invoke_elapsed_ms = (time.perf_counter() - invoke_start) * 1000
            logger.debug(
                "[INFERENCE] azure_openai_test_connection_complete | "
                "model=%s | elapsed=%.2fms",
                config.model or cls.DEFAULT_MODEL,
                invoke_elapsed_ms,
            )
            return True, "Connected to Azure OpenAI."
        except Exception as e:
            return False, str(e)
