from typing import Any

from esdc.providers.anthropic import AnthropicProvider
from esdc.providers.azure_openai import AzureOpenAIProvider
from esdc.providers.base import Provider, ProviderConfig
from esdc.providers.google import GoogleProvider
from esdc.providers.groq import GroqProvider
from esdc.providers.ollama import OllamaProvider
from esdc.providers.ollama_cloud import OllamaCloudProvider
from esdc.providers.openai import OpenAIProvider
from esdc.providers.openai_compatible import OpenAICompatibleProvider

PROVIDER_CLASSES: dict[str, type[Provider]] = {
    "ollama": OllamaProvider,
    "openai": OpenAIProvider,
    "openai_compatible": OpenAICompatibleProvider,
    "anthropic": AnthropicProvider,
    "google": GoogleProvider,
    "azure_openai": AzureOpenAIProvider,
    "groq": GroqProvider,
    "ollama_cloud": OllamaCloudProvider,
}

PROVIDER_NAMES: dict[str, str] = {
    "ollama": "Ollama",
    "openai": "OpenAI",
    "openai_compatible": "OpenAI Compatible API",
    "anthropic": "Anthropic (Claude)",
    "google": "Google (Gemini)",
    "azure_openai": "Azure OpenAI",
    "groq": "Groq",
    "ollama_cloud": "Ollama Cloud",
}


def get_provider(provider_type: str) -> type[Provider] | None:
    """Get provider class by type."""
    return PROVIDER_CLASSES.get(provider_type)


def get_provider_name(provider_type: str) -> str:
    """Get human-readable provider name."""
    return PROVIDER_NAMES.get(provider_type, provider_type)


def list_provider_types() -> list[str]:
    """List all available provider types."""
    return sorted(PROVIDER_CLASSES.keys())


def create_provider(provider_type: str, model: str | None = None, **kwargs) -> Provider:
    """Create a provider instance."""
    provider_class = get_provider(provider_type)
    if not provider_class:
        raise ValueError(f"Unknown provider type: {provider_type}")

    return provider_class()


def create_llm_from_config(config: dict[str, Any]):
    """Create a LangChain LLM from provider config dict."""
    provider_type = config.get("provider_type") or config.get("type")
    if not provider_type:
        raise ValueError("provider_type is required in config")

    provider_class = get_provider(provider_type)
    if not provider_class:
        raise ValueError(f"Unknown provider type: {provider_type}")

    provider_config = ProviderConfig(
        name=str(config.get("name") or provider_type),
        provider_type=provider_type,
        model=str(config.get("model") or ""),
        base_url=str(config.get("base_url") or ""),
        api_key=str(config.get("api_key") or ""),
        auth_method=str(config.get("auth_method") or "api_key"),
        oauth=config.get("oauth") or {},
        reasoning_effort=config.get("reasoning_effort"),
    )

    llm_kwargs: dict[str, Any] = {
        "config": provider_config,
    }
    if provider_config.reasoning_effort is not None:
        llm_kwargs["reasoning_effort"] = provider_config.reasoning_effort

    return provider_class.create_llm(
        model=provider_config.model or None,
        base_url=provider_config.base_url or None,
        api_key=provider_config.api_key or None,
        **llm_kwargs,
    )


__all__ = [
    "Provider",
    "ProviderConfig",
    "OllamaProvider",
    "OllamaCloudProvider",
    "OpenAIProvider",
    "OpenAICompatibleProvider",
    "AnthropicProvider",
    "GoogleProvider",
    "AzureOpenAIProvider",
    "GroqProvider",
    "PROVIDER_CLASSES",
    "PROVIDER_NAMES",
    "get_provider",
    "get_provider_name",
    "list_provider_types",
    "create_provider",
    "create_llm_from_config",
]
