from typing import Any

from esdc.providers.base import Provider, ProviderConfig
from esdc.providers.ollama import OllamaProvider
from esdc.providers.openai import OpenAIProvider
from esdc.providers.openai_compatible import OpenAICompatibleProvider

PROVIDER_CLASSES: dict[str, type[Provider]] = {
    "ollama": OllamaProvider,
    "openai": OpenAIProvider,
    "openai_compatible": OpenAICompatibleProvider,
}

PROVIDER_NAMES: dict[str, str] = {
    "ollama": "Ollama",
    "openai": "OpenAI",
    "openai_compatible": "OpenAI-Compatible",
}


def get_provider(provider_type: str) -> type[Provider] | None:
    """Get provider class by type."""
    return PROVIDER_CLASSES.get(provider_type)


def get_provider_name(provider_type: str) -> str:
    """Get human-readable provider name."""
    return PROVIDER_NAMES.get(provider_type, provider_type)


def list_provider_types() -> list[str]:
    """List all available provider types."""
    return list(PROVIDER_CLASSES.keys())


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
        name=str(config.get("name", provider_type)),
        provider_type=provider_type,
        model=str(config.get("model", "")),
        base_url=str(config.get("base_url", "")),
        api_key=str(config.get("api_key", "")),
        auth_method=str(config.get("auth_method", "api_key")),
        oauth=config.get("oauth", {}),
    )

    model = config.get("model")
    base_url = config.get("base_url")
    api_key = config.get("api_key")

    return provider_class.create_llm(
        model=model if model else "",
        base_url=base_url if base_url else "",
        api_key=api_key if api_key else "",
    )


__all__ = [
    "Provider",
    "ProviderConfig",
    "OllamaProvider",
    "OpenAIProvider",
    "OpenAICompatibleProvider",
    "PROVIDER_CLASSES",
    "PROVIDER_NAMES",
    "get_provider",
    "get_provider_name",
    "list_provider_types",
    "create_provider",
    "create_llm_from_config",
]
