from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable

from esdc.providers.anthropic import AnthropicProvider
from esdc.providers.azure_openai import AzureOpenAIProvider
from esdc.providers.base import Provider, ProviderConfig
from esdc.providers.deepseek import DeepSeekProvider
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
    "deepseek": DeepSeekProvider,
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
    "deepseek": "DeepSeek",
    "ollama_cloud": "Ollama Cloud",
}


class ProviderFallbackChatModel(BaseChatModel):
    """Chat model wrapper that tries configured providers in order."""

    models: list[BaseChatModel]
    provider_names: list[str]
    model_names: list[str]
    last_provider_name: str | None = None
    last_model_name: str | None = None

    @property
    def _llm_type(self) -> str:
        return "esdc-provider-fallback"

    def bind_tools(self, tools: list[Any], **kwargs: Any) -> Runnable:
        """Bind tools to each provider and return a fallback runnable."""
        if not self.models:
            raise ValueError("No provider models configured")

        bound_models = [model.bind_tools(tools, **kwargs) for model in self.models]
        if len(bound_models) == 1:
            return bound_models[0]
        return bound_models[0].with_fallbacks(bound_models[1:])

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        last_error: Exception | None = None
        for model, provider_name, model_name in zip(
            self.models, self.provider_names, self.model_names, strict=False
        ):
            try:
                message = model.invoke(messages, stop=stop, **kwargs)
                self.last_provider_name = provider_name
                self.last_model_name = model_name
                return ChatResult(generations=[ChatGeneration(message=message)])
            except Exception as exc:
                last_error = exc
        if last_error is not None:
            raise last_error
        raise ValueError("No provider models configured")

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        last_error: Exception | None = None
        for model, provider_name, model_name in zip(
            self.models, self.provider_names, self.model_names, strict=False
        ):
            try:
                message = await model.ainvoke(messages, stop=stop, **kwargs)
                self.last_provider_name = provider_name
                self.last_model_name = model_name
                return ChatResult(generations=[ChatGeneration(message=message)])
            except Exception as exc:
                last_error = exc
        if last_error is not None:
            raise last_error
        raise ValueError("No provider models configured")


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


def _create_single_llm_from_config(config: dict[str, Any]):
    """Create a LangChain LLM from a single provider config dict."""
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
    if config.get("temperature") is not None:
        llm_kwargs["temperature"] = config["temperature"]

    llm = provider_class.create_llm(
        model=provider_config.model or None,
        base_url=provider_config.base_url or None,
        api_key=provider_config.api_key or None,
        **llm_kwargs,
    )
    object.__setattr__(llm, "_esdc_provider_name", provider_config.name)
    object.__setattr__(llm, "_esdc_provider_type", provider_config.provider_type)
    object.__setattr__(llm, "_esdc_model_name", provider_config.model)
    object.__setattr__(llm, "_esdc_base_url", provider_config.base_url)
    return llm


def create_llm_from_config(config: dict[str, Any]):
    """Create a LangChain LLM from provider config dict.

    If the config contains ``fallback_configs``, returns a wrapper that tries
    the primary provider first and then each fallback provider in order.
    """
    configs = [config] + list(config.get("fallback_configs") or [])
    llms: list[BaseChatModel] = []
    provider_names: list[str] = []
    model_names: list[str] = []
    errors: list[str] = []

    for cfg in configs:
        try:
            llm = _create_single_llm_from_config(cfg)
            llms.append(llm)
            provider_names.append(str(cfg.get("name") or cfg.get("provider_type")))
            model_names.append(str(cfg.get("model") or ""))
        except Exception as exc:
            provider_name = str(cfg.get("name") or cfg.get("provider_type") or "?")
            errors.append(f"{provider_name}: {exc}")

    if not llms:
        detail = "; ".join(errors) if errors else "no providers configured"
        raise ValueError(f"Failed to create any provider LLM: {detail}")

    if len(llms) == 1:
        return llms[0]

    return ProviderFallbackChatModel(
        models=llms,
        provider_names=provider_names,
        model_names=model_names,
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
    "DeepSeekProvider",
    "PROVIDER_CLASSES",
    "PROVIDER_NAMES",
    "ProviderFallbackChatModel",
    "get_provider",
    "get_provider_name",
    "list_provider_types",
    "create_provider",
    "create_llm_from_config",
]
