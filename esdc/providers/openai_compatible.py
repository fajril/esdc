from typing import Any

from langchain_openai import ChatOpenAI
from openai import OpenAI

from esdc.providers.base import Provider, ProviderConfig


class OpenAICompatibleProvider(Provider):
    """Provider for OpenAI-compatible API servers."""

    NAME = "OpenAI Compatible API"

    CONTEXT_LENGTHS = {
        "qwen3.6": 262144,
        "qwen2.5": 128000,
        "qwen2": 131072,
        "deepseek": 128000,
        "llama3": 128000,
        "mistral": 32000,
        "mixtral": 32000,
        "phi4": 128000,
        "command-r": 128000,
    }

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
        config: ProviderConfig | None = None,
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
            config: Provider config (consumed to prevent leak into ChatOpenAI)
            **kwargs: Additional keyword arguments passed to ChatOpenAI.
        """
        _ = config  # consumed — prevents leak into ChatOpenAI kwargs
        if not model:
            raise ValueError("model is required for OpenAI Compatible API provider")
        if not base_url:
            raise ValueError("base_url is required for OpenAI Compatible API provider")

        if reasoning_effort is not None:
            kwargs["extra_body"] = {
                **kwargs.get("extra_body", {}),
                "reasoning_effort": reasoning_effort,
            }

        llm = ChatOpenAI(
            model=model,
            base_url=base_url,
            api_key=api_key or "none",  # type: ignore[arg-type]
            temperature=temperature,
            **kwargs,
        )
        # Attach metadata (fallback to zero; agent will use static dict if needed)
        val = cls.get_context_length_from_api(model, api_key=api_key, base_url=base_url)
        llm._esdc_context_length = val  # type: ignore[attr-defined]
        return llm

    @classmethod
    def get_context_length_from_api(
        cls, model: str, api_key: str | None = None, base_url: str | None = None
    ) -> int:
        """Fetch context length from OpenAI-compatible /v1/models API.

        Checked in order:

        1. Top-level ``__dict__`` fields (e.g. vLLM exposes ``max_model_len``).
        2. Pydantic ``model_extra`` top-level fields (some servers inject custom
           fields that land here rather than ``__dict__``).
        3. Nested ``model_extra["meta"]`` dict (llama-cpp-python / llama-server
           exposes ``n_ctx_train`` inside a ``meta`` object).
        """
        try:
            if not base_url:
                return 0
            client = OpenAI(base_url=base_url, api_key=api_key or "none")
            for m in client.models.list():
                if m.id == model:
                    ctx = cls._extract_context_length(m)
                    if ctx > 0:
                        return ctx
        except Exception:
            pass
        return 0

    @classmethod
    def _extract_context_length(cls, model_object: Any) -> int:
        """Extract context length from a model object returned by /v1/models.

        Searches multiple locations where OpenAI-compatible servers expose
        context window information.
        """
        # Level 1: top-level __dict__ fields (vLLM style: max_model_len)
        extra = getattr(model_object, "__dict__", {})
        for key in ("max_model_len", "context_length", "max_tokens"):
            if key in extra and extra[key]:
                return int(extra[key])

        # Level 2: Pydantic model_extra top-level fields
        model_extra = getattr(model_object, "model_extra", None) or {}
        for key in ("max_model_len", "context_length", "max_tokens"):
            if key in model_extra and model_extra[key]:
                return int(model_extra[key])

        # Level 3: nested meta object (llama-server / llama-cpp-python)
        meta = model_extra.get("meta", {}) if isinstance(model_extra, dict) else {}
        if isinstance(meta, dict):
            for key in ("n_ctx_train", "context_length", "max_model_len", "max_tokens"):
                if key in meta and meta[key]:
                    return int(meta[key])

        return 0

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
