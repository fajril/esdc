"""Ollama Cloud provider — inherits from OllamaProvider.

Uses ChatOllama (Ollama Native API /api/chat) instead of ChatOpenAI
(OpenAI-compatible /v1/chat/completions) for identical behavior with
Ollama local and reliable empty-IMessage handling.
"""

import logging
import time
from typing import Any

import ollama
from langchain_ollama import ChatOllama

from esdc.providers.base import ProviderConfig, _extract_model_info
from esdc.providers.ollama import OllamaProvider

logger = logging.getLogger(__name__)


class OllamaCloudProvider(OllamaProvider):
    """Provider for Ollama Cloud (https://ollama.com).

    Inherits all Ollama local behavior but targets the cloud endpoint
    and requires API key authentication via request headers.
    """

    NAME = "Ollama Cloud"
    BASE_URL = "https://ollama.com"
    DEFAULT_BASE_URL = "https://ollama.com"
    DEFAULT_MODEL = "llama3.2"

    CONTEXT_LENGTHS: dict[str, int] = {
        "llama3.2": 128000,
        "llama3.1": 128000,
        "deepseek": 128000,
        "qwen": 128000,
        "gpt-oss": 128000,
        "kimi-k2.5": 262144,
        "kimi-k2.6": 262144,
        "deepseek-v4-flash": 1048576,
    }

    @classmethod
    def _resolve_api_key(
        cls,
        api_key: str | None = None,
        config: ProviderConfig | None = None,
    ) -> str | None:
        """Resolve effective API key from args or config."""
        return api_key or (config.api_key if config else None)

    @classmethod
    def _auth_headers(cls, api_key: str | None) -> dict[str, str]:
        """Return Authorization headers dict if api_key is present."""
        if api_key:
            return {"Authorization": f"Bearer {api_key}"}
        return {}

    @classmethod
    def get_context_length_from_api(
        cls,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> int:
        """Fetch context length from Ollama Cloud /api/show with auth."""
        try:
            client = ollama.Client(
                host=base_url or cls.DEFAULT_BASE_URL,
                headers=cls._auth_headers(api_key),
            )
            logger.debug(
                "[INFERENCE] ollama_cloud_show | model=%s",
                model,
            )
            show_start = time.perf_counter()
            info = client.show(model)
            show_elapsed_ms = (time.perf_counter() - show_start) * 1000
            logger.debug(
                "[INFERENCE] ollama_cloud_show_complete | model=%s | elapsed=%.2fms",
                model,
                show_elapsed_ms,
            )

            model_info = _extract_model_info(info)

            for key, value in model_info.items():
                if key.endswith(".context_length") and isinstance(value, (int, float)):
                    ctx_len = int(value)
                    logger.info(
                        "Ollama Cloud context length: %s = %s tokens",
                        model,
                        ctx_len,
                    )
                    return ctx_len

            logger.warning(
                "No context_length in model_info for %s",
                model,
            )
        except Exception as exc:
            logger.warning(
                "Failed to fetch context length: %s",
                exc,
            )

        fallback = cls.get_context_length(model)
        logger.warning(
            "Using fallback context length: %s = %s tokens",
            model,
            fallback,
        )
        return fallback

    @classmethod
    def list_models(cls, base_url: str | None = None, **kwargs: Any) -> list[str]:
        """List models from Ollama Cloud API."""
        api_key = kwargs.get("api_key")
        try:
            client = ollama.Client(
                host=base_url or cls.DEFAULT_BASE_URL,
                headers=cls._auth_headers(api_key),
            )
            result = client.list()
            return [m.model for m in result.models if m.model is not None]
        except Exception:
            return []

    @classmethod
    def get_default_model(cls, base_url: str | None = None, **kwargs: Any) -> str:
        """Return default model for Ollama Cloud (hardcoded)."""
        return cls.DEFAULT_MODEL

    @classmethod
    def is_configured(cls, config: ProviderConfig) -> bool:
        """Check if api_key is configured."""
        return bool(config.api_key)

    @classmethod
    def create_llm(
        cls,
        model: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.0,
        reasoning_effort: str | None = None,
        **kwargs: Any,
    ) -> ChatOllama:
        """Create a ChatOllama instance for Ollama Cloud.

        Note on **not calling super()**:
        ------------------------------------------------------------------------
        ChatOllama is a thin wrapper; all parameters we need (``model``,
        ``base_url``) overlap with the superclass.  Calling ``super()`` and *then*
        injecting ``client_kwargs`` adds brittle indirection — the parent
        OllamaProvider.create_llm injects no auth header yet still yields a
        ChatOllama, so there is no re-use benefit.  Building the instance
        directly keeps authentication, timeout configuration, and metadata
        attachment in one place, making the flow self-contained and
        straightforward to audit.
        """
        api_key = kwargs.pop("api_key", None)
        config = kwargs.pop("config", None)
        effective_api_key = cls._resolve_api_key(api_key, config)

        effective_base_url = base_url or cls.DEFAULT_BASE_URL

        if not model:
            model = cls.get_default_model(effective_base_url)

        if reasoning_effort == "none":
            kwargs["reasoning"] = False
        elif reasoning_effort is not None:
            kwargs["reasoning"] = reasoning_effort
        else:
            kwargs["reasoning"] = True

        client_kwargs: dict[str, Any] = {}
        if effective_api_key:
            client_kwargs["headers"] = {"Authorization": f"Bearer {effective_api_key}"}

        llm = ChatOllama(
            model=model,
            base_url=effective_base_url,
            temperature=temperature,
            client_kwargs=client_kwargs if client_kwargs else None,
            sync_client_kwargs={"timeout": 120},
            async_client_kwargs={"timeout": 120},
            **kwargs,
        )
        ctx_len = cls.get_actual_context_length(
            model,
            api_key=effective_api_key,
            base_url=effective_base_url,
        )
        llm._esdc_context_length = ctx_len  # type: ignore[attr-defined]
        return llm

    @classmethod
    def test_connection(cls, config: ProviderConfig) -> tuple[bool, str]:
        """Test Ollama Cloud connection."""
        if not config.api_key:
            return False, "API key not configured"

        try:
            models = cls.list_models(
                base_url=None,
                api_key=config.api_key,
            )
            if models:
                return True, f"Connected. {len(models)} models available."
            return False, "No models found"
        except Exception as exc:
            return False, str(exc)
