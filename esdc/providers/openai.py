# Standard library
import logging
import time
from typing import Any

# Third-party
from langchain_openai import ChatOpenAI
from openai import OpenAI

logger = logging.getLogger(__name__)

# Local
from esdc.auth import is_token_expired, start_oauth_flow  # noqa: E402
from esdc.providers.base import Provider, ProviderConfig  # noqa: E402


class OpenAIProvider(Provider):
    """Provider implementation for OpenAI API."""

    NAME = "OpenAI"
    DEFAULT_MODEL = "gpt-4o-mini"
    BASE_URL = "https://models.inference.ai.azure.com"

    CONTEXT_LENGTHS = {
        "gpt-4o": 128000,
        "gpt-4o-mini": 128000,
        "gpt-4-turbo": 128000,
        "gpt-4": 8192,
        "gpt-3.5-turbo": 16385,
        "gpt-3.5-turbo-16k": 16385,
        "gpt-4-32k": 32768,
        "o1": 128000,
        "o1-mini": 128000,
        "o3": 200000,
        "o3-mini": 128000,
    }

    @classmethod
    def list_models(cls, api_key: str | None = None, **kwargs: Any) -> list[str]:
        """List available OpenAI models."""
        try:
            client = OpenAI(
                api_key=api_key or "",
                base_url=cls.BASE_URL,
            )

            logger.debug("[INFERENCE] openai_list_models | base_url=%s", cls.BASE_URL)
            list_start = time.perf_counter()

            models = client.models.list()

            elapsed_ms = (time.perf_counter() - list_start) * 1000
            model_list = [m.id for m in models if "gpt" in m.id.lower()]  # type: ignore[union-attr]
            logger.debug(
                "[INFERENCE] openai_list_models_complete | elapsed=%.2fms | models=%d",
                elapsed_ms,
                len(model_list),
            )
            return model_list
        except Exception as e:
            logger.debug(
                "[INFERENCE] openai_list_models_error | error=%s", str(e)[:100]
            )
            return []

    @classmethod
    def get_default_model(cls, **kwargs: Any) -> str:
        """Get default model."""
        return cls.DEFAULT_MODEL

    @classmethod
    def is_configured(cls, config: ProviderConfig) -> bool:
        """Check if OpenAI is configured (OAuth or API key)."""
        return bool(config.api_key or (config.oauth and "access_token" in config.oauth))

    @classmethod
    def create_llm(
        cls,
        model: str | None = None,
        api_key: str | None = None,
        config: ProviderConfig | None = None,
        temperature: float = 0.0,
        reasoning_effort: str | None = None,
        **kwargs,
    ) -> ChatOpenAI:
        """Create a ChatOpenAI instance.

        Args:
            model: Model name (e.g. "gpt-4o-mini", "o3-mini")
            api_key: API key or OAuth access token
            config: Provider configuration (used for OAuth)
            temperature: Sampling temperature
            reasoning_effort: OpenAI reasoning effort level.
                Supported values: "none", "minimal", "low", "medium", "high", "xhigh".
                Only applicable to reasoning models (o-series, GPT-5).
            **kwargs: Additional keyword arguments passed to ChatOpenAI.
        """
        if not model:
            model = cls.get_default_model()

        effective_api_key = api_key

        if config and config.oauth:
            if not is_token_expired(config.oauth):
                effective_api_key = config.oauth.get("access_token")
            else:
                from esdc.auth import refresh_access_token

                refresh_token = config.oauth.get("refresh_token", "")
                new_tokens = refresh_access_token(refresh_token)
                config.oauth.update(new_tokens)
                config.oauth["expires_at"] = int(new_tokens.get("expires_in", 3600))
                effective_api_key = new_tokens.get("access_token")

        if reasoning_effort is not None:
            kwargs["extra_body"] = {
                **kwargs.get("extra_body", {}),
                "reasoning_effort": reasoning_effort,
            }

        return ChatOpenAI(
            model=model,
            api_key=effective_api_key or "",  # type: ignore[arg-type]
            base_url=cls.BASE_URL,
            temperature=temperature,
            **kwargs,
        )

    @classmethod
    def authenticate(cls) -> dict[str, Any]:
        """Start OAuth flow and return tokens."""
        return start_oauth_flow()

    @classmethod
    def test_connection(cls, config: ProviderConfig) -> tuple[bool, str]:
        """Test OpenAI connection."""
        try:
            if config.oauth and "access_token" in config.oauth:
                if is_token_expired(config.oauth):
                    from esdc.auth import refresh_access_token

                    new_tokens = refresh_access_token(
                        config.oauth.get("refresh_token", "")
                    )
                    config.oauth.update(new_tokens)
                    config.oauth["expires_at"] = int(new_tokens.get("expires_in", 3600))

                api_key = config.oauth.get("access_token")
            elif config.api_key:
                api_key = config.api_key
            else:
                return (
                    False,
                    "Not configured. Run 'esdc chat' to set up authentication.",
                )

            models = cls.list_models(api_key)
            if not models:
                return (
                    False,
                    "Connected but no models available. Your account may not have access.",  # noqa: E501
                )

            llm = cls.create_llm(
                model=config.model or cls.DEFAULT_MODEL,
                api_key=api_key,
            )

            logger.debug(
                "[INFERENCE] openai_test_connection_invoke | model=%s",
                config.model or cls.DEFAULT_MODEL,
            )
            invoke_start = time.perf_counter()

            llm.invoke("Hello")

            invoke_elapsed_ms = (time.perf_counter() - invoke_start) * 1000
            logger.debug(
                "[INFERENCE] openai_test_connection_complete | model=%s | elapsed=%.2fms",  # noqa: E501
                config.model or cls.DEFAULT_MODEL,
                invoke_elapsed_ms,
            )
            return True, f"Connected. Available models: {len(models)} models"
        except Exception as e:
            return False, str(e)
