from typing import Any

from langchain_openai import ChatOpenAI
from openai import OpenAI

from esdc.auth import get_valid_token, is_token_expired, start_oauth_flow
from esdc.providers.base import Provider, ProviderConfig


class OpenAIProvider(Provider):
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
            models = client.models.list()
            return [m.id for m in models if "gpt" in m.id.lower()]  # type: ignore[union-attr]
        except Exception:
            return []

    @classmethod
    def get_default_model(cls, **kwargs: Any) -> str:
        """Get default model."""
        return cls.DEFAULT_MODEL

    @classmethod
    def is_configured(cls, config: ProviderConfig) -> bool:
        """Check if OpenAI is configured (OAuth or API key)."""
        if config.api_key:
            return True
        if config.oauth and "access_token" in config.oauth:
            return True
        return False

    @classmethod
    def create_llm(
        cls,
        model: str | None = None,
        api_key: str | None = None,
        config: ProviderConfig | None = None,
        temperature: float = 0.0,
        **kwargs,
    ) -> ChatOpenAI:
        """Create a ChatOpenAI instance."""
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
                    "Connected but no models available. Your account may not have access.",
                )

            llm = cls.create_llm(
                model=config.model or cls.DEFAULT_MODEL,
                api_key=api_key,
            )
            llm.invoke("Hello")
            return True, f"Connected. Available models: {len(models)} models"
        except Exception as e:
            return False, str(e)
