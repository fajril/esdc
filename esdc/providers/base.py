# Standard library
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal

# Third-party
from langchain_core.language_models import BaseChatModel

ProviderType = Literal["ollama", "openai", "openai_compatible"]

DEFAULT_CONTEXT_LENGTH = 4096


@dataclass
class ProviderConfig:
    name: str
    provider_type: ProviderType
    model: str = ""
    base_url: str = ""
    api_key: str = ""
    auth_method: str = "api_key"
    oauth: dict[str, Any] = field(default_factory=dict)


class Provider(ABC):
    NAME: str = ""
    DEFAULT_MODEL: str = ""
    CONTEXT_LENGTHS: dict[str, int] = {}

    @classmethod
    @abstractmethod
    def list_models(cls, **kwargs: Any) -> list[str]:
        """List available models for this provider."""
        pass

    @classmethod
    @abstractmethod
    def get_default_model(cls, **kwargs: Any) -> str:
        """Return default model for this provider."""
        pass

    @classmethod
    @abstractmethod
    def is_configured(cls, config: ProviderConfig) -> bool:
        """Check if provider is properly configured."""
        pass

    @classmethod
    @abstractmethod
    def create_llm(cls, model: str | None = "", **kwargs: Any) -> BaseChatModel:
        """Create a LangChain chat model instance."""
        pass

    @classmethod
    def get_context_length(cls, model: str) -> int:
        """Return context length for a model in tokens.

        Args:
            model: The model name

        Returns:
            Context length in tokens, or DEFAULT_CONTEXT_LENGTH as fallback
        """
        model_lower = model.lower() if model else ""

        for key, length in cls.CONTEXT_LENGTHS.items():
            if key in model_lower:
                return length

        return DEFAULT_CONTEXT_LENGTH

    @classmethod
    def get_name(cls) -> str:
        """Human-readable provider name."""
        return cls.NAME

    @classmethod
    def test_connection(cls, config: ProviderConfig) -> tuple[bool, str]:
        """Test provider connection. Returns (success, message)."""
        try:
            llm = cls.create_llm(
                model=config.model or None,
                base_url=config.base_url or None,
                api_key=config.api_key or None,
            )
            llm.invoke("test")
            return True, "Connection successful"
        except Exception as e:
            return False, str(e)
