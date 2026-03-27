from typing import Any

from langchain_ollama import ChatOllama

from esdc.providers.base import Provider, ProviderConfig


class OllamaProvider(Provider):
    NAME = "Ollama"
    DEFAULT_BASE_URL = "http://localhost:11434"
    DEFAULT_MODEL = "llama3.2"

    CONTEXT_LENGTHS = {
        "llama3.2": 128000,
        "llama3.1": 128000,
        "llama3.0": 128000,
        "llama2": 4096,
        "llama2-70b": 4096,
        "mistral": 32000,
        "mixtral": 32000,
        "qwen2.5": 128000,
        "qwen2": 131072,
        "qwen": 8192,
        "deepseek-coder": 16384,
        "deepseek-v2": 128000,
        "codellama": 16384,
        "phi3": 4096,
        "phi4": 128000,
        "gemma": 8192,
        "gemma2": 8192,
        "command-r": 128000,
        "aya": 8192,
        "OL-4": 128000,
    }

    CODE_CAPABLE_MODELS = [
        "deepseek-coder",
        "codellama",
        "llama3.2",
        "mistral",
        "qwen2.5-coder",
    ]

    @classmethod
    def list_models(cls, base_url: str | None = None) -> list[str]:
        """List available models from Ollama server."""
        try:
            from ollama import list as ollama_list

            result = ollama_list()
            return [m.model for m in result.models]  # type: ignore[union-attr]
        except Exception:
            return []

    @classmethod
    def get_default_model(cls, base_url: str | None = None) -> str:
        """Get default model, preferring code-capable models."""
        models = cls.list_models(base_url)
        if not models:
            return cls.DEFAULT_MODEL

        for preferred in cls.CODE_CAPABLE_MODELS:
            for m in models:
                if preferred.lower() in m.lower():
                    return m

        return models[0] if models else cls.DEFAULT_MODEL

    @classmethod
    def is_configured(cls, config: ProviderConfig) -> bool:
        """Check if Ollama is configured and server is running."""
        if not config.model and not cls.get_default_model(config.base_url or None):
            return False
        return True

    @classmethod
    def create_llm(
        cls,
        model: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.0,
        **kwargs,
    ) -> ChatOllama:
        """Create a ChatOllama instance."""
        if not model:
            model = cls.get_default_model(base_url)

        return ChatOllama(
            model=model,
            base_url=base_url or cls.DEFAULT_BASE_URL,
            temperature=temperature,
            **kwargs,
        )

    @classmethod
    def test_connection(cls, config: ProviderConfig) -> tuple[bool, str]:
        """Test Ollama server connection."""
        try:
            models = cls.list_models(config.base_url)
            if not models:
                return False, "No models found. Pull a model with: ollama pull <model>"

            default_model = cls.get_default_model(config.base_url)
            llm = cls.create_llm(
                model=config.model or default_model,
                base_url=config.base_url,
            )
            llm.invoke("Hello")
            return True, f"Connected. Available models: {', '.join(models)}"
        except Exception as e:
            error_msg = str(e)
            if "ConnectionError" in type(e).__name__:
                return (
                    False,
                    "Ollama server not running. Start with: ollama serve",
                )
            return False, error_msg
