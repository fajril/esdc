# Standard library

# Third-party
from langchain_core.language_models import BaseChatModel

# Local
from esdc.chat.agent import create_agent
from esdc.configs import Config
from esdc.providers import create_llm_from_config


class AgentFactory:
    """Factory for creating ESDC agents."""

    def create_agent(self, llm: BaseChatModel):
        """Create ESDC agent with tools.

        Args:
            llm: Language model to use

        Returns:
            Compiled agent graph
        """
        return create_agent(llm, checkpointer=None)

    def create_llm(self) -> BaseChatModel:
        """Create LLM from config.

        Returns:
            Configured language model

        Raises:
            ValueError: If no provider is configured
        """
        provider_config = Config.get_provider_config()

        if not provider_config:
            raise ValueError(
                "No provider configured. Please run 'esdc provider add' first."
            )

        provider_name = provider_config.get("provider", "ollama")
        model = provider_config.get("model")
        base_url = provider_config.get("base_url")
        api_key = provider_config.get("api_key")

        config = {
            "provider_type": provider_name,
            "model": model,
            "base_url": base_url,
            "api_key": api_key,
        }

        return create_llm_from_config(config)
