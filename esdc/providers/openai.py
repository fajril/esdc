from esdc.providers.base import Provider, ProviderConfig


class OpenAIProvider(Provider):
    def __init__(self, api_key: str = "", model: str = "gpt-4o"):
        self.api_key = api_key
        self.model = model
        self.client = None

    @classmethod
    def from_config(cls, config: ProviderConfig):
        """Create provider from ProviderConfig."""
        return cls(api_key=config.api_key or "", model=config.model or "gpt-4o")

    def generate_sql(self, schema: str, user_query: str) -> str:
        # TODO: Implement actual OpenAI API call
        # For now, return a simple query
        return f"-- SQL for: {user_query}\nSELECT * FROM project_resources LIMIT 10;"

    def chat(self, messages: list[dict]) -> str:
        raise NotImplementedError("Chat not implemented for OpenAIProvider")
