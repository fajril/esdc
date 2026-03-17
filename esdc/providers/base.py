from dataclasses import dataclass
from typing import Literal

ProviderType = Literal["openai", "anthropic", "ollama", "custom"]


@dataclass
class ProviderConfig:
    name: str
    provider_type: ProviderType
    api_key: str = ""
    base_url: str = ""
    model: str = ""


class Provider:
    def __init__(self, config: ProviderConfig):
        self.config = config

    def chat(self, messages: list[dict]) -> str:
        raise NotImplementedError("Subclasses must implement chat()")
