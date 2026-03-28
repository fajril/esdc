from unittest.mock import MagicMock, patch

from esdc.providers.base import Provider, ProviderConfig


def test_ollama_get_context_length_from_api():
    """Test fetching context length from Ollama API."""
    from esdc.providers.ollama import OllamaProvider

    # Mock the ollama client
    mock_client = MagicMock()
    mock_client.show.return_value = {
        "model_info": {
            "gemma3.context_length": 131072,
            "some_other_field": "value",
        }
    }

    with patch("ollama.Client", return_value=mock_client):
        result = OllamaProvider.get_context_length_from_api("gemma3:latest")

        assert result == 131072
        mock_client.show.assert_called_once_with("gemma3:latest")


def test_ollama_get_context_length_from_api_fallback():
    """Test fallback to hardcoded mapping when API fails."""
    from esdc.providers.ollama import OllamaProvider

    # Mock the ollama client to raise exception
    with patch("ollama.Client") as mock_client_class:
        mock_client_class.return_value.show.side_effect = Exception("API error")

        result = OllamaProvider.get_context_length_from_api("llama3.2")

        # Should fallback to hardcoded mapping
        assert result == 128000


def test_provider_config_validation():
    config = ProviderConfig(
        name="openai", provider_type="openai", api_key="sk-test", model="gpt-4o"
    )
    assert config.name == "openai"
    assert config.api_key == "sk-test"
    assert config.provider_type == "openai"
    assert config.model == "gpt-4o"


def test_openai_provider_config():
    from esdc.providers.base import ProviderConfig

    config = ProviderConfig(
        name="test_openai",
        provider_type="openai",
        model="gpt-4o-mini",
    )
    assert config.name == "test_openai"
    assert config.provider_type == "openai"
    assert config.model == "gpt-4o-mini"
