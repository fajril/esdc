from unittest.mock import MagicMock, patch

from esdc.providers.base import ProviderConfig


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


class TestOllamaCreateLLMNoLeak:
    """Ensure config kwarg doesn't leak into ChatOllama constructor."""

    @patch("esdc.providers.ollama.ChatOllama")
    def test_create_llm_config_not_passed_to_chatollama(self, mock_chat_cls):
        from esdc.providers.ollama import OllamaProvider

        mock_instance = MagicMock()
        mock_instance._esdc_context_length = 0
        mock_chat_cls.return_value = mock_instance

        with patch.object(
            OllamaProvider,
            "get_actual_context_length",
            return_value=0,
        ):
            OllamaProvider.create_llm(
                model="llama3.2",
                base_url="http://localhost:11434",
                config=ProviderConfig(
                    name="test",
                    provider_type="ollama",
                    model="llama3.2",
                ),
            )

        call_kwargs = mock_chat_cls.call_args[1]
        assert "config" not in call_kwargs, (
            f"'config' leaked into ChatOllama kwargs: {call_kwargs.keys()}"
        )


class TestCreateLlmFromConfigNoLeak:
    """Verify create_llm_from_config never leaks ProviderConfig into LLMs."""

    @patch("esdc.providers.openai_compatible.ChatOpenAI")
    def test_openai_compatible_no_config_leak(self, mock_cls):
        from esdc.providers import create_llm_from_config

        mock_instance = MagicMock()
        mock_instance._esdc_context_length = 0
        mock_cls.return_value = mock_instance

        with patch(
            "esdc.providers.openai_compatible.OpenAICompatibleProvider"
            ".get_context_length_from_api",
            return_value=0,
        ):
            create_llm_from_config(
                {
                    "provider_type": "openai_compatible",
                    "model": "test-model",
                    "base_url": "http://localhost:11434/v1",
                    "api_key": "test-key",
                }
            )

        call_kwargs = mock_cls.call_args[1]
        assert "config" not in call_kwargs

    @patch("esdc.providers.anthropic.ChatAnthropic")
    def test_anthropic_no_config_leak(self, mock_cls):
        from esdc.providers import create_llm_from_config

        mock_instance = MagicMock()
        mock_instance._esdc_context_length = 0
        mock_cls.return_value = mock_instance

        with patch(
            "esdc.providers.anthropic.AnthropicProvider.get_actual_context_length",
            return_value=0,
        ):
            create_llm_from_config(
                {
                    "provider_type": "anthropic",
                    "model": "claude-sonnet-4-6",
                    "api_key": "test-key",
                }
            )

        call_kwargs = mock_cls.call_args[1]
        assert "config" not in call_kwargs

    @patch("esdc.providers.groq.ChatGroq")
    def test_groq_no_config_leak(self, mock_cls):
        from esdc.providers import create_llm_from_config

        mock_instance = MagicMock()
        mock_instance._esdc_context_length = 0
        mock_cls.return_value = mock_instance

        with patch(
            "esdc.providers.groq.GroqProvider.get_actual_context_length",
            return_value=0,
        ):
            create_llm_from_config(
                {
                    "provider_type": "groq",
                    "model": "llama-3.3-70b-versatile",
                    "api_key": "gsk-test",
                }
            )

        call_kwargs = mock_cls.call_args[1]
        assert "config" not in call_kwargs

    @patch("esdc.providers.ollama.ChatOllama")
    def test_ollama_no_config_leak(self, mock_cls):
        from esdc.providers import create_llm_from_config

        mock_instance = MagicMock()
        mock_instance._esdc_context_length = 0
        mock_cls.return_value = mock_instance

        with patch(
            "esdc.providers.ollama.OllamaProvider.get_actual_context_length",
            return_value=0,
        ):
            create_llm_from_config(
                {
                    "provider_type": "ollama",
                    "model": "llama3.2",
                    "base_url": "http://localhost:11434",
                }
            )

        call_kwargs = mock_cls.call_args[1]
        assert "config" not in call_kwargs
