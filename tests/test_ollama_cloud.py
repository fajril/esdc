"""Tests for OllamaCloudProvider — ChatOllama-based implementation."""

from unittest.mock import MagicMock, patch

from esdc.providers.base import DEFAULT_CONTEXT_LENGTH, ProviderConfig
from esdc.providers.ollama_cloud import OllamaCloudProvider


class TestOllamaCloudProvider:
    def test_name(self):
        assert OllamaCloudProvider.NAME == "Ollama Cloud"

    def test_base_url(self):
        assert OllamaCloudProvider.BASE_URL == "https://ollama.com"

    def test_default_model(self):
        assert OllamaCloudProvider.get_default_model() == "llama3.2"

    def test_is_configured_with_key(self):
        config = ProviderConfig(
            name="test", provider_type="ollama_cloud", api_key="sk-xxx"
        )
        assert OllamaCloudProvider.is_configured(config) is True

    def test_is_configured_without_key(self):
        config = ProviderConfig(name="test", provider_type="ollama_cloud", api_key="")
        assert OllamaCloudProvider.is_configured(config) is False

    def test_context_length_known(self):
        assert OllamaCloudProvider.get_context_length("kimi-k2.5") == 262144

    def test_context_length_unknown(self):
        assert (
            OllamaCloudProvider.get_context_length("unknown-model")
            == DEFAULT_CONTEXT_LENGTH
        )

    @patch("ollama.Client")
    def test_list_models(self, mock_client_cls):
        mock_client = MagicMock()
        mock_model = MagicMock()
        mock_model.model = "llama3.2"
        mock_client.list.return_value.models = [mock_model]
        mock_client_cls.return_value = mock_client

        result = OllamaCloudProvider.list_models(api_key="test-key")
        assert result == ["llama3.2"]

    @patch("ollama.Client")
    def test_list_models_failure(self, mock_client_cls):
        mock_client_cls.side_effect = Exception("API error")
        result = OllamaCloudProvider.list_models(api_key="test-key")
        assert result == []

    @patch("esdc.providers.ollama_cloud.ChatOllama")
    @patch("esdc.providers.ollama_cloud.ollama.Client")
    def test_create_llm(self, mock_ollama_client, mock_chat_cls):
        mock_instance = MagicMock()
        mock_chat_cls.return_value = mock_instance
        mock_ollama_client.return_value.show.return_value = {}

        llm = OllamaCloudProvider.create_llm(
            model="llama3.2",
            api_key="test-key",
            temperature=0.5,
        )
        assert llm is mock_instance
        call_kwargs = mock_chat_cls.call_args[1]
        assert call_kwargs["model"] == "llama3.2"
        assert call_kwargs["base_url"] == "https://ollama.com"
        assert call_kwargs["temperature"] == 0.5
        assert "Authorization" in call_kwargs["client_kwargs"]["headers"]

    @patch("esdc.providers.ollama_cloud.ChatOllama")
    @patch("esdc.providers.ollama_cloud.ollama.Client")
    def test_create_llm_reasoning(self, mock_ollama_client, mock_chat_cls):
        mock_instance = MagicMock()
        mock_chat_cls.return_value = mock_instance
        mock_ollama_client.return_value.show.return_value = {}

        OllamaCloudProvider.create_llm(
            model="llama3.2",
            api_key="test-key",
            reasoning_effort="none",
        )
        call_kwargs = mock_chat_cls.call_args[1]
        assert call_kwargs["reasoning"] is False

    @patch("esdc.providers.ollama_cloud.ChatOllama")
    @patch("esdc.providers.ollama_cloud.ollama.Client")
    def test_create_llm_no_key(self, mock_ollama_client, mock_chat_cls):
        mock_instance = MagicMock()
        mock_chat_cls.return_value = mock_instance
        mock_ollama_client.return_value.show.return_value = {}

        OllamaCloudProvider.create_llm(model="llama3.2")
        call_kwargs = mock_chat_cls.call_args[1]
        # No auth headers when no api_key
        assert call_kwargs.get("client_kwargs") is None

    @patch("ollama.Client")
    def test_test_connection_success(self, mock_client_cls):
        mock_client = MagicMock()
        mock_model = MagicMock()
        mock_model.model = "llama3.2"
        mock_client.list.return_value.models = [mock_model]
        mock_client_cls.return_value = mock_client

        config = ProviderConfig(
            name="test",
            provider_type="ollama_cloud",
            api_key="test-key",
        )
        success, message = OllamaCloudProvider.test_connection(config)
        assert success is True
        assert "1 models" in message

    def test_test_connection_no_key(self):
        config = ProviderConfig(
            name="test",
            provider_type="ollama_cloud",
            api_key="",
        )
        success, message = OllamaCloudProvider.test_connection(config)
        assert success is False
        assert "API key not configured" in message

    @patch("esdc.providers.ollama_cloud.ChatOllama")
    @patch("esdc.providers.ollama_cloud.ollama.Client")
    def test_create_llm_sets_context_length_attribute(
        self, mock_ollama_client, mock_chat_cls
    ):
        """Created LLM instance must carry _esdc_context_length metadata."""
        mock_instance = MagicMock()
        mock_chat_cls.return_value = mock_instance
        mock_ollama_client.return_value.show.return_value = {}

        llm = OllamaCloudProvider.create_llm(
            model="kimi-k2.5",
        )
        assert llm is mock_instance
        assert llm._esdc_context_length == 262144  # type: ignore[attr-defined]

    @patch("esdc.providers.ollama_cloud.ChatOllama")
    @patch("esdc.providers.ollama_cloud.ollama.Client")
    def test_create_llm_context_length_from_api(
        self, mock_ollama_client, mock_chat_cls
    ):
        """Happy path: fetch context length from Ollama API modelinfo."""
        mock_instance = MagicMock()
        mock_chat_cls.return_value = mock_instance

        mock_info = MagicMock()
        mock_info.modelinfo = {"general.context_length": 32000}
        mock_ollama_client.return_value.show.return_value = mock_info

        llm = OllamaCloudProvider.create_llm(
            model="llama-test",
        )
        assert llm._esdc_context_length == 32000  # type: ignore[attr-defined]

    def test_inherits_from_ollama_provider(self):
        from esdc.providers.ollama import OllamaProvider

        assert issubclass(OllamaCloudProvider, OllamaProvider)
