"""Tests for OllamaCloudProvider."""

from unittest.mock import MagicMock, patch

import pytest

from esdc.providers.base import DEFAULT_CONTEXT_LENGTH, ProviderConfig
from esdc.providers.ollama_cloud import OllamaCloudProvider


class TestOllamaCloudProvider:
    """Tests for OllamaCloudProvider."""

    def test_name(self):
        assert OllamaCloudProvider.NAME == "Ollama Cloud"

    def test_base_url(self):
        assert OllamaCloudProvider.BASE_URL == "https://ollama.com/v1"

    def test_default_model(self):
        assert OllamaCloudProvider.get_default_model() == "llama3.2"

    def test_is_configured(self):
        config = ProviderConfig(
            name="test", provider_type="ollama_cloud", api_key="sk-xxx"
        )
        assert OllamaCloudProvider.is_configured(config) is True

        config_missing = ProviderConfig(
            name="test", provider_type="ollama_cloud", api_key=""
        )
        assert OllamaCloudProvider.is_configured(config_missing) is False

    def test_context_length(self):
        assert OllamaCloudProvider.get_context_length("llama3.2") == 128000
        assert (
            OllamaCloudProvider.get_context_length("unknown-model")
            == DEFAULT_CONTEXT_LENGTH
        )

    @patch("esdc.providers.ollama_cloud.OpenAI")
    def test_list_models(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_model = MagicMock()
        mock_model.id = "llama3.2"
        mock_client.models.list.return_value = [mock_model]
        mock_openai_cls.return_value = mock_client

        result = OllamaCloudProvider.list_models(api_key="test-key")
        assert result == ["llama3.2"]
        mock_openai_cls.assert_called_once_with(
            base_url="https://ollama.com/v1",
            api_key="test-key",
        )

    @patch("esdc.providers.ollama_cloud.OpenAI")
    def test_list_models_failure(self, mock_openai_cls):
        mock_openai_cls.side_effect = Exception("API error")
        result = OllamaCloudProvider.list_models(api_key="test-key")
        assert result == []

    @patch("esdc.providers.ollama_cloud.ChatOpenAI")
    def test_create_llm(self, mock_chat_cls):
        mock_instance = MagicMock()
        mock_chat_cls.return_value = mock_instance

        llm = OllamaCloudProvider.create_llm(
            model="llama3.2",
            api_key="test-key",
            temperature=0.5,
        )
        mock_chat_cls.assert_called_once_with(
            model="llama3.2",
            base_url="https://ollama.com/v1",
            api_key="test-key",
            temperature=0.5,
        )
        assert llm is mock_instance

    def test_create_llm_no_model(self):
        with pytest.raises(ValueError, match="model is required"):
            OllamaCloudProvider.create_llm(model="", api_key="test-key")

    @patch("esdc.providers.ollama_cloud.OllamaCloudProvider.list_models")
    def test_test_connection_success(self, mock_list):
        mock_list.return_value = ["llama3.2", "deepseek-v3.1"]
        config = ProviderConfig(
            name="test",
            provider_type="ollama_cloud",
            api_key="test-key",
        )
        success, message = OllamaCloudProvider.test_connection(config)
        assert success is True
        assert "2 models" in message

    @patch("esdc.providers.ollama_cloud.OllamaCloudProvider.list_models")
    def test_test_connection_empty(self, mock_list):
        mock_list.return_value = []
        config = ProviderConfig(
            name="test",
            provider_type="ollama_cloud",
            api_key="test-key",
        )
        success, message = OllamaCloudProvider.test_connection(config)
        assert success is False
        assert "No models" in message

    def test_test_connection_no_key(self):
        config = ProviderConfig(
            name="test",
            provider_type="ollama_cloud",
            api_key="",
        )
        success, message = OllamaCloudProvider.test_connection(config)
        assert success is False
        assert "API key not configured" in message
