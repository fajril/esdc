from unittest.mock import MagicMock, patch

from esdc.providers.base import DEFAULT_CONTEXT_LENGTH, ProviderConfig


class TestAnthropicProvider:
    """Tests for AnthropicProvider."""

    def test_is_configured(self):
        from esdc.providers.anthropic import AnthropicProvider

        config = ProviderConfig(
            name="test", provider_type="anthropic", api_key="sk-xxx"
        )
        assert AnthropicProvider.is_configured(config) is True

        config_missing = ProviderConfig(
            name="test", provider_type="anthropic", api_key=""
        )
        assert AnthropicProvider.is_configured(config_missing) is False

    def test_list_models(self):
        from esdc.providers.anthropic import AnthropicProvider

        models = AnthropicProvider.list_models()
        assert len(models) > 0
        assert any("claude" in m.lower() for m in models)

    def test_get_default_model(self):
        from esdc.providers.anthropic import AnthropicProvider

        assert AnthropicProvider.get_default_model() == AnthropicProvider.DEFAULT_MODEL

    def test_get_context_length(self):
        from esdc.providers.anthropic import AnthropicProvider

        assert AnthropicProvider.get_context_length("claude-3-opus") == 200000
        assert AnthropicProvider.get_context_length("unknown") == DEFAULT_CONTEXT_LENGTH

    @patch("esdc.providers.anthropic.ChatAnthropic")
    def test_create_llm(self, mock_chat_cls):
        from esdc.providers.anthropic import AnthropicProvider

        mock_instance = MagicMock()
        mock_chat_cls.return_value = mock_instance

        llm = AnthropicProvider.create_llm(
            model="claude-3-5-sonnet",
            api_key="sk-xxx",
            temperature=0.5,
        )
        mock_chat_cls.assert_called_once_with(
            model_name="claude-3-5-sonnet",
            api_key="sk-xxx",
            temperature=0.5,
        )
        assert llm is mock_instance

    @patch("esdc.providers.anthropic.ChatAnthropic")
    def test_create_llm_reasoning_effort(self, mock_chat_cls):
        from esdc.providers.anthropic import AnthropicProvider

        mock_instance = MagicMock()
        mock_chat_cls.return_value = mock_instance

        AnthropicProvider.create_llm(
            model="claude-3-5-sonnet",
            api_key="sk-xxx",
            reasoning_effort="medium",
        )
        call_kwargs = mock_chat_cls.call_args.kwargs
        assert call_kwargs["extra_body"]["reasoning_effort"] == "medium"

    @patch("esdc.providers.anthropic.ChatAnthropic")
    def test_create_llm_default_model(self, mock_chat_cls):
        from esdc.providers.anthropic import AnthropicProvider

        AnthropicProvider.create_llm(api_key="sk-xxx")
        call_kwargs = mock_chat_cls.call_args.kwargs
        assert call_kwargs["model_name"] == AnthropicProvider.DEFAULT_MODEL


class TestGoogleProvider:
    """Tests for GoogleProvider."""

    def test_is_configured(self):
        from esdc.providers.google import GoogleProvider

        config = ProviderConfig(name="test", provider_type="google", api_key="sk-xxx")
        assert GoogleProvider.is_configured(config) is True

        config_missing = ProviderConfig(name="test", provider_type="google", api_key="")
        assert GoogleProvider.is_configured(config_missing) is False

    def test_list_models(self):
        from esdc.providers.google import GoogleProvider

        models = GoogleProvider.list_models()
        assert len(models) > 0
        assert any("gemini" in m.lower() for m in models)

    def test_get_default_model(self):
        from esdc.providers.google import GoogleProvider

        assert GoogleProvider.get_default_model() == GoogleProvider.DEFAULT_MODEL

    def test_get_context_length(self):
        from esdc.providers.google import GoogleProvider

        assert GoogleProvider.get_context_length("gemini-1.5-pro") == 2_097_152
        assert GoogleProvider.get_context_length("gemini-1.5-flash") == 1_048_576
        assert GoogleProvider.get_context_length("unknown") == DEFAULT_CONTEXT_LENGTH

    @patch("esdc.providers.google.ChatGoogleGenerativeAI")
    def test_create_llm(self, mock_chat_cls):
        from esdc.providers.google import GoogleProvider

        mock_instance = MagicMock()
        mock_chat_cls.return_value = mock_instance

        llm = GoogleProvider.create_llm(
            model="gemini-1.5-pro",
            api_key="gapi-xxx",
            temperature=0.3,
        )
        mock_chat_cls.assert_called_once_with(
            model="gemini-1.5-pro",
            google_api_key="gapi-xxx",
            temperature=0.3,
        )
        assert llm is mock_instance

    @patch("esdc.providers.google.ChatGoogleGenerativeAI")
    def test_create_llm_ignores_reasoning_effort(self, mock_chat_cls):
        from esdc.providers.google import GoogleProvider

        GoogleProvider.create_llm(
            model="gemini-1.5-pro",
            api_key="gapi-xxx",
            reasoning_effort="medium",
        )
        call_kwargs = mock_chat_cls.call_args.kwargs
        assert "reasoning_effort" not in call_kwargs
        assert "extra_body" not in call_kwargs


class TestAzureOpenAIProvider:
    """Tests for AzureOpenAIProvider."""

    def test_is_configured(self):
        from esdc.providers.azure_openai import AzureOpenAIProvider

        config = ProviderConfig(
            name="test",
            provider_type="azure_openai",
            api_key="sk-xxx",
            base_url="https://test.openai.azure.com/",
        )
        assert AzureOpenAIProvider.is_configured(config) is True

        config_missing = ProviderConfig(
            name="test", provider_type="azure_openai", api_key="", base_url=""
        )
        assert AzureOpenAIProvider.is_configured(config_missing) is False

        config_no_url = ProviderConfig(
            name="test", provider_type="azure_openai", api_key="sk-xxx", base_url=""
        )
        assert AzureOpenAIProvider.is_configured(config_no_url) is False

    def test_list_models(self):
        from esdc.providers.azure_openai import AzureOpenAIProvider

        models = AzureOpenAIProvider.list_models()
        assert len(models) > 0
        assert "gpt-4o" in models

    def test_get_default_model(self):
        from esdc.providers.azure_openai import AzureOpenAIProvider

        assert AzureOpenAIProvider.get_default_model() == "gpt-4o"

    def test_get_context_length(self):
        from esdc.providers.azure_openai import AzureOpenAIProvider

        assert AzureOpenAIProvider.get_context_length("gpt-4o") == 128000
        assert AzureOpenAIProvider.get_context_length("gpt-4") == 8192

    @patch("esdc.providers.azure_openai.AzureChatOpenAI")
    def test_create_llm(self, mock_chat_cls):
        from esdc.providers.azure_openai import AzureOpenAIProvider

        mock_instance = MagicMock()
        mock_chat_cls.return_value = mock_instance

        llm = AzureOpenAIProvider.create_llm(
            model="gpt-4o",
            api_key="sk-xxx",
            config=ProviderConfig(
                name="test",
                provider_type="azure_openai",
                base_url="https://test.openai.azure.com/",
            ),
            temperature=0.2,
        )
        mock_chat_cls.assert_called_once_with(
            azure_endpoint="https://test.openai.azure.com/",
            api_key="sk-xxx",
            azure_deployment="gpt-4o",
            api_version="2024-02-01",
            temperature=0.2,
        )
        assert llm is mock_instance

    @patch("esdc.providers.azure_openai.AzureChatOpenAI")
    def test_create_llm_reasoning_effort(self, mock_chat_cls):
        from esdc.providers.azure_openai import AzureOpenAIProvider

        mock_instance = MagicMock()
        mock_chat_cls.return_value = mock_instance

        AzureOpenAIProvider.create_llm(
            model="gpt-4o",
            api_key="sk-xxx",
            config=ProviderConfig(
                name="test",
                provider_type="azure_openai",
                base_url="https://test.openai.azure.com/",
            ),
            reasoning_effort="high",
        )
        call_kwargs = mock_chat_cls.call_args.kwargs
        assert call_kwargs["extra_body"]["reasoning_effort"] == "high"


class TestGroqProvider:
    """Tests for GroqProvider."""

    def test_is_configured(self):
        from esdc.providers.groq import GroqProvider

        config = ProviderConfig(name="test", provider_type="groq", api_key="gsk-xxx")
        assert GroqProvider.is_configured(config) is True

        config_missing = ProviderConfig(name="test", provider_type="groq", api_key="")
        assert GroqProvider.is_configured(config_missing) is False

    def test_get_default_model(self):
        from esdc.providers.groq import GroqProvider

        assert GroqProvider.get_default_model() == GroqProvider.DEFAULT_MODEL

    def test_get_context_length(self):
        from esdc.providers.groq import GroqProvider

        assert GroqProvider.get_context_length("llama-3.3-70b-versatile") == 128000
        assert GroqProvider.get_context_length("mixtral-8x7b") == 32768
        assert GroqProvider.get_context_length("unknown-model") == 4096

    @patch("esdc.providers.groq.ChatGroq")
    def test_create_llm(self, mock_chat_cls):
        from esdc.providers.groq import GroqProvider

        mock_instance = MagicMock()
        mock_chat_cls.return_value = mock_instance

        llm = GroqProvider.create_llm(
            model="llama-3.3-70b-versatile",
            api_key="gsk-xxx",
            temperature=0.1,
        )
        mock_chat_cls.assert_called_once_with(
            model="llama-3.3-70b-versatile",
            api_key="gsk-xxx",
            base_url=GroqProvider.BASE_URL,
            temperature=0.1,
        )
        assert llm is mock_instance

    @patch("esdc.providers.groq.ChatGroq")
    def test_create_llm_ignores_reasoning_effort(self, mock_chat_cls):
        from esdc.providers.groq import GroqProvider

        GroqProvider.create_llm(
            model="llama-3.3-70b-versatile",
            api_key="gsk-xxx",
            reasoning_effort="low",
        )
        call_kwargs = mock_chat_cls.call_args.kwargs
        assert "reasoning_effort" not in call_kwargs
        assert "extra_body" not in call_kwargs


class TestProviderRegistry:
    """Tests for provider registry."""

    def test_all_providers_registered(self):
        from esdc.providers import PROVIDER_CLASSES, list_provider_types

        expected = {
            "ollama",
            "openai",
            "openai_compatible",
            "anthropic",
            "google",
            "azure_openai",
            "groq",
            "ollama_cloud",
        }
        assert set(PROVIDER_CLASSES.keys()) == expected
        assert set(list_provider_types()) == expected

    def test_provider_names(self):
        from esdc.providers import PROVIDER_NAMES

        assert PROVIDER_NAMES["anthropic"] == "Anthropic (Claude)"
        assert PROVIDER_NAMES["google"] == "Google (Gemini)"
        assert PROVIDER_NAMES["azure_openai"] == "Azure OpenAI"
        assert PROVIDER_NAMES["groq"] == "Groq"

    def test_get_provider(self):
        from esdc.providers import get_provider
        from esdc.providers.anthropic import AnthropicProvider
        from esdc.providers.azure_openai import AzureOpenAIProvider
        from esdc.providers.google import GoogleProvider
        from esdc.providers.groq import GroqProvider

        assert get_provider("anthropic") is AnthropicProvider
        assert get_provider("google") is GoogleProvider
        assert get_provider("azure_openai") is AzureOpenAIProvider
        assert get_provider("groq") is GroqProvider
        assert get_provider("nonexistent") is None

    def test_provider_type_literal(self):
        from esdc.providers.base import ProviderType

        assert "anthropic" in ProviderType.__args__
        assert "google" in ProviderType.__args__
        assert "azure_openai" in ProviderType.__args__
        assert "groq" in ProviderType.__args__
