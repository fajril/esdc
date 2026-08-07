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
        assert (
            GroqProvider.get_context_length("unknown-model") == DEFAULT_CONTEXT_LENGTH
        )

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


class TestDeepSeekProvider:
    """Tests for DeepSeekProvider."""

    def test_is_configured(self):
        from esdc.providers.deepseek import DeepSeekProvider

        config = ProviderConfig(name="test", provider_type="deepseek", api_key="sk-xxx")
        assert DeepSeekProvider.is_configured(config) is True

        config_missing = ProviderConfig(
            name="test", provider_type="deepseek", api_key=""
        )
        assert DeepSeekProvider.is_configured(config_missing) is False

    def test_get_default_model(self):
        from esdc.providers.deepseek import DeepSeekProvider

        assert DeepSeekProvider.get_default_model() == "deepseek-v4-flash"

    def test_get_context_length(self):
        from esdc.providers.deepseek import DeepSeekProvider

        assert DeepSeekProvider.get_context_length("deepseek-v4-flash") == 1_000_000
        assert DeepSeekProvider.get_context_length("deepseek-v4-pro") == 1_000_000
        assert DeepSeekProvider.get_context_length("deepseek-chat") == 1_000_000
        assert DeepSeekProvider.get_context_length("deepseek-reasoner") == 1_000_000
        assert DeepSeekProvider.get_context_length("unknown") == DEFAULT_CONTEXT_LENGTH

    @patch("esdc.providers.deepseek.ChatOpenAI")
    def test_create_llm(self, mock_chat_cls):
        from esdc.providers.deepseek import DeepSeekProvider

        mock_instance = MagicMock()
        mock_chat_cls.return_value = mock_instance

        with patch.object(
            DeepSeekProvider, "get_actual_context_length", return_value=0
        ):
            llm = DeepSeekProvider.create_llm(
                model="deepseek-v4-pro",
                api_key="sk-xxx",
                temperature=0.2,
            )

        mock_chat_cls.assert_called_once_with(
            model="deepseek-v4-pro",
            api_key="sk-xxx",
            base_url=DeepSeekProvider.BASE_URL,
            temperature=0.2,
        )
        assert llm is mock_instance

    @patch("esdc.providers.deepseek.ChatOpenAI")
    def test_create_llm_reasoning_none_disables_thinking(self, mock_chat_cls):
        from esdc.providers.deepseek import DeepSeekProvider

        mock_chat_cls.return_value = MagicMock()

        with patch.object(
            DeepSeekProvider, "get_actual_context_length", return_value=0
        ):
            DeepSeekProvider.create_llm(
                model="deepseek-v4-flash",
                api_key="sk-xxx",
                reasoning_effort="none",
            )

        call_kwargs = mock_chat_cls.call_args.kwargs
        assert call_kwargs["extra_body"]["thinking"]["type"] == "disabled"
        assert "reasoning_effort" not in call_kwargs

    @patch("esdc.providers.deepseek.ChatOpenAI")
    def test_create_llm_reasoning_low_maps_to_high(self, mock_chat_cls):
        from esdc.providers.deepseek import DeepSeekProvider

        mock_chat_cls.return_value = MagicMock()

        with patch.object(
            DeepSeekProvider, "get_actual_context_length", return_value=0
        ):
            DeepSeekProvider.create_llm(
                model="deepseek-v4-flash",
                api_key="sk-xxx",
                reasoning_effort="low",
            )

        call_kwargs = mock_chat_cls.call_args.kwargs
        assert call_kwargs["reasoning_effort"] == "high"
        assert call_kwargs["extra_body"]["thinking"]["type"] == "enabled"

    @patch("esdc.providers.deepseek.ChatOpenAI")
    def test_create_llm_reasoning_xhigh_maps_to_max(self, mock_chat_cls):
        from esdc.providers.deepseek import DeepSeekProvider

        mock_chat_cls.return_value = MagicMock()

        with patch.object(
            DeepSeekProvider, "get_actual_context_length", return_value=0
        ):
            DeepSeekProvider.create_llm(
                model="deepseek-v4-flash",
                api_key="sk-xxx",
                reasoning_effort="xhigh",
            )

        call_kwargs = mock_chat_cls.call_args.kwargs
        assert call_kwargs["reasoning_effort"] == "max"
        assert call_kwargs["extra_body"]["thinking"]["type"] == "enabled"


class TestOpenAICompatibleCreateLLMNoLeak:
    """Ensure config kwarg doesn't leak into ChatOpenAI constructor."""

    @patch("esdc.providers.openai_compatible.ChatOpenAI")
    def test_create_llm_config_not_passed_to_chatopenai(self, mock_chat_cls):
        from esdc.providers.base import ProviderConfig
        from esdc.providers.openai_compatible import OpenAICompatibleProvider

        mock_instance = MagicMock()
        mock_instance._esdc_context_length = 0
        mock_chat_cls.return_value = mock_instance

        with patch.object(
            OpenAICompatibleProvider,
            "get_context_length_from_api",
            return_value=0,
        ):
            OpenAICompatibleProvider.create_llm(
                model="test-model",
                base_url="http://localhost:11434/v1",
                api_key="test-key",
                config=ProviderConfig(
                    name="test",
                    provider_type="openai_compatible",
                    model="test-model",
                    base_url="http://localhost:11434/v1",
                    api_key="test-key",
                ),
            )

        call_kwargs = mock_chat_cls.call_args[1]
        assert "config" not in call_kwargs, (
            f"'config' leaked into ChatOpenAI kwargs: {call_kwargs.keys()}"
        )


class TestAnthropicCreateLLMNoLeak:
    """Ensure config kwarg doesn't leak into ChatAnthropic constructor."""

    @patch("esdc.providers.anthropic.ChatAnthropic")
    def test_create_llm_config_not_passed_to_chatanthropic(self, mock_chat_cls):
        from esdc.providers.anthropic import AnthropicProvider
        from esdc.providers.base import ProviderConfig

        mock_instance = MagicMock()
        mock_instance._esdc_context_length = 0
        mock_chat_cls.return_value = mock_instance

        with patch.object(
            AnthropicProvider,
            "get_actual_context_length",
            return_value=0,
        ):
            AnthropicProvider.create_llm(
                model="claude-sonnet-4-6",
                api_key="test-key",
                config=ProviderConfig(
                    name="test",
                    provider_type="anthropic",
                    model="claude-sonnet-4-6",
                    api_key="test-key",
                ),
            )

        call_kwargs = mock_chat_cls.call_args[1]
        assert "config" not in call_kwargs, (
            f"'config' leaked into ChatAnthropic kwargs: {call_kwargs.keys()}"
        )


class TestGroqCreateLLMNoLeak:
    """Ensure config kwarg doesn't leak into ChatGroq constructor."""

    @patch("esdc.providers.groq.ChatGroq")
    def test_create_llm_config_not_passed_to_chatgroq(self, mock_chat_cls):
        from esdc.providers.base import ProviderConfig
        from esdc.providers.groq import GroqProvider

        mock_instance = MagicMock()
        mock_instance._esdc_context_length = 0
        mock_chat_cls.return_value = mock_instance

        with patch.object(
            GroqProvider,
            "get_actual_context_length",
            return_value=0,
        ):
            GroqProvider.create_llm(
                model="llama-3.3-70b-versatile",
                api_key="gsk-test",
                config=ProviderConfig(
                    name="test",
                    provider_type="groq",
                    model="llama-3.3-70b-versatile",
                    api_key="gsk-test",
                ),
            )

        call_kwargs = mock_chat_cls.call_args[1]
        assert "config" not in call_kwargs, (
            f"'config' leaked into ChatGroq kwargs: {call_kwargs.keys()}"
        )


class TestDeepSeekCreateLLMNoLeak:
    """Ensure config kwarg doesn't leak into ChatOpenAI constructor."""

    @patch("esdc.providers.deepseek.ChatOpenAI")
    def test_create_llm_config_not_passed_to_chatopenai(self, mock_chat_cls):
        from esdc.providers.base import ProviderConfig
        from esdc.providers.deepseek import DeepSeekProvider

        mock_instance = MagicMock()
        mock_instance._esdc_context_length = 0
        mock_chat_cls.return_value = mock_instance

        with patch.object(
            DeepSeekProvider,
            "get_actual_context_length",
            return_value=0,
        ):
            DeepSeekProvider.create_llm(
                model="deepseek-v4-flash",
                api_key="sk-test",
                config=ProviderConfig(
                    name="test",
                    provider_type="deepseek",
                    model="deepseek-v4-flash",
                    api_key="sk-test",
                ),
            )

        call_kwargs = mock_chat_cls.call_args[1]
        assert "config" not in call_kwargs, (
            f"'config' leaked into ChatOpenAI kwargs: {call_kwargs.keys()}"
        )


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
            "deepseek",
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
        assert PROVIDER_NAMES["deepseek"] == "DeepSeek"

    def test_get_provider(self):
        from esdc.providers import get_provider
        from esdc.providers.anthropic import AnthropicProvider
        from esdc.providers.azure_openai import AzureOpenAIProvider
        from esdc.providers.deepseek import DeepSeekProvider
        from esdc.providers.google import GoogleProvider
        from esdc.providers.groq import GroqProvider

        assert get_provider("anthropic") is AnthropicProvider
        assert get_provider("google") is GoogleProvider
        assert get_provider("azure_openai") is AzureOpenAIProvider
        assert get_provider("groq") is GroqProvider
        assert get_provider("deepseek") is DeepSeekProvider
        assert get_provider("nonexistent") is None

    def test_provider_type_literal(self):
        from esdc.providers.base import ProviderType

        assert "anthropic" in ProviderType.__args__
        assert "google" in ProviderType.__args__
        assert "azure_openai" in ProviderType.__args__
        assert "groq" in ProviderType.__args__
        assert "deepseek" in ProviderType.__args__


class TestOpenAICompatibleContextLength:
    """Test context length resolution for OpenAI-compatible providers."""

    @staticmethod
    def _make_model_obj(
        id: str = "test-model",
        dict_extra: dict | None = None,
        model_extra: dict | None = None,
    ):
        class FakeModel:
            pass

        obj = FakeModel()
        obj.id = id
        if dict_extra:
            for k, v in dict_extra.items():
                obj.__dict__[k] = v
        obj.model_extra = model_extra or {}
        return obj

    def test_extract_context_length_from_meta_n_ctx_train(self):
        from esdc.providers.openai_compatible import OpenAICompatibleProvider

        model_obj = self._make_model_obj(
            id="Qwen3.6-27B-Q4_K_M.gguf",
            model_extra={
                "meta": {
                    "n_ctx_train": 262144,
                    "n_vocab": 248320,
                    "n_embd": 5120,
                }
            },
        )

        result = OpenAICompatibleProvider._extract_context_length(model_obj)
        assert result == 262144

    def test_extract_context_length_from_top_level_max_model_len(self):
        from esdc.providers.openai_compatible import OpenAICompatibleProvider

        model_obj = self._make_model_obj(
            dict_extra={"max_model_len": 32768},
            model_extra={},
        )

        result = OpenAICompatibleProvider._extract_context_length(model_obj)
        assert result == 32768

    def test_extract_context_length_from_model_extra_top_level(self):
        from esdc.providers.openai_compatible import OpenAICompatibleProvider

        model_obj = self._make_model_obj(
            model_extra={"max_model_len": 65536},
        )

        result = OpenAICompatibleProvider._extract_context_length(model_obj)
        assert result == 65536

    def test_extract_context_length_returns_zero_when_no_metadata(self):
        from esdc.providers.openai_compatible import OpenAICompatibleProvider

        model_obj = self._make_model_obj(model_extra={})

        result = OpenAICompatibleProvider._extract_context_length(model_obj)
        assert result == 0

    def test_get_context_length_static_dict_prefix_match(self):
        from esdc.providers.openai_compatible import OpenAICompatibleProvider

        assert (
            OpenAICompatibleProvider.get_context_length("Qwen3.6-27B-Q4_K_M.gguf")
            == 262144
        )
        assert (
            OpenAICompatibleProvider.get_context_length("Qwen2.5-Coder-32B") == 128000
        )
        assert OpenAICompatibleProvider.get_context_length("deepseek-v3") == 128000

    def test_get_context_length_falls_back_to_default(self):
        from esdc.providers.openai_compatible import OpenAICompatibleProvider

        assert (
            OpenAICompatibleProvider.get_context_length("unknown-model")
            == DEFAULT_CONTEXT_LENGTH
        )
        assert DEFAULT_CONTEXT_LENGTH == 32768

    @patch("esdc.providers.openai_compatible.OpenAI")
    def test_get_context_length_from_api_reads_n_ctx_train(self, mock_openai_cls):
        from esdc.providers.openai_compatible import OpenAICompatibleProvider

        mock_model = self._make_model_obj(
            id="Qwen3.6-27B-Q4_K_M.gguf",
            model_extra={
                "meta": {
                    "n_ctx_train": 262144,
                    "n_vocab": 248320,
                    "n_embd": 5120,
                }
            },
        )

        mock_client = MagicMock()
        mock_client.models.list.return_value = [mock_model]
        mock_openai_cls.return_value = mock_client

        result = OpenAICompatibleProvider.get_context_length_from_api(
            "Qwen3.6-27B-Q4_K_M.gguf",
            base_url="http://localhost:8888/v1",
        )
        assert result == 262144

    @patch("esdc.providers.openai_compatible.OpenAI")
    def test_get_context_length_api_fallback_to_static(self, mock_openai_cls):
        from esdc.providers.openai_compatible import OpenAICompatibleProvider

        mock_client = MagicMock()
        mock_client.models.list.side_effect = Exception("API error")
        mock_openai_cls.return_value = mock_client

        result = OpenAICompatibleProvider.get_actual_context_length(
            "Qwen3.6-27B-Q4_K_M.gguf",
            base_url="http://localhost:8888/v1",
        )
        assert result == 262144

    @patch("esdc.providers.openai_compatible.OpenAI")
    def test_get_context_length_api_fallback_to_default(self, mock_openai_cls):
        from esdc.providers.openai_compatible import OpenAICompatibleProvider

        mock_client = MagicMock()
        mock_client.models.list.side_effect = Exception("API error")
        mock_openai_cls.return_value = mock_client

        result = OpenAICompatibleProvider.get_actual_context_length(
            "totally-unknown-model",
            base_url="http://localhost:8888/v1",
        )
        assert result == DEFAULT_CONTEXT_LENGTH
        assert result == 32768
