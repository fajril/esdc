"""Tests for the LLM provider layer."""

import time
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from esdc.providers.base import ProviderConfig


class _FailingChatModel(BaseChatModel):
    @property
    def _llm_type(self) -> str:
        return "failing-chat-model"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager=None,
        **kwargs,
    ) -> ChatResult:
        raise RuntimeError("primary failed")


class _StaticChatModel(BaseChatModel):
    content: str

    @property
    def _llm_type(self) -> str:
        return "static-chat-model"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager=None,
        **kwargs,
    ) -> ChatResult:
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=self.content))]
        )


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


class TestCreateLlmFromConfigFallback:
    """Verify create_llm_from_config supports ordered provider fallbacks."""

    @pytest.mark.asyncio
    async def test_runtime_invoke_falls_back_to_next_provider(self):
        from esdc.providers import ProviderFallbackChatModel, create_llm_from_config

        with (
            patch(
                "esdc.providers.openai.OpenAIProvider.create_llm",
                return_value=_FailingChatModel(),
            ),
            patch(
                "esdc.providers.deepseek.DeepSeekProvider.create_llm",
                return_value=_StaticChatModel(content="fallback ok"),
            ),
        ):
            llm = create_llm_from_config(
                {
                    "name": "openai",
                    "provider_type": "openai",
                    "model": "gpt-4o-mini",
                    "api_key": "sk-openai",
                    "fallback_configs": [
                        {
                            "name": "deepseek",
                            "provider_type": "deepseek",
                            "model": "deepseek-v4-flash",
                            "api_key": "sk-deepseek",
                        }
                    ],
                }
            )

            assert isinstance(llm, ProviderFallbackChatModel)
            response = await llm.ainvoke([HumanMessage(content="hello")])
            assert response.content == "fallback ok"
            assert llm.last_provider_name == "deepseek"
            assert llm.last_model_name == "deepseek-v4-flash"

    def test_creation_skips_provider_that_fails_to_construct(self):
        from esdc.providers import create_llm_from_config

        fallback_model = _StaticChatModel(content="constructed fallback")

        with (
            patch(
                "esdc.providers.openai.OpenAIProvider.create_llm",
                side_effect=ValueError("bad config"),
            ),
            patch(
                "esdc.providers.deepseek.DeepSeekProvider.create_llm",
                return_value=fallback_model,
            ),
        ):
            llm = create_llm_from_config(
                {
                    "name": "openai",
                    "provider_type": "openai",
                    "model": "gpt-4o-mini",
                    "api_key": "sk-openai",
                    "fallback_configs": [
                        {
                            "name": "deepseek",
                            "provider_type": "deepseek",
                            "model": "deepseek-v4-flash",
                            "api_key": "sk-deepseek",
                        }
                    ],
                }
            )

        assert llm is fallback_model
        assert llm._esdc_provider_name == "deepseek"
        assert llm._esdc_model_name == "deepseek-v4-flash"

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


class TestTemperaturePassthrough:
    """Verify temperature flows from config to create_llm."""

    def test_temperature_passed_when_set(self):
        from unittest.mock import patch

        from esdc.providers import _create_single_llm_from_config

        captured_kwargs = {}

        class _FakeProvider:
            @classmethod
            def create_llm(cls, model=None, base_url=None, api_key=None, **kwargs):
                captured_kwargs.update(dict(kwargs, model=model))

                class _FakeLLM:
                    pass

                return _FakeLLM()

        with patch("esdc.providers.get_provider", return_value=_FakeProvider):
            _create_single_llm_from_config(
                {"provider_type": "fake", "model": "m", "temperature": 0.9}
            )
        assert captured_kwargs.get("temperature") == 0.9

    def test_temperature_zero_is_honored(self):
        from unittest.mock import patch

        from esdc.providers import _create_single_llm_from_config

        captured_kwargs = {}

        class _FakeProvider:
            @classmethod
            def create_llm(cls, model=None, base_url=None, api_key=None, **kwargs):
                captured_kwargs.update(dict(kwargs, model=model))

                class _FakeLLM:
                    pass

                return _FakeLLM()

        with patch("esdc.providers.get_provider", return_value=_FakeProvider):
            _create_single_llm_from_config(
                {"provider_type": "fake", "model": "m", "temperature": 0.0}
            )
        assert captured_kwargs.get("temperature") == 0.0

    def test_temperature_omitted_when_absent(self):
        from unittest.mock import patch

        from esdc.providers import _create_single_llm_from_config

        captured_kwargs = {}

        class _FakeProvider:
            @classmethod
            def create_llm(cls, model=None, base_url=None, api_key=None, **kwargs):
                captured_kwargs.update(dict(kwargs, model=model))

                class _FakeLLM:
                    pass

                return _FakeLLM()

        with patch("esdc.providers.get_provider", return_value=_FakeProvider):
            _create_single_llm_from_config({"provider_type": "fake", "model": "m"})
        assert "temperature" not in captured_kwargs


class TestOAuthRefreshExpiresAt:
    """Verify refreshed OAuth tokens store an absolute expires_at."""

    def test_oauth_refresh_sets_absolute_expires_at(self):
        """expires_at must be now + expires_in, not the raw expires_in.

        The raw expires_in value is an epoch timestamp in 1970 and would
        always be treated as 'expired'.
        """
        from esdc.providers.openai import OpenAIProvider

        config = ProviderConfig(
            name="openai",
            provider_type="openai",
            model="gpt-4o",
            base_url="",
            api_key="",
            auth_method="oauth",
            oauth={"access_token": "old", "refresh_token": "r1", "expires_at": 1},
        )
        fresh = {"access_token": "new", "expires_in": 3600}

        with (
            patch("esdc.auth.refresh_access_token", return_value=dict(fresh)),
            patch.object(
                OpenAIProvider, "get_actual_context_length", return_value=8192
            ),
            patch("esdc.providers.openai.ChatOpenAI"),
        ):
            OpenAIProvider.create_llm(model="gpt-4o", config=config)

        assert config.oauth["expires_at"] >= int(time.time()) + 3500

    def test_oauth_refresh_sets_absolute_expires_at_in_test_connection(self):
        """Same fix applies to the refresh path inside test_connection."""
        from esdc.providers.openai import OpenAIProvider

        config = ProviderConfig(
            name="openai",
            provider_type="openai",
            model="gpt-4o",
            base_url="",
            api_key="",
            auth_method="oauth",
            oauth={"access_token": "old", "refresh_token": "r1", "expires_at": 1},
        )
        fresh = {"access_token": "new", "expires_in": 3600}

        with (
            patch("esdc.auth.refresh_access_token", return_value=dict(fresh)),
            patch.object(OpenAIProvider, "list_models", return_value=["gpt-4o"]),
            patch.object(OpenAIProvider, "create_llm") as mock_create_llm,
        ):
            mock_create_llm.return_value.invoke.return_value = None
            OpenAIProvider.test_connection(config)

        assert config.oauth["expires_at"] >= int(time.time()) + 3500


class TestOAuthRefreshPersistence:
    """Verify refreshed OAuth tokens are written back to persistent storage.

    A rotated refresh_token must not be kept in memory only.
    """

    def test_create_llm_persists_refreshed_oauth_tokens(self):
        """create_llm's refresh path calls Config.persist_provider_oauth."""
        from esdc.providers.openai import OpenAIProvider

        config = ProviderConfig(
            name="openai",
            provider_type="openai",
            model="gpt-4o",
            base_url="",
            api_key="",
            auth_method="oauth",
            oauth={"access_token": "old", "refresh_token": "r1", "expires_at": 1},
        )
        fresh = {"access_token": "new", "refresh_token": "rotated", "expires_in": 3600}

        with (
            patch("esdc.auth.refresh_access_token", return_value=dict(fresh)),
            patch.object(
                OpenAIProvider, "get_actual_context_length", return_value=8192
            ),
            patch("esdc.providers.openai.ChatOpenAI"),
            patch("esdc.configs.Config.persist_provider_oauth") as mock_persist,
        ):
            OpenAIProvider.create_llm(model="gpt-4o", config=config)

        mock_persist.assert_called_once_with("openai", config.oauth)
        assert config.oauth["refresh_token"] == "rotated"

    def test_test_connection_persists_refreshed_oauth_tokens(self):
        """test_connection's refresh path calls Config.persist_provider_oauth."""
        from esdc.providers.openai import OpenAIProvider

        config = ProviderConfig(
            name="openai",
            provider_type="openai",
            model="gpt-4o",
            base_url="",
            api_key="",
            auth_method="oauth",
            oauth={"access_token": "old", "refresh_token": "r1", "expires_at": 1},
        )
        fresh = {"access_token": "new", "refresh_token": "rotated", "expires_in": 3600}

        with (
            patch("esdc.auth.refresh_access_token", return_value=dict(fresh)),
            patch.object(OpenAIProvider, "list_models", return_value=["gpt-4o"]),
            patch.object(OpenAIProvider, "create_llm") as mock_create_llm,
            patch("esdc.configs.Config.persist_provider_oauth") as mock_persist,
        ):
            mock_create_llm.return_value.invoke.return_value = None
            OpenAIProvider.test_connection(config)

        mock_persist.assert_called_once_with("openai", config.oauth)
        assert config.oauth["refresh_token"] == "rotated"

    def test_create_llm_survives_persistence_failure(self):
        """A disk-persist failure must not break the LLM call."""
        from esdc.providers.openai import OpenAIProvider

        config = ProviderConfig(
            name="openai",
            provider_type="openai",
            model="gpt-4o",
            base_url="",
            api_key="",
            auth_method="oauth",
            oauth={"access_token": "old", "refresh_token": "r1", "expires_at": 1},
        )
        fresh = {"access_token": "new", "expires_in": 3600}

        with (
            patch("esdc.auth.refresh_access_token", return_value=dict(fresh)),
            patch.object(
                OpenAIProvider, "get_actual_context_length", return_value=8192
            ),
            patch("esdc.providers.openai.ChatOpenAI"),
            patch(
                "esdc.configs.Config.persist_provider_oauth",
                side_effect=OSError("disk full"),
            ),
        ):
            llm = OpenAIProvider.create_llm(model="gpt-4o", config=config)

        assert llm is not None
        assert config.oauth["access_token"] == "new"


class TestBoundedLlmKwargs:
    """Provider-type keyed output-cap and timeout kwargs mapping."""

    @pytest.mark.parametrize(
        ("provider_type", "token_key", "timeout_key"),
        [
            ("openai_compatible", "max_completion_tokens", "timeout"),
            ("openai", "max_completion_tokens", "timeout"),
            ("azure_openai", "max_completion_tokens", "timeout"),
            ("deepseek", "max_completion_tokens", "timeout"),
            ("anthropic", "max_tokens_to_sample", "timeout"),
            ("groq", "max_tokens", "timeout"),
            ("google", "max_tokens", "request_timeout"),
        ],
    )
    def test_bounded_llm_kwargs(self, provider_type, token_key, timeout_key):
        from esdc.providers import _bounded_llm_kwargs

        kwargs = _bounded_llm_kwargs(provider_type, 8192, 300)
        assert kwargs[token_key] == 8192
        assert kwargs[timeout_key] == 300.0

    def test_ollama_bounded_llm_kwargs(self):
        from esdc.providers import _bounded_llm_kwargs

        kwargs = _bounded_llm_kwargs("ollama", 8192, 300)
        assert kwargs == {
            "num_predict": 8192,
            "sync_client_kwargs": {"timeout": 300.0},
            "async_client_kwargs": {"timeout": 300.0},
        }

    def test_ollama_cloud_bounded_llm_kwargs(self):
        from esdc.providers import _bounded_llm_kwargs

        kwargs = _bounded_llm_kwargs("ollama_cloud", 2048, 60)
        assert kwargs == {
            "num_predict": 2048,
            "sync_client_kwargs": {"timeout": 60.0},
            "async_client_kwargs": {"timeout": 60.0},
        }

    def test_partial_bounds_disable_each_bound_independently(self):
        from esdc.providers import _bounded_llm_kwargs

        assert _bounded_llm_kwargs("openai", 8192, 0) == {"max_completion_tokens": 8192}
        assert _bounded_llm_kwargs("openai", 0, 300) == {"timeout": 300.0}

    def test_zero_bounds_produce_no_kwargs(self):
        from esdc.providers import _bounded_llm_kwargs

        assert _bounded_llm_kwargs("openai_compatible", 0, 0) == {}

    def test_unknown_provider_type_raises(self):
        from esdc.providers import _bounded_llm_kwargs

        with pytest.raises(ValueError, match="Unknown provider type"):
            _bounded_llm_kwargs("mystery_provider", 8192, 300)

    def test_bounded_kwargs_match_installed_constructor_signatures(self):
        """Dependency drift must fail loudly, not silently drop a bound."""
        import inspect

        from langchain_anthropic import ChatAnthropic
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain_groq import ChatGroq
        from langchain_ollama import ChatOllama
        from langchain_openai import AzureChatOpenAI, ChatOpenAI

        assert "max_completion_tokens" in inspect.signature(ChatOpenAI).parameters
        assert "timeout" in inspect.signature(ChatOpenAI).parameters
        assert "max_completion_tokens" in inspect.signature(AzureChatOpenAI).parameters
        assert "timeout" in inspect.signature(AzureChatOpenAI).parameters
        assert "max_tokens_to_sample" in inspect.signature(ChatAnthropic).parameters
        assert "timeout" in inspect.signature(ChatAnthropic).parameters
        assert "max_tokens" in inspect.signature(ChatGroq).parameters
        assert "timeout" in inspect.signature(ChatGroq).parameters
        assert "num_predict" in inspect.signature(ChatOllama).parameters
        assert "sync_client_kwargs" in inspect.signature(ChatOllama).parameters
        assert "async_client_kwargs" in inspect.signature(ChatOllama).parameters
        assert "max_tokens" in inspect.signature(ChatGoogleGenerativeAI).parameters
        assert "request_timeout" in inspect.signature(ChatGoogleGenerativeAI).parameters


class TestCreateLlmFromConfigBounds:
    """Bounds must reach every primary and fallback provider client."""

    def test_bounds_propagate_to_primary_and_fallback(self):
        from esdc.providers import create_llm_from_config

        captured: list[dict[str, object]] = []

        class _RecordingProvider:
            @classmethod
            def create_llm(
                cls,
                model=None,
                base_url=None,
                api_key=None,
                **kwargs,
            ):
                captured.append(dict(kwargs, model=model))
                return _StaticChatModel(content="ok")

        with patch("esdc.providers.get_provider", return_value=_RecordingProvider):
            create_llm_from_config(
                {
                    "provider_type": "openai",
                    "model": "gpt-4o-mini",
                    "api_key": "sk-primary",
                    "fallback_configs": [
                        {
                            "provider_type": "openai",
                            "model": "gpt-4o",
                            "api_key": "sk-fallback",
                        }
                    ],
                },
                max_output_tokens=8192,
                timeout_seconds=300.0,
            )

        assert len(captured) == 2
        for kwargs in captured:
            assert kwargs["max_completion_tokens"] == 8192
            assert kwargs["timeout"] == 300.0

    def test_no_bounds_leaves_factory_unbounded(self):
        from esdc.providers import create_llm_from_config

        captured: list[dict[str, object]] = []

        class _RecordingProvider:
            @classmethod
            def create_llm(
                cls,
                model=None,
                base_url=None,
                api_key=None,
                **kwargs,
            ):
                captured.append(dict(kwargs, model=model))
                return _StaticChatModel(content="ok")

        with patch("esdc.providers.get_provider", return_value=_RecordingProvider):
            create_llm_from_config(
                {
                    "provider_type": "openai",
                    "model": "gpt-4o-mini",
                    "api_key": "sk-test",
                }
            )

        assert len(captured) == 1
        assert "max_completion_tokens" not in captured[0]
        assert "timeout" not in captured[0]


class TestProviderFallbackChatModelMismatch:
    """Verify ProviderFallbackChatModel rejects mismatched config lists."""

    def test_fallback_model_rejects_mismatched_name_lists(self):
        """A models/provider_names length mismatch must raise.

        It must not silently truncate the fallback chain.
        """
        from esdc.providers import ProviderFallbackChatModel

        model = ProviderFallbackChatModel(
            models=[_FailingChatModel(), _FailingChatModel()],
            provider_names=["a"],  # mismatch: 2 models, 1 name
            model_names=["m1", "m2"],
        )
        with pytest.raises(ValueError):
            model._generate([])
