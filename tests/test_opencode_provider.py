"""Tests for the OpenCode Go provider."""

from types import SimpleNamespace
from unittest.mock import patch

from esdc.providers.base import ProviderConfig
from esdc.providers.opencode import OpencodeProvider


def _fake_model(model_id: str, max_model_len: int | None = None):
    obj = SimpleNamespace(id=model_id)
    obj.__dict__["max_model_len"] = max_model_len
    return obj


class TestOpencodeProviderConfig:
    def test_constants(self):
        assert OpencodeProvider.NAME == "OpenCode Go"
        assert OpencodeProvider.BASE_URL == "https://opencode.ai/zen/go/v1"
        assert OpencodeProvider.DEFAULT_MODEL == "deepseek-v4-flash"

    def test_is_configured(self):
        configured = ProviderConfig(name="oc", provider_type="opencode", api_key="sk-x")
        missing = ProviderConfig(name="oc", provider_type="opencode", api_key="")
        assert OpencodeProvider.is_configured(configured) is True
        assert OpencodeProvider.is_configured(missing) is False

    def test_get_default_model(self):
        assert OpencodeProvider.get_default_model() == "deepseek-v4-flash"


class TestOpencodeHeaders:
    def test_default_headers_shape(self):
        headers = OpencodeProvider._default_headers()
        assert headers["x-opencode-session"].startswith("esdc-")
        assert headers["User-Agent"].startswith("esdc/")

    def test_env_overrides_session(self, monkeypatch):
        monkeypatch.setenv("ESDC_OPENCODE_SESSION", "custom-session-123")
        assert (
            OpencodeProvider._default_headers()["x-opencode-session"]
            == "custom-session-123"
        )


class TestOpencodeCreateLlm:
    @patch("esdc.providers.opencode.OpenAI")
    @patch("esdc.providers.openai_compatible.ChatOpenAI")
    def test_defaults_and_headers(self, mock_chat, mock_openai):
        mock_openai.return_value.models.list.return_value = []

        OpencodeProvider.create_llm(api_key="sk-x")

        kwargs = mock_chat.call_args.kwargs
        assert kwargs["model"] == "deepseek-v4-flash"
        assert kwargs["base_url"] == "https://opencode.ai/zen/go/v1"
        assert kwargs["default_headers"]["x-opencode-session"].startswith("esdc-")
        assert kwargs["default_headers"]["User-Agent"].startswith("esdc/")

    @patch("esdc.providers.opencode.OpenAI")
    @patch("esdc.providers.openai_compatible.ChatOpenAI")
    def test_caller_headers_are_preserved(self, mock_chat, mock_openai):
        mock_openai.return_value.models.list.return_value = []

        OpencodeProvider.create_llm(api_key="sk-x", default_headers={"X-Custom": "1"})

        headers = mock_chat.call_args.kwargs["default_headers"]
        assert headers["X-Custom"] == "1"
        assert "x-opencode-session" in headers


class TestOpencodeListAndContext:
    @patch("esdc.providers.opencode.OpenAI")
    def test_list_models_passes_headers(self, mock_openai):
        mock_openai.return_value.models.list.return_value = [
            SimpleNamespace(id="glm-5.3"),
            SimpleNamespace(id="deepseek-v4-flash"),
        ]

        models = OpencodeProvider.list_models(api_key="sk-x")

        assert models == ["glm-5.3", "deepseek-v4-flash"]
        headers = mock_openai.call_args.kwargs["default_headers"]
        assert "x-opencode-session" in headers
        assert mock_openai.call_args.kwargs["base_url"] == OpencodeProvider.BASE_URL

    @patch("esdc.providers.opencode.OpenAI")
    def test_list_models_swallows_errors(self, mock_openai):
        mock_openai.return_value.models.list.side_effect = RuntimeError("boom")
        assert OpencodeProvider.list_models(api_key="sk-x") == []

    @patch("esdc.providers.opencode.OpenAI")
    def test_context_length_from_api(self, mock_openai):
        mock_openai.return_value.models.list.return_value = [
            _fake_model("deepseek-v4-flash", max_model_len=128000),
        ]

        assert (
            OpencodeProvider.get_context_length_from_api("deepseek-v4-flash", "sk-x")
            == 128000
        )
        headers = mock_openai.call_args.kwargs["default_headers"]
        assert "x-opencode-session" in headers

    @patch("esdc.providers.opencode.OpenAI")
    def test_context_length_api_failure_returns_zero(self, mock_openai):
        mock_openai.return_value.models.list.side_effect = RuntimeError("boom")
        assert OpencodeProvider.get_context_length_from_api("x", "sk-x") == 0
