"""Tests for config_wizard module."""

from unittest.mock import MagicMock, patch

from esdc.config_wizard import (
    _fetch_models,
    _mask_value,
    _prompt_for_config_value,
    run_wizard,
)
from esdc.configs import KEY_DESCRIPTIONS, MODEL_SECTIONS, SETTINGS_SECTIONS, Config


class TestMaskValue:
    """Test the _mask_value helper."""

    def test_api_key(self):
        assert _mask_value("api_key", "sk-1234567890abcdef") == "sk****ef"

    def test_short_api_key(self):
        assert _mask_value("api_key", "abcd") == "****"

    def test_non_sensitive(self):
        assert _mask_value("name", "Alice") == "Alice"


class TestPromptForConfigValue:
    """Test _prompt_for_config_value with mocked questionary."""

    @patch("esdc.config_wizard.questionary.select")
    def test_boolean(self, mock_select):
        mock_select.return_value.ask.return_value = "True"
        result = _prompt_for_config_value("api.verify_ssl", False)
        assert result is True

    @patch("esdc.config_wizard.questionary.select")
    def test_enum(self, mock_select):
        mock_select.return_value.ask.return_value = "DEBUG"
        result = _prompt_for_config_value("logging.level", "INFO")
        assert result == "DEBUG"

    @patch("esdc.config_wizard.questionary.password")
    def test_password(self, mock_pass):
        mock_pass.return_value.ask.return_value = "secret"
        result = _prompt_for_config_value("providers.my.api_key", "old")
        assert result == "secret"

    @patch("esdc.config_wizard.questionary.text")
    def test_text(self, mock_text):
        mock_text.return_value.ask.return_value = "http://new.url"
        result = _prompt_for_config_value("api_url", "http://old.url")
        assert result == "http://new.url"

    @patch("esdc.config_wizard.questionary.text")
    def test_integer(self, mock_text):
        mock_text.return_value.ask.return_value = "42"
        result = _prompt_for_config_value("cache.sql_ttl", 3600)
        assert result == 42


class TestFetchModels:
    """Test model list fetching."""

    @patch("esdc.config_wizard.PROVIDER_CLASSES")
    def test_fetch_success(self, mock_classes):
        mock_provider = MagicMock()
        mock_provider.list_models.return_value = ["gpt-4o", "gpt-4o-mini"]
        mock_provider.get_default_model.return_value = "gpt-4o"
        mock_classes.get.return_value = mock_provider
        result = _fetch_models("openai")
        assert result == ["gpt-4o", "gpt-4o-mini"]

    @patch("esdc.config_wizard.PROVIDER_CLASSES")
    def test_fetch_failure_fallback(self, mock_classes):
        mock_provider = MagicMock()
        mock_provider.list_models.side_effect = Exception("API error")
        mock_provider.get_default_model.return_value = "gpt-4o"
        mock_classes.get.return_value = mock_provider
        result = _fetch_models("openai")
        assert "gpt-4o" in result

    @patch("esdc.config_wizard.PROVIDER_CLASSES")
    def test_empty_list_fallback(self, mock_classes):
        mock_provider = MagicMock()
        mock_provider.list_models.return_value = []
        mock_provider.get_default_model.return_value = "gpt-4o"
        mock_classes.get.return_value = mock_provider
        result = _fetch_models("openai")
        assert "gpt-4o" in result

    @patch("esdc.config_wizard.PROVIDER_CLASSES")
    def test_fetch_with_kwargs(self, mock_classes):
        mock_provider = MagicMock()
        mock_provider.list_models.return_value = ["model-a", "model-b"]
        mock_provider.get_default_model.return_value = "model-a"
        mock_classes.get.return_value = mock_provider

        result = _fetch_models("openai_compatible", base_url="http://localhost:8000")
        mock_provider.list_models.assert_called_once_with(
            base_url="http://localhost:8000"
        )
        assert result == ["model-a", "model-b"]


class TestRunWizard:
    """Test the main run_wizard dispatch logic.

    The wizard is now a nested tree: top menu -> "Models & endpoints" ->
    "Chat providers" -> individual provider actions. Each "__back__" pops
    one level (mocked via questionary.select, since _select_with_back calls
    it under the hood).
    """

    @patch("esdc.config_wizard.Config.init_config")
    @patch("esdc.config_wizard.questionary.select")
    @patch("esdc.config_wizard._add_provider_flow")
    def test_add_provider(self, mock_add, mock_select, mock_init):
        mock_select.return_value.ask.side_effect = [
            "Models & endpoints",
            "Chat providers",
            "Add provider",
            "__back__",
            "__back__",
            "Exit",
        ]
        run_wizard()
        mock_add.assert_called_once()

    @patch("esdc.config_wizard.Config.init_config")
    @patch("esdc.config_wizard.questionary.select")
    @patch("esdc.config_wizard._show_config_flow")
    def test_show_config(self, mock_show, mock_select, mock_init):
        mock_select.return_value.ask.side_effect = ["Show config", "Exit"]
        run_wizard()
        mock_show.assert_called_once()

    @patch("esdc.config_wizard.Config.init_config")
    @patch("esdc.config_wizard.questionary.select")
    def test_exit(self, mock_select, mock_init):
        mock_select.return_value.ask.side_effect = ["Exit"]
        run_wizard()

    @patch("esdc.config_wizard.Config.init_config")
    @patch("esdc.config_wizard.questionary.select")
    @patch("esdc.config_wizard.rich_print")
    def test_keyboard_interrupt(self, mock_print, mock_select, mock_init):
        mock_select.return_value.ask.side_effect = KeyboardInterrupt()
        run_wizard()

    @patch("esdc.config_wizard.Config.init_config")
    @patch("esdc.config_wizard.questionary.select")
    @patch("esdc.config_wizard._edit_provider_flow")
    def test_edit_provider(self, mock_edit, mock_select, mock_init):
        mock_select.return_value.ask.side_effect = [
            "Models & endpoints",
            "Chat providers",
            "Edit provider",
            "__back__",
            "__back__",
            "Exit",
        ]
        run_wizard()
        mock_edit.assert_called_once()


class TestMenuDispatch:
    """Top menu -> submenu routes to the right underlying flow/section."""

    @patch("esdc.config_wizard.Config.init_config")
    @patch("esdc.config_wizard.questionary.select")
    @patch("esdc.config_wizard._edit_section")
    def test_corpus_models_routes_to_edit_section(
        self, mock_edit_section, mock_select, mock_init
    ):
        mock_select.return_value.ask.side_effect = [
            "Models & endpoints",
            "Corpus models",
            "__back__",
            "Exit",
        ]
        run_wizard()
        mock_edit_section.assert_called_once_with(
            "Corpus models", MODEL_SECTIONS["Corpus models"]
        )

    @patch("esdc.config_wizard.Config.init_config")
    @patch("esdc.config_wizard.questionary.select")
    @patch("esdc.config_wizard._edit_section")
    def test_logging_routes_to_edit_section(
        self, mock_edit_section, mock_select, mock_init
    ):
        mock_select.return_value.ask.side_effect = [
            "Settings",
            "Logging",
            "__back__",
            "Exit",
        ]
        run_wizard()
        mock_edit_section.assert_called_once_with(
            "Logging", SETTINGS_SECTIONS["Logging"]
        )


class TestPromptInstructionLine:
    """Descriptions move off the message and onto questionary's instruction=."""

    @patch("esdc.config_wizard.questionary.select")
    def test_corpus_model_uses_instruction_param(self, mock_select):
        mock_select.return_value.ask.return_value = "main"

        _prompt_for_config_value("corpus.metadata_model", "main")

        _, kwargs = mock_select.call_args
        message = (
            mock_select.call_args.args[0] if mock_select.call_args.args else ""
        )
        description = KEY_DESCRIPTIONS["corpus.metadata_model"]
        assert kwargs.get("instruction") == description
        assert description not in message


class TestPhoenixSurfaced:
    """phoenix.* keys appear in Observability even when absent from config.yaml."""

    @patch("esdc.config_wizard._select_with_back", return_value="__back__")
    def test_phoenix_keys_present_with_defaults(
        self, mock_select_back, isolated_config
    ):
        """With an empty config.yaml, phoenix.* still surface from get_defaults."""
        from esdc.config_wizard import _edit_section

        _edit_section("Observability", SETTINGS_SECTIONS["Observability"])

        choices = mock_select_back.call_args.kwargs.get("choices") or (
            mock_select_back.call_args.args[1]
        )
        labels = {c.value: str(c.title) for c in choices}
        assert "phoenix.enabled = False" in labels["phoenix.enabled"]
        assert "http://localhost:4317" in labels["phoenix.collector_endpoint"]
        assert "iris" in labels["phoenix.project_name"]


class TestResetMenu:
    """Reset submenu: single-key reset and reset-all behind a confirm gate."""

    @patch("esdc.config_wizard.Config.reset_config")
    @patch("esdc.config_wizard.questionary.confirm")
    @patch("esdc.config_wizard.questionary.select")
    @patch("esdc.config_wizard.rich_print")
    def test_reset_all_confirmed(
        self, mock_print, mock_select, mock_confirm, mock_reset
    ):
        from esdc.config_wizard import _reset_menu

        mock_select.return_value.ask.side_effect = ["Reset all", "__back__"]
        mock_confirm.return_value.ask.return_value = True

        _reset_menu()

        mock_reset.assert_called_once_with()

    @patch("esdc.config_wizard.Config.reset_config")
    @patch("esdc.config_wizard.questionary.confirm")
    @patch("esdc.config_wizard.questionary.select")
    @patch("esdc.config_wizard.rich_print")
    def test_reset_all_declined(
        self, mock_print, mock_select, mock_confirm, mock_reset
    ):
        from esdc.config_wizard import _reset_menu

        mock_select.return_value.ask.side_effect = ["Reset all", "__back__"]
        mock_confirm.return_value.ask.return_value = False

        _reset_menu()

        mock_reset.assert_not_called()

    @patch("esdc.config_wizard.Config.reset_config")
    @patch(
        "esdc.config_wizard.Config.get_all_config_flat",
        return_value={"logging.level": "INFO"},
    )
    @patch("esdc.config_wizard.questionary.confirm")
    @patch("esdc.config_wizard.questionary.select")
    @patch("esdc.config_wizard.rich_print")
    def test_reset_single_key(
        self, mock_print, mock_select, mock_confirm, mock_flat, mock_reset
    ):
        from esdc.config_wizard import _reset_menu

        mock_select.return_value.ask.side_effect = [
            "Reset key",
            "Logging",
            "logging.level",
            "__back__",
        ]
        mock_confirm.return_value.ask.return_value = True

        _reset_menu()

        mock_reset.assert_called_once_with("logging.level")


class TestProviderCRUDFlows:
    """Test provider add/edit/remove flows."""

    @patch("esdc.config_wizard._fetch_models", return_value=["gpt-4o"])
    @patch("esdc.config_wizard.Config.get_default_provider", return_value="")
    @patch("esdc.config_wizard.Config.get_providers", return_value={})
    @patch("esdc.config_wizard.Config.save_provider")
    @patch("esdc.config_wizard.Config.set_default_provider")
    @patch("esdc.config_wizard.questionary.select")
    @patch("esdc.config_wizard.questionary.text")
    @patch("esdc.config_wizard.questionary.password")
    @patch("esdc.config_wizard.questionary.confirm")
    @patch("esdc.config_wizard.rich_print")
    def test_add_provider_full(
        self,
        mock_print,
        mock_confirm,
        mock_password,
        mock_text,
        mock_select,
        mock_set_default,
        mock_save,
        mock_get_providers,
        mock_get_default,
        mock_fetch_models,
    ):
        from esdc.config_wizard import _add_provider_flow

        mock_select.return_value.ask.side_effect = [
            "openai",
            "(none)",
            "gpt-4o",
        ]
        mock_password.return_value.ask.return_value = "sk-test"
        mock_confirm.return_value.ask.return_value = False

        _add_provider_flow()

        mock_save.assert_called_once()
        mock_set_default.assert_called_once_with("openai")

    @patch("esdc.config_wizard._fetch_models", return_value=["llama3"])
    @patch("esdc.config_wizard.Config.get_default_provider", return_value="openai")
    @patch("esdc.config_wizard.Config.get_providers", return_value={})
    @patch("esdc.config_wizard.Config.save_provider")
    @patch("esdc.config_wizard.Config.set_default_provider")
    @patch("esdc.config_wizard.questionary.select")
    @patch("esdc.config_wizard.questionary.text")
    @patch("esdc.config_wizard.questionary.password")
    @patch("esdc.config_wizard.questionary.confirm")
    @patch("esdc.config_wizard.rich_print")
    def test_add_provider_no_default(
        self,
        mock_print,
        mock_confirm,
        mock_password,
        mock_text,
        mock_select,
        mock_set_default,
        mock_save,
        mock_get_providers,
        mock_get_default,
        mock_fetch_models,
    ):
        from esdc.config_wizard import _add_provider_flow

        mock_select.return_value.ask.side_effect = [
            "ollama",
            "llama3",
        ]
        mock_text.return_value.ask.side_effect = [
            "http://localhost:11434",
        ]
        mock_confirm.return_value.ask.return_value = False

        _add_provider_flow()

        mock_save.assert_called_once()
        mock_set_default.assert_not_called()

    @patch("esdc.config_wizard.Config.get_providers")
    @patch("esdc.config_wizard.Config.set_default_provider")
    @patch("esdc.config_wizard.questionary.select")
    @patch("esdc.config_wizard.rich_print")
    def test_set_default(
        self, mock_print, mock_select, mock_set_default, mock_get_providers
    ):
        from esdc.config_wizard import _set_default_provider_flow

        mock_get_providers.return_value = {"my-openai": {"provider_type": "openai"}}
        mock_select.return_value.ask.return_value = "my-openai"

        _set_default_provider_flow()

        mock_set_default.assert_called_once_with("my-openai")

    @patch("esdc.config_wizard.Config.get_provider_order")
    @patch("esdc.config_wizard.Config.get_providers")
    @patch("esdc.config_wizard.Config.set_provider_order")
    @patch("esdc.config_wizard.questionary.select")
    @patch("esdc.config_wizard.questionary.confirm")
    @patch("esdc.config_wizard.rich_print")
    def test_set_provider_order(
        self,
        mock_print,
        mock_confirm,
        mock_select,
        mock_set_order,
        mock_get_providers,
        mock_get_order,
    ):
        from esdc.config_wizard import _set_provider_order_flow

        mock_get_providers.return_value = {
            "deepseek": {"provider_type": "deepseek"},
            "openai": {"provider_type": "openai"},
        }
        mock_get_order.return_value = []
        mock_select.return_value.ask.side_effect = ["deepseek", "__done__"]
        mock_confirm.return_value.ask.return_value = True

        _set_provider_order_flow()

        mock_set_order.assert_called_once_with(["deepseek", "openai"])

    @patch("esdc.config_wizard.Config.get_providers")
    @patch("esdc.config_wizard.Config.remove_provider")
    @patch("esdc.config_wizard.questionary.select")
    @patch("esdc.config_wizard.questionary.confirm")
    @patch("esdc.config_wizard.rich_print")
    def test_remove_provider_yes(
        self,
        mock_print,
        mock_confirm,
        mock_select,
        mock_remove,
        mock_get_providers,
    ):
        from esdc.config_wizard import _remove_provider_flow

        mock_get_providers.return_value = {"my-openai": {"provider_type": "openai"}}
        mock_select.return_value.ask.return_value = "my-openai"
        mock_confirm.return_value.ask.return_value = True

        _remove_provider_flow()

        mock_remove.assert_called_once_with("my-openai")


class TestConfigFlows:
    """Test general config and show flows."""

    @patch("esdc.config_wizard.Config.get_all_config_flat")
    @patch("esdc.config_wizard.questionary.select")
    @patch("esdc.config_wizard._prompt_for_config_value")
    @patch("esdc.config_wizard.Config.set_config_value")
    @patch("esdc.config_wizard.rich_print")
    def test_edit_general_config(
        self,
        mock_print,
        mock_set,
        mock_prompt,
        mock_select,
        mock_flat,
    ):
        from esdc.config_wizard import _edit_section

        mock_flat.return_value = {"logging.level": "INFO"}
        mock_select.return_value.ask.return_value = "__back__"

        _edit_section("Logging", ["logging.level"])

        # When back is selected, no edits should happen
        mock_prompt.assert_not_called()
        mock_set.assert_not_called()

    @patch("esdc.config_wizard.Config.get_all_config_flat")
    @patch("esdc.config_wizard.rich_print")
    def test_show_config(self, mock_print, mock_flat):
        from esdc.config_wizard import _show_config_flow

        mock_flat.return_value = {
            "api_url": "https://example.com",
            "database_path": "/tmp/db",
        }
        _show_config_flow()

    @patch("esdc.config_wizard.Config.get_all_config_flat")
    @patch("esdc.config_wizard.rich_print")
    def test_show_config_empty(self, mock_print, mock_flat):
        from esdc.config_wizard import _show_config_flow

        mock_flat.return_value = {}
        _show_config_flow()


class TestGeneralConfigListsCorpusKeys:
    """Corpus keys appear in the edit list even when absent from config.yaml."""

    @patch("esdc.config_wizard._select_with_back", return_value="__back__")
    @patch("esdc.config_wizard.Config.get_all_config_flat")
    def test_corpus_defaults_merged(self, mock_flat, mock_select):
        from esdc.config_wizard import _edit_section

        mock_flat.return_value = {"api_url": "http://x"}  # no corpus.* in file
        _edit_section("Corpus models", MODEL_SECTIONS["Corpus models"])

        choices = mock_select.call_args.kwargs.get("choices") or (
            mock_select.call_args.args[1] if len(mock_select.call_args.args) > 1 else []
        )
        values = [getattr(c, "value", None) for c in choices]
        assert "corpus.metadata_model" in values
        assert "corpus.cleanup_model" in values
        assert "corpus.ocr_model" in values


class TestCorpusModelPicker:
    """Corpus model keys get a select of main/ollama/custom, not a text box."""

    @patch("esdc.config_wizard._fetch_models", return_value=["glm-ocr", "qwen3:8b"])
    def test_choices_metadata_model(self, mock_fetch):
        from esdc.config_wizard import _corpus_model_choices

        values = [c.value for c in _corpus_model_choices("corpus.metadata_model")]
        assert values[0] == "main"
        assert "" in values            # image-based prefill fallback
        assert "glm-ocr" in values and "qwen3:8b" in values
        assert "__custom__" in values

    @patch("esdc.config_wizard._fetch_models", return_value=["glm-ocr"])
    def test_choices_ocr_model_has_no_main(self, mock_fetch):
        from esdc.config_wizard import _corpus_model_choices

        values = [c.value for c in _corpus_model_choices("corpus.ocr_model")]
        assert "main" not in values    # vision OCR can't route through chat provider
        assert "glm-ocr" in values and "__custom__" in values

    _FAKE_PROVIDERS = {
        "anthropic": {"provider_type": "anthropic", "model": "claude-haiku-4-5"},
        "work": {"provider_type": "openai", "model": "gpt-5"},
    }

    @patch(
        "esdc.config_wizard.Config.get_providers", return_value=dict(_FAKE_PROVIDERS)
    )
    @patch("esdc.config_wizard._fetch_models", return_value=["qwen3:8b"])
    def test_choices_include_configured_providers(self, mock_fetch, mock_providers):
        from esdc.config_wizard import _corpus_model_choices

        for key in ("corpus.metadata_model", "corpus.cleanup_model"):
            choices = _corpus_model_choices(key)
            values = [c.value for c in choices]
            assert "provider:anthropic" in values
            assert "provider:work" in values
            labels = {
                c.value: c.title for c in choices if str(c.value).startswith("provider:")
            }
            assert "anthropic" in str(labels["provider:anthropic"])
            assert "claude-haiku-4-5" in str(labels["provider:anthropic"])
            assert "work" in str(labels["provider:work"])
            assert "gpt-5" in str(labels["provider:work"])
            # provider entries come right after "main"
            assert values.index("provider:anthropic") > values.index("main")

    @patch(
        "esdc.config_wizard.Config.get_providers", return_value=dict(_FAKE_PROVIDERS)
    )
    @patch("esdc.config_wizard._fetch_models", return_value=["glm-ocr"])
    def test_choices_ocr_model_has_no_providers(self, mock_fetch, mock_providers):
        from esdc.config_wizard import _corpus_model_choices

        values = [c.value for c in _corpus_model_choices("corpus.ocr_model")]
        assert not any(str(v).startswith("provider:") for v in values)

    @patch(
        "esdc.config_wizard.Config.get_providers", return_value=dict(_FAKE_PROVIDERS)
    )
    @patch("esdc.config_wizard._fetch_models", return_value=["qwen3:8b"])
    @patch("esdc.config_wizard.questionary.select")
    def test_current_provider_value_is_default_choice(
        self, mock_select, mock_fetch, mock_providers
    ):
        mock_select.return_value.ask.return_value = "provider:anthropic"
        result = _prompt_for_config_value(
            "corpus.metadata_model", "provider:anthropic"
        )
        assert result == "provider:anthropic"
        default = mock_select.call_args.kwargs["default"]
        assert default is not None
        assert default.value == "provider:anthropic"

    @patch("esdc.config_wizard._fetch_models", return_value=["qwen3:8b"])
    @patch("esdc.config_wizard.questionary.select")
    def test_select_main(self, mock_select, mock_fetch):
        mock_select.return_value.ask.return_value = "main"
        result = _prompt_for_config_value("corpus.metadata_model", "")
        assert result == "main"

    @patch("esdc.config_wizard._fetch_models", return_value=["qwen3:8b"])
    @patch("esdc.config_wizard.questionary.text")
    @patch("esdc.config_wizard.questionary.select")
    def test_custom_falls_through_to_text(self, mock_select, mock_text, mock_fetch):
        mock_select.return_value.ask.return_value = "__custom__"
        mock_text.return_value.ask.return_value = "my-remote-model"
        result = _prompt_for_config_value("corpus.cleanup_model", "main")
        assert result == "my-remote-model"

    @patch("esdc.config_wizard._fetch_models", return_value=[])
    @patch("esdc.config_wizard.questionary.select")
    def test_ollama_down_still_offers_main_and_custom(self, mock_select, mock_fetch):
        mock_select.return_value.ask.return_value = "main"
        result = _prompt_for_config_value("corpus.metadata_model", "main")
        assert result == "main"


class TestRerankKeyWidgets:
    """corpus.rerank* keys must use int/bool/select widgets, not generic text."""

    @patch("esdc.config_wizard.questionary.text")
    def test_rerank_pool_edit_returns_int(self, mock_text):
        mock_text.return_value.ask.return_value = "50"
        result = _prompt_for_config_value("corpus.rerank_pool", 30)
        assert result == 50
        assert isinstance(result, int)

    def test_rerank_pool_in_int_keys(self):
        assert "corpus.rerank_pool" in Config.INT_KEYS

    @patch("esdc.config_wizard.questionary.select")
    def test_rerank_edit_returns_bool(self, mock_select):
        mock_select.return_value.ask.return_value = "True"
        result = _prompt_for_config_value("corpus.rerank", False)
        assert result is True
        assert isinstance(result, bool)

    def test_rerank_in_boolean_keys(self):
        assert "corpus.rerank" in Config.BOOLEAN_KEYS

    @patch("esdc.config_wizard.questionary.select")
    def test_rerank_model_offers_curated_choices(self, mock_select):
        mock_select.return_value.ask.return_value = (
            "jinaai/jina-reranker-v2-base-multilingual"
        )
        result = _prompt_for_config_value(
            "corpus.rerank_model", "jinaai/jina-reranker-v2-base-multilingual"
        )
        choices = mock_select.call_args.kwargs["choices"]
        values = [c.value for c in choices]
        assert "jinaai/jina-reranker-v2-base-multilingual" in values
        assert "BAAI/bge-reranker-v2-m3" in values
        assert "__custom__" in values
        assert result == "jinaai/jina-reranker-v2-base-multilingual"

    @patch("esdc.config_wizard.questionary.select")
    def test_rerank_model_selecting_bge(self, mock_select):
        mock_select.return_value.ask.return_value = "BAAI/bge-reranker-v2-m3"
        result = _prompt_for_config_value(
            "corpus.rerank_model", "jinaai/jina-reranker-v2-base-multilingual"
        )
        assert result == "BAAI/bge-reranker-v2-m3"

    @patch("esdc.config_wizard.questionary.text")
    @patch("esdc.config_wizard.questionary.select")
    def test_rerank_model_custom_falls_through_to_text(
        self, mock_select, mock_text
    ):
        mock_select.return_value.ask.return_value = "__custom__"
        mock_text.return_value.ask.return_value = "my-custom-reranker"
        result = _prompt_for_config_value(
            "corpus.rerank_model", "jinaai/jina-reranker-v2-base-multilingual"
        )
        assert result == "my-custom-reranker"

    def test_coerce_rerank_pool_to_int(self):
        assert Config._coerce_value("corpus.rerank_pool", "30") == 30

    def test_coerce_rerank_to_bool(self):
        assert Config._coerce_value("corpus.rerank", "true") is True
