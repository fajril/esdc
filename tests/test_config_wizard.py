"""Tests for config_wizard module."""

from unittest.mock import MagicMock, patch

from esdc.config_wizard import (
    _fetch_models,
    _mask_value,
    _prompt_for_config_value,
    run_wizard,
)


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


class TestRunWizard:
    """Test the main run_wizard dispatch logic."""

    @patch("esdc.config_wizard.Config.init_config")
    @patch("esdc.config_wizard.questionary.select")
    @patch("esdc.config_wizard._add_provider_flow")
    def test_add_provider(self, mock_add, mock_select, mock_init):
        mock_select.return_value.ask.side_effect = ["Add provider", "Exit"]
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
            "Edit provider",
            "Exit",
        ]
        run_wizard()
        mock_edit.assert_called_once()


class TestProviderCRUDFlows:
    """Test provider add/edit/remove flows."""

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
    ):
        from esdc.config_wizard import _add_provider_flow

        mock_select.return_value.ask.side_effect = [
            "openai",
            "gpt-4o",
        ]
        mock_text.return_value.ask.side_effect = ["my-openai"]
        mock_password.return_value.ask.return_value = "sk-test"
        mock_confirm.side_effect = [True, False, False]

        _add_provider_flow()

        mock_save.assert_called_once()
        mock_set_default.assert_called_once_with("my-openai")

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
    ):
        from esdc.config_wizard import _add_provider_flow

        mock_select.return_value.ask.side_effect = [
            "ollama",
            "llama3",
        ]
        mock_text.return_value.ask.side_effect = [
            "my-ollama",
            "http://localhost:11434",
        ]
        mock_confirm.side_effect = [False, False, False]

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
    @patch("esdc.config_wizard.questionary.confirm")
    @patch("esdc.config_wizard.rich_print")
    def test_edit_general_config(
        self,
        mock_print,
        mock_confirm,
        mock_set,
        mock_prompt,
        mock_select,
        mock_flat,
    ):
        from esdc.config_wizard import _edit_general_config_flow

        mock_flat.return_value = {"logging.level": "INFO"}
        mock_confirm.side_effect = [False]
        mock_select.return_value.ask.return_value = "__back__"

        _edit_general_config_flow()

        # When back is selected, no edits should happen
        mock_prompt.assert_not_called()

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
