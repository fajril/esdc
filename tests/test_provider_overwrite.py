"""Tests for provider overwrite and default provider logic."""

from unittest.mock import patch

from esdc.config_wizard import (
    _check_name_exists,
    _find_provider_by_type,
    _format_provider_label,
)


class TestFindProviderByType:
    """Test _find_provider_by_type() helper."""

    @patch("esdc.config_wizard.Config.get_providers")
    def test_openai_exists(self, mock_get_providers):
        mock_get_providers.return_value = {
            "my-openai": {"provider_type": "openai"},
            "my-claude": {"provider_type": "anthropic"},
        }
        result = _find_provider_by_type("openai")
        assert result == "my-openai"

    @patch("esdc.config_wizard.Config.get_providers")
    def test_anthropic_exists(self, mock_get_providers):
        mock_get_providers.return_value = {
            "my-openai": {"provider_type": "openai"},
            "my-claude": {"provider_type": "anthropic"},
        }
        result = _find_provider_by_type("anthropic")
        assert result == "my-claude"

    @patch("esdc.config_wizard.Config.get_providers")
    def test_not_found(self, mock_get_providers):
        mock_get_providers.return_value = {
            "my-openai": {"provider_type": "openai"},
        }
        result = _find_provider_by_type("groq")
        assert result is None

    @patch("esdc.config_wizard.Config.get_providers")
    def test_openai_compatible_exempt(self, mock_get_providers):
        mock_get_providers.return_value = {
            "custom-api": {"provider_type": "openai_compatible"},
        }
        result = _find_provider_by_type("openai_compatible")
        assert result is None


class TestAddProviderOverwrite:
    """Test overwrite behavior in _add_provider_flow."""

    @patch("esdc.config_wizard.Config.save_provider")
    @patch("esdc.config_wizard.Config.get_providers")
    @patch("esdc.config_wizard.Config.get_default_provider")
    @patch("esdc.config_wizard.Config.remove_provider")
    @patch("esdc.config_wizard.questionary.confirm")
    @patch("esdc.config_wizard.questionary.select")
    @patch("esdc.config_wizard.questionary.text")
    @patch("esdc.config_wizard.questionary.password")
    @patch("esdc.config_wizard.rich_print")
    def test_overwrite_confirms(
        self,
        mock_print,
        mock_password,
        mock_text,
        mock_select,
        mock_confirm,
        mock_remove,
        mock_default,
        mock_providers,
        mock_save,
    ):
        from esdc.config_wizard import _add_provider_flow

        # Existing openai provider
        mock_providers.return_value = {
            "my-openai": {"provider_type": "openai"},
        }
        mock_default.return_value = "my-openai"
        mock_select.return_value.ask.side_effect = [
            "openai",
            "gpt-4o",
            "low",
        ]
        mock_text.return_value.ask.side_effect = [
            "company-openai",
            "",
        ]
        mock_password.return_value.ask.return_value = "sk-test"
        mock_confirm.return_value.ask.side_effect = [
            True,
            False,
            False,
        ]

        _add_provider_flow()

        # Should remove old one
        mock_remove.assert_called_once_with("my-openai")
        # Should save new one
        mock_save.assert_called_once()
        # Name should be provider_type key for non-custom types
        call_args = mock_save.call_args
        assert call_args[0][0] == "openai"

    @patch("esdc.config_wizard.Config.save_provider")
    @patch("esdc.config_wizard.Config.get_providers")
    @patch("esdc.config_wizard.Config.get_default_provider")
    @patch("esdc.config_wizard.Config.remove_provider")
    @patch("esdc.config_wizard.questionary.confirm")
    @patch("esdc.config_wizard.questionary.select")
    @patch("esdc.config_wizard.questionary.text")
    @patch("esdc.config_wizard.questionary.password")
    @patch("esdc.config_wizard.rich_print")
    def test_overwrite_declined(
        self,
        mock_print,
        mock_password,
        mock_text,
        mock_select,
        mock_confirm,
        mock_remove,
        mock_default,
        mock_providers,
        mock_save,
    ):
        from esdc.config_wizard import _add_provider_flow

        mock_providers.return_value = {
            "my-openai": {"provider_type": "openai"},
        }
        mock_default.return_value = "my-openai"
        mock_select.return_value.ask.side_effect = [
            "openai",
            "gpt-4o",
            "low",
        ]
        mock_text.return_value.ask.side_effect = [
            "company-openai",
            "",
        ]
        mock_password.return_value.ask.return_value = "sk-test"
        mock_confirm.return_value.ask.side_effect = [False]

        _add_provider_flow()

        # Should NOT remove or save
        mock_remove.assert_not_called()
        mock_save.assert_not_called()


class TestDefaultProviderAutoSet:
    """Test auto-set default when adding first provider."""

    @patch("esdc.config_wizard.Config.save_provider")
    @patch("esdc.config_wizard.Config.get_providers")
    @patch("esdc.config_wizard.Config.get_default_provider")
    @patch("esdc.config_wizard.Config.set_default_provider")
    @patch("esdc.config_wizard.questionary.confirm")
    @patch("esdc.config_wizard.questionary.select")
    @patch("esdc.config_wizard.questionary.text")
    @patch("esdc.config_wizard.questionary.password")
    @patch("esdc.config_wizard.rich_print")
    def test_auto_set_first_provider(
        self,
        mock_print,
        mock_password,
        mock_text,
        mock_select,
        mock_confirm,
        mock_set_default,
        mock_default,
        mock_providers,
        mock_save,
    ):
        from esdc.config_wizard import _add_provider_flow

        mock_providers.return_value = {}  # No providers yet
        mock_default.return_value = None  # No default set
        mock_select.return_value.ask.side_effect = [
            "openai",
            "gpt-4o",
            "low",
        ]
        mock_text.return_value.ask.side_effect = [
            "my-openai",
            "",
        ]
        mock_password.return_value.ask.return_value = "sk-test"
        mock_confirm.return_value.ask.side_effect = [
            False,
            False,
        ]

        _add_provider_flow()

        # Should auto-set as default (no confirm prompt)
        mock_set_default.assert_called_once_with("openai")


class TestCLISetDefault:
    """Test --set-default-provider CLI flag."""

    @patch("esdc.configs.Config.get_providers")
    @patch("esdc.configs.Config.set_default_provider")
    @patch("builtins.print")
    @patch("typer.echo")
    def test_set_default_valid(self, mock_echo, mock_print, mock_set, mock_get):
        from esdc.commands.configs import configs

        mock_get.return_value = {
            "my-openai": {"provider_type": "openai"},
        }

        configs(
            reset=False,
            reset_key=None,
            show=False,
            set_default_provider="my-openai",
        )

        mock_set.assert_called_once_with("my-openai")
        mock_echo.assert_called_once_with("Default provider set to 'my-openai'.")

    @patch("esdc.configs.Config.get_providers")
    @patch("esdc.configs.Config.set_default_provider")
    @patch("typer.echo")
    def test_set_default_invalid(self, mock_echo, mock_set, mock_get):
        from esdc.commands.configs import configs

        mock_get.return_value = {
            "my-openai": {"provider_type": "openai"},
        }

        configs(
            reset=False,
            reset_key=None,
            show=False,
            set_default_provider="nonexistent",
        )

        mock_set.assert_not_called()
        # Should print error
        assert mock_echo.call_count >= 1


class TestCheckNameExists:
    """Test _check_name_exists() helper."""

    @patch("esdc.config_wizard.Config.get_providers")
    def test_name_taken(self, mock_get_providers):
        mock_get_providers.return_value = {
            "my-qwen": {"provider_type": "openai_compatible"},
            "openai": {"provider_type": "openai"},
        }
        assert _check_name_exists("my-qwen") == "my-qwen"
        assert _check_name_exists("openai") == "openai"

    @patch("esdc.config_wizard.Config.get_providers")
    def test_name_available(self, mock_get_providers):
        mock_get_providers.return_value = {
            "my-qwen": {"provider_type": "openai_compatible"},
        }
        assert _check_name_exists("my-deepseek") is None


class TestFormatProviderLabel:
    """Test _format_provider_label() helper."""

    def test_default_type_name(self):
        label = _format_provider_label("openai", {"provider_type": "openai"})
        assert label == "openai"

    def test_custom_name(self):
        label = _format_provider_label(
            "my-qwen", {"provider_type": "openai_compatible"}
        )
        assert label == "my-qwen (OpenAI Compatible API)"

    def test_custom_name_unknown_type(self):
        label = _format_provider_label("my-api", {"provider_type": "custom_type"})
        assert label == "my-api (custom_type)"


class TestAddOpenaiCompatibleFlow:
    """Test adding OpenAI-compatible provider with custom name."""

    @patch("esdc.config_wizard.Config.save_provider")
    @patch("esdc.config_wizard.Config.get_providers")
    @patch("esdc.config_wizard.Config.get_default_provider")
    @patch("esdc.config_wizard.Config.set_default_provider")
    @patch("esdc.config_wizard.questionary.confirm")
    @patch("esdc.config_wizard.questionary.select")
    @patch("esdc.config_wizard.questionary.text")
    @patch("esdc.config_wizard.questionary.password")
    @patch("esdc.config_wizard.rich_print")
    def test_custom_name_saved(
        self,
        mock_print,
        mock_password,
        mock_text,
        mock_select,
        mock_confirm,
        mock_set_default,
        mock_default,
        mock_providers,
        mock_save,
    ):
        from esdc.config_wizard import _add_provider_flow

        mock_providers.return_value = {}  # No existing providers
        mock_default.return_value = None  # No default set
        mock_select.return_value.ask.side_effect = [
            "openai_compatible",  # select provider type
            "qwen-7b",  # select model
        ]
        mock_text.return_value.ask.side_effect = [
            "my-qwen",  # custom provider name
            "http://qwen:8000",  # base URL
            "",  # unused (model manual entry fallback)
        ]
        mock_password.return_value.ask.return_value = ""  # no api key
        mock_confirm.return_value.ask.side_effect = [
            False,  # don't set as default
            False,  # don't test connection
        ]

        _add_provider_flow()

        mock_save.assert_called_once()
        call_args = mock_save.call_args
        assert call_args[0][0] == "my-qwen"
        saved_config = call_args[0][1]
        assert saved_config["provider_type"] == "openai_compatible"
        assert saved_config["model"] == "qwen-7b"
