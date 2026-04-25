"""Regression tests for cancel/None handling in config wizard."""

from unittest.mock import MagicMock, patch

import pytest

from esdc.config_wizard import (
    WizardCancelledError,
    _prompt,
    _prompt_for_config_value,
    _select_with_back,
)


class TestPromptWrapper:
    """_prompt() must raise WizardCancelledError when questionary returns None."""

    def test_prompt_raises_wizard_exit_on_none(self):
        mock_question = MagicMock()
        mock_question.ask.return_value = None
        with pytest.raises(WizardCancelledError):
            _prompt(mock_question)

    def test_prompt_returns_value_on_success(self):
        mock_question = MagicMock()
        mock_question.ask.return_value = "openai"
        assert _prompt(mock_question) == "openai"


class TestSelectWithBack:
    """_select_with_back() must prepend a Back choice and raise on cancel."""

    @patch("esdc.config_wizard.questionary.select")
    def test_select_returns_value(self, mock_select):
        mock_select.return_value.ask.return_value = "openai"
        result = _select_with_back("Pick:", [MagicMock()], default="openai")
        assert result == "openai"
        # Verify Back choice was prepended
        call_args = mock_select.call_args
        choices = call_args.kwargs["choices"]
        assert choices[0].value == "__back__"

    @patch("esdc.config_wizard.questionary.select")
    def test_select_raises_wizard_exit_on_cancel(self, mock_select):
        mock_select.return_value.ask.return_value = None
        with pytest.raises(WizardCancelledError):
            _select_with_back("Pick:", [MagicMock()])


class TestPromptForConfigValueCancel:
    """When user presses Escape/Control+C, questionary returns None.

    _prompt_for_config_value() must return None (not True/False implicitly).
    """

    @patch("esdc.config_wizard.questionary.select")
    def test_boolean_cancel_returns_none(self, mock_select):
        mock_select.return_value.ask.return_value = None
        result = _prompt_for_config_value("api.verify_ssl", True)
        assert result is None

    @patch("esdc.config_wizard.questionary.select")
    def test_enum_cancel_returns_none(self, mock_select):
        mock_select.return_value.ask.return_value = None
        result = _prompt_for_config_value("tool_format", "native")
        assert result is None

    @patch("esdc.config_wizard.questionary.text")
    def test_integer_cancel_returns_none(self, mock_text):
        mock_text.return_value.ask.return_value = None
        result = _prompt_for_config_value("cache.sql_ttl", 3600)
        assert result is None

    @patch("esdc.config_wizard.questionary.password")
    def test_password_cancel_returns_none(self, mock_password):
        mock_password.return_value.ask.return_value = None
        result = _prompt_for_config_value("providers.x.api_key", "secret")
        assert result is None

    @patch("esdc.config_wizard.questionary.path")
    def test_path_cancel_returns_none(self, mock_path):
        mock_path.return_value.ask.return_value = None
        result = _prompt_for_config_value("database_path", "/tmp/db")
        assert result is None

    @patch("esdc.config_wizard.questionary.text")
    def test_text_cancel_returns_none(self, mock_text):
        mock_text.return_value.ask.return_value = None
        result = _prompt_for_config_value("api_url", "http://old")
        assert result is None


class TestEditGeneralConfigSkipOnCancel:
    """_edit_general_config_flow must skip save when _prompt returns None."""

    @patch("esdc.config_wizard.Config.get_all_config_flat")
    @patch("esdc.config_wizard.Config.set_config_value")
    @patch("esdc.config_wizard._prompt_for_config_value")
    @patch("esdc.config_wizard.questionary.select")
    @patch("esdc.config_wizard.rich_print")
    def test_cancel_skips_save(
        self,
        mock_print,
        mock_select,
        mock_prompt,
        mock_set,
        mock_flat,
    ):
        from esdc.config_wizard import _edit_general_config_flow

        mock_flat.return_value = {"api_url": "https://example.com"}
        mock_select.return_value.ask.side_effect = [
            "api_url",
            "__back__",
        ]
        mock_prompt.return_value = None

        _edit_general_config_flow()

        # Should NOT call set_config_value because prompt returned None
        mock_set.assert_not_called()
        # Should print cancel message
        cancel_calls = [
            call
            for call in mock_print.call_args_list
            if "cancelled" in str(call).lower()
        ]
        assert len(cancel_calls) > 0
