"""Tests for back navigation in config wizard flows."""

from unittest.mock import patch

from esdc.config_wizard import (
    _add_provider_flow,
    _edit_provider_flow,
    _remove_provider_flow,
    _set_default_provider_flow,
    _test_provider_standalone,
)


class TestAddProviderFlowBack:
    """_add_provider_flow must return to main menu on Back."""

    @patch("esdc.config_wizard.questionary.select")
    def test_back_at_provider_type(self, mock_select):
        mock_select.return_value.ask.return_value = "__back__"
        # Should return without raising or saving anything
        _add_provider_flow()

    @patch("esdc.config_wizard.Config.save_provider")
    @patch("esdc.config_wizard._fetch_models")
    @patch("esdc.config_wizard.questionary.password")
    @patch("esdc.config_wizard.questionary.select")
    def test_back_at_model_select(
        self, mock_select, mock_password, mock_fetch, mock_save
    ):
        """If user hits Back at model selection, flow exits without saving."""
        mock_password.return_value.ask.return_value = "sk-test"
        mock_fetch.return_value = ["gpt-4", "gpt-3.5"]
        # First select = provider type (openai), second = model select (__back__)
        mock_select.return_value.ask.side_effect = ["openai", "__back__"]

        _add_provider_flow()
        mock_save.assert_not_called()


class TestEditProviderFlowBack:
    """_edit_provider_flow must handle Back at each level."""

    @patch("esdc.config_wizard.Config.get_providers")
    @patch("esdc.config_wizard.questionary.select")
    def test_back_at_provider_selection(self, mock_select, mock_get_providers):
        mock_get_providers.return_value = {"openai": {"provider_type": "openai"}}
        mock_select.return_value.ask.return_value = "__back__"
        _edit_provider_flow()

    @patch("esdc.config_wizard.Config.save_provider")
    @patch("esdc.config_wizard.Config.get_providers")
    @patch("esdc.config_wizard.questionary.select")
    def test_back_at_field_selection(self, mock_select, mock_get_providers, mock_save):
        mock_get_providers.return_value = {
            "openai": {"provider_type": "openai", "model": "gpt-4"}
        }
        # First = provider, second = field (__back__)
        mock_select.return_value.ask.side_effect = ["openai", "__back__"]
        _edit_provider_flow()
        mock_save.assert_not_called()


class TestRemoveProviderFlowBack:
    """_remove_provider_flow must return on Back."""

    @patch("esdc.config_wizard.Config.get_providers")
    @patch("esdc.config_wizard.questionary.select")
    def test_back_at_provider_selection(self, mock_select, mock_get_providers):
        mock_get_providers.return_value = {"openai": {"provider_type": "openai"}}
        mock_select.return_value.ask.return_value = "__back__"
        _remove_provider_flow()


class TestSetDefaultProviderFlowBack:
    """_set_default_provider_flow must return on Back."""

    @patch("esdc.config_wizard.Config.get_providers")
    @patch("esdc.config_wizard.questionary.select")
    def test_back_at_provider_selection(self, mock_select, mock_get_providers):
        mock_get_providers.return_value = {"openai": {"provider_type": "openai"}}
        mock_select.return_value.ask.return_value = "__back__"
        _set_default_provider_flow()


class TestTestProviderStandaloneBack:
    """_test_provider_standalone must return on Back."""

    @patch("esdc.config_wizard.Config.get_providers")
    @patch("esdc.config_wizard.questionary.select")
    def test_back_at_provider_selection(self, mock_select, mock_get_providers):
        mock_get_providers.return_value = {"openai": {"provider_type": "openai"}}
        mock_select.return_value.ask.return_value = "__back__"
        _test_provider_standalone()
