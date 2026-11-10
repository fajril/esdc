"""Tests for back navigation in config wizard flows."""

from unittest.mock import patch

from esdc.config_wizard import _advanced_menu, run_wizard


class TestAdvancedBackNavigation:
    """Advanced returns to home on Back, and its key editor returns to Advanced."""

    @patch("esdc.config_wizard.Config.reset_config")
    @patch("esdc.config_wizard.rich_print")
    @patch("esdc.config_wizard.questionary.select")
    def test_back_at_advanced_returns_home(self, mock_select, mock_print, mock_reset):
        mock_select.return_value.unsafe_ask.side_effect = ["__back__"]
        _advanced_menu()
        mock_reset.assert_not_called()

    @patch("esdc.config_wizard.Config.set_config_value")
    @patch(
        "esdc.config_wizard.Config.get_all_config_flat",
        return_value={"cache.sql_ttl": 604800},
    )
    @patch("esdc.config_wizard.questionary.select")
    def test_back_at_the_key_list_returns_to_advanced(
        self, mock_select, mock_flat, mock_set
    ):
        """One Back pops the key list, and the next one pops Advanced."""
        mock_select.return_value.unsafe_ask.side_effect = [
            "Cache",
            "__back__",
            "__back__",
        ]
        _advanced_menu()
        mock_set.assert_not_called()

    @patch("esdc.config_wizard.Config.reset_config")
    @patch(
        "esdc.config_wizard.Config.get_all_config_flat",
        return_value={"cache.sql_ttl": 604800},
    )
    @patch("esdc.config_wizard.questionary.confirm")
    @patch("esdc.config_wizard.questionary.select")
    def test_escape_at_the_reset_confirm_resets_nothing(
        self, mock_select, mock_confirm, mock_flat, mock_reset
    ):
        mock_select.return_value.unsafe_ask.side_effect = [
            "Reset a setting",
            "Cache",
            "cache.sql_ttl",
            "__back__",
        ]
        mock_confirm.return_value.unsafe_ask.return_value = None
        _advanced_menu()
        mock_reset.assert_not_called()


class TestHomeBackNavigation:
    """Escape at home ends the run; a submenu Back lands back at home."""

    @patch("esdc.config_wizard.Config.init_config")
    @patch("esdc.config_wizard.rich_print")
    @patch("esdc.config_wizard.questionary.select")
    def test_escape_at_home_exits_from_a_submenu_level(
        self, mock_select, mock_print, mock_init
    ):
        # Advanced -> Back -> home, where Escape ends the run.
        mock_select.return_value.unsafe_ask.side_effect = ["Advanced", "__back__", None]
        run_wizard()
        assert mock_select.call_count == 3
