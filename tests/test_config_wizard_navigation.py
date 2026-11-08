"""Tests for back navigation in config wizard flows."""

from unittest.mock import MagicMock, patch

import pytest

from esdc.config_wizard import (
    PairSelection,
    _add_provider_flow,
    _advanced_menu,
    _edit_provider_flow,
    _models_by_task_menu,
    _remove_provider_flow,
    _set_default_provider_flow,
    _set_provider_order_flow,
    _test_provider_standalone,
    run_wizard,
)
from esdc.model_catalog import ModelCatalog

BACK_FLOWS = [
    _add_provider_flow,
    _edit_provider_flow,
    _remove_provider_flow,
    _set_default_provider_flow,
    _set_provider_order_flow,
    _test_provider_standalone,
]


@pytest.mark.parametrize("flow", BACK_FLOWS, ids=lambda flow: flow.__name__)
def test_back_at_the_first_level_returns_without_writing(flow, monkeypatch):
    """Back at the provider picker stops the flow before it can write."""
    monkeypatch.setattr(
        "esdc.config_wizard.Config.get_providers",
        lambda: {"openai": {"provider_type": "openai"}},
    )
    monkeypatch.setattr("esdc.config_wizard.Config.get_provider_order", lambda: [])
    writers = {
        name: MagicMock()
        for name in (
            "save_provider",
            "remove_provider",
            "set_default_provider",
            "set_provider_order",
            "set_config_value",
            "set_config_values",
        )
    }
    for name, mock in writers.items():
        monkeypatch.setattr(f"esdc.config_wizard.Config.{name}", mock)
    test_provider = MagicMock()
    monkeypatch.setattr("esdc.config_wizard._test_provider", test_provider)
    select = MagicMock()
    select.return_value.unsafe_ask.return_value = "__back__"
    monkeypatch.setattr("esdc.config_wizard.questionary.select", select)
    references = MagicMock(return_value=[])
    monkeypatch.setattr(
        "esdc.config_wizard.Config.get_provider_references", references
    )
    confirm = MagicMock()
    confirm.return_value.unsafe_ask.return_value = False
    monkeypatch.setattr("esdc.config_wizard.questionary.confirm", confirm)

    flow()

    # One prompt only: the flow returned instead of continuing to the next step.
    assert select.call_count == 1
    confirm.assert_not_called()
    references.assert_not_called()
    test_provider.assert_not_called()
    for mock in writers.values():
        mock.assert_not_called()


class TestAddProviderFlowBack:
    """_add_provider_flow must return to main menu on Back."""

    @patch("esdc.config_wizard.Config.save_provider")
    @patch("esdc.config_wizard._fetch_models")
    @patch("esdc.config_wizard.questionary.password")
    @patch("esdc.config_wizard.questionary.select")
    def test_back_at_model_select(
        self, mock_select, mock_password, mock_fetch, mock_save
    ):
        """If user hits Back at model selection, flow exits without saving."""
        mock_password.return_value.unsafe_ask.return_value = "sk-test"
        mock_fetch.return_value = ["gpt-4", "gpt-3.5"]
        # First select = provider type (openai), second = model select (__back__)
        mock_select.return_value.unsafe_ask.side_effect = ["openai", "__back__"]

        _add_provider_flow()
        mock_save.assert_not_called()


class TestEditProviderFlowBack:
    """_edit_provider_flow must handle Back at each level."""

    @patch("esdc.config_wizard.Config.save_provider")
    @patch("esdc.config_wizard.Config.get_providers")
    @patch("esdc.config_wizard.questionary.select")
    def test_back_at_field_selection(self, mock_select, mock_get_providers, mock_save):
        mock_get_providers.return_value = {
            "openai": {"provider_type": "openai", "model": "gpt-4"}
        }
        # First = provider, second = field (__back__)
        mock_select.return_value.unsafe_ask.side_effect = ["openai", "__back__"]
        _edit_provider_flow()
        mock_save.assert_not_called()

    @patch("esdc.config_wizard.Config.save_provider")
    @patch("esdc.config_wizard.Config.get_providers")
    @patch("esdc.config_wizard.questionary.password")
    @patch("esdc.config_wizard.questionary.select")
    def test_back_at_the_action_menu_after_an_edit_writes_nothing(
        self, mock_select, mock_password, mock_get_providers, mock_save
    ):
        """A field edit only updates the draft, so Back at the action menu writes nothing."""
        saved = {
            "openai": {"provider_type": "openai", "model": "gpt-4", "api_key": "old"}
        }
        mock_get_providers.return_value = saved
        mock_select.return_value.unsafe_ask.side_effect = [
            "openai",
            "api_key",
            "__done__",
            "__back__",
        ]
        mock_password.return_value.unsafe_ask.return_value = "new-key"

        _edit_provider_flow()

        mock_save.assert_not_called()
        assert saved["openai"]["api_key"] == "old"


_EMPTY_CATALOG = ModelCatalog()
_NO_PROVIDERS: dict[str, dict] = {}


def _pair(provider: str, model: str) -> PairSelection:
    return PairSelection(provider, model)


class TestTaskMenuBackNavigation:
    """Each Back at a task screen pops exactly one level and writes nothing."""

    @patch("esdc.config_wizard.Config.set_config_values")
    @patch("esdc.config_wizard.Config.set_chat_selection")
    @patch("esdc.config_wizard.discover_models", return_value=_EMPTY_CATALOG)
    @patch("esdc.config_wizard.rich_print")
    @patch("esdc.config_wizard.Config.get_providers", return_value=_NO_PROVIDERS)
    @patch("esdc.config_wizard.Config.get_all_config_flat", return_value={})
    @patch("esdc.config_wizard.questionary.select")
    def test_back_at_the_picker_returns_to_the_task_menu(
        self,
        mock_select,
        mock_flat,
        mock_providers,
        mock_print,
        mock_discover,
        mock_chat,
        mock_values,
    ):
        # Task menu -> Embeddings -> Back at the picker -> Back at the task menu.
        mock_select.return_value.unsafe_ask.side_effect = [
            "embedding",
            "__back__",
            "__back__",
        ]
        _models_by_task_menu()

        assert mock_select.call_count == 3
        mock_chat.assert_not_called()
        mock_values.assert_not_called()

    @patch("esdc.config_wizard.Config.set_config_values")
    @patch("esdc.config_wizard.Config.set_chat_selection")
    @patch("esdc.config_wizard.discover_models", return_value=_EMPTY_CATALOG)
    @patch("esdc.config_wizard.rich_print")
    @patch("esdc.config_wizard.Config.get_providers", return_value=_NO_PROVIDERS)
    @patch("esdc.config_wizard.Config.get_all_config_flat", return_value={})
    @patch("esdc.config_wizard.questionary.select")
    def test_escape_at_a_task_screen_pops_one_level(
        self,
        mock_select,
        mock_flat,
        mock_providers,
        mock_print,
        mock_discover,
        mock_chat,
        mock_values,
    ):
        """Escape reads as Back, so it lands back on the task menu, not at home."""
        mock_select.return_value.unsafe_ask.side_effect = ["chat", None, "__back__"]
        _models_by_task_menu()

        assert mock_select.call_count == 3
        mock_chat.assert_not_called()
        mock_values.assert_not_called()

    @patch("esdc.config_wizard.Config.set_config_values")
    @patch("esdc.config_wizard.discover_models", return_value=_EMPTY_CATALOG)
    @patch("esdc.config_wizard.rich_print")
    @patch("esdc.config_wizard.Config.get_providers", return_value=_NO_PROVIDERS)
    @patch("esdc.config_wizard.Config.get_all_config_flat", return_value={})
    @patch("esdc.config_wizard.questionary.select")
    def test_back_at_the_action_menu_returns_to_the_task_screen(
        self,
        mock_select,
        mock_flat,
        mock_providers,
        mock_print,
        mock_discover,
        mock_values,
    ):
        """Back after a selection returns to the task screen, then one more Back exits."""
        pair = _pair("engine", "m1")
        mock_select.return_value.unsafe_ask.side_effect = [
            "embedding",
            pair,
            "__back__",
            pair,
            "cancel",
            "__back__",
            "__back__",
        ]
        _models_by_task_menu()

        # Two screens were entered, and neither saved.
        assert mock_select.call_count == 7
        mock_values.assert_not_called()

    @patch("esdc.config_wizard.Config.set_config_value")
    @patch("esdc.config_wizard.Config.set_config_values")
    @patch("esdc.config_wizard.discover_models", return_value=_EMPTY_CATALOG)
    @patch("esdc.config_wizard.rich_print")
    @patch("esdc.config_wizard.Config.get_providers", return_value=_NO_PROVIDERS)
    @patch("esdc.config_wizard.Config.get_all_config_flat", return_value={})
    @patch("esdc.config_wizard.questionary.select")
    def test_back_at_the_rerank_menu_exits_without_writing(
        self,
        mock_select,
        mock_flat,
        mock_providers,
        mock_print,
        mock_discover,
        mock_values,
        mock_single,
    ):
        mock_select.return_value.unsafe_ask.side_effect = [
            "rerank",
            "__back__",
            "__back__",
        ]
        _models_by_task_menu()

        mock_values.assert_not_called()
        mock_single.assert_not_called()

    @patch("esdc.config_wizard.Config.set_config_values")
    @patch("esdc.config_wizard.discover_models", return_value=_EMPTY_CATALOG)
    @patch("esdc.config_wizard.rich_print")
    @patch("esdc.config_wizard.Config.get_providers", return_value=_NO_PROVIDERS)
    @patch("esdc.config_wizard.Config.get_all_config_flat", return_value={})
    @patch("esdc.config_wizard.questionary.select")
    def test_escape_at_the_custom_sub_prompts_returns_to_the_picker(
        self,
        mock_select,
        mock_flat,
        mock_providers,
        mock_print,
        mock_discover,
        mock_values,
    ):
        """Escape inside Custom goes back to the picker, not straight out of the task."""
        mock_providers.return_value = {"engine": {"provider_type": "openai"}}
        pair = _pair("engine", "m1")
        mock_select.return_value.unsafe_ask.side_effect = [
            "chat",
            "__custom__",
            None,
            pair,
            "cancel",
            "__back__",
            "__back__",
        ]
        _models_by_task_menu()

        mock_values.assert_not_called()
        assert mock_select.call_count == 7


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
    @patch("esdc.config_wizard.Config.set_config_values")
    @patch("esdc.config_wizard.discover_models", return_value=_EMPTY_CATALOG)
    @patch("esdc.config_wizard.Config.get_providers", return_value=_NO_PROVIDERS)
    @patch("esdc.config_wizard.Config.get_all_config_flat", return_value={})
    @patch("esdc.config_wizard.rich_print")
    @patch("esdc.config_wizard.questionary.select")
    def test_task_menu_back_returns_to_home_then_exit(
        self,
        mock_select,
        mock_print,
        mock_flat,
        mock_providers,
        mock_discover,
        mock_values,
        mock_init,
    ):
        mock_select.return_value.unsafe_ask.side_effect = [
            "Models by task",
            "chat",
            "__back__",
            "__back__",
            "Exit",
        ]
        run_wizard()

        assert mock_select.call_count == 5
        mock_values.assert_not_called()

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
