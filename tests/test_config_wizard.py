"""Tests for config_wizard module."""

import ast
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock, patch

import ollama
import pytest

from esdc.config_fields import field_hint
from esdc.config_wizard import (
    _DIRECT_SETTINGS_VALUE,
    _RETIRED_EMBEDDING_REASON,
    _RETIRED_RERANK_REASON,
    _SAVEABLE_SELECTIONS,
    ChatDefaultSelection,
    CustomSelection,
    DirectSelection,
    LinkedProviderSelection,
    LocalSelection,
    NeedsConfigurationSelection,
    OffSelection,
    PairSelection,
    TaskRole,
    _add_provider_flow,
    _choose_provider_model,
    _current_task_selection,
    _custom_pair_flow,
    _direct_embedding_flow,
    _edit_provider_flow,
    _edit_task,
    _fetch_models,
    _mask_value,
    _prompt_for_config_value,
    _remove_provider_flow,
    _run_task_action,
    _save_task_selection,
    _selectable,
    _selection_label,
    _short_test_message,
    _task_summary,
    _test_task_selection,
    run_wizard,
)
from esdc.configs import (
    ADVANCED_SECTIONS,
    APPLICATION_SECTIONS,
    KEY_DESCRIPTIONS,
    TASK_SECTIONS,
    Config,
)
from esdc.corpus.ocr import OllamaVisionOcr
from esdc.embedders import OllamaEmbedder
from esdc.model_catalog import ModelCatalog, ModelEntry


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
        mock_select.return_value.unsafe_ask.return_value = "True"
        result = _prompt_for_config_value("api.verify_ssl", False)
        assert result is True

    @patch("esdc.config_wizard.questionary.select")
    def test_enum(self, mock_select):
        mock_select.return_value.unsafe_ask.return_value = "DEBUG"
        result = _prompt_for_config_value("logging.level", "INFO")
        assert result == "DEBUG"

    @patch("esdc.config_wizard.questionary.password")
    def test_password(self, mock_pass):
        mock_pass.return_value.unsafe_ask.return_value = "secret"
        result = _prompt_for_config_value("providers.my.api_key", "old")
        assert result == "secret"

    @patch("esdc.config_wizard.questionary.text")
    def test_text(self, mock_text):
        mock_text.return_value.unsafe_ask.return_value = "http://new.url"
        result = _prompt_for_config_value("api_url", "http://old.url")
        assert result == "http://new.url"

    @patch("esdc.config_wizard.questionary.text")
    def test_integer(self, mock_text):
        mock_text.return_value.unsafe_ask.return_value = "42"
        result = _prompt_for_config_value("cache.sql_ttl", 3600)
        assert result == 42

    @patch("esdc.config_wizard.questionary.text")
    def test_integer_corpus_num_ctx(self, mock_text):
        mock_text.return_value.unsafe_ask.return_value = "4096"
        result = _prompt_for_config_value("corpus.num_ctx", 16384)
        assert result == 4096


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

    The wizard is a task-first tree: home -> Models by task (or Providers,
    Application settings, Advanced) -> one task or action. Each "__back__" pops
    one level (mocked via questionary.select, since _select_with_back calls it
    under the hood), and Escape at home returns None and ends the run.
    """

    @patch("esdc.config_wizard.Config.init_config")
    @patch("esdc.config_wizard.questionary.select")
    @patch("esdc.config_wizard._add_connection_flow")
    def test_add_connection(self, mock_add, mock_select, mock_init):
        mock_select.return_value.unsafe_ask.side_effect = [
            "Connections",
            "Add connection",
            "__back__",
            "Exit",
        ]
        run_wizard()
        mock_add.assert_called_once()

    @patch("esdc.config_wizard.Config.init_config")
    @patch("esdc.config_wizard.questionary.select")
    @patch("esdc.config_wizard._show_config_flow")
    def test_show_config(self, mock_show, mock_select, mock_init):
        mock_select.return_value.unsafe_ask.side_effect = ["View settings", "Exit"]
        run_wizard()
        mock_show.assert_called_once()

    @patch("esdc.config_wizard.Config.init_config")
    @patch("esdc.config_wizard.questionary.select")
    @patch("esdc.config_wizard._edit_connection_flow")
    def test_edit_connection(self, mock_edit, mock_select, mock_init):
        mock_select.return_value.unsafe_ask.side_effect = [
            "Connections",
            "Edit connection",
            "__back__",
            "Exit",
        ]
        run_wizard()
        mock_edit.assert_called_once()


class TestHomeMenuDispatch:
    """Home menu: the task-first layout, with exactly its six rows."""

    HOME_ROWS = [
        "Models by task ▸",
        "Connections ▸",
        "Application settings ▸",
        "Advanced ▸",
        "View settings",
        "Exit",
    ]

    @staticmethod
    def _home_rows(mock_select) -> list[str]:
        choices = mock_select.call_args.kwargs["choices"]
        return [str(choice.title) for choice in choices]

    @patch("esdc.config_wizard.Config.init_config")
    @patch("esdc.config_wizard.questionary.select")
    @patch("esdc.config_wizard.rich_print")
    def test_home_rows(self, mock_print, mock_select, mock_init):
        mock_select.return_value.unsafe_ask.side_effect = ["Exit"]
        run_wizard()
        assert self._home_rows(mock_select) == self.HOME_ROWS

    @patch("esdc.config_wizard.Config.init_config")
    @patch("esdc.config_wizard.questionary.select")
    @patch("esdc.config_wizard._models_by_task_menu")
    @patch("esdc.config_wizard.rich_print")
    def test_models_by_task_routes_to_the_task_menu(
        self, mock_print, mock_task_menu, mock_select, mock_init
    ):
        mock_select.return_value.unsafe_ask.side_effect = ["Models by task", "Exit"]
        run_wizard()
        mock_task_menu.assert_called_once()

    @patch("esdc.config_wizard.Config.init_config")
    @patch("esdc.config_wizard.questionary.select")
    @patch("esdc.config_wizard._connections_menu")
    @patch("esdc.config_wizard.rich_print")
    def test_connections_route_to_the_connection_flow(
        self, mock_print, mock_connections_menu, mock_select, mock_init
    ):
        mock_select.return_value.unsafe_ask.side_effect = ["Connections", "Exit"]
        run_wizard()
        mock_connections_menu.assert_called_once()

    @patch("esdc.config_wizard.Config.init_config")
    @patch("esdc.config_wizard.questionary.select")
    @patch("esdc.config_wizard._edit_section")
    @patch("esdc.config_wizard.rich_print")
    def test_application_settings_routes_to_edit_section(
        self, mock_print, mock_edit_section, mock_select, mock_init
    ):
        mock_select.return_value.unsafe_ask.side_effect = [
            "Application settings",
            "__back__",
            "Exit",
        ]
        run_wizard()
        mock_edit_section.assert_called_once_with(
            "Application settings", APPLICATION_SECTIONS["Application settings"]
        )

    @patch("esdc.config_wizard.Config.init_config")
    @patch("esdc.config_wizard.questionary.select")
    @patch("esdc.config_wizard._advanced_menu")
    @patch("esdc.config_wizard.rich_print")
    def test_advanced_routes_to_the_advanced_menu(
        self, mock_print, mock_advanced_menu, mock_select, mock_init
    ):
        mock_select.return_value.unsafe_ask.side_effect = ["Advanced", "Exit"]
        run_wizard()
        mock_advanced_menu.assert_called_once()

    @patch("esdc.config_wizard.Config.init_config")
    @patch("esdc.config_wizard.questionary.select")
    @patch("esdc.config_wizard._show_config_flow")
    @patch("esdc.config_wizard.rich_print")
    def test_view_settings_routes_to_show_config(
        self, mock_print, mock_show, mock_select, mock_init
    ):
        mock_select.return_value.unsafe_ask.side_effect = ["View settings", "Exit"]
        run_wizard()
        mock_show.assert_called_once()

    @patch("esdc.config_wizard.Config.init_config")
    @patch("esdc.config_wizard.questionary.select")
    @patch("esdc.config_wizard._models_by_task_menu")
    @patch("esdc.config_wizard.rich_print")
    def test_escape_at_home_exits(
        self, mock_print, mock_task_menu, mock_select, mock_init
    ):
        """Escape at home has no Back above it, so it ends the run."""
        mock_select.return_value.unsafe_ask.side_effect = [None]
        run_wizard()
        mock_task_menu.assert_not_called()

    def test_the_old_menus_are_gone(self):
        """One UI per area: no competing models/settings/reset menus."""
        import esdc.config_wizard as wizard

        for name in ("_models_menu", "_settings_menu", "_reset_menu"):
            assert getattr(wizard, name, None) is None


class TestModelsByTaskMenu:
    """Models by task: one row per task, then Back."""

    TASK_ROWS = [
        "← Back",
        "Chat",
        "Embeddings",
        "Reranking",
        "OCR",
        "Metadata extraction",
        "Text cleanup",
    ]

    @staticmethod
    def _rows(mock_select) -> list[str]:
        choices = mock_select.call_args.kwargs["choices"]
        return [str(choice.title) for choice in choices]

    @patch("esdc.config_wizard.questionary.select")
    @patch("esdc.config_wizard._edit_task")
    def test_task_rows(self, mock_edit_task, mock_select):
        mock_select.return_value.unsafe_ask.side_effect = ["__back__"]
        from esdc.config_wizard import _models_by_task_menu

        _models_by_task_menu()
        assert self._rows(mock_select) == self.TASK_ROWS

    @patch("esdc.config_wizard.questionary.select")
    @patch("esdc.config_wizard._edit_task")
    def test_each_task_opens_its_task_screen(self, mock_edit_task, mock_select):
        from esdc.config_wizard import _models_by_task_menu

        mock_select.return_value.unsafe_ask.side_effect = [
            "embedding",
            "rerank",
            "ocr",
            "metadata",
            "cleanup",
            "__back__",
        ]
        _models_by_task_menu()
        assert [call.args[0] for call in mock_edit_task.call_args_list] == [
            "embedding",
            "rerank",
            "ocr",
            "metadata",
            "cleanup",
        ]

    @patch("esdc.config_wizard.questionary.select")
    @patch("esdc.config_wizard._edit_task")
    def test_back_and_escape_leave_the_task_menu(self, mock_edit_task, mock_select):
        from esdc.config_wizard import _models_by_task_menu

        # Escape at a selection reads as Back, so nothing is opened or written.
        mock_select.return_value.unsafe_ask.side_effect = [None]
        _models_by_task_menu()
        mock_edit_task.assert_not_called()


class TestPromptInstructionLine:
    """Descriptions move off the message and onto questionary's instruction=."""

    @patch("esdc.config_wizard.questionary.select")
    def test_enum_key_uses_instruction_param(self, mock_select):
        mock_select.return_value.unsafe_ask.return_value = "INFO"

        _prompt_for_config_value("logging.level", "INFO")

        _, kwargs = mock_select.call_args
        message = mock_select.call_args.args[0] if mock_select.call_args.args else ""
        description = KEY_DESCRIPTIONS["logging.level"]
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

        _edit_section("Observability", ADVANCED_SECTIONS["Observability"])

        choices = (
            mock_select_back.call_args.kwargs.get("choices")
            or (mock_select_back.call_args.args[1])
        )
        labels = {c.value: str(c.title) for c in choices}
        assert "phoenix.enabled = False" in labels["phoenix.enabled"]
        assert "http://localhost:4317" in labels["phoenix.collector_endpoint"]
        assert "iris" in labels["phoenix.project_name"]


class TestResetMenu:
    """Reset lives in Advanced: one key, or reset-all behind a confirm gate."""

    @patch("esdc.config_wizard.Config.reset_config")
    @patch("esdc.config_wizard.questionary.confirm")
    @patch("esdc.config_wizard.questionary.select")
    @patch("esdc.config_wizard.rich_print")
    def test_reset_all_confirmed(
        self, mock_print, mock_select, mock_confirm, mock_reset
    ):
        from esdc.config_wizard import _advanced_menu

        mock_select.return_value.unsafe_ask.side_effect = ["Reset all", "__back__"]
        mock_confirm.return_value.unsafe_ask.return_value = True

        _advanced_menu()

        mock_reset.assert_called_once_with()

    @patch("esdc.config_wizard.Config.reset_config")
    @patch("esdc.config_wizard.questionary.confirm")
    @patch("esdc.config_wizard.questionary.select")
    @patch("esdc.config_wizard.rich_print")
    def test_reset_all_declined(
        self, mock_print, mock_select, mock_confirm, mock_reset
    ):
        from esdc.config_wizard import _advanced_menu

        mock_select.return_value.unsafe_ask.side_effect = ["Reset all", "__back__"]
        mock_confirm.return_value.unsafe_ask.return_value = False

        _advanced_menu()

        mock_reset.assert_not_called()

    @patch("esdc.config_wizard.Config.reset_config")
    @patch(
        "esdc.config_wizard.Config.get_all_config_flat",
        return_value={"cache.sql_ttl": 604800},
    )
    @patch("esdc.config_wizard.questionary.confirm")
    @patch("esdc.config_wizard.questionary.select")
    @patch("esdc.config_wizard.rich_print")
    def test_reset_single_key(
        self, mock_print, mock_select, mock_confirm, mock_flat, mock_reset
    ):
        from esdc.config_wizard import _advanced_menu

        mock_select.return_value.unsafe_ask.side_effect = [
            "Reset a setting",
            "Cache",
            "cache.sql_ttl",
            "__back__",
        ]
        mock_confirm.return_value.unsafe_ask.return_value = True

        _advanced_menu()

        mock_reset.assert_called_once_with("cache.sql_ttl")


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

        mock_select.return_value.unsafe_ask.side_effect = [
            "openai",
            "(none)",
            "gpt-4o",
            # The draft action menu is the only write; the old post-save test
            # confirm is gone because Test connection is now a menu choice.
            "save",
        ]
        mock_password.return_value.unsafe_ask.return_value = "sk-test"
        mock_confirm.return_value.unsafe_ask.return_value = False

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

        mock_select.return_value.unsafe_ask.side_effect = [
            "ollama",
            "llama3",
            "save",
        ]
        mock_text.return_value.unsafe_ask.side_effect = [
            "http://localhost:11434",
        ]
        mock_confirm.return_value.unsafe_ask.return_value = False

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
        mock_select.return_value.unsafe_ask.return_value = "my-openai"

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
        mock_select.return_value.unsafe_ask.side_effect = ["deepseek", "__done__"]
        mock_confirm.return_value.unsafe_ask.return_value = True

        _set_provider_order_flow()

        mock_set_order.assert_called_once_with(["deepseek", "openai"])

    @patch("esdc.config_wizard.Config.get_provider_references", return_value=[])
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
        mock_references,
    ):
        from esdc.config_wizard import _remove_provider_flow

        mock_get_providers.return_value = {"my-openai": {"provider_type": "openai"}}
        mock_select.return_value.unsafe_ask.return_value = "my-openai"
        mock_confirm.return_value.unsafe_ask.return_value = True

        _remove_provider_flow()

        # Nothing refers to it, so the confirm is reached and the removal happens.
        mock_references.assert_called_once_with("my-openai")
        mock_remove.assert_called_once_with("my-openai")


class TestProviderRemovalReferences:
    """Removal is blocked while a setting still names the provider.

    Every value here is a placeholder: the warning shows key names only, and these
    tests assert that no stored value can reach the output.
    """

    _SAVED = {
        "provider_type": "openai_compatible",
        "base_url": "http://engine:4000/v1",
        "api_key": "placeholder-not-a-secret",
        "model": "chat-placeholder",
    }

    @patch("esdc.config_wizard.Config.get_provider_references")
    @patch("esdc.config_wizard.Config.get_providers")
    @patch("esdc.config_wizard.Config.remove_provider")
    @patch("esdc.config_wizard.questionary.select")
    @patch("esdc.config_wizard.questionary.confirm")
    @patch("esdc.config_wizard.rich_print")
    def test_blocked_removal_lists_the_dependent_keys(
        self,
        mock_print,
        mock_confirm,
        mock_select,
        mock_remove,
        mock_get_providers,
        mock_references,
    ):
        references = [
            "default_provider",
            "provider_order",
            "corpus.ocr_provider",
        ]
        mock_references.return_value = references
        mock_get_providers.return_value = {"engine": dict(self._SAVED)}
        mock_select.return_value.unsafe_ask.return_value = "engine"

        _remove_provider_flow()

        mock_references.assert_called_once_with("engine")
        printed = " ".join(str(call) for call in mock_print.call_args_list)
        for key in references:
            assert key in printed
        # The user must change those settings first, so nothing is confirmed
        # and nothing is removed.
        mock_confirm.assert_not_called()
        mock_remove.assert_not_called()
        # Key names only: no stored value, so no credential can reach the output.
        assert self._SAVED["api_key"] not in printed
        assert self._SAVED["model"] not in printed

    @patch("esdc.config_wizard.Config.get_provider_references", return_value=[])
    @patch("esdc.config_wizard.Config.get_providers")
    @patch("esdc.config_wizard.Config.remove_provider")
    @patch("esdc.config_wizard.questionary.select")
    @patch("esdc.config_wizard.questionary.confirm")
    @patch("esdc.config_wizard.rich_print")
    def test_reference_check_happens_before_the_confirm(
        self,
        mock_print,
        mock_confirm,
        mock_select,
        mock_remove,
        mock_get_providers,
        mock_references,
    ):
        mock_references.return_value = ["embedding_provider"]
        mock_get_providers.return_value = {"engine": dict(self._SAVED)}
        mock_select.return_value.unsafe_ask.return_value = "engine"

        _remove_provider_flow()

        # The reference check runs before the confirm is even built.
        mock_references.assert_called_once_with("engine")
        mock_confirm.assert_not_called()
        mock_remove.assert_not_called()

    @patch("esdc.config_wizard.Config.get_provider_references", return_value=[])
    @patch("esdc.config_wizard.Config.get_providers")
    @patch("esdc.config_wizard.Config.remove_provider")
    @patch("esdc.config_wizard.questionary.select")
    @patch("esdc.config_wizard.questionary.confirm")
    def test_inactive_reference_also_blocks_removal(
        self,
        mock_confirm,
        mock_select,
        mock_remove,
        mock_get_providers,
        mock_references,
    ):
        # corpus.rerank_provider names it while rerank is off: still a reference.
        mock_references.return_value = ["corpus.rerank_provider"]
        mock_get_providers.return_value = {"engine": dict(self._SAVED)}
        mock_select.return_value.unsafe_ask.return_value = "engine"

        _remove_provider_flow()

        mock_confirm.assert_not_called()
        mock_remove.assert_not_called()

    @patch("esdc.config_wizard.Config.get_provider_references", return_value=[])
    @patch("esdc.config_wizard.Config.get_providers")
    @patch("esdc.config_wizard.Config.remove_provider")
    @patch("esdc.config_wizard.questionary.select")
    @patch("esdc.config_wizard.questionary.confirm")
    def test_removal_proceeds_when_nothing_refers_to_it(
        self,
        mock_confirm,
        mock_select,
        mock_remove,
        mock_get_providers,
        mock_references,
    ):
        mock_get_providers.return_value = {"engine": dict(self._SAVED)}
        mock_select.return_value.unsafe_ask.return_value = "engine"
        mock_confirm.return_value.unsafe_ask.return_value = True

        _remove_provider_flow()

        mock_references.assert_called_once_with("engine")
        mock_remove.assert_called_once_with("engine")

    @patch("esdc.config_wizard.Config.get_provider_references", return_value=[])
    @patch("esdc.config_wizard.Config.get_providers")
    @patch("esdc.config_wizard.Config.remove_provider")
    @patch("esdc.config_wizard.questionary.select")
    @patch("esdc.config_wizard.questionary.confirm")
    def test_cancel_at_the_confirm_removes_nothing(
        self,
        mock_confirm,
        mock_select,
        mock_remove,
        mock_get_providers,
        mock_references,
    ):
        mock_get_providers.return_value = {"engine": dict(self._SAVED)}
        mock_select.return_value.unsafe_ask.return_value = "engine"
        # Escape at the confirm: no removal even when nothing refers to it.
        mock_confirm.return_value.unsafe_ask.return_value = None

        _remove_provider_flow()

        mock_remove.assert_not_called()


class TestSharedConnectionChangeWarning:
    """A changed address or key names the tasks that share the provider.

    Nothing here blocks the write, and no test prints or asserts a stored value:
    the warning is checked for the task names only.
    """

    _SAVED = {
        "provider_type": "openai_compatible",
        "base_url": "http://engine:4000/v1",
        "api_key": "placeholder-not-a-secret",
        "model": "chat-placeholder",
    }

    _REFERENCES = [
        "default_provider",
        "provider_order",
        "embedding_provider",
        "corpus.metadata_provider",
        "corpus.metadata_model",
        "corpus.cleanup_provider",
        "corpus.ocr_provider",
        "corpus.rerank_provider",
    ]

    @patch("esdc.config_wizard.Config.get_provider_references")
    @patch("esdc.config_wizard.Config.save_provider")
    @patch("esdc.config_wizard.Config.get_providers")
    @patch("esdc.config_wizard.questionary.text")
    @patch("esdc.config_wizard.questionary.select")
    @patch("esdc.config_wizard.rich_print")
    def test_address_change_warns_with_the_affected_tasks_and_still_saves(
        self,
        mock_print,
        mock_select,
        mock_text,
        mock_get_providers,
        mock_save,
        mock_references,
    ):
        mock_references.return_value = self._REFERENCES
        mock_get_providers.return_value = {"engine": dict(self._SAVED)}
        mock_select.return_value.unsafe_ask.side_effect = [
            "engine",
            "base_url",
            "__done__",
            "save",
        ]
        mock_text.return_value.unsafe_ask.return_value = "http://desk-box:1234/v1"

        _edit_provider_flow()

        warning = str(mock_print.call_args_list[0])
        for task in (
            "Chat",
            "Embeddings",
            "Metadata extraction",
            "Text cleanup",
            "OCR",
            "Reranking",
        ):
            assert task in warning
        assert "base_url" in warning
        # Key names are mapped to tasks, so no stored value is shown.
        assert self._SAVED["api_key"] not in warning
        # The warning does not block the write.
        mock_save.assert_called_once_with(
            "engine",
            {
                "provider_type": "openai_compatible",
                "base_url": "http://desk-box:1234/v1",
                "api_key": "placeholder-not-a-secret",
                "model": "chat-placeholder",
            },
        )

    @patch("esdc.config_wizard.Config.get_provider_references", return_value=[])
    @patch("esdc.config_wizard.Config.save_provider")
    @patch("esdc.config_wizard.Config.get_providers")
    @patch("esdc.config_wizard.questionary.text")
    @patch("esdc.config_wizard.questionary.select")
    @patch("esdc.config_wizard.rich_print")
    def test_warning_is_skipped_when_nothing_refers_to_the_provider(
        self,
        mock_print,
        mock_select,
        mock_text,
        mock_get_providers,
        mock_save,
        mock_references,
    ):
        mock_get_providers.return_value = {"engine": dict(self._SAVED)}
        mock_select.return_value.unsafe_ask.side_effect = [
            "engine",
            "base_url",
            "__done__",
            "save",
        ]
        mock_text.return_value.unsafe_ask.return_value = "http://desk-box:1234/v1"

        _edit_provider_flow()

        printed = " ".join(str(call) for call in mock_print.call_args_list)
        assert "shared by" not in printed
        mock_save.assert_called_once_with(
            "engine",
            {
                "provider_type": "openai_compatible",
                "base_url": "http://desk-box:1234/v1",
                "api_key": "placeholder-not-a-secret",
                "model": "chat-placeholder",
            },
        )

    @patch("esdc.config_wizard.Config.get_provider_references", return_value=[])
    @patch("esdc.config_wizard.Config.save_provider")
    @patch("esdc.config_wizard.Config.get_providers")
    @patch("esdc.config_wizard._fetch_models", return_value=["chat-b"])
    @patch("esdc.config_wizard.questionary.text")
    @patch("esdc.config_wizard.questionary.select")
    @patch("esdc.config_wizard.rich_print")
    def test_model_only_edit_never_warns(
        self,
        mock_print,
        mock_select,
        mock_text,
        mock_fetch,
        mock_get_providers,
        mock_save,
        mock_references,
    ):
        """A normal task model edit is not a provider connection change."""
        mock_get_providers.return_value = {"engine": dict(self._SAVED)}
        mock_select.return_value.unsafe_ask.side_effect = [
            "engine",
            "model",
            "chat-b",
            "__done__",
            "save",
        ]

        _edit_provider_flow()

        printed = " ".join(str(call) for call in mock_print.call_args_list)
        assert "shared by" not in printed
        assert mock_save.call_args[0][1]["model"] == "chat-b"

    @patch("esdc.config_wizard.Config.rename_provider")
    @patch("esdc.config_wizard.Config.save_provider")
    @patch("esdc.config_wizard.Config.get_providers")
    @patch("esdc.config_wizard.questionary.text")
    @patch("esdc.config_wizard.questionary.password")
    @patch("esdc.config_wizard.questionary.select")
    @patch("esdc.config_wizard.rich_print")
    def test_warning_uses_the_saved_key_while_a_rename_is_pending(
        self,
        mock_print,
        mock_select,
        mock_password,
        mock_text,
        mock_get_providers,
        mock_save,
        mock_rename,
    ):
        mock_get_providers.return_value = {"engine": dict(self._SAVED)}
        mock_select.return_value.unsafe_ask.side_effect = [
            "engine",
            "name",
            "api_key",
            "__done__",
            "save",
        ]
        mock_text.return_value.unsafe_ask.return_value = "desk"
        mock_password.return_value.unsafe_ask.return_value = "placeholder-new-key"

        # Without an explicit patch the reference check would read the real config.
        with patch.object(
            Config,
            "get_provider_references",
            return_value=["corpus.ocr_provider"],
        ) as mock_references:
            _edit_provider_flow()

        warning = str(mock_print.call_args_list[0])
        # The stored reference still names the saved key at the moment of the warning.
        assert mock_references.call_args_list[0][0][0] == "engine"
        assert "OCR" in warning
        assert "engine" in warning
        assert self._SAVED["api_key"] not in warning
        # The rename and the draft fields are one write, so no second save follows.
        mock_rename.assert_called_once_with(
            "engine",
            "desk",
            provider_config={
                "provider_type": "openai_compatible",
                "base_url": "http://engine:4000/v1",
                "api_key": "placeholder-new-key",
                "model": "chat-placeholder",
            },
        )
        mock_save.assert_not_called()

    @patch("esdc.config_wizard.Config.rename_provider")
    @patch("esdc.config_wizard.Config.remove_provider")
    @patch("esdc.config_wizard.Config.get_provider_references")
    @patch("esdc.config_wizard.Config.save_provider")
    @patch("esdc.config_wizard.Config.get_providers")
    @patch("esdc.config_wizard.Config.get_default_provider", return_value="openai")
    @patch("esdc.config_wizard.questionary.confirm")
    @patch("esdc.config_wizard.questionary.password")
    @patch("esdc.config_wizard.questionary.select")
    @patch("esdc.config_wizard._fetch_models", return_value=["chat-b"])
    @patch("esdc.config_wizard.rich_print")
    def test_overwrite_warns_before_the_write(
        self,
        mock_print,
        mock_fetch,
        mock_select,
        mock_password,
        mock_confirm,
        mock_default,
        mock_get_providers,
        mock_save,
        mock_references,
        mock_remove,
        mock_rename,
    ):
        mock_references.return_value = ["provider_order", "embedding_provider"]
        saved = {
            "provider_type": "openai",
            "api_key": "placeholder-not-a-secret",
            "model": "chat-placeholder",
        }
        # The same key is overwritten, so the reference keeps pointing at it.
        mock_get_providers.return_value = {"openai": saved}
        # openai is not exempt from the same-type lookup, so the overwrite
        # confirmation runs for it: type, effort, model, then the action menu.
        mock_select.return_value.unsafe_ask.side_effect = [
            "openai",
            "(none)",
            "chat-b",
            "save",
        ]
        mock_password.return_value.unsafe_ask.return_value = "placeholder-new-key"
        mock_confirm.return_value.unsafe_ask.side_effect = [True, False]

        _add_provider_flow()

        printed = " ".join(str(call) for call in mock_print.call_args_list)
        assert "Chat" in printed
        assert "Embeddings" in printed
        # Only the changed field is named, and no stored value is shown.
        assert "api_key changed" in printed
        assert "base_url" not in printed
        assert "placeholder-not-a-secret" not in printed
        # The write is in place: nothing is deleted first, so references survive.
        mock_save.assert_called_once_with(
            "openai",
            {
                "provider_type": "openai",
                "api_key": "placeholder-new-key",
                "model": "chat-b",
            },
        )
        mock_remove.assert_not_called()
        mock_rename.assert_not_called()


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
        mock_select.return_value.unsafe_ask.return_value = "__back__"

        _edit_section("Logging", ["logging.level"])

        # When back is selected, no edits should happen
        mock_prompt.assert_not_called()
        mock_set.assert_not_called()

class TestGeneralConfigListsCorpusKeys:
    """Corpus keys appear in the edit list even when absent from config.yaml."""

    @patch("esdc.config_wizard._select_with_back", return_value="__back__")
    @patch("esdc.config_wizard.Config.get_all_config_flat")
    def test_corpus_defaults_merged(self, mock_flat, mock_select):
        from esdc.config_wizard import _edit_section

        mock_flat.return_value = {"api_url": "http://x"}  # no corpus.* in file
        _edit_section("Corpus processing", ADVANCED_SECTIONS["Corpus processing"])

        choices = mock_select.call_args.kwargs.get("choices") or (
            mock_select.call_args.args[1] if len(mock_select.call_args.args) > 1 else []
        )
        values = [getattr(c, "value", None) for c in choices]
        assert "corpus.chunk_size" in values
        assert "corpus.ollama_host" not in values
        assert "corpus.num_ctx" in values

    def test_model_keys_live_with_their_task_not_a_flat_section(self):
        """Task screens own the model fields; Advanced owns tuning keys only."""
        assert "corpus.ocr_model" in TASK_SECTIONS["OCR"]
        assert "corpus.metadata_model" in TASK_SECTIONS["Metadata extraction"]
        assert "corpus.cleanup_model" in TASK_SECTIONS["Text cleanup"]
        assert "corpus.ollama_host" not in ADVANCED_SECTIONS["Corpus processing"]
        assert "corpus.num_ctx" in ADVANCED_SECTIONS["Corpus processing"]
        assert (
            "semantic_search.embedding_batch_size"
            in ADVANCED_SECTIONS["Corpus processing"]
        )
        assert "tool_format" in ADVANCED_SECTIONS["Corpus processing"]


class TestRerankKeyWidgets:
    """corpus.rerank* keys must use int/bool/select widgets, not generic text."""

    @patch("esdc.config_wizard.questionary.text")
    def test_rerank_pool_edit_returns_int(self, mock_text):
        mock_text.return_value.unsafe_ask.return_value = "50"
        result = _prompt_for_config_value("corpus.rerank_pool", 30)
        assert result == 50
        assert isinstance(result, int)

    def test_rerank_pool_in_int_keys(self):
        assert "corpus.rerank_pool" in Config.INT_KEYS

    @patch("esdc.config_wizard.questionary.select")
    def test_rerank_edit_returns_bool(self, mock_select):
        mock_select.return_value.unsafe_ask.return_value = "True"
        result = _prompt_for_config_value("corpus.rerank", False)
        assert result is True
        assert isinstance(result, bool)

    def test_rerank_in_boolean_keys(self):
        assert "corpus.rerank" in Config.BOOLEAN_KEYS

    @patch("esdc.config_wizard.questionary.text")
    def test_rerank_model_is_a_plain_text_prompt(self, mock_text):
        """No curated GGUF list.

        A service model id is typed, and any nonempty value is accepted unchanged.
        """
        mock_text.return_value.unsafe_ask.return_value = "my-custom-reranker"
        result = _prompt_for_config_value("corpus.rerank_model", "previous-model")
        assert result == "my-custom-reranker"
        assert mock_text.return_value.unsafe_ask.call_count == 1

    @patch("esdc.config_wizard.questionary.text")
    def test_legacy_gguf_rerank_model_is_accepted_unchanged(self, mock_text):
        """A stored GGUF id is the default and comes back exactly as stored."""
        stored = "ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF"
        mock_text.return_value.unsafe_ask.return_value = stored
        result = _prompt_for_config_value("corpus.rerank_model", stored)
        assert result == stored
        assert mock_text.call_args.kwargs["default"] == stored

    @patch("esdc.config_wizard.questionary.text")
    def test_gguf_rerank_model_prompt_carries_the_hint(self, mock_text):
        """The prompt shows the hint for the stored value it is reading."""
        stored = "ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF"
        flat = {"corpus.rerank_model": stored}
        mock_text.return_value.unsafe_ask.return_value = stored
        _prompt_for_config_value("corpus.rerank_model", stored, flat)
        instruction = mock_text.call_args.kwargs["instruction"]
        assert "needs configuring" in instruction.lower()
        assert "preserved" in instruction

    @patch("esdc.config_wizard.questionary.select")
    def test_rerank_backend_offers_only_a_service_choice(self, mock_select):
        """A stored `local` is never offered as a working choice."""
        mock_select.return_value.unsafe_ask.return_value = "provider"
        result = _prompt_for_config_value("corpus.rerank_backend", "provider")
        choices = mock_select.call_args.kwargs["choices"]
        values = [c.value for c in choices]
        assert values == ["provider"]
        assert [str(c.title) for c in choices] == ["HTTP rerank service"]
        assert result == "provider"

    @patch("esdc.config_wizard.questionary.select")
    def test_retired_rerank_backend_shows_only_the_retired_row(self, mock_select):
        stored = "local"
        mock_select.return_value.unsafe_ask.return_value = stored
        result = _prompt_for_config_value("corpus.rerank_backend", stored)
        choices = mock_select.call_args.kwargs["choices"]
        titles = [str(c.title) for c in choices]
        # The retired row is shown once, and no plain `local` row duplicates it.
        assert titles.count("local (retired — choose a rerank service)") == 1
        assert "local" not in titles
        assert result == stored
        assert mock_select.call_args.kwargs["default"] == stored

    @patch("esdc.config_wizard.questionary.select")
    def test_retired_rerank_backend_choice_resolves_normally(self, mock_select):
        mock_select.return_value.unsafe_ask.return_value = "provider"
        result = _prompt_for_config_value("corpus.rerank_backend", "local")
        assert result == "provider"

    @patch("esdc.config_wizard.questionary.select")
    def test_retired_embedding_backend_shows_only_service_rows(self, mock_select):
        stored = "local"
        mock_select.return_value.unsafe_ask.return_value = stored
        result = _prompt_for_config_value("embedding_backend", stored)
        choices = mock_select.call_args.kwargs["choices"]
        titles = [str(c.title) for c in choices]
        assert titles == [
            "local (retired — choose an HTTP service)",
            "Ollama HTTP service",
            "OpenAI-compatible HTTP service",
        ]
        assert result == stored
        assert mock_select.call_args.kwargs["default"] == stored

    @patch("esdc.config_wizard.questionary.select")
    def test_retired_embedding_backend_choice_resolves_normally(self, mock_select):
        mock_select.return_value.unsafe_ask.return_value = "openai"
        result = _prompt_for_config_value("embedding_backend", "local")
        assert result == "openai"

    @patch("esdc.config_wizard.questionary.select")
    def test_retired_field_prompts_write_nothing(self, mock_select):
        """The prompt is a read: the field hint never changes the stored value.

        Choosing the retired row returns the stored value as it is, so the wizard has
        nothing to write; the draft it read is left untouched.
        """
        flat = {
            "embedding_backend": "local",
            "corpus.rerank_backend": "local",
            "corpus.rerank_model": "ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF",
        }
        mock_select.return_value.unsafe_ask.side_effect = ["local", "local"]
        assert _prompt_for_config_value("embedding_backend", "local", flat) == "local"
        assert (
            _prompt_for_config_value("corpus.rerank_backend", "local", flat) == "local"
        )
        assert flat == {
            "embedding_backend": "local",
            "corpus.rerank_backend": "local",
            "corpus.rerank_model": "ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF",
        }

    def test_coerce_rerank_pool_to_int(self):
        assert Config._coerce_value("corpus.rerank_pool", "30") == 30

    def test_coerce_rerank_to_bool(self):
        assert Config._coerce_value("corpus.rerank", "true") is True


_FAKE_PROVIDERS = {
    "engine": {
        "provider_type": "openai_compatible",
        "base_url": "http://engine:4000/v1",
        "api_key": "sk-fake-not-a-real-secret",
        "model": "chat-a",
    },
    "gpu": {
        "provider_type": "ollama",
        "base_url": "http://gpu-box:11434",
        "model": "ollama-a",
    },
    "desk": {"provider_type": "anthropic", "model": "claude-x"},
}

_ALL_ROLES = ("chat", "embedding", "rerank", "ocr", "metadata", "cleanup")


def _catalog(*model_ids, roles=frozenset(), vision=None, warning=None):
    """One catalog with unknown roles, so every task filter accepts it."""
    return ModelCatalog(
        entries=tuple(
            ModelEntry(model_id, frozenset(roles), vision) for model_id in model_ids
        ),
        warning=warning,
    )


def _row_labels(mock_select) -> list[str]:
    choices = mock_select.call_args.kwargs["choices"]
    return [str(choice.title) for choice in choices]


class TestTaskPicker:
    """One combined searchable provider/model list per task."""

    @patch("esdc.config_wizard.discover_models", return_value=_catalog("m1"))
    @patch("esdc.config_wizard.Config.get_all_config_flat", return_value={})
    @patch(
        "esdc.config_wizard.Config.get_providers", return_value=dict(_FAKE_PROVIDERS)
    )
    @patch("esdc.config_wizard.rich_print")
    @patch("esdc.config_wizard.questionary.select")
    def test_pair_label_is_identical_across_all_six_tasks(
        self, mock_select, mock_print, mock_providers, mock_flat, mock_discover
    ):
        from esdc.config_wizard import _choose_provider_model

        for role in _ALL_ROLES:
            mock_select.return_value.unsafe_ask.side_effect = [None]
            _choose_provider_model(role, None)
            labels = _row_labels(mock_select)
            # Exactly the same label text for the same saved provider/model pair.
            assert "engine / m1" in labels, role

    @patch("esdc.config_wizard.discover_models", return_value=_catalog("m1"))
    @patch("esdc.config_wizard.Config.get_all_config_flat", return_value={})
    @patch(
        "esdc.config_wizard.Config.get_providers", return_value=dict(_FAKE_PROVIDERS)
    )
    @patch("esdc.config_wizard.questionary.select")
    def test_list_is_searchable_with_jk_keys_disabled(
        self, mock_select, mock_providers, mock_flat, mock_discover
    ):
        mock_select.return_value.unsafe_ask.return_value = PairSelection("engine", "m1")
        result = _choose_provider_model("chat", None)

        kwargs = mock_select.call_args.kwargs
        assert kwargs.get("use_search_filter") is True
        assert kwargs.get("use_jk_keys") is False
        assert result == PairSelection("engine", "m1")

    @patch("esdc.config_wizard.discover_models", return_value=_catalog("m1"))
    @patch("esdc.config_wizard.Config.get_all_config_flat", return_value={})
    @patch(
        "esdc.config_wizard.Config.get_providers", return_value=dict(_FAKE_PROVIDERS)
    )
    @patch("esdc.config_wizard.questionary.select")
    def test_one_select_picks_provider_and_model_together(
        self, mock_select, mock_providers, mock_flat, mock_discover
    ):
        """No provider-selection submenu: one list, one choice, both values."""
        from esdc.config_wizard import _choose_provider_model

        _choose_provider_model("embedding", None)
        assert mock_select.call_count == 1
        rows = mock_select.call_args.kwargs["choices"]
        labels = [str(c.title) for c in rows]
        assert "engine / m1" in labels
        assert any(label.startswith("gpu / ") for label in labels)
        # Rows for two different providers sit at the same level of one list.
        pairs = [c.value for c in rows if isinstance(c.value, PairSelection)]
        assert len({p.provider_name for p in pairs}) >= 2

    @patch("esdc.config_wizard.Config.get_all_config_flat", return_value={})
    @patch(
        "esdc.config_wizard.Config.get_providers",
        return_value={"gpu": _FAKE_PROVIDERS["gpu"]},
    )
    @patch("esdc.config_wizard.discover_models")
    @patch("esdc.config_wizard.questionary.select")
    def test_ollama_embedding_row_shows_the_pinned_model_without_discovery(
        self, mock_select, mock_discover, mock_providers, mock_flat
    ):
        """The Ollama embedding backend pins its model, so the row is read-only."""
        mock_discover.return_value = _catalog("m1")
        mock_select.return_value.unsafe_ask.side_effect = [None]
        _choose_provider_model("embedding", None)

        assert f"gpu / {OllamaEmbedder.OLLAMA_TAG}" in _row_labels(mock_select)
        mock_discover.assert_not_called()

    @patch("esdc.config_wizard.Config.get_all_config_flat", return_value={})
    @patch(
        "esdc.config_wizard.Config.get_providers", return_value=dict(_FAKE_PROVIDERS)
    )
    @patch("esdc.config_wizard.discover_models", return_value=_catalog("m1"))
    @patch("esdc.config_wizard.questionary.select")
    def test_back_and_escape_return_none(
        self, mock_select, mock_discover, mock_providers, mock_flat
    ):
        from esdc.config_wizard import _choose_provider_model

        mock_select.return_value.unsafe_ask.side_effect = ["__back__"]
        assert _choose_provider_model("ocr", None) is None

        mock_select.return_value.unsafe_ask.side_effect = [None]
        assert _choose_provider_model("ocr", None) is None

    @patch("esdc.config_wizard.discover_models", return_value=_catalog("m1"))
    @patch("esdc.config_wizard.Config.get_all_config_flat", return_value={})
    @patch(
        "esdc.config_wizard.Config.get_providers", return_value=dict(_FAKE_PROVIDERS)
    )
    @patch("esdc.config_wizard.questionary.select")
    def test_special_rows_are_at_the_same_list_level(
        self, mock_select, mock_providers, mock_flat, mock_discover
    ):
        from esdc.config_wizard import _choose_provider_model

        mock_select.return_value.unsafe_ask.side_effect = [None]
        _choose_provider_model("metadata", None)
        labels = _row_labels(mock_select)
        assert "Chat setting / main" in labels
        assert "Off / image fallback" in labels
        assert "engine / (chat model)" in labels
        assert "Ollama (local) / m1" in labels
        assert "Custom provider / model…" in labels
        assert "Retry discovery" in labels
        assert "← Back" in labels

        mock_select.return_value.unsafe_ask.side_effect = [None]
        _choose_provider_model("cleanup", None)
        labels = _row_labels(mock_select)
        assert "Off" in labels and "Chat setting / main" in labels

    @patch("esdc.config_wizard.discover_models", return_value=_catalog("embed-1"))
    @patch("esdc.config_wizard.Config.get_all_config_flat", return_value={})
    @patch(
        "esdc.config_wizard.Config.get_providers", return_value=dict(_FAKE_PROVIDERS)
    )
    @patch("esdc.config_wizard.questionary.select")
    def test_embedding_rows_are_service_rows(
        self, mock_select, mock_providers, mock_flat, mock_discover
    ):
        """No in-process row: the Ollama service row and a direct endpoint row."""
        mock_select.return_value.unsafe_ask.side_effect = [None]
        _choose_provider_model("embedding", None)
        labels = _row_labels(mock_select)
        assert f"Ollama (local) / {OllamaEmbedder.OLLAMA_TAG}" in labels
        assert not any("GGUF" in label or "in-process" in label for label in labels)

        # A saved direct endpoint with no reference shows its own row.
        flat = {
            "embedding_backend": "openai",
            "embedding_provider": "",
            "embedding_host": "http://direct-box:1234",
            "embedding_model": "embed-1",
        }
        with patch("esdc.config_wizard.Config.get_all_config_flat", return_value=flat):
            mock_select.return_value.unsafe_ask.side_effect = [None]
            _choose_provider_model("embedding", None)
            labels = _row_labels(mock_select)
            direct = "Direct endpoint / http://direct-box:1234 / embed-1"
            assert direct in labels
            row = [
                c for c in mock_select.call_args.kwargs["choices"] if c.title == direct
            ]
            assert row[0].value == DirectSelection(
                "openai", "http://direct-box:1234", "embed-1"
            )

    @patch("esdc.config_wizard.discover_models", return_value=_catalog("m1"))
    @patch("esdc.config_wizard.Config.get_all_config_flat", return_value={})
    @patch(
        "esdc.config_wizard.Config.get_providers", return_value=dict(_FAKE_PROVIDERS)
    )
    @patch("esdc.config_wizard.questionary.select")
    def test_rerank_has_no_local_rows_and_ocr_keeps_its_ollama_row(
        self, mock_select, mock_providers, mock_flat, mock_discover
    ):
        # Rerank rows are saved provider pairs only; OCR keeps its Ollama service row.
        mock_select.return_value.unsafe_ask.side_effect = [None]
        _choose_provider_model("rerank", None)
        labels = _row_labels(mock_select)
        assert not any(label.startswith("Local /") for label in labels)
        assert "engine / m1" in labels

        flat = {"corpus.ocr_model": "vision-x", "corpus.ocr_provider": ""}
        with patch("esdc.config_wizard.Config.get_all_config_flat", return_value=flat):
            mock_select.return_value.unsafe_ask.side_effect = [None]
            _choose_provider_model("ocr", None)
            assert "Ollama (local) / vision-x" in _row_labels(mock_select)


class TestDiscoveryFetchPolicy:
    """Discovery is fetched once per picker open, or on Retry (D6)."""

    _SAVED = {"engine": _FAKE_PROVIDERS["engine"], "gpu": _FAKE_PROVIDERS["gpu"]}

    @patch("esdc.config_wizard.discover_models")
    @patch("esdc.config_wizard.Config.get_all_config_flat", return_value={})
    @patch("esdc.config_wizard.Config.get_providers")
    @patch("esdc.config_wizard.questionary.select")
    def test_fetched_once_per_open(
        self, mock_select, mock_providers, mock_flat, mock_discover
    ):
        mock_providers.return_value = dict(self._SAVED)
        mock_discover.return_value = _catalog("m1")
        mock_select.return_value.unsafe_ask.side_effect = [
            PairSelection("engine", "m1")
        ]

        _choose_provider_model("chat", None)

        assert mock_discover.call_count == len(self._SAVED)

    @patch("esdc.config_wizard.discover_models")
    @patch("esdc.config_wizard.Config.get_all_config_flat", return_value={})
    @patch("esdc.config_wizard.Config.get_providers")
    @patch("esdc.config_wizard.questionary.select")
    def test_retry_clears_the_cache_and_refetches(
        self, mock_select, mock_providers, mock_flat, mock_discover
    ):
        mock_providers.return_value = dict(self._SAVED)
        mock_discover.return_value = _catalog("m1")
        mock_select.return_value.unsafe_ask.side_effect = [
            "__retry__",
            PairSelection("engine", "m1"),
        ]

        _choose_provider_model("chat", None)

        assert mock_discover.call_count == 2 * len(self._SAVED)

    @patch("esdc.config_wizard.discover_models")
    @patch("esdc.config_wizard.Config.get_all_config_flat", return_value={})
    @patch("esdc.config_wizard.Config.get_providers")
    def test_filtering_and_redraw_do_not_refetch(
        self, mock_providers, mock_flat, mock_discover
    ):
        """Rows are rebuilt from the local cache, not from the network."""
        mock_providers.return_value = dict(self._SAVED)
        mock_discover.return_value = _catalog("m1")
        cache = {}

        from esdc.config_wizard import _task_rows

        _task_rows("chat", None, cache)
        _task_rows("chat", None, cache)

        assert mock_discover.call_count == len(self._SAVED)
        assert len(cache) == len(self._SAVED)

    @patch("esdc.config_wizard.discover_models")
    @patch("esdc.config_wizard.Config.get_all_config_flat", return_value={})
    @patch("esdc.config_wizard.Config.get_providers")
    @patch("esdc.config_wizard.questionary.select")
    def test_no_global_cache_between_picker_opens(
        self, mock_select, mock_providers, mock_flat, mock_discover
    ):
        mock_providers.return_value = dict(self._SAVED)
        mock_discover.return_value = _catalog("m1")
        mock_select.return_value.unsafe_ask.side_effect = [
            PairSelection("engine", "m1"),
            PairSelection("gpu", "m1"),
        ]

        _choose_provider_model("chat", None)
        _choose_provider_model("chat", None)

        assert mock_discover.call_count == 2 * len(self._SAVED)

    @patch("esdc.config_wizard.discover_models")
    @patch("esdc.config_wizard.Config.get_all_config_flat", return_value={})
    @patch("esdc.config_wizard.Config.get_providers")
    @patch("esdc.config_wizard.rich_print")
    @patch("esdc.config_wizard.questionary.select")
    def test_discovery_warning_reveals_no_credential(
        self, mock_select, mock_print, mock_providers, mock_flat, mock_discover
    ):
        mock_providers.return_value = {"engine": _FAKE_PROVIDERS["engine"]}
        mock_discover.return_value = _catalog(warning="timeout")
        mock_select.return_value.unsafe_ask.side_effect = [None]

        _choose_provider_model("chat", None)

        printed = " ".join(str(call) for call in mock_print.call_args_list)
        assert "timeout" in printed
        assert _FAKE_PROVIDERS["engine"]["api_key"] not in printed


class TestUnavailableCatalog:
    """A failed catalog keeps the saved and custom choices usable."""

    @patch("esdc.config_wizard.discover_models")
    @patch("esdc.config_wizard.Config.get_all_config_flat", return_value={})
    @patch("esdc.config_wizard.Config.get_providers")
    @patch("esdc.config_wizard.questionary.select")
    def test_saved_pair_stays_visible_and_marked(
        self, mock_select, mock_providers, mock_flat, mock_discover
    ):
        mock_providers.return_value = {"engine": _FAKE_PROVIDERS["engine"]}
        mock_discover.return_value = _catalog(warning="connection error")
        mock_select.return_value.unsafe_ask.side_effect = [None]

        with patch(
            "esdc.config_wizard.Config.get_all_config_flat",
            return_value={"default_provider": "engine"},
        ):
            _choose_provider_model("chat", None)

        labels = _row_labels(mock_select)
        assert "engine / chat-a (saved, unverified)" in labels
        assert "Custom provider / model…" in labels
        assert "Retry discovery" in labels

    @patch("esdc.config_wizard.discover_models")
    @patch("esdc.config_wizard.Config.get_all_config_flat", return_value={})
    @patch("esdc.config_wizard.Config.get_providers")
    @patch("esdc.config_wizard.questionary.select")
    def test_unsupported_provider_type_shows_only_the_saved_pair(
        self, mock_select, mock_providers, mock_flat, mock_discover
    ):
        mock_providers.return_value = {"desk": _FAKE_PROVIDERS["desk"]}
        mock_select.return_value.unsafe_ask.side_effect = [None]

        _choose_provider_model("chat", None)

        labels = _row_labels(mock_select)
        assert labels == [
            "desk / claude-x (saved, unverified)",
            "Custom provider / model…",
            "Retry discovery",
            "← Back",
        ]
        mock_discover.assert_not_called()

    @patch("esdc.config_wizard._select_with_back", return_value="__back__")
    @patch(
        "esdc.config_wizard.Config.get_all_config_flat",
        return_value={"api_url": "http://x"},
    )
    def test_a_basic_setting_edit_is_not_blocked(self, mock_flat, mock_select_back):
        from esdc.config_wizard import _edit_section

        _edit_section(
            "Application settings", APPLICATION_SECTIONS["Application settings"]
        )
        assert mock_select_back.called


class TestCustomPairFlow:
    """Custom resolves to a concrete pair before Save; nothing else is persisted."""

    @patch("esdc.config_wizard.discover_models", return_value=_catalog("m1"))
    @patch("esdc.config_wizard.Config.get_all_config_flat", return_value={})
    @patch(
        "esdc.config_wizard.Config.get_providers",
        return_value={"engine": _FAKE_PROVIDERS["engine"]},
    )
    @patch("esdc.config_wizard.questionary.text")
    @patch("esdc.config_wizard.rich_print")
    @patch("esdc.config_wizard.questionary.select")
    def test_custom_resolves_to_a_concrete_pair(
        self,
        mock_select,
        mock_print,
        mock_text,
        mock_providers,
        mock_flat,
        mock_discover,
    ):
        mock_select.return_value.unsafe_ask.side_effect = ["__custom__", "engine"]
        mock_text.return_value.unsafe_ask.return_value = "typed-model"

        result = _choose_provider_model("chat", None)

        assert result == PairSelection("engine", "typed-model")
        assert not isinstance(result, CustomSelection)

    @patch("esdc.config_wizard.discover_models", return_value=_catalog("m1"))
    @patch("esdc.config_wizard.Config.get_all_config_flat", return_value={})
    @patch(
        "esdc.config_wizard.Config.get_providers",
        return_value={"engine": _FAKE_PROVIDERS["engine"]},
    )
    @patch("esdc.config_wizard.rich_print")
    @patch("esdc.config_wizard.questionary.select")
    def test_escape_at_the_custom_prompts_returns_to_the_picker(
        self, mock_select, mock_print, mock_providers, mock_flat, mock_discover
    ):
        mock_select.return_value.unsafe_ask.side_effect = [
            "__custom__",
            None,
            PairSelection("engine", "m1"),
        ]

        assert _choose_provider_model("chat", None) == PairSelection("engine", "m1")

    def test_custom_is_never_saved(self):
        with pytest.raises(ValueError):
            _save_task_selection("chat", CustomSelection("engine", "typed-model"))


class TestCurrentTaskSelection:
    """The saved config maps back to the same selection shape the picker uses."""

    @patch(
        "esdc.config_wizard.Config.get_providers", return_value=dict(_FAKE_PROVIDERS)
    )
    @patch(
        "esdc.config_wizard.Config.get_all_config_flat",
        return_value={"default_provider": "engine"},
    )
    def test_chat(self, mock_flat, mock_providers):
        assert _current_task_selection("chat") == PairSelection("engine", "chat-a")

    @patch(
        "esdc.config_wizard.Config.get_providers", return_value=dict(_FAKE_PROVIDERS)
    )
    @patch(
        "esdc.config_wizard.Config.get_all_config_flat",
        return_value={
            "embedding_backend": "local",
            "embedding_provider": "engine",
            "embedding_model": "embed-1",
        },
    )
    def test_embedding_local_is_reported_as_needing_configuration(
        self, mock_flat, mock_providers
    ):
        """A retired stored backend is a status, and reading it writes nothing."""
        selection = _current_task_selection("embedding")
        assert isinstance(selection, NeedsConfigurationSelection)
        assert selection.reason == _RETIRED_EMBEDDING_REASON
        assert not _selectable(selection)
        assert mock_flat.return_value["embedding_backend"] == "local"
        assert mock_flat.return_value["embedding_model"] == "embed-1"

    @patch(
        "esdc.config_wizard.Config.get_providers", return_value=dict(_FAKE_PROVIDERS)
    )
    @patch(
        "esdc.config_wizard.Config.get_all_config_flat",
        return_value={
            "embedding_backend": "openai",
            "embedding_provider": "engine",
            "embedding_model": "embed-1",
        },
    )
    def test_embedding_pair(self, mock_flat, mock_providers):
        assert _current_task_selection("embedding") == PairSelection(
            "engine", "embed-1"
        )

    @patch(
        "esdc.config_wizard.Config.get_providers", return_value=dict(_FAKE_PROVIDERS)
    )
    @patch(
        "esdc.config_wizard.Config.get_all_config_flat",
        return_value={
            "embedding_backend": "openai",
            "embedding_provider": "",
            "embedding_host": "http://direct-box:1234",
            "embedding_model": "embed-1",
        },
    )
    def test_embedding_direct_endpoint(self, mock_flat, mock_providers):
        assert _current_task_selection("embedding") == DirectSelection(
            "openai", "http://direct-box:1234", "embed-1"
        )

    @patch(
        "esdc.config_wizard.Config.get_providers", return_value=dict(_FAKE_PROVIDERS)
    )
    @patch(
        "esdc.config_wizard.Config.get_all_config_flat",
        return_value={"embedding_backend": "ollama", "embedding_provider": ""},
    )
    def test_embedding_ollama_without_reference(self, mock_flat, mock_providers):
        # The direct Ollama backend is its own row, distinct from the local GGUF row.
        assert _current_task_selection("embedding") == DirectSelection(
            "ollama", "", OllamaEmbedder.OLLAMA_TAG
        )

    @patch(
        "esdc.config_wizard.Config.get_providers", return_value=dict(_FAKE_PROVIDERS)
    )
    @patch(
        "esdc.config_wizard.Config.get_all_config_flat",
        return_value={
            "corpus.rerank_backend": "local",
            "corpus.rerank_model": "gguf-x",
        },
    )
    def test_rerank_local_is_reported_as_needing_configuration(
        self, mock_flat, mock_providers
    ):
        selection = _current_task_selection("rerank")
        assert selection == NeedsConfigurationSelection(_RETIRED_RERANK_REASON)
        assert isinstance(selection, NeedsConfigurationSelection)
        assert not _selectable(selection)
        # The legacy GGUF id is not read as a model file and is not rewritten.
        assert mock_flat.return_value["corpus.rerank_model"] == "gguf-x"

    @patch(
        "esdc.config_wizard.Config.get_providers", return_value=dict(_FAKE_PROVIDERS)
    )
    @patch(
        "esdc.config_wizard.Config.get_all_config_flat",
        return_value={
            "corpus.rerank_backend": "provider",
            "corpus.rerank_provider": "engine",
            "corpus.rerank_model": "gguf-x",
        },
    )
    def test_rerank_provider_pair(self, mock_flat, mock_providers):
        assert _current_task_selection("rerank") == PairSelection("engine", "gguf-x")

    @patch(
        "esdc.config_wizard.Config.get_providers", return_value=dict(_FAKE_PROVIDERS)
    )
    @patch(
        "esdc.config_wizard.Config.get_all_config_flat",
        return_value={"corpus.ocr_provider": "", "corpus.ocr_model": "vision-x"},
    )
    def test_ocr_local(self, mock_flat, mock_providers):
        assert _current_task_selection("ocr") == LocalSelection("vision-x")

    @patch(
        "esdc.config_wizard.Config.get_providers", return_value=dict(_FAKE_PROVIDERS)
    )
    @patch(
        "esdc.config_wizard.Config.get_all_config_flat",
        return_value={"corpus.ocr_provider": "engine", "corpus.ocr_model": "vision-x"},
    )
    def test_ocr_pair(self, mock_flat, mock_providers):
        assert _current_task_selection("ocr") == PairSelection("engine", "vision-x")

    @patch(
        "esdc.config_wizard.Config.get_providers", return_value=dict(_FAKE_PROVIDERS)
    )
    @patch(
        "esdc.config_wizard.Config.get_all_config_flat",
        return_value={"corpus.metadata_provider": "", "corpus.metadata_model": "main"},
    )
    def test_metadata_chat_default(self, mock_flat, mock_providers):
        assert _current_task_selection("metadata") == ChatDefaultSelection()

    @patch(
        "esdc.config_wizard.Config.get_providers", return_value=dict(_FAKE_PROVIDERS)
    )
    @patch(
        "esdc.config_wizard.Config.get_all_config_flat",
        return_value={
            "corpus.metadata_provider": "",
            "corpus.metadata_model": "provider:engine",
        },
    )
    def test_metadata_linked_provider(self, mock_flat, mock_providers):
        assert _current_task_selection("metadata") == LinkedProviderSelection("engine")

    @patch(
        "esdc.config_wizard.Config.get_providers", return_value=dict(_FAKE_PROVIDERS)
    )
    @patch(
        "esdc.config_wizard.Config.get_all_config_flat",
        return_value={"corpus.cleanup_provider": "", "corpus.cleanup_model": ""},
    )
    def test_cleanup_off(self, mock_flat, mock_providers):
        assert _current_task_selection("cleanup") == OffSelection()

    @patch(
        "esdc.config_wizard.Config.get_providers", return_value=dict(_FAKE_PROVIDERS)
    )
    @patch(
        "esdc.config_wizard.Config.get_all_config_flat",
        return_value={
            "corpus.cleanup_provider": "",
            "corpus.cleanup_model": "qwen3:8b",
        },
    )
    def test_cleanup_bare_ollama_name(self, mock_flat, mock_providers):
        assert _current_task_selection("cleanup") == LocalSelection("qwen3:8b")

    @patch(
        "esdc.config_wizard.Config.get_providers", return_value=dict(_FAKE_PROVIDERS)
    )
    @patch(
        "esdc.config_wizard.Config.get_all_config_flat",
        return_value={
            "corpus.cleanup_provider": "engine",
            "corpus.cleanup_model": "wire-model",
        },
    )
    def test_cleanup_new_reference(self, mock_flat, mock_providers):
        assert _current_task_selection("cleanup") == PairSelection(
            "engine", "wire-model"
        )


class TestSaveMappings:
    """Save writes only the fields of the mapping."""

    @patch("esdc.config_wizard.Config.set_chat_selection")
    @patch("esdc.config_wizard.Config.set_config_values")
    def test_chat_save_is_one_atomic_write(self, mock_values, mock_chat):
        _save_task_selection("chat", PairSelection("engine", "chat-a"))
        mock_chat.assert_called_once_with("engine", "chat-a")
        mock_values.assert_not_called()

    @patch(
        "esdc.config_wizard.Config.get_referenced_provider",
        return_value=dict(_FAKE_PROVIDERS["engine"]),
    )
    @patch("esdc.config_wizard.Config.set_config_values")
    def test_embedding_openai_compatible_pair(self, mock_values, mock_ref):
        _save_task_selection("embedding", PairSelection("engine", "embed-1"))
        assert mock_values.call_args.args[0] == {
            "embedding_backend": "openai",
            "embedding_provider": "engine",
            "embedding_model": "embed-1",
        }

    @patch(
        "esdc.config_wizard.Config.get_referenced_provider",
        return_value=dict(_FAKE_PROVIDERS["gpu"]),
    )
    @patch("esdc.config_wizard.Config.set_config_values")
    def test_embedding_ollama_pair_saves_the_selected_model(
        self, mock_values, mock_ref
    ):
        """D7: the model the row shows is the model written."""
        _save_task_selection("embedding", PairSelection("gpu", "embed-1"))
        assert mock_values.call_args.args[0] == {
            "embedding_backend": "ollama",
            "embedding_provider": "gpu",
            "embedding_model": "embed-1",
        }

    @patch("esdc.config_wizard.Config.set_config_values")
    def test_embedding_needs_configuration_selection_is_never_saved(self, mock_values):
        """A retired stored value is a status: it is refused and writes nothing."""
        with pytest.raises(ValueError):
            _save_task_selection(
                "embedding",
                NeedsConfigurationSelection(_RETIRED_EMBEDDING_REASON),
            )
        mock_values.assert_not_called()

    @patch("esdc.config_wizard.Config.set_config_values")
    def test_embedding_ollama_direct_saves_the_selected_model(self, mock_values):
        _save_task_selection(
            "embedding",
            DirectSelection("ollama", "http://direct-box:1234", "embed-1"),
        )
        assert mock_values.call_args.args[0] == {
            "embedding_backend": "ollama",
            "embedding_provider": "",
            "embedding_host": "http://direct-box:1234",
            "embedding_model": "embed-1",
        }

    @patch("esdc.config_wizard.Config.set_config_values")
    def test_embedding_needs_configuration_selection_is_not_a_saveable_row(
        self, mock_values
    ):
        assert NeedsConfigurationSelection("retired") not in _SAVEABLE_SELECTIONS[
            "embedding"
        ]
        assert LocalSelection("model") not in _SAVEABLE_SELECTIONS["embedding"]
        assert LocalSelection("model") not in _SAVEABLE_SELECTIONS["rerank"]

    @patch("esdc.config_wizard.Config.set_config_values")
    def test_embedding_direct_endpoint_preserves_the_key(self, mock_values):
        _save_task_selection(
            "embedding", DirectSelection("openai", "http://direct-box:1234", "embed-1")
        )
        assert mock_values.call_args.args[0] == {
            "embedding_backend": "openai",
            "embedding_provider": "",
            "embedding_host": "http://direct-box:1234",
            "embedding_model": "embed-1",
        }

    @patch(
        "esdc.config_wizard.Config.get_referenced_provider",
        return_value=dict(_FAKE_PROVIDERS["engine"]),
    )
    @patch("esdc.config_wizard.Config.set_config_values")
    def test_metadata_pair_and_linked_and_default_and_off(self, mock_values, mock_ref):
        _save_task_selection("metadata", PairSelection("engine", "wire-model"))
        assert mock_values.call_args.args[0] == {
            "corpus.metadata_provider": "engine",
            "corpus.metadata_model": "wire-model",
        }

        _save_task_selection("metadata", LinkedProviderSelection("engine"))
        assert mock_values.call_args.args[0] == {
            "corpus.metadata_provider": "",
            "corpus.metadata_model": "provider:engine",
        }

        _save_task_selection("metadata", ChatDefaultSelection())
        assert mock_values.call_args.args[0] == {
            "corpus.metadata_provider": "",
            "corpus.metadata_model": "main",
        }

        _save_task_selection("metadata", OffSelection())
        assert mock_values.call_args.args[0] == {
            "corpus.metadata_provider": "",
            "corpus.metadata_model": "",
        }

        _save_task_selection("cleanup", OffSelection())
        assert mock_values.call_args.args[0] == {
            "corpus.cleanup_provider": "",
            "corpus.cleanup_model": "",
        }

        _save_task_selection("metadata", LocalSelection("qwen3:8b"))
        assert mock_values.call_args.args[0] == {
            "corpus.metadata_provider": "",
            "corpus.metadata_model": "qwen3:8b",
        }

    @patch(
        "esdc.config_wizard.discover_models",
        return_value=_catalog("vision-x", vision=True),
    )
    @patch(
        "esdc.config_wizard.Config.get_providers",
        return_value=dict(_FAKE_PROVIDERS),
    )
    @patch(
        "esdc.config_wizard.Config.get_all_config_flat",
        return_value={"corpus.ollama_host": ""},
    )
    @patch(
        "esdc.config_wizard.Config.get_referenced_provider",
        return_value=dict(_FAKE_PROVIDERS["engine"]),
    )
    @patch("esdc.config_wizard.Config.set_config_values")
    def test_ocr_pair_and_local(
        self, mock_values, mock_ref, mock_flat, mock_providers, mock_discover
    ):
        _save_task_selection("ocr", PairSelection("engine", "vision-x"))
        assert mock_values.call_args.args[0] == {
            "corpus.ocr_provider": "engine",
            "corpus.ocr_model": "vision-x",
        }

        _save_task_selection("ocr", LocalSelection("vision-x"))
        assert mock_values.call_args.args[0] == {
            "corpus.ocr_provider": "",
            "corpus.ocr_model": "vision-x",
        }

    @patch(
        "esdc.config_wizard.Config.get_referenced_provider",
        return_value=dict(_FAKE_PROVIDERS["engine"]),
    )
    @patch("esdc.config_wizard.Config.set_config_values")
    def test_rerank_pair_excludes_the_enable_switch(self, mock_values, mock_ref):
        _save_task_selection("rerank", PairSelection("engine", "rerank-model"))
        assert mock_values.call_args.args[0] == {
            "corpus.rerank_backend": "provider",
            "corpus.rerank_provider": "engine",
            "corpus.rerank_model": "rerank-model",
        }

    @patch(
        "esdc.config_wizard.Config.get_referenced_provider",
        return_value=dict(_FAKE_PROVIDERS["engine"]),
    )
    @patch("esdc.config_wizard.Config.set_config_values")
    def test_rerank_pair_preserves_a_legacy_gguf_value(self, mock_values, mock_ref):
        """A stored GGUF id is saved as it was selected; nothing is rewritten."""
        _save_task_selection("rerank", PairSelection("engine", "gguf-x"))
        assert mock_values.call_args.args[0] == {
            "corpus.rerank_backend": "provider",
            "corpus.rerank_provider": "engine",
            "corpus.rerank_model": "gguf-x",
        }

    @patch("esdc.config_wizard.Config.set_config_values")
    def test_rerank_needs_configuration_selection_is_never_saved(self, mock_values):
        with pytest.raises(ValueError):
            _save_task_selection(
                "rerank", NeedsConfigurationSelection(_RETIRED_RERANK_REASON)
            )
        mock_values.assert_not_called()

    def test_a_selection_from_another_task_is_rejected(self):
        with pytest.raises(ValueError):
            _save_task_selection("chat", LocalSelection("m1"))
        with pytest.raises(ValueError):
            _save_task_selection("embedding", LinkedProviderSelection("engine"))
        with pytest.raises(ValueError):
            _save_task_selection("rerank", OffSelection())


class TestTaskSummaryLine:
    """The summary shows the provider/model string, status and env override."""

    @patch("esdc.config_wizard.discover_models", return_value=_catalog("m1"))
    @patch("esdc.config_wizard.Config.get_all_config_flat", return_value={})
    @patch(
        "esdc.config_wizard.Config.get_providers", return_value=dict(_FAKE_PROVIDERS)
    )
    def test_summary_uses_the_same_label_as_the_rows(
        self, mock_providers, mock_flat, mock_discover
    ):
        from esdc.config_wizard import _task_summary

        assert _task_summary("chat", PairSelection("engine", "chat-a")) == (
            "  Chat: engine / chat-a"
        )
        assert "follows the chat model" in _task_summary(
            "metadata", LinkedProviderSelection("engine")
        )
        assert _task_summary("metadata", OffSelection()) == (
            "  Metadata extraction: Off / image fallback"
        )
        assert _task_summary("cleanup", OffSelection()) == "  Text cleanup: Off"

    @patch("esdc.config_wizard.discover_models", return_value=_catalog("m1"))
    @patch("esdc.config_wizard.Config.get_all_config_flat", return_value={})
    @patch(
        "esdc.config_wizard.Config.get_providers", return_value=dict(_FAKE_PROVIDERS)
    )
    def test_environment_override_is_shown(
        self, mock_providers, mock_flat, mock_discover
    ):
        import os

        from esdc.config_wizard import _task_summary

        os.environ["ESDC_EMBEDDING_MODEL"] = "env-model"
        try:
            summary = _task_summary(
                "embedding", DirectSelection("ollama", "", OllamaEmbedder.OLLAMA_TAG)
            )
            assert "environment override: ESDC_EMBEDDING_MODEL" in summary
        finally:
            del os.environ["ESDC_EMBEDDING_MODEL"]

    @patch("esdc.config_wizard.discover_models", return_value=_catalog("m1"))
    @patch("esdc.config_wizard.Config.get_all_config_flat", return_value={})
    @patch(
        "esdc.config_wizard.Config.get_providers", return_value=dict(_FAKE_PROVIDERS)
    )
    def test_summary_carries_the_retired_reason(
        self, mock_providers, mock_flat, mock_discover
    ):
        """The label carries the reason: the summary needs no extra status word."""
        from esdc.config_wizard import _task_summary

        summary = _task_summary(
            "embedding", NeedsConfigurationSelection(_RETIRED_EMBEDDING_REASON)
        )
        assert summary == (
            "  Embeddings: Needs configuring (retired in-process embedding backend "
            "— choose an HTTP service)"
        )


class TestConnectionMenuKeepsChatControlsOut:
    """Connections has no default-provider or failover rows after unification."""

    def test_provider_fields_are_not_flat_section_keys(self):
        all_keys = {
            k
            for keys in {
                **TASK_SECTIONS,
                **APPLICATION_SECTIONS,
                **ADVANCED_SECTIONS,
            }.values()
            for k in keys
        }
        assert "default_provider" not in all_keys
        assert "provider_order" not in all_keys

    @patch("esdc.config_wizard.Config.init_config")
    @patch("esdc.config_wizard.questionary.select")
    @patch("esdc.config_wizard.rich_print")
    def test_home_has_no_provider_default_or_order_rows(
        self, mock_print, mock_select, mock_init
    ):
        mock_select.return_value.unsafe_ask.side_effect = ["Exit"]
        run_wizard()
        labels = [str(c.title) for c in mock_select.call_args.kwargs["choices"]]
        assert "Set default provider" not in labels
        assert "Set provider failover order" not in labels

    @patch("esdc.config_wizard.questionary.select")
    def test_connection_menu_has_no_default_or_order_rows(self, mock_select):
        from esdc.config_wizard import _connections_menu

        mock_select.return_value.unsafe_ask.side_effect = ["__back__"]
        _connections_menu()
        labels = [str(c.title) for c in mock_select.call_args.kwargs["choices"]]
        assert "Set default provider" not in labels
        assert "Set provider failover order" not in labels
        assert "Chat fallback" not in " ".join(labels)


class TestPlainQuestionaryMessages:
    """No Rich markup reaches a questionary message."""

    _PROMPTS = {"select", "confirm", "text", "password"}

    @staticmethod
    def _literal(node: ast.AST) -> str:
        """The plain-text parts of a message argument, including f-string parts."""
        return "".join(
            part.value
            for part in ast.walk(node)
            if isinstance(part, ast.Constant) and isinstance(part.value, str)
        )

    def _messages(self) -> list[tuple[int, str]]:
        """Every prompt message the wizard can show: prompts, choices, Back selects."""
        source = Path(__file__).resolve().parent.parent / "esdc" / "config_wizard.py"
        tree = ast.parse(source.read_text())
        found: list[tuple[int, str]] = []

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue

            is_prompt = (
                isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "questionary"
                and node.func.attr in self._PROMPTS | {"Choice"}
            )
            is_back_select = isinstance(node.func, ast.Name) and node.func.id in (
                "_select_with_back",
            )
            if not is_prompt and not is_back_select:
                continue

            # The message is the first positional argument, or `message=`/`title=`.
            argument = node.args[0] if node.args else None
            if argument is None:
                for keyword in node.keywords:
                    if keyword.arg in {"message", "title"}:
                        argument = keyword.value
                        break
            if argument is None:
                continue
            found.append((node.lineno, self._literal(argument)))

        return found

    def test_no_markup_in_questionary_messages(self):
        offenders = [
            (lineno, message)
            for lineno, message in self._messages()
            if "[/" in message or "[bold]" in message or "[#" in message
        ]
        assert offenders == []

    def test_provider_prompt_messages_are_covered(self):
        """The scan reaches the provider screens, not only the task screens."""
        messages = {message for _, message in self._messages()}
        for expected in (
            "Select provider to edit:",
            "Select field to edit:",
            "API Key (blank keeps current):",
            "Clear API key",
            "Done editing",
        ):
            assert expected in messages


class TestMenuKeyCoverage:
    """Every editable key has exactly one menu location and a description."""

    def test_every_key_appears_exactly_once(self):
        sections = {
            **TASK_SECTIONS,
            **APPLICATION_SECTIONS,
            **ADVANCED_SECTIONS,
        }
        keys = [key for keys in sections.values() for key in keys]
        # `corpus.n_gpu_layers` is no longer offered: a legacy value still parses and
        # is preserved, but it is not part of the menu.
        assert len(keys) == 43
        assert len(set(keys)) == 43

    def test_the_gpu_layers_key_is_not_offered_but_still_parses(self):
        """Config keeps a stored value; only the menu no longer offers the key."""
        sections = {
            **TASK_SECTIONS,
            **APPLICATION_SECTIONS,
            **ADVANCED_SECTIONS,
        }
        for keys in sections.values():
            assert "corpus.n_gpu_layers" not in keys

        assert "corpus.n_gpu_layers" in Config.INT_KEYS
        assert "n_gpu_layers" not in Config.CORPUS_DEFAULTS
        assert "corpus.n_gpu_layers" in KEY_DESCRIPTIONS
        # Reading it is a pure parse: the stored value comes back unchanged.
        assert Config._coerce_value("corpus.n_gpu_layers", "6") == 6

    def test_every_key_has_a_description(self):
        for keys in {
            **TASK_SECTIONS,
            **APPLICATION_SECTIONS,
            **ADVANCED_SECTIONS,
        }.values():
            for key in keys:
                assert key in KEY_DESCRIPTIONS, key

    def test_the_legacy_aliases_still_cover_the_same_keys(self):
        from esdc.configs import MODEL_SECTIONS, SETTINGS_SECTIONS

        assert MODEL_SECTIONS is TASK_SECTIONS
        merged = {**APPLICATION_SECTIONS, **ADVANCED_SECTIONS}
        assert set(SETTINGS_SECTIONS["Corpus processing"]) == set(
            merged["Corpus processing"]
        )


class TestOcrVisionFilter:
    """OCR keeps known-vision and unknown entries; drops explicit non-vision ones."""

    _ENTRY = {"engine": _FAKE_PROVIDERS["engine"]}

    @staticmethod
    def _labels(mock_select) -> list[str]:
        return [str(c.title) for c in mock_select.call_args.kwargs["choices"]]

    @patch("esdc.config_wizard.Config.get_providers")
    @patch("esdc.config_wizard.Config.get_all_config_flat", return_value={})
    @patch("esdc.config_wizard.questionary.select")
    def test_known_vision_and_unknown_stay_selectable(
        self, mock_select, mock_flat, mock_providers
    ):
        mock_providers.return_value = dict(self._ENTRY)
        entries = [
            ModelEntry("vision-yes", frozenset({"chat"}), True),
            ModelEntry("unknown-role-vision", frozenset(), None),
            ModelEntry("embedding-only", frozenset({"embedding"}), True),
        ]
        with patch(
            "esdc.config_wizard.discover_models",
            return_value=ModelCatalog(entries=tuple(entries)),
        ):
            mock_select.return_value.unsafe_ask.side_effect = [None]
            _choose_provider_model("ocr", None)

        labels = self._labels(mock_select)
        assert "engine / vision-yes" in labels
        assert "engine / unknown-role-vision" in labels
        # A known non-chat entry is still filtered out for OCR.
        assert "engine / embedding-only" not in labels

    @patch("esdc.config_wizard.Config.get_providers")
    @patch("esdc.config_wizard.Config.get_all_config_flat", return_value={})
    @patch("esdc.config_wizard.questionary.select")
    def test_explicit_non_vision_entry_is_filtered_out(
        self, mock_select, mock_flat, mock_providers
    ):
        mock_providers.return_value = dict(self._ENTRY)
        entries = [
            ModelEntry("vision-yes", frozenset({"chat"}), True),
            ModelEntry("text-only", frozenset({"chat"}), False),
            ModelEntry("unknown-vision", frozenset({"chat"}), None),
        ]
        with patch(
            "esdc.config_wizard.discover_models",
            return_value=ModelCatalog(entries=tuple(entries)),
        ):
            mock_select.return_value.unsafe_ask.side_effect = [None]
            _choose_provider_model("ocr", None)

        labels = self._labels(mock_select)
        assert "engine / text-only" not in labels
        assert "engine / vision-yes" in labels
        assert "engine / unknown-vision" in labels

    @patch("esdc.config_wizard.Config.get_providers")
    @patch("esdc.config_wizard.Config.get_all_config_flat", return_value={})
    @patch("esdc.config_wizard.questionary.select")
    def test_without_reliable_metadata_every_entry_stays(
        self, mock_select, mock_flat, mock_providers
    ):
        """No entry reports vision, so nothing is filtered on unknown metadata."""
        mock_providers.return_value = dict(self._ENTRY)
        entries = [
            ModelEntry("m-a", frozenset({"chat"}), None),
            ModelEntry("m-b", frozenset({"chat"}), None),
        ]
        with patch(
            "esdc.config_wizard.discover_models",
            return_value=ModelCatalog(entries=tuple(entries)),
        ):
            mock_select.return_value.unsafe_ask.side_effect = [None]
            _choose_provider_model("ocr", None)

        labels = self._labels(mock_select)
        assert "engine / m-a" in labels and "engine / m-b" in labels

    @patch("esdc.config_wizard.Config.get_providers")
    @patch("esdc.config_wizard.Config.get_all_config_flat", return_value={})
    @patch("esdc.config_wizard.questionary.select")
    def test_the_filter_applies_to_ocr_only(
        self, mock_select, mock_flat, mock_providers
    ):
        """The same non-vision entry is still selectable for metadata extraction."""
        mock_providers.return_value = dict(self._ENTRY)
        entries = [ModelEntry("text-only", frozenset({"chat"}), False)]
        with patch(
            "esdc.config_wizard.discover_models",
            return_value=ModelCatalog(entries=tuple(entries)),
        ):
            mock_select.return_value.unsafe_ask.side_effect = [None]
            _choose_provider_model("metadata", None)
            assert "engine / text-only" in self._labels(mock_select)

            mock_select.return_value.unsafe_ask.side_effect = [None]
            _choose_provider_model("ocr", None)
            assert "engine / text-only" not in self._labels(mock_select)


class TestTestActionNeverSaves:
    """Test connection resolves the draft only; it never writes or enables."""

    @patch("esdc.config_wizard._test_provider")
    @patch("esdc.config_wizard.Config.set_config_values")
    @patch("esdc.config_wizard.Config.set_chat_selection")
    @patch("esdc.config_wizard.Config.get_referenced_provider")
    @patch("esdc.config_wizard.rich_print")
    @patch("esdc.config_wizard.questionary.select")
    def test_test_connection_writes_nothing(
        self,
        mock_select,
        mock_print,
        mock_ref,
        mock_chat,
        mock_values,
        mock_test_provider,
    ):
        saved = {"provider_type": "openai_compatible", "model": "saved-chat-model"}
        mock_ref.return_value = dict(saved)
        mock_select.return_value.unsafe_ask.side_effect = ["test", "cancel"]

        _run_task_action("chat", PairSelection("engine", "draft-model"))

        mock_chat.assert_not_called()
        mock_values.assert_not_called()
        # The draft sent to the test carries the draft model; the saved entry does not.
        assert mock_test_provider.call_args.args[1]["model"] == "draft-model"
        assert saved["model"] == "saved-chat-model"

    @patch("esdc.config_wizard._test_provider")
    @patch("esdc.config_wizard.OpenAIEmbedder")
    @patch("esdc.config_wizard.Config.set_config_values")
    @patch("esdc.config_wizard.Config.get_all_config_flat")
    @patch("esdc.config_wizard.rich_print")
    @patch("esdc.config_wizard.questionary.select")
    def test_direct_endpoint_uses_the_saved_key_without_saving(
        self,
        mock_select,
        mock_print,
        mock_flat,
        mock_values,
        mock_embedder,
        mock_test_provider,
    ):
        mock_flat.return_value = {"embedding_api_key": "sk-saved"}
        mock_select.return_value.unsafe_ask.side_effect = ["test", "cancel"]
        mock_embedder.return_value.generate_embedding.return_value = [0.1]

        _run_task_action(
            "embedding", DirectSelection("openai", "http://direct-box:1234", "embed-1")
        )

        mock_values.assert_not_called()
        # The embedder is built from the draft the action menu resolved: the saved
        # key, the direct host, the picked model. No chat provider test is run.
        assert mock_embedder.call_args.kwargs == {
            "host": "http://direct-box:1234",
            "model": "embed-1",
            "api_key": "sk-saved",
        }
        mock_test_provider.assert_not_called()

    @patch("esdc.config_wizard.rich_print")
    @patch("esdc.config_wizard.Config.set_config_values")
    @patch("esdc.config_wizard.questionary.select")
    def test_retired_and_off_rows_report_nothing_to_test(
        self, mock_select, mock_values, mock_print
    ):
        # Two action menus: one for the retired rerank status, one for the off row.
        mock_select.return_value.unsafe_ask.side_effect = [
            "test",
            "cancel",
            "test",
            "cancel",
        ]

        # A retired stored value has no implementation any more: nothing is loaded,
        # no request is made, and the stored value is left as it is.
        _run_task_action(
            "rerank", NeedsConfigurationSelection(_RETIRED_RERANK_REASON)
        )
        # An off row has no source to reach at all.
        _run_task_action("metadata", OffSelection())

        printed = " ".join(str(call) for call in mock_print.call_args_list)
        assert (
            "Nothing to test: retired local rerank backend "
            "— choose a rerank service." in printed
        )
        assert "Nothing to test" in printed
        mock_values.assert_not_called()

    @patch("esdc.config_wizard.create_ocr_client")
    @patch("esdc.config_wizard.Config.set_config_values")
    @patch("esdc.config_wizard.Config.get_corpus_config", return_value={})
    @patch("esdc.config_wizard.rich_print")
    @patch("esdc.config_wizard.questionary.select")
    def test_ocr_test_uses_the_draft_and_sends_its_own_image(
        self, mock_select, mock_print, mock_corpus, mock_values, mock_client
    ):
        client = MagicMock()
        client.health_check.return_value = True
        mock_client.return_value = client
        mock_select.return_value.unsafe_ask.side_effect = ["test", "cancel"]

        _run_task_action("ocr", PairSelection("engine", "vision-x"))

        mock_values.assert_not_called()
        draft = mock_client.call_args.args[0]
        assert draft["ocr_provider"] == "engine"
        assert draft["ocr_model"] == "vision-x"
        client.health_check.assert_called_once()
        # Availability alone is not the test: the draft also has to read an image.
        client.query_image.assert_called_once()
        png, prompt = client.query_image.call_args.args
        assert png.startswith(b"\x89PNG")
        assert prompt == "Return the text in this image exactly."


class TestRoleProviderTypes:
    """Each task offers only the provider types it can hold a reference for."""

    _PROVIDERS = {
        "engine": {
            "provider_type": "openai_compatible",
            "base_url": "http://engine:4000/v1",
            "model": "chat-a",
        },
        "public": {"provider_type": "openai", "model": "gpt-5"},
        "gpu": {
            "provider_type": "ollama",
            "base_url": "http://gpu-box:11434",
            "model": "ollama-a",
        },
        "desk": {"provider_type": "anthropic", "model": "claude-x"},
    }

    @staticmethod
    def _labels(mock_select) -> list[str]:
        return [str(c.title) for c in mock_select.call_args.kwargs["choices"]]

    def _pick(self, mock_select, role: TaskRole) -> list[str]:
        mock_select.return_value.unsafe_ask.side_effect = [None]
        _choose_provider_model(role, None)
        return self._labels(mock_select)

    @patch("esdc.config_wizard.discover_models", return_value=_catalog("m1"))
    @patch("esdc.config_wizard.Config.get_all_config_flat", return_value={})
    @patch("esdc.config_wizard.Config.get_providers", return_value=dict(_PROVIDERS))
    @patch("esdc.config_wizard.questionary.select")
    def test_embedding_accepts_openai_compatible_and_ollama_only(
        self, mock_select, mock_providers, mock_flat, mock_discover
    ):
        labels = self._pick(mock_select, "embedding")
        assert "engine / m1" in labels
        assert f"gpu / {OllamaEmbedder.OLLAMA_TAG}" in labels
        assert not any(label.startswith("public / ") for label in labels)
        assert not any(label.startswith("desk / ") for label in labels)

    @patch("esdc.config_wizard.discover_models", return_value=_catalog("m1"))
    @patch("esdc.config_wizard.Config.get_all_config_flat", return_value={})
    @patch("esdc.config_wizard.Config.get_providers", return_value=dict(_PROVIDERS))
    @patch("esdc.config_wizard.questionary.select")
    def test_rerank_accepts_openai_compatible_only(
        self, mock_select, mock_providers, mock_flat, mock_discover
    ):
        labels = self._pick(mock_select, "rerank")
        assert "engine / m1" in labels
        assert not any(label.startswith("gpu / ") for label in labels)
        assert not any(label.startswith("public / ") for label in labels)

    @patch("esdc.config_wizard.discover_models", return_value=_catalog("m1"))
    @patch("esdc.config_wizard.Config.get_all_config_flat", return_value={})
    @patch("esdc.config_wizard.Config.get_providers", return_value=dict(_PROVIDERS))
    @patch("esdc.config_wizard.questionary.select")
    def test_ocr_accepts_openai_compatible_and_ollama_only(
        self, mock_select, mock_providers, mock_flat, mock_discover
    ):
        labels = self._pick(mock_select, "ocr")
        assert "engine / m1" in labels
        assert any(label.startswith("gpu / ") for label in labels)
        assert not any(label.startswith("public / ") for label in labels)

    @patch("esdc.config_wizard.discover_models", return_value=_catalog("m1"))
    @patch("esdc.config_wizard.Config.get_all_config_flat", return_value={})
    @patch("esdc.config_wizard.Config.get_providers", return_value=dict(_PROVIDERS))
    @patch("esdc.config_wizard.questionary.select")
    def test_chat_and_text_tasks_accept_every_saved_provider(
        self, mock_select, mock_providers, mock_flat, mock_discover
    ):
        roles: tuple[TaskRole, ...] = ("chat", "metadata", "cleanup")
        for role in roles:
            labels = self._pick(mock_select, role)
            for name in self._PROVIDERS:
                assert any(label.startswith(f"{name} / ") for label in labels), (
                    role,
                    name,
                )

    @patch("esdc.config_wizard.discover_models", return_value=_catalog("m1"))
    @patch(
        "esdc.config_wizard.Config.get_all_config_flat",
        return_value={
            "corpus.rerank_backend": "provider",
            "corpus.rerank_provider": "public",
        },
    )
    @patch("esdc.config_wizard.Config.get_providers", return_value=dict(_PROVIDERS))
    @patch("esdc.config_wizard.questionary.select")
    def test_a_saved_pair_of_an_invalid_type_is_not_shown(
        self, mock_select, mock_providers, mock_flat, mock_discover
    ):
        """The unverified saved pair follows the same type rule as the rows."""
        labels = self._pick(mock_select, "rerank")
        assert not any("public" in label for label in labels)


class TestLinkedRowsNeedASavedModel:
    """A linked row follows the provider's chat model, so it needs one."""

    _PROVIDERS = {
        "withmodel": {"provider_type": "openai_compatible", "model": "chat-a"},
        "nomodel": {"provider_type": "openai_compatible", "model": ""},
    }

    @staticmethod
    def _labels(mock_select) -> list[str]:
        return [str(c.title) for c in mock_select.call_args.kwargs["choices"]]

    @patch("esdc.config_wizard.discover_models", return_value=_catalog())
    @patch("esdc.config_wizard.Config.get_all_config_flat", return_value={})
    @patch("esdc.config_wizard.Config.get_providers", return_value=dict(_PROVIDERS))
    @patch("esdc.config_wizard.questionary.select")
    def test_metadata_linked_row_skips_a_provider_without_a_model(
        self, mock_select, mock_providers, mock_flat, mock_discover
    ):
        mock_select.return_value.unsafe_ask.side_effect = [None]
        _choose_provider_model("metadata", None)
        labels = self._labels(mock_select)
        assert "withmodel / (chat model)" in labels
        assert "nomodel / (chat model)" not in labels

    @patch("esdc.config_wizard.discover_models", return_value=_catalog())
    @patch("esdc.config_wizard.Config.get_all_config_flat", return_value={})
    @patch("esdc.config_wizard.Config.get_providers", return_value=dict(_PROVIDERS))
    @patch("esdc.config_wizard.questionary.select")
    def test_cleanup_linked_row_skips_a_provider_without_a_model(
        self, mock_select, mock_providers, mock_flat, mock_discover
    ):
        mock_select.return_value.unsafe_ask.side_effect = [None]
        _choose_provider_model("cleanup", None)
        labels = self._labels(mock_select)
        assert "withmodel / (chat model)" in labels
        assert "nomodel / (chat model)" not in labels


class TestPinnedEmbeddingModel:
    """An Ollama embedding reference shows the configured model (D7).

    The legacy tag is only the fallback when nothing is configured.
    """

    _FLAT = {
        "embedding_backend": "ollama",
        "embedding_provider": "gpu",
        "embedding_model": "inactive-model",
    }
    _EMPTY_FLAT = {
        "embedding_backend": "ollama",
        "embedding_provider": "gpu",
        "embedding_model": "",
    }
    _PROVIDERS = {"gpu": _FAKE_PROVIDERS["gpu"]}

    @staticmethod
    def _labels(mock_select) -> list[str]:
        return [str(c.title) for c in mock_select.call_args.kwargs["choices"]]

    @patch("esdc.config_wizard.discover_models", return_value=_catalog("m1"))
    @patch("esdc.config_wizard.Config.get_all_config_flat", return_value=dict(_FLAT))
    @patch("esdc.config_wizard.Config.get_providers", return_value=dict(_PROVIDERS))
    @patch("esdc.config_wizard.questionary.select")
    def test_saved_selection_is_the_configured_model(
        self, mock_select, mock_providers, mock_flat, mock_discover
    ):
        assert _current_task_selection("embedding") == PairSelection(
            "gpu", "inactive-model"
        )

        mock_select.return_value.unsafe_ask.side_effect = [None]
        _choose_provider_model("embedding", None)
        labels = self._labels(mock_select)
        assert "gpu / inactive-model" in labels

    @patch("esdc.config_wizard.discover_models", return_value=_catalog("m1"))
    @patch(
        "esdc.config_wizard.Config.get_all_config_flat", return_value=dict(_EMPTY_FLAT)
    )
    @patch("esdc.config_wizard.Config.get_providers", return_value=dict(_PROVIDERS))
    @patch("esdc.config_wizard.questionary.select")
    def test_empty_model_falls_back_to_the_legacy_tag(
        self, mock_select, mock_providers, mock_flat, mock_discover
    ):
        assert _current_task_selection("embedding") == PairSelection(
            "gpu", OllamaEmbedder.OLLAMA_TAG
        )

        mock_select.return_value.unsafe_ask.side_effect = [None]
        _choose_provider_model("embedding", None)
        assert f"gpu / {OllamaEmbedder.OLLAMA_TAG}" in self._labels(mock_select)

    @patch("esdc.config_wizard.discover_models", return_value=_catalog("m1"))
    @patch("esdc.config_wizard.Config.get_all_config_flat", return_value=dict(_FLAT))
    @patch("esdc.config_wizard.Config.get_providers", return_value=dict(_PROVIDERS))
    def test_summary_shows_the_configured_model(
        self, mock_providers, mock_flat, mock_discover
    ):
        assert _task_summary("embedding", _current_task_selection("embedding")) == (
            "  Embeddings: gpu / inactive-model"
        )

    @patch(
        "esdc.config_wizard.Config.get_referenced_provider",
        return_value=dict(_FAKE_PROVIDERS["gpu"]),
    )
    @patch("esdc.config_wizard.Config.set_config_values")
    def test_save_writes_backend_reference_and_model(self, mock_values, mock_ref):
        _save_task_selection(
            "embedding", PairSelection("gpu", OllamaEmbedder.OLLAMA_TAG)
        )
        assert mock_values.call_args.args[0] == {
            "embedding_backend": "ollama",
            "embedding_provider": "gpu",
            "embedding_model": OllamaEmbedder.OLLAMA_TAG,
        }


class TestIncompleteRerankSource:
    """A provider source with no reference is incomplete, not a local model."""

    _FLAT = {
        "corpus.rerank": True,
        "corpus.rerank_backend": "provider",
        "corpus.rerank_provider": "",
        "corpus.rerank_model": "rerank-model",
    }

    @patch(
        "esdc.config_wizard.Config.get_providers", return_value=dict(_FAKE_PROVIDERS)
    )
    @patch("esdc.config_wizard.Config.get_all_config_flat", return_value=dict(_FLAT))
    def test_selection_is_not_read_as_local(self, mock_providers, mock_flat):
        assert _current_task_selection("rerank") is None

    @patch("esdc.config_wizard.rich_print")
    @patch("esdc.config_wizard.Config.set_config_value")
    @patch("esdc.config_wizard.Config.set_config_values")
    @patch("esdc.config_wizard.Config.get_all_config_flat", return_value=dict(_FLAT))
    @patch(
        "esdc.config_wizard.Config.get_providers", return_value=dict(_FAKE_PROVIDERS)
    )
    @patch("esdc.config_wizard.questionary.confirm")
    @patch("esdc.config_wizard.questionary.select")
    def test_enable_refuses_and_writes_nothing(
        self,
        mock_select,
        mock_confirm,
        mock_providers,
        mock_flat,
        mock_values,
        mock_single,
        mock_print,
    ):
        mock_select.return_value.unsafe_ask.side_effect = ["enabled", "__back__"]
        mock_confirm.return_value.unsafe_ask.return_value = True

        _edit_task("rerank")

        mock_single.assert_not_called()
        mock_values.assert_not_called()


class TestCustomPairRoleFilter:
    """Custom offers only provider types this task can save, and needs a model id."""

    _PROVIDERS = {
        "engine": {"provider_type": "openai_compatible", "model": "chat-a"},
        "public": {"provider_type": "openai", "model": "gpt-5"},
        "gpu": {"provider_type": "ollama", "model": "ollama-a"},
    }

    @staticmethod
    def _custom_names(mock_select, role: TaskRole) -> list[str]:
        # The provider list is the first question; cancelling there gets the
        # choices back without needing a model id.
        mock_select.return_value.unsafe_ask.return_value = None
        assert _custom_pair_flow(role, None) is None
        return [str(c.value) for c in mock_select.call_args.kwargs["choices"]]

    @patch("esdc.config_wizard.Config.get_providers", return_value=dict(_PROVIDERS))
    @patch("esdc.config_wizard.questionary.select")
    def test_embedding_custom_allows_ollama_and_openai_compatible(
        self, mock_select, mock_providers
    ):
        assert self._custom_names(mock_select, "embedding") == ["engine", "gpu"]

    @patch("esdc.config_wizard.Config.get_providers", return_value=dict(_PROVIDERS))
    @patch("esdc.config_wizard.questionary.select")
    def test_rerank_custom_is_openai_compatible_only(self, mock_select, mock_providers):
        assert self._custom_names(mock_select, "rerank") == ["engine"]

    @patch("esdc.config_wizard.Config.get_providers", return_value=dict(_PROVIDERS))
    @patch("esdc.config_wizard.questionary.select")
    def test_ocr_custom_allows_openai_compatible_and_ollama(
        self, mock_select, mock_providers
    ):
        assert self._custom_names(mock_select, "ocr") == ["engine", "gpu"]

    @patch("esdc.config_wizard.Config.get_providers", return_value=dict(_PROVIDERS))
    @patch("esdc.config_wizard.questionary.select")
    def test_chat_custom_offers_every_saved_provider(self, mock_select, mock_providers):
        assert self._custom_names(mock_select, "chat") == ["engine", "public", "gpu"]

    @patch("esdc.config_wizard.Config.get_providers", return_value=dict(_PROVIDERS))
    @patch("esdc.config_wizard.rich_print")
    @patch("esdc.config_wizard.questionary.text")
    @patch("esdc.config_wizard.questionary.select")
    def test_an_empty_model_is_rejected(
        self, mock_select, mock_text, mock_print, mock_providers
    ):
        mock_select.return_value.unsafe_ask.return_value = "engine"
        mock_text.return_value.unsafe_ask.return_value = "   "

        assert _custom_pair_flow("chat", None) is None

    @patch("esdc.config_wizard.discover_models", return_value=_catalog("m1"))
    @patch("esdc.config_wizard.Config.get_providers", return_value=dict(_PROVIDERS))
    @patch("esdc.config_wizard.Config.get_all_config_flat", return_value={})
    @patch("esdc.config_wizard.Config.set_config_values")
    @patch("esdc.config_wizard.rich_print")
    @patch("esdc.config_wizard.questionary.text")
    @patch("esdc.config_wizard.questionary.select")
    def test_empty_model_returns_to_the_picker_and_writes_nothing(
        self,
        mock_select,
        mock_text,
        mock_print,
        mock_values,
        mock_flat,
        mock_providers,
        mock_discover,
    ):
        """An empty custom model re-opens the picker; it is not treated as Back."""
        mock_select.return_value.unsafe_ask.side_effect = [
            "__custom__",
            "engine",
            PairSelection("engine", "m1"),
        ]
        mock_text.return_value.unsafe_ask.return_value = ""

        assert _choose_provider_model("chat", None) == PairSelection("engine", "m1")
        assert mock_select.call_count == 3
        mock_values.assert_not_called()

    @patch("esdc.config_wizard.Config.set_chat_selection")
    @patch("esdc.config_wizard.Config.get_referenced_provider")
    @patch("esdc.config_wizard.Config.get_providers", return_value=dict(_PROVIDERS))
    @patch("esdc.config_wizard.questionary.text")
    @patch("esdc.config_wizard.questionary.select")
    def test_a_custom_pair_is_never_one_save_would_reject(
        self, mock_select, mock_text, mock_providers, mock_ref, mock_chat
    ):
        mock_ref.return_value = dict(self._PROVIDERS["engine"])
        mock_select.return_value.unsafe_ask.return_value = "engine"
        mock_text.return_value.unsafe_ask.return_value = "typed-model"
        pair = _custom_pair_flow("chat", None)

        assert isinstance(pair, PairSelection)
        assert pair == PairSelection("engine", "typed-model")
        _save_task_selection("chat", pair)
        mock_chat.assert_called_once_with("engine", "typed-model")


class TestSaveGuardsRejectInvalidPairTypes:
    """Save refuses a pair whose provider type cannot serve the task."""

    _PAIR = PairSelection("public", "m1")

    @patch("esdc.config_wizard.Config.get_referenced_provider")
    @patch("esdc.config_wizard.Config.set_config_values")
    def test_embedding_rejects_an_openai_provider(self, mock_values, mock_ref):
        mock_ref.return_value = {"provider_type": "openai"}
        with pytest.raises(ValueError):
            _save_task_selection("embedding", self._PAIR)
        mock_values.assert_not_called()

    @patch("esdc.config_wizard.Config.get_referenced_provider")
    @patch("esdc.config_wizard.Config.set_config_values")
    def test_rerank_rejects_an_ollama_provider(self, mock_values, mock_ref):
        mock_ref.return_value = {"provider_type": "ollama"}
        with pytest.raises(ValueError):
            _save_task_selection("rerank", PairSelection("gpu", "m1"))
        mock_values.assert_not_called()

    @patch("esdc.config_wizard.Config.get_referenced_provider")
    @patch("esdc.config_wizard.Config.set_config_values")
    def test_ocr_rejects_an_anthropic_provider(self, mock_values, mock_ref):
        mock_ref.return_value = {"provider_type": "anthropic"}
        with pytest.raises(ValueError):
            _save_task_selection("ocr", PairSelection("desk", "m1"))
        mock_values.assert_not_called()

    @patch("esdc.config_wizard.Config.get_referenced_provider")
    @patch("esdc.config_wizard.Config.set_config_values")
    def test_an_unknown_provider_is_rejected(self, mock_values, mock_ref):
        mock_ref.side_effect = ValueError("Unknown provider reference: 'gone'")
        with pytest.raises(ValueError):
            _save_task_selection("rerank", PairSelection("gone", "m1"))
        mock_values.assert_not_called()

    @patch("esdc.config_wizard.Config.get_referenced_provider")
    @patch("esdc.config_wizard.Config.set_config_values")
    def test_a_valid_type_still_saves(self, mock_values, mock_ref):
        mock_ref.return_value = {"provider_type": "openai_compatible"}
        _save_task_selection("rerank", self._PAIR)
        assert mock_values.call_args.args[0] == {
            "corpus.rerank_backend": "provider",
            "corpus.rerank_provider": "public",
            "corpus.rerank_model": "m1",
        }


class TestNoHalfRemovedModelPath:
    """The flat-editor model picker is gone: task screens own those keys."""

    def test_the_dead_path_is_removed(self):
        import esdc.config_wizard as wizard

        assert getattr(wizard, "_corpus_model_choices", None) is None
        assert getattr(wizard, "_CORPUS_MODEL_KEYS", None) is None

    def test_no_task_section_key_is_prompted_through_the_flat_editor(self):
        """A task key has no menu route, so the flat editor never sees one."""
        menu_keys = {
            key
            for keys in {**APPLICATION_SECTIONS, **ADVANCED_SECTIONS}.values()
            for key in keys
        }
        task_keys = {key for keys in TASK_SECTIONS.values() for key in keys}
        assert not (task_keys & menu_keys)
        assert task_keys


class TestEmptyModelIsNeverARow:
    """A saved pair with no model has nothing to select, so it is not a row."""

    # The only saved provider has no model, so every role that reads a saved pair
    # would otherwise show a blank ``provider / `` row.
    _PROVIDERS = {"engine": {"provider_type": "openai_compatible", "model": ""}}
    _FLAT = {
        "default_provider": "engine",
        "corpus.metadata_provider": "engine",
        "corpus.metadata_model": "",
        "corpus.cleanup_provider": "engine",
        "corpus.cleanup_model": "",
        "corpus.ocr_provider": "engine",
        "corpus.ocr_model": "",
    }
    _FIXED_ROWS = ["Custom provider / model…", "Retry discovery", "← Back"]

    def _pick(self, mock_select, role: TaskRole) -> list[str]:
        mock_select.return_value.unsafe_ask.side_effect = [None]
        _choose_provider_model(role, None)
        return _row_labels(mock_select)

    @staticmethod
    def _blank_rows(mock_select) -> list[str]:
        return [
            str(choice.title)
            for choice in mock_select.call_args.kwargs["choices"]
            if isinstance(
                choice.value, (PairSelection, LocalSelection, DirectSelection)
            )
            and not str(choice.value.model_id).strip()
        ]

    @patch("esdc.config_wizard.discover_models", return_value=_catalog())
    @patch("esdc.config_wizard.Config.get_all_config_flat", return_value=dict(_FLAT))
    @patch("esdc.config_wizard.Config.get_providers", return_value=dict(_PROVIDERS))
    @patch("esdc.config_wizard.questionary.select")
    def test_chat_has_no_blank_pair_row(
        self, mock_select, mock_providers, mock_flat, mock_discover
    ):
        assert self._pick(mock_select, "chat") == self._FIXED_ROWS
        assert self._blank_rows(mock_select) == []

    @patch("esdc.config_wizard.discover_models", return_value=_catalog())
    @patch("esdc.config_wizard.Config.get_all_config_flat", return_value=dict(_FLAT))
    @patch("esdc.config_wizard.Config.get_providers", return_value=dict(_PROVIDERS))
    @patch("esdc.config_wizard.questionary.select")
    def test_metadata_keeps_the_fallback_rows_only(
        self, mock_select, mock_providers, mock_flat, mock_discover
    ):
        # Off is the row that means "no model"; a blank pair row is not one.
        assert self._pick(mock_select, "metadata") == [
            "Chat setting / main",
            "Off / image fallback",
            *self._FIXED_ROWS,
        ]
        assert self._blank_rows(mock_select) == []

    @patch("esdc.config_wizard.discover_models", return_value=_catalog())
    @patch("esdc.config_wizard.Config.get_all_config_flat", return_value=dict(_FLAT))
    @patch("esdc.config_wizard.Config.get_providers", return_value=dict(_PROVIDERS))
    @patch("esdc.config_wizard.questionary.select")
    def test_cleanup_keeps_the_fallback_rows_only(
        self, mock_select, mock_providers, mock_flat, mock_discover
    ):
        assert self._pick(mock_select, "cleanup") == [
            "Chat setting / main",
            "Off",
            *self._FIXED_ROWS,
        ]
        assert self._blank_rows(mock_select) == []

    @patch("esdc.config_wizard.discover_models", return_value=_catalog())
    @patch("esdc.config_wizard.Config.get_all_config_flat", return_value=dict(_FLAT))
    @patch("esdc.config_wizard.Config.get_providers", return_value=dict(_PROVIDERS))
    @patch("esdc.config_wizard.questionary.select")
    def test_ocr_has_no_blank_pair_row(
        self, mock_select, mock_providers, mock_flat, mock_discover
    ):
        assert self._pick(mock_select, "ocr") == [
            "Custom local OCR model…",
            *self._FIXED_ROWS,
        ]
        assert self._blank_rows(mock_select) == []

    @patch("esdc.config_wizard.discover_models", return_value=_catalog("m1"))
    @patch("esdc.config_wizard.Config.get_all_config_flat", return_value=dict(_FLAT))
    @patch("esdc.config_wizard.Config.get_providers", return_value=dict(_PROVIDERS))
    @patch("esdc.config_wizard.questionary.select")
    def test_a_reported_model_is_still_a_row(
        self, mock_select, mock_providers, mock_flat, mock_discover
    ):
        assert self._pick(mock_select, "chat") == ["engine / m1", *self._FIXED_ROWS]
        assert self._blank_rows(mock_select) == []


class TestIncompleteSelectionsAreNotSaved:
    """An incomplete concrete selection is refused before anything is written."""

    @patch("esdc.config_wizard.Config.set_chat_selection")
    def test_chat_pair_needs_a_provider_and_a_model(self, mock_chat):
        for selection in (PairSelection("", "m1"), PairSelection("engine", "   ")):
            with pytest.raises(ValueError):
                _save_task_selection("chat", selection)
        mock_chat.assert_not_called()

    @patch(
        "esdc.config_wizard.Config.get_referenced_provider",
        return_value=dict(_FAKE_PROVIDERS["engine"]),
    )
    @patch("esdc.config_wizard.Config.set_config_values")
    def test_embedding_direct_endpoint_needs_a_model_not_a_host(
        self, mock_values, mock_ref
    ):
        with pytest.raises(ValueError):
            _save_task_selection(
                "embedding", DirectSelection("openai", "http://direct-box:1234", "")
            )
        mock_values.assert_not_called()

        # The host may stay empty (the default endpoint), the model may not.
        _save_task_selection("embedding", DirectSelection("openai", "", "embed-1"))
        assert mock_values.call_args.args[0] == {
            "embedding_backend": "openai",
            "embedding_provider": "",
            "embedding_host": "",
            "embedding_model": "embed-1",
        }

    @patch("esdc.config_wizard.Config.set_config_values")
    def test_rerank_needs_a_model_and_a_provider_reference(self, mock_values):
        """A status selection is never saved, and an empty pair is refused."""
        for selection in (
            NeedsConfigurationSelection(_RETIRED_RERANK_REASON),
            PairSelection("engine", "  "),
            PairSelection("", "rerank-model"),
        ):
            with pytest.raises(ValueError):
                _save_task_selection("rerank", selection)
        mock_values.assert_not_called()

    @patch(
        "esdc.config_wizard.Config.get_referenced_provider",
        return_value=dict(_FAKE_PROVIDERS["engine"]),
    )
    @patch("esdc.config_wizard.Config.set_config_values")
    def test_ocr_pair_needs_a_model_and_an_existing_provider(
        self, mock_values, mock_ref
    ):
        with pytest.raises(ValueError):
            _save_task_selection("ocr", PairSelection("engine", ""))
        mock_values.assert_not_called()

        mock_ref.side_effect = ValueError("Unknown provider reference: 'gone'")
        with pytest.raises(ValueError):
            _save_task_selection("ocr", PairSelection("gone", "vision-x"))
        mock_values.assert_not_called()

    @patch(
        "esdc.config_wizard.Config.get_referenced_provider",
        return_value=dict(_FAKE_PROVIDERS["engine"]),
    )
    @patch("esdc.config_wizard.Config.set_config_values")
    def test_metadata_and_cleanup_pairs_need_a_model_and_an_existing_provider(
        self, mock_values, mock_ref
    ):
        for role in ("metadata", "cleanup"):
            with pytest.raises(ValueError):
                _save_task_selection(role, PairSelection("engine", "   "))
        mock_values.assert_not_called()

        mock_ref.side_effect = ValueError("Unknown provider reference: 'gone'")
        with pytest.raises(ValueError):
            _save_task_selection("cleanup", PairSelection("gone", "qwen3:8b"))
        mock_values.assert_not_called()

    @patch("esdc.config_wizard.rich_print")
    @patch("esdc.config_wizard.Config.set_config_values")
    @patch("esdc.config_wizard.Config.get_referenced_provider")
    @patch("esdc.config_wizard.questionary.select")
    def test_the_action_menu_reports_the_rejected_save(
        self, mock_select, mock_ref, mock_values, mock_print
    ):
        mock_ref.return_value = dict(_FAKE_PROVIDERS["engine"])
        mock_select.return_value.unsafe_ask.side_effect = ["save"]

        _run_task_action("ocr", PairSelection("engine", ""))

        printed = " ".join(str(call) for call in mock_print.call_args_list)
        assert "needs a model" in printed
        mock_values.assert_not_called()


class TestTestActionKeepsTheActionMenu:
    """Test connection returns to the same menu, so Save follows without re-picking."""

    @staticmethod
    def _is_action_menu(call) -> bool:
        values = [str(c.value) for c in call.kwargs["choices"]]
        return {"save", "test", "cancel"} <= set(values)

    @patch("esdc.config_wizard._test_provider")
    @patch("esdc.config_wizard.Config.set_chat_selection")
    @patch("esdc.config_wizard.Config.get_referenced_provider")
    @patch("esdc.config_wizard.rich_print")
    @patch("esdc.config_wizard.questionary.select")
    def test_test_then_save_writes_once(
        self, mock_select, mock_print, mock_ref, mock_chat, mock_test_provider
    ):
        mock_ref.return_value = dict(_FAKE_PROVIDERS["engine"])
        mock_select.return_value.unsafe_ask.side_effect = ["test", "save"]

        _run_task_action("chat", PairSelection("engine", "draft-model"))

        # Test came back to this same menu, so Save used the same selection.
        assert mock_select.call_count == 2
        assert all(self._is_action_menu(call) for call in mock_select.call_args_list)
        mock_test_provider.assert_called_once()
        mock_chat.assert_called_once_with("engine", "draft-model")

    @patch("esdc.config_wizard._test_provider")
    @patch("esdc.config_wizard.Config.set_chat_selection")
    @patch("esdc.config_wizard.Config.get_referenced_provider")
    @patch("esdc.config_wizard.rich_print")
    @patch("esdc.config_wizard.questionary.select")
    def test_back_after_test_writes_nothing(
        self, mock_select, mock_print, mock_ref, mock_chat, mock_test_provider
    ):
        mock_ref.return_value = dict(_FAKE_PROVIDERS["engine"])
        mock_select.return_value.unsafe_ask.side_effect = ["test", None]

        _run_task_action("chat", PairSelection("engine", "draft-model"))

        assert mock_select.call_count == 2
        mock_test_provider.assert_called_once()
        mock_chat.assert_not_called()


class TestAddProviderDraft:
    """Collecting a provider writes nothing; only Save from the action menu does."""

    @staticmethod
    def _mocks(mock_select, side_effect):
        mock_select.return_value.unsafe_ask.side_effect = side_effect

    @patch("esdc.config_wizard._test_provider")
    @patch("esdc.config_wizard.Config.set_default_provider")
    @patch("esdc.config_wizard.Config.save_provider")
    @patch("esdc.config_wizard.Config.get_default_provider", return_value="")
    @patch("esdc.config_wizard.Config.get_providers", return_value={})
    @patch("esdc.config_wizard._fetch_models", return_value=["gpt-4o"])
    @patch("esdc.config_wizard.questionary.password")
    @patch("esdc.config_wizard.questionary.select")
    def test_test_then_cancel_writes_nothing(
        self,
        mock_select,
        mock_password,
        mock_fetch,
        mock_providers,
        mock_default,
        mock_save,
        mock_set_default,
        mock_test_provider,
    ):
        # openai: api_key, then reasoning effort, then model, then the action menu.
        self._mocks(mock_select, ["openai", "(none)", "gpt-4o", "test", "cancel"])
        mock_password.return_value.unsafe_ask.return_value = "sk-test"

        _add_provider_flow()

        draft = {"provider_type": "openai", "api_key": "sk-test", "model": "gpt-4o"}
        mock_test_provider.assert_called_once_with("openai", draft)
        mock_save.assert_not_called()
        mock_set_default.assert_not_called()

    @patch("esdc.config_wizard._test_provider")
    @patch("esdc.config_wizard.Config.set_default_provider")
    @patch("esdc.config_wizard.Config.save_provider")
    @patch("esdc.config_wizard.Config.get_default_provider", return_value="")
    @patch("esdc.config_wizard.Config.get_providers", return_value={})
    @patch("esdc.config_wizard._fetch_models", return_value=["gpt-4o"])
    @patch("esdc.config_wizard.questionary.password")
    @patch("esdc.config_wizard.questionary.select")
    def test_test_then_save_writes_the_draft_once(
        self,
        mock_select,
        mock_password,
        mock_fetch,
        mock_providers,
        mock_default,
        mock_save,
        mock_set_default,
        mock_test_provider,
    ):
        self._mocks(mock_select, ["openai", "(none)", "gpt-4o", "test", "save"])
        mock_password.return_value.unsafe_ask.return_value = "sk-test"

        _add_provider_flow()

        draft = {"provider_type": "openai", "api_key": "sk-test", "model": "gpt-4o"}
        # One write, and the test used the same draft: no second save is left over
        # from the old post-save confirm.
        mock_save.assert_called_once_with("openai", draft)
        mock_test_provider.assert_called_once_with("openai", draft)
        mock_set_default.assert_called_once_with("openai")

    @patch("esdc.config_wizard.Config.set_default_provider")
    @patch("esdc.config_wizard.Config.save_provider")
    @patch("esdc.config_wizard.Config.get_default_provider", return_value="")
    @patch("esdc.config_wizard.Config.get_providers", return_value={})
    @patch("esdc.config_wizard._fetch_models", return_value=["gpt-4o"])
    @patch("esdc.config_wizard.questionary.password")
    @patch("esdc.config_wizard.questionary.select")
    def test_back_or_escape_at_the_action_menu_writes_nothing(
        self,
        mock_select,
        mock_password,
        mock_fetch,
        mock_providers,
        mock_default,
        mock_save,
        mock_set_default,
    ):
        for exit_value in ("__back__", None):
            self._mocks(mock_select, ["openai", "(none)", "gpt-4o", exit_value])
            mock_password.return_value.unsafe_ask.return_value = "sk-test"

            _add_provider_flow()

        mock_save.assert_not_called()
        mock_set_default.assert_not_called()

    @patch("esdc.config_wizard.Config.remove_provider")
    @patch("esdc.config_wizard.Config.set_default_provider")
    @patch("esdc.config_wizard.Config.save_provider")
    @patch("esdc.config_wizard.Config.get_default_provider", return_value="my-openai")
    @patch(
        "esdc.config_wizard.Config.get_providers",
        return_value={"my-openai": {"provider_type": "openai"}},
    )
    @patch("esdc.config_wizard._fetch_models", return_value=["gpt-4o"])
    @patch("esdc.config_wizard.questionary.password")
    @patch("esdc.config_wizard.questionary.select")
    def test_cancel_before_save_leaves_the_existing_provider_untouched(
        self,
        mock_select,
        mock_password,
        mock_fetch,
        mock_providers,
        mock_default,
        mock_save,
        mock_set_default,
        mock_remove,
    ):
        """The same-type confirm belongs to Save, so Cancel before it writes nothing."""
        self._mocks(mock_select, ["openai", "(none)", "gpt-4o", "cancel"])
        mock_password.return_value.unsafe_ask.return_value = "sk-test"

        _add_provider_flow()

        # The overwrite question is never reached, so the existing provider stays.
        mock_remove.assert_not_called()
        mock_save.assert_not_called()
        mock_set_default.assert_not_called()

    @patch("esdc.config_wizard._test_provider")
    @patch("esdc.config_wizard.Config.save_provider")
    @patch("esdc.config_wizard.Config.get_default_provider", return_value="")
    @patch("esdc.config_wizard.Config.get_providers", return_value={})
    @patch("esdc.config_wizard._fetch_models", return_value=["gpt-4o"])
    @patch("esdc.config_wizard.questionary.password")
    @patch("esdc.config_wizard.questionary.select")
    def test_test_returns_to_the_same_action_menu(
        self,
        mock_select,
        mock_password,
        mock_fetch,
        mock_providers,
        mock_default,
        mock_save,
        mock_test_provider,
    ):
        self._mocks(
            mock_select, ["openai", "(none)", "gpt-4o", "test", "test", "cancel"]
        )
        mock_password.return_value.unsafe_ask.return_value = "sk-test"

        _add_provider_flow()

        def is_action_menu(call) -> bool:
            values = {str(choice.value) for choice in call.kwargs["choices"]}
            return {"save", "test", "cancel"} <= values

        # Collection prompts, then three turns of one action menu.
        assert mock_select.call_count == 6
        assert all(is_action_menu(call) for call in mock_select.call_args_list[3:])
        assert mock_test_provider.call_count == 2
        mock_save.assert_not_called()

    @patch("esdc.config_wizard._test_provider")
    @patch("esdc.config_wizard.Config.save_provider")
    @patch("esdc.config_wizard.Config.get_default_provider", return_value="existing")
    @patch("esdc.config_wizard.Config.get_providers", return_value={})
    @patch("esdc.config_wizard._fetch_models", return_value=["qwen-7b"])
    @patch("esdc.config_wizard.questionary.confirm")
    @patch("esdc.config_wizard.questionary.text")
    @patch("esdc.config_wizard.questionary.password")
    @patch("esdc.config_wizard.questionary.select")
    def test_test_then_save_uses_the_entered_draft(
        self,
        mock_select,
        mock_password,
        mock_text,
        mock_confirm,
        mock_fetch,
        mock_providers,
        mock_default,
        mock_save,
        mock_test_provider,
    ):
        # openai_compatible: name, base_url, api_key, model, then the action menu.
        # Test first, then Save, so the same draft is used for both.
        self._mocks(mock_select, ["openai_compatible", "qwen-7b", "test", "save"])
        mock_text.return_value.unsafe_ask.side_effect = [
            "desk",
            "http://desk-box:1234/v1",
        ]
        mock_password.return_value.unsafe_ask.return_value = ""
        # Test connection is chosen from the menu, so the old post-save confirm
        # is not asked; the default-provider confirm still is.
        mock_confirm.return_value.unsafe_ask.side_effect = [False]

        _add_provider_flow()

        draft = {
            "provider_type": "openai_compatible",
            "api_key": "",
            "base_url": "http://desk-box:1234/v1",
            "model": "qwen-7b",
        }
        mock_save.assert_called_once_with("desk", draft)
        mock_test_provider.assert_called_once_with("desk", draft)
        mock_confirm.assert_called_once()


class TestEditProviderDraft:
    """Field edits update the draft only; nothing is written until Save."""

    _SAVED = {
        "provider_type": "openai_compatible",
        "base_url": "http://engine:4000/v1",
        "api_key": "sk-saved",
        "model": "chat-a",
        "oauth": {"access_token": "tok", "refresh_token": "refresh-tok"},
        "unknown_key": "keep-me",
    }

    @patch("esdc.config_wizard.Config.remove_provider")
    @patch("esdc.config_wizard.Config.save_provider")
    @patch("esdc.config_wizard.Config.get_providers")
    @patch("esdc.config_wizard._fetch_models", return_value=["chat-b"])
    @patch("esdc.config_wizard.questionary.text")
    @patch("esdc.config_wizard.questionary.password")
    @patch("esdc.config_wizard.questionary.select")
    def test_edits_write_nothing_until_save(
        self,
        mock_select,
        mock_password,
        mock_text,
        mock_fetch,
        mock_get,
        mock_save,
        mock_remove,
    ):
        saved = dict(self._SAVED)
        mock_get.return_value = {"engine": saved}
        mock_select.return_value.unsafe_ask.side_effect = [
            "engine",
            "api_key",
            "base_url",
            "model",
            "chat-b",
            "__done__",
            "save",
        ]
        mock_password.return_value.unsafe_ask.return_value = "sk-new"
        mock_text.return_value.unsafe_ask.return_value = "http://desk-box:1234/v1"

        _edit_provider_flow()

        assert mock_save.call_count == 1
        mock_save.assert_called_once_with(
            "engine",
            {
                "provider_type": "openai_compatible",
                "base_url": "http://desk-box:1234/v1",
                "api_key": "sk-new",
                "model": "chat-b",
                "oauth": {"access_token": "tok", "refresh_token": "refresh-tok"},
                "unknown_key": "keep-me",
            },
        )
        # The saved entry itself was not changed on the way to Save.
        assert saved == self._SAVED
        mock_remove.assert_not_called()

    @patch("esdc.config_wizard.Config.save_provider")
    @patch("esdc.config_wizard.Config.get_providers")
    @patch("esdc.config_wizard.questionary.password")
    @patch("esdc.config_wizard.questionary.select")
    def test_blank_api_key_keeps_the_saved_value(
        self, mock_select, mock_password, mock_get, mock_save
    ):
        mock_get.return_value = {"engine": dict(self._SAVED)}
        mock_select.return_value.unsafe_ask.side_effect = [
            "engine",
            "api_key",
            "__done__",
            "save",
        ]
        mock_password.return_value.unsafe_ask.return_value = ""

        _edit_provider_flow()

        assert mock_save.call_args.args[1]["api_key"] == "sk-saved"

    @patch("esdc.config_wizard.Config.save_provider")
    @patch("esdc.config_wizard.Config.get_providers")
    @patch("esdc.config_wizard.questionary.password")
    @patch("esdc.config_wizard.questionary.select")
    def test_clear_api_key_writes_empty_only_on_save(
        self, mock_select, mock_password, mock_get, mock_save
    ):
        saved = dict(self._SAVED)
        mock_get.return_value = {"engine": saved}
        mock_select.return_value.unsafe_ask.side_effect = [
            "engine",
            "__clear_key__",
            "__done__",
            "cancel",
        ]

        _edit_provider_flow()

        mock_save.assert_not_called()
        assert saved["api_key"] == "sk-saved"

        # The Clear action is the only way to clear it, and it writes on Save.
        mock_select.return_value.unsafe_ask.side_effect = [
            "engine",
            "__clear_key__",
            "__done__",
            "save",
        ]
        _edit_provider_flow()

        assert mock_save.call_args.args[1]["api_key"] == ""
        assert saved["api_key"] == "sk-saved"

    @patch("esdc.config_wizard.Config.save_provider")
    @patch("esdc.config_wizard.Config.get_providers")
    @patch("esdc.config_wizard.questionary.select")
    def test_cancel_leaves_the_saved_provider_unchanged(
        self, mock_select, mock_get, mock_save
    ):
        saved = dict(self._SAVED)
        mock_get.return_value = {"engine": saved}
        mock_select.return_value.unsafe_ask.side_effect = [
            "engine",
            "__done__",
            "cancel",
        ]

        _edit_provider_flow()

        mock_save.assert_not_called()
        assert saved == self._SAVED

    @patch("esdc.config_wizard.Config.save_provider")
    @patch("esdc.config_wizard.Config.get_providers")
    @patch("esdc.config_wizard.questionary.select")
    def test_reasoning_effort_keeps_the_saved_value_as_default(
        self, mock_select, mock_get, mock_save
    ):
        mock_get.return_value = {
            "engine": {
                "provider_type": "openai",
                "api_key": "sk-saved",
                "model": "chat-a",
                "reasoning_effort": "low",
            }
        }
        mock_select.return_value.unsafe_ask.side_effect = [
            "engine",
            "reasoning_effort",
            "high",
            "__done__",
            "save",
        ]

        _edit_provider_flow()

        # The effort prompt opens on the saved value, not on "(none)".
        # Call 0 is the provider pick, call 1 the field menu, call 2 the effort list.
        assert mock_select.call_args_list[2].kwargs["default"] == "low"
        assert mock_save.call_args.args[1]["reasoning_effort"] == "high"

    @patch("esdc.config_wizard.Config.save_provider")
    @patch("esdc.config_wizard.Config.get_providers")
    @patch("esdc.config_wizard.questionary.text")
    @patch("esdc.config_wizard.questionary.select")
    def test_api_version_is_editable_with_the_saved_default(
        self, mock_select, mock_text, mock_get, mock_save
    ):
        mock_get.return_value = {
            "azure": {
                "provider_type": "azure_openai",
                "base_url": "https://azure-box",
                "api_key": "sk-saved",
                "model": "chat-a",
                "api_version": "2024-02-01",
            }
        }
        mock_select.return_value.unsafe_ask.side_effect = [
            "azure",
            "api_version",
            "__done__",
            "save",
        ]
        mock_text.return_value.unsafe_ask.return_value = "2025-01-01"

        _edit_provider_flow()

        assert mock_text.call_args.kwargs["default"] == "2024-02-01"
        assert mock_save.call_args.args[1]["api_version"] == "2025-01-01"

    @patch("esdc.config_wizard.Config.rename_provider")
    @patch("esdc.config_wizard.Config.remove_provider")
    @patch("esdc.config_wizard.Config.save_provider")
    @patch("esdc.config_wizard.Config.get_providers")
    @patch("esdc.config_wizard.questionary.text")
    @patch("esdc.config_wizard.questionary.select")
    def test_name_change_renames_instead_of_delete_and_recreate(
        self, mock_select, mock_text, mock_get, mock_save, mock_remove, mock_rename
    ):
        mock_get.return_value = {"engine": dict(self._SAVED)}
        mock_select.return_value.unsafe_ask.side_effect = [
            "engine",
            "name",
            "__done__",
            "save",
        ]
        mock_text.return_value.unsafe_ask.return_value = "desk"

        _edit_provider_flow()

        # The entry moves through rename_provider, so its references survive.
        # The draft travels with the rename: one call, one write.
        mock_rename.assert_called_once_with(
            "engine", "desk", provider_config=dict(self._SAVED)
        )
        mock_remove.assert_not_called()
        mock_save.assert_not_called()

    @patch("esdc.config_wizard.Config.rename_provider")
    @patch("esdc.config_wizard.Config.remove_provider")
    @patch("esdc.config_wizard.Config.save_provider")
    @patch("esdc.config_wizard.Config.get_providers")
    @patch("esdc.config_wizard.questionary.text")
    @patch("esdc.config_wizard.questionary.select")
    def test_rename_failure_is_reported_and_writes_nothing(
        self, mock_select, mock_text, mock_get, mock_save, mock_remove, mock_rename
    ):
        mock_get.return_value = {"engine": dict(self._SAVED)}
        mock_rename.side_effect = ValueError("Provider name already exists: 'desk'")
        mock_select.return_value.unsafe_ask.side_effect = [
            "engine",
            "name",
            "__done__",
            "save",
        ]
        mock_text.return_value.unsafe_ask.return_value = "desk"

        _edit_provider_flow()

        mock_rename.assert_called_once_with(
            "engine", "desk", provider_config=dict(self._SAVED)
        )
        mock_save.assert_not_called()
        mock_remove.assert_not_called()

    @patch("esdc.config_wizard.Config.get_provider_references", return_value=[])
    @patch("esdc.config_wizard.Config.rename_provider")
    @patch("esdc.config_wizard.Config.save_provider")
    @patch("esdc.config_wizard.Config.get_providers")
    @patch("esdc.config_wizard.questionary.text")
    @patch("esdc.config_wizard.questionary.select")
    def test_renamed_edit_carries_the_draft_in_the_same_call(
        self, mock_select, mock_text, mock_get, mock_save, mock_rename, mock_references
    ):
        mock_get.return_value = {"engine": dict(self._SAVED)}
        mock_select.return_value.unsafe_ask.side_effect = [
            "engine",
            "name",
            "base_url",
            "__done__",
            "save",
        ]
        mock_text.return_value.unsafe_ask.side_effect = [
            "desk",
            "http://desk-box:4000/v1",
        ]

        _edit_provider_flow()

        # One call carries the new name and the edited field, so there is no second
        # write that could fail after the references had already moved.
        mock_rename.assert_called_once_with(
            "engine",
            "desk",
            provider_config={**self._SAVED, "base_url": "http://desk-box:4000/v1"},
        )
        mock_save.assert_not_called()

    @patch("esdc.config_wizard.Config.get_provider_references", return_value=[])
    @patch("esdc.config_wizard.Config.rename_provider")
    @patch("esdc.config_wizard.Config.save_provider")
    @patch("esdc.config_wizard.Config.get_providers")
    @patch("esdc.config_wizard.questionary.text")
    @patch("esdc.config_wizard.questionary.select")
    def test_same_name_edit_still_writes_through_save_provider(
        self, mock_select, mock_text, mock_get, mock_save, mock_rename, mock_references
    ):
        mock_get.return_value = {"engine": dict(self._SAVED)}
        mock_select.return_value.unsafe_ask.side_effect = [
            "engine",
            "base_url",
            "__done__",
            "save",
        ]
        mock_text.return_value.unsafe_ask.return_value = "http://desk-box:4000/v1"

        _edit_provider_flow()

        # A name that did not change is not a rename: the draft is written as before.
        mock_rename.assert_not_called()
        mock_save.assert_called_once_with(
            "engine",
            {**self._SAVED, "base_url": "http://desk-box:4000/v1"},
        )

    @patch("esdc.config_wizard._test_provider")
    @patch("esdc.config_wizard.Config.save_provider")
    @patch("esdc.config_wizard.Config.get_providers")
    @patch("esdc.config_wizard.questionary.password")
    @patch("esdc.config_wizard.questionary.select")
    def test_test_uses_the_draft_and_returns_to_the_same_menu(
        self,
        mock_select,
        mock_password,
        mock_get,
        mock_save,
        mock_test_provider,
    ):
        mock_get.return_value = {"engine": dict(self._SAVED)}
        mock_select.return_value.unsafe_ask.side_effect = [
            "engine",
            "api_key",
            "__done__",
            "test",
            "cancel",
        ]
        mock_password.return_value.unsafe_ask.return_value = "sk-new"

        _edit_provider_flow()

        draft = dict(self._SAVED)
        draft["api_key"] = "sk-new"
        mock_test_provider.assert_called_once_with("engine", draft)
        mock_save.assert_not_called()

    @patch("esdc.config_wizard.Config.save_provider")
    @patch("esdc.config_wizard.Config.get_providers")
    @patch("esdc.config_wizard.questionary.select")
    def test_field_menu_lists_the_editable_fields_and_clear(
        self, mock_select, mock_get, mock_save
    ):
        mock_get.return_value = {"engine": dict(self._SAVED)}
        mock_select.return_value.unsafe_ask.side_effect = ["engine", "__back__"]

        _edit_provider_flow()

        labels = [
            str(choice.title) for choice in mock_select.call_args.kwargs["choices"]
        ]
        assert "model = chat-a" in labels
        assert "base_url = http://engine:4000/v1" in labels
        assert "api_key = sk****ed" in labels
        assert "Clear API key" in labels
        assert "name = engine" in labels
        assert "Done editing" in labels

    @patch("esdc.config_wizard.Config.save_provider")
    @patch("esdc.config_wizard.Config.get_providers")
    @patch("esdc.config_wizard.questionary.select")
    def test_a_saved_key_is_editable_even_when_the_type_has_no_key_field(
        self, mock_select, mock_get, mock_save
    ):
        """`ollama` prompts no key when adding, but a key saved on it is editable."""
        mock_get.return_value = {
            "gpu": {
                "provider_type": "ollama",
                "base_url": "http://gpu-box:11434",
                "api_key": "sk-saved",
                "model": "ollama-a",
            }
        }
        mock_select.return_value.unsafe_ask.side_effect = ["gpu", "__back__"]

        _edit_provider_flow()

        labels = [
            str(choice.title) for choice in mock_select.call_args.kwargs["choices"]
        ]
        assert "api_key = sk****ed" in labels
        assert "Clear API key" in labels
        # ollama has no reasoning_effort or api_version field, so neither is offered.
        assert not any("reasoning_effort" in label for label in labels)

    @patch("esdc.config_wizard.Config.save_provider")
    @patch("esdc.config_wizard.Config.get_providers")
    @patch("esdc.config_wizard.questionary.select")
    def test_clear_is_offered_only_when_a_key_is_present(
        self, mock_select, mock_get, mock_save
    ):
        mock_get.return_value = {
            "desk": {"provider_type": "anthropic", "model": "claude-x"}
        }
        mock_select.return_value.unsafe_ask.side_effect = ["desk", "__back__"]

        _edit_provider_flow()

        labels = [
            str(choice.title) for choice in mock_select.call_args.kwargs["choices"]
        ]
        assert "Clear API key" not in labels
        assert "reasoning_effort = (none)" in labels


class TestEmbeddingTestUsesEnvironmentPrecedence:
    """Environment values apply to an embedding test only, never to config."""

    _SAVED = {
        "provider_type": "openai_compatible",
        "base_url": "http://engine:4000/v1",
        "api_key": "sk-saved",
        "model": "embed-1",
    }

    @patch("esdc.config_wizard._test_provider")
    @patch("esdc.config_wizard.OpenAIEmbedder")
    @patch("esdc.config_wizard.Config.set_config_values")
    @patch("esdc.config_wizard.Config.get_referenced_provider")
    def test_embedding_test_uses_environment_values(
        self, mock_ref, mock_values, mock_embedder, mock_test_provider, monkeypatch
    ):
        saved = dict(self._SAVED)
        mock_ref.return_value = dict(saved)
        mock_embedder.return_value.generate_embedding.return_value = [0.1, 0.2]
        monkeypatch.setenv("ESDC_EMBEDDING_BACKEND", "openai")
        monkeypatch.setenv("ESDC_EMBEDDING_HOST", "http://env-box:1234/v1")
        monkeypatch.setenv("ESDC_EMBEDDING_API_KEY", "sk-env")
        monkeypatch.setenv("ESDC_EMBEDDING_MODEL", "env-model")

        _test_task_selection("embedding", PairSelection("engine", "embed-1"))

        # The embedder is built from the environment-overlaid draft, so the test
        # measures what runtime would use, not what is stored.
        assert mock_embedder.call_args.kwargs == {
            "host": "http://env-box:1234/v1",
            "model": "env-model",
            "api_key": "sk-env",
        }
        mock_test_provider.assert_not_called()
        mock_values.assert_not_called()
        assert saved == self._SAVED

    @patch("esdc.config_wizard._test_provider")
    @patch("esdc.config_wizard.OpenAIEmbedder")
    @patch("esdc.config_wizard.Config.set_config_values")
    @patch("esdc.config_wizard.Config.get_all_config_flat", return_value={})
    @patch("esdc.config_wizard.rich_print")
    def test_direct_endpoint_test_uses_environment_values(
        self,
        mock_print,
        mock_flat,
        mock_values,
        mock_embedder,
        mock_test_provider,
        monkeypatch,
    ):
        monkeypatch.setenv("ESDC_EMBEDDING_HOST", "http://env-box:1234/v1")
        monkeypatch.setenv("ESDC_EMBEDDING_API_KEY", "sk-env")
        mock_embedder.return_value.generate_embedding.return_value = [0.1, 0.2]

        _test_task_selection(
            "embedding", DirectSelection("openai", "http://direct-box:1234", "embed-1")
        )

        assert mock_embedder.call_args.kwargs["host"] == "http://env-box:1234/v1"
        assert mock_embedder.call_args.kwargs["api_key"] == "sk-env"
        mock_test_provider.assert_not_called()
        mock_values.assert_not_called()

    @patch("esdc.config_wizard._test_provider")
    @patch("esdc.config_wizard.Config.get_referenced_provider")
    @patch("esdc.config_wizard.rich_print")
    def test_an_environment_backend_that_disagrees_is_reported(
        self, mock_print, mock_ref, mock_test_provider, monkeypatch
    ):
        mock_ref.return_value = dict(self._SAVED)
        monkeypatch.setenv("ESDC_EMBEDDING_BACKEND", "ollama")

        _test_task_selection("embedding", PairSelection("engine", "embed-1"))

        printed = " ".join(str(call) for call in mock_print.call_args_list)
        assert "ESDC_EMBEDDING_BACKEND" in printed
        mock_test_provider.assert_not_called()

    @patch("esdc.config_wizard._test_provider")
    @patch("esdc.config_wizard.Config.get_referenced_provider")
    @patch("esdc.config_wizard.rich_print")
    def test_a_provider_that_cannot_serve_embeddings_is_reported(
        self, mock_print, mock_ref, mock_test_provider, monkeypatch
    ):
        """The environment backend never switches the source silently."""
        mock_ref.return_value = {"provider_type": "anthropic", "model": "claude-x"}
        monkeypatch.setenv("ESDC_EMBEDDING_BACKEND", "openai")

        _test_task_selection("embedding", PairSelection("desk", "claude-x"))

        printed = " ".join(str(call) for call in mock_print.call_args_list)
        assert "openai_compatible" in printed
        mock_test_provider.assert_not_called()

    @patch("esdc.config_wizard._test_provider")
    @patch("esdc.config_wizard.Config.get_referenced_provider")
    def test_other_tasks_keep_the_saved_values(
        self, mock_ref, mock_test_provider, monkeypatch
    ):
        mock_ref.return_value = dict(self._SAVED)
        monkeypatch.setenv("ESDC_EMBEDDING_HOST", "http://env-box:1234/v1")
        monkeypatch.setenv("ESDC_EMBEDDING_MODEL", "env-model")

        # Chat is not an embedding source, so the environment does not overlay it.
        _test_task_selection("chat", PairSelection("engine", "chat-a"))

        assert mock_test_provider.call_args.args[1] == {
            **self._SAVED,
            "model": "chat-a",
        }


class TestEmbeddingTestUsesTheEmbeddingEndpoint:
    """A chat request is not a connection test for an embedding source."""

    _SAVED = {
        "provider_type": "openai_compatible",
        "base_url": "http://engine:4000/v1",
        "api_key": "sk-saved",
        "model": "embed-1",
    }

    @patch("esdc.config_wizard._test_provider")
    @patch("esdc.config_wizard.OpenAIEmbedder")
    @patch("esdc.config_wizard.Config.set_config_values")
    @patch("esdc.config_wizard.Config.get_all_config_flat", return_value={})
    @patch("esdc.config_wizard.Config.get_referenced_provider")
    @patch("esdc.config_wizard.rich_print")
    def test_openai_compatible_test_calls_the_embedding_endpoint(
        self,
        mock_print,
        mock_ref,
        mock_flat,
        mock_values,
        mock_embedder,
        mock_test_provider,
    ):
        mock_ref.return_value = dict(self._SAVED)
        mock_embedder.return_value.generate_embedding.return_value = [0.1, 0.2, 0.3]

        _test_task_selection("embedding", PairSelection("engine", "embed-1"))

        # The embedder is built from the draft, so the test measures the embeddings
        # route the runtime uses rather than the chat route.
        assert mock_embedder.call_args.kwargs == {
            "host": "http://engine:4000/v1",
            "model": "embed-1",
            "api_key": "sk-saved",
        }
        mock_embedder.return_value.generate_embedding.assert_called_once_with(
            "esdc embedding connection test"
        )
        # An embedding-only deployment answers 404 on the chat route, so a chat
        # completion cannot prove it is reachable.
        mock_test_provider.assert_not_called()
        # Testing never writes and never enables anything.
        mock_values.assert_not_called()
        printed = " ".join(str(call) for call in mock_print.call_args_list)
        assert "Success" in printed
        assert "3-dimensional vector" in printed

    @patch("esdc.config_wizard._test_provider")
    @patch("esdc.config_wizard.OpenAIEmbedder")
    @patch("esdc.config_wizard.Config.set_config_value")
    @patch("esdc.config_wizard.Config.set_config_values")
    @patch("esdc.config_wizard.Config.get_all_config_flat", return_value={})
    @patch("esdc.config_wizard.Config.get_referenced_provider")
    @patch("esdc.config_wizard.rich_print")
    def test_failed_embedding_call_is_reported_and_writes_nothing(
        self,
        mock_print,
        mock_ref,
        mock_flat,
        mock_values,
        mock_set_value,
        mock_embedder,
        mock_test_provider,
    ):
        mock_ref.return_value = dict(self._SAVED)
        class _StatusError(RuntimeError):
            status_code = 404

        failure = _StatusError(
            "SECRET EMBEDDING BODY at http://engine:4000/v1/embeddings"
        )
        mock_embedder.return_value.generate_embedding.side_effect = failure

        _test_task_selection("embedding", PairSelection("engine", "embed-1"))

        printed = " ".join(str(call) for call in mock_print.call_args_list)
        assert "Failed:" in printed
        assert "HTTP 404" in printed
        assert "SECRET" not in printed
        assert "EMBEDDING BODY" not in printed
        assert "http://engine" not in printed
        mock_test_provider.assert_not_called()
        mock_values.assert_not_called()
        mock_set_value.assert_not_called()

    @patch("esdc.config_wizard.OpenAIEmbedder")
    @patch("esdc.config_wizard.Config.set_config_values")
    @patch("esdc.config_wizard.Config.get_all_config_flat", return_value={})
    @patch("esdc.config_wizard.Config.get_referenced_provider")
    @patch("esdc.config_wizard.rich_print")
    def test_an_empty_vector_is_reported_as_a_failure(
        self, mock_print, mock_ref, mock_flat, mock_values, mock_embedder
    ):
        """A response without a vector is not a successful connection test."""
        mock_ref.return_value = dict(self._SAVED)
        mock_embedder.return_value.generate_embedding.return_value = []

        _test_task_selection("embedding", PairSelection("engine", "embed-1"))

        printed = " ".join(str(call) for call in mock_print.call_args_list)
        assert "Failed:" in printed
        assert "no vector" in printed
        mock_values.assert_not_called()

    @patch("esdc.config_wizard._test_provider")
    @patch("esdc.config_wizard.OpenAIEmbedder")
    @patch("esdc.config_wizard.OllamaEmbedder")
    @patch(
        "esdc.config_wizard.Config.get_all_config_flat",
        return_value={"corpus.ollama_host": "http://ollama-box:11434"},
    )
    @patch("esdc.config_wizard.Config.get_referenced_provider")
    @patch("esdc.config_wizard.rich_print")
    def test_ollama_test_uses_the_ollama_embedder(
        self,
        mock_print,
        mock_ref,
        mock_flat,
        mock_ollama,
        mock_openai,
        mock_test_provider,
    ):
        mock_ref.return_value = {
            "provider_type": "ollama",
            "base_url": "http://gpu-box:11434",
            "model": "chat-a",
        }
        mock_ollama.return_value.generate_embedding.return_value = [0.4, 0.5]

        _test_task_selection("embedding", PairSelection("gpu", "qwen3-embedding:0.6b"))

        # The provider address is used, and the model asked for is the pair's own
        # model rather than the provider's chat model.
        assert mock_ollama.call_args.kwargs == {
            "host": "http://gpu-box:11434",
            "model": "qwen3-embedding:0.6b",
        }
        mock_ollama.return_value.generate_embedding.assert_called_once_with(
            "esdc embedding connection test"
        )
        mock_openai.assert_not_called()
        mock_test_provider.assert_not_called()

    @patch("esdc.config_wizard.OllamaEmbedder")
    @patch(
        "esdc.config_wizard.Config.get_all_config_flat",
        return_value={
            "embedding_host": "http://embedding-box:11434",
            "corpus.ollama_host": "http://ollama-box:11434",
        },
    )
    @patch("esdc.config_wizard.Config.get_referenced_provider")
    def test_ollama_test_falls_back_to_the_saved_embedding_host(
        self, mock_ref, mock_flat, mock_ollama
    ):
        mock_ref.return_value = {"provider_type": "ollama", "model": "chat-a"}
        mock_ollama.return_value.generate_embedding.return_value = [0.4]

        _test_task_selection("embedding", PairSelection("gpu", "qwen3-embedding:0.6b"))

        # A provider with no saved address still names one endpoint to test: the
        # runtime embedding host, not the corpus Ollama host that OCR uses. The model
        # is the one the selection names.
        assert mock_ollama.call_args.kwargs == {
            "host": "http://embedding-box:11434",
            "model": "qwen3-embedding:0.6b",
        }

    @patch("esdc.config_wizard._test_provider")
    @patch("esdc.config_wizard.OpenAIEmbedder")
    @patch("esdc.config_wizard.OllamaEmbedder")
    @patch(
        "esdc.config_wizard.Config.get_all_config_flat",
        return_value={"embedding_api_key": "sk-saved"},
    )
    def test_direct_openai_row_tests_the_embedding_endpoint(
        self, mock_flat, mock_ollama, mock_openai, mock_test_provider
    ):
        _test_task_selection(
            "embedding", DirectSelection("openai", "http://direct-box:1234", "embed-1")
        )

        # The row's own endpoint, with the key the selection carries or the saved one.
        assert mock_openai.call_args.kwargs == {
            "host": "http://direct-box:1234",
            "model": "embed-1",
            "api_key": "sk-saved",
        }
        mock_openai.return_value.generate_embedding.assert_called_once_with(
            "esdc embedding connection test"
        )
        mock_ollama.assert_not_called()
        mock_test_provider.assert_not_called()

    @patch("esdc.config_wizard._test_provider")
    @patch("esdc.config_wizard.OpenAIEmbedder")
    @patch("esdc.config_wizard.OllamaEmbedder")
    @patch(
        "esdc.config_wizard.Config.get_all_config_flat",
        return_value={"embedding_api_key": "sk-saved"},
    )
    def test_direct_openai_row_tests_the_key_the_selection_carries(
        self, mock_flat, mock_ollama, mock_openai, mock_test_provider
    ):
        _test_task_selection(
            "embedding",
            DirectSelection(
                "openai", "http://direct-box:1234", "embed-1", api_key="sk-draft"
            ),
        )

        # An editor draft wins over the saved key, and a cleared key tests without one.
        assert mock_openai.call_args.kwargs == {
            "host": "http://direct-box:1234",
            "model": "embed-1",
            "api_key": "sk-draft",
        }
        mock_test_provider.assert_not_called()

    @patch("esdc.config_wizard._test_provider")
    @patch("esdc.config_wizard.OpenAIEmbedder")
    @patch("esdc.config_wizard.OllamaEmbedder")
    @patch(
        "esdc.config_wizard.Config.get_all_config_flat",
        return_value={"corpus.ollama_host": "http://corpus-box:11434"},
    )
    def test_direct_ollama_row_tests_its_own_endpoint_not_the_chat_route(
        self, mock_flat, mock_ollama, mock_openai, mock_test_provider
    ):
        _test_task_selection(
            "embedding",
            DirectSelection(
                "ollama", "http://ollama-box:11434", OllamaEmbedder.OLLAMA_TAG
            ),
        )

        # A direct row names its own address, so the corpus Ollama host does not
        # replace it, and the embedding route is the one asked. The model asked for is
        # the one runtime will use.
        assert mock_ollama.call_args.kwargs == {
            "host": "http://ollama-box:11434",
            "model": OllamaEmbedder.OLLAMA_TAG,
        }
        mock_ollama.return_value.generate_embedding.assert_called_once_with(
            "esdc embedding connection test"
        )
        mock_openai.assert_not_called()
        mock_test_provider.assert_not_called()

    @patch("esdc.config_wizard._test_provider")
    @patch("esdc.config_wizard.OpenAIEmbedder")
    @patch("esdc.config_wizard.OllamaEmbedder")
    @patch(
        "esdc.config_wizard.Config.get_all_config_flat",
        return_value={
            "embedding_backend": "local",
            "embedding_model": "embed-1",
        },
    )
    @patch("esdc.config_wizard.Config.set_config_values")
    @patch("esdc.config_wizard.rich_print")
    def test_retired_embedding_backend_test_loads_nothing_and_writes_nothing(
        self,
        mock_print,
        mock_values,
        mock_flat,
        mock_ollama,
        mock_openai,
        mock_test_provider,
    ):
        """A retired stored backend is a status, so there is nothing to reach.

        The test reports it and stops: no embedder is built, no request is made,
        nothing is written, and the stored value is left as it is.
        """
        _test_task_selection(
            "embedding", NeedsConfigurationSelection(_RETIRED_EMBEDDING_REASON)
        )

        printed = " ".join(str(call) for call in mock_print.call_args_list)
        assert (
            "Nothing to test: retired in-process embedding backend "
            "— choose an HTTP service." in printed
        )
        mock_ollama.assert_not_called()
        mock_openai.assert_not_called()
        mock_test_provider.assert_not_called()
        mock_values.assert_not_called()
        assert mock_flat.return_value["embedding_backend"] == "local"
        assert mock_flat.return_value["embedding_model"] == "embed-1"


_DIRECT_FLAT = {
    "embedding_backend": "openai",
    "embedding_provider": "",
    "embedding_host": "http://direct-box:1234",
    "embedding_model": "embed-1",
    "embedding_api_key": "sk-saved",
}

_DIRECT_OLLAMA_FLAT = {
    "embedding_backend": "ollama",
    "embedding_provider": "",
    "embedding_host": "http://ollama-box:11434",
    "embedding_model": "embed-1",
}


class TestDirectEmbeddingEditor:
    """The direct embedding row edits its own fields, without a saved provider.

    The editor is not a provider submenu: it holds host, model, and the key, and it
    writes nothing itself. Save happens on the task action menu for the selection the
    editor returns, so the key rule decides whether `embedding_api_key` is written:
    None keeps the saved one, "" clears it, a string writes that key.
    """

    @staticmethod
    def _titles(mock_select, index: int = -1) -> list[str]:
        choices = mock_select.call_args_list[index].kwargs["choices"]
        return [str(choice.title) for choice in choices]

    @staticmethod
    def _values(mock_select, index: int = -1) -> list:
        choices = mock_select.call_args_list[index].kwargs["choices"]
        return [choice.value for choice in choices]

    @patch("esdc.config_wizard.discover_models", return_value=_catalog("embed-1"))
    @patch(
        "esdc.config_wizard.Config.get_all_config_flat",
        return_value=dict(_DIRECT_FLAT),
    )
    @patch(
        "esdc.config_wizard.Config.get_providers", return_value=dict(_FAKE_PROVIDERS)
    )
    @patch("esdc.config_wizard.rich_print")
    @patch("esdc.config_wizard.questionary.select")
    def test_direct_settings_row_opens_the_editor_not_a_provider_submenu(
        self, mock_select, mock_print, mock_providers, mock_flat, mock_discover
    ):
        # The row sits in the same searchable list as the provider/model rows, and
        # Escape at the editor returns to it.
        mock_select.return_value.unsafe_ask.side_effect = [
            _DIRECT_SETTINGS_VALUE,
            None,
            None,
        ]
        assert _choose_provider_model("embedding", None) is None

        assert "Direct embedding settings…" in self._titles(mock_select, 0)
        assert _DIRECT_SETTINGS_VALUE in self._values(mock_select, 0)

        # The editor menu is the direct fields at one level, not a provider list.
        assert self._values(mock_select, 1) == [
            "__back__",
            "host",
            "model",
            "api_key",
            "__clear_key__",
            "__done__",
        ]
        assert not any("engine /" in title for title in self._titles(mock_select, 1))

    @patch("esdc.config_wizard.Config.set_config_values")
    @patch(
        "esdc.config_wizard.Config.get_all_config_flat",
        return_value=dict(_DIRECT_FLAT),
    )
    @patch("esdc.config_wizard.questionary.password")
    @patch("esdc.config_wizard.questionary.text")
    @patch("esdc.config_wizard.rich_print")
    @patch("esdc.config_wizard.questionary.select")
    def test_editor_changes_host_model_and_key_and_save_writes_them(
        self, mock_select, mock_print, mock_text, mock_password, mock_flat, mock_values
    ):
        mock_select.return_value.unsafe_ask.side_effect = [
            "host",
            "model",
            "api_key",
            "__done__",
        ]
        mock_text.return_value.unsafe_ask.side_effect = [
            "http://new-box:1234",
            "embed-2",
        ]
        mock_password.return_value.unsafe_ask.return_value = "sk-new"

        selection = _direct_embedding_flow()
        assert selection is not None
        assert selection == DirectSelection(
            "openai", "http://new-box:1234", "embed-2", api_key="sk-new"
        )

        # The editor writes nothing; Save is the task action menu.
        mock_values.assert_not_called()
        _save_task_selection("embedding", selection)
        assert mock_values.call_args.args[0] == {
            "embedding_backend": "openai",
            "embedding_provider": "",
            "embedding_host": "http://new-box:1234",
            "embedding_model": "embed-2",
            "embedding_api_key": "sk-new",
        }

    @patch("esdc.config_wizard.Config.set_config_values")
    @patch(
        "esdc.config_wizard.Config.get_all_config_flat",
        return_value=dict(_DIRECT_FLAT),
    )
    @patch("esdc.config_wizard.questionary.password")
    @patch("esdc.config_wizard.rich_print")
    @patch("esdc.config_wizard.questionary.select")
    def test_blank_key_keeps_the_saved_key_on_save(
        self, mock_select, mock_print, mock_password, mock_flat, mock_values
    ):
        mock_select.return_value.unsafe_ask.side_effect = ["api_key", "__done__"]
        mock_password.return_value.unsafe_ask.return_value = ""

        selection = _direct_embedding_flow()
        assert selection is not None
        # A blank entry keeps the current key, which the menu shows as (kept): the
        # selection carries no key, so Save leaves the saved one alone.
        assert selection.api_key is None
        _save_task_selection("embedding", selection)
        assert "embedding_api_key" not in mock_values.call_args.args[0]

    @patch("esdc.config_wizard.Config.set_config_values")
    @patch(
        "esdc.config_wizard.Config.get_all_config_flat",
        return_value=dict(_DIRECT_FLAT),
    )
    @patch("esdc.config_wizard.questionary.text")
    @patch("esdc.config_wizard.rich_print")
    @patch("esdc.config_wizard.questionary.select")
    def test_a_cancelled_field_keeps_the_draft_value(
        self, mock_select, mock_print, mock_text, mock_flat, mock_values
    ):
        # Escape at one prompt cancels that field only, not the draft.
        mock_select.return_value.unsafe_ask.side_effect = ["host", "model", "__done__"]
        mock_text.return_value.unsafe_ask.side_effect = [None, "embed-2"]

        selection = _direct_embedding_flow()
        assert selection is not None
        assert selection.host == "http://direct-box:1234"
        assert selection.model_id == "embed-2"
        assert selection.api_key is None
        mock_values.assert_not_called()

    @patch("esdc.config_wizard.Config.set_config_values")
    @patch(
        "esdc.config_wizard.Config.get_all_config_flat",
        return_value=dict(_DIRECT_FLAT),
    )
    @patch("esdc.config_wizard.questionary.password")
    @patch("esdc.config_wizard.rich_print")
    @patch("esdc.config_wizard.questionary.select")
    def test_clear_api_key_writes_an_empty_key_on_save(
        self, mock_select, mock_print, mock_password, mock_flat, mock_values
    ):
        mock_select.return_value.unsafe_ask.side_effect = [
            "api_key",
            "__clear_key__",
            "__done__",
        ]
        # A blank entry alone would have kept the saved key; Clear is the way out.
        mock_password.return_value.unsafe_ask.return_value = ""

        selection = _direct_embedding_flow()
        assert selection is not None
        assert selection.api_key == ""
        _save_task_selection("embedding", selection)
        assert mock_values.call_args.args[0]["embedding_api_key"] == ""

    @patch("esdc.config_wizard._run_task_action")
    @patch("esdc.config_wizard.Config.set_config_values")
    @patch("esdc.config_wizard.discover_models", return_value=_catalog("embed-1"))
    @patch(
        "esdc.config_wizard.Config.get_all_config_flat",
        return_value=dict(_DIRECT_FLAT),
    )
    @patch(
        "esdc.config_wizard.Config.get_providers", return_value=dict(_FAKE_PROVIDERS)
    )
    @patch("esdc.config_wizard.rich_print")
    @patch("esdc.config_wizard.questionary.select")
    def test_back_and_escape_write_nothing(
        self,
        mock_select,
        mock_print,
        mock_providers,
        mock_flat,
        mock_discover,
        mock_values,
        mock_action,
    ):
        # Back at the editor menu discards the draft.
        mock_select.return_value.unsafe_ask.side_effect = ["__back__"]
        assert _direct_embedding_flow() is None

        # Escape at the editor menu is read as Back the same way.
        mock_select.return_value.unsafe_ask.side_effect = [None]
        assert _direct_embedding_flow() is None

        # A cancelled editor returns to the list, and Escape there leaves without an
        # action, so no field is written.
        mock_select.return_value.unsafe_ask.side_effect = [
            _DIRECT_SETTINGS_VALUE,
            "__back__",
            None,
        ]
        assert _choose_provider_model("embedding", None) is None
        mock_values.assert_not_called()
        mock_action.assert_not_called()

    @patch("esdc.config_wizard.Config.set_config_values")
    @patch(
        "esdc.config_wizard.Config.get_all_config_flat",
        return_value={
            "embedding_backend": "openai",
            "embedding_provider": "",
            "embedding_host": "http://direct-box:1234",
            "embedding_model": "",
        },
    )
    @patch("esdc.config_wizard.questionary.text")
    @patch("esdc.config_wizard.rich_print")
    @patch("esdc.config_wizard.questionary.select")
    def test_done_refuses_an_empty_model_and_keeps_the_editor_open(
        self, mock_select, mock_print, mock_text, mock_flat, mock_values
    ):
        mock_text.return_value.unsafe_ask.return_value = ""
        mock_select.return_value.unsafe_ask.side_effect = [
            "__done__",
            "model",
            "__done__",
            "__back__",
        ]

        assert _direct_embedding_flow() is None
        printed = " ".join(str(call) for call in mock_print.call_args_list)
        # Refused twice, then Back: nothing was written.
        assert printed.count("need a model") == 2
        mock_values.assert_not_called()

    @patch("esdc.config_wizard.discover_models", return_value=_catalog("embed-1"))
    @patch(
        "esdc.config_wizard.Config.get_all_config_flat",
        return_value=dict(_DIRECT_OLLAMA_FLAT),
    )
    @patch(
        "esdc.config_wizard.Config.get_providers", return_value=dict(_FAKE_PROVIDERS)
    )
    @patch("esdc.config_wizard.rich_print")
    @patch("esdc.config_wizard.questionary.select")
    def test_the_embedding_local_row_is_the_ollama_service_row(
        self, mock_select, mock_print, mock_providers, mock_flat, mock_discover
    ):
        mock_select.return_value.unsafe_ask.side_effect = [None]
        _choose_provider_model("embedding", None)

        labels = _row_labels(mock_select)
        # No in-process row is offered any more: the Ollama service row names the
        # configured model, and a retired stored backend is a status, not a row.
        assert "Ollama (local) / embed-1" in labels
        assert not any(label.startswith("Local /") for label in labels)
        assert not any("GGUF" in label or "in-process" in label for label in labels)

        # The row carries the direct endpoint selection, not a local model selection.
        values = self._values(mock_select, 0)
        assert not any(isinstance(value, LocalSelection) for value in values)
        assert (
            DirectSelection("ollama", "http://ollama-box:11434", "embed-1") in values
        )

        assert _current_task_selection("embedding") == DirectSelection(
            "ollama", "http://ollama-box:11434", "embed-1"
        )

    @patch("esdc.config_wizard.Config.set_config_values")
    def test_direct_ollama_save_writes_backend_host_and_the_selected_model(
        self, mock_values
    ):
        _save_task_selection(
            "embedding",
            DirectSelection(
                "ollama",
                "http://ollama-box:11434",
                "embed-1",
                api_key="sk-new",
            ),
        )

        # The model the row shows is the model written, and an Ollama endpoint carries
        # no key, so the saved `embedding_api_key` is left alone.
        assert mock_values.call_args.args[0] == {
            "embedding_backend": "ollama",
            "embedding_provider": "",
            "embedding_host": "http://ollama-box:11434",
            "embedding_model": "embed-1",
        }


_RERANK_PROVIDER_FLAT = {
    "corpus.rerank": True,
    "corpus.rerank_backend": "provider",
    "corpus.rerank_provider": "engine",
    "corpus.rerank_model": "remote-alias",
}

_RERANK_LOCAL_FLAT = {
    "corpus.rerank": True,
    "corpus.rerank_backend": "local",
    "corpus.rerank_provider": "",
    "corpus.rerank_model": "my-org/my-model:file.gguf",
}


class TestRerankRowsAreProviderPairs:
    """Every rerank row is a saved provider pair; a retired stored value is a status.

    `corpus.rerank_backend` = `local` has no implementation any more, so it is reported
    as needing configuring and stays stored: showing it writes nothing and does not
    switch it. A legacy GGUF id saved on a provider pair keeps its row, stays
    selectable, and is saved exactly as it was stored or typed.
    """

    _LEGACY_MODEL = "my-org/my-model:file.gguf"

    @staticmethod
    def _values(mock_select, index: int = -1) -> list:
        choices = mock_select.call_args_list[index].kwargs["choices"]
        return [choice.value for choice in choices]

    @patch(
        "esdc.config_wizard.Config.get_referenced_provider",
        return_value=dict(_FAKE_PROVIDERS["engine"]),
    )
    @patch("esdc.config_wizard.discover_models", return_value=_catalog("remote-alias"))
    @patch(
        "esdc.config_wizard.Config.get_all_config_flat",
        return_value=dict(_RERANK_PROVIDER_FLAT),
    )
    @patch(
        "esdc.config_wizard.Config.get_providers", return_value=dict(_FAKE_PROVIDERS)
    )
    @patch("esdc.config_wizard.Config.set_config_values")
    @patch("esdc.config_wizard.rich_print")
    @patch("esdc.config_wizard.questionary.select")
    def test_only_provider_pair_rows_are_offered(
        self,
        mock_select,
        mock_print,
        mock_values,
        mock_providers,
        mock_flat,
        mock_discover,
        mock_ref,
    ):
        mock_select.return_value.unsafe_ask.side_effect = [None]
        _choose_provider_model("rerank", None)

        labels = _row_labels(mock_select)
        # The stored model keeps its pair row; there is no in-process row to offer and
        # no typed-GGUF row, because a GGUF file is not a service model id.
        assert "engine / remote-alias" in labels
        assert not any(label.startswith("Local /") for label in labels)
        assert "Local custom model…" not in labels
        assert not any("GGUF" in label or "in-process" in label for label in labels)

        mock_select.return_value.unsafe_ask.side_effect = [
            PairSelection("engine", "remote-alias")
        ]
        selection = _choose_provider_model("rerank", None)
        assert selection == PairSelection("engine", "remote-alias")
        assert selection is not None

        # The row writes nothing by itself; Save is the task action menu, and it writes
        # the model exactly as stored.
        mock_values.assert_not_called()
        _save_task_selection("rerank", selection)
        assert mock_values.call_args.args[0] == {
            "corpus.rerank_backend": "provider",
            "corpus.rerank_provider": "engine",
            "corpus.rerank_model": "remote-alias",
        }

    @patch("esdc.config_wizard.discover_models", return_value=_catalog("m1"))
    @patch(
        "esdc.config_wizard.Config.get_all_config_flat",
        return_value=dict(_RERANK_LOCAL_FLAT),
    )
    @patch(
        "esdc.config_wizard.Config.get_providers", return_value=dict(_FAKE_PROVIDERS)
    )
    @patch("esdc.config_wizard.Config.set_config_values")
    @patch("esdc.config_wizard.rich_print")
    @patch("esdc.config_wizard.questionary.select")
    def test_retired_local_backend_is_a_status_and_never_a_row(
        self,
        mock_select,
        mock_print,
        mock_values,
        mock_providers,
        mock_flat,
        mock_discover,
    ):
        mock_select.return_value.unsafe_ask.side_effect = [None]
        _choose_provider_model("rerank", None)

        labels = _row_labels(mock_select)
        # A retired stored backend is reported by the summary, not offered as a row, so
        # the stored GGUF id is not shown as a choice and never read as a model file.
        assert not any(label.startswith("Local /") for label in labels)
        assert not any("Local custom model…" in label for label in labels)
        assert not any("needs configuring" in label for label in labels)
        assert not any(
            isinstance(value, NeedsConfigurationSelection)
            for value in self._values(mock_select, 0)
        )

        current = _current_task_selection("rerank")
        assert current == NeedsConfigurationSelection(_RETIRED_RERANK_REASON)
        assert isinstance(current, NeedsConfigurationSelection)
        assert not _selectable(current)
        assert _selection_label("rerank", current) == (
            "Needs configuring (retired local rerank backend — choose a rerank service)"
        )

        # Reading the state writes nothing and the stored value is preserved.
        mock_values.assert_not_called()
        assert mock_flat.return_value["corpus.rerank_model"] == self._LEGACY_MODEL
        assert mock_flat.return_value["corpus.rerank_provider"] == ""

    @patch(
        "esdc.config_wizard.Config.get_referenced_provider",
        return_value=dict(_FAKE_PROVIDERS["engine"]),
    )
    @patch("esdc.config_wizard.discover_models", return_value=_catalog(_LEGACY_MODEL))
    @patch(
        "esdc.config_wizard.Config.get_all_config_flat",
        return_value={
            "corpus.rerank": True,
            "corpus.rerank_backend": "provider",
            "corpus.rerank_provider": "engine",
            "corpus.rerank_model": _LEGACY_MODEL,
        },
    )
    @patch(
        "esdc.config_wizard.Config.get_providers", return_value=dict(_FAKE_PROVIDERS)
    )
    @patch("esdc.config_wizard.Config.set_config_values")
    @patch("esdc.config_wizard.questionary.text")
    @patch("esdc.config_wizard.rich_print")
    @patch("esdc.config_wizard.questionary.select")
    def test_a_legacy_gguf_pair_row_stays_selectable_and_is_saved_unchanged(
        self,
        mock_select,
        mock_print,
        mock_text,
        mock_values,
        mock_providers,
        mock_flat,
        mock_discover,
        mock_ref,
    ):
        mock_select.return_value.unsafe_ask.side_effect = [None]
        _choose_provider_model("rerank", None)

        # The label says the value needs configuring, but the row is still a pair row
        # and the stored value is not rewritten to make it one.
        labels = _row_labels(mock_select)
        assert f"engine / {self._LEGACY_MODEL} (needs configuring)" in labels
        assert not any(label.startswith("Local /") for label in labels)

        saved = _current_task_selection("rerank")
        assert saved == PairSelection("engine", self._LEGACY_MODEL)
        assert isinstance(saved, PairSelection)
        assert _selectable(saved)
        assert isinstance(saved, PairSelection)
        assert saved in self._values(mock_select, 0)

        # Typing the same id through the custom pair flow returns it unchanged.
        mock_select.return_value.unsafe_ask.side_effect = ["engine"]
        mock_text.return_value.unsafe_ask.return_value = self._LEGACY_MODEL
        selection = _custom_pair_flow("rerank", None)
        assert selection == PairSelection("engine", self._LEGACY_MODEL)
        assert isinstance(selection, PairSelection)

        mock_values.assert_not_called()
        _save_task_selection("rerank", selection)
        assert mock_values.call_args.args[0] == {
            "corpus.rerank_backend": "provider",
            "corpus.rerank_provider": "engine",
            "corpus.rerank_model": self._LEGACY_MODEL,
        }

    @patch("esdc.config_wizard.discover_models", return_value=_catalog("m1"))
    @patch(
        "esdc.config_wizard.Config.get_all_config_flat",
        return_value=dict(_RERANK_LOCAL_FLAT),
    )
    @patch(
        "esdc.config_wizard.Config.get_providers", return_value=dict(_FAKE_PROVIDERS)
    )
    @patch("esdc.config_wizard.Config.set_config_values")
    @patch("esdc.config_wizard.questionary.text")
    @patch("esdc.config_wizard.rich_print")
    @patch("esdc.config_wizard.questionary.select")
    def test_an_empty_custom_model_is_refused_and_writes_nothing(
        self,
        mock_select,
        mock_print,
        mock_text,
        mock_values,
        mock_providers,
        mock_flat,
        mock_discover,
    ):
        # A blank model is refused and returns to the picker, where Escape leaves.
        mock_text.return_value.unsafe_ask.side_effect = ["   "]
        mock_select.return_value.unsafe_ask.side_effect = ["engine", None]

        assert _custom_pair_flow("rerank", None) is None
        printed = " ".join(str(call) for call in mock_print.call_args_list)
        assert "needs a model id" in printed
        mock_values.assert_not_called()

        # Escape at the prompt is a cancel, not an empty model.
        mock_text.return_value.unsafe_ask.side_effect = [None]
        assert _custom_pair_flow("rerank", None) is None
        mock_values.assert_not_called()

    @patch(
        "esdc.config_wizard.Config.get_referenced_provider",
        return_value=dict(_FAKE_PROVIDERS["engine"]),
    )
    @patch("esdc.config_wizard.Config.set_config_values")
    def test_pair_save_keeps_the_enable_switch(self, mock_values, mock_ref):
        _save_task_selection("rerank", PairSelection("engine", self._LEGACY_MODEL))

        written = mock_values.call_args.args[0]
        # The Enabled switch is its own field and is never part of a source save.
        assert "corpus.rerank" not in written
        assert written == {
            "corpus.rerank_backend": "provider",
            "corpus.rerank_provider": "engine",
            "corpus.rerank_model": self._LEGACY_MODEL,
        }

    @patch("esdc.config_wizard.Config.set_config_values")
    def test_the_retired_status_is_never_saved(self, mock_values):
        # A status is not a selection Save accepts: it carries no reference or model.
        with pytest.raises(ValueError):
            _save_task_selection(
                "rerank", NeedsConfigurationSelection(_RETIRED_RERANK_REASON)
            )
        mock_values.assert_not_called()
_NO_IN_PROCESS_FLAT = {
    "embedding_backend": "ollama",
    "embedding_provider": "gpu",
    "embedding_model": "embed-1",
    "corpus.rerank_backend": "provider",
    "corpus.rerank_provider": "engine",
    "corpus.rerank_model": "rerank-model",
}


class TestNoInProcessChoiceIsOffered:
    """Nothing the wizard offers for embedding or rerank is an in-process GGUF.

    Both the picker rows and the raw field prompts are checked: a retired stored value
    has its own row that reports it, and every offered choice is an HTTP service.
    """

    @staticmethod
    def _rows(mock_select, index: int) -> tuple[list[str], list[Any]]:
        choices = mock_select.call_args_list[index].kwargs["choices"]
        return (
            [str(choice.title) for choice in choices],
            [choice.value for choice in choices],
        )

    @patch("esdc.config_wizard.discover_models", return_value=_catalog("embed-1"))
    @patch(
        "esdc.config_wizard.Config.get_all_config_flat",
        return_value=dict(_NO_IN_PROCESS_FLAT),
    )
    @patch(
        "esdc.config_wizard.Config.get_providers", return_value=dict(_FAKE_PROVIDERS)
    )
    @patch("esdc.config_wizard.questionary.select")
    def test_no_in_process_choice_is_offered(
        self, mock_select, mock_providers, mock_flat, mock_discover
    ):
        # The picker rows for both tasks.
        mock_select.return_value.unsafe_ask.side_effect = [None, None]
        _choose_provider_model("embedding", None)
        embedding_labels, embedding_values = self._rows(mock_select, 0)
        _choose_provider_model("rerank", None)
        rerank_labels, rerank_values = self._rows(mock_select, 1)

        # The backend prompts, the other place an inference choice is offered.
        mock_select.return_value.unsafe_ask.side_effect = ["ollama", "provider"]
        flat = dict(_NO_IN_PROCESS_FLAT)
        _prompt_for_config_value("embedding_backend", "ollama", flat)
        embedding_prompt_labels = self._rows(mock_select, 2)[0]
        _prompt_for_config_value("corpus.rerank_backend", "provider", flat)
        rerank_prompt_labels = self._rows(mock_select, 3)[0]

        labels = (
            embedding_labels
            + rerank_labels
            + embedding_prompt_labels
            + rerank_prompt_labels
        )

        assert not [
            label for label in labels if "GGUF" in label or "in-process" in label
        ]
        assert any("HTTP service" in label for label in labels)
        assert any("HTTP rerank service" in label for label in labels)

        # A retired stored value is not offered as a working choice, and a status is
        # never a row: the picker has no row that reports it.
        assert "local" not in labels
        assert not any(
            isinstance(value, NeedsConfigurationSelection)
            for value in embedding_values + rerank_values
        )

    def test_legacy_gguf_value_is_shown_as_needing_configuration(self):
        """The hint is a read: it reports the stored value and leaves it alone."""
        stored = "ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF"
        flat = {"corpus.rerank_model": stored}
        hint = field_hint("corpus.rerank_model", flat)
        assert hint is not None
        assert "needs configuring" in hint.lower()
        assert "preserved" in hint
        assert flat["corpus.rerank_model"] == stored

        # The same holds for the two retired backends.
        for key in ("embedding_backend", "corpus.rerank_backend"):
            values = {key: "local"}
            hint = field_hint(key, values)
            assert hint is not None
            assert "needs configuring" in hint.lower()
            assert values[key] == "local"
        service = {"corpus.rerank_model": "rerank-model"}
        assert field_hint("corpus.rerank_model", service) is None

class TestRerankTestUsesTheRerankEndpoint:
    """A chat greeting is not a connection test for a rerank source."""

    _SAVED = {
        "provider_type": "openai_compatible",
        "base_url": "http://engine:4000/v1",
        "api_key": "sk-saved",
        "model": "saved-chat-model",
    }

    @patch("esdc.config_wizard._test_provider")
    @patch("esdc.config_wizard.ProviderReranker")
    @patch("esdc.config_wizard.Config.set_config_value")
    @patch("esdc.config_wizard.Config.set_config_values")
    @patch("esdc.config_wizard.Config.get_referenced_provider")
    @patch("esdc.config_wizard.rich_print")
    @patch("esdc.config_wizard.questionary.select")
    def test_rerank_pair_test_calls_the_rerank_endpoint(
        self,
        mock_select,
        mock_print,
        mock_ref,
        mock_values,
        mock_set_value,
        mock_reranker,
        mock_test_provider,
    ):
        saved = dict(self._SAVED)
        mock_ref.return_value = dict(saved)
        mock_reranker.return_value.rerank.return_value = [0.4, 0.9]
        mock_select.return_value.unsafe_ask.side_effect = ["test", "cancel"]

        _run_task_action("rerank", PairSelection("engine", "rerank-model"))

        # The client is built the way runtime builds it: the saved connection and the
        # picked model, never the provider's chat model.
        assert mock_reranker.call_args.args == (saved, "rerank-model")
        mock_reranker.return_value.rerank.assert_called_once_with(
            "esdc rerank connection test",
            ["first test document", "second test document"],
        )
        # A chat success would not prove reranking works, so no chat request is made.
        mock_test_provider.assert_not_called()
        # Testing never writes, and the enable switch is not part of a test.
        mock_values.assert_not_called()
        mock_set_value.assert_not_called()
        assert saved["model"] == "saved-chat-model"

        printed = " ".join(str(call) for call in mock_print.call_args_list)
        assert "Success" in printed
        assert "2 scores" in printed

    @patch("esdc.config_wizard._test_provider")
    @patch("esdc.config_wizard.ProviderReranker")
    @patch("esdc.config_wizard.Config.set_config_value")
    @patch("esdc.config_wizard.Config.set_config_values")
    @patch("esdc.config_wizard.Config.get_referenced_provider")
    @patch("esdc.config_wizard.rich_print")
    def test_failed_rerank_call_is_reported_and_writes_nothing(
        self,
        mock_print,
        mock_ref,
        mock_values,
        mock_set_value,
        mock_reranker,
        mock_test_provider,
    ):
        mock_ref.return_value = dict(self._SAVED)
        class _StatusError(RuntimeError):
            status_code = 404

        failure = _StatusError("SECRET RERANK BODY at http://engine:4000/v1/rerank")
        mock_reranker.return_value.rerank.side_effect = failure

        _test_task_selection("rerank", PairSelection("engine", "rerank-model"))

        printed = " ".join(str(call) for call in mock_print.call_args_list)
        assert "Failed:" in printed
        assert "HTTP 404" in printed
        assert "SECRET" not in printed
        assert "RERANK BODY" not in printed
        assert "/v1/rerank" not in printed
        mock_test_provider.assert_not_called()
        mock_values.assert_not_called()
        mock_set_value.assert_not_called()

    @patch("esdc.config_wizard._test_provider")
    @patch("esdc.config_wizard.ProviderReranker")
    @patch("esdc.config_wizard.Config.set_config_value")
    @patch("esdc.config_wizard.Config.set_config_values")
    @patch("esdc.config_wizard.Config.get_referenced_provider")
    @patch("esdc.config_wizard.rich_print")
    def test_an_incomplete_rerank_result_is_a_failure(
        self,
        mock_print,
        mock_ref,
        mock_values,
        mock_set_value,
        mock_reranker,
        mock_test_provider,
    ):
        mock_ref.return_value = dict(self._SAVED)
        # One score for two documents, so the result is not a complete answer.
        mock_reranker.return_value.rerank.return_value = [0.4]

        _test_task_selection("rerank", PairSelection("engine", "rerank-model"))

        printed = " ".join(str(call) for call in mock_print.call_args_list)
        assert "incomplete result" in printed
        mock_test_provider.assert_not_called()
        mock_values.assert_not_called()
        mock_set_value.assert_not_called()

    @patch("esdc.config_wizard._test_provider")
    @patch("esdc.config_wizard.ProviderReranker")
    @patch("esdc.config_wizard.Config.set_config_values")
    @patch("esdc.config_wizard.Config.get_referenced_provider")
    @patch("esdc.config_wizard.rich_print")
    def test_a_provider_of_another_type_is_reported_without_a_chat_test(
        self, mock_print, mock_ref, mock_values, mock_reranker, mock_test_provider
    ):
        mock_ref.return_value = dict(_FAKE_PROVIDERS["gpu"])

        _test_task_selection("rerank", PairSelection("gpu", "ollama-a"))

        printed = " ".join(str(call) for call in mock_print.call_args_list)
        assert "cannot be a reranking reference" in printed
        mock_reranker.assert_not_called()
        mock_test_provider.assert_not_called()
        mock_values.assert_not_called()

    @patch("esdc.config_wizard._test_provider")
    @patch("esdc.config_wizard.ProviderReranker")
    @patch("esdc.config_wizard.Config.set_config_values")
    @patch("esdc.config_wizard.Config.get_referenced_provider")
    @patch("esdc.config_wizard.rich_print")
    def test_an_empty_model_is_reported_without_a_chat_test(
        self, mock_print, mock_ref, mock_values, mock_reranker, mock_test_provider
    ):
        saved = dict(self._SAVED)
        saved["model"] = ""
        mock_ref.return_value = saved

        # A linked row follows the provider's saved model, so an empty one cannot be
        # tested; the same holds for a pair with no model.
        _test_task_selection("rerank", LinkedProviderSelection("engine"))

        printed = " ".join(str(call) for call in mock_print.call_args_list)
        assert "needs a model" in printed
        mock_reranker.assert_not_called()
        mock_test_provider.assert_not_called()
        mock_values.assert_not_called()

    @patch("esdc.config_wizard._test_provider")
    @patch("esdc.config_wizard.ProviderReranker")
    @patch("esdc.config_wizard.Config.set_config_values")
    @patch("esdc.config_wizard.rich_print")
    def test_retired_rerank_status_test_loads_nothing_and_writes_nothing(
        self,
        mock_print,
        mock_values,
        mock_reranker,
        mock_test_provider,
    ):
        """A status has no endpoint: the test reports it and stops."""
        _test_task_selection(
            "rerank", NeedsConfigurationSelection(_RETIRED_RERANK_REASON)
        )

        printed = " ".join(str(call) for call in mock_print.call_args_list)
        assert (
            "Nothing to test: retired local rerank backend "
            "— choose a rerank service." in printed
        )
        # No client is built, no chat request is made, nothing is written, and the
        # stored value is left as it is.
        mock_reranker.assert_not_called()
        mock_test_provider.assert_not_called()
        mock_values.assert_not_called()


class TestExplicitReferenceWinsForTextTasks:
    """A saved task reference is read before the model string's legacy meanings.

    Runtime resolves an explicit `corpus.<role>_provider` first, so `main`,
    `provider:<name>`, a bare Ollama name, or an empty model is that provider's wire
    model. The chat-default, linked, off, and local readings apply only when no
    reference is saved.
    """

    _METADATA_MAIN = {
        "corpus.metadata_provider": "engine",
        "corpus.metadata_model": "main",
    }
    _CLEANUP_LINKED_STRING = {
        "corpus.cleanup_provider": "engine",
        "corpus.cleanup_model": "provider:other",
    }
    _EMPTY_MODEL = {
        "corpus.metadata_provider": "engine",
        "corpus.metadata_model": "",
    }
    _NO_REFERENCE = {
        "corpus.metadata_provider": "",
        "corpus.metadata_model": "main",
    }

    @patch(
        "esdc.config_wizard.Config.get_referenced_provider",
        return_value=dict(_FAKE_PROVIDERS["engine"]),
    )
    @patch("esdc.config_wizard.Config.set_config_values")
    @patch(
        "esdc.config_wizard.Config.get_all_config_flat",
        return_value=dict(_METADATA_MAIN),
    )
    @patch(
        "esdc.config_wizard.Config.get_providers", return_value=dict(_FAKE_PROVIDERS)
    )
    def test_wire_id_main_behind_a_reference_is_a_pair(
        self, mock_providers, mock_flat, mock_values, mock_ref
    ):
        current = _current_task_selection("metadata")
        assert current is not None
        assert current == PairSelection("engine", "main")
        assert not isinstance(current, ChatDefaultSelection)
        assert "engine / main" in _task_summary("metadata", current)

        # Save writes the pair as it was read: the reference is kept, not cleared.
        _save_task_selection("metadata", current)
        assert mock_values.call_args.args[0] == {
            "corpus.metadata_provider": "engine",
            "corpus.metadata_model": "main",
        }

    @patch(
        "esdc.config_wizard.Config.get_referenced_provider",
        return_value=dict(_FAKE_PROVIDERS["engine"]),
    )
    @patch("esdc.config_wizard.Config.set_config_values")
    @patch(
        "esdc.config_wizard.Config.get_all_config_flat",
        return_value=dict(_CLEANUP_LINKED_STRING),
    )
    @patch(
        "esdc.config_wizard.Config.get_providers", return_value=dict(_FAKE_PROVIDERS)
    )
    def test_provider_prefix_string_behind_a_reference_is_a_pair(
        self, mock_providers, mock_flat, mock_values, mock_ref
    ):
        current = _current_task_selection("cleanup")
        assert current is not None
        assert current == PairSelection("engine", "provider:other")
        assert not isinstance(current, LinkedProviderSelection)
        assert "engine / provider:other" in _task_summary("cleanup", current)

        # The prefix is part of the wire model here, so it is written back unchanged.
        _save_task_selection("cleanup", current)
        assert mock_values.call_args.args[0] == {
            "corpus.cleanup_provider": "engine",
            "corpus.cleanup_model": "provider:other",
        }

    @patch(
        "esdc.config_wizard.Config.get_all_config_flat",
        return_value=dict(_EMPTY_MODEL),
    )
    @patch(
        "esdc.config_wizard.Config.get_providers", return_value=dict(_FAKE_PROVIDERS)
    )
    def test_empty_model_behind_a_reference_is_not_read_as_off(
        self, mock_providers, mock_flat
    ):
        current = _current_task_selection("metadata")
        assert current is not None
        assert current == PairSelection("engine", "")
        assert not isinstance(current, OffSelection)

        # A blank pair names no model, so it is not a row to choose; the saved state
        # is still the reference, and the fallback rows stay where they belong.
        assert not _selectable(current)

    @patch(
        "esdc.config_wizard.Config.get_all_config_flat",
        return_value=dict(_NO_REFERENCE),
    )
    @patch(
        "esdc.config_wizard.Config.get_providers", return_value=dict(_FAKE_PROVIDERS)
    )
    def test_legacy_meanings_stay_without_a_reference(self, mock_providers, mock_flat):
        assert _current_task_selection("metadata") == ChatDefaultSelection()

        for model, expected in (
            ("provider:engine", LinkedProviderSelection("engine")),
            ("", OffSelection()),
            ("qwen3:8b", LocalSelection("qwen3:8b")),
        ):
            flat = {"corpus.metadata_provider": "", "corpus.metadata_model": model}
            with patch(
                "esdc.config_wizard.Config.get_all_config_flat", return_value=flat
            ):
                assert _current_task_selection("metadata") == expected

    @patch("esdc.config_wizard.discover_models", return_value=_catalog())
    @patch(
        "esdc.config_wizard.Config.get_all_config_flat",
        return_value=dict(_METADATA_MAIN),
    )
    @patch(
        "esdc.config_wizard.Config.get_providers", return_value=dict(_FAKE_PROVIDERS)
    )
    @patch("esdc.config_wizard.questionary.select")
    def test_saved_pair_stays_visible_when_discovery_is_unavailable(
        self, mock_select, mock_providers, mock_flat, mock_discover
    ):
        mock_select.return_value.unsafe_ask.side_effect = [None]
        _choose_provider_model("metadata", None)

        labels = _row_labels(mock_select)
        # Discovery reported nothing, so the saved pair remains as an unverified row.
        assert "engine / main (saved, unverified)" in labels
        assert "Chat setting / main" in labels

        values = [choice.value for choice in mock_select.call_args.kwargs["choices"]]
        # The chat default is a row of its own here, not the saved selection.
        assert PairSelection("engine", "main") in values
        assert ChatDefaultSelection() in values

    @patch("esdc.config_wizard.discover_models", return_value=_catalog())
    @patch(
        "esdc.config_wizard.Config.get_all_config_flat",
        return_value=dict(_CLEANUP_LINKED_STRING),
    )
    @patch(
        "esdc.config_wizard.Config.get_providers", return_value=dict(_FAKE_PROVIDERS)
    )
    @patch("esdc.config_wizard.questionary.select")
    def test_prefix_pair_stays_visible_when_discovery_is_unavailable(
        self, mock_select, mock_providers, mock_flat, mock_discover
    ):
        mock_select.return_value.unsafe_ask.side_effect = [None]
        _choose_provider_model("cleanup", None)

        labels = _row_labels(mock_select)
        # A wire model with the legacy prefix is still that provider's model, so the
        # saved pair keeps its row even when discovery reports nothing.
        assert "engine / provider:other (saved, unverified)" in labels
        assert "Off" in labels


class TestOcrSavedPairFollowsVisionMetadata:
    """A saved OCR pair is judged by what discovery reported about that model.

    An entry the provider reports as non-vision has no usable OCR row, so the saved
    pair is not re-added. A known-vision entry stays a normal row, and a model
    discovery did not report stays selectable as the saved, unverified pair.
    """

    _PROVIDERS = {"engine": dict(_FAKE_PROVIDERS["engine"])}
    _ENTRIES = (
        ModelEntry("vision-yes", frozenset({"chat"}), True),
        ModelEntry("text-only", frozenset({"chat"}), False),
        ModelEntry("unknown-vision", frozenset({"chat"}), None),
    )
    _CATALOG = ModelCatalog(entries=_ENTRIES)

    def _pick(self, mock_select, model: str) -> tuple[list[str], list[Any]]:
        flat = {"corpus.ocr_provider": "engine", "corpus.ocr_model": model}
        mock_select.return_value.unsafe_ask.side_effect = [None]
        with patch("esdc.config_wizard.Config.get_all_config_flat", return_value=flat):
            _choose_provider_model("ocr", None)

        choices = mock_select.call_args.kwargs["choices"]
        titles = [str(choice.title) for choice in choices]
        values = [choice.value for choice in choices]
        return titles, values

    @patch("esdc.config_wizard.discover_models", return_value=_CATALOG)
    @patch(
        "esdc.config_wizard.Config.get_providers",
        return_value=dict(_PROVIDERS),
    )
    @patch("esdc.config_wizard.questionary.select")
    def test_explicit_non_vision_saved_pair_is_not_readded(
        self, mock_select, mock_providers, mock_discover
    ):
        labels, values = self._pick(mock_select, "text-only")
        # The model is known non-vision: no row for it, saved or unverified.
        assert "engine / text-only" not in labels
        assert "engine / text-only (saved, unverified)" not in labels
        assert PairSelection("engine", "text-only") not in values
        # The rows that can serve OCR stay.
        assert "engine / vision-yes" in labels
        assert "engine / unknown-vision" in labels

    @patch("esdc.config_wizard.discover_models", return_value=_CATALOG)
    @patch(
        "esdc.config_wizard.Config.get_providers",
        return_value=dict(_PROVIDERS),
    )
    @patch("esdc.config_wizard.questionary.select")
    def test_known_vision_saved_pair_stays_a_normal_row(
        self, mock_select, mock_providers, mock_discover
    ):
        labels, values = self._pick(mock_select, "vision-yes")
        assert "engine / vision-yes" in labels
        assert "engine / vision-yes (saved, unverified)" not in labels
        assert PairSelection("engine", "vision-yes") in values

    @patch("esdc.config_wizard.discover_models", return_value=_CATALOG)
    @patch(
        "esdc.config_wizard.Config.get_providers",
        return_value=dict(_PROVIDERS),
    )
    @patch("esdc.config_wizard.questionary.select")
    def test_unknown_vision_saved_pair_stays_selectable(
        self, mock_select, mock_providers, mock_discover
    ):
        labels, values = self._pick(mock_select, "unknown-vision")
        # Reported but unverified metadata is still a row: unknown is not a refusal.
        assert "engine / unknown-vision" in labels
        assert PairSelection("engine", "unknown-vision") in values

    @patch(
        "esdc.config_wizard.discover_models",
        return_value=ModelCatalog(entries=()),
    )
    @patch(
        "esdc.config_wizard.Config.get_providers",
        return_value=dict(_PROVIDERS),
    )
    @patch("esdc.config_wizard.questionary.select")
    def test_saved_pair_missing_from_discovery_stays_unverified(
        self, mock_select, mock_providers, mock_discover
    ):
        labels, values = self._pick(mock_select, "text-only")
        assert "engine / text-only (saved, unverified)" in labels
        assert PairSelection("engine", "text-only") in values

    @patch(
        "esdc.config_wizard.discover_models",
        return_value=ModelCatalog(
            entries=(ModelEntry("text-only", frozenset({"chat"}), False),)
        ),
    )
    @patch(
        "esdc.config_wizard.Config.get_providers",
        return_value=dict(_PROVIDERS),
    )
    @patch("esdc.config_wizard.questionary.select")
    def test_the_vision_row_rule_applies_to_ocr_only(
        self, mock_select, mock_providers, mock_discover
    ):
        flat = {
            "corpus.metadata_provider": "engine",
            "corpus.metadata_model": "text-only",
        }
        mock_select.return_value.unsafe_ask.side_effect = [None]
        with patch("esdc.config_wizard.Config.get_all_config_flat", return_value=flat):
            _choose_provider_model("metadata", None)

        choices = mock_select.call_args.kwargs["choices"]
        assert "engine / text-only" in _row_labels(mock_select)
        assert PairSelection("engine", "text-only") in [
            choice.value for choice in choices
        ]


class TestOcrSaveChecksTheVisionFlag:
    """OCR Save refuses an explicitly non-vision model and writes nothing."""

    _PROVIDERS = {"engine": dict(_FAKE_PROVIDERS["engine"])}
    _FLAT = {"corpus.ollama_host": "http://gpu-box:11434"}

    def _catalog(self, model: str, vision: bool | None) -> ModelCatalog:
        return ModelCatalog(entries=(ModelEntry(model, frozenset({"chat"}), vision),))

    @patch("esdc.config_wizard.Config.get_providers", return_value=dict(_PROVIDERS))
    @patch(
        "esdc.config_wizard.Config.get_referenced_provider",
        return_value=dict(_FAKE_PROVIDERS["engine"]),
    )
    @patch("esdc.config_wizard.Config.get_all_config_flat", return_value=dict(_FLAT))
    @patch("esdc.config_wizard.Config.set_config_values")
    @patch("esdc.config_wizard.discover_models")
    def test_explicit_non_vision_pair_is_refused(
        self, mock_discover, mock_values, mock_flat, mock_ref, mock_providers
    ):
        mock_discover.return_value = self._catalog("text-only", False)

        with pytest.raises(ValueError, match="corpus.ocr_model"):
            _save_task_selection("ocr", PairSelection("engine", "text-only"))

        mock_values.assert_not_called()

    @patch("esdc.config_wizard.Config.get_providers", return_value=dict(_PROVIDERS))
    @patch(
        "esdc.config_wizard.Config.get_referenced_provider",
        return_value=dict(_FAKE_PROVIDERS["engine"]),
    )
    @patch("esdc.config_wizard.Config.get_all_config_flat", return_value=dict(_FLAT))
    @patch("esdc.config_wizard.Config.set_config_values")
    @patch("esdc.config_wizard.discover_models")
    def test_known_vision_pair_writes_once(
        self, mock_discover, mock_values, mock_flat, mock_ref, mock_providers
    ):
        mock_discover.return_value = self._catalog("vision-yes", True)

        _save_task_selection("ocr", PairSelection("engine", "vision-yes"))

        mock_values.assert_called_once_with(
            {"corpus.ocr_provider": "engine", "corpus.ocr_model": "vision-yes"}
        )

    @patch("esdc.config_wizard.Config.get_providers", return_value=dict(_PROVIDERS))
    @patch(
        "esdc.config_wizard.Config.get_referenced_provider",
        return_value=dict(_FAKE_PROVIDERS["engine"]),
    )
    @patch("esdc.config_wizard.Config.get_all_config_flat", return_value=dict(_FLAT))
    @patch("esdc.config_wizard.Config.set_config_values")
    @patch("esdc.config_wizard.discover_models")
    def test_unknown_vision_pair_writes_once(
        self, mock_discover, mock_values, mock_flat, mock_ref, mock_providers
    ):
        # Unknown metadata, and a model discovery did not report, are both allowed.
        mock_discover.return_value = self._catalog("vision-unknown", None)
        _save_task_selection("ocr", PairSelection("engine", "vision-unknown"))
        assert mock_values.call_args.args[0] == {
            "corpus.ocr_provider": "engine",
            "corpus.ocr_model": "vision-unknown",
        }

        mock_values.reset_mock()
        mock_discover.return_value = ModelCatalog(entries=())
        _save_task_selection("ocr", PairSelection("engine", "not-reported"))
        assert mock_values.call_args.args[0] == {
            "corpus.ocr_provider": "engine",
            "corpus.ocr_model": "not-reported",
        }

    @patch("esdc.config_wizard.Config.get_providers", return_value=dict(_PROVIDERS))
    @patch("esdc.config_wizard.Config.get_all_config_flat", return_value=dict(_FLAT))
    @patch("esdc.config_wizard.Config.set_config_values")
    @patch("esdc.config_wizard.discover_models")
    def test_local_ocr_is_checked_against_the_ollama_host_provider(
        self, mock_discover, mock_values, mock_flat, mock_ref
    ):
        mock_discover.return_value = self._catalog("text-only", False)

        with pytest.raises(ValueError, match="corpus.ocr_model"):
            _save_task_selection("ocr", LocalSelection("text-only"))

        mock_values.assert_not_called()
        # Local OCR has no saved provider, so the synthetic Ollama provider is used.
        assert mock_discover.call_args.args[0] == {
            "provider_type": "ollama",
            "base_url": "http://gpu-box:11434",
        }

    @patch("esdc.config_wizard.Config.get_providers", return_value=dict(_PROVIDERS))
    @patch("esdc.config_wizard.Config.get_all_config_flat", return_value=dict(_FLAT))
    @patch("esdc.config_wizard.Config.set_config_values")
    @patch("esdc.config_wizard.discover_models")
    def test_local_ocr_writes_once_for_known_or_unknown_vision(
        self, mock_discover, mock_values, mock_flat, mock_ref
    ):
        mock_discover.return_value = self._catalog("vision-yes", True)
        _save_task_selection("ocr", LocalSelection("vision-yes"))
        assert mock_values.call_args.args[0] == {
            "corpus.ocr_provider": "",
            "corpus.ocr_model": "vision-yes",
        }

        mock_values.reset_mock()
        mock_discover.return_value = self._catalog("vision-unknown", None)
        _save_task_selection("ocr", LocalSelection("vision-unknown"))
        assert mock_values.call_args.args[0] == {
            "corpus.ocr_provider": "",
            "corpus.ocr_model": "vision-unknown",
        }

    @patch("esdc.config_wizard.Config.get_providers", return_value=dict(_PROVIDERS))
    @patch("esdc.config_wizard.Config.get_all_config_flat", return_value=dict(_FLAT))
    @patch("esdc.config_wizard.rich_print")
    @patch("esdc.config_wizard.Config.set_config_values")
    @patch("esdc.config_wizard.discover_models")
    def test_a_failed_discovery_never_blocks_the_save(
        self, mock_discover, mock_values, mock_print, mock_flat, mock_ref
    ):
        mock_discover.side_effect = RuntimeError("SECRET MODEL LIST BODY")

        _save_task_selection("ocr", LocalSelection("vision-x"))

        printed = " ".join(str(call) for call in mock_print.call_args_list)
        assert "OCR vision metadata" in printed
        assert "RuntimeError" in printed
        assert "SECRET" not in printed
        assert "MODEL LIST BODY" not in printed
        mock_values.assert_called_once_with(
            {"corpus.ocr_provider": "", "corpus.ocr_model": "vision-x"}
        )

    @patch("esdc.config_wizard.Config.get_providers", return_value=dict(_PROVIDERS))
    @patch(
        "esdc.config_wizard.Config.get_referenced_provider",
        return_value=dict(_FAKE_PROVIDERS["engine"]),
    )
    @patch("esdc.config_wizard.Config.get_all_config_flat", return_value=dict(_FLAT))
    @patch("esdc.config_wizard.rich_print")
    @patch("esdc.config_wizard.Config.set_config_values")
    @patch("esdc.config_wizard.discover_models")
    def test_a_catalog_warning_warns_and_stays_unknown(
        self,
        mock_discover,
        mock_values,
        mock_print,
        mock_flat,
        mock_ref,
        mock_providers,
    ):
        # `discover_models` reports a network/HTTP/shape failure as a warning on the
        # catalog rather than raising: one short line, and the save is still allowed.
        mock_discover.return_value = ModelCatalog(
            entries=(),
            warning="Role metadata unavailable (HTTP 403); models are unverified.",
        )

        _save_task_selection("ocr", PairSelection("engine", "vision-x"))

        assert mock_print.call_count == 1
        printed = " ".join(str(call) for call in mock_print.call_args_list)
        assert "OCR vision metadata" in printed
        assert "HTTP 403" in printed
        mock_values.assert_called_once_with(
            {"corpus.ocr_provider": "engine", "corpus.ocr_model": "vision-x"}
        )

    @patch(
        "esdc.config_wizard.discover_models",
        return_value=ModelCatalog(
            entries=(ModelEntry("vision-x", frozenset({"chat"}), False),),
            warning="Role metadata unavailable (HTTP 403); models are unverified.",
        ),
    )
    @patch("esdc.config_wizard.Config.get_providers", return_value=dict(_PROVIDERS))
    @patch(
        "esdc.config_wizard.Config.get_referenced_provider",
        return_value=dict(_FAKE_PROVIDERS["engine"]),
    )
    @patch("esdc.config_wizard.Config.get_all_config_flat", return_value=dict(_FLAT))
    @patch("esdc.config_wizard.rich_print")
    @patch("esdc.config_wizard.Config.set_config_values")
    def test_an_explicit_false_still_blocks_even_with_a_catalog_warning(
        self,
        mock_values,
        mock_print,
        mock_flat,
        mock_ref,
        mock_providers,
        mock_discover,
    ):
        with pytest.raises(ValueError, match="corpus.ocr_model"):
            _save_task_selection("ocr", PairSelection("engine", "vision-x"))

        mock_values.assert_not_called()
        # The warning is reported, and the refusal is the model's own flag.
        assert mock_print.call_count == 1
        assert "HTTP 403" in " ".join(str(call) for call in mock_print.call_args_list)

    @patch("esdc.config_wizard.Config.get_referenced_provider")
    @patch("esdc.config_wizard.Config.set_config_values")
    @patch("esdc.config_wizard.discover_models")
    def test_a_non_ocr_save_does_not_call_discovery(
        self, mock_discover, mock_values, mock_ref
    ):
        mock_ref.return_value = dict(_FAKE_PROVIDERS["engine"])
        _save_task_selection("metadata", PairSelection("engine", "text-only"))

        mock_discover.assert_not_called()
        mock_values.assert_called_once_with(
            {"corpus.metadata_provider": "engine", "corpus.metadata_model": "text-only"}
        )


class TestOcrTestSendsItsOwnImage:
    """OCR Test sends the wizard's own image, so an endpoint that ignores it fails.

    The image is generated in memory from the wizard's test text; no document image is
    submitted, nothing is written, and a failure is reported for this test only.
    """

    _PROMPT = "Return the text in this image exactly."
    _PAIR = PairSelection("engine", "vision-x")
    _ANSWER = "ESDC OCR TEST 12345"

    def _client(self, answer, healthy=True) -> MagicMock:
        client = MagicMock()
        client.health_check.return_value = healthy
        client.query_image.return_value = answer
        return client

    def _run(self, mock_print, mock_values, client: MagicMock, selection=None) -> str:
        with (
            patch("esdc.config_wizard.create_ocr_client", return_value=client),
            patch("esdc.config_wizard.Config.get_corpus_config", return_value={}),
        ):
            _test_task_selection("ocr", selection or self._PAIR)

        return " ".join(str(call) for call in mock_print.call_args_list)

    @patch("esdc.config_wizard.rich_print")
    @patch("esdc.config_wizard.Config.set_config_values")
    def test_sends_a_generated_png_and_the_test_prompt(self, mock_values, mock_print):
        client = self._client(self._ANSWER)
        printed = self._run(mock_print, mock_values, client)

        client.query_image.assert_called_once()
        png, prompt = client.query_image.call_args.args
        assert png.startswith(b"\x89PNG")
        assert prompt == self._PROMPT
        # The reply came from the text the wizard drew, so the endpoint read the image.
        assert "passed" in printed
        mock_values.assert_not_called()

    @patch("esdc.config_wizard.rich_print")
    @patch("esdc.config_wizard.Config.set_config_values")
    def test_the_only_image_is_the_generated_one(self, mock_values, mock_print):
        import esdc.config_wizard as wizard

        client = self._client(self._ANSWER)
        self._run(mock_print, mock_values, client)

        calls = client.query_image.call_args_list
        assert len(calls) == 1
        png, _ = calls[0].args
        # Byte for byte the image the wizard drew, never a page from a document.
        assert png == wizard._ocr_test_image()
        assert len(png) < 200_000

    @pytest.mark.parametrize("answer", ["12345", "esdc ocr test 12345", "ESDC"])
    @patch("esdc.config_wizard.rich_print")
    @patch("esdc.config_wizard.Config.set_config_values")
    def test_a_reply_that_reads_the_text_passes(self, mock_values, mock_print, answer):
        printed = self._run(mock_print, mock_values, self._client(answer))
        assert "passed" in printed

    @pytest.mark.parametrize("answer", ["", "   ", "the document is unreadable"])
    @patch("esdc.config_wizard.rich_print")
    @patch("esdc.config_wizard.Config.set_config_values")
    def test_an_empty_or_generic_reply_fails(self, mock_values, mock_print, answer):
        printed = self._run(mock_print, mock_values, self._client(answer))
        assert "Failed:" in printed
        assert "passed" not in printed
        # The failure line carries the outcome only, never an excerpt of the reply.
        assert "(got:" not in printed
        reply = answer.strip()
        if reply:
            assert reply not in printed
        mock_values.assert_not_called()

    @patch("esdc.config_wizard.rich_print")
    @patch("esdc.config_wizard.Config.set_config_values")
    def test_a_reply_holding_a_secret_is_not_echoed(self, mock_values, mock_print):
        printed = self._run(
            mock_print, mock_values, self._client("SECRET MODEL OUTPUT 98765")
        )

        # Model output is not printed back: a reply can hold anything the model was
        # shown, so only the pass or fail outcome reaches the terminal.
        assert "Failed:" in printed
        assert "did not read the test image" in printed
        assert "passed" not in printed
        assert "(got:" not in printed
        assert "SECRET" not in printed
        assert "MODEL OUTPUT" not in printed
        assert "98765" not in printed
        mock_values.assert_not_called()

    @patch("esdc.config_wizard.rich_print")
    @patch("esdc.config_wizard.Config.set_config_values")
    def test_an_unavailable_endpoint_stops_before_the_image_is_sent(
        self, mock_values, mock_print
    ):
        client = self._client(self._ANSWER, healthy=False)
        printed = self._run(mock_print, mock_values, client)

        client.health_check.assert_called_once()
        client.query_image.assert_not_called()
        assert "not available" in printed
        mock_values.assert_not_called()

    @pytest.mark.parametrize(
        "error",
        [RuntimeError("connection reset"), ValueError("not a PNG")],
    )
    @patch("esdc.config_wizard.rich_print")
    @patch("esdc.config_wizard.Config.set_config_values")
    def test_a_request_failure_is_reported_without_writing(
        self, mock_values, mock_print, error
    ):
        client = MagicMock()
        client.health_check.return_value = True
        client.query_image.side_effect = error
        printed = self._run(mock_print, mock_values, client)

        assert "Failed:" in printed
        mock_values.assert_not_called()

    @patch("esdc.config_wizard.rich_print")
    @patch("esdc.config_wizard.Config.set_config_values")
    def test_a_400_adds_a_safe_vision_hint_without_echoing_the_body(
        self, mock_values, mock_print
    ):
        class _BadRequestError(RuntimeError):
            status_code = 400

        client = self._client(self._ANSWER)
        client.query_image.side_effect = _BadRequestError(
            "SECRET SERVER BODY at http://gpu-box:11434"
        )

        printed = self._run(mock_print, mock_values, client)

        assert "Failed:" in printed
        assert "HTTP 400" in printed
        assert "may not support vision" in printed
        assert "SECRET" not in printed
        assert "SERVER BODY" not in printed
        assert "gpu-box" not in printed
        mock_values.assert_not_called()

    @patch("esdc.config_wizard.rich_print")
    @patch("esdc.config_wizard.Config.set_config_values")
    def test_a_500_does_not_claim_a_vision_failure(self, mock_values, mock_print):
        class _ServerError(RuntimeError):
            status_code = 500

        client = self._client(self._ANSWER)
        client.query_image.side_effect = _ServerError("SECRET SERVER BODY")

        printed = self._run(mock_print, mock_values, client)

        assert "Failed:" in printed
        assert "HTTP 500" in printed
        assert "may not support vision" not in printed
        assert "SECRET" not in printed
        mock_values.assert_not_called()

    @patch("esdc.config_wizard._test_provider")
    @patch("esdc.config_wizard._save_task_selection")
    @patch("esdc.config_wizard.rich_print")
    @patch("esdc.config_wizard.Config.set_config_values")
    @patch("esdc.config_wizard.questionary.select")
    def test_the_test_writes_and_enables_nothing(
        self, mock_select, mock_values, mock_print, mock_save, mock_test_provider
    ):
        # Test keeps the action menu open, then Cancel closes it without a write.
        mock_select.return_value.unsafe_ask.side_effect = ["test", "cancel"]
        client = self._client(self._ANSWER)

        with (
            patch("esdc.config_wizard.create_ocr_client", return_value=client),
            patch("esdc.config_wizard.Config.get_corpus_config", return_value={}),
        ):
            _run_task_action("ocr", self._PAIR)

        # No enable switch, no save, and no fallback to a chat provider test.
        mock_values.assert_not_called()
        mock_save.assert_not_called()
        mock_test_provider.assert_not_called()


class TestShortTestMessageIsSafe:
    """A failed Test reports only a type or HTTP status, never exception text."""

    def test_a_message_is_not_echoed(self):
        class SecretError(Exception):
            pass

        error = SecretError("SECRET SERVER BODY at http://gateway/v1/ocr")

        assert _short_test_message(error) == "SecretError"

    def test_a_numeric_status_is_reported(self):
        class StatusError(Exception):
            status_code = 429

        error = StatusError("SECRET SERVER BODY at http://gateway/v1/ocr")

        assert _short_test_message(error) == "HTTP 429"

    def test_a_status_on_the_cause_is_reported(self):
        class CauseError(Exception):
            status_code = 503

        error = RuntimeError("SECRET SERVER BODY")
        error.__cause__ = CauseError("SECRET SERVER BODY")

        assert _short_test_message(error) == "HTTP 503"


class TestNativeOllamaOcrFailureDoesNotLeak:
    """The native Ollama client's ResponseError body never reaches the terminal."""

    class _FailingClient:
        def show(self, model):
            return {}

        def chat(self, **kwargs):
            raise ollama.ResponseError(
                "SECRET SERVER BODY at http://gpu-box:11434", status_code=500
            )

    @patch("esdc.config_wizard.Config.save_provider")
    @patch("esdc.config_wizard._test_provider")
    @patch("esdc.config_wizard.Config.set_config_value")
    @patch("esdc.config_wizard.Config.set_config_values")
    @patch("esdc.config_wizard.rich_print")
    def test_response_error_reports_failure_without_the_body(
        self, mock_print, mock_values, mock_set_value, mock_test_provider, mock_save
    ):
        # The real OllamaVisionOcr client, with a fake transport that raises the
        # native ResponseError. Health succeeds, so the image request is attempted.
        client = OllamaVisionOcr(
            model="glm-ocr", client=cast(ollama.Client, self._FailingClient())
        )

        with (
            patch("esdc.config_wizard.create_ocr_client", return_value=client),
            patch("esdc.config_wizard.Config.get_corpus_config", return_value={}),
        ):
            _test_task_selection("ocr", PairSelection("engine", "vision-x"))

        printed = " ".join(str(call) for call in mock_print.call_args_list)
        assert "Failed:" in printed
        assert "HTTP 500" in printed
        assert "SECRET" not in printed
        assert "SERVER BODY" not in printed
        assert "gpu-box" not in printed
        mock_values.assert_not_called()
        mock_set_value.assert_not_called()
        mock_test_provider.assert_not_called()
        mock_save.assert_not_called()


class TestEmbeddingTestFollowsRuntimeHostPrecedence:
    """The embedding test resolves its address the way runtime does.

    `corpus.ollama_host` belongs to OCR and to Ollama-named corpus models; it is never
    the address an embedding test uses. For a named provider the order is the
    environment host, then that provider's own address, then the saved embedding host,
    and a named provider with no address at all is reported instead of tested against a
    default host. A direct row keeps its own behavior: the environment host, else the
    host the row carries, and an empty one stays the embedder's default endpoint.
    """

    _FLAT = {
        "embedding_host": "http://embed-host:11434",
        "corpus.ollama_host": "http://corpus-host:11434",
    }
    _PAIR = PairSelection("gpu", "qwen3-embedding:0.6b")
    _DIRECT = DirectSelection(
        "ollama", "http://direct-box:1234", OllamaEmbedder.OLLAMA_TAG
    )

    @staticmethod
    def _printed(mock_print) -> str:
        return " ".join(str(call) for call in mock_print.call_args_list)

    @patch("esdc.config_wizard._test_provider")
    @patch("esdc.config_wizard.OpenAIEmbedder")
    @patch("esdc.config_wizard.OllamaEmbedder")
    @patch("esdc.config_wizard.Config.get_all_config_flat", return_value=dict(_FLAT))
    @patch("esdc.config_wizard.Config.get_referenced_provider")
    def test_named_ollama_provider_uses_the_saved_embedding_host(
        self, mock_ref, mock_flat, mock_ollama, mock_openai, mock_test_provider
    ):
        mock_ref.return_value = {"provider_type": "ollama", "model": "chat-a"}
        mock_ollama.return_value.generate_embedding.return_value = [0.1]

        _test_task_selection("embedding", self._PAIR)

        # The saved embedding host, not the corpus Ollama host.
        assert mock_ollama.call_args.kwargs == {
            "host": "http://embed-host:11434",
            "model": "qwen3-embedding:0.6b",
        }
        mock_openai.assert_not_called()
        mock_test_provider.assert_not_called()

    @patch("esdc.config_wizard._test_provider")
    @patch("esdc.config_wizard.OpenAIEmbedder")
    @patch("esdc.config_wizard.OllamaEmbedder")
    @patch("esdc.config_wizard.Config.get_all_config_flat", return_value=dict(_FLAT))
    @patch("esdc.config_wizard.Config.get_referenced_provider")
    def test_openai_provider_uses_the_saved_embedding_host_when_it_has_no_address(
        self, mock_ref, mock_flat, mock_ollama, mock_openai, mock_test_provider
    ):
        mock_ref.return_value = {
            "provider_type": "openai_compatible",
            "api_key": "sk-saved",
            "model": "embed-1",
        }
        mock_openai.return_value.generate_embedding.return_value = [0.1]

        _test_task_selection("embedding", PairSelection("engine", "embed-1"))

        assert mock_openai.call_args.kwargs == {
            "host": "http://embed-host:11434",
            "model": "embed-1",
            "api_key": "sk-saved",
        }
        mock_ollama.assert_not_called()
        mock_test_provider.assert_not_called()

    @patch("esdc.config_wizard._test_provider")
    @patch("esdc.config_wizard.OpenAIEmbedder")
    @patch("esdc.config_wizard.OllamaEmbedder")
    @patch("esdc.config_wizard.Config.get_all_config_flat", return_value=dict(_FLAT))
    @patch("esdc.config_wizard.Config.get_referenced_provider")
    def test_the_provider_address_beats_the_saved_embedding_host(
        self, mock_ref, mock_flat, mock_ollama, mock_openai, mock_test_provider
    ):
        mock_ref.return_value = {
            "provider_type": "ollama",
            "base_url": "http://gpu-box:11434",
            "model": "chat-a",
        }
        mock_ollama.return_value.generate_embedding.return_value = [0.1]

        _test_task_selection("embedding", self._PAIR)

        # Precedence: environment host, then this provider's address, then the
        # saved embedding host.
        assert mock_ollama.call_args.kwargs == {
            "host": "http://gpu-box:11434",
            "model": "qwen3-embedding:0.6b",
        }

    @patch("esdc.config_wizard._test_provider")
    @patch("esdc.config_wizard.OpenAIEmbedder")
    @patch("esdc.config_wizard.OllamaEmbedder")
    @patch("esdc.config_wizard.Config.get_all_config_flat", return_value=dict(_FLAT))
    @patch("esdc.config_wizard.Config.get_referenced_provider")
    def test_the_environment_host_overrides_both(
        self,
        mock_ref,
        mock_flat,
        mock_ollama,
        mock_openai,
        mock_test_provider,
        monkeypatch,
    ):
        mock_ref.return_value = {
            "provider_type": "ollama",
            "base_url": "http://gpu-box:11434",
            "model": "chat-a",
        }
        mock_ollama.return_value.generate_embedding.return_value = [0.1]
        monkeypatch.setenv("ESDC_EMBEDDING_HOST", "http://env-host:11434")

        _test_task_selection("embedding", self._PAIR)

        # The host is the environment override; the model is the one runtime will use.
        assert mock_ollama.call_args.kwargs == {
            "host": "http://env-host:11434",
            "model": "qwen3-embedding:0.6b",
        }

    @patch("esdc.config_wizard._test_provider")
    @patch("esdc.config_wizard.OpenAIEmbedder")
    @patch("esdc.config_wizard.OllamaEmbedder")
    @patch("esdc.config_wizard.Config.set_config_values")
    @patch(
        "esdc.config_wizard.Config.get_all_config_flat",
        return_value={"corpus.ollama_host": "http://corpus-host:11434"},
    )
    @patch("esdc.config_wizard.Config.get_referenced_provider")
    @patch("esdc.config_wizard.rich_print")
    def test_a_named_provider_with_no_address_is_reported_and_writes_nothing(
        self,
        mock_print,
        mock_ref,
        mock_flat,
        mock_values,
        mock_ollama,
        mock_openai,
        mock_test_provider,
    ):
        mock_ref.return_value = {"provider_type": "ollama", "model": "chat-a"}

        _test_task_selection("embedding", self._PAIR)

        # No address at all: nothing is constructed, and no default host is assumed.
        printed = self._printed(mock_print)
        assert "Failed:" in printed
        assert "embedding address" in printed
        assert "corpus-host" not in printed
        mock_ollama.assert_not_called()
        mock_openai.assert_not_called()
        mock_test_provider.assert_not_called()
        mock_values.assert_not_called()

    @patch("esdc.config_wizard.rich_print")
    @patch("esdc.config_wizard._test_provider")
    @patch("esdc.config_wizard.OpenAIEmbedder")
    @patch("esdc.config_wizard.OllamaEmbedder")
    @patch("esdc.config_wizard.Config.get_all_config_flat", return_value=dict(_FLAT))
    def test_a_direct_row_keeps_its_own_host(
        self, mock_flat, mock_ollama, mock_openai, mock_test_provider, mock_print
    ):
        mock_ollama.return_value.generate_embedding.return_value = [0.1]

        _test_task_selection("embedding", self._DIRECT)

        # A direct row names its own endpoint: neither embedding_host nor a provider
        # stands behind it, and the test still runs.
        assert mock_ollama.call_args.kwargs == {
            "host": "http://direct-box:1234",
            "model": OllamaEmbedder.OLLAMA_TAG,
        }
        assert "Failed:" not in self._printed(mock_print)
        mock_openai.assert_not_called()
        mock_test_provider.assert_not_called()

    @patch("esdc.config_wizard.Config.set_config_values")
    @patch("esdc.config_wizard._test_provider")
    @patch("esdc.config_wizard.OpenAIEmbedder")
    @patch("esdc.config_wizard.OllamaEmbedder")
    @patch("esdc.config_wizard.Config.get_all_config_flat", return_value=dict(_FLAT))
    @patch("esdc.config_wizard.rich_print")
    def test_a_direct_row_with_an_empty_host_keeps_the_local_default(
        self,
        mock_print,
        mock_flat,
        mock_ollama,
        mock_openai,
        mock_test_provider,
        mock_values,
    ):
        mock_ollama.return_value.generate_embedding.return_value = [0.1]

        _test_task_selection(
            "embedding", DirectSelection("ollama", "", OllamaEmbedder.OLLAMA_TAG)
        )

        # Nothing to fall back to here: the embedder keeps its own local default, so
        # neither embedding_host nor corpus.ollama_host is borrowed, and the row is not
        # reported as missing an address. The model asked for is the row's own.
        assert mock_ollama.call_args.kwargs == {
            "host": None,
            "model": OllamaEmbedder.OLLAMA_TAG,
        }
        mock_ollama.return_value.generate_embedding.assert_called_once_with(
            "esdc embedding connection test"
        )
        assert "Failed:" not in self._printed(mock_print)
        mock_openai.assert_not_called()
        mock_test_provider.assert_not_called()
        mock_values.assert_not_called()

    @patch("esdc.config_wizard.Config.set_config_values")
    @patch("esdc.config_wizard._test_provider")
    @patch("esdc.config_wizard.OpenAIEmbedder")
    @patch("esdc.config_wizard.OllamaEmbedder")
    @patch("esdc.config_wizard.Config.get_all_config_flat", return_value=dict(_FLAT))
    @patch("esdc.config_wizard.rich_print")
    def test_a_direct_openai_row_with_an_empty_host_uses_its_own_default(
        self,
        mock_print,
        mock_flat,
        mock_ollama,
        mock_openai,
        mock_test_provider,
        mock_values,
    ):
        mock_openai.return_value.generate_embedding.return_value = [0.1]

        _test_task_selection("embedding", DirectSelection("openai", "", "embed-1"))

        # An empty direct host is passed through as it was: the embedder's own default
        # endpoint, not the saved embedding host.
        assert mock_openai.call_args.kwargs == {
            "host": "",
            "model": "embed-1",
            "api_key": "",
        }
        assert "Failed:" not in self._printed(mock_print)
        mock_ollama.assert_not_called()
        mock_test_provider.assert_not_called()
        mock_values.assert_not_called()

    @patch("esdc.config_wizard._test_provider")
    @patch("esdc.config_wizard.OpenAIEmbedder")
    @patch("esdc.config_wizard.OllamaEmbedder")
    @patch("esdc.config_wizard.Config.get_all_config_flat", return_value=dict(_FLAT))
    def test_a_direct_row_still_takes_the_environment_override(
        self, mock_flat, mock_ollama, mock_openai, mock_test_provider, monkeypatch
    ):
        mock_ollama.return_value.generate_embedding.return_value = [0.1]
        monkeypatch.setenv("ESDC_EMBEDDING_HOST", "http://env-host:11434")

        _test_task_selection("embedding", self._DIRECT)

        # The environment host wins, and the model is the one the row names.
        assert mock_ollama.call_args.kwargs == {
            "host": "http://env-host:11434",
            "model": OllamaEmbedder.OLLAMA_TAG,
        }


def test_bad_ollama_provider_does_not_hide_healthy_task_rows(monkeypatch):
    """A malformed Ollama host must not crash the whole task picker."""
    import esdc.config_wizard as wizard
    from esdc.config_wizard import _CUSTOM_VALUE, _RETRY_VALUE, _task_rows

    providers = {
        "bad-ollama": {"provider_type": "ollama", "base_url": "http://["},
        "healthy": {
            "provider_type": "openai_compatible",
            "base_url": "http://engine:4000/v1",
            "api_key": "fixture-key",
        },
    }
    before = {name: dict(entry) for name, entry in providers.items()}
    monkeypatch.setattr(Config, "get_all_config_flat", lambda: {})
    monkeypatch.setattr(Config, "get_providers", lambda: providers)

    real_discover = wizard.discover_models

    def fake_discover(provider):
        if str(provider.get("provider_type")) == "ollama":
            return real_discover(provider)
        return ModelCatalog(
            entries=(ModelEntry("healthy-chat", roles=frozenset({"chat"})),)
        )

    monkeypatch.setattr(wizard, "discover_models", fake_discover)

    rows = _task_rows("chat", None, {})
    values = [row.value for row in rows]
    pairs = [value for value in values if isinstance(value, PairSelection)]

    assert PairSelection("healthy", "healthy-chat") in pairs
    assert all(pair.provider_name != "bad-ollama" for pair in pairs)
    assert _RETRY_VALUE in values
    assert _CUSTOM_VALUE in values
    assert providers == before


def _ocr_models(rows) -> list[str]:
    return [row.value.model_id for row in rows if isinstance(row.value, LocalSelection)]


def test_local_ocr_lists_corpus_host_vision_and_unknown_models(monkeypatch):
    import esdc.config_wizard as wizard
    from esdc.config_wizard import _task_rows

    monkeypatch.setattr(Config, "get_providers", lambda: {})
    monkeypatch.setattr(
        Config,
        "get_all_config_flat",
        lambda: {"corpus.ollama_host": "http://gpu-box:11434"},
    )
    seen: dict[str, Any] = {}

    def fake_discover(provider):
        seen.update(provider)
        return ModelCatalog(
            entries=(
                ModelEntry("vision-a", supports_vision=True),
                ModelEntry("unknown-b", supports_vision=None),
                ModelEntry("vision-named-text-c", supports_vision=False),
            )
        )

    monkeypatch.setattr(wizard, "discover_models", fake_discover)

    models = _ocr_models(_task_rows("ocr", None, {}))

    assert "vision-a" in models
    assert "unknown-b" in models
    assert "vision-named-text-c" not in models
    assert seen == {"provider_type": "ollama", "base_url": "http://gpu-box:11434"}


def test_local_ocr_discovery_preserves_saved_unverified_model(monkeypatch):
    import esdc.config_wizard as wizard
    from esdc.config_wizard import _task_rows

    monkeypatch.setattr(Config, "get_providers", lambda: {})
    monkeypatch.setattr(
        Config,
        "get_all_config_flat",
        lambda: {
            "corpus.ollama_host": "http://gpu-box:11434",
            "corpus.ocr_model": "saved-ocr",
        },
    )

    monkeypatch.setattr(
        wizard,
        "discover_models",
        lambda provider: ModelCatalog(
            entries=(ModelEntry("other", supports_vision=True),)
        ),
    )
    assert "saved-ocr" in _ocr_models(_task_rows("ocr", None, {}))

    monkeypatch.setattr(
        wizard,
        "discover_models",
        lambda provider: ModelCatalog(warning="Model discovery failed (timeout)."),
    )
    assert "saved-ocr" in _ocr_models(_task_rows("ocr", None, {}))

    monkeypatch.setattr(
        wizard,
        "discover_models",
        lambda provider: ModelCatalog(
            entries=(ModelEntry("saved-ocr", supports_vision=False),)
        ),
    )
    assert "saved-ocr" not in _ocr_models(_task_rows("ocr", None, {}))


def test_local_ocr_custom_model_without_saved_provider(monkeypatch):
    import esdc.config_wizard as wizard
    from esdc.config_wizard import _local_ocr_model_flow

    monkeypatch.setattr(Config, "get_providers", lambda: {})
    discover = MagicMock()
    monkeypatch.setattr(wizard, "discover_models", discover)
    mock_text = MagicMock()
    mock_text.return_value.unsafe_ask.return_value = "  custom-vision  "
    monkeypatch.setattr(wizard.questionary, "text", mock_text)

    result = _local_ocr_model_flow(None)

    assert result == LocalSelection("custom-vision")
    discover.assert_not_called()


def test_local_ocr_custom_cancel_and_blank_do_not_save(
    monkeypatch, tmp_path
):
    import esdc.config_wizard as wizard
    from esdc.config_wizard import _local_ocr_model_flow

    monkeypatch.setenv("ESDC_CONFIG_DIR", str(tmp_path))
    Config._config_cache = None
    Config.init_config()
    config_path = tmp_path / "config.yaml"
    before = config_path.read_bytes()
    save = MagicMock()
    monkeypatch.setattr(Config, "set_config_values", save)

    mock_text = MagicMock()
    mock_text.return_value.unsafe_ask.side_effect = ["", "   ", "model-x"]
    monkeypatch.setattr(wizard.questionary, "text", mock_text)

    assert _local_ocr_model_flow(None) == LocalSelection("model-x")
    assert mock_text.return_value.unsafe_ask.call_count == 3

    cancel_text = MagicMock()
    cancel_text.return_value.unsafe_ask.return_value = None
    monkeypatch.setattr(wizard.questionary, "text", cancel_text)
    assert _local_ocr_model_flow(None) is None
    assert config_path.read_bytes() == before
    save.assert_not_called()


def test_local_ocr_save_changes_only_role_source_and_model(monkeypatch):
    import esdc.config_wizard as wizard
    from esdc.config_wizard import _save_task_selection

    captured: dict[str, Any] = {}
    monkeypatch.setattr(Config, "set_config_values", lambda values: captured.update(values))
    monkeypatch.setattr(
        Config,
        "get_all_config_flat",
        lambda: {"corpus.ollama_host": "http://gpu-box:11434"},
    )
    monkeypatch.setattr(
        wizard,
        "discover_models",
        lambda provider: ModelCatalog(
            entries=(ModelEntry("glm-ocr", supports_vision=True),)
        ),
    )

    _save_task_selection("ocr", LocalSelection("glm-ocr"))

    assert captured == {
        "corpus.ocr_provider": "",
        "corpus.ocr_model": "glm-ocr",
    }


def test_local_ocr_test_uses_saved_corpus_host_without_saving_draft(monkeypatch):
    import esdc.config_wizard as wizard
    from esdc.config_wizard import _test_task_selection

    monkeypatch.setattr(
        Config,
        "get_corpus_config",
        lambda: {
            "ollama_host": "http://gpu-box:11434",
            "ocr_provider": "saved-other",
            "ocr_model": "saved-model",
            "extract_timeout_seconds": 5,
        },
    )
    captured: list[dict[str, Any]] = []

    class _FakeClient:
        def health_check(self) -> bool:
            return True

        def query_image(self, image, prompt) -> str:
            return "ESDC OCR TEST 12345"

    def fake_create_ocr_client(config, **kwargs):
        captured.append(dict(config))
        return _FakeClient()

    monkeypatch.setattr(wizard, "create_ocr_client", fake_create_ocr_client)
    save = MagicMock()
    monkeypatch.setattr(Config, "set_config_values", save)

    _test_task_selection("ocr", LocalSelection("draft-model"))

    assert len(captured) == 1
    assert captured[0]["ollama_host"] == "http://gpu-box:11434"
    assert captured[0]["ocr_provider"] == ""
    assert captured[0]["ocr_model"] == "draft-model"
    save.assert_not_called()


@pytest.mark.parametrize(
    ("key", "service_value"),
    [
        ("embedding_backend", "ollama"),
        ("corpus.rerank_backend", "provider"),
    ],
)
def test_retired_backend_editor_constructs_real_questionary_select(
    monkeypatch, tmp_path, key: str, service_value: str
):
    """The retired default must be a real choice value, not its display label."""
    import questionary

    import esdc.config_wizard as wizard

    monkeypatch.setenv("ESDC_CONFIG_DIR", str(tmp_path))
    Config._config_cache = None
    Config.init_config()
    config_path = tmp_path / "config.yaml"
    before = config_path.read_bytes()
    save = MagicMock()
    monkeypatch.setattr(Config, "set_config_values", save)

    captured: dict[str, Any] = {}
    real_select = questionary.select

    def recording_select(*args, **kwargs):
        captured.update(kwargs)
        return real_select(*args, **kwargs)

    monkeypatch.setattr(wizard.questionary, "select", recording_select)

    # The real constructor above runs before `_ask`: a label default raises there.
    monkeypatch.setattr(wizard, "_ask", lambda question: "local")
    assert wizard._prompt_for_config_value(key, "local", {key: "local"}) == "local"

    default = captured["default"]
    assert default == "local"
    assert any(choice.value == default for choice in captured["choices"])

    monkeypatch.setattr(wizard, "_ask", lambda question: None)
    assert wizard._prompt_for_config_value(key, "local", {key: "local"}) is None

    monkeypatch.setattr(wizard, "_ask", lambda question: service_value)
    assert (
        wizard._prompt_for_config_value(key, "local", {key: "local"})
        == service_value
    )

    assert config_path.read_bytes() == before
    save.assert_not_called()
