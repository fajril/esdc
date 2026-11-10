"""Tests for config_wizard module."""

import ast
from pathlib import Path
from unittest.mock import patch

from esdc.config_wizard import (
    _prompt_for_config_value,
    _short_test_message,
    run_wizard,
)
from esdc.configs import (
    ADVANCED_SECTIONS,
    APPLICATION_SECTIONS,
    KEY_DESCRIPTIONS,
    TASK_SECTIONS,
)


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
        """The generic editor has no model keys at all."""
        assert TASK_SECTIONS == {}
        all_keys = {
            k
            for keys in {
                **APPLICATION_SECTIONS,
                **ADVANCED_SECTIONS,
            }.values()
            for k in keys
        }
        for key in (
            "corpus.ocr_model",
            "corpus.metadata_model",
            "corpus.cleanup_model",
            "corpus.ollama_host",
            "embedding_backend",
            "corpus.rerank_backend",
        ):
            assert key not in all_keys, key
        assert "corpus.num_ctx" in ADVANCED_SECTIONS["Corpus processing"]
        assert (
            "semantic_search.embedding_batch_size"
            in ADVANCED_SECTIONS["Corpus processing"]
        )
        assert "tool_format" in ADVANCED_SECTIONS["Corpus processing"]



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
        """The scan reaches the connection screens, not only the task screens."""
        messages = {message for _, message in self._messages()}
        for expected in (
            "Select connection to edit:",
            "Select field to edit:",
            "API Key (blank keeps current):",
            "Clear API key",
            "Done editing",
        ):
            assert expected in messages


class TestMenuKeyCoverage:
    """Every editable generic key has exactly one menu location and a description."""

    def test_every_key_appears_exactly_once(self):
        sections = {
            **TASK_SECTIONS,
            **APPLICATION_SECTIONS,
            **ADVANCED_SECTIONS,
        }
        keys = [key for keys in sections.values() for key in keys]
        assert keys
        assert len(keys) == len(set(keys))

    def test_no_legacy_model_key_is_offered(self):
        sections = {
            **TASK_SECTIONS,
            **APPLICATION_SECTIONS,
            **ADVANCED_SECTIONS,
        }
        offered = {key for keys in sections.values() for key in keys}
        for key in (
            "embedding_backend",
            "embedding_host",
            "embedding_model",
            "embedding_api_key",
            "embedding_provider",
            "corpus.rerank",
            "corpus.rerank_backend",
            "corpus.rerank_provider",
            "corpus.rerank_model",
            "corpus.ocr_model",
            "corpus.ocr_provider",
            "corpus.metadata_model",
            "corpus.metadata_provider",
            "corpus.cleanup_model",
            "corpus.cleanup_provider",
            "corpus.ollama_host",
            "corpus.n_gpu_layers",
        ):
            assert key not in offered, key

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
