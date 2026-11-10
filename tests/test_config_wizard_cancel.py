"""Cancel handling in the config wizard.

Escape returns None from `_ask`, Control+C propagates as KeyboardInterrupt, and
a cancel never writes. The pipe-input tests drive real prompt_toolkit
applications through a pipe instead of a terminal, so the Escape binding and
Control+C are observed as they actually behave; the mock tests cover the flow
boundaries. No real terminal and no real secret.
"""

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import questionary
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys
from prompt_toolkit.output import DummyOutput

from esdc.config_wizard import (
    WizardCancelledError,
    _ask,
    _escape,
    _prompt_for_config_value,
    _select_with_back,
    run_wizard,
)


def _escape_binding(question):
    """The single Escape binding attached to one question, or None."""
    for binding in question.application.key_bindings.bindings:
        if getattr(binding, "handler", None) is _escape:
            return binding
    return None


class _FakeApp:
    """Just enough of an application to observe what the Escape binding does."""

    def __init__(self):
        self.key_bindings = KeyBindings()
        self.exited_with = "not-run"

    def exit(self, result=None):
        self.exited_with = result


class _FakeQuestion:
    """A question stand-in that dispatches the attached Escape binding."""

    def __init__(self, answer=None, keys=None, error=None):
        self.application = _FakeApp()
        self._answer = answer
        self._keys = keys
        self._error = error
        self.ask_calls = 0

    def ask(self):
        self.ask_calls += 1
        return "ask-was-used"

    def unsafe_ask(self):
        if self._error is not None:
            raise self._error
        if self._keys == "escape":
            binding = _escape_binding(self)
            assert binding is not None, "Escape binding was not attached"
            # The handler only needs an application, so a stand-in event is enough.
            event: Any = SimpleNamespace(app=self.application)
            binding.handler(event)
            return self.application.exited_with
        return self._answer


def _run_pipe(question, keys):
    """Run one real questionary prompt from a pipe input, with no terminal."""
    with create_pipe_input() as pipe:
        question.application.input = pipe
        question.application.output = DummyOutput()
        pipe.send_text(keys)
        return _ask(question)


class TestAsk:
    """`_ask` attaches one eager Escape binding and calls unsafe_ask()."""

    def test_escape_returns_none(self):
        assert _ask(_FakeQuestion(keys="escape")) is None

    def test_success_returns_the_value(self):
        assert _ask(_FakeQuestion(answer="openai")) == "openai"

    def test_keyboard_interrupt_propagates(self):
        with pytest.raises(KeyboardInterrupt):
            _ask(_FakeQuestion(error=KeyboardInterrupt()))

    def test_ask_is_never_used(self):
        question = _FakeQuestion(answer="openai")
        _ask(question)
        assert question.ask_calls == 0

    def test_escape_binding_is_eager_and_attached_once(self):
        question = _FakeQuestion(keys="escape")
        _ask(question)
        _ask(question)

        found = [
            binding
            for binding in question.application.key_bindings.bindings
            if getattr(binding, "handler", None) is _escape
        ]
        assert len(found) == 1
        assert Keys.Escape in found[0].keys
        # prompt_toolkit stores eager as a filter, so the binding is eager when
        # its filter evaluates True.
        assert found[0].eager() is True

    def test_wizard_cancelled_error_is_kept_for_compatibility(self):
        assert issubclass(WizardCancelledError, Exception)


class TestRealPipeInput:
    """Escape and Control+C on real prompt_toolkit applications, pipe-driven."""

    def test_select_escape_returns_none(self):
        question = questionary.select(
            "Pick:", choices=[questionary.Choice("a", value="a")]
        )
        assert _run_pipe(question, "\x1b") is None

    def test_select_control_c_raises(self):
        question = questionary.select(
            "Pick:", choices=[questionary.Choice("a", value="a")]
        )
        with pytest.raises(KeyboardInterrupt):
            _run_pipe(question, "\x03")

    def test_text_escape_returns_none(self):
        question = questionary.text("Name:")
        assert _run_pipe(question, "\x1b") is None

    def test_text_control_c_raises(self):
        question = questionary.text("Name:")
        with pytest.raises(KeyboardInterrupt):
            _run_pipe(question, "\x03")

    def test_pipe_tests_still_answer_normally(self):
        """The harness really runs the prompt: Enter answers, and Escape cancels."""
        select = questionary.select(
            "Pick:", choices=[questionary.Choice("a", value="a")]
        )
        assert _run_pipe(select, "\r") == "a"

        text = questionary.text("Name:")
        assert _run_pipe(text, "chat-a\r") == "chat-a"
        assert _run_pipe(text, "\x1b") is None


class TestSelectWithBack:
    """`_select_with_back` prepends Back and reads a cancel as Back."""

    @patch("esdc.config_wizard.questionary.select")
    def test_select_returns_value(self, mock_select):
        mock_select.return_value.unsafe_ask.return_value = "openai"
        result = _select_with_back("Pick:", [MagicMock()], default="openai")
        assert result == "openai"
        choices = mock_select.call_args.kwargs["choices"]
        assert choices[0].value == "__back__"

    @patch("esdc.config_wizard.questionary.select")
    def test_cancel_maps_to_back(self, mock_select):
        mock_select.return_value.unsafe_ask.return_value = None
        assert _select_with_back("Pick:", [MagicMock()]) == "__back__"


class TestPromptForConfigValueCancel:
    """When Escape is pressed, questionary returns None and the field is cancelled."""

    @patch("esdc.config_wizard.questionary.select")
    def test_boolean_cancel_returns_none(self, mock_select):
        mock_select.return_value.unsafe_ask.return_value = None
        assert _prompt_for_config_value("api.verify_ssl", True) is None

    @patch("esdc.config_wizard.questionary.select")
    def test_enum_cancel_returns_none(self, mock_select):
        mock_select.return_value.unsafe_ask.return_value = None
        assert _prompt_for_config_value("tool_format", "native") is None

    @patch("esdc.config_wizard.questionary.text")
    def test_integer_cancel_returns_none(self, mock_text):
        mock_text.return_value.unsafe_ask.return_value = None
        assert _prompt_for_config_value("cache.sql_ttl", 3600) is None

    @patch("esdc.config_wizard.questionary.password")
    def test_password_cancel_returns_none(self, mock_password):
        mock_password.return_value.unsafe_ask.return_value = None
        assert _prompt_for_config_value("providers.x.api_key", "secret") is None

    @patch("esdc.config_wizard.questionary.path")
    def test_path_cancel_returns_none(self, mock_path):
        mock_path.return_value.unsafe_ask.return_value = None
        assert _prompt_for_config_value("database_path", "/tmp/db") is None

    @patch("esdc.config_wizard.questionary.text")
    def test_text_cancel_returns_none(self, mock_text):
        mock_text.return_value.unsafe_ask.return_value = None
        assert _prompt_for_config_value("api_url", "http://old") is None

    @patch("esdc.config_wizard.questionary.text")
    def test_invalid_value_raises_instead_of_writing(self, mock_text):
        mock_text.return_value.unsafe_ask.return_value = "not-a-number"
        with pytest.raises(ValueError, match="cache.sql_ttl"):
            _prompt_for_config_value("cache.sql_ttl", 3600)

    @patch("esdc.config_wizard.questionary.text")
    def test_valid_value_is_parsed(self, mock_text):
        mock_text.return_value.unsafe_ask.return_value = "4000"
        assert _prompt_for_config_value("cache.sql_ttl", 3600) == 4000


class TestEditSectionCancel:
    """A cancelled or invalid field never reaches a save."""

    @patch("esdc.config_wizard.Config.get_all_config_flat")
    @patch("esdc.config_wizard.Config.set_config_value")
    @patch("esdc.config_wizard._prompt_for_config_value")
    @patch("esdc.config_wizard.questionary.select")
    @patch("esdc.config_wizard.rich_print")
    def test_cancel_skips_save(
        self, mock_print, mock_select, mock_prompt, mock_set, mock_flat
    ):
        from esdc.config_wizard import _edit_section

        mock_flat.return_value = {"api_url": "https://example.com"}
        mock_select.return_value.unsafe_ask.side_effect = ["api_url", "__back__"]
        mock_prompt.return_value = None

        _edit_section("Connection", ["api_url"])

        mock_set.assert_not_called()
        cancel_calls = [
            call
            for call in mock_print.call_args_list
            if "cancelled" in str(call).lower()
        ]
        assert len(cancel_calls) > 0

    @patch("esdc.config_wizard.Config.get_all_config_flat")
    @patch("esdc.config_wizard.Config.set_config_value")
    @patch("esdc.config_wizard.questionary.text")
    @patch("esdc.config_wizard.questionary.select")
    @patch("esdc.config_wizard.rich_print")
    def test_invalid_value_stays_at_the_field_and_does_not_write(
        self, mock_print, mock_select, mock_text, mock_set, mock_flat
    ):
        from esdc.config_wizard import _edit_section

        mock_flat.return_value = {"cache.sql_ttl": 3600}
        # First the invalid input, then Escape at the re-asked field.
        mock_text.return_value.unsafe_ask.side_effect = ["not-a-number", None]
        mock_select.return_value.unsafe_ask.side_effect = ["cache.sql_ttl", "__back__"]

        _edit_section("Cache", ["cache.sql_ttl"])

        mock_set.assert_not_called()
        assert any("cache.sql_ttl" in str(call) for call in mock_print.call_args_list)

    @patch("esdc.config_wizard.Config.get_all_config_flat")
    @patch("esdc.config_wizard.Config.set_config_value")
    @patch("esdc.config_wizard.questionary.text")
    @patch("esdc.config_wizard.questionary.select")
    @patch("esdc.config_wizard.rich_print")
    def test_related_fields_are_checked_against_the_draft(
        self, mock_print, mock_select, mock_text, mock_set, mock_flat
    ):
        from esdc.config_wizard import _edit_section

        mock_flat.return_value = {"corpus.chunk_size": 100, "corpus.chunk_overlap": 50}
        # An overlap larger than the draft chunk size is rejected before saving,
        # then Escape cancels the re-asked field.
        mock_text.return_value.unsafe_ask.side_effect = ["200", None]
        mock_select.return_value.unsafe_ask.side_effect = [
            "corpus.chunk_overlap",
            "__back__",
        ]

        _edit_section("Corpus", ["corpus.chunk_overlap"])

        mock_set.assert_not_called()
        assert any(
            "corpus.chunk_overlap" in str(call) for call in mock_print.call_args_list
        )

    @patch("esdc.config_wizard.Config.get_all_config_flat")
    @patch("esdc.config_wizard.Config.set_config_value")
    @patch("esdc.config_wizard.questionary.text")
    @patch("esdc.config_wizard.questionary.select")
    @patch("esdc.config_wizard.rich_print")
    def test_invalid_input_then_valid_input_writes_only_the_valid_value(
        self, mock_print, mock_select, mock_text, mock_set, mock_flat
    ):
        from esdc.config_wizard import _edit_section

        mock_flat.return_value = {"cache.sql_ttl": 3600}
        mock_text.return_value.unsafe_ask.side_effect = ["not-a-number", "300"]
        mock_select.return_value.unsafe_ask.side_effect = ["cache.sql_ttl", "__back__"]

        _edit_section("Cache", ["cache.sql_ttl"])

        mock_set.assert_called_once_with("cache.sql_ttl", 300)



class TestRunWizardCancel:
    """Control+C ends the run cleanly, and earlier saves remain written."""

    @patch("esdc.config_wizard.Config.init_config")
    @patch("esdc.config_wizard.Config.get_all_config_flat")
    @patch("esdc.config_wizard.traceback.print_exc")
    @patch("esdc.config_wizard.rich_print")
    @patch("esdc.config_wizard.Config.set_config_value")
    @patch("esdc.config_wizard.questionary.select")
    def test_control_c_exits_without_a_traceback_and_keeps_the_save(
        self,
        mock_select,
        mock_set,
        mock_print,
        mock_traceback,
        mock_flat,
        mock_init,
    ):
        # The draft is supplied so the run never reads the live config file.
        mock_flat.return_value = {"logging.level": "INFO"}
        # Home -> Application settings -> the key -> value, then Back, then Control+C.
        mock_select.return_value.unsafe_ask.side_effect = [
            "Application settings",
            "logging.level",
            "DEBUG",
            "__back__",
            KeyboardInterrupt(),
        ]
        mock_print.return_value = None

        run_wizard()

        mock_set.assert_called_once_with("logging.level", "DEBUG")
        mock_traceback.assert_not_called()
        assert any("Goodbye" in str(call) for call in mock_print.call_args_list)
