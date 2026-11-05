"""Tests for provider overwrite and default provider logic.

The overwrite decision belongs to the Save choice of the provider draft, so these
mocks end the draft action menu with "save"; the no-write cases end it with
"cancel". The overwrite confirmation assertions are unchanged; the write path is
now in place, so a differently named provider is renamed first and nothing is
removed before replacement.
"""

import os
from unittest.mock import patch

import yaml

from esdc.config_wizard import (
    _check_name_exists,
    _find_provider_by_type,
    _format_provider_label,
)
from esdc.configs import Config


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

    @patch("esdc.config_wizard.Config.rename_provider")
    @patch("esdc.config_wizard.Config.save_provider")
    @patch("esdc.config_wizard.Config.get_providers")
    @patch("esdc.config_wizard.Config.get_default_provider")
    @patch("esdc.config_wizard.Config.remove_provider")
    @patch("esdc.config_wizard.questionary.confirm")
    @patch("esdc.config_wizard.questionary.select")
    @patch("esdc.config_wizard.questionary.text")
    @patch("esdc.config_wizard.questionary.password")
    @patch("esdc.config_wizard._fetch_models", return_value=["gpt-4o"])
    @patch("esdc.config_wizard.rich_print")
    def test_overwrite_confirms(
        self,
        mock_print,
        mock_fetch_models,
        mock_password,
        mock_text,
        mock_select,
        mock_confirm,
        mock_remove,
        mock_default,
        mock_providers,
        mock_save,
        mock_rename,
    ):
        from esdc.config_wizard import _add_provider_flow

        # Existing openai provider
        mock_providers.return_value = {
            "my-openai": {"provider_type": "openai"},
        }
        mock_default.return_value = "my-openai"
        # Provider type, reasoning effort, model, then the draft action menu.
        # The overwrite confirmation belongs to Save, so a Cancel before it
        # would leave the existing provider untouched.
        mock_select.return_value.unsafe_ask.side_effect = [
            "openai",
            "low",
            "gpt-4o",
            "save",
        ]
        mock_text.return_value.unsafe_ask.side_effect = [
            "company-openai",
            "",
        ]
        mock_password.return_value.unsafe_ask.return_value = "placeholder-key"
        mock_confirm.return_value.unsafe_ask.side_effect = [
            True,
            False,
        ]

        _add_provider_flow()

        # The existing key is renamed, not deleted, so its references survive, and
        # the draft is written by that same call rather than by a second write.
        mock_rename.assert_called_once()
        rename_args = mock_rename.call_args
        assert rename_args.args == ("my-openai", "openai")
        # Name should be provider_type key for non-custom types
        assert rename_args.args[1] == "openai"
        assert rename_args.kwargs["provider_config"]["model"] == "gpt-4o"
        mock_remove.assert_not_called()
        mock_save.assert_not_called()

    @patch("esdc.config_wizard.Config.save_provider")
    @patch("esdc.config_wizard.Config.get_providers")
    @patch("esdc.config_wizard.Config.get_default_provider")
    @patch("esdc.config_wizard.Config.remove_provider")
    @patch("esdc.config_wizard.questionary.confirm")
    @patch("esdc.config_wizard.questionary.select")
    @patch("esdc.config_wizard.questionary.text")
    @patch("esdc.config_wizard.questionary.password")
    @patch("esdc.config_wizard._fetch_models", return_value=["gpt-4o"])
    @patch("esdc.config_wizard.rich_print")
    def test_overwrite_declined(
        self,
        mock_print,
        mock_fetch_models,
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
        # Declining overwrite at Save writes nothing for the draft.
        mock_select.return_value.unsafe_ask.side_effect = [
            "openai",
            "low",
            "gpt-4o",
            "save",
        ]
        mock_text.return_value.unsafe_ask.side_effect = [
            "company-openai",
            "",
        ]
        mock_password.return_value.unsafe_ask.return_value = "sk-test"
        mock_confirm.return_value.unsafe_ask.side_effect = [False]

        _add_provider_flow()

        # Should NOT remove or save
        mock_remove.assert_not_called()
        mock_save.assert_not_called()


class TestOverwriteWritesInPlace:
    """The overwrite replaces the entry in place, so references survive.

    These tests drive the real `Config` against a temporary file, so the write
    path is checked on disk. Every stored value is a placeholder.
    """

    _CONFIG_TEXT = """
default_provider: openai
provider_order:
  - openai
  - other
embedding_provider: openai
embedding_model: embedding-placeholder
corpus:
  metadata_provider: openai
  cleanup_provider: openai
  ocr_provider: openai
  rerank_provider: openai
  metadata_model: provider:openai
providers:
  openai:
    provider_type: openai
    model: chat-placeholder
    reasoning_effort: low
  other:
    provider_type: ollama
    base_url: http://localhost:11434
    model: chat-b
"""

    def _run_add_flow(self, select_values, confirm_values):
        """Run the add flow with the prompt values, ending the menu with Save."""
        from esdc.config_wizard import _add_provider_flow

        with (
            patch("esdc.config_wizard._fetch_models", return_value=["chat-new"]),
            patch("esdc.config_wizard.questionary.select") as mock_select,
            patch("esdc.config_wizard.questionary.password") as mock_password,
            patch("esdc.config_wizard.questionary.confirm") as mock_confirm,
            patch("esdc.config_wizard.rich_print"),
        ):
            # Provider type, reasoning effort, model, then the action menu.
            mock_select.return_value.unsafe_ask.side_effect = list(select_values)
            mock_password.return_value.unsafe_ask.return_value = "placeholder-key"
            mock_confirm.return_value.unsafe_ask.side_effect = list(confirm_values)
            _add_provider_flow()

    def test_same_name_overwrite_writes_in_place(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text(self._CONFIG_TEXT)

        replaces = []
        real_replace = os.replace

        def record_replace(src, dst):
            replaces.append((src, dst))
            return real_replace(src, dst)

        with (
            patch.object(Config, "get_config_dir", return_value=tmp_path),
            patch.object(Config, "remove_provider") as mock_remove,
            patch.object(os, "replace", side_effect=record_replace),
        ):
            self._run_add_flow(
                ["openai", "(none)", "chat-new", "save"],
                # Overwrite confirmed, then the default-provider confirm declined.
                [True, False],
            )

        # Nothing was deleted first: one write replaced the entry.
        mock_remove.assert_not_called()
        assert len(replaces) == 1

        config = yaml.safe_load(config_file.read_text())
        assert config["providers"]["openai"]["model"] == "chat-new"
        # Every reference still names the same provider key.
        assert config["default_provider"] == "openai"
        assert config["provider_order"] == ["openai", "other"]
        assert config["embedding_provider"] == "openai"
        for key in (
            "metadata_provider",
            "cleanup_provider",
            "ocr_provider",
            "rerank_provider",
        ):
            assert config["corpus"][key] == "openai"
        assert config["corpus"]["metadata_model"] == "provider:openai"
        # An unrelated task model and the other provider are untouched.
        assert config["embedding_model"] == "embedding-placeholder"
        assert config["providers"]["other"] == {
            "provider_type": "ollama",
            "base_url": "http://localhost:11434",
            "model": "chat-b",
        }

    def test_name_different_overwrite_renames_before_the_write(self, tmp_path):
        # The same file, but the openai provider is stored under a different key.
        config_text = """
default_provider: engine
provider_order:
  - engine
  - other
embedding_provider: engine
embedding_model: embedding-placeholder
corpus:
  metadata_provider: engine
  cleanup_provider: engine
  ocr_provider: engine
  rerank_provider: engine
  metadata_model: provider:engine
providers:
  engine:
    provider_type: openai
    model: chat-placeholder
    reasoning_effort: low
  other:
    provider_type: ollama
    base_url: http://localhost:11434
    model: chat-b
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(config_text)

        calls = []
        real_rename = Config.rename_provider
        real_save = Config.save_provider

        def record_rename(old_name, new_name, provider_config=None):
            calls.append(
                ("rename", old_name, new_name, "draft" if provider_config else "entry")
            )
            return real_rename(old_name, new_name, provider_config)

        def record_save(name, provider_config):
            calls.append(("save", name))
            return real_save(name, provider_config)

        with (
            patch.object(Config, "get_config_dir", return_value=tmp_path),
            patch.object(Config, "remove_provider") as mock_remove,
            patch.object(Config, "rename_provider", side_effect=record_rename),
            patch.object(Config, "save_provider", side_effect=record_save),
        ):
            # The name of an openai provider is its type key, so the entry named
            # 'engine' is renamed to 'openai' and the draft is written by that same
            # call: one write, no second save that could fail after the move.
            self._run_add_flow(
                ["openai", "(none)", "chat-new", "save"],
                [True, False],
            )

        mock_remove.assert_not_called()
        assert calls == [("rename", "engine", "openai", "draft")]

        config = yaml.safe_load(config_file.read_text())
        # The references moved with the entry, so no task lost its provider.
        assert config["default_provider"] == "openai"
        assert config["provider_order"] == ["openai", "other"]
        assert config["embedding_provider"] == "openai"
        for key in (
            "metadata_provider",
            "cleanup_provider",
            "ocr_provider",
            "rerank_provider",
        ):
            assert config["corpus"][key] == "openai"
        # `provider:engine` is a literal wire model behind a non-empty metadata
        # provider, so only the role provider moves with the rename.
        assert config["corpus"]["metadata_model"] == "provider:engine"
        assert set(config["providers"]) == {"openai", "other"}
        assert config["providers"]["openai"]["model"] == "chat-new"


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
    @patch("esdc.config_wizard._fetch_models", return_value=["gpt-4o"])
    @patch("esdc.config_wizard.rich_print")
    def test_auto_set_first_provider(
        self,
        mock_print,
        mock_fetch_models,
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
        mock_select.return_value.unsafe_ask.side_effect = [
            "openai",
            "low",
            "gpt-4o",
            "save",
        ]
        mock_text.return_value.unsafe_ask.side_effect = [
            "my-openai",
            "",
        ]
        mock_password.return_value.unsafe_ask.return_value = "sk-test"
        mock_confirm.return_value.unsafe_ask.side_effect = [
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
    @patch("esdc.config_wizard._fetch_models", return_value=["qwen-7b"])
    @patch("esdc.config_wizard.rich_print")
    def test_custom_name_saved(
        self,
        mock_print,
        mock_fetch_models,
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
        mock_default.return_value = "other"  # Not the first provider
        mock_select.return_value.unsafe_ask.side_effect = [
            "openai_compatible",  # select provider type
            "qwen-7b",  # select model
            "save",  # the draft action menu is the only write
        ]
        mock_text.return_value.unsafe_ask.side_effect = [
            "my-qwen",  # custom provider name
            "http://qwen:8000",  # base URL
        ]
        mock_password.return_value.unsafe_ask.return_value = ""  # no api key
        mock_confirm.return_value.unsafe_ask.side_effect = [
            False,  # don't set as default
        ]

        _add_provider_flow()

        mock_save.assert_called_once()
        call_args = mock_save.call_args
        assert call_args[0][0] == "my-qwen"
        saved_config = call_args[0][1]
        assert saved_config["provider_type"] == "openai_compatible"
        assert saved_config["model"] == "qwen-7b"
