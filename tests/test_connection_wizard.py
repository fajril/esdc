"""Connection management is model-independent and reference-safe.

These tests drive the wizard's connection flows with mocked prompts. No test
writes the user's config, constructs a chat model, or opens a socket.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from esdc.model_catalog import ModelCatalog, ModelEntry


@patch("esdc.config_wizard.rich_print")
@patch("esdc.config_wizard.questionary.select")
def test_connections_menu_has_only_connection_actions(mock_select, mock_print):
    from esdc.config_wizard import _connections_menu

    mock_select.return_value.unsafe_ask.side_effect = ["__back__"]
    _connections_menu()

    labels = [str(choice.title) for choice in mock_select.call_args.kwargs["choices"]]
    assert labels == [
        "← Back",
        "Add connection",
        "Edit connection",
        "Remove connection",
        "Test connection",
    ]


@patch("esdc.config_wizard.rich_print")
@patch("esdc.config_wizard.Config.save_connection")
@patch("esdc.config_wizard.questionary.text")
@patch("esdc.config_wizard.questionary.select")
def test_every_connection_type_can_have_a_custom_name(
    mock_select, mock_text, mock_save, mock_print
):
    from esdc.config_wizard import _add_connection_flow

    mock_select.return_value.unsafe_ask.side_effect = ["ollama", "save"]
    mock_text.return_value.unsafe_ask.side_effect = [
        "desk-two",
        "http://box:11434",
    ]

    _add_connection_flow()

    mock_save.assert_called_once_with(
        "desk-two", {"type": "ollama", "base_url": "http://box:11434"}
    )
    # No model or reasoning prompt is part of a connection draft.
    select_labels = [
        str(choice.title)
        for call in mock_select.call_args_list
        for choice in call.kwargs["choices"]
    ]
    assert not any("model" in label.lower() for label in select_labels)
    assert not any("reasoning" in label.lower() for label in select_labels)


@patch("esdc.config_wizard.rich_print")
@patch("esdc.config_wizard.Config.get_connections")
@patch("esdc.config_wizard.Config.save_connection")
@patch("esdc.config_wizard.questionary.text")
@patch("esdc.config_wizard.questionary.select")
def test_a_second_connection_of_the_same_type_is_allowed(
    mock_select, mock_text, mock_save, mock_get_connections, mock_print
):
    from esdc.config_wizard import _add_connection_flow

    mock_get_connections.return_value = {
        "desk-one": {"type": "ollama", "base_url": "http://one:11434"}
    }
    mock_select.return_value.unsafe_ask.side_effect = ["ollama", "save"]
    mock_text.return_value.unsafe_ask.side_effect = [
        "desk-two",
        "http://two:11434",
    ]

    _add_connection_flow()

    mock_save.assert_called_once_with(
        "desk-two", {"type": "ollama", "base_url": "http://two:11434"}
    )


@patch("esdc.config_wizard.rich_print")
@patch("esdc.config_wizard.Config.save_connection")
@patch("esdc.config_wizard.questionary.text")
@patch("esdc.config_wizard.questionary.select")
def test_cancel_at_the_action_menu_writes_nothing(
    mock_select, mock_text, mock_save, mock_print
):
    from esdc.config_wizard import _add_connection_flow

    mock_select.return_value.unsafe_ask.side_effect = ["openai", "cancel"]
    mock_text.return_value.unsafe_ask.side_effect = ["work-openai"]
    mock_password = MagicMock()
    with patch("esdc.config_wizard.questionary.password", mock_password):
        mock_password.return_value.unsafe_ask.return_value = "synthetic-key"
        _add_connection_flow()

    mock_save.assert_not_called()


def test_connection_editor_offers_no_model_or_reasoning_fields():
    from esdc.config_wizard import _connection_edit_choices

    draft = {
        "type": "openai",
        "api_key": "synthetic-key",
        "model": "chat-model",
        "reasoning_effort": "high",
    }
    values = [
        choice.value for choice in _connection_edit_choices(draft, "work-openai")
    ]

    assert "model" not in values
    assert "reasoning_effort" not in values
    assert "name" in values
    assert "api_key" in values


@patch("esdc.config_wizard.rich_print")
@patch("esdc.config_wizard.Config.get_connection_references")
@patch("esdc.config_wizard.Config.rename_connection")
@patch("esdc.config_wizard.Config.save_connection")
def test_save_lists_impacted_roles_and_renames_in_one_write(
    mock_save, mock_rename, mock_references, mock_print
):
    from esdc.config_wizard import _save_connection_draft

    mock_references.return_value = [
        "models.chat",
        "models.chat.fallbacks[0]",
        "models.embedding",
    ]
    draft = {
        "type": "openai_compatible",
        "base_url": "http://renamed.example/v1",
        "api_key": "synthetic-key",
    }

    _save_connection_draft("renamed", draft, "engine")

    mock_rename.assert_called_once_with("engine", "renamed", values=draft)
    mock_save.assert_not_called()
    printed = " ".join(str(call.args[0]) for call in mock_print.call_args_list)
    assert "Chat" in printed
    assert "Chat fallback 1" in printed
    assert "Embeddings" in printed


@patch("esdc.config_wizard.rich_print")
@patch("esdc.config_wizard.Config.get_connection_references")
@patch("esdc.config_wizard.Config.save_connection")
def test_same_name_edit_saves_in_place_and_lists_uses(
    mock_save, mock_references, mock_print
):
    from esdc.config_wizard import _save_connection_draft

    mock_references.return_value = ["models.chat"]
    draft = {"type": "openai", "api_key": "synthetic-key"}

    _save_connection_draft("work-openai", draft, "work-openai")

    mock_save.assert_called_once_with("work-openai", draft)
    assert "Chat" in " ".join(
        str(call.args[0]) for call in mock_print.call_args_list
    )


@patch("esdc.config_wizard.rich_print")
@patch("esdc.config_wizard.Config.get_connection_references")
@patch("esdc.config_wizard.Config.get_connections")
@patch("esdc.config_wizard.Config.remove_connection")
@patch("esdc.config_wizard.questionary.select")
def test_remove_refuses_a_referenced_connection_and_names_the_fallback(
    mock_select, mock_remove, mock_get_connections, mock_references, mock_print
):
    from esdc.config_wizard import _remove_connection_flow

    mock_get_connections.return_value = {"engine": {"type": "openai"}}
    mock_references.return_value = [
        "models.chat.fallbacks[0]",
        "models.embedding",
    ]
    mock_select.return_value.unsafe_ask.return_value = "engine"

    _remove_connection_flow()

    mock_remove.assert_not_called()
    printed = " ".join(str(call.args[0]) for call in mock_print.call_args_list)
    assert "Chat fallback 1" in printed
    assert "Embeddings" in printed
    assert "nothing was removed" in printed


@patch("esdc.config_wizard.rich_print")
@patch("esdc.config_wizard.Config.remove_connection")
@patch("esdc.config_wizard.Config.get_connection_references", return_value=[])
@patch("esdc.config_wizard.Config.get_connections")
@patch("esdc.config_wizard.questionary.confirm")
@patch("esdc.config_wizard.questionary.select")
def test_remove_deletes_an_unreferenced_connection(
    mock_select, mock_confirm, mock_get_connections, mock_references, mock_remove, mock_print
):
    from esdc.config_wizard import _remove_connection_flow

    mock_get_connections.return_value = {"spare": {"type": "ollama"}}
    mock_select.return_value.unsafe_ask.return_value = "spare"
    mock_confirm.return_value.unsafe_ask.return_value = True

    _remove_connection_flow()

    mock_remove.assert_called_once_with("spare")


@patch("esdc.config_wizard.rich_print")
@patch("esdc.config_wizard.Config.save_connection")
@patch("esdc.config_wizard.discover_models")
@patch("esdc.config_wizard.Config.get_connections")
@patch("esdc.config_wizard.questionary.select")
def test_test_connection_uses_discovery_and_writes_nothing(
    mock_select, mock_get_connections, mock_discover, mock_save, mock_print
):
    from esdc.config_wizard import _test_connection_standalone

    mock_get_connections.return_value = {
        "engine": {"type": "openai_compatible", "base_url": "http://engine/v1"}
    }
    mock_discover.return_value = ModelCatalog(
        entries=(ModelEntry(model_id="chat-a"), ModelEntry(model_id="embed-b"))
    )
    mock_select.return_value.unsafe_ask.return_value = "engine"

    _test_connection_standalone()

    mock_discover.assert_called_once()
    mock_save.assert_not_called()
    printed = " ".join(str(call.args[0]) for call in mock_print.call_args_list)
    assert "2" in printed


@patch("esdc.config_wizard.rich_print")
@patch("esdc.config_wizard.Config.save_connection")
@patch("esdc.config_wizard.discover_models")
@patch("esdc.config_wizard.Config.get_connections")
@patch("esdc.config_wizard.questionary.select")
def test_test_connection_reports_an_unverified_warning_without_claiming_failure(
    mock_select, mock_get_connections, mock_discover, mock_save, mock_print
):
    from esdc.config_wizard import _test_connection_standalone

    mock_get_connections.return_value = {
        "engine": {"type": "anthropic", "api_key": "synthetic-key"}
    }
    mock_discover.return_value = ModelCatalog(
        warning="Model discovery is unavailable for provider type anthropic."
    )
    mock_select.return_value.unsafe_ask.return_value = "engine"

    _test_connection_standalone()

    printed = " ".join(str(call.args[0]) for call in mock_print.call_args_list)
    assert "unavailable" in printed
    assert "cannot serve" not in printed
