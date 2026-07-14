"""Slash command handling."""


def _make_app(monkeypatch):
    from esdc.chat.app import ESDCChatApp

    app = ESDCChatApp()
    displayed = []
    monkeypatch.setattr(
        app, "display_message", lambda role, content: displayed.append((role, content))
    )
    return app, displayed


def test_non_slash_input_not_handled(monkeypatch):
    app, _ = _make_app(monkeypatch)
    assert app._handle_slash_command("show reserves for 2025") is False


def test_new_command_resets_thread(monkeypatch):
    app, displayed = _make_app(monkeypatch)
    old_thread = app._thread_id
    assert app._handle_slash_command("/new") is True
    assert app._thread_id != old_thread
    assert app._token_count == 0
    assert any("New conversation" in c for _, c in displayed)


def test_unknown_command_shows_help_hint(monkeypatch):
    app, displayed = _make_app(monkeypatch)
    assert app._handle_slash_command("/frobnicate") is True
    assert any("/help" in c for _, c in displayed)


def test_help_lists_commands(monkeypatch):
    app, displayed = _make_app(monkeypatch)
    assert app._handle_slash_command("/help") is True
    assert any("/new" in c for _, c in displayed)
