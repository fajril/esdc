"""Token events must not render directly; a 10Hz flush renders instead."""


class _FakeMarkdown:
    def __init__(self):
        self.update_calls = 0
        self.last_content = ""

    def update(self, content):
        self.update_calls += 1
        self.last_content = content


def _make_app_with_stream():
    from esdc.chat.app import ESDCChatApp

    app = ESDCChatApp()
    fake = _FakeMarkdown()
    app._streaming_message = fake
    app._accumulated_content = ""
    return app, fake


def test_token_events_do_not_render_directly():
    app, fake = _make_app_with_stream()
    for i in range(200):
        app._handle_stream_chunk({"type": "token", "content": f"t{i} "})
    assert fake.update_calls == 0
    assert app._render_dirty is True
    assert "t199" in app._accumulated_content


def test_flush_renders_once_and_clears_dirty():
    app, fake = _make_app_with_stream()
    for _i in range(50):
        app._handle_stream_chunk({"type": "token", "content": "x"})
    app._flush_stream_render()
    assert fake.update_calls == 1
    assert fake.last_content == "x" * 50
    assert app._render_dirty is False
    app._flush_stream_render()  # nothing new -> no render
    assert fake.update_calls == 1
