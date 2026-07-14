"""Widget-level tests for the overhauled TUI."""


class TestStatusBar:
    def test_status_line_contains_all_segments(self):
        from esdc.chat.widgets import IRIS_VERSION, StatusBar

        bar = StatusBar()
        bar.set_status(
            model_name="qwen3:32b",
            thread_id="esdc-abcd1234",
            token_count=21_554,
            context_length=262_144,
        )
        text = str(bar._status_text)
        assert f"IRIS v{IRIS_VERSION}" in text
        assert "qwen3:32b" in text
        assert "esdc-abc" in text
        assert "21,554/262,144" in text
        assert "(8%)" in text

    def test_high_context_usage_is_marked(self):
        from esdc.chat.widgets import StatusBar

        bar = StatusBar()
        bar.set_status(
            model_name="m",
            thread_id="t",
            token_count=80,
            context_length=100,
        )
        assert "[red]" in str(bar._status_text)


class TestContextPanelSlim:
    def test_context_panel_has_no_session_or_context_sections(self):
        import inspect

        from esdc.chat.widgets import ContextPanel

        source = inspect.getsource(ContextPanel.compose)
        assert "Session Info" not in source
        assert "session-section" not in source
        assert "context-section" not in source
