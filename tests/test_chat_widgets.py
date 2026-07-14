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


class TestToolTimeline:
    def test_start_and_finish_lifecycle(self):
        from esdc.chat.widgets import ToolTimeline

        tl = ToolTimeline()
        tl.start_tool("execute_sql")
        assert tl.entries[-1][0] == "execute_sql"
        assert tl.entries[-1][1] == "running"
        tl.finish_tool("execute_sql")
        assert tl.entries[-1][1] == "done"
        assert tl.entries[-1][2] >= 0.0

    def test_reset_clears_entries(self):
        from esdc.chat.widgets import ToolTimeline

        tl = ToolTimeline()
        tl.start_tool("search_documents")
        tl.reset()
        assert tl.entries == []

    def test_caps_at_20_entries(self):
        from esdc.chat.widgets import ToolTimeline

        tl = ToolTimeline()
        for i in range(25):
            tl.start_tool(f"tool_{i}")
        assert len(tl.entries) == 20
        assert tl.entries[-1][0] == "tool_24"


class TestChatDecluttered:
    def test_tool_status_maps_removed_from_app(self):
        import inspect

        import esdc.chat.app as app_mod

        source = inspect.getsource(app_mod)
        assert "TOOL_STATUS_MAP" not in source
        assert "TOOL_COMPLETED_MAP" not in source
