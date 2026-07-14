"""Widget-level tests for the overhauled TUI."""

import pytest


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


class TestRightPanelComposition:
    def test_context_panel_composes_working_state_widgets(self):
        import inspect

        from esdc.chat.widgets import ContextPanel

        source = inspect.getsource(ContextPanel.compose)
        for widget in ("ToolTimeline", "SQLPanel", "ResultsPanel", "QueryHistory"):
            assert widget in source, f"{widget} missing from ContextPanel.compose"


class TestToolResultWiring:
    def test_tool_result_feeds_sql_and_results_panels(self, monkeypatch):
        """Tools register with display names, not python names.

        e.g. 'SQL Executor', not 'execute_sql' — the wiring must match on
        the real name.
        """
        from esdc.chat.app import ESDCChatApp

        app = ESDCChatApp()
        calls = {}

        class _P:
            def set_sql(self, s):
                calls["sql"] = s

            def set_results(self, r):
                calls["results"] = r

        class _CP:
            sql_panel = _P()
            results_panel = _P()

            class timeline:  # noqa: N801
                @staticmethod
                def finish_tool(name):
                    pass

        app._context_panel = _CP()
        app._handle_stream_chunk(
            {
                "type": "tool_result",
                "tool": "SQL Executor",
                "result": "col_a\n1\n2",
                "sql": "SELECT col_a FROM t",
            }
        )
        assert calls["sql"] == "SELECT col_a FROM t"
        assert calls["results"] == "col_a\n1\n2"

    def test_simple_data_query_result_feeds_sql_and_results_panels(self):
        """'Simple Data Query' returns a JSON string with an embedded key.

        The 'sql' key is embedded in the JSON body (no top-level sql chunk
        field) — the wiring must extract it and populate both the SQL and
        results panels.
        """
        from esdc.chat.app import ESDCChatApp

        app = ESDCChatApp()
        calls = {}

        class _P:
            def set_sql(self, s):
                calls["sql"] = s

            def set_results(self, r):
                calls["results"] = r

        class _CP:
            sql_panel = _P()
            results_panel = _P()

            class timeline:  # noqa: N801
                @staticmethod
                def finish_tool(name):
                    pass

        app._context_panel = _CP()
        app._handle_stream_chunk(
            {
                "type": "tool_result",
                "tool": "Simple Data Query",
                "sql": "",
                "result": '{"sql": "SELECT x FROM t", "rows": [[1]]}',
            }
        )
        assert calls["sql"] == "SELECT x FROM t"
        assert calls["results"] == '{"sql": "SELECT x FROM t", "rows": [[1]]}'


class TestToggleAllSections:
    def test_toggle_all_collapses_when_any_expanded(self):
        from esdc.chat.app import ESDCChatApp

        app = ESDCChatApp()

        class _Panel:
            collapsed = False

        class _CP:
            sql_panel = _Panel()
            results_panel = _Panel()

        app._context_panel = _CP()
        app.action_toggle_all_sections()
        assert app._context_panel.sql_panel.collapsed is True
        assert app._context_panel.results_panel.collapsed is True
        app.action_toggle_all_sections()
        assert app._context_panel.sql_panel.collapsed is False


class TestThinkingIndicator:
    def test_append_reasoning_accumulates(self):
        from esdc.chat.widgets import ThinkingIndicator

        ti = ThinkingIndicator()
        ti.append_reasoning("step one. ")
        ti.append_reasoning("step two.")
        assert "step one" in ti._reasoning_text
        assert "step two" in ti._reasoning_text

    def test_mark_done_sets_done_state(self):
        from esdc.chat.widgets import ThinkingIndicator

        ti = ThinkingIndicator()
        ti.append_reasoning("hmm")
        ti.mark_done()
        assert ti._done is True


class TestThinkingIndicatorMountSafety:
    def test_append_before_mount_does_not_crash(self):
        from esdc.chat.widgets import ThinkingIndicator

        ti = ThinkingIndicator()
        ti.append_reasoning("early reasoning")  # before any mount
        ti.mark_done()
        assert ti._done is True

    @pytest.mark.asyncio
    async def test_dynamic_mount_mid_stream_is_safe(self):
        from textual.app import App

        from esdc.chat.widgets import ThinkingIndicator

        class _Host(App):
            pass

        app = _Host()
        async with app.run_test() as pilot:
            ti = ThinkingIndicator()
            app.mount(ti)  # deliberately not awaited — mirrors app.py usage
            ti.append_reasoning("racing text")
            ti.mark_done()
            await pilot.pause()
            assert ti._content_widget is not None
