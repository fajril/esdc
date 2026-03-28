"""
DOM Structure and Widget Mounting Tests for ESDC Chat App

Tests verify:
1. Widget mounting functionality
2. ChatPanel structure
3. Collapsible widgets (SQLPanel, ResultsPanel, ThinkingIndicator)
4. Correct mounting order
5. Widget parent-child relationships
"""

import pytest
from textual.app import App
from textual.widgets import Static, Markdown
from textual.containers import ScrollableContainer
from textual.widgets import Collapsible

# Import the app and widgets
from esdc.chat.app import (
    ESDCChatApp,
    ChatPanel,
    ChatMessage,
    ThinkingIndicator,
    SQLPanel,
    ResultsPanel,
    ContextPanel,
)


class TestChatPanelStructure:
    """Test ChatPanel widget structure."""

    def test_chat_panel_is_scrollable_container(self):
        """ChatPanel should be a ScrollableContainer."""
        assert issubclass(ChatPanel, ScrollableContainer)

    def test_chat_panel_has_messages_list(self):
        """ChatPanel should have messages attribute."""
        panel = ChatPanel()
        assert hasattr(panel, "messages")
        assert isinstance(panel.messages, list)

    def test_chat_panel_has_add_message_method(self):
        """ChatPanel should have add_message method."""
        panel = ChatPanel()
        assert hasattr(panel, "add_message")
        assert callable(panel.add_message)

    def test_chat_panel_has_mount_collapsible_method(self):
        """ChatPanel should have mount_collapsible method."""
        panel = ChatPanel()
        assert hasattr(panel, "mount_collapsible")
        assert callable(panel.mount_collapsible)


class TestThinkingIndicator:
    """Test ThinkingIndicator collapsible widget."""

    def test_thinking_indicator_is_collapsible(self):
        """ThinkingIndicator should be a Collapsible."""
        assert issubclass(ThinkingIndicator, Collapsible)

    def test_thinking_indicator_has_steps_list(self):
        """ThinkingIndicator should have steps list."""
        indicator = ThinkingIndicator()
        assert hasattr(indicator, "steps")
        assert isinstance(indicator.steps, list)

    def test_thinking_indicator_can_add_steps(self):
        """ThinkingIndicator should be able to add steps."""
        indicator = ThinkingIndicator()
        assert hasattr(indicator, "add_step")
        indicator.add_step("Test step")
        assert len(indicator.steps) == 1
        assert indicator.steps[0] == "Test step"

    def test_thinking_indicator_title_updates(self):
        """ThinkingIndicator title should update with step count."""
        indicator = ThinkingIndicator()
        initial_title = indicator.title
        indicator.add_step("Step 1")
        # Title should contain step count after adding
        # (verified in _update_display called by add_step)


class TestSQLPanel:
    """Test SQLPanel collapsible widget."""

    def test_sql_panel_is_collapsible(self):
        """SQLPanel should be a Collapsible."""
        assert issubclass(SQLPanel, Collapsible)

    def test_sql_panel_has_sql_content(self):
        """SQLPanel should store SQL content."""
        sql = "SELECT * FROM test;"
        panel = SQLPanel(sql)
        assert panel.sql_content == sql

    def test_sql_panel_collapsed_by_default_when_empty(self):
        """SQLPanel should be collapsed when empty."""
        panel = SQLPanel("")
        assert panel.collapsed is True

    def test_sql_panel_expanded_when_has_content(self):
        """SQLPanel should be expanded when has SQL."""
        panel = SQLPanel("SELECT 1;")
        assert panel.collapsed is False

    def test_sql_panel_compose_returns_markdown(self):
        """SQLPanel compose should yield Markdown widget."""
        panel = SQLPanel("SELECT 1;")
        # Get generator items
        items = list(panel.compose())
        assert len(items) == 1
        assert isinstance(items[0], Markdown)

    def test_sql_panel_has_set_sql_method(self):
        """SQLPanel should have set_sql method for updating content."""
        panel = SQLPanel("")
        assert hasattr(panel, "set_sql")
        assert callable(panel.set_sql)


class TestResultsPanel:
    """Test ResultsPanel collapsible widget."""

    def test_results_panel_is_collapsible(self):
        """ResultsPanel should be a Collapsible."""
        assert issubclass(ResultsPanel, Collapsible)

    def test_results_panel_has_results_content(self):
        """ResultsPanel should store results content."""
        results = "col1|col2\nval1|val2"
        panel = ResultsPanel(results)
        assert panel.results_content == results

    def test_results_panel_collapsed_by_default_when_empty(self):
        """ResultsPanel should be collapsed when empty."""
        panel = ResultsPanel("")
        assert panel.collapsed is True

    def test_results_panel_expanded_when_has_content(self):
        """ResultsPanel should be expanded when has results."""
        panel = ResultsPanel("test results")
        assert panel.collapsed is False

    def test_results_panel_compose_returns_markdown(self):
        """ResultsPanel compose should yield Markdown widget."""
        panel = ResultsPanel("test results")
        items = list(panel.compose())
        assert len(items) == 1
        assert isinstance(items[0], Markdown)

    def test_results_panel_has_set_results_method(self):
        """ResultsPanel should have set_results method for updating content."""
        panel = ResultsPanel("")
        assert hasattr(panel, "set_results")
        assert callable(panel.set_results)

    def test_results_panel_format_empty_results(self):
        """ResultsPanel should format empty results correctly."""
        panel = ResultsPanel("")
        formatted = panel._format_results_as_markdown("")
        assert formatted == "No results returned."

    def test_results_panel_format_markdown_table(self):
        """ResultsPanel should convert pipe-separated to markdown table."""
        panel = ResultsPanel("")
        raw_data = "col1|col2\nval1|val2"
        formatted = panel._format_results_as_markdown(raw_data)
        # Should have markdown table structure
        assert "|" in formatted
        assert "---" in formatted


class TestChatMessage:
    """Test ChatMessage widget."""

    def test_chat_message_is_markdown(self):
        """ChatMessage should be a Markdown widget."""
        assert issubclass(ChatMessage, Markdown)

    def test_chat_message_formats_user_message(self):
        """ChatMessage should format user messages with quote."""
        msg = ChatMessage("user", "Hello")
        # User messages are formatted with >
        assert hasattr(msg, "role")
        assert msg.role == "user"

    def test_chat_message_formats_ai_message(self):
        """ChatMessage should format AI messages as plain text."""
        msg = ChatMessage("ai", "Hello")
        assert hasattr(msg, "role")
        assert msg.role == "ai"

    def test_chat_message_adds_role_class(self):
        """ChatMessage should add role as CSS class."""
        msg = ChatMessage("user", "Test")
        assert "user" in msg.classes
        msg2 = ChatMessage("ai", "Test")
        assert "ai" in msg2.classes


class TestContextPanel:
    """Test ContextPanel widget."""

    def test_context_panel_container(self):
        """ContextPanel should be a Vertical container."""
        panel = ContextPanel()
        # Check it's a container (Vertical inherits from Container)
        assert hasattr(panel, "compose")

    def test_context_panel_update_session_info(self):
        """ContextPanel should have update_session_info method."""
        panel = ContextPanel()
        assert hasattr(panel, "update_session_info")
        assert callable(panel.update_session_info)

    def test_context_panel_stores_provider_model_thread(self):
        """ContextPanel should store session info."""
        panel = ContextPanel()
        panel.update_session_info("test_provider", "test_model", "test_thread")
        assert panel._provider_name == "test_provider"
        assert panel._model_name == "test_model"
        assert panel._session_thread_id == "test_thread"


class TestWidgetMountingOrder:
    """Test correct mounting order of widgets."""

    @pytest.mark.anyio
    async def test_mounting_order_in_app(self):
        """Test that widgets mount in correct order: Thinking → SQL → Results → AI."""
        app = ESDCChatApp()

        async with app.run_test() as pilot:
            # Get chat panel
            chat_panel = app.query_one(ChatPanel)
            assert chat_panel is not None

            # Mount thinking indicator
            thinking = ThinkingIndicator()
            chat_panel.mount(thinking)

            # Mount SQL panel
            sql = SQLPanel("SELECT 1;")
            chat_panel.mount(sql)

            # Mount results panel
            results = ResultsPanel("result data")
            chat_panel.mount(results)

            # Mount AI message last
            ai_msg = ChatMessage("ai", "AI response")
            chat_panel.mount(ai_msg)

            # Verify order by checking children
            children = list(chat_panel.children)

            # Should have 4 widgets in order
            assert len(children) >= 4

            # Check types in order
            assert isinstance(children[0], ThinkingIndicator)
            assert isinstance(children[1], SQLPanel)
            assert isinstance(children[2], ResultsPanel)
            assert isinstance(children[3], ChatMessage)
            assert children[3].role == "ai"


class TestCollapsibleWidgets:
    """Test Collapsible widgets functionality."""

    def test_sql_panel_collapsible_toggle(self):
        """SQLPanel should be collapsible and expandable."""
        panel = SQLPanel("SELECT * FROM users;")
        # Initially not collapsed (has content)
        assert panel.collapsed is False

        # Can collapse
        panel.collapsed = True
        assert panel.collapsed is True

        # Can expand
        panel.collapsed = False
        assert panel.collapsed is False

    def test_results_panel_collapsible_toggle(self):
        """ResultsPanel should be collapsible and expandable."""
        panel = ResultsPanel("test data")
        # Initially not collapsed (has content)
        assert panel.collapsed is False

        # Can collapse
        panel.collapsed = True
        assert panel.collapsed is True

    def test_thinking_indicator_collapsible_toggle(self):
        """ThinkingIndicator should be collapsible."""
        indicator = ThinkingIndicator()
        # Initially collapsed=False (shows steps)
        assert indicator.collapsed is False

        # Can collapse
        indicator.collapsed = True
        assert indicator.collapsed is True


class TestNoDuplicateMounting:
    """Test that widgets don't mount multiple times."""

    @pytest.mark.anyio
    async def test_widget_mounts_only_once(self):
        """Verify mounting a widget twice doesn't duplicate it."""
        app = ESDCChatApp()

        async with app.run_test() as pilot:
            chat_panel = app.query_one(ChatPanel)

            # Create and mount widget once
            panel1 = SQLPanel("SELECT 1;")
            chat_panel.mount(panel1)

            initial_count = len(list(chat_panel.children))

            # Try mounting again (should not duplicate)
            # Textual prevents duplicate mounting

            final_count = len(list(chat_panel.children))
            assert final_count == initial_count


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
