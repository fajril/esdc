# tests/test_chat_app.py
import pytest
from unittest.mock import patch, MagicMock


class TestChatMessage:
    """Tests for ChatMessage component."""

    def test_chat_message_stores_role(self):
        """Test that ChatMessage stores the role."""
        from esdc.chat.app import ChatMessage

        msg = ChatMessage("user", "Hello world")
        assert msg.role == "user"

    def test_chat_message_role_ai(self):
        """Test ChatMessage with AI role."""
        from esdc.chat.app import ChatMessage

        msg = ChatMessage("ai", "Response content")
        assert msg.role == "ai"

    def test_chat_message_role_system(self):
        """Test ChatMessage with system role."""
        from esdc.chat.app import ChatMessage

        msg = ChatMessage("system", "Tool call info")
        assert msg.role == "system"


class TestSQLPanel:
    """Tests for SQLPanel component."""

    def test_sql_panel_creation(self):
        """Test SQLPanel can be created."""
        from esdc.chat.app import SQLPanel

        panel = SQLPanel()
        assert panel.sql_content == ""

    def test_sql_panel_stores_sql(self):
        """Test SQLPanel stores SQL content."""
        from esdc.chat.app import SQLPanel

        panel = SQLPanel()
        panel.sql_content = "SELECT * FROM table1"
        assert panel.sql_content == "SELECT * FROM table1"


class TestResultsPanel:
    """Tests for ResultsPanel component."""

    def test_results_panel_creation(self):
        """Test ResultsPanel can be created."""
        from esdc.chat.app import ResultsPanel

        panel = ResultsPanel()
        assert panel.results_content == ""

    def test_results_panel_stores_data(self):
        """Test ResultsPanel stores data without calling display (no app needed)."""
        from esdc.chat.app import ResultsPanel

        panel = ResultsPanel()
        panel.results_content = "1 | John\n2 | Jane"
        assert panel.results_content == "1 | John\n2 | Jane"


class TestChatPanel:
    """Tests for ChatPanel container."""

    def test_chat_panel_creation(self):
        """Test ChatPanel can be created."""
        from esdc.chat.app import ChatPanel

        panel = ChatPanel()
        assert panel.sql_panel is not None
        assert panel.results_panel is not None
        assert panel.thinking is not None

    def test_chat_panel_has_sql_panel(self):
        """Test ChatPanel has SQLPanel."""
        from esdc.chat.app import ChatPanel, SQLPanel

        panel = ChatPanel()
        assert isinstance(panel.sql_panel, SQLPanel)

    def test_chat_panel_has_results_panel(self):
        """Test ChatPanel has ResultsPanel."""
        from esdc.chat.app import ChatPanel, ResultsPanel

        panel = ChatPanel()
        assert isinstance(panel.results_panel, ResultsPanel)

    def test_sql_panel_stores_sql(self):
        """Test SQLPanel stores SQL content."""
        from esdc.chat.app import SQLPanel

        panel = SQLPanel()
        panel.sql_content = "SELECT * FROM test"
        assert panel.sql_content == "SELECT * FROM test"

    def test_results_panel_stores_results(self):
        """Test ResultsPanel stores results content."""
        from esdc.chat.app import ResultsPanel

        panel = ResultsPanel()
        panel.results_content = "result data"
        assert panel.results_content == "result data"


class TestContextSection:
    """Tests for ContextSection collapsible widget."""

    def test_context_section_creation(self):
        """Test ContextSection can be created."""
        from esdc.chat.app import ContextSection

        section = ContextSection("Test Section")
        assert section.title == "Test Section"
        assert section.expanded is False

    def test_context_section_expanded(self):
        """Test ContextSection with expanded state."""
        from esdc.chat.app import ContextSection

        section = ContextSection("Test Section", expanded=True)
        assert section.expanded is True

    def test_context_section_toggle(self):
        """Test toggling ContextSection."""
        from esdc.chat.app import ContextSection

        section = ContextSection("Test Section")
        assert section.expanded is False
        section.toggle()
        assert section.expanded is True


class TestTokenUsageWidget:
    """Tests for TokenUsageWidget."""

    def test_token_usage_widget_creation(self):
        """Test TokenUsageWidget can be created."""
        from esdc.chat.app import TokenUsageWidget

        widget = TokenUsageWidget()
        assert widget.token_count == 0
        assert widget.context_length == 4096

    def test_token_usage_widget_update(self):
        """Test updating token count."""
        from esdc.chat.app import TokenUsageWidget

        widget = TokenUsageWidget(context_length=16384)
        widget.update_tokens(5432)
        assert widget.token_count == 5432

    def test_token_usage_widget_percentage(self):
        """Test percentage calculation."""
        from esdc.chat.app import TokenUsageWidget

        widget = TokenUsageWidget(context_length=16384)
        assert widget.get_percentage() == 0

        widget.update_tokens(8192)
        assert widget.get_percentage() == 50

    def test_token_usage_widget_format(self):
        """Test formatted display."""
        from esdc.chat.app import TokenUsageWidget

        widget = TokenUsageWidget(context_length=16384)
        widget.update_tokens(5432)

        formatted = widget.get_formatted()
        assert "5,432" in formatted
        assert "33%" in formatted or "34%" in formatted


class TestToolStatusList:
    """Tests for ToolStatusList widget."""

    def test_tool_status_list_creation(self):
        """Test ToolStatusList can be created."""
        from esdc.chat.app import ToolStatusList

        widget = ToolStatusList()
        assert len(widget.tools) == 3
        assert "execute_sql" in widget.tools

    def test_tool_status_list_mark_used(self):
        """Test marking tools as used."""
        from esdc.chat.app import ToolStatusList

        widget = ToolStatusList()
        widget.mark_used(["execute_sql", "get_schema"])

        assert widget.tools_used == ["execute_sql", "get_schema"]

    def test_tool_status_list_reset(self):
        """Test resetting used tools."""
        from esdc.chat.app import ToolStatusList

        widget = ToolStatusList()
        widget.mark_used(["execute_sql"])
        widget.reset_used()

        assert widget.tools_used == []


class TestQueryHistory:
    """Tests for QueryHistory widget."""

    def test_query_history_creation(self):
        """Test QueryHistory can be created."""
        from esdc.chat.app import QueryHistory

        widget = QueryHistory()
        assert widget.queries == []

    def test_query_history_add(self):
        """Test adding queries."""
        from esdc.chat.app import QueryHistory

        widget = QueryHistory(max_queries=5)
        widget.add_query("SELECT * FROM table1")
        widget.add_query("SELECT name FROM table2")

        assert len(widget.queries) == 2
        assert widget.queries[0] == "SELECT * FROM table1"

    def test_query_history_limit(self):
        """Test query history limit."""
        from esdc.chat.app import QueryHistory

        widget = QueryHistory(max_queries=3)
        widget.add_query("query 1")
        widget.add_query("query 2")
        widget.add_query("query 3")
        widget.add_query("query 4")

        assert len(widget.queries) == 3
        assert widget.queries[0] == "query 2"

    def test_query_history_clear(self):
        """Test clearing history."""
        from esdc.chat.app import QueryHistory

        widget = QueryHistory()
        widget.add_query("query 1")
        widget.clear()

        assert widget.queries == []


class TestContextPanel:
    """Tests for ContextPanel widget (session info only)."""

    def test_context_panel_creation(self):
        """Test ContextPanel can be created."""
        from esdc.chat.app import ContextPanel

        panel = ContextPanel()
        assert panel is not None

    def test_context_panel_has_session_info(self):
        """Test ContextPanel has session info attributes."""
        from esdc.chat.app import ContextPanel

        panel = ContextPanel()
        assert hasattr(panel, "_provider_name")
        assert hasattr(panel, "_model_name")
        assert hasattr(panel, "_session_thread_id")

    def test_context_panel_update_session_info(self):
        """Test updating session info."""
        from esdc.chat.app import ContextPanel

        panel = ContextPanel()
        panel.update_session_info("ollama", "llama3.2", "test-thread-123")

        assert panel._provider_name == "ollama"
        assert panel._model_name == "llama3.2"
        assert panel._session_thread_id == "test-thread-123"


class TestESDCChatAppComposition:
    """Integration tests for ESDCChatApp compose method."""

    def test_app_compose_does_not_raise_mount_error(self):
        """Test that compose() doesn't raise MountError."""
        from esdc.chat.app import ESDCChatApp

        app = ESDCChatApp()

        # This should not raise MountError
        try:
            result = list(app.compose())
            # Verify we get the expected widgets
            assert len(result) == 2
            # First should be Horizontal (main-content)
            assert result[0].id == "main-content"
            # Second should be Footer (contains Input and StatusBar)
            assert result[1].__class__.__name__ == "Footer"
        except Exception as e:
            pytest.fail(f"compose() raised {type(e).__name__}: {e}")

    def test_app_compose_returns_correct_widgets(self):
        """Test that compose returns expected widget types."""
        from esdc.chat.app import ESDCChatApp, Footer

        app = ESDCChatApp()
        result = list(app.compose())

        # Check we have 2 items: main content, footer
        assert len(result) == 2

        # First should be Horizontal (main-content)
        assert result[0].id == "main-content"
        # Second should be Footer (contains Input and StatusBar)
        assert isinstance(result[1], Footer)
