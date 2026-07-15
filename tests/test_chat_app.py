# tests/test_chat_app.py

import pytest


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
        assert panel is not None
        assert hasattr(panel, "messages")
        assert panel.messages == []

    def test_chat_panel_mount_collapsible(self):
        """Test ChatPanel can mount collapsible widgets."""
        from esdc.chat.app import ChatPanel, ThinkingIndicator

        panel = ChatPanel()
        # mount_collapsible method exists
        assert hasattr(panel, "mount_collapsible")

        # ThinkingIndicator is a Collapsible
        thinking = ThinkingIndicator()
        assert thinking is not None

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
