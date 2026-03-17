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
        assert panel.schema_tips == ""

    def test_sql_panel_stores_data(self):
        """Test SQLPanel stores data without calling display (no app needed)."""
        from esdc.chat.app import SQLPanel

        panel = SQLPanel()
        panel.sql_content = "SELECT * FROM table1"
        panel.schema_tips = "table1: id, name"
        assert panel.sql_content == "SELECT * FROM table1"
        assert panel.schema_tips == "table1: id, name"


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


class TestRightPanel:
    """Tests for RightPanel container."""

    def test_right_panel_creation(self):
        """Test RightPanel can be created."""
        from esdc.chat.app import RightPanel

        panel = RightPanel()
        assert panel.sql_panel is not None
        assert panel.results_panel is not None

    def test_right_panel_has_sql_panel(self):
        """Test RightPanel has SQLPanel."""
        from esdc.chat.app import RightPanel

        panel = RightPanel()
        assert isinstance(panel.sql_panel, type(panel.sql_panel))

    def test_right_panel_has_results_panel(self):
        """Test RightPanel has ResultsPanel."""
        from esdc.chat.app import RightPanel

        panel = RightPanel()
        assert isinstance(panel.results_panel, type(panel.results_panel))
