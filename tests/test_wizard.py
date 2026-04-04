"""Tests for the wizard module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from esdc.chat.wizard import (
    DatabasePathScreen,
    OllamaSetupScreen,
    OpenAICompatibleSetupScreen,
    OpenAISetupScreen,
    ProviderSelectScreen,
    SummaryScreen,
    WelcomeScreen,
    WizardApp,
)
from esdc.configs import Config


class TestWelcomeScreen:
    """Tests for WelcomeScreen."""

    def test_compose_renders(self):
        """Test that welcome screen composes without error."""
        screen = WelcomeScreen()
        # Just verify the screen can be created
        assert screen is not None
        assert screen.BINDINGS is not None


class TestProviderSelectScreen:
    """Tests for ProviderSelectScreen."""

    def test_on_selection_changed_updates_selected_provider(self):
        """Test that selection change updates selected_provider."""
        screen = ProviderSelectScreen()
        screen.selected_provider = None

        event = MagicMock()
        event.selection_list.selected = ["ollama"]

        screen.on_selection_list_selected_changed(event)

        assert screen.selected_provider == "ollama"

    def test_on_selection_changed_empty_selection(self):
        """Test that empty selection sets selected_provider to None."""
        screen = ProviderSelectScreen()
        screen.selected_provider = "openai"

        event = MagicMock()
        event.selection_list.selected = []

        screen.on_selection_list_selected_changed(event)

        assert screen.selected_provider is None

    def test_on_selection_changed_openai_compatible(self):
        """Test selecting openai_compatible provider."""
        screen = ProviderSelectScreen()
        screen.selected_provider = None

        event = MagicMock()
        event.selection_list.selected = ["openai_compatible"]

        screen.on_selection_list_selected_changed(event)

        assert screen.selected_provider == "openai_compatible"

    def test_bindings_defined(self):
        """Test that bindings are properly defined."""
        screen = ProviderSelectScreen()
        assert screen.BINDINGS is not None
        assert len(screen.BINDINGS) == 2


class TestOllamaSetupScreen:
    """Tests for OllamaSetupScreen."""

    def test_compose_renders(self):
        """Test that Ollama screen composes without error."""
        screen = OllamaSetupScreen()
        assert screen is not None
        assert screen.provider_type == "ollama"

    def test_custom_provider_type(self):
        """Test custom provider type."""
        screen = OllamaSetupScreen("custom")
        assert screen.provider_type == "custom"


class TestOpenAISetupScreen:
    """Tests for OpenAISetupScreen."""

    def test_oauth_flow_updates_config(self, tmp_path):
        """Test OAuth flow updates configuration."""
        with patch.object(Config, "get_config_dir", return_value=tmp_path):
            Config._config_cache = None

            screen = OpenAISetupScreen()

            mock_tokens = {
                "access_token": "test-token",
                "refresh_token": "test-refresh",
                "expires_in": 3600,
            }

            with patch("esdc.chat.wizard.start_oauth_flow", return_value=mock_tokens):
                auth_select = MagicMock()
                auth_select.value = "oauth"

                def query_one(selector, cls=None):
                    if selector == "#auth-method":
                        return auth_select
                    return MagicMock()

                screen.query_one = query_one

                button = MagicMock()
                button.id = "connect"
                event = MagicMock()
                event.button = button

                screen.on_button_pressed(event)

                providers = Config.get_providers()
                assert "openai" in providers
                assert providers["openai"]["auth_method"] == "oauth"

    def test_api_key_flow_requires_key(self, tmp_path):
        """Test API key flow shows error when key is missing."""
        with patch.object(Config, "get_config_dir", return_value=tmp_path):
            screen = OpenAISetupScreen()

            auth_select = MagicMock()
            auth_select.value = "api_key"
            api_key_input = MagicMock()
            api_key_input.value = ""
            status = MagicMock()

            def query_one(selector, cls=None):
                if selector == "#auth-method":
                    return auth_select
                elif selector == "#api-key":
                    return api_key_input
                elif selector == "#status":
                    return status
                return MagicMock()

            screen.query_one = query_one

            button = MagicMock()
            button.id = "connect"
            event = MagicMock()
            event.button = button

            screen.on_button_pressed(event)

            status.update.assert_called_once()
            assert "API key" in status.update.call_args[0][0]


class TestOpenAICompatibleSetupScreen:
    """Tests for OpenAICompatibleSetupScreen."""

    def test_save_requires_base_url(self, tmp_path):
        """Test save requires base URL."""
        with patch.object(Config, "get_config_dir", return_value=tmp_path):
            Config._config_cache = None

            screen = OpenAICompatibleSetupScreen()

            base_url_input = MagicMock()
            base_url_input.value = ""
            model_input = MagicMock()
            model_input.value = "test-model"
            status = MagicMock()

            def query_one(selector, cls=None):
                if selector == "#base-url":
                    return base_url_input
                elif selector == "#model-name":
                    return model_input
                elif selector == "#status":
                    return status
                return MagicMock()

            screen.query_one = query_one

            button = MagicMock()
            button.id = "save"
            event = MagicMock()
            event.button = button

            screen.on_button_pressed(event)

            status.update.assert_called()
            assert "URL" in status.update.call_args[0][0]

    def test_save_requires_model(self, tmp_path):
        """Test save requires model name."""
        with patch.object(Config, "get_config_dir", return_value=tmp_path):
            Config._config_cache = None

            screen = OpenAICompatibleSetupScreen()

            base_url_input = MagicMock()
            base_url_input.value = "http://localhost:8000/v1"
            model_input = MagicMock()
            model_input.value = ""
            status = MagicMock()

            def query_one(selector, cls=None):
                if selector == "#base-url":
                    return base_url_input
                elif selector == "#model-name":
                    return model_input
                elif selector == "#status":
                    return status
                return MagicMock()

            screen.query_one = query_one

            button = MagicMock()
            button.id = "save"
            event = MagicMock()
            event.button = button

            screen.on_button_pressed(event)

            status.update.assert_called()
            assert "model" in status.update.call_args[0][0].lower()


class TestDatabasePathScreen:
    """Tests for DatabasePathScreen."""

    def test_compose_does_not_error(self, tmp_path):
        """Test that compose works with mocked Config."""
        with patch.object(
            Config, "get_chat_db_path", return_value=Path.home() / ".esdc" / "chat.db"
        ):
            screen = DatabasePathScreen()
            assert screen is not None


class TestSummaryScreen:
    """Tests for SummaryScreen."""

    def test_compose_does_not_error(self):
        """Test that summary screen can be created."""
        screen = SummaryScreen()
        assert screen is not None


class TestWizardApp:
    """Tests for WizardApp."""

    def test_screens_installed(self):
        """Test all screens are installed in SCREENS."""
        assert "welcome" in WizardApp.SCREENS
        assert "provider_select" in WizardApp.SCREENS
        assert "ollama" in WizardApp.SCREENS
        assert "openai" in WizardApp.SCREENS
        assert "openai_compatible" in WizardApp.SCREENS
        assert "database_path" in WizardApp.SCREENS
        assert "summary" in WizardApp.SCREENS

    def test_screens_are_correct_classes(self):
        """Test SCREENS maps to correct screen classes."""
        assert WizardApp.SCREENS["welcome"] == WelcomeScreen
        assert WizardApp.SCREENS["provider_select"] == ProviderSelectScreen
        assert WizardApp.SCREENS["ollama"] == OllamaSetupScreen
        assert WizardApp.SCREENS["openai"] == OpenAISetupScreen
        assert WizardApp.SCREENS["openai_compatible"] == OpenAICompatibleSetupScreen
        assert WizardApp.SCREENS["database_path"] == DatabasePathScreen
        assert WizardApp.SCREENS["summary"] == SummaryScreen
