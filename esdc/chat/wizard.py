import os
from pathlib import Path
from typing import Any

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.screen import ModalScreen, Screen
from textual.widgets import Button, Header, Footer, Static, Input, SelectionList, Select

from esdc.auth import start_oauth_flow
from esdc.configs import Config
from esdc.providers import (
    PROVIDER_NAMES,
    get_provider,
    list_provider_types,
)


class WelcomeScreen(Screen):
    BINDINGS = [
        Binding("enter", "next", "Next"),
        Binding("escape", "cancel", "Cancel"),
    ]

    def compose(self) -> ComposeResult:
        yield Container(
            Static("Welcome to ESDC Chat Setup!", classes="title"),
            Static(
                "This wizard will help you configure your chat provider.\n\n"
                "Press Enter to continue or Escape to exit.",
                classes="content",
            ),
            id="welcome",
        )

    def action_next(self) -> None:
        self.app.push_screen("provider_select")

    def action_cancel(self) -> None:
        self.app.exit()


class ProviderSelectScreen(Screen):
    BINDINGS = [
        Binding("enter", "next", "Next"),
        Binding("escape", "back", "Back"),
    ]

    selected_provider: str | None = None

    def compose(self) -> ComposeResult:
        yield Container(
            Static("Select Provider", classes="title"),
            Static("Choose a provider for your chat:"),
            SelectionList(
                *[(name, id) for id, name in PROVIDER_NAMES.items()],
                id="provider-select",
            ),
            Horizontal(
                Button("Back", variant="default", id="back"),
                Button("Next", variant="primary", id="next"),
            ),
            id="provider-select-screen",
        )

    def on_selection_list_selected_changed(
        self, event: SelectionList.SelectedChanged
    ) -> None:
        self.selected_provider = (
            event.selection_list.selected[0] if event.selection_list.selected else None
        )

    def action_next(self) -> None:
        """Handle Enter key press."""
        if self.selected_provider:
            self.app.push_screen(self.selected_provider)
        else:
            self.app.notify("Please select a provider", severity="warning")

    def action_back(self) -> None:
        """Handle Escape key press."""
        self.app.pop_screen()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "back":
            self.app.pop_screen()
        elif event.button.id == "next":
            if self.selected_provider:
                self.app.push_screen(self.selected_provider)
            else:
                self.app.notify("Please select a provider", severity="warning")


class OllamaSetupScreen(Screen):
    BINDINGS = [
        Binding("escape", "back", "Back"),
    ]

    def __init__(self, provider_type: str = "ollama"):
        super().__init__()
        self.provider_type = provider_type

    def compose(self) -> ComposeResult:
        from esdc.providers.ollama import OllamaProvider

        models = OllamaProvider.list_models()
        default_model = OllamaProvider.get_default_model()

        model_choices = (
            [(m, m) for m in models] if models else [("llama3.2", "llama3.2")]
        )

        yield Container(
            Static("Configure Ollama", classes="title"),
            Static(
                f"Available models: {', '.join(models) if models else 'None found'}"
            ),
            Select(
                model_choices,
                id="model-select",
                value=default_model,
            ),
            Static("Base URL (optional):", classes="label"),
            Input("http://localhost:11434", id="base-url"),
            Horizontal(
                Button("Back", variant="default", id="back"),
                Button("Save", variant="success", id="save"),
            ),
            id="ollama-setup",
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "back":
            self.app.pop_screen()
        elif event.button.id == "save":
            model_select = self.query_one("#model-select", Select)
            base_url_input = self.query_one("#base-url", Input)

            config_data = {
                "provider_type": "ollama",
                "model": model_select.value or "llama3.2",
                "base_url": base_url_input.value or "http://localhost:11434",
            }

            Config.update_provider_config("ollama", config_data)
            Config.set_default_provider("ollama")
            self.app.push_screen("database_path")


class OpenAISetupScreen(Screen):
    BINDINGS = [
        Binding("escape", "back", "Back"),
    ]

    def __init__(self, provider_type: str = "openai"):
        super().__init__()
        self.provider_type = provider_type

    def compose(self) -> ComposeResult:
        yield Container(
            Static("Configure OpenAI", classes="title"),
            Static("Choose authentication method:"),
            Select(
                [
                    ("oauth", "OAuth (recommended - opens browser)"),
                    ("api_key", "API Key (for CI/automation)"),
                ],
                id="auth-method",
                value="oauth",
            ),
            Static("API Key (if using API key method):", classes="label"),
            Input("", id="api-key", password=True),
            Static("", id="status"),
            Horizontal(
                Button("Back", variant="default", id="back"),
                Button("Connect", variant="success", id="connect"),
            ),
            id="openai-setup",
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "back":
            self.app.pop_screen()
        elif event.button.id == "connect":
            auth_select = self.query_one("#auth-method", Select)
            api_key_input = self.query_one("#api-key", Input)
            status = self.query_one("#status", Static)

            if auth_select.value == "oauth":
                try:
                    tokens = start_oauth_flow()
                    config_data = {
                        "provider_type": "openai",
                        "model": "gpt-4o-mini",
                        "auth_method": "oauth",
                        "oauth": tokens,
                    }
                    Config.update_provider_config("openai", config_data)
                    Config.set_default_provider("openai")
                    self.app.push_screen("database_path")
                except Exception as e:
                    status.update(f"Error: {str(e)}")
            else:
                if not api_key_input.value:
                    status.update("Please enter an API key")
                    return

                config_data = {
                    "provider_type": "openai",
                    "model": "gpt-4o-mini",
                    "auth_method": "api_key",
                    "api_key": api_key_input.value,
                }
                Config.update_provider_config("openai", config_data)
                Config.set_default_provider("openai")
                self.app.push_screen("database_path")


class OpenAICompatibleSetupScreen(Screen):
    BINDINGS = [
        Binding("escape", "back", "Back"),
    ]

    def __init__(self, provider_type: str = "openai_compatible"):
        super().__init__()
        self.provider_type = provider_type

    def compose(self) -> ComposeResult:
        yield Container(
            Static("Configure OpenAI-Compatible Server", classes="title"),
            Static("Server URL:", classes="label"),
            Input("http://localhost:8000/v1", id="base-url"),
            Static("API Key (optional):", classes="label"),
            Input("", id="api-key", password=True),
            Static("Model name:", classes="label"),
            Input("", id="model-name"),
            Static("", id="status"),
            Horizontal(
                Button("Back", variant="default", id="back"),
                Button("Test Connection", variant="default", id="test"),
                Button("Save", variant="success", id="save"),
            ),
            id="compatible-setup",
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "back":
            self.app.pop_screen()
        elif event.button.id == "test":
            base_url = self.query_one("#base-url", Input).value
            api_key = self.query_one("#api-key", Input).value
            status = self.query_one("#status", Static)

            from esdc.providers.base import ProviderConfig
            from esdc.providers.openai_compatible import OpenAICompatibleProvider

            test_config = ProviderConfig(
                name="test",
                provider_type="openai_compatible",
                base_url=base_url,
                api_key=api_key,
                model="",
            )
            success, message = OpenAICompatibleProvider.test_connection(test_config)
            status.update(message)
        elif event.button.id == "save":
            base_url = self.query_one("#base-url", Input).value
            api_key = self.query_one("#api-key", Input).value
            model = self.query_one("#model-name", Input).value
            status = self.query_one("#status", Static)

            if not base_url:
                status.update("Please enter a server URL")
                return

            if not model:
                status.update("Please enter a model name")
                return

            config_data = {
                "provider_type": "openai_compatible",
                "base_url": base_url,
                "api_key": api_key,
                "model": model,
            }
            Config.update_provider_config("compatible", config_data)
            Config.set_default_provider("compatible")
            self.app.push_screen("database_path")


class DatabasePathScreen(Screen):
    BINDINGS = [
        Binding("escape", "back", "Back"),
    ]

    def compose(self) -> ComposeResult:
        db_path = Config.get_chat_db_path()

        yield Container(
            Static("Database Path", classes="title"),
            Static(f"Current path: {db_path}"),
            Static("Enter database path:", classes="label"),
            Input(str(db_path), id="db-path"),
            Horizontal(
                Button("Back", variant="default", id="back"),
                Button("Save", variant="success", id="save"),
            ),
            id="database-path",
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "back":
            self.app.pop_screen()
        elif event.button.id == "save":
            db_path = self.query_one("#db-path", Input).value
            Config.set_chat_db_path(Path(db_path))
            self.app.push_screen("summary")


class SummaryScreen(Screen):
    BINDINGS = [
        Binding("escape", "quit", "Quit"),
    ]

    def compose(self) -> ComposeResult:
        providers = Config.get_providers()
        default_provider = Config.get_default_provider()
        db_path = Config.get_chat_db_path()

        summary = "Configuration Summary:\n\n"
        summary += f"Default Provider: {default_provider}\n\n"
        summary += "Providers:\n"
        for name, config in providers.items():
            summary += f"  - {name}: {config.get('provider_type')} ({config.get('model', 'N/A')})\n"
        summary += f"\nDatabase: {db_path}"

        yield Container(
            Static("Setup Complete!", classes="title"),
            Static(summary, classes="summary"),
            Button("Start Chat", variant="primary", id="start"),
            id="summary",
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "start":
            self.app.exit(0)


class WizardApp:
    """Wizard application for setting up ESDC chat configuration."""

    SCREENS = {
        "welcome": WelcomeScreen,
        "provider_select": ProviderSelectScreen,
        "ollama": OllamaSetupScreen,
        "openai": OpenAISetupScreen,
        "openai_compatible": OpenAICompatibleSetupScreen,
        "database_path": DatabasePathScreen,
        "summary": SummaryScreen,
    }

    WIZARD_CSS = """
    Screen {
        background: $surface;
        align: center middle;
    }

    Container {
        width: 60;
        height: auto;
        border: solid $primary;
        padding: 1 2;
    }

    .title {
        text-align: center;
        text-style: bold;
        color: $accent;
        margin-bottom: 1;
    }

    .content {
        margin-bottom: 1;
    }

    .label {
        margin-top: 1;
    }

    .summary {
        width: 100%;
    }

    #provider-select {
        margin: 1 0;
    }
    """

    @staticmethod
    def run() -> None:
        """Run the wizard."""
        from textual.app import App

        class WizardAppInner(App):
            CSS_PATH = None
            CSS = WizardApp.WIZARD_CSS
            SCREENS = WizardApp.SCREENS

            def on_mount(self) -> None:
                self.push_screen(WelcomeScreen())

            def compose(self) -> ComposeResult:
                yield Header()
                yield Footer()

        app = WizardAppInner()
        app.run()


SCREEN_ROUTES = WizardApp.SCREENS
