from pathlib import Path

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal
from textual.screen import Screen
from textual.widgets import Button, Footer, Header, Input, Select, SelectionList, Static

from esdc.auth import start_oauth_flow
from esdc.configs import Config
from esdc.providers import PROVIDER_NAMES


class WelcomeScreen(Screen):
    """Welcome screen for the setup wizard."""

    BINDINGS = [
        Binding("enter", "next", "Next"),
        Binding("escape", "cancel", "Cancel"),
    ]

    def compose(self) -> ComposeResult:
        """Build the welcome screen UI."""
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
        """Navigate to the provider selection screen."""
        self.app.push_screen("provider_select")

    def action_cancel(self) -> None:
        """Exit the setup wizard."""
        self.app.exit()


class ProviderSelectScreen(Screen):
    """Screen for selecting the chat provider."""

    BINDINGS = [
        Binding("enter", "next", "Next"),
        Binding("escape", "back", "Back"),
    ]

    selected_provider: str | None = None

    def compose(self) -> ComposeResult:
        """Build the provider selection UI."""
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
        """Capture the selected provider from the list."""
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
        """Handle button presses for navigation and provider selection."""
        if event.button.id == "back":
            self.app.pop_screen()
        elif event.button.id == "next":
            if self.selected_provider:
                self.app.push_screen(self.selected_provider)
            else:
                self.app.notify("Please select a provider", severity="warning")


class OllamaSetupScreen(Screen):
    """Screen for configuring Ollama provider."""

    BINDINGS = [
        Binding("escape", "back", "Back"),
    ]

    def __init__(self, provider_type: str = "ollama"):
        """Initialize Ollama setup screen.

        Args:
            provider_type: The type of provider to configure (default "ollama").
        """
        super().__init__()
        self.provider_type = provider_type

    def compose(self) -> ComposeResult:
        """Build the Ollama configuration UI."""
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
        """Handle button presses for navigation and saving configuration."""
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
    """Screen for configuring OpenAI provider."""

    BINDINGS = [
        Binding("escape", "back", "Back"),
    ]

    def __init__(self, provider_type: str = "openai"):
        """Initialize OpenAI setup screen.

        Args:
            provider_type: The type of provider to configure (default "openai").
        """
        super().__init__()
        self.provider_type = provider_type

    def compose(self) -> ComposeResult:
        """Build the OpenAI configuration UI."""
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
        """Handle button presses for navigation and connection."""
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
    """Screen for configuring OpenAI-compatible provider."""

    BINDINGS = [
        Binding("escape", "back", "Back"),
    ]

    def __init__(self, provider_type: str = "openai_compatible"):
        """Initialize OpenAI-compatible setup screen.

        Args:
            provider_type: The type of provider to configure
                (default "openai_compatible").
        """
        super().__init__()
        self.provider_type = provider_type

    def compose(self) -> ComposeResult:
        """Build the OpenAI-compatible configuration UI."""
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
        """Handle button presses for navigation, testing, and saving."""
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


class AnthropicSetupScreen(Screen):
    """Screen for configuring Anthropic provider."""

    BINDINGS = [
        Binding("escape", "back", "Back"),
    ]

    def __init__(self, provider_type: str = "anthropic"):
        """Initialize Anthropic setup screen.

        Args:
            provider_type: The type of provider to configure (default "anthropic").
        """
        super().__init__()
        self.provider_type = provider_type

    def compose(self) -> ComposeResult:
        """Build the Anthropic configuration UI."""
        from esdc.providers.anthropic import AnthropicProvider

        models = AnthropicProvider.list_models()
        default_model = AnthropicProvider.get_default_model()

        model_choices = (
            [(m, m) for m in models] if models else [(default_model, default_model)]
        )

        yield Container(
            Static("Configure Anthropic (Claude)", classes="title"),
            Static("API Key:", classes="label"),
            Input("", id="api-key", password=True),
            Select(
                model_choices,
                id="model-select",
                value=default_model,
            ),
            Static("", id="status"),
            Horizontal(
                Button("Back", variant="default", id="back"),
                Button("Test Connection", variant="default", id="test"),
                Button("Save", variant="success", id="save"),
            ),
            id="anthropic-setup",
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses for navigation, testing, and saving."""
        if event.button.id == "back":
            self.app.pop_screen()
        elif event.button.id == "test":
            api_key = self.query_one("#api-key", Input).value
            model = self.query_one("#model-select", Select).value
            status = self.query_one("#status", Static)

            from esdc.providers.anthropic import AnthropicProvider
            from esdc.providers.base import ProviderConfig

            test_config = ProviderConfig(
                name="test",
                provider_type="anthropic",
                api_key=api_key,
                model=str(model) if model else "",
            )
            success, message = AnthropicProvider.test_connection(test_config)
            status.update(message)
        elif event.button.id == "save":
            api_key = self.query_one("#api-key", Input).value
            model = self.query_one("#model-select", Select).value
            status = self.query_one("#status", Static)

            if not api_key:
                status.update("Please enter an API key")
                return

            config_data = {
                "provider_type": "anthropic",
                "api_key": api_key,
                "model": str(model) if model else "",
            }
            Config.update_provider_config("anthropic", config_data)
            Config.set_default_provider("anthropic")
            self.app.push_screen("database_path")


class GeminiSetupScreen(Screen):
    """Screen for configuring Google Gemini provider."""

    BINDINGS = [
        Binding("escape", "back", "Back"),
    ]

    def __init__(self, provider_type: str = "google"):
        """Initialize Gemini setup screen.

        Args:
            provider_type: The type of provider to configure (default "google").
        """
        super().__init__()
        self.provider_type = provider_type

    def compose(self) -> ComposeResult:
        """Build the Gemini configuration UI."""
        from esdc.providers.google import GoogleProvider

        models = GoogleProvider.list_models()
        default_model = GoogleProvider.get_default_model()

        model_choices = (
            [(m, m) for m in models] if models else [(default_model, default_model)]
        )

        yield Container(
            Static("Configure Google (Gemini)", classes="title"),
            Static("API Key:", classes="label"),
            Input("", id="api-key", password=True),
            Select(
                model_choices,
                id="model-select",
                value=default_model,
            ),
            Static("", id="status"),
            Horizontal(
                Button("Back", variant="default", id="back"),
                Button("Test Connection", variant="default", id="test"),
                Button("Save", variant="success", id="save"),
            ),
            id="gemini-setup",
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses for navigation, testing, and saving."""
        if event.button.id == "back":
            self.app.pop_screen()
        elif event.button.id == "test":
            api_key = self.query_one("#api-key", Input).value
            model = self.query_one("#model-select", Select).value
            status = self.query_one("#status", Static)

            from esdc.providers.base import ProviderConfig
            from esdc.providers.google import GoogleProvider

            test_config = ProviderConfig(
                name="test",
                provider_type="google",
                api_key=api_key,
                model=str(model) if model else "",
            )
            success, message = GoogleProvider.test_connection(test_config)
            status.update(message)
        elif event.button.id == "save":
            api_key = self.query_one("#api-key", Input).value
            model = self.query_one("#model-select", Select).value
            status = self.query_one("#status", Static)

            if not api_key:
                status.update("Please enter an API key")
                return

            config_data = {
                "provider_type": "google",
                "api_key": api_key,
                "model": str(model) if model else "",
            }
            Config.update_provider_config("google", config_data)
            Config.set_default_provider("google")
            self.app.push_screen("database_path")


class AzureOpenAISetupScreen(Screen):
    """Screen for configuring Azure OpenAI provider."""

    BINDINGS = [
        Binding("escape", "back", "Back"),
    ]

    def __init__(self, provider_type: str = "azure_openai"):
        """Initialize Azure OpenAI setup screen.

        Args:
            provider_type: The type of provider to configure (default "azure_openai").
        """
        super().__init__()
        self.provider_type = provider_type

    def compose(self) -> ComposeResult:
        """Build the Azure OpenAI configuration UI."""
        yield Container(
            Static("Configure Azure OpenAI", classes="title"),
            Static("Azure Endpoint:", classes="label"),
            Input("https://your-resource.openai.azure.com/", id="base-url"),
            Static("API Key:", classes="label"),
            Input("", id="api-key", password=True),
            Static("Deployment Name:", classes="label"),
            Input("gpt-4o", id="deployment-name"),
            Static("API Version:", classes="label"),
            Input("2024-02-01", id="api-version"),
            Static("", id="status"),
            Horizontal(
                Button("Back", variant="default", id="back"),
                Button("Test Connection", variant="default", id="test"),
                Button("Save", variant="success", id="save"),
            ),
            id="azure-setup",
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses for navigation, testing, and saving."""
        if event.button.id == "back":
            self.app.pop_screen()
        elif event.button.id == "test":
            base_url = self.query_one("#base-url", Input).value
            api_key = self.query_one("#api-key", Input).value
            deployment = self.query_one("#deployment-name", Input).value
            status = self.query_one("#status", Static)

            from esdc.providers.azure_openai import AzureOpenAIProvider
            from esdc.providers.base import ProviderConfig

            test_config = ProviderConfig(
                name="test",
                provider_type="azure_openai",
                base_url=base_url,
                api_key=api_key,
                model=deployment,
            )
            success, message = AzureOpenAIProvider.test_connection(test_config)
            status.update(message)
        elif event.button.id == "save":
            base_url = self.query_one("#base-url", Input).value
            api_key = self.query_one("#api-key", Input).value
            deployment = self.query_one("#deployment-name", Input).value
            api_version = self.query_one("#api-version", Input).value
            status = self.query_one("#status", Static)

            if not base_url:
                status.update("Please enter an Azure endpoint")
                return
            if not api_key:
                status.update("Please enter an API key")
                return
            if not deployment:
                status.update("Please enter a deployment name")
                return

            config_data = {
                "provider_type": "azure_openai",
                "base_url": base_url,
                "api_key": api_key,
                "model": deployment,
                "oauth": {"api_version": api_version},
            }
            Config.update_provider_config("azure_openai", config_data)
            Config.set_default_provider("azure_openai")
            self.app.push_screen("database_path")


class GroqSetupScreen(Screen):
    """Screen for configuring Groq provider."""

    BINDINGS = [
        Binding("escape", "back", "Back"),
    ]

    def __init__(self, provider_type: str = "groq"):
        """Initialize Groq setup screen.

        Args:
            provider_type: The type of provider to configure (default "groq").
        """
        super().__init__()
        self.provider_type = provider_type

    def compose(self) -> ComposeResult:
        """Build the Groq configuration UI."""
        from esdc.providers.groq import GroqProvider

        models = GroqProvider.list_models()
        default_model = GroqProvider.get_default_model()

        model_choices = (
            [(m, m) for m in models] if models else [(default_model, default_model)]
        )

        yield Container(
            Static("Configure Groq", classes="title"),
            Static("API Key:", classes="label"),
            Input("", id="api-key", password=True),
            Select(
                model_choices,
                id="model-select",
                value=default_model,
            ),
            Static("", id="status"),
            Horizontal(
                Button("Back", variant="default", id="back"),
                Button("Test Connection", variant="default", id="test"),
                Button("Save", variant="success", id="save"),
            ),
            id="groq-setup",
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses for navigation, testing, and saving."""
        if event.button.id == "back":
            self.app.pop_screen()
        elif event.button.id == "test":
            api_key = self.query_one("#api-key", Input).value
            model = self.query_one("#model-select", Select).value
            status = self.query_one("#status", Static)

            from esdc.providers.base import ProviderConfig
            from esdc.providers.groq import GroqProvider

            test_config = ProviderConfig(
                name="test",
                provider_type="groq",
                api_key=api_key,
                model=str(model) if model else "",
            )
            success, message = GroqProvider.test_connection(test_config)
            status.update(message)
        elif event.button.id == "save":
            api_key = self.query_one("#api-key", Input).value
            model = self.query_one("#model-select", Select).value
            status = self.query_one("#status", Static)

            if not api_key:
                status.update("Please enter an API key")
                return

            config_data = {
                "provider_type": "groq",
                "api_key": api_key,
                "model": str(model) if model else "",
            }
            Config.update_provider_config("groq", config_data)
            Config.set_default_provider("groq")
            self.app.push_screen("database_path")


class DatabasePathScreen(Screen):
    """Screen for configuring the database path."""

    BINDINGS = [
        Binding("escape", "back", "Back"),
    ]

    def compose(self) -> ComposeResult:
        """Build the database path configuration UI."""
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
        """Handle button presses for navigation and saving the database path."""
        if event.button.id == "back":
            self.app.pop_screen()
        elif event.button.id == "save":
            db_path = self.query_one("#db-path", Input).value
            Config.set_chat_db_path(Path(db_path))
            self.app.push_screen("summary")


class SummaryScreen(Screen):
    """Screen displaying the configuration summary."""

    BINDINGS = [
        Binding("escape", "quit", "Quit"),
    ]

    def compose(self) -> ComposeResult:
        """Build the configuration summary UI."""
        providers = Config.get_providers()
        default_provider = Config.get_default_provider()
        db_path = Config.get_chat_db_path()

        summary = "Configuration Summary:\n\n"
        summary += f"Default Provider: {default_provider}\n\n"
        summary += "Providers:\n"
        for name, config in providers.items():
            summary += f"  - {name}: {config.get('provider_type')} ({config.get('model', 'N/A')})\n"  # noqa: E501
        summary += f"\nDatabase: {db_path}"

        yield Container(
            Static("Setup Complete!", classes="title"),
            Static(summary, classes="summary"),
            Button("Start Chat", variant="primary", id="start"),
            id="summary",
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button press to start the chat application."""
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
        "anthropic": AnthropicSetupScreen,
        "google": GeminiSetupScreen,
        "azure_openai": AzureOpenAISetupScreen,
        "groq": GroqSetupScreen,
        "database_path": DatabasePathScreen,
        "summary": SummaryScreen,
    }

    WIZARD_CSS = """
    /* ===== Minimalist Wizard - Clean & Clear =====
       Occam's Razor: Simplest effective design
       Hick's Law: Minimize choices
       Law of Prägnanz: Simple, clear forms
    ================================================================ */

    Screen {
        background: $background;
        align: center middle;
    }

    Container {
        width: 60;
        height: auto;
        border: none;
        padding: 2 4;
        background: $background;
    }

    .title {
        text-align: center;
        text-style: bold;
        color: $text;
        margin-bottom: 2;
    }

    .content {
        text-align: center;
        color: $text;
        margin-bottom: 2;
    }

    .label {
        margin-top: 1;
        color: $text-muted;
    }

    .summary {
        width: 100%;
        background: transparent;
        padding: 1;
        border: none;
        color: $text;
    }

    #provider-select {
        margin: 2 0;
        padding: 1;
        border: none;
        background: transparent;
    }

    Select {
        margin: 1 0;
        border: solid $primary;
        background: $background;
    }

    Select:focus {
        border: solid $accent;
        background: $background;
    }

    Input {
        margin: 1 0;
        padding: 1 2;
        border: solid $primary;
        background: $background;
    }

    Input:focus {
        border: solid $accent;
        background: $background;
    }

    Button {
        margin: 1 1;
    }

    Horizontal {
        align: center middle;
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
