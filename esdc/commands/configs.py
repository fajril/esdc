from __future__ import annotations

import typer
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal
from textual.screen import Screen
from textual.widgets import Button, Input, OptionList, Select, Static
from textual.widgets.option_list import Option

from esdc.configs import ENUM_CHOICES, KEY_DESCRIPTIONS, Config

configs_app = typer.Typer(invoke_without_command=False, no_args_is_help=False)

SENSITIVE_KEYS = frozenset({"api_key"})

SECTION_ORDER = [
    "api_url",
    "api",
    "database_path",
    "tool_format",
    "default_provider",
    "providers",
    "openwebui",
    "openterminal",
    "cache",
    "logging",
    "semantic_search",
]


def _section_for(key: str) -> int:
    top = key.split(".")[0]
    try:
        return SECTION_ORDER.index(top)
    except ValueError:
        return len(SECTION_ORDER)


def _mask_value(key: str, value: object) -> str:
    for sensitive in SENSITIVE_KEYS:
        if key.endswith(sensitive):
            s = str(value)
            if len(s) <= 4:
                return "****"
            return s[:2] + "****" + s[-2:]
    return str(value)


class ConfigEditScreen(Screen):
    """Screen for editing a single configuration value."""

    BINDINGS = [Binding("escape", "back", "Back")]

    def __init__(self, key: str) -> None:
        """Initialize the config command group."""
        super().__init__()
        self.key = key

    def compose(self) -> ComposeResult:
        """Compose the edit screen layout."""
        assert isinstance(self.app, ConfigsApp)
        value = self.app.config_flat.get(self.key, "")
        masked = _mask_value(self.key, value)
        description = KEY_DESCRIPTIONS.get(self.key, "")

        if self.key in Config.BOOLEAN_KEYS:
            editor: Input | Select = Select(
                [("True", "true"), ("False", "false")],
                value="true" if value else "false",
                id="val",
            )
        elif self.key in ENUM_CHOICES:
            choices = ENUM_CHOICES[self.key]
            str_val = str(value)
            editor = Select(
                [(c, c) for c in choices],
                value=str_val if str_val in choices else choices[0],
                id="val",
            )
        else:
            editor = Input(str(value), id="val", placeholder="New value...")

        yield Container(
            Static(f"[bold]{self.key}[/bold]", id="ek"),
            Static(f"Current: {masked}", id="ev"),
            Static(description, id="edesc") if description else Static("", id="edesc"),
            editor,
            Static("", id="es"),
            Horizontal(
                Button("Back", id="back"),
                Button("Reset Default", id="reset"),
                Button("Save", variant="success", id="save"),
            ),
            id="ed",
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button press events for Back, Reset, and Save."""
        bid = event.button.id
        if bid == "back":
            self.app.pop_screen()
        elif bid == "reset":
            self._reset()
        elif bid == "save":
            self._save()

    def action_back(self) -> None:
        """Navigate back to the config list."""
        self.app.pop_screen()

    def _reset(self) -> None:
        assert isinstance(self.app, ConfigsApp)
        try:
            Config.reset_config(self.key)
            Config._config_cache = None
            self.app.config_flat = Config.get_all_config_flat()
            new_val = self.app.config_flat.get(self.key, "")
            self.query_one("#es", Static).update(
                f"Reset to default: {_mask_value(self.key, new_val)}"
            )
        except KeyError as e:
            self.app.notify(str(e), severity="error")

    def _save(self) -> None:
        """Save the edited config value."""
        assert isinstance(self.app, ConfigsApp)
        if self.key in Config.BOOLEAN_KEYS:
            sel = self.query_one("#val", Select)
            if sel.value is None or sel.value == Select.BLANK:
                self.app.notify("Select a value", severity="warning")
                return
            new_value: str | bool = sel.value == "true"
        elif self.key in ENUM_CHOICES:
            sel = self.query_one("#val", Select)
            if sel.value is None or sel.value == Select.BLANK:
                self.app.notify("Select a value", severity="warning")
                return
            new_value = str(sel.value)
        else:
            new_value = self.query_one("#val", Input).value
            if not new_value.strip():
                self.app.notify("Value cannot be empty", severity="warning")
                return

        Config.set_config_value(self.key, new_value)
        Config._config_cache = None
        self.app.config_flat = Config.get_all_config_flat()
        self.app.pop_screen()


class ConfigListScreen(Screen):
    """Screen listing all configuration keys with values."""

    BINDINGS = [
        Binding("escape", "quit", "Quit"),
        Binding("q", "quit", "Quit"),
    ]

    def compose(self) -> ComposeResult:
        """Compose the config list screen layout."""
        assert isinstance(self.app, ConfigsApp)
        flat = self.app.config_flat
        keys = sorted(flat.keys(), key=lambda k: (_section_for(k), k))

        options = []
        for k in keys:
            label = f"{k} = {_mask_value(k, flat[k])}"
            desc = KEY_DESCRIPTIONS.get(k)
            if desc:
                label = f"{label}  —  {desc}"
            options.append(Option(label, id=k))
        options.append(Option("-- Done / Exit --", id="__exit__"))

        yield Container(
            Static("[bold]ESDC Configuration[/bold]"),
            Static("Arrow keys to navigate, Enter to edit, q/Esc to quit"),
            OptionList(*options, id="ol"),
            Horizontal(Button("Quit", id="quit")),
            id="cfg-list",
        )

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        """Handle option selection to navigate to edit screen."""
        if event.option.id == "__exit__":
            self.app.exit()
            return
        if event.option.id:
            self.app.push_screen(ConfigEditScreen(event.option.id))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle Quit button press."""
        if event.button.id == "quit":
            self.app.exit()

    def action_quit(self) -> None:
        """Quit the application."""
        self.app.exit()


class ConfigsApp(App):
    """Textual application for editing ESDC configuration."""

    TITLE = "ESDC Config Editor"
    CSS = """
#cfg-list, #ed {
    padding: 1 4;
    width: 90;
    max-width: 100%;
}
#ek { text-style: bold; margin-bottom: 1; }
#ev { color: $text-muted; margin-bottom: 1; }
#es { color: $success; margin-top: 1; }
#ol { margin: 1 0; height: auto; max-height: 20; }
Horizontal { align: center middle; height: 3; margin-top: 1; }
Button { margin: 0 1; }
"""

    def __init__(self, config_flat: dict | None = None) -> None:
        """Initialize the config setup interactive wizard."""
        super().__init__()
        self.config_flat: dict = config_flat or {}

    def on_mount(self) -> None:
        """Push the config list screen on mount."""
        self.push_screen(ConfigListScreen())


@configs_app.callback(invoke_without_command=True)
def configs(
    reset: bool = typer.Option(False, "--reset", help="Reset ALL config to defaults."),
    reset_key: str | None = typer.Option(
        None, "--reset-key", help="Reset a specific key to its default value."
    ),
):
    """View and edit ESDC configuration interactively.

    Run without flags to open the interactive editor.
    Use --reset to restore all defaults, or --reset-key <key> for a single key.
    """
    if reset:
        Config.reset_config()
        Config._config_cache = None
        typer.echo("All config reset to defaults.")
        return

    if reset_key:
        try:
            Config.reset_config(reset_key)
            Config._config_cache = None
            typer.echo(f"Reset '{reset_key}' to default.")
        except KeyError as e:
            typer.echo(f"Error: {e}", err=True)
        return

    Config.init_config()
    Config._config_cache = None
    flat = Config.get_all_config_flat()
    app = ConfigsApp(config_flat=flat)
    app.run()
