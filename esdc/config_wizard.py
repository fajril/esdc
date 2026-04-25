"""Interactive configuration wizard using questionary.

Provides a unified, streamlined CLI for managing all ESDC configuration
— providers, general settings, and everything else stored in config.yaml.
"""

# Standard library
import logging
import traceback
from typing import Any

# Third-party
import questionary
from questionary import Style
from rich import print as rich_print
from rich.panel import Panel

# Local
from esdc.configs import ENUM_CHOICES, KEY_DESCRIPTIONS, SENSITIVE_KEYS, Config
from esdc.providers import PROVIDER_CLASSES, PROVIDER_NAMES, get_provider
from esdc.providers.base import ProviderConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Professional color palette (Blue Grey family with soft accents)
# ---------------------------------------------------------------------------
_WIZARD_STYLE = Style(
    [
        ("qmark", "fg:#546e7a bold"),  # question mark
        ("question", "fg:#263238 bold"),  # question text
        ("answer", "fg:#29b6f6 bold"),  # selected answer
        ("pointer", "fg:#29b6f6 bold"),  # arrow cursor
        ("highlighted", "fg:#eceff1 bg:#37474f"),  # highlighted option
        ("separator", "fg:#78909c"),  # separator text
        ("instruction", "fg:#90a4ae"),  # instructions
    ]
)

# Rich markup palette for headers and messages
_KEY_COLOR = "#78909a"  # blue grey 400 — calm labels
_VALUE_COLOR = "#eceff1"  # blue grey 50 — soft white values
_SEP_COLOR = "#455a64"  # blue grey 700 — muted separators
_HEADER_BORDER = "#546e7a"  # blue grey 600 — panel border
_SUCCESS_COLOR = "#66bb6a"  # green 400
_WARNING_COLOR = "#ffb74d"  # orange 300
_ERROR_COLOR = "#ef5350"  # red 400


# ---------------------------------------------------------------------------
# Provider field schema – which fields to prompt for each provider type
# ---------------------------------------------------------------------------
_PROVIDER_FIELDS: dict[str, list[str]] = {
    "ollama": ["base_url", "model"],
    "openai": ["api_key", "model", "reasoning_effort"],
    "openai_compatible": ["base_url", "api_key", "model"],
    "anthropic": ["api_key", "model", "reasoning_effort"],
    "google": ["api_key", "model"],
    "azure_openai": ["base_url", "api_key", "model", "api_version"],
    "groq": ["api_key", "model"],
}


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------
def _mask_value(key: str, value: Any) -> str:
    """Mask sensitive values in the UI."""
    for sensitive in SENSITIVE_KEYS:
        if key.endswith(sensitive):
            s = str(value)
            if len(s) <= 4:
                return "****"
            return s[:2] + "****" + s[-2:]
    return str(value)


def _fetch_models(provider_type: str, **kwargs) -> list[str]:
    """Fetch model list from a provider class.

    Falls back to the provider's default model if the API call fails or
    returns no results. Passes kwargs (base_url, api_key) to support
    providers that require them for listing models.
    """
    provider_class = PROVIDER_CLASSES.get(provider_type)
    if provider_class is None:
        return []

    try:
        models = provider_class.list_models(**kwargs)
        if models:
            return models
    except Exception as exc:
        logger.debug("Failed to fetch models for %s: %s", provider_type, exc)

    default_model = provider_class.get_default_model()
    return [default_model] if default_model else []


def _resolve_default_field(provider_type: str, field: str) -> str:
    """Return a sensible default value for a provider field."""
    provider_class = PROVIDER_CLASSES.get(provider_type)
    if field == "base_url" and provider_class is not None:
        return getattr(provider_class, "BASE_URL", "")
    if field == "model" and provider_class is not None:
        return provider_class.get_default_model()
    return ""


def _find_provider_by_type(provider_type: str) -> str | None:
    """Find existing provider by type.

    Returns provider name if a provider of this type already exists.
    openai_compatible is exempt (always multiple).
    Returns None if no existing provider found.
    """
    if provider_type == "openai_compatible":
        return None

    providers = Config.get_providers()
    for name, cfg in providers.items():
        if cfg.get("provider_type") == provider_type:
            return name
    return None


def _check_default_provider_warning() -> None:
    """Show warning if no default provider is set but providers exist."""
    providers = Config.get_providers()
    if not providers:
        return

    default = Config.get_default_provider()
    if not default:
        rich_print(
            f"[{_WARNING_COLOR}]Warning: No default provider set. "
            f"Chat will not work until you set a default provider.[/{_WARNING_COLOR}]"
        )


def _prompt_for_config_value(key: str, current: Any) -> Any:
    """Prompt the user for a single config value using the best widget."""
    # 1. Boolean key?
    if key in Config.BOOLEAN_KEYS or isinstance(current, bool):
        selected = questionary.select(
            f"[{_KEY_COLOR}]{key}[/{_KEY_COLOR}] — "
            f"[{_VALUE_COLOR}]{KEY_DESCRIPTIONS.get(key, '')}[/{_VALUE_COLOR}]",
            choices=["True", "False"],
            default="True" if current else "False",
            style=_WIZARD_STYLE,
        ).ask()
        if selected is None:
            return None
        return selected == "True"

    # 2. Enum key?
    if key in ENUM_CHOICES:
        choices = ENUM_CHOICES[key]
        default = str(current) if str(current) in choices else choices[0]
        selected = questionary.select(
            f"[{_KEY_COLOR}]{key}[/{_KEY_COLOR}] — "
            f"[{_VALUE_COLOR}]{KEY_DESCRIPTIONS.get(key, '')}[/{_VALUE_COLOR}]",
            choices=choices,
            default=default,
            style=_WIZARD_STYLE,
        ).ask()
        return selected

    # 3. Integer key?
    if key in Config.INT_KEYS:
        default_str = str(current)
        new_str = questionary.text(
            f"[{_KEY_COLOR}]{key}[/{_KEY_COLOR}] — "
            f"[{_VALUE_COLOR}]{KEY_DESCRIPTIONS.get(key, '')}[/{_VALUE_COLOR}]",
            default=default_str,
            style=_WIZARD_STYLE,
        ).ask()
        return int(new_str) if new_str is not None else None

    # 4. Sensitive key?
    if any(s in key for s in SENSITIVE_KEYS):
        new = questionary.password(
            f"[{_KEY_COLOR}]{key}[/{_KEY_COLOR}] — "
            f"[{_VALUE_COLOR}]{KEY_DESCRIPTIONS.get(key, '')}[/{_VALUE_COLOR}]",
            style=_WIZARD_STYLE,
        ).ask()
        return new

    # 5. Database / path key?
    if key == "database_path":
        default = str(current)
        new = questionary.path(
            f"[{_KEY_COLOR}]{key}[/{_KEY_COLOR}] — "
            f"[{_VALUE_COLOR}]{KEY_DESCRIPTIONS.get(key, '')}[/{_VALUE_COLOR}]",
            default=default,
            only_directories=False,
            style=_WIZARD_STYLE,
        ).ask()
        return new

    # 6. Generic text key
    default = str(current)
    new = questionary.text(
        f"[{_KEY_COLOR}]{key}[/{_KEY_COLOR}] — "
        f"[{_VALUE_COLOR}]{KEY_DESCRIPTIONS.get(key, '')}[/{_VALUE_COLOR}]",
        default=default,
        style=_WIZARD_STYLE,
    ).ask()
    return new


# ---------------------------------------------------------------------------
# Provider CRUD flows
# ---------------------------------------------------------------------------
def _add_provider_flow() -> None:
    """Interactive wizard to add a new provider."""
    provider_type = questionary.select(
        "Select provider type:",
        choices=[
            questionary.Choice(title, value=key)
            for key, title in PROVIDER_NAMES.items()
        ],
        style=_WIZARD_STYLE,
    ).ask()
    if not provider_type:
        return

    name = questionary.text(
        "Provider name:", default=provider_type, style=_WIZARD_STYLE
    ).ask()
    if not name:
        return

    config_data: dict[str, Any] = {"provider_type": provider_type}
    fields = _PROVIDER_FIELDS.get(provider_type, ["model"])

    # Pass 1: collect connection fields first (base_url, api_key, api_version)
    for field in fields:
        if field == "api_key":
            value = questionary.password(
                "API Key (optional for local):", style=_WIZARD_STYLE
            ).ask()
            if value is None:
                value = ""
            config_data["api_key"] = value
        elif field == "base_url":
            default = _resolve_default_field(provider_type, "base_url")
            value = questionary.text(
                "Base URL:", default=default, style=_WIZARD_STYLE
            ).ask()
            if value is not None:
                config_data["base_url"] = value
        elif field == "api_version":
            default = _resolve_default_field(provider_type, "api_version")
            value = questionary.text(
                "API Version:",
                default=default or "2024-02-01",
                style=_WIZARD_STYLE,
            ).ask()
            if value is not None:
                config_data["api_version"] = value
        elif field == "reasoning_effort":
            effort = questionary.select(
                "Reasoning effort (optional):",
                choices=["(none)", "low", "medium", "high"],
                default="(none)",
                style=_WIZARD_STYLE,
            ).ask()
            if effort and effort != "(none)":
                config_data["reasoning_effort"] = effort

    # Pass 2: fetch models (needs base_url/api_key for some providers)
    if "model" in fields:
        rich_print(f"[{_SEP_COLOR}]Fetching models…[/{_SEP_COLOR}]")
        # Strip provider_type from kwargs to avoid "got multiple values" error
        list_kwargs = {k: v for k, v in config_data.items() if k != "provider_type"}
        models = _fetch_models(provider_type, **list_kwargs)
        if not models:
            models = ["(manual entry)"]
        default_model = _resolve_default_field(provider_type, "model")
        selected = questionary.select(
            "Select model:",
            choices=models,
            default=default_model,
            style=_WIZARD_STYLE,
        ).ask()
        if selected == "(manual entry)":
            selected = questionary.text(
                "Model name:",
                default=default_model,
                style=_WIZARD_STYLE,
            ).ask()
        config_data["model"] = selected or default_model

    # Check for existing provider of same type (openai can have max 1)
    existing = _find_provider_by_type(provider_type)
    if existing is not None:
        overwrite = questionary.confirm(
            f"Provider '[{_KEY_COLOR}]{existing}[/{_KEY_COLOR}]' "
            f"([{_VALUE_COLOR}]{provider_type}[/{_VALUE_COLOR}]) "
            "already exists. "
            "Overwrite?",
            default=False,
            style=_WIZARD_STYLE,
        ).ask()
        if not overwrite:
            rich_print(
                f"[{_SEP_COLOR}]Cancelled. Existing provider kept.[/{_SEP_COLOR}]"
            )
            return
        Config.remove_provider(existing)
        rich_print(
            f"[{_SEP_COLOR}]Removing '[{_KEY_COLOR}]{existing}[/{_KEY_COLOR}]' "
            f"to replace with '[{_VALUE_COLOR}]{name}[/{_VALUE_COLOR}]'.[/{_SEP_COLOR}]"
        )

    Config.save_provider(name, config_data)
    rich_print(
        f"[{_SUCCESS_COLOR}]Provider '[{_VALUE_COLOR}]{name}[/{_VALUE_COLOR}]' "
        f"saved.[/{_SUCCESS_COLOR}]"
    )

    # Auto-set default if first provider
    default = Config.get_default_provider()
    if not default:
        Config.set_default_provider(name)
        rich_print(
            f"[{_SUCCESS_COLOR}]Set '[{_VALUE_COLOR}]{name}[/{_VALUE_COLOR}]' "
            f"as default provider (first provider).[/{_SUCCESS_COLOR}]"
        )
    elif questionary.confirm(
        "Set as default provider?", default=False, style=_WIZARD_STYLE
    ).ask():
        Config.set_default_provider(name)

    if questionary.confirm("Test connection now?", default=False).ask():
        _test_provider(name, config_data)


def _test_provider_standalone() -> None:
    """Test connection for any existing provider (standalone flow)."""
    providers = Config.get_providers()
    if not providers:
        rich_print(f"[{_WARNING_COLOR}]No providers configured.[/{_WARNING_COLOR}]")
        return

    choices = [
        questionary.Choice(f"{name} ({cfg.get('provider_type')})", value=name)
        for name, cfg in providers.items()
    ]
    choices.append(questionary.Choice("  — Cancel", value="__cancel__"))

    selected = questionary.select(
        "Select provider to test:",
        choices=choices,
        style=_WIZARD_STYLE,
    ).ask()
    if selected is None or selected == "__cancel__":
        return

    cfg = providers[selected]
    _test_provider(selected, cfg)


def _test_provider(name: str, config_data: dict[str, Any]) -> None:
    """Test provider connection and print result."""
    provider_type = config_data.get("provider_type", "")
    provider_class = get_provider(provider_type)
    if provider_class is None:
        rich_print(
            f"[{_ERROR_COLOR}]Unknown provider type: "
            f"'[{_VALUE_COLOR}]{provider_type}[/{_VALUE_COLOR}]'[/{_ERROR_COLOR}]"
        )
        return

    test_config = ProviderConfig(
        name=name,
        provider_type=provider_type,
        model=config_data.get("model", ""),
        base_url=config_data.get("base_url", ""),
        api_key=config_data.get("api_key", ""),
    )
    rich_print(
        f"[{_SEP_COLOR}]Testing '[{_KEY_COLOR}]{provider_type}[/{_KEY_COLOR}]' "
        f"provider '[{_VALUE_COLOR}]{name}[/{_VALUE_COLOR}]'…[/{_SEP_COLOR}]"
    )
    success, message = provider_class.test_connection(test_config)
    if success:
        rich_print(f"[{_SUCCESS_COLOR}]Success: {message}[/{_SUCCESS_COLOR}]")
    else:
        rich_print(f"[{_ERROR_COLOR}]Failed: {message}[/{_ERROR_COLOR}]")


def _edit_provider_flow() -> None:
    """Edit an existing provider."""
    providers = Config.get_providers()
    if not providers:
        rich_print("[yellow]No providers configured.[/yellow]")
        return

    choices = [
        questionary.Choice(f"{name} ({cfg.get('provider_type')})", value=name)
        for name, cfg in providers.items()
    ]
    choices.append(questionary.Choice("← Back", value="__back__"))

    selected = questionary.select("Select provider to edit:", choices=choices).ask()
    if selected is None or selected == "__back__":
        return

    cfg = dict(providers[selected])
    provider_type = cfg.get("provider_type", "")

    # Determine editable fields
    editable = {"model": cfg.get("model", "")}
    if "api_key" in cfg:
        editable["api_key"] = cfg["api_key"]
    if "base_url" in cfg:
        editable["base_url"] = cfg["base_url"]

    field_choices = [
        questionary.Choice(f"{k} = {_mask_value(k, v)}", value=k)
        for k, v in editable.items()
    ]
    field_choices.append(questionary.Choice("← Back", value="__back__"))

    field = questionary.select("Select field to edit:", choices=field_choices).ask()
    if field is None or field == "__back__":
        return

    if field == "model":
        models = _fetch_models(provider_type)
        default = cfg.get("model", "")
        new_model = questionary.select(
            "Select model:", choices=models, default=default
        ).ask()
        if new_model is not None:
            cfg["model"] = new_model
    elif field == "api_key":
        new_key = questionary.password("API Key:").ask()
        if new_key is not None:
            cfg["api_key"] = new_key
    elif field == "base_url":
        default = cfg.get("base_url", "")
        new_url = questionary.text("Base URL:", default=default).ask()
        if new_url is not None:
            cfg["base_url"] = new_url

    Config.save_provider(selected, cfg)
    rich_print(f"[green]Provider '{selected}' updated.[/green]")


def _remove_provider_flow() -> None:
    """Remove a provider."""
    providers = Config.get_providers()
    if not providers:
        rich_print("[yellow]No providers configured.[/yellow]")
        return

    choices = [
        questionary.Choice(f"{name} ({cfg.get('provider_type')})", value=name)
        for name, cfg in providers.items()
    ]
    choices.append(questionary.Choice("← Back", value="__back__"))

    selected = questionary.select("Select provider to remove:", choices=choices).ask()
    if selected is None or selected == "__back__":
        return

    if questionary.confirm(f"Remove provider '{selected}'?", default=False).ask():
        if Config.remove_provider(selected):
            rich_print(f"[green]Provider '{selected}' removed.[/green]")
        else:
            rich_print(f"[red]Provider '{selected}' not found.[/red]")


def _set_default_provider_flow() -> None:
    """Set the default provider."""
    providers = Config.get_providers()
    if not providers:
        rich_print("[yellow]No providers configured.[/yellow]")
        return

    choices = [
        questionary.Choice(f"{name} ({cfg.get('provider_type')})", value=name)
        for name, cfg in providers.items()
    ]
    choices.append(questionary.Choice("← Back", value="__back__"))

    selected = questionary.select("Set default provider:", choices=choices).ask()
    if selected is None or selected == "__back__":
        return

    Config.set_default_provider(selected)
    rich_print(f"[green]Default provider set to '{selected}'.[/green]")


# ---------------------------------------------------------------------------
# General config flows
# ---------------------------------------------------------------------------
def _edit_general_config_flow() -> None:
    """Edit general (non-provider) configuration."""
    while True:
        flat = Config.get_all_config_flat()
        if not flat:
            rich_print("[yellow]No configuration found.[/yellow]")
            break

        # Group keys by top-level section
        groups: dict[str, list[str]] = {}
        for key in sorted(flat.keys()):
            section = key.split(".")[0]
            groups.setdefault(section, []).append(key)

        choices = []
        for section in sorted(groups.keys()):
            choices.append(questionary.Separator(f"── {section} ──"))
            for key in groups[section]:
                value = _mask_value(key, flat[key])
                desc = KEY_DESCRIPTIONS.get(key, "")
                label = f"{key} = {value}"
                if desc:
                    label += f" — {desc}"
                choices.append(questionary.Choice(label, value=key))
        choices.append(questionary.Separator("── Actions ──"))
        choices.append(questionary.Choice("← Back", value="__back__"))

        selected = questionary.select("Select setting to edit:", choices=choices).ask()
        if selected is None or selected == "__back__":
            break

        current = flat[selected]
        new_value = _prompt_for_config_value(selected, current)
        if new_value is None:
            rich_print("[dim]Edit cancelled, no changes made.[/dim]")
            continue

        Config.set_config_value(selected, new_value)
        rich_print(
            f"[green]Saved '{selected}' = {_mask_value(selected, new_value)}[/green]"
        )
        continue


def _show_config_flow() -> None:
    """Print all configuration flat keys (masked)."""
    flat = Config.get_all_config_flat()
    if not flat:
        rich_print(f"[{_WARNING_COLOR}]No configuration found.[/{_WARNING_COLOR}]")
        return

    for key in sorted(flat.keys()):
        value = _mask_value(key, flat[key])
        rich_print(
            f"  [{_KEY_COLOR}]{key}[/{_KEY_COLOR}] "
            f"[{_SEP_COLOR}]=[/{_SEP_COLOR}] "
            f"[{_VALUE_COLOR}]{value}[/{_VALUE_COLOR}]"
        )
    print()


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------
def run_wizard() -> None:
    """Run the interactive configuration wizard."""
    Config.init_config()

    _check_default_provider_warning()

    while True:
        rich_print(
            Panel.fit(
                "[bold]ESDC Configuration Manager[/bold]",
                border_style=_HEADER_BORDER,
                padding=(1, 4),
            )
        )

        selected = questionary.select(
            "What would you like to do?",
            choices=[
                questionary.Separator("Provider Management"),
                questionary.Choice("  — Add new provider", value="Add provider"),
                questionary.Choice("  — Edit provider", value="Edit provider"),
                questionary.Choice("  — Remove provider", value="Remove provider"),
                questionary.Choice(
                    "  — Set default provider", value="Set default provider"
                ),
                questionary.Choice("  — Test connection", value="Test connection"),
                questionary.Separator("Configuration"),
                questionary.Choice(
                    "  — Edit general settings", value="Edit general config"
                ),
                questionary.Choice("  — Show full config", value="Show config"),
                questionary.Separator(""),
                questionary.Choice("Exit", value="Exit"),
            ],
            style=_WIZARD_STYLE,
        ).ask()

        if selected is None or selected == "Exit":
            break

        try:
            if selected == "Add provider":
                _add_provider_flow()
            elif selected == "Edit general config":
                _edit_general_config_flow()
            elif selected == "Edit provider":
                _edit_provider_flow()
            elif selected == "Remove provider":
                _remove_provider_flow()
            elif selected == "Set default provider":
                _set_default_provider_flow()
            elif selected == "Test connection":
                _test_provider_standalone()
            elif selected == "Show config":
                _show_config_flow()
        except KeyboardInterrupt:
            break
        except Exception:
            traceback.print_exc()
            rich_print("[red]An error occurred. Please try again.[/red]")

    rich_print("[dim]Goodbye![/dim]")
