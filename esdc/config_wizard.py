"""Interactive configuration wizard using questionary.

Provides a unified, streamlined CLI for managing all ESDC configuration
— providers, general settings, and everything else stored in config.yaml.
"""

# Standard library
import contextlib
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
    "ollama_cloud": ["base_url", "api_key", "model"],
}


# ---------------------------------------------------------------------------
# Cancel / back helpers
# ---------------------------------------------------------------------------
class WizardCancelledError(Exception):
    """Raised when the user cancels (Ctrl+C or Esc) at any prompt."""


def _prompt(question: Any) -> Any:
    """Wrapper around a questionary prompt that raises WizardCancelledError on cancel.

    questionary's ``.ask()`` returns ``None`` when the user hits Ctrl+C or Esc.
    We treat that as a request to exit the entire wizard.
    """
    result = question.ask()
    if result is None:
        raise WizardCancelledError
    return result


def _select_with_back(
    message: str,
    choices: list,
    default: Any = None,
) -> Any:
    """Prompt a *select* with a leading '← Back' option.

    Returns the selected value (including ``"__back__"`` when the user picks
    *Back*).  A ``None`` / Ctrl+C / Esc raises :class:`WizardCancelledError`.
    """
    all_choices = [questionary.Choice("← Back", value="__back__")] + list(choices)
    return _prompt(
        questionary.select(
            message,
            choices=all_choices,
            default=default,
            style=_WIZARD_STYLE,
        )
    )


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
    provider_type = _select_with_back(
        "Select provider type:",
        choices=[
            questionary.Choice(title, value=key)
            for key, title in PROVIDER_NAMES.items()
        ],
    )
    if provider_type == "__back__":
        return

    name = provider_type
    config_data: dict[str, Any] = {"provider_type": provider_type}
    fields = _PROVIDER_FIELDS.get(provider_type, ["model"])

    # Pass 1: collect connection fields first (base_url, api_key, api_version)
    for field in fields:
        if field == "api_key":
            value = _prompt(
                questionary.password(
                    "API Key (optional for local):", style=_WIZARD_STYLE
                )
            )
            config_data["api_key"] = value
        elif field == "base_url":
            default = _resolve_default_field(provider_type, "base_url")
            value = _prompt(
                questionary.text("Base URL:", default=default, style=_WIZARD_STYLE)
            )
            config_data["base_url"] = value
        elif field == "api_version":
            default = _resolve_default_field(provider_type, "api_version")
            value = _prompt(
                questionary.text(
                    "API Version:",
                    default=default or "2024-02-01",
                    style=_WIZARD_STYLE,
                )
            )
            config_data["api_version"] = value
        elif field == "reasoning_effort":
            effort = _select_with_back(
                "Reasoning effort (optional):",
                choices=[
                    questionary.Choice("(none)", value="(none)"),
                    questionary.Choice("low", value="low"),
                    questionary.Choice("medium", value="medium"),
                    questionary.Choice("high", value="high"),
                ],
                default="(none)",
            )
            if effort == "__back__":
                return
            if effort != "(none)":
                config_data["reasoning_effort"] = effort

    # Pass 2: fetch models (needs base_url/api_key for some providers)
    if "model" in fields:
        rich_print(f"[{_SEP_COLOR}]Fetching models…[/{_SEP_COLOR}]")
        if provider_type == "openai_compatible":
            base = config_data.get("base_url", "")
            if base and not base.rstrip("/").endswith("/v1"):
                config_data["base_url"] = base.rstrip("/") + "/v1"
        list_kwargs = {k: v for k, v in config_data.items() if k != "provider_type"}
        models = _fetch_models(provider_type, **list_kwargs)
        if not models:
            models = ["(manual entry)"]
        default_model = _resolve_default_field(provider_type, "model")
        if default_model not in models and models:
            default_model = models[0]
        choices = [questionary.Choice(m, value=m) for m in models]
        selected = _select_with_back(
            "Select model:",
            choices=choices,
            default=default_model,
        )
        if selected == "__back__":
            return
        if selected == "(manual entry)":
            selected = _prompt(
                questionary.text(
                    "Model name:",
                    default=default_model,
                    style=_WIZARD_STYLE,
                )
            )
        config_data["model"] = selected

    # Check for existing provider of same type
    existing = _find_provider_by_type(provider_type)
    if existing is not None:
        overwrite = _prompt(
            questionary.confirm(
                f"Provider '[{_KEY_COLOR}]{existing}[/{_KEY_COLOR}]' "
                f"([{_VALUE_COLOR}]{provider_type}[/{_VALUE_COLOR}]) "
                "already exists. Overwrite?",
                default=False,
                style=_WIZARD_STYLE,
            )
        )
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
    else:
        set_default = _prompt(
            questionary.confirm(
                "Set as default provider?", default=False, style=_WIZARD_STYLE
            )
        )
        if set_default:
            Config.set_default_provider(name)

    test_now = _prompt(
        questionary.confirm("Test connection now?", default=False, style=_WIZARD_STYLE)
    )
    if test_now:
        _test_provider(name, config_data)


def _test_provider_standalone() -> None:
    """Test connection for any existing provider (standalone flow)."""
    providers = Config.get_providers()
    if not providers:
        rich_print(f"[{_WARNING_COLOR}]No providers configured.[/{_WARNING_COLOR}]")
        return

    choices = [questionary.Choice(name, value=name) for name in providers]
    selected = _select_with_back("Select provider to test:", choices=choices)
    if selected == "__back__":
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
        rich_print(f"[{_WARNING_COLOR}]No providers configured.[/{_WARNING_COLOR}]")
        return

    choices = [questionary.Choice(name, value=name) for name in providers]
    selected = _select_with_back("Select provider to edit:", choices=choices)
    if selected == "__back__":
        return

    cfg = dict(providers[selected])
    provider_type = cfg.get("provider_type", "")

    while True:
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
        field = _select_with_back("Select field to edit:", choices=field_choices)
        if field == "__back__":
            break

        if field == "model":
            # Remove provider_type from kwargs to avoid duplicate argument
            api_kwargs = {k: v for k, v in cfg.items() if k != "provider_type"}
            models = _fetch_models(provider_type, **api_kwargs)
            if not models:
                models = ["(manual entry)"]
            default = cfg.get("model", "")
            if default not in models and models:
                default = models[0]
            choices = [questionary.Choice(m, value=m) for m in models]
            new_model = _select_with_back(
                "Select model:", choices=choices, default=default
            )
            if new_model == "__back__":
                continue
            if new_model == "(manual entry)":
                new_model = _prompt(
                    questionary.text(
                        "Model name:", default=default, style=_WIZARD_STYLE
                    )
                )
            cfg["model"] = new_model
        elif field == "api_key":
            new_key = _prompt(questionary.password("API Key:", style=_WIZARD_STYLE))
            cfg["api_key"] = new_key
        elif field == "base_url":
            default = cfg.get("base_url", "")
            new_url = _prompt(
                questionary.text("Base URL:", default=default, style=_WIZARD_STYLE)
            )
            cfg["base_url"] = new_url

        Config.save_provider(selected, cfg)
        rich_print(
            f"[{_SUCCESS_COLOR}]Provider '[{_VALUE_COLOR}]{selected}[/{_VALUE_COLOR}]' "
            f"updated.[/{_SUCCESS_COLOR}]"
        )

        edit_another = _prompt(
            questionary.confirm(
                "Edit another field?", default=False, style=_WIZARD_STYLE
            )
        )
        if not edit_another:
            break


def _remove_provider_flow() -> None:
    """Remove a provider."""
    providers = Config.get_providers()
    if not providers:
        rich_print(f"[{_WARNING_COLOR}]No providers configured.[/{_WARNING_COLOR}]")
        return

    choices = [questionary.Choice(name, value=name) for name in providers]
    selected = _select_with_back("Select provider to remove:", choices=choices)
    if selected == "__back__":
        return

    confirm = _prompt(
        questionary.confirm(
            f"Remove provider '{selected}'?", default=False, style=_WIZARD_STYLE
        )
    )
    if confirm:
        if Config.remove_provider(selected):
            rich_print(
                f"[{_SUCCESS_COLOR}]Provider "
                f"'[{_VALUE_COLOR}]{selected}[/{_VALUE_COLOR}]' "
                f"removed.[/{_SUCCESS_COLOR}]"
            )
        else:
            rich_print(
                f"[{_ERROR_COLOR}]Provider "
                f"'[{_VALUE_COLOR}]{selected}[/{_VALUE_COLOR}]' "
                f"not found.[/{_ERROR_COLOR}]"
            )


def _set_default_provider_flow() -> None:
    """Set the default provider."""
    providers = Config.get_providers()
    if not providers:
        rich_print(f"[{_WARNING_COLOR}]No providers configured.[/{_WARNING_COLOR}]")
        return

    choices = [questionary.Choice(name, value=name) for name in providers]
    selected = _select_with_back("Set default provider:", choices=choices)
    if selected == "__back__":
        return

    Config.set_default_provider(selected)
    rich_print(
        f"[{_SUCCESS_COLOR}]Default provider set to "
        f"'[{_VALUE_COLOR}]{selected}[/{_VALUE_COLOR}]'.[/{_SUCCESS_COLOR}]"
    )


# ---------------------------------------------------------------------------
# General config flows
# ---------------------------------------------------------------------------
def _edit_general_config_flow() -> None:
    """Edit general (non-provider) configuration."""
    while True:
        flat = Config.get_all_config_flat()
        if not flat:
            rich_print(f"[{_WARNING_COLOR}]No configuration found.[/{_WARNING_COLOR}]")
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

        selected = _select_with_back("Select setting to edit:", choices=choices)
        if selected == "__back__":
            break

        current = flat[selected]
        new_value = _prompt_for_config_value(selected, current)
        if new_value is None:
            rich_print(f"[{_SEP_COLOR}]Edit cancelled, no changes made.[/{_SEP_COLOR}]")
            continue

        Config.set_config_value(selected, new_value)
        masked = _mask_value(selected, new_value)
        rich_print(
            f"[{_SUCCESS_COLOR}]Saved '[{_KEY_COLOR}]{selected}[/{_KEY_COLOR}]' = "
            f"[{_VALUE_COLOR}]{masked}[/{_VALUE_COLOR}][/{_SUCCESS_COLOR}]"
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

    with contextlib.suppress(WizardCancelledError):
        _run_wizard_loop()

    rich_print(f"[{_SEP_COLOR}]Goodbye![/{_SEP_COLOR}]")


def _run_wizard_loop() -> None:
    """Core wizard loop (catches WizardCancelledError to exit cleanly)."""
    while True:
        rich_print(
            Panel.fit(
                "[bold]ESDC Configuration Manager[/bold]",
                border_style=_HEADER_BORDER,
                padding=(1, 4),
            )
        )

        selected = _prompt(
            questionary.select(
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
            )
        )

        if selected == "Exit":
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
        except Exception:
            traceback.print_exc()
            rich_print(
                f"[{_ERROR_COLOR}]An error occurred. Please try again.[/{_ERROR_COLOR}]"
            )
