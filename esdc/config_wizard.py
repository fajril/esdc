"""Interactive configuration wizard using questionary.

Provides a unified, streamlined CLI for managing all ESDC configuration
— providers, general settings, and everything else stored in config.yaml.
"""

# Standard library
import contextlib
import logging
import traceback
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

# Third-party
import questionary
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys
from questionary import Style
from rich import print as rich_print
from rich.panel import Panel

# Local
from esdc.config_fields import (
    format_config_value,
    parse_config_input,
)
from esdc.configs import (
    ADVANCED_SECTIONS,
    APPLICATION_SECTIONS,
    ENUM_CHOICES,
    KEY_DESCRIPTIONS,
    SENSITIVE_KEYS,
    TASK_SECTIONS,
    Config,
)
from esdc.model_catalog import discover_models
from esdc.model_settings import ROLE_LABELS
from esdc.providers import PROVIDER_CLASSES, PROVIDER_NAMES

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

_RESTART_NOTE = "Running ESDC services may need restart to read the saved config."


# ---------------------------------------------------------------------------
# Connection transport/authentication fields
# ---------------------------------------------------------------------------
# Connections own transport and authentication only. A task owns its model and
# model options, so no connection editor asks for a model or reasoning effort.
_CONNECTION_FIELDS: dict[str, list[str]] = {
    "ollama": ["base_url"],
    "openai": ["api_key"],
    "openai_compatible": ["base_url", "api_key"],
    "anthropic": ["api_key"],
    "google": ["api_key"],
    "azure_openai": ["base_url", "api_key", "api_version"],
    "groq": ["api_key"],
    "deepseek": ["api_key"],
    "ollama_cloud": ["base_url", "api_key"],
    "opencode": ["api_key", "base_url"],
}


# ---------------------------------------------------------------------------
# Cancel / back helpers
# ---------------------------------------------------------------------------
class WizardCancelledError(Exception):
    """Kept for compatibility.

    A cancel is no longer an exception: :func:`_ask` returns ``None`` for
    Escape, and Control+C raises :class:`KeyboardInterrupt` which
    :func:`run_wizard` handles at the top level.
    """


# Attribute marking the Escape binding already attached to one question.
_ESCAPE_ATTR = "_escape_binding"


def _escape(event: Any) -> None:
    """Exit the current question with no result — the cancel value."""
    event.app.exit(result=None)


def _attach_escape(question: Any) -> None:
    """Attach exactly one eager Escape binding to one question's application.

    Calling this twice on the same question is a no-op, so the binding is
    never registered twice.
    """
    if getattr(question, _ESCAPE_ATTR, None) is not None:
        return

    bindings = question.application.key_bindings
    for binding in getattr(bindings, "bindings", []):
        if getattr(binding, "handler", None) is _escape:
            setattr(question, _ESCAPE_ATTR, bindings)
            return

    target = bindings
    if not hasattr(target, "add"):
        # questionary hands back a merged registry view, which has no `add`,
        # so a registry of our own is attached to it.
        registries = getattr(target, "registries", None)
        if registries is None:
            return
        target = KeyBindings()
        registries.append(target)

    target.add(Keys.Escape, eager=True)(_escape)
    setattr(question, _ESCAPE_ATTR, target)


def _ask(question: Any) -> Any | None:
    """Ask one question: Escape returns None, Control+C propagates.

    ``unsafe_ask()`` is used so a KeyboardInterrupt is not swallowed and stays
    distinguishable from an Escape cancel (the value ``None``). The result is
    returned unchanged, including ``None``.
    """
    _attach_escape(question)
    return question.unsafe_ask()


def _select_with_back(
    message: str,
    choices: list,
    default: Any = None,
) -> Any:
    """Prompt a *select* with a leading '← Back' option.

    Returns the selected value (including ``"__back__"`` when the user picks
    *Back*). A cancel (Escape, so ``None``) is read as Back for a selection.
    """
    all_choices = [questionary.Choice("← Back", value="__back__")] + list(choices)
    result = _ask(
        questionary.select(
            message,
            choices=all_choices,
            default=default,
            style=_WIZARD_STYLE,
        )
    )
    return "__back__" if result is None else result


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------






def _check_chat_assignment_warning() -> None:
    """Warn when canonical Chat has no model assignment yet.

    Connections can exist without any task using them, so a saved connection is
    not evidence of a Chat setup. The message names the task, never a hidden
    connection-level model.
    """
    try:
        assignment = Config.get_model_assignment("chat")
    except ValueError:
        # A pre-canonical file is reported by the explicit conversion flow, not
        # this informational banner.
        return
    if assignment.mode != "model":
        rich_print(
            f"[{_WARNING_COLOR}]Warning: Chat has no connection/model "
            f"assigned. Choose one in Models by task > Chat.[/{_WARNING_COLOR}]"
        )


def _prompt_for_config_value(
    key: str,
    current: Any,
    values: Mapping[str, Any] | None = None,
) -> Any:
    """Prompt the user for a single config value using the best widget.

    The key's description (if any) is passed via questionary's
    ``instruction=`` so it renders dim beside the prompt instead of being
    concatenated into the message label.

    Returns ``None`` when the user cancels (Escape), which means "cancel this
    field". Typed input is validated through ``parse_config_input``, so an
    invalid value raises a short ``ValueError`` naming the field and never
    reaches a save.

    ``values`` is the flat merged config (the draft being edited) used to
    check fields that depend on each other.
    """
    label = f"[{_KEY_COLOR}]{key}[/{_KEY_COLOR}]"
    instruction = KEY_DESCRIPTIONS.get(key, "")
    draft = values if values is not None else {}

    def resolve(raw: Any) -> Any:
        return None if raw is None else parse_config_input(key, raw, draft)

    # 1. Boolean key?
    if key in Config.BOOLEAN_KEYS or isinstance(current, bool):
        selected = _ask(
            questionary.select(
                label,
                choices=["True", "False"],
                default="True" if current else "False",
                instruction=instruction,
                style=_WIZARD_STYLE,
            )
        )
        return None if selected is None else selected == "True"

    # 2. Enum key?
    if key in ENUM_CHOICES:
        choices = list(ENUM_CHOICES[key])
        rows = [questionary.Choice(choice, value=choice) for choice in choices]
        stored = str(current)
        default = stored if stored in choices else choices[0]
        selected = _ask(
            questionary.select(
                label,
                choices=rows,
                default=default,
                instruction=instruction,
                style=_WIZARD_STYLE,
            )
        )
        return resolve(selected)

    # 3. Integer or decimal key?
    if key in Config.INT_KEYS or key in Config.FLOAT_KEYS:
        return resolve(
            _ask(
                questionary.text(
                    label,
                    default=str(current),
                    instruction=instruction,
                    style=_WIZARD_STYLE,
                )
            )
        )

    # 5. Sensitive key?
    if any(s in key for s in SENSITIVE_KEYS):
        return resolve(
            _ask(
                questionary.password(
                    label,
                    instruction=instruction,
                    style=_WIZARD_STYLE,
                )
            )
        )

    # 6. Database / path key?
    if key == "database_path":
        default = str(current)
        return resolve(
            _ask(
                questionary.path(
                    label,
                    default=default,
                    only_directories=False,
                    instruction=instruction,
                    style=_WIZARD_STYLE,
                )
            )
        )

    # 7. Generic text key
    default = str(current)
    return resolve(
        _ask(
            questionary.text(
                label,
                default=default,
                instruction=instruction,
                style=_WIZARD_STYLE,
            )
        )
    )


# ---------------------------------------------------------------------------
# Canonical connection CRUD flows
# ---------------------------------------------------------------------------
def _connection_default_base_url(provider_type: str) -> str:
    """The provider SDK's own saved default address, never a guessed host."""
    provider_class = PROVIDER_CLASSES.get(provider_type)
    if provider_class is None:
        return ""
    return str(
        getattr(provider_class, "BASE_URL", "")
        or getattr(provider_class, "DEFAULT_BASE_URL", "")
    )


def _format_connection_label(name: str, cfg: Mapping[str, Any]) -> str:
    """One registry row: custom name plus its provider type."""
    provider_type = str(cfg.get("type") or cfg.get("provider_type") or "")
    type_label = PROVIDER_NAMES.get(provider_type, provider_type or "unknown")
    return f"{name} ({type_label})"


def _connection_reference_label(path: str) -> str:
    """Human task/fallback label for one canonical reference path."""
    if not path.startswith("models."):
        return path
    suffix = path[len("models.") :]
    if ".fallbacks[" in suffix:
        role, _, tail = suffix.partition(".fallbacks[")
        index = tail.rstrip("]")
        role_label = ROLE_LABELS.get(role, role)  # type: ignore[arg-type]
        return f"{role_label} fallback {int(index) + 1}"
    return ROLE_LABELS.get(suffix, suffix)  # type: ignore[arg-type]


def _connection_reference_lines(references: list[str]) -> list[str]:
    """Deduplicated labels, keeping the canonical reference order."""
    labels: list[str] = []
    for reference in references:
        label = _connection_reference_label(reference)
        if label not in labels:
            labels.append(label)
    return labels


def _prompt_connection_name(current: str) -> str | None:
    """Prompt a connection name; None cancels this field only."""
    value = _ask(
        questionary.text("Connection name:", default=current, style=_WIZARD_STYLE)
    )
    if value is None:
        return None

    existing_names = Config.get_connections()
    while not value:
        value = _ask(
            questionary.text(
                "Name cannot be empty. Enter a connection name:",
                default="",
                style=_WIZARD_STYLE,
            )
        )
        if value is None:
            return None

    while value != current and value in existing_names:
        rich_print(
            f"[{_WARNING_COLOR}]Name '[{_VALUE_COLOR}]{value}"
            f"[/{_VALUE_COLOR}]' is already taken.[/{_WARNING_COLOR}]"
        )
        value = _ask(
            questionary.text(
                "Enter a different connection name:",
                default="",
                style=_WIZARD_STYLE,
            )
        )
        if value is None:
            return None

    return value


def _prompt_connection_field(
    provider_type: str, field: str, current: Any
) -> Any | None:
    """Prompt one transport/auth field; None cancels this field only."""
    if field == "api_key":
        value = _ask(
            questionary.password("API Key (blank keeps current):", style=_WIZARD_STYLE)
        )
        if value is None:
            return None
        return value or current

    if field == "api_version":
        default = current or "2024-02-01"
        return _ask(
            questionary.text("API Version:", default=default, style=_WIZARD_STYLE)
        )

    if field == "base_url":
        default = current or _connection_default_base_url(provider_type)
        return _ask(questionary.text("Base URL:", default=default, style=_WIZARD_STYLE))

    return None


def _connection_edit_choices(
    draft: Mapping[str, Any], name: str
) -> list[questionary.Choice]:
    """Editable fields of one connection: name and transport/auth only."""
    provider_type = str(draft.get("type") or "")

    def field_choice(field: str) -> questionary.Choice:
        return questionary.Choice(
            f"{field} = {format_config_value(field, draft.get(field, '')) or '(none)'}",
            value=field,
        )

    choices = [questionary.Choice(f"name = {name}", value="name")]

    if "api_key" in draft:
        choices.append(field_choice("api_key"))
        choices.append(questionary.Choice("Clear API key", value="__clear_key__"))

    for field in _CONNECTION_FIELDS.get(provider_type, []):
        if field == "api_key":
            continue
        choices.append(field_choice(field))

    if draft.get("base_url") and "base_url" not in _CONNECTION_FIELDS.get(
        provider_type, []
    ):
        choices.append(field_choice("base_url"))

    choices.append(questionary.Choice("Done editing", value="__done__"))
    return choices


def _test_connection(name: str, entry: Mapping[str, Any]) -> None:
    """Read-only connection check through centralized model discovery."""
    rich_print(
        f"[{_SEP_COLOR}]Checking connection '[{_VALUE_COLOR}]{name}"
        f"[/{_VALUE_COLOR}]'…[/{_SEP_COLOR}]"
    )
    catalog = discover_models(dict(entry))
    if catalog.warning:
        rich_print(f"[{_WARNING_COLOR}]{catalog.warning}[/{_WARNING_COLOR}]")
    if catalog.entries:
        shown = ", ".join(entry.model_id for entry in catalog.entries[:5])
        suffix = "…" if len(catalog.entries) > 5 else ""
        rich_print(
            f"[{_SUCCESS_COLOR}]Discovered {len(catalog.entries)} model(s): "
            f"{shown}{suffix}[/{_SUCCESS_COLOR}]"
        )
    elif catalog.warning is None:
        rich_print(f"[{_WARNING_COLOR}]No models were reported.[/{_WARNING_COLOR}]")


def _save_connection_draft(
    name: str,
    draft: dict[str, Any],
    original_name: str | None = None,
) -> None:
    """List impacted tasks, then write one connection draft in one save.

    A new connection is added; an edited same-name connection is updated in
    place; an edited renamed connection moves its key and every task reference
    in the same atomic write. Cancel/Back never reaches this function.
    """
    references = Config.get_connection_references(original_name or name)
    if references:
        uses = ", ".join(_connection_reference_lines(references))
        rich_print(
            f"[{_WARNING_COLOR}]Connection '[{_VALUE_COLOR}]{original_name or name}"
            f"[/{_VALUE_COLOR}]' is used by: {uses}.[/{_WARNING_COLOR}]"
        )

    try:
        if original_name is not None and name != original_name:
            Config.rename_connection(original_name, name, values=draft)
            rich_print(
                f"[{_SUCCESS_COLOR}]Connection renamed to "
                f"'[{_VALUE_COLOR}]{name}[/{_VALUE_COLOR}]'; task references were "
                f"updated.[/{_SUCCESS_COLOR}]"
            )
        else:
            Config.save_connection(name, draft)
            action = "updated" if original_name is not None else "saved"
            rich_print(
                f"[{_SUCCESS_COLOR}]Connection '[{_VALUE_COLOR}]{name}"
                f"[/{_VALUE_COLOR}]' {action}.[/{_SUCCESS_COLOR}]"
            )
    except ValueError as error:
        rich_print(f"[{_ERROR_COLOR}]{error}[/{_ERROR_COLOR}]")
        return

    rich_print(f"[{_SEP_COLOR}]{_RESTART_NOTE}[/]")


def _connection_draft_actions(
    name: str,
    draft: dict[str, Any],
    original_name: str | None = None,
) -> None:
    """Save / Test / Cancel for one connection draft; only Save writes."""
    while True:
        action = _select_with_back(
            f"Connection '{name}' — what to do",
            choices=[
                questionary.Choice("Save", value="save"),
                questionary.Choice("Test connection", value="test"),
                questionary.Choice("Cancel", value="cancel"),
            ],
        )
        if action is None or action in ("cancel", "__back__"):
            rich_print(f"[{_SEP_COLOR}]Draft discarded, nothing saved.[/{_SEP_COLOR}]")
            return
        if action == "test":
            _test_connection(name, draft)
            continue
        _save_connection_draft(name, draft, original_name)
        return


def collect_connection_draft() -> tuple[str, dict[str, Any]] | None:
    """Collect one connection draft without saving it.

    Used by standalone Connections Add and by the task picker's nested
    ``Add connection…`` action. The returned name/entry can be assigned to a
    task draft; cancel returns None and writes nothing.
    """
    provider_type = _select_with_back(
        "Select connection type:",
        choices=[
            questionary.Choice(title, value=key)
            for key, title in PROVIDER_NAMES.items()
        ],
    )
    if provider_type == "__back__":
        return None

    name = _prompt_connection_name("")
    if name is None:
        return None

    draft: dict[str, Any] = {"type": provider_type}
    for field in _CONNECTION_FIELDS.get(provider_type, []):
        value = _prompt_connection_field(provider_type, field, draft.get(field, ""))
        if value is None:
            return None
        draft[field] = value

    return name, draft


def _add_connection_flow() -> None:
    """Add a connection: type, custom name, transport/auth; Save writes once."""
    collected = collect_connection_draft()
    if collected is None:
        return
    name, draft = collected
    _connection_draft_actions(name, draft)


def _edit_connection_flow() -> None:
    """Edit a connection's transport/auth only, then Save or cancel."""
    connections = Config.get_connections()
    if not connections:
        rich_print(f"[{_WARNING_COLOR}]No connections configured.[/{_WARNING_COLOR}]")
        return

    choices = [
        questionary.Choice(_format_connection_label(name, entry), value=name)
        for name, entry in connections.items()
    ]
    original_name = _select_with_back("Select connection to edit:", choices=choices)
    if original_name == "__back__":
        return

    draft: dict[str, Any] = dict(connections[original_name])
    name = original_name

    while True:
        field = _select_with_back(
            "Select field to edit:",
            choices=_connection_edit_choices(draft, name),
        )
        if field == "__back__":
            rich_print(f"[{_SEP_COLOR}]Draft discarded, nothing saved.[/{_SEP_COLOR}]")
            return
        if field == "__done__":
            break

        if field == "__clear_key__":
            draft["api_key"] = ""
            rich_print(
                f"[{_SEP_COLOR}]API key cleared for the draft; Save writes it."
                f"[/{_SEP_COLOR}]"
            )
            continue

        if field == "name":
            new_name = _prompt_connection_name(name)
            if new_name is None:
                rich_print(
                    f"[{_SEP_COLOR}]Edit cancelled, no changes made.[/{_SEP_COLOR}]"
                )
                continue
            name = new_name
            continue

        value = _prompt_connection_field(
            str(draft.get("type") or ""), field, draft.get(field, "")
        )
        if value is None:
            rich_print(f"[{_SEP_COLOR}]Edit cancelled, no changes made.[/{_SEP_COLOR}]")
            continue
        draft[field] = value

    _connection_draft_actions(name, draft, original_name)


def _remove_connection_flow() -> None:
    """Remove an unreferenced connection; a referenced one is refused."""
    connections = Config.get_connections()
    if not connections:
        rich_print(f"[{_WARNING_COLOR}]No connections configured.[/{_WARNING_COLOR}]")
        return

    choices = [
        questionary.Choice(_format_connection_label(name, entry), value=name)
        for name, entry in connections.items()
    ]
    selected = _select_with_back("Select connection to remove:", choices=choices)
    if selected == "__back__":
        return

    references = Config.get_connection_references(selected)
    if references:
        uses = ", ".join(_connection_reference_lines(references))
        rich_print(
            f"[{_WARNING_COLOR}]Connection '[{_VALUE_COLOR}]{selected}"
            f"[/{_VALUE_COLOR}]' is still used by: {uses}.[/{_WARNING_COLOR}]"
        )
        rich_print(
            f"[{_SEP_COLOR}]Reassign those tasks first; nothing was removed."
            f"[/{_SEP_COLOR}]"
        )
        return

    confirm = _ask(
        questionary.confirm(
            f"Remove connection '{selected}'?", default=False, style=_WIZARD_STYLE
        )
    )
    if confirm is None:
        return
    if not confirm:
        return
    if Config.remove_connection(selected):
        rich_print(
            f"[{_SUCCESS_COLOR}]Connection '[{_VALUE_COLOR}]{selected}"
            f"[/{_VALUE_COLOR}]' removed.[/{_SUCCESS_COLOR}]"
        )
    else:
        rich_print(
            f"[{_ERROR_COLOR}]Connection '[{_VALUE_COLOR}]{selected}"
            f"[/{_VALUE_COLOR}]' not found.[/{_ERROR_COLOR}]"
        )


def _test_connection_standalone() -> None:
    """Pick any saved connection and check it through model discovery."""
    connections = Config.get_connections()
    if not connections:
        rich_print(f"[{_WARNING_COLOR}]No connections configured.[/{_WARNING_COLOR}]")
        return

    choices = [
        questionary.Choice(_format_connection_label(name, entry), value=name)
        for name, entry in connections.items()
    ]
    selected = _select_with_back("Select connection to test:", choices=choices)
    if selected == "__back__":
        return
    _test_connection(selected, connections[selected])


# ---------------------------------------------------------------------------
# Provider CRUD flows
# ---------------------------------------------------------------------------







# ---------------------------------------------------------------------------
# General config flows
# ---------------------------------------------------------------------------
def _get_merged_flat() -> dict[str, Any]:
    """Flat config view with every editable key present.

    Config.get_all_config_flat() merges Config.get_defaults() over the saved
    config, so all default sections (corpus, phoenix, logging, …) are always
    populated even when config.yaml has no such section yet.
    """
    return Config.get_all_config_flat()


def _edit_section(title: str, keys: list[str]) -> None:
    """Generic editor for a fixed, ordered list of config keys.

    Renders one row per key (``key = masked_value``); selecting a row
    reuses ``_prompt_for_config_value`` + ``Config.set_config_value``.
    Loops until the user picks Back.

    An invalid value raises ``ValueError`` before any write, so the field is
    re-asked with its current value and nothing is saved.
    """
    while True:
        flat = _get_merged_flat()
        choices = [
            questionary.Choice(
                f"{key} = {format_config_value(key, flat.get(key))}", value=key
            )
            for key in keys
        ]

        selected = _select_with_back(title, choices=choices)
        if selected == "__back__":
            break

        while True:
            try:
                new_value = _prompt_for_config_value(selected, flat.get(selected), flat)
            except ValueError as error:
                # Stay at the field: the message names it, and nothing is written.
                rich_print(f"[{_ERROR_COLOR}]{error}[/{_ERROR_COLOR}]")
                continue

            if new_value is None:
                rich_print(
                    f"[{_SEP_COLOR}]Edit cancelled, no changes made.[/{_SEP_COLOR}]"
                )
                break

            if new_value == flat.get(selected):
                # The field came back as it was stored (a retired row is read, not
                # switched), so there is nothing to write.
                rich_print(
                    f"[{_SEP_COLOR}]'{selected}' unchanged, nothing written."
                    f"[/{_SEP_COLOR}]"
                )
                break

            Config.set_config_value(selected, new_value)
            rich_print(
                f"[{_SUCCESS_COLOR}]Saved '[{_KEY_COLOR}]{selected}[/{_KEY_COLOR}]' = "
                f"[{_VALUE_COLOR}]{format_config_value(selected, new_value)}"
                f"[/{_VALUE_COLOR}][/{_SUCCESS_COLOR}]"
            )
            break


def _show_config_flow() -> None:
    """Print all configuration flat keys (masked)."""
    flat = Config.get_all_config_flat()
    if not flat:
        rich_print(f"[{_WARNING_COLOR}]No configuration found.[/{_WARNING_COLOR}]")
        return

    for key in sorted(flat.keys()):
        value = format_config_value(key, flat[key])
        rich_print(
            f"  [{_KEY_COLOR}]{key}[/{_KEY_COLOR}] "
            f"[{_SEP_COLOR}]=[/{_SEP_COLOR}] "
            f"[{_VALUE_COLOR}]{value}[/{_VALUE_COLOR}]"
        )
    print()


def _needs_model_conversion() -> bool:
    """Whether the saved file lacks the canonical model-settings marker."""
    config = Config._load_config()
    return bool(config) and "model_settings_version" not in config


def _update_model_settings_flow() -> None:
    """Preview, confirm and apply the one-time model-settings conversion."""
    from esdc.model_config_apply import (
        ModelSettingsApplyError,
        apply_model_settings,
        preview_model_settings,
        render_preview,
    )

    source = Config.get_config_file()
    try:
        preview = preview_model_settings(source)
    except ModelSettingsApplyError as error:
        rich_print(f"[{_ERROR_COLOR}]{error}[/{_ERROR_COLOR}]")
        return

    for line in render_preview(preview):
        print(line)

    if preview.conversion.blockers:
        rich_print(
            f"[{_ERROR_COLOR}]Conversion is blocked; nothing was written."
            f"[/{_ERROR_COLOR}]"
        )
        return
    if preview.noop:
        rich_print(
            f"[{_SEP_COLOR}]Already canonical; nothing was written."
            f"[/{_SEP_COLOR}]"
        )
        return

    backup_value = _ask(
        questionary.text(
            "Backup path:",
            default=f"{source}.pre-unified",
            style=_WIZARD_STYLE,
        )
    )
    if not backup_value:
        rich_print(f"[{_SEP_COLOR}]Cancelled, nothing written.[/{_SEP_COLOR}]")
        return

    confirm = _ask(
        questionary.confirm(
            "Apply model settings conversion?",
            default=False,
            style=_WIZARD_STYLE,
        )
    )
    if not confirm:
        rich_print(f"[{_SEP_COLOR}]Cancelled, nothing written.[/{_SEP_COLOR}]")
        return

    try:
        result = apply_model_settings(source, Path(str(backup_value)).expanduser())
    except ModelSettingsApplyError as error:
        rich_print(f"[{_ERROR_COLOR}]{error}[/{_ERROR_COLOR}]")
        return

    if result.conversion.blockers:
        for blocker in result.conversion.blockers:
            rich_print(f"[{_ERROR_COLOR}]{blocker}[/{_ERROR_COLOR}]")
        return
    if result.noop:
        rich_print(
            f"[{_SEP_COLOR}]Already canonical; nothing was written."
            f"[/{_SEP_COLOR}]"
        )
        return
    rich_print(
        f"[{_SUCCESS_COLOR}]Model settings converted. Backup: "
        f"[{_VALUE_COLOR}]{result.backup}[/{_VALUE_COLOR}].[/{_SUCCESS_COLOR}]"
    )
    rich_print(f"[{_SEP_COLOR}]{_RESTART_NOTE}[/]")


# ---------------------------------------------------------------------------
# Task selection (D4)
# ---------------------------------------------------------------------------
TaskRole = Literal["chat", "embedding", "rerank", "ocr", "metadata", "cleanup"]


_TASK_MENU_ORDER: tuple[TaskRole, ...] = (
    "chat",
    "embedding",
    "rerank",
    "ocr",
    "metadata",
    "cleanup",
)

_TASK_TITLES: dict[TaskRole, str] = {
    "chat": "Chat",
    "embedding": "Embeddings",
    "rerank": "Reranking",
    "ocr": "OCR",
    "metadata": "Metadata extraction",
    "cleanup": "Text cleanup",
}


def _error_status(error: Exception) -> int | None:
    """A numeric HTTP status from the exception, its response, or its cause chain.

    Never reads exception text: server bodies, credentials, and request URLs are
    not safe to display.
    """
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        status = getattr(current, "status_code", None)
        if isinstance(status, int) and not isinstance(status, bool):
            return status
        response = getattr(current, "response", None)
        status = getattr(response, "status_code", None)
        if isinstance(status, int) and not isinstance(status, bool):
            return status
        current = current.__cause__ or current.__context__
    return None


def _short_test_message(error: Exception) -> str:
    """A safe error type or HTTP status for a failed test.

    Exception text can contain server response bodies, credentials, or request
    URLs, so it is never read. A numeric status wins; otherwise only the
    exception type is reported.
    """
    status = _error_status(error)
    if status is not None:
        return f"HTTP {status}"
    return type(error).__name__


def _edit_task(role: TaskRole) -> None:
    """Open the one canonical task picker for a role."""
    from esdc.model_wizard import edit_task

    edit_task(role)




# ---------------------------------------------------------------------------
# Submenus
# ---------------------------------------------------------------------------
def _connections_menu() -> None:
    """Connections submenu: registry management only, no model/failover."""
    while True:
        selected = _select_with_back(
            "Connections",
            choices=[
                questionary.Choice("Add connection", value="Add connection"),
                questionary.Choice("Edit connection", value="Edit connection"),
                questionary.Choice("Remove connection", value="Remove connection"),
                questionary.Choice("Test connection", value="Test connection"),
            ],
        )
        if selected == "__back__":
            break

        if selected == "Add connection":
            _add_connection_flow()
        elif selected == "Edit connection":
            _edit_connection_flow()
        elif selected == "Remove connection":
            _remove_connection_flow()
        elif selected == "Test connection":
            _test_connection_standalone()


def _models_by_task_menu() -> None:
    """Models by task submenu — one row per task, each opening its picker."""
    while True:
        selected = _select_with_back(
            "Models by task",
            choices=[
                questionary.Choice(_TASK_TITLES[role], value=role)
                for role in _TASK_MENU_ORDER
            ],
        )
        if selected == "__back__":
            return

        if selected in _TASK_TITLES:
            _edit_task(selected)


def _advanced_menu() -> None:
    """Advanced submenu — the remaining processing controls and the reset actions."""
    while True:
        selected = _select_with_back(
            "Advanced",
            choices=[
                questionary.Choice(f"{name} ▸", value=name)
                for name in ADVANCED_SECTIONS
            ]
            + [
                questionary.Choice("Reset a setting", value="Reset a setting"),
                questionary.Choice("Reset all", value="Reset all"),
            ],
        )
        if selected == "__back__":
            return

        if selected == "Reset a setting":
            _reset_single_key_flow()
        elif selected == "Reset all":
            _confirm_reset_all()
        elif selected in ADVANCED_SECTIONS:
            _edit_section(selected, ADVANCED_SECTIONS[selected])


def _reset_single_key_flow() -> None:
    """Pick a key from the task/application/advanced tree, then reset it."""
    all_sections: dict[str, list[str]] = {
        **TASK_SECTIONS,
        **APPLICATION_SECTIONS,
        **ADVANCED_SECTIONS,
    }

    while True:
        section = _select_with_back(
            "Reset a single key",
            choices=[questionary.Choice(name, value=name) for name in all_sections],
        )
        if section == "__back__":
            return

        flat = _get_merged_flat()
        key_choices = [
            questionary.Choice(
                f"{key} = {format_config_value(key, flat.get(key))}", value=key
            )
            for key in all_sections[section]
        ]
        key = _select_with_back(section, choices=key_choices)
        if key == "__back__":
            continue

        confirm = _ask(
            questionary.confirm(
                f"Reset '{key}' to its default?",
                default=False,
                style=_WIZARD_STYLE,
            )
        )
        if confirm is None:
            # Cancel at the confirm resets nothing.
            return
        if confirm:
            Config.reset_config(key)
            rich_print(
                f"[{_SUCCESS_COLOR}]Reset '[{_KEY_COLOR}]{key}[/{_KEY_COLOR}]' "
                f"to default.[/{_SUCCESS_COLOR}]"
            )
        return


def _confirm_reset_all() -> None:
    """Reset every setting, behind a confirm gate."""
    confirm = _ask(
        questionary.confirm(
            "Reset ALL configuration to defaults? This cannot be undone.",
            default=False,
            style=_WIZARD_STYLE,
        )
    )
    if confirm is None:
        # Cancel at the confirm resets nothing.
        return
    if confirm:
        Config.reset_config()
        rich_print(
            f"[{_SUCCESS_COLOR}]All configuration reset to defaults.[/{_SUCCESS_COLOR}]"
        )


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------
def run_wizard() -> None:
    """Run the interactive configuration wizard.

    Control+C ends the run here without a traceback, and saves already made by
    an earlier explicit action remain written.
    """
    Config.init_config()
    _check_chat_assignment_warning()

    rich_print(
        Panel.fit(
            "[bold]ESDC Configuration Manager[/bold]",
            border_style=_HEADER_BORDER,
            padding=(1, 4),
        )
    )

    with contextlib.suppress(WizardCancelledError, KeyboardInterrupt):
        _run_wizard_loop()

    rich_print(f"[{_SEP_COLOR}]Goodbye![/{_SEP_COLOR}]")


def _run_wizard_loop() -> None:
    """Core wizard loop. Escape at the top level exits the wizard."""
    while True:
        choices = [
            questionary.Choice("Models by task ▸", value="Models by task"),
            questionary.Choice("Connections ▸", value="Connections"),
            questionary.Choice(
                "Application settings ▸", value="Application settings"
            ),
            questionary.Choice("Advanced ▸", value="Advanced"),
            questionary.Choice("View settings", value="View settings"),
        ]
        if _needs_model_conversion():
            choices.append(
                questionary.Choice(
                    "Update model settings…", value="Update model settings"
                )
            )
        choices.append(questionary.Choice("Exit", value="Exit"))

        selected = _ask(
            questionary.select(
                "What would you like to do?",
                choices=choices,
                style=_WIZARD_STYLE,
            )
        )

        if selected is None:
            # Escape at the top level has no Back above it, so it ends the run.
            break

        if selected == "Exit":
            break

        try:
            if selected == "Models by task":
                _models_by_task_menu()
            elif selected == "Connections":
                _connections_menu()
            elif selected == "Application settings":
                _edit_section(
                    "Application settings", APPLICATION_SECTIONS["Application settings"]
                )
            elif selected == "Advanced":
                _advanced_menu()
            elif selected == "View settings":
                _show_config_flow()
            elif selected == "Update model settings":
                _update_model_settings_flow()
        except Exception:
            traceback.print_exc()
            rich_print(
                f"[{_ERROR_COLOR}]An error occurred. Please try again.[/{_ERROR_COLOR}]"
            )
