"""Interactive configuration wizard using questionary.

Provides a unified, streamlined CLI for managing all ESDC configuration
— providers, general settings, and everything else stored in config.yaml.
"""

# Standard library
import contextlib
import logging
import math
import os
import traceback
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

# Third-party
import fitz
import questionary
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys
from questionary import Style
from rich import print as rich_print
from rich.panel import Panel

# Local
from esdc.config_fields import format_config_value, parse_config_input
from esdc.configs import (
    ADVANCED_SECTIONS,
    APPLICATION_SECTIONS,
    ENUM_CHOICES,
    KEY_DESCRIPTIONS,
    SENSITIVE_KEYS,
    TASK_SECTIONS,
    Config,
)
from esdc.corpus.ocr import create_ocr_client
from esdc.corpus.provider_reranker import ProviderReranker
from esdc.corpus.reranker import DEFAULT_RERANKER
from esdc.embedders import MODEL_ID, OllamaEmbedder, OpenAIEmbedder
from esdc.model_catalog import CatalogRole, ModelCatalog, ModelEntry, discover_models
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
    "deepseek": ["api_key", "model", "reasoning_effort"],
    "ollama_cloud": ["base_url", "api_key", "model"],
    "opencode": ["api_key", "base_url", "model"],
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
def _mask_value(key: str, value: Any) -> str:
    """Mask sensitive values in the UI (thin wrapper of the shared formatter)."""
    return format_config_value(key, value)


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
            return sorted(models)
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


def _check_name_exists(name: str) -> str | None:
    """Check if a provider name is already taken.

    Returns the existing name if found, None otherwise.
    """
    providers = Config.get_providers()
    if name in providers:
        return name
    return None


def _format_provider_label(name: str, cfg: dict) -> str:
    """Format a provider label for display in selection lists.

    Shows just the name if it matches the provider type (e.g. 'openai'),
    or 'name (type)' for custom-named providers.
    """
    provider_type = cfg.get("provider_type", "")
    type_label = PROVIDER_NAMES.get(provider_type, provider_type)
    if name == provider_type:
        return name
    return f"{name} ({type_label})"


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

    # 2. Reranker model key? Curated GGUF choice (llama.cpp registry) +
    # custom escape. Keep in sync with esdc/corpus/reranker.py.
    if key == "corpus.rerank_model":
        choices = [
            questionary.Choice(
                "Qwen3-Reranker-0.6B Q8_0 (default, llama.cpp GGUF)",
                value="ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF",
            ),
            questionary.Choice("Custom…", value="__custom__"),
        ]
        default_choice = next(
            (c for c in choices if c.value == str(current)), choices[0]
        )
        selected = resolve(
            _ask(
                questionary.select(
                    label,
                    choices=choices,
                    default=default_choice,
                    instruction=instruction,
                    style=_WIZARD_STYLE,
                )
            )
        )
        if selected == "__custom__":
            return resolve(
                _ask(
                    questionary.text(
                        f"{label} — model name:",
                        default=str(current),
                        style=_WIZARD_STYLE,
                    )
                )
            )
        return selected

    # 3. Enum key?
    if key in ENUM_CHOICES:
        choices = ENUM_CHOICES[key]
        default = str(current) if str(current) in choices else choices[0]
        return resolve(
            _ask(
                questionary.select(
                    label,
                    choices=choices,
                    default=default,
                    instruction=instruction,
                    style=_WIZARD_STYLE,
                )
            )
        )

    # 4. Integer or decimal key?
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
# Provider draft — collect, test, then one write on Save
# ---------------------------------------------------------------------------
_REASONING_EFFORTS = ("(none)", "low", "medium", "high")

# Task names for the provider reference keys `Config.get_provider_references`
# reports, so a shared address or key change names the tasks without repeating any
# stored value.
_REFERENCE_TASK_NAMES = {
    "default_provider": "Chat",
    "provider_order": "Chat",
    "embedding_provider": "Embeddings",
    "corpus.metadata_provider": "Metadata extraction",
    "corpus.metadata_model": "Metadata extraction",
    "corpus.cleanup_provider": "Text cleanup",
    "corpus.cleanup_model": "Text cleanup",
    "corpus.ocr_provider": "OCR",
    "corpus.rerank_provider": "Reranking",
}

# Environment values that take precedence over the saved embedding fields, in the
# same order as runtime resolution. They are applied to a test draft only: an
# environment secret is never copied into config. `ESDC_EMBEDDING_BACKEND` selects
# the source and is checked separately, so it cannot switch provider silently.
_EMBEDDING_ENV_KEYS = {
    "base_url": "ESDC_EMBEDDING_HOST",
    "api_key": "ESDC_EMBEDDING_API_KEY",
    "model": "ESDC_EMBEDDING_MODEL",
}


def _reasoning_effort_choices() -> list[questionary.Choice]:
    """The same fixed effort list for the add prompt and the edit prompt."""
    return [questionary.Choice(effort, value=effort) for effort in _REASONING_EFFORTS]


def _apply_default_provider_choice(name: str) -> None:
    """The default-provider choice after the first write of a new provider."""
    default = Config.get_default_provider()
    if not default:
        Config.set_default_provider(name)
        rich_print(
            f"[{_SUCCESS_COLOR}]Set '[{_VALUE_COLOR}]{name}[/{_VALUE_COLOR}]' "
            f"as default provider (first provider).[/{_SUCCESS_COLOR}]"
        )
        return

    set_default = _ask(
        questionary.confirm(
            "Set as default provider?", default=False, style=_WIZARD_STYLE
        )
    )
    if set_default is None:
        return
    if set_default:
        Config.set_default_provider(name)


def _reference_task_names(references: list[str]) -> list[str]:
    """The task names a provider reference list affects, in the order reported.

    Only key names are used, so a task is named without showing any stored value.
    Chat covers the default and the failover order; the linked metadata and cleanup
    strings belong to their own task.
    """
    tasks: list[str] = []
    for key in references:
        task = _REFERENCE_TASK_NAMES.get(key)
        if task is not None and task not in tasks:
            tasks.append(task)
    return tasks


def _warn_shared_connection_change(
    saved_name: str, draft: dict[str, Any], saved: dict[str, Any] | None
) -> None:
    """Warn before a write that changes an address or key other tasks share.

    The draft is compared with the saved entry, and the affected tasks are named
    from the references the provider key currently has: while the write is still
    pending, the stored references name the saved key, so that is the key looked up.
    Nothing is blocked: the plan says warn, and a normal task source or model edit
    never reaches this path because it does not touch the provider's own fields.
    """
    if not saved:
        return

    changed = [
        field
        for field in ("base_url", "api_key")
        if str(draft.get(field, "")) != str(saved.get(field, ""))
    ]
    if not changed:
        return

    tasks = _reference_task_names(Config.get_provider_references(saved_name))
    if not tasks:
        return

    rich_print(
        f"[{_WARNING_COLOR}]{' or '.join(changed)} changed for "
        f"'[{_VALUE_COLOR}]{saved_name}[/{_VALUE_COLOR}]'; shared by "
        f"{', '.join(tasks)}.[/{_WARNING_COLOR}]"
    )


def _save_provider_draft(
    name: str,
    draft: dict[str, Any],
    original_name: str | None = None,
) -> None:
    """Write one provider draft once.

    For a draft being added, the same-type overwrite confirmation runs here, at the
    moment of the write, so a Cancel before Save leaves an existing provider
    untouched. The replacement writes in place: the same key is overwritten, or a
    differently named provider is renamed and written in the same single save, so its
    references stay valid, nothing is removed first, and no failure can leave the
    references renamed while the entry still holds the old fields. For an edited
    draft the same rule applies, and `original_name` is the saved key.
    """
    provider_type = str(draft.get("provider_type", ""))
    providers = Config.get_providers()

    if original_name is None:
        existing = _find_provider_by_type(provider_type)

        if existing is None:
            # A new key cannot be referenced yet, so there is nothing to warn about.
            Config.save_provider(name, draft)
            rich_print(
                f"[{_SUCCESS_COLOR}]Provider '[{_VALUE_COLOR}]{name}[/{_VALUE_COLOR}]' "
                f"saved.[/{_SUCCESS_COLOR}]"
            )
            rich_print(f"[{_SEP_COLOR}]{_RESTART_NOTE}[/]")
            _apply_default_provider_choice(name)
            return

        overwrite = _ask(
            questionary.confirm(
                f"Provider '{existing}' ({provider_type}) already exists. Overwrite?",
                default=False,
                style=_WIZARD_STYLE,
            )
        )
        if overwrite is None or not overwrite:
            # Cancel leaves the existing provider untouched.
            rich_print(
                f"[{_SEP_COLOR}]Cancelled. Existing provider kept.[/{_SEP_COLOR}]"
            )
            return

        _warn_shared_connection_change(existing, draft, providers.get(existing))

        if existing == name:
            rich_print(
                f"[{_SEP_COLOR}]Replacing '[{_KEY_COLOR}]{existing}[/{_KEY_COLOR}]' "
                f"in place.[/{_SEP_COLOR}]"
            )
            Config.save_provider(name, draft)
        else:
            try:
                # Key move and draft content in one write.
                Config.rename_provider(existing, name, provider_config=draft)
            except ValueError as error:
                rich_print(f"[{_ERROR_COLOR}]{error}[/{_ERROR_COLOR}]")
                return
            rich_print(
                f"[{_SEP_COLOR}]Renaming '[{_KEY_COLOR}]{existing}[/{_KEY_COLOR}]' "
                f"to '[{_VALUE_COLOR}]{name}[/{_VALUE_COLOR}]' and writing the draft "
                f"in one save.[/{_SEP_COLOR}]"
            )

        rich_print(
            f"[{_SUCCESS_COLOR}]Provider '[{_VALUE_COLOR}]{name}[/{_VALUE_COLOR}]' "
            f"saved.[/{_SUCCESS_COLOR}]"
        )
        rich_print(f"[{_SEP_COLOR}]{_RESTART_NOTE}[/]")
        _apply_default_provider_choice(name)
        return

    _warn_shared_connection_change(original_name, draft, providers.get(original_name))

    if name != original_name:
        try:
            # Key move, references and the draft fields in one write.
            Config.rename_provider(original_name, name, provider_config=draft)
        except ValueError as error:
            rich_print(f"[{_ERROR_COLOR}]{error}[/{_ERROR_COLOR}]")
            return
        rich_print(
            f"[{_SUCCESS_COLOR}]Provider renamed to "
            f"'[{_VALUE_COLOR}]{name}[/{_VALUE_COLOR}]'.[/{_SUCCESS_COLOR}]"
        )
        rich_print(f"[{_SEP_COLOR}]{_RESTART_NOTE}[/]")
        return

    Config.save_provider(name, draft)
    rich_print(
        f"[{_SUCCESS_COLOR}]Provider '[{_VALUE_COLOR}]{name}[/{_VALUE_COLOR}]' "
        f"updated.[/{_SUCCESS_COLOR}]"
    )
    rich_print(f"[{_SEP_COLOR}]{_RESTART_NOTE}[/]")


def _provider_draft_actions(
    name: str,
    draft: dict[str, Any],
    original_name: str | None = None,
) -> None:
    """Save / Test connection / Cancel for one provider draft.

    Test connection keeps this menu open, so the draft can be tested and then
    saved or discarded without collecting the fields again. Save writes the draft
    once; Cancel, Back or Escape writes nothing. `original_name` is the saved key
    for an edited provider, and None for a draft that is being added.
    """
    while True:
        action = _select_with_back(
            f"Provider '{name}' — what to do",
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
            # The draft is tested exactly as entered: no write here.
            _test_provider(name, draft)
            continue

        _save_provider_draft(name, draft, original_name)
        return


def _prompt_provider_name(name: str) -> str | None:
    """Prompt the provider name for a draft; None cancels this field only."""
    value = _ask(questionary.text("Provider name:", default=name, style=_WIZARD_STYLE))
    if value is None:
        return None

    while not value:
        value = _ask(
            questionary.text(
                "Name cannot be empty. Enter a provider name:",
                default="",
                style=_WIZARD_STYLE,
            )
        )
        if value is None:
            return None

    existing = _check_name_exists(value)
    while existing is not None and existing != name:
        rich_print(
            f"[{_WARNING_COLOR}]Name '[{_VALUE_COLOR}]{value}"
            f"[/{_VALUE_COLOR}]' is already taken.[/{_WARNING_COLOR}]"
        )
        value = _ask(
            questionary.text(
                "Enter a different provider name:",
                default="",
                style=_WIZARD_STYLE,
            )
        )
        if value is None:
            return None
        existing = _check_name_exists(value)

    return value


def _prompt_provider_model(draft: dict[str, Any]) -> str | None:
    """Pick the model for a provider draft; None cancels this field only."""
    provider_type = str(draft.get("provider_type", ""))
    models = _fetch_models(
        provider_type, **{k: v for k, v in draft.items() if k != "provider_type"}
    )
    if not models:
        models = ["(manual entry)"]

    default = str(draft.get("model", ""))
    if default not in models and models:
        default = models[0]

    selected = _select_with_back(
        "Select model:",
        choices=[questionary.Choice(model, value=model) for model in models],
        default=default,
    )
    if selected == "__back__":
        return None
    if selected == "(manual entry)":
        return _ask(
            questionary.text("Model name:", default=default, style=_WIZARD_STYLE)
        )
    return selected


def _prompt_provider_field(draft: dict[str, Any], field: str) -> Any:
    """Prompt one provider draft field; None cancels this field only.

    A blank secret keeps the value already in the draft, which is why clearing a
    key needs the explicit Clear API key action.
    """
    provider_type = str(draft.get("provider_type", ""))
    current = draft.get(field, "")

    if field == "api_key":
        value = _ask(
            questionary.password("API Key (blank keeps current):", style=_WIZARD_STYLE)
        )
        if value is None:
            return None
        return value or current

    if field == "reasoning_effort":
        effort = _select_with_back(
            "Reasoning effort (optional):",
            choices=_reasoning_effort_choices(),
            default=current if current in _REASONING_EFFORTS else "(none)",
        )
        return None if effort == "__back__" else effort

    if field == "api_version":
        default = current or _resolve_default_field(provider_type, "api_version")
        return _ask(
            questionary.text(
                "API Version:",
                default=default or "2024-02-01",
                style=_WIZARD_STYLE,
            )
        )

    if field == "base_url":
        default = current or _resolve_default_field(provider_type, "base_url")
        return _ask(questionary.text("Base URL:", default=default, style=_WIZARD_STYLE))

    return None


def _provider_edit_choices(
    draft: dict[str, Any], name: str
) -> list[questionary.Choice]:
    """The editable fields of one provider draft, plus Clear and Done.

    Only fields the provider has are editable: a secret is shown masked, and
    clearing it is a separate action because a blank entry keeps the current key.
    `reasoning_effort` and `api_version` appear only for the types that have them.
    """
    provider_type = str(draft.get("provider_type", ""))

    def field_choice(field: str) -> questionary.Choice:
        # An optional field that is not set is shown as "(none)", the same word the
        # effort list uses for it.
        return questionary.Choice(
            f"{field} = {format_config_value(field, draft.get(field, '')) or '(none)'}",
            value=field,
        )

    choices = [field_choice("model")]

    if "api_key" in draft:
        choices.append(field_choice("api_key"))
        # The only way to clear a key, so a blank entry can keep the current one.
        choices.append(questionary.Choice("Clear API key", value="__clear_key__"))

    if "base_url" in draft:
        choices.append(field_choice("base_url"))

    for field in _PROVIDER_FIELDS.get(provider_type, ["model"]):
        if field in ("model", "api_key", "base_url"):
            continue
        choices.append(field_choice(field))

    if provider_type == "openai_compatible":
        choices.append(questionary.Choice(f"name = {name}", value="name"))

    choices.append(questionary.Choice("Done editing", value="__done__"))
    return choices


def _embedding_test_draft(draft: dict[str, Any]) -> dict[str, Any] | None:
    """One embedding draft with the environment values applied, for testing only.

    The saved values are not changed: a copy is returned. `ESDC_EMBEDDING_BACKEND`
    selects the source, so a draft whose provider type does not agree with the
    environment backend is reported instead of switched, the same rule as runtime
    resolution. When the backend agrees, the draft already holds that provider type.
    """
    test_draft = dict(draft)
    provider_type = str(test_draft.get("provider_type", "")).strip().lower()

    backend = str(os.environ.get("ESDC_EMBEDDING_BACKEND") or "").strip().lower()
    if backend:
        expected = Config.EMBEDDING_PROVIDER_BACKENDS.get(provider_type)
        if expected is None:
            rich_print(
                f"[{_ERROR_COLOR}]ESDC_EMBEDDING_BACKEND '{backend}' needs an "
                f"'openai_compatible' or 'ollama' provider, not "
                f"'{provider_type}'.[/{_ERROR_COLOR}]"
            )
            return None
        if backend != expected:
            rich_print(
                f"[{_ERROR_COLOR}]ESDC_EMBEDDING_BACKEND '{backend}' does not match "
                f"provider type '{provider_type}', which uses "
                f"'{expected}'.[/{_ERROR_COLOR}]"
            )
            return None

    for target, variable in _EMBEDDING_ENV_KEYS.items():
        value = os.environ.get(variable)
        if value:
            test_draft[target] = value

    return test_draft


# ---------------------------------------------------------------------------
# Provider CRUD flows
# ---------------------------------------------------------------------------
def _add_provider_flow() -> None:
    """Interactive wizard to add a new provider.

    A cancel (Escape) at any prompt discards the draft and returns to the
    provider menu; nothing is written for a discarded draft. Collection alone
    writes nothing: the draft is written by the Save choice of the draft action
    menu, which also offers Test connection before anything is stored.
    """
    provider_type = _select_with_back(
        "Select provider type:",
        choices=[
            questionary.Choice(title, value=key)
            for key, title in PROVIDER_NAMES.items()
        ],
    )
    if provider_type == "__back__":
        return

    # Prompt custom name for OpenAI-compatible providers (supports multiple instances)
    if provider_type == "openai_compatible":
        name = _prompt_provider_name("")
        if name is None:
            return
    else:
        name = provider_type

    draft: dict[str, Any] = {"provider_type": provider_type}
    fields = _PROVIDER_FIELDS.get(provider_type, ["model"])

    # Pass 1: collect connection fields first (base_url, api_key, api_version)
    for field in fields:
        if field == "api_key":
            value = _ask(
                questionary.password(
                    "API Key (optional for local):", style=_WIZARD_STYLE
                )
            )
            if value is None:
                return
            draft["api_key"] = value
        elif field == "base_url":
            default = _resolve_default_field(provider_type, "base_url")
            value = _ask(
                questionary.text("Base URL:", default=default, style=_WIZARD_STYLE)
            )
            if value is None:
                return
            draft["base_url"] = value
        elif field == "api_version":
            default = _resolve_default_field(provider_type, "api_version")
            value = _ask(
                questionary.text(
                    "API Version:",
                    default=default or "2024-02-01",
                    style=_WIZARD_STYLE,
                )
            )
            if value is None:
                return
            draft["api_version"] = value
        elif field == "reasoning_effort":
            effort = _select_with_back(
                "Reasoning effort (optional):",
                choices=_reasoning_effort_choices(),
                default="(none)",
            )
            if effort == "__back__":
                return
            if effort != "(none)":
                draft["reasoning_effort"] = effort

    # Pass 2: fetch models (needs base_url/api_key for some providers)
    if "model" in fields:
        rich_print(f"[{_SEP_COLOR}]Fetching models…[/{_SEP_COLOR}]")
        if provider_type == "openai_compatible":
            base = draft.get("base_url", "")
            if base and not base.rstrip("/").endswith("/v1"):
                draft["base_url"] = base.rstrip("/") + "/v1"
        list_kwargs = {k: v for k, v in draft.items() if k != "provider_type"}
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
            selected = _ask(
                questionary.text(
                    "Model name:",
                    default=default_model,
                    style=_WIZARD_STYLE,
                )
            )
            if selected is None:
                return
        draft["model"] = selected

    # Nothing has been written yet: the draft is saved by the Save choice of the
    # action menu, and the same-type overwrite confirmation runs at that moment.
    _provider_draft_actions(name, draft)


def _test_provider_standalone() -> None:
    """Test connection for any existing provider (standalone flow)."""
    providers = Config.get_providers()
    if not providers:
        rich_print(f"[{_WARNING_COLOR}]No providers configured.[/{_WARNING_COLOR}]")
        return

    choices = [
        questionary.Choice(_format_provider_label(name, providers[name]), value=name)
        for name in providers
    ]
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
    """Edit an existing provider as a draft.

    The saved provider is loaded into a draft and every field edit updates the
    draft only: nothing is written until Save on the action menu. A cancel
    (Escape) at a field value prompt cancels that field only and returns to the
    field menu; a cancel at the field menu discards the whole draft, so the
    provider stays as it was. Unknown keys and OAuth data are preserved, and
    environment values are never part of the draft.
    """
    providers = Config.get_providers()
    if not providers:
        rich_print(f"[{_WARNING_COLOR}]No providers configured.[/{_WARNING_COLOR}]")
        return

    choices = [
        questionary.Choice(_format_provider_label(name, providers[name]), value=name)
        for name in providers
    ]
    original_name = _select_with_back("Select provider to edit:", choices=choices)
    if original_name == "__back__":
        return

    # The draft is a copy, so the saved entry is untouched until Save writes it.
    draft: dict[str, Any] = dict(providers[original_name])
    name = original_name

    while True:
        field = _select_with_back(
            "Select field to edit:",
            choices=_provider_edit_choices(draft, name),
        )
        if field == "__back__":
            rich_print(f"[{_SEP_COLOR}]Draft discarded, nothing saved.[/{_SEP_COLOR}]")
            return
        if field == "__done__":
            break

        if field == "__clear_key__":
            # The only way to clear a key: a blank entry keeps the current one.
            draft["api_key"] = ""
            rich_print(
                f"[{_SEP_COLOR}]API key cleared for the draft; Save writes it."
                f"[/{_SEP_COLOR}]"
            )
            continue

        if field == "name":
            new_name = _prompt_provider_name(name)
            if new_name is None:
                rich_print(
                    f"[{_SEP_COLOR}]Edit cancelled, no changes made.[/{_SEP_COLOR}]"
                )
                continue
            name = new_name
            continue

        value = (
            _prompt_provider_model(draft)
            if field == "model"
            else _prompt_provider_field(draft, field)
        )
        if value is None:
            # Cancel this field only: the value already in the draft is kept.
            rich_print(f"[{_SEP_COLOR}]Edit cancelled, no changes made.[/{_SEP_COLOR}]")
            continue

        if field == "reasoning_effort" and value == "(none)":
            # "(none)" means the field is absent, not that it holds that text.
            draft.pop(field, None)
            continue

        draft[field] = value

    _provider_draft_actions(name, draft, original_name)


def _remove_provider_flow() -> None:
    """Remove a provider, but only while nothing refers to it.

    A provider that is still referenced is reported with the settings that depend on
    it and left in place, so the user can change those first. Only key names are
    shown, never a stored value. Cancel removes nothing.
    """
    providers = Config.get_providers()
    if not providers:
        rich_print(f"[{_WARNING_COLOR}]No providers configured.[/{_WARNING_COLOR}]")
        return

    choices = [
        questionary.Choice(_format_provider_label(name, providers[name]), value=name)
        for name in providers
    ]
    selected = _select_with_back("Select provider to remove:", choices=choices)
    if selected == "__back__":
        return

    references = Config.get_provider_references(selected)
    if references:
        rich_print(
            f"[{_WARNING_COLOR}]Provider "
            f"'[{_VALUE_COLOR}]{selected}[/{_VALUE_COLOR}]' is still used "
            f"by: {', '.join(references)}.[/{_WARNING_COLOR}]"
        )
        rich_print(
            f"[{_SEP_COLOR}]Change those settings first; nothing was removed."
            f"[/{_SEP_COLOR}]"
        )
        return

    confirm = _ask(
        questionary.confirm(
            f"Remove provider '{selected}'?", default=False, style=_WIZARD_STYLE
        )
    )
    if confirm is None:
        # Cancel at the confirm does not remove anything.
        return
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

    choices = [
        questionary.Choice(_format_provider_label(name, providers[name]), value=name)
        for name in providers
    ]
    selected = _select_with_back("Set default provider:", choices=choices)
    if selected == "__back__":
        return

    Config.set_default_provider(selected)
    rich_print(
        f"[{_SUCCESS_COLOR}]Default provider set to "
        f"'[{_VALUE_COLOR}]{selected}[/{_VALUE_COLOR}]'.[/{_SUCCESS_COLOR}]"
    )


def _set_provider_order_flow() -> None:
    """Set provider failover order."""
    providers = Config.get_providers()
    if not providers:
        rich_print(f"[{_WARNING_COLOR}]No providers configured.[/{_WARNING_COLOR}]")
        return

    current_order = Config.get_provider_order()
    if current_order:
        rich_print(
            f"[{_SEP_COLOR}]Current order: "
            f"[{_VALUE_COLOR}]{' > '.join(current_order)}[/{_VALUE_COLOR}]"
            f"[/{_SEP_COLOR}]"
        )

    ordered: list[str] = []
    remaining = list(providers.keys())

    while remaining:
        choices = [
            questionary.Choice(
                _format_provider_label(name, providers[name]), value=name
            )
            for name in remaining
        ]
        if ordered:
            choices.append(questionary.Choice("Done", value="__done__"))

        selected = _select_with_back(
            "Select next provider in failover order:",
            choices=choices,
        )
        if selected == "__back__":
            return
        if selected == "__done__":
            break

        ordered.append(selected)
        remaining.remove(selected)
        rich_print(
            f"[{_SEP_COLOR}]Order: "
            f"[{_VALUE_COLOR}]{' > '.join(ordered)}[/{_VALUE_COLOR}]"
            f"[/{_SEP_COLOR}]"
        )

    if not ordered:
        rich_print(f"[{_SEP_COLOR}]No changes made.[/{_SEP_COLOR}]")
        return

    include_remaining = False
    if remaining:
        include_remaining = _ask(
            questionary.confirm(
                "Append unselected providers at the end?",
                default=True,
                style=_WIZARD_STYLE,
            )
        )
        if include_remaining is None:
            # Cancel here writes nothing: the order built so far is discarded.
            rich_print(f"[{_SEP_COLOR}]Cancelled, order not saved.[/{_SEP_COLOR}]")
            return
    if include_remaining:
        ordered.extend(remaining)

    Config.set_provider_order(ordered)
    rich_print(
        f"[{_SUCCESS_COLOR}]Provider order set to "
        f"[{_VALUE_COLOR}]{' > '.join(ordered)}[/{_VALUE_COLOR}]. "
        f"Default provider is '[{_VALUE_COLOR}]{ordered[0]}[/{_VALUE_COLOR}]'."
        f"[/{_SUCCESS_COLOR}]"
    )


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


# ---------------------------------------------------------------------------
# Task selection (D4)
# ---------------------------------------------------------------------------
TaskRole = Literal["chat", "embedding", "rerank", "ocr", "metadata", "cleanup"]


@dataclass(frozen=True)
class PairSelection:
    """One saved provider plus one model id for this task."""

    provider_name: str
    model_id: str


@dataclass(frozen=True)
class LinkedProviderSelection:
    """Follow the provider's chat model through the legacy ``provider:<name>`` form."""

    provider_name: str


@dataclass(frozen=True)
class ChatDefaultSelection:
    """Follow the default chat provider (``main``)."""


@dataclass(frozen=True)
class OffSelection:
    """Task disabled / image fallback (empty model)."""


@dataclass(frozen=True)
class DirectSelection:
    """Direct endpoint with no provider reference.

    ``api_key`` is the key this selection carries: ``None`` means keep the saved
    ``embedding_api_key``, ``""`` means clear it, and a string is the key to write.
    A selection is a value, so the key is written only by Save.
    """

    backend: str
    host: str
    model_id: str
    api_key: str | None = None


@dataclass(frozen=True)
class LocalSelection:
    """Local model, pinned by the task."""

    model_id: str


@dataclass(frozen=True)
class CustomSelection:
    """Input action only. It is resolved to a :class:`PairSelection` before Save."""

    provider_name: str
    model_id: str


TaskSelection = (
    PairSelection
    | LinkedProviderSelection
    | ChatDefaultSelection
    | OffSelection
    | DirectSelection
    | LocalSelection
    | CustomSelection
)

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

# Catalog role used for discovery. OCR, metadata and cleanup use chat entries;
# OCR additionally reads the vision metadata.
_TASK_CATALOG_ROLE: dict[TaskRole, CatalogRole] = {
    "chat": "chat",
    "embedding": "embedding",
    "rerank": "rerank",
    "ocr": "chat",
    "metadata": "chat",
    "cleanup": "chat",
}

# Provider types the catalog can query. Every other saved provider shows only
# its saved pair.
_DISCOVERY_TYPES = frozenset({"openai_compatible", "openai", "ollama"})

# Provider types each task can hold a reference for. A saved provider of another
# type is not offered for that task, and its saved pair is not shown either.
_TASK_PROVIDER_TYPES: dict[TaskRole, frozenset[str]] = {
    "chat": frozenset(PROVIDER_NAMES),
    "embedding": frozenset({"openai_compatible", "ollama"}),
    "rerank": frozenset({"openai_compatible"}),
    "ocr": frozenset({"openai_compatible", "ollama"}),
    "metadata": frozenset(PROVIDER_NAMES),
    "cleanup": frozenset(PROVIDER_NAMES),
}

# Custom pairs are limited to types where a typed model is meaningful: the Ollama
# embedding backend pins its model, so a custom model there would be ignored.
_CUSTOM_PROVIDER_TYPES: dict[TaskRole, frozenset[str]] = {
    **_TASK_PROVIDER_TYPES,
    "embedding": frozenset({"openai_compatible"}),
}

# Row sentinels. A real selection is a dataclass instance, so a row value is
# never a display string.
_CUSTOM_VALUE = "__custom__"
_RETRY_VALUE = "__retry__"
_BACK_VALUE = "__back__"
_DIRECT_SETTINGS_VALUE = "__direct_settings__"
_LOCAL_CUSTOM_VALUE = "__local_custom__"
_CORPUS_OLLAMA_CACHE_KEY = "__corpus_ollama__"

_LOCAL_LABEL = "Local"
_OLLAMA_LOCAL_LABEL = "Ollama (local)"
_DEFAULT_OLLAMA_EMBEDDING_HOST_LABEL = "localhost:1234"

_RESTART_NOTE = "Running ESDC services may need restart to read the saved config."

_RERANK_ENABLE_ERROR = (
    "Cannot enable reranking: the saved source or model is incomplete. "
    "Choose a source and model first."
)

# Selection types Save accepts for each task. Custom is never saved.
_SAVEABLE_SELECTIONS: dict[TaskRole, tuple[type, ...]] = {
    "chat": (PairSelection,),
    "embedding": (PairSelection, LocalSelection, DirectSelection),
    "rerank": (PairSelection, LocalSelection),
    "ocr": (PairSelection, LocalSelection),
    "metadata": (
        PairSelection,
        LinkedProviderSelection,
        ChatDefaultSelection,
        OffSelection,
        LocalSelection,
    ),
    "cleanup": (
        PairSelection,
        LinkedProviderSelection,
        ChatDefaultSelection,
        OffSelection,
        LocalSelection,
    ),
}


def _selection_label(
    role: TaskRole,
    selection: TaskSelection,
    *,
    unverified: bool = False,
) -> str:
    """The one label source, shared by the picker rows and the task summary.

    A pair row is exactly ``provider / model``; the ``(saved, unverified)``
    suffix is added only for a saved pair that discovery did not report.
    """
    if isinstance(selection, PairSelection):
        label = f"{selection.provider_name} / {selection.model_id}"
        return f"{label} (saved, unverified)" if unverified else label
    if isinstance(selection, LinkedProviderSelection):
        return f"{selection.provider_name} / (chat model)"
    if isinstance(selection, ChatDefaultSelection):
        return "Chat setting / main"
    if isinstance(selection, OffSelection):
        return "Off / image fallback" if role == "metadata" else "Off"
    if isinstance(selection, DirectSelection):
        if selection.backend == "ollama":
            # The direct Ollama backend is its own row, not the local GGUF row.
            return f"{_OLLAMA_LOCAL_LABEL} / {selection.model_id}"
        host = selection.host or _DEFAULT_OLLAMA_EMBEDDING_HOST_LABEL
        return f"Direct endpoint / {host} / {selection.model_id}"
    if isinstance(selection, LocalSelection):
        prefix = (
            _OLLAMA_LOCAL_LABEL
            if role in ("ocr", "metadata", "cleanup")
            else _LOCAL_LABEL
        )
        return f"{prefix} / {selection.model_id}"
    raise ValueError(f"No label for selection {type(selection).__name__}")


def _entry_matches_role(entry: ModelEntry, role: TaskRole) -> bool:
    """True when the entry is usable for this task (unknown stays usable)."""
    catalog_role = _TASK_CATALOG_ROLE[role]
    return not entry.roles or catalog_role in entry.roles


def _vision_filter(entries: list[ModelEntry]) -> list[ModelEntry]:
    """OCR rows: drop only explicitly non-vision entries. Unknown stays selectable."""
    if not any(entry.supports_vision is not None for entry in entries):
        return entries
    return [entry for entry in entries if entry.supports_vision is not False]


def _catalog_for(
    cache: dict[str, ModelCatalog], key: str, provider: Mapping[str, Any]
) -> ModelCatalog:
    """One catalog per provider key, fetched at most once per picker open."""
    if key not in cache:
        cache[key] = discover_models(dict(provider))
    return cache[key]


def _provider_type(entry: Mapping[str, Any]) -> str:
    """The stored provider type, normalised."""
    return str(entry.get("provider_type") or entry.get("type") or "").strip().lower()


def _current_task_selection(role: TaskRole) -> TaskSelection | None:
    """The saved selection for one task, read from config."""
    flat = Config.get_all_config_flat()
    providers = Config.get_providers()

    if role == "chat":
        name = str(flat.get("default_provider") or "")
        entry = providers.get(name) if isinstance(providers, Mapping) else None
        model = str((entry or {}).get("model") or "")
        return PairSelection(name, model) if name else None

    if role == "embedding":
        backend = str(flat.get("embedding_backend") or "")
        reference = str(flat.get("embedding_provider") or "")
        model = str(flat.get("embedding_model") or "")
        host = str(flat.get("embedding_host") or "")
        if backend == "local":
            return LocalSelection(MODEL_ID)
        if reference:
            if backend == "ollama":
                # The Ollama embedding backend pins its model: the inactive
                # `embedding_model` is never shown as the selection.
                return PairSelection(reference, OllamaEmbedder.OLLAMA_TAG)
            return PairSelection(reference, model)
        if backend == "openai":
            return DirectSelection("openai", host, model)
        if backend == "ollama":
            # A direct Ollama backend has no provider, so it is its own row rather
            # than the local GGUF row.
            return DirectSelection("ollama", host, OllamaEmbedder.OLLAMA_TAG)
        return None

    if role == "rerank":
        backend = str(flat.get("corpus.rerank_backend") or "")
        model = str(flat.get("corpus.rerank_model") or "") or DEFAULT_RERANKER
        if backend == "provider":
            name = str(flat.get("corpus.rerank_provider") or "")
            # A provider source with no reference stays incomplete: reading it as
            # a local model would let enable pass validation on an incomplete draft.
            return PairSelection(name, model) if name else None
        # A local source names a local GGUF; the default is what the loader uses
        # when the saved value is empty.
        return LocalSelection(model)

    if role == "ocr":
        reference = str(flat.get("corpus.ocr_provider") or "")
        model = str(flat.get("corpus.ocr_model") or "")
        return PairSelection(reference, model) if reference else LocalSelection(model)

    # metadata and cleanup. Runtime gives the explicit reference priority (see
    # `_resolve_text_caller`), so a saved reference is read as a pair first: `main`,
    # `provider:<name>`, or a bare Ollama name is that provider's wire model, not the
    # chat setting or a linked provider.
    reference = str(flat.get(f"corpus.{role}_provider") or "")
    model = str(flat.get(f"corpus.{role}_model") or "")

    if reference:
        return PairSelection(reference, model)

    # Only without a reference do the saved strings keep their legacy meanings.
    if model == "main":
        return ChatDefaultSelection()
    if model.startswith("provider:"):
        return LinkedProviderSelection(model[len("provider:") :])
    if not model:
        return OffSelection()
    return LocalSelection(model)


def _task_summary(role: TaskRole, current: TaskSelection | None) -> str:
    """One plain summary line: selection, status, and any environment override.

    The label already carries the status for an off/fallback selection, so only
    an explicit link and an environment override are added as extra status.
    """
    title = _TASK_TITLES[role]
    if current is None:
        return f"{title}: not set"

    parts = [f"{title}: {_selection_label(role, current)}"]
    if isinstance(current, LinkedProviderSelection):
        parts.append("follows the chat model")
    if role == "embedding" and os.environ.get("ESDC_EMBEDDING_MODEL"):
        parts.append("environment override: ESDC_EMBEDDING_MODEL")
    return "  " + " — ".join(parts)


def _special_selections(
    role: TaskRole,
    flat: Mapping[str, Any],
    providers: Mapping[str, Any],
    cache: dict[str, ModelCatalog],
) -> list[TaskSelection]:
    """The task's own rows, kept at the same list level as the pair rows."""
    selections: list[TaskSelection] = []

    if role == "chat":
        return selections

    if role == "embedding":
        selections.append(LocalSelection(MODEL_ID))

        # Rows for a source with no provider reference: the direct Ollama backend is
        # its own row, distinct from the local GGUF row, and an existing direct
        # OpenAI-compatible endpoint keeps its own row.
        if not str(flat.get("embedding_provider") or ""):
            backend = str(flat.get("embedding_backend") or "")
            host = str(flat.get("embedding_host") or "")
            model = str(flat.get("embedding_model") or "")
            if backend == "openai" and model:
                selections.append(DirectSelection("openai", host, model))
            selections.append(
                DirectSelection("ollama", host, OllamaEmbedder.OLLAMA_TAG)
            )
        return selections

    if role == "rerank":
        # A local row is a local GGUF, so the saved model is only a local row while
        # the saved source is local: a provider source stores a remote model id in the
        # same key, which the loader cannot read as a GGUF.
        saved_local = (
            str(flat.get("corpus.rerank_model") or "")
            if str(flat.get("corpus.rerank_backend") or "") == "local"
            else ""
        )
        # The registry default is always visible, and a saved local model that is not
        # the default keeps its own row next to it.
        selections.append(LocalSelection(DEFAULT_RERANKER))
        if saved_local and saved_local != DEFAULT_RERANKER:
            selections.append(LocalSelection(saved_local))
        return selections

    if role == "ocr":
        model = str(flat.get("corpus.ocr_model") or "")
        if model:
            selections.append(LocalSelection(model))
        return selections

    # metadata and cleanup: the chat-default, off, and linked rows, plus the
    # models the corpus Ollama host reports.
    selections.append(ChatDefaultSelection())
    selections.append(OffSelection())

    # A linked row follows that provider's chat model, so a provider with no saved
    # model has nothing to follow and is not offered.
    for name, entry in providers.items():
        if isinstance(entry, Mapping) and str(entry.get("model") or ""):
            selections.append(LinkedProviderSelection(str(name)))

    ollama_host = str(flat.get("corpus.ollama_host") or "")
    catalog = _catalog_for(
        cache,
        _CORPUS_OLLAMA_CACHE_KEY,
        {"provider_type": "ollama", "base_url": ollama_host},
    )
    for entry in catalog.entries:
        if _entry_matches_role(entry, role):
            selections.append(LocalSelection(entry.model_id))
    return selections


def _selectable(selection: TaskSelection) -> bool:
    """A row that names a model can be chosen; a row with an empty model cannot.

    Rows that follow something else (the chat model, off, the chat default) carry
    no model id of their own and stay selectable.
    """
    if isinstance(selection, (PairSelection, LocalSelection, DirectSelection)):
        return bool(str(selection.model_id).strip())
    return True


def _task_rows(
    role: TaskRole,
    current: TaskSelection | None,
    cache: dict[str, ModelCatalog],
) -> list[questionary.Choice]:
    """One combined searchable list of rows for one task (D4/D6)."""
    flat = Config.get_all_config_flat()
    providers = Config.get_providers()
    saved = _current_task_selection(role)

    rows = [
        questionary.Choice(_selection_label(role, selection), value=selection)
        for selection in _special_selections(role, flat, providers, cache)
        if _selectable(selection)
    ]

    if role == "embedding":
        # The direct endpoint's own fields: a direct configuration has no provider to
        # select, so this row opens an editor rather than a provider submenu.
        rows.append(
            questionary.Choice(
                "Direct embedding settings…", value=_DIRECT_SETTINGS_VALUE
            )
        )

    if role == "rerank":
        # A local GGUF outside the known registry is typed here, at the same level as
        # the local rows. The row is an input action, never a value Save can accept.
        rows.append(
            questionary.Choice("Local custom model…", value=_LOCAL_CUSTOM_VALUE)
        )

    for name, entry in providers.items():
        if not isinstance(entry, Mapping):
            continue

        provider_type = _provider_type(entry)
        if provider_type not in _TASK_PROVIDER_TYPES[role]:
            # This provider cannot serve the task, so neither its rows nor a saved
            # pair pointing at it are shown.
            continue

        # A saved pair with no model is not a row: there is nothing to select.
        saved_pair = (
            saved
            if isinstance(saved, PairSelection)
            and saved.provider_name == name
            and _selectable(saved)
            else None
        )

        if role == "embedding" and provider_type == "ollama":
            # The Ollama embedding backend pins its model, so the row shows the
            # pinned tag and no discovery request is made for it.
            pair = saved_pair or PairSelection(str(name), OllamaEmbedder.OLLAMA_TAG)
            rows.append(questionary.Choice(_selection_label(role, pair), value=pair))
            continue

        if provider_type not in _DISCOVERY_TYPES:
            model = saved_pair.model_id if saved_pair else str(entry.get("model") or "")
            if model:
                pair = saved_pair or PairSelection(str(name), model)
                rows.append(
                    questionary.Choice(
                        _selection_label(role, pair, unverified=True), value=pair
                    )
                )
            continue

        catalog = _catalog_for(cache, str(name), entry)
        all_entries = [
            item for item in catalog.entries if _entry_matches_role(item, role)
        ]
        # OCR shows known-vision and unknown entries only; the saved pair is judged
        # against what discovery reported, not against the rows that passed the filter.
        entries = _vision_filter(all_entries) if role == "ocr" else all_entries

        for item in entries:
            pair = PairSelection(str(name), item.model_id)
            if _selectable(pair):
                rows.append(
                    questionary.Choice(_selection_label(role, pair), value=pair)
                )

        if saved_pair and not any(
            item.model_id == saved_pair.model_id for item in entries
        ):
            # A saved pair stays visible when discovery did not report it, or reported
            # no vision flag. An OCR model that discovery reports as non-vision has no
            # usable row here, so it is not re-added.
            reported = next(
                (item for item in all_entries if item.model_id == saved_pair.model_id),
                None,
            )
            if (
                role != "ocr"
                or reported is None
                or reported.supports_vision is not False
            ):
                rows.append(
                    questionary.Choice(
                        _selection_label(role, saved_pair, unverified=True),
                        value=saved_pair,
                    )
                )

    rows.append(questionary.Choice("Custom provider / model…", value=_CUSTOM_VALUE))
    rows.append(questionary.Choice("Retry discovery", value=_RETRY_VALUE))
    rows.append(questionary.Choice("← Back", value=_BACK_VALUE))
    return rows


def _custom_pair_flow(
    role: TaskRole, current: TaskSelection | None
) -> PairSelection | None:
    """Custom entry: an existing saved provider, then a typed model id.

    Only provider types this task can hold a reference for are offered, and the
    typed model must be non-empty, so the resolved pair is always one Save can
    accept. ``None`` means cancel, which returns to the picker.
    """
    providers = Config.get_providers()
    names = [
        str(name)
        for name, entry in providers.items()
        if isinstance(entry, Mapping)
        and _provider_type(entry) in _CUSTOM_PROVIDER_TYPES[role]
    ]
    if not names:
        rich_print(
            f"[{_WARNING_COLOR}]No saved provider can hold a"
            f" {_TASK_TITLES[role].lower()} reference.[/{_WARNING_COLOR}]"
        )
        return None

    provider_name = _ask(
        questionary.select(
            "Custom pair — select provider:",
            choices=[questionary.Choice(name, value=name) for name in names],
            style=_WIZARD_STYLE,
        )
    )
    if provider_name is None:
        return None

    default_model = current.model_id if isinstance(current, PairSelection) else ""
    model_id = _ask(
        questionary.text(
            "Custom pair — model id:",
            default=default_model,
            style=_WIZARD_STYLE,
        )
    )
    if model_id is None:
        return None

    model_id = str(model_id).strip()
    if not model_id:
        rich_print(f"[{_ERROR_COLOR}]A custom pair needs a model id.[/{_ERROR_COLOR}]")
        return None

    return PairSelection(str(provider_name), model_id)


def _direct_embedding_choices(draft: dict[str, Any]) -> list[questionary.Choice]:
    """The direct endpoint's own fields, plus Clear and Done, at one menu level.

    A key the draft does not carry is shown as "(kept)": a blank entry keeps the
    saved key, so clearing it stays the explicit Clear API key action.
    """
    key = draft.get("api_key")
    if key is None:
        key_display = "(kept)"
    else:
        key_display = _mask_value("api_key", key) or "(cleared)"

    return [
        questionary.Choice(
            f"host = {draft['host'] or _DEFAULT_OLLAMA_EMBEDDING_HOST_LABEL}",
            value="host",
        ),
        questionary.Choice(f"model = {draft['model'] or '(none)'}", value="model"),
        questionary.Choice(f"api_key = {key_display}", value="api_key"),
        questionary.Choice("Clear API key", value="__clear_key__"),
        questionary.Choice("Done editing", value="__done__"),
    ]


def _direct_embedding_flow() -> DirectSelection | None:
    """Edit the direct embedding settings. Nothing is written here.

    The draft starts from the saved direct fields: host, model, and the saved key.
    The draft key is ``None`` while it is only preserving the saved value, so Done
    cannot write it; a typed value replaces it, and Clear API key sets it to ``""``.
    Save happens on the task action menu for the returned selection, exactly as for a
    pair. A cancel (Back or Escape at the menu) discards the draft and returns None;
    a cancel at one field keeps the value already in the draft.
    """
    flat = Config.get_all_config_flat()
    draft: dict[str, Any] = {
        "host": str(flat.get("embedding_host") or ""),
        "model": str(flat.get("embedding_model") or ""),
        # None preserves the saved key; "" is an explicit clear.
        "api_key": None,
    }

    while True:
        field = _select_with_back(
            "Direct embedding settings:",
            choices=_direct_embedding_choices(draft),
        )
        if field == "__back__":
            rich_print(f"[{_SEP_COLOR}]Draft discarded, nothing saved.[/{_SEP_COLOR}]")
            return None

        if field == "__done__":
            if not str(draft["model"]).strip():
                rich_print(
                    f"[{_ERROR_COLOR}]Direct embedding settings need a model "
                    f"before Done.[/{_ERROR_COLOR}]"
                )
                continue
            return DirectSelection(
                "openai",
                str(draft["host"]).strip(),
                str(draft["model"]).strip(),
                api_key=draft["api_key"],
            )

        if field == "__clear_key__":
            # The only way to clear a key: a blank entry keeps the current one.
            draft["api_key"] = ""
            rich_print(
                f"[{_SEP_COLOR}]API key cleared for the draft; Save writes it."
                f"[/{_SEP_COLOR}]"
            )
            continue

        if field == "host":
            value = _ask(
                questionary.text(
                    "Embedding host (blank uses the default endpoint):",
                    default=draft["host"],
                    style=_WIZARD_STYLE,
                )
            )
            if value is None:
                rich_print(
                    f"[{_SEP_COLOR}]Edit cancelled, no changes made.[/{_SEP_COLOR}]"
                )
                continue
            # An empty host is the default endpoint, so "" is accepted.
            draft["host"] = str(value).strip()
            continue

        if field == "api_key":
            value = _ask(
                questionary.password(
                    "API key (blank keeps current):", style=_WIZARD_STYLE
                )
            )
            if value is None:
                rich_print(
                    f"[{_SEP_COLOR}]Edit cancelled, no changes made.[/{_SEP_COLOR}]"
                )
                continue
            if value:
                draft["api_key"] = str(value)
            continue

        # field == "model"
        value = _ask(
            questionary.text(
                "Embedding model:",
                default=draft["model"],
                style=_WIZARD_STYLE,
            )
        )
        if value is None:
            rich_print(f"[{_SEP_COLOR}]Edit cancelled, no changes made.[/{_SEP_COLOR}]")
            continue
        draft["model"] = str(value).strip()


def _local_custom_rerank_flow() -> LocalSelection | None:
    """One typed local rerank GGUF, resolved to a LocalSelection. Nothing is written.

    The value is the loader's own form: `repo_id:filename.gguf`, or a name the rerank
    registry knows. An empty or whitespace-only value is refused and returns to the
    picker, and a cancel (Escape) returns there without a value, so the sentinel row
    itself never reaches Save.
    """
    value = _ask(
        questionary.text(
            "Local rerank model (repo_id:filename.gguf, or a known registry name):",
            style=_WIZARD_STYLE,
        )
    )
    if value is None:
        return None

    model = str(value).strip()
    if not model:
        rich_print(
            f"[{_ERROR_COLOR}]A local rerank model needs a GGUF id "
            f"('repo_id:filename.gguf') or a known registry name.[/{_ERROR_COLOR}]"
        )
        return None
    return LocalSelection(model)


def _choose_provider_model(
    role: TaskRole, current: TaskSelection | None = None
) -> TaskSelection | None:
    """One searchable provider/model list. ``None`` means Back or Escape."""
    cache: dict[str, ModelCatalog] = {}

    while True:
        rows = _task_rows(role, current, cache)
        for key, catalog in cache.items():
            if catalog.warning:
                name = key if key != _CORPUS_OLLAMA_CACHE_KEY else "corpus Ollama host"
                rich_print(
                    f"[{_WARNING_COLOR}]{name}: {catalog.warning}[/{_WARNING_COLOR}]"
                )

        selected = _ask(
            questionary.select(
                f"{_TASK_TITLES[role]} — provider / model",
                choices=rows,
                use_search_filter=True,
                use_jk_keys=False,
                style=_WIZARD_STYLE,
            )
        )

        if selected is None or selected == _BACK_VALUE:
            return None
        if selected == _RETRY_VALUE:
            # Retry clears the local cache and re-opens the picker once.
            cache.clear()
            continue
        if selected == _CUSTOM_VALUE:
            resolved = _custom_pair_flow(role, current)
            if resolved is None:
                continue
            return resolved
        if selected == _DIRECT_SETTINGS_VALUE:
            # The editor writes nothing; Done hands the draft back as a selection,
            # and a cancel returns to this same list.
            resolved = _direct_embedding_flow()
            if resolved is None:
                continue
            return resolved
        if selected == _LOCAL_CUSTOM_VALUE:
            # The row resolves to a LocalSelection; the sentinel itself is never a
            # value Save can accept, and a refused or cancelled input comes back here.
            resolved = _local_custom_rerank_flow()
            if resolved is None:
                continue
            return resolved
        if isinstance(selected, TaskSelection):
            return selected
        return None


def _pair_provider_type(role: TaskRole, selection: PairSelection) -> str:
    """The saved provider type behind a pair, guarded for this task.

    Raises:
        ValueError: the provider is unknown, or its type cannot hold a reference
            for this task.
    """
    provider = Config.get_referenced_provider(selection.provider_name)
    provider_type = _provider_type(provider)
    if provider_type not in _TASK_PROVIDER_TYPES[role]:
        raise ValueError(
            f"{selection.provider_name!r} is a {provider_type or 'unknown'} provider"
            f" and cannot be a {_TASK_TITLES[role].lower()} reference"
        )
    return provider_type


def _ocr_vision_flag(
    selection: PairSelection | LocalSelection,
) -> bool | None:
    """What discovery reports about one OCR model: True, False, or None (unknown).

    A pair is checked against its saved provider; a local OCR selection against the
    synthetic Ollama provider built from `corpus.ollama_host`, which is the source the
    runtime would use. Discovery is best effort: a catalog that comes back with a
    warning, an unknown model, or an unknown flag stays unknown and may print one short
    warning line, so a save is never blocked on a metadata gap. Only an explicit False
    blocks, and that decision belongs to the caller.
    """
    if isinstance(selection, PairSelection):
        providers = Config.get_providers()
        entry = (
            providers.get(selection.provider_name)
            if isinstance(providers, Mapping)
            else None
        )
        if not isinstance(entry, Mapping):
            return None
        provider: dict[str, Any] = dict(entry)
    else:
        flat = Config.get_all_config_flat()
        provider = {
            "provider_type": "ollama",
            "base_url": str(flat.get("corpus.ollama_host") or ""),
        }

    try:
        catalog = discover_models(provider)
        entries = list(catalog.entries)
        warning = str(catalog.warning or "")
    except Exception as error:
        rich_print(
            f"[{_SEP_COLOR}]OCR vision metadata: {_short_test_message(error)}.[/]"
        )
        return None

    if warning:
        # `discover_models` does not raise: a network, HTTP, or shape failure comes
        # back as a catalog with a warning. Report it and keep the model unknown, so a
        # metadata gap never blocks the save. The warning text is credential-free.
        rich_print(f"[{_SEP_COLOR}]OCR vision metadata: {warning}.[/]")

    for item in entries:
        if item.model_id == selection.model_id:
            return item.supports_vision
    return None


def _save_task_selection(role: TaskRole, selection: TaskSelection) -> None:
    """Save one task's fields only; everything else is preserved.

    Only concrete values reach config: no union tag and no display label, and an
    incomplete concrete selection is rejected before anything is written.

    Raises:
        ValueError: the selection does not belong to this task, it is incomplete,
            its provider type cannot hold a reference for this task, or for OCR the
            model is one the provider reports as non-vision.
    """
    if not isinstance(selection, _SAVEABLE_SELECTIONS[role]):
        raise ValueError(f"{type(selection).__name__} is not a {role} selection")

    # A row that cannot be chosen should never reach Save, but a value from
    # another entry point can: an incomplete one is refused, not written.
    if isinstance(selection, (PairSelection, LocalSelection, DirectSelection)):
        if not str(selection.model_id).strip():
            raise ValueError(f"{_TASK_TITLES[role]} selection needs a model")
        if (
            isinstance(selection, PairSelection)
            and not str(selection.provider_name).strip()
        ):
            raise ValueError(f"{_TASK_TITLES[role]} selection needs a provider")

    if role == "chat":
        if not isinstance(selection, PairSelection):
            raise ValueError(f"{type(selection).__name__} is not a chat selection")
        Config.set_chat_selection(selection.provider_name, selection.model_id)
        return

    if role == "embedding":
        if isinstance(selection, PairSelection):
            provider_type = _pair_provider_type(role, selection)
            if provider_type == "ollama":
                # The Ollama backend pins its model: embedding_model is preserved.
                values = {
                    "embedding_backend": "ollama",
                    "embedding_provider": selection.provider_name,
                }
            else:
                values = {
                    "embedding_backend": "openai",
                    "embedding_provider": selection.provider_name,
                    "embedding_model": selection.model_id,
                }
        elif isinstance(selection, LocalSelection):
            # Service fields stay as they are.
            values = {"embedding_backend": "local"}
        elif isinstance(selection, DirectSelection):
            # A direct selection clears the reference and writes its own endpoint
            # fields. The key is written only when the selection carries one: None
            # means the saved `embedding_api_key` is preserved.
            if selection.backend == "ollama":
                # The Ollama backend pins its model, so the saved model and key stay.
                values = {
                    "embedding_backend": "ollama",
                    "embedding_provider": "",
                    "embedding_host": selection.host,
                }
            else:
                values = {
                    "embedding_backend": "openai",
                    "embedding_provider": "",
                    "embedding_host": selection.host,
                    "embedding_model": selection.model_id,
                }
                if selection.api_key is not None:
                    values["embedding_api_key"] = selection.api_key
        else:
            raise ValueError(
                f"{type(selection).__name__} is not an embedding selection"
            )

        Config.set_config_values(values)
        return

    if role == "rerank":
        if isinstance(selection, PairSelection):
            _pair_provider_type(role, selection)
            values = {
                "corpus.rerank_backend": "provider",
                "corpus.rerank_provider": selection.provider_name,
                "corpus.rerank_model": selection.model_id,
            }
        elif isinstance(selection, LocalSelection):
            values = {
                "corpus.rerank_backend": "local",
                "corpus.rerank_provider": "",
                "corpus.rerank_model": selection.model_id,
            }
        else:
            raise ValueError(f"{type(selection).__name__} is not a rerank selection")

        # The enable switch is never part of a source/model save.
        Config.set_config_values(values)
        return

    provider_key = f"corpus.{role}_provider"
    model_key = f"corpus.{role}_model"

    if isinstance(selection, PairSelection):
        # A wire-model pair names a saved provider, so that provider must exist.
        if role == "ocr":
            _pair_provider_type(role, selection)
        else:
            Config.get_referenced_provider(selection.provider_name)
        values = {provider_key: selection.provider_name, model_key: selection.model_id}
    elif isinstance(selection, LinkedProviderSelection):
        values = {provider_key: "", model_key: f"provider:{selection.provider_name}"}
    elif isinstance(selection, ChatDefaultSelection):
        values = {provider_key: "", model_key: "main"}
    elif isinstance(selection, OffSelection):
        values = {provider_key: "", model_key: ""}
    elif isinstance(selection, LocalSelection):
        values = {provider_key: "", model_key: selection.model_id}
    else:
        raise ValueError(f"{type(selection).__name__} is not a {role} selection")

    # OCR needs a vision-capable model. Only an explicit `false` from discovery refuses
    # the save: unknown metadata never blocks, and nothing is written when it does.
    if (
        role == "ocr"
        and isinstance(selection, (PairSelection, LocalSelection))
        and _ocr_vision_flag(selection) is False
    ):
        raise ValueError(
            f"OCR model {selection.model_id!r} is reported as non-vision by the "
            "provider, so it cannot be saved as corpus.ocr_model"
        )

    Config.set_config_values(values)


# One short text for the embedding connection test. It is not PROBE_TEXT: the test
# only proves the endpoint answers, it never compares against a stored vector.
_EMBEDDING_TEST_TEXT = "esdc embedding connection test"

# The OCR test sends its own image, so a reply that reads it back proves the endpoint
# took an image and looked at it. Answering with nothing, or ignoring the image, fails.
_OCR_TEST_PROMPT = "Return the text in this image exactly."
_OCR_TEST_TEXT = "ESDC OCR TEST 12345"
_OCR_TEST_MARKER = "12345"


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


def _test_embedding_selection(role: TaskRole, selection: TaskSelection) -> None:
    """Test one embedding selection through the embedding endpoint.

    The embedding backend has its own route, so the provider connection test is not
    a valid test for it: an embedding-only deployment answers 404 for the chat
    route. This path builds the same embedder runtime builds and asks it for one
    vector, so a success means the embedding endpoint really responded.

    The test draft is resolved with `_embedding_test_draft`, so the runtime
    environment values apply here as they do at runtime. The address follows the same
    precedence runtime uses: `ESDC_EMBEDDING_HOST`, then the selected provider's own
    `base_url`, then the saved `embedding_host`; `corpus.ollama_host` is never a
    fallback for an embedding test, and a named provider with no address at all is
    reported instead of tested against a default host. A direct row keeps its existing
    behavior, so an empty host stays the embedder's own default endpoint. Nothing is
    saved, nothing is enabled, and a local embedding model is never loaded for a test.
    """
    if isinstance(selection, LocalSelection):
        rich_print(
            f"[{_SEP_COLOR}]Local embedding model; no remote endpoint to test."
            f"[/{_SEP_COLOR}]"
        )
        return

    flat = Config.get_all_config_flat()

    if isinstance(selection, (PairSelection, LinkedProviderSelection)):
        try:
            draft = Config.get_referenced_provider(selection.provider_name)
        except ValueError as error:
            rich_print(f"[{_ERROR_COLOR}]{error}[/{_ERROR_COLOR}]")
            return
        if isinstance(selection, PairSelection):
            draft["model"] = selection.model_id
        source = selection.provider_name
        # Runtime precedence for a named provider: the environment host, then this
        # provider's own address, then the saved embedding host. `corpus.ollama_host`
        # belongs to OCR and to Ollama-named corpus models, so it is not a fallback
        # for the embedding source.
        host_fallback = str(flat.get("embedding_host") or "").strip()
    elif isinstance(selection, DirectSelection):
        draft = {
            "provider_type": (
                "ollama" if selection.backend == "ollama" else "openai_compatible"
            ),
            "base_url": selection.host,
            # None on the selection means the saved key, "" means it was cleared.
            "api_key": (
                selection.api_key
                if selection.api_key is not None
                else str(flat.get("embedding_api_key") or "")
            ),
            "model": selection.model_id,
        }
        source = "direct endpoint"
        # A direct row names its own endpoint: no provider and no saved host stands
        # behind it, and an empty host stays the embedder's own default.
        host_fallback = ""
    else:
        rich_print(
            f"[{_SEP_COLOR}]Nothing to test for "
            f"'{_selection_label(role, selection)}'.[/{_SEP_COLOR}]"
        )
        return

    test_draft = _embedding_test_draft(draft)
    if test_draft is None:
        return

    provider_type = str(test_draft.get("provider_type", "")).strip().lower()
    # `_embedding_test_draft` applied ESDC_EMBEDDING_HOST, so the draft already holds
    # the value that wins; the saved host is the only thing after it.
    host = str(test_draft.get("base_url") or "").strip() or host_fallback
    model = str(test_draft.get("model") or "").strip()

    if not host and isinstance(selection, (PairSelection, LinkedProviderSelection)):
        # A named provider with no address anywhere is reported, not assumed: no
        # embedder is built, and no default or corpus host is substituted for it. A
        # direct row keeps its own behavior and tests the embedder's default endpoint.
        rich_print(
            f"[{_ERROR_COLOR}]Failed: no embedding address for "
            f"'[{_VALUE_COLOR}]{source}[/{_VALUE_COLOR}]'; the provider has no "
            f"base_url and embedding_host is empty.[/{_ERROR_COLOR}]"
        )
        return

    rich_print(
        f"[{_SEP_COLOR}]Testing '[{_KEY_COLOR}]{provider_type}[/{_KEY_COLOR}]' "
        f"embedding endpoint '[{_VALUE_COLOR}]{source}[/{_VALUE_COLOR}]'…"
        f"[/{_SEP_COLOR}]"
    )

    try:
        if provider_type == "ollama":
            # OllamaEmbedder pins its own model tag, so only the resolved address comes
            # from the draft. A named provider always has one by the time this runs; an
            # empty direct host stays the embedder's local default.
            embedder = OllamaEmbedder(host=host or None)
        else:
            embedder = OpenAIEmbedder(
                host=host,
                model=model,
                api_key=str(test_draft.get("api_key") or ""),
            )
        vector = embedder.generate_embedding(_EMBEDDING_TEST_TEXT)
    except Exception as error:
        message = _short_test_message(error)
        rich_print(f"[{_ERROR_COLOR}]Failed: {message}[/{_ERROR_COLOR}]")
        return

    if not vector or not all(isinstance(value, (int, float)) for value in vector):
        label = model or provider_type
        rich_print(
            f"[{_ERROR_COLOR}]Failed: the embedding endpoint returned no vector "
            f"for '[{_VALUE_COLOR}]{label}[/{_VALUE_COLOR}]'.[/{_ERROR_COLOR}]"
        )
        return

    rich_print(
        f"[{_SUCCESS_COLOR}]Success: embedding endpoint responded with a "
        f"{len(vector)}-dimensional vector.[/{_SUCCESS_COLOR}]"
    )


# One short query and two documents for the rerank connection test. The test only
# proves the rerank route answers with one score per input; it never compares scores
# against a stored value, and it is not a chat request.
_RERANK_TEST_QUERY = "esdc rerank connection test"
_RERANK_TEST_DOCUMENTS = ("first test document", "second test document")


def _is_finite_score(value: Any) -> bool:
    """One numeric score that is finite; a bool is not a score."""
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _test_rerank_selection(role: TaskRole, selection: TaskSelection) -> None:
    """Test one rerank selection through the rerank endpoint.

    A chat greeting is not a connection test for a rerank-only deployment: a valid
    rerank model can fail a chat request, and a chat success says nothing about
    rerank scoring. This path builds the same client runtime builds (`ProviderReranker`)
    and asks for one score per test document, so a success means the rerank route
    answered with a complete numeric result.

    A local GGUF is never loaded for a test, nothing is saved, and `corpus.rerank` is
    never touched: the enable switch is not part of a connection test.
    """
    if isinstance(selection, LocalSelection):
        # A local reranker is a model file, not an endpoint; a test would load it.
        rich_print(
            f"[{_SEP_COLOR}]Local reranker; no remote endpoint to test.[/{_SEP_COLOR}]"
        )
        return

    if not isinstance(selection, (PairSelection, LinkedProviderSelection)):
        rich_print(
            f"[{_SEP_COLOR}]Nothing to test for "
            f"'{_selection_label(role, selection)}'.[/{_SEP_COLOR}]"
        )
        return

    try:
        provider = Config.get_referenced_provider(selection.provider_name)
    except ValueError as error:
        rich_print(f"[{_ERROR_COLOR}]{error}[/{_ERROR_COLOR}]")
        return

    source = selection.provider_name
    provider_type = _provider_type(provider)
    if provider_type not in _TASK_PROVIDER_TYPES[role]:
        rich_print(
            f"[{_ERROR_COLOR}]'[{_VALUE_COLOR}]{source}[/{_VALUE_COLOR}]' is a "
            f"{provider_type or 'unknown'} provider and cannot be a "
            f"{_TASK_TITLES[role].lower()} reference.[/{_ERROR_COLOR}]"
        )
        return

    # The pair carries the picked model; a linked row follows the saved chat model.
    picked = selection.model_id if isinstance(selection, PairSelection) else None
    model = str(picked if picked is not None else provider.get("model") or "").strip()
    if not model:
        rich_print(
            f"[{_ERROR_COLOR}]Rerank test needs a model for "
            f"'[{_VALUE_COLOR}]{source}[/{_VALUE_COLOR}]'.[/{_ERROR_COLOR}]"
        )
        return

    try:
        client = ProviderReranker(dict(provider), model)
    except ValueError as error:
        rich_print(f"[{_ERROR_COLOR}]{error}[/{_ERROR_COLOR}]")
        return

    rich_print(
        f"[{_SEP_COLOR}]Testing '[{_KEY_COLOR}]{provider_type}[/{_KEY_COLOR}]' "
        f"rerank endpoint '[{_VALUE_COLOR}]{source}[/{_VALUE_COLOR}]'…[/{_SEP_COLOR}]"
    )

    try:
        scores = client.rerank(_RERANK_TEST_QUERY, list(_RERANK_TEST_DOCUMENTS))
    except Exception as error:
        message = _short_test_message(error)
        rich_print(f"[{_ERROR_COLOR}]Failed: {message}[/{_ERROR_COLOR}]")
        return

    if len(scores) != len(_RERANK_TEST_DOCUMENTS) or not all(
        _is_finite_score(score) for score in scores
    ):
        rich_print(
            f"[{_ERROR_COLOR}]Failed: the rerank endpoint returned an incomplete "
            f"result for '[{_VALUE_COLOR}]{model}[/{_VALUE_COLOR}]'.[/{_ERROR_COLOR}]"
        )
        return

    rich_print(
        f"[{_SUCCESS_COLOR}]Success: rerank endpoint responded with "
        f"{len(scores)} scores.[/{_SUCCESS_COLOR}]"
    )


def _ocr_test_image() -> bytes:
    """One small in-memory PNG that carries the wizard's own test text.

    Nothing is written to disk and no document image is submitted: the reply has to
    read the text the wizard drew, so an endpoint that accepts an image and ignores it
    cannot pass.
    """
    doc = fitz.open()
    try:
        page = doc.new_page(width=220, height=120)
        page.insert_text((20, 60), _OCR_TEST_TEXT, fontsize=18)
        return page.get_pixmap(dpi=150).tobytes("png")
    finally:
        doc.close()


def _test_task_selection(role: TaskRole, selection: TaskSelection) -> None:
    """Resolve the draft and test it. Nothing is saved and nothing is enabled.

    Embeddings are tested through the embedding endpoint (see
    `_test_embedding_selection`) and reranking through the rerank endpoint (see
    `_test_rerank_selection`), because a chat request is not a connection test for a
    deployment that serves only those routes. OCR is tested by sending the wizard's own
    small image (see `_ocr_test_image`) and checking that the reply reads it, so an
    endpoint that accepts an image and ignores it cannot pass. Every other role uses the
    provider connection test. For the embedding task the runtime environment values are
    applied to the test draft (see `_embedding_test_draft`); they are never written to
    config.
    """
    if role == "ocr":
        draft = dict(Config.get_corpus_config())
        if isinstance(selection, PairSelection):
            draft["ocr_provider"] = selection.provider_name
            draft["ocr_model"] = selection.model_id
        elif isinstance(selection, LocalSelection):
            draft["ocr_provider"] = ""
            draft["ocr_model"] = selection.model_id

        label = _selection_label(role, selection)
        try:
            client = create_ocr_client(draft)
            if not client.health_check():
                # Availability is checked first: an endpoint that does not answer is
                # reported as it was before, and no image is sent.
                rich_print(
                    f"[{_WARNING_COLOR}]OCR endpoint is not available for "
                    f"'{label}'.[/{_WARNING_COLOR}]"
                )
                return

            answer = client.query_image(_ocr_test_image(), _OCR_TEST_PROMPT)
        except Exception as error:
            # Construction, request, response, and image errors stop this test only:
            # nothing is saved, nothing is enabled, and no other provider is tried.
            detail = f"{_short_test_message(error)}."
            if _error_status(error) == 400:
                # A 400 on an image request usually means the model is text-only.
                detail += (
                    " The gateway rejected the image; the model may not support "
                    "vision."
                )
            rich_print(f"[{_ERROR_COLOR}]Failed: {detail}[/]")
            return

        text = str(answer or "").strip()
        if _OCR_TEST_MARKER in text or "esdc" in text.lower():
            rich_print(f"[{_SUCCESS_COLOR}]OCR image test passed for '{label}'.[/]")
        else:
            # An endpoint that accepts an image and ignores it is not a vision model.
            # Only the outcome is printed: the reply text can hold whatever the model
            # was shown, so no excerpt of it belongs in a warning line.
            rich_print(
                f"[{_WARNING_COLOR}]Failed: the reply did not read the test image."
                f"[/{_WARNING_COLOR}]"
            )
        return

    if role == "embedding":
        _test_embedding_selection(role, selection)
        return

    if role == "rerank":
        # A chat request cannot verify a rerank-only model, so this role has its own
        # test path (see `_test_rerank_selection`).
        _test_rerank_selection(role, selection)
        return

    if isinstance(selection, (PairSelection, LinkedProviderSelection)):
        try:
            draft = Config.get_referenced_provider(selection.provider_name)
        except ValueError as error:
            rich_print(f"[{_ERROR_COLOR}]{error}[/{_ERROR_COLOR}]")
            return
        if isinstance(selection, PairSelection):
            draft["model"] = selection.model_id
        _test_provider(selection.provider_name, draft)
        return

    if isinstance(selection, DirectSelection):
        flat = Config.get_all_config_flat()
        draft = {
            "provider_type": "openai_compatible",
            "base_url": selection.host,
            "api_key": str(flat.get("embedding_api_key") or ""),
            "model": selection.model_id,
        }
        _test_provider("direct endpoint", draft)
        return

    rich_print(
        f"[{_SEP_COLOR}]Nothing to test for "
        f"'{_selection_label(role, selection)}'.[/{_SEP_COLOR}]"
    )


def _run_task_action(role: TaskRole, selection: TaskSelection) -> None:
    """Save / Test connection / Cancel for one resolved selection.

    Test connection keeps this menu open, so the user can test and then Save or
    Cancel without picking the pair again. Save or Cancel closes the menu, and so
    does Back or Escape; Test never writes.
    """
    while True:
        action = _select_with_back(
            f"{_TASK_TITLES[role]} — what to do",
            choices=[
                questionary.Choice("Save", value="save"),
                questionary.Choice("Test connection", value="test"),
                questionary.Choice("Cancel", value="cancel"),
            ],
        )

        if action is None or action in ("cancel", "__back__"):
            rich_print(
                f"[{_SEP_COLOR}]Selection discarded, nothing saved.[/{_SEP_COLOR}]"
            )
            return

        if action == "test":
            # No write here: the next turn of the menu is still this selection.
            _test_task_selection(role, selection)
            continue

        try:
            _save_task_selection(role, selection)
        except ValueError as error:
            rich_print(f"[{_ERROR_COLOR}]{error}[/{_ERROR_COLOR}]")
            return

        rich_print(
            f"[{_SUCCESS_COLOR}]Saved {_selection_label(role, selection)}."
            f"[/{_SUCCESS_COLOR}]"
        )
        rich_print(f"[{_SEP_COLOR}]{_RESTART_NOTE}[/]")
        return


def _edit_task(role: TaskRole) -> None:
    """Task screen: current selection, the combined picker, then the action menu."""
    if role == "rerank":
        _edit_rerank_task()
        return

    while True:
        current = _current_task_selection(role)
        rich_print(f"[{_KEY_COLOR}]{_task_summary(role, current)}[/]")

        selection = _choose_provider_model(role, current)
        if selection is None:
            # Back or Escape: nothing is written.
            return

        _run_task_action(role, selection)


def _rerank_saved_selection_is_valid(selection: TaskSelection | None) -> bool:
    """Enable only validates what is already saved; it does not choose a model."""
    if not isinstance(selection, (LocalSelection, PairSelection)):
        return False
    if isinstance(selection, LocalSelection):
        return bool(selection.model_id)
    if not selection.model_id:
        return False
    providers = Config.get_providers()
    entry = providers.get(selection.provider_name)
    if not isinstance(entry, Mapping):
        return False
    return _provider_type(entry) in _TASK_PROVIDER_TYPES["rerank"]


def _edit_rerank_task() -> None:
    """Reranking: the Enabled switch is separate from source and model."""
    while True:
        current = _current_task_selection("rerank")
        rich_print(f"[{_KEY_COLOR}]{_task_summary('rerank', current)}[/]")

        action = _select_with_back(
            "Reranking — what to do",
            choices=[
                questionary.Choice("Enabled (yes/no)", value="enabled"),
                questionary.Choice("Source and model", value="source"),
            ],
        )
        if action == "__back__":
            return

        if action == "source":
            selection = _choose_provider_model("rerank", current)
            if selection is None:
                return
            _run_task_action("rerank", selection)
            continue

        enabled = _ask(
            questionary.confirm(
                "Enable reranking?",
                default=bool(Config.get_all_config_flat().get("corpus.rerank")),
                style=_WIZARD_STYLE,
            )
        )
        if enabled is None:
            rich_print(f"[{_SEP_COLOR}]Cancelled, nothing saved.[/{_SEP_COLOR}]")
            return
        if not enabled:
            # Disabling writes only the switch.
            Config.set_config_value("corpus.rerank", False)
            rich_print(f"[{_SUCCESS_COLOR}]Reranking disabled.[/{_SUCCESS_COLOR}]")
            rich_print(f"[{_SEP_COLOR}]{_RESTART_NOTE}[/]")
            continue

        saved = _current_task_selection("rerank")
        if saved is None or not _rerank_saved_selection_is_valid(saved):
            # An incomplete provider source is not treated as a local model.
            rich_print(f"[{_ERROR_COLOR}]{_RERANK_ENABLE_ERROR}[/]")
            continue

        # Enabling validates the saved selection and writes only the switch.
        Config.set_config_value("corpus.rerank", True)
        rich_print(
            f"[{_SUCCESS_COLOR}]Reranking enabled for "
            f"{_selection_label('rerank', saved)}.[/{_SUCCESS_COLOR}]"
        )
        rich_print(f"[{_SEP_COLOR}]{_RESTART_NOTE}[/]")
        return


# ---------------------------------------------------------------------------
# Submenus
# ---------------------------------------------------------------------------
def _providers_menu() -> None:
    """Chat providers submenu — groups the existing provider CRUD flows."""
    while True:
        selected = _select_with_back(
            "Chat providers",
            choices=[
                questionary.Choice("Add provider", value="Add provider"),
                questionary.Choice("Edit provider", value="Edit provider"),
                questionary.Choice("Remove provider", value="Remove provider"),
                questionary.Choice(
                    "Set default provider", value="Set default provider"
                ),
                questionary.Choice(
                    "Set provider failover order", value="Set provider order"
                ),
                questionary.Choice("Test connection", value="Test connection"),
            ],
        )
        if selected == "__back__":
            break

        if selected == "Add provider":
            _add_provider_flow()
        elif selected == "Edit provider":
            _edit_provider_flow()
        elif selected == "Remove provider":
            _remove_provider_flow()
        elif selected == "Set default provider":
            _set_default_provider_flow()
        elif selected == "Set provider order":
            _set_provider_order_flow()
        elif selected == "Test connection":
            _test_provider_standalone()


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
    _check_default_provider_warning()

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
        selected = _ask(
            questionary.select(
                "What would you like to do?",
                choices=[
                    questionary.Choice("Models by task ▸", value="Models by task"),
                    questionary.Choice("Providers ▸", value="Providers"),
                    questionary.Choice(
                        "Application settings ▸", value="Application settings"
                    ),
                    questionary.Choice("Advanced ▸", value="Advanced"),
                    questionary.Choice("View settings", value="View settings"),
                    questionary.Choice("Exit", value="Exit"),
                ],
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
            elif selected == "Providers":
                _providers_menu()
            elif selected == "Application settings":
                _edit_section(
                    "Application settings", APPLICATION_SECTIONS["Application settings"]
                )
            elif selected == "Advanced":
                _advanced_menu()
            elif selected == "View settings":
                _show_config_flow()
        except Exception:
            traceback.print_exc()
            rich_print(
                f"[{_ERROR_COLOR}]An error occurred. Please try again.[/{_ERROR_COLOR}]"
            )
