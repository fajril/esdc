"""One canonical connection/model picker and draft flow for every task.

The picker consumes a canonical :class:`TaskAssignment`, discovers models from
the eligible connection registry, and returns a draft. Nothing is written until
the shared action menu's explicit Save; Test uses the same resolver as runtime
and never persists credentials or enables a task.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any

import questionary
from rich import print as rich_print

from esdc.config_wizard import (
    _ERROR_COLOR,
    _KEY_COLOR,
    _RESTART_NOTE,
    _SEP_COLOR,
    _SUCCESS_COLOR,
    _VALUE_COLOR,
    _WARNING_COLOR,
    _WIZARD_STYLE,
    _ask,
    _format_connection_label,
    _select_with_back,
    _short_test_message,
    collect_connection_draft,
)
from esdc.configs import Config
from esdc.embedders import OllamaEmbedder, OpenAIEmbedder
from esdc.model_catalog import ModelEntry, discover_models
from esdc.model_runtime import ResolvedTask, resolve_task
from esdc.model_settings import (
    ROLE_CONNECTION_TYPES,
    ROLE_LABELS,
    ModelTarget,
    TaskAssignment,
    TaskRole,
    with_chat_primary,
)

_RESTRICTED_OPTION_ROLES = frozenset({"chat", "metadata", "cleanup"})

_TASK_TITLES: dict[TaskRole, str] = dict(ROLE_LABELS)

_TASK_CATALOG_ROLE = {
    "chat": "chat",
    "embedding": "embedding",
    "rerank": "rerank",
    "ocr": "chat",
    "metadata": "chat",
    "cleanup": "chat",
}

_EMBEDDING_TEST_TEXT = "esdc embedding connection test"
_RERANK_TEST_QUERY = "esdc rerank connection test"
_RERANK_TEST_DOCUMENTS = ("first test document", "second test document")
_OCR_TEST_PROMPT = "Return the text in this image exactly."
_OCR_TEST_TEXT = "ESDC OCR TEST 12345"
_OCR_TEST_MARKER = "12345"
_TEXT_TEST_PROMPT = "Reply with the single word OK."

_BACK = "__back__"
_REFRESH = "__refresh__"
_MANUAL = "__manual__"
_ADD_CONNECTION = "__add_connection__"
_MODE_CHAT = "__mode_chat__"
_MODE_OCR = "__mode_ocr__"
_MODE_OFF = "__mode_off__"
_MANAGE_FALLBACKS = "__manage_fallbacks__"


@dataclass(frozen=True)
class TaskDraft:
    """A task assignment plus any connections collected inside this draft."""

    role: TaskRole
    assignment: TaskAssignment
    new_connections: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)


class _CatalogCache:
    """One picker-open discovery cache, invalidated by a changed URL."""

    def __init__(self) -> None:
        self._entries: dict[tuple[str, str, str], Any] = {}

    def get(self, name: str, connection: Mapping[str, Any]):
        key = (
            name,
            _provider_type(connection),
            str(connection.get("base_url") or "").strip(),
        )
        if key not in self._entries:
            self._entries[key] = discover_models(dict(connection))
        return self._entries[key]

    def clear(self) -> None:
        self._entries.clear()


def _provider_type(entry: Mapping[str, Any]) -> str:
    return str(entry.get("type") or entry.get("provider_type") or "").strip().lower()


def _connection_entries(draft: TaskDraft) -> dict[str, dict[str, Any]]:
    """Saved connections followed by connections collected in this draft."""
    entries = {
        name: dict(entry) for name, entry in Config.get_connections().items()
    }
    for name, entry in draft.new_connections.items():
        entries[name] = dict(entry)
    return entries


def _eligible_names(
    role: TaskRole, entries: Mapping[str, Mapping[str, Any]]
) -> list[str]:
    allowed = ROLE_CONNECTION_TYPES[role]
    return [
        name
        for name, entry in entries.items()
        if _provider_type(entry) in allowed
    ]


def _entry_matches_role(entry: ModelEntry, role: TaskRole) -> bool:
    catalog_role = _TASK_CATALOG_ROLE[role]
    return not entry.roles or catalog_role in entry.roles


def _target_label(role: TaskRole, target: ModelTarget, *, unverified: bool) -> str:
    label = f"{target.connection} / {target.model}"
    if unverified:
        label += " (saved, unverified)"
    return label


def _assignment_label(role: TaskRole, assignment: TaskAssignment) -> str:
    if assignment.mode == "model" and assignment.target is not None:
        label = _target_label(role, assignment.target, unverified=False)
        if role == "chat" and assignment.fallbacks:
            label += f" (+{len(assignment.fallbacks)} fallback(s))"
        return label
    if assignment.mode == "chat":
        return "Use chat model"
    if assignment.mode == "ocr":
        return "Use OCR result"
    if assignment.mode == "off":
        return "Off"
    return "Not set"


def _active_override_names(role: TaskRole) -> tuple[str, ...]:
    if role != "embedding":
        return ()
    try:
        return Config.resolve_model_task("embedding").overrides
    except ValueError:
        return ()


def task_summary(role: TaskRole) -> str:
    """One plain line: current assignment plus the rerank switch/overrides."""
    assignment = Config.get_model_assignment(role)
    parts = [f"{_TASK_TITLES[role]}: {_assignment_label(role, assignment)}"]
    if role == "rerank":
        parts.append("enabled" if assignment.enabled else "disabled")
    for name in _active_override_names(role):
        parts.append(f"environment override: {name}")
    return "  " + " — ".join(parts)


def _resolve_draft(draft: TaskDraft) -> ResolvedTask:
    """Resolve the draft through the same resolver runtime uses."""
    entries = _connection_entries(draft)

    def load_connection(name: str) -> dict[str, Any]:
        if name in entries:
            return dict(entries[name])
        return Config.get_connection(name)

    def load_assignment(role: TaskRole) -> TaskAssignment:
        if role == draft.role:
            return draft.assignment
        return Config.get_model_assignment(role)

    return resolve_task(
        draft.role,
        load_assignment=load_assignment,
        load_connection=load_connection,
        apply_overrides=True,
    )


def _expired_oauth(provider: Mapping[str, Any]) -> bool:
    oauth = provider.get("oauth")
    if not isinstance(oauth, Mapping) or not oauth.get("access_token"):
        return False
    from esdc.auth.oauth import is_token_expired

    return is_token_expired(dict(oauth))


def _check_readonly_credentials(task: ResolvedTask) -> bool:
    """False when Test must stop for reauthentication instead of refreshing."""
    targets = [task.primary, *task.fallbacks]
    for target in targets:
        if target is not None and _expired_oauth(target.provider):
            rich_print(
                f"[{_WARNING_COLOR}]Credential expired for "
                f"'[{_VALUE_COLOR}]{target.connection}[/{_VALUE_COLOR}]'. "
                "Reauthenticate before testing; nothing was refreshed, saved or "
                f"retried as another account.[/{_WARNING_COLOR}]"
            )
            return False
    return True


def _test_embedding(task: ResolvedTask) -> None:
    target = task.primary
    if target is None:
        rich_print(f"[{_SEP_COLOR}]Nothing to test.[/{_SEP_COLOR}]")
        return
    provider_type = _provider_type(target.provider)
    host = str(target.provider.get("base_url") or "").strip() or None
    try:
        if provider_type == "ollama":
            embedder = OllamaEmbedder(host=host, model=target.model)
        elif provider_type == "openai_compatible":
            embedder = OpenAIEmbedder(
                host=host,
                model=target.model,
                api_key=str(target.provider.get("api_key") or ""),
            )
        else:
            rich_print(
                f"[{_ERROR_COLOR}]Embeddings are not supported for connection type "
                f"{provider_type or 'unknown'}.[/{_ERROR_COLOR}]"
            )
            return
        vector = embedder.generate_embedding(_EMBEDDING_TEST_TEXT)
    except Exception as error:
        rich_print(
            f"[{_ERROR_COLOR}]Failed: {_short_test_message(error)}[/{_ERROR_COLOR}]"
        )
        return
    if not vector or not all(isinstance(value, (int, float)) for value in vector):
        rich_print(
            f"[{_ERROR_COLOR}]Failed: the embedding endpoint returned no "
            f"vector.[/{_ERROR_COLOR}]"
        )
        return
    rich_print(
        f"[{_SUCCESS_COLOR}]Success: embedding endpoint responded with a "
        f"{len(vector)}-dimensional vector.[/{_SUCCESS_COLOR}]"
    )


def _test_rerank(task: ResolvedTask) -> None:
    target = task.primary
    if target is None:
        rich_print(f"[{_SEP_COLOR}]Nothing to test.[/{_SEP_COLOR}]")
        return
    provider_type = _provider_type(target.provider)
    if provider_type != "openai_compatible":
        rich_print(
            f"[{_ERROR_COLOR}]Reranking is not supported for connection type "
            f"{provider_type or 'unknown'}.[/{_ERROR_COLOR}]"
        )
        return
    try:
        from esdc.corpus.provider_reranker import ProviderReranker

        client = ProviderReranker(dict(target.provider), target.model)
        scores = client.rerank(_RERANK_TEST_QUERY, list(_RERANK_TEST_DOCUMENTS))
    except Exception as error:
        rich_print(
            f"[{_ERROR_COLOR}]Failed: {_short_test_message(error)}[/{_ERROR_COLOR}]"
        )
        return
    if len(scores) != len(_RERANK_TEST_DOCUMENTS) or not all(
        isinstance(score, (int, float))
        and not isinstance(score, bool)
        and math.isfinite(float(score))
        for score in scores
    ):
        rich_print(
            f"[{_ERROR_COLOR}]Failed: the rerank endpoint returned an incomplete "
            f"result.[/{_ERROR_COLOR}]"
        )
        return
    rich_print(
        f"[{_SUCCESS_COLOR}]Success: rerank endpoint responded with "
        f"{len(scores)} scores.[/{_SUCCESS_COLOR}]"
    )


def _ocr_test_image() -> bytes:
    import fitz

    doc = fitz.open()
    try:
        page = doc.new_page(width=220, height=120)
        page.insert_text((20, 60), _OCR_TEST_TEXT, fontsize=18)
        return page.get_pixmap(dpi=150).tobytes("png")
    finally:
        doc.close()


def _test_ocr(task: ResolvedTask) -> None:
    from esdc.corpus.ocr import create_ocr_client

    try:
        client = create_ocr_client(Config.get_corpus_config(), resolved=task)
        if not client.health_check():
            rich_print(
                f"[{_WARNING_COLOR}]OCR endpoint is not available.[/{_WARNING_COLOR}]"
            )
            return
        answer = client.query_image(_ocr_test_image(), _OCR_TEST_PROMPT)
    except Exception as error:
        rich_print(
            f"[{_ERROR_COLOR}]Failed: {_short_test_message(error)}[/{_ERROR_COLOR}]"
        )
        return
    text = str(answer or "").strip()
    if _OCR_TEST_MARKER in text or "esdc" in text.lower():
        rich_print(f"[{_SUCCESS_COLOR}]OCR image test passed.[/{_SUCCESS_COLOR}]")
    else:
        rich_print(
            f"[{_WARNING_COLOR}]Failed: the reply did not read the test image."
            f"[/{_WARNING_COLOR}]"
        )


def _test_text(task: ResolvedTask) -> None:
    from esdc.providers import create_llm_from_resolved_task

    try:
        llm = create_llm_from_resolved_task(task)
        llm.invoke(_TEXT_TEST_PROMPT)
    except Exception as error:
        rich_print(
            f"[{_ERROR_COLOR}]Failed: {_short_test_message(error)}[/{_ERROR_COLOR}]"
        )
        return
    rich_print(f"[{_SUCCESS_COLOR}]Text inference responded.[/{_SUCCESS_COLOR}]")


def _test_draft(draft: TaskDraft) -> None:
    """Test the proposed assignment through the shared resolver; never writes."""
    try:
        task = _resolve_draft(draft)
    except ValueError as error:
        rich_print(f"[{_ERROR_COLOR}]{error}[/{_ERROR_COLOR}]")
        return

    if not _check_readonly_credentials(task):
        return
    if task.mode != "model" or task.primary is None:
        if task.mode == "ocr":
            rich_print(
                f"[{_SEP_COLOR}]Metadata follows the OCR result; no model call is "
                f"made.[/{_SEP_COLOR}]"
            )
        elif task.mode == "off":
            rich_print(
                f"[{_SEP_COLOR}]Text cleanup is off; nothing to test.[/{_SEP_COLOR}]"
            )
        elif task.mode == "chat":
            rich_print(
                f"[{_SEP_COLOR}]Chat is not configured; choose a Chat model before "
                f"testing this relationship.[/{_SEP_COLOR}]"
            )
        else:
            rich_print(
                f"[{_SEP_COLOR}]Nothing to test: this task is not configured."
                f"[/{_SEP_COLOR}]"
            )
        return

    if draft.role == "embedding":
        _test_embedding(task)
    elif draft.role == "rerank":
        _test_rerank(task)
    elif draft.role == "ocr":
        _test_ocr(task)
    else:
        _test_text(task)


def _target_options_flow(
    role: TaskRole, target: ModelTarget
) -> ModelTarget | None:
    """Edit reasoning effort on one text target, or return None to keep it."""
    if role not in _RESTRICTED_OPTION_ROLES:
        rich_print(f"[{_SEP_COLOR}]This task has no model options.[/{_SEP_COLOR}]")
        return None
    current = str(target.options.get("reasoning_effort") or "")
    choice = _select_with_back(
        "Reasoning effort:",
        choices=[
            questionary.Choice("(none)", value=""),
            questionary.Choice("low", value="low"),
            questionary.Choice("medium", value="medium"),
            questionary.Choice("high", value="high"),
        ],
        default=current or "",
    )
    if choice is None or choice == _BACK:
        return None
    options = dict(target.options)
    if choice:
        options["reasoning_effort"] = choice
    else:
        options.pop("reasoning_effort", None)
    return replace(target, options=options)


def _manual_target(
    draft: TaskDraft,
    entries: Mapping[str, Mapping[str, Any]],
    cache: _CatalogCache,
) -> ModelTarget | None:
    names = _eligible_names(draft.role, entries)
    if not names:
        rich_print(
            f"[{_WARNING_COLOR}]No eligible connection. Add a connection first."
            f"[/{_WARNING_COLOR}]"
        )
        return None
    connection = _ask(
        questionary.select(
            "Connection for the model ID:",
            choices=[
                questionary.Choice(
                    _format_connection_label(name, entries[name]), value=name
                )
                for name in names
            ],
            use_search_filter=True,
            use_jk_keys=False,
            style=_WIZARD_STYLE,
        )
    )
    if connection is None:
        return None
    current = draft.assignment.target
    default = current.model if current and current.connection == connection else ""
    while True:
        value = _ask(
            questionary.text(
                "Model ID:", default=default, style=_WIZARD_STYLE
            )
        )
        if value is None:
            return None
        model = str(value).strip()
        if not model:
            rich_print(
                f"[{_ERROR_COLOR}]A model ID cannot be empty.[/{_ERROR_COLOR}]"
            )
            continue
        if (
            draft.role == "ocr"
            and _known_vision(cache, entries, connection, model) is False
        ):
            rich_print(
                f"[{_ERROR_COLOR}]OCR model {model!r} is reported as non-vision by "
                f"this connection.[/{_ERROR_COLOR}]"
            )
            return None
        options = (
            dict(current.options)
            if current and current.connection == connection and current.model == model
            else {}
        )
        return ModelTarget(connection=connection, model=model, options=options)


def _known_vision(
    cache: _CatalogCache,
    entries: Mapping[str, Mapping[str, Any]],
    connection: str,
    model: str,
) -> bool | None:
    catalog = cache.get(connection, entries[connection])
    for entry in catalog.entries:
        if entry.model_id == model:
            return entry.supports_vision
    return None


def _target_rows(
    draft: TaskDraft,
    cache: _CatalogCache,
    entries: dict[str, dict[str, Any]],
) -> list[questionary.Choice]:
    rows: list[questionary.Choice] = []
    seen: set[tuple[str, str]] = set()
    current = draft.assignment.target

    for name in _eligible_names(draft.role, entries):
        catalog = cache.get(name, entries[name])
        if catalog.warning:
            rich_print(
                f"[{_WARNING_COLOR}]{name}: {catalog.warning}[/{_WARNING_COLOR}]"
            )
        candidates = [
            entry for entry in catalog.entries if _entry_matches_role(entry, draft.role)
        ]
        if draft.role == "ocr":
            candidates = [
                entry
                for entry in candidates
                if entry.supports_vision is not False
            ]
        for entry in candidates:
            target = ModelTarget(name, entry.model_id, {})
            if current and (current.connection, current.model) == (
                target.connection,
                target.model,
            ):
                target = replace(target, options=dict(current.options))
            rows.append(
                questionary.Choice(
                    _target_label(draft.role, target, unverified=False), value=target
                )
            )
            seen.add((target.connection, target.model))

    if current is not None and (current.connection, current.model) not in seen:
        known_vision = (
            _known_vision(cache, entries, current.connection, current.model)
            if draft.role == "ocr"
            else None
        )
        if known_vision is not False:
            rows.insert(
                0,
                questionary.Choice(
                    _target_label(draft.role, current, unverified=True),
                    value=current,
                ),
            )

    return rows


def _pick_target(
    draft: TaskDraft, cache: _CatalogCache
) -> tuple[TaskDraft, ModelTarget] | None:
    """Pick one connection/model target, allowing nested connection drafts."""
    while True:
        entries = _connection_entries(draft)
        rows = _target_rows(draft, cache, entries)
        rows.extend(
            [
                questionary.Choice("Enter model ID…", value=_MANUAL),
                questionary.Choice("Add connection…", value=_ADD_CONNECTION),
                questionary.Choice("Refresh models", value=_REFRESH),
                questionary.Choice("← Back", value=_BACK),
            ]
        )
        selected = _ask(
            questionary.select(
                f"{_TASK_TITLES[draft.role]} — connection / model",
                choices=rows,
                use_search_filter=True,
                use_jk_keys=False,
                style=_WIZARD_STYLE,
            )
        )
        if selected is None or selected == _BACK:
            return None
        if selected == _REFRESH:
            cache.clear()
            continue
        if selected == _ADD_CONNECTION:
            added = collect_connection_draft()
            if added is None:
                continue
            name, entry = added
            if name in entries:
                rich_print(
                    f"[{_WARNING_COLOR}]Connection '{name}' is already in this "
                    f"draft.[/{_WARNING_COLOR}]"
                )
                continue
            connections = dict(draft.new_connections)
            connections[name] = entry
            draft = replace(draft, new_connections=connections)
            continue
        if selected == _MANUAL:
            target = _manual_target(draft, entries, cache)
            if target is None:
                continue
            return draft, target
        if isinstance(selected, ModelTarget):
            return draft, selected


def _manage_fallbacks(draft: TaskDraft, cache: _CatalogCache) -> TaskDraft | None:
    """Add, remove or edit Chat fallback targets; never touches the primary."""
    while True:
        assignment = draft.assignment
        if assignment.mode != "model" or assignment.target is None:
            rich_print(
                f"[{_WARNING_COLOR}]Choose a Chat primary model before adding "
                f"fallbacks.[/{_WARNING_COLOR}]"
            )
            return None
        choices: list[questionary.Choice] = []
        for index, fallback in enumerate(assignment.fallbacks):
            label = _target_label("chat", fallback, unverified=False)
            choices.append(
                questionary.Choice(
                    f"Fallback {index + 1} options: {label}",
                    value=("options", index),
                )
            )
            choices.append(
                questionary.Choice(
                    f"Remove fallback {index + 1}: {label}",
                    value=("remove", index),
                )
            )
        choices.append(questionary.Choice("Add fallback…", value=("add", None)))
        choices.append(questionary.Choice("← Back", value=_BACK))
        selected = _select_with_back("Chat fallbacks", choices=choices)
        if selected == _BACK or selected is None:
            return draft
        action, index = selected
        if action == "add":
            picked = _pick_target(draft, cache)
            if picked is None:
                continue
            draft, target = picked
            primary = draft.assignment.target
            existing = {
                (item.connection, item.model)
                for item in (
                    *((primary,) if primary is not None else ()),
                    *draft.assignment.fallbacks,
                )
            }
            if (target.connection, target.model) in existing:
                rich_print(
                    f"[{_WARNING_COLOR}]That connection/model is already the primary "
                    f"or a fallback.[/{_WARNING_COLOR}]"
                )
                continue
            draft = replace(
                draft,
                assignment=replace(
                    draft.assignment,
                    fallbacks=(*draft.assignment.fallbacks, target),
                ),
            )
            continue
        fallbacks = list(draft.assignment.fallbacks)
        if action == "remove" and index is not None:
            del fallbacks[index]
            draft = replace(
                draft, assignment=replace(draft.assignment, fallbacks=tuple(fallbacks))
            )
            continue
        if action == "options" and index is not None:
            edited = _target_options_flow("chat", fallbacks[index])
            if edited is not None:
                fallbacks[index] = edited
                draft = replace(
                    draft,
                    assignment=replace(
                        draft.assignment, fallbacks=tuple(fallbacks)
                    ),
                )


def _assignment_for_target(
    draft: TaskDraft, target: ModelTarget
) -> TaskAssignment:
    """Chat promotion owns fallback cleanup; other roles just set the target."""
    if draft.role == "chat":
        return with_chat_primary(draft.assignment, target)
    return replace(
        draft.assignment,
        mode="model",
        target=target,
        enabled=draft.assignment.enabled,
    )


def _pick_assignment(
    draft: TaskDraft, cache: _CatalogCache
) -> TaskDraft | None:
    """The single combined picker: models, modes, manual, add, refresh, back."""
    while True:
        entries = _connection_entries(draft)
        rows = _target_rows(draft, cache, entries)

        if draft.role == "metadata":
            rows.extend(
                [
                    questionary.Choice("Use chat model", value=_MODE_CHAT),
                    questionary.Choice("Use OCR result", value=_MODE_OCR),
                ]
            )
        elif draft.role == "cleanup":
            rows.extend(
                [
                    questionary.Choice("Use chat model", value=_MODE_CHAT),
                    questionary.Choice("Off", value=_MODE_OFF),
                ]
            )
        elif draft.role == "chat":
            rows.append(
                questionary.Choice("Manage fallbacks…", value=_MANAGE_FALLBACKS)
            )

        rows.extend(
            [
                questionary.Choice("Enter model ID…", value=_MANUAL),
                questionary.Choice("Add connection…", value=_ADD_CONNECTION),
                questionary.Choice("Refresh models", value=_REFRESH),
                questionary.Choice("← Back", value=_BACK),
            ]
        )

        selected = _ask(
            questionary.select(
                f"{_TASK_TITLES[draft.role]} — connection / model",
                choices=rows,
                use_search_filter=True,
                use_jk_keys=False,
                style=_WIZARD_STYLE,
            )
        )
        if selected is None or selected == _BACK:
            return None
        if selected == _REFRESH:
            cache.clear()
            continue
        if selected == _ADD_CONNECTION:
            added = collect_connection_draft()
            if added is None:
                continue
            name, entry = added
            if name in entries:
                rich_print(
                    f"[{_WARNING_COLOR}]Connection '{name}' is already in this "
                    f"draft.[/{_WARNING_COLOR}]"
                )
                continue
            connections = dict(draft.new_connections)
            connections[name] = entry
            draft = replace(draft, new_connections=connections)
            continue
        if selected == _MANUAL:
            target = _manual_target(draft, entries, cache)
            if target is None:
                continue
            return replace(
                draft,
                assignment=_assignment_for_target(draft, target),
            )
        if selected == _MODE_CHAT:
            return replace(
                draft,
                assignment=TaskAssignment(mode="chat", target=None, enabled=True),
            )
        if selected == _MODE_OCR:
            return replace(
                draft,
                assignment=TaskAssignment(mode="ocr", target=None, enabled=True),
            )
        if selected == _MODE_OFF:
            return replace(
                draft,
                assignment=TaskAssignment(mode="off", target=None, enabled=True),
            )
        if selected == _MANAGE_FALLBACKS:
            updated = _manage_fallbacks(draft, cache)
            if updated is not None:
                return updated
            continue
        if isinstance(selected, ModelTarget):
            return replace(
                draft,
                assignment=_assignment_for_target(draft, selected),
            )


def _save_draft(draft: TaskDraft) -> bool:
    if draft.role == "embedding":
        saved = Config.get_model_assignment("embedding")
        proposed = draft.assignment.target
        changed = (
            saved.mode != "model"
            or saved.target is None
            or proposed is None
            or (saved.target.connection, saved.target.model)
            != (proposed.connection, proposed.model)
        )
        if changed:
            rich_print(
                f"[{_WARNING_COLOR}]Changing the embedding vector space may require "
                f"rebuilding embeddings (`esdc reload --embeddings-only`). "
                f"No re-embed was started.[/{_WARNING_COLOR}]"
            )
    try:
        Config.save_model_assignment(
            draft.role,
            draft.assignment,
            new_connections=draft.new_connections,
        )
    except ValueError as error:
        rich_print(f"[{_ERROR_COLOR}]{error}[/{_ERROR_COLOR}]")
        return False
    rich_print(
        f"[{_SUCCESS_COLOR}]Saved {_TASK_TITLES[draft.role]}: "
        f"{_assignment_label(draft.role, draft.assignment)}.[/{_SUCCESS_COLOR}]"
    )
    rich_print(f"[{_SEP_COLOR}]{_RESTART_NOTE}[/]")
    return True


def _run_actions(draft: TaskDraft) -> str:
    """Save / Test / model options / Cancel for one draft.

    Returns ``"saved"``, ``"cancel"`` or ``"failed"``. A failed save keeps the
    draft and returns to the action menu without writing anything.
    """
    while True:
        choices = [questionary.Choice("Save", value="save")]
        if (
            draft.assignment.mode == "model"
            and draft.assignment.target is not None
            and draft.role in _RESTRICTED_OPTION_ROLES
        ):
            choices.append(questionary.Choice("Model options…", value="options"))
        choices.extend(
            [
                questionary.Choice("Test", value="test"),
                questionary.Choice("Cancel", value="cancel"),
            ]
        )
        rich_print(
            f"[{_KEY_COLOR}]Proposed {_TASK_TITLES[draft.role]}: "
            f"{_assignment_label(draft.role, draft.assignment)}[/]"
        )
        action = _select_with_back(
            f"{_TASK_TITLES[draft.role]} — what to do", choices=choices
        )
        if action is None or action in ("cancel", _BACK):
            rich_print(
                f"[{_SEP_COLOR}]Draft discarded, nothing saved.[/{_SEP_COLOR}]"
            )
            return "cancel"
        if action == "test":
            try:
                _test_draft(draft)
            except Exception as error:  # a failed test never erases the draft
                rich_print(
                    f"[{_ERROR_COLOR}]Failed: {_short_test_message(error)}"
                    f"[/{_ERROR_COLOR}]"
                )
            continue
        if action == "options":
            target = draft.assignment.target
            assert target is not None
            edited = _target_options_flow(draft.role, target)
            if edited is not None:
                draft = replace(
                    draft, assignment=replace(draft.assignment, target=edited)
                )
            continue
        return "saved" if _save_draft(draft) else "failed"


def _edit_task_loop(role: TaskRole) -> None:
    while True:
        rich_print(f"[{_KEY_COLOR}]{task_summary(role)}[/]")
        draft = TaskDraft(
            role=role, assignment=Config.get_model_assignment(role)
        )
        picked = _pick_assignment(draft, _CatalogCache())
        if picked is None:
            return
        draft = picked
        while True:
            status = _run_actions(draft)
            if status in {"saved", "cancel"}:
                return


def _edit_rerank_task() -> None:
    while True:
        assignment = Config.get_model_assignment("rerank")
        rich_print(f"[{_KEY_COLOR}]{task_summary('rerank')}[/]")
        action = _select_with_back(
            "Reranking — what to do",
            choices=[
                questionary.Choice("Source and model", value="source"),
                questionary.Choice("Enabled (yes/no)", value="enabled"),
            ],
        )
        if action is None or action == _BACK:
            return
        if action == "source":
            _edit_task_loop("rerank")
            continue
        enabled = _ask(
            questionary.confirm(
                "Enable reranking?",
                default=assignment.enabled,
                style=_WIZARD_STYLE,
            )
        )
        if enabled is None:
            rich_print(f"[{_SEP_COLOR}]Cancelled, nothing saved.[/{_SEP_COLOR}]")
            continue
        if enabled and (assignment.mode != "model" or assignment.target is None):
            rich_print(
                f"[{_ERROR_COLOR}]Cannot enable reranking: choose a connection and "
                f"model first. The switch was not changed.[/{_ERROR_COLOR}]"
            )
            continue
        Config.save_model_assignment(
            "rerank", replace(assignment, enabled=bool(enabled))
        )
        rich_print(
            f"[{_SUCCESS_COLOR}]Reranking "
            f"{'enabled' if enabled else 'disabled'}.[/{_SUCCESS_COLOR}]"
        )
        rich_print(f"[{_SEP_COLOR}]{_RESTART_NOTE}[/]")


def edit_task(role: TaskRole) -> None:
    """Open one canonical task draft, action menu and explicit save."""
    if role == "rerank":
        _edit_rerank_task()
        return
    _edit_task_loop(role)
