"""Observable end-to-end flows for the one canonical task model picker.

Answers are supplied at the Questionary boundary: every test drives the real
wizard functions, but no terminal, network, database or live config is used.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any
from unittest.mock import MagicMock

import pytest
import questionary

import esdc.model_wizard as model_wizard
from esdc.configs import Config
from esdc.model_catalog import ModelCatalog, ModelEntry
from esdc.model_settings import ModelTarget, TaskAssignment, TaskRole


class _PromptPlan:
    """Answer every Questionary prompt from one ordered answer list."""

    def __init__(self, answers: Sequence[Any]) -> None:
        self.answers = list(answers)
        self.select_calls: list[tuple[str, list[str]]] = []
        self.text_calls: list[str] = []

    def _next(self) -> Any:
        if not self.answers:
            raise AssertionError("no prompt answer left")
        return self.answers.pop(0)

    @staticmethod
    def _resolve(answer: Any, choices: Sequence[Any]) -> Any:
        if callable(answer):
            return answer(choices)
        for choice in choices:
            if str(getattr(choice, "title", "")) == answer:
                return choice.value
        for choice in choices:
            if getattr(choice, "value", object()) == answer:
                return answer
        raise AssertionError(
            f"answer {answer!r} not in {[str(c.title) for c in choices]}"
        )

    def select(self, message: str, choices: Sequence[Any], **_kwargs: Any):
        self.select_calls.append((message, [str(c.title) for c in choices]))
        value = self._resolve(self._next(), choices)
        question = MagicMock()
        question.unsafe_ask.return_value = value
        return question

    def text(self, message: str, default: str = "", **_kwargs: Any):
        self.text_calls.append(message)
        value = self._next()
        if value is _USE_DEFAULT:
            value = default
        question = MagicMock()
        question.unsafe_ask.return_value = value
        return question

    def password(self, message: str, **_kwargs: Any):
        return self.text(message)

    def confirm(self, message: str, default: bool = False, **_kwargs: Any):
        value = self._next()
        if value is _USE_DEFAULT:
            value = default
        question = MagicMock()
        question.unsafe_ask.return_value = value
        return question


_USE_DEFAULT = object()


@pytest.fixture(autouse=True)
def isolated_config(tmp_path, monkeypatch) -> Iterator[None]:
    monkeypatch.setenv("ESDC_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("ESDC_DB_DIR", str(tmp_path))
    monkeypatch.setenv("ESDC_DB_FILE", str(tmp_path / "esdc.duckdb"))
    for name in (
        "ESDC_EMBEDDING_BACKEND",
        "ESDC_EMBEDDING_HOST",
        "ESDC_EMBEDDING_MODEL",
        "ESDC_EMBEDDING_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    Config._config_cache = None
    yield
    Config._config_cache = None


@pytest.fixture
def prompts(monkeypatch):
    plans: list[_PromptPlan] = []

    def install(answers: Sequence[Any]) -> _PromptPlan:
        plan = _PromptPlan(answers)
        plans.append(plan)
        monkeypatch.setattr(questionary, "select", plan.select)
        monkeypatch.setattr(questionary, "text", plan.text)
        monkeypatch.setattr(questionary, "password", plan.password)
        monkeypatch.setattr(questionary, "confirm", plan.confirm)
        return plan

    yield install


def _connection(
    name: str = "gateway",
    provider_type: str = "openai_compatible",
    base_url: str | None = "http://gateway.example/v1",
    **extra: Any,
) -> None:
    values: dict[str, Any] = {"type": provider_type}
    if base_url is not None:
        values["base_url"] = base_url
    values.update(extra)
    Config.save_connection(name, values)


def _assign(
    role: TaskRole,
    *,
    mode: str = "model",
    connection: str | None = None,
    model: str | None = None,
    options: dict[str, Any] | None = None,
    fallbacks: tuple[ModelTarget, ...] = (),
    enabled: bool | None = None,
) -> None:
    target = (
        ModelTarget(connection, model, dict(options or {}))
        if connection is not None and model is not None
        else None
    )
    if role == "rerank":
        assignment = TaskAssignment(
            mode=mode,  # type: ignore[arg-type]
            target=target,
            enabled=bool(enabled),
            fallbacks=(),
        )
    else:
        assignment = TaskAssignment(
            mode=mode,  # type: ignore[arg-type]
            target=target,
            fallbacks=fallbacks,
        )
    Config.save_model_assignment(role, assignment)


def _discover(monkeypatch, *entries: ModelEntry) -> MagicMock:
    fake = MagicMock(return_value=ModelCatalog(entries=tuple(entries)))
    monkeypatch.setattr(model_wizard, "discover_models", fake)
    return fake


TASK_PRELUDE: dict[TaskRole, list[Any]] = {
    "chat": [],
    "embedding": [],
    "rerank": ["Source and model"],
    "ocr": [],
    "metadata": [],
    "cleanup": [],
}


@pytest.mark.parametrize(
    "role",
    ["chat", "embedding", "rerank", "ocr", "metadata", "cleanup"],
)
def test_discovered_selection_saves_the_role_assignment(
    monkeypatch, prompts, role
):
    _connection()
    _discover(monkeypatch, ModelEntry("model-a"))
    plan = prompts([*TASK_PRELUDE[role], "gateway / model-a", "Save", "← Back"])

    model_wizard.edit_task(role)

    assignment = Config.get_model_assignment(role)
    assert assignment.mode == "model"
    assert assignment.target == ModelTarget("gateway", "model-a", {})
    if role == "rerank":
        assert assignment.enabled is False
    labels = " ".join(
        label for _message, titles in plan.select_calls for label in titles
    )
    assert "gateway / model-a" in labels


@pytest.mark.parametrize("role", ["chat", "embedding", "ocr", "metadata"])
def test_existing_saved_model_stays_visible_when_discovery_cannot_confirm_it(
    monkeypatch, prompts, role
):
    _connection()
    _assign(role, connection="gateway", model="saved-model", options={"x": 1})
    _discover(monkeypatch, ModelEntry("other-model"))
    prompts(
        [
            *TASK_PRELUDE[role],
            "gateway / saved-model (saved, unverified)",
            "Save",
            "← Back",
        ]
    )

    model_wizard.edit_task(role)

    target = Config.get_model_assignment(role).target
    assert target is not None
    assert (target.connection, target.model) == ("gateway", "saved-model")
    assert target.options == {"x": 1}


def test_manual_model_id_works_for_an_ollama_embedding_connection(prompts):
    _connection(
        "desk", provider_type="ollama", base_url="http://desk.example:11434"
    )
    prompts(
        [
            "Enter model ID…",
            "desk (Ollama)",
            "qwen3-embedding:0.6b",
            "Save",
        ]
    )

    model_wizard.edit_task("embedding")

    target = Config.get_model_assignment("embedding").target
    assert target is not None
    assert (target.connection, target.model) == (
        "desk",
        "qwen3-embedding:0.6b",
    )


def test_add_connection_is_a_draft_and_saves_both_parts_in_one_write(
    monkeypatch, prompts
):
    _discover(monkeypatch, ModelEntry("new-model"))
    monkeypatch.setattr(
        model_wizard,
        "collect_connection_draft",
        lambda: ("desk", {"type": "ollama", "base_url": "http://new:11434"}),
    )
    prompts(["Add connection…", "desk / new-model", "Save"])

    model_wizard.edit_task("embedding")

    assert Config.get_connections()["desk"]["type"] == "ollama"
    target = Config.get_model_assignment("embedding").target
    assert target is not None
    assert (target.connection, target.model) == ("desk", "new-model")


def test_cancel_after_adding_a_draft_connection_writes_nothing(
    monkeypatch, prompts
):
    monkeypatch.setattr(model_wizard, "collect_connection_draft", lambda: None)
    prompts(["Add connection…", "← Back"])

    model_wizard.edit_task("chat")

    assert Config.get_connections() == {}
    assert Config.get_model_assignment("chat").mode == "unset"


def test_testing_a_draft_connection_does_not_add_it_to_the_registry(
    monkeypatch, prompts
):
    _discover(monkeypatch, ModelEntry("new-model"))
    monkeypatch.setattr(
        model_wizard,
        "collect_connection_draft",
        lambda: ("desk", {"type": "ollama", "base_url": "http://new:11434"}),
    )
    tested = MagicMock()
    monkeypatch.setattr(model_wizard, "_test_text", tested)
    prompts(["Add connection…", "desk / new-model", "Test", "Cancel"])

    model_wizard.edit_task("chat")

    tested.assert_called_once()
    assert Config.get_connections() == {}


def test_failed_save_keeps_the_draft_connection_unsaved(
    monkeypatch, prompts
):
    _discover(monkeypatch, ModelEntry("new-model"))
    monkeypatch.setattr(
        model_wizard,
        "collect_connection_draft",
        lambda: ("desk", {"type": "ollama", "base_url": "http://new:11434"}),
    )
    monkeypatch.setattr(
        Config,
        "save_model_assignment",
        MagicMock(side_effect=ValueError("synthetic save failure")),
    )
    prompts(["Add connection…", "desk / new-model", "Save", "Cancel"])

    model_wizard.edit_task("embedding")

    assert Config.get_connections() == {}


def test_refresh_models_replaces_a_failed_catalog(monkeypatch, prompts):
    _connection()
    catalogs = [
        ModelCatalog(warning="Model discovery failed (timeout)."),
        ModelCatalog(entries=(ModelEntry("fresh-model"),)),
    ]
    monkeypatch.setattr(
        model_wizard, "discover_models", MagicMock(side_effect=catalogs)
    )
    prompts(["Refresh models", "gateway / fresh-model", "Save"])

    model_wizard.edit_task("chat")

    target = Config.get_model_assignment("chat").target
    assert target is not None
    assert target.model == "fresh-model"


def test_back_at_the_picker_writes_nothing(prompts):
    prompts(["← Back"])

    model_wizard.edit_task("chat")

    assert Config.get_connections() == {}
    assert Config.get_model_assignment("chat").mode == "unset"


def test_cancel_at_the_action_menu_writes_nothing(monkeypatch, prompts):
    _connection()
    _discover(monkeypatch, ModelEntry("model-a"))
    prompts(["gateway / model-a", "Cancel"])

    model_wizard.edit_task("chat")

    assert Config.get_model_assignment("chat").mode == "unset"


def test_test_then_save_keeps_the_same_draft(monkeypatch, prompts):
    _connection()
    _discover(monkeypatch, ModelEntry("model-a"))
    tested = MagicMock()
    monkeypatch.setattr(model_wizard, "_test_text", tested)
    prompts(["gateway / model-a", "Test", "Save"])

    model_wizard.edit_task("chat")

    tested.assert_called_once()
    target = Config.get_model_assignment("chat").target
    assert target is not None
    assert (target.connection, target.model) == ("gateway", "model-a")


def test_test_failure_keeps_the_draft_and_test_never_saves(
    monkeypatch, prompts
):
    _connection()
    _discover(monkeypatch, ModelEntry("model-a"))
    monkeypatch.setattr(
        model_wizard,
        "_test_text",
        MagicMock(side_effect=RuntimeError("synthetic test failure")),
    )
    save = MagicMock()
    monkeypatch.setattr(Config, "save_model_assignment", save)
    prompts(["gateway / model-a", "Test", "Cancel"])

    model_wizard.edit_task("chat")

    save.assert_not_called()
    assert Config.get_model_assignment("chat").mode == "unset"


def test_chat_primary_change_preserves_distinct_fallbacks(monkeypatch, prompts):
    _connection()
    _connection("backup", base_url="http://backup.example/v1")
    _assign(
        "chat",
        connection="gateway",
        model="old-primary",
        fallbacks=(ModelTarget("backup", "fallback-model", {}),),
    )
    _discover(monkeypatch, ModelEntry("new-primary"))
    prompts(["gateway / new-primary", "Save"])

    model_wizard.edit_task("chat")

    assignment = Config.get_model_assignment("chat")
    assert assignment.target is not None
    assert assignment.target.model == "new-primary"
    assert assignment.fallbacks == (ModelTarget("backup", "fallback-model", {}),)


def test_chat_fallback_can_be_added_and_removed_with_the_same_picker(
    monkeypatch, prompts
):
    _connection()
    _connection("backup", base_url="http://backup.example/v1")
    _assign("chat", connection="gateway", model="primary")
    _discover(monkeypatch, ModelEntry("new-primary"), ModelEntry("fallback-a"))
    prompts(
        [
            "Manage fallbacks…",
            "Add fallback…",
            "gateway / fallback-a",
            "← Back",
            "Save",
        ]
    )

    model_wizard.edit_task("chat")

    assignment = Config.get_model_assignment("chat")
    assert assignment.fallbacks == (ModelTarget("gateway", "fallback-a", {}),)


@pytest.mark.parametrize(
    ("role", "row", "mode"),
    [
        ("metadata", "Use chat model", "chat"),
        ("metadata", "Use OCR result", "ocr"),
        ("cleanup", "Use chat model", "chat"),
        ("cleanup", "Off", "off"),
    ],
)
def test_metadata_and_cleanup_relationship_modes_save(prompts, role, row, mode):
    _connection()
    prompts([row, "Save"])

    model_wizard.edit_task(role)

    assignment = Config.get_model_assignment(role)
    assert assignment.mode == mode
    assert assignment.target is None


def test_rerank_selection_does_not_change_the_enabled_switch(monkeypatch, prompts):
    _connection()
    _assign("rerank", connection="gateway", model="old", enabled=False)
    _discover(monkeypatch, ModelEntry("rank-new"))
    prompts(["Source and model", "gateway / rank-new", "Save", "← Back"])

    model_wizard.edit_task("rerank")

    assignment = Config.get_model_assignment("rerank")
    assert assignment.target is not None
    assert assignment.target.model == "rank-new"
    assert assignment.enabled is False


def test_rerank_enable_requires_a_model_and_only_changes_the_switch(
    monkeypatch, prompts
):
    _connection()
    _assign("rerank", connection="gateway", model="rank-model", enabled=False)
    prompts(["Enabled (yes/no)", True, "← Back"])

    model_wizard.edit_task("rerank")

    assignment = Config.get_model_assignment("rerank")
    assert assignment.enabled is True
    assert assignment.target is not None
    assert assignment.target.model == "rank-model"

    prompts(["Enabled (yes/no)", False, "← Back"])
    model_wizard.edit_task("rerank")
    assert Config.get_model_assignment("rerank").enabled is False


def test_enable_rerank_without_a_model_is_refused(prompts):
    prompts(["Enabled (yes/no)", True, "← Back"])

    model_wizard.edit_task("rerank")

    assert Config.get_model_assignment("rerank").enabled is False


def test_positively_non_vision_ocr_model_is_not_offered(monkeypatch, prompts):
    _connection()
    _discover(
        monkeypatch,
        ModelEntry("vision-model", supports_vision=True),
        ModelEntry("text-only", supports_vision=False),
    )
    plan = prompts(["← Back"])

    model_wizard.edit_task("ocr")

    offered = " ".join(plan.select_calls[0][1])
    assert "vision-model" in offered
    assert "text-only" not in offered


def test_saved_non_vision_ocr_model_is_not_reoffered(
    monkeypatch, prompts
):
    _connection()
    _assign("ocr", connection="gateway", model="text-only")
    _discover(
        monkeypatch,
        ModelEntry("vision-model", supports_vision=True),
        ModelEntry("text-only", supports_vision=False),
    )
    plan = prompts(["← Back"])

    model_wizard.edit_task("ocr")

    offered = " ".join(plan.select_calls[0][1])
    assert "text-only" not in offered
    assert "vision-model" in offered


def test_typed_non_vision_ocr_model_is_refused(monkeypatch, prompts):
    _connection()
    _discover(
        monkeypatch,
        ModelEntry("vision-model", supports_vision=True),
        ModelEntry("text-only", supports_vision=False),
    )
    prompts(
        [
            "Enter model ID…",
            "gateway (OpenAI Compatible API)",
            "text-only",
            "← Back",
        ]
    )

    model_wizard.edit_task("ocr")

    assert Config.get_model_assignment("ocr").mode == "unset"


def test_reasoning_effort_is_edited_on_the_chat_target(monkeypatch, prompts):
    _connection()
    _assign("chat", connection="gateway", model="chat-model")
    _discover(monkeypatch, ModelEntry("chat-model"))
    prompts(["gateway / chat-model", "Model options…", "medium", "Save"])

    model_wizard.edit_task("chat")

    target = Config.get_model_assignment("chat").target
    assert target is not None
    assert target.options == {"reasoning_effort": "medium"}


def test_test_adapters_use_the_resolved_role_target(monkeypatch):
    _connection()
    _assign("chat", connection="gateway", model="chat-model")
    fake_llm = MagicMock()
    create = MagicMock(return_value=fake_llm)
    monkeypatch.setattr(
        "esdc.providers.create_llm_from_resolved_task", create
    )

    draft = model_wizard.TaskDraft(
        role="chat", assignment=Config.get_model_assignment("chat")
    )
    model_wizard._test_draft(draft)

    create.assert_called_once()
    fake_llm.invoke.assert_called_once()


def test_expired_oauth_is_reported_without_refresh_or_constructing_a_model(
    monkeypatch, prompts
):
    _connection(
        "oauth",
        provider_type="openai",
        base_url=None,
        oauth={
            "access_token": "expired-token",
            "refresh_token": "refresh-token",
            "expires_at": 1,
        },
    )
    _assign("chat", connection="oauth", model="gpt-test")
    create = MagicMock()
    monkeypatch.setattr(
        "esdc.providers.create_llm_from_resolved_task", create
    )

    draft = model_wizard.TaskDraft(
        role="chat", assignment=Config.get_model_assignment("chat")
    )
    model_wizard._test_draft(draft)

    create.assert_not_called()
    assert Config.get_connection("oauth")["oauth"]["access_token"] == "expired-token"


def test_ocr_test_uses_the_resolved_ocr_connection(monkeypatch):
    _connection()
    _assign("ocr", connection="gateway", model="vision-model")
    client = MagicMock()
    client.health_check.return_value = True
    client.query_image.return_value = "ESDC OCR TEST 12345"
    create = MagicMock(return_value=client)
    monkeypatch.setattr("esdc.corpus.ocr.create_ocr_client", create)

    draft = model_wizard.TaskDraft(
        role="ocr", assignment=Config.get_model_assignment("ocr")
    )
    model_wizard._test_draft(draft)

    create.assert_called_once()
    assert create.call_args.kwargs["resolved"].primary.model == "vision-model"
    client.query_image.assert_called_once()


def test_rerank_test_uses_the_rerank_endpoint(monkeypatch):
    _connection()
    _assign("rerank", connection="gateway", model="rank-model")
    client = MagicMock()
    client.rerank.return_value = [0.1, 0.2]
    monkeypatch.setattr(
        "esdc.corpus.provider_reranker.ProviderReranker", MagicMock(return_value=client)
    )

    draft = model_wizard.TaskDraft(
        role="rerank", assignment=Config.get_model_assignment("rerank")
    )
    model_wizard._test_draft(draft)

    client.rerank.assert_called_once()


@pytest.mark.parametrize("selection", ["discovered", "manual"])
def test_wizard_promotes_existing_chat_fallback(monkeypatch, prompts, selection):
    _connection()
    _connection("backup", base_url="http://backup.example/v1")
    _connection("other", base_url="http://other.example/v1")
    _assign(
        "chat",
        connection="gateway",
        model="a",
        fallbacks=(
            ModelTarget("backup", "b", {"reasoning_effort": "high"}),
            ModelTarget("gateway", "c", {"reasoning_effort": "low"}),
            ModelTarget("other", "d", {}),
        ),
    )
    _discover(
        monkeypatch,
        ModelEntry("a"),
        ModelEntry("b"),
        ModelEntry("c"),
        ModelEntry("d"),
    )
    if selection == "discovered":
        prompts(["backup / b", "Save"])
    else:
        prompts(
            [
                "Enter model ID…",
                "backup (OpenAI Compatible API)",
                "b",
                "Save",
            ]
        )

    model_wizard.edit_task("chat")

    Config._config_cache = None
    assignment = Config.get_model_assignment("chat")
    assert assignment.target == ModelTarget(
        "backup", "b", {"reasoning_effort": "high"}
    )
    assert [(pair.connection, pair.model) for pair in assignment.fallbacks] == [
        ("gateway", "c"),
        ("other", "d"),
    ]
    assert assignment.fallbacks[0].options == {"reasoning_effort": "low"}
    assert assignment.fallbacks[1].options == {}


def _catalog_role_for(role: TaskRole) -> str:
    return model_wizard._TASK_CATALOG_ROLE[role]


def _incompatible_entry(role: TaskRole, model: str = "incompatible") -> ModelEntry:
    other_role = "embedding" if _catalog_role_for(role) != "embedding" else "chat"
    return ModelEntry(model, roles=frozenset({other_role}))


def _row_targets(rows) -> list[ModelTarget]:
    return [choice.value for choice in rows if isinstance(choice.value, ModelTarget)]


def _assert_target_absent(rows, model: str = "incompatible") -> None:
    assert all(
        not isinstance(choice.value, ModelTarget) or choice.value.model != model
        for choice in rows
    ), [str(choice.title) for choice in rows]


@pytest.mark.parametrize(
    "role",
    ["chat", "embedding", "rerank", "ocr", "metadata", "cleanup"],
)
@pytest.mark.parametrize("path", ["discovered", "manual", "saved"])
def test_known_incompatible_model_is_rejected_in_all_picker_paths(
    monkeypatch, prompts, role, path
):
    if path == "saved":
        _connection()
        _assign(role, connection="gateway", model="incompatible")
        entry = _incompatible_entry(role)
        _discover(monkeypatch, entry)
        answers: list[Any] = ["Source and model"] if role == "rerank" else []
        answers += ["← Back"]
        if role == "rerank":
            answers += ["← Back"]
        plan = prompts(answers)

        model_wizard.edit_task(role)

        Config._config_cache = None
        assignment = Config.get_model_assignment(role)
        assert assignment.target is not None
        assert assignment.target.model == "incompatible"
        assert all(
            "gateway / incompatible" not in label
            for _message, labels in plan.select_calls
            for label in labels
        )
        return

    entries = {
        "gateway": {
            "type": "openai_compatible",
            "base_url": "http://gateway.example/v1",
        }
    }
    entry = _incompatible_entry(role)
    monkeypatch.setattr(
        model_wizard,
        "discover_models",
        MagicMock(return_value=ModelCatalog(entries=(entry,))),
    )
    draft = model_wizard.TaskDraft(
        role=role, assignment=TaskAssignment(mode="unset")
    )
    cache = model_wizard._CatalogCache()

    if path == "discovered":
        rows = model_wizard._target_rows(draft, cache, entries)
        _assert_target_absent(rows)
        return

    prompts(["gateway (OpenAI Compatible API)", "incompatible"])
    assert model_wizard._manual_target(draft, entries, cache) is None


@pytest.mark.parametrize(
    "role",
    ["chat", "embedding", "rerank", "ocr", "metadata", "cleanup"],
)
@pytest.mark.parametrize("kind", ["empty_roles", "absent", "warning"])
def test_manual_unknown_model_remains_usable(prompts, role, kind):
    entries = {
        "gateway": {
            "type": "openai_compatible",
            "base_url": "http://gateway.example/v1",
        }
    }
    if kind == "empty_roles":
        catalog = ModelCatalog(entries=(ModelEntry("unknown", roles=frozenset()),))
    elif kind == "absent":
        catalog = ModelCatalog(entries=(ModelEntry("other"),))
    else:
        catalog = ModelCatalog(warning="discovery failed")
    cache = model_wizard._CatalogCache()
    cache._entries[("gateway", "openai_compatible", "http://gateway.example/v1")] = catalog
    draft = model_wizard.TaskDraft(
        role=role,
        assignment=TaskAssignment(mode="unset", enabled=False)
        if role == "rerank"
        else TaskAssignment(mode="unset"),
    )
    prompts(["gateway (OpenAI Compatible API)", "unknown"])

    target = model_wizard._manual_target(draft, entries, cache)

    assert target == ModelTarget("gateway", "unknown", {})
    if role == "rerank":
        assert draft.assignment.enabled is False


@pytest.mark.parametrize("path", ["discovered", "manual", "saved"])
def test_ocr_known_nonvision_model_is_rejected_in_all_picker_paths(
    monkeypatch, prompts, path
):
    entry = ModelEntry(
        "text-only", roles=frozenset({"chat"}), supports_vision=False
    )
    if path == "saved":
        _connection()
        _assign("ocr", connection="gateway", model="text-only")
        _discover(monkeypatch, entry)
        plan = prompts(["← Back"])

        model_wizard.edit_task("ocr")

        Config._config_cache = None
        assert Config.get_model_assignment("ocr").target is not None
        assert all(
            "gateway / text-only" not in label
            for _message, labels in plan.select_calls
            for label in labels
        )
        return

    entries = {
        "gateway": {
            "type": "openai_compatible",
            "base_url": "http://gateway.example/v1",
        }
    }
    monkeypatch.setattr(
        model_wizard,
        "discover_models",
        MagicMock(return_value=ModelCatalog(entries=(entry,))),
    )
    draft = model_wizard.TaskDraft(
        role="ocr", assignment=TaskAssignment(mode="unset")
    )
    cache = model_wizard._CatalogCache()
    if path == "discovered":
        rows = model_wizard._target_rows(draft, cache, entries)
        _assert_target_absent(rows, "text-only")
        return

    prompts(["gateway (OpenAI Compatible API)", "text-only"])
    assert model_wizard._manual_target(draft, entries, cache) is None


def test_ocr_vision_true_and_unknown_remain_selectable(monkeypatch):
    entries = {
        "gateway": {
            "type": "openai_compatible",
            "base_url": "http://gateway.example/v1",
        }
    }
    monkeypatch.setattr(
        model_wizard,
        "discover_models",
        MagicMock(
            return_value=ModelCatalog(
                entries=(
                    ModelEntry(
                        "vision", roles=frozenset({"chat"}), supports_vision=True
                    ),
                    ModelEntry(
                        "unknown", roles=frozenset({"chat"}), supports_vision=None
                    ),
                )
            )
        ),
    )
    draft = model_wizard.TaskDraft(
        role="ocr", assignment=TaskAssignment(mode="unset")
    )
    rows = model_wizard._target_rows(
        draft, model_wizard._CatalogCache(), entries
    )
    assert {target.model for target in _row_targets(rows)} == {"vision", "unknown"}


def test_capability_rejection_keeps_other_model_and_cancel_usable(
    monkeypatch, prompts
):
    _connection()
    _discover(
        monkeypatch,
        ModelEntry("good", roles=frozenset()),
        ModelEntry("bad", roles=frozenset({"embedding"})),
    )
    prompts(
        [
            "Enter model ID…",
            "gateway (OpenAI Compatible API)",
            "bad",
            "gateway / good",
            "Save",
        ]
    )

    model_wizard.edit_task("chat")

    Config._config_cache = None
    assignment = Config.get_model_assignment("chat")
    assert assignment.target == ModelTarget("gateway", "good", {})


def test_conflicting_duplicate_catalog_entries_reject_positive_incompatibility(
    prompts,
):
    entries = {
        "gateway": {
            "type": "openai_compatible",
            "base_url": "http://gateway.example/v1",
        }
    }
    catalog = ModelCatalog(
        entries=(
            ModelEntry("dup", roles=frozenset()),
            ModelEntry("dup", roles=frozenset({"embedding"})),
        )
    )
    cache = model_wizard._CatalogCache()
    cache._entries[("gateway", "openai_compatible", "http://gateway.example/v1")] = catalog
    draft = model_wizard.TaskDraft(
        role="chat", assignment=TaskAssignment(mode="unset")
    )

    rows = model_wizard._target_rows(draft, cache, entries)
    _assert_target_absent(rows, "dup")

    prompts(["gateway (OpenAI Compatible API)", "dup"])
    assert model_wizard._manual_target(draft, entries, cache) is None


@pytest.mark.parametrize(
    "role",
    ["chat", "embedding", "rerank", "ocr", "metadata", "cleanup"],
)
@pytest.mark.parametrize("connection", ["missing", "ineligible"])
def test_saved_missing_or_ineligible_connection_is_not_reinserted(
    monkeypatch, role, connection
):
    entries = {
        "gateway": {
            "type": "openai_compatible",
            "base_url": "http://gateway.example/v1",
        }
    }
    if connection == "ineligible":
        entries["ollama"] = {"type": "ollama", "base_url": "http://ollama:11434"}
    target = ModelTarget(connection, "saved")
    draft = model_wizard.TaskDraft(
        role=role,
        assignment=TaskAssignment(mode="model", target=target),
    )
    monkeypatch.setattr(
        model_wizard,
        "discover_models",
        MagicMock(return_value=ModelCatalog(entries=())),
    )

    rows = model_wizard._target_rows(
        draft, model_wizard._CatalogCache(), entries
    )

    _assert_target_absent(rows, "saved")
